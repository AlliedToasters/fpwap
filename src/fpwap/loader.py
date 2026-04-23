from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn


def resolve_snapshot_dir(model: str) -> Path:
    """Resolve `model` to a local HF snapshot directory.

    - If `model` is an existing directory, return it as a Path.
    - Otherwise treat it as a hub id and resolve via the local HF cache
      (`snapshot_download(..., local_files_only=True)`). If the model isn't
      cached, re-raise with an actionable message that names the id.

    Centralizing this lets `Sweep(model="meta-llama/...")` Just Work for
    consumers who have the model cached, without every call site
    re-implementing the dance.
    """
    p = Path(model)
    if p.is_dir():
        return p
    try:
        return Path(snapshot_download(model, local_files_only=True))
    except Exception as exc:
        raise FileNotFoundError(
            f"fpwap could not resolve model {model!r} to a local snapshot. "
            f"Either pass an existing snapshot directory, or pre-cache the "
            f"model with `huggingface-cli download {model}` (or equivalent "
            f"`snapshot_download({model!r})`). Underlying error: {exc}"
        ) from exc

_SAFE_TO_TORCH_DTYPE: dict[str, str] = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def build_accel_index_from_hf_cache(snapshot_dir: Path) -> dict[str, dict[str, Any]]:
    """Convert HF's model.safetensors.index.json to accelerate's loader format.

    The resulting index maps each weight name to an entry accelerate's
    OffloadedWeightsLoader understands: (safetensors_file, weight_name, dtype,
    shape). dtype is the torch attribute name (e.g. "bfloat16"), not the
    safetensors wire name (e.g. "BF16") — see SPEC D.1.
    """
    snapshot_dir = Path(snapshot_dir)
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
    else:
        # Single-shard models skip the index and ship one model.safetensors.
        single = snapshot_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(
                f"no safetensors index or single shard at {snapshot_dir}"
            )
        with safe_open(str(single), framework="pt") as st:  # type: ignore[no-untyped-call]
            weight_map = {k: "model.safetensors" for k in st.keys()}

    accel_index: dict[str, dict[str, Any]] = {}
    # Cache safe_open handles per shard to avoid O(n_weights) file opens.
    handles: dict[str, Any] = {}
    try:
        for weight_name, shard_file in weight_map.items():
            shard_path = str(snapshot_dir / shard_file)
            st = handles.get(shard_path)
            if st is None:
                st = safe_open(shard_path, framework="pt").__enter__()  # type: ignore[no-untyped-call]
                handles[shard_path] = st
            sl = st.get_slice(weight_name)
            accel_index[weight_name] = {
                "safetensors_file": shard_path,
                "weight_name": weight_name,
                "dtype": _SAFE_TO_TORCH_DTYPE[sl.get_dtype()],
                "shape": list(sl.get_shape()),
            }
    finally:
        for st in handles.values():
            st.__exit__(None, None, None)
    return accel_index


def alias_tied_weights_in_index(
    model: nn.Module,
    accel_index: dict[str, dict[str, Any]],
) -> None:
    """Add aliases for tied weights to the index. Requires model.tie_weights() first.

    Tied parameters (e.g. `lm_head.weight` ↔ `model.embed_tokens.weight`) are
    only stored once in the safetensors shards, so the accel_index built from
    those shards is missing one of the names. accelerate's hook lookup is by
    absolute module-parameter path, so both names have to resolve.

    Detection is by object identity on `named_parameters(remove_duplicate=False)`;
    the default `named_parameters()` hides aliases and silently misses the case.
    """
    by_id: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        by_id.setdefault(id(param), []).append(name)
    for names in by_id.values():
        if len(names) < 2:
            continue
        sources = [n for n in names if n in accel_index]
        targets = [n for n in names if n not in accel_index]
        if sources and targets:
            source_entry = accel_index[sources[0]]
            for t in targets:
                accel_index[t] = source_entry


def build_empty_model_and_index(
    model_id: str,
    snapshot_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, dict[str, dict[str, Any]]]:
    """Construct an empty-weights model and the accelerate index for its shards.

    This is the lower-level helper fpwap's engine uses directly: the model
    stays on meta device, and the returned index is suitable for constructing
    an OffloadedWeightsLoader that mmap's weights from the HF cache. No
    AlignDevicesHook is installed.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)  # type: ignore[no-untyped-call]
    model.tie_weights()  # MUST precede index construction — SPEC D.2

    accel_index = build_accel_index_from_hf_cache(Path(snapshot_dir))
    alias_tied_weights_in_index(model, accel_index)
    return model, accel_index


def load_from_cache(
    model_id: str,
    snapshot_dir: Path,
    offload_dir: Path,
    execution_device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load a model larger than CPU RAM via mmap-from-HF-cache (SPEC §12.3).

    The escape hatch in disk_offload: if `offload_dir/index.json` exists before
    disk_offload is called, it skips the RAM-materializing copy of the
    state_dict and attaches hooks directly. Combined with an index whose
    entries point at safetensors shards in the HF cache, the model never
    lands in CPU RAM.

    Index write order is load-bearing: `OffloadedWeightsLoader` caches the
    index at construction (SPEC D.3), so writing after disk_offload silently
    has no effect.
    """
    from accelerate import disk_offload

    if execution_device is None:
        execution_device = torch.device("cuda:0")

    model, accel_index = build_empty_model_and_index(
        model_id=model_id, snapshot_dir=snapshot_dir, dtype=dtype
    )

    offload_dir = Path(offload_dir)
    offload_dir.mkdir(parents=True, exist_ok=True)
    with open(offload_dir / "index.json", "w") as f:
        json.dump(accel_index, f)

    result = disk_offload(
        model,
        offload_dir=str(offload_dir),
        execution_device=execution_device,
    )
    return result  # type: ignore[no-any-return]


def _load_layer(
    model: nn.Module,
    layer_idx: int,
    plumbing: Any,
    loader: Any,
    device: torch.device,
) -> None:
    """Materialize layer `layer_idx` weights onto the execution device.

    Approach A from SPEC §12.4: fetch each param directly from the
    OffloadedWeightsLoader and install it via accelerate's
    `set_module_tensor_to_device`, which handles the meta→real transition
    that plain `param.data = tensor` refuses across device boundaries.
    No AlignDevicesHook is installed or fired.
    """
    from accelerate.utils import set_module_tensor_to_device

    layer = plumbing.layer_modules(model)[layer_idx]
    prefix = plumbing.layer_prefix(layer_idx)
    non_blocking = device.type == "cuda"
    for rel_name, _ in layer.named_parameters():
        full_name = f"{prefix}.{rel_name}"
        tensor = loader[full_name]
        set_module_tensor_to_device(
            layer,
            rel_name,
            device,
            value=tensor,
            non_blocking=non_blocking,
        )


def _load_named_param(
    model: nn.Module,
    full_name: str,
    loader: Any,
    device: torch.device,
) -> None:
    """Materialize a single param (by absolute name) onto the execution device.

    Used for pass-0 embedding weights, which are kept resident across all
    layers rather than streamed per-layer (SPEC D.5).
    """
    from accelerate.utils import set_module_tensor_to_device

    submod_path, _, param_name = full_name.rpartition(".")
    submod = model.get_submodule(submod_path) if submod_path else model
    tensor = loader[full_name]
    non_blocking = device.type == "cuda"
    set_module_tensor_to_device(
        submod, param_name, device, value=tensor, non_blocking=non_blocking
    )


def _unload_layer(model: nn.Module, layer_idx: int, plumbing: Any) -> None:
    """Release layer `layer_idx` weights back to the meta device.

    Accelerate's `set_module_tensor_to_device(..., device="meta")` is the
    counterpart to _load_layer — it re-installs zero-size meta placeholders
    on the same parameter objects so references held by hooks stay valid.
    """
    from accelerate.utils import set_module_tensor_to_device

    layer = plumbing.layer_modules(model)[layer_idx]
    for rel_name, _ in list(layer.named_parameters()):
        set_module_tensor_to_device(layer, rel_name, "meta")
