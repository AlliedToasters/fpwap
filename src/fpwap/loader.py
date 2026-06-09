from __future__ import annotations

import copy
import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn

_HAS_POSIX_FADVISE = hasattr(os, "posix_fadvise")


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


@dataclass
class WeightConversionPlan:
    """Recipe to realize one converted module param from raw checkpoint keys.

    `converter` is the transformers WeightTransform that owns the conversion
    (e.g. MergeModulelist + Concatenate for fused MoE experts). It is a
    template: transformers' transforms accumulate per-load state, so every
    materialization deep-copies it first. `sources` preserves the natural-sort
    order `from_pretrained` collects tensors in — expert stacking order
    depends on it. `targets` lists every full module-param name the convert
    realizes (one-to-many converters like qkv-split produce several).
    """

    converter: Any
    sources: list[tuple[str, str]] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)


def build_conversion_plans(
    model: nn.Module,
    accel_index: dict[str, dict[str, Any]],
) -> dict[str, WeightConversionPlan]:
    """Map module param names to conversion plans over raw checkpoint keys.

    transformers ≥5 reconciles checkpoints whose on-disk layout differs from
    the instantiated modules (all current HF MoE checkpoints: per-expert
    tensors on disk, fused stacked params in the module) through a global
    WeightConverter mapping applied inside `from_pretrained`. fpwap's index
    is built from raw safetensors keys, so module params produced by such a
    conversion are missing from it — resolve them here so the streaming
    loader can fuse on demand (issue #77).

    Returns {} when the installed transformers has no conversion machinery
    (<5.x — layouts match by construction there) or when every module param
    resolves directly against the index. Keys are full module-param names;
    plans for one-to-many converters are shared across their targets.
    """
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import (
            WeightConverter,
            WeightRenaming,
            dot_natural_key,
            rename_source_key,
        )
    except ImportError:
        return {}

    # Annotated for PreTrainedModel upstream, but only walks named_modules —
    # safe for any nn.Module (non-HF modules just get the legacy renames).
    weight_mapping = get_model_conversion_mapping(cast(Any, model))
    renamings = [m for m in weight_mapping if isinstance(m, WeightRenaming)]
    converters = [m for m in weight_mapping if isinstance(m, WeightConverter)]
    if not renamings and not converters:
        return {}

    meta_state_dict = model.state_dict()
    prefix = getattr(model, "base_model_prefix", None)
    pattern_to_converter = {
        pattern: converter
        for converter in converters
        for pattern in converter.source_patterns
    }

    grouped: dict[str, WeightConversionPlan] = {}
    for original_key in sorted(accel_index, key=dot_natural_key):
        renamed_key, source_pattern = rename_source_key(
            original_key, renamings, converters, prefix, meta_state_dict
        )
        if renamed_key not in meta_state_dict and original_key in meta_state_dict:
            # Mirrors from_pretrained: the key matched a pattern but the
            # renamed form doesn't exist — it shouldn't have been renamed.
            renamed_key, source_pattern = rename_source_key(
                original_key, [], [], prefix, meta_state_dict
            )
        if renamed_key not in meta_state_dict:
            continue
        if source_pattern is None:
            if renamed_key != original_key and renamed_key not in accel_index:
                # Pure renaming (legacy keys, prefix moves): same bytes under
                # a new name — alias the index entry like tied weights.
                accel_index[renamed_key] = accel_index[original_key]
            continue
        converter = pattern_to_converter[source_pattern]
        plan = grouped.setdefault(renamed_key, WeightConversionPlan(converter))
        plan.sources.append((original_key, source_pattern))

    plans: dict[str, WeightConversionPlan] = {}
    for first_target, plan in grouped.items():
        # One-to-many converters realize every target in one convert() call;
        # expose the plan under each full target name (transformers derives
        # them the same way for its unexpected-keys accounting).
        target_patterns = plan.converter.target_patterns
        plan.targets = [
            first_target.replace(target_patterns[0], pattern)
            for pattern in target_patterns
        ]
        for target in plan.targets:
            plans[target] = plan
    return plans


class ConvertingWeightsLoader:
    """Dict-like loader that resolves converted module params on demand.

    Raw checkpoint keys pass straight through to the wrapped
    OffloadedWeightsLoader; module param names that only exist after
    transformers' checkpoint conversion (fused MoE experts, split qkv, ...)
    are materialized by loading their source tensors and running the
    registered conversion, exactly as `from_pretrained` would. Sibling
    outputs of one-to-many converts are cached until first read so each
    source tensor is read from disk once.
    """

    def __init__(
        self,
        base: Any,
        plans: dict[str, WeightConversionPlan],
        config: Any = None,
    ) -> None:
        self._base = base
        self._plans = plans
        self._config = config
        self._realized: dict[str, torch.Tensor] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._base or key in self._plans

    def keys(self) -> list[str]:
        return list(self._base.keys()) + list(self._plans.keys())

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._base:
            return self._base[key]  # type: ignore[no-any-return]
        if key in self._realized:
            return self._realized.pop(key)
        plan = self._plans.get(key)
        if plan is None:
            raise KeyError(
                f"{key!r} is neither a checkpoint weight nor a converted "
                f"module param — the snapshot may not match the instantiated "
                f"model under the installed transformers version"
            )
        converter = copy.deepcopy(plan.converter)
        first_target = plan.targets[0]
        for source_key, source_pattern in plan.sources:
            # Callables → transformers' sync materialization path; the read
            # from disk happens inside convert(), one source at a time.
            converter.add_tensor(
                first_target,
                source_key,
                source_pattern,
                lambda source_key=source_key: self._base[source_key],
            )
        realized = converter.convert(first_target, config=self._config)
        for name, value in realized.items():
            self._realized[name] = value[0] if isinstance(value, list) else value
        return self._realized.pop(key)


def build_empty_model_and_index(
    model_id: str,
    snapshot_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, dict[str, dict[str, Any]], dict[str, float]]:
    """Construct an empty-weights model and the accelerate index for its shards.

    This is the lower-level helper fpwap's engine uses directly: the model
    stays on meta device, and the returned index is suitable for constructing
    an OffloadedWeightsLoader that mmap's weights from the HF cache. No
    AlignDevicesHook is installed.

    Returns (model, accel_index, timing_dict) where timing_dict has keys
    config_s, model_s, index_s for sub-phase breakdowns.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    t0 = time.perf_counter_ns()
    config = AutoConfig.from_pretrained(model_id)
    config_s = (time.perf_counter_ns() - t0) / 1e9

    t0 = time.perf_counter_ns()
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)  # type: ignore[no-untyped-call]
    model.tie_weights()  # MUST precede index construction — SPEC D.2
    model_s = (time.perf_counter_ns() - t0) / 1e9

    t0 = time.perf_counter_ns()
    accel_index = build_accel_index_from_hf_cache(Path(snapshot_dir))
    alias_tied_weights_in_index(model, accel_index)
    index_s = (time.perf_counter_ns() - t0) / 1e9

    timing = {"config_s": config_s, "model_s": model_s, "index_s": index_s}
    return model, accel_index, timing


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

    model, accel_index, _ = build_empty_model_and_index(
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


def _parse_safetensors_offsets(path: str) -> dict[str, tuple[int, int]]:
    """Parse a safetensors header to get absolute byte offsets per tensor.

    Returns {tensor_name: (abs_start, abs_end)} where offsets are relative
    to the start of the file (not the data region).
    """
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)

    data_start = 8 + header_size
    header = json.loads(header_bytes)

    offsets: dict[str, tuple[int, int]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        offsets[name] = (data_start + start, data_start + end)
    return offsets


class ShardPageAdvisor:
    """Advises the kernel about page cache for safetensors shards.

    Uses posix_fadvise(DONTNEED) after a layer is unloaded so the kernel
    can reclaim those pages for upcoming layers. On non-Linux platforms
    (no posix_fadvise), all methods are silent no-ops.

    `virtual_sources` maps module param names that don't exist on disk
    (converted weights, e.g. fused MoE experts) to the raw checkpoint keys
    backing them, so advising on the module name reaches the real byte
    ranges.
    """

    def __init__(
        self,
        accel_index: dict[str, dict[str, Any]],
        virtual_sources: dict[str, list[str]] | None = None,
    ) -> None:
        self._offsets: dict[str, tuple[str, int, int]] = {}
        self._virtual_sources = virtual_sources or {}
        if not _HAS_POSIX_FADVISE:
            return

        shard_paths: set[str] = set()
        for entry in accel_index.values():
            shard_paths.add(entry["safetensors_file"])

        shard_offsets: dict[str, dict[str, tuple[int, int]]] = {}
        for path in shard_paths:
            shard_offsets[path] = _parse_safetensors_offsets(path)

        for weight_name, entry in accel_index.items():
            shard_path = entry["safetensors_file"]
            st_name = entry["weight_name"]
            if st_name in shard_offsets.get(shard_path, {}):
                start, end = shard_offsets[shard_path][st_name]
                self._offsets[weight_name] = (shard_path, start, end)

    def _advise(self, weight_names: list[str], advice: int) -> None:
        if not _HAS_POSIX_FADVISE:
            return
        by_shard: dict[str, list[tuple[int, int]]] = {}
        expanded: list[str] = []
        for name in weight_names:
            expanded.extend(self._virtual_sources.get(name, (name,)))
        for name in expanded:
            if name not in self._offsets:
                continue
            path, start, end = self._offsets[name]
            by_shard.setdefault(path, []).append((start, end))

        for path, ranges in by_shard.items():
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    for start, end in ranges:
                        os.posix_fadvise(fd, start, end - start, advice)
                finally:
                    os.close(fd)
            except OSError:
                pass

    def advise_dontneed(self, weight_names: list[str]) -> None:
        self._advise(weight_names, os.POSIX_FADV_DONTNEED)

    def advise_willneed(self, weight_names: list[str]) -> None:
        self._advise(weight_names, os.POSIX_FADV_WILLNEED)
