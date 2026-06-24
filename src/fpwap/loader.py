from __future__ import annotations

import copy
import ctypes
import ctypes.util
import json
import os
import resource
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

# Page-pinning machinery for the resident-weight cache. preadv-based loading
# makes a weight "resident" only by leaving its pages in the OS page cache;
# omitting DONTNEED is advisory, and under real memory pressure (no free RAM +
# the surprise sweep's emit/staging buffers) the kernel reclaims those clean
# file-backed pages anyway, silently re-faulting the "resident" set from disk
# every sweep. mlock'ing the pages makes residency binding; mincore lets us
# detect (and warn about) eviction when pinning is off. All optional — absent
# libc/posix_fadvise, the advisor falls back to the advisory-only behavior.
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_PROT_READ = 0x1
_MAP_SHARED = 0x01
_MAP_POPULATE = 0x8000  # Linux-specific: prefault the range on mmap
_MAP_FAILED = ctypes.c_void_p(-1).value
# mincore watchdog (advisory-only mode): sample residency every N advise calls
# (~2 sweeps of a 48-layer model) and warn once if the resident set has been
# evicted below this fraction.
_RESIDENCY_CHECK_EVERY = 96
_RESIDENCY_WARN_BELOW = 0.5
# Headroom under RLIMIT_MEMLOCK left for torch/CUDA pinned staging buffers
# (the staged loader pins ~one layer; CUDA pins context buffers) so weight
# pinning doesn't starve them and trip mlock ENOMEM.
_PIN_HEADROOM = 2 << 30

# Process-global registry of mlock'd weight-page regions, keyed by
# (shard_path, aligned_offset, aligned_len) -> mapped address. Pins persist
# across Sweep/streamer teardown: lens issues one Sweep per unit (shard_tokens=
# 1), so a per-advisor pin would be locked and released every unit and never
# reused. Holding them process-globally keeps the resident weight set warm
# across units. release_resident_pages() tears it down (tests / model switch).
_PINNED_REGIONS: dict[tuple[str, int, int], int] = {}
_PINNED_TOTAL = 0
# One-time warning keys already emitted this process (the advisor is rebuilt
# per Sweep — one per unit at shard_tokens=1 — so per-instance guards would spam).
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(message, flush=True)


def _pin_cap_bytes() -> int:
    """How many bytes may be globally mlock'd: RLIMIT_MEMLOCK minus headroom."""
    return max(0, _memlock_limit_bytes() - _PIN_HEADROOM)


def release_resident_pages() -> None:
    """munlock + munmap every globally pinned weight region (process-wide)."""
    global _PINNED_TOTAL
    if _LIBC is not None:
        for (_path, _start, length), addr in _PINNED_REGIONS.items():
            try:
                _LIBC.munlock(ctypes.c_void_p(addr), length)
                _LIBC.munmap(ctypes.c_void_p(addr), length)
            except OSError:
                pass
    _PINNED_REGIONS.clear()
    _PINNED_TOTAL = 0


def _load_libc() -> ctypes.CDLL | None:
    if not _HAS_POSIX_FADVISE:  # gates on Linux-ish platforms
        return None
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    try:
        libc = ctypes.CDLL(name, use_errno=True)
    except OSError:
        return None
    try:
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_long,
        ]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    except AttributeError:
        return None
    return libc


_LIBC = _load_libc()


def _memlock_limit_bytes() -> int:
    """Soft RLIMIT_MEMLOCK in bytes; a large sentinel for 'unlimited'."""
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (ValueError, OSError, AttributeError):
        return 0
    if soft == resource.RLIM_INFINITY:
        return 1 << 62
    return int(soft)


def _phys_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


def _resident_page_budget() -> int:
    """Byte budget for weight pages held resident in the page cache.

    ``DONTNEED``-after-unload frees a layer's pages so the kernel can reclaim
    them for the next layer. That is necessary when the *touched* weight set
    exceeds host RAM, but actively harmful when it fits: it forces every sweep
    to re-read the shards from disk instead of reusing the warm page cache, so
    a workload that streams many shards (each a full layer sweep) runs
    disk-bound at the snapshot's read rate.

    The advisor keeps touched layers resident up to this budget and only
    DONTNEEDs beyond it (see ``ShardPageAdvisor.advise_dontneed``). Truncated
    sweeps (an activations-only profile stops at the deepest probed layer) and
    repeated reads then run compute-bound; a full-depth sweep larger than the
    budget fills it and evicts the rest gracefully. The budget is a fraction
    of total RAM rather than the whole snapshot, so the policy is correct even
    when the full snapshot is larger than RAM but the swept prefix is not.

    ``0`` disables resident caching (restores the always-DONTNEED behavior).
    Override the whole policy with ``FPWAP_PAGE_RESIDENT=0`` (off) or set the
    budget directly with ``FPWAP_PAGE_RESIDENT_GB=<n>``.
    """
    if os.environ.get("FPWAP_PAGE_RESIDENT") in ("0", "false", "False"):
        return 0
    gb = os.environ.get("FPWAP_PAGE_RESIDENT_GB")
    if gb is not None:
        try:
            return int(float(gb) * 1e9)
        except ValueError:
            pass
    ram = _phys_ram_bytes()
    # Leave ~40% of RAM for activation/emit buffers, pinned staging buffers,
    # other processes, and the kernel's own reclaim headroom.
    return int(0.6 * ram) if ram > 0 else 0


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


_STAGED_ALIGN = 64  # byte alignment for tensors packed into the staging buffer
_STAGED_READ_THREADS = 8  # preadv workers; reads release the GIL


@dataclass
class _SourceRange:
    """One contiguous byte range of checkpoint data backing (part of) a param."""

    path: str
    start: int
    nbytes: int


def _close_fds(fds: dict[str, int]) -> None:
    for fd in fds.values():
        os.close(fd)
    fds.clear()


class StagedLayerLoader:
    """Pinned-staging layer loader: direct byte reads + one async H2D per layer.

    The per-tensor path (`_load_layer`) is load-bound twice over: each
    param goes through safetensors `get_tensor` (a fresh pageable CPU
    tensor per call — and for transformers ≥5 fused-MoE params, through
    the whole checkpoint-conversion machinery per layer), then through a
    `set_module_tensor_to_device(..., non_blocking=True)` whose
    `non_blocking` is silently synchronous from pageable memory. Measured
    on a Qwen3-30B-A3B MoE layer: ~1 GB/s effective against a gen5 x16
    link that does ~25× that from pinned memory, with the whole sweep
    load-bound.

    This loader cuts both costs:

    - **Reads**: each param's bytes are `preadv`'d straight from the
      safetensors shards into a reused pinned host buffer on a small
      thread pool (the read releases the GIL; offsets come from the shard
      headers). Fused params produced by checkpoint conversion are
      assembled by concatenating their source ranges in plan order — and
      the first time each param pattern is assembled, the result is
      verified bitwise against the conversion machinery's output; a
      pattern that doesn't match byte-concatenation permanently falls
      back to the tensor path (`loader[name]` → copy into pinned), so the
      fast path can never silently produce different weights.
    - **Copies**: ONE genuinely-async H2D from the pinned buffer into a
      fresh contiguous device buffer on a dedicated copy stream; params
      install as views into that buffer.

    Contract with the engine:

    - `load_layer` returns a `torch.cuda.Event` recorded on the copy
      stream after the H2D copy, or None when it fell back to the
      per-tensor path. The caller must make its compute stream wait on the
      event (`Sweep.run` does, via `streamer.wait_load_ready`) before
      running the layer forward.
    - Unloading is unchanged: `_unload_layer`'s meta re-install drops the
      param views, freeing the device buffer back to the caching
      allocator. The buffer is `record_stream`-marked against the copy
      stream so the allocator won't recycle it under an in-flight copy.
    - Install semantics mirror `set_module_tensor_to_device` for the
      plain-Parameter meta→device case: checkpoint values are cast to the
      meta param's dtype, shape mismatches raise, and a fresh
      `nn.Parameter` replaces the placeholder. Layers with exotic param
      subclasses (quantized wrappers etc.) fall back to `_load_layer`,
      as does the whole loader if pinned allocation fails.

    Loads are serialized by the engine (one outstanding prefetch), so a
    single pinned buffer suffices: each fill waits host-side on the
    previous layer's copy event before reusing the buffer.
    """

    def __init__(
        self,
        loader: Any,
        device: torch.device,
        accel_index: dict[str, dict[str, Any]] | None = None,
        plans: dict[str, WeightConversionPlan] | None = None,
    ) -> None:
        self._loader = loader
        self._device = device
        self._accel_index = accel_index or {}
        self._plans = plans or {}
        self._copy_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        self._pinned: torch.Tensor | None = None
        self._pending: torch.cuda.Event | None = None
        self._broken = False  # pinned alloc failed; permanent fallback
        self._offsets: dict[str, dict[str, tuple[int, int]]] = {}
        self._fds: dict[str, int] = {}
        self._pool: Any | None = None
        # Byte-assembly trust state, keyed by layer-relative param name
        # (layers are structurally identical, so one verification covers
        # the pattern across all layers).
        self._verified: set[str] = set()
        self._tensor_path: set[str] = set()
        # Safety net: close shard fds on GC if close() is never reached
        # (holds the dict, not self, so the finalizer doesn't pin the loader).
        import weakref

        weakref.finalize(self, _close_fds, self._fds)

    def close(self) -> None:
        # The last H2D may still be in flight on the copy stream; it reads
        # from the pinned buffer this loader owns, so drain it before the
        # buffer can be garbage-collected.
        if self._pending is not None:
            self._pending.synchronize()
            self._pending = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        _close_fds(self._fds)

    def _range_of_key(self, key: str) -> _SourceRange | None:
        entry = self._accel_index.get(key)
        if entry is None:
            return None
        path = entry["safetensors_file"]
        st_name = entry.get("weight_name", key)
        if path not in self._offsets:
            self._offsets[path] = _parse_safetensors_offsets(path)
        rng = self._offsets[path].get(st_name)
        if rng is None:
            return None
        start, end = rng
        return _SourceRange(path, start, end - start)

    def _ranges_for(self, full_name: str, param: nn.Parameter) -> list[_SourceRange] | None:
        """Byte ranges whose concatenation should equal the param's bytes.

        None when the param can't be byte-assembled (unknown key, dtype
        mismatch vs the module, one-to-many conversion, size mismatch) —
        callers then use the tensor path.
        """
        plan = self._plans.get(full_name)
        if plan is not None:
            if len(plan.targets) != 1:
                return None  # one-to-many (qkv split etc.): not concatenation
            keys = [k for k, _ in plan.sources]
        else:
            keys = [full_name]
        ranges: list[_SourceRange] = []
        module_dtype = str(param.dtype).removeprefix("torch.")
        for key in keys:
            entry = self._accel_index.get(key)
            if entry is None or str(entry.get("dtype", "")) != module_dtype:
                return None
            rng = self._range_of_key(key)
            if rng is None:
                return None
            ranges.append(rng)
        if sum(r.nbytes for r in ranges) != param.element_size() * param.numel():
            return None
        return ranges

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def load_layer(
        self, model: nn.Module, layer_idx: int, plumbing: Any
    ) -> torch.cuda.Event | None:
        layer = plumbing.layer_modules(model)[layer_idx]
        prefix = plumbing.layer_prefix(layer_idx)

        # Pack plan: (rel_name, meta_param, byte_offset, nbytes).
        entries: list[tuple[str, nn.Parameter, int, int]] = []
        total = 0
        fallback = self._broken
        for rel_name, param in layer.named_parameters():
            if type(param) is not nn.Parameter:
                fallback = True
                break
            offset = -(-total // _STAGED_ALIGN) * _STAGED_ALIGN
            nbytes = param.element_size() * param.numel()
            entries.append((rel_name, param, offset, nbytes))
            total = offset + nbytes
        if fallback:
            _load_layer(model, layer_idx, plumbing, self._loader, self._device)
            return None

        # Host fill: wait for the previous layer's H2D to release the pinned
        # buffer, then pack every tensor back-to-back at the param's dtype.
        if self._pending is not None:
            self._pending.synchronize()
            self._pending = None
        if self._pinned is None or self._pinned.numel() < total:
            self._pinned = None
            try:
                self._pinned = torch.empty(
                    total, dtype=torch.uint8, pin_memory=True
                )
            except RuntimeError:
                self._broken = True
                _load_layer(model, layer_idx, plumbing, self._loader, self._device)
                return None
        self._fill_pinned(entries, prefix)

        assert self._pinned is not None
        # One async H2D into a fresh device buffer on the copy stream. The
        # buffer's home stream is this thread's current (compute) stream, so
        # post-unload reuse by compute-stream allocations is ordered; the
        # record_stream covers the cross-stream write.
        dev = torch.empty(total, dtype=torch.uint8, device=self._device)
        event = torch.cuda.Event()  # type: ignore[no-untyped-call]
        with torch.cuda.stream(self._copy_stream):
            dev.copy_(self._pinned[:total], non_blocking=True)
            event.record(self._copy_stream)
        dev.record_stream(self._copy_stream)

        # Install params as views into the device buffer (the meta→real
        # transition only needs a fresh Parameter in module._parameters,
        # exactly what set_module_tensor_to_device does underneath).
        for rel_name, param, offset, nbytes in entries:
            view = dev[offset : offset + nbytes].view(param.dtype).view(param.shape)
            submod_path, _, leaf = rel_name.rpartition(".")
            submod = layer.get_submodule(submod_path) if submod_path else layer
            submod._parameters[leaf] = nn.Parameter(
                view, requires_grad=param.requires_grad
            )
        self._pending = event
        return event

    def _fill_pinned(
        self, entries: list[tuple[str, nn.Parameter, int, int]], prefix: str
    ) -> None:
        """Pack every param's bytes into the pinned buffer.

        Byte-assemblable params are read with preadv on a thread pool;
        the rest go through `loader[name]` (conversion machinery / dtype
        cast) on this thread while the pool drains. First-time byte
        assemblies are verified bitwise against the tensor path before
        the pattern is trusted.
        """
        import concurrent.futures as _cf

        assert self._pinned is not None
        pinned_mv = memoryview(self._pinned.numpy())  # type: ignore[arg-type]

        byte_jobs: list[tuple[str, nn.Parameter, int, int, list[_SourceRange]]] = []
        tensor_jobs: list[tuple[str, nn.Parameter, int, int]] = []
        for rel_name, param, offset, nbytes in entries:
            ranges = (
                None
                if rel_name in self._tensor_path
                else self._ranges_for(f"{prefix}.{rel_name}", param)
            )
            if ranges is None:
                tensor_jobs.append((rel_name, param, offset, nbytes))
            else:
                byte_jobs.append((rel_name, param, offset, nbytes, ranges))

        futures: list[Any] = []
        if byte_jobs:
            if self._pool is None:
                self._pool = _cf.ThreadPoolExecutor(
                    max_workers=_STAGED_READ_THREADS,
                    thread_name_prefix="fpwap-staged-read",
                )

            def _read(path: str, start: int, dst_lo: int, dst_hi: int) -> None:
                os.preadv(self._fd(path), [pinned_mv[dst_lo:dst_hi]], start)

            for _, _, offset, _, ranges in byte_jobs:
                dst = offset
                for rng in ranges:
                    futures.append(
                        self._pool.submit(_read, rng.path, rng.start, dst, dst + rng.nbytes)
                    )
                    dst += rng.nbytes

        for rel_name, param, offset, nbytes in tensor_jobs:
            value = self._loader[f"{prefix}.{rel_name}"]
            if value.shape != param.shape:
                raise ValueError(
                    f"checkpoint tensor {prefix}.{rel_name} has shape "
                    f"{tuple(value.shape)}, module expects {tuple(param.shape)}"
                )
            staged = self._pinned[offset : offset + nbytes]
            staged.view(param.dtype).view(param.shape).copy_(value)

        for fut in futures:
            fut.result()

        # First use of each pattern: prove byte concatenation reproduces the
        # conversion machinery's output before trusting it for later layers.
        for rel_name, param, offset, nbytes, _ in byte_jobs:
            if rel_name in self._verified:
                continue
            staged = self._pinned[offset : offset + nbytes]
            view = staged.view(param.dtype).view(param.shape)
            reference = self._loader[f"{prefix}.{rel_name}"]
            if reference.shape == param.shape and torch.equal(
                view, reference.to(param.dtype)
            ):
                self._verified.add(rel_name)
            else:
                self._tensor_path.add(rel_name)
                if reference.shape != param.shape:
                    raise ValueError(
                        f"checkpoint tensor {prefix}.{rel_name} has shape "
                        f"{tuple(reference.shape)}, module expects "
                        f"{tuple(param.shape)}"
                    )
                view.copy_(reference)


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

    Uses posix_fadvise(DONTNEED) after a layer is unloaded so the kernel can
    reclaim those pages for upcoming layers — but only once the resident
    weight set exceeds the page budget (see ``_resident_page_budget``). Touched
    layers within the budget are kept resident so the page cache stays warm
    across sweeps and repeated / truncated multi-shard reads run compute-bound
    rather than re-reading shards from disk each sweep. On non-Linux platforms
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
        self._budget = _resident_page_budget()
        self._resident_keys: set[frozenset[str]] = set()
        self._resident_bytes = 0
        # mlock pinning: opt-in, makes residency binding instead of advisory.
        self._pin_enabled = (
            self._budget > 0
            and _LIBC is not None
            and os.environ.get("FPWAP_PAGE_RESIDENT_PIN") in ("1", "true", "True")
        )
        self._pin_fds: dict[str, int] = {}
        # mincore watchdog state (advisory-only mode)
        self._advise_calls = 0
        self._warned_evict = False
        if not _HAS_POSIX_FADVISE:
            self._budget = 0
            self._pin_enabled = False
            return
        if self._pin_enabled:
            # Can only lock as much as RLIMIT_MEMLOCK allows; clamp the resident
            # budget to it (minus headroom) so the pinned prefix fits and the
            # rest is DONTNEED'd. Unlimited rlimit leaves the budget untouched.
            cap = _pin_cap_bytes()
            if cap < self._budget:
                want_gb = self._budget / 1e9
                self._budget = max(0, cap)
                _warn_once(
                    "clamp",
                    f"fpwap: FPWAP_PAGE_RESIDENT_PIN=1 but RLIMIT_MEMLOCK only allows "
                    f"~{max(0, cap) / 1e9:.1f} GB of the requested {want_gb:.1f} GB "
                    f"resident budget; pinning the prefix that fits and DONTNEED'ing "
                    f"the rest. Raise `ulimit -l` (or grant CAP_IPC_LOCK) to pin more.",
                )
            if self._budget <= 0:
                self._pin_enabled = False

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

    def _bytes_for(self, weight_names: list[str]) -> int:
        expanded: list[str] = []
        for name in weight_names:
            expanded.extend(self._virtual_sources.get(name, (name,)))
        return sum(
            end - start
            for name in expanded
            if name in self._offsets
            for _, start, end in (self._offsets[name],)
        )

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
        # Hold touched layers resident in the page cache up to the budget so
        # repeated / multi-shard sweeps reuse warm pages instead of re-reading
        # from disk; only evict once the resident set would exceed the budget.
        #
        # Eviction is keep-prefix, not LRU: the first layers seen fill the
        # budget and stay resident for the run; layers encountered after it
        # fills are DONTNEED'd on every sweep. This is deliberate — lens-style
        # workloads sweep the *same* layer set over N shards, a cyclic access
        # pattern on which LRU is pessimal (Belady) while keep-prefix retains
        # the largest cacheable contiguous prefix. A full-depth sweep larger
        # than the budget thus stays bounded; a truncated/repeated sweep that
        # fits runs fully warm.
        #
        # With FPWAP_PAGE_RESIDENT_PIN the resident prefix is mlock'd so the
        # kernel cannot reclaim it under memory pressure — without it, "keep
        # resident" is only advisory (omit DONTNEED) and a memory-pressured box
        # silently re-faults the set from disk every sweep, which the mincore
        # watchdog below detects and warns about.
        # Tick the residency watchdog on every advise call (not just resident
        # ones) so its cadence tracks sweeps regardless of how much of the
        # model fits the budget.
        self._maybe_check_residency()
        if self._budget:
            key = frozenset(weight_names)
            if key in self._resident_keys:
                return
            layer_bytes = self._bytes_for(weight_names)
            if self._resident_bytes + layer_bytes <= self._budget:
                if not self._pin_enabled or self._pin(weight_names):
                    self._resident_keys.add(key)
                    self._resident_bytes += layer_bytes
                    return
                # Pinning was requested but failed (rlimit/OOM): don't pretend
                # the layer is resident — evict it explicitly so behavior is
                # predictable, and note the exhaustion once.
                self._note_pin_failed()
        self._advise(weight_names, os.POSIX_FADV_DONTNEED)

    def advise_willneed(self, weight_names: list[str]) -> None:
        self._advise(weight_names, os.POSIX_FADV_WILLNEED)

    # -- pinning / residency verification --------------------------------

    def _ranges_by_shard(self, weight_names: list[str]) -> dict[str, list[tuple[int, int]]]:
        """Resolve weight names (expanding virtual sources) to merged byte
        ranges per shard file."""
        expanded: list[str] = []
        for name in weight_names:
            expanded.extend(self._virtual_sources.get(name, (name,)))
        by_shard: dict[str, list[tuple[int, int]]] = {}
        for name in expanded:
            if name not in self._offsets:
                continue
            path, start, end = self._offsets[name]
            by_shard.setdefault(path, []).append((start, end))
        for path, ranges in by_shard.items():
            ranges.sort()
            merged: list[tuple[int, int]] = []
            for start, end in ranges:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            by_shard[path] = merged
        return by_shard

    def _pin_fd(self, path: str) -> int:
        fd = self._pin_fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._pin_fds[path] = fd
        return fd

    def _pin(self, weight_names: list[str]) -> bool:
        """mmap(MAP_SHARED|MAP_POPULATE) + mlock each of *weight_names*' byte
        ranges into the process-global registry so the page cache pages backing
        them cannot be reclaimed. preadv of the same ranges then always hits the
        locked pages (page cache is per-inode, shared across fds). Ranges
        already pinned by an earlier sweep are reused, not re-locked — this is
        what makes the resident set survive the per-unit Sweep teardown. Returns
        False if a range can't be locked (rlimit/OOM) so the caller falls back
        to DONTNEED; regions locked before the failure stay pinned (still
        useful)."""
        assert _LIBC is not None
        global _PINNED_TOTAL
        cap = _pin_cap_bytes()
        for path, ranges in self._ranges_by_shard(weight_names).items():
            for start, end in ranges:
                a_start = start - (start % _PAGE_SIZE)
                a_len = end - a_start
                a_len = ((a_len + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE
                key = (path, a_start, a_len)
                if key in _PINNED_REGIONS:
                    continue  # already locked by an earlier sweep — reuse
                if _PINNED_TOTAL + a_len > cap:
                    return False  # would exceed the memlock budget
                fd = os.open(path, os.O_RDONLY)
                try:
                    addr = _LIBC.mmap(
                        None, a_len, _PROT_READ, _MAP_SHARED | _MAP_POPULATE, fd, a_start
                    )
                finally:
                    os.close(fd)  # mapping + lock survive fd close
                if addr is None or addr == _MAP_FAILED:
                    return False
                if _LIBC.mlock(ctypes.c_void_p(addr), a_len) != 0:
                    _LIBC.munmap(ctypes.c_void_p(addr), a_len)
                    return False
                _PINNED_REGIONS[key] = addr
                _PINNED_TOTAL += a_len
        return True

    def _note_pin_failed(self) -> None:
        self._warned_evict = True
        _warn_once(
            "pin_failed",
            f"fpwap: page-resident pinning hit a limit at "
            f"{self._resident_bytes / 1e9:.1f} GB (RLIMIT_MEMLOCK / available RAM); "
            f"remaining layers will be DONTNEED'd and re-read each sweep. Raise "
            f"`ulimit -l` to pin the full weight set.",
        )

    def _maybe_check_residency(self) -> None:
        """Advisory-only mode: periodically verify (via mincore) that the
        resident set the advisor *believes* warm is actually in the page cache,
        and warn once if the kernel has reclaimed it (the silent I/O-bound
        regime). Pinned mode skips this — mlock guarantees residency."""
        if self._pin_enabled or self._warned_evict or _LIBC is None:
            return
        self._advise_calls += 1
        if self._advise_calls % _RESIDENCY_CHECK_EVERY or not self._resident_keys:
            return
        resident, total = 0, 0
        for key in self._resident_keys:
            for path, ranges in self._ranges_by_shard(list(key)).items():
                fd = self._pin_fd(path)
                for start, end in ranges:
                    r, n = self._mincore_resident_pages(fd, start, end)
                    resident += r
                    total += n
        if total and resident / total < _RESIDENCY_WARN_BELOW:
            self._warned_evict = True
            print(
                f"fpwap: page-resident weights re-faulting — only "
                f"{resident / total:.0%} of the {self._resident_bytes / 1e9:.1f} GB "
                f"resident set is still in the page cache; this capture is I/O-bound "
                f"(weights re-read from disk every sweep). Raise RAM headroom, reduce "
                f"the swept footprint, or set FPWAP_PAGE_RESIDENT_PIN=1 (needs "
                f"`ulimit -l`).",
                flush=True,
            )

    def _mincore_resident_pages(self, fd: int, start: int, end: int) -> tuple[int, int]:
        """(resident_pages, total_pages) for [start, end) of *fd* via mincore.
        A read-only MAP_SHARED mapping (no POPULATE) reflects current cache
        state without faulting pages in."""
        assert _LIBC is not None
        a_start = start - (start % _PAGE_SIZE)
        a_len = end - a_start
        a_len = ((a_len + _PAGE_SIZE - 1) // _PAGE_SIZE) * _PAGE_SIZE
        npages = a_len // _PAGE_SIZE
        addr = _LIBC.mmap(None, a_len, _PROT_READ, _MAP_SHARED, fd, a_start)
        if addr is None or addr == _MAP_FAILED:
            return 0, npages
        vec = ctypes.create_string_buffer(npages)
        resident = 0
        if _LIBC.mincore(ctypes.c_void_p(addr), a_len, vec) == 0:
            resident = sum(1 for b in vec.raw if b & 1)
        _LIBC.munmap(ctypes.c_void_p(addr), a_len)
        return resident, npages

    def close(self) -> None:
        # Globally pinned weight regions deliberately survive teardown so the
        # resident set stays warm across the per-unit sweeps; release them with
        # release_resident_pages(). Here we only drop this advisor's mincore fds.
        _close_fds(self._pin_fds)
        self._pin_fds = {}
