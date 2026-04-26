from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from fpwap.buffer import ResidualBuffer
from fpwap.callbacks.base import Callback
from fpwap.loader import (
    ShardPageAdvisor,
    _load_layer,
    _load_named_param,
    _unload_layer,
    build_empty_model_and_index,
)
from fpwap.models import ModelPlumbing, get_plumbing
from fpwap.preflight import PreflightReport
from fpwap.storage import StorageBackend
from fpwap.types import (
    Artifact,
    Context,
    Emit,
    HookName,
    LoadingStrategy,
    PaddingMode,
    RaggedTensor,
    ResultArtifact,
    WriteBack,
)

ProgressReporter = Callable[["ProgressEvent"], None]


@dataclass(frozen=True)
class ProgressEvent:
    """Emitted by the engine at layer and batch boundaries.

    Receivers should be cheap — they run on the hot loop. Heavy I/O (wandb flush,
    file writes) should be queued to a background thread by the receiver itself.
    """

    kind: str
    layer_idx: int
    batch_idx: int
    n_batches: int
    wall_s: float


@dataclass
class LayerTiming:
    """Per-layer wall-clock breakdown. All times in seconds, measured with
    perf_counter_ns at phase boundaries and aggregated after the run — no
    per-op synchronization on the hot path."""

    load_s: float = 0.0
    forward_s: float = 0.0
    callback_s: float = 0.0
    write_s: float = 0.0
    emit_s: float = 0.0
    bytes_weights: int = 0
    bytes_buffer: int = 0


@dataclass
class SetupTiming:
    """Breakdown of the setup phase before the main loop starts.

    Captures where wall-clock goes during model resolution, empty-model
    construction, index building, buffer allocation, and embedding.
    """

    resolve_s: float = 0.0
    config_s: float = 0.0
    model_s: float = 0.0
    index_s: float = 0.0
    total_s: float = 0.0


@dataclass
class PreloopTiming:
    """Breakdown of the pre-loop phase between run() entry and embedding.

    Captures where wall-clock goes during dataset resolution, embedding
    weight loading, buffer/segment allocation, and callback initialization.
    """

    resolve_dataset_s: float = 0.0
    ensure_embedding_s: float = 0.0
    build_segments_s: float = 0.0
    storage_start_s: float = 0.0
    callbacks_start_s: float = 0.0
    total_s: float = 0.0


@dataclass
class TeardownTiming:
    """Breakdown of the teardown phase after the main loop completes.

    Captures where wall-clock goes during streamer shutdown, callback
    finalization, residual buffer flush, and storage backend flush.
    """

    streamer_close_s: float = 0.0
    callbacks_s: float = 0.0
    buffer_flush_s: float = 0.0
    storage_flush_s: float = 0.0
    total_s: float = 0.0


@dataclass
class ProfileReport:
    """Always-on profile of an fpwap run. Target overhead: < 1% wall-clock.

    Built by the engine during .run() and attached to Result. Surface is
    designed for answering "where did the time go?" — not a debug dump.
    """

    total_wall_s: float = 0.0
    total_tokens: int = 0
    per_layer: dict[int, LayerTiming] = field(default_factory=dict)
    setup: SetupTiming | None = None
    preloop: PreloopTiming | None = None
    embed_s: float = 0.0
    embed_sync_s: float = 0.0
    loop_setup_s: float = 0.0
    loop_s: float = 0.0
    drain_sync_s: float = 0.0
    emit_drain_s: float = 0.0
    unload_s: float = 0.0
    teardown: TeardownTiming | None = None

    @property
    def preloop_s(self) -> float:
        if self.preloop is None:
            return 0.0
        return self.preloop.total_s

    @property
    def teardown_s(self) -> float:
        if self.teardown is None:
            return 0.0
        return self.teardown.total_s

    def throughput_tok_per_s(self) -> float:
        """End-to-end tokens per second — total tokens (N * seq_len) divided
        by wall-clock from .run() entry to return. This is the throughput
        definition the SPEC and README headline targets are stated against.
        """
        if self.total_wall_s <= 0.0:
            return 0.0
        return self.total_tokens / self.total_wall_s

    def weight_bandwidth_gb_per_s(self) -> float:
        """Aggregate weight I/O divided by total wall-clock. On the streaming
        path this is the effective NVMe+PCIe throughput the engine achieved;
        on the preloaded path it's 0."""
        if self.total_wall_s <= 0.0:
            return 0.0
        total_bytes = sum(t.bytes_weights for t in self.per_layer.values())
        return total_bytes / self.total_wall_s / 1e9

    def summary(self) -> str:
        tps = self.throughput_tok_per_s()
        hdr = (
            f"total wall {self.total_wall_s:.3f}s  "
            f"tokens {self.total_tokens}  "
            f"throughput {tps:,.1f} tok/s  "
            f"across {len(self.per_layer)} layers"
        )
        lines = [hdr]
        if self.setup is not None:
            s = self.setup
            lines.append(
                f"  setup {s.total_s:.3f}s  "
                f"(resolve {s.resolve_s:.3f}s  config {s.config_s:.3f}s  "
                f"model {s.model_s:.3f}s  index {s.index_s:.3f}s)"
            )
        if self.preloop is not None and self.preloop.total_s > 0:
            pl = self.preloop
            pl_parts: list[str] = []
            for name, val in [
                ("resolve_dataset", pl.resolve_dataset_s),
                ("ensure_embedding", pl.ensure_embedding_s),
                ("build_segments", pl.build_segments_s),
                ("storage_start", pl.storage_start_s),
                ("callbacks_start", pl.callbacks_start_s),
            ]:
                if val > 0:
                    pl_parts.append(f"{name} {val:.3f}s")
            pl_detail = f"  ({', '.join(pl_parts)})" if pl_parts else ""
            lines.append(f"  preloop {pl.total_s:.3f}s{pl_detail}")
        if self.embed_s > 0:
            embed_parts: list[str] = []
            if self.embed_sync_s > 0:
                embed_parts.append(f"sync {self.embed_sync_s:.3f}s")
            embed_detail = f"  (includes {', '.join(embed_parts)})" if embed_parts else ""
            lines.append(f"  embed {self.embed_s:.3f}s{embed_detail}")
        if self.loop_setup_s > 0:
            lines.append(f"  loop_setup {self.loop_setup_s:.3f}s")
        if self.loop_s > 0:
            loop_parts: list[str] = []
            if self.drain_sync_s > 0:
                loop_parts.append(f"drain_sync {self.drain_sync_s:.3f}s")
            if self.emit_drain_s > 0:
                loop_parts.append(f"emit_drain {self.emit_drain_s:.3f}s")
            if self.unload_s > 0:
                loop_parts.append(f"unload {self.unload_s:.3f}s")
            loop_detail = f"  (includes {', '.join(loop_parts)})" if loop_parts else ""
            lines.append(f"  loop  {self.loop_s:.3f}s{loop_detail}")
        if self.teardown is not None and self.teardown.total_s > 0:
            td = self.teardown
            parts: list[str] = []
            for name, val in [
                ("streamer_close", td.streamer_close_s),
                ("callbacks", td.callbacks_s),
                ("buffer_flush", td.buffer_flush_s),
                ("storage_flush", td.storage_flush_s),
            ]:
                if val > 0:
                    parts.append(f"{name} {val:.3f}s")
            detail = f"  ({', '.join(parts)})" if parts else ""
            lines.append(f"  teardown {td.total_s:.3f}s{detail}")
        for i, t in sorted(self.per_layer.items()):
            emit_part = f"  emit {t.emit_s:.3f}s" if t.emit_s > 0 else ""
            lines.append(
                f"  layer {i}: load {t.load_s:.3f}s  fwd {t.forward_s:.3f}s  "
                f"cb {t.callback_s:.3f}s  write {t.write_s:.3f}s{emit_part}"
            )
        return "\n".join(lines)

    def by_phase(self) -> dict[str, list[float]]:
        phases: dict[str, list[float]] = {
            "load": [],
            "forward": [],
            "callback": [],
            "write": [],
            "emit": [],
        }
        for _, t in sorted(self.per_layer.items()):
            phases["load"].append(t.load_s)
            phases["forward"].append(t.forward_s)
            phases["callback"].append(t.callback_s)
            phases["write"].append(t.write_s)
            phases["emit"].append(t.emit_s)
        return phases

    def slowest_layer(self) -> tuple[int, str]:
        if not self.per_layer:
            return (-1, "none")
        phase_names = ("load_s", "forward_s", "callback_s", "write_s", "emit_s")
        worst_i = -1
        worst_phase = "none"
        worst_s = -1.0
        for i, t in self.per_layer.items():
            for name in phase_names:
                v = getattr(t, name)
                if v > worst_s:
                    worst_s = v
                    worst_i = i
                    worst_phase = name.removesuffix("_s")
        return (worst_i, worst_phase)

    def bytes_moved(self) -> dict[str, int]:
        w = sum(t.bytes_weights for t in self.per_layer.values())
        b = sum(t.bytes_buffer for t in self.per_layer.values())
        return {"weights": w, "buffer": b}


@dataclass
class Result:
    sweep_id: str
    artifacts: dict[tuple[str, int], Artifact] = field(default_factory=dict)
    storage: StorageBackend | None = None
    profile: ProfileReport = field(default_factory=ProfileReport)
    seq_len: int | None = None
    _left_padded: bool = True
    # When no StorageBackend is set, Emit outputs are collected in memory per
    # (layer, hook). Fine for moderate-size sweeps; for dataset-scale
    # extraction, swap in a storage backend to avoid RAM pressure.
    # Each entry is (sample_ids, tensor, sample_lengths_or_None). When the
    # last element is non-None, the microbatch was a ragged emit (#65).
    _emits: dict[
        tuple[int, str], list[tuple[Tensor, Tensor, Tensor | None]]
    ] = field(default_factory=dict)

    def artifact(self, kind: str, layer: int) -> Artifact:
        return self.artifacts[(kind, layer)]

    def activations(
        self,
        layer: int,
        hook: HookName,
        *,
        as_path: bool | str | Path = False,
    ) -> Tensor | RaggedTensor | ResultArtifact:
        """Concatenate emitted activations for (layer, hook) in sample-id order.

        Requires at least one read-phase callback (e.g. RawActivations) to have
        emitted for this (layer, hook). With a StorageBackend wired, reads the
        full corpus back from disk (the backend owns shape/dtype); otherwise
        concatenates the in-memory microbatch emits.

        Return type is a union: a dense `Tensor` for ordinary emits, or a
        `RaggedTensor(flat, offsets)` when the callback supplied
        `Emit.sample_lengths`. Callers that may receive either layout must
        dispatch on `isinstance(result, RaggedTensor)`.

        Args:
            as_path: If False (default), return the materialized tensor.
                If True, return a `ResultArtifact` pointing at the backend's
                in-place file (caller mmap-reads on demand — issue #70).
                If a Path/str, hardlink (or copy on EXDEV) the data file
                and sidecar into that directory and return a `ResultArtifact`
                pointing there. Backend keeps ownership of its original.
                Requires a StorageBackend; raises otherwise.
        """
        # Catch both `as_path=""` (falsy — would silently fall through to
        # the in-memory branch and confuse the caller) and `as_path=Path("")`
        # (truthy — would resolve to cwd and hardlink into the working dir).
        # Path("").parts is () — distinguishes the empty Path from explicit
        # Path(".") which a caller might genuinely intend.
        if (
            (isinstance(as_path, str) and not as_path)
            or (isinstance(as_path, Path) and not as_path.parts)
        ):
            raise ValueError(
                f"as_path string/Path must be non-empty (got {as_path!r}); "
                "pass True for an in-place handle or an actual path"
            )
        if as_path:
            if self.storage is None:
                raise RuntimeError(
                    "activations(as_path=...) requires a StorageBackend on "
                    "the Sweep; no backend is wired so there is no on-disk "
                    "file to hand out"
                )
            dest = Path(as_path) if not isinstance(as_path, bool) else None
            return self.storage.path_for(layer, hook, dest=dest)
        if self.storage is not None:
            return self.storage.read_all(layer, hook)
        key = (layer, hook)
        if key not in self._emits or not self._emits[key]:
            raise KeyError(
                f"no activations collected for layer={layer}, hook={hook!r}; "
                f"add a read-phase callback that returns Emit (e.g. RawActivations)"
            )
        parts = self._emits[key]
        ragged = any(p[2] is not None for p in parts)
        if ragged:
            if not all(p[2] is not None for p in parts):
                raise RuntimeError(
                    f"layer={layer} hook={hook!r}: mixed dense + ragged emits"
                )
            # Concatenate microbatches in arrival order.
            ids_concat = torch.cat([p[0] for p in parts], dim=0)
            lengths_concat = torch.cat(
                [p[2] for p in parts if p[2] is not None], dim=0
            ).to(torch.int64)
            flat_concat = torch.cat([p[1] for p in parts], dim=0)
            n = int(ids_concat.shape[0])
            total = int(lengths_concat.sum().item())

            # Reorder samples to true sample-id order across microbatches.
            order = torch.argsort(ids_concat, stable=True)
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(n, dtype=order.dtype)

            sorted_lengths = lengths_concat[order]
            offsets = torch.zeros(n + 1, dtype=torch.int64)
            offsets[1:] = torch.cumsum(sorted_lengths, dim=0)

            in_offsets = torch.zeros(n + 1, dtype=torch.int64)
            in_offsets[1:] = torch.cumsum(lengths_concat, dim=0)

            # Vectorized scatter: build dst_for_row[total] in one pass.
            # sample_of_row[r] = which input-order sample owns input row r.
            sample_of_row = torch.repeat_interleave(
                torch.arange(n, dtype=torch.int64), lengths_concat
            )
            dst_starts_per_sample = offsets[inv_order]  # [n]
            src_starts_per_sample = in_offsets[:n]  # [n]
            row_idx = torch.arange(total, dtype=torch.int64)
            dst_for_row = (
                dst_starts_per_sample[sample_of_row]
                + (row_idx - src_starts_per_sample[sample_of_row])
            )

            out_flat = torch.empty(
                (total, *flat_concat.shape[1:]), dtype=flat_concat.dtype
            )
            out_flat[dst_for_row] = flat_concat
            return RaggedTensor(flat=out_flat, offsets=offsets)
        ordered = sorted(parts, key=lambda p: int(p[0][0]))
        tensors = [t for _, t, _ in ordered]
        if tensors and tensors[0].dim() >= 2:
            seq_dims = {t.shape[1] for t in tensors}
            if len(seq_dims) > 1:
                target = self.seq_len if self.seq_len is not None else max(seq_dims)
                padded = []
                for t in tensors:
                    gap = target - t.shape[1]
                    if gap > 0:
                        pad_shape = (t.shape[0], gap, *t.shape[2:])
                        z = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                        t = torch.cat([z, t] if self._left_padded else [t, z], dim=1)
                    padded.append(t)
                tensors = padded
        return torch.cat(tensors, dim=0)


def _resolve_dataset(dataset: Iterable[Any]) -> list[Any]:
    return list(dataset)


class _LayerStreamer:
    """Abstracts per-layer weight movement for the engine loop.

    Two impls: `_PreloadedStreamer` (no-op, for pre-loaded nn.Module inputs)
    and `_OffloadStreamer` (manual load/unload via OffloadedWeightsLoader,
    for string model IDs). The engine hot loop is uniform across both.

    Optional `prefetch_load(model, layer_idx, plumbing)` schedules the load
    of an upcoming layer on a worker thread, so safetensors read + H2D for
    layer L+1 can overlap with compute for layer L. Streamers that don't
    support prefetch return None; the engine falls back to sync load.
    """

    execution_device: torch.device | None = None
    last_load_bytes: int = 0

    def ensure_embedding_loaded(self, model: nn.Module, plumbing: ModelPlumbing) -> None:
        return None

    def load_layer(
        self, model: nn.Module, layer_idx: int, plumbing: ModelPlumbing
    ) -> None:
        return None

    def unload_layer(
        self, model: nn.Module, layer_idx: int, plumbing: ModelPlumbing
    ) -> None:
        return None

    def prefetch_load(
        self, model: nn.Module, layer_idx: int, plumbing: ModelPlumbing
    ) -> Any | None:
        """Return a handle (future-like) that the engine can wait on, or
        None if this streamer doesn't do prefetch."""
        return None

    def prefetch_chunk(
        self, model: nn.Module, layer_indices: range, plumbing: ModelPlumbing
    ) -> Any | None:
        """Prefetch all layers in a chunk. Returns a future or None."""
        return None

    def close(self) -> None:
        return None


class _PreloadedStreamer(_LayerStreamer):
    def __init__(self, execution_device: torch.device | None) -> None:
        self.execution_device = execution_device
        self.last_load_bytes = 0


class _OffloadStreamer(_LayerStreamer):
    """Manual streaming via accelerate's OffloadedWeightsLoader.

    Bypasses AlignDevicesHook entirely — that's the per-forward streaming
    pattern fpwap exists to avoid (SPEC §12.4 Approach A).
    """

    execution_device: torch.device  # non-None; overrides base's Optional

    def __init__(self, accel_index: dict[str, Any], execution_device: torch.device) -> None:
        from accelerate.utils import OffloadedWeightsLoader

        self.execution_device = execution_device
        self._loader = OffloadedWeightsLoader(index=accel_index)
        self._accel_index = accel_index
        self.last_load_bytes = 0
        self._advisor = ShardPageAdvisor(accel_index)
        # Single-worker pool: next layer's load runs on a worker thread so
        # safetensors read + H2D overlap with the main thread's compute on
        # the current layer. Modern GPUs have a separate copy engine, so
        # worker-stream H2D and main-stream compute progress concurrently.
        import concurrent.futures as _cf
        import weakref

        self._prefetch_pool: _cf.ThreadPoolExecutor | None = _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fpwap-prefetch"
        )
        # Safety net: if Sweep.run() raises before reaching the explicit
        # close() at the end, the finalizer shuts the pool down on GC so a
        # caller looping Sweep constructions doesn't leak worker threads.
        weakref.finalize(self, _cf.ThreadPoolExecutor.shutdown, self._prefetch_pool, True)

    def ensure_embedding_loaded(self, model: nn.Module, plumbing: ModelPlumbing) -> None:
        device = self.execution_device
        for name in plumbing.embedding_param_names(model):
            _load_named_param(model, name, self._loader, device)

    def load_layer(
        self, model: nn.Module, layer_idx: int, plumbing: ModelPlumbing
    ) -> None:
        _load_layer(model, layer_idx, plumbing, self._loader, self.execution_device)
        layer = plumbing.layer_modules(model)[layer_idx]
        bytes_loaded = 0
        for _, p in layer.named_parameters():
            bytes_loaded += p.element_size() * p.numel()
        self.last_load_bytes = bytes_loaded

    def unload_layer(
        self, model: nn.Module, layer_idx: int, plumbing: ModelPlumbing
    ) -> None:
        prefix = plumbing.layer_prefix(layer_idx)
        layer = plumbing.layer_modules(model)[layer_idx]
        weight_names = [
            f"{prefix}.{rel}" for rel, _ in layer.named_parameters()
        ]
        _unload_layer(model, layer_idx, plumbing)
        self._advisor.advise_dontneed(weight_names)

    def prefetch_load(
        self, model: nn.Module, layer_idx: int, plumbing: ModelPlumbing
    ) -> Any | None:
        """Schedule layer_idx's load on the prefetch worker. Returns a future."""
        if self._prefetch_pool is None:
            return None
        return self._prefetch_pool.submit(
            self.load_layer, model, layer_idx, plumbing
        )

    def prefetch_chunk(
        self, model: nn.Module, layer_indices: range, plumbing: ModelPlumbing
    ) -> Any | None:
        if self._prefetch_pool is None:
            return None

        def _load_all() -> None:
            for li in layer_indices:
                self.load_layer(model, li, plumbing)

        return self._prefetch_pool.submit(_load_all)

    def close(self) -> None:
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=True)
            self._prefetch_pool = None


def _make_chunks(n_layers: int, chunk_size: int) -> list[range]:
    """Partition [0, n_layers) into consecutive chunks of at most chunk_size."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    chunks: list[range] = []
    for start in range(0, n_layers, chunk_size):
        end = min(start + chunk_size, n_layers)
        chunks.append(range(start, end))
    return chunks


def _callback_applies(cb: Callback, layer_idx: int, hook: HookName) -> bool:
    if cb.target_layers != "all" and layer_idx not in cb.target_layers:
        return False
    return hook in cb.target_hooks


def _max_capture_layer(callbacks: Sequence[Callback], n_layers: int) -> int:
    """Deepest layer any callback needs.  Returns n_layers - 1 if any targets 'all'."""
    if not callbacks:
        return n_layers - 1
    deepest = -1
    for cb in callbacks:
        if cb.target_layers == "all":
            return n_layers - 1
        for layer in cb.target_layers:
            if layer > deepest:
                deepest = layer
    return deepest if deepest >= 0 else n_layers - 1


def _packed_real_input_ids(items: list[Any]) -> Tensor:
    """Concatenate per-item real-token ids into a flat [sum(L_i)] vector.

    Each item is `{"input_ids": [1, max_seq], "attention_mask": [1, max_seq]}`.
    Pad positions (mask == 0) are dropped. Used to feed the embed pass when
    `Sweep(pack=True)`.
    """
    chunks: list[Tensor] = []
    for it in items:
        ids = it["input_ids"].squeeze(0)
        mask = it["attention_mask"].squeeze(0).to(torch.bool)
        chunks.append(ids[mask])
    return torch.cat(chunks, dim=0)


def _stack_field(items: list[Any], key: str) -> Tensor:
    """Collate a slice of dataset items along `key` into a single `(mb, seq)` tensor."""
    parts: list[Tensor] = []
    for item in items:
        t = item[key]
        if t.dim() == 1:
            t = t.unsqueeze(0)
        parts.append(t)
    return torch.cat(parts, dim=0)


@dataclass
class _Segment:
    """A group of items with the same padded seq_len for the engine loop.

    When `is_packed=True`, the segment holds a single packed buffer covering
    all items (no padding). `cu_seqlens` is `[N+1]` int64 with sample i's
    rows at `flat[cu_seqlens[i]:cu_seqlens[i+1]]`. `seq_len` then represents
    the maximum sample length (for shapes that need a scalar bound) and
    `mask_buffer` is unused.

    The `packed_*` tensors are flat segment-wide invariants computed once
    at segment-build time and sliced per microbatch. Avoids rebuilding ids /
    position_ids / doc_ids on every embed pass and every layer.
    """

    seq_len: int
    items: list[Any]
    orig_indices: list[int]
    buffer: ResidualBuffer
    mask_buffer: Tensor | None
    mb_size: int
    is_packed: bool = False
    cu_seqlens: Tensor | None = None
    packed_input_ids: Tensor | None = None
    packed_pos_ids: Tensor | None = None
    packed_doc_ids: Tensor | None = None

    @property
    def n_samples(self) -> int:
        return len(self.items)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _detect_left_padding(items: list[Any]) -> bool:
    for item in items:
        mask = item["attention_mask"]
        if isinstance(mask, Tensor):
            if mask.dim() == 2:
                mask = mask.squeeze(0)
            real_len = int(mask.sum().item())
            if real_len < mask.shape[0]:
                return int(mask[0].item()) == 0
    return True


def _trim_to_length(item: dict[str, Any], target_len: int, left_padded: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, val in item.items():
        if key not in ("input_ids", "attention_mask") or not isinstance(val, Tensor):
            result[key] = val
            continue
        t = val
        was_2d = t.dim() == 2
        if was_2d:
            t = t.squeeze(0)
        cur = t.shape[0]
        if cur > target_len:
            t = t[-target_len:] if left_padded else t[:target_len]
        elif cur < target_len:
            pad = torch.zeros(target_len - cur, dtype=t.dtype, device=t.device)
            t = torch.cat([pad, t]) if left_padded else torch.cat([t, pad])
        if was_2d:
            t = t.unsqueeze(0)
        result[key] = t
    return result


def _build_bucketed_segments(
    items: list[Any],
    max_seq_len: int,
    hidden: int,
    transport_dtype: torch.dtype,
    buf_device: torch.device,
    mb_size_override: int | None,
    config: Any,
    exec_device: torch.device,
    buffer_path: Path | None = None,
) -> list[_Segment]:
    real_lengths: list[int] = []
    for item in items:
        mask = item["attention_mask"]
        if isinstance(mask, Tensor):
            if mask.dim() == 2:
                mask = mask.squeeze(0)
            real_lengths.append(int(mask.sum().item()))
        else:
            real_lengths.append(max_seq_len)

    left_padded = _detect_left_padding(items)

    bucket_assign: dict[int, list[tuple[int, Any]]] = {}
    for orig_idx, (item, rl) in enumerate(zip(items, real_lengths, strict=True)):
        bseq = min(_next_power_of_2(max(rl, 16)), max_seq_len)
        bucket_assign.setdefault(bseq, []).append((orig_idx, item))

    segments: list[_Segment] = []
    for bseq in sorted(bucket_assign.keys()):
        entries = bucket_assign[bseq]
        orig_indices = [e[0] for e in entries]
        trimmed_items = [_trim_to_length(e[1], bseq, left_padded) for e in entries]
        n = len(trimmed_items)

        seg_path: Path | None = None
        if buffer_path is not None:
            seg_path = buffer_path.with_stem(f"{buffer_path.stem}_seq{bseq}")
        buf = ResidualBuffer(
            n_samples=n,
            seq_len=bseq,
            hidden=hidden,
            dtype=transport_dtype,
            device=buf_device,
            path=seg_path,
        )
        pin = buf_device.type == "cpu" and torch.cuda.is_available()
        mask_buf = torch.zeros(
            (n, bseq), dtype=torch.int64, device=buf_device, pin_memory=pin,
        )

        if mb_size_override is not None:
            seg_mb = mb_size_override
        else:
            if isinstance(buf_device, torch.device):
                buf_on_gpu = buf_device.type == "cuda"
            else:
                buf_on_gpu = "cuda" in str(buf_device)
            seg_mb = estimate_max_microbatch(
                config=config,
                n_samples=n,
                seq_len=bseq,
                transport_dtype=transport_dtype,
                exec_device=exec_device,
                buf_on_gpu=buf_on_gpu,
            )

        segments.append(_Segment(
            seq_len=bseq,
            items=trimmed_items,
            orig_indices=orig_indices,
            buffer=buf,
            mask_buffer=mask_buf,
            mb_size=seg_mb,
        ))

    return segments


def _run_naive_baseline(
    model: nn.Module,
    plumbing: ModelPlumbing,
    items: list[Any],
    exec_device: torch.device,
    mb_size: int,
) -> dict[int, Tensor]:
    """One naive forward pass with per-block forward hooks capturing residual_post.

    Used by `verify=True`. Output is a dict `layer_idx -> [N, seq, H]` CPU tensor
    concatenated across microbatches. Microbatches match fpwap's so bf16
    reductions are deterministic (see bf16_microbatch_determinism memo).
    """
    layer_modules = plumbing.layer_modules(model)
    captured: dict[int, list[Tensor]] = {i: [] for i in range(len(layer_modules))}

    def make_hook(i: int) -> Any:
        def _h(_m: nn.Module, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            captured[i].append(t.detach().to("cpu", copy=True))

        return _h

    handles = [
        b.register_forward_hook(make_hook(i)) for i, b in enumerate(layer_modules)
    ]
    try:
        has_mask = _has_attention_mask(items)
        with torch.no_grad():
            for start in range(0, len(items), mb_size):
                stop = min(start + mb_size, len(items))
                kwargs: dict[str, Tensor] = {
                    "input_ids": _stack_field(items[start:stop], "input_ids").to(
                        exec_device
                    )
                }
                if has_mask:
                    kwargs["attention_mask"] = _stack_field(
                        items[start:stop], "attention_mask"
                    ).to(exec_device)
                model(**kwargs)
    finally:
        for h in handles:
            h.remove()
    return {i: torch.cat(parts, dim=0) for i, parts in captured.items() if parts}


def _has_attention_mask(items: list[Any]) -> bool:
    """All-or-nothing contract: every item carries an attention_mask, or none do.

    Mixed datasets silently drop the mask on items that don't declare one, which
    would make padded-batch correctness a subtle footgun.
    """
    present = ["attention_mask" in item for item in items]
    if all(present):
        return True
    if any(present):
        raise ValueError(
            "dataset items are inconsistent: some have attention_mask and others "
            "don't. Either include attention_mask on every item or omit it from "
            "all of them."
        )
    return False


def estimate_max_microbatch(
    config: Any,
    n_samples: int,
    seq_len: int,
    transport_dtype: torch.dtype,
    exec_device: torch.device,
    buf_on_gpu: bool = True,
    safety_factor: float = 0.85,
) -> int:
    """Estimate the largest microbatch size that fits in VRAM.

    Uses model config attributes and total device memory to compute a
    conservative upper bound.  Returns *n_samples* on CPU or when config
    is missing required attributes.
    """
    if exec_device.type != "cuda":
        return n_samples

    hidden = int(getattr(config, "hidden_size", 0))
    if hidden == 0:
        return n_samples
    intermediate = int(getattr(config, "intermediate_size", None) or hidden * 4)
    n_heads = int(getattr(config, "num_attention_heads", None) or hidden // 128)
    vocab = int(getattr(config, "vocab_size", 32000))

    dtype_bytes = torch.empty((), dtype=transport_dtype).element_size()

    buf_bytes = n_samples * seq_len * hidden * dtype_bytes if buf_on_gpu else 0
    layer_bytes = (4 * hidden * hidden + 3 * hidden * intermediate) * dtype_bytes
    embed_bytes = vocab * hidden * dtype_bytes
    # Two layers may coexist on GPU during prefetch overlap.
    reserved = buf_bytes + 2 * layer_bytes + embed_bytes

    total_vram = torch.cuda.get_device_properties(exec_device).total_memory
    available = max(0, int(total_vram * safety_factor) - reserved)

    # Per-sample peak activation: MLP gate+up alive simultaneously + residual.
    per_sample = seq_len * dtype_bytes * (hidden + 3 * intermediate)
    per_sample += n_heads * seq_len * seq_len * dtype_bytes

    if per_sample <= 0:
        return n_samples

    max_mb = max(1, available // per_sample)
    if max_mb >= 2:
        max_mb = 1 << (max_mb.bit_length() - 1)
    return min(max_mb, n_samples)


class Sweep:
    """The engine. Inverts the inference loop: for each layer, run the dataset.

    Construction is cheap; call .preflight() to plan, .run() to execute.
    """

    def __init__(
        self,
        model: str | Any,
        dataset: Iterable[Any],
        seq_len: int,
        callbacks: Sequence[Callback],
        storage: StorageBackend | None = None,
        transport_dtype: torch.dtype = torch.bfloat16,
        loading_strategy: LoadingStrategy | None = None,
        verify: bool = False,
        progress: bool | ProgressReporter = True,
        seed: int = 0,
        microbatch_size: int | Literal["auto"] | None = None,
        snapshot_dir: str | None = None,
        offload_dir: str | None = None,
        execution_device: torch.device | str | None = None,
        buffer_device: torch.device | str | None = None,
        buffer_path: Any | None = None,
        apply_final_norm: bool = True,
        padding: PaddingMode = "fixed",
        chunk_size: int = 1,
        pack: bool = False,
        _accel_index: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.seq_len = seq_len
        self.callbacks = callbacks
        self.storage = storage
        self.transport_dtype = transport_dtype
        self.loading_strategy = loading_strategy
        self.verify = verify
        self.progress = progress
        self.seed = seed
        self.microbatch_size: int | Literal["auto"] | None = microbatch_size
        self.snapshot_dir = snapshot_dir
        self.offload_dir = offload_dir
        self.apply_final_norm = apply_final_norm
        self.padding: PaddingMode = padding
        self.chunk_size = chunk_size
        self.pack = pack
        self._accel_index = _accel_index
        self.execution_device = (
            torch.device(execution_device) if execution_device is not None else None
        )
        self.buffer_device = (
            torch.device(buffer_device) if buffer_device is not None else None
        )
        self.buffer_path = Path(buffer_path) if buffer_path is not None else None

    def preflight(self) -> PreflightReport:
        """Feasibility gate + cost-model prediction.

        Runs the embed pass + a single-layer dry-run, measures weight-load
        and forward times separately, then feeds them into the cost model to
        predict throughput and recommend prefetch on/off.
        """
        from fpwap.cost_model import CostModelInput, recommend
        from fpwap.preflight import PreflightReport as _Report

        items = _resolve_dataset(self.dataset)
        n_samples = len(items)
        if n_samples == 0:
            return _Report(
                feasible=False,
                microbatch_size=0,
                residual_buffer_gb=0.0,
                per_layer_peak_vram_gb=0.0,
                estimated_wall_clock_s=0.0,
                estimated_weight_io_gb=0.0,
                loading_strategy="cpu_offload",
                blockers=["dataset is empty"],
            )

        model, streamer, _ = self._resolve_model_and_streamer()
        model.eval()
        plumbing = get_plumbing(model)
        exec_device = streamer.execution_device or items[0]["input_ids"].device
        buf_device = self.buffer_device or exec_device
        is_preloaded = isinstance(streamer, _PreloadedStreamer)

        config = getattr(model, "config", None)
        hidden_attr = getattr(config, "hidden_size", None) if config is not None else None
        hidden = int(hidden_attr) if hidden_attr is not None else 0
        n_layers = len(plumbing.layer_modules(model))
        if self.microbatch_size == "auto":
            if isinstance(buf_device, torch.device):
                buf_on_gpu = buf_device.type == "cuda"
            else:
                buf_on_gpu = "cuda" in str(buf_device)
            if isinstance(exec_device, torch.device):
                dev = exec_device
            else:
                dev = torch.device(exec_device)
            mb_size = estimate_max_microbatch(
                config=config,
                n_samples=n_samples,
                seq_len=self.seq_len,
                transport_dtype=self.transport_dtype,
                exec_device=dev,
                buf_on_gpu=buf_on_gpu,
            )
        elif self.microbatch_size is not None:
            mb_size = self.microbatch_size
        else:
            mb_size = min(n_samples, 8)

        element_bytes = (
            torch.zeros((), dtype=self.transport_dtype).element_size() if hidden else 2
        )
        residual_gb = n_samples * self.seq_len * hidden * element_bytes / 1e9

        # --- Probe: measure embed, weight-load, and forward separately ---
        streamer.ensure_embedding_loaded(model, plumbing)

        input_ids = _stack_field(items[:mb_size], "input_ids").to(exec_device)

        # Warmup: untimed forward through embed + layer 0 to warm JIT/page cache
        with torch.no_grad():
            _warmup_hs = plumbing.embed(model, input_ids)
            _ = plumbing.layer_forward_with_hooks(
                model, plumbing.layer_modules(model)[0], _warmup_hs
            )
        del _warmup_hs
        if exec_device.type == "cuda":
            torch.cuda.synchronize()

        # Embed timing
        if exec_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            hidden_states = plumbing.embed(model, input_ids)
        if exec_device.type == "cuda":
            torch.cuda.synchronize()
        embed_s = (time.perf_counter_ns() - t0) / 1e9

        # Weight-load timing (unload first so the load is realistic)
        streamer.unload_layer(model, 0, plumbing)
        if exec_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        streamer.load_layer(model, 0, plumbing)
        if exec_device.type == "cuda":
            torch.cuda.synchronize()
        weight_load_s = (time.perf_counter_ns() - t0) / 1e9

        # Forward timing (single microbatch)
        if exec_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            _ = plumbing.layer_forward_with_hooks(
                model, plumbing.layer_modules(model)[0], hidden_states
            )
        if exec_device.type == "cuda":
            torch.cuda.synchronize()
        fwd_per_microbatch_s = (time.perf_counter_ns() - t0) / 1e9

        layer_weight_bytes = streamer.last_load_bytes
        streamer.unload_layer(model, 0, plumbing)
        del input_ids, hidden_states

        # --- Cost model ---
        n_microbatches = (n_samples + mb_size - 1) // mb_size
        # Scale embed_s from probe microbatch to full dataset
        embed_total_s = embed_s * n_microbatches

        inp = CostModelInput(
            n_layers=n_layers,
            n_samples=n_samples,
            seq_len=self.seq_len,
            microbatch_size=mb_size,
            weight_load_s=weight_load_s,
            fwd_per_microbatch_s=fwd_per_microbatch_s,
            embed_s=embed_total_s,
            layer_weight_bytes=layer_weight_bytes,
        )

        candidates: list[tuple[CostModelInput, bool]] = [(inp, False)]
        if not is_preloaded:
            candidates.append((inp, True))

        rec = recommend(candidates)

        return _Report(
            feasible=True,
            microbatch_size=mb_size,
            residual_buffer_gb=residual_gb,
            per_layer_peak_vram_gb=0.0,
            estimated_wall_clock_s=rec.prediction.total_wall_s,
            estimated_weight_io_gb=rec.prediction.weight_io_gb,
            loading_strategy="cpu_offload",
            prediction=rec.prediction,
            recommended_prefetch=rec.prefetch,
            recommended_buffer_device="cpu" if buf_device != exec_device else "cuda",
        )

    def _resolve_model_and_streamer(
        self,
    ) -> tuple[nn.Module, _LayerStreamer, SetupTiming | None]:
        """Turn `self.model` into (concrete nn.Module, streamer, setup_timing).

        Pre-loaded nn.Module → _PreloadedStreamer (no-op per-layer).
        Pre-loaded nn.Module + _accel_index → _OffloadStreamer (Extractor reuse).
        String snapshot path → build empty-weights model + _OffloadStreamer
        backed by OffloadedWeightsLoader.
        """
        if isinstance(self.model, nn.Module):
            if self._accel_index is not None:
                if self.execution_device is None:
                    raise ValueError(
                        "execution_device is required when using a pre-built accel_index "
                        "(Extractor path)"
                    )
                return self.model, _OffloadStreamer(self._accel_index, self.execution_device), None
            return self.model, _PreloadedStreamer(self.execution_device), None
        if isinstance(self.model, str):
            if self.execution_device is None:
                raise ValueError(
                    "execution_device is required when model is a string ID"
                )
            from fpwap.loader import resolve_snapshot_dir

            t0_resolve = time.perf_counter_ns()
            if self.snapshot_dir is not None:
                snapshot_dir = Path(self.snapshot_dir)
            else:
                snapshot_dir = resolve_snapshot_dir(self.model)
            resolve_s = (time.perf_counter_ns() - t0_resolve) / 1e9

            model, accel_index, build_timing = build_empty_model_and_index(
                model_id=self.model,
                snapshot_dir=snapshot_dir,
                dtype=self.transport_dtype,
            )
            setup = SetupTiming(
                resolve_s=resolve_s,
                config_s=build_timing["config_s"],
                model_s=build_timing["model_s"],
                index_s=build_timing["index_s"],
                total_s=resolve_s + sum(build_timing.values()),
            )
            streamer = _OffloadStreamer(accel_index, self.execution_device)
            return model, streamer, setup
        got = type(self.model).__name__
        raise TypeError(
            f"model must be str (model ID / snapshot path) or nn.Module, got {got}"
        )

    def run(self) -> Result:
        if self.verify and not isinstance(self.model, nn.Module):
            raise NotImplementedError(
                "verify=True requires a pre-loaded nn.Module; streaming verify "
                "would require running cpu_offload in parallel and isn't wired. "
                "For streaming correctness, see tests/gpu/test_streaming_bit_exact.py."
            )
        if self.verify and self.padding == "bucketed":
            raise NotImplementedError(
                "verify=True is not supported with padding='bucketed'. "
                "Bucketed padding uses per-bucket seq_len which changes the "
                "naive baseline shape. Use the dedicated bucketed correctness "
                "tests instead."
            )
        t0_run = time.perf_counter_ns()

        # Set up progress reporter early so pre-loop events are visible.
        progress_reporter: ProgressReporter | None = None
        if callable(self.progress):
            progress_reporter = self.progress

        def _emit_progress(
            kind: str, layer_idx: int, batch_idx: int, n_batches: int
        ) -> None:
            if progress_reporter is None:
                return
            progress_reporter(
                ProgressEvent(
                    kind=kind,
                    layer_idx=layer_idx,
                    batch_idx=batch_idx,
                    n_batches=n_batches,
                    wall_s=(time.perf_counter_ns() - t0_run) / 1e9,
                )
            )

        # --- Pre-loop: timed per sub-phase ---
        pl = PreloopTiming()
        t0_preloop = time.perf_counter_ns()

        model, streamer, setup_timing = self._resolve_model_and_streamer()
        model.eval()
        plumbing = get_plumbing(model)

        if self.pack:
            if not getattr(plumbing, "supports_packed", False):
                raise RuntimeError(
                    f"pack=True requested but {type(plumbing).__name__} does "
                    f"not support packed forward "
                    f"(supports_packed=False). Use pack=False or run with a "
                    f"Llama-family model."
                )
            non_packed = [
                type(cb).__name__ for cb in self.callbacks
                if not getattr(cb, "accepts_packed", False)
            ]
            if non_packed:
                raise RuntimeError(
                    f"pack=True requires every callback to set "
                    f"accepts_packed=True (acts is delivered as RaggedTensor). "
                    f"Callbacks missing the flag: {non_packed}"
                )
            if self.verify:
                import warnings

                warnings.warn(
                    "Sweep(pack=True, verify=True) runs an extra full padded "
                    "forward through every layer to compare against the packed "
                    "path. Cost is roughly 2x a normal sweep; turn verify off "
                    "for production runs.",
                    UserWarning,
                    stacklevel=2,
                )

        _emit_progress("preloop_resolve_dataset", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write("fpwap: resolving dataset...\n")
            sys.stderr.flush()
        t0 = time.perf_counter_ns()
        items = _resolve_dataset(self.dataset)
        n_samples = len(items)
        if n_samples == 0:
            raise ValueError("fpwap dataset is empty")
        first_ids = items[0]["input_ids"]
        exec_device = streamer.execution_device or first_ids.device
        buf_device = self.buffer_device or exec_device
        config = getattr(model, "config", None)
        hidden_attr = getattr(config, "hidden_size", None) if config is not None else None
        if self.microbatch_size == "auto":
            if isinstance(buf_device, torch.device):
                buf_on_gpu = buf_device.type == "cuda"
            else:
                buf_on_gpu = "cuda" in str(buf_device)
            if isinstance(exec_device, torch.device):
                dev = exec_device
            else:
                dev = torch.device(exec_device)
            mb_size = estimate_max_microbatch(
                config=config,
                n_samples=n_samples,
                seq_len=self.seq_len,
                transport_dtype=self.transport_dtype,
                exec_device=dev,
                buf_on_gpu=buf_on_gpu,
            )
        elif self.microbatch_size is not None:
            mb_size = self.microbatch_size
        else:
            mb_size = n_samples
        verify_baseline: dict[int, Tensor] | None = None
        if self.verify:
            verify_baseline = _run_naive_baseline(
                model, plumbing, items, exec_device, mb_size
            )
        if hidden_attr is None:
            raise NotImplementedError(
                "model.config.hidden_size is required to size the residual buffer"
            )
        hidden = int(hidden_attr)
        pl.resolve_dataset_s = (time.perf_counter_ns() - t0) / 1e9

        _emit_progress("preloop_ensure_embedding", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write("fpwap: loading embedding weights...\n")
            sys.stderr.flush()
        t0 = time.perf_counter_ns()
        streamer.ensure_embedding_loaded(model, plumbing)
        final_norm: nn.Module | None = None
        if self.apply_final_norm:
            final_norm = plumbing.final_norm_module(model)
            if final_norm is not None and isinstance(streamer, _OffloadStreamer):
                for name in plumbing.final_norm_param_names(model):
                    _load_named_param(model, name, streamer._loader, exec_device)
        pl.ensure_embedding_s = (time.perf_counter_ns() - t0) / 1e9

        sweep_id = uuid.uuid4().hex[:12]
        ctx = Context(
            sweep_id=sweep_id,
            n_samples=n_samples,
            seq_len=self.seq_len,
            hidden=hidden,
            transport_dtype=self.transport_dtype,
        )

        has_mask = _has_attention_mask(items)

        _emit_progress("preloop_build_segments", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write("fpwap: building segments and buffers...\n")
            sys.stderr.flush()
        t0 = time.perf_counter_ns()
        if self.pack:
            if not has_mask:
                raise ValueError(
                    "pack=True requires attention_mask on every dataset item "
                    "to determine real sequence lengths."
                )
            real_lengths = [
                int(it["attention_mask"].sum().item()) for it in items
            ]
            cu_list: list[int] = [0]
            for L in real_lengths:
                cu_list.append(cu_list[-1] + L)
            cu = torch.tensor(cu_list, dtype=torch.int64)
            # Segment-wide flat tensors — built once, sliced per MB. The
            # embed pass and per-MB hot path (BlockMask + position_embeddings
            # build) read slices of these instead of re-deriving from items.
            packed_ids = _packed_real_input_ids(items)
            packed_pos_ids = torch.cat(
                [torch.arange(L, dtype=torch.int64) for L in real_lengths],
                dim=0,
            )
            packed_doc_ids = torch.repeat_interleave(
                torch.arange(1, len(real_lengths) + 1, dtype=torch.int64),
                torch.tensor(real_lengths, dtype=torch.int64),
            )
            buffer = ResidualBuffer(
                n_samples=n_samples,
                hidden=hidden,
                dtype=self.transport_dtype,
                device=buf_device,
                path=self.buffer_path,
                layout="packed",
                cu_seqlens=cu,
            )
            segments = [_Segment(
                seq_len=max(real_lengths) if real_lengths else 0,
                items=items,
                orig_indices=list(range(n_samples)),
                buffer=buffer,
                mask_buffer=None,
                mb_size=mb_size,
                is_packed=True,
                cu_seqlens=cu,
                packed_input_ids=packed_ids,
                packed_pos_ids=packed_pos_ids,
                packed_doc_ids=packed_doc_ids,
            )]
            left_padded = False
        elif self.padding == "bucketed":
            if not has_mask:
                raise ValueError(
                    "padding='bucketed' requires attention_mask on all dataset "
                    "items to determine real sequence lengths. Either add "
                    "attention_mask or use padding='fixed'."
                )
            if plumbing.uses_learned_positions:
                import warnings

                warnings.warn(
                    "padding='bucketed' with learned positional embeddings "
                    f"({type(plumbing).__name__}): real tokens receive different "
                    "position IDs than with padding='fixed'. Outputs will differ "
                    "from fixed padding at positions where the bucket seq_len "
                    "differs from the original seq_len. For RoPE models (Llama, "
                    "Mistral), this is not an issue.",
                    UserWarning,
                    stacklevel=2,
                )
            left_padded = _detect_left_padding(items)
            mb_override: int | None = None if self.microbatch_size == "auto" else mb_size
            segments = _build_bucketed_segments(
                items, self.seq_len, hidden, self.transport_dtype,
                buf_device, mb_override, config, exec_device,
                buffer_path=self.buffer_path,
            )
        else:
            left_padded = True
            buffer = ResidualBuffer(
                n_samples=n_samples,
                seq_len=self.seq_len,
                hidden=hidden,
                dtype=self.transport_dtype,
                device=buf_device,
                path=self.buffer_path,
            )
            mask_pin = buf_device.type == "cpu" and torch.cuda.is_available()
            mask_buffer: Tensor | None = (
                torch.zeros(
                    (n_samples, self.seq_len),
                    dtype=torch.int64,
                    device=buf_device,
                    pin_memory=mask_pin,
                )
                if has_mask
                else None
            )
            segments = [_Segment(
                seq_len=self.seq_len,
                items=items,
                orig_indices=list(range(n_samples)),
                buffer=buffer,
                mask_buffer=mask_buffer,
                mb_size=mb_size,
            )]
        pl.build_segments_s = (time.perf_counter_ns() - t0) / 1e9

        emits_sink: dict[
            tuple[int, str], list[tuple[Tensor, Tensor, Tensor | None]]
        ] = {}
        layer_artifacts: dict[tuple[str, int], Artifact] = {}

        _emit_progress("preloop_storage_start", -1, 0, 0)
        t0 = time.perf_counter_ns()
        if self.storage is not None:
            self.storage.on_sweep_start(sweep_id, n_samples)
        pl.storage_start_s = (time.perf_counter_ns() - t0) / 1e9

        _emit_progress("preloop_callbacks_start", -1, 0, 0)
        t0 = time.perf_counter_ns()
        for cb in self.callbacks:
            cb.on_sweep_start(ctx)
        pl.callbacks_start_s = (time.perf_counter_ns() - t0) / 1e9

        pl.total_s = (time.perf_counter_ns() - t0_preloop) / 1e9
        _emit_progress("preloop_end", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write(
                f"fpwap: preloop complete ({pl.total_s:.1f}s)\n"
            )
            sys.stderr.flush()

        _emit_progress("embed_start", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write("fpwap: embedding pass...\n")
            sys.stderr.flush()
        t0_embed = time.perf_counter_ns()

        # Pass 0: embedding over the whole dataset. Contiguous write_slice
        # into the pinned CPU buffer goes through the CUDA copy engine.
        with torch.no_grad():
            for seg in segments:
                for start in range(0, seg.n_samples, seg.mb_size):
                    stop = min(start + seg.mb_size, seg.n_samples)
                    if seg.is_packed:
                        # Pack the microbatch's real tokens into [1, T_real, ...].
                        # Slice the segment-wide flat ids by row-range —
                        # cu_seqlens translates the sample range for us.
                        assert seg.cu_seqlens is not None
                        assert seg.packed_input_ids is not None
                        row_start = int(seg.cu_seqlens[start].item())
                        row_stop = int(seg.cu_seqlens[stop].item())
                        ids_real = seg.packed_input_ids[row_start:row_stop].to(
                            exec_device
                        )
                        embedded = plumbing.embed(model, ids_real.unsqueeze(0))
                        # write_slice translates [start, stop) sample-range to
                        # rows via cu_seqlens — squeeze the batch dim away.
                        seg.buffer.write_slice(start, stop, embedded.squeeze(0))
                    else:
                        input_ids = _stack_field(
                            seg.items[start:stop], "input_ids"
                        ).to(exec_device)
                        embedded = plumbing.embed(model, input_ids)
                        seg.buffer.write_slice(start, stop, embedded)
                        if seg.mask_buffer is not None:
                            seg.mask_buffer[start:stop] = _stack_field(
                                seg.items[start:stop], "attention_mask"
                            ).to(dtype=torch.int64)
        t0_embed_sync = time.perf_counter_ns()
        if exec_device.type == "cuda":
            torch.cuda.synchronize()
        embed_sync_s = (time.perf_counter_ns() - t0_embed_sync) / 1e9
        embed_s = (time.perf_counter_ns() - t0_embed) / 1e9
        _emit_progress("embed_end", -1, 0, 0)

        # --- Loop setup: hook computation, tqdm init, chunk building ---
        t0_loop_setup = time.perf_counter_ns()

        wanted_hooks: set[str] = set()
        for cb in self.callbacks:
            for h in cb.target_hooks:
                wanted_hooks.add(h)
        sub_hooks: frozenset[str] = frozenset(
            wanted_hooks & {"attn_out", "mlp_out"}
        )
        dispatch_residual_pre = "residual_pre" in wanted_hooks
        needs_inline_sublayer_dispatch = any(
            cb.phase == "write"
            and any(h in {"attn_out", "mlp_out"} for h in cb.target_hooks)
            for cb in self.callbacks
        )

        layer_modules = plumbing.layer_modules(model)
        n_layers = len(layer_modules)

        max_cap = _max_capture_layer(self.callbacks, n_layers)
        effective_n_layers = min(max_cap + 1, n_layers)

        if final_norm is not None and effective_n_layers < n_layers:
            final_norm = None

        profile = ProfileReport()

        layer_iter: Any = None
        if self.progress is True:
            from tqdm.auto import tqdm

            layer_iter = tqdm(total=effective_n_layers, desc="fpwap layers", dynamic_ncols=True)

        loop_setup_s = (time.perf_counter_ns() - t0_loop_setup) / 1e9

        chunks = _make_chunks(effective_n_layers, self.chunk_size)

        # When chunk_size > 1 and the buffer lives on a different device
        # from the execution device, we allocate a GPU-resident scratch
        # tensor per segment so intermediate residuals stay on GPU within
        # a chunk (avoiding H2D/D2H round-trips for every layer boundary).
        use_gpu_scratch = (
            self.chunk_size > 1
            and buf_device.type != exec_device.type
        )

        # Prefetch: per-layer, same as chunk_size=1. The difference with
        # chunk_size > 1 is that unloads are deferred to the chunk boundary,
        # so multiple layers coexist on GPU for the chunk's duration.
        prefetch_future: Any | None = None

        t0_loop = time.perf_counter_ns()
        for chunk in chunks:
            # Allocate GPU scratch for inter-layer residuals within chunk.
            gpu_scratch: dict[int, Tensor] = {}
            if use_gpu_scratch and len(chunk) > 1:
                for seg in segments:
                    gpu_scratch[id(seg)] = torch.empty(
                        seg.n_samples, seg.seq_len, hidden,
                        dtype=self.transport_dtype, device=exec_device,
                    )

            # Per-MB invariants for packed segments — built lazily on first
            # touch within a chunk and reused across all layers in the chunk.
            # Avoids rebuilding BlockMask + position_embeddings 80× per MB on
            # an 80-layer model. Keyed by (id(seg), start).
            packed_mb_cache: dict[tuple[int, int], dict[str, Any]] = {}

            n_batches = sum(
                (seg.n_samples + seg.mb_size - 1) // seg.mb_size
                for seg in segments
            )

            for layer_idx in chunk:
                timing = LayerTiming()
                profile.per_layer[layer_idx] = timing

                is_first_in_chunk = layer_idx == chunk.start
                is_last_in_chunk = layer_idx == chunk.stop - 1
                has_scratch = bool(gpu_scratch)

                # Load this layer (wait for prefetch, or sync load).
                t_load = time.perf_counter_ns()
                if prefetch_future is not None:
                    prefetch_future.result()
                    prefetch_future = None
                else:
                    streamer.load_layer(model, layer_idx, plumbing)
                timing.load_s = (time.perf_counter_ns() - t_load) / 1e9
                timing.bytes_weights = streamer.last_load_bytes

                # Prefetch next layer so its load overlaps with this layer's
                # compute. Works across chunk boundaries: the last layer of
                # this chunk prefetches the first layer of the next chunk.
                if layer_idx + 1 < effective_n_layers:
                    prefetch_future = streamer.prefetch_load(
                        model, layer_idx + 1, plumbing
                    )

                for cb in self.callbacks:
                    cb.on_layer_start(layer_idx)

                _emit_progress("layer_start", layer_idx, 0, n_batches)
                block = layer_modules[layer_idx]
                batch_counter = 0

                for seg in segments:
                    for start in range(0, seg.n_samples, seg.mb_size):
                        stop = min(start + seg.mb_size, seg.n_samples)
                        sample_ids_exec = torch.tensor(
                            seg.orig_indices[start:stop],
                            device=exec_device,
                            dtype=torch.long,
                        )

                        # Local cu_seqlens for the packed microbatch — set inside
                        # the packed forward branch and reused in the post-forward
                        # callback dispatch. Stays None on the dense path.
                        mb_cu: Tensor | None = None
                        t_fwd = time.perf_counter_ns()
                        with torch.no_grad():
                            if has_scratch and not is_first_in_chunk:
                                hidden_states = gpu_scratch[id(seg)][start:stop]
                            else:
                                hidden_states = seg.buffer.read_slice(start, stop).to(
                                    exec_device, non_blocking=True
                                )
                            mb_mask = (
                                seg.mask_buffer[start:stop].to(
                                    exec_device, non_blocking=True
                                )
                                if seg.mask_buffer is not None
                                else None
                            )
                            if seg.is_packed:
                                # Packed path: hidden_states is [T_real, H].
                                # All per-MB invariants (cu_seqlens slice,
                                # position_ids, BlockMask, RoPE cos/sin) are
                                # cached for the duration of the chunk so each
                                # layer just reads them.
                                assert seg.cu_seqlens is not None
                                cache_key = (id(seg), start)
                                cached = packed_mb_cache.get(cache_key)
                                if cached is None:
                                    from transformers.integrations.flex_attention import (
                                        make_flex_block_causal_mask,
                                    )
                                    assert seg.packed_pos_ids is not None
                                    assert seg.packed_doc_ids is not None
                                    row_start = int(seg.cu_seqlens[start].item())
                                    row_stop = int(seg.cu_seqlens[stop].item())
                                    mb_cu_local = (
                                        seg.cu_seqlens[start : stop + 1]
                                        - seg.cu_seqlens[start]
                                    ).to(torch.int64)
                                    mb_pos_ids = (
                                        seg.packed_pos_ids[row_start:row_stop]
                                        .to(exec_device, non_blocking=True)
                                        .unsqueeze(0)
                                    )
                                    mb_doc_ids = (
                                        seg.packed_doc_ids[row_start:row_stop]
                                        .to(exec_device, non_blocking=True)
                                        .unsqueeze(0)
                                    )
                                    mb_block_mask = make_flex_block_causal_mask(
                                        mb_doc_ids
                                    )
                                    rotary_emb = (
                                        cast(Any, model).model.rotary_emb
                                    )
                                    mb_pos_emb = rotary_emb(
                                        hidden_states.unsqueeze(0), mb_pos_ids
                                    )
                                    cached = {
                                        "mb_cu": mb_cu_local,
                                        "pos_ids": mb_pos_ids,
                                        "block_mask": mb_block_mask,
                                        "pos_emb": mb_pos_emb,
                                    }
                                    packed_mb_cache[cache_key] = cached
                                mb_cu = cached["mb_cu"]
                                pos_ids = cached["pos_ids"]
                                mb_block_mask = cached["block_mask"]
                                mb_pos_emb = cached["pos_emb"]

                                acts_rt_pre = RaggedTensor(
                                    flat=hidden_states, offsets=mb_cu.to(hidden_states.device)
                                )
                                if dispatch_residual_pre:
                                    pre_acts, _pre_emit = self._dispatch_callbacks(
                                        layer_idx,
                                        "residual_pre",
                                        acts_rt_pre,
                                        sample_ids_exec,
                                        emits_sink=emits_sink,
                                        is_packed=True,
                                    )
                                    timing.emit_s += _pre_emit
                                    if isinstance(pre_acts, RaggedTensor):
                                        hidden_states = pre_acts.flat
                                hidden_states_3d = hidden_states.unsqueeze(0)
                                # layer_forward_packed lives only on plumbings
                                # whose supports_packed=True. Dispatch via getattr
                                # because the structural Protocol doesn't carry it.
                                _packed_fwd = plumbing.layer_forward_packed  # type: ignore[attr-defined]
                                hidden_states_3d, extras = _packed_fwd(
                                    model,
                                    block,
                                    hidden_states_3d,
                                    position_ids=pos_ids,
                                    block_mask=mb_block_mask,
                                    position_embeddings=mb_pos_emb,
                                    wanted_hooks=sub_hooks,
                                    dispatch_fn=None,
                                )
                                hidden_states = hidden_states_3d.squeeze(0)
                            else:
                                if dispatch_residual_pre:
                                    pre_acts, _pre_emit = self._dispatch_callbacks(
                                        layer_idx,
                                        "residual_pre",
                                        hidden_states,
                                        sample_ids_exec,
                                        emits_sink=emits_sink,
                                    )
                                    timing.emit_s += _pre_emit
                                    if isinstance(pre_acts, Tensor):
                                        hidden_states = pre_acts
                                inline_dispatch = None
                                if needs_inline_sublayer_dispatch:
                                    def _inline(
                                        hook_name: str,
                                        tensor: Tensor,
                                        _layer_idx: int = layer_idx,
                                        _sample_ids: Tensor = sample_ids_exec,
                                    ) -> Tensor:
                                        t, _e = self._dispatch_callbacks(
                                            _layer_idx,
                                            hook_name,  # type: ignore[arg-type]
                                            tensor,
                                            _sample_ids,
                                            emits_sink=emits_sink,
                                            write_allowed=True,
                                        )
                                        return t  # type: ignore[return-value]

                                    inline_dispatch = _inline

                                hidden_states, extras = plumbing.layer_forward_with_hooks(
                                    model,
                                    block,
                                    hidden_states,
                                    attention_mask=mb_mask,
                                    wanted_hooks=sub_hooks,
                                    dispatch_fn=inline_dispatch,
                                )
                        timing.forward_s += (time.perf_counter_ns() - t_fwd) / 1e9

                        if final_norm is not None and layer_idx == n_layers - 1:
                            hidden_states = final_norm(hidden_states)

                        t_cb = time.perf_counter_ns()
                        mb_emit_s = 0.0
                        if seg.is_packed:
                            # Wrap each post-forward tensor as RaggedTensor before
                            # handing to callbacks. mb_cu is in [N+1] form on the
                            # exec device; emits use the same offsets so callbacks
                            # can extract per-sample slices.
                            assert mb_cu is not None
                            mb_cu_for_emit = mb_cu.to(hidden_states.device)
                            for sub_hook, sub_tensor in extras.items():
                                sub_flat = sub_tensor.squeeze(0)
                                _, _sub_emit = self._dispatch_callbacks(
                                    layer_idx,
                                    sub_hook,
                                    RaggedTensor(flat=sub_flat, offsets=mb_cu_for_emit),
                                    sample_ids_exec,
                                    emits_sink=emits_sink,
                                    write_allowed=False,
                                    is_packed=True,
                                )
                                mb_emit_s += _sub_emit
                            post_acts, _post_emit = self._dispatch_callbacks(
                                layer_idx,
                                "residual_post",
                                RaggedTensor(flat=hidden_states, offsets=mb_cu_for_emit),
                                sample_ids_exec,
                                emits_sink=emits_sink,
                                is_packed=True,
                            )
                            if isinstance(post_acts, RaggedTensor):
                                hidden_states = post_acts.flat
                            mb_emit_s += _post_emit
                        else:
                            for sub_hook, sub_tensor in extras.items():
                                _, _sub_emit = self._dispatch_callbacks(
                                    layer_idx,
                                    sub_hook,
                                    sub_tensor,
                                    sample_ids_exec,
                                    emits_sink=emits_sink,
                                    write_allowed=False,
                                )
                                mb_emit_s += _sub_emit
                            post_acts, _post_emit = self._dispatch_callbacks(
                                layer_idx,
                                "residual_post",
                                hidden_states,
                                sample_ids_exec,
                                emits_sink=emits_sink,
                            )
                            if isinstance(post_acts, Tensor):
                                hidden_states = post_acts
                            mb_emit_s += _post_emit
                        timing.emit_s += mb_emit_s
                        if verify_baseline is not None and not seg.is_packed:
                            assert isinstance(hidden_states, Tensor)
                            expected = verify_baseline[layer_idx][start:stop].to(
                                device=hidden_states.device, dtype=hidden_states.dtype
                            )
                            if mb_mask is not None:
                                m = mb_mask.bool().unsqueeze(-1)
                                got_real = hidden_states.masked_select(m)
                                exp_real = expected.masked_select(m)
                                ok = torch.equal(got_real, exp_real)
                            else:
                                ok = torch.equal(hidden_states, expected)
                            if not ok:
                                diff = (hidden_states.float() - expected.float()).abs().max().item()
                                raise RuntimeError(
                                    f"verify: layer {layer_idx} microbatch [{start}:{stop}] "
                                    f"diverged from naive baseline (max abs diff {diff}). "
                                    f"For bf16 this usually means microbatch_size != dataset "
                                    f"size (see bf16_microbatch_determinism); for fp32 it "
                                    f"indicates a plumbing bug."
                                )
                        timing.callback_s += (time.perf_counter_ns() - t_cb) / 1e9

                        t_w = time.perf_counter_ns()
                        if has_scratch and not is_last_in_chunk:
                            gpu_scratch[id(seg)][start:stop].copy_(hidden_states)
                        else:
                            seg.buffer.write_slice(start, stop, hidden_states)
                            timing.bytes_buffer += (
                                hidden_states.element_size() * hidden_states.numel()
                            )
                        mb_write_s = (time.perf_counter_ns() - t_w) / 1e9
                        timing.write_s += mb_write_s
                        batch_counter += 1
                        if is_last_in_chunk and hasattr(layer_iter, "set_postfix"):
                            layer_iter.set_postfix({
                                "mb": f"{batch_counter}/{n_batches}",
                                "write": f"{mb_write_s * 1e3:.0f}ms",
                                "emit": f"{mb_emit_s * 1e3:.0f}ms",
                            })
                        _emit_progress(
                            "microbatch_end",
                            layer_idx,
                            batch_counter,
                            n_batches,
                        )

                _emit_progress("layer_end", layer_idx, n_batches, n_batches)

                if hasattr(layer_iter, "set_postfix"):
                    elapsed = (time.perf_counter_ns() - t0_run) / 1e9
                    tps = profile.total_tokens / elapsed if elapsed > 0 else 0
                    layer_iter.set_postfix({"tok/s": f"{tps:,.0f}"})

                for cb in self.callbacks:
                    layer_art = cb.on_layer_end(layer_idx)
                    if layer_art is not None:
                        from fpwap.types import ArtifactKey as _Key

                        layer_artifacts[(layer_art.kind, layer_idx)] = Artifact(
                            key=_Key(
                                sweep_id=sweep_id,
                                layer_idx=layer_idx,
                                hook="residual_post",
                                kind=layer_art.kind,
                            ),
                            payload=layer_art.payload,
                        )

                if hasattr(layer_iter, "update"):
                    layer_iter.update(1)

            _emit_progress("chunk_drain_sync", -1, 0, 0)
            if self.progress is True:
                sys.stderr.write(
                    f"fpwap: draining async writes (chunk {chunk})...\n"
                )
                sys.stderr.flush()
            t0_drain = time.perf_counter_ns()
            if exec_device.type == "cuda":
                torch.cuda.synchronize()
            profile.drain_sync_s += (time.perf_counter_ns() - t0_drain) / 1e9

            if self.storage is not None:
                t0_edrain = time.perf_counter_ns()
                self.storage.drain_emits()
                profile.emit_drain_s += (time.perf_counter_ns() - t0_edrain) / 1e9

            _emit_progress("chunk_unload", -1, 0, 0)
            if self.progress is True:
                sys.stderr.write(
                    f"fpwap: unloading chunk {chunk} weights...\n"
                )
                sys.stderr.flush()
            t0_unload = time.perf_counter_ns()
            if gpu_scratch:
                del gpu_scratch
            packed_mb_cache.clear()
            for li in chunk:
                streamer.unload_layer(model, li, plumbing)
            profile.unload_s += (time.perf_counter_ns() - t0_unload) / 1e9

        if layer_iter is not None and hasattr(layer_iter, "close"):
            layer_iter.close()

        loop_s = (time.perf_counter_ns() - t0_loop) / 1e9

        # --- Teardown: timed per sub-phase, with progress events ---
        td = TeardownTiming()
        t0_teardown = time.perf_counter_ns()

        _emit_progress("teardown_streamer_close", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write("fpwap: flushing (streamer close)...\n")
            sys.stderr.flush()
        t0 = time.perf_counter_ns()
        streamer.close()
        td.streamer_close_s = (time.perf_counter_ns() - t0) / 1e9

        _emit_progress("teardown_callbacks", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write("fpwap: flushing (callbacks)...\n")
            sys.stderr.flush()
        t0 = time.perf_counter_ns()
        artifacts: dict[tuple[str, int], Artifact] = dict(layer_artifacts)
        for cb in self.callbacks:
            art = cb.on_sweep_end()
            if art is not None:
                artifacts[(art.key.kind, art.key.layer_idx)] = art
        td.callbacks_s = (time.perf_counter_ns() - t0) / 1e9

        _emit_progress("teardown_buffer_flush", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write("fpwap: flushing (residual buffer)...\n")
            sys.stderr.flush()
        t0 = time.perf_counter_ns()
        for seg in segments:
            seg.buffer.flush()
        td.buffer_flush_s = (time.perf_counter_ns() - t0) / 1e9

        _emit_progress("teardown_storage_flush", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write("fpwap: flushing (storage backend)...\n")
            sys.stderr.flush()
        t0 = time.perf_counter_ns()
        if self.storage is not None:
            self.storage.on_sweep_end()
        td.storage_flush_s = (time.perf_counter_ns() - t0) / 1e9

        td.total_s = (time.perf_counter_ns() - t0_teardown) / 1e9
        _emit_progress("teardown_end", -1, 0, 0)
        if self.progress is True:
            sys.stderr.write(
                f"fpwap: teardown complete ({td.total_s:.1f}s: "
                f"buffer_flush {td.buffer_flush_s:.1f}s, "
                f"storage_flush {td.storage_flush_s:.1f}s)\n"
            )
            sys.stderr.flush()

        profile.total_wall_s = (time.perf_counter_ns() - t0_run) / 1e9
        profile.total_tokens = sum(
            seg.n_samples * seg.seq_len for seg in segments
        )
        profile.setup = setup_timing
        profile.preloop = pl
        profile.embed_s = embed_s
        profile.embed_sync_s = embed_sync_s
        profile.loop_setup_s = loop_setup_s
        profile.loop_s = loop_s
        profile.teardown = td
        return Result(
            sweep_id=sweep_id,
            artifacts=artifacts,
            storage=self.storage,
            profile=profile,
            seq_len=self.seq_len,
            _left_padded=left_padded,
            _emits=emits_sink,
        )

    def _dispatch_callbacks(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor | RaggedTensor,
        sample_ids: Tensor,
        emits_sink: dict[
            tuple[int, str], list[tuple[Tensor, Tensor, Tensor | None]]
        ] | None = None,
        write_allowed: bool = True,
        is_packed: bool = False,
    ) -> tuple[Tensor | RaggedTensor, float]:
        """Phase-ordered dispatch: read → write (sequential) → read_after_write.

        Returns (possibly modified activations, emit_write_seconds). Under
        `is_packed=True`, `acts` is a `RaggedTensor` (engine ran a packed
        forward) and the auto-pad-to-seq_len pad block is bypassed because
        the emit is already at native shape.
        """
        emit_s = 0.0

        # read
        for cb in self.callbacks:
            if cb.phase != "read" or not _callback_applies(cb, layer_idx, hook):
                continue
            result = cb.on_batch(layer_idx, hook, acts, sample_ids)  # type: ignore[arg-type]
            if isinstance(result, WriteBack):
                raise ValueError(
                    f"read-phase callback {type(cb).__name__} returned WriteBack"
                )
            if isinstance(result, Emit):
                ragged_lengths = result.sample_lengths
                if is_packed and ragged_lengths is None:
                    raise RuntimeError(
                        f"callback {type(cb).__name__} returned an Emit "
                        f"without sample_lengths under Sweep(pack=True); "
                        f"ragged emit lengths are required to reassemble "
                        f"per-sample slices in packed mode"
                    )
                if self.storage is not None:
                    emit_tensor = result.tensor
                    if not is_packed and ragged_lengths is None and (
                        emit_tensor.dim() >= 2
                        and emit_tensor.shape[1] < self.seq_len
                    ):
                        gap = self.seq_len - emit_tensor.shape[1]
                        pad_shape = (emit_tensor.shape[0], gap, *emit_tensor.shape[2:])
                        z = torch.zeros(
                            pad_shape, dtype=emit_tensor.dtype, device=emit_tensor.device,
                        )
                        emit_tensor = torch.cat([z, emit_tensor], dim=1)
                    t0_emit = time.perf_counter_ns()
                    self.storage.write_emit(
                        layer_idx, hook, sample_ids, emit_tensor,
                        sample_lengths=ragged_lengths,
                    )
                    emit_s += (time.perf_counter_ns() - t0_emit) / 1e9
                elif emits_sink is not None:
                    key = (layer_idx, hook)
                    emits_sink.setdefault(key, []).append(
                        (
                            sample_ids.detach().to("cpu", copy=True),
                            result.tensor.detach().to("cpu", copy=True),
                            (
                                ragged_lengths.detach().to("cpu", copy=True)
                                if ragged_lengths is not None
                                else None
                            ),
                        )
                    )

        # write
        for cb in self.callbacks:
            if cb.phase != "write" or not _callback_applies(cb, layer_idx, hook):
                continue
            if not write_allowed:
                raise NotImplementedError(
                    f"write-phase callback {type(cb).__name__} targets hook "
                    f"{hook!r}, but WriteBack at sub-layer hooks isn't wired. "
                    f"Use residual_pre or residual_post for steering."
                )
            if is_packed:
                raise NotImplementedError(
                    f"write-phase callback {type(cb).__name__} is not "
                    "supported under Sweep(pack=True) in the pack pilot. "
                    "Use pack=False or move steering to a read-phase setup."
                )
            result = cb.on_batch(layer_idx, hook, acts, sample_ids)  # type: ignore[arg-type]
            if isinstance(result, WriteBack):
                acts = result.tensor

        # read_after_write
        for cb in self.callbacks:
            if cb.phase != "read_after_write" or not _callback_applies(cb, layer_idx, hook):
                continue
            result = cb.on_batch(layer_idx, hook, acts, sample_ids)  # type: ignore[arg-type]
            if isinstance(result, WriteBack):
                raise ValueError(
                    f"read_after_write-phase callback {type(cb).__name__} returned WriteBack"
                )

        return acts, emit_s
