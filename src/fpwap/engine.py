from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from fpwap.buffer import ResidualBuffer
from fpwap.callbacks.base import Callback
from fpwap.loader import (
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
    bytes_weights: int = 0
    bytes_buffer: int = 0


@dataclass
class ProfileReport:
    """Always-on profile of an fpwap run. Target overhead: < 1% wall-clock.

    Built by the engine during .run() and attached to Result. Surface is
    designed for answering "where did the time go?" — not a debug dump.
    """

    total_wall_s: float = 0.0
    total_tokens: int = 0
    per_layer: dict[int, LayerTiming] = field(default_factory=dict)

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
        for i, t in sorted(self.per_layer.items()):
            lines.append(
                f"  layer {i}: load {t.load_s:.3f}s  fwd {t.forward_s:.3f}s  "
                f"cb {t.callback_s:.3f}s  write {t.write_s:.3f}s"
            )
        return "\n".join(lines)

    def by_phase(self) -> dict[str, list[float]]:
        phases: dict[str, list[float]] = {
            "load": [],
            "forward": [],
            "callback": [],
            "write": [],
        }
        for _, t in sorted(self.per_layer.items()):
            phases["load"].append(t.load_s)
            phases["forward"].append(t.forward_s)
            phases["callback"].append(t.callback_s)
            phases["write"].append(t.write_s)
        return phases

    def slowest_layer(self) -> tuple[int, str]:
        if not self.per_layer:
            return (-1, "none")
        phase_names = ("load_s", "forward_s", "callback_s", "write_s")
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
    # When no StorageBackend is set, Emit outputs are collected in memory per
    # (layer, hook). Fine for moderate-size sweeps; for dataset-scale
    # extraction, swap in a storage backend to avoid RAM pressure.
    _emits: dict[tuple[int, str], list[tuple[Tensor, Tensor]]] = field(
        default_factory=dict
    )

    def artifact(self, kind: str, layer: int) -> Artifact:
        return self.artifacts[(kind, layer)]

    def activations(self, layer: int, hook: HookName) -> Tensor:
        """Concatenate emitted activations for (layer, hook) in sample-id order.

        Requires at least one read-phase callback (e.g. RawActivations) to have
        emitted for this (layer, hook). With a StorageBackend wired, reads the
        full corpus back from disk (the backend owns shape/dtype); otherwise
        concatenates the in-memory microbatch emits.
        """
        if self.storage is not None:
            return self.storage.read_all(layer, hook)
        key = (layer, hook)
        if key not in self._emits or not self._emits[key]:
            raise KeyError(
                f"no activations collected for layer={layer}, hook={hook!r}; "
                f"add a read-phase callback that returns Emit (e.g. RawActivations)"
            )
        parts = self._emits[key]
        ordered = sorted(parts, key=lambda p: int(p[0][0]))
        tensors = [t for _, t in ordered]
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
        _unload_layer(model, layer_idx, plumbing)

    def prefetch_load(
        self, model: nn.Module, layer_idx: int, plumbing: ModelPlumbing
    ) -> Any | None:
        """Schedule layer_idx's load on the prefetch worker. Returns a future."""
        if self._prefetch_pool is None:
            return None
        return self._prefetch_pool.submit(
            self.load_layer, model, layer_idx, plumbing
        )

    def close(self) -> None:
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=True)
            self._prefetch_pool = None


def _callback_applies(cb: Callback, layer_idx: int, hook: HookName) -> bool:
    if cb.target_layers != "all" and layer_idx not in cb.target_layers:
        return False
    return hook in cb.target_hooks


def _stack_field(items: list[Any], key: str) -> Tensor:
    """Collate a slice of dataset items along `key` into a single `(mb, seq)` tensor."""
    parts: list[Tensor] = []
    for item in items:
        t = item[key]
        if t.dim() == 1:
            t = t.unsqueeze(0)
        parts.append(t)
    return torch.cat(parts, dim=0)


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
        microbatch_size: int | None = None,
        snapshot_dir: str | None = None,
        offload_dir: str | None = None,
        execution_device: torch.device | str | None = None,
        buffer_device: torch.device | str | None = None,
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
        self.microbatch_size = microbatch_size
        self.snapshot_dir = snapshot_dir
        self.offload_dir = offload_dir
        self.execution_device = (
            torch.device(execution_device) if execution_device is not None else None
        )
        self.buffer_device = (
            torch.device(buffer_device) if buffer_device is not None else None
        )

    def preflight(self) -> PreflightReport:
        """Minimum-viable feasibility + wall-clock estimate.

        Runs the embed pass + a single-layer dry-run on a small slice of the
        dataset, measures wall-clock, and extrapolates. Not the full SPEC §10
        planner (no microbatch binary search, no VRAM static analysis); this
        is the "does this configuration even start" gate plus a rough ETA.
        """
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

        model, streamer = self._resolve_model_and_streamer()
        model.eval()
        plumbing = get_plumbing(model)
        exec_device = streamer.execution_device or items[0]["input_ids"].device
        buf_device = self.buffer_device or exec_device

        config = getattr(model, "config", None)
        hidden_attr = getattr(config, "hidden_size", None) if config is not None else None
        hidden = int(hidden_attr) if hidden_attr is not None else 0
        n_layers = len(plumbing.layer_modules(model))
        mb_size = self.microbatch_size or min(n_samples, 8)

        element_bytes = (
            torch.zeros((), dtype=self.transport_dtype).element_size() if hidden else 2
        )
        residual_gb = n_samples * self.seq_len * hidden * element_bytes / 1e9

        streamer.ensure_embedding_loaded(model, plumbing)
        streamer.load_layer(model, 0, plumbing)

        probe_ids = torch.arange(mb_size, device=buf_device)
        input_ids = _stack_field(items[:mb_size], "input_ids").to(exec_device)
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            hidden_states = plumbing.embed(model, input_ids)
            _ = plumbing.layer_forward_with_hooks(
                model, plumbing.layer_modules(model)[0], hidden_states
            )
        if exec_device.type == "cuda":
            torch.cuda.synchronize()
        probe_s = (time.perf_counter_ns() - t0) / 1e9
        streamer.unload_layer(model, 0, plumbing)
        del probe_ids, input_ids, hidden_states

        n_microbatches = (n_samples + mb_size - 1) // mb_size
        est_wall = probe_s * n_layers * n_microbatches
        est_weight_io_gb = streamer.last_load_bytes * n_layers / 1e9

        return _Report(
            feasible=True,
            microbatch_size=mb_size,
            residual_buffer_gb=residual_gb,
            per_layer_peak_vram_gb=0.0,  # static analysis deferred
            estimated_wall_clock_s=est_wall,
            estimated_weight_io_gb=est_weight_io_gb,
            loading_strategy="cpu_offload",
            warnings=(
                ["preflight is a minimal wall-clock estimate, not a full planner"]
            ),
        )

    def _resolve_model_and_streamer(self) -> tuple[nn.Module, _LayerStreamer]:
        """Turn `self.model` into (concrete nn.Module, streamer).

        Pre-loaded nn.Module → _PreloadedStreamer (no-op per-layer).
        String snapshot path → build empty-weights model + _OffloadStreamer
        backed by OffloadedWeightsLoader.
        """
        if isinstance(self.model, nn.Module):
            return self.model, _PreloadedStreamer(self.execution_device)
        if isinstance(self.model, str):
            if self.execution_device is None:
                raise ValueError(
                    "execution_device is required when model is a string ID"
                )
            from pathlib import Path

            snapshot_dir = Path(self.snapshot_dir) if self.snapshot_dir else Path(self.model)
            model, accel_index = build_empty_model_and_index(
                model_id=self.model,
                snapshot_dir=snapshot_dir,
                dtype=self.transport_dtype,
            )
            streamer = _OffloadStreamer(accel_index, self.execution_device)
            return model, streamer
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
        model, streamer = self._resolve_model_and_streamer()
        model.eval()  # fpwap is inference-only; dropout/etc. must be off.
        plumbing = get_plumbing(model)

        items = _resolve_dataset(self.dataset)
        n_samples = len(items)
        if n_samples == 0:
            raise ValueError("fpwap dataset is empty")

        mb_size = self.microbatch_size or n_samples

        first_ids = items[0]["input_ids"]
        exec_device = streamer.execution_device or first_ids.device
        buf_device = self.buffer_device or exec_device

        # verify=True: one-shot naive forward over the dataset, capturing
        # every block's residual_post. The main loop diffs its output
        # per-microbatch against this baseline (fail-fast).
        verify_baseline: dict[int, Tensor] | None = None
        if self.verify:
            verify_baseline = _run_naive_baseline(
                model, plumbing, items, exec_device, mb_size
            )
        config = getattr(model, "config", None)
        hidden_attr = getattr(config, "hidden_size", None) if config is not None else None
        if hidden_attr is None:
            raise NotImplementedError(
                "model.config.hidden_size is required to size the residual buffer"
            )
        hidden = int(hidden_attr)

        # Streaming path: load pass-0 embedding weights onto the execution device.
        streamer.ensure_embedding_loaded(model, plumbing)

        sweep_id = uuid.uuid4().hex[:12]
        ctx = Context(
            sweep_id=sweep_id,
            n_samples=n_samples,
            seq_len=self.seq_len,
            hidden=hidden,
            transport_dtype=self.transport_dtype,
        )
        buffer = ResidualBuffer(
            n_samples=n_samples,
            seq_len=self.seq_len,
            hidden=hidden,
            dtype=self.transport_dtype,
            device=buf_device,
        )

        # Optional parallel mask buffer: (N, seq) int8 when any item carries
        # attention_mask, else None. Stored once; reused every layer so the
        # 2D→additive conversion happens inside layer_forward per microbatch.
        has_mask = _has_attention_mask(items)
        mask_pin = buf_device.type == "cpu"
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

        # Emit sink: storage backend (disk) if wired, else in-memory.
        emits_sink: dict[tuple[int, str], list[tuple[Tensor, Tensor]]] = {}
        # Per-layer artifacts from on_layer_end returns, merged with on_sweep_end
        # artifacts at run end. Both land in Result.artifacts keyed by (kind, layer).
        layer_artifacts: dict[tuple[str, int], Artifact] = {}
        if self.storage is not None:
            self.storage.on_sweep_start(sweep_id, n_samples)

        for cb in self.callbacks:
            cb.on_sweep_start(ctx)

        t0_run = time.perf_counter_ns()

        # Pass 0: embedding over the whole dataset. Contiguous write_slice
        # into the pinned CPU buffer goes through the CUDA copy engine.
        with torch.no_grad():
            for start in range(0, n_samples, mb_size):
                stop = min(start + mb_size, n_samples)
                input_ids = _stack_field(items[start:stop], "input_ids").to(exec_device)
                embedded = plumbing.embed(model, input_ids)
                buffer.write_slice(start, stop, embedded)
                if mask_buffer is not None:
                    mask_buffer[start:stop] = _stack_field(
                        items[start:stop], "attention_mask"
                    ).to(dtype=torch.int64)
        if exec_device.type == "cuda":
            torch.cuda.synchronize()

        # Which hooks do any callbacks care about? Drives whether to take the
        # fast-path (direct block forward) or decompose the block to expose
        # attn_out / mlp_out, and whether to dispatch residual_pre at all.
        wanted_hooks: set[str] = set()
        for cb in self.callbacks:
            for h in cb.target_hooks:
                wanted_hooks.add(h)
        sub_hooks: frozenset[str] = frozenset(
            wanted_hooks & {"attn_out", "mlp_out"}
        )
        dispatch_residual_pre = "residual_pre" in wanted_hooks
        # If any write-phase callback targets a sub-layer hook, we need to
        # thread a dispatcher into the plumbing so the WriteBack takes effect
        # mid-block (instead of being applied uselessly after residual add).
        needs_inline_sublayer_dispatch = any(
            cb.phase == "write"
            and any(h in {"attn_out", "mlp_out"} for h in cb.target_hooks)
            for cb in self.callbacks
        )

        # Main loop: for each layer, for each microbatch.
        layer_modules = plumbing.layer_modules(model)
        n_layers = len(layer_modules)
        profile = ProfileReport()

        layer_iter: Iterable[int]
        progress_reporter: ProgressReporter | None = None
        if self.progress is True:
            from tqdm.auto import tqdm

            layer_iter = tqdm(range(n_layers), desc="fpwap layers", dynamic_ncols=True)
        elif self.progress is False:
            layer_iter = range(n_layers)
        elif callable(self.progress):
            # Callable reporter: bypass tqdm, emit ProgressEvents at layer
            # and microbatch boundaries into the user's sink (wandb, rich, …).
            layer_iter = range(n_layers)
            progress_reporter = self.progress
        else:
            layer_iter = range(n_layers)

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

        # Prefetch: a future for the NEXT layer's load, submitted at the end
        # of the current layer so it runs in parallel with this layer's
        # microbatch loop. On first iteration, load sync.
        prefetch_future: Any | None = None

        for layer_idx in layer_iter:
            timing = LayerTiming()
            profile.per_layer[layer_idx] = timing

            t_load = time.perf_counter_ns()
            if prefetch_future is not None:
                # Wait for the async prefetch to finish; any time that shows
                # up here is the portion of load that wasn't covered by the
                # previous layer's compute. Worker-thread H2Ds are CPU-
                # blocking (source tensors from safetensors mmap aren't
                # pinned), so `future.result()` returning means the data is
                # on GPU; the end-of-previous-layer cuda.synchronize() has
                # already drained the main stream, so no extra sync needed.
                prefetch_future.result()
                prefetch_future = None
            else:
                streamer.load_layer(model, layer_idx, plumbing)
            timing.load_s += (time.perf_counter_ns() - t_load) / 1e9
            timing.bytes_weights += streamer.last_load_bytes

            for cb in self.callbacks:
                cb.on_layer_start(layer_idx)

            n_batches = (n_samples + mb_size - 1) // mb_size
            _emit_progress("layer_start", layer_idx, 0, n_batches)
            block = layer_modules[layer_idx]
            for start in range(0, n_samples, mb_size):
                stop = min(start + mb_size, n_samples)
                sample_ids_exec = torch.arange(start, stop, device=exec_device)

                t_fwd = time.perf_counter_ns()
                with torch.no_grad():
                    # Contiguous slice reads exploit the pinned CPU buffer for
                    # async H2D; index reads (buffer[tensor]) would allocate a
                    # non-pinned intermediate and block the transfer.
                    hidden_states = buffer.read_slice(start, stop).to(
                        exec_device, non_blocking=True
                    )
                    mb_mask = (
                        mask_buffer[start:stop].to(exec_device, non_blocking=True)
                        if mask_buffer is not None
                        else None
                    )
                    if dispatch_residual_pre:
                        hidden_states = self._dispatch_callbacks(
                            layer_idx,
                            "residual_pre",
                            hidden_states,
                            sample_ids_exec,
                            emits_sink=emits_sink,
                        )
                    inline_dispatch = None
                    if needs_inline_sublayer_dispatch:
                        # Bind this microbatch's identifiers via default args
                        # so the closure doesn't pick up later iterations'
                        # values. Called synchronously inside plumbing for
                        # each sub-layer hook that has callbacks wired.
                        def _inline(
                            hook_name: str,
                            tensor: Tensor,
                            _layer_idx: int = layer_idx,
                            _sample_ids: Tensor = sample_ids_exec,
                        ) -> Tensor:
                            return self._dispatch_callbacks(
                                _layer_idx,
                                hook_name,  # type: ignore[arg-type]
                                tensor,
                                _sample_ids,
                                emits_sink=emits_sink,
                                write_allowed=True,
                            )

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

                t_cb = time.perf_counter_ns()
                # Sub-layer extras that the plumbing didn't dispatch inline
                # (pure-read case) go here as read-only. When inline dispatch
                # was active, extras is empty by construction.
                for sub_hook, sub_tensor in extras.items():
                    self._dispatch_callbacks(
                        layer_idx,
                        sub_hook,  # type: ignore[arg-type]
                        sub_tensor,
                        sample_ids_exec,
                        emits_sink=emits_sink,
                        write_allowed=False,
                    )
                hidden_states = self._dispatch_callbacks(
                    layer_idx,
                    "residual_post",
                    hidden_states,
                    sample_ids_exec,
                    emits_sink=emits_sink,
                )
                if verify_baseline is not None:
                    expected = verify_baseline[layer_idx][start:stop].to(
                        device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    # On padded datasets, the pad positions are "don't care"
                    # (HF's own output there is undefined); compare only at
                    # real tokens, like tests/integration/test_padded_batch.py.
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
                            f"For bf16 this usually means microbatch_size != dataset size "
                            f"(see bf16_microbatch_determinism); for fp32 it indicates "
                            f"a plumbing bug."
                        )
                timing.callback_s += (time.perf_counter_ns() - t_cb) / 1e9

                t_w = time.perf_counter_ns()
                buffer.write_slice(start, stop, hidden_states)
                timing.write_s += (time.perf_counter_ns() - t_w) / 1e9
                timing.bytes_buffer += hidden_states.element_size() * hidden_states.numel()
                _emit_progress(
                    "microbatch_end",
                    layer_idx,
                    start // mb_size + 1,
                    n_batches,
                )

            # Drain pending async D2H writes so the next layer's reads see
            # a coherent buffer. One sync per layer is negligible vs the
            # wins from letting writes overlap with compute within the layer.
            if exec_device.type == "cuda":
                torch.cuda.synchronize()
            _emit_progress("layer_end", layer_idx, n_batches, n_batches)

            for cb in self.callbacks:
                layer_art = cb.on_layer_end(layer_idx)
                if layer_art is not None:
                    # Per-layer artifacts land in Result.artifacts keyed by
                    # (kind, layer). Synthesize the full ArtifactKey so
                    # callers can walk metadata if they want.
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

            streamer.unload_layer(model, layer_idx, plumbing)

            # Kick off prefetch of the next layer right after unload, so the
            # worker starts its safetensors read while the next iteration's
            # end-of-layer synchronization (from cuda.synchronize above) has
            # just drained the main stream. This overlaps worker I/O with
            # Python-side accounting on the main thread; empirically the
            # submit-after-unload placement beats submit-before-fwd on the
            # 1024 × 128 bench (see earlier v2 measurement) — possibly
            # because moving the submit earlier causes some contention on
            # the CUDA driver lock with the main stream's kernel launches.
            if layer_idx + 1 < n_layers:
                prefetch_future = streamer.prefetch_load(
                    model, layer_idx + 1, plumbing
                )

        # Drain the prefetch pool so the worker thread exits cleanly.
        streamer.close()

        artifacts: dict[tuple[str, int], Artifact] = dict(layer_artifacts)
        for cb in self.callbacks:
            art = cb.on_sweep_end()
            if art is not None:
                artifacts[(art.key.kind, art.key.layer_idx)] = art

        profile.total_wall_s = (time.perf_counter_ns() - t0_run) / 1e9
        profile.total_tokens = n_samples * self.seq_len
        if self.storage is not None:
            self.storage.on_sweep_end()
        return Result(
            sweep_id=sweep_id,
            artifacts=artifacts,
            storage=self.storage,
            profile=profile,
            _emits=emits_sink,
        )

    def _dispatch_callbacks(
        self,
        layer_idx: int,
        hook: HookName,
        acts: Tensor,
        sample_ids: Tensor,
        emits_sink: dict[tuple[int, str], list[tuple[Tensor, Tensor]]] | None = None,
        write_allowed: bool = True,
    ) -> Tensor:
        """Phase-ordered dispatch: read → write (sequential) → read_after_write.

        Returns the (possibly modified) activations. When `write_allowed` is
        False (sub-layer hooks: attn_out / mlp_out), a callback whose phase is
        "write" and whose targets include this hook raises — threading a
        modified sub-layer tensor back into the block mid-forward isn't
        supported yet.
        """
        # read
        for cb in self.callbacks:
            if cb.phase != "read" or not _callback_applies(cb, layer_idx, hook):
                continue
            result = cb.on_batch(layer_idx, hook, acts, sample_ids)
            if isinstance(result, WriteBack):
                raise ValueError(
                    f"read-phase callback {type(cb).__name__} returned WriteBack"
                )
            if isinstance(result, Emit):
                if self.storage is not None:
                    self.storage.write_emit(
                        layer_idx, hook, sample_ids, result.tensor
                    )
                elif emits_sink is not None:
                    key = (layer_idx, hook)
                    # Clone on capture: residual_pre Emits alias the residual
                    # buffer (read_slice returns a view); the same slice gets
                    # overwritten by this layer's residual_post write, which
                    # would corrupt the captured tensor without this clone.
                    emits_sink.setdefault(key, []).append(
                        (
                            sample_ids.detach().to("cpu", copy=True),
                            result.tensor.detach().to("cpu", copy=True),
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
            result = cb.on_batch(layer_idx, hook, acts, sample_ids)
            if isinstance(result, WriteBack):
                acts = result.tensor

        # read_after_write
        for cb in self.callbacks:
            if cb.phase != "read_after_write" or not _callback_applies(cb, layer_idx, hook):
                continue
            result = cb.on_batch(layer_idx, hook, acts, sample_ids)
            if isinstance(result, WriteBack):
                raise ValueError(
                    f"read_after_write-phase callback {type(cb).__name__} returned WriteBack"
                )

        return acts
