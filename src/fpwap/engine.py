from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from fpwap.callbacks.base import fpwapCallback
from fpwap.preflight import PreflightReport
from fpwap.storage import StorageBackend
from fpwap.types import HookName, LoadingStrategy, fpwapArtifact

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

    Built by the engine during .run() and attached to fpwapResult. Surface is
    designed for answering "where did the time go?" — not a debug dump.
    """

    total_wall_s: float = 0.0
    per_layer: dict[int, LayerTiming] = field(default_factory=dict)

    def summary(self) -> str:
        raise NotImplementedError

    def by_phase(self) -> dict[str, list[float]]:
        raise NotImplementedError

    def slowest_layer(self) -> tuple[int, str]:
        raise NotImplementedError

    def bytes_moved(self) -> dict[str, int]:
        raise NotImplementedError


@dataclass
class fpwapResult:
    fpwap_id: str
    artifacts: dict[tuple[str, int], fpwapArtifact] = field(default_factory=dict)
    storage: StorageBackend | None = None
    profile: ProfileReport = field(default_factory=ProfileReport)

    def artifact(self, kind: str, layer: int) -> fpwapArtifact:
        raise NotImplementedError

    def activations(self, layer: int, hook: HookName) -> Tensor:
        raise NotImplementedError


class fpwap:
    """The engine. Inverts the inference loop: for each layer, run the dataset.

    Construction is cheap; call .preflight() to plan, .run() to execute.
    """

    def __init__(
        self,
        model: str | Any,
        dataset: Iterable[Any],
        seq_len: int,
        callbacks: Sequence[fpwapCallback],
        storage: StorageBackend | None = None,
        transport_dtype: torch.dtype = torch.bfloat16,
        loading_strategy: LoadingStrategy | None = None,
        verify: bool = False,
        progress: bool | ProgressReporter = True,
        seed: int = 0,
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

    def preflight(self) -> PreflightReport:
        raise NotImplementedError

    def run(self) -> fpwapResult:
        raise NotImplementedError
