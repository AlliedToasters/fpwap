from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from fpwap.callbacks.base import fpwapCallback
from fpwap.preflight import PreflightReport
from fpwap.storage import StorageBackend
from fpwap.types import HookName, LoadingStrategy, fpwapArtifact


@dataclass
class fpwapResult:
    fpwap_id: str
    artifacts: dict[tuple[str, int], fpwapArtifact] = field(default_factory=dict)
    storage: StorageBackend | None = None

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
        self.seed = seed

    def preflight(self) -> PreflightReport:
        raise NotImplementedError

    def run(self) -> fpwapResult:
        raise NotImplementedError
