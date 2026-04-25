from __future__ import annotations

from typing import Protocol

from torch import Tensor

from fpwap.types import HookName, RaggedTensor


class StorageBackend(Protocol):
    """Persist per-sample emits across a sweep.

    Pragmatic shape: engine calls on_sweep_start once, write_emit per
    microbatch that produces an Emit, and read_all when the user reaches for
    `result.activations(layer, hook)`. on_sweep_end gives the backend a
    chance to flush. Backends MAY lazily size their storage on first
    write_emit (shape comes from the tensor itself).

    Ragged emits (#65): when a callback returns Emit with sample_lengths set,
    the engine forwards them via the optional sample_lengths kwarg. The
    backend stores rows as a flat `[sum(lengths), *trailing]` tensor; read_all
    returns a RaggedTensor (flat + per-sample offsets) for that shard.
    """

    def on_sweep_start(self, sweep_id: str, n_samples: int) -> None: ...

    def write_emit(
        self,
        layer_idx: int,
        hook: HookName,
        sample_ids: Tensor,
        tensor: Tensor,
        sample_lengths: Tensor | None = None,
    ) -> None: ...

    def read_all(
        self, layer_idx: int, hook: HookName
    ) -> Tensor | RaggedTensor: ...

    def drain_emits(self) -> None: ...

    def on_sweep_end(self) -> None: ...


__all__ = ["StorageBackend"]
