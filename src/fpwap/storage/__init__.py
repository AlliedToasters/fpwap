from __future__ import annotations

from pathlib import Path
from typing import Protocol

from torch import Tensor

from fpwap.types import HookName, RaggedTensor, ResultArtifact


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
    ) -> Tensor | RaggedTensor:
        """Read the full corpus for one (layer, hook).

        Returns a `Tensor` for dense shards (written without
        `sample_lengths`) and a `RaggedTensor` for ragged shards. Callers
        iterating over multiple (layer, hook) pairs that may mix layouts
        must dispatch on `isinstance(result, RaggedTensor)`.
        """
        ...

    def drain_emits(self) -> None: ...

    def on_sweep_end(self) -> None: ...

    def path_for(
        self,
        layer_idx: int,
        hook: HookName,
        dest: Path | None = None,
    ) -> ResultArtifact:
        """Return an on-disk handle for one (layer, hook), bypassing the
        materialize-into-RAM round-trip in `read_all` (issue #70).

        Modes:
            * `dest=None` — return a handle pointing at the backend's own
              file. Caller mmap-reads on demand. Backend retains ownership;
              the file lifetime is tied to the backend (e.g. a sweep
              cleanup may delete it later).
            * `dest=Path(dir)` — hardlink (fallback to copy on EXDEV) the
              data file and its sidecar into `dir`, and return a handle
              pointing at the hardlinked copy. Backend keeps the original.
              Caller owns `dir` and decides when to delete.

        Backends MUST flush pending writes (e.g. drain staging buffers,
        finalize ragged shards) before returning so the file is readable.
        """
        ...


__all__ = ["StorageBackend"]
