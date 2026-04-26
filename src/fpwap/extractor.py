from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from fpwap.callbacks.base import Callback
from fpwap.storage import StorageBackend
from fpwap.types import LoadingStrategy, PaddingMode

if TYPE_CHECKING:
    from fpwap.engine import Sweep


class Extractor:
    """Reusable handle that owns a pre-built empty model + accelerate index.

    Multiple Sweeps against the same model reuse the cached artifacts instead
    of rebuilding ``build_empty_model_and_index`` on each call — for 70B that
    rebuild is real wall-clock burned on every eval dataset.

    Sequential use only: do not start a new Sweep while a previous one is
    still in-flight on the same Extractor.
    """

    def __init__(
        self,
        model: nn.Module,
        accel_index: dict[str, dict[str, Any]],
        snapshot_dir: Path,
        model_id: str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._model = model
        self._accel_index = accel_index
        self._snapshot_dir = snapshot_dir
        self._model_id = model_id
        self._dtype = dtype

    @classmethod
    def from_hf(
        cls,
        model_id: str,
        snapshot_dir: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Extractor:
        """Build an Extractor from an HF model ID or local snapshot path."""
        from fpwap.loader import build_empty_model_and_index, resolve_snapshot_dir

        if snapshot_dir is not None:
            sdir = Path(snapshot_dir)
        else:
            sdir = resolve_snapshot_dir(model_id)

        model, accel_index, _ = build_empty_model_and_index(
            model_id=model_id, snapshot_dir=sdir, dtype=dtype,
        )
        return cls(
            model=model,
            accel_index=accel_index,
            snapshot_dir=sdir,
            model_id=model_id,
            dtype=dtype,
        )

    def sweep(
        self,
        dataset: Iterable[Any],
        seq_len: int,
        callbacks: Sequence[Callback],
        storage: StorageBackend | None = None,
        transport_dtype: torch.dtype | None = None,
        loading_strategy: LoadingStrategy | None = None,
        verify: bool = False,
        progress: bool | Any = True,
        seed: int = 0,
        microbatch_size: int | Literal["auto"] | None = None,
        offload_dir: str | None = None,
        execution_device: torch.device | str | None = None,
        buffer_device: torch.device | str | None = None,
        apply_final_norm: bool = True,
        chunk_size: int = 1,
        padding: PaddingMode = "fixed",
        buffer_path: str | Path | None = None,
        pack: bool = False,
    ) -> Sweep:
        """Create a Sweep that reuses this Extractor's model and index."""
        from fpwap.engine import Sweep

        return Sweep(
            model=self._model,
            dataset=dataset,
            seq_len=seq_len,
            callbacks=callbacks,
            storage=storage,
            transport_dtype=transport_dtype if transport_dtype is not None else self._dtype,
            loading_strategy=loading_strategy,
            verify=verify,
            progress=progress,
            seed=seed,
            microbatch_size=microbatch_size,
            snapshot_dir=str(self._snapshot_dir),
            offload_dir=offload_dir,
            execution_device=execution_device,
            buffer_device=buffer_device,
            apply_final_norm=apply_final_norm,
            chunk_size=chunk_size,
            padding=padding,
            buffer_path=buffer_path,
            pack=pack,
            _accel_index=self._accel_index,
        )
