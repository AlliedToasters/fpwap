from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


class ResidualBuffer:
    """Memmap-backed inter-layer transport. Sized N_samples x seq x hidden x dtype."""

    def __init__(
        self,
        path: Path,
        n_samples: int,
        seq_len: int,
        hidden: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.path = path
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.hidden = hidden
        self.dtype = dtype

    def __getitem__(self, sample_ids: Tensor) -> Tensor:
        raise NotImplementedError

    def __setitem__(self, sample_ids: Tensor, values: Tensor) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
