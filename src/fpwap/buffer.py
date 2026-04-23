from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


class ResidualBuffer:
    """Inter-layer transport for the fpwap loop.

    For now, an in-memory tensor. Spec calls for NVMe-backed memmap so residual
    state can exceed CPU RAM; that swap-in comes behind a unit test against
    this same get/set contract (see SPEC §3.4).

    When resident on CPU, the buffer is page-locked (`pin_memory=True`) so
    `read_slice` / `write_slice` can issue genuinely async H2D / D2H copies
    through the CUDA copy engine. Without that, D2H writes are stuck at
    unpaged transfer rates (~3 GB/s on PCIe 5.0 vs ~40 GB/s pinned) and they
    dominate the hero-scale wall-clock.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        hidden: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        path: Path | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.hidden = hidden
        self.dtype = dtype
        self.device = torch.device(device)
        self.path = path
        pin = self.device.type == "cpu" and torch.cuda.is_available()
        self._data: Tensor = torch.zeros(
            (n_samples, seq_len, hidden),
            dtype=dtype,
            device=self.device,
            pin_memory=pin,
        )

    def __getitem__(self, sample_ids: Tensor) -> Tensor:
        return self._data[sample_ids]

    def __setitem__(self, sample_ids: Tensor, values: Tensor) -> None:
        self._data[sample_ids] = values.to(dtype=self.dtype, device=self.device)

    def read_slice(self, start: int, stop: int) -> Tensor:
        """Contiguous view into the buffer — pinned when on CPU so callers
        can chain `.to(exec_device, non_blocking=True)` for async H2D.
        """
        return self._data[start:stop]

    def write_slice(self, start: int, stop: int, values: Tensor) -> None:
        """Async in-place copy into the buffer slice `[start:stop]`.

        On CPU-resident buffers this is a non-blocking D2H copy into pinned
        memory; the caller is responsible for synchronizing on the exec
        device's stream before reading the slice again (the engine does this
        at layer boundaries).
        """
        if values.dtype != self.dtype:
            values = values.to(dtype=self.dtype)
        self._data[start:stop].copy_(values, non_blocking=True)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None
