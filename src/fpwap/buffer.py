from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class ResidualBuffer:
    """Inter-layer transport for the fpwap loop.

    Two modes:
    - In-memory (path=None): pinned torch tensor. Fast async D2H via the CUDA
      copy engine. Default for workloads that fit in host RAM.
    - Disk-backed (path=<file>): numpy memmap. The OS page cache manages
      residency; the full [N, seq, H] corpus never needs to fit in RAM at once.
      bf16 is stored as uint16 bit-patterns (numpy has no bf16 dtype).
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
        self._shape = (n_samples, seq_len, hidden)

        if path is not None:
            self._bf16_as_u16 = dtype == torch.bfloat16
            np_dtype = _torch_to_numpy(dtype)
            self._np_dtype: type | None = np_dtype
            self._mm: np.memmap | None = np.memmap(
                path, dtype=np_dtype, mode="w+", shape=self._shape
            )
            if hasattr(os, "posix_madvise"):
                try:
                    os.posix_madvise(
                        self._mm.ctypes.data, self._mm.nbytes, os.POSIX_MADV_SEQUENTIAL
                    )
                except OSError:
                    pass
            self._data: Tensor | None = None
        else:
            self._bf16_as_u16 = False
            self._np_dtype = None
            self._mm = None
            pin = self.device.type == "cpu" and torch.cuda.is_available()
            self._data = torch.zeros(
                self._shape, dtype=dtype, device=self.device, pin_memory=pin,
            )

    def _mm_to_tensor(self, arr: np.ndarray) -> Tensor:
        t = torch.from_numpy(arr)
        if self._bf16_as_u16:
            t = t.view(torch.bfloat16)
        return t

    def _tensor_to_np(self, values: Tensor) -> np.ndarray:
        host = values.detach().to(device="cpu", dtype=self.dtype)
        if self._bf16_as_u16:
            host = host.view(torch.uint16)
        return host.numpy()

    def __getitem__(self, sample_ids: Tensor) -> Tensor:
        if self._data is not None:
            return self._data[sample_ids]
        assert self._mm is not None
        ids_np = sample_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        return self._mm_to_tensor(np.asarray(self._mm[ids_np]).copy())

    def __setitem__(self, sample_ids: Tensor, values: Tensor) -> None:
        if self._data is not None:
            self._data[sample_ids] = values.to(dtype=self.dtype, device=self.device)
            return
        assert self._mm is not None
        ids_np = sample_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        self._mm[ids_np] = self._tensor_to_np(values)

    def read_slice(self, start: int, stop: int) -> Tensor:
        if self._data is not None:
            return self._data[start:stop]
        assert self._mm is not None
        return self._mm_to_tensor(np.asarray(self._mm[start:stop]).copy())

    def write_slice(self, start: int, stop: int, values: Tensor) -> None:
        if self._data is not None:
            if values.dtype != self.dtype:
                values = values.to(dtype=self.dtype)
            self._data[start:stop].copy_(values, non_blocking=True)
            return
        assert self._mm is not None
        self._mm[start:stop] = self._tensor_to_np(values)

    def flush(self) -> None:
        if self._mm is not None:
            self._mm.flush()

    def close(self) -> None:
        if self._mm is not None:
            self._mm.flush()
            del self._mm
            self._mm = None


_TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.bfloat16: np.uint16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
}


def _torch_to_numpy(dtype: torch.dtype) -> type:
    np_dtype = _TORCH_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported dtype for memmap: {dtype}")
    return np_dtype
