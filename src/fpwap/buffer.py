from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

BufferLayout = Literal["dense", "packed"]


class ResidualBuffer:
    """Inter-layer transport for the fpwap loop.

    Two layouts:
    - ``layout="dense"`` (default): shape ``[n_samples, seq_len, hidden]``.
      Every sample contributes the same fixed-length row.
    - ``layout="packed"``: shape ``[total_real_tokens, hidden]`` with a
      ``cu_seqlens [N+1]`` sidecar. Sample ``i`` lives at flat rows
      ``[cu_seqlens[i], cu_seqlens[i+1])``. Pad positions are never
      allocated. Used when ``Sweep(pack=True)``.

    Two storage modes (orthogonal to layout):
    - In-memory (``path=None``): pinned torch tensor. Fast async D2H via
      the CUDA copy engine. Default for workloads that fit in host RAM.
    - Disk-backed (``path=<file>``): numpy memmap. The OS page cache
      manages residency. bf16 is stored as uint16 bit-patterns (numpy
      has no bf16 dtype).

    The hot path (``read_slice`` / ``write_slice`` over a contiguous
    sample range) is shape-shared between layouts: packed translates
    ``[start, stop)`` through ``cu_seqlens`` to a row-range and emits a
    contiguous slice, exactly like dense.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int | None = None,
        hidden: int = 0,  # required at call site; default keeps it after the optional seq_len
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        path: Path | None = None,
        layout: BufferLayout = "dense",
        cu_seqlens: Tensor | None = None,
    ) -> None:
        if hidden <= 0:
            raise ValueError(f"hidden must be > 0; got {hidden}")
        if layout == "dense":
            if seq_len is None:
                raise ValueError("dense layout requires seq_len")
            shape: tuple[int, ...] = (n_samples, seq_len, hidden)
            total_tokens = n_samples * seq_len
            cu_cpu: Tensor | None = None
        elif layout == "packed":
            if cu_seqlens is None:
                raise ValueError("packed layout requires cu_seqlens")
            cu_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).contiguous()
            if cu_cpu.shape != (n_samples + 1,):
                raise ValueError(
                    f"cu_seqlens shape {tuple(cu_cpu.shape)} does not match "
                    f"n_samples+1 = {n_samples + 1}"
                )
            total_tokens = int(cu_cpu[-1].item())
            shape = (total_tokens, hidden)
        else:
            raise ValueError(f"unknown layout: {layout!r}")

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.hidden = hidden
        self.dtype = dtype
        self.device = torch.device(device)
        self.path = path
        self.layout: BufferLayout = layout
        self.cu_seqlens = cu_cpu
        self.total_tokens = total_tokens
        self._shape = shape

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
                        self._mm.ctypes.data, self._mm.nbytes,
                        os.POSIX_MADV_SEQUENTIAL,  # type: ignore[attr-defined]
                    )
                except OSError:
                    pass
            self._staging: Tensor | None = None
            self._data: Tensor | None = None
        else:
            self._bf16_as_u16 = False
            self._np_dtype = None
            self._mm = None
            self._staging = None
            pin = self.device.type == "cpu" and torch.cuda.is_available()
            self._data = torch.zeros(
                self._shape, dtype=dtype, device=self.device, pin_memory=pin,
            )

    def _row_range(self, start: int, stop: int) -> tuple[int, int]:
        """Translate a sample range to a flat-row range for the active layout."""
        if self.layout == "dense":
            return start, stop
        assert self.cu_seqlens is not None
        return int(self.cu_seqlens[start].item()), int(self.cu_seqlens[stop].item())

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
        if self.layout == "packed":
            raise NotImplementedError(
                "non-contiguous gather is not implemented for packed layout; "
                "use read_slice over a contiguous sample range"
            )
        if self._data is not None:
            return self._data[sample_ids]
        assert self._mm is not None
        ids_np = sample_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        return self._mm_to_tensor(np.asarray(self._mm[ids_np]).copy())

    def __setitem__(self, sample_ids: Tensor, values: Tensor) -> None:
        if self.layout == "packed":
            raise NotImplementedError(
                "non-contiguous scatter is not implemented for packed layout; "
                "use write_slice over a contiguous sample range"
            )
        if self._data is not None:
            self._data[sample_ids] = values.to(dtype=self.dtype, device=self.device)
            return
        assert self._mm is not None
        ids_np = sample_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        if values.device.type == "cuda":
            staging = self._ensure_staging(values.shape)
            if values.dtype != self.dtype:
                values = values.to(dtype=self.dtype)
            staging.copy_(values, non_blocking=True)
            torch.cuda.synchronize()
            host = staging
        else:
            host = values.detach().to(device="cpu", dtype=self.dtype)
        if self._bf16_as_u16:
            host = host.view(torch.uint16)
        self._mm[ids_np] = host.numpy()

    def read_slice(self, start: int, stop: int) -> Tensor:
        row_start, row_stop = self._row_range(start, stop)
        if self._data is not None:
            return self._data[row_start:row_stop]
        assert self._mm is not None
        return self._mm_to_tensor(np.asarray(self._mm[row_start:row_stop]).copy())

    def _ensure_staging(self, shape: tuple[int, ...]) -> Tensor:
        if self._staging is not None and self._staging.shape == shape:
            return self._staging
        self._staging = torch.zeros(shape, dtype=self.dtype, pin_memory=True)
        return self._staging

    def write_slice(self, start: int, stop: int, values: Tensor) -> None:
        row_start, row_stop = self._row_range(start, stop)
        if self._data is not None:
            if values.dtype != self.dtype:
                values = values.to(dtype=self.dtype)
            self._data[row_start:row_stop].copy_(values, non_blocking=True)
            return
        assert self._mm is not None
        if values.device.type == "cuda":
            staging = self._ensure_staging(values.shape)
            if values.dtype != self.dtype:
                values = values.to(dtype=self.dtype)
            staging.copy_(values, non_blocking=True)
            torch.cuda.synchronize()
            host = staging
        else:
            host = values.detach().to(device="cpu", dtype=self.dtype)
        if self._bf16_as_u16:
            host = host.view(torch.uint16)
        self._mm[row_start:row_stop] = host.numpy()

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
