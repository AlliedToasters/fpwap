"""Memmap-backed storage for per-sample emits.

Each (layer_idx, hook) gets its own memmap file sized `[N, *per_row_shape]`,
lazily created on first emit. Indexed writes per microbatch land into the
slots named by `sample_ids`. At read time the whole file is mapped as a
torch tensor — the OS page cache is the de-facto memory budget, so the
full [N, S, H] corpus never lives in Python memory at once.

This is the path that keeps `RawActivations(last_token_only=False)` tractable
on the SPEC §17 target workload (Llama-70B × 10k prompts × 128 tokens ≈
1.7 TB of residuals — not a RAM-resident object).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fpwap.types import HookName

_HAS_POSIX_FADVISE = hasattr(os, "posix_fadvise")

_TORCH_TO_NUMPY: dict[torch.dtype, Any] = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.bfloat16: None,  # numpy has no bf16; we store as uint16
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
}


def _shard_basename(layer_idx: int, hook: HookName) -> str:
    return f"layer_{layer_idx:04d}_{hook}"


class _Shard:
    """One memmap file + its per-row shape/dtype, lazily sized."""

    def __init__(self, path: Path, n_samples: int) -> None:
        self.path = path
        self.n_samples = n_samples
        self._mm: np.memmap | None = None
        self.per_row_shape: tuple[int, ...] | None = None
        self.torch_dtype: torch.dtype | None = None
        self._np_dtype: Any = None
        self._stores_bf16_as_u16: bool = False
        self._dirty: bool = False

    def _ensure(self, sample_tensor: Tensor) -> np.memmap:
        if self._mm is not None:
            return self._mm
        self.torch_dtype = sample_tensor.dtype
        self.per_row_shape = tuple(sample_tensor.shape[1:])
        np_dtype = _TORCH_TO_NUMPY.get(sample_tensor.dtype)
        if np_dtype is None:
            # bf16: store bit-pattern as uint16 so the file is a standalone
            # artifact (numpy can't dtype bf16 directly, but the raw bits
            # round-trip through view).
            self._stores_bf16_as_u16 = True
            np_dtype = np.uint16
        self._np_dtype = np_dtype
        shape = (self.n_samples, *self.per_row_shape)
        self._mm = np.memmap(self.path, dtype=np_dtype, mode="w+", shape=shape)
        meta_path = self.path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {
                    "n_samples": self.n_samples,
                    "per_row_shape": list(self.per_row_shape),
                    "dtype": str(sample_tensor.dtype).removeprefix("torch."),
                    "bf16_as_u16": self._stores_bf16_as_u16,
                }
            )
        )
        return self._mm

    def write(self, sample_ids: Tensor, tensor: Tensor) -> None:
        mm = self._ensure(tensor)
        ids_np = sample_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        host = tensor.detach().to(device="cpu")
        if self._stores_bf16_as_u16:
            host = host.view(torch.uint16)
        mm[ids_np] = host.numpy()
        self._dirty = True

    def drain(self) -> None:
        """Flush dirty pages to disk and evict from page cache.

        Called at chunk boundaries instead of per-microbatch (#61).
        """
        if not self._dirty:
            return
        self._flush_and_evict()
        self._dirty = False

    def _flush_and_evict(self) -> None:
        """Flush dirty pages to disk, then advise the kernel to reclaim them.

        Emit shards are write-once-read-at-end; without this, written pages
        stay hot in page cache for the entire sweep, stealing budget from
        weight and residual streaming (see #50).
        """
        if self._mm is None:
            return
        self._mm.flush()
        if _HAS_POSIX_FADVISE:
            try:
                fd = os.open(str(self.path), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass

    def read(self) -> Tensor:
        if self._mm is None:
            raise RuntimeError(f"shard {self.path.name!r} was never written to")
        # _dirty intentionally not cleared: drain() still needs fadvise for eviction.
        self._mm.flush()
        arr = np.asarray(self._mm)
        t = torch.from_numpy(arr)
        if self._stores_bf16_as_u16:
            t = t.view(torch.bfloat16)
        return t



class MemmapBackend:
    """Default disk-backed storage. One memmap per (layer, hook).

    Usage:
        Sweep(..., storage=MemmapBackend(root=Path("/path/to/sweep_out")))

    Files land at `root/layer_{i:04d}_{hook}.bin` with a `.json` sidecar
    describing shape and dtype.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._n_samples: int | None = None
        self._shards: dict[tuple[int, str], _Shard] = {}

    def on_sweep_start(self, sweep_id: str, n_samples: int) -> None:
        self._n_samples = n_samples

    def _shard(self, layer_idx: int, hook: HookName) -> _Shard:
        if self._n_samples is None:
            raise RuntimeError(
                "MemmapBackend.on_sweep_start must be called before write_emit"
            )
        key = (layer_idx, hook)
        if key not in self._shards:
            path = self.root / f"{_shard_basename(layer_idx, hook)}.bin"
            self._shards[key] = _Shard(path, self._n_samples)
        return self._shards[key]

    def write_emit(
        self,
        layer_idx: int,
        hook: HookName,
        sample_ids: Tensor,
        tensor: Tensor,
    ) -> None:
        self._shard(layer_idx, hook).write(sample_ids, tensor)

    def read_all(self, layer_idx: int, hook: HookName) -> Tensor:
        key = (layer_idx, hook)
        if key not in self._shards:
            raise KeyError(
                f"no emits recorded for layer={layer_idx} hook={hook!r}"
            )
        return self._shards[key].read()

    def drain_emits(self) -> None:
        for shard in self._shards.values():
            shard.drain()

    def on_sweep_end(self) -> None:
        self.drain_emits()
