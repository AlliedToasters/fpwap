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

import errno
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fpwap.types import HookName, RaggedTensor, ResultArtifact

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
    """One memmap file + its per-row shape/dtype, lazily sized.

    Two layouts, decided on first write and locked thereafter:

    * **dense** — `[n_samples, *per_row_shape]`. Default, writes by sample_id.
    * **ragged** (#65) — variable-length per sample. On write, the flat
      `[sum(sample_lengths), *trailing]` chunk is appended to a temporary
      `.raw.bin` and a per-arrival metadata row is recorded. At drain/read
      time the data is reordered into sample-id order in the final `.bin`,
      and `read()` returns a `RaggedTensor`.
    """

    def __init__(
        self, path: Path, n_samples: int, max_staging_bytes: int = 0
    ) -> None:
        self.path = path
        self.n_samples = n_samples
        self._mm: np.memmap | None = None
        self.per_row_shape: tuple[int, ...] | None = None
        self.torch_dtype: torch.dtype | None = None
        self._np_dtype: Any = None
        self._stores_bf16_as_u16: bool = False
        self._dirty: bool = False

        self._max_staging_bytes = max_staging_bytes
        self._staging: Tensor | None = None
        self._max_staging_rows: int = 0
        self._staging_cursor: int = 0
        self._pending: list[tuple[np.ndarray, int, int]] = []

        # Ragged-layout state (None until first write determines layout).
        self._layout: str | None = None  # "dense" | "ragged"
        self._raw_path: Path = path.with_suffix(".raw.bin")
        self._raw_mm: np.memmap | None = None
        self._raw_capacity_rows: int = 0
        self._raw_cursor: int = 0
        self._sample_raw_offsets: np.ndarray | None = None  # int64[n_samples]
        self._sample_lengths_arr: np.ndarray | None = None  # int64[n_samples]
        self._sample_written: np.ndarray | None = None  # bool[n_samples]
        self._final_built: bool = False
        self._final_offsets: np.ndarray | None = None
        self._finalized: bool = False

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

    def _ensure_staging(self, tensor: Tensor) -> Tensor:
        if self._staging is not None:
            return self._staging
        per_row_bytes = tensor[0:1].nelement() * tensor.element_size()
        self._max_staging_rows = max(1, self._max_staging_bytes // per_row_bytes)
        shape = (self._max_staging_rows, *tensor.shape[1:])
        self._staging = torch.zeros(shape, dtype=tensor.dtype, pin_memory=True)
        return self._staging

    def write(
        self,
        sample_ids: Tensor,
        tensor: Tensor,
        sample_lengths: Tensor | None = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError(
                f"shard {self.path.name!r} was finalized (sweep ended); "
                "no further writes accepted. The .raw.bin scratch was "
                "dropped — re-running the sweep is the only path forward."
            )
        if sample_lengths is not None:
            self._enter_ragged()
            self._write_ragged(sample_ids, tensor, sample_lengths)
            return
        if self._layout == "ragged":
            raise RuntimeError(
                f"shard {self.path.name!r} was written ragged; cannot mix in "
                "dense writes (omit sample_lengths once a shard is ragged)"
            )
        self._layout = "dense"

        mm = self._ensure(tensor)
        ids_np = sample_ids.detach().to(device="cpu", dtype=torch.int64).numpy()

        if (
            tensor.device.type == "cuda"
            and self._max_staging_bytes > 0
        ):
            n_rows = tensor.shape[0]
            staging = self._ensure_staging(tensor)

            if n_rows > self._max_staging_rows:
                host = tensor.detach().to(device="cpu")
                if self._stores_bf16_as_u16:
                    host = host.view(torch.uint16)
                mm[ids_np] = host.numpy()
                self._dirty = True
                return

            if self._staging_cursor + n_rows > self._max_staging_rows:
                self._drain_staging()

            start = self._staging_cursor
            stop = start + n_rows
            staging[start:stop].copy_(tensor, non_blocking=True)
            self._pending.append((ids_np, start, stop))
            self._staging_cursor = stop
        else:
            host = tensor.detach().to(device="cpu")
            if self._stores_bf16_as_u16:
                host = host.view(torch.uint16)
            mm[ids_np] = host.numpy()

        self._dirty = True

    def _enter_ragged(self) -> None:
        if self._layout == "dense":
            raise RuntimeError(
                f"shard {self.path.name!r} was written dense; cannot mix in "
                "ragged writes (must supply sample_lengths from the first write)"
            )
        if self._layout is None:
            self._layout = "ragged"
            self._sample_raw_offsets = np.full(self.n_samples, -1, dtype=np.int64)
            self._sample_lengths_arr = np.zeros(self.n_samples, dtype=np.int64)
            self._sample_written = np.zeros(self.n_samples, dtype=bool)

    def _ensure_raw(self, tensor: Tensor, n_rows: int) -> np.memmap:
        """Reserve raw-file capacity for `n_rows` more rows.

        Raw file holds chunks in arrival order (out-of-order by sample_id);
        reordering happens at drain/read.
        """
        if self.per_row_shape is None:
            self.torch_dtype = tensor.dtype
            self.per_row_shape = tuple(tensor.shape[1:])
            np_dtype = _TORCH_TO_NUMPY.get(tensor.dtype)
            if np_dtype is None:
                self._stores_bf16_as_u16 = True
                np_dtype = np.uint16
            self._np_dtype = np_dtype

        needed = self._raw_cursor + n_rows
        if self._raw_mm is not None and needed <= self._raw_capacity_rows:
            return self._raw_mm

        grow = self._raw_capacity_rows * 2 if self._raw_capacity_rows else n_rows * 4
        new_capacity = max(needed, grow)
        if self._raw_mm is not None:
            self._raw_mm.flush()
            del self._raw_mm
            self._raw_mm = None
        shape = (new_capacity, *self.per_row_shape)
        if self._raw_path.exists():
            self._raw_mm = np.memmap(
                self._raw_path, dtype=self._np_dtype, mode="r+", shape=shape
            )
        else:
            self._raw_mm = np.memmap(
                self._raw_path, dtype=self._np_dtype, mode="w+", shape=shape
            )
        self._raw_capacity_rows = new_capacity
        return self._raw_mm

    def _write_ragged(
        self, sample_ids: Tensor, tensor: Tensor, sample_lengths: Tensor
    ) -> None:
        ids_np = sample_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        lengths_np = sample_lengths.detach().to(device="cpu", dtype=torch.int64).numpy()
        if int(lengths_np.sum()) != int(tensor.shape[0]):
            raise ValueError(
                f"sum(sample_lengths)={int(lengths_np.sum())} but tensor has "
                f"{int(tensor.shape[0])} rows"
            )
        if ids_np.shape[0] != lengths_np.shape[0]:
            raise ValueError(
                f"sample_ids ({ids_np.shape[0]}) and sample_lengths "
                f"({lengths_np.shape[0]}) differ"
            )

        host = tensor.detach().to(device="cpu")
        if self._stores_bf16_as_u16 or host.dtype == torch.bfloat16:
            self._stores_bf16_as_u16 = True
            host = host.view(torch.uint16)

        n_rows = int(host.shape[0])
        raw_mm = self._ensure_raw(tensor, n_rows)
        write_start = self._raw_cursor
        raw_mm[write_start : write_start + n_rows] = host.numpy()
        self._raw_cursor += n_rows

        cursor = write_start
        assert self._sample_raw_offsets is not None
        assert self._sample_lengths_arr is not None
        assert self._sample_written is not None
        for i in range(ids_np.shape[0]):
            sid = int(ids_np[i])
            length = int(lengths_np[i])
            if self._sample_written[sid]:
                raise RuntimeError(
                    f"sample {sid} written twice to ragged shard {self.path.name!r}"
                )
            self._sample_raw_offsets[sid] = cursor
            self._sample_lengths_arr[sid] = length
            self._sample_written[sid] = True
            cursor += length

        self._dirty = True
        self._final_built = False

    def _drain_staging(self) -> None:
        if not self._pending:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        assert self._staging is not None
        assert self._mm is not None
        for ids_np, start, stop in self._pending:
            host = self._staging[start:stop]
            if self._stores_bf16_as_u16:
                host = host.view(torch.uint16)
            self._mm[ids_np] = host.numpy()
        self._pending.clear()
        self._staging_cursor = 0

    def drain(self) -> None:
        """Flush dirty pages to disk and evict from page cache.

        Called at chunk boundaries instead of per-microbatch (#61).
        """
        if not self._dirty:
            return
        if self._layout == "ragged":
            # Drain only flushes the arrival-order .raw.bin and advises the
            # kernel to reclaim its pages. The sample-id-ordered final .bin
            # is built lazily on read() / finalize() — rebuilding on every
            # chunk-boundary drain is O(N_microbatches × total_bytes) wasted
            # I/O on multi-layer-capture sweeps.
            if self._raw_mm is not None:
                self._raw_mm.flush()
                if _HAS_POSIX_FADVISE:
                    try:
                        fd = os.open(str(self._raw_path), os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        finally:
                            os.close(fd)
                    except OSError:
                        pass
            self._dirty = False
            return
        self._drain_staging()
        self._flush_and_evict()
        self._dirty = False

    def _build_final_ragged(self) -> None:
        """Reorder raw arrival-order data into sample-id order on the final .bin.

        Called lazily on read() / finalize(). Writes the per-sample sidecar
        with the final offsets so the file is self-describing.
        """
        if self._final_built:
            return
        if (
            self._raw_mm is None
            or self._sample_lengths_arr is None
            or self._sample_raw_offsets is None
            or self._sample_written is None
            or self.per_row_shape is None
        ):
            raise RuntimeError(
                f"shard {self.path.name!r} has no ragged data to materialize"
            )

        self._raw_mm.flush()

        lengths = self._sample_lengths_arr
        offsets = np.zeros(self.n_samples + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])

        shape = (total, *self.per_row_shape)
        # Reuse the existing final file if it's already the right size; only
        # truncate (mode="w+") on first build or when total grew.
        if (
            self._mm is not None
            and self.path.exists()
            and self._mm.shape == shape
        ):
            final_mm = np.memmap(
                self.path, dtype=self._np_dtype, mode="r+", shape=shape
            )
        else:
            if self._mm is not None:
                self._mm.flush()
                del self._mm
                self._mm = None
            final_mm = np.memmap(
                self.path, dtype=self._np_dtype, mode="w+", shape=shape
            )
        for sid in range(self.n_samples):
            length = int(lengths[sid])
            if length == 0:
                continue
            raw_off = int(self._sample_raw_offsets[sid])
            final_mm[int(offsets[sid]) : int(offsets[sid]) + length] = (
                self._raw_mm[raw_off : raw_off + length]
            )
        final_mm.flush()
        self._mm = final_mm

        torch_dtype_label = "bfloat16" if self._stores_bf16_as_u16 else (
            str(self.torch_dtype).removeprefix("torch.") if self.torch_dtype else "unknown"
        )
        meta_path = self.path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {
                    "layout": "ragged",
                    "n_samples": self.n_samples,
                    "per_row_shape": list(self.per_row_shape),
                    "dtype": torch_dtype_label,
                    "bf16_as_u16": self._stores_bf16_as_u16,
                    "offsets": offsets.tolist(),
                }
            )
        )
        self._final_offsets = offsets
        self._final_built = True

    def finalize(self) -> None:
        """Build the final sample-id-ordered .bin and drop the raw scratch file.

        Called once at sweep end (no further writes expected). Subsequent
        writes raise — by design, since the raw scratch is gone.
        """
        if self._layout != "ragged":
            return
        if self._raw_mm is None:
            self._finalized = True
            return
        self._build_final_ragged()
        # Drop the raw scratch file: ~doubles disk usage if left around
        # (raw + final coexist for the rest of the sweep).
        self._raw_mm.flush()
        del self._raw_mm
        self._raw_mm = None
        self._raw_capacity_rows = 0
        try:
            self._raw_path.unlink()
        except OSError:
            pass
        self._finalized = True

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

    def read(self) -> Tensor | RaggedTensor:
        if self._layout == "ragged":
            self._build_final_ragged()
            assert self._mm is not None
            assert self._final_offsets is not None
            self._mm.flush()
            arr = np.asarray(self._mm)
            flat = torch.from_numpy(arr)
            if self._stores_bf16_as_u16:
                flat = flat.view(torch.bfloat16)
            offsets = torch.from_numpy(self._final_offsets.copy())
            return RaggedTensor(flat=flat, offsets=offsets)

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

    def __init__(
        self,
        root: Path | str,
        max_staging_bytes: int = 2 * 1024**3,
    ) -> None:
        """
        Args:
            root: Directory for memmap shard files.
            max_staging_bytes: Per-shard pinned-host staging budget for async
                GPU→CPU copies. Total pinned memory = n_captured_shards × this
                value. Set to 0 to disable staging (synchronous writes).
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._n_samples: int | None = None
        self._shards: dict[tuple[int, str], _Shard] = {}
        self._max_staging_bytes = max_staging_bytes

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
            self._shards[key] = _Shard(
                path, self._n_samples, self._max_staging_bytes
            )
        return self._shards[key]

    def write_emit(
        self,
        layer_idx: int,
        hook: HookName,
        sample_ids: Tensor,
        tensor: Tensor,
        sample_lengths: Tensor | None = None,
    ) -> None:
        self._shard(layer_idx, hook).write(sample_ids, tensor, sample_lengths)

    def read_all(self, layer_idx: int, hook: HookName) -> Tensor | RaggedTensor:
        """Read the full corpus for one (layer, hook).

        Returns:
            * `Tensor` of shape `[N_samples, *per_row_shape]` for shards
              written via dense `Emit(tensor=...)`. The tensor is backed by
              the on-disk memmap, so the OS page cache is the budget — the
              full corpus does not need to fit in RAM.
            * `RaggedTensor(flat, offsets)` for shards written with
              `Emit(tensor=..., sample_lengths=...)`. `flat[offsets[i]:
              offsets[i+1]]` is sample `i`'s contribution. `flat` is also
              memmap-backed.

        The return type is a union — callers iterating over multiple
        (layer, hook) pairs that may mix layouts must dispatch on
        `isinstance(result, RaggedTensor)`.
        """
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
        # No more writes after sweep end: build the sample-id-ordered final
        # file for each ragged shard and drop its .raw.bin scratch.
        for shard in self._shards.values():
            shard.finalize()

    def path_for(
        self,
        layer_idx: int,
        hook: HookName,
        dest: Path | None = None,
    ) -> ResultArtifact:
        """Return an on-disk handle for one (layer, hook) — issue #70.

        Skips the materialize-into-host-RAM round-trip in `read_all`. Useful
        when the caller already plans to mmap-read or move the file
        (cumulative emits across sweeps that won't fit in RAM).
        """
        key = (layer_idx, hook)
        if key not in self._shards:
            raise KeyError(
                f"no emits recorded for layer={layer_idx} hook={hook!r}"
            )
        shard = self._shards[key]

        # Make the on-disk artifact readable: drain pending staged writes,
        # and for ragged build the final sample-id-ordered .bin + offsets
        # sidecar. Mid-sweep callers may hit this before on_sweep_end.
        shard.drain()
        if shard._layout == "ragged":
            shard._build_final_ragged()
        elif shard._mm is None:
            raise RuntimeError(
                f"shard {shard.path.name!r} was never written to"
            )
        else:
            shard._mm.flush()

        data_path: Path = shard.path
        sidecar_path: Path | None = shard.path.with_suffix(".json")
        layout: str = shard._layout or "dense"
        assert shard.torch_dtype is not None
        dtype = shard.torch_dtype
        shape: tuple[int, ...] | None
        if layout == "dense":
            assert shard.per_row_shape is not None
            shape = (shard.n_samples, *shard.per_row_shape)
        else:
            shape = None

        if dest is not None:
            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=True)
            data_path = _link_or_copy(shard.path, dest / shard.path.name)
            src_sidecar = shard.path.with_suffix(".json")
            sidecar_path = _link_or_copy(
                src_sidecar, dest / src_sidecar.name
            ) if src_sidecar.exists() else None

        return ResultArtifact(
            data_path=data_path,
            sidecar_path=sidecar_path,
            layout=layout,  # type: ignore[arg-type]
            dtype=dtype,
            shape=shape,
        )


def _link_or_copy(src: Path, dst: Path) -> Path:
    """Hardlink src → dst; fall back to copy on EXDEV (cross-filesystem).

    Hardlink is the cheap path: backend keeps writing into its file, the
    user holds a separate path with the same inode. If the destination
    is on another filesystem the kernel rejects the link with EXDEV; copy
    is the only option there.
    """
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(src, dst)
    return dst
