"""Packed-layout `ResidualBuffer` — phase 2 of the pack pilot.

Adds a `layout="packed"` mode that stores `[total_real_tokens, hidden]` flat
+ `cu_seqlens [N+1]`. The hot path (`read_slice` / `write_slice` over a
contiguous sample range) translates `start, stop` through cu_seqlens — the
returned tensor is contiguous, same as dense, just with a variable
sample-range row count. Pinned host, async-D2H staging, bf16-as-u16, and
memmap lifecycle stay shared with dense.

Phase 2 deliberately leaves non-contiguous gather (`buffer[sample_ids]`)
unimplemented for packed; the engine's hot path doesn't use it. Adding it
later returns a `RaggedTensor`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fpwap.buffer import ResidualBuffer

HIDDEN = 4


class TestPackedConstruction:
    def test_default_layout_is_dense(self) -> None:
        """Existing call shape stays dense — backwards compat."""
        buf = ResidualBuffer(n_samples=3, seq_len=5, hidden=HIDDEN)
        assert buf.layout == "dense"

    def test_packed_requires_cu_seqlens(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            ResidualBuffer(n_samples=3, hidden=HIDDEN, layout="packed")

    def test_packed_storage_shape_matches_total_tokens(self) -> None:
        cu = torch.tensor([0, 2, 5, 6], dtype=torch.int64)  # lengths 2, 3, 1
        buf = ResidualBuffer(
            n_samples=3, hidden=HIDDEN, layout="packed", cu_seqlens=cu
        )
        assert buf.layout == "packed"
        assert buf.total_tokens == 6
        # Don't assert the underlying tensor's exact attribute name; instead
        # check that read_slice over the whole range returns [6, H].
        out = buf.read_slice(0, 3)
        assert out.shape == (6, HIDDEN)


class TestPackedReadWriteInMemory:
    def test_write_then_read_full_range(self) -> None:
        cu = torch.tensor([0, 2, 5, 6], dtype=torch.int64)  # lengths 2, 3, 1
        buf = ResidualBuffer(
            n_samples=3, hidden=HIDDEN, layout="packed", cu_seqlens=cu,
            dtype=torch.float32,
        )
        flat = torch.arange(6 * HIDDEN, dtype=torch.float32).reshape(6, HIDDEN)
        buf.write_slice(0, 3, flat)
        out = buf.read_slice(0, 3)
        assert torch.equal(out, flat)

    def test_write_then_read_sample_range(self) -> None:
        """Read sample-range [1, 3) → tokens [cu[1], cu[3]) = rows 2..6."""
        cu = torch.tensor([0, 2, 5, 6], dtype=torch.int64)
        buf = ResidualBuffer(
            n_samples=3, hidden=HIDDEN, layout="packed", cu_seqlens=cu,
            dtype=torch.float32,
        )
        flat = torch.arange(6 * HIDDEN, dtype=torch.float32).reshape(6, HIDDEN)
        buf.write_slice(0, 3, flat)
        out = buf.read_slice(1, 3)
        assert out.shape == (4, HIDDEN)
        assert torch.equal(out, flat[2:6])

    def test_write_sample_range_only(self) -> None:
        """write_slice over a sub-range targets the right cu_seqlens region."""
        cu = torch.tensor([0, 2, 5, 6], dtype=torch.int64)
        buf = ResidualBuffer(
            n_samples=3, hidden=HIDDEN, layout="packed", cu_seqlens=cu,
            dtype=torch.float32,
        )
        # Write samples [1, 3) — that's 4 rows (lengths 3 and 1).
        sub = torch.full((4, HIDDEN), 7.0, dtype=torch.float32)
        buf.write_slice(1, 3, sub)
        out = buf.read_slice(0, 3)
        # Sample 0 (rows 0..2) untouched (zeros), samples 1+2 (rows 2..6) = 7.
        assert torch.equal(out[0:2], torch.zeros(2, HIDDEN))
        assert torch.equal(out[2:6], sub)


class TestPackedDiskBacked:
    def test_disk_roundtrip_fp32(self, tmp_path: Path) -> None:
        cu = torch.tensor([0, 3, 4, 7], dtype=torch.int64)
        buf = ResidualBuffer(
            n_samples=3, hidden=HIDDEN, layout="packed", cu_seqlens=cu,
            dtype=torch.float32, path=tmp_path / "packed.bin",
        )
        flat = torch.randn(7, HIDDEN, dtype=torch.float32)
        buf.write_slice(0, 3, flat)
        buf.flush()
        out = buf.read_slice(0, 3)
        assert torch.equal(out, flat)

    def test_disk_roundtrip_bf16(self, tmp_path: Path) -> None:
        """bf16 stays bit-equal across the disk path (memmap stores u16 stride)."""
        cu = torch.tensor([0, 2, 5], dtype=torch.int64)
        buf = ResidualBuffer(
            n_samples=2, hidden=HIDDEN, layout="packed", cu_seqlens=cu,
            dtype=torch.bfloat16, path=tmp_path / "packed_bf16.bin",
        )
        flat = torch.randn(5, HIDDEN).to(torch.bfloat16)
        buf.write_slice(0, 2, flat)
        buf.flush()
        out = buf.read_slice(0, 2)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, flat)

    def test_disk_partial_write(self, tmp_path: Path) -> None:
        """Sub-range write on disk-backed packed buffer hits the right rows."""
        cu = torch.tensor([0, 3, 5], dtype=torch.int64)
        buf = ResidualBuffer(
            n_samples=2, hidden=HIDDEN, layout="packed", cu_seqlens=cu,
            dtype=torch.float32, path=tmp_path / "packed_part.bin",
        )
        full = torch.arange(5 * HIDDEN, dtype=torch.float32).reshape(5, HIDDEN)
        buf.write_slice(0, 2, full)
        # Overwrite only sample 1 (rows 3..5).
        sub = torch.full((2, HIDDEN), -1.0, dtype=torch.float32)
        buf.write_slice(1, 2, sub)
        buf.flush()
        out = buf.read_slice(0, 2)
        assert torch.equal(out[:3], full[:3])
        assert torch.equal(out[3:5], sub)


class TestDenseUnchanged:
    """Phase 2 must not regress dense-layout behavior — backwards compat lock."""

    def test_dense_in_memory_roundtrip(self) -> None:
        buf = ResidualBuffer(n_samples=4, seq_len=3, hidden=HIDDEN, dtype=torch.float32)
        t = torch.randn(4, 3, HIDDEN, dtype=torch.float32)
        buf.write_slice(0, 4, t)
        assert torch.equal(buf.read_slice(0, 4), t)

    def test_dense_disk_roundtrip(self, tmp_path: Path) -> None:
        buf = ResidualBuffer(
            n_samples=4, seq_len=3, hidden=HIDDEN,
            dtype=torch.bfloat16, path=tmp_path / "dense.bin",
        )
        t = torch.randn(4, 3, HIDDEN).to(torch.bfloat16)
        buf.write_slice(0, 4, t)
        buf.flush()
        out = buf.read_slice(0, 4)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, t)
