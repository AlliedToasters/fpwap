"""GPU tests for pinned-host staging in _Shard (#63).

Verifies that CUDA tensors flow through the async staging path and
round-trip correctly, including backpressure when staging overflows.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fpwap.storage.memmap import _Shard

pytestmark = pytest.mark.gpu


@pytest.fixture
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestCudaStagingPath:
    def test_cuda_write_populates_pending(
        self, tmp_path: Path, cuda_available
    ) -> None:
        """CUDA write enqueues to staging — memmap not written yet."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4, max_staging_bytes=1 << 20)
        t = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        shard.write(torch.tensor([0, 1]), t)
        assert len(shard._pending) == 1
        assert shard._staging is not None
        assert shard._staging.is_pinned()

    def test_cuda_roundtrip(self, tmp_path: Path, cuda_available) -> None:
        """CUDA write → drain → read produces correct data."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4, max_staging_bytes=1 << 20)
        t1 = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        t2 = torch.randn(2, 8, dtype=torch.float32, device="cuda")
        shard.write(torch.tensor([0, 1]), t1)
        shard.write(torch.tensor([2, 3]), t2)
        shard.drain()
        got = shard.read()
        assert torch.equal(got[0:2], t1.cpu())
        assert torch.equal(got[2:4], t2.cpu())

    def test_bf16_cuda_roundtrip(self, tmp_path: Path, cuda_available) -> None:
        shard = _Shard(tmp_path / "test.bin", n_samples=4, max_staging_bytes=1 << 20)
        t = torch.randn(2, 8, dtype=torch.bfloat16, device="cuda")
        shard.write(torch.tensor([0, 1]), t)
        shard.drain()
        got = shard.read()
        assert torch.equal(got[0:2], t.cpu())


class TestStagingBackpressure:
    def test_backpressure_correctness(
        self, tmp_path: Path, cuda_available
    ) -> None:
        """Overflow staging capacity; all data still round-trips correctly."""
        per_row_bytes = 8 * 4  # 8 floats × 4 bytes
        max_staging = per_row_bytes * 3  # room for 3 rows
        shard = _Shard(
            tmp_path / "test.bin", n_samples=8, max_staging_bytes=max_staging
        )
        expected = {}
        for i in range(8):
            t = torch.randn(1, 8, dtype=torch.float32, device="cuda")
            shard.write(torch.tensor([i]), t)
            expected[i] = t.cpu()
        shard.drain()
        got = shard.read()
        for i, exp in expected.items():
            assert torch.equal(got[i : i + 1], exp), f"row {i} mismatch"

    def test_backpressure_forces_drain(
        self, tmp_path: Path, cuda_available
    ) -> None:
        """When staging is full, forced drain clears pending before next write."""
        per_row_bytes = 8 * 4
        max_staging = per_row_bytes * 2  # room for 2 rows only
        shard = _Shard(
            tmp_path / "test.bin", n_samples=6, max_staging_bytes=max_staging
        )
        t = torch.randn(1, 8, dtype=torch.float32, device="cuda")
        shard.write(torch.tensor([0]), t)
        shard.write(torch.tensor([1]), t)
        assert shard._staging_cursor == 2
        # Third write should trigger forced drain, resetting cursor
        shard.write(torch.tensor([2]), t)
        assert shard._staging_cursor == 1  # reset + 1 new row
