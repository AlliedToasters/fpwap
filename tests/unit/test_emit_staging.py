"""Unit tests for pinned-host staging in _Shard / MemmapBackend (#63).

The async D2H path (CUDA tensors) can only be tested on GPU hosts.
These CI-safe tests verify:
- The new max_staging_bytes parameter is accepted and wired through
- CPU tensor writes bypass staging entirely (existing behavior preserved)
- Round-trips remain correct with staging disabled (max_staging_bytes=0)
"""
from __future__ import annotations

from pathlib import Path

import torch

from fpwap.storage.memmap import MemmapBackend, _Shard


class TestShardStagingConstructor:
    def test_shard_accepts_max_staging_bytes(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "test.bin", n_samples=4, max_staging_bytes=1024)
        assert shard._max_staging_bytes == 1024
        assert shard._staging is None

    def test_shard_default_staging_zero(self, tmp_path: Path) -> None:
        """Default max_staging_bytes=0 disables staging (backward compat)."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        assert shard._max_staging_bytes == 0


class TestCpuBypassesStaging:
    def test_cpu_tensor_no_staging_allocated(self, tmp_path: Path) -> None:
        """CPU tensor writes go directly to memmap; staging is never allocated."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4, max_staging_bytes=4096)
        t = torch.randn(2, 8, dtype=torch.float32)
        shard.write(torch.tensor([0, 1]), t)
        assert shard._staging is None

    def test_cpu_tensor_no_pending(self, tmp_path: Path) -> None:
        """CPU tensor writes commit immediately — nothing pending."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4, max_staging_bytes=4096)
        t = torch.randn(2, 8, dtype=torch.float32)
        shard.write(torch.tensor([0, 1]), t)
        assert len(shard._pending) == 0


class TestRoundtripStagingDisabled:
    def test_roundtrip_float32_staging_zero(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "test.bin", n_samples=4, max_staging_bytes=0)
        t1 = torch.randn(2, 8, dtype=torch.float32)
        t2 = torch.randn(2, 8, dtype=torch.float32)
        shard.write(torch.tensor([0, 1]), t1)
        shard.write(torch.tensor([2, 3]), t2)
        shard.drain()
        got = shard.read()
        assert torch.equal(got[0:2], t1)
        assert torch.equal(got[2:4], t2)

    def test_roundtrip_bf16_staging_zero(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "test.bin", n_samples=4, max_staging_bytes=0)
        t = torch.randn(2, 8, dtype=torch.bfloat16)
        shard.write(torch.tensor([0, 1]), t)
        shard.drain()
        got = shard.read()
        assert torch.equal(got[0:2], t)


class TestBackendStagingWiring:
    def test_backend_accepts_max_staging_bytes(self, tmp_path: Path) -> None:
        backend = MemmapBackend(root=tmp_path, max_staging_bytes=4096)
        assert backend._max_staging_bytes == 4096

    def test_backend_passes_staging_to_shards(self, tmp_path: Path) -> None:
        backend = MemmapBackend(root=tmp_path, max_staging_bytes=4096)
        backend.on_sweep_start("test", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)
        backend.write_emit(0, "residual_post", torch.tensor([0, 1]), t)
        shard = backend._shards[(0, "residual_post")]
        assert shard._max_staging_bytes == 4096
