"""Unit tests for chunk-boundary emit drain (#61).

Emit writes should NOT flush+fadvise per microbatch. Instead, writes
accumulate in page cache and a single drain() at chunk boundary does
the flush + DONTNEED eviction.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from fpwap.storage.memmap import MemmapBackend, _Shard


class TestShardDeferredFlush:
    def test_write_does_not_flush(self, tmp_path: Path) -> None:
        """_Shard.write() must NOT call _flush_and_evict."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)
        ids = torch.tensor([0, 2])

        with patch.object(shard, "_flush_and_evict") as mock_flush:
            shard.write(ids, t)
            mock_flush.assert_not_called()

    def test_drain_calls_flush_and_evict(self, tmp_path: Path) -> None:
        """_Shard.drain() should call _flush_and_evict exactly once."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)
        shard.write(torch.tensor([0, 1]), t)
        shard.write(torch.tensor([2, 3]), t)

        with patch.object(shard, "_flush_and_evict") as mock_flush:
            shard.drain()
            mock_flush.assert_called_once()

    def test_drain_noop_when_no_writes(self, tmp_path: Path) -> None:
        """drain() on a never-written shard is a no-op (no crash)."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        shard.drain()

    def test_roundtrip_correct_after_drain(self, tmp_path: Path) -> None:
        """Data written + drained round-trips correctly through read()."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t1 = torch.randn(2, 8, dtype=torch.float32)
        t2 = torch.randn(2, 8, dtype=torch.float32)
        shard.write(torch.tensor([0, 1]), t1)
        shard.write(torch.tensor([2, 3]), t2)
        shard.drain()
        got = shard.read()
        assert torch.equal(got[0:2], t1)
        assert torch.equal(got[2:4], t2)

    def test_fadvise_only_on_drain(self, tmp_path: Path) -> None:
        """posix_fadvise is called once on drain, not per write."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)

        with patch("fpwap.storage.memmap.os.posix_fadvise") as mock_fadvise:
            shard.write(torch.tensor([0, 1]), t)
            shard.write(torch.tensor([2, 3]), t)
            assert mock_fadvise.call_count == 0
            shard.drain()
            assert mock_fadvise.call_count == 1


class TestMemmapBackendDrainEmits:
    def test_drain_emits_drains_all_shards(self, tmp_path: Path) -> None:
        """drain_emits() calls drain() on every shard."""
        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)

        backend.write_emit(0, "residual_post", torch.tensor([0, 1]), t)
        backend.write_emit(1, "residual_post", torch.tensor([0, 1]), t)

        with patch("fpwap.storage.memmap.os.posix_fadvise") as mock_fadvise:
            backend.drain_emits()
            assert mock_fadvise.call_count == 2  # one per shard

    def test_write_emit_no_fadvise(self, tmp_path: Path) -> None:
        """write_emit() must not trigger fadvise (deferred to drain)."""
        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)

        with patch("fpwap.storage.memmap.os.posix_fadvise") as mock_fadvise:
            backend.write_emit(0, "residual_post", torch.tensor([0, 1]), t)
            backend.write_emit(0, "residual_post", torch.tensor([2, 3]), t)
            assert mock_fadvise.call_count == 0

    def test_on_sweep_end_includes_drain(self, tmp_path: Path) -> None:
        """on_sweep_end() must drain before final flush."""
        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)
        backend.write_emit(0, "residual_post", torch.tensor([0, 1]), t)

        with patch("fpwap.storage.memmap.os.posix_fadvise") as mock_fadvise:
            backend.on_sweep_end()
            assert mock_fadvise.call_count == 1
