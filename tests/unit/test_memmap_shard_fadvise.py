"""Unit tests for posix_fadvise(DONTNEED) on MemmapBackend emit shards (#50)."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import torch

from fpwap.storage.memmap import MemmapBackend, _Shard


class TestShardFlushAndEvict:
    def test_fadvise_called_on_drain(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)
        ids = torch.tensor([0, 2])

        with patch("fpwap.storage.memmap.os.posix_fadvise") as mock_fadvise:
            shard.write(ids, t)
            assert mock_fadvise.call_count == 0
            shard.drain()
            assert mock_fadvise.call_count == 1
            fd, offset, length, advice = mock_fadvise.call_args.args
            assert offset == 0
            assert length == 0
            assert advice == os.POSIX_FADV_DONTNEED

    def test_multiple_writes_single_drain(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t1 = torch.randn(2, 8, dtype=torch.float32)
        t2 = torch.randn(2, 8, dtype=torch.float32)

        with patch("fpwap.storage.memmap.os.posix_fadvise") as mock_fadvise:
            shard.write(torch.tensor([0, 1]), t1)
            shard.write(torch.tensor([2, 3]), t2)
            assert mock_fadvise.call_count == 0
            shard.drain()
            assert mock_fadvise.call_count == 1

    def test_noop_when_posix_fadvise_unavailable(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)

        with patch("fpwap.storage.memmap._HAS_POSIX_FADVISE", False):
            shard.write(torch.tensor([0, 1]), t)
            shard.drain()

        got = shard.read()
        assert torch.equal(got[0:2], t)

    def test_oserror_is_silenced(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)

        shard.write(torch.tensor([0, 1]), t)
        with patch(
            "fpwap.storage.memmap.os.posix_fadvise",
            side_effect=OSError("not supported"),
        ):
            shard.drain()

        got = shard.read()
        assert torch.equal(got[0:2], t)

    def test_roundtrip_still_correct(self, tmp_path: Path) -> None:
        """Write + drain eviction doesn't corrupt data on read-back."""
        shard = _Shard(tmp_path / "test.bin", n_samples=4)
        t1 = torch.randn(2, 8, dtype=torch.float32)
        t2 = torch.randn(2, 8, dtype=torch.float32)
        shard.write(torch.tensor([0, 1]), t1)
        shard.write(torch.tensor([2, 3]), t2)
        shard.drain()
        got = shard.read()
        assert torch.equal(got[0:2], t1)
        assert torch.equal(got[2:4], t2)


class TestMemmapBackendFadvise:
    def test_drain_emits_triggers_fadvise(self, tmp_path: Path) -> None:
        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=4)
        t = torch.randn(2, 8, dtype=torch.float32)

        with patch("fpwap.storage.memmap.os.posix_fadvise") as mock_fadvise:
            backend.write_emit(0, "residual_post", torch.tensor([0, 1]), t)
            assert mock_fadvise.call_count == 0
            backend.drain_emits()
            assert mock_fadvise.call_count == 1
