"""Unit tests: _Shard.read() must return a memmap-backed tensor, not a copy (#52).

MemmapBackend exists so the full [N, S, H] corpus never lives in Python
memory at once. A .copy() in read() defeats that — it materializes the
entire shard in RAM, which OOMs when the shard is bigger than host memory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fpwap.storage.memmap import _Shard


class TestShardReadZeroCopy:
    def test_float32_shares_memory(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "f32.bin", n_samples=4)
        data = torch.randn(4, 8, dtype=torch.float32)
        shard.write(torch.arange(4), data)

        t = shard.read()
        assert torch.equal(t, data)
        assert np.shares_memory(t.numpy(), np.asarray(shard._mm))

    def test_bf16_shares_memory(self, tmp_path: Path) -> None:
        shard = _Shard(tmp_path / "bf16.bin", n_samples=4)
        data = torch.randn(4, 8, dtype=torch.bfloat16)
        shard.write(torch.arange(4), data)

        t = shard.read()
        assert t.dtype == torch.bfloat16
        assert torch.equal(t, data)
        # bf16 stored as uint16 on disk; view shares the same buffer
        assert np.shares_memory(
            t.view(torch.uint16).numpy(), np.asarray(shard._mm)
        )
