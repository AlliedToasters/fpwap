"""Ragged-emit support — `Emit.sample_lengths` + `RaggedTensor` (#65).

These cover the CI-safe parts of the contract: dataclass shape, `RaggedTensor`
slicing, `_Shard` ragged round-trip, and `MemmapBackend.read_all` returning a
`RaggedTensor` for shards written with `sample_lengths`.

Engine-side dispatch (callback returning ragged `Emit` → storage receives
`sample_lengths`, pad block is bypassed) lives in tests/integration since it
needs a full sweep run.
"""
from __future__ import annotations

from pathlib import Path

import torch

from fpwap import Emit, RaggedTensor
from fpwap.storage.memmap import MemmapBackend, _Shard


class TestEmitDataclass:
    def test_sample_lengths_default_none(self) -> None:
        e = Emit(tensor=torch.zeros(2, 3, 4))
        assert e.sample_lengths is None

    def test_sample_lengths_field_accepts_tensor(self) -> None:
        flat = torch.zeros(7, 4)
        lengths = torch.tensor([3, 4], dtype=torch.int64)
        e = Emit(tensor=flat, sample_lengths=lengths)
        assert e.sample_lengths is lengths


class TestRaggedTensor:
    def test_len_and_getitem(self) -> None:
        flat = torch.arange(7 * 4, dtype=torch.float32).reshape(7, 4)
        offsets = torch.tensor([0, 3, 7], dtype=torch.int64)
        rt = RaggedTensor(flat=flat, offsets=offsets)
        assert len(rt) == 2
        assert torch.equal(rt[0], flat[0:3])
        assert torch.equal(rt[1], flat[3:7])

    def test_lengths_property(self) -> None:
        flat = torch.zeros(10, 2)
        offsets = torch.tensor([0, 2, 5, 10], dtype=torch.int64)
        rt = RaggedTensor(flat=flat, offsets=offsets)
        assert torch.equal(rt.lengths, torch.tensor([2, 3, 5], dtype=torch.int64))


class TestShardRagged:
    def test_ragged_write_roundtrip_single_microbatch(self, tmp_path: Path) -> None:
        """One microbatch covers all samples; ragged round-trip equals input."""
        shard = _Shard(tmp_path / "rag.bin", n_samples=3)
        # samples 0,1,2 have lengths 2, 4, 1 — total 7 rows of width 4
        flat = torch.arange(7 * 4, dtype=torch.float32).reshape(7, 4)
        ids = torch.tensor([0, 1, 2])
        lengths = torch.tensor([2, 4, 1], dtype=torch.int64)

        shard.write(ids, flat, sample_lengths=lengths)
        shard.drain()
        rt = shard.read()
        assert isinstance(rt, RaggedTensor)
        assert torch.equal(rt.lengths, lengths)
        assert torch.equal(rt[0], flat[0:2])
        assert torch.equal(rt[1], flat[2:6])
        assert torch.equal(rt[2], flat[6:7])

    def test_ragged_multi_microbatch_arrives_out_of_order(self, tmp_path: Path) -> None:
        """Two microbatches arrive — second covers earlier sample ids; read
        result must be in sample-id order."""
        shard = _Shard(tmp_path / "rag.bin", n_samples=4)
        # Microbatch 1: samples [2, 3], lengths [1, 3]
        mb1 = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
        shard.write(
            torch.tensor([2, 3]),
            mb1,
            sample_lengths=torch.tensor([1, 3], dtype=torch.int64),
        )
        # Microbatch 2: samples [0, 1], lengths [2, 2]
        mb2 = (torch.arange(4 * 2, dtype=torch.float32) + 100).reshape(4, 2)
        shard.write(
            torch.tensor([0, 1]),
            mb2,
            sample_lengths=torch.tensor([2, 2], dtype=torch.int64),
        )
        shard.drain()

        rt = shard.read()
        assert isinstance(rt, RaggedTensor)
        assert torch.equal(rt.lengths, torch.tensor([2, 2, 1, 3], dtype=torch.int64))
        # sample 0: rows 0:2 of mb2
        assert torch.equal(rt[0], mb2[0:2])
        # sample 1: rows 2:4 of mb2
        assert torch.equal(rt[1], mb2[2:4])
        # sample 2: row 0 of mb1
        assert torch.equal(rt[2], mb1[0:1])
        # sample 3: rows 1:4 of mb1
        assert torch.equal(rt[3], mb1[1:4])

    def test_ragged_bf16_roundtrip(self, tmp_path: Path) -> None:
        """bf16 stays bit-equal across the ragged disk path."""
        shard = _Shard(tmp_path / "rag_bf16.bin", n_samples=2)
        flat = torch.randn(5, 8).to(torch.bfloat16)
        shard.write(
            torch.tensor([0, 1]),
            flat,
            sample_lengths=torch.tensor([2, 3], dtype=torch.int64),
        )
        shard.drain()
        rt = shard.read()
        assert rt.flat.dtype == torch.bfloat16
        assert torch.equal(rt[0], flat[0:2])
        assert torch.equal(rt[1], flat[2:5])

    def test_dense_after_ragged_rejected(self, tmp_path: Path) -> None:
        """Once a shard is ragged, a subsequent dense write is an error."""
        shard = _Shard(tmp_path / "mode.bin", n_samples=2)
        shard.write(
            torch.tensor([0]),
            torch.zeros(2, 4),
            sample_lengths=torch.tensor([2], dtype=torch.int64),
        )
        try:
            shard.write(torch.tensor([1]), torch.zeros(1, 4))
        except (ValueError, RuntimeError):
            return
        raise AssertionError("expected an error when mixing dense + ragged on one shard")

    def test_ragged_after_dense_rejected(self, tmp_path: Path) -> None:
        """Mixing modes the other direction is also an error."""
        shard = _Shard(tmp_path / "mode2.bin", n_samples=2)
        shard.write(torch.tensor([0]), torch.zeros(1, 4))
        try:
            shard.write(
                torch.tensor([1]),
                torch.zeros(2, 4),
                sample_lengths=torch.tensor([2], dtype=torch.int64),
            )
        except (ValueError, RuntimeError):
            return
        raise AssertionError("expected an error when mixing dense + ragged on one shard")


class TestMemmapBackendRagged:
    def test_write_emit_sample_lengths_kwarg(self, tmp_path: Path) -> None:
        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=3)

        flat = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        backend.write_emit(
            layer_idx=0,
            hook="residual_post",
            sample_ids=torch.tensor([0, 1, 2]),
            tensor=flat,
            sample_lengths=torch.tensor([1, 2, 3], dtype=torch.int64),
        )
        backend.on_sweep_end()

        rt = backend.read_all(0, "residual_post")
        assert isinstance(rt, RaggedTensor)
        assert len(rt) == 3
        assert torch.equal(rt.lengths, torch.tensor([1, 2, 3], dtype=torch.int64))
        assert torch.equal(rt[0], flat[0:1])
        assert torch.equal(rt[1], flat[1:3])
        assert torch.equal(rt[2], flat[3:6])

    def test_dense_path_still_returns_tensor(self, tmp_path: Path) -> None:
        """Backwards-compat: write_emit without sample_lengths returns Tensor."""
        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=2)
        t = torch.randn(2, 4, 8)
        backend.write_emit(0, "residual_post", torch.tensor([0, 1]), t)
        backend.on_sweep_end()
        out = backend.read_all(0, "residual_post")
        assert isinstance(out, torch.Tensor)
        assert torch.equal(out, t)

    def test_ragged_raw_scratch_dropped_after_sweep_end(self, tmp_path: Path) -> None:
        """After on_sweep_end the .raw.bin scratch must not survive on disk."""
        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=2)
        backend.write_emit(
            0,
            "residual_post",
            torch.tensor([0, 1]),
            torch.zeros(3, 4),
            sample_lengths=torch.tensor([1, 2], dtype=torch.int64),
        )
        # Mid-sweep drain: raw scratch may still be present.
        backend.drain_emits()
        backend.on_sweep_end()

        raw_files = list(tmp_path.glob("*.raw.bin"))
        assert raw_files == [], f"orphaned raw scratch: {raw_files}"
        # Final .bin and sidecar must exist.
        assert list(tmp_path.glob("*.bin")) != []
        assert list(tmp_path.glob("*.json")) != []

    def test_write_after_finalize_raises_clearly(self, tmp_path: Path) -> None:
        """A write after sweep_end / finalize() must raise a clear error,
        not corrupt mid-pipeline. Avoids confusing failures from the missing
        raw scratch file."""
        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=2)
        backend.write_emit(
            0,
            "residual_post",
            torch.tensor([0]),
            torch.zeros(1, 4),
            sample_lengths=torch.tensor([1], dtype=torch.int64),
        )
        backend.on_sweep_end()

        try:
            backend.write_emit(
                0,
                "residual_post",
                torch.tensor([1]),
                torch.zeros(2, 4),
                sample_lengths=torch.tensor([2], dtype=torch.int64),
            )
        except RuntimeError as exc:
            assert "finalized" in str(exc).lower()
            return
        raise AssertionError("expected RuntimeError on write after finalize()")

    def test_ragged_drain_does_not_rebuild_final(self, tmp_path: Path) -> None:
        """drain() for ragged shards must NOT call _build_final_ragged.

        Rebuilding on every chunk-boundary drain is O(N_microbatches × bytes)
        wasted I/O on multi-layer-capture sweeps. Final rebuild is deferred
        to read() / finalize().
        """
        from unittest.mock import patch

        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=2)
        backend.write_emit(
            0,
            "residual_post",
            torch.tensor([0, 1]),
            torch.zeros(3, 4),
            sample_lengths=torch.tensor([1, 2], dtype=torch.int64),
        )

        shard = backend._shards[(0, "residual_post")]
        with patch.object(shard, "_build_final_ragged") as mock_build:
            backend.drain_emits()
            mock_build.assert_not_called()

        # finalize at sweep_end must build exactly once.
        with patch.object(
            shard, "_build_final_ragged", wraps=shard._build_final_ragged
        ) as mock_build:
            backend.on_sweep_end()
            mock_build.assert_called_once()

    def test_ragged_sidecar_records_layout(self, tmp_path: Path) -> None:
        """Sidecar JSON must record ragged layout so the file is self-describing."""
        import json

        backend = MemmapBackend(root=tmp_path)
        backend.on_sweep_start("test", n_samples=2)
        backend.write_emit(
            0,
            "residual_post",
            torch.tensor([0, 1]),
            torch.zeros(3, 4),
            sample_lengths=torch.tensor([1, 2], dtype=torch.int64),
        )
        backend.on_sweep_end()

        meta_path = next(tmp_path.glob("*.json"))
        meta = json.loads(meta_path.read_text())
        assert meta.get("layout") == "ragged"
        assert meta.get("n_samples") == 2
