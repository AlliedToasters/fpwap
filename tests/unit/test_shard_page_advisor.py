"""Unit tests for ShardPageAdvisor and safetensors offset parsing — CI-safe."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from fpwap.loader import (
    ShardPageAdvisor,
    _parse_safetensors_offsets,
    _phys_ram_bytes,
    _resident_page_budget,
)


def _make_shard(tmp_path: Path, name: str = "model.safetensors") -> Path:
    """Create a tiny safetensors shard with known tensors."""
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.v_proj.weight": torch.ones(4, 8, dtype=torch.bfloat16),
        "model.layers.1.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
    }
    path = tmp_path / name
    save_file(tensors, path)
    return path


def _make_accel_index(shard_path: Path) -> dict[str, dict]:
    """Build a minimal accel_index pointing at the given shard."""
    return {
        "model.layers.0.self_attn.q_proj.weight": {
            "safetensors_file": str(shard_path),
            "weight_name": "model.layers.0.self_attn.q_proj.weight",
            "dtype": "bfloat16",
            "shape": [4, 8],
        },
        "model.layers.0.self_attn.v_proj.weight": {
            "safetensors_file": str(shard_path),
            "weight_name": "model.layers.0.self_attn.v_proj.weight",
            "dtype": "bfloat16",
            "shape": [4, 8],
        },
        "model.layers.1.self_attn.q_proj.weight": {
            "safetensors_file": str(shard_path),
            "weight_name": "model.layers.1.self_attn.q_proj.weight",
            "dtype": "bfloat16",
            "shape": [8, 8],
        },
    }


class TestParseSafetensorsOffsets:
    def test_returns_offsets_for_all_tensors(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        offsets = _parse_safetensors_offsets(str(shard_path))
        assert "model.layers.0.self_attn.q_proj.weight" in offsets
        assert "model.layers.0.self_attn.v_proj.weight" in offsets
        assert "model.layers.1.self_attn.q_proj.weight" in offsets

    def test_offsets_are_within_file(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        file_size = shard_path.stat().st_size
        offsets = _parse_safetensors_offsets(str(shard_path))
        for _name, (start, end) in offsets.items():
            assert start >= 0
            assert end <= file_size
            assert start < end

    def test_offset_sizes_match_tensor_sizes(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        offsets = _parse_safetensors_offsets(str(shard_path))
        # 4×8 bf16 = 64 bytes
        start, end = offsets["model.layers.0.self_attn.q_proj.weight"]
        assert end - start == 4 * 8 * 2
        # 8×8 bf16 = 128 bytes
        start, end = offsets["model.layers.1.self_attn.q_proj.weight"]
        assert end - start == 8 * 8 * 2

    def test_no_metadata_key(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        offsets = _parse_safetensors_offsets(str(shard_path))
        assert "__metadata__" not in offsets


class TestShardPageAdvisor:
    def test_construction_builds_offset_map(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        index = _make_accel_index(shard_path)
        advisor = ShardPageAdvisor(index)
        assert len(advisor._offsets) == 3

    def test_advise_dontneed_calls_posix_fadvise(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        index = _make_accel_index(shard_path)
        # Disable resident caching so DONTNEED is issued eagerly (the regime
        # this test exercises); residency is covered in TestResidentPageCache.
        with patch.dict(os.environ, {"FPWAP_PAGE_RESIDENT": "0"}):
            advisor = ShardPageAdvisor(index)

        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed([
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            ])
            assert mock_fadvise.call_count == 2
            for call in mock_fadvise.call_args_list:
                fd, offset, length, advice = call.args
                assert advice == os.POSIX_FADV_DONTNEED
                assert length > 0

    def test_advise_willneed_calls_posix_fadvise(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        index = _make_accel_index(shard_path)
        advisor = ShardPageAdvisor(index)

        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_willneed([
                "model.layers.1.self_attn.q_proj.weight",
            ])
            assert mock_fadvise.call_count == 1
            _, _, _, advice = mock_fadvise.call_args.args
            assert advice == os.POSIX_FADV_WILLNEED

    def test_unknown_weight_name_is_noop(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        index = _make_accel_index(shard_path)
        advisor = ShardPageAdvisor(index)

        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed(["nonexistent.weight"])
            mock_fadvise.assert_not_called()

    def test_noop_when_posix_fadvise_unavailable(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        index = _make_accel_index(shard_path)

        with patch("fpwap.loader._HAS_POSIX_FADVISE", False):
            advisor = ShardPageAdvisor(index)
            assert len(advisor._offsets) == 0
            advisor.advise_dontneed([
                "model.layers.0.self_attn.q_proj.weight",
            ])

    def test_oserror_is_silenced(self, tmp_path: Path) -> None:
        shard_path = _make_shard(tmp_path)
        index = _make_accel_index(shard_path)
        with patch.dict(os.environ, {"FPWAP_PAGE_RESIDENT": "0"}):
            advisor = ShardPageAdvisor(index)

        with patch(
            "fpwap.loader.os.posix_fadvise",
            side_effect=OSError("not supported"),
        ):
            advisor.advise_dontneed([
                "model.layers.0.self_attn.q_proj.weight",
            ])

    def test_multi_shard_groups_by_file(self, tmp_path: Path) -> None:
        shard1 = tmp_path / "shard1.safetensors"
        shard2 = tmp_path / "shard2.safetensors"
        save_file(
            {"w0": torch.zeros(2, 2, dtype=torch.bfloat16)},
            shard1,
        )
        save_file(
            {"w1": torch.zeros(4, 4, dtype=torch.bfloat16)},
            shard2,
        )
        index = {
            "w0": {
                "safetensors_file": str(shard1),
                "weight_name": "w0",
                "dtype": "bfloat16",
                "shape": [2, 2],
            },
            "w1": {
                "safetensors_file": str(shard2),
                "weight_name": "w1",
                "dtype": "bfloat16",
                "shape": [4, 4],
            },
        }
        with patch.dict(os.environ, {"FPWAP_PAGE_RESIDENT": "0"}):
            advisor = ShardPageAdvisor(index)

        fds_opened: list[str] = []
        original_open = os.open

        def tracking_open(path, flags, *args, **kwargs):
            fds_opened.append(path)
            return original_open(path, flags, *args, **kwargs)

        with patch("fpwap.loader.os.open", side_effect=tracking_open):
            with patch("fpwap.loader.os.posix_fadvise"):
                advisor.advise_dontneed(["w0", "w1"])

        assert len(fds_opened) == 2


class TestResidentPageBudget:
    def test_default_budget_is_ram_fraction(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FPWAP_PAGE_RESIDENT", None)
            os.environ.pop("FPWAP_PAGE_RESIDENT_GB", None)
            budget = _resident_page_budget()
        assert budget == int(0.6 * _phys_ram_bytes())

    def test_env_disable_returns_zero(self) -> None:
        with patch.dict(os.environ, {"FPWAP_PAGE_RESIDENT": "0"}):
            assert _resident_page_budget() == 0

    def test_env_gb_override(self) -> None:
        with patch.dict(os.environ, {"FPWAP_PAGE_RESIDENT_GB": "2"}):
            os.environ.pop("FPWAP_PAGE_RESIDENT", None)
            assert _resident_page_budget() == 2_000_000_000

    def test_env_disable_wins_over_gb(self) -> None:
        with patch.dict(
            os.environ,
            {"FPWAP_PAGE_RESIDENT": "0", "FPWAP_PAGE_RESIDENT_GB": "8"},
        ):
            assert _resident_page_budget() == 0


class TestResidentPageCache:
    def test_within_budget_suppresses_dontneed(self, tmp_path: Path) -> None:
        index = _make_accel_index(_make_shard(tmp_path))
        advisor = ShardPageAdvisor(index)
        advisor._budget = 10_000  # bytes — far exceeds the tiny shard

        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed(["model.layers.0.self_attn.q_proj.weight"])
            mock_fadvise.assert_not_called()

        # 4×8 bf16 = 64 bytes held resident.
        assert advisor._resident_bytes == 4 * 8 * 2

    def test_resident_layer_revisit_counts_once(self, tmp_path: Path) -> None:
        # A layer re-unloaded on the next sweep stays resident without being
        # re-counted or evicted — the cross-shard reuse this feature exists for.
        index = _make_accel_index(_make_shard(tmp_path))
        advisor = ShardPageAdvisor(index)
        advisor._budget = 10_000
        names = ["model.layers.0.self_attn.q_proj.weight"]

        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed(names)
            advisor.advise_dontneed(names)
            mock_fadvise.assert_not_called()

        assert advisor._resident_bytes == 4 * 8 * 2

    def test_evicts_once_over_budget(self, tmp_path: Path) -> None:
        # Keep-prefix: the 64-byte layer0 fits and stays resident; the 128-byte
        # layer1 pushes over budget and is DONTNEED'd.
        index = _make_accel_index(_make_shard(tmp_path))
        advisor = ShardPageAdvisor(index)
        advisor._budget = 4 * 8 * 2  # exactly layer0's size

        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed(["model.layers.0.self_attn.q_proj.weight"])
            assert mock_fadvise.call_count == 0
            advisor.advise_dontneed(["model.layers.1.self_attn.q_proj.weight"])
            assert mock_fadvise.call_count == 1

        assert advisor._resident_bytes == 4 * 8 * 2

    def test_disabled_budget_always_evicts(self, tmp_path: Path) -> None:
        index = _make_accel_index(_make_shard(tmp_path))
        with patch.dict(os.environ, {"FPWAP_PAGE_RESIDENT": "0"}):
            advisor = ShardPageAdvisor(index)
        assert advisor._budget == 0

        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed(["model.layers.0.self_attn.q_proj.weight"])
            assert mock_fadvise.call_count == 1
