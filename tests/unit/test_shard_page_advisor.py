"""Unit tests for ShardPageAdvisor and safetensors offset parsing — CI-safe."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

import fpwap.loader as loader
from fpwap.loader import (
    _LIBC,
    ShardPageAdvisor,
    _parse_safetensors_offsets,
    _phys_ram_bytes,
    _resident_page_budget,
    release_resident_pages,
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


@pytest.mark.skipif(_LIBC is None, reason="needs libc mmap/mlock/mincore")
class TestResidentPagePinning:
    """FPWAP_PAGE_RESIDENT_PIN: make residency binding via mlock so the kernel
    cannot reclaim the resident set under memory pressure (issue #21)."""

    @pytest.fixture(autouse=True)
    def _clear_global_pins(self):
        release_resident_pages()
        loader._WARNED.clear()
        yield
        release_resident_pages()
        loader._WARNED.clear()

    def _pin_advisor(self, tmp_path: Path, budget_gb: str = "1") -> ShardPageAdvisor:
        index = _make_accel_index(_make_shard(tmp_path))
        with patch.dict(
            os.environ,
            {
                "FPWAP_PAGE_RESIDENT": "1",
                "FPWAP_PAGE_RESIDENT_GB": budget_gb,
                "FPWAP_PAGE_RESIDENT_PIN": "1",
            },
        ):
            return ShardPageAdvisor(index)

    def test_pin_enabled_mlocks_and_is_resident(self, tmp_path: Path) -> None:
        advisor = self._pin_advisor(tmp_path)
        assert advisor._pin_enabled
        name = "model.layers.0.self_attn.q_proj.weight"
        # No DONTNEED in the resident branch; instead the bytes get mlock'd.
        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed([name])
            mock_fadvise.assert_not_called()
        assert frozenset([name]) in advisor._resident_keys
        assert loader._PINNED_REGIONS, "pin must mmap+mlock the layer's bytes globally"
        # mincore: the pinned bytes are fully resident.
        path, start, end = advisor._offsets[name]
        resident, total = advisor._mincore_resident_pages(
            advisor._pin_fd(path), start, end
        )
        assert total and resident == total
        advisor.close()

    def test_pins_persist_across_advisors(self, tmp_path: Path) -> None:
        # The crux of issue #21: a second sweep's advisor reuses the first's
        # pins instead of re-locking — so the resident set survives the
        # per-unit Sweep teardown rather than being rebuilt every unit.
        name = "model.layers.0.self_attn.q_proj.weight"
        a1 = self._pin_advisor(tmp_path)
        a1.advise_dontneed([name])
        n_after_first = len(loader._PINNED_REGIONS)
        total_after_first = loader._PINNED_TOTAL
        a1.close()  # streamer teardown must NOT release the global pins
        assert loader._PINNED_REGIONS, "close() must keep global pins resident"

        a2 = self._pin_advisor(tmp_path)
        a2.advise_dontneed([name])  # same range → reused, not re-locked
        assert len(loader._PINNED_REGIONS) == n_after_first
        assert loader._PINNED_TOTAL == total_after_first
        a2.close()

    def test_close_keeps_global_pins_drops_fds(self, tmp_path: Path) -> None:
        advisor = self._pin_advisor(tmp_path)
        advisor.advise_dontneed(["model.layers.0.self_attn.q_proj.weight"])
        assert loader._PINNED_REGIONS
        advisor.close()
        assert loader._PINNED_REGIONS  # globals survive teardown
        assert advisor._pin_fds == {}
        release_resident_pages()
        assert not loader._PINNED_REGIONS  # explicit release tears them down

    def test_pin_disabled_keeps_legacy_advisory(self, tmp_path: Path) -> None:
        index = _make_accel_index(_make_shard(tmp_path))
        advisor = ShardPageAdvisor(index)  # PIN unset → advisory-only
        advisor._budget = 10_000
        assert not advisor._pin_enabled
        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed(["model.layers.0.self_attn.q_proj.weight"])
            mock_fadvise.assert_not_called()
        assert not loader._PINNED_REGIONS  # tracked resident, but never mlock'd

    def test_memlock_limit_clamps_budget(self, tmp_path: Path) -> None:
        # rlimit below the requested budget → clamp + warn, pin stays enabled.
        index = _make_accel_index(_make_shard(tmp_path))
        with patch("fpwap.loader._memlock_limit_bytes", return_value=(4 << 30)):
            with patch.dict(
                os.environ,
                {
                    "FPWAP_PAGE_RESIDENT": "1",
                    "FPWAP_PAGE_RESIDENT_GB": "8",  # 8 GB requested
                    "FPWAP_PAGE_RESIDENT_PIN": "1",
                },
            ):
                advisor = ShardPageAdvisor(index)
        assert advisor._pin_enabled
        # clamped to rlimit minus the pinned-buffer headroom
        assert advisor._budget == (4 << 30) - (2 << 30)

    def test_pin_failure_falls_back_to_dontneed(self, tmp_path: Path) -> None:
        advisor = self._pin_advisor(tmp_path)
        name = "model.layers.0.self_attn.q_proj.weight"
        with patch.object(advisor, "_pin", return_value=False):
            with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
                advisor.advise_dontneed([name])
                assert mock_fadvise.call_count == 1  # evicted, not pretended resident
        assert frozenset([name]) not in advisor._resident_keys
        assert advisor._warned_evict  # one-time exhaustion note
        advisor.close()

    def test_mincore_watchdog_warns_on_eviction(self, tmp_path: Path, capsys) -> None:
        # Advisory-only mode: when the resident set has been reclaimed, the
        # periodic mincore check emits a one-time I/O-bound warning.
        index = _make_accel_index(_make_shard(tmp_path))
        advisor = ShardPageAdvisor(index)
        advisor._budget = 10_000
        name = "model.layers.0.self_attn.q_proj.weight"
        advisor.advise_dontneed([name])  # marks resident (advise_calls=1)
        from fpwap.loader import _RESIDENCY_CHECK_EVERY

        advisor._advise_calls = _RESIDENCY_CHECK_EVERY - 1
        with patch.object(advisor, "_mincore_resident_pages", return_value=(0, 10)):
            advisor.advise_dontneed([name])  # tick → check fires, sees 0% resident
        out = capsys.readouterr().out
        assert "re-faulting" in out
        assert advisor._warned_evict
