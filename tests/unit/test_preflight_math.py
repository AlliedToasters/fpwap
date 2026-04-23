"""Strategy-selection arithmetic from SPEC.md §12.2.

Implemented in the skeleton; tests verify the three decision branches.
"""
from __future__ import annotations

from fpwap.preflight import _select_loading_strategy

GB = 1024**3


def test_small_model_picks_cpu_offload() -> None:
    # 70B-ish weights in bf16 are ~140 GB; plenty of headroom in 512 GB RAM.
    assert (
        _select_loading_strategy(
            model_size_bytes=140 * GB,
            cpu_ram_bytes=512 * GB,
            nvme_free_bytes=2000 * GB,
        )
        == "cpu_offload"
    )


def test_model_over_ram_with_nvme_headroom_picks_disk_offload() -> None:
    # Model is 1.2x RAM (over 0.7x, under 1.5x); NVMe has >2x model size free.
    assert (
        _select_loading_strategy(
            model_size_bytes=150 * GB,
            cpu_ram_bytes=128 * GB,
            nvme_free_bytes=500 * GB,
        )
        == "disk_offload"
    )


def test_model_much_larger_than_ram_picks_mmap_from_cache() -> None:
    # 405B in bf16 (~810 GB) vs 128 GB RAM.
    assert (
        _select_loading_strategy(
            model_size_bytes=810 * GB,
            cpu_ram_bytes=128 * GB,
            nvme_free_bytes=2000 * GB,
        )
        == "mmap_from_cache"
    )


def test_disk_offload_requires_nvme_headroom() -> None:
    # Would otherwise fit the disk_offload band, but NVMe is too tight.
    assert (
        _select_loading_strategy(
            model_size_bytes=150 * GB,
            cpu_ram_bytes=128 * GB,
            nvme_free_bytes=100 * GB,
        )
        == "mmap_from_cache"
    )


def test_boundary_at_exactly_70_percent() -> None:
    # Exactly 70% of RAM should still pick cpu_offload (<= in the spec).
    assert (
        _select_loading_strategy(
            model_size_bytes=70 * GB,
            cpu_ram_bytes=100 * GB,
            nvme_free_bytes=1000 * GB,
        )
        == "cpu_offload"
    )
