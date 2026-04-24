"""ProfileReport throughput + bandwidth math (unit-only, no GPU, no model)."""
from __future__ import annotations

from fpwap.engine import LayerTiming, ProfileReport


def test_throughput_zero_wall_clock_is_zero() -> None:
    r = ProfileReport(total_wall_s=0.0, total_tokens=1_000)
    assert r.throughput_tok_per_s() == 0.0


def test_throughput_simple_ratio() -> None:
    r = ProfileReport(total_wall_s=2.0, total_tokens=10_000)
    assert r.throughput_tok_per_s() == 5_000.0


def test_bandwidth_sums_per_layer_weight_bytes() -> None:
    r = ProfileReport(
        total_wall_s=1.0,
        per_layer={
            0: LayerTiming(bytes_weights=2_000_000_000),
            1: LayerTiming(bytes_weights=3_000_000_000),
        },
    )
    assert r.weight_bandwidth_gb_per_s() == 5.0


def test_summary_includes_throughput_line() -> None:
    r = ProfileReport(
        total_wall_s=2.0,
        total_tokens=2_000,
        per_layer={0: LayerTiming(load_s=0.1, forward_s=0.5)},
    )
    s = r.summary()
    assert "throughput" in s
    assert "1,000.0 tok/s" in s
    assert "tokens 2000" in s


def test_peak_vram_gb() -> None:
    r = ProfileReport(
        per_layer={
            0: LayerTiming(peak_vram_bytes=10_000_000_000),
            1: LayerTiming(peak_vram_bytes=12_000_000_000),
            2: LayerTiming(peak_vram_bytes=11_000_000_000),
        },
    )
    assert r.peak_vram_gb() == 12.0


def test_peak_vram_in_summary() -> None:
    r = ProfileReport(
        total_wall_s=1.0,
        total_tokens=1_000,
        per_layer={
            0: LayerTiming(
                load_s=0.1, forward_s=0.5, peak_vram_bytes=8_500_000_000
            ),
        },
    )
    s = r.summary()
    assert "peak VRAM 8.5 GB" in s
