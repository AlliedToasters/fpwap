"""Unit tests for bench_warm_start.summarize_timings — CI-safe, no model."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from bench_warm_start import summarize_timings


def test_summarize_basic() -> None:
    stats = summarize_timings(setup_s=1.0, sweep_times=[2.0, 3.0, 4.0])
    assert stats["n_sweeps"] == 3.0
    assert stats["setup_s"] == 1.0
    assert stats["total_run_s"] == 9.0
    assert stats["total_s"] == 10.0
    assert stats["median_sweep_s"] == 3.0
    assert abs(stats["mean_sweep_s"] - 3.0) < 1e-9
    assert stats["min_sweep_s"] == 2.0
    assert stats["max_sweep_s"] == 4.0
    assert abs(stats["setup_fraction"] - 0.1) < 1e-9


def test_summarize_single_sweep() -> None:
    stats = summarize_timings(setup_s=0.5, sweep_times=[1.5])
    assert stats["n_sweeps"] == 1.0
    assert stats["median_sweep_s"] == 1.5
    assert stats["mean_sweep_s"] == 1.5
    assert stats["total_s"] == 2.0
    assert abs(stats["setup_fraction"] - 0.25) < 1e-9


def test_summarize_zero_setup() -> None:
    stats = summarize_timings(setup_s=0.0, sweep_times=[1.0, 2.0])
    assert stats["setup_s"] == 0.0
    assert stats["setup_fraction"] == 0.0
    assert stats["total_s"] == 3.0


def test_summarize_empty_sweeps() -> None:
    stats = summarize_timings(setup_s=1.0, sweep_times=[])
    assert stats["n_sweeps"] == 0.0
    assert stats["median_sweep_s"] == 0.0
    assert stats["total_run_s"] == 0.0
    assert stats["total_s"] == 1.0
    assert abs(stats["setup_fraction"] - 1.0) < 1e-9


def test_summarize_even_count_median() -> None:
    stats = summarize_timings(setup_s=0.0, sweep_times=[1.0, 2.0, 3.0, 4.0])
    assert stats["median_sweep_s"] == 2.5


def test_summarize_identical_sweeps() -> None:
    stats = summarize_timings(setup_s=2.0, sweep_times=[1.0] * 15)
    assert stats["n_sweeps"] == 15.0
    assert stats["median_sweep_s"] == 1.0
    assert stats["mean_sweep_s"] == 1.0
    assert stats["min_sweep_s"] == 1.0
    assert stats["max_sweep_s"] == 1.0
    assert stats["total_run_s"] == 15.0
    assert stats["total_s"] == 17.0
