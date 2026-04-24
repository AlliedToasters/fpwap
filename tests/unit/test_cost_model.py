"""Cost model: pure arithmetic, CI-safe.

Tests cover predict() under three regimes (compute-bound, load-bound,
balanced) and the prefetch overlap model. recommend() picks highest
throughput from a candidate set.
"""
from __future__ import annotations

import math

import pytest

from fpwap.cost_model import CostModelInput, predict, recommend

# ---------------------------------------------------------------------------
# Fixtures: representative inputs
# ---------------------------------------------------------------------------

def _base_input(**overrides: float | int) -> CostModelInput:
    defaults: dict[str, float | int] = dict(
        n_layers=32,
        n_samples=1024,
        seq_len=128,
        microbatch_size=64,
        weight_load_s=0.5,
        fwd_per_microbatch_s=0.02,
        embed_s=0.1,
        layer_weight_bytes=500_000_000,
    )
    defaults.update(overrides)
    return CostModelInput(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# predict() — no prefetch
# ---------------------------------------------------------------------------


def test_predict_compute_bound_no_prefetch() -> None:
    """When compute >> load, per-layer ≈ compute time (no prefetch)."""
    inp = _base_input(weight_load_s=0.01, fwd_per_microbatch_s=0.1)
    pred = predict(inp, prefetch=False)

    n_mb = math.ceil(1024 / 64)  # 16
    expected_compute = 0.1 * n_mb  # 1.6
    expected_per_layer = 0.01 + expected_compute  # no overlap → 1.61

    assert pred.per_layer_s == pytest.approx(expected_per_layer, rel=1e-6)
    assert pred.bottleneck == "compute"
    assert pred.compute_pct > pred.load_pct



def test_predict_load_bound_no_prefetch() -> None:
    """When load >> compute, per-layer ≈ load time (no prefetch)."""
    inp = _base_input(weight_load_s=2.0, fwd_per_microbatch_s=0.001)
    pred = predict(inp, prefetch=False)

    n_mb = math.ceil(1024 / 64)
    expected_compute = 0.001 * n_mb  # 0.016
    expected_per_layer = 2.0 + expected_compute

    assert pred.per_layer_s == pytest.approx(expected_per_layer, rel=1e-6)
    assert pred.bottleneck == "load"
    assert pred.load_pct > pred.compute_pct



def test_predict_balanced_no_prefetch() -> None:
    """When load ≈ compute, bottleneck is 'balanced'."""
    n_mb = math.ceil(1024 / 64)  # 16
    fwd = 0.5 / n_mb  # 0.03125 → compute = 0.5
    inp = _base_input(weight_load_s=0.5, fwd_per_microbatch_s=fwd)
    pred = predict(inp, prefetch=False)

    assert pred.bottleneck == "balanced"
    assert pred.load_pct == pytest.approx(pred.compute_pct, rel=0.1)


# ---------------------------------------------------------------------------
# predict() — with prefetch
# ---------------------------------------------------------------------------


def test_predict_prefetch_overlaps_load() -> None:
    """With prefetch, per_layer = max(load, compute), not sum."""
    inp = _base_input(weight_load_s=0.5, fwd_per_microbatch_s=0.1)
    pred = predict(inp, prefetch=True)

    n_mb = math.ceil(1024 / 64)
    compute = 0.1 * n_mb  # 1.6
    expected_per_layer = max(0.5, compute)  # 1.6

    assert pred.per_layer_s == pytest.approx(expected_per_layer, rel=1e-6)
    assert pred.prefetch is True



def test_predict_prefetch_load_dominant() -> None:
    """When load > compute even with prefetch, load sets the pace."""
    inp = _base_input(weight_load_s=5.0, fwd_per_microbatch_s=0.001)
    pred = predict(inp, prefetch=True)

    n_mb = math.ceil(1024 / 64)
    compute = 0.001 * n_mb
    expected_per_layer = max(5.0, compute)  # 5.0

    assert pred.per_layer_s == pytest.approx(expected_per_layer, rel=1e-6)
    assert pred.bottleneck == "load"


# ---------------------------------------------------------------------------
# predict() — total wall clock and throughput
# ---------------------------------------------------------------------------


def test_predict_total_wall_and_throughput() -> None:
    inp = _base_input(weight_load_s=0.5, fwd_per_microbatch_s=0.02)
    pred = predict(inp, prefetch=False)

    n_mb = math.ceil(1024 / 64)
    compute = 0.02 * n_mb  # 0.32
    per_layer = 0.5 + compute  # 0.82
    total = 0.1 + per_layer * 32  # embed + layers
    total_tokens = 1024 * 128

    assert pred.total_wall_s == pytest.approx(total, rel=1e-6)
    assert pred.throughput_tok_s == pytest.approx(total_tokens / total, rel=1e-6)



def test_predict_weight_io_gb() -> None:
    inp = _base_input(layer_weight_bytes=500_000_000)
    pred = predict(inp, prefetch=False)
    expected_io = 500_000_000 * 32 / 1e9
    assert pred.weight_io_gb == pytest.approx(expected_io, rel=1e-6)


# ---------------------------------------------------------------------------
# recommend()
# ---------------------------------------------------------------------------


def test_recommend_picks_highest_throughput() -> None:
    """recommend() should pick the candidate with highest throughput."""
    slow = _base_input(weight_load_s=2.0, fwd_per_microbatch_s=0.1)
    fast = _base_input(weight_load_s=0.1, fwd_per_microbatch_s=0.02)

    rec = recommend([
        (slow, False),
        (slow, True),
        (fast, False),
        (fast, True),
    ])

    assert rec.prediction.throughput_tok_s > 0
    assert rec.prefetch is True
    assert rec.input is fast



def test_recommend_single_candidate() -> None:
    """With one candidate, recommend() returns it regardless."""
    inp = _base_input()
    rec = recommend([(inp, False)])
    assert rec.input is inp
    assert rec.prefetch is False



def test_recommend_empty_raises() -> None:
    with pytest.raises(ValueError):
        recommend([])
