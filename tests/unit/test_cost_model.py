"""Unit tests for cost_model — CI-safe, no GPU required."""

from __future__ import annotations

import pytest

from fpwap.cost_model import (
    CandidateConfig,
    CostModelInput,
    CostModelPrediction,
    predict,
    recommend,
)
from fpwap.preflight import PreflightReport


def _make_input(
    *,
    n_layers: int = 80,
    n_samples: int = 10_000,
    seq_len: int = 128,
    microbatch_size: int = 64,
    weight_load_s: float = 0.5,
    fwd_per_microbatch_s: float = 0.01,
    embed_s: float = 0.5,
    layer_weight_bytes: int = 1_750_000_000,
) -> CostModelInput:
    return CostModelInput(
        n_layers=n_layers,
        n_samples=n_samples,
        seq_len=seq_len,
        microbatch_size=microbatch_size,
        weight_load_s=weight_load_s,
        fwd_per_microbatch_s=fwd_per_microbatch_s,
        embed_s=embed_s,
        layer_weight_bytes=layer_weight_bytes,
    )


class TestPredict:
    def test_compute_bound(self) -> None:
        inp = _make_input(weight_load_s=0.1, fwd_per_microbatch_s=0.05)
        pred = predict(inp, prefetch=True)
        assert pred.bottleneck == "compute"
        assert pred.compute_pct > pred.load_pct

    def test_load_bound(self) -> None:
        inp = _make_input(weight_load_s=2.0, fwd_per_microbatch_s=0.001)
        pred = predict(inp, prefetch=True)
        assert pred.bottleneck == "load"
        assert pred.load_pct > pred.compute_pct

    def test_balanced(self) -> None:
        inp = _make_input(weight_load_s=1.0, fwd_per_microbatch_s=0.00625)
        n_mb = -(-10_000 // 64)
        compute = 0.00625 * n_mb
        assert abs(compute - 1.0) < 0.05
        pred = predict(inp, prefetch=True)
        assert pred.bottleneck == "balanced"

    def test_prefetch_overlaps_load(self) -> None:
        inp = _make_input(weight_load_s=1.0, fwd_per_microbatch_s=0.05)
        with_pf = predict(inp, prefetch=True)
        without_pf = predict(inp, prefetch=False)
        n_mb = -(-10_000 // 64)
        compute = 0.05 * n_mb

        assert with_pf.per_layer_s == pytest.approx(max(1.0, compute))
        assert without_pf.per_layer_s == pytest.approx(1.0 + compute)
        assert with_pf.total_wall_s < without_pf.total_wall_s

    def test_no_prefetch_sequential(self) -> None:
        inp = _make_input(weight_load_s=0.5, fwd_per_microbatch_s=0.01)
        pred = predict(inp, prefetch=False)
        n_mb = -(-10_000 // 64)
        expected_per_layer = 0.5 + 0.01 * n_mb
        assert pred.per_layer_s == pytest.approx(expected_per_layer)

    def test_throughput_calculation(self) -> None:
        inp = _make_input(n_samples=1000, seq_len=128, embed_s=0.0)
        pred = predict(inp, prefetch=False)
        expected_tokens = 1000 * 128
        assert pred.throughput_tok_s == pytest.approx(
            expected_tokens / pred.total_wall_s
        )

    def test_zero_load_preloaded(self) -> None:
        inp = _make_input(weight_load_s=0.0, fwd_per_microbatch_s=0.01)
        pred = predict(inp, prefetch=True)
        n_mb = -(-10_000 // 64)
        assert pred.per_layer_s == pytest.approx(0.01 * n_mb)
        assert pred.load_pct == 0.0
        assert pred.bottleneck == "compute"

    def test_weight_io_gb(self) -> None:
        inp = _make_input(n_layers=80, layer_weight_bytes=1_750_000_000)
        pred = predict(inp, prefetch=True)
        assert pred.weight_io_gb == pytest.approx(80 * 1.75)

    def test_ceil_div_microbatches(self) -> None:
        inp = _make_input(n_samples=100, microbatch_size=64, fwd_per_microbatch_s=1.0)
        pred = predict(inp, prefetch=False)
        assert pred.per_layer_s == pytest.approx(inp.weight_load_s + 2.0)

    def test_exact_div_microbatches(self) -> None:
        inp = _make_input(n_samples=128, microbatch_size=64, fwd_per_microbatch_s=1.0)
        pred = predict(inp, prefetch=False)
        assert pred.per_layer_s == pytest.approx(inp.weight_load_s + 2.0)

    def test_zero_total_returns_zero_throughput(self) -> None:
        inp = _make_input(
            weight_load_s=0.0, fwd_per_microbatch_s=0.0, embed_s=0.0
        )
        pred = predict(inp, prefetch=True)
        assert pred.throughput_tok_s == 0.0

    def test_prediction_has_prefetch_flag(self) -> None:
        inp = _make_input()
        assert predict(inp, prefetch=True).prefetch is True
        assert predict(inp, prefetch=False).prefetch is False

    def test_load_pct_plus_compute_pct_sums_to_one(self) -> None:
        inp = _make_input(weight_load_s=0.3, fwd_per_microbatch_s=0.02)
        pred = predict(inp, prefetch=True)
        assert pred.load_pct + pred.compute_pct == pytest.approx(1.0)


class TestRecommend:
    def test_picks_highest_throughput(self) -> None:
        fast = _make_input(weight_load_s=0.1, fwd_per_microbatch_s=0.005)
        slow = _make_input(weight_load_s=2.0, fwd_per_microbatch_s=0.05)
        rec = recommend([
            CandidateConfig(cost_input=fast, buffer_device="cpu", prefetch=True),
            CandidateConfig(cost_input=slow, buffer_device="cuda", prefetch=True),
        ])
        assert rec.buffer_device == "cpu"
        assert rec.prediction.throughput_tok_s > 0

    def test_prefetch_beats_no_prefetch(self) -> None:
        inp = _make_input(weight_load_s=1.0, fwd_per_microbatch_s=0.01)
        rec = recommend([
            CandidateConfig(cost_input=inp, buffer_device="cuda", prefetch=True),
            CandidateConfig(cost_input=inp, buffer_device="cuda", prefetch=False),
        ])
        assert rec.prefetch is True

    def test_empty_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="no candidate"):
            recommend([])

    def test_single_candidate(self) -> None:
        inp = _make_input()
        rec = recommend([
            CandidateConfig(cost_input=inp, buffer_device="cuda", prefetch=True),
        ])
        assert rec.microbatch_size == inp.microbatch_size
        assert rec.prediction.total_wall_s > 0

    def test_larger_microbatch_preferred_when_faster(self) -> None:
        small_mb = _make_input(microbatch_size=8, fwd_per_microbatch_s=0.002)
        large_mb = _make_input(microbatch_size=64, fwd_per_microbatch_s=0.01)
        rec = recommend([
            CandidateConfig(cost_input=small_mb, buffer_device="cuda", prefetch=True),
            CandidateConfig(cost_input=large_mb, buffer_device="cuda", prefetch=True),
        ])
        assert rec.microbatch_size == 64


class TestPreflightSummary:
    def test_summary_with_prediction(self) -> None:
        pred = CostModelPrediction(
            per_layer_s=1.0,
            total_wall_s=120.0,
            throughput_tok_s=1000.0,
            bottleneck="compute",
            load_pct=0.3,
            compute_pct=0.7,
            weight_io_gb=140.0,
            prefetch=True,
        )
        report = PreflightReport(
            feasible=True,
            microbatch_size=64,
            residual_buffer_gb=20.0,
            per_layer_peak_vram_gb=0.0,
            estimated_wall_clock_s=120.0,
            estimated_weight_io_gb=140.0,
            loading_strategy="disk_offload",
            prediction=pred,
            recommended_buffer_device="cpu",
            recommended_prefetch=True,
        )
        s = report.summary()
        assert "feasible" in s
        assert "microbatch_size=64" in s
        assert "2.0 min" in s
        assert "1,000 tok/s" in s
        assert "compute" in s
        assert "140.0 GB" in s
        assert "buffer_device='cpu'" in s
        assert "prefetch=on" in s

    def test_summary_without_prediction(self) -> None:
        report = PreflightReport(
            feasible=True,
            microbatch_size=32,
            residual_buffer_gb=10.0,
            per_layer_peak_vram_gb=0.0,
            estimated_wall_clock_s=60.0,
            estimated_weight_io_gb=16.0,
            loading_strategy="cpu_offload",
        )
        s = report.summary()
        assert "feasible" in s
        assert "60.0s" in s
        assert "16.0 GB" in s

    def test_summary_infeasible(self) -> None:
        report = PreflightReport(
            feasible=False,
            microbatch_size=0,
            residual_buffer_gb=0.0,
            per_layer_peak_vram_gb=0.0,
            estimated_wall_clock_s=0.0,
            estimated_weight_io_gb=0.0,
            loading_strategy="cpu_offload",
            blockers=["dataset is empty"],
        )
        s = report.summary()
        assert "INFEASIBLE" in s
        assert "dataset is empty" in s

    def test_summary_sub_minute(self) -> None:
        pred = CostModelPrediction(
            per_layer_s=0.1,
            total_wall_s=30.0,
            throughput_tok_s=5000.0,
            bottleneck="compute",
            load_pct=0.1,
            compute_pct=0.9,
            weight_io_gb=16.0,
            prefetch=True,
        )
        report = PreflightReport(
            feasible=True,
            microbatch_size=128,
            residual_buffer_gb=5.0,
            per_layer_peak_vram_gb=0.0,
            estimated_wall_clock_s=30.0,
            estimated_weight_io_gb=16.0,
            loading_strategy="cpu_offload",
            prediction=pred,
        )
        s = report.summary()
        assert "30.0s" in s
        assert "min" not in s
