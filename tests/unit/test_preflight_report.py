"""PreflightReport: cost-model integration fields and summary()."""
from __future__ import annotations

from fpwap.cost_model import CostModelPrediction
from fpwap.preflight import PreflightReport


def _report_with_prediction(**overrides: object) -> PreflightReport:
    pred = CostModelPrediction(
        per_layer_s=0.82,
        total_wall_s=26.34,
        throughput_tok_s=4972.7,
        bottleneck="load",
        load_pct=0.61,
        compute_pct=0.39,
        weight_io_gb=16.0,
        prefetch=True,
    )
    defaults: dict[str, object] = dict(
        feasible=True,
        microbatch_size=64,
        residual_buffer_gb=0.5,
        per_layer_peak_vram_gb=2.1,
        estimated_wall_clock_s=26.34,
        estimated_weight_io_gb=16.0,
        loading_strategy="cpu_offload",
        prediction=pred,
        recommended_prefetch=True,
        recommended_buffer_device="cpu",
    )
    defaults.update(overrides)
    return PreflightReport(**defaults)  # type: ignore[arg-type]



def test_report_has_prediction_field() -> None:
    r = _report_with_prediction()
    assert r.prediction is not None
    assert r.prediction.bottleneck == "load"
    assert r.prediction.throughput_tok_s > 0



def test_report_has_recommended_fields() -> None:
    r = _report_with_prediction()
    assert r.recommended_prefetch is True
    assert r.recommended_buffer_device == "cpu"



def test_summary_contains_key_info() -> None:
    r = _report_with_prediction()
    s = r.summary()
    assert "feasible" in s.lower()
    assert "4,97" in s or "4972" in s  # throughput
    assert "load" in s.lower()  # bottleneck
    assert "prefetch" in s.lower()
    assert "cpu" in s.lower()  # buffer device



def test_summary_infeasible() -> None:
    r = _report_with_prediction(feasible=False, blockers=["not enough VRAM"])
    s = r.summary()
    assert "infeasible" in s.lower() or "not feasible" in s.lower()
    assert "VRAM" in s



def test_summary_without_prediction() -> None:
    """summary() works even when prediction is None (legacy preflight)."""
    r = PreflightReport(
        feasible=True,
        microbatch_size=64,
        residual_buffer_gb=0.5,
        per_layer_peak_vram_gb=0.0,
        estimated_wall_clock_s=30.0,
        estimated_weight_io_gb=16.0,
        loading_strategy="cpu_offload",
    )
    s = r.summary()
    assert "feasible" in s.lower()
    assert "30" in s
