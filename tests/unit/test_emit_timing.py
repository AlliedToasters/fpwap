"""Unit tests for emit_s field on LayerTiming and summary rendering."""
from __future__ import annotations

from fpwap.engine import LayerTiming, ProfileReport


def test_layer_timing_emit_s_default() -> None:
    t = LayerTiming()
    assert t.emit_s == 0.0


def test_layer_timing_emit_s_set() -> None:
    t = LayerTiming(emit_s=1.5)
    assert t.emit_s == 1.5


def test_summary_shows_emit_s() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={
            0: LayerTiming(
                load_s=0.1, forward_s=0.5, callback_s=0.2,
                write_s=1.0, emit_s=0.8,
            ),
        },
    )
    s = r.summary()
    assert "emit 0.800s" in s


def test_summary_omits_zero_emit() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={0: LayerTiming(forward_s=0.5)},
    )
    s = r.summary()
    assert "emit" not in s


def test_by_phase_includes_emit() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={
            0: LayerTiming(emit_s=0.5),
            1: LayerTiming(emit_s=0.3),
        },
    )
    phases = r.by_phase()
    assert "emit" in phases
    assert phases["emit"] == [0.5, 0.3]


def test_slowest_layer_considers_emit() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={
            0: LayerTiming(forward_s=0.1),
            1: LayerTiming(emit_s=5.0),
        },
    )
    idx, phase = r.slowest_layer()
    assert idx == 1
    assert phase == "emit"
