"""Unit tests for SetupTiming and full-phase ProfileReport profiling."""
from __future__ import annotations

from fpwap.engine import LayerTiming, ProfileReport, SetupTiming


def test_setup_timing_fields() -> None:
    t = SetupTiming(config_s=0.1, model_s=2.5, index_s=0.3, total_s=2.9)
    assert t.config_s == 0.1
    assert t.model_s == 2.5
    assert t.index_s == 0.3
    assert t.total_s == 2.9


def test_setup_timing_defaults() -> None:
    t = SetupTiming()
    assert t.config_s == 0.0
    assert t.model_s == 0.0
    assert t.index_s == 0.0
    assert t.total_s == 0.0


def test_profile_report_setup_default_none() -> None:
    r = ProfileReport(total_wall_s=1.0, total_tokens=100)
    assert r.setup is None


def test_profile_report_phase_defaults() -> None:
    r = ProfileReport(total_wall_s=1.0, total_tokens=100)
    assert r.embed_s == 0.0
    assert r.loop_s == 0.0
    assert r.teardown_s == 0.0


def test_profile_report_setup_attached() -> None:
    setup = SetupTiming(config_s=0.1, model_s=2.0, index_s=0.5, total_s=2.6)
    r = ProfileReport(total_wall_s=5.0, total_tokens=100, setup=setup)
    assert r.setup is not None
    assert r.setup.model_s == 2.0


def test_profile_report_phases_attached() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=1000,
        embed_s=0.5,
        loop_s=8.0,
        teardown_s=0.1,
    )
    assert r.embed_s == 0.5
    assert r.loop_s == 8.0
    assert r.teardown_s == 0.1


def test_summary_includes_setup_when_present() -> None:
    setup = SetupTiming(config_s=0.1, model_s=2.0, index_s=0.5, total_s=2.6)
    r = ProfileReport(
        total_wall_s=5.0,
        total_tokens=100,
        per_layer={0: LayerTiming(load_s=0.1, forward_s=0.5)},
        setup=setup,
    )
    s = r.summary()
    assert "setup" in s
    assert "model 2.000s" in s
    assert "config 0.100s" in s
    assert "index 0.500s" in s


def test_summary_omits_setup_when_none() -> None:
    r = ProfileReport(
        total_wall_s=5.0,
        total_tokens=100,
        per_layer={0: LayerTiming(load_s=0.1, forward_s=0.5)},
    )
    s = r.summary()
    assert "setup" not in s


def test_summary_includes_embed_and_loop() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=1000,
        per_layer={0: LayerTiming()},
        embed_s=0.5,
        loop_s=8.0,
        teardown_s=0.1,
    )
    s = r.summary()
    assert "embed 0.500s" in s
    assert "loop  8.000s" in s
    assert "teardown 0.100s" in s


def test_summary_omits_zero_phases() -> None:
    r = ProfileReport(
        total_wall_s=5.0,
        total_tokens=100,
        per_layer={0: LayerTiming()},
    )
    s = r.summary()
    assert "embed" not in s
    assert "loop" not in s
    assert "teardown" not in s
