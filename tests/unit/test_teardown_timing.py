"""Unit tests for TeardownTiming sub-phase breakdown in ProfileReport."""
from __future__ import annotations

from fpwap.engine import LayerTiming, ProfileReport, TeardownTiming


def test_teardown_timing_fields() -> None:
    t = TeardownTiming(
        streamer_close_s=0.01,
        callbacks_s=0.05,
        buffer_flush_s=12.3,
        storage_flush_s=4.5,
        total_s=16.86,
    )
    assert t.streamer_close_s == 0.01
    assert t.callbacks_s == 0.05
    assert t.buffer_flush_s == 12.3
    assert t.storage_flush_s == 4.5
    assert t.total_s == 16.86


def test_teardown_timing_defaults() -> None:
    t = TeardownTiming()
    assert t.streamer_close_s == 0.0
    assert t.callbacks_s == 0.0
    assert t.buffer_flush_s == 0.0
    assert t.storage_flush_s == 0.0
    assert t.total_s == 0.0


def test_profile_report_teardown_is_teardown_timing() -> None:
    td = TeardownTiming(buffer_flush_s=5.0, total_s=5.0)
    r = ProfileReport(total_wall_s=10.0, total_tokens=100, teardown=td)
    assert r.teardown is td
    assert r.teardown_s == 5.0


def test_teardown_s_backward_compat_default() -> None:
    """teardown_s returns 0.0 when no teardown timing is attached."""
    r = ProfileReport(total_wall_s=1.0, total_tokens=100)
    assert r.teardown_s == 0.0


def test_summary_shows_teardown_subphases() -> None:
    td = TeardownTiming(
        streamer_close_s=0.01,
        callbacks_s=0.02,
        buffer_flush_s=12.0,
        storage_flush_s=4.0,
        total_s=16.03,
    )
    r = ProfileReport(
        total_wall_s=30.0,
        total_tokens=1000,
        per_layer={0: LayerTiming()},
        teardown=td,
    )
    s = r.summary()
    assert "teardown 16.030s" in s
    assert "buffer_flush 12.000s" in s
    assert "storage_flush 4.000s" in s


def test_summary_omits_zero_teardown_subphases() -> None:
    td = TeardownTiming(buffer_flush_s=5.0, total_s=5.0)
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={0: LayerTiming()},
        teardown=td,
    )
    s = r.summary()
    assert "teardown 5.000s" in s
    assert "buffer_flush 5.000s" in s
    assert "streamer_close" not in s
    assert "callbacks" not in s
    assert "storage_flush" not in s


def test_summary_omits_teardown_when_zero() -> None:
    r = ProfileReport(
        total_wall_s=5.0,
        total_tokens=100,
        per_layer={0: LayerTiming()},
    )
    s = r.summary()
    assert "teardown" not in s
