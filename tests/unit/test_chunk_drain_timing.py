"""Unit tests for per-chunk drain_sync and unload timing in ProfileReport."""
from __future__ import annotations

from fpwap.engine import LayerTiming, ProfileReport


def test_profile_report_drain_sync_default() -> None:
    r = ProfileReport(total_wall_s=1.0, total_tokens=100)
    assert r.drain_sync_s == 0.0


def test_profile_report_unload_default() -> None:
    r = ProfileReport(total_wall_s=1.0, total_tokens=100)
    assert r.unload_s == 0.0


def test_profile_report_drain_sync_set() -> None:
    r = ProfileReport(total_wall_s=10.0, total_tokens=100, drain_sync_s=5.0)
    assert r.drain_sync_s == 5.0


def test_profile_report_unload_set() -> None:
    r = ProfileReport(total_wall_s=10.0, total_tokens=100, unload_s=2.0)
    assert r.unload_s == 2.0


def test_summary_shows_drain_sync_when_nonzero() -> None:
    r = ProfileReport(
        total_wall_s=30.0,
        total_tokens=1000,
        per_layer={0: LayerTiming()},
        loop_s=20.0,
        drain_sync_s=8.0,
        unload_s=1.5,
    )
    s = r.summary()
    assert "drain_sync 8.000s" in s
    assert "unload 1.500s" in s


def test_summary_omits_drain_sync_when_zero() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={0: LayerTiming()},
        loop_s=5.0,
    )
    s = r.summary()
    assert "drain_sync" not in s
    assert "unload" not in s
