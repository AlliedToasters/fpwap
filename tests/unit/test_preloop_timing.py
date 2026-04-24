"""Unit tests for PreloopTiming and related ProfileReport fields."""
from __future__ import annotations

from fpwap.engine import LayerTiming, PreloopTiming, ProfileReport


def test_preloop_timing_fields() -> None:
    t = PreloopTiming(
        resolve_dataset_s=0.5,
        ensure_embedding_s=1.2,
        build_segments_s=0.3,
        storage_start_s=0.01,
        callbacks_start_s=0.02,
        total_s=2.03,
    )
    assert t.resolve_dataset_s == 0.5
    assert t.ensure_embedding_s == 1.2
    assert t.build_segments_s == 0.3
    assert t.storage_start_s == 0.01
    assert t.callbacks_start_s == 0.02
    assert t.total_s == 2.03


def test_preloop_timing_defaults() -> None:
    t = PreloopTiming()
    assert t.resolve_dataset_s == 0.0
    assert t.ensure_embedding_s == 0.0
    assert t.build_segments_s == 0.0
    assert t.storage_start_s == 0.0
    assert t.callbacks_start_s == 0.0
    assert t.total_s == 0.0


def test_profile_report_preloop_default() -> None:
    r = ProfileReport(total_wall_s=1.0, total_tokens=100)
    assert r.preloop is None
    assert r.preloop_s == 0.0


def test_profile_report_preloop_set() -> None:
    pl = PreloopTiming(total_s=2.0)
    r = ProfileReport(total_wall_s=10.0, total_tokens=100, preloop=pl)
    assert r.preloop is pl
    assert r.preloop_s == 2.0


def test_profile_report_embed_sync_default() -> None:
    r = ProfileReport(total_wall_s=1.0, total_tokens=100)
    assert r.embed_sync_s == 0.0


def test_profile_report_loop_setup_default() -> None:
    r = ProfileReport(total_wall_s=1.0, total_tokens=100)
    assert r.loop_setup_s == 0.0


def test_summary_shows_preloop_subphases() -> None:
    pl = PreloopTiming(
        resolve_dataset_s=0.5,
        ensure_embedding_s=1.2,
        build_segments_s=0.3,
        total_s=2.0,
    )
    r = ProfileReport(
        total_wall_s=20.0,
        total_tokens=1000,
        per_layer={0: LayerTiming()},
        preloop=pl,
    )
    s = r.summary()
    assert "preloop 2.000s" in s
    assert "resolve_dataset 0.500s" in s
    assert "ensure_embedding 1.200s" in s
    assert "build_segments 0.300s" in s


def test_summary_omits_zero_preloop_subphases() -> None:
    pl = PreloopTiming(ensure_embedding_s=1.0, total_s=1.0)
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={0: LayerTiming()},
        preloop=pl,
    )
    s = r.summary()
    assert "preloop 1.000s" in s
    assert "ensure_embedding 1.000s" in s
    assert "resolve_dataset" not in s
    assert "storage_start" not in s


def test_summary_omits_preloop_when_none() -> None:
    r = ProfileReport(
        total_wall_s=5.0,
        total_tokens=100,
        per_layer={0: LayerTiming()},
    )
    s = r.summary()
    assert "preloop" not in s


def test_summary_shows_embed_sync() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={0: LayerTiming()},
        embed_s=2.0,
        embed_sync_s=0.5,
    )
    s = r.summary()
    assert "embed 2.000s" in s
    assert "sync 0.500s" in s


def test_summary_shows_loop_setup() -> None:
    r = ProfileReport(
        total_wall_s=10.0,
        total_tokens=100,
        per_layer={0: LayerTiming()},
        loop_setup_s=0.05,
    )
    s = r.summary()
    assert "loop_setup 0.050s" in s
