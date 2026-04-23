"""Unit tests for hub-id → local snapshot-dir resolution.

`Sweep(model=<hub-id>)` needs to resolve the id via the HF cache without
the consumer having to call `snapshot_download` themselves. This is a
pure-function test on the resolver — no network, no real weights — so
we can gate the contract in CI without pulling checkpoints.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_existing_local_directory_returned_as_is(tmp_path: Path) -> None:
    from fpwap.loader import resolve_snapshot_dir

    snap = tmp_path / "my_snapshot"
    snap.mkdir()
    resolved = resolve_snapshot_dir(str(snap))
    assert resolved == snap


def test_hub_id_uses_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-directory string → calls snapshot_download(..., local_files_only=True)."""
    from fpwap import loader as loader_mod

    fake_snapshot = tmp_path / "cached" / "llama-3.3-70b"
    fake_snapshot.mkdir(parents=True)

    calls: list[tuple[str, bool]] = []

    def fake_snapshot_download(repo_id: str, local_files_only: bool = False, **_: object) -> str:
        calls.append((repo_id, local_files_only))
        return str(fake_snapshot)

    monkeypatch.setattr(loader_mod, "snapshot_download", fake_snapshot_download)

    resolved = loader_mod.resolve_snapshot_dir("meta-llama/Llama-3.3-70B-Instruct")

    assert resolved == fake_snapshot
    assert calls == [("meta-llama/Llama-3.3-70B-Instruct", True)]


def test_hub_id_not_cached_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model isn't in the HF cache, the error must mention the
    model id and tell the user how to pre-cache. Opaque errors from deep
    inside huggingface_hub have burned users before."""
    from fpwap import loader as loader_mod

    def raising_snapshot_download(repo_id: str, **_: object) -> str:
        raise FileNotFoundError(f"not cached: {repo_id}")

    monkeypatch.setattr(loader_mod, "snapshot_download", raising_snapshot_download)

    with pytest.raises(FileNotFoundError) as excinfo:
        loader_mod.resolve_snapshot_dir("meta-llama/Llama-3.3-70B-Instruct")

    msg = str(excinfo.value)
    assert "meta-llama/Llama-3.3-70B-Instruct" in msg
    assert "hf" in msg.lower() or "huggingface" in msg.lower() or "snapshot_download" in msg


def test_sweep_string_model_with_explicit_snapshot_dir_wins(tmp_path: Path) -> None:
    """Explicit snapshot_dir kwarg wins over implicit resolution. This
    preserves the pre-change behavior for consumers who already pass
    snapshot_dir, and lets users override when their cache layout is
    non-standard."""
    from fpwap import Sweep

    explicit = tmp_path / "explicit"
    explicit.mkdir()

    sweep = Sweep(
        model="meta-llama/Llama-3.3-70B-Instruct",
        dataset=[],
        seq_len=4,
        callbacks=[],
        execution_device="cpu",
        snapshot_dir=str(explicit),
    )
    assert sweep.snapshot_dir == str(explicit)
