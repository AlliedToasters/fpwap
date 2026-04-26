"""ResultArtifact dataclass shape — CI-safe (no GPU, no model).

Issue #70 introduces a small handle returned by `Result.activations(..., as_path=True)`
that bundles the data path with sidecar/dtype/shape metadata so callers can
mmap-read on demand instead of materializing a host-RAM tensor.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_result_artifact_dense_fields(tmp_path: Path) -> None:
    from fpwap import ResultArtifact

    data = tmp_path / "layer_0001_residual_post.bin"
    sidecar = tmp_path / "layer_0001_residual_post.json"
    data.touch()
    sidecar.touch()

    art = ResultArtifact(
        data_path=data,
        sidecar_path=sidecar,
        layout="dense",
        dtype=torch.bfloat16,
        shape=(10_000, 128, 8192),
    )
    assert art.data_path == data
    assert art.sidecar_path == sidecar
    assert art.layout == "dense"
    assert art.dtype == torch.bfloat16
    assert art.shape == (10_000, 128, 8192)


def test_result_artifact_ragged_shape_is_none(tmp_path: Path) -> None:
    """Ragged emits don't have a uniform [N, ...] shape — sample lengths
    differ. The sidecar JSON carries per-sample offsets; `shape` is None
    so callers know to use the sidecar / `RaggedTensor.lengths`."""
    from fpwap import ResultArtifact

    art = ResultArtifact(
        data_path=tmp_path / "x.bin",
        sidecar_path=tmp_path / "x.json",
        layout="ragged",
        dtype=torch.float32,
        shape=None,
    )
    assert art.layout == "ragged"
    assert art.shape is None


def test_result_artifact_is_frozen(tmp_path: Path) -> None:
    """Frozen so callers can use it as a stable handle / dict key."""
    from dataclasses import FrozenInstanceError

    from fpwap import ResultArtifact

    art = ResultArtifact(
        data_path=tmp_path / "x.bin",
        sidecar_path=None,
        layout="dense",
        dtype=torch.float32,
        shape=(1, 1),
    )
    with pytest.raises(FrozenInstanceError):
        art.data_path = tmp_path / "y.bin"  # type: ignore[misc]
