"""Unit tests for harness_adapter.compute_fingerprint — CI-safe, no model."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from harness_adapter import compute_fingerprint


def test_fingerprint_no_mask_shape() -> None:
    pooled = {0: torch.ones(2, 8)}
    fp = compute_fingerprint(pooled)
    assert isinstance(fp, torch.Tensor)
    assert fp.shape == (2, 8)


def test_fingerprint_no_mask_values() -> None:
    pooled = {0: torch.ones(2, 8)}
    fp = compute_fingerprint(pooled)
    assert torch.allclose(fp, torch.ones(2, 8))


def test_fingerprint_multi_layer_shape() -> None:
    pooled = {
        0: torch.full((3, 4), 2.0),
        1: torch.full((3, 4), 6.0),
    }
    fp = compute_fingerprint(pooled)
    assert fp.shape == (3, 8)


def test_fingerprint_multi_layer_concat() -> None:
    pooled = {
        0: torch.full((1, 4), 2.0),
        1: torch.full((1, 4), 6.0),
    }
    fp = compute_fingerprint(pooled)
    expected = torch.tensor([[2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0]])
    assert torch.allclose(fp, expected)


def test_fingerprint_empty() -> None:
    fp = compute_fingerprint({})
    assert fp.numel() == 0


def test_fingerprint_deterministic() -> None:
    torch.manual_seed(42)
    pooled = {0: torch.randn(8, 32)}
    fp1 = compute_fingerprint(pooled)
    fp2 = compute_fingerprint(pooled)
    assert torch.equal(fp1, fp2)
