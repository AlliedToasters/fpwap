"""Unit tests for harness_adapter.compute_fingerprint — CI-safe, no model."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from harness_adapter import compute_fingerprint


def test_fingerprint_no_mask() -> None:
    acts = {0: torch.ones(2, 4, 8)}
    fp = compute_fingerprint(acts, attention_mask=None)
    assert fp == 1.0


def test_fingerprint_with_mask_excludes_pad() -> None:
    acts = {0: torch.full((2, 4, 8), 5.0)}
    acts[0][0, :2, :] = 0.0
    mask = torch.ones(2, 4, dtype=torch.long)
    mask[0, :2] = 0
    fp = compute_fingerprint(acts, attention_mask=mask)
    expected = (0 * 16 + 5.0 * 48) / 48
    assert abs(fp - expected) < 1e-6


def test_fingerprint_multi_layer() -> None:
    acts = {
        0: torch.full((1, 2, 4), 2.0),
        1: torch.full((1, 2, 4), 6.0),
    }
    fp = compute_fingerprint(acts, attention_mask=None)
    assert abs(fp - 4.0) < 1e-6


def test_fingerprint_empty() -> None:
    fp = compute_fingerprint({}, attention_mask=None)
    assert fp == 0.0


def test_fingerprint_deterministic() -> None:
    torch.manual_seed(42)
    acts = {0: torch.randn(8, 16, 32)}
    fp1 = compute_fingerprint(acts, attention_mask=None)
    fp2 = compute_fingerprint(acts, attention_mask=None)
    assert fp1 == fp2
    assert math.isfinite(fp1)
