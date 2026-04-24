"""Unit tests for _make_chunks — CI-safe, pure arithmetic."""
from __future__ import annotations

import pytest

from fpwap.engine import _make_chunks


def test_exact_division() -> None:
    chunks = _make_chunks(8, 4)
    assert chunks == [range(0, 4), range(4, 8)]


def test_uneven_division() -> None:
    chunks = _make_chunks(10, 3)
    assert chunks == [range(0, 3), range(3, 6), range(6, 9), range(9, 10)]


def test_chunk_size_1() -> None:
    chunks = _make_chunks(5, 1)
    assert len(chunks) == 5
    for i, c in enumerate(chunks):
        assert c == range(i, i + 1)


def test_chunk_size_exceeds_n_layers() -> None:
    chunks = _make_chunks(3, 8)
    assert chunks == [range(0, 3)]


def test_chunk_size_equals_n_layers() -> None:
    chunks = _make_chunks(4, 4)
    assert chunks == [range(0, 4)]


def test_single_layer() -> None:
    chunks = _make_chunks(1, 1)
    assert chunks == [range(0, 1)]


def test_single_layer_large_chunk() -> None:
    chunks = _make_chunks(1, 10)
    assert chunks == [range(0, 1)]


def test_all_layers_covered() -> None:
    for n in [1, 2, 5, 10, 32, 80]:
        for cs in [1, 2, 3, 4, 7, 16, 100]:
            chunks = _make_chunks(n, cs)
            all_indices = []
            for c in chunks:
                all_indices.extend(c)
            assert all_indices == list(range(n)), f"n={n}, cs={cs}"


def test_invalid_chunk_size_zero() -> None:
    with pytest.raises(ValueError):
        _make_chunks(5, 0)


def test_invalid_chunk_size_negative() -> None:
    with pytest.raises(ValueError):
        _make_chunks(5, -1)
