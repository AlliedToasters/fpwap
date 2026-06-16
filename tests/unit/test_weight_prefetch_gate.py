from __future__ import annotations

import pytest

from fpwap.engine import _weight_prefetch_enabled


def test_weight_prefetch_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FPWAP_PREFETCH_LOAD", raising=False)

    assert _weight_prefetch_enabled() is False


def test_weight_prefetch_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPWAP_PREFETCH_LOAD", "1")

    assert _weight_prefetch_enabled() is True


def test_weight_prefetch_false_spellings(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("0", "false", "False"):
        monkeypatch.setenv("FPWAP_PREFETCH_LOAD", value)

        assert _weight_prefetch_enabled() is False
