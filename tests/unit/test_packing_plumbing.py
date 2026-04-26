"""Per-family `supports_packed` capability flag — phase 1 of the pack pilot.

The engine consults this flag before enabling `Sweep(pack=True)` for a model;
families that don't have a packed forward fall back to dense (or raise, when
pack is requested explicitly). Locked here so adding a new family forces the
implementer to declare intent.
"""
from __future__ import annotations


def test_llama_plumbing_supports_packed() -> None:
    from fpwap.models.llama import LlamaPlumbing

    assert LlamaPlumbing.supports_packed is True


def test_gpt2_plumbing_does_not_support_packed() -> None:
    from fpwap.models.gpt2 import GPT2Plumbing

    assert GPT2Plumbing.supports_packed is False


def test_protocol_declares_supports_packed() -> None:
    """Protocol must declare the attr so mypy catches new families that omit it."""
    from fpwap.models.base import ModelPlumbing

    assert "supports_packed" in ModelPlumbing.__annotations__
