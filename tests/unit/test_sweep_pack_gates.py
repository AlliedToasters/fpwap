"""`Sweep(pack=True)` capability + opt-in gates — phase 3 of the pack pilot.

Cheap CI-safe contract: pack=True must raise a clear error when either the
model family doesn't support packed forward, or any callback hasn't declared
that it can handle packed `acts` shapes (`RaggedTensor` instead of dense
`[mb, seq, H]`). No GPU / no Llama instantiation needed — the gates fire in
`Sweep.run()` before the heavy work.
"""
from __future__ import annotations

import pytest
import torch

from fpwap import Callback, Sweep


class _DummyDenseCallback(Callback):
    """Default-shape callback (accepts_packed left at False)."""


class _DummyPackedCallback(Callback):
    accepts_packed = True


def _gpt2_tiny() -> torch.nn.Module:
    """GPT-2 plumbing has supports_packed=False — used to test the family gate."""
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=32, n_positions=8, n_embd=16, n_layer=1, n_head=2
    )
    torch.manual_seed(0)
    m = GPT2LMHeadModel(cfg)
    m.eval()
    return m


def _llama_tiny_flex() -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=8, attn_implementation="flex_attention",
    )
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg)
    m.eval()
    return m


def _items() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]]),
        },
        {
            "input_ids": torch.tensor([[4, 5, 6, 7, 8, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        },
    ]


def test_callback_default_does_not_accept_packed() -> None:
    """`accepts_packed` defaults to False so existing callbacks stay safe."""
    cb = _DummyDenseCallback()
    assert cb.accepts_packed is False


def test_pack_true_unsupported_family_raises() -> None:
    """GPT-2 has no packed forward — pack=True must raise clearly, not silently fall back."""
    sweep = Sweep(
        model=_gpt2_tiny(),
        dataset=_items(),
        seq_len=8,
        callbacks=[_DummyPackedCallback()],
        pack=True,
        progress=False,
        transport_dtype=torch.float32,
    )
    with pytest.raises((RuntimeError, ValueError), match="(?i)pack"):
        sweep.run()


def test_pack_true_callback_not_opted_in_raises() -> None:
    """If any callback hasn't set accepts_packed=True, pack=True must raise."""
    sweep = Sweep(
        model=_llama_tiny_flex(),
        dataset=_items(),
        seq_len=8,
        callbacks=[_DummyDenseCallback()],
        pack=True,
        progress=False,
        transport_dtype=torch.float32,
    )
    with pytest.raises((RuntimeError, ValueError), match="(?i)accepts_packed"):
        sweep.run()


def test_pack_false_default_unchanged() -> None:
    """The existing Sweep API stays identical — pack defaults to False."""
    sweep = Sweep(
        model=_gpt2_tiny(),
        dataset=_items(),
        seq_len=8,
        callbacks=[_DummyDenseCallback()],
        progress=False,
        transport_dtype=torch.float32,
    )
    assert getattr(sweep, "pack", False) is False
