"""Unit tests for final_norm_module / final_norm_param_names on each plumbing family.

CI-safe: constructs tiny models on CPU, no GPU required.
"""
from __future__ import annotations

import torch
from torch import nn

from fpwap.models import get_plumbing
from fpwap.models.gpt2 import GPT2Plumbing
from fpwap.models.llama import LlamaPlumbing


def _make_tiny_gpt2() -> nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=40, n_positions=8, n_embd=16, n_layer=2, n_head=2
    )
    torch.manual_seed(0)
    return GPT2LMHeadModel(config).eval()


def _make_tiny_llama() -> nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


class TestGPT2FinalNorm:
    def test_returns_ln_f(self) -> None:
        model = _make_tiny_gpt2()
        plumbing = GPT2Plumbing()
        norm = plumbing.final_norm_module(model)
        assert norm is model.transformer.ln_f  # type: ignore[union-attr]

    def test_param_names(self) -> None:
        model = _make_tiny_gpt2()
        plumbing = GPT2Plumbing()
        names = plumbing.final_norm_param_names(model)
        assert "transformer.ln_f.weight" in names
        assert "transformer.ln_f.bias" in names
        for name in names:
            parts = name.split(".")
            obj = model
            for part in parts:
                obj = getattr(obj, part)
            assert isinstance(obj, (nn.Parameter, torch.Tensor))


class TestLlamaFinalNorm:
    def test_returns_norm(self) -> None:
        model = _make_tiny_llama()
        plumbing = LlamaPlumbing()
        norm = plumbing.final_norm_module(model)
        assert norm is model.model.norm  # type: ignore[union-attr]

    def test_param_names(self) -> None:
        model = _make_tiny_llama()
        plumbing = LlamaPlumbing()
        names = plumbing.final_norm_param_names(model)
        assert "model.norm.weight" in names
        for name in names:
            parts = name.split(".")
            obj = model
            for part in parts:
                obj = getattr(obj, part)
            assert isinstance(obj, (nn.Parameter, torch.Tensor))


def _make_tiny_deepseek_v2() -> nn.Module:
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

    config = DeepseekV2Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        topk_method="greedy",
        norm_topk_prob=True,
    )
    torch.manual_seed(0)
    return DeepseekV2ForCausalLM(config).eval()


class TestDeepSeekV2FinalNorm:
    def test_returns_norm(self) -> None:
        model = _make_tiny_deepseek_v2()
        plumbing = LlamaPlumbing()
        norm = plumbing.final_norm_module(model)
        assert norm is model.model.norm  # type: ignore[union-attr]

    def test_param_names(self) -> None:
        model = _make_tiny_deepseek_v2()
        plumbing = LlamaPlumbing()
        names = plumbing.final_norm_param_names(model)
        assert "model.norm.weight" in names
        for name in names:
            parts = name.split(".")
            obj = model
            for part in parts:
                obj = getattr(obj, part)
            assert isinstance(obj, (nn.Parameter, torch.Tensor))


def test_get_plumbing_dispatches_final_norm() -> None:
    gpt2 = _make_tiny_gpt2()
    llama = _make_tiny_llama()
    deepseek = _make_tiny_deepseek_v2()
    assert get_plumbing(gpt2).final_norm_module(gpt2) is not None
    assert get_plumbing(llama).final_norm_module(llama) is not None
    assert get_plumbing(deepseek).final_norm_module(deepseek) is not None
