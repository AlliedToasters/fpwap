"""Cross-family parity: Mistral and Qwen2 ride on LlamaPlumbing's structural matcher.

LlamaPlumbing.matches() fires on any `{model}.model.{layers, embed_tokens,
rotary_emb}` layout — Mistral and Qwen2 both satisfy this and their
decoder-block forward signatures are byte-identical to Llama's. These
tests lock that coverage in so a future HF refactor to either family
fails loudly rather than silently drifting.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 8
HIDDEN = 32
INTER = 64
N_LAYERS = 2
N_HEAD = 4
N_KV_HEAD = 2
VOCAB = 64


def _capture_per_block(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[
    int, torch.Tensor
]:
    captured: dict[int, torch.Tensor] = {}

    def make_hook(i: int):
        def _h(_m, _i, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out).detach().clone()

        return _h

    handles = [
        b.register_forward_hook(make_hook(i)) for i, b in enumerate(model.model.layers)
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    return captured


def _run_fpwap(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[
    int, torch.Tensor
]:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    raw = RawActivations(layers="all", last_token_only=False, out_dtype=torch.float32)
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    )
    result = sweep.run()
    return {i: result.activations(layer=i, hook="residual_post") for i in range(N_LAYERS)}


@pytest.mark.integration
def test_mistral_matches_naive() -> None:
    from transformers import MistralConfig, MistralForCausalLM

    from fpwap.models import LlamaPlumbing, get_plumbing

    config = MistralConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEAD,
        num_key_value_heads=N_KV_HEAD,
        max_position_embeddings=SEQ_LEN,
        sliding_window=None,  # disable SWA for simple verification
    )
    torch.manual_seed(SEED)
    model = MistralForCausalLM(config)
    model.eval()

    assert isinstance(get_plumbing(model), LlamaPlumbing)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    expected = _capture_per_block(model, input_ids)
    got = _run_fpwap(model, input_ids)

    for i in range(N_LAYERS):
        assert torch.equal(got[i], expected[i].float()), (
            f"mistral layer {i}: max diff {(got[i] - expected[i].float()).abs().max().item()}"
        )


@pytest.mark.integration
def test_qwen2_matches_naive() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from fpwap.models import LlamaPlumbing, get_plumbing

    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEAD,
        num_key_value_heads=N_KV_HEAD,
        max_position_embeddings=SEQ_LEN,
    )
    torch.manual_seed(SEED)
    model = Qwen2ForCausalLM(config)
    model.eval()

    assert isinstance(get_plumbing(model), LlamaPlumbing)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    expected = _capture_per_block(model, input_ids)
    got = _run_fpwap(model, input_ids)

    for i in range(N_LAYERS):
        assert torch.equal(got[i], expected[i].float()), (
            f"qwen2 layer {i}: max diff {(got[i] - expected[i].float()).abs().max().item()}"
        )


@pytest.mark.integration
def test_gemma_matches_naive() -> None:
    from transformers import GemmaConfig, GemmaForCausalLM

    from fpwap.models import LlamaPlumbing, get_plumbing

    config = GemmaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEAD,
        num_key_value_heads=N_KV_HEAD,
        max_position_embeddings=SEQ_LEN,
        head_dim=HIDDEN // N_HEAD,
    )
    torch.manual_seed(SEED)
    model = GemmaForCausalLM(config)
    model.eval()

    assert isinstance(get_plumbing(model), LlamaPlumbing)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    expected = _capture_per_block(model, input_ids)
    got = _run_fpwap(model, input_ids)

    for i in range(N_LAYERS):
        assert torch.equal(got[i], expected[i].float()), (
            f"gemma layer {i}: max diff {(got[i] - expected[i].float()).abs().max().item()}"
        )
