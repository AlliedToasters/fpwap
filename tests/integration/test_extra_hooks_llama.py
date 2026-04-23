"""Per-sub-module hooks for Llama family — CPU tiny-Llama parity.

Confirms layer_forward_with_hooks decomposition on LlamaDecoderLayer matches
what you'd capture with register_forward_hook on self_attn / mlp. Runs on
CPU with a tiny random-init Llama so it can live alongside the GPT-2
parametrized test in the integration tier (no GPU, no model downloads).
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


def _tiny_llama() -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEAD,
        num_key_value_heads=N_KV_HEAD,
        max_position_embeddings=SEQ_LEN,
    )
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _capture_naive(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[
    tuple[int, str], torch.Tensor
]:
    captured: dict[tuple[int, str], torch.Tensor] = {}

    def make_block_pre(i: int):
        def _hook(_mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            captured[(i, "residual_pre")] = h.detach().clone()

        return _hook

    def make_attn_out(i: int):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[(i, "attn_out")] = t.detach().clone()

        return _hook

    def make_mlp_out(i: int):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[(i, "mlp_out")] = t.detach().clone()

        return _hook

    def make_block_post(i: int):
        def _hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[(i, "residual_post")] = t.detach().clone()

        return _hook

    handles = []
    for i, block in enumerate(model.model.layers):
        handles.append(
            block.register_forward_pre_hook(make_block_pre(i), with_kwargs=True)
        )
        handles.append(block.self_attn.register_forward_hook(make_attn_out(i)))
        handles.append(block.mlp.register_forward_hook(make_mlp_out(i)))
        handles.append(block.register_forward_hook(make_block_post(i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    return captured


@pytest.mark.integration
@pytest.mark.parametrize("hook", ["residual_pre", "attn_out", "mlp_out", "residual_post"])
def test_extra_hooks_match_naive_llama(hook: str) -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_llama()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    naive = _capture_naive(model, input_ids)

    raw = RawActivations(
        layers="all", hook=hook, last_token_only=False, out_dtype=torch.float32  # type: ignore[arg-type]
    )
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
        apply_final_norm=False,
    )
    result = sweep.run()

    for layer_idx in range(N_LAYERS):
        got = result.activations(layer=layer_idx, hook=hook)  # type: ignore[arg-type]
        expected = naive[(layer_idx, hook)].float()
        assert got.shape == expected.shape
        max_diff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5), (
            f"hook={hook} layer={layer_idx} max abs diff {max_diff}"
        )
