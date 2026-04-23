"""Per-sub-module hooks: residual_pre, attn_out, mlp_out, residual_post.

Ground truth comes from register_forward_pre_hook on the block (for
residual_pre) and register_forward_hook on block.attn / block.mlp (for
attn_out / mlp_out — the sub-layer outputs BEFORE the residual add,
matching the README's "attention sub-layer output" definition).

fpwap must capture bit-for-bit the same tensors via a RawActivations
callback targeted at each hook.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 6
SEQ_LEN = 8
HIDDEN = 32
N_LAYERS = 2
N_HEAD = 2
VOCAB = 100


def _tiny_gpt2() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=N_HEAD,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config)
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
            return None

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
    for i, block in enumerate(model.transformer.h):
        handles.append(
            block.register_forward_pre_hook(make_block_pre(i), with_kwargs=True)
        )
        handles.append(block.attn.register_forward_hook(make_attn_out(i)))
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
def test_extra_hooks_match_naive(hook: str) -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
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
        microbatch_size=2,
        seed=SEED,
        progress=False,
        apply_final_norm=False,
    )
    result = sweep.run()

    for layer_idx in range(N_LAYERS):
        got = result.activations(layer=layer_idx, hook=hook)  # type: ignore[arg-type]
        expected = naive[(layer_idx, hook)].float()
        assert got.shape == expected.shape, (
            f"shape mismatch layer={layer_idx} hook={hook}: {got.shape} vs {expected.shape}"
        )
        max_diff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5), (
            f"hook={hook} layer={layer_idx} max abs diff {max_diff}"
        )
