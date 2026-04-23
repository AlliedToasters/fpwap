"""RawActivations round-trip: callback emits → result.activations() returns.

Locks the README's `result.activations(layer=..., hook=...)` contract
for the in-memory path (no StorageBackend). Mirrors the forcing test's
ground truth: pool last-token residual_post from a naive forward.
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


@pytest.mark.integration
def test_raw_activations_last_token_matches_naive() -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    # Ground truth: naive forward, capture per-block output, pool last-token.
    per_layer_last: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            per_layer_last[layer_idx] = h[:, -1, :].detach().clone()

        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(_make_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    raw = RawActivations(layers="all", last_token_only=True, out_dtype=torch.float32)
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        microbatch_size=2,  # exercise the multi-microbatch concat path
        seed=SEED,
        apply_final_norm=False,
    )
    result = sweep.run()

    for layer_idx in range(N_LAYERS):
        got = result.activations(layer=layer_idx, hook="residual_post")
        assert got.shape == (N_SAMPLES, HIDDEN)
        expected = per_layer_last[layer_idx].float()
        max_diff = (got - expected).abs().max().item()
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5), (
            f"RawActivations layer {layer_idx} mismatch: max abs diff {max_diff}"
        )
