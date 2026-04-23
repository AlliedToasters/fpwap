"""Regression: residual_pre Emit captures must survive the layer's own
residual_post write without corruption.

The pinned-buffer fast path makes `buffer.read_slice(...)` return a view
into the residual buffer. When the same layer writes residual_post back
to that slice, an un-cloned emit would silently alias the overwritten
memory. This test drives residual_pre + residual_post RawActivations in
the same sweep and asserts both capture what they should.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 6
SEQ_LEN = 8
HIDDEN = 32
N_LAYERS = 3
N_HEAD = 2
VOCAB = 64


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
def test_residual_pre_and_post_both_correct_in_same_sweep() -> None:
    """Dual-hook sweep: residual_pre (aliasing-prone) + residual_post."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    # Naive baseline: pre-hook for residual_pre, post-hook for residual_post.
    pre: dict[int, torch.Tensor] = {}
    post: dict[int, torch.Tensor] = {}

    def pre_hook(i: int):
        def _h(_m, args, kwargs):
            t = args[0] if args else kwargs["hidden_states"]
            pre[i] = t.detach().clone()

        return _h

    def post_hook(i: int):
        def _h(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            post[i] = t.detach().clone()

        return _h

    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(
            block.register_forward_pre_hook(pre_hook(i), with_kwargs=True)
        )
        handles.append(block.register_forward_hook(post_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    pre_cb = RawActivations(
        layers="all", hook="residual_pre", last_token_only=False, out_dtype=torch.float32
    )
    post_cb = RawActivations(
        layers="all", hook="residual_post", last_token_only=False, out_dtype=torch.float32
    )
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[pre_cb, post_cb],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
        apply_final_norm=False,
    )
    result = sweep.run()

    for layer_idx in range(N_LAYERS):
        got_pre = result.activations(layer=layer_idx, hook="residual_pre")
        got_post = result.activations(layer=layer_idx, hook="residual_post")
        assert torch.allclose(got_pre, pre[layer_idx].float(), atol=1e-5, rtol=1e-5), (
            f"layer {layer_idx} residual_pre mismatch — the aliasing bug returned"
        )
        assert torch.allclose(got_post, post[layer_idx].float(), atol=1e-5, rtol=1e-5), (
            f"layer {layer_idx} residual_post mismatch"
        )
