"""Regression: a truncated sweep must capture shallow layers identically to a
full-depth sweep.

fpwap stops the layer loop at the deepest layer any callback targets
(``effective_n_layers = max_capture + 1``) and skips final norm + LM head.
That truncation must be invisible to any *shallower* capture: in a standard
transformer ``residual_post[L]`` is a pure function of blocks ``<= L``, so the
value stored for layer ``L`` cannot depend on how many layers run after it.

This guards against a recurring suspicion (a downstream "truncation perturbs
shallow layers" report) by pinning the property bitwise at both fp32 and bf16
transport, on the default ``chunk_size=1`` / ``microbatch_size=1`` path that
activation-only readers use.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 16
HIDDEN = 32
N_LAYERS = 6
N_HEAD = 2
VOCAB = 64
PROBE = 2  # the fixed shallow layer we compare across truncation depths


def _tiny_gpt2(dtype: torch.dtype) -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB, n_positions=SEQ_LEN, n_embd=HIDDEN,
        n_layer=N_LAYERS, n_head=N_HEAD,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model.to(dtype)


def _capture_probe(model, ids, deepest, transport) -> torch.Tensor:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    layers = sorted({PROBE, deepest})
    cb = RawActivations(
        layers=layers, hook="residual_post", last_token_only=False,
        out_dtype=torch.float32,
    )
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[cb],
        transport_dtype=transport,
        microbatch_size=1,
        seed=SEED,
        progress=False,
        apply_final_norm=False,
    )
    return sweep.run().activations(layer=PROBE, hook="residual_post")


@pytest.mark.integration
@pytest.mark.parametrize("transport", [torch.float32, torch.bfloat16])
def test_truncation_does_not_perturb_shallow_capture(transport: torch.dtype) -> None:
    """residual_post[PROBE] is bitwise-identical whether the sweep stops at
    PROBE (hard truncation) or runs to the last layer (full depth)."""
    model = _tiny_gpt2(transport)
    torch.manual_seed(SEED)
    ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    truncated = _capture_probe(model, ids, deepest=PROBE, transport=transport)
    full = _capture_probe(model, ids, deepest=N_LAYERS - 1, transport=transport)

    assert torch.equal(truncated, full), (
        f"truncation perturbed a shallow capture (transport={transport}): "
        f"max abs diff "
        f"{(truncated.float() - full.float()).abs().max().item():.3e}"
    )
