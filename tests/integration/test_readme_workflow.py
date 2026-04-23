"""End-to-end README workflow: multi-callback sweep → artifact → steering pass.

Runs the exact shape shown at the top of README.md:
  1. IncrementalPCA fits a basis per layer
  2. result.artifact("pca_basis", layer=...) returns a usable handle
  3. SteerInBasis in a second pass consumes that handle

On a tiny GPT-2 so it stays CPU-only. Goal: if anyone refactors the
artifact flow or callback protocol, they break this test and have to
fix the documented shape rather than silently diverge from it.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 8
SEQ_LEN = 6
HIDDEN = 16
N_LAYERS = 3
VOCAB = 48


def _tiny_gpt2() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=2,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.mark.integration
def test_readme_workflow_pca_then_steer() -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import (
        IncrementalPCA,
        RawActivations,
        SteerInBasis,
    )

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    # Pass 1: capture activations at a middle layer + fit PCA over all layers.
    target_layer = 1
    run = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[
            RawActivations(layers=[target_layer]),
            IncrementalPCA(layers="all", n_components=4),
        ],
        transport_dtype=torch.float32,
        microbatch_size=4,
        seed=SEED,
        progress=False,
    )
    plan = run.preflight()
    assert plan.feasible

    result = run.run()
    acts = result.activations(layer=target_layer, hook="residual_post")
    assert acts.shape[0] == N_SAMPLES  # pooled last-token by default
    basis = result.artifact("pca_basis", layer=target_layer)
    assert basis.payload["basis"].shape == (HIDDEN, 4)

    # Pass 2: steer in the basis fit during pass 1.
    steer_run = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[
            SteerInBasis(
                basis_artifact=basis,
                direction_idx=0,
                alpha=2.0,
                layers=[target_layer],
            ),
        ],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    )
    steered = steer_run.run()
    # The steering pass must complete without error and produce a profile.
    assert steered.profile.total_wall_s > 0
    assert steered.profile.total_tokens == N_SAMPLES * SEQ_LEN
