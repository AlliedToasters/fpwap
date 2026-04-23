"""Minimum preflight: dry-run probe + extrapolation.

Not the full SPEC §10 planner (microbatch binary search, VRAM static
analysis) — just the feasibility gate + ETA that runs in seconds.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 8
HIDDEN = 32
N_LAYERS = 2
N_HEAD = 2
VOCAB = 40


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
def test_preflight_reports_feasible() -> None:
    from fpwap import Sweep

    model = _tiny_gpt2()
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
    )
    report = sweep.preflight()

    assert report.feasible
    assert report.microbatch_size == 2
    assert report.residual_buffer_gb > 0
    assert report.estimated_wall_clock_s > 0
    assert not report.blockers


@pytest.mark.integration
def test_preflight_blocks_empty_dataset() -> None:
    from fpwap import Sweep

    model = _tiny_gpt2()
    sweep = Sweep(
        model=model,
        dataset=[],
        seq_len=SEQ_LEN,
        callbacks=[],
        transport_dtype=torch.float32,
        progress=False,
    )
    report = sweep.preflight()

    assert not report.feasible
    assert report.blockers
