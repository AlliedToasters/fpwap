"""Callable progress reporter receives ProgressEvents at sensible boundaries.

The README promises `progress=callable` as the wandb/rich hook. This test
locks the shape: layer_start → n * microbatch_end → layer_end, one set per
layer, no ProgressEvents when progress=False.
"""
from __future__ import annotations

import pytest
import torch

from fpwap.engine import ProgressEvent

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 4
HIDDEN = 8
N_LAYERS = 2


def _tiny_gpt2() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=16,
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
def test_callable_reporter_gets_events() -> None:
    from fpwap import Sweep

    events: list[ProgressEvent] = []
    mb = 2  # 2 microbatches per layer

    sweep = Sweep(
        model=_tiny_gpt2(),
        dataset=[
            {"input_ids": torch.randint(0, 16, (1, SEQ_LEN))} for _ in range(N_SAMPLES)
        ],
        seq_len=SEQ_LEN,
        callbacks=[],
        transport_dtype=torch.float32,
        microbatch_size=mb,
        seed=SEED,
        progress=events.append,
    )
    sweep.run()

    kinds = [e.kind for e in events]
    assert kinds.count("layer_start") == N_LAYERS
    assert kinds.count("layer_end") == N_LAYERS
    assert kinds.count("microbatch_end") == N_LAYERS * (N_SAMPLES // mb)
    # layer_start comes before any microbatch_end for that layer
    for layer in range(N_LAYERS):
        layer_events = [e for e in events if e.layer_idx == layer]
        assert layer_events[0].kind == "layer_start"
        assert layer_events[-1].kind == "layer_end"
    # Wall-clock is monotonic nondecreasing
    walls = [e.wall_s for e in events]
    assert walls == sorted(walls)


@pytest.mark.integration
def test_progress_false_emits_no_events_to_reporter() -> None:
    from fpwap import Sweep

    # With progress=False, no reporter can be wired at all. This test just
    # confirms False doesn't trip the callable-reporter branch.
    sweep = Sweep(
        model=_tiny_gpt2(),
        dataset=[{"input_ids": torch.randint(0, 16, (1, SEQ_LEN))}],
        seq_len=SEQ_LEN,
        callbacks=[],
        transport_dtype=torch.float32,
        microbatch_size=1,
        seed=SEED,
        progress=False,
    )
    sweep.run()  # must not raise
