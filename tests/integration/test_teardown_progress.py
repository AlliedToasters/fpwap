"""Callable progress reporter receives teardown-phase events.

After the last layer_end, the reporter should receive events for each
teardown sub-phase so users/integrators have visibility into finalization.
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
def test_teardown_events_emitted() -> None:
    from fpwap import Sweep

    events: list[ProgressEvent] = []

    sweep = Sweep(
        model=_tiny_gpt2(),
        dataset=[
            {"input_ids": torch.randint(0, 16, (1, SEQ_LEN))} for _ in range(N_SAMPLES)
        ],
        seq_len=SEQ_LEN,
        callbacks=[],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=events.append,
    )
    sweep.run()

    teardown_events = [e for e in events if e.kind.startswith("teardown")]
    assert len(teardown_events) >= 1, "expected at least one teardown progress event"

    teardown_kinds = [e.kind for e in teardown_events]
    assert "teardown_buffer_flush" in teardown_kinds
    assert "teardown_end" in teardown_kinds

    last_layer_end = max(
        i for i, e in enumerate(events) if e.kind == "layer_end"
    )
    first_teardown = min(
        i for i, e in enumerate(events) if e.kind.startswith("teardown")
    )
    assert first_teardown > last_layer_end

    walls = [e.wall_s for e in events]
    assert walls == sorted(walls)


@pytest.mark.integration
def test_profile_teardown_is_teardown_timing() -> None:
    from fpwap import Sweep
    from fpwap.engine import TeardownTiming

    sweep = Sweep(
        model=_tiny_gpt2(),
        dataset=[
            {"input_ids": torch.randint(0, 16, (1, SEQ_LEN))} for _ in range(N_SAMPLES)
        ],
        seq_len=SEQ_LEN,
        callbacks=[],
        transport_dtype=torch.float32,
        microbatch_size=2,
        seed=SEED,
        progress=False,
    )
    result = sweep.run()

    assert isinstance(result.profile.teardown, TeardownTiming)
    assert result.profile.teardown.total_s > 0
    assert result.profile.teardown_s == result.profile.teardown.total_s
