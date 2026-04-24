"""Callable progress reporter receives chunk-boundary drain/unload events."""
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
def test_chunk_drain_events_emitted() -> None:
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

    kinds = [e.kind for e in events]
    assert "chunk_drain_sync" in kinds
    assert "chunk_unload" in kinds


@pytest.mark.integration
def test_profile_has_drain_sync_timing() -> None:
    from fpwap import Sweep

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

    assert result.profile.drain_sync_s >= 0
    assert result.profile.unload_s >= 0


@pytest.mark.integration
def test_preloop_events_emitted() -> None:
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

    kinds = [e.kind for e in events]
    assert "preloop_resolve_dataset" in kinds
    assert "preloop_build_segments" in kinds
    assert "preloop_end" in kinds
    assert "embed_start" in kinds
    assert "embed_end" in kinds

    preloop_end_idx = kinds.index("preloop_end")
    embed_start_idx = kinds.index("embed_start")
    assert embed_start_idx > preloop_end_idx

    walls = [e.wall_s for e in events]
    assert walls == sorted(walls)


@pytest.mark.integration
def test_profile_has_preloop_timing() -> None:
    from fpwap import Sweep
    from fpwap.engine import PreloopTiming

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
    p = result.profile

    assert isinstance(p.preloop, PreloopTiming)
    assert p.preloop.total_s > 0
    assert p.preloop_s == p.preloop.total_s
    assert p.embed_sync_s >= 0
    assert p.loop_setup_s >= 0


@pytest.mark.integration
def test_all_wall_clock_attributed() -> None:
    """Every second of wall-clock is attributed to a named phase."""
    from fpwap import Sweep

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
    p = result.profile

    attributed = (
        (p.setup.total_s if p.setup else 0.0)
        + p.preloop_s
        + p.embed_s
        + p.loop_setup_s
        + p.loop_s
        + p.teardown_s
    )
    assert attributed <= p.total_wall_s * 1.01
    assert attributed >= p.total_wall_s * 0.80
