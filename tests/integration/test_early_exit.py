"""Lazy forward: exit sweep once all requested captures are emitted (#48).

When every callback declares explicit target_layers (not "all"), the engine
should stop after the deepest requested layer — layers beyond it produce no
downstream output.  Verifies:

1. Activations at the captured layer are bit-exact with a full naive forward.
2. The profile only contains timings for layers 0..max_capture_layer.
3. No early exit when any callback targets "all" layers.
4. With multiple callbacks, the deepest target across all of them governs.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 6
SEQ_LEN = 8
HIDDEN = 32
N_LAYERS = 4
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


def _naive_per_layer_residual_post(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> dict[int, torch.Tensor]:
    captures: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            captures[layer_idx] = hidden.detach().clone()
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
    return captures



@pytest.mark.integration
def test_early_exit_activations_bit_exact() -> None:
    """Capture at layer 1 of 4. Activations must match naive forward."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    baseline = _naive_per_layer_residual_post(model, input_ids)

    raw = RawActivations(layers=[1], last_token_only=False, out_dtype=torch.float32)
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        apply_final_norm=False,
    )
    result = sweep.run()

    got = result.activations(layer=1, hook="residual_post")
    expected = baseline[1]
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6), (
        f"early-exit activations mismatch: max diff "
        f"{(got - expected).abs().max().item()}"
    )

    # Key assertion: only layers 0 and 1 were processed.
    assert set(result.profile.per_layer.keys()) == {0, 1}



@pytest.mark.integration
def test_early_exit_skips_profile_for_unused_layers() -> None:
    """Profile must not contain entries for layers beyond the capture plan."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    raw = RawActivations(layers=[0], last_token_only=True, out_dtype=torch.float32)
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        apply_final_norm=False,
    )
    result = sweep.run()

    assert set(result.profile.per_layer.keys()) == {0}


@pytest.mark.integration
def test_no_early_exit_when_target_all() -> None:
    """target_layers='all' means every layer runs — no early exit."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    raw = RawActivations(layers="all", last_token_only=True, out_dtype=torch.float32)
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        apply_final_norm=False,
    )
    result = sweep.run()

    assert set(result.profile.per_layer.keys()) == set(range(N_LAYERS))



@pytest.mark.integration
def test_early_exit_multi_callback_uses_deepest() -> None:
    """Two callbacks at layers [0,1] and [2]. Engine must exit after layer 2."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    baseline = _naive_per_layer_residual_post(model, input_ids)

    raw_early = RawActivations(
        layers=[0, 1], last_token_only=True, out_dtype=torch.float32,
    )
    raw_late = RawActivations(
        layers=[2], last_token_only=True, out_dtype=torch.float32,
    )
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw_early, raw_late],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        apply_final_norm=False,
    )
    result = sweep.run()

    # All three capture layers present, layer 3 skipped.
    assert set(result.profile.per_layer.keys()) == {0, 1, 2}

    # Bit-exact at each captured layer.
    for layer_idx in [0, 1, 2]:
        got = result.activations(layer=layer_idx, hook="residual_post")
        expected = baseline[layer_idx][:, -1, :]
        assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6), (
            f"layer {layer_idx} mismatch: max diff "
            f"{(got - expected).abs().max().item()}"
        )



@pytest.mark.integration
def test_early_exit_with_chunk_size() -> None:
    """Early exit works correctly with chunk_size > 1."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    baseline = _naive_per_layer_residual_post(model, input_ids)

    raw = RawActivations(layers=[1], last_token_only=False, out_dtype=torch.float32)
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[raw],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        apply_final_norm=False,
        chunk_size=3,
    )
    result = sweep.run()

    got = result.activations(layer=1, hook="residual_post")
    expected = baseline[1]
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
    assert set(result.profile.per_layer.keys()) == {0, 1}
