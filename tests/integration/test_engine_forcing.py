"""The forcing function for the engine milestone.

Drives the inverted-loop engine, embedding pass, model plumbing, residual
buffer, and callback dispatch into existence by comparing fpwap's per-layer
`residual_post` against a naive forward pass on an in-memory GPT-2.

CPU + fp32 for determinism and speed — the weight-streaming path is not
under test here (that is the GPU bit-perfect test in tests/gpu/). This test
exercises the loop inversion, pass-0 embedding, per-layer dispatch, and
callback wiring.

Marked `integration` so razor-thin CI skips it (see pyproject addopts).
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


def _naive_per_layer_residual_post(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> dict[int, torch.Tensor]:
    captures: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_mod: torch.nn.Module, _inp: tuple, out: object) -> None:
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
def test_fpwap_matches_naive_forward_cpu_gpt2() -> None:
    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    torch.manual_seed(SEED)
    model = _tiny_gpt2()
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    baseline = _naive_per_layer_residual_post(model, input_ids)

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.acts: dict[int, torch.Tensor] = {
                i: torch.zeros(N_SAMPLES, SEQ_LEN, HIDDEN) for i in range(N_LAYERS)
            }

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            self.acts[layer_idx][sample_ids] = acts.detach().float().cpu()
            return None

    cap = Capture()
    run = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.float32,
        seed=SEED,
    )
    run.run()

    for layer_idx, baseline_hidden in baseline.items():
        got = cap.acts[layer_idx]
        max_diff = (got - baseline_hidden).abs().max().item()
        assert torch.allclose(got, baseline_hidden, atol=1e-6, rtol=1e-6), (
            f"residual_post mismatch at layer {layer_idx}: max abs diff {max_diff}"
        )
