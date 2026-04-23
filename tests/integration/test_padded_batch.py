"""Dataset contract: attention_mask must propagate through the sweep.

Without an attention_mask, GPT-2 attends over padded positions, which
leaks into per-layer residual_post for real tokens. This is the
~3e-3 drift hazard flagged by lmprobe's SDPA bit-exactness scaffold
(Issue #284 in lmprobe). Locking the contract before any non-GPT-2
family lands stops that class of bug from recurring per family.

Items in the dataset may carry `attention_mask` alongside `input_ids`.
The engine collates both, and the plumbing converts the 2D mask into
whatever form the block expects.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 8
HIDDEN = 16
N_LAYERS = 2
VOCAB = 32
PAD_TOKEN = 0


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


def _build_padded_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Build LEFT-padded input_ids + attention_mask with real lengths 3,5,7,8.

    Left-padding is the form that actually exposes mask-propagation bugs
    for causal LMs: pad tokens precede real tokens in the sequence, so
    the causal attention window at a real position covers pad positions
    unless the mask suppresses them.
    """
    torch.manual_seed(SEED + 1)
    real_lengths = torch.tensor([3, 5, 7, SEQ_LEN])
    input_ids = torch.full((N_SAMPLES, SEQ_LEN), PAD_TOKEN, dtype=torch.long)
    attention_mask = torch.zeros((N_SAMPLES, SEQ_LEN), dtype=torch.long)
    for i, L in enumerate(real_lengths):
        input_ids[i, SEQ_LEN - L :] = torch.randint(1, VOCAB, (L,))  # avoid PAD_TOKEN
        attention_mask[i, SEQ_LEN - L :] = 1
    return input_ids, attention_mask


def _naive_per_layer_residual_post(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
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
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captures


@pytest.mark.integration
def test_padded_batch_matches_naive_at_real_positions() -> None:
    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    model = _tiny_gpt2()
    input_ids, attention_mask = _build_padded_batch()

    baseline = _naive_per_layer_residual_post(model, input_ids, attention_mask)

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
        dataset=[
            {
                "input_ids": input_ids[i : i + 1],
                "attention_mask": attention_mask[i : i + 1],
            }
            for i in range(N_SAMPLES)
        ],
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.float32,
        seed=SEED,
        apply_final_norm=False,
    )
    run.run()

    # Compare only at real (non-padded) positions — the padded slots are
    # "don't care" and HF's own output at those positions is undefined.
    for layer_idx, baseline_hidden in baseline.items():
        got = cap.acts[layer_idx]
        mask_expanded = attention_mask.bool().unsqueeze(-1)
        real_got = got.masked_select(mask_expanded)
        real_base = baseline_hidden.masked_select(mask_expanded)
        max_diff = (real_got - real_base).abs().max().item()
        assert torch.allclose(real_got, real_base, atol=1e-5, rtol=1e-5), (
            f"residual_post mismatch at layer {layer_idx} on real positions: "
            f"max abs diff {max_diff}"
        )
