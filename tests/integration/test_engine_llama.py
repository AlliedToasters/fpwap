"""Llama-family plumbing: per-layer residual_post matches naive forward.

Locks the second model family against the same contract as GPT-2:
padded-batch correctness, attention_mask propagation, RoPE-specific
position_ids / position_embeddings wiring. This is the test that
protects against the rotary/norm bug class flagged by lmprobe #284.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 8
HIDDEN = 32
INTERMEDIATE = 64
N_LAYERS = 2
N_HEAD = 4
N_KV_HEAD = 2
VOCAB = 40
PAD_TOKEN = 0


def _tiny_llama() -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEAD,
        num_key_value_heads=N_KV_HEAD,
        max_position_embeddings=SEQ_LEN,
        pad_token_id=PAD_TOKEN,
    )
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _build_left_padded() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED + 1)
    real_lengths = torch.tensor([3, 5, 7, SEQ_LEN])
    input_ids = torch.full((N_SAMPLES, SEQ_LEN), PAD_TOKEN, dtype=torch.long)
    attention_mask = torch.zeros((N_SAMPLES, SEQ_LEN), dtype=torch.long)
    for i, L in enumerate(real_lengths):
        input_ids[i, SEQ_LEN - L :] = torch.randint(1, VOCAB, (L,))
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
    for i, block in enumerate(model.model.layers):
        handles.append(block.register_forward_hook(_make_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captures


@pytest.mark.integration
def test_llama_padded_batch_matches_naive() -> None:
    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    model = _tiny_llama()
    input_ids, attention_mask = _build_left_padded()

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
    )
    run.run()

    for layer_idx, baseline_hidden in baseline.items():
        got = cap.acts[layer_idx]
        mask_expanded = attention_mask.bool().unsqueeze(-1)
        real_got = got.masked_select(mask_expanded)
        real_base = baseline_hidden.masked_select(mask_expanded)
        max_diff = (real_got - real_base).abs().max().item()
        assert torch.allclose(real_got, real_base, atol=1e-5, rtol=1e-5), (
            f"residual_post mismatch at llama layer {layer_idx}: max abs diff {max_diff}"
        )
