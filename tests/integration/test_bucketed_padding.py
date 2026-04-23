"""Bucketed padding: variable-length sequences grouped by real length.

Verifies that padding='bucketed' produces correct activations at real
(unmasked) positions. Each bucket's output is compared against HF's
naive forward on the same trimmed inputs — this is the true correctness
baseline (not the fixed-padding forward, which uses different positions).

Covers issue #10: relax fixed seq_len assumption.
"""
from __future__ import annotations

import pytest
import torch

SEED = 42
SEQ_LEN = 64
HIDDEN = 16
N_LAYERS = 2
VOCAB = 32
PAD_TOKEN = 0
N_SHORT = 4
N_LONG = 4
N_SAMPLES = N_SHORT + N_LONG


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


def _mixed_length_dataset() -> list[dict[str, torch.Tensor]]:
    """Left-padded dataset: 4 short (~8 tok) + 4 long (~48 tok) sequences."""
    torch.manual_seed(SEED + 1)
    short_lengths = [5, 7, 9, 11]
    long_lengths = [40, 45, 50, 55]
    all_lengths = short_lengths + long_lengths

    items: list[dict[str, torch.Tensor]] = []
    for L in all_lengths:
        input_ids = torch.full((1, SEQ_LEN), PAD_TOKEN, dtype=torch.long)
        attention_mask = torch.zeros((1, SEQ_LEN), dtype=torch.long)
        input_ids[0, SEQ_LEN - L :] = torch.randint(1, VOCAB, (L,))
        attention_mask[0, SEQ_LEN - L :] = 1
        items.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return items


@pytest.mark.integration

def test_bucketed_padding_basic() -> None:
    """padding='bucketed' runs to completion and returns per-sample activations."""
    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    model = _tiny_gpt2()
    items = _mixed_length_dataset()

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.per_sample: dict[int, dict[int, torch.Tensor]] = {}

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            for i, sid in enumerate(sample_ids.cpu().tolist()):
                self.per_sample.setdefault(sid, {})[layer_idx] = (
                    acts[i].detach().float().cpu()
                )
            return None

    cap = Capture()
    sweep = Sweep(
        model=model,
        dataset=items,
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.float32,
        padding="bucketed",
        apply_final_norm=False,
    )
    result = sweep.run()

    for i in range(N_SAMPLES):
        assert i in cap.per_sample, f"sample {i} missing from output"
        for layer in range(N_LAYERS):
            assert layer in cap.per_sample[i], f"sample {i} missing layer {layer}"
            acts = cap.per_sample[i][layer]
            assert acts.shape[-1] == HIDDEN
            assert torch.isfinite(acts).all()


@pytest.mark.integration

def test_bucketed_matches_per_bucket_naive() -> None:
    """Each bucket's activations match HF's naive forward on the same trimmed inputs.

    This is the real correctness guarantee: given the same (trimmed) input_ids
    and attention_mask, fpwap's bucketed output equals HF's own forward at
    every real (unmasked) position.
    """
    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    model = _tiny_gpt2()
    items = _mixed_length_dataset()

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.per_sample: dict[int, dict[int, torch.Tensor]] = {}

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            for i, sid in enumerate(sample_ids.cpu().tolist()):
                self.per_sample.setdefault(sid, {})[layer_idx] = (
                    acts[i].detach().float().cpu()
                )
            return None

    cap = Capture()
    sweep = Sweep(
        model=model,
        dataset=items,
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.float32,
        padding="bucketed",
        apply_final_norm=False,
    )
    sweep.run()

    # Naive baseline: run each sample individually through HF at its own
    # bucket-trimmed length. Compare at real positions.
    def _next_po2(n: int) -> int:
        return 1 << max(n - 1, 0).bit_length() if n > 1 else 1

    for i, item in enumerate(items):
        mask = item["attention_mask"].squeeze(0)
        real_len = int(mask.sum().item())
        bseq = min(_next_po2(max(real_len, 16)), SEQ_LEN)

        # Trim (left-padded → take rightmost bseq tokens)
        ids_trimmed = item["input_ids"][:, -bseq:]
        mask_trimmed = item["attention_mask"][:, -bseq:]

        # Naive HF forward on the trimmed input
        captures: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def hook(_mod, _inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                captures[layer_idx] = hidden.detach().clone()

            return hook

        handles = [
            block.register_forward_hook(_make_hook(li))
            for li, block in enumerate(model.transformer.h)
        ]
        try:
            with torch.no_grad():
                model(input_ids=ids_trimmed, attention_mask=mask_trimmed)
        finally:
            for h in handles:
                h.remove()

        # Compare at real positions
        mask_1d = mask_trimmed.squeeze(0).bool()
        for layer_idx in range(N_LAYERS):
            got = cap.per_sample[i][layer_idx]
            expected = captures[layer_idx].squeeze(0).float()
            got_real = got[mask_1d]
            exp_real = expected[mask_1d]
            max_diff = (got_real - exp_real).abs().max().item()
            assert torch.allclose(got_real, exp_real, atol=1e-5, rtol=1e-5), (
                f"sample {i} layer {layer_idx}: bucketed vs naive mismatch "
                f"at real positions (max diff {max_diff})"
            )


@pytest.mark.integration

def test_bucketed_warns_learned_positions() -> None:
    """padding='bucketed' emits a UserWarning for GPT-2 (learned positional embeddings)."""
    from fpwap import Callback, Sweep
    from fpwap.types import HookName

    model = _tiny_gpt2()
    items = _mixed_length_dataset()

    class Noop(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

    with pytest.warns(UserWarning, match="learned positional"):
        sweep = Sweep(
            model=model,
            dataset=items,
            seq_len=SEQ_LEN,
            callbacks=[Noop()],
            transport_dtype=torch.float32,
            padding="bucketed",
            apply_final_norm=False,
        )
        sweep.run()


@pytest.mark.integration

def test_bucketed_requires_attention_mask() -> None:
    """padding='bucketed' raises ValueError when items lack attention_mask."""
    from fpwap import Callback, Sweep
    from fpwap.types import HookName

    model = _tiny_gpt2()
    items = [
        {"input_ids": torch.randint(1, VOCAB, (1, SEQ_LEN))} for _ in range(4)
    ]

    class Noop(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

    sweep = Sweep(
        model=model,
        dataset=items,
        seq_len=SEQ_LEN,
        callbacks=[Noop()],
        transport_dtype=torch.float32,
        padding="bucketed",
        apply_final_norm=False,
    )
    with pytest.raises(ValueError, match="attention_mask"):
        sweep.run()
