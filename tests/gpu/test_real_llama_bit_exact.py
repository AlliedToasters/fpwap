"""Real Llama-family model on CUDA: fpwap matches naive forward bit-for-bit.

The tiny-random tests prove the plumbing; this test proves a *real* cached
Llama runs bit-exact through fpwap. Skipped cleanly if the model isn't in
the HF cache locally (there's no universe where we want to download a
multi-GB model during `pytest`).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SEQ_LEN = 32
N_SAMPLES = 4


def _snapshot_dir_or_skip() -> Path:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(MODEL_ID, local_files_only=True))
    except Exception as e:
        pytest.skip(f"{MODEL_ID} not in local HF cache: {e}")


@pytest.mark.gpu
def test_real_llama_1b_residual_post_bit_exact() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    snapshot_dir = _snapshot_dir_or_skip()

    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda:0")
    model.eval()

    prompts = [
        "The quick brown fox jumps",
        "In the beginning was the word",
        "Hello",
        "Attention is all you need.",
    ]
    batch = tokenizer(
        prompts,
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to("cuda:0")
    attention_mask = batch["attention_mask"].to("cuda:0")

    n_layers = len(model.model.layers)
    baseline: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            baseline[layer_idx] = h.detach().clone()

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

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.acts: dict[int, torch.Tensor] = {
                i: torch.zeros(
                    N_SAMPLES,
                    SEQ_LEN,
                    model.config.hidden_size,
                    dtype=torch.bfloat16,
                    device="cuda:0",
                )
                for i in range(n_layers)
            }

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            self.acts[layer_idx][sample_ids] = acts.detach()
            return None

    cap = Capture()
    sweep = Sweep(
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
        transport_dtype=torch.bfloat16,
        # Match naive's batch size; bf16 reductions are batch-size-sensitive,
        # so a mismatched microbatch would diverge by LSB noise that amplifies
        # through 16 layers. Users will tune this to fit VRAM — correctness
        # within a given microbatch_size is what we're verifying here.
        microbatch_size=N_SAMPLES,
        seed=0,
    )
    sweep.run()

    # With microbatch == naive batch size, bf16 matmul is deterministic and
    # outputs should be bit-identical at real positions.
    mask_expanded = attention_mask.bool().unsqueeze(-1)
    for layer_idx, base in baseline.items():
        got = cap.acts[layer_idx]
        real_got = got.masked_select(mask_expanded)
        real_base = base.masked_select(mask_expanded)
        max_diff = (real_got.float() - real_base.float()).abs().max().item()
        assert torch.equal(real_got, real_base), (
            f"real Llama layer {layer_idx} mismatch: max abs diff {max_diff}"
        )
