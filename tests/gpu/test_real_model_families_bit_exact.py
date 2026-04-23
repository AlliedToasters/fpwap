"""Real-checkpoint bit-exact tests across the five-family parametrization.

The tiny-random integration tests (test_model_families.py) prove the
structural matcher; these tests prove that *real* cached checkpoints run
bit-exact through fpwap against a naive HF forward. Opt-in via
``--run-gpu-large`` because each checkpoint is multi-GB.

The five-checkpoint parametrization mirrors the lmprobe PR #285 audit
surface. The 1-ULP rotary drift class (lmprobe #284) only surfaces at
real scale — tiny random configs don't carry real ``rope_scaling``.

Skips cleanly if the model isn't in the local HF cache.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "deepseek-ai/DeepSeek-V2-Lite",
]
SEQ_LEN = 32
N_SAMPLES = 4
PROMPTS = [
    "The quick brown fox jumps",
    "In the beginning was the word",
    "Hello",
    "Attention is all you need.",
]


def _snapshot_dir_or_skip(model_id: str) -> Path:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True))
    except Exception as e:
        pytest.skip(f"{model_id} not in local HF cache: {e}")


def _naive_forward(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(i: int):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h.detach().clone()

        return hook

    layers = model.model.layers  # type: ignore[union-attr]
    handles = [b.register_forward_hook(_make_hook(i)) for i, b in enumerate(layers)]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captured


@pytest.mark.gpu
@pytest.mark.gpu_large
@pytest.mark.parametrize("model_id", MODELS, ids=[m.split("/")[-1] for m in MODELS])
def test_real_checkpoint_residual_post_bit_exact(model_id: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    snapshot_dir = _snapshot_dir_or_skip(model_id)

    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda:0")
    model.eval()

    batch = tokenizer(
        PROMPTS,
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to("cuda:0")
    attention_mask = batch["attention_mask"].to("cuda:0")

    baseline = _naive_forward(model, input_ids, attention_mask)

    n_layers = len(model.model.layers)  # type: ignore[union-attr]
    hidden_size = model.config.hidden_size

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.acts: dict[int, torch.Tensor] = {
                i: torch.zeros(
                    N_SAMPLES, SEQ_LEN, hidden_size,
                    dtype=torch.bfloat16, device="cuda:0",
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
        microbatch_size=N_SAMPLES,
        seed=0,
        progress=False,
        apply_final_norm=False,
    )
    sweep.run()

    mask_expanded = attention_mask.bool().unsqueeze(-1)
    for layer_idx, base in baseline.items():
        got = cap.acts[layer_idx]
        real_got = got.masked_select(mask_expanded)
        real_base = base.masked_select(mask_expanded)
        max_diff = (real_got.float() - real_base.float()).abs().max().item()
        assert torch.equal(real_got, real_base), (
            f"{model_id} layer {layer_idx} mismatch: max abs diff {max_diff}"
        )
