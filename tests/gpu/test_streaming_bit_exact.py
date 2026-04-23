"""Streaming path bit-exactness on a real cached Llama.

Cross-checks the mmap-from-HF-cache + manual _load_layer / _unload_layer
path against a pre-loaded model running the same weights through the
same fpwap loop. Both runs use bfloat16 on CUDA with matched microbatch
size; the streaming path has no forgiveness budget — identical inputs
through identical math must produce identical outputs.

Skipped if Llama-3.2-1B isn't in the HF cache locally.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SEQ_LEN = 16
N_SAMPLES = 4


def _snapshot_dir_or_skip() -> Path:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(MODEL_ID, local_files_only=True))
    except Exception as e:
        pytest.skip(f"{MODEL_ID} not in local HF cache: {e}")


@pytest.mark.gpu
def test_streaming_matches_preloaded_on_real_llama() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    snapshot_dir = _snapshot_dir_or_skip()
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = ["Quick fox", "Hello world", "Foo bar baz", "Attention is all"]
    batch = tokenizer(
        prompts,
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to("cuda:0")
    attention_mask = batch["attention_mask"].to("cuda:0")

    # Run A: pre-loaded model (all weights on GPU, no streaming).
    preloaded_model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda:0")
    preloaded_model.eval()
    n_layers = len(preloaded_model.model.layers)
    hidden_size = preloaded_model.config.hidden_size

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.acts: dict[int, torch.Tensor] = {
                i: torch.zeros(
                    N_SAMPLES,
                    SEQ_LEN,
                    hidden_size,
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

    dataset = [
        {
            "input_ids": input_ids[i : i + 1],
            "attention_mask": attention_mask[i : i + 1],
        }
        for i in range(N_SAMPLES)
    ]

    cap_pre = Capture()
    Sweep(
        model=preloaded_model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[cap_pre],
        transport_dtype=torch.bfloat16,
        microbatch_size=N_SAMPLES,
        seed=0,
        progress=False,
    ).run()

    # Free the preloaded model before the streaming run to keep VRAM sane.
    del preloaded_model
    torch.cuda.empty_cache()

    # Run B: streaming via mmap-from-HF-cache.
    cap_stream = Capture()
    Sweep(
        model=str(snapshot_dir),
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[cap_stream],
        transport_dtype=torch.bfloat16,
        microbatch_size=N_SAMPLES,
        execution_device="cuda:0",
        seed=0,
        progress=False,
    ).run()

    mask_expanded = attention_mask.bool().unsqueeze(-1)
    for layer_idx in range(n_layers):
        pre = cap_pre.acts[layer_idx].masked_select(mask_expanded)
        stm = cap_stream.acts[layer_idx].masked_select(mask_expanded)
        max_diff = (pre.float() - stm.float()).abs().max().item()
        assert torch.equal(pre, stm), (
            f"preloaded vs streaming mismatch at layer {layer_idx}: max abs diff {max_diff}"
        )
