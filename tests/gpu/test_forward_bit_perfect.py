"""Canonical correctness contract: fpwap vs. accelerate.cpu_offload.

Per SPEC.md §14.2 and CLAUDE.md principle 2, every forward pass must match
bit-for-bit (within dtype tolerance) against the naive cpu_offload baseline.

This test is the definition of "done" for the engine's first milestone.
Stays xfail-strict until the engine is implemented.
"""
from __future__ import annotations

import pytest
import torch

TEST_MODEL_ID = "sshleifer/tiny-gpt2"
SEQ_LEN = 16
N_SAMPLES = 8
SEED = 0


@pytest.mark.gpu
@pytest.mark.xfail(strict=True, reason="engine not yet implemented")
def test_residual_post_matches_cpu_offload_baseline() -> None:
    """Run N_SAMPLES through both paths; residual_post must match at every layer."""
    from accelerate import cpu_offload
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fpwap import Emit, fpwap, fpwapCallback
    from fpwap.types import BatchResult, HookName

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(TEST_MODEL_ID)
    prompts = [f"sample {i}" for i in range(N_SAMPLES)]
    batch = tok(prompts, padding="max_length", max_length=SEQ_LEN, return_tensors="pt")
    input_ids = batch["input_ids"].cuda()

    baseline_model = AutoModelForCausalLM.from_pretrained(
        TEST_MODEL_ID, torch_dtype=torch.bfloat16
    )
    baseline_model = cpu_offload(baseline_model, execution_device=torch.device("cuda:0"))
    baseline_per_layer: dict[int, torch.Tensor] = {}

    def _capture(layer_idx: int) -> torch.utils.hooks.RemovableHandle:
        def hook(_mod: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            baseline_per_layer[layer_idx] = out.detach().clone()

        return baseline_model.transformer.h[layer_idx].register_forward_hook(hook)

    handles = [_capture(i) for i in range(len(baseline_model.transformer.h))]
    with torch.no_grad():
        baseline_model(input_ids=input_ids)
    for h in handles:
        h.remove()

    class Capture(fpwapCallback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.per_layer: dict[int, torch.Tensor] = {}

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            self.per_layer[layer_idx] = acts.detach().clone()
            return Emit(acts)

    cap = Capture()
    run = fpwap(
        model=TEST_MODEL_ID,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.bfloat16,
        seed=SEED,
    )
    run.run()

    for layer_idx, baseline in baseline_per_layer.items():
        got = cap.per_layer[layer_idx]
        assert torch.allclose(got, baseline, atol=1e-2, rtol=1e-2), (
            f"residual_post mismatch at layer {layer_idx}"
        )
