"""GPU bit-exact: apply_final_norm vs HF's output_hidden_states.

Issue #1: without apply_final_norm, the last layer's residual_post is the raw
block output (pre-model.norm). HF's output_hidden_states[-1] applies the final
norm, causing a ~100x drift cliff that users will correctly interpret as a bug.

Tests:
  - apply_final_norm=True  → last-layer residual_post matches HF's hidden_states[-1]
  - apply_final_norm=False → last-layer residual_post diverges (raw post-block)
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 16
HIDDEN = 64
N_LAYERS = 2
N_HEAD = 2
VOCAB = 128


def _tiny_gpt2_cuda_bf16() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=N_HEAD,
    )
    torch.manual_seed(SEED)
    return GPT2LMHeadModel(config).to(dtype=torch.bfloat16, device="cuda:0").eval()


def _tiny_llama_cuda_bf16() -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEAD,
        max_position_embeddings=SEQ_LEN,
    )
    torch.manual_seed(SEED)
    return LlamaForCausalLM(config).to(dtype=torch.bfloat16, device="cuda:0").eval()


def _hf_last_hidden_state(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[-1].detach()


def _fpwap_last_layer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    apply_final_norm: bool,
) -> torch.Tensor:
    from fpwap import Callback, Emit, Sweep
    from fpwap.types import BatchResult, HookName

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def on_batch(
            self,
            layer_idx: int,
            hook: HookName,
            acts: torch.Tensor,
            sample_ids: torch.Tensor,
        ) -> BatchResult:
            return Emit(tensor=acts)

    cap = Capture()
    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(input_ids.shape[0])],
        seq_len=input_ids.shape[1],
        callbacks=[cap],
        transport_dtype=torch.bfloat16,
        seed=SEED,
        apply_final_norm=apply_final_norm,
        progress=False,
    )
    result = sweep.run()
    last_layer = N_LAYERS - 1
    return result.activations(last_layer, "residual_post")


@pytest.mark.gpu
def test_apply_final_norm_true_matches_hf_gpt2() -> None:
    model = _tiny_gpt2_cuda_bf16()
    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN), device="cuda:0")

    hf_last = _hf_last_hidden_state(model, input_ids)
    fpwap_last = _fpwap_last_layer(model, input_ids, apply_final_norm=True)

    fpwap_last_dev = fpwap_last.to(device=hf_last.device, dtype=hf_last.dtype)
    assert torch.equal(hf_last, fpwap_last_dev), (
        f"apply_final_norm=True should match HF output_hidden_states[-1]; "
        f"max diff = {(hf_last.float() - fpwap_last_dev.float()).abs().max().item()}"
    )


@pytest.mark.gpu
def test_apply_final_norm_false_diverges_gpt2() -> None:
    model = _tiny_gpt2_cuda_bf16()
    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN), device="cuda:0")

    hf_last = _hf_last_hidden_state(model, input_ids)
    fpwap_raw = _fpwap_last_layer(model, input_ids, apply_final_norm=False)

    fpwap_raw_dev = fpwap_raw.to(device=hf_last.device, dtype=hf_last.dtype)
    diff = (hf_last.float() - fpwap_raw_dev.float()).abs().max().item()
    assert not torch.equal(hf_last, fpwap_raw_dev), (
        "apply_final_norm=False should NOT match HF output_hidden_states[-1]"
    )
    assert diff > 0.01, (
        f"expected significant divergence without final norm, got max diff = {diff}"
    )


@pytest.mark.gpu
def test_apply_final_norm_true_matches_hf_llama() -> None:
    model = _tiny_llama_cuda_bf16()
    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN), device="cuda:0")

    hf_last = _hf_last_hidden_state(model, input_ids)
    fpwap_last = _fpwap_last_layer(model, input_ids, apply_final_norm=True)

    fpwap_last_dev = fpwap_last.to(device=hf_last.device, dtype=hf_last.dtype)
    assert torch.equal(hf_last, fpwap_last_dev), (
        f"apply_final_norm=True should match HF output_hidden_states[-1]; "
        f"max diff = {(hf_last.float() - fpwap_last_dev.float()).abs().max().item()}"
    )


@pytest.mark.gpu
def test_apply_final_norm_false_diverges_llama() -> None:
    model = _tiny_llama_cuda_bf16()
    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN), device="cuda:0")

    hf_last = _hf_last_hidden_state(model, input_ids)
    fpwap_raw = _fpwap_last_layer(model, input_ids, apply_final_norm=False)

    fpwap_raw_dev = fpwap_raw.to(device=hf_last.device, dtype=hf_last.dtype)
    diff = (hf_last.float() - fpwap_raw_dev.float()).abs().max().item()
    assert not torch.equal(hf_last, fpwap_raw_dev), (
        "apply_final_norm=False should NOT match HF output_hidden_states[-1]"
    )
    assert diff > 0.01, (
        f"expected significant divergence without final norm, got max diff = {diff}"
    )
