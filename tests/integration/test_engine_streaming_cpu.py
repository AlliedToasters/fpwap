"""CPU-only streaming-path forcing test.

Exercises the full engine loop with `model=<snapshot_dir>`: the model is
constructed on meta, weights are mmap'd from safetensors and installed
per-layer via _load_layer / _unload_layer. Compares per-layer
residual_post against a naive forward of the same weights loaded
conventionally.

Bypasses AlignDevicesHook entirely — this is the path that unlocks
running oversized models on a single consumer GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 6
HIDDEN = 16
N_LAYERS = 2
N_HEAD = 2
VOCAB = 32


def _write_tiny_gpt2_snapshot(snapshot_dir: Path) -> torch.nn.Module:
    """Self-contained HF-style snapshot + return a ground-truth model."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=N_HEAD,
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(snapshot_dir)

    torch.manual_seed(SEED)
    src = GPT2LMHeadModel(config)
    src.eval()

    state_dict = {
        k: v.contiguous() for k, v in src.state_dict().items() if k != "lm_head.weight"
    }
    save_file(state_dict, snapshot_dir / "model.safetensors")
    hf_index = {
        "metadata": {"total_size": 0},
        "weight_map": {k: "model.safetensors" for k in state_dict},
    }
    (snapshot_dir / "model.safetensors.index.json").write_text(json.dumps(hf_index))
    return src


def _naive_per_layer_residual_post(
    model: torch.nn.Module, input_ids: torch.Tensor
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
def test_streaming_path_matches_naive(tmp_path: Path) -> None:
    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    snapshot_dir = tmp_path / "snapshot"
    src = _write_tiny_gpt2_snapshot(snapshot_dir)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    # Naive reference: use the already-loaded source model with real weights.
    baseline = _naive_per_layer_residual_post(src, input_ids)

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
        model=str(snapshot_dir),
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[cap],
        transport_dtype=torch.float32,
        execution_device="cpu",
        seed=SEED,
        apply_final_norm=False,
    )
    result = run.run()

    for layer_idx, baseline_hidden in baseline.items():
        got = cap.acts[layer_idx]
        max_diff = (got - baseline_hidden).abs().max().item()
        assert torch.allclose(got, baseline_hidden, atol=1e-5, rtol=1e-5), (
            f"residual_post mismatch at layer {layer_idx}: max abs diff {max_diff}"
        )

    # Profile should have recorded per-layer weight bytes since this is the
    # streaming path, not the pre-loaded one.
    assert result.profile.bytes_moved()["weights"] > 0
