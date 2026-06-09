"""Streaming-path bit-exactness on an MoE model with a converted checkpoint.

Issue #77: HF MoE checkpoints store per-expert weights on disk while the
transformers 5.x modules hold fused stacked parameters. The streaming loader
must apply the same checkpoint conversion `from_pretrained` does, then the
loop must produce activations identical to the pre-loaded model.

Uses a tiny Qwen3-MoE built in-test and saved in the on-disk per-expert
layout, so it runs in seconds and needs no downloads.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SEQ_LEN = 8
N_SAMPLES = 4
HIDDEN = 16
N_LAYERS = 2


def _write_tiny_qwen3_moe_snapshot(tmp_path: Path) -> tuple[torch.nn.Module, Path]:
    """Save a tiny Qwen3-MoE in the per-expert on-disk layout (see unit twin)."""
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        head_dim=8,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    src = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    src.eval()

    m = config.moe_intermediate_size
    shard: dict[str, torch.Tensor] = {}
    for name, value in src.state_dict().items():
        if name.endswith("mlp.experts.gate_up_proj"):
            base = name.rsplit(".gate_up_proj", 1)[0]
            for e in range(config.num_experts):
                shard[f"{base}.{e}.gate_proj.weight"] = value[e, :m, :].contiguous()
                shard[f"{base}.{e}.up_proj.weight"] = value[e, m:, :].contiguous()
        elif name.endswith("mlp.experts.down_proj"):
            base = name.rsplit(".down_proj", 1)[0]
            for e in range(config.num_experts):
                shard[f"{base}.{e}.down_proj.weight"] = value[e].contiguous()
        else:
            shard[name] = value.contiguous()

    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    save_file(shard, snapshot_dir / "model.safetensors")
    hf_index = {
        "metadata": {"total_size": 0},
        "weight_map": {k: "model.safetensors" for k in shard},
    }
    (snapshot_dir / "model.safetensors.index.json").write_text(json.dumps(hf_index))
    config.save_pretrained(snapshot_dir)
    return src, snapshot_dir


@pytest.mark.gpu
def test_moe_streaming_matches_preloaded(tmp_path: Path) -> None:
    pytest.importorskip("transformers.conversion_mapping")

    from fpwap import Callback, Sweep
    from fpwap.types import BatchResult, HookName

    src, snapshot_dir = _write_tiny_qwen3_moe_snapshot(tmp_path)

    torch.manual_seed(1)
    input_ids = torch.randint(0, 64, (N_SAMPLES, SEQ_LEN), device="cuda:0")
    attention_mask = torch.ones_like(input_ids)
    dataset = [
        {
            "input_ids": input_ids[i : i + 1],
            "attention_mask": attention_mask[i : i + 1],
        }
        for i in range(N_SAMPLES)
    ]

    class Capture(Callback):
        phase = "read"
        target_layers = "all"
        target_hooks: tuple[HookName, ...] = ("residual_post",)

        def __init__(self) -> None:
            self.acts: dict[int, torch.Tensor] = {
                i: torch.zeros(
                    N_SAMPLES,
                    SEQ_LEN,
                    HIDDEN,
                    dtype=torch.bfloat16,
                    device="cuda:0",
                )
                for i in range(N_LAYERS)
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

    # Run A: pre-loaded model (weights fused by from_pretrained itself).
    preloaded = src.to("cuda:0")
    cap_pre = Capture()
    Sweep(
        model=preloaded,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[cap_pre],
        transport_dtype=torch.bfloat16,
        microbatch_size=N_SAMPLES,
        seed=0,
        progress=False,
    ).run()

    del preloaded, src
    torch.cuda.empty_cache()

    # Run B: streaming from the per-expert snapshot — the #77 repro path.
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

    for layer_idx in range(N_LAYERS):
        pre = cap_pre.acts[layer_idx]
        stm = cap_stream.acts[layer_idx]
        max_diff = (pre.float() - stm.float()).abs().max().item()
        assert torch.equal(pre, stm), (
            f"preloaded vs streaming mismatch at layer {layer_idx}: "
            f"max abs diff {max_diff}"
        )
