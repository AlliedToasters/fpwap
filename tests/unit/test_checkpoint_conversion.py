"""Contract for transformers 5.x checkpoint conversion in the loader (issue #77).

HF MoE checkpoints store per-expert weights on disk
(`...mlp.experts.E.{gate_proj,up_proj,down_proj}.weight`) while the
instantiated transformers 5.x modules hold fused stacked parameters
(`...mlp.experts.{gate_up_proj,down_proj}`). `from_pretrained` reconciles the
two through the global WeightConverter mapping; fpwap builds its accel index
from raw safetensors keys, so module param names must be resolved through the
same conversion mapping or `_load_layer` KeyErrors on every MoE model.

These tests exercise the conversion plumbing on a tiny Qwen3-MoE checkpoint
written in the on-disk per-expert layout. CPU-only; CI-safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers.conversion_mapping")


def _write_tiny_qwen3_moe_snapshot(tmp_path: Path) -> tuple[torch.nn.Module, Path]:
    """Save a tiny Qwen3-MoE in the on-disk per-expert layout.

    Returns (source_model_with_real_fused_weights, snapshot_dir). The shard
    keys deliberately mismatch the module param names — exactly the layout
    `from_pretrained` reconciles via MergeModulelist + Concatenate.
    """
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
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
    src = Qwen3MoeForCausalLM(config)
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


def _build_empty(snapshot_dir: Path):
    from fpwap.loader import build_empty_model_and_index

    return build_empty_model_and_index(
        model_id=str(snapshot_dir), snapshot_dir=snapshot_dir, dtype=torch.float32
    )


class TestBuildConversionPlans:
    def test_plans_cover_fused_moe_params(self, tmp_path: Path) -> None:
        from fpwap.loader import build_conversion_plans

        _, snapshot_dir = _write_tiny_qwen3_moe_snapshot(tmp_path)
        model, accel_index, _ = _build_empty(snapshot_dir)
        plans = build_conversion_plans(model, accel_index)

        for layer in range(2):
            for fused in ("gate_up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.experts.{fused}"
                assert name in plans, f"no conversion plan for {name}"
                assert name not in accel_index

    def test_plan_sources_are_raw_checkpoint_keys_in_expert_order(
        self, tmp_path: Path
    ) -> None:
        from fpwap.loader import build_conversion_plans

        _, snapshot_dir = _write_tiny_qwen3_moe_snapshot(tmp_path)
        model, accel_index, _ = _build_empty(snapshot_dir)
        plans = build_conversion_plans(model, accel_index)

        plan = plans["model.layers.0.mlp.experts.down_proj"]
        source_keys = [key for key, _pattern in plan.sources]
        assert source_keys == [
            f"model.layers.0.mlp.experts.{e}.down_proj.weight" for e in range(4)
        ]
        for key in source_keys:
            assert key in accel_index

    def test_no_plans_for_plain_dense_model(self, tmp_path: Path) -> None:
        """A model whose checkpoint layout matches its module layout needs no
        conversion — the fast path must stay plan-free."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from fpwap.loader import build_conversion_plans

        config = GPT2Config(
            vocab_size=40, n_positions=8, n_embd=16, n_layer=2, n_head=2
        )
        torch.manual_seed(0)
        src = GPT2LMHeadModel(config)
        state_dict = {
            k: v.contiguous()
            for k, v in src.state_dict().items()
            if k != "lm_head.weight"
        }
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        save_file(state_dict, snapshot_dir / "model.safetensors")
        hf_index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model.safetensors" for k in state_dict},
        }
        (snapshot_dir / "model.safetensors.index.json").write_text(
            json.dumps(hf_index)
        )
        config.save_pretrained(snapshot_dir)

        model, accel_index, _ = _build_empty(snapshot_dir)
        assert build_conversion_plans(model, accel_index) == {}


class TestConvertingWeightsLoader:
    def _make_loader(self, snapshot_dir: Path):
        from accelerate.utils import OffloadedWeightsLoader

        from fpwap.loader import ConvertingWeightsLoader, build_conversion_plans

        model, accel_index, _ = _build_empty(snapshot_dir)
        plans = build_conversion_plans(model, accel_index)
        base = OffloadedWeightsLoader(index=accel_index)
        return model, ConvertingWeightsLoader(base, plans, config=model.config)

    def test_fuses_gate_up_and_down_on_demand(self, tmp_path: Path) -> None:
        src, snapshot_dir = _write_tiny_qwen3_moe_snapshot(tmp_path)
        _, loader = self._make_loader(snapshot_dir)

        for layer in range(2):
            experts = src.model.layers[layer].mlp.experts
            fused_gate_up = loader[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
            fused_down = loader[f"model.layers.{layer}.mlp.experts.down_proj"]
            assert torch.equal(fused_gate_up, experts.gate_up_proj.data)
            assert torch.equal(fused_down, experts.down_proj.data)

    def test_raw_keys_pass_through(self, tmp_path: Path) -> None:
        src, snapshot_dir = _write_tiny_qwen3_moe_snapshot(tmp_path)
        _, loader = self._make_loader(snapshot_dir)

        embed = loader["model.embed_tokens.weight"]
        assert torch.equal(embed, src.model.embed_tokens.weight.data)

    def test_unknown_key_raises_keyerror(self, tmp_path: Path) -> None:
        _, snapshot_dir = _write_tiny_qwen3_moe_snapshot(tmp_path)
        _, loader = self._make_loader(snapshot_dir)

        with pytest.raises(KeyError):
            loader["model.layers.0.mlp.experts.nonexistent"]


class TestLoadLayerWithConversion:
    def test_load_layer_materializes_moe_layer(self, tmp_path: Path) -> None:
        """The issue-#77 repro: _load_layer over an MoE block must resolve the
        fused module param names instead of KeyError'ing on them."""
        from fpwap.engine import _OffloadStreamer
        from fpwap.models import get_plumbing

        src, snapshot_dir = _write_tiny_qwen3_moe_snapshot(tmp_path)
        model, accel_index, _ = _build_empty(snapshot_dir)
        streamer = _OffloadStreamer(
            accel_index, torch.device("cpu"), model=model
        )
        try:
            plumbing = get_plumbing(model)
            streamer.load_layer(model, 0, plumbing)

            src_layer = src.model.layers[0]
            dst_layer = model.model.layers[0]
            for (src_name, src_p), (dst_name, dst_p) in zip(
                src_layer.named_parameters(),
                dst_layer.named_parameters(),
                strict=True,
            ):
                assert src_name == dst_name
                assert dst_p.device.type != "meta", f"{dst_name} still on meta"
                assert torch.equal(src_p, dst_p), f"value mismatch at {dst_name}"
        finally:
            streamer.close()


class TestAdvisorVirtualSources:
    def test_fused_name_advises_source_ranges(self, tmp_path: Path) -> None:
        """fadvise on a fused module param name must hit the underlying
        per-expert checkpoint ranges, or MoE sweeps silently lose page-cache
        discipline (the #61/#69 regime)."""
        from unittest.mock import patch

        from fpwap.loader import ShardPageAdvisor

        _, snapshot_dir = _write_tiny_qwen3_moe_snapshot(tmp_path)
        from fpwap.loader import build_accel_index_from_hf_cache

        accel_index = build_accel_index_from_hf_cache(snapshot_dir)
        sources = [
            f"model.layers.0.mlp.experts.{e}.down_proj.weight" for e in range(4)
        ]
        advisor = ShardPageAdvisor(
            accel_index,
            virtual_sources={"model.layers.0.mlp.experts.down_proj": sources},
        )

        with patch("fpwap.loader.os.posix_fadvise") as mock_fadvise:
            advisor.advise_dontneed(["model.layers.0.mlp.experts.down_proj"])
        assert mock_fadvise.call_count == 4
