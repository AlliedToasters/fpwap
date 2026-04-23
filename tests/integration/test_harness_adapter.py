"""Integration test for harness_adapter.run_fpwap with a tiny GPT-2 snapshot.

Proves the harness contract tuple (load_s, extract_times, fingerprint,
peak_VRAM) has the right shape and values are sane. CPU-only, no GPU
required.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from harness_adapter import run_fpwap

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 6
HIDDEN = 16
N_LAYERS = 2
N_HEAD = 2
VOCAB = 32


def _write_tiny_gpt2_snapshot(snapshot_dir: Path) -> None:
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


@pytest.mark.integration
def test_run_fpwap_contract(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    hf_layers = list(range(N_LAYERS))

    load_s, extract_times, fingerprint, peak_vram = run_fpwap(
        model_id=str(snapshot_dir),
        pretokenized_batch=input_ids,
        hf_layers=hf_layers,
        iters=2,
        device="cpu",
        dtype=torch.float32,
    )

    assert isinstance(load_s, float) and load_s > 0
    assert isinstance(extract_times, list) and len(extract_times) == 2
    assert all(t > 0 for t in extract_times)
    assert isinstance(fingerprint, float) and math.isfinite(fingerprint)
    assert isinstance(peak_vram, float)


@pytest.mark.integration
def test_run_fpwap_fingerprint_stable(tmp_path: Path) -> None:
    """Same inputs -> same fingerprint across iters."""
    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    _, times1, fp1, _ = run_fpwap(
        model_id=str(snapshot_dir),
        pretokenized_batch=input_ids,
        hf_layers=[0, 1],
        iters=1,
        device="cpu",
        dtype=torch.float32,
    )
    _, times2, fp2, _ = run_fpwap(
        model_id=str(snapshot_dir),
        pretokenized_batch=input_ids,
        hf_layers=[0, 1],
        iters=1,
        device="cpu",
        dtype=torch.float32,
    )

    assert fp1 == fp2, f"fingerprint drifted: {fp1} vs {fp2}"


@pytest.mark.integration
def test_run_fpwap_with_attention_mask(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    _write_tiny_gpt2_snapshot(snapshot_dir)

    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    attention_mask = torch.ones(N_SAMPLES, SEQ_LEN, dtype=torch.long)
    attention_mask[:, :2] = 0

    load_s, extract_times, fingerprint, peak_vram = run_fpwap(
        model_id=str(snapshot_dir),
        pretokenized_batch=input_ids,
        hf_layers=[N_LAYERS - 1],
        iters=1,
        device="cpu",
        dtype=torch.float32,
        attention_mask=attention_mask,
    )

    assert math.isfinite(fingerprint)
    assert len(extract_times) == 1
