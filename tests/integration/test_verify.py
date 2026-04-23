"""verify=True: fpwap diffs residual_post against a captured naive baseline.

Pre-loaded model path only (streaming verify would need cpu_offload in
parallel). The fp32 path must be bit-exact; bf16 requires microbatch_size
== n_samples (per the bf16_microbatch_determinism memo).
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 8
HIDDEN = 32
N_LAYERS = 2
N_HEAD = 2
VOCAB = 64


def _tiny_gpt2() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=N_HEAD,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.mark.integration
def test_verify_passes_when_fpwap_matches_naive() -> None:
    from fpwap import Sweep

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    sweep = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
        verify=True,
    )
    sweep.run()  # must not raise


@pytest.mark.integration
def test_verify_with_padded_mask_masks_pad_positions() -> None:
    """verify=True must ignore pad positions (HF's output is undefined there);
    compare only at real tokens, matching the padded_batch test's wisdom.
    """
    from fpwap import Sweep

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    attention_mask = torch.ones((N_SAMPLES, SEQ_LEN), dtype=torch.long)
    attention_mask[:, :2] = 0

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
        callbacks=[],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
        verify=True,
    )
    sweep.run()  # must not raise; pad positions excluded from compare


@pytest.mark.integration
def test_verify_raises_on_streaming_model(tmp_path) -> None:
    from fpwap import Sweep

    sweep = Sweep(
        model=str(tmp_path),  # string path triggers the streaming branch
        dataset=[{"input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long)}],
        seq_len=SEQ_LEN,
        callbacks=[],
        transport_dtype=torch.float32,
        seed=SEED,
        progress=False,
        verify=True,
    )
    with pytest.raises(NotImplementedError, match="pre-loaded nn.Module"):
        sweep.run()
