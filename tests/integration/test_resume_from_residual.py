"""Resume-from-residual is bit-exact when the seed dtype matches the reader
(CPU streaming path).

The seek primitive: seed the residual buffer with a stored residual (the input
to block ``start_layer``) and run only blocks ``[start_layer, N)``. Re-running a
sub-range of blocks over a bit-identical seed reproduces the cold pass *bitwise*
— the seed is the only thing that can perturb a resume.

So the whole correctness question collapses to the seed's dtype, which is the
caller's to guarantee:

- ``test_resume_is_bit_exact`` — seed stored at the reader's residual dtype
  (fp32 reader / fp32 store, and bf16 reader / bf16 store, the real lens case):
  ``max_abs == 0`` across every re-run layer.
- ``test_undermatched_seed_diverges`` — seed stored *below* the reader's dtype
  (fp32 reader, bf16 store): resume is no longer exact. This is the case lens's
  keyframe-eligibility filter must exclude, pinned here so the invariant can't
  silently rot.

Proven structurally on a tiny model on CPU; the GPU / fp8 receipts live with the
consumers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SEED = 0
N_SAMPLES = 3
SEQ_LEN = 7
HIDDEN = 16
N_LAYERS = 4
N_HEAD = 2
VOCAB = 32


def _write_tiny_gpt2_snapshot(snapshot_dir: Path) -> None:
    """Self-contained HF-style snapshot (no lm_head; fpwap never runs it)."""
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
    (snapshot_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {k: "model.safetensors" for k in state_dict},
            }
        )
    )


def _cold_residual_posts(
    snapshot_dir: Path,
    input_ids: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    store_dtype: torch.dtype,
    apply_final_norm: bool = False,
) -> dict[int, torch.Tensor]:
    """One cold sweep capturing residual_post at every layer, stored at
    ``store_dtype`` (what a keyframe on disk would hold)."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    sweep = Sweep(
        model=str(snapshot_dir),
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[
            RawActivations(
                layers="all",
                hook="residual_post",
                last_token_only=False,
                out_dtype=store_dtype,
            )
        ],
        transport_dtype=compute_dtype,
        execution_device="cpu",
        microbatch_size=1,
        apply_final_norm=apply_final_norm,
        progress=False,
    )
    result = sweep.run()
    return {
        layer: result.activations(layer=layer, hook="residual_post")
        for layer in range(N_LAYERS)
    }


def _resume(
    snapshot_dir: Path,
    input_ids: torch.Tensor,
    seed: torch.Tensor,
    start_layer: int,
    *,
    compute_dtype: torch.dtype,
    store_dtype: torch.dtype,
    apply_final_norm: bool = False,
) -> tuple[dict[int, torch.Tensor], int]:
    """Seed block ``start_layer`` and run to the end, capturing residual_post.

    Returns the per-layer captures and the weight bytes the streamer moved
    (fewer than cold, since the layers below start_layer never load)."""
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    dataset = [
        {"input_ids": input_ids[i : i + 1], "residual": seed[i]}
        for i in range(N_SAMPLES)
    ]
    sweep = Sweep(
        model=str(snapshot_dir),
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[
            RawActivations(
                layers=list(range(start_layer, N_LAYERS)),
                hook="residual_post",
                last_token_only=False,
                out_dtype=store_dtype,
            )
        ],
        transport_dtype=compute_dtype,
        execution_device="cpu",
        microbatch_size=1,
        apply_final_norm=apply_final_norm,
        start_layer=start_layer,
        progress=False,
    )
    result = sweep.run()
    captures = {
        layer: result.activations(layer=layer, hook="residual_post")
        for layer in range(start_layer, N_LAYERS)
    }
    return captures, result.profile.bytes_moved()["weights"]


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("tiny_gpt2") / "snapshot"
    _write_tiny_gpt2_snapshot(d)
    return d


# (compute dtype, keyframe store dtype) — both at the reader's residual dtype,
# so the seed is bit-identical to what the cold pass carried.
MATCHED_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, torch.bfloat16, id="bf16"),
]


@pytest.mark.integration
@pytest.mark.parametrize("compute_dtype,store_dtype", MATCHED_DTYPES)
@pytest.mark.parametrize("start_layer", [1, 2, 3])
def test_resume_is_bit_exact(
    snapshot: Path,
    start_layer: int,
    compute_dtype: torch.dtype,
    store_dtype: torch.dtype,
) -> None:
    torch.manual_seed(SEED + 1)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    cold = _cold_residual_posts(
        snapshot, input_ids, compute_dtype=compute_dtype, store_dtype=store_dtype
    )
    # The input to block start_layer is residual_post of block start_layer - 1.
    seed = cold[start_layer - 1]
    resumed, _ = _resume(
        snapshot,
        input_ids,
        seed,
        start_layer,
        compute_dtype=compute_dtype,
        store_dtype=store_dtype,
    )

    for layer in range(start_layer, N_LAYERS):
        max_abs = (resumed[layer].float() - cold[layer].float()).abs().max().item()
        assert torch.equal(resumed[layer], cold[layer]), (
            f"resume from layer {start_layer}: residual_post at layer {layer} "
            f"diverged from cold (max_abs={max_abs})"
        )


@pytest.mark.integration
@pytest.mark.parametrize("compute_dtype,store_dtype", MATCHED_DTYPES)
def test_resume_reproduces_final_norm(
    snapshot: Path, compute_dtype: torch.dtype, store_dtype: torch.dtype
) -> None:
    """The surprise path's seed: resume to full depth reproduces the final
    layer's post-final-norm output bitwise (what the LM head consumes)."""
    torch.manual_seed(SEED + 2)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    raw = _cold_residual_posts(
        snapshot, input_ids, compute_dtype=compute_dtype, store_dtype=store_dtype
    )
    cold = _cold_residual_posts(
        snapshot,
        input_ids,
        compute_dtype=compute_dtype,
        store_dtype=store_dtype,
        apply_final_norm=True,
    )
    start_layer = 2
    resumed, _ = _resume(
        snapshot,
        input_ids,
        raw[start_layer - 1],
        start_layer,
        compute_dtype=compute_dtype,
        store_dtype=store_dtype,
        apply_final_norm=True,
    )

    last = N_LAYERS - 1
    max_abs = (resumed[last].float() - cold[last].float()).abs().max().item()
    assert torch.equal(resumed[last], cold[last]), (
        f"resume final-norm output diverged from cold (max_abs={max_abs})"
    )


@pytest.mark.integration
def test_undermatched_seed_diverges(snapshot: Path) -> None:
    """The eligibility guard, executable: an fp32 reader seeded from a bf16
    keyframe (store dtype below the residual dtype) is NOT bit-exact. lens must
    route this to a cold pass; pinned here so the invariant can't silently rot."""
    torch.manual_seed(SEED + 5)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    # fp32 compute everywhere, but the keyframe was stored bf16.
    cold_true = _cold_residual_posts(
        snapshot, input_ids, compute_dtype=torch.float32, store_dtype=torch.float32
    )
    cold_lossy = _cold_residual_posts(
        snapshot, input_ids, compute_dtype=torch.float32, store_dtype=torch.bfloat16
    )
    start_layer = 2
    resumed, _ = _resume(
        snapshot,
        input_ids,
        cold_lossy[start_layer - 1],  # bf16 seed into an fp32 reader
        start_layer,
        compute_dtype=torch.float32,
        store_dtype=torch.float32,
    )
    assert not torch.equal(resumed[start_layer], cold_true[start_layer]), (
        "a bf16 seed into an fp32 reader unexpectedly reproduced cold bitwise; "
        "the dtype invariant would be vacuous"
    )


@pytest.mark.integration
def test_resume_skips_lower_layer_weights(snapshot: Path) -> None:
    """A resume must not pay for the layers below the seed — it loads strictly
    fewer layer weights the deeper it seeds."""
    torch.manual_seed(SEED + 3)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    cold = _cold_residual_posts(
        snapshot, input_ids, compute_dtype=torch.bfloat16, store_dtype=torch.bfloat16
    )
    _, shallow_bytes = _resume(
        snapshot, input_ids, cold[0], 1,
        compute_dtype=torch.bfloat16, store_dtype=torch.bfloat16,
    )
    _, deep_bytes = _resume(
        snapshot, input_ids, cold[2], 3,
        compute_dtype=torch.bfloat16, store_dtype=torch.bfloat16,
    )
    # Resuming at layer 3 runs one block; at layer 1 runs three. Strictly fewer.
    assert deep_bytes < shallow_bytes


@pytest.mark.integration
def test_seed_requires_residual(snapshot: Path) -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    torch.manual_seed(SEED + 4)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    sweep = Sweep(
        model=str(snapshot),
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[RawActivations(layers=[2, 3], hook="residual_post")],
        transport_dtype=torch.bfloat16,
        execution_device="cpu",
        microbatch_size=1,
        start_layer=2,
        progress=False,
    )
    with pytest.raises(ValueError, match="residual"):
        sweep.run()
