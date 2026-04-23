"""IncrementalPCA: streaming mean + X^T X, eigendecomp at layer end.

Compare against torch's direct PCA over the full dataset. The incremental
accumulation should land at the same covariance (up to fp32 rounding), so
the top eigenvectors match up to a sign flip per axis.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 20
SEQ_LEN = 6
HIDDEN = 16
N_LAYERS = 2
N_HEAD = 2
VOCAB = 64
N_COMPONENTS = 4


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


def _direct_pca(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA via eigh on the full covariance. Returns (basis, mean, eigvals desc)."""
    x = x.to(torch.float32)
    mean = x.mean(dim=0)
    xc = x - mean
    cov = (xc.T @ xc) / x.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return eigvecs[:, order][:, :k].contiguous(), mean, eigvals[order][:k]


@pytest.mark.integration
def test_incremental_pca_matches_direct_pca() -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import IncrementalPCA, RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    # Ground truth: pull all last-token residuals, direct PCA.
    capture = RawActivations(
        layers="all", last_token_only=True, out_dtype=torch.float32
    )
    truth_result = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[capture],
        transport_dtype=torch.float32,
        microbatch_size=4,
        seed=SEED,
        progress=False,
    ).run()

    # Compute ground-truth PCA per layer.
    expected: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for layer in range(N_LAYERS):
        x = truth_result.activations(layer=layer, hook="residual_post")
        expected[layer] = _direct_pca(x, N_COMPONENTS)

    # Streaming PCA
    pca = IncrementalPCA(layers="all", n_components=N_COMPONENTS, last_token_only=True)
    result = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[pca],
        transport_dtype=torch.float32,
        microbatch_size=4,
        seed=SEED,
        progress=False,
    ).run()

    for layer in range(N_LAYERS):
        artifact = result.artifact(kind="pca_basis", layer=layer)
        basis = artifact.payload["basis"]
        mean = artifact.payload["mean"]
        evar = artifact.payload["explained_variance"]
        exp_basis, exp_mean, exp_evar = expected[layer]

        # Mean is dtype-deterministic.
        assert torch.allclose(mean, exp_mean, atol=1e-5, rtol=1e-5), (
            f"layer {layer}: mean diff {(mean - exp_mean).abs().max()}"
        )
        # Eigenvalues (variance explained) are sign-invariant.
        assert torch.allclose(evar, exp_evar, atol=1e-4, rtol=1e-4), (
            f"layer {layer}: explained_variance diff "
            f"{(evar - exp_evar).abs().max()}"
        )
        # Eigenvectors match up to a per-component sign flip. Compare abs dot.
        for k in range(N_COMPONENTS):
            dot = (basis[:, k] * exp_basis[:, k]).sum().abs().item()
            assert abs(dot - 1.0) < 1e-3, (
                f"layer {layer} component {k}: |<basis, exp>| = {dot}, expected ~1"
            )


@pytest.mark.integration
def test_incremental_pca_produces_orthonormal_basis() -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import IncrementalPCA

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))

    pca = IncrementalPCA(layers=[1], n_components=N_COMPONENTS)
    result = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[pca],
        transport_dtype=torch.float32,
        microbatch_size=4,
        seed=SEED,
        progress=False,
    ).run()

    basis = result.artifact(kind="pca_basis", layer=1).payload["basis"]
    assert basis.shape == (HIDDEN, N_COMPONENTS)
    gram = basis.T @ basis
    eye = torch.eye(N_COMPONENTS)
    assert torch.allclose(gram, eye, atol=1e-4), (
        f"basis not orthonormal: gram-eye diff {(gram - eye).abs().max()}"
    )
