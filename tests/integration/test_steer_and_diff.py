"""SteerInBasis + DiffOfMeans reference callbacks end-to-end.

SteerInBasis: pairs with IncrementalPCA from a prior sweep — apply `alpha *
basis[:, 0]` to layer L's residual and confirm the layer-L residual_post
actually moves by `alpha * basis_direction` relative to the baseline.

DiffOfMeans: user-supplied binary labels → per-class running means,
returned as a per-layer artifact. Compare against a direct computation on
the full captured activations.
"""
from __future__ import annotations

import pytest
import torch

SEED = 0
N_SAMPLES = 8
SEQ_LEN = 6
HIDDEN = 16
N_LAYERS = 2
N_HEAD = 2
VOCAB = 48


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
def test_steer_in_basis_shifts_residual_by_alpha_times_direction() -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import (
        IncrementalPCA,
        RawActivations,
        SteerInBasis,
    )

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    # Pass 1: fit PCA on layer 0.
    pca = IncrementalPCA(layers=[0], n_components=4)
    pca_result = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[pca],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run()
    basis_art = pca_result.artifact(kind="pca_basis", layer=0)

    # Pass 2: steer at layer 0 residual_post with alpha=1.5 along component 0.
    # RawActivations at residual_pre of layer 1 reads buffer[0], which
    # contains the post-WriteBack value — the canonical place to observe
    # the effect of a layer-0 residual_post WriteBack.
    alpha = 1.5
    direction = basis_art.payload["basis"][:, 0]
    steer = SteerInBasis(
        basis_artifact=basis_art,
        direction_idx=0,
        alpha=alpha,
        layers=[0],
    )
    pre_capture_steered = RawActivations(
        layers=[1], hook="residual_pre", last_token_only=True, out_dtype=torch.float32
    )
    pre_capture_baseline = RawActivations(
        layers=[1], hook="residual_pre", last_token_only=True, out_dtype=torch.float32
    )

    baseline_pre = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[pre_capture_baseline],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run().activations(layer=1, hook="residual_pre")

    steered_pre = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[steer, pre_capture_steered],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run().activations(layer=1, hook="residual_pre")

    # layer-1 residual_pre = (layer-0 residual_post, written to buffer) so the
    # steered minus baseline should equal exactly alpha * direction.
    delta = steered_pre - baseline_pre
    expected = alpha * direction
    assert torch.allclose(delta, expected.expand_as(delta), atol=1e-5, rtol=1e-5), (
        f"steering shifted residual by a different direction than basis[:, 0]: "
        f"max diff {(delta - expected).abs().max().item()}"
    )


@pytest.mark.integration
def test_diff_of_means_direction_composes_with_steer() -> None:
    """End-to-end probing recipe: DoM fits a direction, SteerInBasis applies it.

    Locks the composition: DiffOfMeans' payload["direction"] is 1D [H],
    and SteerInBasis treats it as the single steer direction (no column
    slicing needed).
    """
    from fpwap import Sweep
    from fpwap.callbacks.common import DiffOfMeans, RawActivations, SteerInBasis

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    # Pass 1: fit DoM at layer 0.
    dom_result = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[DiffOfMeans(labels=labels, layers=[0])],
        transport_dtype=torch.float32,
        microbatch_size=4,
        seed=SEED,
        progress=False,
    ).run()
    dom_art = dom_result.artifact(kind="diff_of_means", layer=0)
    direction = dom_art.payload["direction"]
    assert direction.shape == (HIDDEN,)

    # Pass 2: steer at layer 0 residual_post with the DoM direction, then
    # confirm layer-1 residual_pre moved by exactly alpha * direction.
    alpha = 0.75
    steer = SteerInBasis(
        basis_artifact=dom_art, direction_idx=0, alpha=alpha, layers=[0]
    )
    pre_cap = RawActivations(
        layers=[1], hook="residual_pre", last_token_only=True, out_dtype=torch.float32
    )
    steered_pre = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[steer, pre_cap],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run().activations(layer=1, hook="residual_pre")

    pre_cap_base = RawActivations(
        layers=[1], hook="residual_pre", last_token_only=True, out_dtype=torch.float32
    )
    baseline_pre = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[pre_cap_base],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run().activations(layer=1, hook="residual_pre")

    delta = steered_pre - baseline_pre
    expected = (alpha * direction).expand_as(delta)
    assert torch.allclose(delta, expected, atol=1e-5, rtol=1e-5), (
        f"DoM → SteerInBasis composition broken: max diff "
        f"{(delta - expected).abs().max().item()}"
    )


@pytest.mark.integration
def test_diff_of_means_matches_direct_computation() -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import DiffOfMeans, RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    # Capture last-token activations, compute DoM directly.
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
    expected_directions: dict[int, torch.Tensor] = {}
    for layer in range(N_LAYERS):
        x = truth_result.activations(layer=layer, hook="residual_post")
        m0 = x[labels == 0].mean(dim=0)
        m1 = x[labels == 1].mean(dim=0)
        expected_directions[layer] = m1 - m0

    # Streaming DoM
    dom = DiffOfMeans(labels=labels, layers="all")
    result = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[dom],
        transport_dtype=torch.float32,
        microbatch_size=4,
        seed=SEED,
        progress=False,
    ).run()

    for layer in range(N_LAYERS):
        art = result.artifact(kind="diff_of_means", layer=layer)
        got = art.payload["direction"]
        exp = expected_directions[layer]
        assert art.payload["counts"] == {0: 4, 1: 4}
        assert torch.allclose(got, exp, atol=1e-5, rtol=1e-5), (
            f"layer {layer}: direction diff {(got - exp).abs().max()}"
        )
