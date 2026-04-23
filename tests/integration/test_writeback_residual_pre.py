"""WriteBack at residual_pre: a write-phase callback can modify the residual
stream feeding into a layer. This is the steering pattern at layer input
(vs residual_post which intervenes on layer output).
"""
from __future__ import annotations

import pytest
import torch

from fpwap.callbacks.base import Callback
from fpwap.types import BatchResult, HookName, WriteBack

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


class ZeroResidualPre(Callback):
    """At the target layer, blanks residual_pre to zero before layer_forward."""

    phase = "write"
    target_hooks: tuple[HookName, ...] = ("residual_pre",)

    def __init__(self, layer_idx: int) -> None:
        self.target_layers = [layer_idx]

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> BatchResult:
        return WriteBack(torch.zeros_like(acts))


@pytest.mark.integration
def test_writeback_at_residual_pre_changes_downstream_output() -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    # Baseline: no intervention.
    capture = RawActivations(
        layers="all", hook="residual_post", last_token_only=False, out_dtype=torch.float32
    )
    baseline = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[capture],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run()
    baseline_layer0 = baseline.activations(layer=0, hook="residual_post").clone()
    baseline_layer1 = baseline.activations(layer=1, hook="residual_post").clone()

    # Intervention: zero residual_pre at layer 0. Layer 0's residual_post is
    # entirely determined by a zero input + block(0) weights, so it should
    # differ from the baseline. Layer 1 inherits layer 0's output and also
    # diverges.
    zero_pre = ZeroResidualPre(layer_idx=0)
    capture2 = RawActivations(
        layers="all", hook="residual_post", last_token_only=False, out_dtype=torch.float32
    )
    out = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[zero_pre, capture2],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run()
    steered_layer0 = out.activations(layer=0, hook="residual_post")
    steered_layer1 = out.activations(layer=1, hook="residual_post")

    assert not torch.allclose(steered_layer0, baseline_layer0), (
        "zero-ing residual_pre at layer 0 should change layer 0's residual_post"
    )
    assert not torch.allclose(steered_layer1, baseline_layer1), (
        "intervention should also propagate to layer 1"
    )
