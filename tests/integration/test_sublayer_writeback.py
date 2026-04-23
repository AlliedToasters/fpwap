"""WriteBack at sub-layer hooks (attn_out, mlp_out).

Previously these hooks were read-only — the engine dispatched their
outputs AFTER the block had already residual-added them. For
interpretability intervention (e.g. zero attention at layer L and see
what happens), write-phase callbacks must be able to modify the sub-layer
output BEFORE it's added back to the residual stream.

Test strategy: WriteBack that zeros `attn_out` at layer 0. The downstream
residual_post at layer 0 must equal `residual + 0 + mlp(ln_2(residual))`,
which differs from the baseline residual_post (which has a nonzero
attention contribution).
"""
from __future__ import annotations

import pytest
import torch

from fpwap.callbacks.base import Callback
from fpwap.types import BatchResult, HookName, WriteBack

SEED = 0
N_SAMPLES = 4
SEQ_LEN = 6
HIDDEN = 16
N_LAYERS = 2
VOCAB = 32


def _tiny_gpt2() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=2,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


class ZeroSubLayer(Callback):
    """Zero a sub-layer output at the target layer (write-phase)."""

    phase = "write"

    def __init__(self, layer_idx: int, hook: HookName) -> None:
        self.target_layers = [layer_idx]
        self.target_hooks = (hook,)

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> BatchResult:
        return WriteBack(torch.zeros_like(acts))


@pytest.mark.integration
@pytest.mark.parametrize("hook", ["attn_out", "mlp_out"])
def test_writeback_at_sublayer_changes_layer_output(hook: str) -> None:
    from fpwap import Sweep
    from fpwap.callbacks.common import RawActivations

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, VOCAB, (N_SAMPLES, SEQ_LEN))
    dataset = [{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)]

    # Baseline: no intervention, capture layer-0 residual_post (pool last token).
    baseline = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[
            RawActivations(
                layers=[0], last_token_only=True, out_dtype=torch.float32
            )
        ],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run().activations(layer=0, hook="residual_post")

    # Intervention: zero the sub-layer at layer 0. The layer's residual_post
    # must differ from baseline (attn and mlp both contribute meaningfully on
    # random-init GPT-2).
    zero_cb = ZeroSubLayer(layer_idx=0, hook=hook)  # type: ignore[arg-type]
    steered = Sweep(
        model=model,
        dataset=dataset,
        seq_len=SEQ_LEN,
        callbacks=[
            zero_cb,
            RawActivations(
                layers=[0], last_token_only=True, out_dtype=torch.float32
            ),
        ],
        transport_dtype=torch.float32,
        microbatch_size=N_SAMPLES,
        seed=SEED,
        progress=False,
    ).run().activations(layer=0, hook="residual_post")

    assert not torch.allclose(baseline, steered, atol=1e-4), (
        f"zeroing {hook} at layer 0 didn't change residual_post — "
        f"WriteBack wasn't threaded through the block"
    )
    # Diff magnitude should be on the order of the sub-layer contribution,
    # i.e. not tiny noise.
    max_diff = (baseline - steered).abs().max().item()
    assert max_diff > 1e-3, (
        f"{hook} WriteBack diff {max_diff} suspiciously small"
    )
