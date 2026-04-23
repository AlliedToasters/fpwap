"""on_layer_end returns must land in Result.artifacts.

Previously the engine discarded the return value, so callbacks accumulating
per-layer state had nowhere to ship it via the standard flow. This locks
the contract the API already promised.
"""
from __future__ import annotations

import pytest
import torch

from fpwap.callbacks.base import Callback
from fpwap.types import BatchResult, HookName, LayerArtifact

SEED = 0
N_SAMPLES = 3
SEQ_LEN = 4
HIDDEN = 8
N_LAYERS = 2


def _tiny_gpt2() -> torch.nn.Module:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=16,
        n_positions=SEQ_LEN,
        n_embd=HIDDEN,
        n_layer=N_LAYERS,
        n_head=2,
    )
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


class EmitPerLayerCount(Callback):
    """Emits one LayerArtifact per layer carrying the microbatch-count seen."""

    phase = "read"
    target_layers = "all"
    target_hooks: tuple[HookName, ...] = ("residual_post",)

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}

    def on_batch(
        self,
        layer_idx: int,
        hook: HookName,
        acts: torch.Tensor,
        sample_ids: torch.Tensor,
    ) -> BatchResult:
        self.counts[layer_idx] = self.counts.get(layer_idx, 0) + 1
        return None

    def on_layer_end(self, layer_idx: int) -> LayerArtifact | None:
        return LayerArtifact(kind="mb_count", payload=self.counts[layer_idx])


@pytest.mark.integration
def test_on_layer_end_return_lands_in_artifacts() -> None:
    from fpwap import Sweep

    model = _tiny_gpt2()
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 16, (N_SAMPLES, SEQ_LEN))

    cb = EmitPerLayerCount()
    result = Sweep(
        model=model,
        dataset=[{"input_ids": input_ids[i : i + 1]} for i in range(N_SAMPLES)],
        seq_len=SEQ_LEN,
        callbacks=[cb],
        transport_dtype=torch.float32,
        microbatch_size=1,  # 3 microbatches per layer
        seed=SEED,
        progress=False,
    ).run()

    for i in range(N_LAYERS):
        art = result.artifact(kind="mb_count", layer=i)
        assert art.payload == 3
        assert art.key.layer_idx == i
        assert art.key.kind == "mb_count"
