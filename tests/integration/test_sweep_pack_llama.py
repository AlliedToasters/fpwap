"""End-to-end `Sweep(pack=True)` on tiny Llama — phase 3 of the pack pilot.

Asserts that running the engine with `pack=True` against a tiny Llama
produces the same per-real-token activations as the same-kernel padded
forward (FlexAttention with a per-batch causal+padding BlockMask). This is
the integration-level contract for packed mode: input layout and dispatch
plumbing change, but the math at real-token positions is preserved.
"""
from __future__ import annotations

from typing import Any
from typing import cast as _cast

import pytest
import torch
from torch import Tensor

from fpwap import Callback, Emit, RaggedTensor, Sweep
from fpwap.types import HookName

SEED = 0
HIDDEN = 32
INTERMEDIATE = 64
N_HEAD = 4
N_KV_HEAD = 2
VOCAB = 64
MAX_SEQ = 16
PAD_TOKEN = 0


def _tiny_llama_flex() -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=2, num_attention_heads=N_HEAD,
        num_key_value_heads=N_KV_HEAD, max_position_embeddings=MAX_SEQ,
        pad_token_id=PAD_TOKEN, attn_implementation="flex_attention",
    )
    torch.manual_seed(SEED)
    m = LlamaForCausalLM(cfg)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _padded_dataset(real_lengths: list[int], max_seq: int) -> list[dict[str, Tensor]]:
    torch.manual_seed(SEED + 1)
    items = []
    for L in real_lengths:
        ids = torch.full((1, max_seq), PAD_TOKEN, dtype=torch.long)
        ids[0, :L] = torch.randint(1, VOCAB, (L,))
        mask = torch.zeros((1, max_seq), dtype=torch.long)
        mask[0, :L] = 1
        items.append({"input_ids": ids, "attention_mask": mask})
    return items


class _PackedRaw(Callback):
    """Captures residual_post as ragged emits under pack mode."""

    accepts_packed = True
    target_layers = "all"
    target_hooks = ("residual_post",)
    phase = "read"

    def on_batch(
        self, layer_idx: int, hook: HookName, acts: Tensor, sample_ids: Tensor
    ) -> Emit:
        # Under pack mode, acts is a RaggedTensor.
        rt = _cast(RaggedTensor, acts)
        return Emit(
            tensor=rt.flat.detach().to(torch.float32).cpu(),
            sample_lengths=rt.lengths.detach().cpu().to(torch.int64),
        )


def _padded_reference_per_layer(
    model: torch.nn.Module, items: list[dict[str, Tensor]]
) -> dict[int, Tensor]:
    """Run a same-kernel padded forward block-by-block, capturing residual_post."""
    from transformers.integrations.flex_attention import make_flex_block_causal_mask

    from fpwap.models.llama import LlamaPlumbing

    pl = LlamaPlumbing()
    inner = _cast(Any, model).model
    rotary_emb = inner.rotary_emb

    input_ids = torch.cat([it["input_ids"] for it in items], dim=0)
    attn_mask = torch.cat([it["attention_mask"] for it in items], dim=0)
    n, max_seq = input_ids.shape

    out_per_layer: dict[int, Tensor] = {}
    with torch.no_grad():
        h = pl.embed(model, input_ids)
        pos_ids = (
            torch.arange(max_seq, dtype=torch.long).unsqueeze(0).expand(n, -1).contiguous()
        )
        pos_emb = rotary_emb(h, pos_ids)
        block_mask = make_flex_block_causal_mask(attn_mask)
        for layer_idx, block in enumerate(pl.layer_modules(model)):
            out = block(
                h,
                attention_mask=block_mask,
                position_ids=pos_ids,
                position_embeddings=pos_emb,
            )
            h = _cast(Tensor, out[0] if isinstance(out, tuple) else out)
            out_per_layer[layer_idx] = h.detach().clone()
    return out_per_layer


@pytest.mark.integration
def test_sweep_pack_true_matches_padded_reference() -> None:
    real_lengths = [3, 5, 7]
    items = _padded_dataset(real_lengths, MAX_SEQ)
    model = _tiny_llama_flex()

    cb = _PackedRaw()
    result = Sweep(
        model=model,
        dataset=items,
        seq_len=MAX_SEQ,
        callbacks=[cb],
        pack=True,
        microbatch_size=3,
        progress=False,
        transport_dtype=torch.float32,
        seed=SEED,
        apply_final_norm=False,  # ref builds raw post-block residuals; match.
    ).run()

    # Reference: same-kernel padded forward.
    ref = _padded_reference_per_layer(model, items)

    for layer_idx in (0, 1):
        rt = result.activations(layer=layer_idx, hook="residual_post")
        assert isinstance(rt, RaggedTensor), (
            f"layer {layer_idx}: expected RaggedTensor, got {type(rt).__name__}"
        )
        assert torch.equal(
            rt.lengths, torch.tensor(real_lengths, dtype=torch.int64)
        )
        for i, L in enumerate(real_lengths):
            ref_real = ref[layer_idx][i, :L, :].to(torch.float32)
            got = rt[i]
            max_diff = (got - ref_real).abs().max().item()
            assert torch.allclose(got, ref_real, atol=1e-5, rtol=1e-5), (
                f"layer {layer_idx} sample {i} (L={L}): max diff {max_diff}"
            )
