"""Llama packed forward: per-real-token equivalence vs padded forward.

Same kernel (FlexAttention) on both sides — only the input layout differs.
Padded forward is `[N, max_seq, H]` with a per-sample causal mask; packed
forward is `[1, sum(L_i), H]` with a document+causal block mask derived from
`cu_seqlens`. At real-token positions the two paths must produce equal outputs
(within fp32 reduction-order tolerance — same arithmetic, possibly different
tile orderings).

Phase-1 contract test for `LlamaPlumbing.layer_forward_packed`. CPU-only;
FlexAttention falls back to a Python reference impl on CPU which is slow but
correct, fine for these tiny shapes.
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

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

    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=N_HEAD,
        num_key_value_heads=N_KV_HEAD,
        max_position_embeddings=MAX_SEQ,
        pad_token_id=PAD_TOKEN,
        attn_implementation="flex_attention",
    )
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _build_padded_inputs(real_lengths: list[int]) -> tuple[Tensor, Tensor, Tensor]:
    """Right-padded ids + 2D attention_mask + flat real_ids for the packed path."""
    n = len(real_lengths)
    max_seq = max(real_lengths)
    input_ids = torch.full((n, max_seq), PAD_TOKEN, dtype=torch.long)
    attention_mask = torch.zeros((n, max_seq), dtype=torch.long)
    torch.manual_seed(SEED + 1)
    real_chunks = []
    for i, L in enumerate(real_lengths):
        ids = torch.randint(1, VOCAB, (L,))
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        real_chunks.append(ids)
    real_concat = torch.cat(real_chunks, dim=0)  # [sum(L), ]
    return input_ids, attention_mask, real_concat


@pytest.mark.integration
def test_llama_packed_equals_padded_at_real_positions() -> None:
    """Packed forward output equals padded forward at real-token positions.

    Both sides call the same `block(...)` with flex_attention. Only the input
    layout and BlockMask differ. Padded uses HF's per-batch causal+padding
    BlockMask; packed uses a document+causal BlockMask built from cu_seqlens.
    The arithmetic at real-token positions must match within fp32 reduction
    tolerance.
    """
    from typing import Any
    from typing import cast as _cast

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

    from fpwap.models.llama import LlamaPlumbing

    plumbing = LlamaPlumbing()
    model = _tiny_llama_flex()
    # FlexAttention on CPU rejects autograd-tracked inputs; freeze params.
    for p in model.parameters():
        p.requires_grad_(False)
    block = plumbing.layer_modules(model)[0]
    rotary_emb = _cast(Any, model).model.rotary_emb

    real_lengths = [3, 5, 7]
    input_ids_padded, attn_mask, real_ids = _build_padded_inputs(real_lengths)
    n, max_seq = input_ids_padded.shape

    with torch.no_grad():
        # ---- padded path: same block, BlockMask built from 2D attention_mask ----
        h_padded = plumbing.embed(model, input_ids_padded)  # [N, max_seq, H]
        padded_pos_ids = (
            torch.arange(max_seq, dtype=torch.long).unsqueeze(0).expand(n, -1).contiguous()
        )
        padded_pos_emb = rotary_emb(h_padded, padded_pos_ids)
        # HF helper expects [batch, total_seq] with non-zero ints for valid tokens.
        # For unpacked, the standard 0/1 attention_mask is interpreted as doc id 1 vs pad.
        padded_block_mask = make_flex_block_causal_mask(attn_mask)
        out_padded = block(
            h_padded,
            attention_mask=padded_block_mask,
            position_ids=padded_pos_ids,
            position_embeddings=padded_pos_emb,
        )
        out_padded_t = _cast(
            torch.Tensor, out_padded[0] if isinstance(out_padded, tuple) else out_padded
        )

        # ---- packed path through LlamaPlumbing.layer_forward_packed ----
        h_packed = plumbing.embed(model, real_ids.unsqueeze(0))  # [1, sum(L), H]
        cu_seqlens = torch.tensor(
            [0, *list(_cumulative(real_lengths))], dtype=torch.int32
        )
        position_ids = torch.cat(
            [torch.arange(L, dtype=torch.long) for L in real_lengths], dim=0
        ).unsqueeze(0)
        out_packed, _ = plumbing.layer_forward_packed(
            model,
            block,
            h_packed,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
        )

    assert out_packed.shape == (1, sum(real_lengths), HIDDEN)

    for i, L in enumerate(real_lengths):
        start = sum(real_lengths[:i])
        padded_real = out_padded_t[i, :L, :]
        packed_real = out_packed[0, start : start + L, :]
        max_diff = (packed_real - padded_real).abs().max().item()
        assert torch.allclose(packed_real, padded_real, atol=1e-5, rtol=1e-5), (
            f"sample {i} (L={L}): max diff {max_diff}"
        )


def _cumulative(xs: list[int]) -> list[int]:
    s = 0
    out = []
    for x in xs:
        s += x
        out.append(s)
    return out
