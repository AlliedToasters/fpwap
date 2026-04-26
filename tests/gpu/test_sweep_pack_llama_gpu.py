"""GPU bit-equivalence for `Sweep(pack=True)` on Llama via FlexAttention.

Mirrors `tests/integration/test_sweep_pack_llama.py` but on CUDA + bf16, with
two design knobs that the CPU mirror doesn't exercise:

- `microbatch_size < n_samples` so the per-MB cache (built on first layer of
  a chunk, reused on subsequent layers) actually fires for multiple keys.
- `chunk_size > 1` so we hit the chunk-boundary cache clear.

If the cache plumbing or the `block_mask` / `position_embeddings` reuse path
is wrong on GPU, the packed activations will diverge from the same-kernel
padded reference at real-token positions.
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
HIDDEN = 64
INTERMEDIATE = 128
N_HEAD = 4
N_KV_HEAD = 2
N_LAYERS = 4
VOCAB = 128
MAX_SEQ = 32
PAD_TOKEN = 0


def _llama_cuda_bf16() -> torch.nn.Module:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEAD,
        num_key_value_heads=N_KV_HEAD,
        max_position_embeddings=MAX_SEQ,
        pad_token_id=PAD_TOKEN,
        attn_implementation="flex_attention",
    )
    torch.manual_seed(SEED)
    m = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16, device="cuda:0")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _padded_dataset(real_lengths: list[int]) -> list[dict[str, Tensor]]:
    torch.manual_seed(SEED + 1)
    items = []
    for L in real_lengths:
        ids = torch.full((1, MAX_SEQ), PAD_TOKEN, dtype=torch.long, device="cuda:0")
        ids[0, :L] = torch.randint(1, VOCAB, (L,), device="cuda:0")
        mask = torch.zeros((1, MAX_SEQ), dtype=torch.long, device="cuda:0")
        mask[0, :L] = 1
        items.append({"input_ids": ids, "attention_mask": mask})
    return items


class _PackedRaw(Callback):
    accepts_packed = True
    target_layers = "all"
    target_hooks: tuple[HookName, ...] = ("residual_post",)
    phase = "read"

    def on_batch(
        self, layer_idx: int, hook: HookName, acts: Tensor, sample_ids: Tensor
    ) -> Emit:
        rt = _cast(RaggedTensor, acts)
        return Emit(
            tensor=rt.flat.detach().to(torch.float32).cpu(),
            sample_lengths=rt.lengths.detach().cpu().to(torch.int64),
        )


def _padded_reference_per_layer(
    model: torch.nn.Module, items: list[dict[str, Tensor]]
) -> dict[int, Tensor]:
    """Same-kernel padded forward, block-by-block, capturing residual_post."""
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
            torch.arange(max_seq, dtype=torch.long, device="cuda:0")
            .unsqueeze(0)
            .expand(n, -1)
            .contiguous()
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


@pytest.mark.gpu
def test_sweep_pack_true_matches_padded_reference_cuda_bf16() -> None:
    """Packed Sweep on CUDA+bf16 matches a same-kernel padded forward.

    Microbatch and chunk sizes are picked so the per-MB cache is built once
    per (chunk, MB) and reused across the chunk's layers, then cleared at
    the chunk boundary. If the hoisted invariants drift on GPU, this test
    catches it.
    """
    real_lengths = [4, 8, 12, 16, 5, 11, 7, 9]
    items = _padded_dataset(real_lengths)
    model = _llama_cuda_bf16()

    cb = _PackedRaw()
    result = Sweep(
        model=model,
        dataset=items,
        seq_len=MAX_SEQ,
        callbacks=[cb],
        pack=True,
        microbatch_size=4,  # 2 microbatches per layer — cache hit path fires
        chunk_size=2,  # 2 chunks of 2 layers — cache cleared at boundary
        progress=False,
        transport_dtype=torch.bfloat16,
        seed=SEED,
        apply_final_norm=False,
    ).run()

    ref = _padded_reference_per_layer(model, items)

    for layer_idx in range(N_LAYERS):
        rt = result.activations(layer=layer_idx, hook="residual_post")
        assert isinstance(rt, RaggedTensor), (
            f"layer {layer_idx}: expected RaggedTensor, got {type(rt).__name__}"
        )
        assert torch.equal(
            rt.lengths, torch.tensor(real_lengths, dtype=torch.int64)
        )
        for i, L in enumerate(real_lengths):
            ref_real = ref[layer_idx][i, :L, :].to(torch.float32).cpu()
            got = rt[i].to(torch.float32)
            max_diff = (got - ref_real).abs().max().item()
            # bf16 + flex_attention tolerance — same envelope as the GPT-2
            # forward bit-perfect test.
            assert torch.allclose(got, ref_real, atol=5e-3, rtol=5e-3), (
                f"layer {layer_idx} sample {i} (L={L}): max diff {max_diff}"
            )
