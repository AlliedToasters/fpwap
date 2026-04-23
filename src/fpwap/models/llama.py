from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn


class LlamaPlumbing:
    """Hook plumbing for HF Llama and the RoPE-decoder family it covers
    structurally: Llama, Mistral, Qwen2, and any future model exposing the
    same `model.{embed_tokens, layers, rotary_emb}` layout with decoder blocks
    that accept `(hidden_states, attention_mask, position_ids, position_embeddings)`.

    Named for Llama for history; the matcher is intentionally structural so
    adjacent families ride for free rather than duplicating plumbing.

    Covers the RoPE-specific wiring that would otherwise surface as the ~1e-3
    drift class lmprobe #284 catalogued: each block needs `position_ids` and
    `position_embeddings` (a `(cos, sin)` tuple from `model.model.rotary_emb`)
    on top of the standard hidden_states + attention_mask signature.
    """

    def matches(self, model: nn.Module) -> bool:
        inner = getattr(model, "model", None)
        return (
            inner is not None
            and hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(inner, "rotary_emb")
        )

    def layer_modules(self, model: nn.Module) -> Sequence[nn.Module]:
        inner = cast(Any, model).model
        return cast(Sequence[nn.Module], inner.layers)

    def layer_prefix(self, layer_idx: int) -> str:
        return f"model.layers.{layer_idx}"

    def embedding_param_names(self, model: nn.Module) -> Sequence[str]:
        # Embed table. rotary_emb.inv_freq is a buffer, not a parameter,
        # and Llama has no learned positional embedding.
        return ["model.embed_tokens.weight"]

    def embed(self, model: nn.Module, input_ids: Tensor) -> Tensor:
        inner = cast(Any, model).model
        return cast(Tensor, inner.embed_tokens(input_ids))

    def final_norm_module(self, model: nn.Module) -> nn.Module | None:
        return cast(nn.Module, cast(Any, model).model.norm)

    def final_norm_param_names(self, model: nn.Module) -> Sequence[str]:
        return ["model.norm.weight"]

    def layer_forward_with_hooks(
        self,
        model: nn.Module,
        block: nn.Module,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        wanted_hooks: frozenset[str] = frozenset(),
        dispatch_fn: Any = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        device = hidden_states.device

        # Sequential position ids — HF's default (LlamaModel.forward uses
        # cache_position.unsqueeze(0)). Padding is handled via attention_mask,
        # not by remapping positions.
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        rotary_emb = cast(Any, model).model.rotary_emb
        position_embeddings = rotary_emb(hidden_states, position_ids)

        # 2D padding mask → combined causal+padding 4D for SDPA.
        if attention_mask is not None and attention_mask.dim() == 2:
            from transformers.modeling_attn_mask_utils import (
                _prepare_4d_causal_attention_mask_for_sdpa,
            )

            ext_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape=(bsz, seq_len),
                inputs_embeds=hidden_states,
                past_key_values_length=0,
            )
        else:
            ext_mask = None

        needs_decompose = bool({"attn_out", "mlp_out"} & wanted_hooks)
        if not needs_decompose:
            out = block(
                hidden_states,
                attention_mask=ext_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            return cast(Tensor, out[0] if isinstance(out, tuple) else out), {}

        # Manual decomposition mirrors LlamaDecoderLayer.forward exactly.
        # Sub-layer outputs captured BEFORE the residual add. When
        # dispatch_fn is provided, calling it lets a write-phase callback
        # replace the tensor before it rejoins the residual stream.
        b = cast(Any, block)
        extras: dict[str, Tensor] = {}

        residual = hidden_states
        h = b.input_layernorm(hidden_states)
        attn_output, _ = b.self_attn(
            hidden_states=h,
            attention_mask=ext_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        if "attn_out" in wanted_hooks:
            if dispatch_fn is not None:
                attn_output = dispatch_fn("attn_out", attn_output)
            else:
                extras["attn_out"] = attn_output
        h = residual + attn_output

        residual = h
        h = b.post_attention_layernorm(h)
        mlp_output = b.mlp(h)
        if "mlp_out" in wanted_hooks:
            if dispatch_fn is not None:
                mlp_output = dispatch_fn("mlp_out", mlp_output)
            else:
                extras["mlp_out"] = mlp_output
        h = residual + mlp_output

        return h, extras
