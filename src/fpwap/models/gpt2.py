from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn


class GPT2Plumbing:
    """Hook plumbing for HuggingFace GPT-2 family (`GPT2LMHeadModel`, `GPT2Model`)."""

    def matches(self, model: nn.Module) -> bool:
        transformer = getattr(model, "transformer", None)
        return transformer is not None and hasattr(transformer, "h") and hasattr(transformer, "wte")

    def layer_modules(self, model: nn.Module) -> Sequence[nn.Module]:
        t = cast(Any, model).transformer
        return cast(Sequence[nn.Module], t.h)

    def layer_prefix(self, layer_idx: int) -> str:
        return f"transformer.h.{layer_idx}"

    def embedding_param_names(self, model: nn.Module) -> Sequence[str]:
        # wte: token embedding. wpe: learned positional embedding. No bias.
        # drop (dropout) has no params; ln_f (final layernorm) isn't used
        # by the per-layer loop so it doesn't need to be kept resident.
        return ["transformer.wte.weight", "transformer.wpe.weight"]

    def embed(self, model: nn.Module, input_ids: Tensor) -> Tensor:
        t = cast(Any, model).transformer
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        hidden: Tensor = t.wte(input_ids) + t.wpe(position_ids)
        return cast(Tensor, t.drop(hidden))

    def layer_forward_with_hooks(
        self,
        model: nn.Module,
        block: nn.Module,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        wanted_hooks: frozenset[str] = frozenset(),
        dispatch_fn: Any = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # GPT-2's SDPA-path implementation expects a combined causal+padding 4D
        # mask (same thing GPT2Model.forward builds internally). Delegate to
        # HF's own prep helper so we match whichever attention impl is active.
        if attention_mask is not None and attention_mask.dim() == 2:
            from transformers.modeling_attn_mask_utils import (
                _prepare_4d_causal_attention_mask_for_sdpa,
            )

            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape=(hidden_states.shape[0], hidden_states.shape[1]),
                inputs_embeds=hidden_states,
                past_key_values_length=0,
            )

        needs_decompose = bool({"attn_out", "mlp_out"} & wanted_hooks)
        if not needs_decompose:
            out = block(hidden_states, attention_mask=attention_mask)
            return cast(Tensor, out[0] if isinstance(out, tuple) else out), {}

        # Manual decomposition mirrors GPT2Block.forward exactly. Sub-layer
        # outputs captured BEFORE the residual add (matches README's
        # "attention sub-layer output" / "MLP sub-layer output" definition).
        # When dispatch_fn is given, the plumbing calls it at each sub-layer
        # boundary so a write-phase callback can modify the tensor; its
        # return value replaces the one flowing into the residual add.
        b = cast(Any, block)
        extras: dict[str, Tensor] = {}

        residual = hidden_states
        h = b.ln_1(hidden_states)
        attn_output, _ = b.attn(h, attention_mask=attention_mask)
        if "attn_out" in wanted_hooks:
            if dispatch_fn is not None:
                attn_output = dispatch_fn("attn_out", attn_output)
            else:
                extras["attn_out"] = attn_output
        h = attn_output + residual

        residual = h
        h = b.ln_2(h)
        feed_forward_hidden_states = b.mlp(h)
        if "mlp_out" in wanted_hooks:
            if dispatch_fn is not None:
                feed_forward_hidden_states = dispatch_fn(
                    "mlp_out", feed_forward_hidden_states
                )
            else:
                extras["mlp_out"] = feed_forward_hidden_states
        h = residual + feed_forward_hidden_states

        return h, extras
