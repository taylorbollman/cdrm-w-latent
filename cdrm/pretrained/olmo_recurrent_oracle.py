"""Independent explicit-history RT oracle over pristine original OLMo blocks.

This diagnostic implementation rebuilds every historical memory source for each
query and computes attention with explicit FP32 matmul/softmax. It neither uses
the adapter's scan/cache nor calls SDPA for recurrent attention. Native source
LayerNorm, projections and value-then-gate SwiGLU remain the authoritative block
primitives. There is no Q/K normalization in original OLMo.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class OLMoRecurrentOracleOutput:
    logits: Tensor | None
    last_hidden_state: Tensor


def _coordinates(x: Tensor, alpha: float, positions: Tensor | None, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    if x.ndim != 3 or min(x.shape) == 0 or not x.is_floating_point():
        raise ValueError("x must be a nonempty floating point [batch, length, width] tensor")
    if isinstance(alpha, (bool, Tensor)) or not isinstance(alpha, (float, int)):
        raise TypeError("alpha must be an immutable real scalar")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    shape = x.shape[:2]
    if mask is None:
        valid = torch.ones(shape, device=x.device, dtype=torch.bool)
    else:
        if mask.shape != shape or mask.device != x.device:
            raise ValueError("attention_mask must be [batch, length] on the input device")
        if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("attention_mask must be boolean or binary")
        valid = mask.bool()
    if positions is None:
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    elif positions.shape != shape or positions.device != x.device or positions.dtype != torch.long or bool((positions < 0).any()):
        raise ValueError("position_ids must be nonnegative int64 [batch, length] on the input device")
    return positions, valid


def _rotate(x: Tensor, positions: Tensor, block: nn.Module) -> Tensor:
    """Use native cached FP32 phases at explicit coordinates, then rotate."""
    # The pristine native helper constructs phases; indexing supplies the
    # arbitrary positions that its ordinary contiguous-position API lacks.
    x_float = x.float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        sine, cosine = block.rotary_emb.get_rotary_embedding(int(positions.max()) + 1, x.device)
        sine, cosine = sine[0, 0][positions].unsqueeze(1), cosine[0, 0][positions].unsqueeze(1)
        return block.rotary_emb.apply_rotary_pos_emb(sine, cosine, x_float).to(x.dtype)


def olmo_recurrent_block_oracle(
    block: nn.Module, x: Tensor, *, alpha: float,
    position_ids: Tensor | None = None, attention_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate full causal RT history, with a temporary current-position entry.

    ``block`` must come from build_olmo_reference. Parameter/runtime precision
    follows the native primitives; FP32 attention accumulation is deliberate.
    BF16 comparisons are observational rather than bitwise source-parity claims.
    """
    positions, valid = _coordinates(x, alpha, position_ids, attention_mask)
    if block.config.multi_query_attention or block.q_norm is not None or block.k_norm is not None:
        raise ValueError("The selected original OLMo oracle requires full MHA without Q/K normalization")
    if block.config.residual_dropout or block.config.attention_dropout:
        raise ValueError("The selected original OLMo oracle requires zero dropout")
    outputs = []
    batch, length, width = x.shape
    heads = block.config.n_heads
    head_dim = width // heads
    for t in range(length):
        current = x[:, t:t + 1]
        if t:
            history = (1.0 - alpha) * x[:, :t] + alpha * torch.cat(outputs, dim=1)
            source = torch.cat((history, current), dim=1)
        else:
            source = current
        query, key, value = block.att_proj(block.attn_norm(source)).split(block.fused_dims, dim=-1)
        query = query[:, -1:].reshape(batch, 1, heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, t + 1, heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, t + 1, heads, head_dim).transpose(1, 2)
        query = _rotate(query, positions[:, t:t + 1], block)
        key = _rotate(key, positions[:, :t + 1], block)
        visible = valid[:, None, None, :t + 1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            score = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(head_dim)
            score = score.masked_fill(~visible, -torch.inf)
            any_visible = visible.any(-1, keepdim=True)
            score = torch.where(any_visible, score, torch.zeros_like(score))
            weights = score.softmax(-1) * any_visible
            attended = torch.matmul(weights, value.float())
        attended = attended.to(value.dtype).transpose(1, 2).reshape(batch, 1, width)
        residual = current + block.dropout(block.attn_out(attended))
        update = block.ff_out(block.act(block.ff_proj(block.ff_norm(residual))))
        outputs.append(residual + block.dropout(update))
    return torch.cat(outputs, dim=1)


def olmo_recurrent_model_oracle(
    model: nn.Module, input_ids: Tensor | None = None, *,
    inputs_embeds: Tensor | None = None, selected_layers: tuple[int, ...] = (0,),
    alpha: float = 1.0, position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None, return_logits: bool = True,
) -> OLMoRecurrentOracleOutput:
    """Run native ordinary blocks around independently reconstructed RT blocks."""
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Provide exactly one of input_ids and inputs_embeds")
    blocks = model.transformer.blocks
    if any(type(i) is not int or not 0 <= i < len(blocks) for i in selected_layers):
        raise ValueError("selected_layers must contain valid layer indices")
    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError("selected_layers must be unique")
    x = model.transformer.wte(input_ids) if input_ids is not None else inputs_embeds
    assert x is not None
    _coordinates(x, alpha, position_ids, attention_mask)
    x = model.transformer.emb_drop(x)
    for index, block in enumerate(blocks):
        if index in selected_layers or position_ids is not None or attention_mask is not None:
            x = olmo_recurrent_block_oracle(
                block, x, alpha=alpha if index in selected_layers else 0.0,
                position_ids=position_ids, attention_mask=attention_mask,
            )
        else:
            x = block(x)[0]
    hidden = model.transformer.ln_f(x)
    logits = F.linear(hidden, model.transformer.wte.weight) if return_logits else None
    return OLMoRecurrentOracleOutput(logits, hidden)
