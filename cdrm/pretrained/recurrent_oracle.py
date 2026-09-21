"""Independent, deliberately slow oracle for native OpenELM recurrence.

Pass a block or model made by ``reference.build_corenet_reference``. Native
projection, normalization and MLP operations then execute the pinned CoreNet
source, independently of the production OpenELM adapter and RT scan. Attention
is explicit matmul/softmax; it does not call SDPA. Every query reconstructs all
earlier memory sources from their completed outputs instead of maintaining the
production scan's permanent KV cache. This costs extra work and is intended for
bounded semantic/gradient checks, not training or performance measurement.

Memory at s < t is A((1-alpha)*x_s + alpha*z_s), where A is the native input
RMSNorm and z_s is the *complete block* output. Position t always attends to
temporary K/V from A(x_t), never a permanent write from its not-yet-known z_t.
Keys are head-normalized before RoPE and expanded only for grouped attention.
The oracle intentionally has no cache or detached-history shortcut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RecurrentOracleOutput:
    logits: Tensor | None
    last_hidden_state: Tensor


def _validate_inputs(
    x: Tensor, alpha: float, position_ids: Tensor | None, attention_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if x.ndim != 3 or min(x.shape) == 0 or not x.is_floating_point():
        raise ValueError("x must be a nonempty floating-point [batch, length, width] tensor")
    if isinstance(alpha, (bool, Tensor)) or not isinstance(alpha, (float, int)):
        raise TypeError("alpha must be an immutable real scalar")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    shape = x.shape[:2]
    if attention_mask is None:
        valid = torch.ones(shape, device=x.device, dtype=torch.bool)
    else:
        if attention_mask.shape != shape or attention_mask.device != x.device:
            raise ValueError("attention_mask must be [batch, length] on the input device")
        if attention_mask.dtype != torch.bool and not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must be boolean or binary")
        valid = attention_mask.bool()
    if position_ids is None:
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    else:
        if position_ids.shape != shape or position_ids.dtype != torch.long or position_ids.device != x.device:
            raise ValueError("position_ids must be int64 [batch, length] on the input device")
        if bool((position_ids < 0).any()):
            raise ValueError("position_ids must be nonnegative")
        positions = position_ids
    return positions, valid


def _rotate_with_native_frequencies(x: Tensor, positions: Tensor, rotary: nn.Module) -> Tensor:
    """Explicit split-half rotation using the pinned native frequency buffer."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        angles = torch.einsum("bt,d->btd", positions.float(), rotary.inv_freq.float())
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        x_float = x.float()
        left, right = x_float.chunk(2, dim=-1)
        result = x_float * angles.cos() + torch.cat((-right, left), dim=-1) * angles.sin()
    return result.type_as(x)


def recurrent_block_oracle(
    block: nn.Module,
    x: Tensor,
    *,
    alpha: float,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate one native CoreNet block with explicit recurrent attention.

    Parameters stay in FP32. BF16 autocast may surround this call; projection
    and MLP operations then use native autocast behavior, while explicit
    attention intermediates use FP32 and its output returns to the value dtype,
    matching the arithmetic intent of PyTorch's math SDPA backend. This is not
    a bitwise promise for fused SDPA or a full FP64 derivative oracle.
    """
    positions, valid = _validate_inputs(x, alpha, position_ids, attention_mask)
    attn = block.attn
    outputs: list[Tensor] = []
    batch, length, _ = x.shape
    for t in range(length):
        current = x[:, t:t + 1]
        # Rebuild history from completed outputs. Both interpolation branches
        # remain attached; unlike a production cache there are no stored K/V.
        if t:
            completed = torch.cat(outputs, dim=1)
            memory = (1.0 - alpha) * x[:, :t] + alpha * completed
            source = torch.cat((memory, current), dim=1)
        else:
            source = current
        projected = attn.qkv_proj(block.attn_norm(source)).reshape(
            batch, t + 1, attn.num_q_heads + attn.num_k_heads + attn.num_v_heads,
            attn.head_dim,
        ).transpose(1, 2)
        queries, keys, values = projected.split(
            (attn.num_q_heads, attn.num_k_heads, attn.num_v_heads), dim=1,
        )
        queries = attn.q_norm(queries[:, :, -1:])
        keys = attn.k_norm(keys)
        queries = _rotate_with_native_frequencies(queries, positions[:, t:t + 1], attn.pos_embedding)
        keys = _rotate_with_native_frequencies(keys, positions[:, :t + 1], attn.pos_embedding)
        keys = keys.repeat_interleave(attn.num_groups, dim=1)
        values = values.repeat_interleave(attn.num_groups, dim=1)
        visible = valid[:, None, None, :t + 1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2)) / math.sqrt(attn.head_dim)
            scores = scores.masked_fill(~visible, -torch.inf)
            # SDPA returns zero attention for a query whose keys are all masked.
            any_visible = visible.any(dim=-1, keepdim=True)
            scores = torch.where(any_visible, scores, torch.zeros_like(scores))
            weights = scores.softmax(dim=-1) * any_visible
            attended = torch.matmul(weights, values.float())
        attended = attended.to(values.dtype).transpose(1, 2).reshape(batch, 1, attn.num_q_heads * attn.head_dim)
        residual = current + attn.out_proj(attended)
        outputs.append(residual + block.ffn(block.ffn_norm(residual)))
    return torch.cat(outputs, dim=1)


def recurrent_model_oracle(
    model: nn.Module,
    input_ids: Tensor | None = None,
    *,
    inputs_embeds: Tensor | None = None,
    selected_layers: tuple[int, ...] = (0,),
    alpha: float = 1.0,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    return_logits: bool = True,
) -> RecurrentOracleOutput:
    """Run a pinned native model with selected blocks replaced by the oracle.

    Unmodified blocks execute their original forward. When explicit masks or
    positions extend the native model's interface, those blocks instead use the
    independent attention above with alpha=0. Final normalization and tied
    readout preserve the native model's operations. No FBT or NextLat is added.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Provide exactly one of input_ids and inputs_embeds")
    if any(type(i) is not int or not 0 <= i < len(model.layers) for i in selected_layers):
        raise ValueError("selected_layers must contain valid layer indices")
    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError("selected_layers must be unique")
    x = model.token_embeddings(input_ids) if input_ids is not None else inputs_embeds
    assert x is not None
    _validate_inputs(x, alpha, position_ids, attention_mask)
    for index, block in enumerate(model.layers):
        if index in selected_layers or position_ids is not None or attention_mask is not None:
            x = recurrent_block_oracle(
                block, x, alpha=alpha if index in selected_layers else 0.0,
                position_ids=position_ids, attention_mask=attention_mask,
            )
        else:
            x = block(x, is_causal=True)[0]
    hidden = model.norm(x)
    logits = None
    if return_logits:
        logits = F.linear(hidden, model.token_embeddings.weight) if model.classifier is None else model.classifier(hidden)
    return RecurrentOracleOutput(logits, hidden)
