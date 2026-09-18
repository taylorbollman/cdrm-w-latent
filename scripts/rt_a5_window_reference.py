"""Independent two-record FP32 oracle for the bounded A5 RT window pilot.

The operational implementation keeps the tiled RT and applies a per-block
attention mask.  This reference never constructs the masked historical cache:
it physically carries only the previous output's permanent K/V pair, then
concatenates the current input's temporary K/V pair.  It uses ordinary
differentiable PyTorch operations, without the vendor's pre/post-attention
helpers or custom backward.  It is a correctness oracle, not a training arm.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from olmo.model import OLMoRecurrentAutogradBlock


def direct_window2(block: OLMoRecurrentAutogradBlock, x: torch.Tensor,
                   attention_bias: torch.Tensor | None = None, *,
                   trace: dict[str, Any] | None = None) -> torch.Tensor:
    """Return a connected scan with self-temporary and predecessor-permanent K/V.

    ``attention_bias`` may be the original full ALiBi bias or the operational
    windowed bias.  Only the one or two actually present entries are read.
    Optional ``trace`` receives connected tensors and the actual key indices;
    retained permanent-write gradients expose the recurrent credit chain.
    """
    cfg = block.config
    if (x.ndim != 3 or x.shape[1] < 1 or x.shape[-1] != cfg.d_model
            or x.dtype != torch.float32):
        raise ValueError("The window oracle requires nonempty FP32 [B, T, D] inputs")
    if (cfg.norm_after or cfg.recurrent_write_rho != 1.0 or cfg.rope
            or cfg.clip_qkv is not None or cfg.residual_dropout != 0.0
            or cfg.attention_dropout != 0.0 or cfg.activation_type != "gelu"
            or cfg.layer_norm_type != "default"
            or cfg.effective_n_kv_heads != cfg.n_heads):
        raise ValueError("The window oracle supports only the frozen pre-norm A5 RT recipe")
    if torch.is_autocast_enabled(x.device.type):
        raise ValueError("The window oracle requires autocast off")
    batch, length, width = x.shape
    heads, head_width = cfg.n_heads, width // cfg.n_heads
    if attention_bias is not None:
        if (attention_bias.ndim != 4 or attention_bias.shape[-2:] != (length, length)
                or attention_bias.shape[0] not in (1, batch)
                or attention_bias.shape[1] not in (1, heads)
                or attention_bias.requires_grad or not attention_bias.is_floating_point()):
            raise ValueError("Bias must be a fixed broadcastable [B, H, T, T] tensor")

    def norm(value, module):
        return F.layer_norm(value, module.normalized_shape, module.weight,
                            module.bias, module.eps)

    def linear(value, module):
        return F.linear(value, module.weight, module.bias)

    def split_heads(value):
        return value.reshape(batch, -1, heads, head_width).transpose(1, 2)

    normalized = norm(x, block.attn_norm)
    query = linear(normalized, block.q_proj)
    initial_key, initial_value = linear(normalized, block.kv_proj).split(width, dim=-1)
    if block.q_norm is not None and block.k_norm is not None:
        query = norm(query, block.q_norm)
        initial_key = norm(initial_key, block.k_norm)
    query, initial_key, initial_value = map(split_heads, (query, initial_key, initial_value))
    previous_key = previous_value = None
    outputs = []
    if trace is not None:
        trace.clear()
        trace.update(key_indices=[], probabilities=[], permanent_keys=[], permanent_values=[], outputs=[])

    for position in range(length):
        key = initial_key[:, :, position:position + 1]
        value = initial_value[:, :, position:position + 1]
        if position:
            key = torch.cat((previous_key, key), dim=-2)
            value = torch.cat((previous_value, value), dim=-2)
        scores = (query[:, :, position:position + 1] @ key.transpose(-2, -1)) / math.sqrt(head_width)
        first_key = max(0, position - 1)
        if attention_bias is not None:
            scores = scores + attention_bias[:, :, position:position + 1, first_key:position + 1]
        probabilities = scores.softmax(dim=-1)
        attention = (probabilities @ value).transpose(1, 2).reshape(batch, 1, width)
        residual = x[:, position:position + 1] + linear(attention, block.attn_out)
        output = residual + linear(F.gelu(linear(norm(residual, block.ff_norm), block.ff_proj),
                                         approximate="none"), block.ff_out)
        outputs.append(output)

        # The terminal permanent write is retained for the credit check but is
        # never read.  Earlier writes remain connected to all later outputs.
        permanent_key, permanent_value = linear(norm(output, block.attn_norm), block.kv_proj).split(width, dim=-1)
        if block.k_norm is not None and block.q_norm is not None:
            permanent_key = norm(permanent_key, block.k_norm)
        previous_key, previous_value = split_heads(permanent_key), split_heads(permanent_value)
        if trace is not None:
            for record in (previous_key, previous_value):
                if record.requires_grad:
                    record.retain_grad()
            trace["key_indices"].append(list(range(first_key, position + 1)))
            trace["probabilities"].append(probabilities)
            trace["permanent_keys"].append(previous_key)
            trace["permanent_values"].append(previous_value)
            trace["outputs"].append(output)
    return torch.cat(outputs, dim=1)


class DirectWindow2RecurrentBlock(OLMoRecurrentAutogradBlock):
    """Swap only an already constructed second block to the reference scan.

    ``block.__class__ = DirectWindow2RecurrentBlock`` preserves all owning
    parameter keys and module objects.  Do this only to a separate diagnostic
    model; it deliberately bypasses the production block's window-mask code.
    """

    def _real_forward(self, x: torch.Tensor,
                      attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        return direct_window2(self, x, attention_bias)
