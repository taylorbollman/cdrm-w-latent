"""Invocation-owned native RoPE tables, with no model buffers or lazy cache.

Positions are the actual caller coordinates, not an inferred contiguous range.
Prepared execution owns these tensors across layers, finite FBT passes and
backward recomputation. The ordinary helper in ``olmo`` remains an independent
reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class RopeTables:
    """FP32 cosine/sine tensors shaped [batch, 1, positions, head_dim].

    Frozen fields prevent accidental reassignment. Tensor contents are owned by
    the caller; a prepared layout separately guards their storage and mutation.
    """
    cos: Tensor
    sin: Tensor

    def __post_init__(self):
        if (self.cos.ndim != 4 or self.cos.shape[1] != 1
                or self.sin.shape != self.cos.shape
                or self.cos.dtype != torch.float32 or self.sin.dtype != torch.float32
                or self.cos.device != self.sin.device
                or self.cos.requires_grad or self.sin.requires_grad):
            raise ValueError("RoPE tables must be matching, fixed FP32 [batch, 1, positions, head_dim] tensors")

    def slice(self, start: int | None, stop: int | None) -> RopeTables:
        """View a token interval without rebuilding frequencies or trig tables."""
        return RopeTables(self.cos[:, :, start:stop], self.sin[:, :, start:stop])


def build_rope_tables(positions: Tensor, head_dim: int, base: float) -> RopeTables:
    """Use the same FP32 operation order as native ``_apply_rope``.

    The public model input or static layout validates positional values. This
    helper checks metadata only and performs no device-to-host scalar reads.
    """
    if positions.ndim != 2 or positions.dtype != torch.long:
        raise ValueError("RoPE positions must be int64 [batch, positions]")
    if type(head_dim) is not int or head_dim < 2 or head_dim % 2:
        raise ValueError("RoPE head_dim must be a positive even integer")
    if isinstance(base, bool) or not isinstance(base, (float, int)) or not math.isfinite(base) or base <= 0:
        raise ValueError("RoPE base must be finite and positive")
    with torch.autocast(device_type=positions.device.type, enabled=False):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
        phases = positions.float().unsqueeze(-1) * inv_freq
        phases = torch.cat((phases, phases), dim=-1).unsqueeze(1)
        return RopeTables(phases.cos(), phases.sin())


def apply_rope_tables(x: Tensor, tables: RopeTables) -> Tensor:
    """Apply native split-half rotation and restore Q/K dtype exactly."""
    if (x.ndim != 4 or x.shape[0] != tables.cos.shape[0]
            or x.shape[-2:] != tables.cos.shape[-2:] or x.device != tables.cos.device):
        raise ValueError("RoPE input and table batch/position/head dimensions or device differ")
    # Retain one shared cast for both branches, including backward accumulation.
    x_float = x.float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        first, second = x_float.chunk(2, dim=-1)
        rotated = x_float * tables.cos + torch.cat((-second, first), dim=-1) * tables.sin
    return rotated.to(x.dtype)
