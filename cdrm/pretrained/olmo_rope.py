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


@dataclass(frozen=True)
class DaoRopeTables:
    """Compact fixed native tables for Dao's two-dimensional lookup interface.

    Row ``b * length + t`` contains the exact native phase at batch/token
    ``(b, t)``. The offsets select that row even for arbitrary, nonconsecutive
    native position IDs. These are invocation/layout-owned tensors, not model
    buffers, and are prepared before capture and shared across ordinary layers.
    """
    cos: Tensor
    sin: Tensor
    offsets: Tensor
    batch_size: int
    sequence_length: int

    def __post_init__(self):
        if (type(self.batch_size) is not int or self.batch_size < 1
                or type(self.sequence_length) is not int or self.sequence_length < 1
                or self.cos.ndim != 2 or self.cos.shape[0] != self.batch_size * self.sequence_length
                or self.cos.shape[1] < 1 or self.sin.shape != self.cos.shape
                or self.cos.dtype != torch.float32 or self.sin.dtype != torch.float32
                or self.offsets.shape != (self.batch_size,) or self.offsets.dtype != torch.long
                or any(value.device != self.cos.device or value.requires_grad or not value.is_contiguous()
                       for value in (self.cos, self.sin, self.offsets))):
            raise ValueError("Dao RoPE requires fixed contiguous FP32 compact tables and int64 batch offsets")


def build_dao_rope_tables(tables: RopeTables) -> DaoRopeTables:
    """Compact canonical native tables outside capture; never rebuild phases.

    The Dao kernel shares each cosine/sine between the two rotary halves, so
    reject noncanonical hand-built tables instead of silently dropping values.
    These value checks belong at preparation, not inside a captured forward.
    """
    if not isinstance(tables, RopeTables):
        raise TypeError("Expected native RopeTables")
    batch, _, length, width = tables.cos.shape
    if not batch or not length or width < 2 or width % 2:
        raise ValueError("Dao RoPE requires nonempty even-width native tables")
    half = width // 2
    if not all(torch.equal(value[..., :half], value[..., half:]) for value in (tables.cos, tables.sin)):
        raise ValueError("Dao RoPE requires identical native split-half cosine/sine tables")
    cos = tables.cos[:, 0, :, :half].reshape(batch * length, half).contiguous()
    sin = tables.sin[:, 0, :, :half].reshape(batch * length, half).contiguous()
    offsets = torch.arange(batch, dtype=torch.long, device=tables.cos.device) * length
    return DaoRopeTables(cos, sin, offsets, batch, length)


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
