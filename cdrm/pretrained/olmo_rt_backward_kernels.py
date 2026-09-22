"""Opt-in mixed-precision historical dK/dV tile for native OLMo RT.

The caller owns reverse recurrent scheduling, reconstruction and accumulation
into FP32 adjoint buffers. This helper only replaces one historical rectangle's
three GEMMs and intervening arithmetic. Explicit eager BF16 cast boundaries are
preserved; accumulation order is not promised bitwise identical to cuBLAS.
"""

from functools import lru_cache
import math

import torch
from torch import Tensor


def _validate_backward_tile_metadata(probabilities, grad_attention, values, query, dot):
    """Check metadata only; CPU fixtures can exercise this without CUDA/Triton."""
    tensors = (probabilities, grad_attention, values, query, dot)
    if not all(isinstance(value, Tensor) for value in tensors):
        raise TypeError("historical backward tile inputs must all be tensors")
    if any(value.layout != torch.strided for value in tensors):
        raise ValueError("historical backward tile inputs must have strided layout")
    if any(value.device != probabilities.device for value in tensors):
        raise ValueError("historical backward tile inputs must share one device")
    if probabilities.ndim != 4 or grad_attention.ndim != 4 or values.ndim != 4 or query.ndim != 4:
        raise ValueError("historical backward tile matrix inputs must be rank four")
    batch, heads, rows, columns = probabilities.shape
    dim = query.shape[-1]
    if batch < 1 or heads < 1:
        raise ValueError("historical backward tile batch and head counts must be positive")
    if not (1 <= rows <= 256 and 1 <= columns <= 256):
        raise ValueError("historical backward tile row/column lengths must be in [1, 256]")
    if dim not in (16, 32, 64, 128):
        raise ValueError("historical backward tile head dimension must be 16, 32, 64 or 128")
    if query.shape != (batch, heads, rows, dim) or grad_attention.shape != query.shape:
        raise ValueError("historical backward tile query/grad_attention shapes must match probability rows")
    if values.shape != (batch, heads, columns, dim):
        raise ValueError("historical backward tile values must match probability columns and query dim")
    if dot.shape != (batch, heads, rows):
        raise ValueError("historical backward tile dot must be [batch, heads, rows]")
    if any(value.dtype != torch.float32 for value in (probabilities, grad_attention, dot)):
        raise ValueError("historical backward tile probabilities, grad_attention and dot must be FP32")
    if any(value.dtype != torch.bfloat16 for value in (values, query)):
        raise ValueError("historical backward tile values and query must be BF16")
    if torch.is_grad_enabled() and any(value.requires_grad for value in tensors):
        raise ValueError("historical backward tile has no autograd; invoke inside the first-order RT VJP")
    return batch, heads, rows, columns, dim


@lru_cache(maxsize=1)
def _get_backward_tile_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def historical_backward_kernel(P, G, V, Q, DOT, DK, DV,
            spb, sph, spr, spc, sgb, sgh, sgr, sgd,
            svb, svh, svc, svd, sqb, sqh, sqr, sqd, sdb, sdh, sdr,
            HEADS: tl.constexpr, ROWS: tl.constexpr, COLUMNS: tl.constexpr,
            DIM: tl.constexpr, SCALE: tl.constexpr,
            BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr):
        bh = tl.program_id(1)
        batch, head = bh // HEADS, bh % HEADS
        columns = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        rows = tl.arange(0, BLOCK_R)
        features = tl.arange(0, DIM)
        p_transposed = tl.load(P + batch * spb + head * sph
            + rows[None, :] * spr + columns[:, None] * spc,
            (rows[None, :] < ROWS) & (columns[:, None] < COLUMNS), other=0.0)
        g = tl.load(G + batch * sgb + head * sgh
            + rows[:, None] * sgr + features[None, :] * sgd,
            rows[:, None] < ROWS, other=0.0).to(tl.bfloat16)
        v = tl.load(V + batch * svb + head * svh
            + columns[:, None] * svc + features[None, :] * svd,
            columns[:, None] < COLUMNS, other=0.0)
        # Compute (G V^T)^T, with the same complete inner-dimension BF16
        # rounding boundary as eager. Only storage orientation is transposed.
        gv_transposed = tl.dot(v, tl.trans(g)).to(tl.bfloat16).to(tl.float32)
        dot = tl.load(DOT + batch * sdb + head * sdh + rows * sdr,
                      rows < ROWS, other=0.0)
        # Probabilities retain FP32 here, independently of their BF16 dV use.
        error_transposed = p_transposed * (gv_transposed - dot[None, :])
        q = tl.load(Q + batch * sqb + head * sqh
            + rows[:, None] * sqr + features[None, :] * sqd,
            rows[:, None] < ROWS, other=0.0)
        # Each dot covers ALL query rows. Cast only after the full reduction,
        # never after partial chunks or after combining with an existing adjoint.
        dkey = tl.dot(error_transposed.to(tl.bfloat16), q).to(tl.bfloat16).to(tl.float32) * SCALE
        dvalue = tl.dot(p_transposed.to(tl.bfloat16), g).to(tl.bfloat16).to(tl.float32)
        offsets = (bh * COLUMNS + columns[:, None]) * DIM + features[None, :]
        tl.store(DK + offsets, dkey, columns[:, None] < COLUMNS)
        tl.store(DV + offsets, dvalue, columns[:, None] < COLUMNS)

    return historical_backward_kernel


def backward_tile(probabilities: Tensor, grad_attention: Tensor, values: Tensor,
                  query: Tensor, dot: Tensor):
    """Return fresh FP32 ``(dkey, dvalue)`` for a historical [rows, columns] tile.

    All inputs must share one CUDA device. Probabilities are FP32 [B,H,R,C],
    attention cotangents FP32 [B,H,R,D], stored values BF16 [B,H,C,D], already
    rotated queries BF16 [B,H,R,D], and dot FP32 [B,H,R]. Both tile extents are
    1..256; D is 16/32/64/128. Arbitrary strided and broadcast read-only views
    are supported. Zero/masked probability entries contribute zero for finite
    operands; the caller supplies the original probability masks.

    No synchronization, value inspection, input mutation, fallback or autograd
    is hidden here. The RT VJP adds these returned increments to its existing
    FP32 adjoints. Warm the required shapes before CUDA graph capture.
    """
    batch, heads, rows, columns, dim = _validate_backward_tile_metadata(
        probabilities, grad_attention, values, query, dot)
    if probabilities.device.type != "cuda":
        raise ValueError("historical backward tile requires CUDA; no CPU fallback is provided")
    kernel = _get_backward_tile_kernel()
    dkey = torch.empty((batch, heads, columns, dim), device=query.device, dtype=torch.float32)
    dvalue = torch.empty_like(dkey)
    block_c = 16
    block_r = max(16, 1 << (rows - 1).bit_length())
    kernel[((columns + block_c - 1) // block_c, batch * heads)](
        probabilities, grad_attention, values, query, dot, dkey, dvalue,
        *probabilities.stride(), *grad_attention.stride(), *values.stride(),
        *query.stride(), *dot.stride(), HEADS=heads, ROWS=rows, COLUMNS=columns,
        DIM=dim, SCALE=1.0 / math.sqrt(dim), BLOCK_C=block_c, BLOCK_R=block_r,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return dkey, dvalue
