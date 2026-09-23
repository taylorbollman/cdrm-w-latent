"""Bounded-workspace historical OLMo RT backward with probability recomputation.

This replaces the probability input of the F3c historical dK/dV helper with
queries, keys and retained complete-row softmax statistics. Each CTA owns at
most 16 keys and loops over query chunks; no probability/error rectangle is
allocated in global memory. The caller still owns recurrent scheduling, row
normalizers, temporary-self handling and accumulation into existing adjoints.
"""

from functools import lru_cache
import math

import torch
from torch import Tensor


def _validate_recomputed_tile_metadata(query, key, values, grad_attention, dot,
                                       row_maximum, row_denominator, key_valid):
    """Validate metadata without loading CUDA, Triton or tensor values."""
    tensors = (query, key, values, grad_attention, dot, row_maximum,
               row_denominator, key_valid)
    if not all(isinstance(value, Tensor) for value in tensors):
        raise TypeError("recomputed backward tile inputs must all be tensors")
    if any(value.layout != torch.strided for value in tensors):
        raise ValueError("recomputed backward tile inputs must have strided layout")
    if any(value.device != query.device for value in tensors):
        raise ValueError("recomputed backward tile inputs must share one device")
    if any(value.ndim != 4 for value in (query, key, values, grad_attention)):
        raise ValueError("recomputed backward tile matrix inputs must be rank four")
    batch, heads, rows, dim = query.shape
    columns = key.shape[2]
    if batch < 1 or heads < 1:
        raise ValueError("recomputed backward tile batch and head counts must be positive")
    if not (1 <= rows <= 2048 and 1 <= columns <= 2048):
        raise ValueError("recomputed backward tile row/column lengths must be in [1, 2048]")
    if dim not in (16, 32, 64, 128):
        raise ValueError("recomputed backward tile head dimension must be 16, 32, 64 or 128")
    if grad_attention.shape != query.shape:
        raise ValueError("recomputed backward tile query/grad_attention shapes must match")
    if key.shape != (batch, heads, columns, dim) or values.shape != key.shape:
        raise ValueError("recomputed backward tile keys and values must match batch, heads and dim")
    if any(value.shape != (batch, heads, rows)
           for value in (dot, row_maximum, row_denominator)):
        raise ValueError("recomputed backward tile row statistics must be [batch, heads, rows]")
    if key_valid.shape != (batch, columns) or key_valid.dtype != torch.bool:
        raise ValueError("recomputed backward tile key_valid must be bool [batch, columns]")
    if any(value.dtype != torch.bfloat16 for value in (query, key, values)):
        raise ValueError("recomputed backward tile query, key and values must be BF16")
    if any(value.dtype != torch.float32 for value in
           (grad_attention, dot, row_maximum, row_denominator)):
        raise ValueError("recomputed backward tile gradients and row statistics must be FP32")
    if torch.is_grad_enabled() and any(value.requires_grad for value in tensors):
        raise ValueError("recomputed backward tile has no autograd; invoke inside the first-order RT VJP")
    return batch, heads, rows, columns, dim


@lru_cache(maxsize=1)
def _get_recomputed_backward_tile_kernel():
    global tl, libdevice
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def historical_recomputed_backward_kernel(Q, K, V, G, DOT, MAXIMUM, DENOMINATOR,
            VALID, DK, DV,
            sqb, sqh, sqr, sqd, skb, skh, skc, skd, svb, svh, svc, svd,
            sgb, sgh, sgr, sgd, sdb, sdh, sdr, smb, smh, smr, snb, snh, snr,
            svalidb, svalidc,
            HEADS: tl.constexpr, ROWS: tl.constexpr, COLUMNS: tl.constexpr,
            DIM: tl.constexpr, SCALE: tl.constexpr,
            BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr):
        bh = tl.program_id(1)
        batch, head = bh // HEADS, bh % HEADS
        columns = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        features = tl.arange(0, DIM)
        k = tl.load(K + batch * skb + head * skh
            + columns[:, None] * skc + features[None, :] * skd,
            columns[:, None] < COLUMNS, other=0.0)
        v = tl.load(V + batch * svb + head * svh
            + columns[:, None] * svc + features[None, :] * svd,
            columns[:, None] < COLUMNS, other=0.0)
        valid = tl.load(VALID + batch * svalidb + columns * svalidc,
                        columns < COLUMNS, other=False)
        dkey = tl.full((BLOCK_C, DIM), 0.0, tl.float32)
        dvalue = tl.full((BLOCK_C, DIM), 0.0, tl.float32)
        for row_start in range(0, tl.cdiv(ROWS, BLOCK_R)):
            rows = row_start * BLOCK_R + tl.arange(0, BLOCK_R)
            q = tl.load(Q + batch * sqb + head * sqh
                + rows[:, None] * sqr + features[None, :] * sqd,
                rows[:, None] < ROWS, other=0.0)
            g = tl.load(G + batch * sgb + head * sgh
                + rows[:, None] * sgr + features[None, :] * sgd,
                rows[:, None] < ROWS, other=0.0).to(tl.bfloat16)
            maximum = tl.load(MAXIMUM + batch * smb + head * smh + rows * smr,
                              rows < ROWS, other=0.0)
            denominator = tl.load(DENOMINATOR + batch * snb + head * snh + rows * snr,
                                  rows < ROWS, other=0.0)
            dot = tl.load(DOT + batch * sdb + head * sdh + rows * sdr,
                          rows < ROWS, other=0.0)
            # Transposed storage changes no complete-inner-dimension product
            # boundary. QK and GV each round once to BF16 before FP32 math.
            scores_transposed = tl.dot(k, tl.trans(q)).to(tl.bfloat16).to(tl.float32) * SCALE
            probability_transposed = libdevice.exp(scores_transposed - maximum[None, :])
            safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
            probability_transposed = tl.div_rn(probability_transposed, safe_denominator[None, :])
            active = valid[:, None] & (rows[None, :] < ROWS) & (denominator[None, :] > 0.0)
            probability_transposed = tl.where(active, probability_transposed, 0.0)
            gv_transposed = tl.dot(v, tl.trans(g)).to(tl.bfloat16).to(tl.float32)
            error_transposed = probability_transposed * (gv_transposed - dot[None, :])
            # Accumulate every row chunk in FP32. Rounding partial products
            # would add BF16 boundaries absent from the materialized VJP.
            dkey = tl.dot(error_transposed.to(tl.bfloat16), q, dkey)
            dvalue = tl.dot(probability_transposed.to(tl.bfloat16), g, dvalue)
        dkey = dkey.to(tl.bfloat16).to(tl.float32) * SCALE
        dvalue = dvalue.to(tl.bfloat16).to(tl.float32)
        offsets = (bh * COLUMNS + columns[:, None]) * DIM + features[None, :]
        tl.store(DK + offsets, dkey, columns[:, None] < COLUMNS)
        tl.store(DV + offsets, dvalue, columns[:, None] < COLUMNS)

    return historical_recomputed_backward_kernel


def backward_recomputed_tile(query: Tensor, key: Tensor, values: Tensor,
                             grad_attention: Tensor, dot: Tensor,
                             row_maximum: Tensor, row_denominator: Tensor,
                             key_valid: Tensor):
    """Return fresh FP32 ``(dkey, dvalue)`` without storing a probability tile.

    Q/G are [B,H,R,D], K/V [B,H,C,D], dot/max/denominator [B,H,R], valid [B,C].
    Q/K/V are BF16, G/dot/maximum/denominator FP32, valid bool. R/C are 1..2048
    and D is 16/32/64/128. All inputs share a CUDA device; arbitrary strided or
    broadcast read-only inputs are supported. The supplied rows' softmax
    statistics include their *entire* attention domain, including temporary
    self keys. This helper's columns must all be historical relative to its
    rows; it applies key_valid, but no additional causal or diagonal mask.

    A nonpositive denominator denotes an empty/masked row. Finite operands and
    valid nonempty-row statistics are the caller's responsibility. No hidden
    synchronization, value inspection, input mutation, CPU fallback or autograd
    is performed. Warm required shapes before graph capture.
    """
    batch, heads, rows, columns, dim = _validate_recomputed_tile_metadata(
        query, key, values, grad_attention, dot, row_maximum, row_denominator, key_valid)
    if query.device.type != "cuda":
        raise ValueError("recomputed backward tile requires CUDA; no CPU fallback is provided")
    kernel = _get_recomputed_backward_tile_kernel()
    dkey = torch.empty((batch, heads, columns, dim), device=query.device, dtype=torch.float32)
    dvalue = torch.empty_like(dkey)
    block_c, block_r = 16, 32
    kernel[((columns + block_c - 1) // block_c, batch * heads)](
        query, key, values, grad_attention, dot, row_maximum, row_denominator,
        key_valid, dkey, dvalue,
        *query.stride(), *key.stride(), *values.stride(), *grad_attention.stride(),
        *dot.stride(), *row_maximum.stride(), *row_denominator.stride(), *key_valid.stride(),
        HEADS=heads, ROWS=rows, COLUMNS=columns, DIM=dim, SCALE=1.0 / math.sqrt(dim),
        BLOCK_C=block_c, BLOCK_R=block_r,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return dkey, dvalue
