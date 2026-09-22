"""Opt-in fused historical RT tile; the caller owns recurrence and its VJP.

This CUDA-only helper implements the mixed arithmetic of ``olmo_tiled._add_tile``.
It preserves the explicit BF16 QK/PV rounding boundaries, not a claim of bitwise
cuBLAS/reduction equivalence. Temporary self attention, RoPE, scheduling and
backward reconstruction remain outside this module. Triton is imported lazily.
"""

from functools import lru_cache
import math

import torch
from torch import Tensor


def _validate_tile_metadata(query, key, value, valid, numerator, maximum, denominator):
    """Validate shapes/dtypes/strides without reading a tensor value or syncing.

    Device type is checked separately by the public CUDA-only entry point. This
    metadata checker can consequently be exercised on ordinary CPU fixtures.
    Inputs may be noncontiguous or broadcast views; outputs are newly allocated.
    """
    tensors = (query, key, value, valid, numerator, maximum, denominator)
    if not all(isinstance(item, Tensor) for item in tensors):
        raise TypeError("historical tile inputs must all be tensors")
    if any(item.layout != torch.strided for item in tensors):
        raise ValueError("historical tile inputs must have strided layout")
    if any(item.device != query.device for item in tensors):
        raise ValueError("historical tile inputs must share one device")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key and value must have shape [batch, heads, time, dim]")
    batch, heads, target, dim = query.shape
    source = key.shape[2]
    if batch < 1 or heads < 1:
        raise ValueError("historical tile batch and head counts must be positive")
    if not (1 <= source <= 256 and 1 <= target <= 256):
        raise ValueError("historical tile source and target lengths must be in [1, 256]")
    if dim not in (16, 32, 64, 128):
        raise ValueError("historical tile head dimension must be 16, 32, 64 or 128")
    if key.shape != (batch, heads, source, dim) or value.shape != key.shape:
        raise ValueError("historical tile key/value shapes must match query batch, heads and dim")
    if valid.shape != (batch, source) or valid.dtype != torch.bool:
        raise ValueError("historical tile valid mask must be bool [batch, source]")
    if numerator.shape != query.shape:
        raise ValueError("historical tile numerator must have the query shape")
    if maximum.shape != query.shape[:-1] or denominator.shape != maximum.shape:
        raise ValueError("historical tile maximum/denominator must be [batch, heads, target]")
    if any(item.dtype != torch.bfloat16 for item in (query, key, value)):
        raise ValueError("historical tile query, key and value must be BF16")
    if any(item.dtype != torch.float32 for item in (numerator, maximum, denominator)):
        raise ValueError("historical tile numerator, maximum and denominator must be FP32")
    if torch.is_grad_enabled() and any(item.requires_grad for item in tensors):
        raise ValueError("fused historical tile is forward-only; use it inside the RT custom VJP")
    return batch, heads, target, source, dim


@lru_cache(maxsize=1)
def _get_tile_kernel():
    # Keep importing the CPU model independent of an installed Triton/CUDA stack.
    # These globals are resolved by Triton's JIT when compiling the nested kernel.
    global tl, libdevice
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice

    @triton.jit
    def kernel(Q, K, V, VALID, N, M, D, OUT_N, OUT_M, OUT_D,
               sqb, sqh, sqt, sqd, skb, skh, skt, skd,
               svb, svh, svt, svd, smb, smt,
               snb, snh, snt, snd, sxb, sxh, sxt, sdb, sdh, sdt,
               HEADS: tl.constexpr, TARGET: tl.constexpr, SOURCE: tl.constexpr,
               DIM: tl.constexpr, SCALE: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        bh = tl.program_id(1)
        batch, head = bh // HEADS, bh % HEADS
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        features = tl.arange(0, DIM)
        q = tl.load(Q + batch * sqb + head * sqh + rows[:, None] * sqt
                    + features[None, :] * sqd, rows[:, None] < TARGET, other=0)
        k = tl.load(K + batch * skb + head * skh + cols[None, :] * skt
                    + features[:, None] * skd, cols[None, :] < SOURCE, other=0)
        # The eager mixed path rounds the COMPLETE QK result before scaling it.
        score = tl.dot(q, k).to(tl.bfloat16).to(tl.float32) * SCALE
        key_valid = tl.load(VALID + batch * smb + cols * smt, cols < SOURCE, other=0)
        score = tl.where((rows[:, None] < TARGET) & key_valid[None, :], score, -float("inf"))
        old_max = tl.load(M + batch * sxb + head * sxh + rows * sxt,
                          rows < TARGET, other=-float("inf"))
        merged_max = tl.maximum(old_max, tl.max(score, axis=1))
        finite = (merged_max != float("inf")) & (merged_max != -float("inf")) & (merged_max == merged_max)
        safe_max = tl.where(finite, merged_max, 0.0)
        # libdevice.exp retains ordinary float32 exp rather than opting into
        # the faster approximate exp2 pattern used by many attention kernels.
        factor = libdevice.exp(old_max - safe_max)
        weights = libdevice.exp(score - safe_max[:, None])
        old_denom = tl.load(D + batch * sdb + head * sdh + rows * sdt,
                            rows < TARGET, other=0.0)
        new_denom = old_denom * factor + tl.sum(weights, axis=1)
        v = tl.load(V + batch * svb + head * svh + cols[:, None] * svt
                    + features[None, :] * svd, cols[:, None] < SOURCE, other=0)
        # One dot covers the ENTIRE source tile. Never round separate partial
        # products before their reduction; that would change the eager contract.
        history = tl.dot(weights.to(tl.bfloat16), v).to(tl.bfloat16).to(tl.float32)
        old_num = tl.load(N + batch * snb + head * snh + rows[:, None] * snt
                         + features[None, :] * snd, rows[:, None] < TARGET, other=0.0)
        new_num = old_num * factor[:, None] + history
        scalar_offsets = bh * TARGET + rows
        output_offsets = scalar_offsets[:, None] * DIM + features[None, :]
        tl.store(OUT_N + output_offsets, new_num, rows[:, None] < TARGET)
        tl.store(OUT_M + scalar_offsets, merged_max, rows < TARGET)
        tl.store(OUT_D + scalar_offsets, new_denom, rows < TARGET)

    return kernel


def add_tile(query: Tensor, key: Tensor, value: Tensor, valid: Tensor,
             numerator: Tensor, maximum: Tensor, denominator: Tensor):
    """Return updated FP32 ``(numerator, maximum, denominator)`` for one tile.

    Q/K are already rotated native RoPE operands. All tensors must be on the same
    CUDA device. Q/K/V are BF16, state is FP32, lengths are 1..256 and head width
    is 16/32/64/128. Arbitrary strided read-only inputs, including expanded views,
    are supported. An all-false key mask contributes zero, including when the
    incoming state is empty (zero numerator/denominator, maximum -inf).

    This function deliberately has no fallback and no autograd implementation.
    Unsupported metadata is rejected before allocation or Triton import. The
    owning RT custom Function supplies backward. The first call compiles kernels;
    warm supported shapes before CUDA graph capture.
    """
    batch, heads, target, source, dim = _validate_tile_metadata(
        query, key, value, valid, numerator, maximum, denominator)
    if query.device.type != "cuda":
        raise ValueError("fused historical tile requires CUDA; no CPU fallback is provided")
    kernel = _get_tile_kernel()
    out_n = torch.empty(query.shape, device=query.device, dtype=torch.float32)
    out_m = torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32)
    out_d = torch.empty_like(out_m)
    # Pad each dot's contracting/source extent to a tensor-core supported size.
    block_m = 16
    block_n = max(16, 1 << (source - 1).bit_length())
    kernel[((target + block_m - 1) // block_m, batch * heads)](
        query, key, value, valid, numerator, maximum, denominator, out_n, out_m, out_d,
        *query.stride(), *key.stride(), *value.stride(), *valid.stride(),
        *numerator.stride(), *maximum.stride(), *denominator.stride(),
        HEADS=heads, TARGET=target, SOURCE=source, DIM=dim,
        SCALE=1.0 / math.sqrt(dim), BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return out_n, out_m, out_d
