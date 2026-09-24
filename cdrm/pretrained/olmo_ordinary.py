"""Opt-in ordinary-block execution helpers, with no parameters or buffers.

These helpers are not used by the native RT scan, tiled recurrence or author
port. External kernels are imported only when requested, after the caller's
device/determinism setup. Compilation covers only the pure SwiGLU expression;
the separate Dao RoPE adapter retains native FP32 casts and phase tables.
Projections, residuals, normalization and checkpoint ownership stay unchanged.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional as F

from .olmo_rope import DaoRopeTables


def validate_ordinary_options(attention_backend, pointwise_backend, *, sdpa_backend, rope_backend="native"):
    if attention_backend not in ("sdpa", "fa4"):
        raise ValueError("ordinary_attention_backend must be sdpa or fa4")
    if pointwise_backend not in ("eager", "compiled"):
        raise ValueError("ordinary_pointwise_backend must be eager or compiled")
    if attention_backend == "fa4" and sdpa_backend != "sdpa":
        raise ValueError("ordinary FA4 requires attention_backend='sdpa'; it cannot override the math oracle")
    if rope_backend not in ("native", "dao"):
        raise ValueError("ordinary_rope_backend must be native or dao")


@lru_cache(maxsize=1)
def _load_dao_rope():
    # Installed flash_attn 2.7.4.post1+git5231d95fe1 rotary helper, independent
    # of the attention backend. Import only after device/determinism setup.
    from flash_attn.layers.rotary import apply_rotary_emb
    return apply_rotary_emb


def _validate_dao_rope_inputs(x: Tensor, tables: DaoRopeTables):
    if x.device.type != "cuda":
        raise ValueError("Ordinary Dao RoPE requires CUDA; there is no CPU or native fallback")
    if (not isinstance(tables, DaoRopeTables) or x.ndim != 4
            or x.shape[0] != tables.batch_size or x.shape[2] != tables.sequence_length
            or x.shape[3] != 2 * tables.cos.shape[1] or x.shape[3] > 256
            or x.device != tables.cos.device or x.dtype not in (torch.float32, torch.bfloat16, torch.float16)):
        raise ValueError("Dao RoPE requires matching [batch, heads, length, dim<=256] projections and prepared tables")


def apply_dao_rope(x: Tensor, tables: DaoRopeTables) -> Tensor:
    """Preserve native FP32 input, tables and backward accumulation boundaries.

    Dao's installed kernel requires matching input/table dtypes. Passing BF16
    tables would change native OLMo, so promote the projection once, use FP32
    out-of-place rotation, then restore its dtype. Conjugate backward therefore
    accumulates in FP32 before the input cast's backward returns to BF16.
    The unmodified kernel may contract FP32 products/addition into FMA, so
    cross-backend bitwise equality is not promised. No cache tensor is mutated.
    """
    _validate_dao_rope_inputs(x, tables)
    with torch.autocast(device_type=x.device.type, enabled=False):
        rotated = _load_dao_rope()(x.float().transpose(1, 2), tables.cos, tables.sin,
            interleaved=False, inplace=False, seqlen_offsets=tables.offsets)
    return rotated.transpose(1, 2).to(x.dtype)


def validate_checkpoint_layers(layers, *, num_layers, enabled):
    if layers is None:
        return
    if not isinstance(layers, tuple) or any(type(index) is not int for index in layers):
        raise TypeError("ordinary_checkpoint_layers must be a tuple of integer indices or None")
    if tuple(sorted(set(layers))) != layers or any(not 0 <= index < num_layers for index in layers):
        raise ValueError("ordinary_checkpoint_layers must be sorted unique indices inside the model")
    if not enabled:
        raise ValueError("ordinary_checkpoint_layers requires ordinary_activation_checkpointing=True")


def _validate_fa4_inputs(query: Tensor, key: Tensor, value: Tensor):
    if any(tensor.device.type != "cuda" for tensor in (query, key, value)):
        raise ValueError("Ordinary FA4 requires CUDA; there is no CPU or SDPA fallback")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Ordinary FA4 requires BF16 or FP16 projected Q/K/V")
    if any(tensor.ndim != 4 or tensor.shape != query.shape or tensor.dtype != query.dtype
           or tensor.device != query.device or tensor.stride(-1) != 1 for tensor in (query, key, value)):
        raise ValueError("Ordinary FA4 requires matching dense [batch, heads, length, dim] Q/K/V with contiguous head dimensions")


@lru_cache(maxsize=1)
def _load_fa4():
    from flash_attn.cute.interface import flash_attn_func
    return flash_attn_func


def flash_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """Dense causal full-sequence attention; caller rejects masks and caches.

    The pinned FA4 b20 interface returns (output, optional_lse). Retain the
    actual projection layouts: these are views, not contiguous layout copies.
    """
    _validate_fa4_inputs(query, key, value)
    result = _load_fa4()(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                         causal=True, deterministic=True)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("Expected the pinned FA4 interface to return (output, optional_lse)")
    return result[0].transpose(1, 2)


def _swiglu(projected: Tensor) -> Tensor:
    up, gate = projected.chunk(2, dim=-1)
    return F.silu(gate) * up


@lru_cache(maxsize=1)
def _compiled_swiglu():
    # fullgraph rejects graph breaks instead of quietly accepting partial
    # compilation. Warm both forward/backward before CUDA-graph capture.
    # Retain eager's BF16 SiLU output rounding before multiplication while
    # still fusing the pointwise kernels. This option is local to this helper.
    return torch.compile(_swiglu, fullgraph=True, dynamic=False,
                         options={"emulate_precision_casts": True})


def ordinary_swiglu(projected: Tensor, *, backend: str) -> Tensor:
    if backend == "eager":
        return _swiglu(projected)
    if backend == "compiled":
        return _compiled_swiglu()(projected)
    raise ValueError("ordinary_pointwise_backend must be eager or compiled")
