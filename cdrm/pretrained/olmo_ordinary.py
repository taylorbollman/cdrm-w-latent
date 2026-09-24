"""Opt-in ordinary-block execution helpers, with no parameters or buffers.

Neither helper is used by the native RT scan, tiled recurrence or author port.
FA4 is imported only when requested, after the caller's device/determinism
setup. Compilation covers the pure SwiGLU pointwise expression only; projections,
native FP32 residuals/RoPE, normalization and checkpoint ownership stay outside.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional as F


def validate_ordinary_options(attention_backend, pointwise_backend, *, sdpa_backend):
    if attention_backend not in ("sdpa", "fa4"):
        raise ValueError("ordinary_attention_backend must be sdpa or fa4")
    if pointwise_backend not in ("eager", "compiled"):
        raise ValueError("ordinary_pointwise_backend must be eager or compiled")
    if attention_backend == "fa4" and sdpa_backend != "sdpa":
        raise ValueError("ordinary FA4 requires attention_backend='sdpa'; it cannot override the math oracle")


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
    return torch.compile(_swiglu, fullgraph=True, dynamic=False)


def ordinary_swiglu(projected: Tensor, *, backend: str) -> Tensor:
    if backend == "eager":
        return _swiglu(projected)
    if backend == "compiled":
        return _compiled_swiglu()(projected)
    raise ValueError("ordinary_pointwise_backend must be eager or compiled")
