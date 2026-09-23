"""Bounded-workspace reconstruction for the native RT first-order VJP.

Only row statistics and linear-sized outputs survive reconstruction. Query
chunks retain the complete key reduction; key chunks retain the complete query
reduction. This preserves the BF16 whole-product boundary of the reference VJP
without a global query-by-key probability/error tensor.
"""
from __future__ import annotations

import math
import torch

WORKSPACE_CHUNK = 32


def _mm(a, b, spec, dtype):
    from .olmo_tiled import _mm as reference_mm
    return reference_mm(a, b, spec, dtype)


def probability_tile(query, key, row_maximum, row_denominator, key_valid, spec, dtype):
    """Historical probabilities using complete-row statistics (no causal mask)."""
    scores = _mm(query, key.transpose(-1, -2), spec, dtype) / math.sqrt(spec.config.head_dim)
    scores.masked_fill_(~key_valid[:, None, None, :], -torch.inf)
    maximum = torch.where(row_denominator > 0, row_maximum, torch.zeros_like(row_maximum))
    weights = torch.exp(scores - maximum.unsqueeze(-1))
    return weights / row_denominator.clamp_min(1e-30).unsqueeze(-1)


def attention_from_completed(query, temporary_key, temporary_value, permanent_key,
                             permanent_value, valid, prefix_length, spec, dtype):
    """Return (maximum, denominator, self probability, attention), all O(T)."""
    batch, heads, length, dim = query.shape
    maximum = torch.empty((batch, heads, length), device=query.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    diagonal_probability = torch.empty_like(maximum)
    attention = torch.empty((batch, heads, length, dim), device=query.device, dtype=torch.float32)
    key_indices = torch.arange(permanent_key.shape[-2], device=query.device)
    for start in range(0, length, WORKSPACE_CHUNK):
        stop = min(length, start + WORKSPACE_CHUNK)
        rows = slice(start, stop)
        indices = torch.arange(start, stop, device=query.device)
        local_indices = torch.arange(stop - start, device=query.device)
        scores = _mm(query[:, :, rows], permanent_key.transpose(-1, -2), spec, dtype) / math.sqrt(dim)
        diagonal = (query[:, :, rows].float() * temporary_key[:, :, rows].float()).sum(-1) / math.sqrt(dim)
        scores[:, :, local_indices, prefix_length + indices] = diagonal
        allowed = ((key_indices[None, :] <= prefix_length + indices[:, None])[None, None]
                   & valid[:, None, None, :])
        scores.masked_fill_(~allowed, -torch.inf)
        row_max = scores.max(-1).values
        safe_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        probabilities = torch.exp(scores - safe_max.unsqueeze(-1))
        row_den = probabilities.sum(-1)
        probabilities = probabilities / row_den.clamp_min(1e-30).unsqueeze(-1)
        diagonal_p = probabilities[:, :, local_indices, prefix_length + indices]
        # Advanced indexing returns a copy, so zeroing P does not erase self P.
        diagonal_probability[:, :, rows] = diagonal_p
        # A diagonal view avoids advanced-index scalar assignment, which may
        # stage a CPU scalar and is not legal inside CUDA graph capture.
        probabilities.diagonal(offset=prefix_length + start, dim1=-2, dim2=-1).zero_()
        attended = _mm(probabilities, permanent_value, spec, dtype)
        attended += diagonal_p.unsqueeze(-1) * temporary_value[:, :, rows].float()
        attention[:, :, rows] = attended
        maximum[:, :, rows], denominator[:, :, rows] = row_max, row_den
    return maximum, denominator, diagonal_probability, attention


def historical_backward(query, key, values, grad_attention, dot,
                        row_maximum, row_denominator, key_valid, spec, dtype):
    """Recompute history P while retaining whole-query dK/dV reductions."""
    if (spec.backward_tile_backend == "triton" and query.device.type == "cuda"
            and spec.attention_precision == "mixed" and dtype == torch.bfloat16
            and query.dtype == key.dtype == values.dtype == torch.bfloat16
            and grad_attention.dtype == dot.dtype == row_maximum.dtype == row_denominator.dtype == torch.float32
            and spec.config.head_dim in (16, 32, 64, 128)
            and 0 < min(query.shape[-2], key.shape[-2])
            and max(query.shape[-2], key.shape[-2]) <= 2048):
        from .olmo_rt_recompute_kernels import backward_recomputed_tile
        return backward_recomputed_tile(query, key, values, grad_attention, dot,
                                        row_maximum, row_denominator, key_valid)
    dkey = torch.empty_like(key, dtype=torch.float32)
    dvalue = torch.empty_like(values, dtype=torch.float32)
    for start in range(0, key.shape[-2], WORKSPACE_CHUNK):
        stop = min(key.shape[-2], start + WORKSPACE_CHUNK)
        columns = slice(start, stop)
        probabilities = probability_tile(query, key[:, :, columns], row_maximum,
                                         row_denominator, key_valid[:, columns], spec, dtype)
        dvalue[:, :, columns] = _mm(probabilities.transpose(-1, -2), grad_attention, spec, dtype)
        error = probabilities * (_mm(grad_attention, values[:, :, columns].transpose(-1, -2), spec, dtype)
                                 - dot.unsqueeze(-1))
        dkey[:, :, columns] = _mm(error.transpose(-1, -2), query, spec, dtype) / math.sqrt(spec.config.head_dim)
    return dkey, dvalue


def query_and_prefix_backward(query, temporary_key, temporary_value, all_keys, all_values,
                              grad_attention, dot, row_maximum, row_denominator,
                              diagonal_probability, valid, prefix_length, spec, dtype):
    """Final query/self/prefix adjoints with bounded attention scratch."""
    length, dim = query.shape[-2:]
    scale = math.sqrt(dim)
    self_error = diagonal_probability * ((grad_attention * temporary_value.float()).sum(-1) - dot)
    dkt = self_error.unsqueeze(-1) * query.float() / scale
    dvt = diagonal_probability.unsqueeze(-1) * grad_attention
    dq = torch.empty_like(query, dtype=torch.float32)
    key_indices = torch.arange(all_keys.shape[-2], device=query.device)
    for start in range(0, length, WORKSPACE_CHUNK):
        stop = min(length, start + WORKSPACE_CHUNK)
        rows = slice(start, stop)
        probabilities = probability_tile(query[:, :, rows], all_keys, row_maximum[:, :, rows],
                                         row_denominator[:, :, rows], valid, spec, dtype)
        indices = torch.arange(start, stop, device=query.device)
        # Permanent self is absent; its independent temporary contribution is below.
        probabilities.masked_fill_(key_indices[None, :] >= (prefix_length + indices[:, None]), 0)
        error = probabilities * (_mm(grad_attention[:, :, rows], all_values.transpose(-1, -2), spec, dtype)
                                 - dot[:, :, rows].unsqueeze(-1))
        dq[:, :, rows] = _mm(error, all_keys, spec, dtype) / scale
    dq += self_error.unsqueeze(-1) * temporary_key.float() / scale
    dpk, dpv = historical_backward(query, all_keys[:, :, :prefix_length], all_values[:, :, :prefix_length],
                                   grad_attention, dot, row_maximum, row_denominator,
                                   valid[:, :prefix_length], spec, dtype)
    return dq, dkt, dvt, dpk, dpv
