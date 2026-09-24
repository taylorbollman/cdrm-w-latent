"""Dyadically tiled native OLMo recurrence with explicit-parameter backward.

The custom Function retains layer inputs and completed outputs, not the scan's
activation graphs. Backward reconstructs projections and attention in parallel,
propagates recurrent credit in reverse with frozen-weight local VJPs, and uses
one batched VJP to return input/parameter/cache gradients. It never writes .grad
or replays a sequential recurrent forward. Higher-order derivatives are outside
this first-order backend's contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .olmo import OLMoBlock, OLMoConfig, _apply_rope
from .olmo_rope import RopeTables, apply_rope_tables, build_rope_tables
from .olmo_ordinary import validate_checkpoint_layers, validate_ordinary_options
from .olmo_recurrent import OLMoRTForCausalLM
from .recurrent import RTExecutionContext, RTMode


@dataclass(frozen=True)
class _Invocation:
    config: OLMoConfig
    alpha: float
    attention_precision: str
    autocast_enabled: bool
    autocast_dtype: torch.dtype
    cast_weights_once: bool = False
    tile_backend: str = "eager"
    backward_tile_backend: str = "eager"
    backward_memory: str = "materialized"
    kv_only_writes: bool = False


def _replay(spec: _Invocation, device: torch.device):
    return torch.autocast(device.type, enabled=spec.autocast_enabled, dtype=spec.autocast_dtype)


def _project(x: Tensor, weight: Tensor, config: OLMoConfig):
    normalized = F.layer_norm(x, (config.model_dim,), eps=config.layer_norm_eps)
    parts = F.linear(normalized, weight).split(config.model_dim, dim=-1)
    return tuple(part.view(x.shape[0], x.shape[1], config.num_heads, config.head_dim).transpose(1, 2) for part in parts)


def _project_kv(x: Tensor, weight: Tensor, config: OLMoConfig):
    """Read the existing packed parameter's K/V rows; never copy parameters."""
    normalized = F.layer_norm(x, (config.model_dim,), eps=config.layer_norm_eps)
    parts = F.linear(normalized, weight[config.model_dim:]).split(config.model_dim, dim=-1)
    return tuple(part.view(x.shape[0], x.shape[1], config.num_heads, config.head_dim).transpose(1, 2) for part in parts)


def _project_memory(x, weight, config, kv_only):
    return _project_kv(x, weight, config) if kv_only else _project(x, weight, config)[1:]


def _rotate(x, positions, config, tables):
    return (_apply_rope(x, positions, config.rope_freq_constant) if tables is None
            else apply_rope_tables(x, tables))


def _finish(x, attended, out_weight, up_gate_weight, ff_out_weight, config, projection_dtype):
    attended = attended.to(projection_dtype).transpose(1, 2).contiguous().view(x.shape)
    residual = x + F.linear(attended, out_weight)
    normalized = F.layer_norm(residual, (config.model_dim,), eps=config.layer_norm_eps)
    up, gate = F.linear(normalized, up_gate_weight).chunk(2, dim=-1)
    return residual + F.linear(F.silu(gate) * up, ff_out_weight)


def _mm(a: Tensor, b: Tensor, spec: _Invocation, projection_dtype: torch.dtype) -> Tensor:
    """Explicit matrix arithmetic; online state and reductions stay FP32.

    Mixed uses the projection dtype for matrix operands/results, like the
    paper's mixed path. FP32 promotes attention matrices only; dense native
    projections/MLPs retain the caller's autocast behavior.
    """
    dtype = torch.float32 if spec.attention_precision == "fp32" else projection_dtype
    with torch.autocast(a.device.type, enabled=False):
        return torch.matmul(a.to(dtype), b.to(dtype)).float()


def _add_tile(query, key, value, valid, numerator, maximum, denominator, spec, dtype):
    if (spec.tile_backend == "triton" and query.device.type == "cuda"
            and dtype == torch.bfloat16 and spec.attention_precision == "mixed"
            and all(tensor.dtype == torch.bfloat16 for tensor in (query, key, value))
            and spec.config.head_dim in (16, 32, 64, 128)
            and max(query.shape[-2], key.shape[-2]) <= 256):
        from .olmo_rt_kernels import add_tile
        return add_tile(query, key, value, valid, numerator, maximum, denominator)
    score = _mm(query, key.transpose(-1, -2), spec, dtype) / math.sqrt(spec.config.head_dim)
    score = score.masked_fill(~valid[:, None, None, :], -torch.inf)
    merged_max = torch.maximum(maximum, score.max(-1).values)
    # Empty/all-masked histories must produce zero, including left padding.
    safe_max = torch.where(torch.isfinite(merged_max), merged_max, torch.zeros_like(merged_max))
    old_factor = torch.exp(maximum - safe_max)
    weights = torch.exp(score - safe_max.unsqueeze(-1))
    numerator = numerator * old_factor.unsqueeze(-1) + _mm(weights, value, spec, dtype)
    denominator = denominator * old_factor + weights.sum(-1)
    return numerator, merged_max, denominator


def _attention_from_completed(query, temporary_key, temporary_value, permanent_key,
                              permanent_value, valid, prefix_length, spec, dtype):
    """Reconstruct complete causal attention once all completed states are known."""
    length = query.shape[-2]
    scores = _mm(query, permanent_key.transpose(-1, -2), spec, dtype) / math.sqrt(spec.config.head_dim)
    indices = torch.arange(length, device=query.device)
    diagonal = (query.float() * temporary_key.float()).sum(-1) / math.sqrt(spec.config.head_dim)
    scores[:, :, indices, prefix_length + indices] = diagonal
    keys = torch.arange(permanent_key.shape[-2], device=query.device)
    causal = keys[None, :] <= (prefix_length + indices[:, None])
    allowed = causal[None, None] & valid[:, None, None, :]
    scores = scores.masked_fill(~allowed, -torch.inf)
    any_valid = allowed.any(-1, keepdim=True)
    scores = torch.where(any_valid, scores, torch.zeros_like(scores))
    probabilities = scores.softmax(-1) * any_valid
    diagonal_probability = probabilities[:, :, indices, prefix_length + indices]
    historical_probability = probabilities.clone()
    historical_probability.diagonal(offset=prefix_length, dim1=-2, dim2=-1).zero_()
    attention = _mm(historical_probability, permanent_value, spec, dtype)
    # Keep the temporary diagonal separate. Subtracting a permanent diagonal
    # after a BF16 PV matmul would mix rounded and unrounded probabilities.
    attention += diagonal_probability.unsqueeze(-1) * temporary_value.float()
    return probabilities, attention


def _historical_backward_tile(probabilities, grad_attention, values, query, dot, spec, dtype):
    """Reverse dyadic history update; the caller owns FP32 adjoint accumulation.

    The optional kernel preserves BF16 matrix operand/result boundaries and
    FP32 probability/error arithmetic. It consumes already reconstructed P;
    this helper alone does not remove quadratic backward intermediates.
    """
    if (spec.backward_tile_backend == "triton" and query.device.type == "cuda"
            and spec.attention_precision == "mixed" and dtype == torch.bfloat16
            and query.dtype == values.dtype == torch.bfloat16
            and probabilities.dtype == grad_attention.dtype == dot.dtype == torch.float32
            and spec.config.head_dim in (16, 32, 64, 128)
            and max(query.shape[-2], values.shape[-2]) <= 256):
        from .olmo_rt_backward_kernels import backward_tile
        return backward_tile(probabilities, grad_attention, values, query, dot)
    dvalue = _mm(probabilities.transpose(-1, -2), grad_attention, spec, dtype)
    error = probabilities * (_mm(grad_attention, values.transpose(-1, -2), spec, dtype)
                             - dot.unsqueeze(-1))
    dkey = _mm(error.transpose(-1, -2), query, spec, dtype) / math.sqrt(spec.config.head_dim)
    return dkey, dvalue


class _TiledRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wq, wo, wup, wdown, past_key, past_value,
                positions, key_positions, valid, spec, query_cos, query_sin, key_cos, key_sin):
        config = spec.config
        query_rope = None if query_cos is None else RopeTables(query_cos, query_sin)
        key_rope = None if key_cos is None else RopeTables(key_cos, key_sin)
        batch, length, _ = x.shape
        prefix = past_key.shape[-2]
        query, temporary_key, temporary_value = _project(x, wq, config)
        dtype = query.dtype
        # A custom forward records no dense activation graph. Cast once per
        # invocation rather than copying full FP32 matrices at every token.
        # Original parameter tensors are still saved for the batched VJP;
        # captured casts execute again and reread updated weights on replay.
        forward_weights = tuple(w.to(dtype) for w in (wq, wo, wup, wdown)) if (
            spec.cast_weights_once and spec.autocast_enabled
        ) else (wq, wo, wup, wdown)
        query = _rotate(query, positions, config, query_rope)
        temporary_key = _rotate(temporary_key, positions, config, query_rope)
        with torch.autocast(x.device.type, enabled=False):
            maximum = (query.float() * temporary_key.float()).sum(-1) / math.sqrt(config.head_dim)
            self_valid = valid[:, None, prefix:]
            maximum = maximum.masked_fill(~self_valid, -torch.inf)
            denominator = self_valid.expand(batch, config.num_heads, length).float().clone()
            numerator = temporary_value.float() * self_valid.unsqueeze(-1)
        if prefix:
            prefix_key = _rotate(past_key, key_positions[:, :prefix], config,
                                 None if key_rope is None else key_rope.slice(0, prefix))
            numerator, maximum, denominator = _add_tile(
                query, prefix_key, past_value, valid[:, :prefix],
                numerator, maximum, denominator, spec, dtype,
            )
        shape = (batch, config.num_heads, length, config.head_dim)
        keys = torch.empty(shape, device=x.device, dtype=dtype)
        values = torch.empty_like(keys)
        rotated_keys = torch.empty_like(keys)
        outputs = []
        for index in range(length):
            attention = numerator[:, :, index:index + 1] / denominator[:, :, index:index + 1].clamp_min(1e-30).unsqueeze(-1)
            current = x[:, index:index + 1]
            completed = _finish(current, attention, *forward_weights[1:], config, dtype)
            outputs.append(completed)
            source = (1.0 - spec.alpha) * current + spec.alpha * completed
            key, value = _project_memory(source, forward_weights[0], config, spec.kv_only_writes)
            keys[:, :, index:index + 1] = key
            values[:, :, index:index + 1] = value
            rotated_keys[:, :, index:index + 1] = _rotate(key, positions[:, index:index + 1], config,
                None if query_rope is None else query_rope.slice(index, index + 1))
            boundary = index + 1
            if boundary == length:
                continue
            width = boundary & -boundary
            stop = min(length, boundary + width)
            target = slice(boundary, stop)
            source_slice = slice(boundary - width, boundary)
            n, m, d = _add_tile(
                query[:, :, target], rotated_keys[:, :, source_slice], values[:, :, source_slice],
                valid[:, prefix + boundary - width:prefix + boundary],
                numerator[:, :, target], maximum[:, :, target], denominator[:, :, target], spec, dtype,
            )
            numerator[:, :, target], maximum[:, :, target], denominator[:, :, target] = n, m, d
        completed = torch.cat(outputs, dim=1)
        ctx.save_for_backward(x, completed, wq, wo, wup, wdown, past_key, past_value,
                              positions, key_positions, valid, query_cos, query_sin, key_cos, key_sin)
        ctx.spec, ctx.projection_dtype = spec, dtype
        ctx.set_materialize_grads(False)
        return completed, keys, values

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_new_key, grad_new_value):
        (x, completed, wq, wo, wup, wdown, past_key, past_value, positions, key_positions, valid,
         query_cos, query_sin, key_cos, key_sin) = ctx.saved_tensors
        query_rope = None if query_cos is None else RopeTables(query_cos, query_sin)
        key_rope = None if key_cos is None else RopeTables(key_cos, key_sin)
        spec, dtype = ctx.spec, ctx.projection_dtype
        config, alpha = spec.config, spec.alpha
        length, prefix = x.shape[1], past_key.shape[-2]
        shape = (x.shape[0], config.num_heads, length, config.head_dim)
        grad_output = torch.zeros_like(completed) if grad_output is None else grad_output
        grad_new_key = torch.zeros(shape, device=x.device, dtype=dtype) if grad_new_key is None else grad_new_key
        grad_new_value = torch.zeros_like(grad_new_key) if grad_new_value is None else grad_new_value
        # Detached local leaves isolate this custom VJP from outer .grad fields.
        xg = x.detach().requires_grad_(True)
        weights = tuple(w.detach().requires_grad_(True) for w in (wq, wo, wup, wdown))
        # Autocast does not cache casts of detached, requires_grad=False weights.
        # Cast each local-Jacobian weight once, not once per sequence position.
        # Batched parameter VJPs below still use the original FP32 weight leaves.
        frozen = tuple(w.detach().to(dtype) if spec.autocast_enabled else w.detach() for w in weights)
        pkg, pvg = past_key.detach().requires_grad_(True), past_value.detach().requires_grad_(True)
        with torch.enable_grad(), _replay(spec, x.device):
            q, kt, vt = _project(xg, weights[0], config)
            q = _rotate(q, positions, config, query_rope)
            kt = _rotate(kt, positions, config, query_rope)
            memory_source = (1.0 - alpha) * xg + alpha * completed.detach()
            kr, vr = _project_memory(memory_source, weights[0], config, spec.kv_only_writes)
            rotated_current = _rotate(kr, positions, config, query_rope)
            rotated_prefix = _rotate(pkg, key_positions[:, :prefix], config,
                                     None if key_rope is None else key_rope.slice(0, prefix))
            all_keys = torch.cat((rotated_prefix, rotated_current), dim=-2)
            all_values = torch.cat((pvg, vr), dim=-2)
            with torch.no_grad():
                if spec.backward_memory == "recompute":
                    from . import olmo_rt_memory
                    # Match the reference matrix operand casts once, including
                    # prefixes whose raw cache dtype differs from projections.
                    attention_dtype = torch.float32 if spec.attention_precision == "fp32" else dtype
                    memory_keys = all_keys.detach().to(attention_dtype)
                    memory_values = all_values.detach().to(attention_dtype)
                    maximum, denominator, diagonal_p, attention = olmo_rt_memory.attention_from_completed(
                        q.detach(), kt.detach(), vt.detach(), memory_keys, memory_values,
                        valid, prefix, spec, dtype,
                    )
                else:
                    probabilities, attention = _attention_from_completed(
                        q.detach(), kt.detach(), vt.detach(), all_keys.detach(), all_values.detach(),
                        valid, prefix, spec, dtype,
                    )
                dk = torch.zeros(shape, device=x.device, dtype=torch.float32)
                dv = torch.zeros_like(dk)
                ga = torch.zeros_like(dk)
                gamma = torch.empty_like(completed)
                dot = torch.zeros(shape[:-1], device=x.device, dtype=torch.float32)
            # Only local Jacobian-vector products are sequential. No completed
            # state is regenerated from preceding states, and no weight VJP is
            # evaluated per token. Permanent projection/attention graphs above
            # and parameter VJPs below are batched over the complete sequence.
            for index in range(length - 1, -1, -1):
                source = memory_source[:, index:index + 1].detach().requires_grad_(True)
                local_key, local_value = _project_memory(source, frozen[0], config, spec.kv_only_writes)
                local_rotated = _rotate(local_key, positions[:, index:index + 1], config,
                    None if query_rope is None else query_rope.slice(index, index + 1))
                du = torch.autograd.grad(
                    (local_rotated, local_key, local_value), source,
                    grad_outputs=(dk[:, :, index:index + 1], grad_new_key[:, :, index:index + 1],
                                  dv[:, :, index:index + 1] + grad_new_value[:, :, index:index + 1]),
                )[0]
                total = grad_output[:, index:index + 1] + alpha * du
                local_attention = attention[:, :, index:index + 1].detach().requires_grad_(True)
                local_finished = _finish(x[:, index:index + 1].detach(), local_attention,
                                         *frozen[1:], config, dtype)
                g = torch.autograd.grad(local_finished, local_attention, grad_outputs=total)[0]
                with torch.no_grad():
                    gamma[:, index:index + 1] = total
                    ga[:, :, index:index + 1] = g
                    dot[:, :, index:index + 1] = (g * attention[:, :, index:index + 1]).sum(-1)
                    if index:
                        width = index & -index
                        queries = slice(index, min(length, index + width))
                        keys_slice = slice(index - width, index)
                        if spec.backward_memory == "recompute":
                            history = slice(prefix + index - width, prefix + index)
                            dkey, dvalue = olmo_rt_memory.historical_backward(
                                q.detach()[:, :, queries], memory_keys[:, :, history], memory_values[:, :, history],
                                ga[:, :, queries], dot[:, :, queries], maximum[:, :, queries],
                                denominator[:, :, queries], valid[:, history], spec, dtype,
                            )
                        else:
                            p = probabilities[:, :, queries, prefix + index - width:prefix + index]
                            dkey, dvalue = _historical_backward_tile(
                                p, ga[:, :, queries], vr.detach()[:, :, keys_slice],
                                q.detach()[:, :, queries], dot[:, :, queries], spec, dtype,
                            )
                        dv[:, :, keys_slice] += dvalue
                        dk[:, :, keys_slice] += dkey
            with torch.no_grad():
                if spec.backward_memory == "recompute":
                    dq, dkt, dvt, dpk, dpv = olmo_rt_memory.query_and_prefix_backward(
                        q.detach(), kt.detach(), vt.detach(), memory_keys, memory_values,
                        ga, dot, maximum, denominator, diagonal_p, valid, prefix, spec, dtype,
                    )
                else:
                    indices = torch.arange(length, device=x.device)
                    diagonal_p = probabilities[:, :, indices, prefix + indices]
                    self_error = diagonal_p * ((ga * vt.detach().float()).sum(-1) - dot)
                    dkt = self_error.unsqueeze(-1) * q.detach().float() / math.sqrt(config.head_dim)
                    dvt = diagonal_p.unsqueeze(-1) * ga
                    error = probabilities * (_mm(ga, all_values.detach().transpose(-1, -2), spec, dtype) - dot.unsqueeze(-1))
                    error.diagonal(offset=prefix, dim1=-2, dim2=-1).zero_()
                    dq = _mm(error, all_keys.detach(), spec, dtype) / math.sqrt(config.head_dim)
                    dq += self_error.unsqueeze(-1) * kt.detach().float() / math.sqrt(config.head_dim)
                    dpk = _mm(error[:, :, :, :prefix].transpose(-1, -2), q.detach(), spec, dtype) / math.sqrt(config.head_dim)
                    dpv = _mm(probabilities[:, :, :, :prefix].transpose(-1, -2), ga, spec, dtype)
            replayed_finished = _finish(xg, attention.detach(), *weights[1:], config, dtype)
            gradients = torch.autograd.grad(
                (q, kt, vt, rotated_current, kr, vr, rotated_prefix, pvg, replayed_finished),
                (xg, *weights, pkg, pvg),
                grad_outputs=(dq, dkt, dvt, dk, grad_new_key, dv + grad_new_value,
                              dpk, dpv, gamma),
                allow_unused=True,
            )
        # The autograd engine owns accumulation, including shared multi-call and
        # distributed consumers. Frozen inputs receive no fabricated .grad.
        return tuple(g if needed else None for g, needed in zip(gradients, ctx.needs_input_grad[:7])) + (None,) * 8


def tiled_recurrent_layer(layer: OLMoBlock, x: Tensor, *, alpha: float,
                          past: tuple[Tensor, Tensor] | None,
                          query_positions: Tensor, key_positions: Tensor, key_valid: Tensor,
                          attention_precision: str = "mixed", cast_weights_once: bool = False,
                          tile_backend: str = "eager", backward_tile_backend: str = "eager",
                          backward_memory: str = "materialized", reuse_rope: bool = False,
                          kv_only_writes: bool = False, query_rope: RopeTables | None = None,
                          key_rope: RopeTables | None = None):
    if attention_precision not in ("mixed", "fp32"):
        raise ValueError("attention_precision must be 'mixed' or 'fp32'")
    if type(cast_weights_once) is not bool:
        raise TypeError("cast_weights_once must be boolean")
    if type(reuse_rope) is not bool or type(kv_only_writes) is not bool:
        raise TypeError("reuse_rope and kv_only_writes must be boolean")
    if tile_backend not in ("eager", "triton"):
        raise ValueError("tile_backend must be eager or triton")
    if tile_backend == "triton" and x.device.type != "cuda":
        raise ValueError("Triton tiles require CUDA; choose eager explicitly on CPU")
    if backward_tile_backend not in ("eager", "triton"):
        raise ValueError("backward_tile_backend must be eager or triton")
    if backward_tile_backend == "triton" and x.device.type != "cuda":
        raise ValueError("Triton backward tiles require CUDA; choose eager explicitly on CPU")
    if backward_memory not in ("materialized", "recompute"):
        raise ValueError("backward_memory must be materialized or recompute")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0,1]")
    if x.ndim != 3 or min(x.shape) == 0 or x.shape[-1] != layer.config.model_dim:
        raise ValueError("x must be nonempty [batch, length, model_dim]")
    if past is None:
        shape = (x.shape[0], layer.config.num_heads, 0, layer.config.head_dim)
        pk, pv = x.new_empty(shape), x.new_empty(shape)
    else:
        pk, pv = past
    if not reuse_rope and (query_rope is not None or key_rope is not None):
        raise ValueError("Prepared RoPE tables require reuse_rope")
    if (query_rope is None) != (key_rope is None):
        raise ValueError("Provide both query and key RoPE tables")
    if reuse_rope and query_rope is None:
        query_rope = build_rope_tables(query_positions, layer.config.head_dim, layer.config.rope_freq_constant)
        key_rope = query_rope if key_positions is query_positions else build_rope_tables(
            key_positions, layer.config.head_dim, layer.config.rope_freq_constant)
    spec = _Invocation(layer.config, float(alpha), attention_precision,
                       torch.is_autocast_enabled(x.device.type), torch.get_autocast_dtype(x.device.type),
                       cast_weights_once, tile_backend, backward_tile_backend, backward_memory, kv_only_writes)
    z, k, v = _TiledRecurrence.apply(
        x, layer.att_proj.weight, layer.attn_out.weight, layer.ff_proj.weight, layer.ff_out.weight,
        pk, pv, query_positions.clone(), key_positions.clone(), key_valid.clone(), spec,
        None if query_rope is None else query_rope.cos, None if query_rope is None else query_rope.sin,
        None if key_rope is None else key_rope.cos, None if key_rope is None else key_rope.sin,
    )
    memory = (k, v) if past is None else (torch.cat((pk, k), dim=-2), torch.cat((pv, v), dim=-2))
    return z, memory


@dataclass(frozen=True)
class OLMoTiledCache:
    key_values: tuple[tuple[Tensor, Tensor], ...]
    attention_mask: Tensor
    position_ids: Tensor
    mode: RTMode
    parameter_versions: tuple[tuple[int, int], ...] = field(repr=False)
    model_generation: int = field(repr=False)
    execution_context: RTExecutionContext
    attention_precision: str
    cast_weights_once: bool = False
    tile_backend: str = "eager"
    backward_tile_backend: str = "eager"
    backward_memory: str = "materialized"
    reuse_rope: bool = False
    kv_only_writes: bool = False
    rt_implementation: str = "native"
    author_precision: str = "author_legacy"
    author_compiled_helpers: bool = True
    author_bwd_mlp_chunks: int = 4
    author_autocast_cache: bool = True
    ordinary_attention_backend: str = "sdpa"
    ordinary_pointwise_backend: str = "eager"
    ordinary_checkpoint_layers: tuple[int, ...] | None = None

    @property
    def sequence_length(self):
        return self.attention_mask.shape[1]


@dataclass
class OLMoTiledOutput:
    logits: Tensor | None
    last_hidden_state: Tensor
    past_key_values: OLMoTiledCache | None = None


def _ordinary_block_hidden(x, query_positions, key_positions, mask, *, layer,
                           is_causal, attention_backend, query_rope=None, key_rope=None,
                           ordinary_attention_backend="sdpa", ordinary_pointwise_backend="eager"):
    """Cache-free ordinary block, with only its hidden output retained."""
    hidden, _ = layer(x, past=None, query_positions=query_positions,
                      key_positions=key_positions, mask=mask,
                      is_causal=is_causal, attention_backend=attention_backend,
                      query_rope=query_rope, key_rope=key_rope,
                      ordinary_attention_backend=ordinary_attention_backend,
                      ordinary_pointwise_backend=ordinary_pointwise_backend)
    return hidden


class OLMoTiledRTForCausalLM(OLMoRTForCausalLM):
    """Tiled RT with optional ordinary-block activation checkpointing.

    The opt-in flag is execution metadata, not part of the native checkpoint
    layout. Callers must record it in their run/resume configuration. During
    grad-enabled training it checkpoints only ordinary block calls; selected
    RT blocks retain their existing input/completed-output reconstruction.
    Checkpointed training is cache-free. Evaluation and no-grad calls use the
    unchanged ordinary/cache path.

    ``rt_implementation="author"`` opts selected RT blocks into the author's
    writer-VJP schedule. It requires reused FP32 RoPE tables, full-strength
    recurrence and unpadded, cache-free calls. The separate author precision,
    compilation, MLP-chunk and autocast-cache options describe that backend;
    native tile/backward/attention-precision options do not change its math.
    Ordinary blocks, including FBT's bootstrap, use separate opt-in attention
    and pointwise backends. ``ordinary_checkpoint_layers=None`` preserves the
    historical all/none boolean policy; a tuple selects ordinary block indices
    when that boolean is enabled. Selected RT blocks never use that checkpoint.
    """
    def __init__(self, config, *, attention_backend="sdpa", attention_precision="mixed",
                 ordinary_activation_checkpointing=False, cast_weights_once=False,
                 tile_backend="eager", backward_tile_backend="eager", backward_memory="materialized",
                 reuse_rope=False, kv_only_writes=False,
                 rt_implementation="native", author_precision="author_legacy",
                 author_compiled_helpers=True, author_bwd_mlp_chunks=4,
                 author_autocast_cache=True,
                 ordinary_attention_backend="sdpa", ordinary_pointwise_backend="eager",
                 ordinary_checkpoint_layers=None,
                 device=None, dtype=None):
        if attention_precision not in ("mixed", "fp32"):
            raise ValueError("attention_precision must be 'mixed' or 'fp32'")
        if type(ordinary_activation_checkpointing) is not bool:
            raise TypeError("ordinary_activation_checkpointing must be boolean")
        validate_ordinary_options(ordinary_attention_backend, ordinary_pointwise_backend,
                                  sdpa_backend=attention_backend)
        validate_checkpoint_layers(ordinary_checkpoint_layers, num_layers=config.num_layers,
                                   enabled=ordinary_activation_checkpointing)
        if type(cast_weights_once) is not bool:
            raise TypeError("cast_weights_once must be boolean")
        if type(reuse_rope) is not bool or type(kv_only_writes) is not bool:
            raise TypeError("reuse_rope and kv_only_writes must be boolean")
        if tile_backend not in ("eager", "triton"):
            raise ValueError("tile_backend must be eager or triton")
        if backward_tile_backend not in ("eager", "triton"):
            raise ValueError("backward_tile_backend must be eager or triton")
        if backward_memory not in ("materialized", "recompute"):
            raise ValueError("backward_memory must be materialized or recompute")
        if rt_implementation not in ("native", "author"):
            raise ValueError("rt_implementation must be native or author")
        if author_precision not in ("author_legacy", "fp32_state"):
            raise ValueError("author_precision must be author_legacy or fp32_state")
        if type(author_compiled_helpers) is not bool or type(author_autocast_cache) is not bool:
            raise TypeError("author_compiled_helpers and author_autocast_cache must be boolean")
        if type(author_bwd_mlp_chunks) is not int or author_bwd_mlp_chunks < 1:
            raise ValueError("author_bwd_mlp_chunks must be a positive integer")
        if rt_implementation == "author" and not reuse_rope:
            raise ValueError("Author RT requires explicit reuse_rope=True")
        super().__init__(config, attention_backend=attention_backend, device=device, dtype=dtype)
        self.attention_precision = attention_precision
        self.ordinary_activation_checkpointing = ordinary_activation_checkpointing
        self.ordinary_attention_backend = ordinary_attention_backend
        self.ordinary_pointwise_backend = ordinary_pointwise_backend
        self.ordinary_checkpoint_layers = ordinary_checkpoint_layers
        self.cast_weights_once = cast_weights_once
        self.tile_backend = tile_backend
        self.backward_tile_backend = backward_tile_backend
        self.backward_memory = backward_memory
        self.reuse_rope = reuse_rope
        self.kv_only_writes = kv_only_writes
        self.rt_implementation = rt_implementation
        self.author_precision = author_precision
        self.author_compiled_helpers = author_compiled_helpers
        self.author_bwd_mlp_chunks = author_bwd_mlp_chunks
        self.author_autocast_cache = author_autocast_cache

    def _validate_ordinary_scope(self, mode, *, attention_mask, is_causal,
                                 past_key_values=None, use_cache=False):
        validate_ordinary_options(self.ordinary_attention_backend, self.ordinary_pointwise_backend,
                                  sdpa_backend=self.attention_backend)
        validate_checkpoint_layers(self.ordinary_checkpoint_layers, num_layers=self.config.num_layers,
                                   enabled=self.ordinary_activation_checkpointing)
        has_ordinary = len(mode.selected_layers) < self.config.num_layers
        if has_ordinary and self.ordinary_attention_backend == "fa4" and (
                attention_mask is not None or not is_causal or past_key_values is not None or use_cache):
            raise ValueError("Ordinary FA4 supports only dense causal full sequences without masks or prefix/exported caches")

    def _checkpoint_ordinary_enabled(self):
        return (self.ordinary_activation_checkpointing and self.ordinary_checkpoint_layers != ()
                and self.training and torch.is_grad_enabled())

    def _validate_author_scope(self, mode, *, all_tokens_valid,
                               past_key_values=None, use_cache=False):
        """Check fixed author scope without reading any device tensor.

        Public callers prove validity outside capture. Prepared layouts supply
        their owned CPU-derived flag and validate it before graph preparation.
        An empty selection remains the ordinary implementation in either mode.
        """
        if self.rt_implementation not in ("native", "author"):
            raise ValueError("rt_implementation must be native or author")
        if self.rt_implementation != "author" or not mode.selected_layers:
            return
        if not self.reuse_rope:
            raise ValueError("Author RT requires explicit reuse_rope=True")
        if mode.alpha != 1.0:
            raise ValueError("Author RT supports only full-strength alpha=1 recurrence")
        if past_key_values is not None or use_cache:
            raise ValueError("Author RT does not support a prefix or exported cache")
        if type(all_tokens_valid) is not bool or not all_tokens_valid:
            raise ValueError("Author RT requires a validated unpadded layout (all_tokens_valid=True)")
        if self.author_compiled_helpers and self.readout_weight.device.type != "cuda":
            raise ValueError("Compiled author helpers require CUDA; explicitly disable them for CPU checks")
        if any(weight.dtype != torch.float32 for index in mode.selected_layers
               for weight in self.layers[index].parameters()):
            raise ValueError("Author RT requires unchanged native FP32 projection weights")
        device_type = self.readout_weight.device.type
        if torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type) != torch.bfloat16:
            raise ValueError("Author RT supports FP32 or BF16 mixed precision only")

    def forward(self, input_ids=None, *, mode=RTMode(), inputs_embeds=None,
                attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, return_logits=True):
        if not isinstance(mode, RTMode):
            raise TypeError("mode must be an RTMode")
        if any(index >= self.config.num_layers for index in mode.selected_layers):
            raise ValueError("selected_layers contains an index outside the model")
        checkpoint_ordinary = self._checkpoint_ordinary_enabled()
        if checkpoint_ordinary and (use_cache or past_key_values is not None):
            raise ValueError("Ordinary activation checkpointing requires cache-free grad-enabled training")
        x, valid, positions, key_positions, mask, causal = self._prepare_inputs(
            input_ids, inputs_embeds, attention_mask, position_ids, past_key_values,
            cache_type=OLMoTiledCache,
        )
        # This dynamic validation boundary is outside prepared graph bodies.
        # The common mask-free case needs no device reduction/synchronization.
        all_tokens_valid = None
        if self.rt_implementation == "author" and mode.selected_layers:
            all_tokens_valid = True if attention_mask is None and past_key_values is None else bool(valid.all())
            self._validate_author_scope(mode, all_tokens_valid=all_tokens_valid,
                                        past_key_values=past_key_values, use_cache=use_cache)
        versions = self._parameter_versions() if use_cache or past_key_values is not None else ()
        generation = getattr(self, "_rt_cache_generation", 0)
        context = self._execution_context(x.device)
        if past_key_values is not None:
            if past_key_values.mode != mode:
                raise ValueError("Cached RT mode differs from the requested mode")
            if past_key_values.parameter_versions != versions:
                raise ValueError("Cached weights differ from the current model or were modified")
            if past_key_values.model_generation != generation:
                raise ValueError("Cached model generation changed after a model conversion")
            if past_key_values.execution_context != context or past_key_values.attention_precision != self.attention_precision:
                raise ValueError("Cached execution context or attention precision differs")
            if (past_key_values.cast_weights_once != self.cast_weights_once
                    or past_key_values.tile_backend != self.tile_backend
                    or past_key_values.backward_tile_backend != self.backward_tile_backend
                    or past_key_values.backward_memory != self.backward_memory
                    or past_key_values.reuse_rope != self.reuse_rope
                    or past_key_values.kv_only_writes != self.kv_only_writes
                    or past_key_values.rt_implementation != self.rt_implementation
                    or past_key_values.author_precision != self.author_precision
                    or past_key_values.author_compiled_helpers != self.author_compiled_helpers
                    or past_key_values.author_bwd_mlp_chunks != self.author_bwd_mlp_chunks
                    or past_key_values.author_autocast_cache != self.author_autocast_cache):
                raise ValueError("Cached RT kernel execution differs")
            if (past_key_values.ordinary_attention_backend != self.ordinary_attention_backend
                    or past_key_values.ordinary_pointwise_backend != self.ordinary_pointwise_backend
                    or past_key_values.ordinary_checkpoint_layers != self.ordinary_checkpoint_layers):
                raise ValueError("Cached ordinary execution differs")
        hidden, present = self._forward_prepared(
            x, mode=mode, positions=positions, key_positions=key_positions,
            key_valid=valid, attention_mask=mask, is_causal=causal,
            past_key_values=past_key_values, use_cache=use_cache,
            checkpoint_ordinary=checkpoint_ordinary,
            all_tokens_valid=all_tokens_valid,
        )
        cache = OLMoTiledCache(tuple(present), valid.clone(), key_positions.clone(), mode,
                              versions, generation, context, self.attention_precision,
                              self.cast_weights_once, self.tile_backend,
                              self.backward_tile_backend, self.backward_memory,
                              self.reuse_rope, self.kv_only_writes, self.rt_implementation,
                              self.author_precision, self.author_compiled_helpers,
                              self.author_bwd_mlp_chunks, self.author_autocast_cache,
                              self.ordinary_attention_backend, self.ordinary_pointwise_backend,
                              self.ordinary_checkpoint_layers) if use_cache else None
        return OLMoTiledOutput(self.project_logits(hidden) if return_logits else None, hidden, cache)

    def _forward_prepared(self, x, *, mode, positions, key_positions, key_valid,
                          attention_mask, is_causal, past_key_values=None,
                          use_cache=False, checkpoint_ordinary=False,
                          query_rope=None, key_rope=None, all_tokens_valid=None):
        """Native layer loop after caller validation; static entries are cache-free.

        The public forward retains all input/cache guards and cache construction.
        Prepared static callers validate their owned fixed layout before capture
        and between replays. This helper changes no layer or checkpoint math.
        """
        self._validate_author_scope(mode, all_tokens_valid=all_tokens_valid,
                                    past_key_values=past_key_values, use_cache=use_cache)
        self._validate_ordinary_scope(mode, attention_mask=attention_mask, is_causal=is_causal,
                                      past_key_values=past_key_values, use_cache=use_cache)
        if not self.reuse_rope and (query_rope is not None or key_rope is not None):
            raise ValueError("Prepared RoPE tables require reuse_rope")
        if (query_rope is None) != (key_rope is None):
            raise ValueError("Provide both query and key RoPE tables")
        if self.reuse_rope and query_rope is None:
            query_rope = build_rope_tables(positions, self.config.head_dim, self.config.rope_freq_constant)
            key_rope = query_rope if key_positions is positions else build_rope_tables(
                key_positions, self.config.head_dim, self.config.rope_freq_constant)
        present = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values.key_values[index]
            if index in mode.selected_layers and self.rt_implementation == "author":
                # Import only when this opt-in backend actually executes. It
                # owns no model parameters and returns gradients to these exact
                # existing packed weights through explicit Function inputs.
                from .olmo_author import author_tiled_recurrent_layer
                x = author_tiled_recurrent_layer(layer, x, query_rope,
                    precision_policy=self.author_precision,
                    compiled_helpers=self.author_compiled_helpers,
                    bwd_mlp_chunks=self.author_bwd_mlp_chunks,
                    autocast_cache=self.author_autocast_cache)
                pair = None  # Author scope above forbids any exported cache.
            elif index in mode.selected_layers:
                x, pair = tiled_recurrent_layer(
                    layer, x, alpha=mode.alpha, past=past, query_positions=positions,
                    key_positions=key_positions, key_valid=key_valid,
                    attention_precision=self.attention_precision,
                    cast_weights_once=self.cast_weights_once, tile_backend=self.tile_backend,
                    backward_tile_backend=self.backward_tile_backend, backward_memory=self.backward_memory,
                    reuse_rope=self.reuse_rope, kv_only_writes=self.kv_only_writes,
                    query_rope=query_rope, key_rope=key_rope,
                )
            elif checkpoint_ordinary and (self.ordinary_checkpoint_layers is None
                                           or index in self.ordinary_checkpoint_layers):
                # Bind this invocation's layer/backend/causality now: a closure
                # over the loop's `layer` would replay the wrong shared block.
                ordinary = partial(_ordinary_block_hidden, layer=layer,
                                   is_causal=is_causal, attention_backend=self.attention_backend,
                                   query_rope=query_rope, key_rope=key_rope,
                                   ordinary_attention_backend=self.ordinary_attention_backend,
                                   ordinary_pointwise_backend=self.ordinary_pointwise_backend)
                x = checkpoint(ordinary, x, positions, key_positions, attention_mask,
                               use_reentrant=False)
                pair = None
            else:
                x, pair = layer(x, past=past, query_positions=positions, key_positions=key_positions,
                                mask=attention_mask, is_causal=is_causal, attention_backend=self.attention_backend,
                                query_rope=query_rope, key_rope=key_rope,
                                ordinary_attention_backend=self.ordinary_attention_backend,
                                ordinary_pointwise_backend=self.ordinary_pointwise_backend)
            if use_cache:
                present.append(pair)
        hidden = self.norm(x)
        return hidden, tuple(present)
