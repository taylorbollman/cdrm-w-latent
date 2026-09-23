"""Author-derived recurrent tiling adapted to native OLMo weights and RoPE.

Adapted from geniucos/recurrent-transformer ``olmo/model.py`` (Apache-2.0),
upstream a21b42d2bc292edb86ed1b62cee4bcab809a9d21 and the project's repaired
fork 824767f9a6f8e29959a0c486d169ce10b7194d41. Substantive adaptations: functional
native pre-LN/SwiGLU weights, explicit FP32 RoPE, private parameter-gradient
ownership and fixed unpadded/full-strength/no-prefix scope. The position-major
dyadic schedule, full backward attention matrix, per-token writer weight VJPs,
chunked MLP-input VJPs and batched final parameter VJPs follow the author code.

``author_legacy`` deliberately retains the author's low-precision arithmetic:
tiled Q is prescaled, diagonal products and materialized probabilities use its
projection dtype. The optional ``fp32_state`` policy reproduces the local fork's
diagnostic promotion of attention and adjoints; it is not the author baseline.
Neither path calls Flash Attention. Higher derivatives are unsupported.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .olmo import OLMoBlock
from .olmo_rope import RopeTables, apply_rope_tables


AUTHOR_SOURCE = {
    "upstream_commit": "a21b42d2bc292edb86ed1b62cee4bcab809a9d21",
    "upstream_model_sha256": "8b552064f233b93f9ec348d93a55b972aac7b62a191460353de6fbdbd7bd3acb",
    "fork_commit": "824767f9a6f8e29959a0c486d169ce10b7194d41",
    "fork_model_sha256": "dfc37f10c8da1480265e8b90c9245a5120c8e596d6550057d9769c588eb2c9f5",
    "license": "Apache-2.0",
    "path": "recurrent-transformer/olmo/model.py",
}
COMPILED_HELPER_BOUNDARIES = {
    "pre_attention": {"dynamic": False},
    "block_attention_add": {"dynamic": False},
    "recompute_alphas": {"fullgraph": True},
    "recompute_atts": {"fullgraph": True},
    "mlp_batched_body": {"fullgraph": True},
}


@dataclass(frozen=True)
class _Spec:
    width: int
    heads: int
    eps: float
    compiled_helpers: bool
    bwd_mlp_chunks: int
    fp32_state: bool
    autocast_enabled: bool
    autocast_dtype: torch.dtype
    autocast_cache: bool


def _validate(layer, x, tables, precision_policy, compiled_helpers, bwd_mlp_chunks, autocast_cache):
    if not isinstance(layer, OLMoBlock):
        raise TypeError("Author adapter requires a native OLMoBlock")
    if precision_policy not in ("author_legacy", "fp32_state"):
        raise ValueError("precision_policy must be author_legacy or fp32_state")
    if type(compiled_helpers) is not bool or type(autocast_cache) is not bool:
        raise TypeError("compiled_helpers and autocast_cache must be boolean")
    if type(bwd_mlp_chunks) is not int or bwd_mlp_chunks < 1:
        raise ValueError("bwd_mlp_chunks must be a positive integer")
    if x.ndim != 3 or min(x.shape) < 1 or x.shape[-1] != layer.config.model_dim:
        raise ValueError("Author input must be nonempty [batch, length, native width]")
    if x.device.type not in ("cpu", "cuda") or x.dtype != torch.float32:
        raise ValueError("Author adapter requires a CPU/CUDA FP32 residual stream")
    if compiled_helpers and x.device.type != "cuda":
        raise ValueError("Compiled author helpers require CUDA; explicitly disable them for CPU checks")
    if not isinstance(tables, RopeTables):
        raise TypeError("Explicit native RopeTables are required")
    if (tables.cos.shape != (x.shape[0], 1, x.shape[1], layer.config.head_dim)
            or tables.cos.device != x.device):
        raise ValueError("RoPE table shape/device differs from the author input")
    weights = (layer.att_proj.weight, layer.attn_out.weight, layer.ff_proj.weight, layer.ff_out.weight)
    expected = ((3 * layer.config.model_dim, layer.config.model_dim),
                (layer.config.model_dim, layer.config.model_dim),
                (2 * layer.config.mlp_intermediate_size, layer.config.model_dim),
                (layer.config.model_dim, layer.config.mlp_intermediate_size))
    if any(w.shape != shape or w.device != x.device or w.dtype != torch.float32 for w, shape in zip(weights, expected)):
        raise ValueError("Author adapter requires unchanged native FP32 projection weights on the input device")
    if any(module.bias is not None for module in (layer.att_proj, layer.attn_out, layer.ff_proj, layer.ff_out)):
        raise ValueError("Author native mapping does not support projection biases")
    if any(module.weight is not None or module.bias is not None
           or module.eps != layer.config.layer_norm_eps for module in (layer.attn_norm, layer.ff_norm)):
        raise ValueError("Author native mapping requires native nonaffine LayerNorm")
    enabled = torch.is_autocast_enabled(x.device.type)
    dtype = torch.get_autocast_dtype(x.device.type)
    if enabled and dtype != torch.bfloat16:
        raise ValueError("Author adapter supports FP32 or BF16 mixed precision only")
    return _Spec(layer.config.model_dim, layer.config.num_heads, layer.config.layer_norm_eps,
        compiled_helpers, bwd_mlp_chunks, precision_policy == "fp32_state", enabled, dtype, autocast_cache)


def _dense_scope(spec, device):
    return torch.autocast(device.type, enabled=spec.autocast_enabled,
                         dtype=spec.autocast_dtype, cache_enabled=spec.autocast_cache)


def _pre_attention(x, wq, wkv, heads, eps):
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    # Preserve the author's separate KV-then-Q projections.
    kv, q = F.linear(normalized, wkv), F.linear(normalized, wq)
    k, v = kv.chunk(2, dim=-1)
    return tuple(t.view(x.shape[0], x.shape[1], heads, x.shape[-1] // heads).transpose(1, 2)
                 for t in (q, k, v))


def _writer(x, wkv, heads, eps, tables):
    k, v = F.linear(F.layer_norm(x, (x.shape[-1],), eps=eps), wkv).chunk(2, dim=-1)
    k, v = (t.view(x.shape[0], x.shape[1], heads, x.shape[-1] // heads).transpose(1, 2) for t in (k, v))
    return apply_rope_tables(k, tables), v


def _finish(x, att, wo, wup, wdown, eps):
    residual = x + F.linear(att, wo)
    up, gate = F.linear(F.layer_norm(residual, (x.shape[-1],), eps=eps), wup).chunk(2, dim=-1)
    return residual + F.linear(F.silu(gate) * up, wdown)


def author_recurrent_reference(layer: OLMoBlock, x: Tensor, rope_tables: RopeTables, *,
                               precision_policy="author_legacy", autocast_cache=True) -> Tensor:
    """Ordinary-autograd scan; legacy scan scales after its dot, as upstream.

    All positions are valid, full-strength recurrence is fixed, no prefix or
    exported cache is supported. The tiled legacy implementation instead
    prescales Q: that original mixed-precision distinction is intentional.
    """
    spec = _validate(layer, x, rope_tables, precision_policy, False, 1, autocast_cache)
    wq, wkv = layer.att_proj.weight[:spec.width], layer.att_proj.weight[spec.width:]
    wo, wup, wdown = layer.attn_out.weight, layer.ff_proj.weight, layer.ff_out.weight
    with _dense_scope(spec, x.device):
        # Packed views are not autocast-cacheable leaves. Explicit once-per-call
        # graph-connected casts retain ordinary autograd ownership in the scan.
        if spec.autocast_enabled and spec.autocast_cache:
            wq, wkv, wo, wup, wdown = (w.to(spec.autocast_dtype) for w in (wq, wkv, wo, wup, wdown))
        q, k, v = _pre_attention(x, wq, wkv, spec.heads, spec.eps)
        q, k = apply_rope_tables(q, rope_tables), apply_rope_tables(k, rope_tables)
        dtype = q.dtype
        q_math = q.float() * (1.0 / math.sqrt(q.shape[-1])) if spec.fp32_state else q
        k_math, v_math = (k.float(), v.float()) if spec.fp32_state else (k, v)
        outputs, keys, values = [], [], []
        for t in range(x.shape[1]):
            big_k = torch.cat([*keys, k_math[:, :, t:t + 1]], dim=-2)
            big_v = torch.cat([*values, v_math[:, :, t:t + 1]], dim=-2)
            with torch.autocast(x.device.type, enabled=False) if spec.fp32_state else nullcontext():
                logits = big_k @ q_math[:, :, t:t + 1].transpose(-2, -1)
                if not spec.fp32_state:
                    logits = logits / math.sqrt(q.shape[-1])
                probabilities = torch.softmax(logits.transpose(-2, -1), dim=-1)
                attention = probabilities @ big_v
            attention = attention.to(dtype).transpose(1, 2).contiguous().view(x.shape[0], 1, spec.width)
            completed = _finish(x[:, t:t + 1], attention, wo, wup, wdown, spec.eps)
            outputs.append(completed)
            kt, vt = _writer(completed, wkv, spec.heads, spec.eps, rope_tables.slice(t, t + 1))
            keys.append(kt.float() if spec.fp32_state else kt)
            values.append(vt.float() if spec.fp32_state else vt)
    return torch.cat(outputs, dim=1)


def _block_attention_add(att0, maximum0, total0, k, v, q, fp32_state):
    with torch.autocast(q.device.type, enabled=False) if fp32_state else nullcontext():
        if fp32_state:
            q, k, v = q.float(), k.float(), v.float()
            att0, maximum0, total0 = att0.float(), maximum0.float(), total0.float()
        weights = q @ k.transpose(-2, -1)
        maximum = torch.maximum(maximum0.permute(1, 2, 0), weights.max(dim=-1).values)
        weights = torch.exp(weights - maximum.unsqueeze(-1))
        att = weights @ v
        maximum = maximum.permute(2, 0, 1)
        rescale = torch.exp(maximum0 - maximum)
        return (att0 * rescale.unsqueeze(-1) + att.permute(2, 0, 1, 3),
                maximum, total0 * rescale + weights.sum(dim=-1).permute(2, 0, 1))


def _recompute_alphas(final_k, k_init, q, fp32_state):
    with torch.autocast(q.device.type, enabled=False) if fp32_state else nullcontext():
        if fp32_state:
            final_k, k_init, q = final_k.float(), k_init.float(), q.float()
        alphas = final_k.permute(1, 2, 0, 3) @ q.permute(1, 2, 3, 0)
        alphas.diagonal(dim1=2, dim2=3).copy_((q * k_init).sum(-1).permute(1, 2, 0))
        lower = torch.ones((alphas.shape[-1], alphas.shape[-1]), device=q.device, dtype=torch.bool).tril(-1)
        alphas.masked_fill_(lower, -torch.inf)
        return torch.softmax(alphas, dim=-2, dtype=torch.float32).to(q.dtype)


def _recompute_atts(final_v, v_init, alphas, fp32_state):
    with torch.autocast(alphas.device.type, enabled=False) if fp32_state else nullcontext():
        if fp32_state:
            final_v, v_init, alphas = final_v.float(), v_init.float(), alphas.float()
        atts = (final_v.permute(1, 2, 3, 0) @ alphas).permute(3, 0, 1, 2)
        return atts + (v_init - final_v) * alphas.diagonal(dim1=2, dim2=3).permute(2, 0, 1).unsqueeze(-1)


def _mlp_batched_body(atts, x, wo, wup, wdown, eps, dtype, autocast_enabled, autocast_cache, fp32_state):
    with torch.autocast(x.device.type, enabled=autocast_enabled, dtype=dtype, cache_enabled=autocast_cache):
        atts = atts.transpose(0, 1).reshape(x.shape)
        if fp32_state:
            atts = atts.to(dtype if autocast_enabled else x.dtype)
        return _finish(x, atts, wo, wup, wdown, eps)


# These are the author's existing compilation boundaries, with Tensor operands
# and immutable scalars instead of mutable module proxies. Compilation is lazy.
_COMPILED = {
    "pre_attention": torch.compile(_pre_attention, dynamic=False),
    "block_attention_add": torch.compile(_block_attention_add, dynamic=False),
    "recompute_alphas": torch.compile(_recompute_alphas, fullgraph=True),
    "recompute_atts": torch.compile(_recompute_atts, fullgraph=True),
    "mlp_batched_body": torch.compile(_mlp_batched_body, fullgraph=True),
}
_EAGER = {"pre_attention": _pre_attention, "block_attention_add": _block_attention_add,
          "recompute_alphas": _recompute_alphas, "recompute_atts": _recompute_atts,
          "mlp_batched_body": _mlp_batched_body}


def _helper(spec, name, *args):
    return (_COMPILED if spec.compiled_helpers else _EAGER)[name](*args)


def _private_weights(packed, wo, wup, wdown, width):
    # Independent TensorImpl leaves on the original storage: no parameter copies
    # or registrations. Fresh leaves per phase prevent cached no-grad forward
    # casts from being reused for graph-connected backward reconstruction.
    return tuple(weight.detach().requires_grad_(True) for weight in
                 (packed[:width], packed[width:], wo, wup, wdown))


def _position_major(value):
    return value.permute(2, 0, 1, 3)


class _AuthorTiled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, wo, wup, wdown, cos, sin, spec):
        tables = RopeTables(cos, sin)
        batch, length, width = x.shape
        weights = _private_weights(packed, wo, wup, wdown, width)
        wq_local, wkv_local, wo_local, wup_local, wdown_local = weights
        with _dense_scope(spec, x.device):
            q, kt, vt = _helper(spec, "pre_attention", x, wq_local, wkv_local, spec.heads, spec.eps)
            q, kt = apply_rope_tables(q, tables), apply_rope_tables(kt, tables)
            projection_dtype = q.dtype
            q = _position_major(q.float() if spec.fp32_state else q).contiguous() * (1.0 / math.sqrt(q.shape[-1]))
            kt = _position_major(kt)
            atts = _position_major(vt).to(x.dtype).contiguous()
            with torch.autocast(x.device.type, enabled=False) if spec.fp32_state else nullcontext():
                maximum = ((kt.float() if spec.fp32_state else kt) * q).sum(-1)
            total = torch.ones((length, batch, spec.heads), dtype=x.dtype, device=x.device)
            del kt, vt
            final_k = torch.empty((length, batch, spec.heads, width // spec.heads), dtype=projection_dtype, device=x.device)
            final_v = torch.empty_like(final_k)
            outputs = []
            for t in range(length):
                with torch.autocast(x.device.type, enabled=False) if spec.fp32_state else nullcontext():
                    att = (atts[t:t + 1] / total[t:t + 1].unsqueeze(-1)).to(projection_dtype)
                att = att.transpose(0, 1).reshape(batch, 1, width)
                completed = _finish(x[:, t:t + 1], att, wo_local, wup_local, wdown_local, spec.eps)
                outputs.append(completed)
                k, v = _writer(completed, wkv_local, spec.heads, spec.eps, tables.slice(t, t + 1))
                final_k[t:t + 1], final_v[t:t + 1] = _position_major(k), _position_major(v)
                boundary = t + 1
                if boundary == length:
                    continue
                tile = boundary & -boundary
                target = slice(boundary, boundary + tile)
                source = slice(boundary - tile, boundary)
                atts[target], maximum[target], total[target] = _helper(spec, "block_attention_add",
                    atts[target], maximum[target], total[target], final_k[source].permute(1, 2, 0, 3),
                    final_v[source].permute(1, 2, 0, 3), q[target].permute(1, 2, 0, 3), spec.fp32_state)
        output = torch.cat(outputs, dim=1)
        ctx.save_for_backward(x, output, packed, wo, wup, wdown, cos, sin)
        ctx.spec, ctx.projection_dtype = spec, projection_dtype
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 8
        with torch.autocast(grad_output.device.type, enabled=False):
            return _author_backward(ctx, grad_output)


def _author_backward(ctx, grad_output):
    x_saved, outs, packed, wo, wup, wdown, cos, sin = ctx.saved_tensors
    spec, projection_dtype = ctx.spec, ctx.projection_dtype
    tables = RopeTables(cos, sin)
    batch, length, width = x_saved.shape
    x = x_saved.detach().requires_grad_(True)
    wq, wkv, wo, wup, wdown = _private_weights(packed, wo, wup, wdown, width)
    with _dense_scope(spec, x.device), torch.enable_grad():
        q, kt, vt = _helper(spec, "pre_attention", x, wq, wkv, spec.heads, spec.eps)
        q, kt = apply_rope_tables(q, tables), apply_rope_tables(kt, tables)
        if spec.fp32_state:
            q, kt, vt = q.float(), kt.float(), vt.float()
        q = _position_major(q).contiguous() * (1.0 / math.sqrt(q.shape[-1]))
        kt, vt = _position_major(kt).contiguous(), _position_major(vt).contiguous()
    final_ks, final_vs, out_leaves = [], [], []
    # Keep all writer weight VJPs separate, as in the author implementation.
    with _dense_scope(spec, x.device), torch.enable_grad():
        for t in range(length):
            leaf = outs[:, t:t + 1].detach().requires_grad_(True)
            out_leaves.append(leaf)
            k, v = _writer(leaf, wkv, spec.heads, spec.eps, tables.slice(t, t + 1))
            k, v = _position_major(k), _position_major(v)
            final_ks.append(k.float() if spec.fp32_state else k)
            final_vs.append(v.float() if spec.fp32_state else v)
    final_k = torch.cat([k.detach() for k in final_ks], dim=0)
    final_v = torch.cat([v.detach() for v in final_vs], dim=0)
    alphas = _helper(spec, "recompute_alphas", final_k, kt, q, spec.fp32_state)
    atts = _helper(spec, "recompute_atts", final_v, vt, alphas, spec.fp32_state)
    k_grads = torch.zeros_like(final_k, dtype=x.dtype)
    v_grads = torch.zeros_like(final_v, dtype=x.dtype)
    gs = torch.empty_like(q)
    g_dot_atts = torch.empty(q.shape[:-1], dtype=q.dtype, device=q.device)
    final_v = final_v.permute(1, 2, 0, 3).contiguous()
    all_grads, mlp_outputs, att_leaves = [], {}, {}
    segment = (length + spec.bwd_mlp_chunks - 1) // spec.bwd_mlp_chunks
    for t in range(length - 1, -1, -1):
        torch.autograd.backward((final_ks[t], final_vs[t]), (k_grads[t:t + 1], v_grads[t:t + 1]))
        if t not in mlp_outputs:
            mlp_outputs.clear()
            att_leaves.clear()
            with _dense_scope(spec, x.device), torch.enable_grad():
                for j in range(t, max(-1, t - segment), -1):
                    att_leaves[j] = atts[j:j + 1].detach().requires_grad_(True)
                    attention = att_leaves[j].transpose(0, 1).reshape(batch, 1, width)
                    if spec.fp32_state:
                        attention = attention.to(projection_dtype)
                    mlp_outputs[j] = _finish(x[:, j:j + 1].detach(), attention, wo, wup, wdown, spec.eps)
        propagated = grad_output[:, t:t + 1] + out_leaves[t].grad
        g = torch.autograd.grad(mlp_outputs[t], (att_leaves[t],), grad_outputs=propagated)[0]
        all_grads.append(propagated)
        g_dot_atts[t:t + 1] = (att_leaves[t] * g).sum(-1)
        gs[t:t + 1] = g
        if t == 0:
            continue
        tile = t & -t
        target, source = slice(t - tile, t), slice(t, t + tile)
        if tile > 1:
            v_grads[target] += (alphas[:, :, target, source] @ gs[source].permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
            alphas[:, :, target, source] *= (final_v[:, :, target] @ gs[source].permute(1, 2, 3, 0)
                - g_dot_atts[source].permute(1, 2, 0).unsqueeze(-2))
            k_grads[target] += (alphas[:, :, target, source] @ q[source].permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
        else:
            v_grads[t - 1] += alphas[:, :, t - 1, t:t + 1] * gs[t]
            alphas[:, :, t - 1, t] *= (final_v[:, :, t - 1] * gs[t]).sum(-1) - g_dot_atts[t]
            k_grads[t - 1] += alphas[:, :, t - 1, t:t + 1] * q[t]
    del final_ks, final_vs, out_leaves, mlp_outputs, att_leaves, final_v, k_grads, v_grads
    v_init_grad = alphas.diagonal(dim1=2, dim2=3).unsqueeze(-1).permute(2, 0, 1, 3) * gs
    alpha_self = alphas.diagonal(dim1=2, dim2=3).permute(2, 0, 1) * ((vt * gs).sum(-1) - g_dot_atts)
    k_init_grad = alpha_self.unsqueeze(-1) * q
    q_grad = alpha_self.unsqueeze(-1) * kt
    del gs
    alphas.diagonal(dim1=2, dim2=3).zero_()
    q_grad += (alphas.permute(0, 1, 3, 2) @ final_k.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
    # Preserve the author's release points: the quadratic probabilities and
    # full-sequence Q/K/V buffers must not overlap the final batched MLP replay.
    del alphas, final_k
    torch.autograd.backward((q, kt, vt), (q_grad, k_init_grad, v_init_grad))
    del q_grad, k_init_grad, v_init_grad, q, kt, vt
    big_grads = torch.cat(list(reversed(all_grads)), dim=1)
    del all_grads
    with torch.enable_grad():
        mlp_output = _helper(spec, "mlp_batched_body", atts, x, wo, wup, wdown, spec.eps,
            spec.autocast_dtype, spec.autocast_enabled, spec.autocast_cache, spec.fp32_state)
    torch.autograd.backward(mlp_output, big_grads)
    gradients = (x.grad, torch.cat((wq.grad, wkv.grad), dim=0), wo.grad, wup.grad, wdown.grad)
    return tuple(value if needed else None for value, needed in zip(gradients, ctx.needs_input_grad[:5])) + (None,) * 3


def author_tiled_recurrent_layer(layer: OLMoBlock, x: Tensor, rope_tables: RopeTables, *,
        compiled_helpers=True, bwd_mlp_chunks=4, precision_policy="author_legacy", autocast_cache=True) -> Tensor:
    """Author dyadic RT with explicit gradients to unchanged native parameters.

    Only invocation-local detached leaves receive nested ``.grad`` updates.
    Original parameters and ``x`` receive their gradients solely through the
    outer autograd engine, including shared, frozen and ``autograd.grad`` use.
    """
    spec = _validate(layer, x, rope_tables, precision_policy, compiled_helpers, bwd_mlp_chunks, autocast_cache)
    return _AuthorTiled.apply(x, layer.att_proj.weight, layer.attn_out.weight, layer.ff_proj.weight,
        layer.ff_out.weight, rope_tables.cos, rope_tables.sin, spec)
