"""FP32 permanent-value injection using the frozen RT tiled recurrence.

The caller supplies the already scaled, graph-connected ``value_bypass``.
Only permanent writes receive it; provisional self K/V and contextual keys keep
their original projections. No parameters, global hooks, or global counters are
introduced here. Projection ownership belongs to the calling backbone.
"""
from __future__ import annotations

import copy

import torch
from torch.autograd.function import once_differentiable

from olmo.model import (
    BufferCache, OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled,
    OLMoRecurrentBlockTiledFunction,
)


def _validate(block, x, bypass, attention_bias):
    if x.ndim != 3 or x.shape[1] < 1 or bypass.shape != x.shape:
        raise ValueError("Value bypass must match nonempty [batch, length, width] inputs")
    if x.dtype != torch.float32 or bypass.dtype != torch.float32 or bypass.device != x.device:
        raise ValueError("This bounded value-bypass pilot requires colocated FP32 tensors")
    if torch.is_autocast_enabled(x.device.type):
        raise ValueError("The value-bypass pilot does not support autocast")
    cfg = block.config
    if (not cfg.reference_eager or cfg.norm_after or cfg.clip_qkv is not None
            or cfg.recurrent_write_rho != 1.0 or cfg.rope
            or cfg.effective_n_kv_heads != cfg.n_heads
            or cfg.attention_dropout != 0 or cfg.residual_dropout != 0
            or cfg.bwd_mlp_chunks < 1):
        raise ValueError("Require the original eager pre-norm, unclipped, attached FP32 RT recipe")
    if block.pre_attention_block.kv_proj is not block.kv_proj:
        raise ValueError("The provisional KV view must retain the original owned projection")
    if any(p.dtype != torch.float32 for p in block.parameters()):
        raise ValueError("Value-bypass block parameters must be FP32")
    if attention_bias is not None and attention_bias.requires_grad:
        raise ValueError("Learned attention bias is unsupported")


class _PermanentValueProjection:
    """One call-local stream of permanent writes, in forward token order.

Both frozen tiled passes call ``block.kv_proj`` once per output token; the
provisional projection is reached through ``pre_attention_block`` instead.
The ordinary-autograd scan has this same separation. Fail closed if it changes.
"""

    def __init__(self, block, bypass):
        self.projection = block.kv_proj
        self.bypass = bypass
        self.dimensions = tuple(block.fused_dims[1:])
        self.position = 0

    def __call__(self, normalized_output):
        batch, length, width = self.bypass.shape
        if (normalized_output.shape != (batch, 1, width)
                or self.position >= length or self.dimensions != (width, width)):
            raise ValueError("Unexpected permanent KV projection order/shape")
        key, value = self.projection(normalized_output).split(self.dimensions, dim=-1)
        addition = self.bypass[:, self.position:self.position + 1]
        self.position += 1
        return torch.cat((key, value + addition), dim=-1)

    def finish(self):
        if self.position != self.bypass.shape[1]:
            raise ValueError("Permanent KV replay did not visit every token exactly once")


class _WriteProxy:
    """Non-module per-call view: own no parameter aliases or persistent tensors."""

    def __init__(self, block, bypass):
        self.original = block
        self.kv_proj = _PermanentValueProjection(block, bypass)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _ContextWithBypass:
    """Append the explicit bypass to the vendor's single saved-tensor call."""

    def __init__(self, context, bypass):
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_bypass", bypass)

    def __getattr__(self, name):
        return getattr(self._context, name)

    def __setattr__(self, name, value):
        setattr(self._context, name, value)

    def save_for_backward(self, *tensors):
        if len(tensors) != 2:
            raise ValueError("Frozen tiled saved-state interface changed")
        self._context.save_for_backward(*tensors, self._bypass)


class _ValueBypassTiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bypass, attention_bias, block):
        proxy = _WriteProxy(block, bypass)
        output = OLMoRecurrentBlockTiledFunction.forward(
            _ContextWithBypass(ctx, bypass), x, attention_bias, proxy)
        proxy.kv_proj.finish()
        ctx.original_block = block
        # Retain tensor inputs via save_for_backward only, never through a proxy.
        ctx.block = None
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) != 3:
            raise ValueError("Value-bypass saved-state interface changed")
        bypass = ctx.saved_tensors[2].detach().requires_grad_(True)
        proxy = _WriteProxy(ctx.original_block, bypass)
        ctx.block = proxy
        try:
            # The vendor reads saved tensors0/1 and computes the complete RT
            # adjoint, including accumulation through future permanent reads.
            grad_x, grad_bias, grad_block = OLMoRecurrentBlockTiledFunction.backward(ctx, grad_output)
            proxy.kv_proj.finish()
            if grad_bias is not None or grad_block is not None or bypass.grad is None:
                raise ValueError("Frozen tiled backward/bypass interface changed")
            grad_bypass = bypass.grad
        finally:
            ctx.block = None
        # Returning this explicit gradient connects the separately computed
        # projection and original raw token embeddings to the outer graph.
        return grad_x, grad_bypass, None, None


class _ValueBypassMixin:
    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False,
                max_doc_len=None, cu_doc_lens=None, *, value_bypass=None):
        if value_bypass is None:
            return super().forward(x, attention_bias, layer_past, use_cache,
                                   max_doc_len, cu_doc_lens)
        if layer_past is not None or use_cache or max_doc_len is not None or cu_doc_lens is not None:
            raise ValueError("Value-bypass recurrence does not support inference caches or packed documents")
        _validate(self, x, value_bypass, attention_bias)
        if isinstance(self, OLMoRecurrentBlockTiled):
            if x.device.type != "cuda":
                raise ValueError("Tiled value bypass requires CUDA; use the explicit naive reference on CPU")
            if torch.is_grad_enabled() and not x.requires_grad:
                raise ValueError("Tiled training requires input gradients")
            output = _ValueBypassTiledFunction.apply(x, value_bypass, attention_bias, self)
        else:
            proxy = _WriteProxy(self, value_bypass)
            output = OLMoRecurrentAutogradBlock._real_forward(proxy, x, attention_bias)
            proxy.kv_proj.finish()
        return output, None


class ValueBypassRecurrentBlockTiled(_ValueBypassMixin, OLMoRecurrentBlockTiled):
    """Existing tiled forward/backward, with an explicit permanent-value input."""


class ValueBypassRecurrentAutogradBlock(_ValueBypassMixin, OLMoRecurrentAutogradBlock):
    """Ordinary-autograd semantic reference for the identical value write rule."""


def replace_value_bypass_block(original):
    """Replace only the full-prefix block at index1, preserving every tensor."""
    if original.layer_id != 1 or type(original) not in (OLMoRecurrentBlockTiled, OLMoRecurrentAutogradBlock):
        raise ValueError("Only the unchanged full-prefix RT block at index1 can receive this bypass")
    target = (ValueBypassRecurrentBlockTiled if isinstance(original, OLMoRecurrentBlockTiled)
              else ValueBypassRecurrentAutogradBlock)
    parameter = next(original.parameters())
    # Constructor initialization must not consume the caller's random stream.
    config = copy.deepcopy(original.config)
    config.init_device = "cpu"
    with torch.random.fork_rng(devices=[]):
        replacement = target(original.layer_id, config, BufferCache())
    replacement.to(device=parameter.device, dtype=parameter.dtype)
    replacement.load_state_dict(original.state_dict(), strict=True)
    replacement.train(original.training)
    if (list(dict(replacement.named_parameters())) != list(dict(original.named_parameters()))
            or any(not torch.equal(value, replacement.state_dict()[name])
                   for name, value in original.state_dict().items())):
        raise AssertionError("Value-bypass replacement changed parameter ownership or tensors")
    return replacement
