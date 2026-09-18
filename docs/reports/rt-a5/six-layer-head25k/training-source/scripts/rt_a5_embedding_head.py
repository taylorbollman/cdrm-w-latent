"""Replace one permanent RT value head with explicit embedding values.

Queries, contextual keys, provisional self K/V, and the cache dimensions retain
the existing RT implementation. Only one head's historical value write changes.
The explicit tensor input carries gradients back to the raw embeddings and the
reused value-projection rows outside the vendor's custom autograd function.
"""
from __future__ import annotations

import copy

import torch
from torch.autograd.function import once_differentiable

from olmo.model import (
    BufferCache, OLMoRecurrentAutogradBlock, OLMoRecurrentBlockTiled,
    OLMoRecurrentBlockTiledFunction,
)
from scripts.rt_a5_value_bypass import _ContextWithBypass


def _validate(block, x, values, attention_bias):
    cfg = block.config
    head_dim = cfg.d_model // cfg.n_heads
    if x.ndim != 3 or x.shape[1] < 1 or values.shape != (*x.shape[:2], head_dim):
        raise ValueError("Embedding values must have one head per token")
    if x.dtype != torch.float32 or values.dtype != torch.float32 or values.device != x.device:
        raise ValueError("Embedding-head diagnostics require colocated FP32 inputs")
    if torch.is_autocast_enabled(x.device.type):
        raise ValueError("Autocast is outside this diagnostic")
    if (not cfg.reference_eager or cfg.norm_after or cfg.clip_qkv is not None
            or cfg.recurrent_write_rho != 1.0 or cfg.rope
            or cfg.effective_n_kv_heads != cfg.n_heads
            or cfg.attention_dropout != 0 or cfg.residual_dropout != 0
            or cfg.bwd_mlp_chunks < 1):
        raise ValueError("Require the original attached pre-norm eager RT recipe")
    if block.pre_attention_block.kv_proj is not block.kv_proj:
        raise ValueError("Provisional KV must use the original owned projection")
    if any(p.dtype != torch.float32 for p in block.parameters()):
        raise ValueError("Embedding-head parameters must be FP32")
    if attention_bias is not None and attention_bias.requires_grad:
        raise ValueError("Learned attention bias is unsupported")


class _PermanentEmbeddingHeadProjection:
    def __init__(self, block, values):
        self.projection = block.kv_proj
        self.values = values
        self.width = block.config.d_model
        self.head_dim = self.width // block.config.n_heads
        self.start = block.embedding_head_index * self.head_dim
        self.end = self.start + self.head_dim
        self.position = 0
        if tuple(block.fused_dims[1:]) != (self.width, self.width):
            raise ValueError("Expected the unchanged full-width K/V projections")

    def __call__(self, normalized_output):
        batch, length, _ = self.values.shape
        if normalized_output.shape != (batch, 1, self.width) or self.position >= length:
            raise ValueError("Unexpected permanent KV write order/shape")
        key, value = self.projection(normalized_output).split(self.width, dim=-1)
        replacement = self.values[:, self.position:self.position + 1]
        self.position += 1
        # Replacement, not addition: the selected historical head no longer
        # stores its contextual value. All other slices and all keys are intact.
        value = torch.cat((value[..., :self.start], replacement, value[..., self.end:]), dim=-1)
        return torch.cat((key, value), dim=-1)

    def finish(self):
        if self.position != self.values.shape[1]:
            raise ValueError("Permanent replay must visit every token exactly once")


class _WriteProxy:
    def __init__(self, block, values):
        self.original = block
        self.kv_proj = _PermanentEmbeddingHeadProjection(block, values)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _EmbeddingHeadTiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, values, attention_bias, block):
        proxy = _WriteProxy(block, values)
        output = OLMoRecurrentBlockTiledFunction.forward(
            _ContextWithBypass(ctx, values), x, attention_bias, proxy)
        proxy.kv_proj.finish()
        ctx.original_block = block
        ctx.block = None
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) != 3:
            raise ValueError("Frozen tiled saved-state interface changed")
        values = ctx.saved_tensors[2].detach().requires_grad_(True)
        proxy = _WriteProxy(ctx.original_block, values)
        ctx.block = proxy
        try:
            grad_x, grad_bias, grad_block = OLMoRecurrentBlockTiledFunction.backward(ctx, grad_output)
            proxy.kv_proj.finish()
            if grad_bias is not None or grad_block is not None or values.grad is None:
                raise ValueError("Frozen tiled embedding-head adjoint interface changed")
            grad_values = values.grad
        finally:
            ctx.block = None
        return grad_x, grad_values, None, None


class _EmbeddingHeadMixin:
    def forward(self, x, attention_bias=None, layer_past=None, use_cache=False,
                max_doc_len=None, cu_doc_lens=None, *, embedding_values=None):
        if embedding_values is None:
            return super().forward(x, attention_bias, layer_past, use_cache,
                                   max_doc_len, cu_doc_lens)
        if layer_past is not None or use_cache or max_doc_len is not None or cu_doc_lens is not None:
            raise ValueError("Cached decoding and packed documents are outside this diagnostic")
        _validate(self, x, embedding_values, attention_bias)
        if isinstance(self, OLMoRecurrentBlockTiled):
            if x.device.type != "cuda":
                raise ValueError("Tiled embedding head requires CUDA; use the explicit naive reference on CPU")
            if torch.is_grad_enabled() and not x.requires_grad:
                raise ValueError("Tiled training requires input gradients")
            output = _EmbeddingHeadTiledFunction.apply(x, embedding_values, attention_bias, self)
        else:
            proxy = _WriteProxy(self, embedding_values)
            output = OLMoRecurrentAutogradBlock._real_forward(proxy, x, attention_bias)
            proxy.kv_proj.finish()
        return output, None


class EmbeddingHeadRecurrentBlockTiled(_EmbeddingHeadMixin, OLMoRecurrentBlockTiled):
    pass


class EmbeddingHeadRecurrentAutogradBlock(_EmbeddingHeadMixin, OLMoRecurrentAutogradBlock):
    pass


def replace_embedding_head_block(original):
    if original.layer_id != 1 or type(original) not in (OLMoRecurrentBlockTiled, OLMoRecurrentAutogradBlock):
        raise ValueError("Only the original full-prefix RT block at index1 may be replaced")
    target = (EmbeddingHeadRecurrentBlockTiled if isinstance(original, OLMoRecurrentBlockTiled)
              else EmbeddingHeadRecurrentAutogradBlock)
    parameter = next(original.parameters())
    config = copy.deepcopy(original.config)
    config.init_device = "cpu"
    with torch.random.fork_rng(devices=[]):
        replacement = target(original.layer_id, config, BufferCache())
    replacement.to(device=parameter.device, dtype=parameter.dtype)
    replacement.load_state_dict(original.state_dict(), strict=True)
    replacement.embedding_head_index = config.n_heads - 1
    replacement.train(original.training)
    if (list(dict(replacement.named_parameters())) != list(dict(original.named_parameters()))
            or any(not torch.equal(value, replacement.state_dict()[name])
                   for name, value in original.state_dict().items())):
        raise AssertionError("Head reassignment changed parameter ownership or initial tensors")
    return replacement
