"""Opt-in, fixed-shape CUDA capture of RT forward and complete loss backward.

The optimizer stays outside capture. Unlike make_graphed_callables, this captures
the tiled custom backward's internal parameter-gradient accumulation as well as
the ordinary autograd edges. The model/module tree and parameter names stay intact.
"""
from __future__ import annotations

import dataclasses
import time

import torch

from rt_batch_profile import head_chunked_backward


class CapturedRTBackward:
    """One physical batch per replay, with persistent FP32 gradient buffers.

    Capture performs no optimizer updates. Replay overwrites (does not add to)
    the preceding batch's gradients. Call clip_grad_norm_ and optimizer.step()
    after replay; do not replace parameters or call zero_grad(set_to_none=True).
    Only dense fixed-length, all-tiled, dropout-free training is supported here.
    Returned loss storage is reused by the next replay.
    """

    def __init__(self, model, example_ids, *, head_chunk_size=2, bf16=True, warmup=3,
                 precision_policy='bf16_fp32_state'):
        from olmo.model import OLMoRecurrentBlockTiled

        if example_ids.device.type != 'cuda' or example_ids.dtype != torch.long:
            raise ValueError('Capture requires CUDA int64 token IDs')
        if (example_ids.ndim != 2 or example_ids.shape[0] < 1 or example_ids.shape[1] < 2
                or head_chunk_size < 1 or warmup < 2):
            raise ValueError('Require [B,T>=2], positive head chunks and at least two warmups')
        cfg = model.config
        if not model.training or not torch.is_grad_enabled() or torch.is_autocast_enabled('cuda'):
            raise ValueError('Capture requires grad-enabled training with no enclosing autocast scope')
        if model.activation_checkpointing_strategy is not None or cfg.cdrm_enabled:
            raise ValueError('This capture path excludes outer checkpointing and CDRM')
        if not all(isinstance(b, OLMoRecurrentBlockTiled) for b in model.transformer.blocks):
            raise ValueError('This capture path requires all layers to be tiled recurrent')
        if any(getattr(cfg, key) for key in ('attention_dropout', 'residual_dropout', 'embedding_dropout')):
            raise ValueError('This capture path requires zero dropout')
        if (precision_policy not in ('bf16_fp32_state', 'legacy')
                or cfg.recurrent_precision_policy != precision_policy or cfg.precision is not None):
            raise ValueError('Use explicit autocast and the explicitly selected precision policy')
        if any(b.config.recurrent_precision_policy != precision_policy
               for b in model.transformer.blocks):
            raise ValueError('Every recurrent layer must use the selected precision policy')
        if not cfg.alibi or cfg.rope or cfg.weight_tying or cfg.block_group_size != 1:
            raise ValueError('This capture path requires ALiBi, no RoPE, untied head and ungrouped blocks')
        if any(getattr(m, '_recurrent_precision_observer', None) for m in model.modules()):
            raise ValueError('Remove Python precision observers before capture')
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()):
            raise ValueError('Remove module hooks before capture')
        self.model = model
        self.config_snapshot = dataclasses.asdict(cfg)
        self.modules = tuple(model.modules())
        self.module_ids = tuple(id(m) for m in self.modules)
        self.module_configs = tuple((m, dataclasses.asdict(m.config)) for m in self.modules
                                    if hasattr(m, 'config') and dataclasses.is_dataclass(m.config))
        self.parameters = tuple(model.parameters())
        self.parameter_ids = tuple(id(p) for p in self.parameters)
        if any(p.device != example_ids.device or p.dtype != torch.float32 or not p.requires_grad
               for p in self.parameters):
            raise ValueError('Require trainable FP32 parameters on the input CUDA device')
        if any(p.grad is not None for p in self.parameters):
            raise ValueError('Clear existing gradients before constructing the captured backward')
        self.parameter_ptrs = tuple(p.data_ptr() for p in self.parameters)
        self.static_ids = example_ids.clone()
        self.shape = tuple(example_ids.shape)
        self.head_chunk_size = head_chunk_size
        self.bf16 = bf16
        self.precision_policy = precision_policy
        self.gradients = tuple(torch.zeros_like(p) for p in self.parameters)
        for parameter, gradient in zip(self.parameters, self.gradients):
            parameter.grad = gradient
        self.gradient_ptrs = tuple(g.data_ptr() for g in self.gradients)
        self.graph = torch.cuda.CUDAGraph()
        # Eager reference work can leave a large cache on the caller's stream.
        # Side-stream warmup cannot directly reuse those cached blocks. Release
        # unused blocks first to avoid a needless temporary reservation spike.
        torch.cuda.synchronize(example_ids.device)
        torch.cuda.empty_cache()
        stream = torch.cuda.Stream(device=example_ids.device)
        stream.wait_stream(torch.cuda.current_stream(example_ids.device))
        started = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._body()
        torch.cuda.current_stream(example_ids.device).wait_stream(stream)
        torch.cuda.synchronize(example_ids.device)
        self.warmup_seconds = time.perf_counter() - started
        # Release unused eager allocations before creating the graph-private pool.
        torch.cuda.empty_cache()
        started = time.perf_counter()
        with torch.cuda.graph(self.graph, stream=stream):
            self.loss = self._body()
        torch.cuda.current_stream(example_ids.device).wait_stream(stream)
        torch.cuda.synchronize(example_ids.device)
        self.capture_seconds = time.perf_counter() - started
        self.replay_count = 0
        self._check_model()

    def _body(self):
        for gradient in self.gradients:
            gradient.zero_()
        return head_chunked_backward(self.model, self.static_ids,
                                     chunk_size=self.head_chunk_size, bf16=self.bf16)

    def _check_model(self):
        if (not self.model.training or not torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
                or any(not m.training for m in self.modules)):
            raise RuntimeError('Captured backward only supports grad-enabled training')
        if (tuple(id(m) for m in self.model.modules()) != self.module_ids
                or tuple(id(p) for p in self.model.parameters()) != self.parameter_ids):
            raise RuntimeError('Model structure changed after capture')
        if dataclasses.asdict(self.model.config) != self.config_snapshot:
            raise RuntimeError('Model configuration changed after capture')
        if any(dataclasses.asdict(m.config) != config for m, config in self.module_configs):
            raise RuntimeError('Layer configuration changed after capture')
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
               or getattr(m, '_recurrent_precision_observer', None) for m in self.modules):
            raise RuntimeError('Module hooks and observers are unsupported during replay')
        for parameter, pointer, gradient, grad_pointer in zip(
                self.parameters, self.parameter_ptrs, self.gradients, self.gradient_ptrs):
            if parameter.data_ptr() != pointer or not parameter.requires_grad or parameter.dtype != torch.float32:
                raise RuntimeError('Captured parameter storage or precision changed')
            if parameter._backward_hooks:
                raise RuntimeError('Parameter hooks are unsupported during replay')
            if parameter.grad is not gradient or parameter.grad.data_ptr() != grad_pointer:
                raise RuntimeError('Captured gradient storage changed; do not set gradients to None')

    def replay(self, ids):
        self._check_model()
        if tuple(ids.shape) != self.shape or ids.device != self.static_ids.device or ids.dtype != torch.long:
            raise ValueError('Replay input shape, dtype and device must match capture')
        self.static_ids.copy_(ids)
        self.graph.replay()
        self.replay_count += 1
        return self.loss

    def metadata(self):
        return {'enabled': True, 'scope': 'Full model forward, head-chunked CE and full backward',
                'optimizer_captured': False, 'clip_captured': False,
                'module_tree_preserved': True, 'causal_mask_via_native_model_forward': True,
                'gradient_buffers': 'Persistent FP32; zeroed inside each replay',
                'warmup_seconds': self.warmup_seconds, 'capture_seconds': self.capture_seconds,
                'replays': self.replay_count, 'shape': list(self.shape),
                'recurrent_precision_policy': self.precision_policy,
                'autocast': 'bf16' if self.bf16 else 'disabled'}
