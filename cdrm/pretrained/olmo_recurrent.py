"""Attached sequential RT reference over unchanged original OLMo parameters.

Only selected blocks scan over positions. Each token reads earlier completed
memory writes and its own input-derived temporary K/V, then publishes a write
from A((1-alpha)*x + alpha*z). No Q/K norm, feedback, tiled backward or auxiliary
objective is introduced. This is the correctness reference, not a fast backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .olmo import OLMoBlock, OLMoForCausalLM
from .recurrent import RTExecutionContext, RTMode


@dataclass(frozen=True)
class OLMoRecurrentCache:
    """Attached unrotated native KV with immutable model/mode provenance.

    Parameter identity/version and model conversion generation reject stale
    caches. Autocast and autograd context cannot change within a history.
    Metadata is copied on construction; callers must not mutate KV tensors.
    Unsupported parameter.data mutation is outside this contract.
    """

    key_values: tuple[tuple[Tensor, Tensor], ...]
    attention_mask: Tensor
    position_ids: Tensor
    mode: RTMode
    parameter_versions: tuple[tuple[int, int], ...] = field(repr=False)
    model_generation: int = field(repr=False)
    execution_context: RTExecutionContext

    @property
    def sequence_length(self) -> int:
        return self.attention_mask.shape[1]


@dataclass
class OLMoRecurrentOutput:
    logits: Tensor | None
    last_hidden_state: Tensor
    past_key_values: OLMoRecurrentCache | None = None


def recurrent_layer_reference(
    layer: OLMoBlock, x: Tensor, *, alpha: float,
    past: tuple[Tensor, Tensor] | None,
    query_positions: Tensor, key_positions: Tensor, key_valid: Tensor,
    attention_backend: str,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Scan one native block, preserving both fractional-alpha gradient paths."""
    past_length = 0 if past is None else past[0].shape[-2]
    memory, outputs = past, []
    for index in range(x.shape[1]):
        current = x[:, index:index + 1]
        stop = past_length + index + 1
        # At this point historical entries are persistent and the current
        # entry is temporary, produced inside the ordinary native block.
        completed, _ = layer(
            current, past=memory,
            query_positions=query_positions[:, index:index + 1],
            key_positions=key_positions[:, :stop],
            mask=key_valid[:, None, None, :stop],
            is_causal=False, attention_backend=attention_backend,
        )
        outputs.append(completed)
        source = (1.0 - alpha) * current + alpha * completed
        _, key, value = layer.project_qkv(layer.attn_norm(source))
        memory = (key, value) if memory is None else (
            torch.cat((memory[0], key), dim=-2),
            torch.cat((memory[1], value), dim=-2),
        )
    assert memory is not None  # Public forward rejects empty sequences.
    return torch.cat(outputs, dim=1), memory


class OLMoRTForCausalLM(OLMoForCausalLM):
    """Native parameter ownership with immutable per-forward recurrence mode."""

    def _apply(self, fn, recurse=True):
        # Even a lossy dtype roundtrip may preserve Parameter identity/version.
        self._rt_cache_generation = getattr(self, "_rt_cache_generation", 0) + 1
        return super()._apply(fn, recurse=recurse)

    def _parameter_versions(self) -> tuple[tuple[int, int], ...]:
        return tuple((id(parameter), parameter._version) for parameter in self.parameters())

    def _execution_context(self, device: torch.device) -> RTExecutionContext:
        return RTExecutionContext(
            autocast_enabled=torch.is_autocast_enabled(device.type),
            autocast_dtype=torch.get_autocast_dtype(device.type),
            grad_enabled=torch.is_grad_enabled(),
            inference_mode=torch.is_inference_mode_enabled(),
            attention_backend=self.attention_backend,
        )

    def forward(
        self, input_ids: Tensor | None = None, *, mode: RTMode = RTMode(),
        inputs_embeds: Tensor | None = None, attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: OLMoRecurrentCache | None = None,
        use_cache: bool = False, return_logits: bool = True,
    ) -> OLMoRecurrentOutput:
        """Run one ordinary/recurrent stack pass with attached causal history.

        Alpha zero deliberately executes each selected scan. Ordinary cache
        objects cannot supply a recurrent history even when alpha is zero.
        Cached prefix masks, weight ownership and execution context are fixed.
        """
        if not isinstance(mode, RTMode):
            raise TypeError("mode must be an RTMode")
        if any(index >= self.config.num_layers for index in mode.selected_layers):
            raise ValueError("selected_layers contains an index outside the model")
        x, valid, positions, key_positions, mask, causal = self._prepare_inputs(
            input_ids, inputs_embeds, attention_mask, position_ids, past_key_values,
            cache_type=OLMoRecurrentCache,
        )
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
            if past_key_values.execution_context != context:
                raise ValueError("Cached execution context differs: autocast, gradient/inference mode or attention backend changed")
        present = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values.key_values[index]
            if index in mode.selected_layers:
                x, pair = recurrent_layer_reference(
                    layer, x, alpha=mode.alpha, past=past,
                    query_positions=positions, key_positions=key_positions,
                    key_valid=valid, attention_backend=self.attention_backend,
                )
            else:
                x, pair = layer(x, past=past, query_positions=positions,
                                key_positions=key_positions, mask=mask,
                                is_causal=causal, attention_backend=self.attention_backend)
            if use_cache:
                present.append(pair)
        hidden = self.norm(x)
        cache = OLMoRecurrentCache(
            tuple(present), valid.clone(), key_positions.clone(), mode,
            versions, generation, context,
        ) if use_cache else None
        return OLMoRecurrentOutput(self.project_logits(hidden) if return_logits else None, hidden, cache)
