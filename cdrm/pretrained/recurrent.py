"""Differentiable sequential RT reference over unchanged native OpenELM blocks.

This is the semantic reference, not the efficient tiled training implementation.
No modules or parameters are added to the ordinary checkpoint layout. Temporary
Q/K/V come from the current layer input; only completed block outputs can enter
historical memory. Both temporary and persistent keys use native headwise
normalization before RoPE, and retained caches contain unrotated native KV heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor

from .openelm import (
    OpenELMCache,
    OpenELMDecoderLayer,
    OpenELMModel,
)


@dataclass(frozen=True)
class RTMode:
    """Immutable per-forward recurrence selection and memory-source bridge.

    ``selected_layers=()`` executes the ordinary model. ``alpha=0`` still runs
    the selected sequential scans: its ordinary-model limit is a tested
    mathematical property, not an implementation bypass. Alpha is a fixed
    configuration scalar, not a learned gate or mutable model attribute.
    """

    selected_layers: tuple[int, ...] = (0,)
    alpha: float = 1.0

    def __post_init__(self) -> None:
        selected = tuple(self.selected_layers)
        if any(type(index) is not int or index < 0 for index in selected):
            raise ValueError("selected_layers must contain nonnegative integer indices")
        if len(set(selected)) != len(selected):
            raise ValueError("selected_layers must not contain duplicate indices")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise TypeError("alpha must be a real configuration scalar")
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")
        object.__setattr__(self, "selected_layers", tuple(sorted(selected)))
        object.__setattr__(self, "alpha", float(self.alpha))


@dataclass(frozen=True)
class RTExecutionContext:
    """Forward arithmetic and autograd settings required by a live cache."""

    autocast_enabled: bool
    autocast_dtype: torch.dtype
    grad_enabled: bool
    inference_mode: bool
    attention_backend: str


@dataclass(frozen=True)
class OpenELMRecurrentCache:
    """Attached cache restricted to its producing weights, mode and context.

    Parameter identity and PyTorch version counters reject another model or a
    cache reused after an ordinary optimizer/load-state update. The model's
    conversion generation additionally rejects caches after ``model.to()`` and
    related conversions, including a lossy dtype roundtrip. Execution metadata
    prevents mixing gradient-bearing history with detached inference history,
    or mixing different precision/backend settings within one cache.

    This deliberately does not inherit ``OpenELMCache``: passing recurrent
    writes into the ordinary Stage A model must fail its cache-type check.
    Masks/positions are owned copies; returned cache tensors must not be
    mutated. As for PyTorch autograd itself, unsupported ``parameter.data``
    writes can bypass version counters and must not occur with a live cache.
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
class OpenELMRecurrentOutput:
    """The ordinary output fields, with an explicitly recurrent cache type."""

    logits: Tensor | None
    last_hidden_state: Tensor
    past_key_values: OpenELMRecurrentCache | None = None


def recurrent_layer_reference(
    layer: OpenELMDecoderLayer,
    x: Tensor,
    *,
    alpha: float,
    past: tuple[Tensor, Tensor] | None,
    query_positions: Tensor,
    key_positions: Tensor,
    key_valid: Tensor,
    attention_backend: str,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Scan one layer with attached memory writes and the native block function.

    ``key_positions`` and ``key_valid`` cover past plus current tokens. Every
    current query reads previous persistent K/V and its own temporary K/V. Its
    persistent write is published only after its residual attention and MLP
    complete, before the model's final normalization. No gradient is detached.
    """
    batch, length, _ = x.shape
    past_length = 0 if past is None else past[0].shape[-2]
    memory = past
    outputs = []
    attention = layer.attn
    for index in range(length):
        current = x[:, index:index + 1]
        stop = past_length + index + 1
        # All supplied keys are causally available. An explicit validity mask
        # also handles left/right padding and wholly masked rows correctly.
        output, _ = layer(
            current,
            past=memory,
            query_positions=query_positions[:, index:index + 1],
            key_positions=key_positions[:, :stop],
            mask=key_valid[:, None, None, :stop],
            is_causal=False,
            attention_backend=attention_backend,
        )
        outputs.append(output)
        source = (1.0 - alpha) * current + alpha * output
        projected = attention.qkv_proj(layer.attn_norm(source)).reshape(
            batch, 1, attention.num_q_heads + 2 * attention.num_k_heads,
            attention.head_dim,
        ).transpose(1, 2)
        _, key, value = projected.split(
            (attention.num_q_heads, attention.num_k_heads, attention.num_k_heads), dim=1,
        )
        key = attention.k_norm(key)
        memory = (key, value) if memory is None else (
            torch.cat((memory[0], key), dim=-2),
            torch.cat((memory[1], value), dim=-2),
        )
    assert memory is not None  # The model rejects empty sequences.
    return torch.cat(outputs, dim=1), memory


class OpenELMRecurrentModel(OpenELMModel):
    """Native OpenELM parameters with an explicit, per-call sequential RT mode.

    All ordinary layers retain the validated Stage A block implementation.
    Inherited initialization, strict state loading, tied readout and parameter
    names remain unchanged. There is no global recurrence toggle, tiled custom
    backward, feedback pass, auxiliary loss or optimizer inside this model.
    """

    def _apply(self, fn, recurse=True):
        # Module._apply can replace parameter storage without changing either
        # its Python identity or its version counter. Even converting BF16 and
        # back to FP32 changes the weights, so invalidate before any conversion.
        # This also conservatively invalidates no-op conversions and failures.
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
        self,
        input_ids: Tensor | None = None,
        *,
        mode: RTMode = RTMode(),
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: OpenELMRecurrentCache | None = None,
        use_cache: bool = False,
        return_logits: bool = True,
    ) -> OpenELMRecurrentOutput:
        """Run one stack pass, with ordinary or recurrent layers as specified.

        Masking and positions follow ``OpenELMModel.forward`` exactly. A mask
        covers the full cached plus current sequence; cached prefix validity
        cannot change. Inferred positions count valid tokens, continuing after
        the maximum valid cached coordinate. Padded queries must be excluded
        from loss. Cache continuation requires the same mode and unchanged
        model weights, conversion generation and execution context. Autocast,
        gradient/inference mode and configured backend cannot change across a
        cached continuation. Cache tensors are attached and must not be mutated.
        """
        if not isinstance(mode, RTMode):
            raise TypeError("mode must be an RTMode")
        if any(index >= len(self.layers) for index in mode.selected_layers):
            raise ValueError("selected_layers contains an index outside the model")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        if input_ids is not None:
            if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must be an integer [batch, length] tensor")
            if input_ids.device != self.token_embeddings.weight.device:
                raise ValueError("input_ids and model must be on the same device")
            x = self.token_embeddings(input_ids)
        else:
            assert inputs_embeds is not None
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.model_dim or not inputs_embeds.is_floating_point():
                raise ValueError("inputs_embeds must be floating point [batch, length, model_dim]")
            if inputs_embeds.device != self.token_embeddings.weight.device:
                raise ValueError("inputs_embeds and model must be on the same device")
            x = inputs_embeds
        batch, length, _ = x.shape
        if batch == 0 or length == 0:
            raise ValueError("Empty batches or sequences are unsupported")
        # These identities also detect loading new parameters with assign=True;
        # version counters detect normal in-place optimizer and load-state writes.
        versions = self._parameter_versions() if use_cache or past_key_values is not None else ()
        generation = getattr(self, "_rt_cache_generation", 0)
        execution_context = self._execution_context(x.device)
        past_length = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, OpenELMRecurrentCache):
                raise TypeError("past_key_values must be an OpenELMRecurrentCache")
            if past_key_values.mode != mode:
                raise ValueError("Cached RT mode differs from the requested mode")
            if past_key_values.parameter_versions != versions:
                raise ValueError("Cached weights differ from the current model or were modified")
            if past_key_values.model_generation != generation:
                raise ValueError("Cached model generation changed after a model conversion")
            if past_key_values.execution_context != execution_context:
                raise ValueError("Cached execution context differs: autocast, gradient/inference mode or attention backend changed")
            # Reuse Stage A's shape/device validation without allowing its
            # ordinary forward to accept recurrent cache objects by inheritance.
            self._validate_cache(OpenELMCache(
                past_key_values.key_values, past_key_values.attention_mask,
                past_key_values.position_ids,
            ), batch, x.device)
            past_length = past_key_values.sequence_length
        key_length = past_length + length
        if attention_mask is None:
            current_valid = torch.ones((batch, length), dtype=torch.bool, device=x.device)
            key_valid = current_valid if past_key_values is None else torch.cat((past_key_values.attention_mask, current_valid), dim=1)
        else:
            if attention_mask.shape != (batch, key_length) or attention_mask.device != x.device:
                raise ValueError("attention_mask must cover the full cached plus current sequence on the input device")
            if attention_mask.dtype != torch.bool and not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must be boolean or contain only 0/1")
            key_valid = attention_mask.to(torch.bool)
            if past_key_values is not None and not torch.equal(key_valid[:, :past_length], past_key_values.attention_mask):
                raise ValueError("A cached prefix's attention mask cannot be changed")
        if position_ids is None:
            current_positions = key_valid[:, past_length:].long().cumsum(dim=-1) - 1
            if past_key_values is not None:
                previous = past_key_values.position_ids.masked_fill(~past_key_values.attention_mask, -1)
                offset = previous.max(dim=-1).values + 1 if past_length else torch.zeros(batch, dtype=torch.long, device=x.device)
                current_positions = current_positions + offset.unsqueeze(-1)
            current_positions = current_positions.clamp_min(0)
        else:
            if position_ids.shape != (batch, length) or position_ids.device != x.device or position_ids.dtype != torch.long:
                raise ValueError("position_ids must be int64 [batch, current_length] on the input device")
            if bool((position_ids < 0).any()):
                raise ValueError("position_ids must be nonnegative")
            current_positions = position_ids
        key_positions = current_positions if past_key_values is None else torch.cat((past_key_values.position_ids, current_positions), dim=1)
        if past_key_values is None and attention_mask is None:
            causal_mask, is_causal = None, True
        else:
            queries = torch.arange(past_length, key_length, device=x.device)
            keys = torch.arange(key_length, device=x.device)
            causal_mask = (keys.unsqueeze(0) <= queries.unsqueeze(1))[None, None] & key_valid[:, None, None, :]
            is_causal = False
        present = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values.key_values[index]
            if index in mode.selected_layers:
                x, pair = recurrent_layer_reference(
                    layer, x, alpha=mode.alpha, past=past,
                    query_positions=current_positions, key_positions=key_positions,
                    key_valid=key_valid, attention_backend=self.attention_backend,
                )
            else:
                x, pair = layer(
                    x, past=past, query_positions=current_positions, key_positions=key_positions,
                    mask=causal_mask, is_causal=is_causal, attention_backend=self.attention_backend,
                )
            if use_cache:
                present.append(pair)
        hidden = self.norm(x)
        cache = OpenELMRecurrentCache(
            tuple(present), key_valid.clone(), key_positions.clone(), mode,
            versions, generation, execution_context,
        ) if use_cache else None
        return OpenELMRecurrentOutput(self.project_logits(hidden) if return_logits else None, hidden, cache)
