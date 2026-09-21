"""Ordinary OpenELM with the native CoreNet checkpoint layout.

The computational reference is CoreNet revision
f9f83e616a34d02c422733a06a3fe5bde63ae575, ``general_gpt.py`` and its RMSNorm/
RoPE layers. This module preserves those tensor names, the complete native
vocabulary and one owned embedding/readout parameter. It adds an explicit,
causal cache/masking interface; there is no RT, FBT or auxiliary objective.

Keys in the cache are normalized and *unrotated*, as in the native model.
Padding is masked only when an attention_mask is provided. Merely passing the
padding token ID preserves the native unmasked forward behavior. Native pad
lookup gradients are suppressed, but its tied readout row remains trainable.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


CORENET_REVISION = "f9f83e616a34d02c422733a06a3fe5bde63ae575"


@dataclass(frozen=True)
class OpenELMConfig:
    """Resolved dimensions, with no implicit resizing or normalization changes."""

    vocab_size: int
    padding_idx: int | None
    model_dim: int
    head_dim: int
    num_query_heads: tuple[int, ...]
    num_kv_heads: tuple[int, ...]
    ffn_intermediate_sizes: tuple[int, ...]
    rms_norm_eps: float = 1e-6
    rope_freq_constant: float = 10000.0
    max_context_length: int = 2048

    def __post_init__(self) -> None:
        for name in ("vocab_size", "model_dim", "head_dim", "max_context_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if self.padding_idx is not None and (
            type(self.padding_idx) is not int or not 0 <= self.padding_idx < self.vocab_size
        ):
            raise ValueError("padding_idx must be None or a valid vocabulary row")
        for name in ("num_query_heads", "num_kv_heads", "ffn_intermediate_sizes"):
            values = tuple(getattr(self, name))
            if not values or any(type(value) is not int or value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers")
            object.__setattr__(self, name, values)
        if len({len(self.num_query_heads), len(self.num_kv_heads), len(self.ffn_intermediate_sizes)}) != 1:
            raise ValueError("Every layer must have query heads, KV heads and an FFN size")
        if any(q % k for q, k in zip(self.num_query_heads, self.num_kv_heads)):
            raise ValueError("Query heads must be divisible by KV heads in every layer")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if not math.isfinite(self.rope_freq_constant) or self.rope_freq_constant <= 0:
            raise ValueError("rope_freq_constant must be finite and positive")

    @property
    def num_transformer_layers(self) -> int:
        return len(self.num_query_heads)

    @classmethod
    def native_1_1b(cls) -> OpenELMConfig:
        """Individual native checkpoint geometry, including 128 extra rows."""
        query_heads = (16,) * 3 + (20,) * 7 + (24,) * 8 + (28,) * 6 + (32,) * 4
        return cls(
            vocab_size=32128, padding_idx=32000, model_dim=2048, head_dim=64,
            num_query_heads=query_heads, num_kv_heads=tuple(q // 4 for q in query_heads),
            ffn_intermediate_sizes=(
                1024, 1280, 1536, 1792, 2048, 2304, 2560,
                2816, 3072, 3328, 3584, 3840, 4096, 4608,
                4608, 5120, 5376, 5632, 5888, 6144, 6400,
                6656, 6912, 7168, 7424, 7680, 7936, 8192,
            ),
        )

    @classmethod
    def tiny(cls) -> OpenELMConfig:
        """Small nonuniform GQA fixture; never a substitute for the real import."""
        return cls(
            vocab_size=67, padding_idx=64, model_dim=32, head_dim=8,
            num_query_heads=(2, 4), num_kv_heads=(1, 2),
            ffn_intermediate_sizes=(24, 48), max_context_length=32,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> OpenELMConfig:
        # Dataclass construction deliberately rejects unknown fields.
        return cls(**values)


@dataclass(frozen=True)
class OpenELMCache:
    """Native-head KV tensors plus the masks/positions needed for continuation.

    A cache belongs to the same weights and mode that produced it. It is not
    detached, mutated in place or expanded to the query-head count.
    """

    key_values: tuple[tuple[Tensor, Tensor], ...]
    attention_mask: Tensor
    position_ids: Tensor

    @property
    def sequence_length(self) -> int:
        return self.attention_mask.shape[1]


@dataclass
class OpenELMOutput:
    logits: Tensor | None
    last_hidden_state: Tensor
    past_key_values: OpenELMCache | None = None


class OpenELMRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float, *, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Keep CoreNet's cast *before* the learned multiplication. In autocast
        # the FP32 gain can promote this result; changing the order changes it.
        # Share this cast across both branches, as CoreNet does: separate casts
        # round their backward contributions to BF16 before they are added.
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.type_as(x) * self.weight


def _apply_rope(x: Tensor, position_ids: Tensor, base: float) -> Tensor:
    """Native split-half RoPE, with phase and rotation computed in FP32."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, x.shape[-1], 2, dtype=torch.float32, device=x.device) / x.shape[-1])
        )
        phases = position_ids.float().unsqueeze(-1) * inv_freq
        phases = torch.cat((phases, phases), dim=-1).unsqueeze(1)
        x_float = x.float()
        first, second = x_float.chunk(2, dim=-1)
        rotated = x_float * phases.cos() + torch.cat((-second, first), dim=-1) * phases.sin()
    return rotated.type_as(x)


class OpenELMAttention(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_index: int, *, device=None, dtype=None):
        super().__init__()
        self.num_q_heads = config.num_query_heads[layer_index]
        self.num_k_heads = self.num_v_heads = config.num_kv_heads[layer_index]
        self.num_groups = self.num_q_heads // self.num_k_heads
        self.head_dim = config.head_dim
        self.rope_freq_constant = config.rope_freq_constant
        factory = {"device": device, "dtype": dtype}
        self.qkv_proj = nn.Linear(config.model_dim, (self.num_q_heads + 2 * self.num_k_heads) * self.head_dim, bias=False, **factory)
        self.q_norm = OpenELMRMSNorm(self.head_dim, config.rms_norm_eps, **factory)
        self.k_norm = OpenELMRMSNorm(self.head_dim, config.rms_norm_eps, **factory)
        self.out_proj = nn.Linear(self.num_q_heads * self.head_dim, config.model_dim, bias=False, **factory)

    def forward(
        self, x: Tensor, *, past: tuple[Tensor, Tensor] | None,
        query_positions: Tensor, key_positions: Tensor, mask: Tensor | None,
        is_causal: bool, attention_backend: str,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, _ = x.shape
        projected = self.qkv_proj(x).reshape(batch, length, self.num_q_heads + 2 * self.num_k_heads, self.head_dim).transpose(1, 2)
        queries, keys, values = projected.split((self.num_q_heads, self.num_k_heads, self.num_k_heads), dim=1)
        queries, keys = self.q_norm(queries), self.k_norm(keys)
        if past is not None:
            keys, values = torch.cat((past[0], keys), dim=-2), torch.cat((past[1], values), dim=-2)
        # Capture before RoPE and head expansion, without detaching history.
        present = (keys, values)
        queries = _apply_rope(queries, query_positions, self.rope_freq_constant)
        keys = _apply_rope(keys, key_positions, self.rope_freq_constant)
        if self.num_groups != 1:
            # Match the native reference. Expansion applies to activations only;
            # parameters and the retained cache keep their native KV dimensions.
            keys = keys.repeat_interleave(self.num_groups, dim=1)
            values = values.repeat_interleave(self.num_groups, dim=1)
        context = sdpa_kernel(SDPBackend.MATH) if attention_backend == "math" else nullcontext()
        with context:
            attended = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=mask, dropout_p=0.0, is_causal=is_causal,
            )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, self.num_q_heads * self.head_dim)
        return self.out_proj(attended), present


class OpenELMFeedForward(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_index: int, *, device=None, dtype=None):
        super().__init__()
        intermediate = config.ffn_intermediate_sizes[layer_index]
        self.proj_1 = nn.Linear(config.model_dim, 2 * intermediate, bias=False, device=device, dtype=dtype)
        self.proj_2 = nn.Linear(intermediate, config.model_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.proj_1(x).chunk(2, dim=-1)
        return self.proj_2(F.silu(gate) * value)


class OpenELMDecoderLayer(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_index: int, *, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.attn = OpenELMAttention(config, layer_index, **factory)
        self.ffn = OpenELMFeedForward(config, layer_index, **factory)
        self.ffn_norm = OpenELMRMSNorm(config.model_dim, config.rms_norm_eps, **factory)
        self.attn_norm = OpenELMRMSNorm(config.model_dim, config.rms_norm_eps, **factory)

    def forward(self, x: Tensor, **attention_args) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, present = self.attn(self.attn_norm(x), **attention_args)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x)), present


class OpenELMModel(nn.Module):
    """Ordinary tied OpenELM; state_dict names match native CoreNet exactly."""

    def __init__(
        self, config: OpenELMConfig, *, attention_backend: str = "sdpa", device=None, dtype=None,
    ):
        super().__init__()
        if not isinstance(config, OpenELMConfig):
            raise TypeError("config must be an OpenELMConfig")
        if attention_backend not in ("sdpa", "math"):
            raise ValueError("attention_backend must be 'sdpa' or 'math'")
        self.config = config
        self.attention_backend = attention_backend
        factory = {"device": device, "dtype": dtype}
        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=config.padding_idx, **factory)
        self.layers = nn.ModuleList(OpenELMDecoderLayer(config, i, **factory) for i in range(config.num_transformer_layers))
        self.norm = OpenELMRMSNorm(config.model_dim, config.rms_norm_eps, **factory)
        # No duplicate Parameter or readout module; strict loading needs one key.
        self.classifier = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=module.in_features ** -0.5)
            elif isinstance(module, nn.Embedding):
                # Native GeneralGPTModel's reset overwrites the initial pad zero.
                nn.init.normal_(module.weight, std=module.embedding_dim ** -0.5)
            elif isinstance(module, OpenELMRMSNorm):
                nn.init.ones_(module.weight)
        residual_std = self.config.model_dim ** -0.5 * (2 * self.config.num_transformer_layers) ** -0.5
        for name, parameter in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ffn.proj_2.weight"):
                nn.init.normal_(parameter, std=residual_std)

    @property
    def readout_weight(self) -> nn.Parameter:
        return self.token_embeddings.weight

    def project_logits(self, hidden_states: Tensor) -> Tensor:
        """Project all native vocabulary rows, without softcap or cropping."""
        return F.linear(hidden_states, self.readout_weight)

    def _validate_cache(self, cache: OpenELMCache, batch: int, device: torch.device) -> None:
        if not isinstance(cache, OpenELMCache):
            raise TypeError("past_key_values must be an OpenELMCache")
        if cache.attention_mask.ndim != 2 or cache.attention_mask.shape[0] != batch:
            raise ValueError("Cached attention mask must have shape [batch, past_length]")
        if cache.attention_mask.dtype != torch.bool or cache.attention_mask.device != device:
            raise ValueError("Cached attention mask must be boolean on the input device")
        if cache.position_ids.shape != cache.attention_mask.shape or cache.position_ids.dtype != torch.long or cache.position_ids.device != device:
            raise ValueError("Cached positions must be int64 with the same shape/device as the cached mask")
        if len(cache.key_values) != self.config.num_transformer_layers:
            raise ValueError("Cached layer count differs from model")
        for i, pair in enumerate(cache.key_values):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each cached layer must contain a key/value pair")
            expected = (batch, self.config.num_kv_heads[i], cache.sequence_length, self.config.head_dim)
            if any(tensor.shape != expected or tensor.device != device for tensor in pair):
                raise ValueError(f"Cached layer {i} must have native KV shape {expected} on the input device")
            if any(not tensor.is_floating_point() for tensor in pair):
                raise ValueError("Cached keys and values must be floating point")

    def forward(
        self, input_ids: Tensor | None = None, *, inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None, position_ids: Tensor | None = None,
        past_key_values: OpenELMCache | None = None, use_cache: bool = False,
        return_logits: bool = True,
    ) -> OpenELMOutput:
        """Causal forward with optional padding and arbitrary cached chunk size.

        attention_mask is a boolean or binary [B, past_length + current_length]
        key-validity mask. Cached prefix validity cannot be changed. When omitted
        on continuation, stored validity is retained and new positions are valid.
        position_ids optionally gives absolute RoPE coordinates [B, current_length].
        Without it, valid tokens count up from zero (padding does not advance RoPE).
        Padding-query outputs are not meaningful and should be excluded from loss.
        max_context_length records the pretrained context, not a RoPE allocation cap.
        """
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
        past_length = 0
        if past_key_values is not None:
            self._validate_cache(past_key_values, batch, x.device)
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
            # Preserve explicit cached offsets too: continuation starts after the
            # last valid cached coordinate, not necessarily after its token count.
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
        # Do not trust SDPA's upper-left causal alignment for unequal Q/K lengths.
        # The native unpadded prefill route retains the efficient causal flag.
        if past_key_values is None and attention_mask is None:
            causal_mask, is_causal = None, True
        else:
            queries = torch.arange(past_length, key_length, device=x.device)
            keys = torch.arange(key_length, device=x.device)
            causal_mask = (keys.unsqueeze(0) <= queries.unsqueeze(1))[None, None] & key_valid[:, None, None, :]
            is_causal = False
        present = []
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values.key_values[i]
            x, pair = layer(
                x, past=past, query_positions=current_positions, key_positions=key_positions,
                mask=causal_mask, is_causal=is_causal, attention_backend=self.attention_backend,
            )
            if use_cache:
                present.append(pair)
        hidden = self.norm(x)
        cache = OpenELMCache(tuple(present), key_valid, key_positions) if use_cache else None
        return OpenELMOutput(self.project_logits(hidden) if return_logits else None, hidden, cache)
