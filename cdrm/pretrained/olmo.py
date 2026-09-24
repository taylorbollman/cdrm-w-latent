"""Original OLMo-1B with its native fused checkpoint parameter layout.

Source authority: AllenAI OLMo b3741bc21f1dd504838b7dbd9878ee077ded63bd.
The adapter preserves nonaffine LayerNorm, value-then-gate SwiGLU, full MHA,
FP32 split-half RoPE and one owned embedding/readout matrix. Cached keys are
unrotated. Explicit masks govern padding; the native embedding has no
padding_idx and its configured pad row remains trainable through lookup.
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

from .olmo_rope import RopeTables, apply_rope_tables
from .olmo_ordinary import flash_attention, ordinary_swiglu, validate_ordinary_options


OLMO_REVISION = "b3741bc21f1dd504838b7dbd9878ee077ded63bd"


@dataclass(frozen=True)
class OLMoConfig:
    """Resolved native dimensions; vocab_size includes every output row."""

    vocab_size: int = 50304
    tokenizer_vocab_size: int = 50280
    model_dim: int = 2048
    num_layers: int = 16
    num_heads: int = 16
    mlp_intermediate_size: int = 8192
    layer_norm_eps: float = 1e-5
    rope_freq_constant: float = 10000.0
    max_context_length: int = 2048
    pad_token_id: int = 1
    eos_token_id: int = 50279

    def __post_init__(self) -> None:
        for name in ("vocab_size", "tokenizer_vocab_size", "model_dim", "num_layers", "num_heads", "mlp_intermediate_size", "max_context_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.tokenizer_vocab_size > self.vocab_size:
            raise ValueError("tokenizer_vocab_size cannot exceed the model vocabulary")
        if self.model_dim % self.num_heads or self.head_dim % 2:
            raise ValueError("model_dim must be divisible by num_heads with even head_dim")
        for name in ("pad_token_id", "eos_token_id"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} must be a valid vocabulary row")
        for name in ("layer_norm_eps", "rope_freq_constant"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    @property
    def d_model(self) -> int:
        return self.model_dim

    @property
    def n_layers(self) -> int:
        return self.num_layers

    @property
    def n_heads(self) -> int:
        return self.num_heads

    @classmethod
    def native_1b(cls) -> OLMoConfig:
        return cls()

    @classmethod
    def tiny(cls) -> OLMoConfig:
        return cls(vocab_size=67, tokenizer_vocab_size=61, model_dim=32,
                   num_layers=2, num_heads=4, mlp_intermediate_size=64,
                   max_context_length=32, pad_token_id=1, eos_token_id=60)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> OLMoConfig:
        return cls(**values)


@dataclass(frozen=True)
class OLMoCache:
    key_values: tuple[tuple[Tensor, Tensor], ...]
    attention_mask: Tensor
    position_ids: Tensor

    @property
    def sequence_length(self) -> int:
        return self.attention_mask.shape[1]


@dataclass
class OLMoOutput:
    logits: Tensor | None
    last_hidden_state: Tensor
    past_key_values: OLMoCache | None = None


class OLMoLayerNorm(nn.Module):
    """Native default LayerNorm: no affine parameters and native autocast."""

    def __init__(self, width: int, eps: float):
        super().__init__()
        self.normalized_shape = (width,)
        self.eps = eps
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, weight=None, bias=None, eps=self.eps)


def _apply_rope(x: Tensor, positions: Tensor, base: float) -> Tensor:
    """Native full-precision split-half rotation, then restore Q/K dtype."""
    # A single cast feeds both branches, preserving its backward accumulation.
    x_float = x.float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        inv_freq = 1.0 / (base ** (torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32) / x.shape[-1]))
        phases = positions.float().unsqueeze(-1) * inv_freq
        phases = torch.cat((phases, phases), dim=-1).unsqueeze(1)
        first, second = x_float.chunk(2, dim=-1)
        rotated = x_float * phases.cos() + torch.cat((-second, first), dim=-1) * phases.sin()
    return rotated.to(x.dtype)


class OLMoBlock(nn.Module):
    """Native sequential block with fused QKV and value/gate projections."""

    def __init__(self, config: OLMoConfig, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        factory = {"device": device, "dtype": dtype}
        self.att_proj = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False, **factory)
        self.attn_out = nn.Linear(config.model_dim, config.model_dim, bias=False, **factory)
        self.ff_proj = nn.Linear(config.model_dim, 2 * config.mlp_intermediate_size, bias=False, **factory)
        self.ff_out = nn.Linear(config.mlp_intermediate_size, config.model_dim, bias=False, **factory)
        self.attn_norm = OLMoLayerNorm(config.model_dim, config.layer_norm_eps)
        self.ff_norm = OLMoLayerNorm(config.model_dim, config.layer_norm_eps)

    def project_qkv(self, normalized: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = normalized.shape
        # Split the fused feature dimension before reshaping exactly as native.
        parts = self.att_proj(normalized).split(self.config.model_dim, dim=-1)
        return tuple(part.view(batch, length, self.config.num_heads, self.config.head_dim).transpose(1, 2) for part in parts)

    def forward(self, x: Tensor, *, past: tuple[Tensor, Tensor] | None,
                query_positions: Tensor, key_positions: Tensor,
                mask: Tensor | None, is_causal: bool, attention_backend: str,
                query_rope: RopeTables | None = None, key_rope: RopeTables | None = None,
                ordinary_attention_backend: str = "sdpa", ordinary_pointwise_backend: str = "eager",
                ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        validate_ordinary_options(ordinary_attention_backend, ordinary_pointwise_backend,
                                  sdpa_backend=attention_backend)
        if ordinary_attention_backend == "fa4" and (past is not None or mask is not None or not is_causal):
            raise ValueError("Ordinary FA4 supports only dense causal full sequences without masks or prefix caches")
        query, key, value = self.project_qkv(self.attn_norm(x))
        if past is not None:
            key, value = torch.cat((past[0], key), dim=-2), torch.cat((past[1], value), dim=-2)
        present = (key, value)
        query = (_apply_rope(query, query_positions, self.config.rope_freq_constant)
                 if query_rope is None else apply_rope_tables(query, query_rope))
        key = (_apply_rope(key, key_positions, self.config.rope_freq_constant)
               if key_rope is None else apply_rope_tables(key, key_rope))
        if ordinary_attention_backend == "fa4":
            attended = flash_attention(query, key, value)
        else:
            context = sdpa_kernel(SDPBackend.MATH) if attention_backend == "math" else nullcontext()
            with context:
                attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)
        attended = attended.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.config.model_dim)
        residual = x + self.attn_out(attended)
        projected = self.ff_proj(self.ff_norm(residual))
        return residual + self.ff_out(ordinary_swiglu(projected, backend=ordinary_pointwise_backend)), present


class OLMoForCausalLM(nn.Module):
    """Ordinary native OLMo; state names exclude only the HF wrapper 'model.'."""

    def __init__(self, config: OLMoConfig, *, attention_backend="sdpa", device=None, dtype=None):
        super().__init__()
        if not isinstance(config, OLMoConfig):
            raise TypeError("config must be an OLMoConfig")
        if attention_backend not in ("sdpa", "math"):
            raise ValueError("attention_backend must be 'sdpa' or 'math'")
        self.config, self.attention_backend = config, attention_backend
        factory = {"device": device, "dtype": dtype}
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.model_dim, **factory),
            "ln_f": OLMoLayerNorm(config.model_dim, config.layer_norm_eps),
            "blocks": nn.ModuleList(OLMoBlock(config, **factory) for _ in range(config.num_layers)),
        })
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Mitchell initialization for fixtures; imported parameters replace all."""
        def initialize(weight, std):
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        initialize(self.token_embeddings.weight, self.config.model_dim ** -0.5)
        for i, block in enumerate(self.layers):
            initialize(block.att_proj.weight, self.config.model_dim ** -0.5)
            initialize(block.ff_proj.weight, self.config.model_dim ** -0.5)
            initialize(block.attn_out.weight, (2 * self.config.model_dim * (i + 1)) ** -0.5)
            initialize(block.ff_out.weight, (2 * self.config.mlp_intermediate_size * (i + 1)) ** -0.5)

    @property
    def token_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    @property
    def layers(self) -> nn.ModuleList:
        return self.transformer.blocks

    @property
    def norm(self) -> OLMoLayerNorm:
        return self.transformer.ln_f

    @property
    def readout_weight(self) -> nn.Parameter:
        return self.transformer.wte.weight

    def project_logits(self, hidden_states: Tensor) -> Tensor:
        return F.linear(hidden_states, self.readout_weight)

    def _validate_cache(self, cache, batch: int, device: torch.device, *, cache_type=OLMoCache) -> None:
        if not isinstance(cache, cache_type):
            raise TypeError(f"past_key_values must be an {cache_type.__name__}")
        if cache.attention_mask.ndim != 2 or cache.attention_mask.shape[0] != batch:
            raise ValueError("Cached attention mask must have shape [batch, past_length]")
        if cache.attention_mask.dtype != torch.bool or cache.attention_mask.device != device:
            raise ValueError("Cached attention mask must be boolean on the input device")
        if cache.position_ids.shape != cache.attention_mask.shape or cache.position_ids.dtype != torch.long or cache.position_ids.device != device:
            raise ValueError("Cached positions must be int64 with the same shape/device as the cached mask")
        if len(cache.key_values) != self.config.num_layers:
            raise ValueError("Cached layer count differs from model")
        expected = (batch, self.config.num_heads, cache.sequence_length, self.config.head_dim)
        for pair in cache.key_values:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each cached layer must contain a key/value pair")
            if any(tensor.shape != expected or tensor.device != device for tensor in pair):
                raise ValueError(f"Cached layer must have native KV shape {expected} on the input device")
            if any(not tensor.is_floating_point() for tensor in pair):
                raise ValueError("Cached keys and values must be floating point")

    def _prepare_inputs(self, input_ids, inputs_embeds, attention_mask, position_ids,
                        past_key_values, *, cache_type=OLMoCache):
        """Shared ordinary/RT input and causal-position contract, without math changes."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        if input_ids is not None:
            if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must be an integer [batch, length] tensor")
            if input_ids.device != self.readout_weight.device:
                raise ValueError("input_ids and model must be on the same device")
            x = self.token_embeddings(input_ids)
        else:
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.model_dim or not inputs_embeds.is_floating_point():
                raise ValueError("inputs_embeds must be floating point [batch, length, model_dim]")
            if inputs_embeds.device != self.readout_weight.device:
                raise ValueError("inputs_embeds and model must be on the same device")
            x = inputs_embeds
        batch, length, _ = x.shape
        if not batch or not length:
            raise ValueError("Empty batches or sequences are unsupported")
        past_length = 0
        if past_key_values is not None:
            self._validate_cache(past_key_values, batch, x.device, cache_type=cache_type)
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
            current_positions = key_valid[:, past_length:].long().cumsum(-1) - 1
            if past_key_values is not None:
                previous = past_key_values.position_ids.masked_fill(~past_key_values.attention_mask, -1)
                offset = previous.max(-1).values + 1 if past_length else torch.zeros(batch, dtype=torch.long, device=x.device)
                current_positions = current_positions + offset.unsqueeze(-1)
            current_positions = current_positions.clamp_min(0)
        else:
            if position_ids.shape != (batch, length) or position_ids.dtype != torch.long or position_ids.device != x.device:
                raise ValueError("position_ids must be int64 [batch, current_length] on the input device")
            if bool((position_ids < 0).any()):
                raise ValueError("position_ids must be nonnegative")
            current_positions = position_ids
        key_positions = current_positions if past_key_values is None else torch.cat((past_key_values.position_ids, current_positions), dim=1)
        if past_key_values is None and attention_mask is None:
            mask, is_causal = None, True
        else:
            queries = torch.arange(past_length, key_length, device=x.device)
            keys = torch.arange(key_length, device=x.device)
            mask = (keys[None] <= queries[:, None])[None, None] & key_valid[:, None, None, :]
            is_causal = False
        return x, key_valid, current_positions, key_positions, mask, is_causal

    def forward(self, input_ids: Tensor | None = None, *, inputs_embeds: Tensor | None = None,
                attention_mask: Tensor | None = None, position_ids: Tensor | None = None,
                past_key_values: OLMoCache | None = None, use_cache: bool = False,
                return_logits: bool = True) -> OLMoOutput:
        """Full or chunked causal forward; positions count valid tokens by default.

        The mask covers cached plus current keys. Padding queries must be
        excluded from loss. Explicit current positions preserve offset RoPE.
        max_context_length documents native training context, not a hard cap.
        KV tensors stay attached and must not be mutated by the caller.
        """
        x, valid, positions, key_positions, mask, causal = self._prepare_inputs(
            input_ids, inputs_embeds, attention_mask, position_ids, past_key_values,
        )
        present = []
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values.key_values[i]
            x, pair = layer(x, past=past, query_positions=positions, key_positions=key_positions,
                            mask=mask, is_causal=causal, attention_backend=self.attention_backend)
            if use_cache:
                present.append(pair)
        hidden = self.norm(x)
        cache = OLMoCache(tuple(present), valid.clone(), key_positions.clone()) if use_cache else None
        return OLMoOutput(self.project_logits(hidden) if return_logits else None, hidden, cache)
