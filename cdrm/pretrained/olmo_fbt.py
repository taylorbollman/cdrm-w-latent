"""Bounded full-bandwidth feedback over the unchanged native OLMo stack.

Finite passes are Jacobi updates from the previous pass. ``forward_online``
instead consumes the freshly completed previous-token state. Neither API
turns the training-only NextLat predictor into an inference recurrence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import torch
from torch import Tensor, nn
from .olmo import OLMoCache, OLMoForCausalLM
from .olmo_recurrent import OLMoRTForCausalLM
from .recurrent import RTMode


def _unit_interval(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a real configuration scalar")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class FBTConfig:
    """New-branch settings; native backbone norms and initialization are intact.

    Explicit epsilon and FP32 RMS reductions are OLMo adaptations of the
    pinned gate_product source. The scale is a fixed checkpoint buffer measured
    once over every element of the original native embedding/readout matrix.
    """

    norm_eps: float = 1e-5
    seed: int = 20260922

    def __post_init__(self):
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


@dataclass(frozen=True)
class FBTMode:
    """K includes the ordinary first pass; K1 is always ordinary when enabled.

    Disabled FBT executes exactly one standalone RT/ordinary pass. Beta zero
    disables feedback, independently of RT writes on any extra passes.
    """

    enabled: bool = True
    num_passes: int = 2
    beta: float = 1.0
    rt_mode: RTMode = RTMode(())

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be boolean")
        if type(self.num_passes) is not int or self.num_passes < 1:
            raise ValueError("num_passes must be a positive integer")
        if not isinstance(self.rt_mode, RTMode):
            raise TypeError("rt_mode must be an RTMode")
        object.__setattr__(self, "beta", _unit_interval(self.beta, "beta"))


@dataclass(frozen=True)
class FBTOnlineMode:
    """Exact online feedback has no pass count or ordinary bootstrap pass."""

    beta: float = 1.0
    rt_mode: RTMode = RTMode(())

    def __post_init__(self):
        if not isinstance(self.rt_mode, RTMode):
            raise TypeError("rt_mode must be an RTMode")
        object.__setattr__(self, "beta", _unit_interval(self.beta, "beta"))


class FBTGateProduct(nn.Module):
    """Asymmetric state projection and sigmoid token gate, with two RMS norms."""

    def __init__(self, embedding_weight: Tensor, config: FBTConfig):
        super().__init__()
        self.norm_eps = config.norm_eps
        width = embedding_weight.shape[1]
        # Meta construction plus an explicit generator avoids consuming any
        # caller CPU/CUDA RNG, including nn.Linear's default initialization.
        self.state_proj = nn.Linear(width, width, bias=False, device="meta")
        self.token_gate = nn.Linear(width, width, bias=False, device="meta")
        self.to_empty(device=embedding_weight.device)
        self.to(dtype=embedding_weight.dtype)
        generator = torch.Generator(device=embedding_weight.device)
        generator.manual_seed(config.seed)
        bound = math.sqrt(3.0 / width)
        nn.init.uniform_(self.state_proj.weight, -bound, bound, generator=generator)
        nn.init.uniform_(self.token_gate.weight, -bound, bound, generator=generator)
        with torch.no_grad():
            scale = embedding_weight.detach().float().square().mean().sqrt()
        if not bool(torch.isfinite(scale)) or not bool(scale > 0):
            raise ValueError("Native embedding RMS must be finite and positive")
        self.register_buffer("output_scale", scale)

    def _rms_norm(self, x):
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.norm_eps)
        return normalized.to(x.dtype)

    def forward(self, previous_hidden: Tensor, token_input: Tensor) -> Tensor:
        value = self.state_proj(previous_hidden)
        gate = torch.sigmoid(self.token_gate(self._rms_norm(token_input)))
        result = self._rms_norm(value * gate) * self.output_scale
        return result.to(token_input.dtype)


@dataclass
class FBTOutput:
    logits: Tensor | None
    last_hidden_state: Tensor
    pass_hidden_states: tuple[Tensor, ...]


@dataclass(frozen=True)
class FBTOnlineCache:
    """Attached exact-online history, never a finite-pass or ordinary prefix.

    Own immutable metadata and parameter/buffer versions reject stale histories.
    KV/previous-state tensors must not be mutated; unsupported ``.data`` writes
    are outside this contract, as with the existing RT caches.
    """

    backbone_cache: object
    previous_hidden: Tensor
    document_ids: Tensor
    mode: FBTOnlineMode
    tensor_versions: tuple[tuple, ...] = field(repr=False)
    storage_references: tuple[Tensor, ...] = field(repr=False)
    owner: int = field(repr=False)
    generation: int = field(repr=False)
    execution_context: tuple = field(repr=False)

    @property
    def attention_mask(self):
        return self.backbone_cache.attention_mask

    @property
    def position_ids(self):
        return self.backbone_cache.position_ids

    @property
    def sequence_length(self):
        return self.backbone_cache.sequence_length


@dataclass
class FBTOnlineOutput:
    logits: Tensor | None
    last_hidden_state: Tensor
    past_key_values: FBTOnlineCache | None = None


class OLMoFBT(nn.Module):
    """Shared native stack plus opt-in feedback, without backbone mutation.

    One document per row, optionally padded, is supported. Loss masks alone
    cannot isolate packed documents, so multi-document rows are rejected.
    Finite passes use fresh layer histories; caching belongs only to the
    explicitly separate exact-online API. That API requires an RT-capable
    backbone (ordinary execution uses an empty selection) for its stronger
    native cache provenance. Prefix-mode switching is unsupported.
    """

    def __init__(self, backbone: OLMoForCausalLM, fusion_config: FBTConfig = FBTConfig()):
        super().__init__()
        if not isinstance(backbone, OLMoForCausalLM):
            raise TypeError("backbone must be a native OLMo model")
        if not isinstance(fusion_config, FBTConfig):
            raise TypeError("fusion_config must be an FBTConfig")
        self.backbone = backbone
        self.fusion_config = fusion_config
        self.fusion = FBTGateProduct(backbone.readout_weight, fusion_config)
        self._cache_generation = 0

    @property
    def config(self):
        return self.backbone.config

    @property
    def token_embeddings(self):
        return self.backbone.token_embeddings

    @property
    def readout_weight(self):
        return self.backbone.readout_weight

    def project_logits(self, hidden_states):
        return self.backbone.project_logits(hidden_states)

    def _apply(self, fn, recurse=True):
        self._cache_generation = getattr(self, "_cache_generation", 0) + 1
        return super()._apply(fn, recurse=recurse)

    def _validate_rt_mode(self, mode):
        if any(index >= self.config.num_layers for index in mode.selected_layers):
            raise ValueError("selected_layers contains an index outside the model")
        if mode.selected_layers and not isinstance(self.backbone, OLMoRTForCausalLM):
            raise ValueError("Selected RT blocks require an RT-capable backbone")

    def _stack(self, embeddings, mode, **kwargs):
        if isinstance(self.backbone, OLMoRTForCausalLM):
            kwargs["mode"] = mode
        return self.backbone(inputs_embeds=embeddings, return_logits=False, **kwargs)

    @staticmethod
    def _documents(valid, document_ids):
        if document_ids is None:
            documents = torch.zeros_like(valid, dtype=torch.long)
        else:
            if document_ids.shape != valid.shape or document_ids.device != valid.device or document_ids.dtype != torch.long:
                raise ValueError("document_ids must be int64 with the full attention-mask shape/device")
            documents = document_ids
        if bool((documents[valid] < 0).any()):
            raise ValueError("Valid tokens require nonnegative document IDs")
        for row, active in zip(documents, valid):
            if row[active].unique().numel() > 1:
                raise ValueError("Packed multi-document rows are unsupported; attention would cross documents")
        return documents.masked_fill(~valid, -1)

    def _blend(self, previous_hidden, embeddings, beta, eligible):
        # The exact endpoint must bypass projection and both new norms.
        if beta == 0.0:
            return embeddings
        fused = self.fusion(previous_hidden, embeddings)
        blended = fused if beta == 1.0 else (1.0 - beta) * embeddings + beta * fused
        return torch.where(eligible.unsqueeze(-1), blended, embeddings)

    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None,
                document_ids=None, position_ids=None, mode: FBTMode = FBTMode(),
                return_logits=True, past_key_values=None, use_cache=False) -> FBTOutput:
        if not isinstance(mode, FBTMode):
            raise TypeError("mode must be an FBTMode")
        if past_key_values is not None or use_cache:
            raise ValueError("Finite FBT passes do not accept caches; use forward_online explicitly")
        self._validate_rt_mode(mode.rt_mode)
        embeddings, valid, positions, _, _, _ = self.backbone._prepare_inputs(
            input_ids, inputs_embeds, attention_mask, position_ids, None,
        )
        documents = self._documents(valid, document_ids)
        first_mode = RTMode(()) if mode.enabled else mode.rt_mode
        hidden = self._stack(embeddings, first_mode, attention_mask=valid,
                             position_ids=positions).last_hidden_state
        states = [hidden]
        if mode.enabled:
            eligible = valid[:, 1:] & valid[:, :-1] & (documents[:, 1:] == documents[:, :-1])
            for _ in range(1, mode.num_passes):
                suffix = self._blend(hidden[:, :-1], embeddings[:, 1:], mode.beta, eligible)
                fused_inputs = torch.cat((embeddings[:, :1], suffix), dim=1)
                hidden = self._stack(fused_inputs, mode.rt_mode, attention_mask=valid,
                                     position_ids=positions).last_hidden_state
                states.append(hidden)
        return FBTOutput(self.project_logits(hidden) if return_logits else None,
                         hidden, tuple(states))

    def _tensor_versions(self):
        return tuple((id(value), value._version, value.data_ptr(), value.dtype, value.device)
                     for value in (*self.parameters(), *self.buffers()))

    def _context(self, device):
        return (torch.is_autocast_enabled(device.type), torch.get_autocast_dtype(device.type),
                torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                self.backbone.attention_backend, getattr(self.backbone, "attention_precision", None),
                getattr(self.backbone, "_rt_cache_generation", 0), self.fusion_config,
                self.fusion.norm_eps)

    def forward_online(self, input_ids=None, *, inputs_embeds=None, attention_mask=None,
                       document_ids=None, position_ids=None,
                       mode: FBTOnlineMode = FBTOnlineMode(),
                       past_key_values: FBTOnlineCache | None = None,
                       use_cache=True, return_logits=True) -> FBTOnlineOutput:
        """Slow exact token sweep, with attached chunk-to-chunk feedback/KV.

        The attention mask and optional document IDs cover prefix plus current
        tokens; explicit positions cover current tokens only. No external
        ordinary-prefill cache or mode switch is accepted. This is a reference,
        not a production generation API or a finite-prefill equivalence claim.
        """
        if not isinstance(mode, FBTOnlineMode):
            raise TypeError("mode must be an FBTOnlineMode")
        if not isinstance(self.backbone, OLMoRTForCausalLM):
            raise ValueError("Exact online execution requires an RT-capable backbone for cache provenance; use an empty RT selection for ordinary attention")
        self._validate_rt_mode(mode.rt_mode)
        if past_key_values is not None and not isinstance(past_key_values, FBTOnlineCache):
            raise TypeError("past_key_values must be an FBTOnlineCache")
        past = past_key_values
        inner = None if past is None else past.backbone_cache
        embeddings, valid, positions, _, _, _ = self.backbone._prepare_inputs(
            input_ids, inputs_embeds, attention_mask, position_ids, inner,
            cache_type=OLMoCache if inner is None else type(inner),
        )
        versions, context = self._tensor_versions(), self._context(embeddings.device)
        if past is not None:
            if past.owner != id(self) or past.tensor_versions != versions:
                raise ValueError("Cached model/fusion weights or scale changed")
            if past.generation != self._cache_generation or past.execution_context != context:
                raise ValueError("Cached execution context or model generation changed")
            if past.mode != mode:
                raise ValueError("Cached online feedback/RT mode differs")
            if document_ids is None:
                row_document = past.document_ids.max(-1).values.clamp_min(0)
                document_ids = row_document[:, None].expand_as(valid)
        documents = self._documents(valid, document_ids)
        prefix_length = 0 if past is None else past.sequence_length
        if past is not None and not torch.equal(documents[:, :prefix_length], past.document_ids):
            raise ValueError("Cached document IDs cannot change")
        previous = None if past is None else past.previous_hidden
        states = []
        for index in range(embeddings.shape[1]):
            absolute = prefix_length + index
            current = embeddings[:, index:index + 1]
            if previous is not None:
                eligible = (valid[:, absolute:absolute + 1] & valid[:, absolute - 1:absolute]
                            & (documents[:, absolute:absolute + 1] == documents[:, absolute - 1:absolute]))
                current = self._blend(previous, current, mode.beta, eligible)
            output = self._stack(current, mode.rt_mode, attention_mask=valid[:, :absolute + 1],
                                 position_ids=positions[:, index:index + 1],
                                 past_key_values=inner, use_cache=True)
            previous, inner = output.last_hidden_state, output.past_key_values
            states.append(previous)
        hidden = torch.cat(states, dim=1)
        # Detached views hold the original storage without copying its bytes.
        # Consequently a direct child-module dtype/device roundtrip cannot
        # reclaim/reuse its old pointer and evade the cache signature.
        storage_references = tuple(value.detach() for value in (*self.parameters(), *self.buffers())) if use_cache else ()
        cache = FBTOnlineCache(inner, previous, documents.clone(), mode, versions,
                               storage_references, id(self), self._cache_generation, context) if use_cache else None
        return FBTOnlineOutput(self.project_logits(hidden) if return_logits else None, hidden, cache)
