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
    """K includes the first pass, with a versioned, explicit RT policy.

    Disabled FBT executes exactly one standalone RT/ordinary pass. Beta zero
    disables feedback independently of RT. ``ordinary-v1`` preserves historical
    checkpoints, including ordinary K1; ``configured-rt-v1`` executes rt_mode on
    every pass. Training jitter requires externally supplied unit-uniform noise,
    so assigning noise to logical examples is independent of microbatch shape.
    """

    enabled: bool = True
    num_passes: int = 2
    beta: float = 1.0
    rt_mode: RTMode = RTMode(())
    first_pass_policy: str = "ordinary-v1"
    feedback_jitter: float = 0.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be boolean")
        if type(self.num_passes) is not int or self.num_passes < 1:
            raise ValueError("num_passes must be a positive integer")
        if not isinstance(self.rt_mode, RTMode):
            raise TypeError("rt_mode must be an RTMode")
        if self.first_pass_policy not in ("ordinary-v1", "configured-rt-v1"):
            raise ValueError("first_pass_policy must be ordinary-v1 or configured-rt-v1")
        if (isinstance(self.feedback_jitter, bool)
                or not isinstance(self.feedback_jitter, (float, int))
                or not math.isfinite(self.feedback_jitter) or self.feedback_jitter < 0):
            raise ValueError("feedback_jitter must be finite and nonnegative")
        object.__setattr__(self, "feedback_jitter", float(self.feedback_jitter))
        object.__setattr__(self, "beta", _unit_interval(self.beta, "beta"))

    @property
    def initial_rt_mode(self):
        if self.enabled and self.first_pass_policy == "ordinary-v1":
            return RTMode(())
        return self.rt_mode


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

    def _validate_feedback_noise(self, feedback_noise, mode, shape, device, dtype, *, check_values=True):
        """Validate explicit unit noise outside capture; no model-owned RNG.

        Values must be finite and in [-1, 1]. The caller samples independently
        from Uniform[-1,1] keyed to logical examples/pass and reuses the same
        draw for recomputation. Shape is [B,T-1,D], one tensor per feedback pass.
        Tensor metadata checks are capture-safe; value checks are external.
        """
        active = (self.training and mode.enabled and mode.num_passes > 1
                  and mode.beta > 0 and mode.feedback_jitter > 0)
        if not active:
            if feedback_noise is not None:
                raise ValueError("feedback_noise is only accepted for active training jitter")
            return None
        if not isinstance(feedback_noise, (tuple, list)) or len(feedback_noise) != mode.num_passes - 1:
            raise ValueError("Training feedback jitter requires one external noise tensor per feedback pass")
        for noise in feedback_noise:
            if (not isinstance(noise, Tensor) or tuple(noise.shape) != tuple(shape)
                    or noise.device != device or noise.dtype != dtype or noise.requires_grad):
                raise ValueError("feedback_noise must match shifted feedback shape/device/dtype and have no gradient")
            if check_values and (not bool(torch.isfinite(noise).all()) or bool((noise.abs() > 1).any())):
                raise ValueError("feedback_noise must be finite unit noise in [-1, 1]")
        return tuple(feedback_noise)

    @staticmethod
    def _jitter_feedback(previous_hidden, mode, eligible, noise):
        if noise is None:
            return previous_hidden
        return previous_hidden + torch.where(eligible.unsqueeze(-1),
            noise * mode.feedback_jitter, torch.zeros_like(noise))

    def forward(self, input_ids=None, *, inputs_embeds=None, attention_mask=None,
                document_ids=None, position_ids=None, mode: FBTMode = FBTMode(),
                return_logits=True, past_key_values=None, use_cache=False,
                full_valid_causal: bool = False, feedback_noise=None,
                right_padded_causal: bool = False) -> FBTOutput:
        """Finite passes, optionally using implicit causal attention for full rows.

        ``full_valid_causal`` is an explicit eager-dispatch option, not permission
        to ignore padding or document boundaries. Prove all tokens valid after
        normal input/document validation, then omit the redundant all-valid
        mask in every fresh stack call. This permits Flash SDPA dispatch while
        preserving supplied RoPE positions. Cached/padded sequences cannot opt in.

        ``right_padded_causal`` additionally permits valid-prefix rows, including
        empty rows. Ordinary attention uses an implicit causal mask; native RT
        retains token validity and every stack zeros invalid output queries.
        This opt-in requires the native tiled backbone and no packed documents.

        Nonzero training ``mode.feedback_jitter`` requires external unit noise;
        evaluation is deterministic and rejects supplied noise. No RNG state is
        consumed by this forward path, including activation recomputation.
        """
        if type(full_valid_causal) is not bool:
            raise TypeError("full_valid_causal must be boolean")
        if type(right_padded_causal) is not bool:
            raise TypeError("right_padded_causal must be boolean")
        if right_padded_causal and full_valid_causal:
            raise ValueError("Choose either right_padded_causal or full_valid_causal")
        if right_padded_causal:
            from .olmo_tiled import OLMoTiledRTForCausalLM
            if not isinstance(self.backbone, OLMoTiledRTForCausalLM):
                raise ValueError("right_padded_causal requires the native tiled backbone")
        if not isinstance(mode, FBTMode):
            raise TypeError("mode must be an FBTMode")
        if past_key_values is not None or use_cache:
            raise ValueError("Finite FBT passes do not accept caches; use forward_online explicitly")
        self._validate_rt_mode(mode.rt_mode)
        if right_padded_causal:
            embeddings, valid, positions, _, _, _ = self.backbone._prepare_right_padded_inputs(
                input_ids, inputs_embeds, attention_mask, position_ids)
        else:
            embeddings, valid, positions, _, _, _ = self.backbone._prepare_inputs(
                input_ids, inputs_embeds, attention_mask, position_ids, None,
            )
        documents = self._documents(valid, document_ids)
        noise = self._validate_feedback_noise(feedback_noise, mode,
            embeddings[:, 1:].shape, embeddings.device, embeddings.dtype)
        if full_valid_causal and not bool(valid.all()):
            raise ValueError("full_valid_causal requires all tokens valid; padding is unsupported")
        stack_mask = None if full_valid_causal else valid
        stack_options = {"right_padded_causal": True} if right_padded_causal else {}
        hidden = self._stack(embeddings, mode.initial_rt_mode, attention_mask=stack_mask,
                             position_ids=positions, **stack_options).last_hidden_state
        states = [hidden]
        if mode.enabled:
            eligible = valid[:, 1:] & valid[:, :-1] & (documents[:, 1:] == documents[:, :-1])
            for pass_index in range(1, mode.num_passes):
                previous = self._jitter_feedback(hidden[:, :-1], mode, eligible,
                    None if noise is None else noise[pass_index - 1])
                suffix = self._blend(previous, embeddings[:, 1:], mode.beta, eligible)
                fused_inputs = torch.cat((embeddings[:, :1], suffix), dim=1)
                hidden = self._stack(fused_inputs, mode.rt_mode, attention_mask=stack_mask,
                                     position_ids=positions, **stack_options).last_hidden_state
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
