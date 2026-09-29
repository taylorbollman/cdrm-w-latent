"""Fixed-capacity NextLat sums with replaceable selection masks.

The shape, objective weights and predictor participation stay fixed; selected
rows/positions can change between replays. All target counts remain host-side
metadata. A caller applies its device-resident normalization coefficients to
these raw sums, and must not capture ``layout.counts`` as denominators.

This adapter deliberately computes unselected positions before masking their
individual losses. It trades additional arithmetic for a stable CUDA graph.
It does not provide attention isolation, token buffers, or a training runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .nextlat import (
    NextLatBatch, NextLatConfig, NextLatPredictor, _objective_weights,
    build_nextlat_masks,
)
from .static_nextlat import _cpu_batch


_TERMS = ("ce", "latent", "kl")
_BUFFERS = ("pair_source_indices", "pair_target_indices", "kl_prediction_indices",
            "kl_teacher_indices", "ce_weights", "latent_weights", "kl_weights")


def _buffer_signature(value):
    return (id(value), value.data_ptr(), tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device, value.requires_grad)


@dataclass(frozen=True)
class DynamicNextLatLayout:
    """Owned fixed-size buffers; use ``load_batch`` between complete replays.

    Exposed tensor buffers must not be replaced or modified directly. External
    validation catches mutations before the runner executes/replays its graph.
    ``load_batch`` copies masks in place and refreshes counts without replacing
    graph input storage. The input token buffer belongs to the caller.

    T>=3 is an explicit initial scope, including completely masked dummy rows.
    Packed documents require the explicit continuous-stream-v1 config; CE then
    crosses document boundaries while auxiliary losses keep same-document masks.
    """

    shape: tuple[int, int]
    device: torch.device
    config: NextLatConfig
    enabled: bool
    pair_source_indices: Tensor = field(repr=False)
    pair_target_indices: Tensor = field(repr=False)
    kl_prediction_indices: Tensor = field(repr=False)
    kl_teacher_indices: Tensor = field(repr=False)
    ce_weights: Tensor = field(repr=False)
    latent_weights: Tensor = field(repr=False)
    kl_weights: Tensor = field(repr=False)
    _count_values: tuple[int, int, int] = field(repr=False)
    _buffer_signatures: tuple = field(repr=False)
    _buffer_versions: tuple[int, ...] = field(repr=False)

    @classmethod
    def from_batch(cls, batch: NextLatBatch, config: NextLatConfig, *,
                   enabled: bool = True, device=None):
        if not isinstance(config, NextLatConfig):
            raise TypeError("config must be NextLatConfig")
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean")
        if config.dropout:
            raise ValueError("Dynamic NextLat requires zero predictor dropout")
        cpu = _cpu_batch(batch, document_policy=config.document_policy)
        shape = tuple(cpu.input_ids.shape)
        if shape[1] < 3:
            raise ValueError("Dynamic NextLat requires sequence length >= 3")
        device = batch.input_ids.device if device is None else torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Dynamic NextLat requires CPU or CUDA storage")
        # Normalize an unspecified CUDA index to its actual allocation device.
        pair_source = torch.arange(cpu.input_ids.numel(), device="cpu").reshape(shape)[:, :-1].reshape(-1)
        pair_target = pair_source + 1
        kl_prediction = torch.arange(shape[0]*(shape[1]-1), device="cpu").reshape(shape[0], shape[1]-1)[:, :-1].reshape(-1)
        kl_teacher = torch.arange(cpu.input_ids.numel(), device="cpu").reshape(shape)[:, 1:-1].reshape(-1)
        weights = _objective_weights(config, enabled)
        masks = build_nextlat_masks(cpu, document_policy=config.document_policy)
        mask_buffers = tuple((masks[term] if weights[term] else torch.zeros_like(masks[term]))
                             .reshape(-1).float() for term in _TERMS)
        counts = tuple(int(value.sum()) for value in mask_buffers)
        buffers = tuple(value.to(device=device).contiguous().clone() for value in (
            pair_source, pair_target, kl_prediction, kl_teacher, *mask_buffers))
        return cls(shape, buffers[0].device, config, enabled, *buffers, counts,
                   tuple(_buffer_signature(value) for value in buffers),
                   tuple(value._version for value in buffers))

    @property
    def counts(self) -> dict[str, int]:
        """Current loaded selection counts; read outside captured execution."""
        return dict(zip(_TERMS, self._count_values))

    @property
    def weights(self) -> dict[str, float]:
        return _objective_weights(self.config, self.enabled)

    def validate_integrity(self):
        """Metadata/version guard outside capture; no device scalar reads."""
        buffers = tuple(getattr(self, name) for name in _BUFFERS)
        if (tuple(_buffer_signature(value) for value in buffers) != self._buffer_signatures
                or tuple(value._version for value in buffers) != self._buffer_versions):
            raise ValueError("Dynamic NextLat buffers were modified outside load_batch")

    def _prepare_batch(self, batch):
        self.validate_integrity()
        if not isinstance(batch, NextLatBatch):
            raise TypeError("batch must be NextLatBatch")
        if (batch.input_ids.shape != self.shape
                or batch.input_ids.device not in (torch.device("cpu"), self.device)):
            raise ValueError("Dynamic NextLat requires its fixed shape and CPU or execution device")
        cpu = _cpu_batch(batch, document_policy=self.config.document_policy)
        masks = build_nextlat_masks(cpu, document_policy=self.config.document_policy)
        weights = self.weights
        buffers = tuple((masks[term] if weights[term] else torch.zeros_like(masks[term]))
                        .reshape(-1).float() for term in _TERMS)
        return buffers, tuple(int(value.sum()) for value in buffers)

    def validate_batch(self, batch):
        """Check a proposed replacement without mutating any input storage."""
        self._prepare_batch(batch)

    def load_batch(self, batch):
        """Validate on host, then copy all three masks into owned storage."""
        masks, counts = self._prepare_batch(batch)
        for name, mask in zip(("ce_weights", "latent_weights", "kl_weights"), masks):
            getattr(self, name).copy_(mask)
        object.__setattr__(self, "_count_values", counts)
        object.__setattr__(self, "_buffer_versions",
                           tuple(getattr(self, name)._version for name in _BUFFERS))

    def validate_execution(self, input_ids, config, *, enabled):
        """Capture-safe metadata validation; call integrity checks externally."""
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean")
        if config != self.config or enabled != self.enabled:
            raise ValueError("Dynamic NextLat configuration/enabled flag changed")
        if (input_ids.shape != self.shape or input_ids.dtype != torch.long
                or input_ids.device != self.device):
            raise ValueError("input_ids must match dynamic layout shape/device and be int64")


def _weighted_ce_chunk(states, readout, targets, mask):
    # Every projection uses the full native vocabulary. Mask each position's
    # FP32 CE before summing, never multiply an already reduced scalar by mask.
    safe_targets = torch.where(mask.bool(), targets, torch.zeros_like(targets))
    per_position = F.cross_entropy(F.linear(states, readout).float(), safe_targets, reduction="none")
    return (per_position * mask).sum()


def _weighted_kl_chunk(predicted, teacher, readout, mask):
    with torch.no_grad():
        log_teacher = F.log_softmax(F.linear(teacher, readout).float(), dim=-1)
    log_student = F.log_softmax(F.linear(predicted, readout).float(), dim=-1)
    per_position = F.kl_div(log_student, log_teacher, log_target=True, reduction="none").sum(-1)
    return (per_position * mask).sum()


def _weighted_chunked_sum(function, states, other, third, mask, chunk_size, *, weight_second):
    result = states.sum() * 0.0
    for start in range(0, states.shape[0], chunk_size):
        end = start + chunk_size
        args = ((states[start:end], other, third[start:end], mask[start:end]) if weight_second else
                (states[start:end], other[start:end], third, mask[start:end]))
        if torch.is_grad_enabled() and any(value.requires_grad for value in args):
            loss = checkpoint(function, *args, use_reentrant=False)
        else:
            loss = function(*args)
        result = result + loss
    return result


def compute_dynamic_nextlat_loss_sums(
    hidden_states: Tensor, token_embeddings: Tensor, readout_weight: Tensor,
    input_ids: Tensor, predictor: NextLatPredictor | None, config: NextLatConfig,
    layout: DynamicNextLatLayout, *, enabled: bool = True,
) -> dict[str, Tensor]:
    """Return graph-safe raw CE/latent/KL sums with dense fixed-capacity work.

    Counts/normalization do not occur in this tensor-only body. The caller must
    validate/load a complete batch and buffer integrity outside replay. Enabled
    objectives execute even for an empty local selection: participating weights
    receive explicit zero gradients instead of disappearing from DDP buckets.
    Globally absent objectives require a separate optimizer participation policy.
    """
    if not isinstance(layout, DynamicNextLatLayout):
        raise TypeError("layout must be DynamicNextLatLayout")
    layout.validate_execution(input_ids, config, enabled=enabled)
    shape = (*layout.shape, config.model_dim)
    if hidden_states.shape != shape or token_embeddings.shape != shape:
        raise ValueError("Hidden states and embeddings must have [batch, sequence, model_dim] shape")
    if hidden_states.device != layout.device or token_embeddings.device != layout.device:
        raise ValueError("Hidden states, embeddings, and layout must share device")
    if (readout_weight.ndim != 2 or readout_weight.shape[1] != config.model_dim
            or readout_weight.device != layout.device):
        raise ValueError("readout_weight must have [vocabulary, model_dim] shape on the layout device")
    hidden = hidden_states.reshape(-1, config.model_dim)
    embeddings = token_embeddings.reshape(-1, config.model_dim)
    tokens = input_ids.reshape(-1)
    source = hidden.index_select(0, layout.pair_source_indices)
    zero = hidden_states.sum() * 0.0
    sums = {"ce": _weighted_chunked_sum(_weighted_ce_chunk, source, readout_weight,
                tokens.index_select(0, layout.pair_target_indices), layout.ce_weights,
                config.effective_ce_chunk_size, weight_second=True), "latent": zero, "kl": zero}
    weights = layout.weights
    if weights["latent"] or weights["kl"]:
        if predictor is None:
            raise ValueError("An enabled auxiliary objective requires a predictor")
        predicted = predictor(source, embeddings.index_select(0, layout.pair_target_indices))
        if weights["latent"]:
            targets = hidden.index_select(0, layout.pair_target_indices).detach()
            per_pair = F.smooth_l1_loss(predicted.float(), targets.float(), reduction="none").mean(-1)
            sums["latent"] = (per_pair * layout.latent_weights).sum()
        if weights["kl"]:
            sums["kl"] = _weighted_chunked_sum(_weighted_kl_chunk,
                predicted.index_select(0, layout.kl_prediction_indices),
                hidden.index_select(0, layout.kl_teacher_indices).detach(),
                readout_weight.detach(), layout.kl_weights, config.vocab_chunk_size, weight_second=False)
    return sums
