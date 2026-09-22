"""Prepared, fixed-selection execution of the canonical NextLat loss.

Layout preparation and validation are deliberately outside CUDA capture. The
tensor-only execution path replaces Boolean selection with fixed integer
gathers; it keeps the predictor subset/order, vocabulary chunks, detachments,
and each objective's independent denominator from :mod:`nextlat`.

A layout fixes the shape, device, document structure, masks, configuration and
enabled flag. Token values may change. It is not a general dynamic-mask graph
interface: validate every replacement batch before replay, and prepare/capture
a new layout when any fixed structure changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F

from .nextlat import (
    NextLatBatch, NextLatConfig, NextLatLosses, NextLatPredictor,
    _ce_chunk, _chunked_sum, _kl_chunk, _objective_weights, _validate_batch,
    build_nextlat_masks,
)


_TERMS = ("ce", "latent", "kl")
_STRUCTURE = ("valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask")
_INDICES = ("ce_source_indices", "needed_source_indices", "latent_target_indices",
            "latent_prediction_indices", "kl_teacher_indices", "kl_prediction_indices")


def _cpu_batch(batch: NextLatBatch) -> NextLatBatch:
    """Snapshot caller-owned tensors before host-side structural validation."""
    if not isinstance(batch, NextLatBatch):
        raise TypeError("batch must be NextLatBatch")
    # Check the original devices/types before copying to a common CPU device;
    # otherwise a malformed mixed-device input would be silently accepted.
    ids = batch.input_ids
    if not isinstance(ids, Tensor) or ids.ndim != 2 or ids.dtype != torch.long or min(ids.shape) < 1:
        raise ValueError("input_ids must be a nonempty int64 [batch, sequence] tensor")
    values = {"input_ids": ids.detach().cpu().clone()}
    for name in _STRUCTURE:
        value = getattr(batch, name)
        if value is None and name not in ("valid_mask", "document_ids"):
            values[name] = None
            continue
        dtype = torch.long if name == "document_ids" else torch.bool
        if not isinstance(value, Tensor) or value.shape != ids.shape or value.dtype != dtype or value.device != ids.device:
            raise ValueError(f"{name} must be {dtype} with input_ids shape/device")
        values[name] = value.detach().cpu().clone()
    result = NextLatBatch(**values)
    _validate_batch(result, one_document_per_row=True)
    return result


@dataclass(frozen=True)
class PreparedNextLatLayout:
    """Fixed integer selections; construct through :meth:`from_batch`.

    Index tensors are graph inputs and must not be modified. ``counts`` and
    ``weights`` return fresh dictionaries. ``validate_batch`` also detects an
    in-place change to an exposed index tensor, before an unsafe replay.
    """

    shape: tuple[int, int]
    device: torch.device
    config: NextLatConfig
    enabled: bool
    ce_source_indices: Tensor = field(repr=False)
    needed_source_indices: Tensor = field(repr=False)
    latent_target_indices: Tensor = field(repr=False)
    latent_prediction_indices: Tensor = field(repr=False)
    kl_teacher_indices: Tensor = field(repr=False)
    kl_prediction_indices: Tensor = field(repr=False)
    _count_values: tuple[int, int, int] = field(repr=False)
    _structure: tuple[Tensor | None, ...] = field(repr=False)
    _index_versions: tuple[int, ...] = field(repr=False)
    _structure_versions: tuple[int | None, ...] = field(repr=False)

    @classmethod
    def from_batch(cls, batch: NextLatBatch, config: NextLatConfig, *, enabled: bool = True):
        if not isinstance(config, NextLatConfig):
            raise TypeError("config must be NextLatConfig")
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean")
        cpu = _cpu_batch(batch)
        masks = build_nextlat_masks(cpu)
        weights = _objective_weights(config, enabled)
        counts = tuple(int(masks[name].sum()) if weights[name] else 0 for name in _TERMS)
        latent = masks["latent"] if weights["latent"] else torch.zeros_like(masks["latent"])
        kl = masks["kl"] if weights["kl"] else torch.zeros_like(masks["kl"])
        # For T1 the canonical path skips auxiliary selection entirely. Avoid
        # padding its empty triple mask into a nonexistent pair dimension.
        kl_pair = F.pad(kl, (0, 1)) if cpu.input_ids.shape[1] > 1 else torch.zeros_like(latent)
        needed = latent | kl_pair
        full = torch.arange(cpu.input_ids.numel(), dtype=torch.long).reshape(cpu.input_ids.shape)
        indices = (
            full[:, :-1][masks["ce"]],
            full[:, :-1][needed],
            full[:, 1:][latent],
            torch.nonzero(latent[needed], as_tuple=False).flatten(),
            full[:, 1:-1][kl],
            torch.nonzero(kl_pair[needed], as_tuple=False).flatten(),
        )
        indices = tuple(value.to(batch.input_ids.device).contiguous() for value in indices)
        structure = tuple(getattr(cpu, name) for name in _STRUCTURE)
        return cls(tuple(batch.input_ids.shape), batch.input_ids.device, config, enabled,
                   *indices, counts, structure, tuple(value._version for value in indices),
                   tuple(None if value is None else value._version for value in structure))

    @property
    def counts(self) -> dict[str, int]:
        return dict(zip(_TERMS, self._count_values))

    @property
    def weights(self) -> dict[str, float]:
        return _objective_weights(self.config, self.enabled)

    def validate_batch(self, batch: NextLatBatch) -> None:
        """Host-side guard for replacement token values, never call in capture."""
        if not isinstance(batch, NextLatBatch):
            raise TypeError("batch must be NextLatBatch")
        if batch.input_ids.shape != self.shape or batch.input_ids.device not in (torch.device("cpu"), self.device):
            raise ValueError("Prepared layout requires the same batch shape and CPU or execution device")
        self.validate_integrity()
        cpu = _cpu_batch(batch)
        for name, expected in zip(_STRUCTURE, self._structure):
            actual = getattr(cpu, name)
            if (actual is None) != (expected is None) or (actual is not None and not torch.equal(actual, expected)):
                raise ValueError(f"Prepared layout {name} changed; prepare a new layout and graph")
    def validate_integrity(self) -> None:
        """Reject in-place layout changes before replay; no device scalar reads."""
        if tuple(getattr(self, name)._version for name in _INDICES) != self._index_versions:
            raise ValueError("Prepared layout index tensors were modified; prepare a new layout and graph")
        versions = tuple(None if value is None else value._version for value in self._structure)
        if versions != self._structure_versions:
            raise ValueError("Prepared layout structure snapshots were modified; prepare a new layout and graph")

    def validate_execution(self, input_ids: Tensor, config: NextLatConfig, *, enabled: bool) -> None:
        """Metadata-only checks; no tensor reads or synchronization in capture."""
        if type(enabled) is not bool:
            raise TypeError("enabled must be boolean")
        if config != self.config or enabled != self.enabled:
            raise ValueError("Prepared layout configuration/enabled flag changed")
        if input_ids.shape != self.shape or input_ids.device != self.device or input_ids.dtype != torch.long:
            raise ValueError("input_ids must match the prepared shape/device and be int64")


def compute_static_nextlat_loss_sums(
    hidden_states: Tensor, token_embeddings: Tensor, readout_weight: Tensor,
    input_ids: Tensor, predictor: NextLatPredictor | None, config: NextLatConfig,
    layout: PreparedNextLatLayout, *, enabled: bool = True,
) -> NextLatLosses:
    """Execute the unchanged selected-position objective with static gathers.

    The caller validates the complete batch with ``layout.validate_batch``
    before replay. The execution path receives tokens only, so cannot verify
    document/mask metadata itself. Flattening complete [B,T,D] tensors avoids
    allocating a dense copy of every shifted [B,T-1,D] slice at large batches.
    """
    if not isinstance(layout, PreparedNextLatLayout):
        raise TypeError("layout must be PreparedNextLatLayout")
    layout.validate_execution(input_ids, config, enabled=enabled)
    shape = (*layout.shape, config.model_dim)
    if hidden_states.shape != shape or token_embeddings.shape != shape:
        raise ValueError("Hidden states and embeddings must have [batch, sequence, model_dim] shape")
    if hidden_states.device != layout.device or token_embeddings.device != layout.device:
        raise ValueError("Hidden states, embeddings, and layout must share device")
    if readout_weight.ndim != 2 or readout_weight.shape[1] != config.model_dim or readout_weight.device != layout.device:
        raise ValueError("readout_weight must have [vocabulary, model_dim] shape on the hidden-state device")
    counts, weights = layout.counts, layout.weights
    hidden = hidden_states.reshape(-1, config.model_dim)
    embeddings = token_embeddings.reshape(-1, config.model_dim)
    tokens = input_ids.reshape(-1)
    zero = hidden_states.sum() * 0.0
    sums = {name: zero for name in _TERMS}
    if counts["ce"]:
        sums["ce"] = _chunked_sum(
            _ce_chunk, hidden.index_select(0, layout.ce_source_indices), readout_weight,
            tokens.index_select(0, layout.ce_source_indices + 1), config.vocab_chunk_size,
            weight_second=True,
        )
    if counts["latent"] or counts["kl"]:
        if predictor is None:
            raise ValueError("An enabled auxiliary objective requires a predictor")
        predicted = predictor(hidden.index_select(0, layout.needed_source_indices),
                              embeddings.index_select(0, layout.needed_source_indices + 1))
        if counts["latent"]:
            selected = predicted.index_select(0, layout.latent_prediction_indices)
            targets = hidden.index_select(0, layout.latent_target_indices).detach()
            sums["latent"] = F.smooth_l1_loss(selected.float(), targets.float(), reduction="none").mean(-1).sum()
        if counts["kl"]:
            sums["kl"] = _chunked_sum(
                _kl_chunk, predicted.index_select(0, layout.kl_prediction_indices),
                hidden.index_select(0, layout.kl_teacher_indices).detach(), readout_weight.detach(),
                config.vocab_chunk_size, weight_second=False,
            )
    return NextLatLosses(sums, counts, weights)
