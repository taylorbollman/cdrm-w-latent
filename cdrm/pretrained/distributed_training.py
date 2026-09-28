"""Forward and objective preparation for a future distributed LM runner.

This module performs no collectives and makes no two-GPU readiness claim.
The caller supplies counts summed over all ranks and all microbatches in one
optimizer update. Default DDP averages gradients, so each rank differentiates
``world_size * local_sum / global_count`` independently for CE, latent and KL.
Clipping belongs after gradient reduction; optimizer, recovery, CUDA graphs,
collective failure handling and sharding remain the runner's responsibility.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn

from .fbt_training import FBTNextLatLM
from .lm_training import TERMS, _counts
from .nextlat import NextLatBatch, NextLatLosses
from .olmo_fbt import FBTMode


def validate_microbatch_inputs(model: FBTNextLatLM, batch: NextLatBatch, *,
                               backbone_kwargs=None, tensor_inputs=None,
                               training: bool | None = None) -> dict:
    """Validate rank-local tensor inputs separately from shared configuration.

    Currently the only input is ``feedback_noise``: one [B,T-1,D] FP32 tensor
    per active feedback pass. These tensors are never configuration and must
    never enter object collectives or checkpoint metadata. Eager DDP calls this
    for *every* microbatch in coordinated preflight before its first forward.
    The adapter repeats validation for callers outside that trainer.

    ``training=True`` lets trainer preflight validate its upcoming train-mode
    execution without changing module modes before all ranks accept the update.
    Empty local slots still supply correctly shaped finite noise (usually zeros).
    """
    if not isinstance(model, FBTNextLatLM):
        raise TypeError("Expected canonical FBTNextLatLM")
    kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
    if "feedback_noise" in kwargs:
        raise ValueError("feedback_noise belongs in per-microbatch tensor_inputs, not shared backbone_kwargs")
    if tensor_inputs is None:
        inputs = {}
    elif isinstance(tensor_inputs, Mapping):
        inputs = dict(tensor_inputs)
    else:
        raise TypeError("Each microbatch tensor_inputs entry must be a mapping or None")
    if set(inputs) - {"feedback_noise"}:
        raise ValueError("Unknown per-microbatch tensor input; only feedback_noise is supported")
    mode = kwargs.get("mode", FBTMode())
    if not isinstance(mode, FBTMode):
        raise TypeError("mode must be FBTMode")
    if training is None:
        training = model.backbone.training
    if type(training) is not bool:
        raise TypeError("training must be boolean")
    active = training and mode.enabled and mode.num_passes > 1 and mode.beta > 0 and mode.feedback_jitter > 0
    noise = inputs.get("feedback_noise")
    if not active:
        if noise is not None:
            raise ValueError("feedback_noise is only accepted for active training jitter")
        return {}
    if not isinstance(noise, (tuple, list)) or len(noise) != mode.num_passes - 1:
        raise ValueError("Training feedback jitter requires one external noise tensor per feedback pass")
    expected_shape = (*batch.input_ids.shape[:1], batch.input_ids.shape[1] - 1, model.config.model_dim)
    weight = model.backbone.readout_weight
    for value in noise:
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape
                or value.device != weight.device or value.dtype != weight.dtype or value.requires_grad):
            raise ValueError("feedback_noise must match shifted feedback shape/device/dtype and have no gradient")
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 1).any()):
            raise ValueError("feedback_noise must be finite unit noise in [-1, 1]")
    return {"feedback_noise": tuple(noise)}


def sum_objective_counts(counts: Sequence[Mapping[str, int]]) -> dict[str, int]:
    """Combine precomputed counts without conflating tasks, passes or ranks.

    This pure helper is also useful for a one-process accumulation reference.
    It is not a replacement for the future cross-rank count all-reduce.
    """
    records = [_counts(record) for record in counts]
    if not records:
        raise ValueError("At least one objective-count record is required")
    return {term: sum(record[term] for record in records) for term in TERMS}


def _normalization_contract(local_counts, weights, global_counts, world_size):
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    local_counts, global_counts = _counts(local_counts), _counts(global_counts)
    if set(weights) != set(TERMS) or any(
        isinstance(weights[t], bool) or not isinstance(weights[t], (int, float))
        or not math.isfinite(weights[t]) or weights[t] < 0 for t in TERMS
    ):
        raise ValueError("Objective weights must be finite nonnegative ce/latent/kl values")
    if any(local_counts[t] > global_counts[t] for t in TERMS):
        raise ValueError("A local objective count exceeds its global denominator")
    if any(not weights[t] and global_counts[t] for t in TERMS):
        raise ValueError("Disabled objectives must have zero global counts")
    if not any(weights[t] and global_counts[t] for t in TERMS):
        raise ValueError("Update has no valid positively weighted objective")
    return local_counts, global_counts


def ddp_normalized_objective(result: NextLatLosses, *,
                             global_counts: Mapping[str, int], world_size: int) -> torch.Tensor:
    """Preserve canonical FBT/NextLat sums under default DDP averaging.

    FBT's selected per-term pass weighting is already in ``result.sums``;
    do not multiply counts by K or average per-rank means. A locally empty but
    globally active term keeps its canonical attached zero, allowing that rank
    to participate without inventing parameter gradients. Globally empty terms
    are omitted. An entirely empty global update is rejected on every rank.

    This eager preparation helper does not check finite tensor values or run
    collectives. A distributed runner must coordinate health checks before
    stepping any rank, and ensure all ranks agree on weights and world size.
    It assumes DDP's default average, not a custom sum-only communication hook.
    """
    if not isinstance(result, NextLatLosses):
        raise TypeError("Expected canonical NextLatLosses")
    _, denominators = _normalization_contract(
        result.counts, result.weights, global_counts, world_size)
    if set(result.sums) != set(TERMS) or any(
        not isinstance(result.sums[t], torch.Tensor) or result.sums[t].ndim != 0
        for t in TERMS
    ):
        raise ValueError("Objective sums must be scalar ce/latent/kl tensors")
    # Keep the existing multiply's operation order at world_size=1, including
    # each term's independent denominator and omission of globally empty terms.
    return sum(result.sums[t] * (result.weights[t] * world_size / denominators[t])
               for t in TERMS if denominators[t] and result.weights[t])


class ObjectiveForwardAdapter(nn.Module):
    """Expose the complete differentiable objective through ``forward``.

    A future DDP caller must invoke ``ddp(batch, ...)``, never
    ``ddp.module.model.loss_sums(...)``. Only ``output['objective']`` retains an
    autograd graph. Detached diagnostic losses must not tell DDP's unused-
    parameter traversal that an otherwise unused loss branch will be backwarded.

    The canonical model is registered exactly once as ``adapter.model``. Build
    and save the optimizer/checkpoint against that original model to retain its
    existing parameter names and runtime configuration. Wrapping neither copies
    parameters nor adds model state; an adapter state_dict has a ``model.``
    prefix and is deliberately not the canonical checkpoint format.

    This is an eager forward adapter, not a StaticFBTTraining or graph adapter.
    Caller autocast and checkpoint/runtime flags continue to govern the original
    model. Dynamic masks/modes or zero coefficients may change used parameters;
    no static-gradient-participation promise is made.
    """

    def __init__(self, model: FBTNextLatLM):
        super().__init__()
        if not isinstance(model, FBTNextLatLM):
            raise TypeError("ObjectiveForwardAdapter requires FBTNextLatLM")
        self.model = model
        self.training = model.training

    def forward(self, batch: NextLatBatch, *, global_counts: Mapping[str, int],
                world_size: int = 1, backbone_kwargs=None, tensor_inputs=None) -> dict:
        local_counts = self.model.counts(batch)
        weights = self.model.objective_weights()
        _, denominators = _normalization_contract(local_counts, weights, global_counts, world_size)
        inputs = validate_microbatch_inputs(self.model, batch, backbone_kwargs=backbone_kwargs,
                                           tensor_inputs=tensor_inputs)
        kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
        result = self.model.loss_sums(batch, backbone_kwargs={**kwargs, **inputs})
        if result.counts != local_counts or result.weights != weights:
            raise ValueError("Forward objectives differ from precomputed counts or weights")
        objective = ddp_normalized_objective(result, global_counts=denominators, world_size=world_size)
        return {
            "objective": objective,
            "loss_sums": {t: result.sums[t].detach() for t in TERMS},
            "counts": dict(result.counts),
            "global_counts": denominators,
            "objective_weights": dict(result.weights),
            "pass_loss_sums": tuple({t: loss.sums[t].detach() for t in TERMS}
                                    for loss in result.pass_losses),
            "pass_coefficients": result.pass_coefficients,
            "term_pass_coefficients": result.term_pass_coefficients,
            "pass_loss_policy": result.pass_loss_policy,
        }
