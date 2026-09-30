"""Bounded K4 loss geometry on fixed weights, without optimizer updates.

The four contributions reuse the campaign's prepared forward, dynamic losses
and canonical pass aggregation. A fifth, fresh forward uses CampaignObjective
itself as the joint-gradient reference. Gradients are kept on CPU; all geometry
reductions use bounded FP64 chunks. This is local FP32/no-jitter geometry, not
a reconstruction of a large-batch BF16 training update or an Adam update.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import math

import torch

from .campaign_losses import compute_dynamic_nextlat_loss_sums
from .campaign_training import CampaignObjective
from .fbt_training import FBTNextLatLosses, aggregate_pass_losses
from .lm_training import TERMS
from .nextlat import NextLatLosses


CONTRIBUTIONS = ("ce_first", "ce_later", "latent", "kl")
GRADIENTS = (*CONTRIBUTIONS, "joint")
COMPONENTS = ("all", "backbone", "fusion", "predictor")
RECONSTRUCTION_RELATIVE_L2_LIMIT = 1e-4


def parameter_component(name: str) -> str:
    if name.startswith("predictor."):
        return "predictor"
    if name.startswith("backbone.fusion."):
        return "fusion"
    if name.startswith("backbone."):
        return "backbone"
    raise ValueError(f"Unknown campaign parameter ownership: {name}")


def weighted_feedback_terms(losses: FBTNextLatLosses,
                            normalization: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Split existing K4 aggregate metadata without changing loss semantics.

    ``normalization`` contains the adapter's actual device-resident coefficients
    (objective weight / diagnostic count for world size one). It is deliberately
    not recomputed from scalar loss values or from a common denominator.
    """
    if (not isinstance(losses, FBTNextLatLosses) or losses.pass_loss_policy != "campaign_v1"
            or len(losses.pass_losses) != 4 or set(normalization) != set(TERMS)):
        raise ValueError("Feedback decomposition requires campaign_v1 K4 losses")
    ce = losses.term_pass_coefficients["ce"]
    if ce != (.5, 1/6, 1/6, 1/6) or any(
            losses.term_pass_coefficients[t] != (.25,)*4 for t in ("latent", "kl")):
        raise ValueError("Unexpected campaign pass coefficients")
    return {
        "ce_first": losses.pass_losses[0].sums["ce"] * ce[0] * normalization["ce"],
        "ce_later": sum(p.sums["ce"] * c for p, c in zip(losses.pass_losses[1:], ce[1:]))
                    * normalization["ce"],
        "latent": losses.sums["latent"] * normalization["latent"],
        "kl": losses.sums["kl"] * normalization["kl"],
    }


def _prepared_terms(adapter):
    output = adapter.forward_layout.forward(adapter.batch.input_ids, adapter.mode,
                                            feedback_noise=adapter.feedback_noise)
    passes = tuple(NextLatLosses(
        compute_dynamic_nextlat_loss_sums(hidden, output.embeddings,
            adapter.model.backbone.readout_weight, adapter.batch.input_ids,
            adapter.model.predictor, adapter.model.config, adapter.loss_layout,
            enabled=adapter.model.enabled), dict(adapter.counts), dict(adapter.weights))
        for hidden in output.pass_hidden_states)
    losses = aggregate_pass_losses(passes, gamma=adapter.model.gamma,
                                    pass_loss_policy=adapter.model.pass_loss_policy)
    return weighted_feedback_terms(losses, dict(zip(TERMS, adapter.coefficients)))


def _validate_adapter(adapter):
    if not isinstance(adapter, CampaignObjective):
        raise TypeError("Expected an existing CampaignObjective")
    adapter.validate_execution()
    if (adapter.world_size != 1 or adapter.config.precision != "fp32"
            or not adapter.mode.enabled or adapter.mode.num_passes != 4 or adapter.mode.beta != 1.0
            or adapter.mode.feedback_jitter != 0 or adapter.feedback_noise is not None
            or not adapter.model.enabled or adapter.counts != adapter.global_counts
            or any(adapter.counts[t] <= 0 or adapter.weights[t] <= 0 for t in TERMS)):
        raise ValueError("Probe requires K4 beta=1 FP32/no-jitter, world_size=1 and positive local diagnostic denominators")
    if torch.is_autocast_enabled(adapter.device.type):
        raise ValueError("Diagnostic autocast must be disabled")
    if torch.distributed.is_initialized():
        raise ValueError("Gradient probe is local and does not participate in DDP")
    parameters = dict(adapter.model.named_parameters())
    if not parameters or any(not p.requires_grad or p.grad is not None for p in parameters.values()):
        raise ValueError("All model parameters must require gradients and start with grad=None")
    for name in parameters:
        parameter_component(name)
    return parameters


def _geometry(gram):
    norms = {name: math.sqrt(max(0., float(gram[i, i]))) for i, name in enumerate(GRADIENTS)}
    dots = {a: {b: float(gram[i, j]) for j, b in enumerate(GRADIENTS)}
            for i, a in enumerate(GRADIENTS)}
    cosines = {a: {b: (max(-1., min(1., dots[a][b]/(norms[a]*norms[b])))
                       if norms[a] and norms[b] else None) for b in GRADIENTS} for a in GRADIENTS}
    ce_squared = float(gram[:2, :2].sum())
    aux_squared = float(gram[2:4, 2:4].sum())
    ce_norm, aux_norm = math.sqrt(max(0., ce_squared)), math.sqrt(max(0., aux_squared))
    dot = float(gram[:2, 2:4].sum())
    return {"norms": norms, "dot_products": dots, "cosines": cosines,
        "ce_vs_aux": {"ce_norm": ce_norm, "aux_norm": aux_norm, "dot": dot,
            "cosine": max(-1., min(1., dot/(ce_norm*aux_norm))) if ce_norm and aux_norm else None,
            "opposes": dot < 0, "cosine_defined": bool(ce_norm and aux_norm)}}


def gradient_accounting(gradients: Mapping[str, Mapping[str, torch.Tensor | None]],
                        parameter_sizes: Mapping[str, int], *, chunk_elements: int = 262144) -> dict:
    """Reduce five CPU gradient dictionaries; absent gradients mean exact zeros.

    No model-sized concatenation or FP64 copy is made. Reconstruction uses an
    explicit FP64 sum of the four measured FP32 component VJPs, compared with
    the independently differentiated joint objective. Only the global relative
    L2 residual is gated; per-component residuals are descriptive.
    """
    if (set(gradients) != set(GRADIENTS) or not parameter_sizes
            or type(chunk_elements) is not int or chunk_elements <= 0):
        raise ValueError("Require five gradient records and a positive chunk size")
    if any(set(record) != set(parameter_sizes) for record in gradients.values()):
        raise ValueError("Gradient parameter ownership differs")
    totals = {g: {"gram": torch.zeros((5, 5), dtype=torch.float64), "difference_squared": 0.,
                  "max_abs_difference": 0., "parameter_tensors": 0, "parameter_elements": 0}
              for g in COMPONENTS}
    for name, size in parameter_sizes.items():
        if type(size) is not int or size <= 0:
            raise ValueError("Invalid parameter size")
        component = parameter_component(name)
        values = [gradients[key][name] for key in GRADIENTS]
        for value in values:
            if value is not None and (value.device.type != "cpu" or value.dtype != torch.float32
                    or value.numel() != size or value.requires_grad):
                raise ValueError("Expected detached FP32 CPU gradients of matching size")
        flat = [None if value is None else value.reshape(-1) for value in values]
        gram = torch.zeros((5, 5), dtype=torch.float64)
        squared, maximum = 0., 0.
        for start in range(0, size, chunk_elements):
            end = min(size, start+chunk_elements)
            chunk = torch.stack([torch.zeros(end-start, dtype=torch.float64) if value is None
                                 else value[start:end].double() for value in flat])
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("Nonfinite diagnostic gradient")
            gram += chunk @ chunk.T
            difference = chunk[:4].sum(dim=0)-chunk[4]
            squared += float(difference.dot(difference))
            maximum = max(maximum, float(difference.abs().max()))
        for group in ("all", component):
            item = totals[group]
            item["gram"] += gram
            item["difference_squared"] += squared
            item["max_abs_difference"] = max(item["max_abs_difference"], maximum)
            item["parameter_tensors"] += 1
            item["parameter_elements"] += size
    groups = {}
    for group, values in totals.items():
        row = _geometry(values["gram"])
        residual = math.sqrt(values["difference_squared"])
        norm = row["norms"]["joint"]
        row.update(parameter_tensors=values["parameter_tensors"], parameter_elements=values["parameter_elements"],
            reconstruction={"difference_norm": residual, "joint_norm": norm,
                "relative_l2_to_joint": residual/norm if norm else None,
                "max_abs_difference": values["max_abs_difference"],
                "zero_joint_and_zero_difference": norm == 0 and residual == 0})
        groups[group] = row
    reconstruction = groups["all"]["reconstruction"]
    relative = reconstruction["relative_l2_to_joint"]
    checks = {
        "finite_gradients": True,
        "global_reconstruction_within_limit": (relative <= RECONSTRUCTION_RELATIVE_L2_LIMIT if relative is not None
                                               else reconstruction["zero_joint_and_zero_difference"]),
        "first_ce_fusion_zero": groups["fusion"]["norms"]["ce_first"] == 0,
        "first_ce_predictor_zero": groups["predictor"]["norms"]["ce_first"] == 0,
        "later_ce_predictor_zero": groups["predictor"]["norms"]["ce_later"] == 0,
    }
    return {"components": groups, "checks": checks, "passed": all(checks.values()),
        "reconstruction_relative_l2_limit": RECONSTRUCTION_RELATIVE_L2_LIMIT,
        "reconstruction_gate_scope": "Global only; component residuals descriptive; no elementwise acceptance grid",
        "geometry_precision": "FP64 CPU chunk reductions over FP32 gradients", "chunk_elements": chunk_elements}


def feedback_gradient_probe(adapter: CampaignObjective, *, publish: Callable[[dict], None] | None = None,
                            chunk_elements: int = 262144) -> dict:
    """Five fresh forwards/backwards on immutable weights; return JSON metadata.

    Caller owns checkpoint/fixture pins, full-state integrity, execution backend
    and final artifact persistence. This helper never changes precision flags,
    trainability, optimizer state or .grad buffers. ``publish`` receives scalar
    progress after each completed contribution. Four CPU model-sized gradient
    sets remain live, plus the fifth during final accounting (at most ~25.4GB
    for 1.27B FP32 parameters; structurally absent gradients use no storage).
    """
    parameters = _validate_adapter(adapter)
    versions = {name: p._version for name, p in parameters.items()}
    snapshots, objectives = {}, {}
    sizes = {name: p.numel() for name, p in parameters.items()}
    for selected in GRADIENTS:
        adapter.validate_execution()
        with torch.autocast(adapter.device.type, enabled=False):
            if selected == "joint":
                objective = adapter()["objective"]
            else:
                terms = _prepared_terms(adapter)
                objective = terms[selected]
                del terms
            value = float(objective.detach())
            if not math.isfinite(value):
                raise ValueError("Nonfinite diagnostic objective")
            gradients = torch.autograd.grad(objective, tuple(parameters.values()), allow_unused=True)
        snapshots[selected] = {name: None if gradient is None else gradient.detach().to("cpu", copy=True)
                               for name, gradient in zip(parameters, gradients)}
        objectives[selected] = value
        del objective, gradients
        if publish is not None:
            publish({"stage": "gradient_contribution_completed", "contribution": selected,
                "objective": value, "completed_backwards": len(snapshots),
                "cpu_gradient_bytes": sum(v.numel()*v.element_size() for row in snapshots.values()
                                          for v in row.values() if v is not None)})
    result = gradient_accounting(snapshots, sizes, chunk_elements=chunk_elements)
    del snapshots
    adapter.validate_execution()
    integrity = (all(p.grad is None and p._version == versions[name] for name, p in parameters.items()))
    if not integrity:
        raise AssertionError("Diagnostic changed parameter versions or .grad buffers")
    result.update(schema="olmo-feedback-gradient-probe-v1", objectives=objectives,
        objective_decomposition_difference=sum(objectives[t] for t in CONTRIBUTIONS)-objectives["joint"],
        counts=dict(adapter.counts), global_counts=dict(adapter.global_counts), weights=dict(adapter.weights),
        coefficients={t: float(adapter.coefficients[i]) for i, t in enumerate(TERMS)},
        pass_weights={"ce": [.5, 1/6, 1/6, 1/6], "latent": [.25]*4, "kl": [.25]*4},
        forward_backward_pairs=5, optimizer_updates=0, parameter_versions_and_grad_buffers_unchanged=integrity,
        precision="fp32_no_jitter", world_size=1,
        scope="Local fixed-batch loss geometry; not cohort gradient norms, BF16 equivalence or Adam updates")
    return result
