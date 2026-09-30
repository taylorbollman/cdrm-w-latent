"""Explicit KL-only objective branches of an existing campaign checkpoint.

This module neither loads state nor changes checkpoint validation. The caller
must restore a parent with its original strict configuration/fingerprint, audit
the complete boundary, then change KL before constructing any objective adapter
or graph. A branch resume instead configures its declared KL before loading its
own exact identity; it does not migrate the parent again.
"""
from __future__ import annotations

from dataclasses import asdict, replace

import torch

from cdrm.pretrained.lm_training import optimizer_ownership, parameter_layout
from cdrm.pretrained.nextlat import NextLatConfig
from scripts.olmo_campaign_execution import CURSOR_SCHEMA
from scripts.olmo_campaign_manifest import pin


SCHEMA = "olmo-kl-objective-transition-v1"
ALLOWED_KL = (1.0, 0.1)


def _weight(value):
    if type(value) not in (int, float) or value not in ALLOWED_KL:
        raise ValueError("KL branch weight must be explicitly 1.0 or 0.1")
    return float(value)


def _model_configs(model):
    if (not model.enabled or model.predictor is None
            or not isinstance(model.config, NextLatConfig)
            or not isinstance(model.predictor.config, NextLatConfig)
            or model.gamma != 1 or model.pass_loss_policy != "campaign_v1"
            or model.config.lambda_latent != 1
            or model.predictor.config.lambda_latent != 1):
        raise ValueError("Require enabled campaign NextLat with unchanged latent/CE weights")
    old = _weight(model.config.lambda_kl)
    if (_weight(model.predictor.config.lambda_kl) != old
            or model.objective_weights() != {"ce": 1., "latent": 1., "kl": old}):
        raise ValueError("Model/predictor KL configuration or actual objective differs")
    return model.config, model.predictor.config


def _ownership(model):
    def tensor(value):
        return (id(value), value.data_ptr(), value._version, tuple(value.shape),
                tuple(value.stride()), value.dtype, value.device, value.requires_grad)
    return {
        "parameters": tuple((name, tensor(p), id(p.grad) if p.grad is not None else None)
                            for name, p in model.named_parameters()),
        "buffers": tuple((name, tensor(v)) for name, v in model.named_buffers()),
        "modules": tuple((name, id(module), type(module), module.training)
                         for name, module in model.named_modules()),
    }


def set_kl_weight(model, weight):
    """Replace only two immutable config fields, at an uncaptured boundary.

No tensor arithmetic, optimizer/RNG operations, or storage replacement occurs.
The lightweight ownership/version checks here complement the caller's complete
before/after tensor, Adam, schedule, counters, cursor and RNG boundary hashes.
External adapters cannot be discovered reliably: the caller must ensure none
is live. Existing CampaignObjective validation rejects a later config change.
"""
    weight = _weight(weight)
    model_config, predictor_config = _model_configs(model)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("KL transition requires a cleared-gradient boundary")
    before = _ownership(model)
    old = {"model": model_config.to_dict(), "predictor": predictor_config.to_dict()}
    weights_before = dict(model.objective_weights())
    # Preserve each config independently: historical imports may differ in
    # document policy, which is unrelated to the requested scalar transition.
    replacements = (replace(model_config, lambda_kl=weight),
                    replace(predictor_config, lambda_kl=weight))
    model.config, model.predictor.config = replacements
    try:
        new = {"model": model.config.to_dict(), "predictor": model.predictor.config.to_dict()}
        after = _ownership(model)
        checks = {key+"_unchanged": before[key] == after[key] for key in before}
        checks["only_lambda_kl_changed"] = all(
            {k: v for k, v in old[owner].items() if k != "lambda_kl"}
            == {k: v for k, v in new[owner].items() if k != "lambda_kl"}
            for owner in old)
        checks["actual_objective_matches"] = model.objective_weights() == {
            "ce": 1., "latent": 1., "kl": weight}
        if not all(checks.values()):
            raise ValueError("KL transition changed unrelated ownership/configuration")
    except BaseException:
        model.config, model.predictor.config = model_config, predictor_config
        raise
    return {"schema": SCHEMA, "configuration_before": old,
            "configuration_after": new, "weights_before": weights_before,
            "weights_after": dict(model.objective_weights()), "checks": checks,
            "scope": "Config-only ownership/version checks; caller audits complete restored state",
            "before_graph_construction_required": True}


def branch_model_contract(model, recipe, optimizer=None, *, kl_weight):
    """Accepted ownership checks with only the declared positive KL exception.

This is a narrow copy of olmo_campaign_execution.model_contract. Keeping it
separate preserves the accepted runtime's all-one objective assertion.
"""
    kl_weight = _weight(kl_weight)
    if recipe.arm not in ("NF", "NFR") or model.pass_loss_policy != "campaign_v1":
        raise ValueError("KL continuation supports NF/NFR campaign ownership only")
    _model_configs(model)
    params = dict(model.named_parameters())
    components = {key: set() for key in ("backbone", "fusion", "predictor")}
    for name in params:
        if name.startswith("backbone.backbone."):
            components["backbone"].add(name)
        elif name.startswith("backbone.fusion."):
            components["fusion"].add(name)
        elif name.startswith("predictor."):
            components["predictor"].add(name)
        else:
            raise ValueError("Unaccounted model parameter")
    expected = components["backbone"] | components["fusion"] | components["predictor"]
    active = {name for name, p in params.items() if p.requires_grad}
    weights = {"ce": 1., "latent": 1., "kl": kl_weight}
    if (active != expected or not components["backbone"] or len(components["fusion"]) != 2
            or not components["predictor"] or not model.enabled or model.predictor is None
            or model.objective_weights() != weights or model.gamma != 1
            or model.config.document_policy != recipe.document_policy
            or any(p.dtype != torch.float32 for p in params.values())
            or model.backbone.readout_weight is not model.backbone.token_embeddings.weight):
        raise ValueError("Active parameter ownership or declared branch objective differs")
    ownership = None
    if optimizer is not None:
        ownership = optimizer_ownership(model, optimizer)
        if {group.get("component") for group in optimizer.param_groups} != set(components):
            raise ValueError("Optimizer component telemetry differs")
        for group, names in zip(optimizer.param_groups, ownership):
            if set(names)-components[group["component"]]:
                raise ValueError("Optimizer group includes another component")
    return {"arm": recipe.arm, "mode": asdict(recipe.mode()), "weights": weights,
            "resident_parameters": sum(p.numel() for p in params.values()),
            "trainable_parameters": sum(params[n].numel() for n in active),
            "component_parameters": {k: sum(params[n].numel() for n in names)
                                     for k, names in components.items()},
            "dormant_fusion_parameters": 0, "parameter_layout": parameter_layout(model),
            "optimizer_ownership": ownership, "tied_readout": True, "cursor_schema": CURSOR_SCHEMA}


def declared_recipe(recipe, *, kl_weight, parent_manifest_sha256):
    """Serialize the operative objective and inherited state without a fresh claim."""
    kl_weight = _weight(kl_weight)
    pin(parent_manifest_sha256)
    if recipe.arm not in ("NF", "NFR"):
        raise ValueError("KL continuation requires NF/NFR")
    result = recipe.to_dict()
    result["auxiliary"]["kl"] = kl_weight
    result["optimizer_state"] = "inherited_exact_parent_checkpoint"
    result["objective_transition"] = {
        "schema": SCHEMA, "parent_manifest_sha256": parent_manifest_sha256,
        "parent_kl_weight": 1., "kl_weight": kl_weight,
        "changed_field": "lambda_kl", "same_weight_control": kl_weight == 1.,
        "retained_state": ["model", "optimizer", "scheduler", "counters", "data_cursor", "rng"],
        "resume": "exact_branch_identity_without_reapplying_parent_transition",
    }
    return result
