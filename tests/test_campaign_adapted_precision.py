"""CPU checks that adapted-state comparison keeps the original NF experiment."""
import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.olmo_campaign_adapted_precision import case_health, fixed_contract_checks
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_precision_bridge import capture_passes, forward_geometry
from scripts.olmo_campaign_precision_components import component_backward, record_gradients
from scripts.olmo_campaign_recurrence_precision import arm_contract, fixture_pins, state_pins


def test_changed_backbone_and_saved_scale_keep_current_nf_objective_and_predictor():
    torch.set_num_threads(1)
    model, recipe, _, ids, eos = construct(SimpleNamespace(scale="tiny", length=16), "NF", torch.device("cpu"))
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
        length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]

    def info():
        return {"initial_state": state_pins(model), "contract": arm_contract(model, recipe),
                "fixture_inputs": fixture_pins(fixtures), "fixture_metadata": global_fixture_metadata(model, fixtures)}

    cold = info()
    with torch.no_grad():
        next(model.backbone.backbone.parameters()).add_(0.001)
        model.backbone.fusion.output_scale.mul_(0.7)
    adapted = info()
    assert adapted["initial_state"]["backbone"] != cold["initial_state"]["backbone"]
    assert adapted["initial_state"]["fusion"] != cold["initial_state"]["fusion"]
    assert all(fixed_contract_checks(adapted, cold).values())
    with capture_passes(model, fixtures) as observed:
        metrics = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    grads, _ = record_gradients(model, scope="Tiny CPU health contract only")
    geometry = forward_geometry(observed)
    assert all(case_health(metrics, grads, geometry, observed, adapted["fixture_metadata"]).values())
    assert grads["groups"]["backbone"]["norm"] > 0
    assert grads["groups"]["fusion"]["norm"] > 0
    assert grads["groups"]["predictor"]["norm"] == 0
    assert info() == adapted
    # The supposedly untouched predictor and noise are both part of the gate.
    altered = copy.deepcopy(adapted)
    altered["initial_state"]["predictor"] = {"parameters": "different", "buffers": "different"}
    assert not fixed_contract_checks(altered, cold)["predictor_state_exact"]
    altered = copy.deepcopy(adapted)
    altered["fixture_inputs"] = "different noise"
    assert not fixed_contract_checks(altered, cold)["fixture_inputs_exact"]
    # Missing materialized gradients or accidental auxiliary objectives cannot
    # be reported as healthy merely because the loss was finite.
    bad = copy.deepcopy(grads)
    bad["participation_intact"] = False
    assert not case_health(metrics, bad, geometry, observed, adapted["fixture_metadata"])["gradient_participation"]
    bad["groups"]["predictor"]["norm"] = 1.0
    assert not case_health(metrics, bad, geometry, observed, adapted["fixture_metadata"])["predictor_zero_cotangent"]


@pytest.mark.parametrize("key", ["input_tokens", "microbatches", "documents", "counts"])
def test_case_health_rejects_changed_global_normalization_metadata(key):
    metadata = {"input_tokens": 29, "microbatches": 2, "documents": 4,
                "counts": {"ce": 25, "latent": 25, "kl": 21}}
    metrics = {**copy.deepcopy(metadata), "objective": 1.0, "loss_sums": {"ce": 25.0}}
    metrics[key] = {} if key == "counts" else metadata[key] + 1
    gradients = {"finite": True, "participation_intact": True, "groups": {"predictor": {"norm": 0.0}}}
    geometry = [{"hidden": {"finite": True}, "total_incoming_cotangent": {"finite": True}}]
    observed = [{"pass_hidden_states": [None]*4}]*2
    assert not case_health(metrics, gradients, geometry, observed, metadata)["counts_exact"]
