"""Probe gradients must remain real even though warmup training freezes weights."""
from types import SimpleNamespace

import pytest
import torch

from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, state_pins
from scripts.olmo_fusion_startup_probe import measure_case
from scripts import olmo_fusion_startup_probe as probe


@pytest.fixture(autouse=True)
def cpu_rng_observer(monkeypatch):
    # The executable requires CUDA; unit tests substitute only its CUDA RNG
    # observation, while exercising real CPU model forward/backward arithmetic.
    monkeypatch.setattr(probe, "rng_snapshot", lambda: torch.get_rng_state().clone())
    monkeypatch.setattr(probe, "rng_unchanged", lambda state: torch.equal(state, torch.get_rng_state()))


def setup():
    torch.set_num_threads(1)
    model, recipe, _, ids, eos = construct(
        SimpleNamespace(scale="tiny", length=16), "NF", torch.device("cpu"))
    model.backbone.backbone.attention_precision = "mixed"
    flags = {name: getattr(model.backbone.backbone, name) for name in RUNTIME_FLAGS}
    fixtures = [fixture_for_update(recipe, model.config.model_dim, r, 0,
        length=16, token_ids=ids, eos_id=eos, batch_size=2) for r in range(2)]
    return model, recipe, flags, fixtures


def test_repeated_cpu_fp32_probe_is_exact_and_keeps_backbone_sensitivity():
    model, recipe, flags, fixtures = setup()
    before = state_pins(model)
    first, reference, states = measure_case(model, recipe, fixtures,
        path=FP32, original_flags=flags)
    second, _, _ = measure_case(model, recipe, fixtures, path=FP32,
        original_flags=flags, reference_gradients=reference, reference_forward=states)
    assert first["passed"] and second["passed"]
    geometry = second["gradients_vs_fp32"]["geometry"]
    assert geometry["backbone"]["reference_gradient_norm"] > 0
    assert geometry["fusion"]["reference_gradient_norm"] > 0
    assert geometry["predictor"]["reference_is_zero"]
    assert geometry["all"]["difference_norm"] == 0
    assert state_pins(model) == before
    assert all(x["hidden"]["difference_norm"] == 0 for x in second["forward_vs_fp32"])
    assert first["forward_fingerprints"] == second["forward_fingerprints"]


def test_probe_rejects_frozen_training_contract_instead_of_reporting_zero_backbone_error():
    model, recipe, flags, fixtures = setup()
    for parameter in model.backbone.backbone.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="trainability"):
        measure_case(model, recipe, fixtures, path=FP32, original_flags=flags)


def test_probe_requires_complete_pair_of_comparison_references():
    model, recipe, flags, fixtures = setup()
    with pytest.raises(ValueError, match="together"):
        measure_case(model, recipe, fixtures, path=FP32, original_flags=flags,
            reference_forward=[])
