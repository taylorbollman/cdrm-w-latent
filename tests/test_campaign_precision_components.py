"""CPU contracts for the bounded precision/component diagnostic."""
import math

import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_campaign_ddp_probe import fixture_for_update
from scripts.olmo_campaign_fp32_localize import configure_full_fp32
from scripts.olmo_campaign_precision_components import (
    OBJECTIVES, TERMS, component_backward, gradient_geometry, parse_args, pass_sums, record_gradients,
)


def test_pass_weights_are_literal_campaign_weights_and_keep_gradients():
    passes = [{term: torch.tensor(float(index+1), requires_grad=True) for term in TERMS} for index in range(4)]
    sums = pass_sums(passes)
    assert float(sums["ce"].detach()) == pytest.approx(2.)
    assert float(sums["latent"].detach()) == pytest.approx(2.5)
    assert float(sums["kl"].detach()) == pytest.approx(2.5)
    sum(sums.values()).backward()
    for index, row in enumerate(passes):
        assert float(row["ce"].grad) == pytest.approx(.5 if index == 0 else 1/6)
        assert float(row["latent"].grad) == pytest.approx(.25)
        assert float(row["kl"].grad) == pytest.approx(.25)
    assert pass_sums([passes[0]])["ce"] == passes[0]["ce"]
    with pytest.raises(ValueError):
        pass_sums(passes[:2])


def test_gradient_geometry_measures_norm_error_and_angle_and_aligns_zero_groups():
    reference = {"backbone.weight": torch.tensor([1., 0.]),
                 "backbone.fusion.weight": torch.zeros(2), "predictor.weight": torch.tensor([0., 2.])}
    actual = {"backbone.weight": torch.tensor([0., 1.]),
              "backbone.fusion.weight": torch.zeros(2), "predictor.weight": torch.tensor([0., 4.])}
    result = gradient_geometry(actual, reference)
    assert result["backbone"]["angle_degrees"] == pytest.approx(90.)
    assert result["backbone"]["relative_l2"] == pytest.approx(math.sqrt(2))
    assert result["predictor"]["cosine"] == pytest.approx(1.)
    assert result["predictor"]["norm_ratio"] == pytest.approx(2.)
    assert result["fusion"]["reference_is_zero"] and result["fusion"]["actual_is_zero"]
    assert result["fusion"]["cosine"] is None
    assert result["all"]["relative_l2"] == pytest.approx(math.sqrt(6/5))
    assert result["all"]["reference_gradient_norm"] == pytest.approx(math.sqrt(5))
    with pytest.raises(ValueError, match="names"):
        gradient_geometry({}, reference)
    with pytest.raises(ValueError, match="shape/dtype"):
        gradient_geometry({key: value.double() for key, value in actual.items()}, reference)


def test_components_sum_to_combined_without_reconfiguring_model_and_dense_matches_sparse():
    torch.set_num_threads(1)
    torch.manual_seed(20260929)
    recipe = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0, 1))
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32",
        tile_backend="eager", backward_tile_backend="eager", backward_memory="recompute",
        reuse_rope=True, kv_only_writes=True)
    model = build_campaign_model(base, recipe).train()
    configure_full_fp32(model)
    original_config = model.config
    original_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0, length=8,
        token_ids=list(range(2, 59)), eos_id=60, batch_size=2) for rank in range(2)]
    reference = {}
    metrics_by_term = {}
    for objective in OBJECTIVES:
        metrics_by_term[objective] = component_backward(model, recipe, fixtures,
            precision="fp32", layout="sparse", objective=objective)
        record, reference[objective] = record_gradients(model, save_cpu=True, scope="CPU oracle")
        assert record["finite"] and record["participation_intact"]
        dense = component_backward(model, recipe, fixtures, precision="fp32", layout="prepared", objective=objective)
        record, _ = record_gradients(model, reference[objective], scope="CPU oracle")
        assert record["finite"] and record["participation_intact"]
        assert record["comparison"]["all_parameters_close"]
        assert not record["comparison"]["gating"]
        assert dense["objective"] == pytest.approx(metrics_by_term[objective]["objective"], rel=3e-6, abs=1e-6)
        if objective == "ce":
            assert all(torch.count_nonzero(value) == 0 for name, value in reference[objective].items() if name.startswith("predictor."))
        assert model.config is original_config and model.enabled
        assert model.objective_weights() == {"ce": 1., "latent": 1., "kl": 1.}
    for name in reference["combined"]:
        torch.testing.assert_close(sum(reference[term][name] for term in TERMS), reference["combined"][name], atol=3e-6, rtol=3e-5)
    assert sum(metrics_by_term[t]["objective"] for t in TERMS) == pytest.approx(metrics_by_term["combined"]["objective"], rel=1e-6)
    assert all(torch.equal(p, original_weights[name]) for name, p in model.named_parameters())


@pytest.mark.parametrize("extra", [["--steps", "5"], ["--length", "1024"], ["--tiny"], ["--batch-size", "8"]])
def test_cli_scope_remains_bounded(extra):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/tmp/precision-components", *extra])
