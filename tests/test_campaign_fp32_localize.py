"""Bounded scope/backend/acceptance guards for the FP32 localization probe."""
import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_campaign_ddp_probe import canonical_backward, fixture_for_update, prepared_backward
from scripts.olmo_campaign_fp32_localize import acceptance, configure_full_fp32, parse_args
from scripts.olmo_campaign_graph_probe import compare_metrics
from scripts.olmo_campaign_probe import gradient_record


def test_configuration_selects_full_fp32_math_eager_without_changing_weights():
    torch.set_num_threads(1)
    recipe = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0, 1))
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="sdpa",
        attention_precision="mixed", tile_backend="eager", backward_tile_backend="eager",
        backward_memory="recompute", reuse_rope=True, kv_only_writes=True)
    model = build_campaign_model(base, recipe).train()
    weights = {n: p.clone() for n, p in model.named_parameters()}
    record = configure_full_fp32(model)
    assert record["runtime_flags"]["attention_precision"] == "fp32"
    assert record["ordinary_sdpa_dispatch"] == "forced MATH"
    assert record["runtime_flags"]["tile_backend"] == "eager"
    assert record["runtime_flags"]["backward_tile_backend"] == "eager"
    assert not record["ddp"] and not record["cuda_graphs"] and record["optimizer_updates"] == 0
    assert all(torch.equal(p, weights[n]) for n, p in model.named_parameters())
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
        length=8, token_ids=list(range(2, 59)), eos_id=60, batch_size=2) for rank in range(2)]
    canonical = canonical_backward(model, recipe, fixtures, precision="fp32")
    _, reference = gradient_record(model, save_cpu=True)
    prepared = prepared_backward(model, recipe, fixtures, precision="fp32")
    gradients, _ = gradient_record(model, reference)
    result = acceptance(compare_metrics(prepared, canonical), gradients,
        weights_unchanged=all(torch.equal(p, weights[n]) for n, p in model.named_parameters()), rng_preserved=True)
    assert result["passed"]
    model.to(torch.float64)
    with pytest.raises(ValueError, match="FP32 master"):
        configure_full_fp32(model)


def test_acceptance_requires_loss_gradient_finiteness_and_state_guards():
    metrics = {"passed": True}
    gradients = {"finite": True, "comparison": {"all_parameters_close": True}}
    assert acceptance(metrics, gradients, weights_unchanged=True, rng_preserved=True)["passed"]
    for loss, gradient, weights, rng in (
        ({"passed": False}, gradients, True, True),
        (metrics, {"finite": False, "comparison": {"all_parameters_close": True}}, True, True),
        (metrics, {"finite": True, "comparison": {"all_parameters_close": False}}, True, True),
        (metrics, gradients, False, True), (metrics, gradients, True, False),
    ):
        assert not acceptance(loss, gradient, weights_unchanged=weights, rng_preserved=rng)["passed"]


@pytest.mark.parametrize("extra", [["--length", "1024"], ["--tiny"], ["--updates", "2"], ["--precision", "bf16_mixed"]])
def test_cli_does_not_allow_unbounded_or_different_diagnostic(extra):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/tmp/fp32-localize", *extra])
