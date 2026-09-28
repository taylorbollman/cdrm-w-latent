"""Campaign ownership, token clocks, and partition-invariant jitter contracts."""
import copy
from dataclasses import replace
import json

import pytest
import torch

from cdrm.pretrained.campaign_recipe import (
    ARMS, CampaignRecipe, CampaignTokenSchedule, build_campaign_adamw,
    build_campaign_model, feedback_noise_for_rows,
)
from cdrm.pretrained.lm_training import optimizer_ownership
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(1804)


def recipe(arm="NFR", **kwargs):
    return CampaignRecipe(arm, sequence_length=8, rt_layers=(0, 1), **kwargs)


def backbone():
    return OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
                                attention_precision="fp32")


@pytest.mark.parametrize("arm", ARMS)
def test_all_arms_preserve_backbone_and_have_explicit_contracts(arm):
    r = recipe(arm)
    base = backbone()
    before = {n: p.detach().clone() for n, p in base.named_parameters()}
    rng = torch.get_rng_state().clone()
    model = build_campaign_model(base, r)
    assert torch.equal(rng, torch.get_rng_state())
    for name, parameter in base.named_parameters():
        assert torch.equal(before[name], parameter)
    mode = r.mode()
    assert mode.num_passes == (4 if "F" in arm else 1)
    assert mode.rt_mode.selected_layers == ((0, 1) if "R" in arm else ())
    assert mode.first_pass_policy == "configured-rt-v1"
    assert model.pass_loss_policy == "campaign_v1"
    assert (model.predictor is not None) == ("N" in arm)
    assert all(p.requires_grad == ("F" in arm) for p in model.backbone.fusion.parameters())
    assert json.loads(json.dumps(r.to_dict())) == r.to_dict()


def test_optimizer_excludes_tied_embedding_and_owns_every_trainable_weight_once():
    r = recipe()
    model = build_campaign_model(backbone(), r)
    optimizer = build_campaign_adamw(model, r, fused=False)
    ownership = optimizer_ownership(model, optimizer)
    assert sum(map(len, ownership)) == len([p for p in model.parameters() if p.requires_grad])
    embed = model.backbone.readout_weight
    groups = [g for g in optimizer.param_groups if any(p is embed for p in g["params"])]
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0
    assert optimizer.defaults["eps"] == 1e-5
    assert not optimizer.state
    assert {g["component"] for g in optimizer.param_groups} == {"backbone", "fusion", "predictor"}
    old = {id(p): p.detach().clone() for p in model.parameters()}
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    optimizer.step()
    assert torch.equal(embed, old[id(embed)])
    for group in optimizer.param_groups:
        for p in group["params"]:
            expected = old[id(p)] * (1 - r.plateau_lr * group["weight_decay"])
            torch.testing.assert_close(p, expected, rtol=0, atol=0)


def test_disabled_fusion_is_not_owned_by_baseline_optimizer():
    r = recipe("B")
    model = build_campaign_model(backbone(), r)
    opt = build_campaign_adamw(model, r, fused=False)
    assert {g["component"] for g in opt.param_groups} == {"backbone"}
    assert all("fusion" not in name for names in optimizer_ownership(model, opt) for name in names)


def test_schedule_uses_completed_valid_tokens_and_roundtrips():
    parameter = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([parameter], lr=2e-4)
    schedule = CampaignTokenSchedule(opt, [10, 20, 5, 7], warmup_tokens=25)
    used = []
    for _ in range(4):
        used.append(opt.param_groups[0]["lr"])
        opt.step(); schedule.step()
    assert used == pytest.approx([2e-5, 2e-4 * .46, 2e-4, 2e-4])
    assert schedule.completed_tokens == 42
    state = copy.deepcopy(schedule.state_dict())
    other_opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=2e-4)
    other = CampaignTokenSchedule(other_opt, [10, 20, 5, 7], warmup_tokens=25)
    other_opt.load_state_dict(opt.state_dict())
    other.load_state_dict(state)
    assert other.state_dict() == state
    wrong = CampaignTokenSchedule(torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=2e-4),
                                  [10, 20, 6, 6], warmup_tokens=25)
    with pytest.raises(ValueError, match="token_prefix"):
        wrong.load_state_dict(state)


def test_noise_same_for_repartitioned_rows_and_padding_no_global_rng_use():
    r = recipe()
    before = torch.get_rng_state().clone()
    full = feedback_noise_for_rows(r, ["a", "b", "c"], logical_update=7, sequence_length=8, width=32)
    reordered = feedback_noise_for_rows(r, ["c", "a"], logical_update=7, sequence_length=8, width=32)
    short = feedback_noise_for_rows(r, ["b"], logical_update=7, sequence_length=4, width=32)
    later = feedback_noise_for_rows(r, ["a"], logical_update=8, sequence_length=8, width=32)
    assert torch.equal(before, torch.get_rng_state())
    for p in range(3):
        assert torch.equal(full[p][[2, 0]], reordered[p])
        assert torch.equal(full[p][1:2, :3], short[p])
        assert not torch.equal(full[p][:1], later[p])
        assert full[p].abs().max() <= 1
    assert not torch.equal(full[0], full[1])
    assert feedback_noise_for_rows(recipe("N"), ["a"], logical_update=0, sequence_length=8, width=32) is None


@pytest.mark.parametrize("kw", [{"arm": "FN"}, {"plateau_lr": float("nan")},
    {"warmup_tokens": -1}, {"rt_layers": (0, 0)}, {"betas": (.9, 1)},
    {"feedback_jitter": -1}, {"warmup_start_fraction": 0}, {"effective_valid_tokens": True}])
def test_invalid_recipe_fails_before_training(kw):
    with pytest.raises(ValueError):
        CampaignRecipe(**{"arm": "NFR", **kw})


def test_recipe_fingerprints_include_scientific_and_optimization_choices():
    r = recipe()
    assert r.sha256 != replace(r, plateau_lr=1e-4).sha256
    assert r.sha256 != replace(r, arm="NF").sha256
    assert r.sha256 != replace(r, feedback_jitter=0).sha256
    assert r.sha256 == recipe().sha256
