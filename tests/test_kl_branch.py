"""Bounded CPU checks of the explicit objective fork; no training campaign."""
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained import distributed_checkpoint as checkpoint
from cdrm.pretrained.campaign_recipe import (
    CampaignRecipe, CampaignTokenSchedule, build_campaign_adamw, build_campaign_model,
)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_campaign_execution import model_contract
from scripts.olmo_campaign_restart import boundary
from scripts.olmo_kl_branch import branch_model_contract, declared_recipe, set_kl_weight


@pytest.fixture(autouse=True)
def cpu_setup():
    torch.set_num_threads(1)
    torch.manual_seed(173)


def objects(arm="NF"):
    recipe = CampaignRecipe(arm, sequence_length=6, rt_layers=(0, 1), feedback_jitter=0,
                            document_policy="continuous-stream-v1")
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=True)
    model = build_campaign_model(base, recipe).train()
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    scheduler = CampaignTokenSchedule(optimizer, [12]*128, warmup_tokens=1200)
    return recipe, model, optimizer, scheduler


def batch():
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 8, 9, 2, 0, 0]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2])
    docs = torch.tensor([[1, 1, 1, 2, 2, 2], [3, 3, 3, 3, -1, -1]])
    masks = [valid.clone() for _ in range(3)]
    masks[0][0, 1] = False
    masks[1][1, 1] = False
    return NextLatBatch(ids, valid, docs, *masks)


@pytest.mark.parametrize("arm", ["NF", "NFR"])
def test_contract_preserves_accepted_ownership_and_only_changes_declared_kl(arm):
    recipe, model, optimizer, _ = objects(arm)
    reference = model_contract(model, recipe, optimizer)
    assert branch_model_contract(model, recipe, optimizer, kl_weight=1.) == reference
    receipt = set_kl_weight(model, .1)
    assert all(receipt["checks"].values())
    candidate = branch_model_contract(model, recipe, optimizer, kl_weight=.1)
    assert candidate == {**reference, "weights": {"ce": 1., "latent": 1., "kl": .1}}
    with pytest.raises(ValueError):
        model_contract(model, recipe, optimizer)
    with pytest.raises(ValueError):
        branch_model_contract(model, recipe, optimizer, kl_weight=1.)
    # Parent strict loading can temporarily restore the original objective.
    set_kl_weight(model, 1.)
    assert branch_model_contract(model, recipe, optimizer, kl_weight=1.) == reference


def test_scalar_change_preserves_raw_losses_counts_and_recalculates_actual_adapter():
    recipe, model, _, _ = objects()
    sample = batch()
    original = CampaignObjective(model, sample, mode=recipe.mode(), global_counts=model.counts(sample))
    first = original.forward()
    set_kl_weight(model, .1)
    with pytest.raises(ValueError, match="objective settings changed"):
        original.validate_execution()
    candidate = CampaignObjective(model, sample, mode=recipe.mode(), global_counts=model.counts(sample))
    second = candidate.forward()
    assert candidate.counts == original.counts
    assert candidate.weights == {"ce": 1., "latent": 1., "kl": .1}
    for term in first["loss_sums"]:
        torch.testing.assert_close(first["loss_sums"][term], second["loss_sums"][term], rtol=0, atol=0)
    torch.testing.assert_close(first["objective"]-second["objective"],
        .9*first["loss_sums"]["kl"]/candidate.counts["kl"], rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(candidate.coefficients,
        torch.tensor([1/candidate.counts["ce"], 1/candidate.counts["latent"], .1/candidate.counts["kl"]]))


def test_transition_preserves_independent_configs_and_rejects_invalid_boundaries():
    _, model, _, _ = objects()
    model.predictor.config = replace(model.predictor.config, document_policy="isolated-v1")
    receipt = set_kl_weight(model, .1)
    assert model.config.document_policy == "continuous-stream-v1"
    assert model.predictor.config.document_policy == "isolated-v1"
    assert all(receipt["checks"].values())
    for invalid in (True, 0, .2, float("nan"), "0.1"):
        before = (model.config, model.predictor.config)
        with pytest.raises(ValueError):
            set_kl_weight(model, invalid)
        assert before == (model.config, model.predictor.config)
    parameter = next(model.parameters())
    parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(ValueError, match="cleared-gradient"):
        set_kl_weight(model, 1.)
    assert model.config.lambda_kl == .1
    parameter.grad = None
    model.predictor.config = replace(model.predictor.config, lambda_kl=1.)
    with pytest.raises(ValueError, match="Model/predictor"):
        set_kl_weight(model, 1.)


def test_declaration_honestly_records_inheritance_without_changing_recipe():
    recipe, _, _, _ = objects()
    original = recipe.to_dict()
    parent_pin = "a"*64
    control = declared_recipe(recipe, kl_weight=1., parent_manifest_sha256=parent_pin)
    branch = declared_recipe(recipe, kl_weight=.1, parent_manifest_sha256=parent_pin)
    assert recipe.to_dict() == original and original["optimizer_state"] == "fresh"
    assert branch["auxiliary"] == {**original["auxiliary"], "kl": .1}
    assert branch["optimizer_state"] == control["optimizer_state"] == "inherited_exact_parent_checkpoint"
    assert branch["objective_transition"]["parent_manifest_sha256"] == parent_pin
    assert control["objective_transition"]["same_weight_control"] is True
    assert branch["objective_transition"]["same_weight_control"] is False
    for field in ("warmup_tokens", "plateau_lr", "effective_valid_tokens", "betas", "precision"):
        assert branch[field] == original[field]
    with pytest.raises(ValueError):
        declared_recipe(recipe, kl_weight=.1, parent_manifest_sha256="missing")


def test_original_strict_load_then_branch_save_and_exact_resume_preserves_complete_state(tmp_path, monkeypatch):
    # A local tiny checkpoint fixture exercises the unchanged serialization and
    # strict metadata loader. It does not claim native two-rank GPU recovery.
    monkeypatch.setattr(checkpoint, "_context", lambda group: (0, 1))
    monkeypatch.setattr(checkpoint, "_gather", lambda value, group: [value])
    recipe, model, optimizer, scheduler = objects()
    for _ in range(2):
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    counters = TrainingCounters(optimizer_updates=2, input_tokens=24)
    cursor = {"tiny_fixture_offset": 2}
    generators = {"local": torch.Generator().manual_seed(91)}
    device = torch.device("cpu")
    parent_config = {"recipe": recipe.to_dict(), "model": model.config.to_dict(), "fixture": "local-cpu"}
    parent_source = {"identity": "parent", "sha256": "a"*64}
    parent = tmp_path/"parent"
    expected = boundary(model, optimizer, scheduler, counters, cursor, device, generators)
    record = checkpoint.save_distributed_checkpoint(parent, model, optimizer, scheduler=scheduler,
        counters=counters, data_cursor=cursor, configuration=parent_config,
        source_fingerprint=parent_source, generators=generators, device=device)
    _, model, optimizer, scheduler = objects()
    loaded = checkpoint.load_distributed_checkpoint(parent, model, optimizer, scheduler=scheduler,
        configuration=parent_config, source_fingerprint=parent_source, generators=generators,
        expected_manifest_sha256=record["manifest_sha256"], device=device)
    assert boundary(model, optimizer, scheduler, loaded["counters"], loaded["data_cursor"], device, generators) == expected
    set_kl_weight(model, .1)
    assert boundary(model, optimizer, scheduler, loaded["counters"], loaded["data_cursor"], device, generators) == expected
    branch_config = {**parent_config, "model": model.config.to_dict(),
        "recipe": declared_recipe(recipe, kl_weight=.1, parent_manifest_sha256=record["manifest_sha256"])}
    branch_source = {"identity": "kl-0.1-child", "sha256": "b"*64,
                     "parent_manifest_sha256": record["manifest_sha256"]}
    # Parent state cannot be loaded while claiming the new branch identity.
    with pytest.raises(checkpoint.DistributedCheckpointError):
        checkpoint.load_distributed_checkpoint(parent, model, optimizer, scheduler=scheduler,
            configuration=branch_config, source_fingerprint=branch_source, generators=generators, device=device)
    branch = tmp_path/"branch"
    saved = checkpoint.save_distributed_checkpoint(branch, model, optimizer, scheduler=scheduler,
        counters=loaded["counters"], data_cursor=loaded["data_cursor"], configuration=branch_config,
        source_fingerprint=branch_source, generators=generators, device=device)
    _, resumed, resume_optimizer, resume_scheduler = objects()
    set_kl_weight(resumed, .1)
    restored = checkpoint.load_distributed_checkpoint(branch, resumed, resume_optimizer,
        scheduler=resume_scheduler, configuration=branch_config, source_fingerprint=branch_source,
        generators=generators, expected_manifest_sha256=saved["manifest_sha256"], device=device)
    # Exact child restore needs no second parent migration or optimizer reset.
    assert resumed.objective_weights() == {"ce": 1., "latent": 1., "kl": .1}
    assert boundary(resumed, resume_optimizer, resume_scheduler, restored["counters"],
                    restored["data_cursor"], device, generators) == expected
    assert resume_scheduler.last_epoch == 2 and len(resume_scheduler.token_prefix) == 129
    assert all(state["step"].item() == 2 for state in resume_optimizer.state.values())
