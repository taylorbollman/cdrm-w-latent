"""Native smoke contract guards, exercised on CPU without native weights or CUDA."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cdrm.pretrained.campaign_recipe import build_campaign_adamw, build_campaign_model
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_topology_native_smoke as smoke
from scripts.olmo_allocation_acceptance import materialize
from cdrm.pretrained.campaign_data import DocumentWindow


def plan_and_ranks():
    rows = [SimpleNamespace(key=f"row{i}") for i in range(48)]
    counts = SimpleNamespace(valid_tokens=49152, packed_rows=48, ce_targets=49104,
                             latent_pairs=49056, kl_triples=48912)
    plan = SimpleNamespace(rows=rows, counts=counts,
        start_cursor=TrainingCounters())  # Any dataclass suffices for the input evidence test.
    ranks = []
    for rank in range(2):
        ranks.append({"slots": 2, "input_tokens": 24576,
            "literal_counts": {"ce": 24552, "latent": 24528, "kl": 24456},
            "model_counts": {"ce": 24552, "latent": 24528, "kl": 24456},
            "rows": [{"key": f"row{i}", "input_sha256": str(i), "masks_sha256": "mask"}
                     for i in range(rank * 24, (rank + 1) * 24)]})
    return plan, ranks


def test_all_four_modes_retain_expected_active_module_ownership():
    for arm in smoke.ARMS:
        recipe = smoke.smoke_recipe(arm)
        # The native recipe and objective are checked with a dimension-compatible
        # CPU fixture; this does not execute a CUDA graph or native RT kernel.
        from dataclasses import replace
        tiny_recipe = replace(recipe, sequence_length=8, rt_layers=(0, 1))
        base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32",
            tile_backend="eager", backward_tile_backend="eager", backward_memory="recompute")
        model = build_campaign_model(base, tiny_recipe)
        smoke.apply_smoke_objective(model)
        optimizer = build_campaign_adamw(model, tiny_recipe, fused=False)
        audited = smoke.state_audit(model, optimizer, update=0)
        names = [n for group in audited["ownership"] for n in group]
        assert any(n.startswith("predictor.") for n in names) == ("N" in arm)
        assert any(n.startswith("backbone.fusion.") for n in names) == ("F" in arm)
        assert model.objective_weights()["kl"] == (0.1 if "N" in arm else 0)
        assert model.objective_weights()["latent"] == (1 if "N" in arm else 0)
        assert recipe.mode().num_passes == (4 if "F" in arm else 1)
        assert recipe.mode().rt_mode.selected_layers == ((0, 15) if "R" in arm else ())
        assert recipe.document_policy == "continuous-stream-v1"
        assert recipe.effective_valid_tokens == 49152 and recipe.sequence_length == 1024
        assert not audited["adam"]


def test_membership_and_counts_are_global_not_local_or_duplicated():
    plan, ranks = plan_and_ranks()
    evidence = smoke.validate_inputs(plan, ranks, nextlat=True)
    assert evidence["counts"] == {"ce": 49104, "latent": 49056, "kl": 48912}
    assert [r["key"] for r in evidence["rows"]] == [r.key for r in plan.rows]
    assert evidence["microbatches"] == 4 and evidence["input_tokens"] == 49152
    duplicated = copy.deepcopy(ranks)
    duplicated[1]["rows"][0] = duplicated[0]["rows"][0]
    with pytest.raises(ValueError, match="duplicated"):
        smoke.validate_inputs(plan, duplicated, nextlat=True)
    wrong = copy.deepcopy(ranks)
    wrong[1]["model_counts"]["kl"] += 1
    with pytest.raises(ValueError, match="denominators"):
        smoke.validate_inputs(plan, wrong, nextlat=True)
    ranks[0]["slots"] = 1
    with pytest.raises(ValueError, match="two physical slots"):
        smoke.validate_inputs(plan, ranks, nextlat=True)


def test_inactive_auxiliary_denominators_are_exactly_zero():
    plan, ranks = plan_and_ranks()
    for rank in ranks:
        for field in ("literal_counts", "model_counts"):
            rank[field].update(latent=0, kl=0)
    assert smoke.validate_inputs(plan, ranks, nextlat=False)["counts"] == {"ce": 49104, "latent": 0, "kl": 0}
    plan.counts.valid_tokens -= 1
    with pytest.raises(ValueError, match="complete T1024"):
        smoke.validate_inputs(plan, ranks, nextlat=False)


def test_literal_masks_distinguish_ce_crossing_eos_from_auxiliary_boundary():
    row = DocumentWindow("test", "train", 0, 0, tuple(range(2, 8)))
    batch = materialize([row], physical_batch_size=1, length=8)
    batch.ce_mask.copy_(batch.valid_mask)
    batch.latent_mask.copy_(batch.valid_mask)
    batch.kl_mask.copy_(batch.valid_mask)
    # Boundary is before token2. Continuous-stream CE retains its next-token
    # target, while latent pair and KL triple may not cross document identity.
    assert smoke.literal_counts([batch], nextlat=True) == {"ce": 5, "latent": 4, "kl": 2}
    assert smoke.literal_counts([batch], nextlat=False) == {"ce": 5, "latent": 0, "kl": 0}


def test_completed_update_rejects_stale_counts_nonfinite_and_wrong_counter():
    plan, ranks = plan_and_ranks()
    evidence = smoke.validate_inputs(plan, ranks, nextlat=True)
    counters = TrainingCounters(optimizer_updates=1, microbatches=4, documents=48, input_tokens=49152,
        ce_positions=49104, latent_pairs=49056, kl_triples=48912)
    metrics = {k: evidence[k] for k in ("counts", "input_tokens", "documents", "microbatches")}
    metrics.update(world_size=2, local_microbatches=2, objective=1., gradient_norm_before_clip=3.,
        loss_sums={"ce": 2., "latent": 3., "kl": 4.}, lr_used=[2e-5], lr_next=[2e-5], counters=asdict(counters))
    smoke.validate_update(metrics, evidence, counters, update=1, previous_counters=asdict(TrainingCounters()))
    bad = copy.deepcopy(metrics)
    bad["gradient_norm_before_clip"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        smoke.validate_update(bad, evidence, counters, update=1, previous_counters=asdict(TrainingCounters()))
    bad = copy.deepcopy(metrics)
    bad["counters"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="counters"):
        smoke.validate_update(bad, evidence, counters, update=1, previous_counters=asdict(TrainingCounters()))


def test_state_audit_proves_fresh_adam_then_checks_finite_moments_and_steps():
    model = nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    before = smoke.state_audit(model, optimizer, update=0)
    model(torch.tensor([[1., 2.]])).square().sum().backward()
    assert smoke.gradient_audit(model)
    optimizer.step()
    after = smoke.state_audit(model, optimizer, update=1)
    assert smoke.changed_components(model, before, after)["backbone"] == 1
    with pytest.raises(ValueError, match="fresh empty"):
        smoke.state_audit(model, optimizer, update=0)
    with pytest.raises(ValueError, match="counter"):
        smoke.state_audit(model, optimizer, update=2)
    first = next(iter(optimizer.state.values()))
    first["exp_avg"][0, 0] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite"):
        smoke.state_audit(model, optimizer, update=1)


def test_state_audit_rejects_foreign_ownership_missing_gradient_and_changed_frozen_parameter():
    model = nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW([model.weight], lr=.01)
    before = smoke.state_audit(model, optimizer, update=0)
    with pytest.raises(ValueError, match="Missing active"):
        smoke.gradient_audit(model)
    with torch.no_grad():
        model.bias.add_(1)
    after = smoke.state_audit(model, optimizer, update=0)
    with pytest.raises(ValueError, match="Frozen parameter"):
        smoke.changed_components(model, before, after)
    optimizer.param_groups[0]["params"].append(model.bias)
    with pytest.raises(ValueError, match="frozen"):
        smoke.state_audit(model, optimizer, update=0)


def test_no_changes_cannot_be_mistaken_for_success_and_bf16_masters_rejected():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    before = smoke.state_audit(model, optimizer, update=0)
    with pytest.raises(ValueError, match="no parameter change"):
        smoke.changed_components(model, before, before)
    with pytest.raises(ValueError, match="FP32"):
        smoke.tensor_record(torch.ones(2, dtype=torch.bfloat16))


def test_cli_does_not_allow_scope_expansion_or_unpinned_data(tmp_path):
    base = ["--arm", "N", "--output-dir", str(smoke.ROOT / ".runtime/test-smoke-new"),
            "--corpus", str(tmp_path), "--index", str(tmp_path), "--index-sha256", "a" * 64]
    args = smoke.parse_args(base)
    assert args.arm == "N"
    with pytest.raises(SystemExit):
        smoke.parse_args(base + ["--updates", "3"])
    with pytest.raises(SystemExit):
        smoke.parse_args(base[:-1] + ["invalid"])
    with pytest.raises(SystemExit):
        smoke.parse_args(base + ["--output-dir", str(tmp_path)])


@pytest.mark.parametrize("arm", smoke.ARMS)
def test_tiny_real_objective_has_finite_owned_gradients_and_actual_component_updates(arm):
    from dataclasses import replace
    from scripts.olmo_allocation_acceptance import fixture_for_rank
    recipe = replace(smoke.smoke_recipe(arm), sequence_length=8, rt_layers=(0, 1))
    torch.manual_seed(15)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32",
        tile_backend="eager", backward_tile_backend="eager", backward_memory="recompute")
    model = build_campaign_model(base, recipe)
    smoke.apply_smoke_objective(model)
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    before = smoke.state_audit(model, optimizer, update=0)
    batches, noises, _ = fixture_for_rank(recipe, 0, rank=0, world_size=1)
    result = model.loss_sums(batches[0], backbone_kwargs={"mode": recipe.mode(),
        "feedback_noise": noises[0], "right_padded_causal": True})
    objective = sum(result.sums[t] * result.weights[t] / result.counts[t]
                    for t in ("ce", "latent", "kl") if result.weights[t])
    objective.backward()
    assert smoke.gradient_audit(model)
    optimizer.step()
    after = smoke.state_audit(model, optimizer, update=1)
    changed = smoke.changed_components(model, before, after)
    assert changed["backbone"] > 0
    assert (changed["predictor"] > 0) == ("N" in arm)
    assert (changed["fusion"] > 0) == ("F" in arm)


@pytest.mark.parametrize("arm", smoke.ARMS)
def test_actual_constructor_common_seed_ignores_ambient_rank_rng_and_preserves_backbone(arm):
    from dataclasses import replace
    config = OLMoConfig.tiny()
    recipe = replace(smoke.smoke_recipe(arm), sequence_length=8, rt_layers=(0, 1))
    torch.manual_seed(8765)
    original = OLMoTiledRTForCausalLM(config, attention_backend="math", attention_precision="fp32",
        tile_backend="eager", backward_tile_backend="eager", backward_memory="recompute").state_dict()
    pins = {name: smoke.tensor_record(value) for name, value in original.items()}
    records, rng = [], []
    for ambient_seed in (111, 999):
        torch.manual_seed(ambient_seed)
        model, optimizer = smoke.construct_model(original, recipe, "cpu", model_config=config)
        records.append(smoke.state_audit(model, optimizer, update=0))
        rng.append(torch.get_rng_state().clone())
        loaded = model.backbone.backbone.state_dict()
        assert {name: smoke.tensor_record(value) for name, value in loaded.items()} == pins
        assert {name: smoke.tensor_record(value) for name, value in original.items()} == pins
    assert records[0] == records[1]
    assert torch.equal(rng[0], rng[1])
    assert torch.initial_seed() == smoke.INITIALIZATION_SEED
