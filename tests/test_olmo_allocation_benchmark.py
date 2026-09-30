"""Allocation accounting and authenticated performance-clone boundary tests."""
import copy
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.lm_training import optimizer_ownership, parameter_layout
from scripts.olmo_allocation_benchmark import (
    ROOT, allocation, construct_tiny, parse_args, recipe_from_record,
    summary_from_updates, tiny_batches, validate_clone_payload,
)


@pytest.mark.parametrize("world,slots,dummies", [(1,43,4),(2,22,16),(8,6,64)])
def test_native_nfr_slot_accounting(world, slots, dummies):
    result = allocation(512, 12, world)
    assert result["microsteps_per_rank"] == slots
    assert result["dummy_rows"] == dummies
    assert result["physical_rows"] == 512 + dummies


@pytest.mark.parametrize("value", [0,-1,True,1.5])
def test_allocation_rejects_invalid_integer(value):
    with pytest.raises(ValueError):
        allocation(512, 12, value)


def test_tiny_partition_keeps_row_membership_and_jitter_at_one_two_eight_ranks():
    recipe = CampaignRecipe("NFR", sequence_length=6, rt_layers=(0,1), document_policy="continuous-stream-v1")
    observed = []
    for world in (1,2,8):
        rows = {}
        for rank in range(world):
            batches, noises, keys = tiny_batches(recipe=recipe, logical_update=3, rank=rank,
                world_size=world, batch_size=2, effective_rows=7, width=32)
            assert len(batches) == allocation(7,2,world)["microsteps_per_rank"]
            for batch, noise, names in zip(batches,noises,keys):
                assert not bool(batch.valid_mask[len(names):].any())
                assert all(not bool(value[len(names):].any()) for value in noise)
                for index,name in enumerate(names):
                    assert name not in rows
                    rows[name] = (batch.input_ids[index], tuple(value[index] for value in noise))
        assert len(rows) == 7
        observed.append(rows)
    for actual in observed[1:]:
        for name,(tokens,noise) in observed[0].items():
            assert torch.equal(tokens,actual[name][0])
            assert all(torch.equal(a,b) for a,b in zip(noise,actual[name][1]))


def test_recipe_retains_runtime_fields_but_not_old_execution_metadata():
    recipe = CampaignRecipe("NFR", document_policy="continuous-stream-v1")
    record = recipe.to_dict()
    record.update(optimizer_state="inherited_exact_parent_checkpoint", objective_transition={"test":True})
    record["auxiliary"]["kl"] = .1
    assert recipe_from_record(record) == recipe


@pytest.fixture
def populated():
    args = SimpleNamespace(arm="NFR",length=6,effective_rows=7)
    model,optimizer,recipe,_ = construct_tiny(args,torch.device("cpu"))
    # Distinct nonzero moments and one genuine optimizer step; the objective
    # math itself is qualified by separate reducer acceptance tests.
    for index,parameter in enumerate(model.parameters()):
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter,(index+1)*.001)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metadata = {"parameter_layout":parameter_layout(model),"optimizer_ownership":optimizer_ownership(model,optimizer)}
    payload = {"metadata":metadata,"counters":{"optimizer_updates":1},
               "model":copy.deepcopy(model.state_dict()),"optimizer":copy.deepcopy(optimizer.state_dict())}
    manifest = {"metadata":copy.deepcopy(metadata),"counters":{"optimizer_updates":1}}
    return model,optimizer,payload,manifest


def test_clone_payload_accepts_complete_owned_adam_without_mutating_model(populated):
    model,optimizer,payload,manifest = populated
    before = {name:value.clone() for name,value in model.state_dict().items()}
    validate_clone_payload(payload,manifest,model,optimizer)
    assert all(torch.equal(before[name],value) for name,value in model.state_dict().items())


@pytest.mark.parametrize("fault", ["metadata","clock","ownership","empty_state","shape","model_shape"])
def test_clone_rejects_invalid_state_before_mutation(populated,fault):
    model,optimizer,payload,manifest = populated
    first = next(iter(payload["optimizer"]["state"]))
    if fault == "metadata":
        payload["counters"]["optimizer_updates"] = 2
    elif fault == "clock":
        payload["optimizer"]["state"][first]["step"].fill_(2)
    elif fault == "ownership":
        next(group for group in payload["optimizer"]["param_groups"] if len(group["param_names"])>1)["param_names"].reverse()
    elif fault == "empty_state":
        payload["optimizer"]["state"].pop(first)
    elif fault == "shape":
        payload["optimizer"]["state"][first]["exp_avg"] = torch.zeros(1)
    else:
        name = next(iter(payload["model"]))
        payload["model"][name] = torch.zeros(1)
    with pytest.raises(ValueError):
        validate_clone_payload(payload,manifest,model,optimizer)


def test_summary_counts_only_measured_useful_tokens_and_conservative_rank_times():
    rows = []
    for index,phase in enumerate(("warmup","measured","measured")):
        rows.append({"phase":phase,"metrics":{"input_tokens":100,"gradient_norm_before_clip":.1,"objective":2.},
            "timing_by_rank":[{"materialization":1.,"backward":2.,"optimizer":1.},
                              {"materialization":1.,"backward":3.,"optimizer":1.}],
            "started_unix":index*6.,"finished_unix":index*6.+5.})
    summary=summary_from_updates(rows)
    assert summary["real_input_tokens"] == 200
    assert summary["selected_compute_materialization_seconds"] == 10.
    assert summary["measured_window_seconds"] == 11.
    assert summary["finite_updates"]


def test_cli_bounds_native_scope_and_gate_identity():
    baseline = ["--arm","B","--scale","tiny","--length","6","--effective-rows","7",
                "--batch-size","2","--output-dir",str(ROOT/".runtime/not-created-by-allocation-test")]
    assert parse_args(baseline).measured_updates == 4
    with pytest.raises(SystemExit):
        parse_args(baseline+["--measured-updates","13"])
    with pytest.raises(SystemExit):
        parse_args(baseline+["--gate-id","not-paired-with-path"])
