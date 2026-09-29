"""Actual tiny objective participation and fixed-forward six-case controls."""
import copy
from types import SimpleNamespace
import json

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from scripts import olmo_fusion_startup_component_probe as bridge
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, state_pins, fixture_pins
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def explicit_cpu(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(bridge,"rng_snapshot",lambda:torch.get_rng_state().clone())
    monkeypatch.setattr(bridge,"rng_unchanged",lambda before:torch.equal(before,torch.get_rng_state()))


def tiny():
    model,recipe,_,ids,eos = construct(SimpleNamespace(scale="tiny",length=8),"NF",torch.device("cpu"))
    model.backbone.backbone.attention_precision = "mixed"
    fixtures = [fixture_for_update(recipe,model.config.model_dim,rank,0,length=8,
        token_ids=ids,eos_id=eos,batch_size=2) for rank in range(2)]
    flags = {name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS}
    return model,recipe,fixtures,flags


def test_rt_selection_changes_only_recipe_arm_and_keeps_full_ownership():
    model,recipe,_,_ = tiny()
    state = state_pins(model)
    changed,record = bridge.enable_native_rt(model,recipe)
    assert changed.arm == "NFR" and changed.mode().rt_mode.selected_layers == (0,1)
    assert recipe.arm == "NF" and recipe.mode().rt_mode.selected_layers == ()
    assert state_pins(model) == state and all(record["checks"].values())
    assert all(parameter.requires_grad for parameter in model.parameters())
    contract = bridge.objective_contract(model,changed,"combined")
    assert contract["objective"] == "combined" and contract["objective_weights"] == dict.fromkeys(TERMS,1.)
    assert contract["auxiliary_cotangents"] == {"latent":1.,"kl":1.}
    assert contract["auxiliary_pass_weights"] == [.25]*4
    assert "auxiliary_pass_weights_before_zero_cotangent" not in contract
    with pytest.raises(ValueError,match="starts from isolated NF"):
        bridge.enable_native_rt(model,changed)


@pytest.mark.parametrize("arm",["NF","NFR"])
def test_combined_gradient_matches_literal_loss_formula_and_predictor_is_active(arm):
    model,recipe,fixtures,flags = tiny()
    if arm == "NFR":
        recipe,_ = bridge.enable_native_rt(model,recipe)
    row,actual,_ = bridge.measure_case(model,recipe,fixtures,objective="combined",path=FP32,original_flags=flags)
    counts = global_fixture_metadata(model,fixtures)["counts"]
    model.zero_grad(set_to_none=True)
    value = 0.
    # Independent objective construction directly from original per-pass losses.
    for (batch,),(noise,) in fixtures:
        with bridge.sdpa_kernel(bridge.SDPBackend.MATH):
            losses = model.loss_sums(batch,backbone_kwargs={"mode":recipe.mode(),"feedback_noise":noise,"right_padded_causal":True})
            ce = losses.pass_losses[0].sums["ce"]*.5 + sum(p.sums["ce"]*(1/6) for p in losses.pass_losses[1:])
            latent = sum(p.sums["latent"]*.25 for p in losses.pass_losses)
            kl = sum(p.sums["kl"]*.25 for p in losses.pass_losses)
            objective = ce/counts["ce"] + latent/counts["latent"] + kl/counts["kl"]
            objective.backward()
        value += float(objective.detach())
    assert row["metrics"]["objective"] == pytest.approx(value,rel=2e-7,abs=1e-7)
    for name,parameter in model.named_parameters():
        # Scalar sum grouping can differ while the independently constructed
        # objective/gradients must agree at strict FP32 roundoff scale.
        torch.testing.assert_close(parameter.grad,actual[name],rtol=2e-5,atol=2e-6)
    assert row["gradient_norms_descriptive_only"]["predictor"] > 0
    assert all(row["health"].values())


def test_six_cases_forward_controls_actual_predictor_participation_and_cleanup(monkeypatch):
    model,recipe,fixtures,flags = tiny()
    anchors = {}
    for path in (FP32,BF16):
        row,_,_ = bridge.measure_case(model,recipe,fixtures,objective="ce",path=path,original_flags=flags)
        anchors[path] = row
    model.zero_grad(set_to_none=True)
    start,inputs = state_pins(model),fixture_pins(fixtures)
    calls = []
    original = bridge.component_backward
    def count(*args,**kwargs):
        calls.append((args[1].arm,kwargs["objective"],kwargs["precision"]))
        return original(*args,**kwargs)
    monkeypatch.setattr(bridge,"component_backward",count)
    result = bridge.measure_bridge(model,recipe,fixtures,original_flags=flags,nf_ce_anchors=anchors,publish=lambda row:None)
    assert len(calls) == 6
    assert calls == [(arm,objective,precision) for arm,objective in bridge.PLAN for precision in ("fp32","bf16_mixed")]
    assert all(result["integrity"].values()) and len(result["transitions"]) == 1
    for row in result["rows"]:
        assert row["passed"] and all(row["health"].values())
        if row["objective"] == "combined":
            assert row["objective_forward_control"]["passed"]
            assert row["gradient_norms_descriptive_only"]["predictor"] > 0
        else:
            assert row["gradient_norms_descriptive_only"]["predictor"] == 0
    assert state_pins(model) == start and fixture_pins(fixtures) == inputs
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(getattr(model.backbone.backbone,name) == value for name,value in flags.items())


def test_bad_forward_anchor_stops_after_first_case_and_clears_gradients():
    model,recipe,fixtures,flags = tiny()
    anchors = {}
    for path in (FP32,BF16):
        row,_,_ = bridge.measure_case(model,recipe,fixtures,objective="ce",path=path,original_flags=flags)
        anchors[path] = row
    model.zero_grad(set_to_none=True)
    anchors[FP32]["metrics"]["loss_sums"]["ce"] += 1
    published = []
    with pytest.raises(AssertionError,match="Objective-only"):
        bridge.measure_bridge(model,recipe,fixtures,original_flags=flags,nf_ce_anchors=anchors,publish=published.append)
    assert len(published) == 1 and not published[0]["passed"]
    assert all(parameter.grad is None for parameter in model.parameters())


def reference():
    return {"schema":"olmo-fusion-startup-context-probe-v1","fixture_kind":"long","state":"startup","status":"passed_operational_diagnostic",
        "passed":True,"checkpoint_sha256":"a"*64,"fixture_sha256":"b"*64,"optimizer_updates":0,
        "aggregate_backwards":2,"physical_backwards":4,"import":{"counters":{"optimizer_updates":128}},
        "determinism":{"deterministic_algorithms":True},"integrity":{"state":True},"pair_integrity":{"state":True},
        "sources":{"old.py":"c"*64},"rows":[{"path":path,"objective":"ce","passed":True,"health":{"finite":True}}
            for path in (FP32,BF16)]}


@pytest.mark.parametrize("mutation",[None,"checkpoint","fixture","kind","cold","source","count","updates","endpoint","health","integrity","determinism","duplicate"])
def test_reference_requires_exact_saved_endpoint_and_healthy_two_precision_authority(tmp_path,mutation):
    report = reference()
    if mutation == "checkpoint": report["checkpoint_sha256"] = "d"*64
    elif mutation == "fixture": report["fixture_sha256"] = "d"*64
    elif mutation == "kind": report["fixture_kind"] = "packed"
    elif mutation == "cold": report["state"] = "cold"
    elif mutation == "source": report["sources"]["old.py"] = "d"*64
    elif mutation == "count": report["physical_backwards"] = 8
    elif mutation == "updates": report["optimizer_updates"] = 1
    elif mutation == "endpoint": report["import"]["counters"]["optimizer_updates"] = 32
    elif mutation == "health": report["rows"][0]["health"] = {}
    elif mutation == "integrity": report["pair_integrity"]["state"] = False
    elif mutation == "determinism": report["determinism"]["deterministic_algorithms"] = False
    elif mutation == "duplicate": report["rows"].append(copy.deepcopy(report["rows"][0]))
    path = tmp_path/"reference.json";path.write_text(json.dumps(report))
    kwargs = {"fixture_sha256":"b"*64,"checkpoint_sha256":"a"*64}
    if mutation:
        with pytest.raises(ValueError):
            bridge.load_nf_reference(path,sha256_file(path),{"old.py":"c"*64,"new.py":"e"*64},**kwargs)
    else:
        actual,anchors = bridge.load_nf_reference(path,sha256_file(path),{"old.py":"c"*64,"new.py":"e"*64},**kwargs)
        assert actual == report and set(anchors) == {FP32,BF16}


def test_health_rejects_zero_predictor_for_combined_or_active_predictor_for_ce():
    model,recipe,fixtures,flags = tiny()
    row,_,observed = bridge.measure_case(model,recipe,fixtures,objective="combined",path=FP32,original_flags=flags)
    metadata = global_fixture_metadata(model,fixtures)
    assert not bridge.case_health(row["metrics"],row["gradients"],row["forward_vs_fp32"],observed,metadata,"ce")["predictor_objective_participation"]
    changed = copy.deepcopy(row["gradients"]);changed["groups"]["predictor"]["norm"] = 0
    assert not bridge.case_health(row["metrics"],changed,row["forward_vs_fp32"],observed,metadata,"combined")["predictor_objective_participation"]


def test_reference_contract_normalizes_json_tuples_without_accepting_real_changes():
    model,recipe,fixtures,_ = tiny()
    current = {key:{"marker":1} for key in ("source_checkpoint","runtime","determinism")}
    current.update(initial_state=state_pins(model),contract=bridge.arm_contract(model,recipe),
        fixture_pins=fixture_pins(fixtures),fixture_metadata=global_fixture_metadata(model,fixtures),
        math_sdpa_reduced_precision_reduction=False,bf16_matmul_reduced_precision_reduction=True,
        fixture_provenance={"training_manifest_sha256":"a"*64},
        **{"import":{"configuration":{"data_manifest_sha256":"a"*64}}})
    restored = json.loads(json.dumps(current))
    assert current["contract"] != restored["contract"]  # Actual tuple/list trap.
    assert all(bridge.reference_checks(current,restored).values())
    restored["contract"]["mode"]["rt_mode"]["selected_layers"] = [0,1]
    assert not bridge.reference_checks(current,restored)["contract"]
    current["import"]["configuration"]["data_manifest_sha256"] = "b"*64
    assert not bridge.reference_checks(current,restored)["training_data_authority_exact"]
