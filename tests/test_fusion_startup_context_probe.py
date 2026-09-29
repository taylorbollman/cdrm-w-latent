"""Policy transition preserves imported weights while enabling stream CE."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.document_policy import CONTINUOUS_STREAM, feedback_eligibility
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from scripts import olmo_fusion_startup_context_probe as packed
from scripts import olmo_fusion_startup_probe as probe
from scripts.olmo_campaign_ddp_probe import construct
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, state_pins, fixture_pins


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def tiny():
    model,recipe=construct(SimpleNamespace(scale="tiny",length=16),"NF",torch.device("cpu"))[:2]
    model.backbone.backbone.attention_precision="mixed"
    return model,recipe


def test_policy_transition_preserves_model_and_rng_and_changes_real_boundary_semantics():
    model,recipe=tiny()
    before=state_pins(model); rng=torch.get_rng_state().clone()
    ids=torch.tensor([[2,3,60,4,5,6,60,7]])
    valid=torch.ones_like(ids,dtype=torch.bool)
    docs=torch.tensor([[0,0,0,1,1,1,1,2]])
    targets=valid.clone();targets[:,0]=False
    batch=NextLatBatch(ids,valid,docs,targets,targets.clone(),targets.clone())
    isolated=build_nextlat_masks(batch)
    changed,record=packed.transition_policy(model,recipe)
    masks=build_nextlat_masks(batch,document_policy=changed.document_policy)
    assert changed.document_policy==model.config.document_policy==CONTINUOUS_STREAM
    assert all(record["checks"].values()) and state_pins(model)==before
    assert torch.equal(rng,torch.get_rng_state())
    assert int(isolated["ce"].sum())==5 and int(masks["ce"].sum())==7
    assert torch.equal(isolated["latent"],masks["latent"]) and torch.equal(isolated["kl"],masks["kl"])
    assert bool(feedback_eligibility(valid,docs,CONTINUOUS_STREAM).all())
    assert recipe.document_policy=="isolated-v1"
    with pytest.raises(ValueError,match="original isolated NF"):
        packed.transition_policy(model,changed)


def test_tiny_packed_gradient_repeats_exactly_and_captures_cross_document_ce(monkeypatch):
    model,recipe=tiny(); recipe,_=packed.transition_policy(model,recipe)
    monkeypatch.setattr(probe,"rng_snapshot",lambda:torch.get_rng_state().clone())
    monkeypatch.setattr(probe,"rng_unchanged",lambda old:torch.equal(old,torch.get_rng_state()))
    fixtures=[]
    for offset in (0,4):
        ids=torch.tensor([[2+offset,3+offset,60,4+offset,5+offset,6+offset,60,7+offset]])
        valid=torch.ones_like(ids,dtype=torch.bool);targets=valid.clone();targets[:,0]=False
        docs=torch.tensor([[0,0,0,1,1,1,1,2]])+offset
        batch=NextLatBatch(ids,valid,docs,targets,targets.clone(),targets.clone())
        noise=tuple(torch.full((1,7,model.config.model_dim),.5) for _ in range(3))
        fixtures.append(((batch,),(noise,)))
    initial=state_pins(model);inputs=fixture_pins(fixtures)
    flags={n:getattr(model.backbone.backbone,n) for n in RUNTIME_FLAGS}
    first,gradients,states=probe.measure_case(model,recipe,fixtures,path=FP32,original_flags=flags)
    repeat,_,_=probe.measure_case(model,recipe,fixtures,path=FP32,original_flags=flags,
        reference_gradients=gradients,reference_forward=states)
    assert first["metrics"]["counts"]["ce"]==14
    assert first["metrics"]["counts"]["latent"]==10
    assert first["metrics"]["counts"]["kl"]==6
    assert repeat["gradients_vs_fp32"]["geometry"]["all"]["difference_norm"]==0
    assert first["forward_fingerprints"]==repeat["forward_fingerprints"]
    assert first["position_geometry"]["document_policy"]==CONTINUOUS_STREAM
    assert state_pins(model)==initial and fixture_pins(fixtures)==inputs
    model.zero_grad(set_to_none=True)


def cold_report():
    return {"schema":packed.SCHEMA,"fixture_kind":"packed","state":"cold","passed":True,"status":"passed_operational_diagnostic",
        "optimizer_updates":0,"aggregate_backwards":2,"physical_backwards":4,"determinism":{"deterministic_algorithms":True},
        "fixture_sha256":"a"*64,"integrity":{"unchanged":True},"pair_integrity":{"unchanged":True},
        "sources":{"x":"b"*64},"rows":[{"path":p,"objective":"ce","passed":True,"health":{"finite":True}}
            for p in (FP32,BF16)]}


@pytest.mark.parametrize("mutation",[None,"source","fixture","state","health","pair","duplicate","updates","backwards","determinism"])
def test_cold_reference_is_bound_to_fixture_sources_and_both_precisions(tmp_path,mutation):
    report=cold_report()
    if mutation=="source":report["sources"]["x"]="c"*64
    elif mutation=="fixture":report["fixture_sha256"]="c"*64
    elif mutation=="state":report["state"]="startup"
    elif mutation=="health":report["rows"][0]["health"]={}
    elif mutation=="pair":report["pair_integrity"]["unchanged"]=False
    elif mutation=="duplicate":report["rows"].append(report["rows"][0])
    elif mutation=="updates":report["optimizer_updates"]=1
    elif mutation=="backwards":report["physical_backwards"]=8
    elif mutation=="determinism":report["determinism"]["deterministic_algorithms"]=False
    path=tmp_path/"report.json";path.write_text(json.dumps(report))
    if mutation:
        with pytest.raises(ValueError):packed.load_cold(path,sha256_file(path),{"x":"b"*64},"a"*64)
    else:
        loaded,rows=packed.load_cold(path,sha256_file(path),{"x":"b"*64},"a"*64)
        assert loaded==report and set(rows)=={FP32,BF16}


def test_json_contract_normalizes_tuple_representation_without_ignoring_semantics():
    original={"mode":{"rt_mode":{"selected_layers":()}},"beta":1.0}
    serialized=json.loads(json.dumps(original))
    assert original != serialized
    assert packed.json_contract(original)==serialized
    changed={"mode":{"rt_mode":{"selected_layers":(0,15)}},"beta":1.0}
    assert packed.json_contract(changed)!=serialized
    assert packed.json_contract({**original,"beta":.5})!=serialized
