"""Actual tiny N-only objective, cold NF first-pass anchors and cleanup."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts import olmo_fusion_startup_packed_baseline as probe
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, state_pins, fixture_pins
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_oracle(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(probe,"rng_snapshot",lambda:torch.get_rng_state().clone())
    monkeypatch.setattr(probe,"rng_unchanged",lambda previous:torch.equal(previous,torch.get_rng_state()))


def tiny():
    model,recipe,_,ids,eos=construct(SimpleNamespace(scale="tiny",length=8,document_policy="continuous-stream-v1"),"NF",torch.device("cpu"))
    model.backbone.backbone.attention_precision="mixed"
    fixtures=[fixture_for_update(recipe,32,rank,0,length=8,token_ids=ids,eos_id=eos,batch_size=2) for rank in (0,1)]
    flags={name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS}
    return model,recipe,fixtures,flags


def anchors(model,recipe,fixtures,flags):
    result={}
    for path in (FP32,BF16):
        probe.configure_path(model,flags,path)
        with probe.capture_passes(model,fixtures) as observed, probe.sdpa_kernel(probe.SDPBackend.MATH), torch.autocast("cpu",dtype=torch.bfloat16,enabled=path==BF16,cache_enabled=False):
            for (batch,),(noise,) in fixtures:
                model.loss_sums(batch,backbone_kwargs={"mode":recipe.mode(),"feedback_noise":noise,"right_padded_causal":True})
        result[path]={"path":path,"forward_fingerprints":tree_digests([{key:row[key] for key in ("batch","token_embeddings","pass_hidden_states")} for row in observed])}
    probe.configure_path(model,flags,BF16)
    return result


def test_transition_retains_exact_state_rows_and_owners_only_freezes_fusion():
    model,recipe,fixtures,_=tiny()
    before=state_pins(model);inputs=fixture_pins(fixtures)
    ordinary,rows,record=probe.ordinary_transition(model,recipe,fixtures)
    assert all(record["checks"].values()) and state_pins(model)==before and fixture_pins(fixtures)==inputs
    assert ordinary.arm=="N" and not ordinary.mode().enabled and ordinary.mode().num_passes==1
    assert not ordinary.mode().rt_mode.selected_layers and ordinary.mode().feedback_jitter==0
    assert recipe.arm=="NF" and all(noise==(None,) for _,noise in rows)
    assert all(p.requires_grad == (not name.startswith("backbone.fusion.")) for name,p in model.named_parameters())
    with pytest.raises(ValueError,match="validated packed NF"):
        probe.ordinary_transition(model,ordinary,rows)


@pytest.mark.parametrize("path",[FP32,BF16])
def test_single_pass_matches_same_precision_nf_output_and_literal_ce_gradient(path):
    model,recipe,fixtures,flags=tiny();reference=anchors(model,recipe,fixtures,flags)
    recipe,rows,_=probe.ordinary_transition(model,recipe,fixtures)
    row,gradients,_=probe.measure_case(model,recipe,rows,path=path,original_flags=flags)
    assert row["passed"] and all(row["health"].values())
    assert probe.first_pass_anchor(row["forward_fingerprints"],reference[path])["passed"]
    assert row["gradient_norms_descriptive_only"]["backbone"]>0
    assert row["gradient_norms_descriptive_only"]["predictor"]==0
    assert row["gradient_norms_descriptive_only"]["fusion"]==0
    counts=global_fixture_metadata(model,rows)["counts"]
    model.zero_grad(set_to_none=True);total=0.
    for (batch,),_ in rows:
        with probe.sdpa_kernel(probe.SDPBackend.MATH):
            with torch.autocast("cpu",dtype=torch.bfloat16,enabled=path==BF16,cache_enabled=False):
                loss=model.loss_sums(batch,backbone_kwargs={"mode":recipe.mode(),"feedback_noise":None,"right_padded_causal":True})
                objective=loss.pass_losses[0].sums["ce"]/counts["ce"]
            objective.backward();total+=float(objective.detach())
    assert row["metrics"]["objective"]==total
    for name,p in model.named_parameters():
        if name.startswith("backbone.backbone."):
            assert tree_digests(p.grad)==tree_digests(gradients[name])
    assert all(p.grad is None for p in model.backbone.fusion.parameters())


def test_pair_runs_two_cases_with_original_first_pass_pins_and_restores_flags(monkeypatch):
    model,recipe,fixtures,flags=tiny();reference=anchors(model,recipe,fixtures,flags)
    recipe,rows,_=probe.ordinary_transition(model,recipe,fixtures)
    before=state_pins(model);calls=[];original=probe.component_backward
    def count(*args,**kwargs):
        calls.append((args[1].arm,kwargs["precision"],kwargs["objective"]))
        return original(*args,**kwargs)
    monkeypatch.setattr(probe,"component_backward",count)
    result=probe.measure_pair(model,recipe,rows,original_flags=flags,anchors=reference,publish=lambda row:None)
    assert calls==[("N","fp32","ce"),("N","bf16_mixed","ce")]
    assert len(result["rows"])==2 and all(result["integrity"].values())
    assert all(row["cold_nf_first_pass_anchor"]["passed"] for row in result["rows"])
    assert state_pins(model)==before and all(p.grad is None for p in model.parameters())
    assert all(getattr(model.backbone.backbone,n)==v for n,v in flags.items())


def test_bad_anchor_fails_first_case_then_clears_gradients_and_restores_runtime():
    model,recipe,fixtures,flags=tiny();reference=anchors(model,recipe,fixtures,flags)
    recipe,rows,_=probe.ordinary_transition(model,recipe,fixtures)
    reference[FP32]["forward_fingerprints"][0]["pass_hidden_states"][0]["sha256"]="0"*64
    seen=[]
    with pytest.raises(AssertionError,match="cold NF first pass"):
        probe.measure_pair(model,recipe,rows,original_flags=flags,anchors=reference,publish=seen.append)
    assert len(seen)==1 and not seen[0]["passed"]
    assert all(p.grad is None for p in model.parameters())
    assert all(getattr(model.backbone.backbone,n)==v for n,v in flags.items())


def test_reject_noise_or_wrong_reference_precision_before_backward():
    model,recipe,fixtures,flags=tiny();reference=anchors(model,recipe,fixtures,flags)
    recipe,rows,_=probe.ordinary_transition(model,recipe,fixtures)
    with pytest.raises(ValueError,match="must not receive feedback noise"):
        probe.measure_case(model,recipe,fixtures,path=FP32,original_flags=flags)
    reference[FP32]["path"]=BF16
    with pytest.raises(ValueError,match="Anchor precision"):
        probe.measure_pair(model,recipe,rows,original_flags=flags,anchors=reference,publish=lambda row:None)


def test_cli_requires_reference_and_fixture_pins():
    common=["--fixture","fixture.json","--fixture-sha256","a"*64,
        "--cold-report","report.json","--cold-report-sha256","b"*64,
        "--output-dir",str(probe.ROOT/".runtime/packed-baseline-cli")]
    assert probe.parse_args(common).cold_report_sha256=="b"*64
    with pytest.raises(SystemExit):probe.parse_args(common+["--fixture-sha256","invalid"])
    with pytest.raises(SystemExit):probe.parse_args(common+["--output-dir","/tmp/nonpersistent"])
