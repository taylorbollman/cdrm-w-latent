"""Ordinary-B ownership, actual packed clocks, and retained lifecycle contracts."""
from dataclasses import asdict, replace
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained import document_shards
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.campaign_recipe import build_campaign_adamw, CampaignTokenSchedule
from cdrm.pretrained.campaign_training import CampaignObjective, CampaignGraphTraining
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.packed_campaign_data import PackedCampaignData, build_packed_index
from scripts import olmo_campaign_base_loop as runner
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def fixed_cpu_math():
    torch.set_num_threads(1)
    with sdpa_kernel(SDPBackend.MATH):
        yield


@pytest.fixture
def actual_data(tmp_path, monkeypatch):
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return SimpleNamespace(ids=[100+b for b in text.encode()])
    monkeypatch.setattr(document_shards, "_load_tokenizer", lambda _: Tokenizer())
    raw = b"".join((json.dumps({"id":str(i), "text":str(i)+"x"*1100})+"\n").encode()
                   for i in range(100))
    path=tmp_path/"source.jsonl";path.write_bytes(raw)
    source=LocalJSONLSource(SourcePin("fixture", "https://example.invalid/pinned", "fixed",
                                      hashlib.sha256(raw).hexdigest()),path)
    corpus,index=tmp_path/"corpus",tmp_path/"index"
    document_shards.prepare_document_shards([source],corpus,tokenizer_path="unused",
        split_policy=SplitPolicy(19,(("train",1),)),max_documents_per_shard=10)
    build_packed_index(corpus,index,split="train",length=1024)
    return corpus,index


def test_actual_t1024_plan_unique_rows_m1_m2_m3_and_ce_only_clock(actual_data):
    with PackedCampaignData(*actual_data) as data:
        origin=data.cursor();plans=runner.plan_updates(data)
        assert data.cursor()==origin
        assert [p.counts.valid_tokens for p in plans]==[16384,32768,49152]
        assert [len(p.rows) for p in plans]==[16,32,48]
        allkeys=[];crossings=0
        for index,plan in enumerate(plans):
            batches=[data.rank_batches(plan,rank=rank,world_size=2,physical_batch_size=8)
                     for rank in range(2)]
            assert [len(row.batches) for row in batches]==[index+1]*2
            for rank in batches:
                for keys,batch in zip(rank.keys,rank.batches):
                    allkeys.extend(keys)
                    assert batch.input_ids.shape==(8,1024) and batch.valid_mask.all()
                    crossings+=int((batch.document_ids[:,1:]!=batch.document_ids[:,:-1]).sum())
            data.commit(plan.start_cursor,plan)
        assert len(allkeys)==len(set(allkeys))==96 and crossings>0
        expected=runner.expected_counters(plans,3)
        assert asdict(expected)==dict(optimizer_updates=3,microbatches=12,documents=96,
            input_tokens=98304,ce_positions=98208,latent_pairs=0,kl_triples=0)
        saved=runner.cursor_record(data.cursor(),rank=1,batch_size=8)
        assert runner.validate_cursor(data,saved,expected,plans,1)==plans[-1].next_cursor


def test_cursor_restores_only_committed_prefix_and_rejects_aux_or_physical_drift(actual_data):
    with PackedCampaignData(*actual_data) as data:
        plans=runner.plan_updates(data);data.commit(plans[0].start_cursor,plans[0])
        expected=runner.expected_counters(plans,1)
        saved=runner.cursor_record(data.cursor(),rank=1,batch_size=8)
        data.peek_update(plans[1].next_cursor,runner.TARGETS[2])
        for changed in (replace(expected,latent_pairs=1),replace(expected,ce_positions=1),
                        replace(expected,microbatches=1),replace(expected,documents=1)):
            with pytest.raises(ValueError,match="counters"):
                runner.validate_cursor(data,saved,changed,plans,1)
        for changed in (dict(saved,rank=0),dict(saved,physical_batch_per_rank=12)):
            with pytest.raises(ValueError):runner.validate_cursor(data,changed,expected,plans,1)
    with PackedCampaignData(*actual_data) as fresh:
        assert runner.validate_cursor(fresh,saved,expected,plans,1,restore=True)==plans[0].next_cursor
        assert fresh.peek_update(fresh.cursor(),runner.TARGETS[1])==plans[1]
        with pytest.raises(ValueError,match="fresh pure"):
            runner.plan_updates(fresh)


def tiny():
    args=SimpleNamespace(scale="tiny",length=8,document_policy=runner.POLICY)
    model,recipe,*_=runner.construct(args,"B",torch.device("cpu"))
    return model,recipe


def batches(number):
    result=[]
    for index in range(number):
        ids=(torch.arange(16).reshape(2,8)+index*3)%55+2
        valid=torch.ones_like(ids,dtype=torch.bool)
        docs=torch.zeros_like(ids);docs[:,4:]=1  # Real explicit stream boundary.
        result.append(NextLatBatch(ids,valid,docs,valid.clone(),valid.clone(),valid.clone()))
    return tuple(result)


def test_ordinary_prepared_objective_matches_literal_ce_and_native_gradients():
    model,recipe=tiny();oracle=copy.deepcopy(model)
    optimizer=build_campaign_adamw(model,recipe,fused=False)
    contract=runner.model_contract(model,recipe,optimizer)
    assert contract["predictor_parameters"]==0
    assert contract["resident_parameters"]==contract["trainable_parameters"]+contract["dormant_fusion_parameters"]
    fixture=batches(2)
    adapter=CampaignObjective(model,fixture[0],mode=recipe.mode(),global_counts={"ce":28,"latent":0,"kl":0},
                              config=LMTrainingConfig(precision="fp32"))
    local=CampaignGraphTraining(adapter)
    measured=local.backward(fixture,replay=False)
    denominator=sum(int(batch.valid_mask[:,1:].sum()) for batch in fixture)
    loss=0
    for batch in fixture:
        output=oracle.backbone.backbone(batch.input_ids,attention_mask=batch.valid_mask,
                                       mode=recipe.mode().rt_mode,return_logits=True)
        loss=loss+F.cross_entropy(output.logits[:,:-1].reshape(-1,output.logits.shape[-1]),
                                 batch.input_ids[:,1:].reshape(-1),reduction="sum")/denominator
    loss.backward()
    assert measured["counts"]=={"ce":28,"latent":0,"kl":0}
    assert measured["objective"]==pytest.approx(float(loss.detach()),rel=2e-7)
    assert model.predictor is None and oracle.predictor is None
    for (name,p),(_,q) in zip(model.named_parameters(),oracle.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=1e-7,msg=name)
        else:assert p.grad is None and q.grad is None


def test_native_updates_keep_complete_dormant_fusion_exact_and_schedule_token_based():
    model,recipe=tiny();optimizer=build_campaign_adamw(model,recipe,fused=False)
    fixture=batches(2);scheduler=CampaignTokenSchedule(optimizer,[32,64,96],
        warmup_tokens=recipe.warmup_tokens,start_fraction=recipe.warmup_start_fraction)
    adapter=CampaignObjective(model,fixture[0],mode=recipe.mode(),global_counts={"ce":28,"latent":0,"kl":0},
                              config=LMTrainingConfig(precision="fp32"))
    local=CampaignGraphTraining(adapter);counters=TrainingCounters()
    native_before=tree_digests(model.backbone.backbone.state_dict())
    dormant=tree_digests(model.backbone.fusion.state_dict())
    for number in (2,4,6):
        local.optimizer_step(optimizer,batches(number),scheduler=scheduler,counters=counters)
        assert tree_digests(model.backbone.fusion.state_dict())==dormant
        runner.model_contract(model,recipe,optimizer)
        assert all(p.grad is None for p in model.backbone.fusion.parameters())
    assert counters.input_tokens==scheduler.completed_tokens==192
    assert counters.optimizer_updates==3 and counters.ce_positions==168
    assert counters.latent_pairs==counters.kl_triples==0
    assert tree_digests(model.backbone.backbone.state_dict())!=native_before
    assert len(optimizer.state)==len([p for p in model.parameters() if p.requires_grad])
    with pytest.raises(ValueError,match="exhausted"):scheduler.validate_next_update(32)


@pytest.mark.parametrize("mutation",["frozen_native","active_fusion","wrong_component","wrong_arm"])
def test_model_contract_rejects_silent_ownership_changes(mutation):
    model,recipe=tiny();optimizer=build_campaign_adamw(model,recipe,fused=False)
    if mutation=="frozen_native":next(model.backbone.backbone.parameters()).requires_grad_(False)
    elif mutation=="active_fusion":model.backbone.fusion.requires_grad_(True)
    elif mutation=="wrong_component":optimizer.param_groups[0]["component"]="fusion"
    else:recipe=replace(recipe,arm="N")
    with pytest.raises(ValueError):runner.model_contract(model,recipe,optimizer)


def test_pinned_reference_requires_verified_update1_and_exact_sources_configuration(tmp_path):
    sources={"source.py":"a"*64};config={"tokens":list(runner.TARGETS)};manifest="b"*64
    report={"schema":runner.SCHEMA,"phase":"reference","status":"passed","sources":sources,
        "configuration":config,"updates":{str(i):[{"rank":0},{"rank":1}] for i in (1,2,3)},
        "local_checkpoints":[{"optimizer_update":1,"receipt":{"manifest_sha256":manifest},
                              "boundary_by_rank":[{"rank":0},{"rank":1}]}],
        "published_checkpoints":[{"manifest_sha256":manifest,"retention":{"download_sha256_verified":True}}]}
    path=tmp_path/"reference.json"
    def load(value,**kwargs):
        path.write_text(json.dumps(value));sha=hashlib.sha256(path.read_bytes()).hexdigest()
        return runner.load_reference(path,sha,kwargs.get("sources",sources),kwargs.get("configuration",config),manifest)
    actual,boundaries=load(report)
    assert actual==report and boundaries==[{"rank":0},{"rank":1}]
    for key,value in (("phase","resume"),("status","stopped_at_boundary"),("updates",{"1":[]}),
                      ("local_checkpoints",[]),("published_checkpoints",[])):
        with pytest.raises(ValueError):load({**report,key:value})
    for changes in ({"sources":{}},{"configuration":{}}):
        with pytest.raises(ValueError):load(report,**changes)
    bad=copy.deepcopy(report);bad["published_checkpoints"][0]["retention"]["download_sha256_verified"]=False
    with pytest.raises(ValueError,match="verified"):load(bad)


@pytest.mark.parametrize("start,checkpoints",[(0,[1,3]),(1,[3])])
def test_same_loop_policy_saves_only_completed_1_3_never_duplicate_origin(start,checkpoints):
    class Coordinator:
        def same(self,phase,value):return value
        def gather(self,value):return [value,value]
        def call(self,phase,function,*,rank_zero=False):return function()
    number=[start];saved=[];published=[]
    def update():number[0]+=1;return {"update":number[0]}
    def save(n,reason):saved.append(n);return {"number":n}
    result=runner.run_loop(coordinator=Coordinator(),policy=runner.LoopPolicy(3,600,(1,3),save_initial=False),
        completed=lambda:number[0],update=update,log=lambda _:None,save=save,publish_checkpoint=published.append,
        stop=runner.StopRequest(),restored=True,clock=lambda:0.)
    assert saved==checkpoints and [r["number"] for r in published]==checkpoints
    assert result["completed_update"]==3 and result["start_update"]==start


def test_cli_freezes_shape_phase_and_requires_all_resume_authority():
    common=["--corpus","corpus","--index","index","--index-sha256","a"*64,
        "--output-dir",str(runner.ROOT/".runtime/base-cli"),"--checkpoint-root",str(runner.ROOT/".runtime/base-checkpoints"),
        "--storage-prefix","gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/base"]
    args=runner.parse_args(common+["--phase","reference"])
    assert (args.scale,args.length,args.document_policy)==("pretrained",1024,"continuous-stream-v1")
    for extra in (["--phase","resume"],["--phase","reference","--resume","checkpoint"],
                  ["--phase","reference","--batch-size","4"],
                  ["--phase","reference","--checkpoint-root","/mnt/localssd/checkpoints"]):
        with pytest.raises(SystemExit):runner.parse_args(common+extra)
    args=runner.parse_args(common+["--phase","resume","--resume","checkpoint","--resume-manifest-sha256","b"*64,
        "--reference-report","reference.json","--reference-sha256","c"*64])
    assert args.phase=="resume"
