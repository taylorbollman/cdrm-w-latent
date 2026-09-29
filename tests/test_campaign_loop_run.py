"""Concrete packed-data, model and resume contracts of the tiny loop CLI."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained import document_shards
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.packed_campaign_data import PackedCampaignData, build_packed_index
from scripts import olmo_campaign_loop_run as runner
from scripts.olmo_lm_common import tree_digests


@pytest.fixture
def actual_data(tmp_path,monkeypatch):
    class Tokenizer:
        def encode(self,text,*,add_special_tokens):
            assert not add_special_tokens
            return SimpleNamespace(ids=[100+b for b in text.encode()])
    monkeypatch.setattr(document_shards,"_load_tokenizer",lambda _:Tokenizer())
    raw = b"".join((json.dumps({"id":str(i),"text":str(i)+"real "+"x"*(80+i)})+"\n").encode() for i in range(8))
    path=tmp_path/"source.jsonl";path.write_bytes(raw)
    source=LocalJSONLSource(SourcePin("fixture","https://example.invalid/fixture","pinned",hashlib.sha256(raw).hexdigest()),path)
    corpus,index=tmp_path/"corpus",tmp_path/"index"
    document_shards.prepare_document_shards([source],corpus,tokenizer_path="unused",
        split_policy=SplitPolicy(19,(("train",1),)),max_documents_per_shard=2)
    build_packed_index(corpus,index,split="train",length=16)
    return corpus,index


def test_bounded_plan_pure_peek_m1_m2_m3_and_cursor_checks(actual_data):
    with PackedCampaignData(*actual_data) as data:
        original=data.cursor()
        plans=runner.plan_updates(data)
        assert data.cursor()==original
        assert tuple(p.counts.valid_tokens for p in plans)==runner.TARGETS
        assert [len(data.rank_batches(p,rank=0,world_size=2,physical_batch_size=2).batches) for p in plans]==[1,2,3]
        first=plans[0];data.commit(original,first)
        counts=TrainingCounters(optimizer_updates=1,input_tokens=64,documents=4,microbatches=2)
        saved=runner.cursor_record(data,rank=1)
        assert runner.validate_cursor(data,saved,counts,plans,1)==first.next_cursor
        for bad in (dict(saved,rank=0),dict(saved,physical_batch_per_rank=3)):
            with pytest.raises(ValueError,match="accounting"):
                runner.validate_cursor(data,bad,counts,plans,1)
        for bad in (replace(counts,microbatches=3),replace(counts,input_tokens=65),replace(counts,documents=3)):
            with pytest.raises(ValueError,match="accounting"):
                runner.validate_cursor(data,saved,bad,plans,1)
    with PackedCampaignData(*actual_data) as fresh:
        assert runner.validate_cursor(fresh,saved,counts,plans,1,restore=True)==first.next_cursor
        assert fresh.peek_update(fresh.cursor(),128)==plans[1]


@pytest.mark.parametrize("arm",["B","NFR"])
def test_actual_tiny_model_accepts_native_tokens_and_objective_has_gradients(actual_data,arm):
    torch.set_num_threads(1)
    with PackedCampaignData(*actual_data) as data:
        model,recipe=runner.construct_tiny(data,arm,torch.device("cpu"))
        assert model.config.model_dim==32
        assert model.backbone.backbone.config.vocab_size==data.manifest["vocab_size"]
        assert model.backbone.backbone.config.eos_token_id==50279
        plan=runner.plan_updates(data)[0]
        packed=data.rank_batches(plan,rank=0,world_size=2,physical_batch_size=2)
        assert int(packed.batches[0].input_ids.max())>67
        noise=runner.feedback_noise_for_rows(recipe,packed.keys[0],logical_update=0,
            sequence_length=16,width=32,physical_batch_size=2)
        loss=model.loss_sums(packed.batches[0],backbone_kwargs={"mode":recipe.mode(),
            "feedback_noise":noise,"right_padded_causal":True})
        objective=sum(loss.sums[term]*loss.weights[term]/loss.counts[term]
                      for term in loss.sums if loss.counts[term])
        objective.backward()
        assert torch.isfinite(objective)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters() if parameter.requires_grad)
        before=tree_digests(model.state_dict())
        again,_=runner.construct_tiny(data,arm,torch.device("cpu"))
        assert tree_digests(again.state_dict())==before


def test_cli_requires_pins_and_persistent_scope():
    common=["--corpus","corpus","--index","index","--index-sha256","a"*64,
        "--output-dir",str(runner.ROOT/".runtime/loop-cli"),
        "--checkpoint-root",str(runner.ROOT/".runtime/loop-cli-checkpoints")]
    assert runner.parse_args(common).max_updates==3
    for extra in (["--resume","checkpoint"],["--reference-report","report.json"],
                  ["--request-stop-after","1"],["--checkpoint-seconds","601"],
                  ["--max-updates","4"],["--checkpoint-root","/mnt/localssd/checkpoints"]):
        with pytest.raises(SystemExit):
            runner.parse_args(common+extra)
    args=runner.parse_args(common+["--resume","checkpoint","--resume-manifest-sha256","b"*64])
    assert args.resume==Path("checkpoint")


def test_reference_rejects_incomplete_or_mismatched_lineage(tmp_path):
    path=tmp_path/"report.json"
    values={"schema":runner.SCHEMA,"status":"passed","sources":{"a":"b"},
        "configuration":{"config":1},"updates":{str(i):[] for i in (1,2,3)}}
    path.write_text(json.dumps(values));pin=hashlib.sha256(path.read_bytes()).hexdigest()
    assert runner.load_reference(path,pin,values["sources"],values["configuration"])==values
    for sources,config in (({},values["configuration"]),(values["sources"],{})):
        with pytest.raises(ValueError,match="matching"):
            runner.load_reference(path,pin,sources,config)
    values["status"]="failed";path.write_text(json.dumps(values))
    with pytest.raises(ValueError,match="matching"):
        runner.load_reference(path,hashlib.sha256(path.read_bytes()).hexdigest(),values["sources"],values["configuration"])


def test_cloud_retention_requires_distinct_project_prefix():
    prefix="gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/loop"
    assert runner.storage_location(prefix)==["fast-chunks",prefix.split("/",3)[3]]
    for bad in ("gs://other/bucket",prefix+"/../x",prefix+"//x",prefix+"/"):
        with pytest.raises(ValueError):
            runner.storage_location(bad)


def test_tracking_finish_failure_cannot_leave_passed_report(tmp_path):
    from scripts.olmo_campaign_loop import LifecycleError
    class LocalCoordinator:
        def call(self,phase,function,*,rank_zero=False):
            try:
                return function()
            except Exception as exc:
                raise LifecycleError(str(exc)) from exc
    class FailingTracker:
        record={"status":"running"}
        def finish(self,*,succeeded):
            assert succeeded
            raise OSError("finish failure")
    report={"status":"passed"}
    with pytest.raises(LifecycleError,match="finish failure"):
        runner.finalize_report(tmp_path,LocalCoordinator(),report,FailingTracker(),succeeded=True)
    saved=json.loads((tmp_path/"report.json").read_text())
    assert saved["status"]=="failed" and "finalization_error" in saved


def test_replica_gate_checks_shared_state_but_preserves_rank_local_rng():
    import copy
    from scripts.olmo_campaign_loop import LifecycleError
    first={"raw_gradients":{"g":"same"},"metrics":{"loss":1.},
        "boundary":{"state":{"adam":"same"},"rng":"rank0"},"input":"rank0"}
    second=copy.deepcopy(first);second["boundary"]["rng"]="rank1";second["input"]="rank1"
    runner.validate_replica_rows([first,second])
    for key in ("raw_gradients","metrics","state"):
        bad=copy.deepcopy(second)
        if key=="state":
            bad["boundary"][key]={"changed":True}
        else:
            bad[key]={"changed":True}
        with pytest.raises(LifecycleError,match="replicas differ"):
            runner.validate_replica_rows([first,bad])
