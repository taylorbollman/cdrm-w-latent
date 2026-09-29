"""Real tiny fusion updates, immutable continuation imports and fresh replay."""
import copy
from contextlib import contextmanager
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import TrainingCounters, _rng_state, _restore_rng
from scripts import olmo_fusion_startup_continue as run
from scripts import olmo_fusion_startup_train as train
from scripts import olmo_fusion_startup_update_probe as probe
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_recurrence_precision import arm_contract
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_contract(monkeypatch):
    torch.set_num_threads(1)
    # Shorten only the CPU oracle; production has no corresponding CLI knobs.
    monkeypatch.setattr(run,"PLAN",{**run.PLAN,"start_update":1,"end_update":3,
        "checkpoint_updates":[2,3],"evaluation_updates":[1,2,3],"length":16,
        "physical_batch_size":2,"ce_targets_per_update":25})
    monkeypatch.setitem(train.TRAINING,"ce_targets_per_update",25)
    monkeypatch.setitem(train.TRAINING,"warmup_updates",1)
    # Production's existing preservation helper intentionally requires CUDA.
    @contextmanager
    def cpu_rng():
        state=copy.deepcopy(_rng_state(None))
        try:yield
        finally:_restore_rng(state,None)
    monkeypatch.setattr(run,"preserve_local_rng",cpu_rng)


class FixedData:
    """A bounded cursor oracle; real StartupData ordering has its own tests."""
    manifest_sha256="b"*64
    def __init__(self,batches):
        self.metadata={"microbatches":len(batches),
            "documents":sum(int(b.valid_mask.any(-1).sum()) for b in batches),
            "input_tokens":sum(int(b.valid_mask.sum()) for b in batches)}
    def update_metadata(self,index):
        assert 0 <= index < 3
        return copy.deepcopy(self.metadata)
    def cursor(self,update):
        return {"schema":"olmo-fusion-startup-data-v1","manifest_sha256":self.manifest_sha256,"next_update":update}
    def restore_cursor(self,value):
        if value != self.cursor(value.get("next_update")) or type(value["next_update"]) is not int:
            raise ValueError("Wrong cursor")
        return value["next_update"]


def prepared(tmp_path,precision="bf16_mixed"):
    model,recipe,source,ids,eos=construct(SimpleNamespace(scale="tiny",length=16),"NF",torch.device("cpu"))
    model.backbone.backbone.attention_precision="mixed"
    original={key:getattr(model.backbone.backbone,key) for key in run.RUNTIME_FLAGS}
    source={**source,"sha256":"a"*64}
    fixtures=[fixture_for_update(recipe,model.config.model_dim,rank,0,length=16,token_ids=ids,eos_id=eos,batch_size=2)
              for rank in range(2)]
    batches=tuple(batch for bs,_ in fixtures for batch in bs)
    noises=tuple(noise for _,ns in fixtures for noise in ns)
    data=FixedData(batches)
    contract=arm_contract(model,recipe)
    execution=train.configure_full_fp32(model);train.freeze_for_startup(model)
    config=train.checkpoint_configuration(model,recipe,source,data_manifest={"CPU":"fixed cursor oracle"},
        data_manifest_sha256=data.manifest_sha256,sources={"fixture.py":"c"*64},determinism={"cpu":True},
        runtime={"cpu":True},execution=execution,initial_contract=contract)
    fingerprint={"checkpoint_sha256":source["sha256"],"code":config["sources"],"data_manifest_sha256":data.manifest_sha256}
    optimizer,scheduler=train.build_optimizer(model);counters=TrainingCounters()
    train.train_update(model,recipe,batches,noises,optimizer,scheduler,counters,ce_targets=25)
    origin_path=tmp_path/"origin.pt";tmp_path.mkdir(parents=True,exist_ok=True)
    record=train.save_fusion_checkpoint(origin_path,model,optimizer,scheduler,counters,
        configuration=config,source_fingerprint=fingerprint,data_cursor=data.cursor(1))
    origin=train._checkpoint_payload(model,origin_path,source,expected_sha256=record["sha256"])
    run.configure_path(model,original,run.PATHS[precision])
    configuration=run.configuration_for(model,recipe,origin,origin_sha256=record["sha256"],data=data,
        fixture_sha256="d"*64,precision=precision,sources={"continue.py":"e"*64},runtime={"cpu":True},
        determinism={"cpu":True},original_flags=original)
    continuation_fingerprint={"checkpoint_sha256":source["sha256"],
        "origin_checkpoint_sha256":record["sha256"],"code":configuration["sources"]}
    return SimpleNamespace(**locals())


def advance(context):
    return run.update(context.model,context.recipe,context.batches,context.noises,context.optimizer,
        context.scheduler,context.counters,precision=context.precision,original_flags=context.original,ce_targets=25)


def save(context,path):
    return run.save_continuation(path,context.model,context.optimizer,context.scheduler,context.counters,
        configuration=context.configuration,source_fingerprint=context.continuation_fingerprint,data=context.data,origin=context.origin)


def load(context,path,digest):
    return run.load_continuation(path,context.model,context.optimizer,context.scheduler,expected_sha256=digest,
        configuration=context.configuration,source_fingerprint=context.continuation_fingerprint,data=context.data,origin=context.origin)


@pytest.mark.parametrize("precision",list(run.PATHS))
def test_forward_only_autocast_and_exact_existing_gradient_oracle(tmp_path,precision,monkeypatch):
    context=prepared(tmp_path,precision)
    expected=probe.ce_gradients(context.model,context.recipe,context.fixtures,
        path=run.PATHS[precision],original_flags=context.original)
    gradients=tree_digests(run.fusion_values(context.model,gradients=True))
    previous=run.sdpa_kernel;seen=[];active=[]
    @contextmanager
    def dispatch(backend):
        with previous(backend):
            active.append(backend)
            try:yield
            finally:active.pop()
    def observe(gradient):
        assert active == [run.SDPBackend.MATH]
        assert not torch.is_autocast_enabled("cpu")
        seen.append(True)
    monkeypatch.setattr(run,"sdpa_kernel",dispatch)
    handles=[p.register_hook(observe) for p in train.assert_fusion_only(context.model)]
    try:
        with torch.autocast("cpu",dtype=torch.bfloat16):
            result=run.backward_ce(context.model,context.recipe,context.batches,context.noises,
                precision=precision,original_flags=context.original,ce_targets=25)
            assert torch.is_autocast_enabled("cpu")
    finally:
        for handle in handles:handle.remove()
    assert len(seen)==4 and not active
    assert result["objective"] == expected["objective"]
    assert tree_digests(run.fusion_values(context.model,gradients=True)) == gradients


@pytest.mark.parametrize("precision",list(run.PATHS))
def test_fresh_model_continuation_replays_exact_actual_update(tmp_path,precision):
    context=prepared(tmp_path/"first",precision)
    frozen=train.frozen_state_pins(context.model)
    second=advance(context)
    assert second["lr_used"] == second["lr_next"] == [1e-4]
    assert second["raw_gradient_geometry"]["all"]["norm"] > 0
    assert second["actual_master_delta_geometry"]["all"]["norm"] > 0
    path=tmp_path/"continuation.pt";record=save(context,path)
    saved_boundary=train.boundary_digests(context.model,context.optimizer,context.scheduler,context.counters,context.data.cursor(2))
    golden=advance(context)
    expected=train.boundary_digests(context.model,context.optimizer,context.scheduler,context.counters,context.data.cursor(3))
    other=prepared(tmp_path/"fresh",precision)
    # torch.save filenames influence archive bytes; the new configuration pins
    # the SAME independently authenticated origin, not this fixture's copy.
    other.configuration=copy.deepcopy(context.configuration)
    other.continuation_fingerprint=copy.deepcopy(context.continuation_fingerprint)
    torch.rand(11)
    restored=load(other,path,record["sha256"])
    other.counters=restored["counters"]
    assert train.boundary_digests(other.model,other.optimizer,other.scheduler,other.counters,other.data.cursor(2)) == saved_boundary
    assert advance(other) == golden
    assert train.boundary_digests(other.model,other.optimizer,other.scheduler,other.counters,other.data.cursor(3)) == expected
    assert train.frozen_state_pins(other.model) == frozen
    assert other.counters.optimizer_updates == 3 and other.counters.ce_positions == 75
    assert other.counters.latent_pairs == other.counters.kl_triples == 0
    with pytest.raises(FileExistsError):save(other,path)


@pytest.mark.parametrize("cache",[False,True])
def test_common_fp32_evaluation_preserves_all_state_and_caller_context(tmp_path,cache):
    context=prepared(tmp_path)
    before=tree_digests(context.model.state_dict())
    boundary=train.boundary_digests(context.model,context.optimizer,context.scheduler,context.counters,context.data.cursor(1))
    flags={key:getattr(context.model.backbone.backbone,key) for key in run.RUNTIME_FLAGS}
    with torch.autocast("cpu",dtype=torch.bfloat16,cache_enabled=cache):
        result=run.evaluate_fp32(context.model,context.recipe,context.fixtures,original_flags=context.original)
        assert torch.is_autocast_enabled("cpu") and torch.is_autocast_cache_enabled() is cache
    assert all(result["checks"].values()) and result["ce_targets"] == 25
    assert tree_digests(context.model.state_dict()) == before
    assert train.boundary_digests(context.model,context.optimizer,context.scheduler,context.counters,context.data.cursor(1)) == boundary
    assert {key:getattr(context.model.backbone.backbone,key) for key in run.RUNTIME_FLAGS} == flags
    run.configure_path(context.model,context.original,run.FP32)
    with torch.no_grad(),run.sdpa_kernel(run.SDPBackend.MATH):
        expected=sum(float(train.ce_loss_sums(context.model,context.recipe,batch,noise).sums["ce"])
            for batch,noise in zip(context.batches,context.noises))
    assert result["ce_sum"] == expected


def test_evaluation_exception_still_restores_flags_rng_and_autocast(tmp_path,monkeypatch):
    context=prepared(tmp_path)
    flags={key:getattr(context.model.backbone.backbone,key) for key in run.RUNTIME_FLAGS}
    rng=tree_digests(_rng_state(None))
    def fail(*args,**kwargs):
        torch.rand(5)
        raise RuntimeError("injected evaluation error")
    monkeypatch.setattr(run,"ce_loss_sums",fail)
    with torch.autocast("cpu",dtype=torch.bfloat16,cache_enabled=True):
        with pytest.raises(RuntimeError,match="injected"):
            run.evaluate_fp32(context.model,context.recipe,context.fixtures,original_flags=context.original)
        assert torch.is_autocast_enabled("cpu") and torch.is_autocast_cache_enabled()
    assert flags == {key:getattr(context.model.backbone.backbone,key) for key in run.RUNTIME_FLAGS}
    assert rng == tree_digests(_rng_state(None))


@pytest.mark.parametrize("corruption",["kind","precision","origin","sources","frozen","counter","cursor",
    "ownership","foreign_moment","step","moment_nan","negative_variance","lr","scheduler","scale","dtype","mode"])
def test_new_checkpoint_corruption_rejected_before_mutation(tmp_path,corruption):
    context=prepared(tmp_path/"fixture");advance(context)
    path=tmp_path/"good.pt";save(context,path)
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if corruption in ("kind","precision","origin","sources","frozen"):
        key={"kind":"kind","precision":"precision","origin":"origin_checkpoint_sha256",
             "sources":"sources","frozen":"frozen_state_pins"}[corruption]
        payload["configuration"][key]="wrong"
    elif corruption=="counter":payload["counters"]["input_tokens"]+=1
    elif corruption=="cursor":payload["data_cursor"]["next_update"]=1
    elif corruption=="ownership":payload["optimizer_ownership"][0].reverse()
    elif corruption=="foreign_moment":payload["optimizer"]["state"][77]={}
    elif corruption=="step":next(iter(payload["optimizer"]["state"].values()))["step"]+=1
    elif corruption=="moment_nan":next(iter(payload["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
    elif corruption=="negative_variance":next(iter(payload["optimizer"]["state"].values()))["exp_avg_sq"].fill_(-1)
    elif corruption=="lr":payload["optimizer"]["param_groups"][0]["lr"]*=2
    elif corruption=="scheduler":payload["scheduler"]["last_epoch"]+=1
    elif corruption=="scale":payload["model"]["output_scale"]+=.1
    elif corruption=="dtype":payload["model"]["state_proj.weight"]=payload["model"]["state_proj.weight"].bfloat16()
    elif corruption=="mode":payload["module_training"][""]=False
    changed=tmp_path/"bad.pt";torch.save(payload,changed)
    before=tree_digests({"model":context.model.state_dict(),"optimizer":context.optimizer.state_dict(),
        "scheduler":context.scheduler.state_dict(),"rng":_rng_state(None)})
    with pytest.raises(ValueError):load(context,changed,sha256_file(changed))
    assert tree_digests({"model":context.model.state_dict(),"optimizer":context.optimizer.state_dict(),
        "scheduler":context.scheduler.state_dict(),"rng":_rng_state(None)}) == before


def test_progress_and_cli_hard_bounds(tmp_path):
    context=prepared(tmp_path)
    run.validate_progress(context.data,context.counters,context.data.cursor(1),context.origin["counters"])
    with pytest.raises(ValueError):run.validate_progress(context.data,context.counters,context.data.cursor(2),context.origin["counters"])
    arguments=["--precision","bf16_mixed","--origin-checkpoint",str(tmp_path/"origin.pt"),
        "--origin-checkpoint-sha256","a"*64,"--fixture",str(tmp_path/"fixture.json"),
        "--fixture-sha256","b"*64,"--output-dir",str(run.ROOT/".runtime/cpu-unexecuted")]
    assert run.parse_args(arguments).precision == "bf16_mixed"
    for extra in (["--end-update","500"],["--golden-report",str(tmp_path/"golden.json")],
                  ["--resume",str(tmp_path/"resume.pt"),"--resume-sha256","bad"]):
        with pytest.raises(SystemExit):run.parse_args(arguments+extra)


def test_golden_report_requires_complete_exact_trajectory(tmp_path):
    config={"kind":run.KIND};sources={"source":"a"*64}
    report={"schema":run.KIND,"status":"completed_segment","passed":True,"precision":"bf16_mixed",
        "configuration":config,"sources":sources,"starting_update":1,"final_counters":{"optimizer_updates":3},
        "physical_optimizer_updates":2,"updates":[{"exact":{"metrics":{"update":i}}} for i in (2,3)],
        "evaluations":[{"update":i} for i in (1,2,3)],"integrity":{"all":True}}
    path=tmp_path/"golden.json";path.write_text(json.dumps(report))
    assert run.load_golden(path,sha256_file(path),configuration=config,sources=sources) == report
    for key,value in (("physical_optimizer_updates",1),("starting_update",2),("evaluations",[]),("integrity",{})):
        bad={**report,key:value};path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):run.load_golden(path,sha256_file(path),configuration=config,sources=sources)
