"""CPU-only authentication, rank mapping and strict destination recovery tests."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import CampaignRecipe, CampaignTokenSchedule, build_campaign_adamw, build_campaign_model
from cdrm.pretrained.distributed_checkpoint import (
    _local_rng, _metadata, _write_checkpoint, load_distributed_checkpoint, save_distributed_checkpoint,
)
from cdrm.pretrained.lm_training import TrainingCounters, _plain
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_campaign_execution_restore import committed_metadata, MANIFEST_FIELDS
from scripts.olmo_topology_contract import (
    canonical_parent_cursor, cursor_record, destination_configuration, destination_fingerprint,
    expected_counters_since_origin, make_topology_contract, validate_topology_contract,
)
from scripts.olmo_topology_checkpoint import (
    ROOT, load_topology_checkpoint, mapped_rng_state, validate_parent_payload,
)


def make_model():
    recipe = CampaignRecipe("NFR", sequence_length=6, effective_valid_tokens=42, rt_layers=(0,1),
                            warmup_tokens=84, document_policy="continuous-stream-v1")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(985)
        model = build_campaign_model(OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
            attention_precision="fp32"), recipe)
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    scheduler = CampaignTokenSchedule(optimizer,[42]*4,warmup_tokens=84,start_fraction=.1)
    return model,optimizer,scheduler,recipe


def generator_pair():
    return {"data":torch.Generator().manual_seed(45),"local":torch.Generator().manual_seed(91)}


def source_payload(world=2):
    model,optimizer,scheduler,recipe = make_model()
    for index,parameter in enumerate(model.parameters()):
        if parameter.requires_grad:
            parameter.grad=torch.full_like(parameter,.001*(index+1))
    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    counters=TrainingCounters(optimizer_updates=1,microbatches=world*((7+world*2-1)//(world*2)),
        documents=7,input_tokens=42,ce_positions=35,latent_pairs=35,kl_triples=28)
    configuration={"schema":"tiny-topology-source-v1","world_size":world,"model":model.config.to_dict(),
        "backbone":model.backbone.backbone.config.to_dict(),"recipe":recipe.to_dict(),
        "training":{"precision":"fp32","max_grad_norm":1.},"schedule":scheduler.checkpoint_contract(),
        "ddp":{"static_graph":True},"execution":{"precision":"fp32","ordinary_attention":"math"},
        "data":{"schema":"tiny-ordered-fixture","index_manifest_sha256":"a"*64}}
    metadata=_metadata(model,optimizer,scheduler,configuration,{"sha256":"b"*64},world)
    rank_states=[]
    for rank in range(world):
        rng=_local_rng(torch.device("cpu"),generator_pair())
        rng["torch_cpu"]=torch.Generator().manual_seed(100+rank).get_state()
        rank_states.append({"rank":rank,"rng":rng,"data_cursor":cursor_record(
            {"manifest_sha256":"a"*64,"split":"train","next_chunk":7,"next_update":1},
            rank=rank,world_size=world,batch_size=2)})
    payload={"metadata":metadata,"counters":asdict(counters),"module_training":{n:m.training for n,m in model.named_modules()},
        "scheduler_state":_plain(scheduler.state_dict()),
        "optimizer_groups":_plain([{k:v for k,v in g.items() if k!="params"} for g in optimizer.param_groups]),
        "model":copy.deepcopy(model.state_dict()),"optimizer":copy.deepcopy(optimizer.state_dict()),
        "scheduler":copy.deepcopy(scheduler.state_dict()),"rank_states":rank_states}
    manifest={"schema":metadata["schema"],"world_size":world,"metadata":copy.deepcopy(metadata),
        "counters":asdict(counters),"rank_cursors":[copy.deepcopy(r["data_cursor"]) for r in rank_states],
        "state":{"sha256":"c"*64},"manifest_sha256":"d"*64}
    return payload,manifest


def contract(manifest,world=1,rank_map=None):
    name="scripts/olmo_topology_contract.py"
    return make_topology_contract(manifest,world_size=world,physical_batch_per_rank=2,
        lineage="topology-unit-test",sources={name:sha256_file(ROOT/name)},rng_rank_map=rank_map)


@pytest.mark.parametrize("source,destination",[(1,2),(2,1),(2,2),(2,8)])
def test_declared_topology_retains_cursor_schedule_and_root_identity(source,destination):
    _,manifest=source_payload(source)
    migration=contract(manifest,destination)
    assert validate_topology_contract(migration,manifest)==migration
    config=destination_configuration(manifest["metadata"]["configuration"],migration)
    assert config["world_size"]==destination
    assert config["schedule"]==manifest["metadata"]["configuration"]["schedule"]
    assert config["data"]==manifest["metadata"]["configuration"]["data"]
    assert migration["origin_cursor"]==canonical_parent_cursor(manifest)
    child=copy.deepcopy(manifest)
    child["world_size"]=child["metadata"]["world_size"]=destination
    child["rank_cursors"]=[cursor_record(migration["origin_cursor"],rank=rank,
        world_size=destination,batch_size=2) for rank in range(destination)]
    child["metadata"]["configuration"]=config
    child["manifest_sha256"]="e"*64
    assert contract(child,destination)["root_manifest_sha256"]==manifest["manifest_sha256"]


@pytest.mark.parametrize("fault",["counters","cursor","rank","rng_map","extra","schedule"])
def test_contract_rejects_undeclared_or_inconsistent_change(fault):
    _,manifest=source_payload()
    migration=contract(manifest)
    if fault=="counters": migration["origin_counters"]["input_tokens"]+=1
    if fault=="cursor": manifest["rank_cursors"][1]["cursor"]["next_chunk"]+=1
    if fault=="rank": manifest["rank_cursors"][1]["rank"]=0
    if fault=="rng_map": migration["rng"]["source_rank_for_destination"]=[2]
    if fault=="extra": migration["skip_validation"]=True
    if fault=="schedule": migration["schedule_contract"]["planned_updates"]+=1
    with pytest.raises(ValueError): validate_topology_contract(migration,manifest)


def test_physical_microbatch_counter_keeps_historical_origin():
    origin=TrainingCounters(optimizer_updates=127,microbatches=5588,documents=65024,input_tokens=66584576)
    plan=SimpleNamespace(rows=[None]*512,counts=SimpleNamespace(valid_tokens=524288,packed_rows=512,
        ce_targets=512*1023,latent_pairs=1,kl_triples=1))
    recipe=CampaignRecipe("NFR")
    one=expected_counters_since_origin(origin,[None]*127+[plan],128,recipe,world_size=1,batch_size=12)
    two=expected_counters_since_origin(origin,[None]*127+[plan],128,recipe,world_size=2,batch_size=12)
    assert one.microbatches==5631 and two.microbatches==5632
    a,b=asdict(one),asdict(two);a.pop("microbatches");b.pop("microbatches")
    assert a==b and one.input_tokens==67108864


def test_retained_rank_one_maps_to_local_rank_zero_without_rng_draw():
    payload,manifest=source_payload()
    migration=contract(manifest,1,[1])
    before=torch.get_rng_state().clone()
    state,policy=mapped_rng_state(payload,migration,rank=0,device="cpu",generators=generator_pair())
    assert policy["source_rank"]==1
    assert torch.equal(state["torch_cpu"],payload["rank_states"][1]["rng"]["torch_cpu"])
    assert torch.equal(before,torch.get_rng_state())


def test_added_rank_seed_is_explicit_deterministic_and_not_global_rng():
    payload,manifest=source_payload(1)
    migration=contract(manifest,2)
    before=tree_digests(_local_rng(torch.device("cpu"),generator_pair()))
    first,policy=mapped_rng_state(payload,migration,rank=1,device="cpu",generators=generator_pair())
    second,_=mapped_rng_state(payload,migration,rank=1,device="cpu",generators=generator_pair())
    assert policy["kind"]=="seeded-additional-rank"
    assert tree_digests(first)==tree_digests(second)
    assert before==tree_digests(_local_rng(torch.device("cpu"),generator_pair()))
    assert not torch.equal(first["torch_cpu"],payload["rank_states"][0]["rng"]["torch_cpu"])


def test_cannot_reseed_a_surviving_rank_or_reuse_one_stream_twice():
    _,manifest=source_payload(2)
    with pytest.raises(ValueError,match="surviving"):
        contract(manifest,1,[None])
    with pytest.raises(ValueError,match="at most once"):
        contract(manifest,2,[0,0])


def test_fresh_destination_identity_matches_unchanged_storage_contract():
    _,source=source_payload(2)
    migration=contract(source,1)
    manifest={key:copy.deepcopy(source[key]) for key in MANIFEST_FIELDS}
    manifest["world_size"]=manifest["metadata"]["world_size"]=1
    manifest["metadata"]["configuration"]=destination_configuration(source["metadata"]["configuration"],migration)
    manifest["metadata"]["source_fingerprint"]=destination_fingerprint(migration)
    manifest["rank_cursors"]=[cursor_record(migration["origin_cursor"],rank=0,world_size=1,batch_size=2)]
    manifest["state"].update(filename="state.pt",size_bytes=100)
    identity=committed_metadata(manifest)
    assert identity["payload"]["schema"]=="olmo-topology-execution-payload-v1"
    assert identity["payload"]["partition"]["world_size"]==1
    assert identity["sha256"]==destination_fingerprint(migration)["execution_identity_sha256"]


def test_exact_comparison_origin_tracks_boundary_while_family_root_remains_stable():
    _,source=source_payload(2)
    first=contract(source,1)
    child=copy.deepcopy(source)
    child["manifest_sha256"]="e"*64
    child["world_size"]=child["metadata"]["world_size"]=1
    child["metadata"]["configuration"]=destination_configuration(source["metadata"]["configuration"],first)
    child["rank_cursors"]=[cursor_record(first["origin_cursor"],rank=0,world_size=1,batch_size=2)]
    unadvanced=contract(child,2)
    assert unadvanced["comparison_origin_manifest_sha256"]==source["manifest_sha256"]
    assert unadvanced["parent_manifest_sha256"]==child["manifest_sha256"]
    child["manifest_sha256"]="f"*64
    for name in child["counters"]:
        child["counters"][name]*=2
    child["rank_cursors"][0]["cursor"].update(next_update=2,next_chunk=14)
    advanced=contract(child,2)
    assert advanced["comparison_origin_manifest_sha256"]==child["manifest_sha256"]
    assert advanced["root_manifest_sha256"]==source["manifest_sha256"]


@pytest.mark.parametrize("fault",[None,"model","clock","missing_moment","moment_type","foreign_state","schedule","group","loss_config"])
def test_parent_validation_before_any_live_mutation(fault):
    payload,manifest=source_payload()
    migration=contract(manifest)
    model,optimizer,scheduler,_=make_model()
    before=tree_digests({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"schedule":scheduler.state_dict()})
    first=next(iter(payload["optimizer"]["state"]))
    if fault=="model": payload["model"].pop(next(iter(payload["model"])))
    if fault=="clock": payload["optimizer"]["state"][first]["step"].fill_(9)
    if fault=="missing_moment": payload["optimizer"]["state"][first].pop("exp_avg")
    if fault=="moment_type": payload["optimizer"]["state"][first]["exp_avg"]="invalid"
    if fault=="foreign_state": payload["optimizer"]["state"][987654]=copy.deepcopy(payload["optimizer"]["state"][first])
    if fault=="schedule": payload["scheduler"]["token_prefix"]=(0,42,85,126,168)
    if fault=="group": payload["optimizer"]["param_groups"][0]["param_names"]=["foreign"]
    if fault=="loss_config": model.config=replace(model.config,lambda_kl=.1)
    if fault is None:
        assert validate_parent_payload(payload,manifest,model,optimizer,scheduler,migration).optimizer_updates==1
    else:
        with pytest.raises(ValueError): validate_parent_payload(payload,manifest,model,optimizer,scheduler,migration)
    after=tree_digests({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"schedule":scheduler.state_dict()})
    assert before==after


@pytest.mark.skipif(not dist.is_gloo_available(),reason="Gloo unavailable")
def test_one_rank_import_then_unchanged_strict_loader_fresh_model(tmp_path):
    payload,_=source_payload(2)
    manifest=_write_checkpoint(tmp_path/"source",payload)
    migration=contract(manifest,1,[1])
    dist.init_process_group("gloo",rank=0,world_size=1,init_method=f"file://{tmp_path/'group'}",
                            timeout=timedelta(seconds=30))
    try:
        model,optimizer,scheduler,_=make_model()
        generators=generator_pair()
        loaded=load_topology_checkpoint(tmp_path/"source",model,optimizer,scheduler=scheduler,migration=migration,
            expected_manifest_sha256=manifest["manifest_sha256"],generators=generators,device="cpu")
        assert loaded["counters"]==TrainingCounters(**payload["counters"])
        assert tree_digests(model.state_dict())==tree_digests(payload["model"])
        assert tree_digests(optimizer.state_dict())==tree_digests(payload["optimizer"])
        assert tree_digests(scheduler.state_dict())==tree_digests(payload["scheduler"])
        assert torch.equal(torch.get_rng_state(),payload["rank_states"][1]["rng"]["torch_cpu"])
        saved=save_distributed_checkpoint(tmp_path/"destination",model,optimizer,scheduler=scheduler,
            counters=loaded["counters"],data_cursor=loaded["data_cursor"],configuration=loaded["configuration"],
            source_fingerprint=loaded["source_fingerprint"],generators=generators,device="cpu")
        assert committed_metadata({key:saved[key] for key in MANIFEST_FIELDS}) == loaded["configuration"]["execution_identity"]
        fresh,new_optimizer,new_scheduler,_=make_model()
        restored=load_distributed_checkpoint(tmp_path/"destination",fresh,new_optimizer,scheduler=new_scheduler,
            configuration=loaded["configuration"],source_fingerprint=loaded["source_fingerprint"],
            expected_manifest_sha256=saved["manifest_sha256"],generators=generator_pair(),device="cpu")
        assert restored["counters"]==loaded["counters"]
        assert tree_digests(fresh.state_dict())==tree_digests(model.state_dict())
        assert tree_digests(new_optimizer.state_dict())==tree_digests(optimizer.state_dict())
        assert new_scheduler.state_dict()==scheduler.state_dict()
    finally:
        dist.destroy_process_group()


def _added_rank_worker(rank,root,source_manifest):
    root=Path(root)
    torch.set_num_threads(1)
    dist.init_process_group("gloo",rank=rank,world_size=2,init_method=f"file://{root/'group-two'}",
                            timeout=timedelta(seconds=60))
    try:
        model,optimizer,scheduler,_=make_model()
        migration=contract(source_manifest,2)
        generators=generator_pair()
        loaded=load_topology_checkpoint(root/"source-one",model,optimizer,scheduler=scheduler,migration=migration,
            expected_manifest_sha256=source_manifest["manifest_sha256"],generators=generators,device="cpu")
        assert loaded["data_cursor"]["rank"]==rank
        assert loaded["counters"].optimizer_updates==1
        assert loaded["migration_receipt"]["rng_mapping_by_rank"][1]["kind"]=="seeded-additional-rank"
        hashes=[None,None]
        dist.all_gather_object(hashes,tree_digests({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                                                 "schedule":scheduler.state_dict()}))
        assert hashes[0]==hashes[1]
        save_distributed_checkpoint(root/"destination-two",model,optimizer,scheduler=scheduler,
            counters=loaded["counters"],data_cursor=loaded["data_cursor"],configuration=loaded["configuration"],
            source_fingerprint=loaded["source_fingerprint"],generators=generators,device="cpu")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(),reason="Gloo unavailable")
def test_actual_two_rank_import_with_added_rng_stream(tmp_path):
    payload,_=source_payload(1)
    manifest=_write_checkpoint(tmp_path/"source-one",payload)
    mp.spawn(_added_rank_worker,args=(str(tmp_path),manifest),nprocs=2,join=True)
    assert (tmp_path/"destination-two"/"manifest.json").is_file()
