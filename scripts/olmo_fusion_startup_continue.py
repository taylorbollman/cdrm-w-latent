#!/usr/bin/env python3
"""Bounded matched fusion-only CE continuation, plus fresh-process BF16 replay.

Each trajectory starts from strict startup128. New checkpoints have a separate
continuation kind; no historical loader or checkpoint contract is weakened.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import (CHECKPOINT_SCHEMA, TrainingCounters, _rng_state,
    load_training_checkpoint, save_training_checkpoint, optimizer_ownership, parameter_layout)
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct
from scripts.olmo_campaign_precision_bridge import configure_path
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, fixture_pins
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_data import StartupData, DEFAULT_ROOT, DEFAULT_MANIFEST_SHA256
from scripts.olmo_fusion_startup_long_probe import load_long_fixture
from scripts.olmo_fusion_startup_train import (TRAINING, FUSION_NAMES, _checkpoint_payload,
    assert_fusion_only, build_optimizer, ce_loss_sums, freeze_for_startup, frozen_state_pins,
    load_fusion_checkpoint, boundary_digests, retain_checkpoint, source_hashes as training_sources)
from scripts.olmo_fusion_startup_update_probe import (fusion_values, moment_values, difference, geometry,
    source_hashes as update_sources)
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

KIND = "olmo-fusion-startup-continuation-v1"
PLAN = {"start_update":128,"end_update":144,"checkpoint_updates":[136,144],
        "evaluation_updates":[128,136,144],"length":128,"physical_batch_size":8,
        "ce_targets_per_update":8192,"checkpoint_interval_seconds":600}
PATHS = {"fp32":FP32,"bf16_mixed":BF16}


def source_hashes():
    sources = update_sources()
    for name in ("scripts/olmo_fusion_startup_continue.py","tests/test_fusion_startup_continue.py",
                 "docs/reports/olmo-fusion-startup/continuation-protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def normalized_sha(value):
    return hashlib.sha256(json.dumps(tree_digests(value),sort_keys=True,separators=(",",":")).encode()).hexdigest()


def validate_progress(data,counters,cursor,origin_counters):
    update = counters.optimizer_updates
    if not PLAN["start_update"] <= update <= PLAN["end_update"] or data.restore_cursor(cursor) != update:
        raise ValueError("Continuation cursor lies outside its fixed completed-update range")
    expected = copy.deepcopy(origin_counters)
    for index in range(PLAN["start_update"],update):
        metadata = data.update_metadata(index)
        expected["optimizer_updates"] += 1
        expected["ce_positions"] += PLAN["ce_targets_per_update"]
        for name in ("microbatches","documents","input_tokens"):
            expected[name] += metadata[name]
    if asdict(counters) != expected or counters.latent_pairs or counters.kl_triples:
        raise ValueError("Continuation counters differ from the inherited origin and exact data prefix")


def validate_adam_payload(payload,origin):
    """Check complete finite moments, fixed hyperparameters and inherited steps."""
    update = payload["counters"]["optimizer_updates"]
    if payload["optimizer_ownership"] != [["state_proj.weight","token_gate.weight"]]:
        raise ValueError("Continuation Adam ownership differs")
    groups = payload["optimizer"]["param_groups"]
    original_groups = origin["optimizer"]["param_groups"]
    if len(groups) != 1 or len(original_groups) != 1:
        raise ValueError("Continuation requires the original single fusion Adam group")
    if {k:v for k,v in groups[0].items() if k != "params"} != {k:v for k,v in original_groups[0].items() if k != "params"}:
        raise ValueError("Continuation Adam hyperparameters changed")
    identifiers = groups[0]["params"]
    if len(identifiers) != 2 or len(set(identifiers)) != 2 or set(payload["optimizer"]["state"]) != set(identifiers):
        raise ValueError("Continuation Adam moments are missing or foreign")
    for identifier,name in zip(identifiers,("state_proj.weight","token_gate.weight")):
        state = payload["optimizer"]["state"][identifier]
        if set(state) != {"step","exp_avg","exp_avg_sq"}:
            raise ValueError("Continuation Adam moment inventory differs")
        if not isinstance(state["step"],torch.Tensor) or state["step"].numel() != 1 or float(state["step"]) != update:
            raise ValueError("Continuation Adam step differs from completed counter")
        for term in ("exp_avg","exp_avg_sq"):
            value = state[term]
            if (not isinstance(value,torch.Tensor) or value.dtype != torch.float32
                    or value.shape != payload["model"][name].shape or not bool(torch.isfinite(value).all())
                    or (term == "exp_avg_sq" and bool((value < 0).any()))):
                raise ValueError("Continuation Adam moments have invalid shape/dtype/value")
    expected_scheduler = copy.deepcopy(origin["scheduler"])
    delta = update-PLAN["start_update"]
    for name in ("last_epoch","_step_count"):
        expected_scheduler[name] += delta
    if payload["scheduler"] != expected_scheduler:
        raise ValueError("Continuation scheduler differs from unchanged inherited schedule")


def configuration_for(model,recipe,origin,*,origin_sha256,data,fixture_sha256,precision,sources,runtime,determinism,original_flags):
    if precision not in PATHS:
        raise ValueError("Unknown continuation precision")
    return {"kind":KIND,"plan":copy.deepcopy(PLAN),"precision":precision,
        "optimizer_training_settings":copy.deepcopy(TRAINING),"origin_checkpoint_sha256":origin_sha256,
        "origin_configuration_sha256":normalized_sha(origin["configuration"]),"origin_counters":origin["counters"],
        "origin_scheduler":origin["scheduler"],"source_checkpoint":origin["configuration"]["source_checkpoint"],
        "model_config":model.backbone.config.to_dict(),"nextlat_config":model.config.to_dict(),
        "fusion_config":model.backbone.fusion_config.to_dict(),"recipe":recipe.to_dict(),
        "frozen_state_pins":frozen_state_pins(model),"full_module_training":{name:m.training for name,m in model.named_modules()},
        "full_parameter_layout":parameter_layout(model),"production_flags":original_flags,
        "data_manifest_sha256":data.manifest_sha256,"heldout_fixture_sha256":fixture_sha256,
        "sources":sources,"runtime":runtime,"determinism":determinism,
        "evaluation":"same fixed long dev CE under FP32/math, no gradients, fixed noise and preserved RNG"}


def validate_continuation_payload(model,payload,*,configuration,source_fingerprint,data,origin):
    """Supplement the generic loader before any live mutation or RNG restore."""
    if (payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("configuration") != tree_digests(configuration)
            or payload.get("source_fingerprint") != tree_digests(source_fingerprint)
            or configuration.get("kind") != KIND or configuration.get("plan") != PLAN
            or configuration.get("frozen_state_pins") != frozen_state_pins(model)
            or payload.get("parameter_layout") != parameter_layout(model.backbone.fusion)
            or payload.get("module_training") != origin["module_training"]
            or {name:m.training for name,m in model.named_modules()} != configuration["full_module_training"]):
        raise ValueError("Continuation source/configuration/frozen ownership contract differs")
    current = model.backbone.fusion.state_dict()
    if set(payload.get("model",{})) != set(current):
        raise ValueError("Continuation fusion tensor inventory differs")
    for name,tensor in payload["model"].items():
        if (not isinstance(tensor,torch.Tensor) or tensor.dtype != torch.float32 or tensor.dtype != current[name].dtype
                or tensor.shape != current[name].shape or not bool(torch.isfinite(tensor).all())):
            raise ValueError("Continuation fusion tensor shape/dtype/finiteness differs")
    if tree_digests(payload["model"]["output_scale"]) != tree_digests(origin["model"]["output_scale"]):
        raise ValueError("Continuation changed frozen fusion output scale")
    validate_progress(data,TrainingCounters(**payload["counters"]),payload["data_cursor"],origin["counters"])
    validate_adam_payload(payload,origin)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Continuation import requires cleared gradients")
    assert_fusion_only(model)


def load_continuation(path,model,optimizer,scheduler,*,expected_sha256,configuration,source_fingerprint,data,origin):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("Continuation checkpoint differs from immutable SHA256")
    payload = torch.load(path,map_location="cpu",weights_only=True)
    if sha256_file(path) != expected_sha256:
        raise ValueError("Continuation checkpoint changed while reading")
    validate_continuation_payload(model,payload,configuration=configuration,source_fingerprint=source_fingerprint,data=data,origin=origin)
    ids = {name:id(parameter) for name,parameter in model.named_parameters()}
    flags = {name:parameter.requires_grad for name,parameter in model.named_parameters()}
    restored = load_training_checkpoint(path,model.backbone.fusion,optimizer,scheduler=scheduler,
        configuration=configuration,source_fingerprint=source_fingerprint,expected_sha256=expected_sha256)
    if (ids != {name:id(parameter) for name,parameter in model.named_parameters()}
            or flags != {name:parameter.requires_grad for name,parameter in model.named_parameters()}
            or frozen_state_pins(model) != configuration["frozen_state_pins"]
            or configuration["full_module_training"] != {name:m.training for name,m in model.named_modules()}
            or tree_digests(model.backbone.fusion.state_dict()) != tree_digests(payload["model"])):
        raise AssertionError("Continuation import changed ownership or frozen state")
    return restored


def save_continuation(path,model,optimizer,scheduler,counters,*,configuration,source_fingerprint,data,origin):
    assert_fusion_only(model)
    cursor = data.cursor(counters.optimizer_updates)
    validate_progress(data,counters,cursor,origin["counters"])
    if (frozen_state_pins(model) != configuration["frozen_state_pins"]
            or configuration["full_module_training"] != {name:m.training for name,m in model.named_modules()}):
        raise AssertionError("Frozen state or module mode changed before continuation checkpoint")
    # Validate the moment/schedule boundary before publication as well as load.
    validate_adam_payload({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
        "model":model.backbone.fusion.state_dict(),"optimizer_ownership":optimizer_ownership(model.backbone.fusion,optimizer),
        "counters":asdict(counters)},origin)
    return save_training_checkpoint(path,model.backbone.fusion,optimizer,scheduler=scheduler,counters=counters,
        data_cursor=cursor,configuration=configuration,source_fingerprint=source_fingerprint)


def backward_ce(model,recipe,batches,noises,*,precision,original_flags,ce_targets):
    assert_fusion_only(model)
    if (precision not in PATHS or not batches or len(batches) != len(noises)
            or type(ce_targets) is not int or ce_targets < 1):
        raise ValueError("Continuation requires matched nonempty batch/noise records")
    count = sum(int(build_nextlat_masks(batch,document_policy="isolated-v1")["ce"].sum()) for batch in batches)
    if count != ce_targets:
        raise ValueError("Continuation CE denominator differs from actual masks")
    configure_path(model,original_flags,PATHS[precision])
    device = next(model.parameters()).device
    backend = SDPBackend.FLASH_ATTENTION if precision == "bf16_mixed" and device.type == "cuda" else SDPBackend.MATH
    model.zero_grad(set_to_none=True)
    metrics = {"objective":0.,"ce_sum":0.,"ce_targets":count,"microbatches":len(batches),
        "documents":sum(int(batch.valid_mask.any(-1).sum()) for batch in batches),
        "input_tokens":sum(int(batch.valid_mask.sum()) for batch in batches),"pass_ce_sums":[0.]*4}
    for batch,noise in zip(batches,noises):
        with sdpa_kernel(backend),torch.autocast(device.type,enabled=False):
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=precision == "bf16_mixed",cache_enabled=False):
                losses = ce_loss_sums(model,recipe,batch.to(device),tuple(value.to(device) for value in noise))
                objective = losses.sums["ce"]/ce_targets
            if not bool(torch.isfinite(objective)) or not objective.requires_grad:
                raise FloatingPointError("Continuation loss nonfinite or detached")
            objective.backward()
        metrics["objective"] += float(objective.detach());metrics["ce_sum"] += float(losses.sums["ce"].detach())
        for index,loss in enumerate(losses.pass_losses):metrics["pass_ce_sums"][index] += float(loss.sums["ce"].detach())
    fusion_values(model,gradients=True)
    assert_fusion_only(model)
    return metrics


def update(model,recipe,batches,noises,optimizer,scheduler,counters,*,precision,original_flags,ce_targets):
    parameters = assert_fusion_only(model)
    if optimizer_ownership(model.backbone.fusion,optimizer) != [["state_proj.weight","token_gate.weight"]]:
        raise ValueError("Continuation optimizer ownership differs before update")
    before = fusion_values(model)
    rates = [group["lr"] for group in optimizer.param_groups]
    try:
        metrics = backward_ce(model,recipe,batches,noises,precision=precision,original_flags=original_flags,ce_targets=ce_targets)
        raw = fusion_values(model,gradients=True)
        if any(not bool(value.any()) for value in raw.values()):
            raise FloatingPointError("A fusion matrix has a zero raw gradient")
        norm = torch.nn.utils.clip_grad_norm_(parameters,TRAINING["max_grad_norm"],error_if_nonfinite=True,foreach=False)
        clipped_pins = tree_digests(fusion_values(model,gradients=True))
        optimizer.step();scheduler.step()
        after = fusion_values(model)
        _,steps = moment_values(model,optimizer)
        if any(value != counters.optimizer_updates+1 for value in steps.values()):
            raise AssertionError("Continuation Adam did not advance exactly one inherited step")
        delta = difference(after,before)
        if any(not bool(value.any()) for value in delta.values()):
            raise FloatingPointError("A fusion matrix failed to update")
    finally:model.zero_grad(set_to_none=True)
    counters.optimizer_updates += 1;counters.ce_positions += ce_targets
    for key in ("microbatches","documents","input_tokens"):setattr(counters,key,getattr(counters,key)+metrics[key])
    return {**metrics,"update":counters.optimizer_updates,"counters":asdict(counters),"lr_used":rates,
        "lr_next":[group["lr"] for group in optimizer.param_groups],"gradient_norm_before_clip":float(norm),
        "clip_scale":min(1.,TRAINING["max_grad_norm"]/(float(norm)+1e-6)),
        "raw_gradient_pins":tree_digests(raw),"clipped_gradient_pins":clipped_pins,
        "raw_gradient_geometry":geometry(raw),"actual_master_delta_geometry":geometry(delta),
        "actual_master_delta_pins":tree_digests(delta)}


def evaluate_fp32(model,recipe,fixtures,*,original_flags):
    """Common FP32 read-only CE, keeping train/eval modes and caller state."""
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Evaluation requires cleared gradients")
    before = tree_digests(model.backbone.fusion.state_dict())
    frozen = frozen_state_pins(model)
    flags = {name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS}
    modes = {name:module.training for name,module in model.named_modules()}
    rng = tree_digests(_rng_state(None));inputs = fixture_pins(fixtures)
    cache = torch.is_autocast_cache_enabled()
    count = sum(int(build_nextlat_masks(batch,document_policy="isolated-v1")["ce"].sum())
                for batches,_ in fixtures for batch in batches)
    total,passes = 0.,[0.]*4
    device = next(model.parameters()).device
    try:
        configure_path(model,original_flags,FP32)
        with preserve_local_rng(),torch.no_grad(),sdpa_kernel(SDPBackend.MATH),torch.autocast(device.type,enabled=False):
            for batches,noises in fixtures:
                for batch,noise in zip(batches,noises):
                    losses = ce_loss_sums(model,recipe,batch.to(device),tuple(value.to(device) for value in noise))
                    total += float(losses.sums["ce"])
                    for index,loss in enumerate(losses.pass_losses):passes[index] += float(loss.sums["ce"])
    finally:
        for name,value in flags.items():setattr(model.backbone.backbone,name,value)
    checks = {"fusion_unchanged":before == tree_digests(model.backbone.fusion.state_dict()),
        "frozen_state_unchanged":frozen == frozen_state_pins(model),
        "rng_unchanged":rng == tree_digests(_rng_state(None)),"fixture_unchanged":inputs == fixture_pins(fixtures),
        "modes_unchanged":modes == {name:module.training for name,module in model.named_modules()},
        "flags_restored":flags == {name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS},
        "autocast_cache_restored":cache == torch.is_autocast_cache_enabled(),
        "gradients_absent":all(parameter.grad is None for parameter in model.parameters()),
        "finite_loss":all(torch.isfinite(torch.tensor(value)) for value in (total,*passes))}
    if count <= 0 or not all(checks.values()):raise AssertionError("Common FP32 evaluation violated read-only contract")
    return {"precision":"fp32_math","ce_targets":count,"ce_sum":total,"ce_mean":total/count,
        "pass_ce_means":[value/count for value in passes],"checks":checks}


def load_golden(path,digest,*,configuration,sources):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024*1024 or sha256_file(path) != digest:
        raise ValueError("Golden continuation report differs from immutable pin")
    report = json.loads(path.read_text())
    if (report.get("schema") != KIND or report.get("status") != "completed_segment" or report.get("passed") is not True
            or report.get("precision") != "bf16_mixed" or report.get("configuration") != tree_digests(configuration)
            or report.get("sources") != sources or report.get("starting_update") != PLAN["start_update"]
            or report.get("final_counters",{}).get("optimizer_updates") != PLAN["end_update"]
            or report.get("physical_optimizer_updates") != PLAN["end_update"]-PLAN["start_update"]
            or [row.get("exact",{}).get("metrics",{}).get("update") for row in report.get("updates",[])]
                != list(range(PLAN["start_update"]+1,PLAN["end_update"]+1))
            or [row.get("update") for row in report.get("evaluations",[])] != PLAN["evaluation_updates"]
            or not report.get("integrity") or not all(report["integrity"].values()) or sha256_file(path) != digest):
        raise ValueError("Require complete matching BF16 golden trajectory")
    return report


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision",choices=tuple(PATHS),required=True)
    for name in ("origin-checkpoint","fixture"):
        p.add_argument("--"+name,type=Path,required=True);p.add_argument("--"+name+"-sha256",required=True)
    for name in ("resume","golden-report"):
        p.add_argument("--"+name,type=Path);p.add_argument("--"+name+"-sha256")
    p.add_argument("--artifacts",type=Path,default=ROOT/".runtime/olmo1b-step60000/artifacts")
    p.add_argument("--data-root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--storage-prefix");p.add_argument("--stop-file",type=Path)
    args=p.parse_args(argv)
    for path,digest in ((args.origin_checkpoint,args.origin_checkpoint_sha256),(args.fixture,args.fixture_sha256),
                        (args.resume,args.resume_sha256),(args.golden_report,args.golden_report_sha256)):
        if (path is None) != (digest is None) or (digest is not None and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest))):
            p.error("Paths require independent lowercase SHA256 pins")
    if args.golden_report and (args.precision != "bf16_mixed" or not args.resume):p.error("Golden comparison requires BF16 fresh resume")
    args.output_dir=args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):p.error("Evidence/checkpoints must remain on persistent project storage")
    return args


def main(argv=None):
    args=parse_args(argv);determinism=configure_determinism(True);runtime=require_container_gpu()
    if torch.distributed.is_initialized():raise RuntimeError("Continuation is one GPU/process, no DDP")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True,exist_ok=False);sources=source_hashes()
    for name in sources:
        destination=args.output_dir/"source-snapshot"/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    report={"schema":KIND,"status":"running","passed":False,"precision":args.precision,"sources":sources,
        "runtime":runtime,"determinism":determinism,"plan":copy.deepcopy(PLAN),"updates":[],"evaluations":[],"checkpoints":[],
        "physical_optimizer_updates":0,"started_utc":datetime.now(timezone.utc).isoformat(),
        "qualification":"Bounded fusion-only CE continuation; no quality, full-backbone, RT/auxiliary or production BF16 clearance"}
    tracker=OnlineTracker(project="pretrained-fbt-rt-nextlat",output_dir=args.output_dir,group="olmo-fusion-startup",
        name=args.output_dir.name,preserve_state=preserve_local_rng)
    started,failure=time.monotonic(),None
    requested=[];previous={number:signal.getsignal(number) for number in (signal.SIGINT,signal.SIGTERM)}
    for number in previous:signal.signal(number,lambda signum,frame:requested.append(signum))
    def persist(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-started,wandb=tracker.record);write_json(args.output_dir/"report.json",report)
    try:
        tracker.start({"precision":args.precision,"plan":PLAN,"qualification":report["qualification"]});persist("construct")
        model,recipe,source,_,_=construct(SimpleNamespace(scale="pretrained",length=16,artifacts=args.artifacts),"NF",torch.device("cuda"))
        original={name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS};freeze_for_startup(model)
        origin=_checkpoint_payload(model,args.origin_checkpoint,source,expected_sha256=args.origin_checkpoint_sha256)
        if origin["counters"]["optimizer_updates"] != PLAN["start_update"] or origin["configuration"]["sources"] != training_sources():
            raise ValueError("Require unchanged-source original startup128 authority")
        optimizer,scheduler=build_optimizer(model)
        report["origin_import"]=load_fusion_checkpoint(model,args.origin_checkpoint,source,expected_sha256=args.origin_checkpoint_sha256,
            configuration=origin["configuration"],source_fingerprint=origin["source_fingerprint"],optimizer=optimizer,scheduler=scheduler)
        data=StartupData.from_prepared(args.data_root,expected_manifest_sha256=DEFAULT_MANIFEST_SHA256,
            length=PLAN["length"],ce_per_update=PLAN["ce_targets_per_update"],physical_batch_size=PLAN["physical_batch_size"])
        fixtures,metadata=load_long_fixture(args.fixture,expected_sha256=args.fixture_sha256,recipe=recipe,width=model.config.model_dim)
        if (data.manifest_sha256 != origin["configuration"]["data_manifest_sha256"]
                or metadata["training_manifest_sha256"] != data.manifest_sha256 or data.total_updates < PLAN["end_update"]):
            raise ValueError("Continuation data/held-out authority differs or schedule is insufficient")
        expected_execution = configure_path(model,original,PATHS[args.precision])["runtime_flags"]
        config=configuration_for(model,recipe,origin,origin_sha256=args.origin_checkpoint_sha256,data=data,
            fixture_sha256=args.fixture_sha256,precision=args.precision,sources=sources,runtime=runtime,determinism=determinism,original_flags=original)
        fingerprint={"checkpoint_sha256":source["sha256"],"origin_checkpoint_sha256":args.origin_checkpoint_sha256,
            "code":sources,"data_manifest_sha256":data.manifest_sha256,"fixture_sha256":args.fixture_sha256}
        report.update(configuration=config,source_fingerprint=fingerprint,fixture_provenance=metadata)
        counters=TrainingCounters(**origin["counters"])
        golden=None
        if args.resume:
            restored=load_continuation(args.resume,model,optimizer,scheduler,expected_sha256=args.resume_sha256,
                configuration=config,source_fingerprint=fingerprint,data=data,origin=origin)
            counters=restored["counters"];report["resumed_checkpoint_sha256"]=args.resume_sha256
            if args.golden_report:
                if counters.optimizer_updates != PLAN["checkpoint_updates"][0]:raise ValueError("Golden replay starts at136")
                golden=load_golden(args.golden_report,args.golden_report_sha256,configuration=config,sources=sources)
        if counters.optimizer_updates >= PLAN["end_update"]:raise ValueError("No continuation updates remain")
        report["starting_boundary"]=boundary_digests(model,optimizer,scheduler,counters,data.cursor(counters.optimizer_updates))
        report["starting_update"]=counters.optimizer_updates
        last_save=time.monotonic()
        def checkpoint(reason):
            nonlocal last_save
            number=counters.optimizer_updates
            previous_record=next((item for item in report["checkpoints"] if item["optimizer_updates"] == number),None)
            if previous_record is not None:return previous_record
            record=save_continuation(args.output_dir/f"update-{number:06d}.pt",model,optimizer,scheduler,counters,
                configuration=config,source_fingerprint=fingerprint,data=data,origin=origin)
            record.update(reason=reason,boundary=boundary_digests(model,optimizer,scheduler,counters,data.cursor(number)))
            report["checkpoints"].append(record);last_save=time.monotonic();persist("checkpoint/"+str(number))
            if args.storage_prefix:
                record["storage"]=retain_checkpoint(record["path"],args.storage_prefix,record["sha256"])
                write_json(Path(record["path"]).with_suffix(".receipt.json"),record);persist("checkpoint/retained")
            return record
        def evaluation():
            boundary = boundary_digests(model,optimizer,scheduler,counters,data.cursor(counters.optimizer_updates))
            row={"update":counters.optimizer_updates,**evaluate_fp32(model,recipe,fixtures,original_flags=original)}
            if boundary != boundary_digests(model,optimizer,scheduler,counters,data.cursor(counters.optimizer_updates)):
                raise AssertionError("Common FP32 evaluation changed optimizer/counters/cursor boundary")
            report["evaluations"].append(row);persist("evaluation/observed")
            if golden is not None:
                expected=next(item for item in golden["evaluations"] if item["update"] == counters.optimizer_updates)
                if row != expected:raise AssertionError("Fresh BF16 replay held-out FP32 evaluation differs")
            tracker.log(scalar_metrics(row,"dev_common_fp32"),step=counters.optimizer_updates);persist("evaluation")
        if counters.optimizer_updates in PLAN["evaluation_updates"]:evaluation()
        if not args.resume:checkpoint("origin_boundary_new_kind")
        while counters.optimizer_updates < PLAN["end_update"]:
            if requested or (args.stop_file and args.stop_file.exists()):
                checkpoint("requested_stop");report.update(status="paused_at_boundary",passed=True);break
            index=counters.optimizer_updates;batches,noises=data.update_batches(index,recipe,model.config.model_dim,device="cpu")
            pins=tree_digests({"batches":[vars(batch) for batch in batches],"noise":noises});planned=data.update_metadata(index)
            persist("update/"+str(index+1));begin=time.monotonic()
            metrics=update(model,recipe,batches,noises,optimizer,scheduler,counters,precision=args.precision,
                original_flags=original,ce_targets=PLAN["ce_targets_per_update"])
            report["physical_optimizer_updates"]+=1
            if (any(metrics[key] != planned[key] for key in ("microbatches","documents","input_tokens"))
                    or pins != tree_digests({"batches":[vars(batch) for batch in batches],"noise":noises})):
                raise AssertionError("Continuation data/count/noise contract changed")
            validate_progress(data,counters,data.cursor(counters.optimizer_updates),origin["counters"])
            exact={"metrics":metrics,"input_pins":pins,"data_metadata":planned,
                "boundary":boundary_digests(model,optimizer,scheduler,counters,data.cursor(counters.optimizer_updates))}
            row={"exact":exact,"elapsed_seconds":time.monotonic()-begin,"memory":memory()};report["updates"].append(row)
            persist("update/observed")
            if golden is not None:
                expected=next(item["exact"] for item in golden["updates"] if item["exact"]["metrics"]["update"] == counters.optimizer_updates)
                if exact != expected:raise AssertionError("Fresh BF16 continuation differs bitwise from golden")
            tracker.log({"update":counters.optimizer_updates,**scalar_metrics(metrics,"train")},step=counters.optimizer_updates)
            persist("update/complete")
            if counters.optimizer_updates in PLAN["evaluation_updates"]:evaluation()
            if counters.optimizer_updates in PLAN["checkpoint_updates"] or time.monotonic()-last_save >= PLAN["checkpoint_interval_seconds"]:
                checkpoint("scheduled")
            del batches,noises
        report["integrity"]={"sources_unchanged":sources == source_hashes(),
            "origin_unchanged":sha256_file(args.origin_checkpoint) == args.origin_checkpoint_sha256,
            "fixture_unchanged":sha256_file(args.fixture) == args.fixture_sha256,
            "frozen_state_exact":frozen_state_pins(model) == config["frozen_state_pins"],
            "gradients_absent":all(parameter.grad is None for parameter in model.parameters()),
            "optimizer_ownership":optimizer_ownership(model.backbone.fusion,optimizer) == [["state_proj.weight","token_gate.weight"]],
            "physical_step_count_exact":report["physical_optimizer_updates"] == counters.optimizer_updates-report["starting_update"],
            "execution_flags_restored":all(getattr(model.backbone.backbone,name) == value for name,value in expected_execution.items())}
        if args.resume:report["integrity"]["resume_checkpoint_unchanged"]=sha256_file(args.resume) == args.resume_sha256
        if args.golden_report:report["integrity"]["golden_report_unchanged"]=sha256_file(args.golden_report) == args.golden_report_sha256
        if not all(report["integrity"].values()):raise AssertionError("Continuation final integrity failed")
        report.update(final_counters=asdict(counters),final_cursor=data.cursor(counters.optimizer_updates),
            final_boundary=boundary_digests(model,optimizer,scheduler,counters,data.cursor(counters.optimizer_updates)))
        if report["status"] != "paused_at_boundary":report.update(status="completed_segment",passed=True)
        persist("complete")
    except BaseException as error:
        failure=error;report.update(status="failed",passed=False,error={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()});raise
    finally:
        for number,handler in previous.items():signal.signal(number,handler)
        report["finished_utc"]=datetime.now(timezone.utc).isoformat();persist(report.get("stage","setup"))
        try:tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed",passed=False,tracking_finish_error={"type":type(error).__name__})
            if failure is None:raise
        finally:persist(report.get("stage","setup"))


if __name__ == "__main__":main()
