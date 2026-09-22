#!/usr/bin/env python3
"""Freeze the matched FBT-only pilot after real-data capacity and scale checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.lm_data import load_lm_data
from cdrm.pretrained.lm_schedule import build_schedule
from cdrm.pretrained.lm_training import LMTrainingConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict,validate_prepared_manifest
from scripts.olmo_lm_common import state_digests
from scripts.olmo_o5b_common import (PilotTracker,preserve_rng,source_hashes,build_model,build_optimizer,
    make_mode,observed_step,microbatches,evaluation,online_evaluation,finite_optimizer)
from scripts.olmo_validation import require_container_gpu


def profile(model,batch,physical,config):
    before=state_digests(model)
    optimizer=build_optimizer(model,config,zero_lr=True)
    training=LMTrainingConfig(precision=config["precision"],max_grad_norm=1)
    samples=microbatches(batch,physical)
    def step():
        return observed_step(model,optimizer,samples,config=training,backbone_kwargs={"mode":make_mode(1)})
    for _ in range(2): step()
    torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    seconds=[]
    for _ in range(3):
        began=time.monotonic();last=step();torch.cuda.synchronize();seconds.append(time.monotonic()-began)
    peak=torch.cuda.max_memory_allocated()/2**30
    unchanged=state_digests(model)==before
    finite=finite_optimizer(model,optimizer)
    row={"physical_batch_size":physical,"effective_batch_size":batch.input_ids.shape[0],
         "length":batch.input_ids.shape[1],"warmup_steps":2,"timed_steps":3,"wall_seconds":seconds,
         "median_wall_seconds":statistics.median(seconds),"peak_allocated_gib":peak,
         "input_tokens_per_second":int(batch.valid_mask.sum())/statistics.median(seconds),
         "weights_unchanged":unchanged,"optimizer_finite":finite,"last_step":last,
         "passed":unchanged and finite}
    del optimizer
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    runtime=require_container_gpu()
    torch.set_num_threads(8);torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={"schema":"olmo-o5b-preflight-v1","status":"running","runtime":runtime,
            "started_utc":datetime.now(timezone.utc).isoformat(),"source_hashes":source_hashes(),"profile":[]}
    tracker=PilotTracker(project="pretrained-fbt-rt-nextlat",output_dir=args.output_dir,
                        name="olmo-1b-o5b-preflight",group="olmo1b-o5b-code-pilot",preserve_state=preserve_rng)
    try:
        native=validate_prepared_manifest(args.artifacts);corpus=load_lm_data(args.data)
        report.update(checkpoint=native["checkpoint"],data_manifest_sha256=corpus.manifest_sha256)
        tracker.start({"scope":"FBT K2 real-data zero-LR complete steps and baseline evaluation",
                       "checkpoint_sha256":native["checkpoint"]["sha256"],"data_manifest_sha256":corpus.manifest_sha256})
        report["wandb"]=tracker.record
        state=load_native_state_dict(args.artifacts);model=build_model(state);del state
        schedule=build_schedule(corpus.train_lengths,batch_size=32,warmup_updates=100,
                                ramp_min_tokens=10_000_000,ramp_min_updates=200)
        config={"schema":"olmo-o5b-pilot-config-v1","source_hashes":report["source_hashes"],
                "data_manifest_sha256":corpus.manifest_sha256,"checkpoint_sha256":native["checkpoint"]["sha256"],
                "seed":20260922,"model_config":model.backbone.config.to_dict(),"fusion_config":model.backbone.fusion_config.to_dict(),
                "fusion_output_scale":float(model.backbone.fusion.output_scale),"nextlat_config":model.config.to_dict(),
                "nextlat_enabled":False,"num_passes":2,"gamma":1.0,"rt_layers":[],
                "precision":"bf16_mixed","attention_backend":"sdpa","attention_precision":"mixed",
                "lr":1e-5,"fusion_lr":1e-4,"fusion_warmup_updates":100,
                "fusion_lr_rule":"first_nonzero_beta_update_starts_100_active_update_linear_warmup",
                "betas":[.9,.95],"eps":1e-8,"weight_decay":.1,"max_grad_norm":1.,
                "schedule":schedule,"schedule_control":"beta_uses_existing_alpha_schedule_fields",
                "eval_every_tokens":1_000_000,"eval_rows":128,"final_eval_rows":512,"eval_batch_size":8,
                "online_eval_rows":32,"online_eval_length":64,"health_gate_update":50,
                "catastrophic_nll_increase":1.5,"catastrophic_consecutive_evals":2,"checkpoint_interval_seconds":1800,
                "prefix_mixin":False,"hidden_jitter":0.0,"compile":False,"cuda_graphs":False,"distributed":False,
                "masking":"all valid same-document targets; independent windows; starts plain; no cross-window cache"}
        indices=np.flatnonzero(corpus.train_lengths==512)[:32]
        if len(indices)!=32:raise ValueError("Need 32 full-length real-data training windows")
        batch=corpus.batch("train",indices,device="cuda")
        chosen=None
        # Prefer unchanged effective/physical B32, fall back only to accumulation.
        for physical in (32,16):
            try:
                row=profile(model,batch,physical,config);report["profile"].append(row)
                tracker.log({"profile/physical_batch":physical,"profile/seconds":row["median_wall_seconds"],
                             "profile/peak_allocated_gib":row["peak_allocated_gib"]})
                if not row["passed"]:raise AssertionError("Zero-LR preflight integrity failed")
                if row["peak_allocated_gib"]<60:chosen=physical
            except torch.cuda.OutOfMemoryError:
                report["profile"].append({"physical_batch_size":physical,"status":"out_of_memory"})
            finally:
                model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
            write_json(args.output_dir/"report.json",report)
            if chosen is not None:break
        if chosen is None:raise RuntimeError("No candidate meets the fixed60GiB complete-step memory ceiling")
        config["physical_batch_size"]=chosen
        del batch
        report["baseline"]=evaluation(model,corpus,config,make_mode(0),512,documents=True)
        report["cold_feedback"]=evaluation(model,corpus,config,make_mode(1),128)
        report["initial_online"]=online_evaluation(model,corpus,config,0)
        for section in ("baseline","cold_feedback"):
            for split,value in report[section].items():
                for p,item in enumerate(value["passes"]):
                    tracker.log({f"{section}/{split}/pass_{p}/nll":item["mean_nll"]})
                print({"section":section,"split":split,"nll":[p["mean_nll"] for p in value["passes"]]},flush=True)
        if report["source_hashes"]!=source_hashes():raise AssertionError("Preflight sources changed during execution")
        write_json(args.output_dir/"configuration.json",config)
        report.update(status="passed",selected_batch_size=32,physical_batch_size=chosen,schedule=schedule)
        tracker.summary({"preflight/status":"passed","preflight/physical_batch":chosen,
                         "preflight/total_updates":schedule["total_updates"]})
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__);raise
    finally:
        try:tracker.finish(succeeded=report["status"]=="passed")
        finally:
            report.update(wandb=tracker.record,finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json",report)
    print({"status":"passed","physical_batch":chosen,"wandb":tracker.record["run_url"]},flush=True)


if __name__=="__main__":main()
