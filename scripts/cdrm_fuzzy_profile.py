#!/usr/bin/env python3
"""Bounded same-runner H100 fit/timing measurement; no research checkpoint."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import cdrm_fuzzy_common as common
from cdrm.mad_data import load_dataset
from experiment_tracking import OnlineTracker, add_wandb_arguments
from stage_b_train import atomic_json, compiler_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("cdrm", "seq"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--batch", type=int, choices=(32,64,128), required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project="cdrm-150m-fuzzy-recall",wandb_group="20260908T031418Z")
    args = parser.parse_args()
    if not 1 <= args.warmup < args.updates <= 20:
        parser.error("Bounded profiling requires 1<=warmup<updates<=20")
    if args.output_dir.exists():
        raise FileExistsError("Use a new profile directory")
    if not args.wandb_project:
        parser.error("Online W&B is required")
    initial = common.load_initial(args.checkpoint)
    data = {split:load_dataset(args.data_root,common.TASK,split,verify=True) for split in ("train","dev")}
    for split,dataset in data.items():
        common.validate_data(dataset,split,length=args.length)
        if len(dataset) < args.batch or len(dataset) % args.batch:
            raise ValueError("Numerical profile fixture must contain complete batches")
    args.output_dir.mkdir(parents=True)
    tracked = common.sources(("scripts/cdrm_fuzzy_profile.py",))
    common.snapshot_sources(args.output_dir,tracked)
    report = {"schema":"cdrm-fuzzy-profile-v1","status":"running","arm":args.arm,
              "scope":"Short operational fit/timing; repeated numerical batches, discarded updates, no task-performance claim",
              "checkpoint":common.reference(args.checkpoint),"source_sha256":tracked,
              "shape":[args.batch,args.length],"precision":args.precision,"warmup_updates":args.warmup,
              "data":{split:common.dataset_identity(dataset) for split,dataset in data.items()},"updates":[]}
    tracker = None;started = time.monotonic()
    try:
        report["runtime"] = common.setup(initial["initialization"]["training_seed"])
        tracker = OnlineTracker(project=args.wandb_project,entity=args.wandb_entity,group=args.wandb_group,
            name=args.wandb_run_name,output_dir=args.output_dir,preserve_state=common.preserve_tracking_rng)
        report["wandb"] = tracker.record
        tracker.start({"evidence":"OPS-profile","arm":args.arm,"batch":args.batch,"length":args.length,
                       "precision":args.precision,"checkpoint_sha256":report["checkpoint"]["sha256"]})
        model,report["construction"] = common.build_model(args.arm,args.precision,"tiled",checkpoint=initial,length=args.length)
        optimizer = common.optimizer_for(model,5e-4)
        for update in range(args.updates):
            begin = (update*args.batch)%len(data["train"])
            batch = data["train"].take(slice(begin,begin+args.batch))
            if update == args.warmup:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            row = common.update(model,optimizer,torch.as_tensor(batch.input_ids,device="cuda"),
                torch.as_tensor(batch.labels,device="cuda"),torch.as_tensor(batch.answer_labels,device="cuda"),
                precision=args.precision,monitor=False)
            row.update(update=update+1,warmup=update<args.warmup,batch_sha256=batch.sha256)
            report["updates"].append(row)
            tracker.log({"update":update+1,"benchmark/update_seconds":row["seconds"],
                         "benchmark/tokens_per_second":args.batch*args.length/row["seconds"],
                         "benchmark/peak_allocated_bytes":torch.cuda.max_memory_allocated()})
            atomic_json(args.output_dir/"progress.json",report,replace=True)
            print(json.dumps({"arm":args.arm,"batch":args.batch,"update":update+1,"seconds":row["seconds"],
                              "peak_allocated_GiB":torch.cuda.max_memory_allocated()/2**30}),flush=True)
        report["training_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["training_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["final_precision"] = common.effective_precision(model,optimizer,require_gradients=True)
        report["development"] = common.evaluate(model,data["dev"],args.batch,precision=args.precision)
        report["compiler"] = compiler_audit(True,require_graphs=args.arm=="cdrm")
        times = [row["seconds"] for row in report["updates"] if not row["warmup"]]
        report["timing"] = {"mean_seconds":statistics.mean(times),"median_seconds":statistics.median(times),
            "min_seconds":min(times),"max_seconds":max(times),"steady_updates":len(times),
            "tokens_per_second":args.batch*args.length/statistics.mean(times),
            "includes":"The actual training update helper, including its gradient/dtype/finite checks and metrics; excludes W&B/save overhead"}
        tracker.summary({"profile/median_seconds":report["timing"]["median_seconds"],
            "profile/training_peak_allocated_bytes":report["training_peak_allocated_bytes"],
            "profile/training_peak_reserved_bytes":report["training_peak_reserved_bytes"]})
        common.verify_sources(tracked)
        common.verify_sources(initial["identity"]["source_sha256"])
        if common.reference(args.checkpoint)!=report["checkpoint"]:
            raise AssertionError("Initial checkpoint changed")
        for split,dataset in data.items():
            if load_dataset(args.data_root,common.TASK,split,verify=True).sha256 != dataset.sha256:
                raise AssertionError("Profile dataset changed")
        report["status"] = "profile_complete"
    except BaseException as error:
        report.update(status="execution_failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        try:
            if tracker:tracker.finish(succeeded=report["status"]=="profile_complete")
        except BaseException as error:
            report.update(status="execution_failed",final_sync_error_type=type(error).__name__)
            raise
        finally:atomic_json(args.output_dir/"report.json",report)


if __name__ == "__main__":
    main()
