#!/usr/bin/env python3
"""Bounded device profile of the validated F3 path before native RT changes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cdrm.pretrained import olmo_tiled
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import TrainingCounters
from scripts.olmo_f3_graph_training import (
    SOURCE_FILES as F3_SOURCES, configure_determinism, require_container_gpu,
    validate_prepared_manifest, load_native_state_dict, load_native_tokenizer,
    case_for, new_plan, build_optimizer, changed_batch, backend_context,
    state_health, timed, OnlineTracker,
)
from scripts.olmo_f3b_validate import new_plan as variant_plan

SOURCES = tuple(sorted(set(F3_SOURCES) | {"scripts/olmo_f3b_profile.py",
    "scripts/olmo_f3b_validate.py", "cdrm/pretrained/olmo_rt_kernels.py"}))


@contextmanager
def phase_annotations():
    """Observer-only labels; wrappers add no tensor operations."""
    originals = {}
    for name in ("_project", "_finish", "_add_tile", "_attention_from_completed"):
        original = getattr(olmo_tiled, name)
        originals[name] = original
        def decorate(function, label):
            @wraps(function)
            def annotated(*args, **kwargs):
                with torch.profiler.record_function("RT/"+label):
                    return function(*args, **kwargs)
            return annotated
        setattr(olmo_tiled, name, decorate(original, name))
    try:
        yield
    finally:
        for name, function in originals.items():
            setattr(olmo_tiled, name, function)


def profile_rows(profile):
    rows = []
    for event in profile.key_averages():
        rows.append({"name": event.key, "count": event.count,
            "cpu_total_us": event.cpu_time_total,
            "self_cpu_us": event.self_cpu_time_total,
            "device_total_us": event.device_time_total,
            "self_device_us": event.self_device_time_total,
            "device_type": str(event.device_type)})
    return sorted(rows, key=lambda row: row["self_device_us"], reverse=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("rt", "combined"), required=True)
    parser.add_argument("--variant", choices=("reference", "cast_once", "triton"), default="reference")
    parser.add_argument("--batch-size", type=int, choices=(32,64,128), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    args.checkpointing = "on"
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4); torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        path = args.output_dir/"source-snapshot"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, path)
    report = {"schema": "olmo-f3b-profile-v1", "status": "running",
        "runtime": {k:str(v) for k,v in runtime.items()},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"case":args.case,"variant":args.variant,"batch_size":args.batch_size,"length":512,
            "rt_layers":[0],"checkpointing":True,"precision":"bf16_mixed",
            "ordinary_backend":"flash","autocast_weight_cache":False,**determinism},
        "source_hashes": {p:sha256_file(ROOT/p) for p in SOURCES}}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f3b-rt-kernel", name="olmo-1b-"+args.output_dir.name)
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration":report["configuration"],"checkpoint":report["checkpoint"]})
        print({"wandb":tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        case = case_for(args.case,batch=args.batch_size,length=512)
        with backend_context("flash"):
            from scripts.olmo_f1_common import build_model
            model = build_model(state,case)
            plan = variant_plan(model,changed_batch(tokenizer,case,0),case.mode(),args.variant)
            optimizer,scheduler = build_optimizer(model)
            counters = TrainingCounters()
            for step in range(3):
                plan.optimizer_step(optimizer,changed_batch(tokenizer,case,step),
                    scheduler=scheduler,counters=counters)
            plan.capture(warmup=10)
            records=[]
            # Pre-create CPU fixture batches so fixture construction is not timed.
            batches=[changed_batch(tokenizer,case,3+i) for i in range(3)]
            def complete_timed():
                records.append(plan.optimizer_step(optimizer,batches[len(records)],
                    replay=True,scheduler=scheduler,counters=counters))
            report["full_step"] = timed(complete_timed,3)
            report["input_tokens_per_second"] = plan.input_tokens/report["full_step"]["median_wall_seconds"]
            report["records"] = records
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            for replay in (False, True):
                activity=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
                with torch.profiler.profile(activities=activity) as profile:
                    if replay:
                        plan.backward(replay=True)
                    else:
                        with phase_annotations():
                            plan.backward(replay=False)
                    torch.cuda.synchronize()
                report["graph_profile" if replay else "annotated_eager_profile"] = profile_rows(profile)
                del profile
            report["health"] = state_health(model,optimizer)
            assert report["health"]["passed"]
        assert report["source_hashes"] == {p:sha256_file(ROOT/p) for p in SOURCES}
        tracker.log({"profile/input_tokens_per_second":report["input_tokens_per_second"],
            "profile/peak_allocated_gib":report["peak_allocated_gib"]},step=6)
        report["status"] = "passed"
        print({"status":"passed","tokens_per_second":report["input_tokens_per_second"],
               "phases":[r for r in report["annotated_eager_profile"] if r["name"].startswith("RT/")]},flush=True)
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error_message=str(error))
        raise
    finally:
        tracker.finish(succeeded=report["status"]=="passed")
        report.update(wandb=tracker.record,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir/"report.json",report)


if __name__ == "__main__":
    main()
