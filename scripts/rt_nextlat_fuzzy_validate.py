#!/usr/bin/env python3
"""Bounded FP32 integration and real-shape profile for the width-128 pilot."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cdrm.mad_data import FUZZY_TASK, load_dataset
from cdrm.rt_nextlat_tasks import build_model, task_loss, task_logits
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_train import atomic_json, preserve_rng, file_sha256
from scripts.rt_nextlat_fuzzy_train import train_step
from scripts.stage_a_common import require_cuda_container
from rt_a5_validate import compare_tensors, finite_state


def gradient_packet(model):
    return {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
            if p.grad is not None}


def small_check(config):
    generator = torch.Generator(device="cpu").manual_seed(2026091610)
    x = torch.randint(0, 16, (2, 17), generator=generator).cuda() + 60
    y = torch.randint(0, 16, (2, 17), generator=generator).cuda()
    packets, checks = {}, {}
    for backend in ("naive", "tiled"):
        model = build_model(config, device="cuda", backend=backend)
        with fp32_context("cuda"):
            result = task_loss(model, x, y)
            result["loss"].backward()
        state = finite_state(model, require_gradients=True)
        packets[backend] = {key: result[key].detach().cpu() for key in ("loss", "ce", "latent", "logits")}
        packets[backend]["gradients"] = gradient_packet(model)
        altered = x.clone()
        altered[:, 8:] = 60 + ((altered[:, 8:] - 60 + 1) % 16)
        with torch.no_grad(), fp32_context("cuda"):
            original = task_logits(model, x)
            changed = task_logits(model, altered)
            # Unrelated forward calls must not leave persistent recurrent state.
            repeated = task_logits(model, x)
        checks[backend] = {"finite": state,
                           "causality": compare_tensors(original[:, :8], changed[:, :8]),
                           "independent_calls": compare_tensors(original, repeated)}
        del model, result
        gc.collect()
        torch.cuda.empty_cache()
    a, b = packets["naive"], packets["tiled"]
    if a["gradients"].keys() != b["gradients"].keys():
        raise AssertionError("Reference and tiled parameter gradient coverage differ")
    compared = {key: compare_tensors(a[key], b[key]) for key in ("loss", "ce", "latent", "logits")}
    compared["gradients"] = {key: compare_tensors(value, b["gradients"][key])
                              for key, value in a["gradients"].items()}
    passed = all(item["passed"] for name, item in compared.items() if name != "gradients")
    passed &= all(item["passed"] for item in compared["gradients"].values())
    passed &= all(item["passed"] for backend in checks.values() for item in backend.values())
    return {"passed": passed, "batch": 2, "length": 17, "width": 128, "heads": 16,
            "scope": "Combined CE+NextLat; inherited FP32 atol2e-6 rtol2e-5",
            "comparison": compared, "backends": checks}


def profile(config, data, tracker, *, batch, microbatch, warmup, measured):
    dataset = load_dataset(data, FUZZY_TASK, "train")
    model = build_model(config, device="cuda")
    optimizer = make_optimizer(model)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(warmup + measured):
        indices = np.arange(index * batch, (index + 1) * batch) % len(dataset)
        x = torch.as_tensor(np.array(dataset.input_ids[indices], dtype=np.int64), device="cuda") + 60
        y = torch.as_tensor(np.array(dataset.labels[indices], dtype=np.int64), device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        values = train_step(model, optimizer, x, y, microbatch=microbatch)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        row = {"update": index + 1, "warmup": index < warmup, "seconds": seconds, **values}
        rows.append(row)
        tracker.log({"update": index + 1, "benchmark/update_seconds": seconds,
                     **{f"train/{key}": value for key, value in values.items()}})
        print(json.dumps({"event": "profile_update", **row}), flush=True)
    state = finite_state(model, optimizer, require_gradients=True)
    times = [r["seconds"] for r in rows if not r["warmup"]]
    out = {"passed": state["passed"], "batch": batch, "microbatch": microbatch,
           "length": int(dataset.input_ids.shape[1]), "discarded_updates": len(rows), "rows": rows,
           "finite_state": state, "median_update_seconds": statistics.median(times),
           "mean_update_seconds": statistics.mean(times),
           "estimated_10000_update_training_hours": statistics.mean(times) * 10000 / 3600,
           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
           "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
           "scope": "Synchronized FP32 loss/backward/clip/Adam; training estimate excludes evaluation and checkpoint overhead"}
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rt_nextlat_tasks/fuzzy_d128.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--measured", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "rt-nextlat-fuzzy-validation-v1", "status": "running",
              "hardware": require_cuda_container(), "runtime": configure_fp32_runtime(),
              "scope": "Discarded fixtures and updates, no confirmation evaluation"}
    tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", output_dir=args.output,
                            group=args.output.parent.name, name="d128-t400-preflight", preserve_state=preserve_rng)
    try:
        tracker.start({"model": "L1R-RT2+NextLat", "width": 128, "heads": 16, "length": 400,
                       "scope": report["scope"], **report["runtime"]})
        report["wandb"] = tracker.record
        atomic_json(args.output / "report.json", report)
        print(json.dumps({"event": "started", "wandb": tracker.record["run_url"]}), flush=True)
        report["small"] = small_check(args.config)
        atomic_json(args.output / "report.json", report)
        if not report["small"]["passed"]:
            raise AssertionError("Bounded same-state FP32 integration check failed")
        report["profile"] = profile(args.config, args.data, tracker, batch=args.batch_size,
                                    microbatch=args.microbatch, warmup=args.warmup, measured=args.measured)
        if not report["profile"]["passed"]:
            raise AssertionError("Actual-shape optimizer state failed finite FP32 checks")
        report["validation_script_sha256"] = file_sha256(__file__)
        shutil.copy2(__file__, args.output / Path(__file__).name)
        report["status"] = "passed"
        tracker.summary({"passed": True, "median_update_seconds": report["profile"]["median_update_seconds"]})
        tracker.finish(succeeded=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        tracker.finish(succeeded=False)
        raise
    finally:
        report["wandb"] = tracker.record
        atomic_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "profile": {key: value for key, value in report["profile"].items()
                                                               if key not in ("rows", "finite_state")}}), flush=True)


if __name__ == "__main__":
    main()
