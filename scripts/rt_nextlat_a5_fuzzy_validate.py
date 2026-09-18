#!/usr/bin/env python3
"""Bounded changed-objective checks; existing D128 FP32 RT kernels are retained."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from cdrm.mad_data import FUZZY_TASK, load_dataset
from cdrm.rt_nextlat_tasks import build_model, encode_inputs, task_logits, task_loss
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import A5Metrics, configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_data import load_split
from scripts.rt_a5_train import atomic_json, batch_tensors, preserve_rng, file_sha256, json_value
from scripts.rt_nextlat_a5_fuzzy_train import train_step
from scripts.stage_a_common import require_cuda_container
from rt_a5_validate import compare_tensors, finite_state


def gradient_check(config):
    generator = torch.Generator().manual_seed(2026091622)
    batches = {}
    for task, vocab, length in (("a5", 60, 12), ("fuzzy", 16, 17)):
        x = torch.randint(vocab, (2, length), generator=generator).cuda()
        y = torch.randint(vocab, (2, length), generator=generator).cuda()
        batches[task] = (encode_inputs(x, task), y)
    packets = []
    for reference in (True, False):
        model = build_model(config, device="cuda")
        if reference:
            with fp32_context("cuda"):
                losses = [task_loss(model, x, y, task=task)["loss"]
                          for task, (x, y) in batches.items()]
                objective = .5 * (losses[0] + losses[1])
                objective.backward()
            scalar = objective.item()
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            result = train_step(model, optimizer, batches, mode="mixed", microbatch=2, clip_norm=1e9)
            scalar = result["loss"]
        packet = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
        state = finite_state(model, require_gradients=True)
        if not state["passed"]:
            raise AssertionError("Invalid FP32 gradient state in mixed objective check")
        packets.append((scalar, packet))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    a, b = packets
    if a[1].keys() != b[1].keys():
        raise AssertionError("Mixed objective gradient coverage differs")
    comparisons = {name: compare_tensors(value, b[1][name]) for name, value in a[1].items()}
    loss = compare_tensors(torch.tensor(a[0]), torch.tensor(b[0]))
    return {"passed": loss["passed"] and all(v["passed"] for v in comparisons.values()),
            "loss": loss, "parameters": comparisons,
            "scope": "Direct half-weighted sum vs sequential task backward; two examples/task T12/T17, same-state tiled FP32; inherited tolerance"}


def actual_shapes(args, tracker):
    a5_x, a5_y = load_split(args.a5_data, "train")
    fuzzy = load_dataset(args.fuzzy_data, FUZZY_TASK, "train")
    output = {}
    for mode in ("a5-only", "mixed"):
        model = build_model(args.config, device="cuda")
        original = json.loads(Path(args.fuzzy_control_report).read_text())["initialization"]
        if json_value(model.initialization) != original:
            raise AssertionError("New arm is not paired with original Fuzzy initialization")
        optimizer = make_optimizer(model)
        rows = []
        torch.cuda.reset_peak_memory_stats()
        for index in range(15):
            selection = np.arange(index * 128, (index + 1) * 128)
            x, y = batch_tensors(a5_x, a5_y, selection, "cuda")
            batches = {"a5": (encode_inputs(x, "a5"), y)}
            if mode == "mixed":
                x, y = batch_tensors(fuzzy.input_ids, fuzzy.labels, selection, "cuda")
                batches["fuzzy"] = (encode_inputs(x, "fuzzy"), y)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = train_step(model, optimizer, batches, mode=mode, microbatch=128)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            rows.append({"update": index + 1, "seconds": seconds, "loss": result["loss"]})
            tracker.log({"update": (15 if mode == "mixed" else 0) + index + 1,
                         f"benchmark/{mode}/seconds": seconds, f"train/{mode}/loss": result["loss"]})
        state = finite_state(model, optimizer, require_gradients=True)
        if not state["passed"]:
            raise AssertionError("Actual-shape model/gradient/Adam state is invalid")
        output[mode] = {"passed": True, "batch_per_task": 128, "rows": rows,
                        "measured_mean_seconds": statistics.mean(row["seconds"] for row in rows[10:]),
                        "warmup_updates": 10, "measured_updates": 5,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "finite_state": state,
                        "original_fuzzy_initialization_exact": True}
        if mode == "a5-only":
            x36, y36 = load_split(args.a5_data, "ood_dev")
            x, y = batch_tensors(x36, y36, np.arange(1024), "cuda")
            metric = A5Metrics()
            with torch.no_grad(), fp32_context("cuda"):
                logits = task_logits(model, encode_inputs(x, "a5"), "a5")
                metric.update(logits, y)
            output[mode]["actual_evaluation_shape"] = {"batch": 1024, "length": 36,
                                                         "finite": bool(torch.isfinite(logits).all()),
                                                         "whole_word_correct": int(metric.prefix_correct[-1])}
            if not output[mode]["actual_evaluation_shape"]["finite"]:
                raise AssertionError("Nonfinite actual A5 evaluation logits")
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rt_nextlat_tasks/fuzzy_d128.json")
    parser.add_argument("--a5-data", required=True)
    parser.add_argument("--fuzzy-data", required=True)
    parser.add_argument("--fuzzy-control-report", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "rt-nextlat-a5-fuzzy-validation-v1", "status": "running",
              "hardware": require_cuda_container(), "runtime": configure_fp32_runtime(),
              "scope": "Discarded integration fixtures; no new precision/backend study or confirmation evaluation"}
    tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", output_dir=args.output,
                            group=args.output.parent.name, name="d128-a5-mixed-preflight", preserve_state=preserve_rng)
    try:
        tracker.start({"scope": report["scope"], "width": 128, "heads": 16, **report["runtime"]})
        report["wandb"] = tracker.record
        atomic_json(args.output / "report.json", report)
        report["gradient_check"] = gradient_check(args.config)
        atomic_json(args.output / "report.json", report)
        if not report["gradient_check"]["passed"]:
            raise AssertionError("Mixed objective differs from its direct mathematical sum")
        report["actual_shapes"] = actual_shapes(args, tracker)
        report.update(status="passed", validation_source_sha256=file_sha256(__file__))
        tracker.summary({"passed": True})
        tracker.finish(succeeded=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        tracker.finish(succeeded=False)
        raise
    finally:
        report["wandb"] = tracker.record
        atomic_json(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "timings": {k:v["measured_mean_seconds"]
                     for k,v in report["actual_shapes"].items()}, "wandb": tracker.record["run_url"]}), flush=True)


if __name__ == "__main__":
    main()
