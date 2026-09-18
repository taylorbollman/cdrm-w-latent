#!/usr/bin/env python3
"""Bounded mixed A5/Fuzzy throughput trials; all resulting weights are discarded.

Every candidate starts from the same D128 initialization and the beginning of
the same two task streams. A candidate's physical microbatch equals its batch
per task. This measures execution, not learning quality or numerical parity.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from cdrm.mad_data import FUZZY_TASK, IGNORE_INDEX, load_dataset
from cdrm.rt_nextlat_tasks import CONFIG_PATH, build_model, encode_inputs, read_configuration
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.rt_a5_common import configure_fp32_runtime, make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_train import WordOrder, atomic_json, batch_tensors, file_sha256, json_sha256, json_value, preserve_rng
from scripts import rt_nextlat_a5_fuzzy_train as trainer
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-a5-fuzzy-batch-profile-v1"
TASK_LENGTHS = {"a5": 12, "fuzzy": 400}
ORDER_SEEDS = {"a5": 5432, "fuzzy": 2026091604}


def validate_arguments(args):
    if (not args.batches or any(type(value) is not int or value < 1 for value in args.batches)
            or len(set(args.batches)) != len(args.batches)):
        raise ValueError("Candidate batches must be distinct positive integers")
    for name in ("warmup_updates", "timed_updates"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name} must be a positive integer")


def summarize_durations(durations, *, batch_per_task, lengths=None):
    """Aggregate work/elapsed throughput, never mean the reciprocal timings."""
    if type(batch_per_task) is not int or batch_per_task < 1:
        raise ValueError("batch_per_task must be a positive integer")
    lengths = TASK_LENGTHS if lengths is None else lengths
    if (not lengths or any(not isinstance(name, str) or type(length) is not int or length < 1
                           for name, length in lengths.items())):
        raise ValueError("Task lengths must be positive integers")
    values = np.asarray(durations, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Timing samples must be a nonempty vector of finite positive seconds")
    elapsed = float(values.sum())
    mean = float(values.mean())
    examples = len(values) * batch_per_task
    return {"timed_updates": len(values), "timed_seconds": elapsed,
            "mean_seconds": mean, "median_seconds": float(np.median(values)),
            "p10_seconds": float(np.quantile(values, 0.1, method="linear")),
            "p90_seconds": float(np.quantile(values, 0.9, method="linear")),
            "total_examples_per_second": examples * len(lengths) / elapsed,
            "total_tokens_per_second": examples * sum(lengths.values()) / elapsed,
            "per_task": {task: {"examples": examples, "tokens": examples * length,
                                "examples_per_second": examples / elapsed,
                                "tokens_per_second": examples * length / elapsed}
                         for task, length in lengths.items()},
            "quantile_method": "linear interpolation (NumPy method=linear)",
            "throughput_reduction": "sum of measured work / sum of measured elapsed seconds"}


def source_manifest():
    sources = dict(trainer.source_manifest())
    relative = "scripts/rt_nextlat_a5_fuzzy_batch_profile.py"
    sources[relative] = file_sha256(ROOT / relative)
    return dict(sorted(sources.items()))


def _profile_candidate(args, batch, *, config, arrays, expected_initialization,
                       tracker, trial_index, history, progress):
    """Catch only CUDA allocation failures; release this trial before returning."""
    record = {"batch_per_task": batch, "total_batch_size": batch * 2,
              "microbatch": batch, "requested_warmup_updates": args.warmup_updates,
              "requested_timed_updates": args.timed_updates, "completed_updates": 0,
              "completed_warmup_updates": 0, "completed_timed_updates": 0,
              "status": "running", "phase": "initializing", "history": [],
              "weights_discarded": True, "checkpoint_written": False}
    model = optimizer = batches = result = None
    completed = 0
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        model = build_model(config, seed=1234, predictor_seed=1235, fuzzy_seed=1236, device="cuda")
        record["initialization"] = json_value(model.initialization)
        if expected_initialization is not None and record["initialization"] != expected_initialization:
            raise AssertionError("Candidate initialization differs from previous trial")
        optimizer = make_optimizer(model, lr=1e-4)
        orders = {task: WordOrder(len(arrays[task][0]), ORDER_SEEDS[task]) for task in TASK_LENGTHS}
        chains = {task: trainer.ORDER_INITIAL[task] for task in TASK_LENGTHS}
        prefix_hashes = {task: hashlib.sha256() for task in TASK_LENGTHS}
        # Hash the first 128 real row IDs independently of candidate boundaries.
        record["first_128_ordered_rows_sha256"] = {
            task: hashlib.sha256(orders[task].indices(0, 128).astype("<i8").tobytes()).hexdigest()
            for task in TASK_LENGTHS}
        record["phase"] = "warmup"
        torch.cuda.reset_peak_memory_stats()
        progress(record)
        for index in range(args.warmup_updates + args.timed_updates):
            measured = index >= args.warmup_updates
            if index == args.warmup_updates:
                torch.cuda.synchronize()
                record["warmup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                record["warmup_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                record["phase"] = "timed"
                progress(record)
            # Remove the last update's input tensors before allocating a new
            # batch, so trials do not retain an accidental extra logical batch.
            batches = None
            torch.cuda.synchronize()
            started = time.perf_counter()
            batches, next_chains = {}, {}
            for task in TASK_LENGTHS:
                indices = orders[task].indices(index * batch, batch)
                serialized = indices.astype("<i8").tobytes()
                next_chains[task] = hashlib.sha256(bytes.fromhex(chains[task]) + serialized).hexdigest()
                prefix_hashes[task].update(serialized)
                x, y = batch_tensors(*arrays[task], indices, "cuda")
                batches[task] = (encode_inputs(x, task=task), y)
                del x, y
            result = trainer.train_step(model, optimizer, batches, mode="mixed", microbatch=batch)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Invalid synchronized elapsed time")
            completed = index + 1
            chains = next_chains
            phase_index = completed - args.warmup_updates if measured else completed
            row = {"candidate_batch_per_task": batch, "phase": "timed" if measured else "warmup",
                   "completed_update": completed, "phase_update": phase_index, "seconds": seconds,
                   "examples_per_task": batch, "examples_total": batch * 2,
                   "tokens_per_task": {task: batch * length for task, length in TASK_LENGTHS.items()},
                   "examples_seen_per_task": completed * batch, "loss": result["loss"],
                   "grad_norm": result["grad_norm"], "tasks": result["tasks"],
                   "order_chains": dict(chains)}
            record["history"].append(row)
            record.update(completed_updates=completed,
                          completed_warmup_updates=min(completed, args.warmup_updates),
                          completed_timed_updates=max(0, completed - args.warmup_updates))
            history.write(json.dumps(row, allow_nan=False) + "\n")
            # Tracker work is outside the synchronization/timing region.
            tracking_update = trial_index * (args.warmup_updates + args.timed_updates) + completed
            tracker.log({"update": tracking_update, "benchmark/batch_per_task": batch,
                         "benchmark/is_timed": measured, "benchmark/completed_update": completed,
                         "benchmark/phase_update": phase_index,
                         f"benchmark/batch_{batch}/{row['phase']}_seconds": seconds,
                         f"benchmark/batch_{batch}/loss": result["loss"]})
        record["finite_state"] = trainer.fuzzy_base.finite_state(model, optimizer, completed)
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        record["peak_allocated_gib"] = record["peak_allocated_bytes"] / 2**30
        record["peak_reserved_gib"] = record["peak_reserved_bytes"] / 2**30
        record["order_chains"] = chains
        record["ordered_rows_sha256"] = {task: value.hexdigest() for task, value in prefix_hashes.items()}
        record["examples_seen_per_task"] = completed * batch
        record["summary"] = summarize_durations(
            [row["seconds"] for row in record["history"] if row["phase"] == "timed"], batch_per_task=batch)
        record.update(status="complete", phase="complete")
    except torch.cuda.OutOfMemoryError:
        record.update(status="oom", phase="failed", failure_phase=record["phase"],
                      error_type="OutOfMemoryError", failed_update=completed + 1,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      finite_state={"checked": False, "reason": "candidate allocation failure"})
    except BaseException as error:
        record.update(status="failed", phase="failed", error_type=type(error).__name__,
                      failure_phase=record["phase"], failed_update=completed + 1)
        progress(record)
        raise
    finally:
        model = optimizer = batches = result = None
        gc.collect()
        torch.cuda.empty_cache()
    progress(record)
    return record


def _publish_charts(tracker, records):
    """One W&B point per successful candidate, with batch size as the x axis."""
    def publish():
        import wandb
        successes = [row for row in records if row["status"] == "complete"]
        if not successes:
            return
        table = wandb.Table(columns=["batch_per_task", "total_examples_per_second", "median_seconds",
                                    "mean_seconds", "peak_allocated_gib", "peak_reserved_gib"],
                            data=[[row["batch_per_task"], row["summary"]["total_examples_per_second"],
                                   row["summary"]["median_seconds"], row["summary"]["mean_seconds"],
                                   row["peak_allocated_gib"], row["peak_reserved_gib"]] for row in successes])
        tracker._run.log({"benchmark/candidate_table": table,
                          "benchmark/throughput_by_batch": wandb.plot.line(
                              table, "batch_per_task", "total_examples_per_second",
                              title="Mixed examples per second by examples per task"),
                          "benchmark/update_time_by_batch": wandb.plot.line(
                              table, "batch_per_task", "median_seconds", title="Median full-update seconds"),
                          "benchmark/memory_by_batch": wandb.plot.line(
                              table, "batch_per_task", "peak_allocated_gib", title="Peak allocated GiB")})
    tracker._call("batch-profile charts", publish)


def run(args):
    validate_arguments(args)
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Profile outputs require a fresh directory")
    config = read_configuration(args.config)
    raw = config["backbone"]
    if (raw["d_model"], raw["n_heads"], raw["mlp_hidden_size"]) != (128, 16, 512):
        raise ValueError("This profile retains the existing D128/H16/FFN512 model")
    # Integrity checks and dataset loading intentionally precede all update clocks.
    a5_root, fuzzy_root = Path(args.a5_data).resolve(), Path(args.fuzzy_data).resolve()
    a5_manifest = validate_manifest(a5_root)
    if file_sha256(a5_root / "manifest.json") != trainer.A5_MANIFEST_SHA256:
        raise ValueError("This profile requires the frozen A5 corpus")
    a5 = load_split(a5_root, "train")
    fuzzy = load_dataset(fuzzy_root, FUZZY_TASK, "train")
    arrays = {"a5": a5, "fuzzy": (fuzzy.input_ids, fuzzy.labels)}
    if any(arrays[task][0].shape[1] != length for task, length in TASK_LENGTHS.items()):
        raise ValueError("This profile requires actual A5 T12 and Fuzzy T400")
    if np.any(fuzzy.labels == IGNORE_INDEX):
        raise ValueError("Fuzzy training targets must retain native dense supervision")
    sources = source_manifest()
    data_identity = {"a5": {"manifest": a5_manifest, "manifest_sha256": trainer.A5_MANIFEST_SHA256},
                     "fuzzy": {"train": fuzzy.manifest,
                               "preparation_manifest_sha256": file_sha256(fuzzy_root / "manifest.json")}}
    output.mkdir(parents=True)
    for relative, expected in sources.items():
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if file_sha256(target) != expected:
            raise RuntimeError("Source changed while snapshotting profile dependencies")
    shutil.copy2(args.config, output / "model-config.json")
    atomic_json(output / "source-manifest.json", sources)
    atomic_json(output / "data-identity.json", data_identity)
    atomic_json(output / "invocation.json", vars(args))
    contract = {"schema": SCHEMA, "mode": "mixed", "model_config": config,
                "configuration_file_sha256": file_sha256(args.config),
                "source_sha256": json_sha256(sources), "data_sha256": json_sha256(data_identity),
                "hardware": hardware, "runtime": runtime, "batches_per_task": args.batches,
                "warmup_updates": args.warmup_updates, "timed_updates": args.timed_updates,
                "initialization_seeds": {"backbone": 1234, "predictor": 1235, "fuzzy_rows": 1236},
                "order_seeds": ORDER_SEEDS, "task_lengths": TASK_LENGTHS,
                "streams": {task: {"train_rows": len(arrays[task][0]), "length": length,
                                    "order_seed": ORDER_SEEDS[task]} for task, length in TASK_LENGTHS.items()},
                "optimizer": {"type": "AdamW", "lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8,
                              "matrix_decay": 0.01, "vector_decay": 0.0, "clip_norm": 1.0},
                "objective": "equal task-local means of CE + NextLat(weight1); one clip and Adam step",
                "timing_scope": "synchronized full update including row selection/order hashes, host-to-device "
                                "copies, input encoding, both tasks' forward/backward, diagnostics, clipping and Adam",
                "excluded_from_timing": ["data verification", "model/optimizer construction", "W&B/local logging",
                                         "finite-state endpoint audit", "cache cleanup"],
                "trial_semantics": "fresh identical initial weights and ordered stream prefixes; variable batch sizes "
                                   "consume different total examples and do not match learning trajectories",
                "checkpoint_written": False, "learning_comparison": False}
    report = {"schema": SCHEMA, "status": "running", "phase": "startup", "contract": contract,
              "candidates": [], "confirmation_evaluated": False, "checkpoint_written": False}
    tracker = OnlineTracker(project=args.wandb_project, entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name=args.wandb_run_name, preserve_state=preserve_rng)
    report["wandb"] = tracker.record
    atomic_json(output / "report.json", report)
    started = time.perf_counter()
    succeeded = False
    try:
        tracker.start(contract)
        with (output / "history.jsonl").open("x", buffering=1) as history:
            initialization = None
            first_prefix = None
            for trial_index, batch in enumerate(args.batches):
                def progress(record):
                    report.update(phase=record["phase"], current_candidate=record,
                                  elapsed_seconds=time.perf_counter() - started)
                    atomic_json(output / "report.json", report)
                    print(json.dumps({"event": "candidate_phase", "batch_per_task": batch,
                                      "phase": record["phase"], "status": record["status"],
                                      "completed_warmup_updates": record["completed_warmup_updates"],
                                      "completed_timed_updates": record["completed_timed_updates"],
                                      "summary": record.get("summary"),
                                      "wandb": tracker.record.get("run_url")}, allow_nan=False), flush=True)

                candidate = _profile_candidate(args, batch, config=config, arrays=arrays,
                                               expected_initialization=initialization, tracker=tracker,
                                               trial_index=trial_index, history=history, progress=progress)
                if "initialization" in candidate:
                    initialization = candidate["initialization"]
                if "first_128_ordered_rows_sha256" in candidate:
                    if first_prefix is not None and candidate["first_128_ordered_rows_sha256"] != first_prefix:
                        raise AssertionError("Candidates did not start from the same ordered task examples")
                    first_prefix = candidate["first_128_ordered_rows_sha256"]
                report["candidates"].append(candidate)
                if candidate["status"] == "complete":
                    tracker.summary(scalar_metrics(candidate["summary"], f"batch_{batch}"))
                atomic_json(output / "report.json", report)
        if source_manifest() != sources or file_sha256(args.config) != contract["configuration_file_sha256"]:
            raise RuntimeError("Executed sources or model configuration changed during profiling")
        _publish_charts(tracker, report["candidates"])
        report.update(status="complete", phase="complete", elapsed_seconds=time.perf_counter() - started,
                      successful_candidates=sum(row["status"] == "complete" for row in report["candidates"]),
                      oom_candidates=sum(row["status"] == "oom" for row in report["candidates"]))
        report.pop("current_candidate", None)
        tracker.summary({key: report[key] for key in ("status", "successful_candidates", "oom_candidates")})
        succeeded = True
    except BaseException as error:
        report.update(status="failed", phase="failed", error_type=type(error).__name__,
                      elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        try:
            tracker.finish(succeeded=succeeded)
        finally:
            report["wandb"] = tracker.record
            atomic_json(output / "report.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--a5-data", type=Path, required=True)
    p.add_argument("--fuzzy-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--warmup-updates", type=int, default=10)
    p.add_argument("--timed-updates", type=int, default=30)
    p.add_argument("--wandb-project", default="rt-nextlat-fuzzy-a5")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-run-name", default="d128-a5-fuzzy-mixed-batch-throughput")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
