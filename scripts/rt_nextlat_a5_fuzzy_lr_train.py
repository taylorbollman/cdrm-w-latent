#!/usr/bin/env python3
"""Matched FP32 mixed-task pilot with a short linear learning-rate warmup.

The original training step, task losses, evaluators, and data ordering are
imported unchanged. Only the global-update learning rate and its logging,
checkpoint identity, and strict continuation contract differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch

from cdrm.mad_data import FUZZY_TASK, IGNORE_INDEX, load_dataset
from cdrm.rt_nextlat_tasks import CONFIG_PATH, build_model, encode_inputs, read_configuration, task_loss
from scripts import rt_nextlat_fuzzy_train as fuzzy_base
from scripts import rt_nextlat_a5_fuzzy_train as original
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.rt_a5_common import A5Metrics, configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_nextlat_train import _validate_optimizer
from scripts.rt_a5_train import (
    WordOrder, atomic_json, batch_tensors, cpu_tree, file_sha256, json_sha256,
    json_value, optimizer_names, preserve_rng, restore_rng, rng_state,
    validate_model_state,
)
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-a5-fuzzy-lr-training-v1"
A5_MANIFEST_SHA256 = "944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb"
ORDER_INITIAL = {"a5": hashlib.sha256(b"rt-nextlat-a5-word-order-v1").hexdigest(),
                 "fuzzy": fuzzy_base.ORDER_INITIAL}
TASK_METRICS = ("loss", "ce", "latent", "weighted_latent", "token_accuracy",
                "scored_sequence_exact")


# These are the exact historical function objects, not copied implementations.
active_tasks = original.active_tasks
train_step = original.train_step
evaluate_a5 = original.evaluate_a5
cursors = original.cursors
_validate_chains = original._validate_chains
evaluation_metrics = original.evaluation_metrics
inherited_positive_gate = original.inherited_positive_gate


def source_manifest():
    sources = dict(original.source_manifest())
    relative = "scripts/rt_nextlat_a5_fuzzy_lr_train.py"
    sources[relative] = file_sha256(ROOT / relative)
    return dict(sorted(sources.items()))


def schedule_configuration(start_lr=1e-4, peak_lr=3e-4, warmup_updates=100):
    """The first optimizer update uses start_lr; update warmup_updates uses peak."""
    if (isinstance(start_lr, bool) or isinstance(peak_lr, bool)
            or not isinstance(start_lr, (int, float)) or not isinstance(peak_lr, (int, float))
            or not math.isfinite(start_lr) or not math.isfinite(peak_lr)
            or not 0 < start_lr <= peak_lr):
        raise ValueError("Learning rates must be finite with 0 < start_lr <= peak_lr")
    if type(warmup_updates) is not int or warmup_updates < 2:
        raise ValueError("warmup_updates must be an integer of at least two")
    return {"schema": "rt-nextlat-a5-fuzzy-linear-lr-v1", "kind": "linear_warmup_then_constant",
            "start_lr": float(start_lr), "peak_lr": float(peak_lr), "warmup_updates": warmup_updates,
            "first_update": 1, "peak_update": warmup_updates,
            "interpolation": "start_lr + (peak_lr-start_lr)*(update-1)/(warmup_updates-1)",
            "progress": "global completed optimizer update counter, never reset on resume",
            "checkpoint_convention": "step0 stores start_lr; stepN stores the LR applied at N"}


def validate_schedule(schedule):
    if not isinstance(schedule, dict):
        raise ValueError("A saved learning-rate schedule is required")
    expected = schedule_configuration(schedule.get("start_lr"), schedule.get("peak_lr"),
                                      schedule.get("warmup_updates"))
    if schedule != expected:
        raise ValueError("Learning-rate schedule contract is malformed or unsupported")


def learning_rate(update, *, start_lr=1e-4, peak_lr=3e-4, warmup_updates=100):
    """Actual applied LR: update 1 = start, update 100 = peak, then constant."""
    schedule_configuration(start_lr, peak_lr, warmup_updates)
    if type(update) is not int or update < 1:
        raise ValueError("Optimizer update must be a positive integer")
    if update == 1:
        return float(start_lr)
    if update >= warmup_updates:
        return float(peak_lr)
    return float(start_lr + (peak_lr - start_lr) * (update - 1) / (warmup_updates - 1))


def schedule_rate(update, schedule):
    validate_schedule(schedule)
    return learning_rate(update, start_lr=schedule["start_lr"], peak_lr=schedule["peak_lr"],
                         warmup_updates=schedule["warmup_updates"])


def learning_rate_state(completed, schedule):
    if type(completed) is not int or completed < 0:
        raise ValueError("Completed update counter must be a nonnegative integer")
    return {"completed_updates": completed, "optimizer_lr": schedule_rate(max(1, completed), schedule),
            "next_update_lr": schedule_rate(completed + 1, schedule)}


def validate_optimizer_lr(saved_optimizer, expected_lr):
    groups = saved_optimizer.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Optimizer parameter groups are missing")
    if any(group.get("lr") != expected_lr for group in groups):
        raise ValueError("Optimizer LR differs from completed schedule position")


def set_learning_rate(optimizer, update, schedule):
    value = schedule_rate(update, schedule)
    for group in optimizer.param_groups:
        group["lr"] = value
    return value


def validate_scheduled_optimizer(packet, model, optimizer, completed, contract):
    """Validate scheduled LR separately; retain every original Adam state check.

    Only the LR is normalized for the historical strict group comparison. The
    saved LR has already been checked against the global completed update.
    Neither the caller's optimizer nor the packet is changed before validation.
    """
    expected = learning_rate_state(completed, contract["learning_rate_schedule"])
    if packet.get("learning_rate_state") != expected:
        raise ValueError("Checkpoint learning-rate state differs from completed schedule position")
    validate_optimizer_lr(packet["optimizer"], expected["optimizer_lr"])
    saved_groups = packet["optimizer"]["param_groups"]
    live_groups = optimizer.state_dict()["param_groups"]
    if len(saved_groups) != len(live_groups):
        raise ValueError("Optimizer group count differs")
    comparable = {**packet, "optimizer": {**packet["optimizer"], "param_groups": [
        {**saved, "lr": live["lr"]} for saved, live in zip(saved_groups, live_groups)]}}
    _validate_optimizer(comparable, model, optimizer, completed)


def save_checkpoint(path, *, model, optimizer, contract, completed, order_chains,
                    initialization):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tasks = active_tasks(contract["mode"])
    if type(completed) is not int or completed < 0 or not _validate_chains(order_chains, tasks):
        raise ValueError("Invalid update counter/order chains")
    schedule_state = learning_rate_state(completed, contract["learning_rate_schedule"])
    validate_optimizer_lr(optimizer.state_dict(), schedule_state["optimizer_lr"])
    audit = fuzzy_base.finite_state(model, optimizer, completed)
    seen = {task: completed * contract["batch_per_task"] for task in tasks}
    packet = {"schema": SCHEMA, "contract": contract, "completed_updates": completed,
              "examples_seen": seen, "next_cursors": cursors(completed, contract),
              "order_chains": dict(order_chains), "model": cpu_tree(model.state_dict()),
              "optimizer": cpu_tree(optimizer.state_dict()),
              "optimizer_parameter_names": optimizer_names(model, optimizer),
              "rng": rng_state(), "initialization": json_value(initialization), "finite_state": audit,
              "learning_rate_state": schedule_state}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(packet, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "bytes": path.stat().st_size, "completed_updates": completed,
            "examples_seen": seen, **audit}


def load_checkpoint(path, *, model, optimizer, contract):
    # Only locally generated/retained full-state checkpoint files are supported.
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if packet.get("schema") != SCHEMA or packet.get("contract") != contract:
        raise ValueError("Checkpoint model/data/source/runtime/optimizer contract differs")
    tasks = active_tasks(contract["mode"])
    completed = packet.get("completed_updates")
    if (type(completed) is not int or completed < 0
            or packet.get("examples_seen") != {task: completed * contract["batch_per_task"] for task in tasks}
            or packet.get("next_cursors") != cursors(completed, contract)
            or not _validate_chains(packet.get("order_chains"), tasks)):
        raise ValueError("Checkpoint update/cursors/order chains differ")
    if packet.get("initialization") != json_value(model.initialization):
        raise ValueError("Checkpoint canonical initialization differs")
    validate_model_state(packet["model"], model.state_dict())
    validate_scheduled_optimizer(packet, model, optimizer, completed, contract)
    if not isinstance(packet.get("rng"), dict) or set(packet["rng"]) != set(rng_state()):
        raise ValueError("Invalid checkpoint RNG state")
    with preserve_rng():
        restore_rng(packet["rng"])
    model.load_state_dict(packet["model"], strict=True)
    optimizer.load_state_dict(packet["optimizer"])
    restore_rng(packet["rng"])
    return packet


def run(args):
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    schedule = schedule_configuration(args.warmup_start_lr, args.learning_rate, args.warmup_updates)
    tasks = active_tasks(args.mode)
    counts = (args.updates, args.batch_per_task, args.microbatch, args.eval_microbatch,
              args.fuzzy_eval_microbatch, args.eval_every, args.eval_rows,
              args.fuzzy_eval_rows, args.full_eval_rows, args.log_every)
    seeds = (args.seed, args.predictor_seed, args.fuzzy_seed, args.a5_order_seed, args.fuzzy_order_seed)
    if min(counts) < 1 or min(seeds) < 0 or any(s < 0 for s in args.checkpoint_steps + args.full_eval_steps):
        raise ValueError("Positive counts, nonnegative seeds and evaluation/checkpoint steps required")
    if args.wandb_mode == "disabled" and args.updates > 10:
        raise ValueError("Disabled W&B is restricted to at most ten integration-check updates")
    if args.mode == "mixed" and args.fuzzy_data is None:
        raise ValueError("mixed requires --fuzzy-data")
    output, a5_root = Path(args.output).resolve(), Path(args.a5_data).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; resume creates a new lineage")
    config = read_configuration(args.config)
    manifest = validate_manifest(a5_root)
    a5_hash = file_sha256(a5_root / "manifest.json")
    if a5_hash != A5_MANIFEST_SHA256:
        raise ValueError("This milestone requires the frozen A5 corpus manifest")
    train_arrays = {"a5": load_split(a5_root, "train")}
    a5_dev = {role: load_split(a5_root, role) for role in ("dev", "ood_dev")}
    if train_arrays["a5"][0].shape[1] != 12 or a5_dev["ood_dev"][0].shape[1] != 36:
        raise ValueError("Expected A5 train length 12 and OOD length 36")
    data_identity = {"a5": {"manifest_sha256": a5_hash, "manifest": manifest}}
    streams = {"a5": {"train_rows": len(train_arrays["a5"][0]), "length": 12,
                       "order_seed": args.a5_order_seed}}
    fuzzy_dev = None
    if "fuzzy" in tasks:
        fuzzy_root = Path(args.fuzzy_data).resolve()
        fuzzy_train = load_dataset(fuzzy_root, FUZZY_TASK, "train")
        fuzzy_dev = load_dataset(fuzzy_root, FUZZY_TASK, "dev")
        if np.any(fuzzy_train.labels == IGNORE_INDEX):
            raise ValueError("Native Fuzzy training labels must remain dense")
        if fuzzy_train.input_ids.shape[1] != fuzzy_dev.input_ids.shape[1]:
            raise ValueError("Fuzzy train/dev lengths differ")
        train_arrays["fuzzy"] = (fuzzy_train.input_ids, fuzzy_train.labels)
        data_identity["fuzzy"] = {"train": fuzzy_train.manifest, "dev": fuzzy_dev.manifest,
                                  "preparation_manifest_sha256": file_sha256(fuzzy_root / "manifest.json")}
        streams["fuzzy"] = {"train_rows": len(fuzzy_train), "length": fuzzy_train.input_ids.shape[1],
                            "order_seed": args.fuzzy_order_seed}
    if max(value["length"] for value in streams.values()) > config["backbone"]["max_sequence_length"]:
        raise ValueError("Training length exceeds model configuration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(config, seed=args.seed, predictor_seed=args.predictor_seed,
                        fuzzy_seed=args.fuzzy_seed, device="cuda")
    optimizer = make_optimizer(model, lr=schedule["start_lr"])
    sources = source_manifest()
    config_path = Path(args.config).resolve()
    contract = {"schema": SCHEMA, "mode": args.mode, "model_config": json_value(config),
                "configuration_file_sha256": file_sha256(config_path),
                "source_sha256": json_sha256(sources), "data_sha256": json_sha256(data_identity),
                "initialization": json_value(model.initialization), "streams": streams,
                "batch_per_task": args.batch_per_task, "microbatch": args.microbatch,
                "runtime": runtime, "device_capability": hardware["capability"],
                "objective": {"ce": "task-local aligned labels; native dense Fuzzy; same-position A5",
                              "latent": "SmoothL1 beta1; all T-1 native transitions; detached target",
                              "latent_weight": 1.0, "task_weights": {task: 1 / len(tasks) for task in tasks},
                              "reduction": "separate task means; one global clip and Adam step"},
                "optimizer": {"type": "AdamW", "lr": args.learning_rate, "initial_lr": args.warmup_start_lr, "betas": [0.9, 0.95],
                              "eps": 1e-8, "matrix_decay": 0.01, "vector_decay": 0.0, "clip_norm": 1.0},
                "learning_rate_schedule": schedule,
                "word_order": "independent per-task shuffled epochs with remainder carried"}
    completed, chains, parent = 0, {task: ORDER_INITIAL[task] for task in tasks}, None
    prior_gate = None
    if args.resume:
        packet = load_checkpoint(args.resume, model=model, optimizer=optimizer, contract=contract)
        completed, chains = packet["completed_updates"], packet["order_chains"]
        if completed >= args.updates:
            raise ValueError("Continuation endpoint must exceed restored update")
        parent = {"path": str(Path(args.resume).resolve()), "sha256": file_sha256(args.resume)}
        prior_gate = inherited_positive_gate(args.resume, contract=contract, completed=completed)
    output.mkdir(parents=True)
    for relative, expected in sources.items():
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if file_sha256(target) != expected:
            raise RuntimeError(f"Source changed while being snapshotted: {relative}")
    shutil.copy2(config_path, output / "model-config.json")
    atomic_json(output / "source-manifest.json", sources)
    atomic_json(output / "data-identity.json", data_identity)
    atomic_json(output / "invocation.json", vars(args))
    report = {"schema": SCHEMA, "status": "running", "contract": contract, "hardware": hardware,
              "parent_checkpoint": parent, "start_update": completed, "completed_updates": completed,
              "requested_endpoint": args.updates, "initialization": json_value(model.initialization),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "checkpoints": [], "evaluations": [], "confirmation_evaluated": False,
              "latent_rollout_evaluated": False,
              "a5_positive_gate": prior_gate or {
                  "eligible": False, "observed_by_3000": False, "length": 36,
                  "criterion": "full development L36 exact-word count > 0"}}
    tracker = (fuzzy_base._DisabledTracker() if args.wandb_mode == "disabled" else OnlineTracker(
        project=args.wandb_project, entity="taylorbollman", output_dir=output,
        group=args.wandb_group, name=args.wandb_run_name or f"l1r-rt2-nextlat-d128-{args.mode}",
        preserve_state=preserve_rng))
    report["wandb"] = tracker.record
    atomic_json(output / "report.json", report)
    checkpoint_steps = set(args.checkpoint_steps) | {args.updates}
    full_eval_steps = set(args.full_eval_steps) | {args.updates}
    checkpointed = {}
    orders = {task: WordOrder(streams[task]["train_rows"], streams[task]["order_seed"]) for task in tasks}
    started, train_seconds = time.perf_counter(), 0.0
    stop_requested, handlers = [], {}

    def request_stop(number, frame):
        stop_requested.append(signal.Signals(number).name)

    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, request_stop)

    def checkpoint():
        if completed not in checkpointed:
            record = save_checkpoint(
                output / f"checkpoints/step-{completed:06d}.pt", model=model, optimizer=optimizer,
                contract=contract, completed=completed, order_chains=chains,
                initialization=model.initialization)
            checkpointed[completed] = record
            report["checkpoints"].append(record)
        return {key: checkpointed[completed][key] for key in ("path", "sha256", "completed_updates")}

    def record_evaluation(result, task, role, scope, identity):
        result.update(update=completed, task=task, role=role, scope=scope, checkpoint=identity)
        report["evaluations"].append(result)
        tracker.log({"update": completed, **evaluation_metrics(result)})
        print(json.dumps({"event": "evaluation", **result}, allow_nan=False), flush=True)
        if (task == "a5" and role == "ood_dev" and scope == "full"
                and result["evaluated_rows"] >= 102400 and result["whole_word_exact_count"] > 0
                and not report["a5_positive_gate"]["eligible"]):
            report["a5_positive_gate"].update(
                eligible=True, first_positive_update=completed,
                observed_by_3000=completed <= 3000, length=36,
                whole_word_exact_count=result["whole_word_exact_count"],
                whole_word_exact_match=result["whole_word_exact_match"],
                evaluated_rows=result["evaluated_rows"], checkpoint=identity)
            atomic_json(output / "a5-positive-gate.json", report["a5_positive_gate"])

    def evaluate_checkpoint(force_full=False):
        identity = checkpoint()
        full = force_full or completed in full_eval_steps
        limit = args.full_eval_rows if full else args.eval_rows
        scope = "full" if limit >= 102400 else "subset"
        for role, arrays in a5_dev.items():
            result = evaluate_a5(model, *arrays, microbatch=args.eval_microbatch, limit=limit)
            record_evaluation(result, "a5", role, scope, identity)
            if (role == "ood_dev" and scope != "full" and result["whole_word_exact_count"] > 0
                    and not report["a5_positive_gate"]["eligible"]):
                confirmed = evaluate_a5(model, *arrays, microbatch=args.eval_microbatch,
                                        limit=args.full_eval_rows)
                record_evaluation(confirmed, "a5", role,
                                  "full" if confirmed["evaluated_rows"] >= 102400 else "subset", identity)
        if fuzzy_dev is not None:
            result = fuzzy_base.evaluate(model, fuzzy_dev, microbatch=args.fuzzy_eval_microbatch,
                                          limit=args.fuzzy_eval_rows)
            record_evaluation(result, "fuzzy", "dev", "full" if args.fuzzy_eval_rows >= len(fuzzy_dev)
                              else "subset", identity)
        atomic_json(output / "report.json", report)

    succeeded = False
    history = (output / "history.jsonl").open("x", buffering=1)
    try:
        tracker.start(contract)
        print(json.dumps({"event": "started", "wandb": tracker.record.get("run_url"),
                          "parameters": report["parameter_count"], "endpoint": args.updates,
                          "mode": args.mode}), flush=True)
        checkpoint()
        atomic_json(output / "report.json", report)
        window = []
        while completed < args.updates:
            if stop_requested or (args.stop_file and Path(args.stop_file).exists()):
                report["stop_reason"] = stop_requested or ["stop_file"]
                break
            fuzzy_base._sync("cuda")
            begin = time.perf_counter()
            batches, next_chains = {}, {}
            for task in tasks:
                indices = orders[task].indices(completed * args.batch_per_task, args.batch_per_task)
                next_chains[task] = hashlib.sha256(
                    bytes.fromhex(chains[task]) + indices.astype("<i8").tobytes()).hexdigest()
                x, y = batch_tensors(*train_arrays[task], indices, "cuda")
                batches[task] = (encode_inputs(x, task=task), y)
            actual_lr = set_learning_rate(optimizer, completed + 1, schedule)
            values = train_step(model, optimizer, batches, mode=args.mode, microbatch=args.microbatch)
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            completed, chains = completed + 1, next_chains
            row = {"update": completed, "examples_seen": {task: completed * args.batch_per_task for task in tasks},
                   "seconds": elapsed, "order_chains": dict(chains), "learning_rate": actual_lr,
                   "gradient_clipped": values["grad_norm"] > contract["optimizer"]["clip_norm"], **values}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            window.append(row)
            if completed % args.log_every == 0 or completed == args.updates:
                logged = {"update": completed,
                          "train/seconds_per_update": sum(r["seconds"] for r in window) / len(window),
                          "train/loss": sum(r["loss"] for r in window) / len(window),
                          "train/grad_norm": sum(r["grad_norm"] for r in window) / len(window),
                          "train/learning_rate": actual_lr,
                          "train/clip_fraction": sum(r["gradient_clipped"] for r in window) / len(window),
                          "train/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "train/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
                for task in tasks:
                    logged[f"train/{task}/examples_seen"] = completed * args.batch_per_task
                    for key in TASK_METRICS:
                        logged[f"train/{task}/{key}"] = sum(r["tasks"][task][key] for r in window) / len(window)
                tracker.log(logged)
                print(json.dumps({"event": "train", **logged}), flush=True)
                window.clear()
            if completed in checkpoint_steps:
                checkpoint()
            if completed % args.eval_every == 0 or completed in checkpoint_steps or completed in full_eval_steps:
                evaluate_checkpoint()
            report.update(completed_updates=completed, order_chains=dict(chains),
                          examples_seen={task: completed * args.batch_per_task for task in tasks},
                          next_cursors=cursors(completed, contract), train_seconds=train_seconds,
                          elapsed_seconds=time.perf_counter() - started)
            if completed % args.log_every == 0 or completed in checkpoint_steps:
                atomic_json(output / "report.json", report)
        checkpoint()
        endpoint_full = any(result["update"] == completed and result["task"] == "a5"
                            and result["role"] == "ood_dev" and result["scope"] == "full"
                            for result in report["evaluations"])
        if not endpoint_full:
            evaluate_checkpoint(force_full=True)
        if source_manifest() != sources or file_sha256(config_path) != contract["configuration_file_sha256"]:
            raise RuntimeError("Executed sources or configuration changed during training")
        report.update(status="complete" if completed == args.updates else "stopped",
                      completed_updates=completed, order_chains=dict(chains),
                      examples_seen={task: completed * args.batch_per_task for task in tasks},
                      next_cursors=cursors(completed, contract), train_seconds=train_seconds,
                      elapsed_seconds=time.perf_counter() - started)
        final_values = {"completed_updates": completed, "status": report["status"],
                        "confirmation_evaluated": False,
                        **scalar_metrics(report["a5_positive_gate"], "a5_positive_gate")}
        for result in report["evaluations"]:
            if result["update"] == completed:
                final_values.update(evaluation_metrics(result))
        tracker.summary(final_values)
        atomic_json(output / "a5-positive-gate.json", report["a5_positive_gate"])
        succeeded = True
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, completed_updates=completed,
                      order_chains=dict(chains), elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        history.flush()
        os.fsync(history.fileno())
        history.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        try:
            tracker.finish(succeeded=succeeded)
        finally:
            report["wandb"] = tracker.record
            atomic_json(output / "report.json", report)
    print(json.dumps({"event": "finished", "status": report["status"], "completed_updates": completed,
                      "a5_positive_gate": report["a5_positive_gate"],
                      "wandb": tracker.record.get("run_url")}), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("a5-only", "mixed"), required=True)
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--a5-data", type=Path, required=True)
    p.add_argument("--fuzzy-data", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=10000)
    p.add_argument("--batch-per-task", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=128)
    p.add_argument("--eval-microbatch", type=int, default=1024)
    p.add_argument("--fuzzy-eval-microbatch", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-rows", type=int, default=4096)
    p.add_argument("--fuzzy-eval-rows", type=int, default=1280)
    p.add_argument("--full-eval-rows", type=int, default=102400)
    p.add_argument("--full-eval-steps", type=int, nargs="+", default=[1000, 3000, 5000, 10000])
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[0, 1000, 3000, 5000, 10000])
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--predictor-seed", type=int, default=1235)
    p.add_argument("--fuzzy-seed", type=int, default=1236)
    p.add_argument("--a5-order-seed", type=int, default=5432)
    p.add_argument("--fuzzy-order-seed", type=int, default=2026091604)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-start-lr", type=float, default=1e-4)
    p.add_argument("--warmup-updates", type=int, default=100)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    p.add_argument("--wandb-project", default="rt-nextlat-fuzzy-a5")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-run-name")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
