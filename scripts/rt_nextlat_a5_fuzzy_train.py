#!/usr/bin/env python3
"""A5-only and equally weighted A5/Fuzzy FP32 pilots with independent streams.

This imports the frozen Fuzzy pilot's helpers without changing that lineage.
One mixed update contains equal task example counts, separate task-normalized
CE/NextLat means, and exactly one global clipping/Adam operation. Confirmation
data and autonomous latent rollout are deliberately absent from this CLI.
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
SCHEMA = "rt-nextlat-a5-fuzzy-training-v1"
A5_MANIFEST_SHA256 = "944c7a2e86a9329611c0fee74aaad59dfec1e77604c58a7ae7a465c8d529f9eb"
ORDER_INITIAL = {"a5": hashlib.sha256(b"rt-nextlat-a5-word-order-v1").hexdigest(),
                 "fuzzy": fuzzy_base.ORDER_INITIAL}
TASK_METRICS = ("loss", "ce", "latent", "weighted_latent", "token_accuracy",
                "scored_sequence_exact")


def active_tasks(mode):
    if mode == "a5-only":
        return ("a5",)
    if mode == "mixed":
        return ("a5", "fuzzy")
    raise ValueError("mode must be 'a5-only' or 'mixed'")


def source_manifest():
    sources = dict(fuzzy_base.source_manifest())
    relative = "scripts/rt_nextlat_a5_fuzzy_train.py"
    sources[relative] = file_sha256(ROOT / relative)
    return dict(sorted(sources.items()))


def train_step(model, optimizer, task_batches, *, mode="mixed", microbatch=128,
               latent_weight=1.0, clip_norm=1.0):
    """Update once from encoded inputs/local labels, weighting tasks equally.

    CE microbatch weights use scored tokens within that task; latent weights
    use examples, since each task has a fixed within-task sequence length.
    Thus longer Fuzzy words never receive extra objective weight simply by
    having more tokens. Uneven physical microbatches retain the same means.
    """
    tasks = active_tasks(mode)
    if set(task_batches) != set(tasks):
        raise ValueError("Batch tasks differ from mode")
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    if not math.isfinite(latent_weight) or latent_weight < 0:
        raise ValueError("latent_weight must be finite and nonnegative")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    counts, scored_totals, devices = {}, {}, set()
    for task in tasks:
        x, y = task_batches[task]
        if (x.ndim != 2 or y.shape != x.shape or len(x) < 1 or x.shape[1] < 2
                or x.dtype != torch.long or y.dtype != torch.long or x.device != y.device):
            raise ValueError("Task inputs/labels must be nonempty int64 [B,T>=2] on one device")
        counts[task] = len(x)
        devices.add(x.device)
        scored_totals[task] = int(y.ne(IGNORE_INDEX).sum().item())
        if not scored_totals[task]:
            raise ValueError("No scored targets in a task batch")
        if any(not y[s:s + microbatch].ne(IGNORE_INDEX).any().item()
               for s in range(0, len(x), microbatch)):
            raise ValueError("A training microbatch has no scored targets")
    if len(set(counts.values())) != 1 or len(devices) != 1:
        raise ValueError("Mixed tasks require equal example counts and one device")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = {}
    task_weight = 1.0 / len(tasks)
    for task in tasks:
        inputs, labels = task_batches[task]
        sums = torch.zeros(4, dtype=torch.float64, device=inputs.device)
        for start in range(0, len(inputs), microbatch):
            x, y = inputs[start:start + microbatch], labels[start:start + microbatch]
            scored = int(y.ne(IGNORE_INDEX).sum().item())
            ce_weight = scored / scored_totals[task]
            row_weight = len(x) / len(inputs)
            with fp32_context(inputs.device):
                result = task_loss(model, x, y, task=task, latent_weight=latent_weight)
                loss = task_weight * (result["ce"] * ce_weight
                                      + latent_weight * result["latent"] * row_weight)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite accumulated task objective")
                loss.backward()
            with torch.no_grad():
                good = result["logits"].argmax(-1).eq(y) & y.ne(IGNORE_INDEX)
                sums[0] += result["ce"].detach().double() * ce_weight
                sums[1] += result["latent"].detach().double() * row_weight
                sums[2] += good.sum()
                sums[3] += (good | y.eq(IGNORE_INDEX)).all(-1).sum()
        ce, latent, correct, exact = sums.cpu().tolist()
        values[task] = {"loss": ce + latent_weight * latent, "ce": ce, "latent": latent,
                        "weighted_latent": latent_weight * latent,
                        "token_accuracy": correct / scored_totals[task],
                        "scored_sequence_exact": exact / counts[task],
                        "examples": counts[task], "scored_tokens": scored_totals[task]}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    fuzzy_base._sync(next(iter(devices)))
    return {"loss": sum(v["loss"] for v in values.values()) * task_weight,
            "grad_norm": fuzzy_base._finite_scalar(norm, "gradient norm"),
            "examples": sum(counts.values()), "tasks": values}


def evaluate_a5(model, inputs, labels, *, microbatch=1024, device="cuda", limit=None):
    """Same-position A5 states, exact integer whole-word counts, no rollout."""
    rows = len(inputs) if limit is None else min(len(inputs), limit)
    if rows < 1 or microbatch < 1 or inputs.shape != labels.shape:
        raise ValueError("Evaluation requires aligned nonempty arrays and positive counts")
    metrics = A5Metrics()
    latent_sum = 0.0
    training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        with preserve_rng(), torch.no_grad(), fp32_context(device):
            for start in range(0, rows, microbatch):
                x, y = batch_tensors(inputs, labels, slice(start, min(rows, start + microbatch)), device)
                result = task_loss(model, encode_inputs(x, task="a5"), y, task="a5", latent_weight=1.0)
                metrics.update(result["logits"], y)
                latent_sum += fuzzy_base._finite_scalar(result["latent"], "A5 dev NextLat") * len(x)
            fuzzy_base._sync(device)
        result = metrics.compute()
        result.update(whole_word_exact_count=int(metrics.prefix_correct[-1].item()),
                      whole_word_correct=int(metrics.prefix_correct[-1].item()),
                      per_position_prefix_correct=metrics.prefix_correct.tolist(),
                      per_position_state_correct=metrics.correct.tolist(),
                      latent=latent_sum / rows, weighted_latent=latent_sum / rows,
                      evaluated_rows=rows, evaluation_seconds=time.perf_counter() - started,
                      route="backbone logits; teacher-conditioned NextLat diagnostic; no rollout")
        return result
    finally:
        model.train(training)


def cursors(completed, contract):
    result = {}
    for task in active_tasks(contract["mode"]):
        offset = completed * contract["batch_per_task"]
        epoch, position = divmod(offset, contract["streams"][task]["train_rows"])
        result[task] = {"absolute_example_offset": offset, "epoch": epoch, "position": position}
    return result


def _validate_chains(chains, tasks):
    return (isinstance(chains, dict) and set(chains) == set(tasks)
            and all(fuzzy_base._valid_chain(v) for v in chains.values()))


def save_checkpoint(path, *, model, optimizer, contract, completed, order_chains,
                    initialization):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tasks = active_tasks(contract["mode"])
    if type(completed) is not int or completed < 0 or not _validate_chains(order_chains, tasks):
        raise ValueError("Invalid update counter/order chains")
    audit = fuzzy_base.finite_state(model, optimizer, completed)
    seen = {task: completed * contract["batch_per_task"] for task in tasks}
    packet = {"schema": SCHEMA, "contract": contract, "completed_updates": completed,
              "examples_seen": seen, "next_cursors": cursors(completed, contract),
              "order_chains": dict(order_chains), "model": cpu_tree(model.state_dict()),
              "optimizer": cpu_tree(optimizer.state_dict()),
              "optimizer_parameter_names": optimizer_names(model, optimizer),
              "rng": rng_state(), "initialization": json_value(initialization), "finite_state": audit}
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
    _validate_optimizer(packet, model, optimizer, completed)
    if not isinstance(packet.get("rng"), dict) or set(packet["rng"]) != set(rng_state()):
        raise ValueError("Invalid checkpoint RNG state")
    with preserve_rng():
        restore_rng(packet["rng"])
    model.load_state_dict(packet["model"], strict=True)
    optimizer.load_state_dict(packet["optimizer"])
    restore_rng(packet["rng"])
    return packet


def evaluation_metrics(result):
    prefix = f"dev/{result['task']}/{result['role']}"
    values = scalar_metrics(result, prefix)
    # Explicit position keys preserve the familiar A5 cumulative-prefix plots.
    for name in ("isolated_state_accuracy", "cumulative_prefix_exactness", "per_position_ce"):
        for position, value in enumerate(result.get(name, ()), 1):
            values[f"{prefix}/{name}/position_{position}"] = value
    return values


def inherited_positive_gate(resume, *, contract, completed):
    """Carry an earlier positive gate only with retained checkpoint evidence.

    This is reporting history, not optimizer state. A parent run may have
    trained beyond the chosen resume checkpoint, so later evidence cannot
    establish eligibility in a replayed child before it is actually observed.
    """
    parent_report_path = Path(resume).resolve().parent.parent / "report.json"
    if not parent_report_path.exists():
        return None
    parent_report = json.loads(parent_report_path.read_text())
    if parent_report.get("contract") != contract:
        raise ValueError("Parent report contract differs from resume checkpoint")
    gate = parent_report.get("a5_positive_gate", {})
    if not gate.get("eligible") or gate.get("first_positive_update", completed + 1) > completed:
        return None
    identity = gate.get("checkpoint", {})
    if (type(gate.get("first_positive_update")) is not int
            or gate.get("whole_word_exact_count", 0) < 1
            or gate.get("evaluated_rows", 0) < 102400
            or gate.get("length") != 36
            or identity.get("completed_updates") != gate["first_positive_update"]
            or not Path(identity.get("path", "")).is_file()
            or file_sha256(identity["path"]) != identity.get("sha256")):
        raise ValueError("Parent positive gate lacks retained full-L36 checkpoint evidence")
    result = dict(gate)
    result["inherited_from"] = {"report": str(parent_report_path),
                               "report_sha256": file_sha256(parent_report_path),
                               "restored_update": completed}
    return result


def run(args):
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
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
    optimizer = make_optimizer(model, lr=args.learning_rate)
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
                "optimizer": {"type": "AdamW", "lr": args.learning_rate, "betas": [0.9, 0.95],
                              "eps": 1e-8, "matrix_decay": 0.01, "vector_decay": 0.0, "clip_norm": 1.0},
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
            values = train_step(model, optimizer, batches, mode=args.mode, microbatch=args.microbatch)
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            completed, chains = completed + 1, next_chains
            row = {"update": completed, "examples_seen": {task: completed * args.batch_per_task for task in tasks},
                   "seconds": elapsed, "order_chains": dict(chains), **values}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            window.append(row)
            if completed % args.log_every == 0 or completed == args.updates:
                logged = {"update": completed,
                          "train/seconds_per_update": sum(r["seconds"] for r in window) / len(window),
                          "train/loss": sum(r["loss"] for r in window) / len(window),
                          "train/grad_norm": sum(r["grad_norm"] for r in window) / len(window),
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
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    p.add_argument("--wandb-project", default="rt-nextlat-fuzzy-a5")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-run-name")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
