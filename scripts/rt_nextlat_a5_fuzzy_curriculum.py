#!/usr/bin/env python3
"""A5-warm-start control and phase-local Fuzzy-weight curriculum, FP32 only.

The fixed A5 10k checkpoint supplies model, Adam, RNG and the continued A5
stream. Fuzzy starts at its own offset zero. This stage has a separate strict
resume schema; it never weakens or edits the historical trainer's contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from cdrm.mad_data import FUZZY_TASK, IGNORE_INDEX, load_dataset
from cdrm.rt_nextlat_tasks import CONFIG_PATH, build_model, encode_inputs, read_configuration, task_loss
from scripts import rt_nextlat_a5_fuzzy_train as base
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_nextlat_train import _validate_optimizer
from scripts.rt_a5_train import (
    WordOrder, atomic_json, batch_tensors, cpu_tree, file_sha256, json_sha256,
    json_value, optimizer_names, preserve_rng, restore_rng, rng_state, validate_model_state,
)
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-a5-fuzzy-curriculum-v1"
PARENT_UPDATES = 10000
PARENT_SHA256 = "0f246e2f39f8a26c8a39fcb336ee1bac4da12990f14bdfaf5676b45ce1364701"
PARENT_PATH = ROOT / ".runtime/rt-nextlat-fuzzy-a5/20260916T182200Z-d128-a5-mixed/train-a5/checkpoints/step-010000.pt"
MODES = ("a5-control", "mixed-curriculum")


def task_weights(mode, phase_update, ramp_updates=3000):
    if mode not in MODES:
        raise ValueError("Unsupported curriculum mode")
    if type(phase_update) is not int or phase_update < 0:
        raise ValueError("phase_update must be a nonnegative integer")
    if type(ramp_updates) is not int or ramp_updates < 1:
        raise ValueError("ramp_updates must be a positive integer")
    fuzzy = 0.0 if mode == "a5-control" else 0.5 * min(phase_update / ramp_updates, 1.0)
    return {"a5": 1.0 - fuzzy, "fuzzy": fuzzy}


def active_tasks(mode):
    task_weights(mode, 0)
    return ("a5",) if mode == "a5-control" else ("a5", "fuzzy")


def train_step(model, optimizer, task_batches, *, mode="mixed-curriculum", phase_update,
               ramp_updates=3000, microbatch=128, clip_norm=1.0):
    """Weight each task's CE+NextLat mean, then clip and update Adam once.

    The control and the zero-Fuzzy-weight boundary use the unchanged A5 step.
    Every actual mixed-stage update uses phase_update >= 1; phase zero is the
    saved/evaluated starting state, not an extra optimizer update.
    """
    weights = task_weights(mode, phase_update, ramp_updates)
    if set(task_batches) != set(active_tasks(mode)):
        raise ValueError("Batch tasks differ from curriculum mode")
    if weights["fuzzy"] == 0:
        result = base.train_step(model, optimizer, {"a5": task_batches["a5"]}, mode="a5-only",
                                 microbatch=microbatch, clip_norm=clip_norm)
        result["task_weights"] = weights
        return result
    if type(microbatch) is not int or microbatch < 1 or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("Positive microbatch and finite positive clip_norm required")
    counts, totals, devices = {}, {}, set()
    for task, (x, y) in task_batches.items():
        if (x.ndim != 2 or x.shape != y.shape or len(x) < 1 or x.shape[1] < 2
                or x.dtype != torch.long or y.dtype != torch.long or x.device != y.device):
            raise ValueError("Expected aligned nonempty int64 [B,T>=2] inputs and labels")
        counts[task] = len(x)
        devices.add(x.device)
        totals[task] = int(y.ne(IGNORE_INDEX).sum().item())
        if not totals[task] or any(not y[s:s + microbatch].ne(IGNORE_INDEX).any().item()
                                   for s in range(0, len(y), microbatch)):
            raise ValueError("Every training microbatch must have scored labels")
    if len(set(counts.values())) != 1 or len(devices) != 1:
        raise ValueError("Mixed tasks require equal example counts on one device")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values = {}
    for task in ("a5", "fuzzy"):
        inputs, labels = task_batches[task]
        sums = torch.zeros(4, dtype=torch.float64, device=inputs.device)
        for start in range(0, len(inputs), microbatch):
            x, y = inputs[start:start + microbatch], labels[start:start + microbatch]
            scored = int(y.ne(IGNORE_INDEX).sum().item())
            ce_weight, latent_weight = scored / totals[task], len(x) / len(inputs)
            with fp32_context(inputs.device):
                result = task_loss(model, x, y, task=task, latent_weight=1.0)
                loss = weights[task] * (result["ce"] * ce_weight + result["latent"] * latent_weight)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite weighted curriculum objective")
                loss.backward()
            with torch.no_grad():
                correct = result["logits"].argmax(-1).eq(y) & y.ne(IGNORE_INDEX)
                sums[0] += result["ce"].detach().double() * ce_weight
                sums[1] += result["latent"].detach().double() * latent_weight
                sums[2] += correct.sum()
                sums[3] += (correct | y.eq(IGNORE_INDEX)).all(-1).sum()
        ce, latent, correct, exact = sums.cpu().tolist()
        values[task] = {"loss": ce + latent, "ce": ce, "latent": latent, "weighted_latent": latent,
                        "token_accuracy": correct / totals[task], "scored_sequence_exact": exact / counts[task],
                        "examples": counts[task], "scored_tokens": totals[task]}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    base.fuzzy_base._sync(next(iter(devices)))
    return {"loss": sum(weights[task] * values[task]["loss"] for task in values),
            "grad_norm": base.fuzzy_base._finite_scalar(norm, "gradient norm"),
            "examples": sum(counts.values()), "tasks": values, "task_weights": weights}


def source_manifest():
    sources = dict(base.source_manifest())
    relative = "scripts/rt_nextlat_a5_fuzzy_curriculum.py"
    sources[relative] = file_sha256(ROOT / relative)
    return dict(sorted(sources.items()))


def task_offsets(phase_updates, contract):
    if type(phase_updates) is not int or phase_updates < 0:
        raise ValueError("Invalid phase update counter")
    batch = contract["batch_per_task"]
    return {"a5": contract["initial_task_offsets"]["a5"] + phase_updates * batch,
            "fuzzy": contract["initial_task_offsets"]["fuzzy"] + (
                phase_updates * batch if contract["mode"] == "mixed-curriculum" else 0)}


def cursors(phase_updates, contract):
    offsets = task_offsets(phase_updates, contract)
    result = {}
    for task, offset in offsets.items():
        epoch, position = divmod(offset, contract["streams"][task]["train_rows"])
        result[task] = {"absolute_example_offset": offset, "epoch": epoch, "position": position}
    return result


def load_parent_checkpoint(path, *, model, optimizer, config, config_path, a5_data_identity,
                           streams, runtime, hardware):
    """Restore the approved parent through the original strict checkpoint loader."""
    if file_sha256(path) != PARENT_SHA256:
        raise ValueError("Warm-start parent is not the approved retained A5 10k checkpoint")
    packet = torch.load(path, map_location="cpu", weights_only=False)
    contract = packet.get("contract", {})
    checks = {"mode": "a5-only", "schema": base.SCHEMA, "batch_per_task": 128, "microbatch": 128,
              "model_config": json_value(config), "configuration_file_sha256": file_sha256(config_path),
              "source_sha256": json_sha256(base.source_manifest()),
              "data_sha256": json_sha256({"a5": a5_data_identity}),
              "streams": {"a5": streams["a5"]}, "runtime": runtime,
              "device_capability": hardware["capability"],
              "initialization": json_value(model.initialization),
              "optimizer": {"type": "AdamW", "lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8,
                            "matrix_decay": 0.01, "vector_decay": 0.0, "clip_norm": 1.0}}
    if any(contract.get(key) != expected for key, expected in checks.items()):
        mismatches = [key for key, expected in checks.items() if contract.get(key) != expected]
        raise ValueError(f"Parent and curriculum configuration differ: {mismatches}")
    if (packet.get("completed_updates") != PARENT_UPDATES
            or contract.get("objective", {}).get("latent_weight") != 1.0
            or contract.get("objective", {}).get("task_weights") != {"a5": 1.0}):
        raise ValueError("Expected the positive-NextLat A5-only 10k parent")
    # The old schema's source/cursor/optimizer/RNG checks remain unchanged.
    return base.load_checkpoint(path, model=model, optimizer=optimizer, contract=contract)


def _validate_stage_counters(packet, contract):
    phase = packet.get("phase_updates")
    if type(phase) is not int or phase < 0:
        raise ValueError("Invalid phase update counter")
    global_update = contract["global_start_update"] + phase
    expected_offsets = task_offsets(phase, contract)
    if (packet.get("completed_updates") != global_update or packet.get("global_updates") != global_update
            or packet.get("task_offsets") != expected_offsets or packet.get("next_cursors") != cursors(phase, contract)
            or packet.get("phase_examples_seen") != {
                task: expected_offsets[task] - contract["initial_task_offsets"][task] for task in expected_offsets}
            or not base._validate_chains(packet.get("order_chains"), ("a5", "fuzzy"))
            or packet.get("task_weights") != task_weights(contract["mode"], phase, contract["ramp_updates"])):
        raise ValueError("Curriculum phase/global counters, offsets, chains or schedule differ")
    if contract["mode"] == "a5-control" and packet["order_chains"]["fuzzy"] != contract["initial_order_chains"]["fuzzy"]:
        raise ValueError("A5-only control unexpectedly advanced the Fuzzy stream")
    if phase == 0 and packet["order_chains"] != contract["initial_order_chains"]:
        raise ValueError("Phase-zero order chains differ from the warm-start parent")
    return phase, global_update


def save_checkpoint(path, *, model, optimizer, contract, phase_updates, order_chains, initialization):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    offsets = task_offsets(phase_updates, contract)
    global_update = contract["global_start_update"] + phase_updates
    packet = {"schema": SCHEMA, "contract": contract, "phase_updates": phase_updates,
              "completed_updates": global_update, "global_updates": global_update,
              "task_offsets": offsets, "next_cursors": cursors(phase_updates, contract),
              "phase_examples_seen": {task: offsets[task] - contract["initial_task_offsets"][task] for task in offsets},
              "order_chains": dict(order_chains),
              "task_weights": task_weights(contract["mode"], phase_updates, contract["ramp_updates"])}
    _validate_stage_counters(packet, contract)
    audit = base.fuzzy_base.finite_state(model, optimizer, global_update)
    packet.update(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                  optimizer_parameter_names=optimizer_names(model, optimizer), rng=rng_state(),
                  initialization=json_value(initialization), finite_state=audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(packet, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size,
            "completed_updates": global_update, "global_updates": global_update,
            "phase_updates": phase_updates, "task_offsets": offsets, **audit}


def load_checkpoint(path, *, model, optimizer, contract):
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if packet.get("schema") != SCHEMA or packet.get("contract") != contract:
        raise ValueError("Curriculum checkpoint source/data/parent/model/runtime contract differs")
    _, global_update = _validate_stage_counters(packet, contract)
    if packet.get("initialization") != json_value(model.initialization):
        raise ValueError("Curriculum original initialization identity differs")
    validate_model_state(packet["model"], model.state_dict())
    _validate_optimizer(packet, model, optimizer, global_update)
    if not isinstance(packet.get("rng"), dict) or set(packet["rng"]) != set(rng_state()):
        raise ValueError("Invalid curriculum RNG state")
    with preserve_rng():
        restore_rng(packet["rng"])
    model.load_state_dict(packet["model"], strict=True)
    optimizer.load_state_dict(packet["optimizer"])
    restore_rng(packet["rng"])
    return packet


def _tracking_axes(tracker):
    if isinstance(tracker, base.fuzzy_base._DisabledTracker):
        return
    def define():
        tracker._run.define_metric("optimizer_update")
        tracker._run.define_metric("phase_update")
        tracker._run.define_metric("train/*", step_metric="optimizer_update")
        tracker._run.define_metric("dev/*", step_metric="optimizer_update")
        tracker._run.define_metric("phase/*", step_metric="phase_update")
    tracker._call("curriculum phase axes", define)


def _validate_args(args):
    if args.mode not in MODES or args.batch_per_task != 128 or args.microbatch != 128:
        raise ValueError("This milestone keeps batch and microbatch 128 per task")
    if min(args.updates, args.ramp_updates, args.eval_every, args.eval_rows, args.full_eval_rows,
           args.eval_microbatch, args.fuzzy_eval_rows, args.fuzzy_eval_microbatch, args.log_every) < 1:
        raise ValueError("All training/evaluation counts must be positive")
    if any(value < 0 for value in args.checkpoint_steps + args.full_eval_steps + args.early_eval_steps):
        raise ValueError("Phase checkpoint/evaluation steps must be nonnegative")
    if args.wandb_mode == "disabled" and args.updates > 10:
        raise ValueError("Disabled W&B is limited to at most ten phase updates for integration checks")


def run(args):
    _validate_args(args)
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    output, a5_root, fuzzy_root = Path(args.output).resolve(), Path(args.a5_data).resolve(), Path(args.fuzzy_data).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh curriculum output directory, including for continuation")
    config = read_configuration(args.config)
    if (config["backbone"]["d_model"], config["backbone"]["n_heads"], config["backbone"]["mlp_hidden_size"]) != (128, 16, 512):
        raise ValueError("This milestone retains the D128/H16/FFN512 model")
    a5_manifest = validate_manifest(a5_root)
    a5_hash = file_sha256(a5_root / "manifest.json")
    if a5_hash != base.A5_MANIFEST_SHA256:
        raise ValueError("Expected the frozen A5 corpus")
    train_arrays = {"a5": load_split(a5_root, "train")}
    a5_dev = {role: load_split(a5_root, role) for role in ("dev", "ood_dev")}
    fuzzy_train = load_dataset(fuzzy_root, FUZZY_TASK, "train")
    fuzzy_dev = load_dataset(fuzzy_root, FUZZY_TASK, "dev")
    train_arrays["fuzzy"] = (fuzzy_train.input_ids, fuzzy_train.labels)
    if (train_arrays["a5"][0].shape[1] != 12 or a5_dev["ood_dev"][0].shape[1] != 36
            or fuzzy_train.input_ids.shape[1] != 400 or fuzzy_dev.input_ids.shape[1] != 400
            or np.any(fuzzy_train.labels == IGNORE_INDEX)):
        raise ValueError("Expected A5 T12/T36 and native densely supervised Fuzzy T400")
    streams = {task: {"train_rows": len(train_arrays[task][0]), "length": train_arrays[task][0].shape[1],
                      "order_seed": seed} for task, seed in (("a5", 5432), ("fuzzy", 2026091604))}
    data_identity = {"a5": {"manifest_sha256": a5_hash, "manifest": a5_manifest},
                     "fuzzy": {"train": fuzzy_train.manifest, "dev": fuzzy_dev.manifest,
                               "preparation_manifest_sha256": file_sha256(fuzzy_root / "manifest.json")}}
    model = build_model(config, seed=1234, predictor_seed=1235, fuzzy_seed=1236, device="cuda")
    optimizer = make_optimizer(model, lr=1e-4)
    parent = load_parent_checkpoint(args.parent_checkpoint, model=model, optimizer=optimizer,
                                   config=config, config_path=args.config, a5_data_identity=data_identity["a5"],
                                   streams=streams, runtime=runtime, hardware=hardware)
    sources = source_manifest()
    contract = {"schema": SCHEMA, "mode": args.mode, "model_config": json_value(config),
                "configuration_file_sha256": file_sha256(args.config), "source_sha256": json_sha256(sources),
                "data_sha256": json_sha256(data_identity), "initialization": json_value(model.initialization),
                "parent_checkpoint": {"sha256": PARENT_SHA256, "completed_updates": PARENT_UPDATES,
                                      "schema": base.SCHEMA},
                "parent_contract_sha256": json_sha256(parent["contract"]),
                "global_start_update": PARENT_UPDATES, "ramp_updates": args.ramp_updates,
                "initial_task_offsets": {"a5": parent["next_cursors"]["a5"]["absolute_example_offset"], "fuzzy": 0},
                "initial_order_chains": {"a5": parent["order_chains"]["a5"], "fuzzy": base.ORDER_INITIAL["fuzzy"]},
                "streams": streams, "batch_per_task": 128, "microbatch": 128,
                "runtime": runtime, "device_capability": hardware["capability"],
                "optimizer": parent["contract"]["optimizer"],
                "tracking_axes": {"train/*": "optimizer_update", "dev/*": "optimizer_update",
                                  "phase/*": "phase_update", "update": "global optimizer-update alias"},
                "objective": {"latent_weight": 1.0, "task_loss": "task-local CE + mean NextLat SmoothL1 beta1",
                              "reduction": "task means weighted before one global clipping/Adam step",
                              "fuzzy_weight": "0.5 * min(phase_update / ramp_updates, 1)" if args.mode == "mixed-curriculum" else "0",
                              "a5_weight": "1 - fuzzy_weight", "target_fuzzy_weight": 0.5 if args.mode == "mixed-curriculum" else 0.0},
                "word_order": "continue parent A5 absolute offset/order chain; start independent Fuzzy stream at zero"}
    phase = 0
    chains = dict(contract["initial_order_chains"])
    resume_identity = None
    if args.resume:
        restored = load_checkpoint(args.resume, model=model, optimizer=optimizer, contract=contract)
        phase, chains = restored["phase_updates"], restored["order_chains"]
        if phase >= args.updates:
            raise ValueError("Requested phase endpoint must exceed restored phase update")
        resume_identity = {"path": str(Path(args.resume).resolve()), "sha256": file_sha256(args.resume),
                           "phase_updates": phase, "completed_updates": PARENT_UPDATES + phase}
    output.mkdir(parents=True)
    for relative, expected in sources.items():
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if file_sha256(target) != expected:
            raise RuntimeError(f"Source changed during snapshot: {relative}")
    shutil.copy2(args.config, output / "model-config.json")
    atomic_json(output / "source-manifest.json", sources)
    atomic_json(output / "data-identity.json", data_identity)
    atomic_json(output / "invocation.json", vars(args))
    report = {"schema": SCHEMA, "status": "running", "contract": contract, "hardware": hardware,
              "parent_training_checkpoint": {"path": str(Path(args.parent_checkpoint).resolve()),
                                             "sha256": PARENT_SHA256, "completed_updates": PARENT_UPDATES},
              "parent_checkpoint": resume_identity, "start_update": PARENT_UPDATES + phase,
              "start_phase_update": phase, "requested_phase_endpoint": args.updates,
              "requested_endpoint": PARENT_UPDATES + args.updates,
              "completed_updates": PARENT_UPDATES + phase, "global_updates": PARENT_UPDATES + phase,
              "phase_updates": phase, "task_offsets": task_offsets(phase, contract),
              "initialization": json_value(model.initialization), "checkpoints": [], "evaluations": [],
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "confirmation_evaluated": False, "latent_rollout_evaluated": False}
    tracker = (base.fuzzy_base._DisabledTracker() if args.wandb_mode == "disabled" else OnlineTracker(
        project=args.wandb_project, entity="taylorbollman", output_dir=output, group=args.wandb_group,
        name=args.wandb_run_name or f"d128-{args.mode}", preserve_state=preserve_rng))
    report["wandb"] = tracker.record
    atomic_json(output / "report.json", report)
    orders = {task: WordOrder(value["train_rows"], value["order_seed"]) for task, value in streams.items()}
    checkpoint_steps = set(args.checkpoint_steps) | {0, args.updates}
    full_steps = set(args.full_eval_steps) | {0, args.updates}
    early_steps = set(args.early_eval_steps)
    checkpointed, stop_requested, handlers = {}, [], {}
    started, train_seconds = time.perf_counter(), 0.0

    def update_report():
        report.update(completed_updates=PARENT_UPDATES + phase, global_updates=PARENT_UPDATES + phase,
                      phase_updates=phase, task_offsets=task_offsets(phase, contract), next_cursors=cursors(phase, contract),
                      order_chains=dict(chains), task_weights=task_weights(args.mode, phase, args.ramp_updates),
                      train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - started)

    def checkpoint():
        if phase not in checkpointed:
            value = save_checkpoint(output / f"checkpoints/phase-{phase:06d}.pt", model=model, optimizer=optimizer,
                                    contract=contract, phase_updates=phase, order_chains=chains,
                                    initialization=model.initialization)
            checkpointed[phase] = value
            report["checkpoints"].append(value)
        return {key: checkpointed[phase][key] for key in ("path", "sha256", "completed_updates", "global_updates", "phase_updates")}

    def record_evaluation(result, task, role, scope, identity):
        result.update(update=PARENT_UPDATES + phase, global_update=PARENT_UPDATES + phase,
                      phase_update=phase, task=task, role=role, scope=scope, checkpoint=identity)
        report["evaluations"].append(result)
        metrics = base.evaluation_metrics(result)
        phase_metrics = {f"phase/{name}": value for name, value in metrics.items()}
        tracker.log({"update": PARENT_UPDATES + phase, "optimizer_update": PARENT_UPDATES + phase,
                     "phase_update": phase, **metrics, **phase_metrics})
        print(json.dumps({"event": "evaluation", **result}, allow_nan=False), flush=True)

    def evaluate_checkpoint(force_full=False):
        identity = checkpoint()
        rows = args.full_eval_rows if force_full or phase in full_steps else args.eval_rows
        for role, arrays in a5_dev.items():
            result = base.evaluate_a5(model, *arrays, microbatch=args.eval_microbatch, limit=rows)
            record_evaluation(result, "a5", role, "full" if result["evaluated_rows"] >= 102400 else "subset", identity)
        result = base.fuzzy_base.evaluate(model, fuzzy_dev, microbatch=args.fuzzy_eval_microbatch, limit=args.fuzzy_eval_rows)
        record_evaluation(result, "fuzzy", "dev", "full" if result["evaluated_rows"] == len(fuzzy_dev) else "subset", identity)
        update_report()
        atomic_json(output / "report.json", report)

    def request_stop(number, frame):
        stop_requested.append(signal.Signals(number).name)

    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, request_stop)
    history = (output / "history.jsonl").open("x", buffering=1)
    succeeded = False
    try:
        tracker.start(contract)
        _tracking_axes(tracker)
        print(json.dumps({"event": "started", "mode": args.mode, "start_phase_update": phase,
                          "global_update": PARENT_UPDATES + phase, "phase_endpoint": args.updates,
                          "wandb": tracker.record.get("run_url")}), flush=True)
        # Evaluation preserves RNG. Both arms have a real, same-state full
        # phase-zero A5/Fuzzy baseline before consuming a new training example.
        evaluate_checkpoint(force_full=True)
        window = []
        while phase < args.updates:
            if stop_requested or (args.stop_file and Path(args.stop_file).exists()):
                report["stop_reason"] = stop_requested or ["stop_file"]
                break
            base.fuzzy_base._sync("cuda")
            begin = time.perf_counter()
            offsets = task_offsets(phase, contract)
            batches, next_chains = {}, dict(chains)
            for task in active_tasks(args.mode):
                indices = orders[task].indices(offsets[task], 128)
                next_chains[task] = hashlib.sha256(bytes.fromhex(chains[task]) + indices.astype("<i8").tobytes()).hexdigest()
                x, y = batch_tensors(*train_arrays[task], indices, "cuda")
                batches[task] = (encode_inputs(x, task=task), y)
            values = train_step(model, optimizer, batches, mode=args.mode, phase_update=phase + 1,
                                ramp_updates=args.ramp_updates, microbatch=128)
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            phase, chains = phase + 1, next_chains
            row = {"update": PARENT_UPDATES + phase, "global_update": PARENT_UPDATES + phase,
                   "phase_update": phase, "seconds": elapsed, "task_offsets": task_offsets(phase, contract),
                   "order_chains": dict(chains), **values}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            window.append(row)
            if phase % args.log_every == 0 or phase == args.updates or phase in early_steps:
                logged = {"update": PARENT_UPDATES + phase, "optimizer_update": PARENT_UPDATES + phase,
                          "phase_update": phase,
                          "train/seconds_per_update": sum(r["seconds"] for r in window) / len(window),
                          "train/loss": sum(r["loss"] for r in window) / len(window),
                          "train/grad_norm": sum(r["grad_norm"] for r in window) / len(window),
                          "train/fuzzy_weight": values["task_weights"]["fuzzy"],
                          "train/a5_weight": values["task_weights"]["a5"],
                          "train/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "train/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
                for task in active_tasks(args.mode):
                    for key in base.TASK_METRICS:
                        logged[f"train/{task}/{key}"] = sum(r["tasks"][task][key] for r in window) / len(window)
                    logged[f"train/{task}/absolute_examples_seen"] = task_offsets(phase, contract)[task]
                logged.update({f"phase/{name}": value for name, value in list(logged.items()) if name.startswith("train/")})
                tracker.log(logged)
                print(json.dumps({"event": "train", **logged}), flush=True)
                window.clear()
            if phase in checkpoint_steps:
                checkpoint()
            if phase % args.eval_every == 0 or phase in checkpoint_steps or phase in full_steps or phase in early_steps:
                evaluate_checkpoint()
            update_report()
            if phase % args.log_every == 0 or phase in checkpoint_steps:
                atomic_json(output / "report.json", report)
        checkpoint()
        endpoint_full = all(any(row["phase_update"] == phase and row["task"] == "a5" and row["role"] == role
                                and row["evaluated_rows"] >= min(args.full_eval_rows, len(a5_dev[role][0]))
                                for row in report["evaluations"]) for role in a5_dev)
        if not endpoint_full:
            evaluate_checkpoint(force_full=True)
        if source_manifest() != sources or file_sha256(args.config) != contract["configuration_file_sha256"]:
            raise RuntimeError("Curriculum sources or configuration changed during training")
        report["status"] = "complete" if phase == args.updates else "stopped"
        update_report()
        summary = {"status": report["status"], "phase_updates": phase, "global_updates": PARENT_UPDATES + phase,
                   "confirmation_evaluated": False}
        for result in report["evaluations"]:
            if result["phase_update"] == phase:
                summary.update(base.evaluation_metrics(result))
        tracker.summary(summary)
        succeeded = True
    except BaseException as error:
        update_report()
        report.update(status="failed", error_type=type(error).__name__)
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
    print(json.dumps({"event": "finished", "status": report["status"], "phase_update": phase,
                      "global_update": PARENT_UPDATES + phase, "wandb": tracker.record.get("run_url")}), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--parent-checkpoint", type=Path, default=PARENT_PATH)
    p.add_argument("--a5-data", type=Path, required=True)
    p.add_argument("--fuzzy-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=10000, help="absolute phase-update endpoint, not global updates")
    p.add_argument("--batch-per-task", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=128)
    p.add_argument("--ramp-updates", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--early-eval-steps", type=int, nargs="+", default=[50, 100])
    p.add_argument("--eval-rows", type=int, default=4096)
    p.add_argument("--full-eval-rows", type=int, default=102400)
    p.add_argument("--eval-microbatch", type=int, default=1024)
    p.add_argument("--fuzzy-eval-rows", type=int, default=1280)
    p.add_argument("--fuzzy-eval-microbatch", type=int, default=128)
    p.add_argument("--full-eval-steps", type=int, nargs="+", default=[0, 1000, 3000, 5000, 10000])
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[0, 50, 100, 1000, 3000, 5000, 10000])
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    p.add_argument("--wandb-project", default="rt-nextlat-fuzzy-a5")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-run-name")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
