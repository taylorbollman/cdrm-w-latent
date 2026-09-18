#!/usr/bin/env python3
"""Repeat the original restricted-first A5/NextLat run in protected BF16 mixed.

The historical training loop is retained with explicit precision functions,
independent checkpoint/source identity, and a graceful caller-owned stop file.
All existing historical source files remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_bf16 import (
    PRECISION_CONTRACT, build_model, configure_runtime, evaluate_arrays,
    evaluate_diagnostics, train_step,
)
from scripts.rt_a5_depth_order_train import source_manifest as original_source_manifest
from scripts.rt_a5_nextlat_train import (
    TRAIN_METRICS, _valid_order_chain, _validate_optimizer,
    make_contract as baseline_make_contract,
)
from scripts.rt_a5_train import (
    WordOrder, atomic_json, batch_tensors, cpu_tree,
    file_sha256, flatten_eval, json_sha256, json_value, optimizer_names,
    preserve_rng, restore_rng, rng_state, validate_model_state,
)
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-bf16-training-v1"
VARIANT_ARCHITECTURES = {"rt_window2_first": "rt"}


def source_manifest():
    """Exact frozen FP32 closure plus the two new executed precision sources."""
    sources = dict(original_source_manifest())
    for name in ("scripts/rt_a5_bf16.py", "scripts/rt_a5_bf16_train.py"):
        sources[name] = file_sha256(ROOT / name)
    return dict(sorted(sources.items()))


def stop_request(path, completed):
    if path is None or not Path(path).is_file():
        return None
    target = Path(path).resolve()
    return {"reason": "user_stop_file", "path": str(target),
            "sha256": file_sha256(target), "observed_after_update": completed}


def save_checkpoint(path, *, model, optimizer, contract, completed, order_chain,
                    initialization):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint exists: {path}")
    if type(completed) is not int or completed < 0 or not _valid_order_chain(order_chain):
        raise ValueError("Invalid checkpoint update or order chain")
    packet = {
        "schema": SCHEMA, "contract": contract, "completed_updates": completed,
        "examples_seen": completed * contract["batch_size"], "order_chain": order_chain,
        "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
        "optimizer_parameter_names": optimizer_names(model, optimizer),
        "rng": rng_state(), "initialization": initialization,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(packet, temporary)
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "bytes": path.stat().st_size, "completed_updates": completed,
            "examples_seen": packet["examples_seen"]}


def load_checkpoint(path, *, model, optimizer, contract):
    """Restore only this experiment's complete, strictly matching checkpoints."""
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if packet.get("schema") != SCHEMA or packet.get("contract") != contract:
        raise ValueError("BF16 checkpoint source/data/model/runtime/objective contract differs")
    completed = packet.get("completed_updates")
    if (type(completed) is not int or completed < 0
            or packet.get("examples_seen") != completed * contract["batch_size"]
            or not _valid_order_chain(packet.get("order_chain"))):
        raise ValueError("Checkpoint update/word offset/order chain is inconsistent")
    if packet.get("initialization") != json_value(model.nextlat_initialization):
        raise ValueError("BF16 initialization differs")
    validate_model_state(packet["model"], model.state_dict())
    _validate_optimizer(packet, model, optimizer, completed)
    # Check RNG restoration before mutating model/Adam; then restore it last.
    if not isinstance(packet.get("rng"), dict) or set(packet["rng"]) != set(rng_state()):
        raise ValueError("Invalid checkpoint RNG state")
    with preserve_rng():
        restore_rng(packet["rng"])
    model.load_state_dict(packet["model"], strict=True)
    optimizer.load_state_dict(packet["optimizer"])
    restore_rng(packet["rng"])
    return packet


def make_contract(args, *, model, sources, data_root, train_x, hardware):
    values = dict(vars(args))
    values["architecture"] = VARIANT_ARCHITECTURES[args.variant]
    resolved = argparse.Namespace(**values)
    contract = baseline_make_contract(
        resolved, model=model, sources=sources, data_root=data_root,
        train_x=train_x, hardware=hardware)
    contract.update(
        schema=SCHEMA,
        variant=args.variant,
        experiment_config=json_value(model.experiment_config),
        training_step="scripts.rt_a5_bf16.train_step",
        evaluation="scripts.rt_a5_bf16.evaluate_arrays; unchanged A5Metrics on FP32 logits",
        one_step_diagnostics="scripts.rt_a5_bf16.evaluate_diagnostics",
        precision="bf16_mixed", precision_contract=json_value(PRECISION_CONTRACT),
    )
    return contract


def run(args):
    if args.variant not in VARIANT_ARCHITECTURES:
        raise ValueError("Unsupported BF16 repeat variant")
    args.architecture = VARIANT_ARCHITECTURES[args.variant]
    hardware = require_cuda_container()
    configure_runtime()
    if min(args.updates, args.batch_size, args.eval_every, args.eval_rows,
           args.full_eval_rows, args.log_every, args.diagnostic_rows) < 1:
        raise ValueError("Update/batch/evaluation/log counts must be positive")
    if not math.isfinite(args.latent_weight) or args.latent_weight <= 0:
        raise ValueError("Joint training requires finite positive latent weight")
    if min(args.seed, args.predictor_seed, args.data_order_seed) < 0:
        raise ValueError("Seeds must be nonnegative")
    if any(step < 1 for step in args.checkpoint_steps):
        raise ValueError("Trained checkpoint steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; resume writes a new continuation directory")
    data_root = Path(args.data_dir).resolve()
    validate_manifest(data_root)
    train_x, train_y = load_split(data_root, "train")
    dev = {role: load_split(data_root, role) for role in ("dev", "ood_dev")}
    sources = source_manifest()
    model_kwargs = {"width": args.width, "seed": args.seed,
                    "predictor_seed": args.predictor_seed, "device": "cuda"}
    if args.predictor_hidden_width is not None:
        model_kwargs["predictor_hidden_width"] = args.predictor_hidden_width
    model = build_model(args.variant, **model_kwargs)
    optimizer = make_optimizer(model)
    initial = json_value(model.nextlat_initialization)
    contract = make_contract(args, model=model, sources=sources, data_root=data_root,
                             train_x=train_x, hardware=hardware)
    completed, order_chain = 0, hashlib.sha256(b"rt-a5-word-order-v1").hexdigest()
    parent = None
    if args.resume:
        packet = load_checkpoint(args.resume, model=model, optimizer=optimizer, contract=contract)
        completed, order_chain = packet["completed_updates"], packet["order_chain"]
        initial = packet["initialization"]
        parent = {"path": str(Path(args.resume).resolve()), "sha256": file_sha256(args.resume)}
        if completed >= args.updates:
            raise ValueError("Resume endpoint must exceed completed updates")
    output.mkdir(parents=True)
    for relative in sources:
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    report = {
        "schema": SCHEMA, "status": "running", "contract": contract,
        "hardware": hardware, "source_files": sources, "initialization": initial,
        "parent_checkpoint": parent, "start_update": completed, "endpoint": args.updates,
        "checkpoint_selection": "fixed matched update budget or explicit user stop; confirmation not evaluated",
        "stop_request": None, "requested_endpoint_reached": False,
        "evaluations": [], "one_step_diagnostics": [], "checkpoints": [],
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
    }
    atomic_json(output / "config.json", vars(args))
    atomic_json(output / "report.json", report)
    tracker = OnlineTracker(
        project=args.wandb_project, entity="taylorbollman", output_dir=output,
        group=args.wandb_group,
        name=args.wandb_run_name or (
            f"{args.variant}-nextlat-bf16-d{args.width}-seed{args.seed}"),
        preserve_state=preserve_rng)
    history = (output / "history.jsonl").open("x", buffering=1)
    started = time.perf_counter()
    train_seconds = 0.0
    try:
        tracker.start(contract)
        report["wandb"] = tracker.record
        atomic_json(output / "report.json", report)
        print(json.dumps({"event": "started", "wandb": tracker.record["run_url"],
                          "architecture": args.architecture, "nextlat": True,
                          "experiment": model.experiment_config,
                          "parameter_count": sum(p.numel() for p in model.parameters())}), flush=True)
        if completed == 0:
            report["checkpoints"].append(save_checkpoint(
                output / "checkpoints/step-000000.pt", model=model, optimizer=optimizer,
                contract=contract, completed=0, order_chain=order_chain, initialization=initial))
            atomic_json(output / "report.json", report)
        order = WordOrder(len(train_x), args.data_order_seed)
        checkpoint_steps = set(args.checkpoint_steps) | {args.updates}
        window = []
        for update in range(completed + 1, args.updates + 1):
            begin = time.perf_counter()
            indices = order.indices((update - 1) * args.batch_size, args.batch_size)
            order_chain = hashlib.sha256(bytes.fromhex(order_chain)
                                         + indices.astype("<i8").tobytes()).hexdigest()
            x, y = batch_tensors(train_x, train_y, indices, "cuda")
            should_log = update % args.log_every == 0 or update == args.updates
            values = train_step(model, optimizer, x, y, latent_weight=args.latent_weight,
                                diagnostics=should_log)
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            completed = update
            row = {"update": update, "examples_seen": update * args.batch_size,
                   "seconds": elapsed, "order_chain": order_chain, **values}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            window.append(row)
            if should_log:
                logged = {"update": update, "train/examples_seen": update * args.batch_size,
                          "train/seconds_per_update": sum(v["seconds"] for v in window) / len(window)}
                for key in TRAIN_METRICS:
                    logged[f"train/{key}"] = sum(v[key] for v in window) / len(window)
                logged.update({f"train/one_step_diagnostics/{key}": value
                               for key, value in values.items() if key not in TRAIN_METRICS})
                tracker.log(logged)
                print(json.dumps({"event": "train", **logged}), flush=True)
                window.clear()
            requested_stop = stop_request(args.stop_file, completed)
            if update % args.eval_every == 0 or update in checkpoint_steps or requested_stop:
                limit = args.full_eval_rows if update in checkpoint_steps or requested_stop else args.eval_rows
                for role, (dx, dy) in dev.items():
                    # Wrapper.forward is backbone-only: no predictor evaluation here.
                    result = evaluate_arrays(model, dx, dy, batch_size=args.batch_size, limit=limit)
                    result.update(role=role, update=update, route="backbone_only")
                    report["evaluations"].append(result)
                    tracker.log({"update": update, **flatten_eval(result, role)})
                    diagnostic = evaluate_diagnostics(
                        model, dx, dy, rows=args.diagnostic_rows, latent_weight=args.latent_weight)
                    diagnostic.update(role=role, update=update)
                    report["one_step_diagnostics"].append(diagnostic)
                    diagnostic_scalars = {key: value for key, value in diagnostic.items()
                                          if isinstance(value, (float, int))}
                    diagnostic_scalars.update(diagnostic["diagnostics"])
                    tracker.log({"update": update, **{
                        f"dev/{role}/one_step_diagnostics/{key}": value
                        for key, value in diagnostic_scalars.items()}})
                    print(json.dumps({"event": "evaluation", **result}), flush=True)
                    print(json.dumps({"event": "one_step_diagnostics", **diagnostic}), flush=True)
            if update in checkpoint_steps or requested_stop:
                report["checkpoints"].append(save_checkpoint(
                    output / f"checkpoints/step-{update:06d}.pt", model=model,
                    optimizer=optimizer, contract=contract, completed=update,
                    order_chain=order_chain, initialization=initial))
                report.update(completed_updates=completed, order_chain=order_chain,
                              train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - started)
                atomic_json(output / "report.json", report)
            if requested_stop:
                report["stop_request"] = requested_stop
                break
        report.update(status="stopped" if report["stop_request"] else "complete",
                      requested_endpoint_reached=completed >= args.updates,
                      completed_updates=completed, order_chain=order_chain,
                      train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - started)
        tracker.summary({"completed_updates": completed, "confirmation_evaluated": False,
                         "latent_rollout_evaluated": False, "train_seconds": train_seconds,
                         "order_chain": order_chain})
        tracker.finish(succeeded=True)
    except BaseException as error:
        report.update(status="failed", completed_updates=completed, error_type=type(error).__name__,
                      elapsed_seconds=time.perf_counter() - started)
        try:
            tracker.finish(succeeded=False)
        except Exception:
            pass
        raise
    finally:
        history.close()
        report["wandb"] = tracker.record
        atomic_json(output / "report.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", choices=tuple(VARIANT_ARCHITECTURES), required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--predictor-seed", type=int, default=1235)
    p.add_argument("--predictor-hidden-width", type=int)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--data-order-seed", type=int, default=1234)
    p.add_argument("--updates", type=int, default=80000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-rows", type=int, default=4096)
    p.add_argument("--full-eval-rows", type=int, default=102400)
    p.add_argument("--diagnostic-rows", type=int, default=1024)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[1000, 5000, 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000])
    p.add_argument("--resume")
    p.add_argument("--stop-file")
    p.add_argument("--wandb-project", default="rt-a5-state-tracking")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-run-name")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
