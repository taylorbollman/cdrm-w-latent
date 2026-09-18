#!/usr/bin/env python3
"""Fresh six-layer permanent-value bypass, with constant or linear coefficient.

Original data/order, optimizer, NextLat update and evaluators are unchanged.
A caller-selected stop file ends gracefully after a completed update, retaining
that update's full development evaluation and resumable model/Adam/RNG state.
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
from scripts.rt_a5_common import configure_fp32_runtime, make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_six_layer_value import SCHEDULES, build_model, coefficient, set_update
from scripts.rt_a5_embedding_injection_train import source_manifest as injection_source_manifest
from scripts.rt_a5_nextlat_train import (
    TRAIN_METRICS, _valid_order_chain, _validate_optimizer, evaluate_diagnostics,
    make_contract as baseline_make_contract,
    train_step,
)
from scripts.rt_a5_train import (
    WordOrder, atomic_json, batch_tensors, cpu_tree, evaluate_arrays,
    file_sha256, flatten_eval, json_sha256, json_value, optimizer_names,
    preserve_rng, restore_rng, rng_state, validate_model_state,
)
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-six-layer-value-training-v1"
VARIANT_ARCHITECTURES = {"constant": "rt", "linear": "rt"}


def source_manifest():
    """Preserve the historical 62 sources and add only this factory/driver/config."""
    sources = dict(injection_source_manifest())
    additions = [ROOT / "scripts" / name for name in
                 ("rt_a5_six_layer_value.py", "rt_a5_six_layer_value_train.py")]
    additions += list((ROOT / "configs/rt_a5_six_layer_value").glob("*.json"))
    sources.update({str(path.relative_to(ROOT)): file_sha256(path) for path in additions})
    return dict(sorted(sources.items()))


def stop_request(path, completed):
    """A retained caller-created file is the sole graceful-stop signal."""
    if path is None or not Path(path).is_file():
        return None
    target = Path(path).resolve()
    return {"reason": "user_stop_file", "path": str(target),
            "sha256": file_sha256(target), "observed_after_update": completed}


def validate_schedule_state(model, completed):
    if (model.backbone.config.n_layers != 6 or model.backbone.injection_variant != "value"
            or model.experiment_config["schedule"] != SCHEDULES[model.experiment_config["variant"]]
            or model.schedule_update != completed
            or model.backbone.injection_coefficient != coefficient(completed, model.experiment_config["variant"])):
        raise ValueError("Model schedule position/coefficient differs from completed updates")


def save_checkpoint(path, *, model, optimizer, contract, completed, order_chain,
                    initialization):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint exists: {path}")
    if type(completed) is not int or completed < 0 or not _valid_order_chain(order_chain):
        raise ValueError("Invalid checkpoint update or order chain")
    validate_schedule_state(model, completed)
    packet = {
        "schema": SCHEMA, "contract": contract, "completed_updates": completed,
        "examples_seen": completed * contract["batch_size"], "order_chain": order_chain,
        "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
        "optimizer_parameter_names": optimizer_names(model, optimizer),
        "rng": rng_state(), "initialization": initialization,
        "schedule": json_value(SCHEDULES[model.experiment_config["variant"]]), "schedule_update": completed,
        "injection_coefficient": coefficient(completed, model.experiment_config["variant"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(packet, temporary)
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path),
            "bytes": path.stat().st_size, "completed_updates": completed,
            "examples_seen": packet["examples_seen"], "injection_coefficient": coefficient(completed, model.experiment_config["variant"])}


def load_checkpoint(path, *, model, optimizer, contract):
    """Restore only this experiment's complete, strictly matching checkpoints."""
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if packet.get("schema") != SCHEMA or packet.get("contract") != contract:
        raise ValueError("Six-layer value checkpoint source/data/model/runtime/objective contract differs")
    completed = packet.get("completed_updates")
    if (type(completed) is not int or completed < 0
            or packet.get("examples_seen") != completed * contract["batch_size"]
            or not _valid_order_chain(packet.get("order_chain"))):
        raise ValueError("Checkpoint update/word offset/order chain is inconsistent")
    if (packet.get("schedule") != SCHEDULES[model.experiment_config["variant"]] or packet.get("schedule_update") != completed
            or packet.get("injection_coefficient") != coefficient(completed, model.experiment_config["variant"])):
        raise ValueError("Checkpoint value schedule or current coefficient differs")
    if packet.get("initialization") != json_value(model.nextlat_initialization):
        raise ValueError("Six-layer value initialization differs")
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
    set_update(model, completed)
    return packet


def make_contract(args, *, model, sources, data_root, train_x, hardware):
    if args.variant != model.experiment_config["variant"]:
        raise ValueError("Requested variant differs from the constructed value model")
    values = dict(vars(args))
    values["architecture"] = VARIANT_ARCHITECTURES[args.variant]
    resolved = argparse.Namespace(**values)
    contract = baseline_make_contract(
        resolved, model=model, sources=sources, data_root=data_root,
        train_x=train_x, hardware=hardware)
    contract.update(
        schema=SCHEMA,
        variant=args.variant,
        n_layers=6,
        projection_seed=args.projection_seed,
        injection_routing="value",
        injection_schedule=json_value(SCHEDULES[model.experiment_config["variant"]]),
        experiment_config=json_value(model.experiment_config),
        training_step="scripts.rt_a5_nextlat_train.train_step (same function object)",
        evaluation="scripts.rt_a5_train.evaluate_arrays (same function object)",
        one_step_diagnostics="scripts.rt_a5_nextlat_train.evaluate_diagnostics (same function object)",
    )
    return contract


def run(args):
    if args.variant not in VARIANT_ARCHITECTURES:
        raise ValueError("Unsupported Six-layer value variant")
    args.architecture = VARIANT_ARCHITECTURES[args.variant]
    hardware = require_cuda_container()
    configure_fp32_runtime()
    if min(args.updates, args.batch_size, args.eval_every, args.eval_rows,
           args.full_eval_rows, args.log_every, args.diagnostic_rows) < 1:
        raise ValueError("Update/batch/evaluation/log counts must be positive")
    if not math.isfinite(args.latent_weight) or args.latent_weight <= 0:
        raise ValueError("Joint training requires finite positive latent weight")
    if min(args.seed, args.predictor_seed, args.projection_seed, args.data_order_seed) < 0:
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
                    "predictor_seed": args.predictor_seed, "device": "cuda",
                    "projection_seed": args.projection_seed, "variant": args.variant}
    if args.predictor_hidden_width is not None:
        model_kwargs["predictor_hidden_width"] = args.predictor_hidden_width
    model = build_model(**model_kwargs)
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
        "checkpoint_selection": "caller-selected endpoint; constant 0.01 or linear 0.01 at 20k; user may stop after completed updates; confirmation not evaluated",
        "schedule": json_value(SCHEDULES[model.experiment_config["variant"]]), "injection_coefficient": coefficient(completed, model.experiment_config["variant"]),
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
            f"{args.variant}-nextlat-d{args.width}-seed{args.seed}"),
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
        stopping = None
        for update in range(completed + 1, args.updates + 1):
            begin = time.perf_counter()
            indices = order.indices((update - 1) * args.batch_size, args.batch_size)
            order_chain = hashlib.sha256(bytes.fromhex(order_chain)
                                         + indices.astype("<i8").tobytes()).hexdigest()
            x, y = batch_tensors(train_x, train_y, indices, "cuda")
            should_log = update % args.log_every == 0 or update == args.updates
            current_coefficient = set_update(model, update)
            values = train_step(model, optimizer, x, y, latent_weight=args.latent_weight,
                                diagnostics=should_log)
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            completed = update
            stopping = stop_request(args.stop_file, completed)
            should_log = should_log or stopping is not None
            row = {"update": update, "examples_seen": update * args.batch_size,
                   "seconds": elapsed, "order_chain": order_chain,
                   "injection_coefficient": current_coefficient, **values}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            window.append(row)
            if should_log:
                logged = {"update": update, "train/injection_coefficient": current_coefficient,
                          "train/examples_seen": update * args.batch_size,
                          "train/seconds_per_update": sum(v["seconds"] for v in window) / len(window)}
                for key in TRAIN_METRICS:
                    logged[f"train/{key}"] = sum(v[key] for v in window) / len(window)
                logged.update({f"train/one_step_diagnostics/{key}": value
                               for key, value in values.items() if key not in TRAIN_METRICS})
                tracker.log(logged)
                print(json.dumps({"event": "train", **logged}), flush=True)
                window.clear()
            if update % args.eval_every == 0 or update in checkpoint_steps or stopping is not None:
                limit = args.full_eval_rows if update in checkpoint_steps or stopping is not None else args.eval_rows
                for role, (dx, dy) in dev.items():
                    # Wrapper.forward is backbone-only: no predictor evaluation here.
                    result = evaluate_arrays(model, dx, dy, batch_size=args.batch_size, limit=limit)
                    result.update(role=role, update=update, route="backbone_only",
                                  injection_coefficient=current_coefficient)
                    report["evaluations"].append(result)
                    tracker.log({"update": update, "schedule/injection_coefficient": current_coefficient,
                                 **flatten_eval(result, role)})
                    diagnostic = evaluate_diagnostics(
                        model, dx, dy, rows=args.diagnostic_rows, latent_weight=args.latent_weight)
                    diagnostic.update(role=role, update=update, injection_coefficient=current_coefficient)
                    report["one_step_diagnostics"].append(diagnostic)
                    diagnostic_scalars = {key: value for key, value in diagnostic.items()
                                          if isinstance(value, (float, int))}
                    diagnostic_scalars.update(diagnostic["diagnostics"])
                    tracker.log({"update": update, **{
                        f"dev/{role}/one_step_diagnostics/{key}": value
                        for key, value in diagnostic_scalars.items()}})
                    print(json.dumps({"event": "evaluation", **result}), flush=True)
                    print(json.dumps({"event": "one_step_diagnostics", **diagnostic}), flush=True)
            if update in checkpoint_steps or stopping is not None:
                report["checkpoints"].append(save_checkpoint(
                    output / f"checkpoints/step-{update:06d}.pt", model=model,
                    optimizer=optimizer, contract=contract, completed=update,
                    order_chain=order_chain, initialization=initial))
                report.update(completed_updates=completed, order_chain=order_chain,
                              injection_coefficient=current_coefficient,
                              train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - started)
                atomic_json(output / "report.json", report)
            if stopping is not None:
                break
        reached_endpoint = completed == args.updates
        report.update(status="complete" if reached_endpoint else "stopped",
                      stop_request=stopping, requested_endpoint_reached=reached_endpoint,
                      injection_coefficient=coefficient(completed, model.experiment_config["variant"]),
                      completed_updates=completed, order_chain=order_chain,
                      train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - started)
        tracker.summary({"completed_updates": completed, "requested_endpoint": args.updates,
                         "requested_endpoint_reached": reached_endpoint,
                         "stopped_by_user": stopping is not None,
                         "injection_coefficient": coefficient(completed, model.experiment_config["variant"]), "confirmation_evaluated": False,
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
    p.add_argument("--variant", choices=tuple(VARIANT_ARCHITECTURES), default="constant")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--projection-seed", type=int, default=1236)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--predictor-seed", type=int, default=1235)
    p.add_argument("--predictor-hidden-width", type=int)
    p.add_argument("--latent-weight", type=float, default=1.0)
    p.add_argument("--data-order-seed", type=int, default=1234)
    p.add_argument("--updates", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-rows", type=int, default=4096)
    p.add_argument("--full-eval-rows", type=int, default=102400)
    p.add_argument("--diagnostic-rows", type=int, default=1024)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000])
    p.add_argument("--resume")
    p.add_argument("--stop-file", help="Caller-created file requesting a saved, evaluated stop after a completed update")
    p.add_argument("--wandb-project", default="rt-a5-state-tracking")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-run-name")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
