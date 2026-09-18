#!/usr/bin/env python3
"""Joint FP32 A5/NextLat training; evaluation uses the backbone only."""
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
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_nextlat import build_nextlat_model, nextlat_objective
from scripts.rt_a5_train import (
    WordOrder, atomic_json, batch_tensors, cpu_tree, evaluate_arrays,
    file_sha256, flatten_eval, json_sha256, json_value, optimizer_names,
    preserve_rng, restore_rng, rng_state, source_manifest as base_source_manifest,
    validate_model_state,
)
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-nextlat-training-v1"
TRAIN_METRICS = ("loss", "state_loss", "latent_loss", "weighted_latent_loss",
                 "token_accuracy", "whole_word_exact", "grad_norm")


def source_manifest():
    """Cover new executed sources without changing historical resume identity."""
    sources = dict(base_source_manifest())
    additions = [ROOT / "scripts" / name for name in
                 ("rt_a5_nextlat.py", "rt_a5_nextlat_train.py")]
    additions += list((ROOT / "configs/rt_a5_nextlat").glob("*.json"))
    sources.update({str(path.relative_to(ROOT)): file_sha256(path) for path in additions})
    return dict(sorted(sources.items()))


def _scalar(value, name):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar diagnostic: {name}")
        value = value.detach().item()
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite NextLat metric: {name}")
    return value


def objective_metrics(result, latent_weight):
    values = {key: _scalar(result[key], key) for key in
              ("loss", "state_loss", "latent_loss")}
    values["weighted_latent_loss"] = latent_weight * values["latent_loss"]
    values["diagnostics"] = {
        key: _scalar(value, key) for key, value in result["diagnostics"].items()}
    return values


def train_step(model, optimizer, inputs, labels, *, latent_weight=1.0,
               clip_norm=1.0, diagnostics=False):
    """One combined backward and one clip/Adam update for all shared parameters."""
    if not math.isfinite(latent_weight) or latent_weight < 0:
        raise ValueError("latent_weight must be finite and nonnegative")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with fp32_context(inputs.device):
        result = nextlat_objective(model, inputs, labels,
                                   latent_weight=latent_weight, diagnostics=diagnostics)
        values = objective_metrics(result, latent_weight)
        result["loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm,
                                             error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        correct = result["logits"].argmax(-1).eq(labels)
        values.update(token_accuracy=correct.float().mean().item(),
                      whole_word_exact=correct.all(dim=1).float().mean().item(),
                      grad_norm=grad_norm.item())
    values.update(values.pop("diagnostics"))
    return values


def evaluate_diagnostics(model, inputs, labels, *, rows=1024, device="cuda",
                         latent_weight=1.0):
    """A bounded teacher-conditioned loss probe; never carry predicted latents."""
    rows = min(rows, len(inputs))
    if rows < 1:
        raise ValueError("Diagnostic evaluation needs positive rows")
    training = model.training
    model.eval()
    try:
        with torch.no_grad(), fp32_context(device):
            x, y = batch_tensors(inputs, labels, slice(0, rows), device)
            result = nextlat_objective(model, x, y, latent_weight=latent_weight,
                                       diagnostics=True)
            values = objective_metrics(result, latent_weight)
        return {"rows": rows, "length": int(x.shape[1]),
                "route": "teacher_conditioned_one_step_diagnostics", **values}
    finally:
        model.train(training)


def _valid_order_chain(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


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


def _validate_optimizer(packet, model, optimizer, completed):
    if packet.get("optimizer_parameter_names") != optimizer_names(model, optimizer):
        raise ValueError("Optimizer parameter mapping differs")
    saved = packet["optimizer"]
    expected = optimizer.state_dict()
    if len(saved["param_groups"]) != len(expected["param_groups"]):
        raise ValueError("Optimizer group count differs")
    id_to_parameter = {}
    for actual, reference, live in zip(saved["param_groups"], expected["param_groups"],
                                        optimizer.param_groups):
        if actual != reference:
            raise ValueError("Optimizer options or parameter IDs differ")
        id_to_parameter.update(zip(actual["params"], live["params"]))
    if not set(saved["state"]).issubset(id_to_parameter):
        raise ValueError("Optimizer has unexpected parameter state")
    for parameter_id, state in saved["state"].items():
        parameter = id_to_parameter[parameter_id]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Invalid Adam state keys")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid FP32 optimizer state: {key}")
        if (state["exp_avg_sq"] < 0).any():
            raise ValueError("Negative Adam second moment")
        step = state["step"]
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or step.dtype != torch.float32 or not torch.isfinite(step).all()
                or step.item() < 1 or step.item() > completed
                or step.item() != int(step.item())):
            raise ValueError("Invalid optimizer update counter")
    # Every trainable tensor participates in the standard positive-weight recipe.
    # Weight-zero integration checks intentionally leave the predictor dormant.
    positive_auxiliary = packet["contract"].get("objective", {}).get("latent_weight", 0) > 0
    if completed and positive_auxiliary and set(saved["state"]) != set(id_to_parameter):
        raise ValueError("Missing optimizer state for an active parameter")
    if completed and positive_auxiliary and any(
            state["step"].item() != completed for state in saved["state"].values()):
        raise ValueError("Active parameter optimizer counter differs from completed updates")
    if not completed and saved["state"]:
        raise ValueError("Initial checkpoint unexpectedly contains Adam moments")


def load_checkpoint(path, *, model, optimizer, contract):
    """Restore only this experiment's complete, strictly matching checkpoints."""
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if packet.get("schema") != SCHEMA or packet.get("contract") != contract:
        raise ValueError("NextLat checkpoint source/data/model/runtime/objective contract differs")
    completed = packet.get("completed_updates")
    if (type(completed) is not int or completed < 0
            or packet.get("examples_seen") != completed * contract["batch_size"]
            or not _valid_order_chain(packet.get("order_chain"))):
        raise ValueError("Checkpoint update/word offset/order chain is inconsistent")
    if packet.get("initialization") != json_value(model.nextlat_initialization):
        raise ValueError("NextLat initialization differs")
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
    return {
        "schema": SCHEMA, "architecture": args.architecture, "width": args.width,
        "seed": args.seed, "predictor_seed": args.predictor_seed,
        "data_order_seed": args.data_order_seed, "batch_size": args.batch_size,
        "train_rows": len(train_x), "length": int(train_x.shape[1]),
        "data_manifest_sha256": file_sha256(data_root / "manifest.json"),
        "source_sha256": json_sha256(sources),
        "model_config": json_value(model.backbone.config),
        "nextlat_config": json_value(model.nextlat_config),
        "objective": {"state_loss": "same-position CE mean over B*T",
                      "latent_loss": "SmoothL1 beta=1 mean over B*(T-1)*D",
                      "latent_weight": args.latent_weight, "horizon": 1,
                      "target_detached": True, "source_and_embedding_attached": True,
                      "kl_weight": 0.0, "predicted_state_ce_weight": 0.0},
        "evaluation_route": "backbone_only", "latent_rollout_evaluated": False,
        "precision": "fp32", "tf32": False, "compile": False, "cuda_graphs": False,
        "optimizer": "AdamW-lr1e-4-betas0.9,0.95-eps1e-8-wd0.01-matrices-clip1-all-hybrid",
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device_capability": hardware["capability"],
        "word_order": "independent shuffled epochs, remainder carried into next batch",
    }


def run(args):
    hardware = require_cuda_container()
    configure_fp32_runtime()
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
    model = build_nextlat_model(args.architecture, **model_kwargs)
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
        "checkpoint_selection": "fixed matched update budget; confirmation not evaluated",
        "evaluations": [], "one_step_diagnostics": [], "checkpoints": [],
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
    }
    atomic_json(output / "config.json", vars(args))
    atomic_json(output / "report.json", report)
    tracker = OnlineTracker(
        project=args.wandb_project, entity="taylorbollman", output_dir=output,
        group=args.wandb_group,
        name=args.wandb_run_name or f"{args.architecture}-nextlat-d{args.width}-seed{args.seed}",
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
            if update % args.eval_every == 0 or update in checkpoint_steps:
                limit = args.full_eval_rows if update in checkpoint_steps else args.eval_rows
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
            if update in checkpoint_steps:
                report["checkpoints"].append(save_checkpoint(
                    output / f"checkpoints/step-{update:06d}.pt", model=model,
                    optimizer=optimizer, contract=contract, completed=update,
                    order_chain=order_chain, initialization=initial))
                report.update(completed_updates=completed, order_chain=order_chain,
                              train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - started)
                atomic_json(output / "report.json", report)
        report.update(status="complete", completed_updates=completed, order_chain=order_chain,
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
    p.add_argument("--architecture", choices=("seq", "rt"), required=True)
    p.add_argument("--width", type=int, default=512)
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
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[1000, 5000, 10000])
    p.add_argument("--resume")
    p.add_argument("--wandb-project", default="rt-a5-state-tracking")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-run-name")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
