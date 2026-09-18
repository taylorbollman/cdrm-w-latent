#!/usr/bin/env python3
"""Bounded FP32 Fuzzy-only RT + NextLat pilot with exact optimizer continuation.

The native MAD labels are already aligned. Inputs are mapped into the shared
76-symbol interface; CE is over the native 16-class readout. Final confirmation
is deliberately absent from this training interface.
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
import torch.nn.functional as F

from cdrm.mad_data import FUZZY_TASK, IGNORE_INDEX, load_dataset
from cdrm.rt_nextlat_fuzzy_metrics import FuzzyMetrics
from cdrm.rt_nextlat_tasks import build_model, encode_inputs, read_configuration, task_loss
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_nextlat_train import _validate_optimizer
from scripts.rt_a5_train import (
    WordOrder, atomic_json, batch_tensors, cpu_tree, file_sha256, json_sha256,
    json_value, optimizer_names, preserve_rng, restore_rng, rng_state,
    validate_model_state,
)
from scripts.stage_a_common import require_cuda_container


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-fuzzy-training-v1"
ORDER_INITIAL = hashlib.sha256(b"rt-nextlat-fuzzy-word-order-v1").hexdigest()


def source_manifest():
    """Snapshot executed dependencies; reporting additions do not alter resume."""
    names = (
        "scripts/rt_nextlat_fuzzy_train.py", "scripts/rt_a5_train.py",
        "scripts/rt_a5_nextlat_train.py", "scripts/rt_a5_nextlat.py",
        "scripts/rt_a5_common.py", "scripts/rt_a5_data.py", "scripts/rt_a5_window.py",
        "scripts/rt_a5_nextlat_variant.py",
        "scripts/experiment_tracking.py", "scripts/stage_a_common.py",
        "cdrm/rt_nextlat_tasks.py", "cdrm/rt_nextlat_fuzzy_metrics.py",
        "cdrm/mad_data.py", "configs/rt_a5/base.json",
        "configs/rt_a5_nextlat/base.json", "configs/rt_a5_window/base.json",
        "configs/rt_a5_nextlat_variant/base.json", "vendors/mad-lab/mad/data/instances.py",
        "vendors/mad-lab/configs/tasks/fuzzy-in-context-recall.yml",
    )
    paths = {ROOT / name for name in names}
    paths.update((ROOT / "recurrent-transformer/olmo").rglob("*.py"))
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(paths)}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _finite_scalar(value, name):
    value = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
    if not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite training diagnostic: {name}")
    return value


def train_step(model, optimizer, inputs, labels, *, microbatch=128,
               latent_weight=1.0, clip_norm=1.0):
    """One logical update: separately normalize CE and NextLat, clip/step once.

    All examples have the same length. CE weights use scored-token counts,
    whereas the native all-transition latent mean uses example counts. This
    also handles an uneven final physical microbatch without overweighting it.
    """
    if inputs.ndim != 2 or labels.shape != inputs.shape or len(inputs) < 1:
        raise ValueError("Expected nonempty equally shaped [B,T] inputs and labels")
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    if not math.isfinite(latent_weight) or latent_weight < 0:
        raise ValueError("latent_weight must be finite and nonnegative")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    total_scored = int(labels.ne(IGNORE_INDEX).sum().item())
    if not total_scored:
        raise ValueError("No scored targets in the logical batch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(4, dtype=torch.float64, device=inputs.device)
    for start in range(0, len(inputs), microbatch):
        x, y = inputs[start:start + microbatch], labels[start:start + microbatch]
        scored = int(y.ne(IGNORE_INDEX).sum().item())
        if not scored:
            raise ValueError("A training microbatch has no scored targets")
        row_weight, ce_weight = len(x) / len(inputs), scored / total_scored
        with fp32_context(inputs.device):
            result = task_loss(model, x, y, task="fuzzy", latent_weight=latent_weight)
            loss = result["ce"] * ce_weight + latent_weight * result["latent"] * row_weight
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite accumulated objective")
            loss.backward()
        with torch.no_grad():
            good = result["logits"].argmax(-1).eq(y) & y.ne(IGNORE_INDEX)
            sums[0] += result["ce"].detach().double() * ce_weight
            sums[1] += result["latent"].detach().double() * row_weight
            sums[2] += good.sum()
            sums[3] += (good | y.eq(IGNORE_INDEX)).all(-1).sum()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm,
                                              error_if_nonfinite=True)
    optimizer.step()
    _sync(inputs.device)
    ce, latent, correct, exact = sums.cpu().tolist()
    return {"loss": ce + latent_weight * latent, "ce": ce, "latent": latent,
            "weighted_latent": latent_weight * latent,
            "token_accuracy": correct / total_scored,
            "scored_sequence_exact": exact / len(inputs),
            "grad_norm": _finite_scalar(grad_norm, "gradient norm"),
            "examples": len(inputs), "scored_tokens": total_scored}


def evaluate(model, dataset, *, microbatch=128, device="cuda", latent_weight=1.0,
             limit=None):
    """Native answer metrics plus separately named teacher-conditioned loss."""
    rows = len(dataset) if limit is None else min(len(dataset), limit)
    if rows < 1 or microbatch < 1:
        raise ValueError("Evaluation requires positive rows and microbatch")
    metrics = FuzzyMetrics(dataset)
    ce_sum = latent_sum = 0.0
    scored = 0
    training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        with preserve_rng(), torch.no_grad(), fp32_context(device):
            for start in range(0, rows, microbatch):
                indices = np.arange(start, min(rows, start + microbatch))
                x, y = batch_tensors(dataset.input_ids, dataset.answer_labels, indices, device)
                result = task_loss(model, encode_inputs(x, task="fuzzy"), y,
                                   task="fuzzy", latent_weight=latent_weight)
                per_token = F.cross_entropy(result["logits"].transpose(1, 2), y,
                                            ignore_index=IGNORE_INDEX, reduction="none")
                metrics.update(result["logits"].argmax(-1).cpu().numpy(),
                               per_token.cpu().numpy(), indices=indices)
                count = int(y.ne(IGNORE_INDEX).sum().item())
                ce_sum += _finite_scalar(result["ce"], "dev CE") * count
                latent_sum += _finite_scalar(result["latent"], "dev NextLat") * len(indices)
                scored += count
            _sync(device)
    finally:
        model.train(training)
    return {**metrics.compute(), "ce": ce_sum / scored, "latent": latent_sum / rows,
            "weighted_latent": latent_weight * latent_sum / rows,
            "evaluated_rows": rows, "evaluation_seconds": time.perf_counter() - started,
            "ce_scope": "native scored answer tokens",
            "latent_scope": "all native within-example transitions, including native padding",
            "route": "backbone logits; teacher-conditioned NextLat diagnostic; no rollout"}


def _valid_chain(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _cursor(completed, contract):
    offset = completed * contract["batch_size"]
    epoch, position = divmod(offset, contract["train_rows"])
    return {"absolute_example_offset": offset, "epoch": epoch, "position": position}


def finite_state(model, optimizer, completed):
    """Check parameters, retained gradients, and each Adam tensor before saving."""
    parameters = dict(model.named_parameters())
    for name, value in parameters.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise FloatingPointError(f"Invalid FP32 parameter: {name}")
        if value.grad is not None and (value.grad.dtype != torch.float32 or
                                      not torch.isfinite(value.grad).all()):
            raise FloatingPointError(f"Invalid FP32 gradient: {name}")
    packet = {"optimizer": optimizer.state_dict(),
              "optimizer_parameter_names": optimizer_names(model, optimizer),
              "contract": {"objective": {"latent_weight": 1.0}}}
    _validate_optimizer(packet, model, optimizer, completed)
    return {"parameters_finite_fp32": True, "gradients_finite_fp32": True,
            "adam_finite_fp32": True, "parameter_tensors": len(parameters),
            "adam_parameter_states": len(optimizer.state)}


def save_checkpoint(path, *, model, optimizer, contract, completed, order_chain,
                    initialization):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if type(completed) is not int or completed < 0 or not _valid_chain(order_chain):
        raise ValueError("Invalid update counter/order hash")
    audit = finite_state(model, optimizer, completed)
    packet = {"schema": SCHEMA, "contract": contract, "completed_updates": completed,
              "examples_seen": completed * contract["batch_size"],
              "next_cursor": _cursor(completed, contract), "order_chain": order_chain,
              "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
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
            "bytes": path.stat().st_size, "completed_updates": completed, **audit}


def load_checkpoint(path, *, model, optimizer, contract):
    # Our locally generated or retained full-state checkpoints only.
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if packet.get("schema") != SCHEMA or packet.get("contract") != contract:
        raise ValueError("Checkpoint model/data/source/runtime/optimizer contract differs")
    completed = packet.get("completed_updates")
    if (type(completed) is not int or completed < 0 or
            packet.get("examples_seen") != completed * contract["batch_size"] or
            packet.get("next_cursor") != _cursor(completed, contract) or
            not _valid_chain(packet.get("order_chain"))):
        raise ValueError("Checkpoint update/cursor/order chain differs")
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


class _DisabledTracker:
    """Explicit opt-out for short integration checks, never an online fallback."""
    record = {"enabled": False, "mode": "disabled", "run_url": None}

    def start(self, config):
        pass

    def log(self, metrics):
        pass

    def summary(self, metrics):
        pass

    def finish(self, *, succeeded):
        pass


def run(args):
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    counts = (args.updates, args.batch_size, args.microbatch, args.eval_microbatch,
              args.eval_every, args.eval_rows, args.log_every)
    if min(counts) < 1 or min(args.seed, args.predictor_seed, args.fuzzy_seed, args.order_seed) < 0:
        raise ValueError("Positive counts and nonnegative seeds required")
    if any(s < 0 for s in args.checkpoint_steps):
        raise ValueError("Checkpoint steps must be nonnegative")
    if args.wandb_mode == "disabled" and args.updates > 10:
        raise ValueError("Disabled W&B is restricted to at most ten integration-check updates")
    output, data_root = Path(args.output).resolve(), Path(args.data).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; resume creates a new lineage")
    config = read_configuration(args.config)
    train = load_dataset(data_root, FUZZY_TASK, "train")
    dev = load_dataset(data_root, FUZZY_TASK, "dev")
    if np.any(train.labels == IGNORE_INDEX):
        raise ValueError("This native Fuzzy pilot requires dense training labels")
    if train.input_ids.shape[1] != dev.input_ids.shape[1]:
        raise ValueError("Training and development lengths must match")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(config, seed=args.seed, predictor_seed=args.predictor_seed,
                        fuzzy_seed=args.fuzzy_seed, device="cuda")
    optimizer = make_optimizer(model, lr=args.learning_rate)
    sources = source_manifest()
    config_path = Path(args.config).resolve()
    data_identity = {split: dataset.manifest for split, dataset in (("train", train), ("dev", dev))}
    preparation_manifest = data_root / "manifest.json"
    contract = {"schema": SCHEMA, "model_config": json_value(config),
                "configuration_file_sha256": file_sha256(config_path),
                "source_sha256": json_sha256(sources), "data_sha256": json_sha256(data_identity),
                "preparation_manifest_sha256": file_sha256(preparation_manifest)
                    if preparation_manifest.exists() else None,
                "initialization": json_value(model.initialization),
                "batch_size": args.batch_size, "microbatch": args.microbatch,
                "train_rows": len(train), "length": train.input_ids.shape[1],
                "order_seed": args.order_seed, "runtime": runtime,
                "device_capability": hardware["capability"],
                "objective": {"ce": "dense native aligned local16 targets; no second shift",
                              "latent": "SmoothL1 beta1; all T-1 native transitions; detached target",
                              "latent_weight": 1.0},
                "optimizer": {"type": "AdamW", "lr": args.learning_rate,
                              "betas": [0.9, 0.95], "eps": 1e-8,
                              "matrix_decay": 0.01, "vector_decay": 0.0, "clip_norm": 1.0},
                "word_order": "independent shuffled epochs with remainder carried"}
    completed, order_chain, parent = 0, ORDER_INITIAL, None
    if args.resume:
        packet = load_checkpoint(args.resume, model=model, optimizer=optimizer, contract=contract)
        completed, order_chain = packet["completed_updates"], packet["order_chain"]
        if completed >= args.updates:
            raise ValueError("Continuation endpoint must exceed the restored update")
        parent = {"path": str(Path(args.resume).resolve()), "sha256": file_sha256(args.resume)}
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
    report = {"schema": SCHEMA, "status": "running", "contract": contract,
              "hardware": hardware, "parent_checkpoint": parent,
              "start_update": completed, "completed_updates": completed,
              "requested_endpoint": args.updates, "initialization": json_value(model.initialization),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "checkpoints": [], "evaluations": [], "confirmation_evaluated": False,
              "latent_rollout_evaluated": False}
    tracker = (_DisabledTracker() if args.wandb_mode == "disabled" else OnlineTracker(
        project=args.wandb_project, entity="taylorbollman", output_dir=output,
        group=args.wandb_group, name=args.wandb_run_name or "l1r-rt2-nextlat-d128-t400",
        preserve_state=preserve_rng))
    report["wandb"] = tracker.record
    atomic_json(output / "report.json", report)
    checkpoint_steps = set(args.checkpoint_steps) | {args.updates}
    checkpointed = set()
    order = WordOrder(len(train), args.order_seed)
    started = time.perf_counter()
    train_seconds = 0.0
    stop_requested = []
    handlers = {}

    def request_stop(number, frame):
        stop_requested.append(signal.Signals(number).name)

    for signum in (signal.SIGTERM, signal.SIGINT):
        handlers[signum] = signal.signal(signum, request_stop)

    def checkpoint():
        if completed in checkpointed:
            return
        report["checkpoints"].append(save_checkpoint(
            output / f"checkpoints/step-{completed:06d}.pt", model=model, optimizer=optimizer,
            contract=contract, completed=completed, order_chain=order_chain,
            initialization=model.initialization))
        checkpointed.add(completed)

    succeeded = False
    history = (output / "history.jsonl").open("x", buffering=1)
    try:
        tracker.start(contract)
        print(json.dumps({"event": "started", "wandb": tracker.record.get("run_url"),
                          "parameters": report["parameter_count"], "endpoint": args.updates}), flush=True)
        if completed == 0:
            checkpoint()
        atomic_json(output / "report.json", report)
        window = []
        while completed < args.updates:
            if stop_requested or (args.stop_file and Path(args.stop_file).exists()):
                report["stop_reason"] = stop_requested or ["stop_file"]
                break
            update = completed + 1
            _sync("cuda")
            begin = time.perf_counter()
            indices = order.indices(completed * args.batch_size, args.batch_size)
            next_chain = hashlib.sha256(bytes.fromhex(order_chain) + indices.astype("<i8").tobytes()).hexdigest()
            x, y = batch_tensors(train.input_ids, train.labels, indices, "cuda")
            values = train_step(model, optimizer, encode_inputs(x, task="fuzzy"), y,
                                microbatch=args.microbatch)
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            completed, order_chain = update, next_chain
            row = {"update": update, "examples_seen": update * args.batch_size,
                   "seconds": elapsed, "order_chain": order_chain, **values}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            window.append(row)
            if update % args.log_every == 0 or update == args.updates:
                logged = {"update": update, "train/examples_seen": update * args.batch_size,
                          "train/seconds_per_update": sum(r["seconds"] for r in window) / len(window),
                          "train/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "train/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
                for key in ("loss", "ce", "latent", "weighted_latent", "token_accuracy",
                            "scored_sequence_exact", "grad_norm"):
                    logged[f"train/{key}"] = sum(r[key] for r in window) / len(window)
                tracker.log(logged)
                print(json.dumps({"event": "train", **logged}), flush=True)
                window.clear()
            if update in checkpoint_steps:
                checkpoint()
            if update % args.eval_every == 0 or update in checkpoint_steps:
                result = evaluate(model, dev, microbatch=args.eval_microbatch,
                                  limit=args.eval_rows, device="cuda")
                result["update"] = update
                report["evaluations"].append(result)
                tracker.log({"update": update, **scalar_metrics(result, "dev/fuzzy")})
                print(json.dumps({"event": "evaluation", **result}, allow_nan=False), flush=True)
            report.update(completed_updates=completed, order_chain=order_chain,
                          next_cursor=_cursor(completed, contract), train_seconds=train_seconds,
                          elapsed_seconds=time.perf_counter() - started)
            if update % args.log_every == 0 or update in checkpoint_steps:
                atomic_json(output / "report.json", report)
        checkpoint()
        if not report["evaluations"] or report["evaluations"][-1]["update"] != completed:
            result = evaluate(model, dev, microbatch=args.eval_microbatch,
                              limit=args.eval_rows, device="cuda")
            result["update"] = completed
            report["evaluations"].append(result)
            tracker.log({"update": completed, **scalar_metrics(result, "dev/fuzzy")})
        if source_manifest() != sources or file_sha256(config_path) != contract["configuration_file_sha256"]:
            raise RuntimeError("Executed sources or configuration changed during training")
        report.update(status="complete" if completed == args.updates else "stopped",
                      completed_updates=completed, order_chain=order_chain,
                      next_cursor=_cursor(completed, contract), train_seconds=train_seconds,
                      elapsed_seconds=time.perf_counter() - started)
        tracker.summary({"completed_updates": completed, "confirmation_evaluated": False,
                         "status": report["status"], **scalar_metrics(report["evaluations"][-1], "final/dev")})
        succeeded = True
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, completed_updates=completed,
                      order_chain=order_chain, elapsed_seconds=time.perf_counter() - started)
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
    print(json.dumps({"event": "finished", "status": report["status"],
                      "completed_updates": completed, "wandb": tracker.record.get("run_url")}), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=128)
    p.add_argument("--eval-microbatch", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-rows", type=int, default=1280)
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[0, 1000, 5000, 10000])
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--predictor-seed", type=int, default=1235)
    p.add_argument("--fuzzy-seed", type=int, default=1236)
    p.add_argument("--order-seed", type=int, default=5432)
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
