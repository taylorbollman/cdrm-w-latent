#!/usr/bin/env python3
"""Uncaptured FP32 A5 prefix-state training, with paired order and exact resume."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import (
    A5Metrics, build_model, configure_fp32_runtime, fp32_context,
    make_optimizer, task_loss,
)
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.stage_a_common import require_cuda_container

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-training-v1"


def json_value(value):
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (Path, torch.dtype)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_manifest():
    # Enumerate executed dependencies. A later reporting-only script must not
    # invalidate an otherwise identical training resume.
    sources = [ROOT / "scripts" / name for name in (
        "rt_a5_common.py", "rt_a5_data.py", "rt_a5_train.py", "rt_a5_eval.py")]
    sources += [ROOT / "scripts/experiment_tracking.py", ROOT / "scripts/stage_a_common.py"]
    sources += list((ROOT / "recurrent-transformer/olmo").rglob("*.py"))
    sources += list((ROOT / "configs/rt_a5").glob("*.json"))
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(set(sources))}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


class WordOrder:
    """Shuffle every epoch, carrying its remainder into the next full batch.

    Each epoch's permutation is independently derivable from (seed, epoch).
    An absolute word offset therefore resumes without replaying training or
    depending on evaluation/W&B RNG consumption. No words are dropped.
    """

    def __init__(self, rows, seed):
        if type(rows) is not int or rows < 1 or type(seed) is not int or seed < 0:
            raise ValueError("Word order needs positive rows and a nonnegative seed")
        self.rows, self.seed = rows, seed
        self._epoch = None
        self._permutation = None

    def indices(self, start, count):
        if type(start) is not int or start < 0 or type(count) is not int or count < 1:
            raise ValueError("Invalid absolute word window")
        chunks = []
        while count:
            epoch, offset = divmod(start, self.rows)
            if epoch != self._epoch:
                self._permutation = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(self.rows)
                self._epoch = epoch
            take = min(count, self.rows - offset)
            chunks.append(self._permutation[offset:offset + take])
            start += take
            count -= take
        return np.concatenate(chunks)


def batch_tensors(inputs, labels, indices, device):
    # Explicit copies also support read-only .npy memmaps without unsafe aliasing.
    return (torch.from_numpy(np.array(inputs[indices], dtype=np.int64, copy=True)).to(device),
            torch.from_numpy(np.array(labels[indices], dtype=np.int64, copy=True)).to(device))


def train_step(model, optimizer, inputs, labels, *, clip_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with fp32_context(inputs.device):
        logits = model(inputs).logits
        loss = task_loss(logits, labels)
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Nonfinite A5 training loss")
    optimizer.step()
    with torch.no_grad():
        correct = logits.argmax(-1).eq(labels)
        token_acc = correct.float().mean().item()
        exact = correct.all(dim=1).float().mean().item()
    return {"loss": loss.item(), "token_accuracy": token_acc,
            "whole_word_exact": exact, "grad_norm": grad_norm.item()}


def evaluate_arrays(model, inputs, labels, *, batch_size=1024, limit=None, device="cuda"):
    rows = len(inputs) if limit is None else min(len(inputs), limit)
    if rows < 1 or batch_size < 1:
        raise ValueError("Evaluation needs nonempty data and positive batch size")
    metrics = A5Metrics()
    training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        with torch.no_grad(), fp32_context(device):
            for begin in range(0, rows, batch_size):
                x, y = batch_tensors(inputs, labels, slice(begin, min(begin + batch_size, rows)), device)
                metrics.update(model(x).logits, y)
        result = metrics.compute()
    finally:
        model.train(training)
    result["evaluation_seconds"] = time.perf_counter() - started
    result["evaluated_rows"] = rows
    return result


def optimizer_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def save_checkpoint(path, *, model, optimizer, contract, completed, order_chain, initialization):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    packet = {"schema": SCHEMA, "contract": contract, "completed_updates": completed,
              "examples_seen": completed * contract["batch_size"], "order_chain": order_chain,
              "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
              "optimizer_parameter_names": optimizer_names(model, optimizer),
              "rng": rng_state(), "initialization": initialization}
    temporary = path.with_suffix(".tmp")
    torch.save(packet, temporary)
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size,
            "completed_updates": completed, "examples_seen": packet["examples_seen"]}


def load_checkpoint(path, *, model, optimizer, contract):
    # Only restore our own retained training checkpoints, never untrusted pickle files.
    packet = torch.load(path, map_location="cpu", weights_only=False)
    if packet.get("schema") != SCHEMA or packet.get("contract") != contract:
        raise ValueError("A5 checkpoint source/data/model/runtime contract differs")
    completed = packet.get("completed_updates")
    if type(completed) is not int or completed < 0 or packet.get("examples_seen") != completed * contract["batch_size"]:
        raise ValueError("Checkpoint update/word offset is inconsistent")
    if packet.get("optimizer_parameter_names") != optimizer_names(model, optimizer):
        raise ValueError("Optimizer parameter mapping differs")
    validate_model_state(packet["model"], model.state_dict())
    expected_groups = optimizer.state_dict()["param_groups"]
    groups = packet["optimizer"]["param_groups"]
    if len(groups) != len(expected_groups):
        raise ValueError("Optimizer group count differs")
    for actual, expected in zip(groups, expected_groups):
        for key in ("params", "lr", "betas", "eps", "weight_decay", "foreach", "fused"):
            if actual[key] != expected[key]:
                raise ValueError(f"Optimizer option differs: {key}")
    model.load_state_dict(packet["model"], strict=True)
    optimizer.load_state_dict(packet["optimizer"])
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and (state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all()):
                raise ValueError(f"Invalid FP32 optimizer state: {key}")
    restore_rng(packet["rng"])
    return packet


def validate_model_state(weights, expected):
    if set(weights) != set(expected):
        raise ValueError("Model state keys differ")
    for name, value in weights.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid FP32 model checkpoint tensor: {name}")


def flatten_eval(result, role):
    scalars = {f"dev/{role}/{key}": value for key, value in result.items()
               if isinstance(value, (int, float))}
    for key, values in result.items():
        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
            scalars.update({f"dev/{role}/{key}/position_{i + 1:02d}": v for i, v in enumerate(values)})
    return scalars


def run(args):
    hardware = require_cuda_container()
    configure_fp32_runtime()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory; resume writes a new continuation directory")
    if min(args.updates, args.batch_size, args.eval_every, args.eval_rows, args.full_eval_rows, args.log_every) < 1:
        raise ValueError("Update/batch/evaluation/log counts must be positive")
    output.mkdir(parents=True)
    data_root = Path(args.data_dir).resolve()
    validate_manifest(data_root)
    train_x, train_y = load_split(data_root, "train")
    dev = {role: load_split(data_root, role) for role in ("dev", "ood_dev")}
    sources = source_manifest()
    for relative in sources:
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    model = build_model(args.architecture, width=args.width, seed=args.seed, device="cuda")
    optimizer = make_optimizer(model)
    initial = json_value(model.a5_initialization)
    contract = {"schema": SCHEMA, "architecture": args.architecture, "width": args.width,
                "seed": args.seed, "data_order_seed": args.data_order_seed,
                "batch_size": args.batch_size, "train_rows": len(train_x), "length": train_x.shape[1],
                "data_manifest_sha256": file_sha256(data_root / "manifest.json"),
                "source_sha256": json_sha256(sources), "model_config": json_value(model.config),
                "precision": "fp32", "tf32": False, "compile": False, "cuda_graphs": False,
                "optimizer": "AdamW-lr1e-4-betas0.9,0.95-eps1e-8-wd0.01-matrices-clip1",
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device_capability": hardware["capability"],
                "word_order": "independent shuffled epochs, remainder carried into next batch"}
    completed, order_chain = 0, hashlib.sha256(b"rt-a5-word-order-v1").hexdigest()
    parent = None
    if args.resume:
        packet = load_checkpoint(args.resume, model=model, optimizer=optimizer, contract=contract)
        completed, order_chain, initial = packet["completed_updates"], packet["order_chain"], packet["initialization"]
        parent = {"path": str(Path(args.resume).resolve()), "sha256": file_sha256(args.resume)}
        if completed >= args.updates:
            raise ValueError("Resume endpoint must exceed completed updates")
    report = {"schema": SCHEMA, "status": "running", "contract": contract,
              "hardware": hardware, "source_files": sources, "initialization": initial,
              "parent_checkpoint": parent, "start_update": completed, "endpoint": args.updates,
              "checkpoint_selection": "fixed matched update budget; confirmation not evaluated",
              "evaluations": [], "checkpoints": [], "confirmation_evaluated": False}
    atomic_json(output / "config.json", vars(args))
    atomic_json(output / "report.json", report)
    tracker = OnlineTracker(project=args.wandb_project, entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name=args.wandb_run_name or f"{args.architecture}-d{args.width}-seed{args.seed}",
                            preserve_state=preserve_rng)
    history = (output / "history.jsonl").open("x", buffering=1)
    started = time.perf_counter()
    train_seconds = 0.0
    try:
        tracker.start(contract)
        report["wandb"] = tracker.record
        atomic_json(output / "report.json", report)
        print(json.dumps({"event": "started", "wandb": tracker.record["run_url"], "architecture": args.architecture,
                          "parameter_count": sum(p.numel() for p in model.parameters())}), flush=True)
        if completed == 0:
            report["checkpoints"].append(save_checkpoint(output / "checkpoints/step-000000.pt", model=model,
                optimizer=optimizer, contract=contract, completed=0, order_chain=order_chain, initialization=initial))
        order = WordOrder(len(train_x), args.data_order_seed)
        checkpoint_steps = set(args.checkpoint_steps) | {args.updates}
        window = []
        for update in range(completed + 1, args.updates + 1):
            begin = time.perf_counter()
            indices = order.indices((update - 1) * args.batch_size, args.batch_size)
            order_chain = hashlib.sha256(bytes.fromhex(order_chain) + indices.astype("<i8").tobytes()).hexdigest()
            x, y = batch_tensors(train_x, train_y, indices, "cuda")
            values = train_step(model, optimizer, x, y)
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            completed = update
            row = {"update": update, "examples_seen": update * args.batch_size,
                   "seconds": elapsed, "order_chain": order_chain, **values}
            history.write(json.dumps(row, allow_nan=False) + "\n")
            window.append(row)
            if update % args.log_every == 0 or update == args.updates:
                logged = {"update": update, "train/examples_seen": update * args.batch_size,
                          "train/seconds_per_update": sum(v["seconds"] for v in window) / len(window)}
                for key in ("loss", "token_accuracy", "whole_word_exact", "grad_norm"):
                    logged[f"train/{key}"] = sum(v[key] for v in window) / len(window)
                tracker.log(logged)
                print(json.dumps({"event": "train", **logged}), flush=True)
                window.clear()
            if update % args.eval_every == 0 or update == args.updates:
                full = update in checkpoint_steps and update >= 5000
                limit = args.full_eval_rows if full else args.eval_rows
                for role, (dx, dy) in dev.items():
                    result = evaluate_arrays(model, dx, dy, batch_size=args.batch_size, limit=limit)
                    result.update(role=role, update=update)
                    report["evaluations"].append(result)
                    tracker.log({"update": update, **flatten_eval(result, role)})
                    print(json.dumps({"event": "evaluation", **result}), flush=True)
            if update in checkpoint_steps:
                report["checkpoints"].append(save_checkpoint(output / f"checkpoints/step-{update:06d}.pt",
                    model=model, optimizer=optimizer, contract=contract, completed=update,
                    order_chain=order_chain, initialization=initial))
                report.update(completed_updates=completed, order_chain=order_chain, train_seconds=train_seconds,
                              elapsed_seconds=time.perf_counter() - started)
                atomic_json(output / "report.json", report)
        report.update(status="complete", completed_updates=completed, order_chain=order_chain,
                      train_seconds=train_seconds, elapsed_seconds=time.perf_counter() - started)
        tracker.summary({"completed_updates": completed, "confirmation_evaluated": False,
                         "train_seconds": train_seconds, "order_chain": order_chain})
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
    p.add_argument("--data-order-seed", type=int, default=1234)
    p.add_argument("--updates", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-rows", type=int, default=4096)
    p.add_argument("--full-eval-rows", type=int, default=102400)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[1000, 5000, 10000])
    p.add_argument("--resume")
    p.add_argument("--wandb-project", default="rt-a5-state-tracking")
    p.add_argument("--wandb-group")
    p.add_argument("--wandb-run-name")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
