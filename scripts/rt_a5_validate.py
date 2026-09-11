#!/usr/bin/env python3
"""Bounded A5 plumbing/backward checks, easy-set overfit and FP32 timing.

Run through scripts/docker_shell.sh. These discarded updates are validation
fixtures, not development comparisons or resumable research checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import numpy as np
import torch

import rt_a5_common as common
from experiment_tracking import OnlineTracker, add_wandb_arguments, scalar_metrics
from stage_a_common import require_cuda_container

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 2e-6, 2e-5


def compare_tensors(reference, actual, *, atol=ATOL, rtol=RTOL):
    """Check every coordinate; keep norm errors descriptive, including zeros."""
    if reference.shape != actual.shape:
        raise ValueError("Comparison shapes differ")
    ref, got = reference.detach().float().cpu(), actual.detach().float().cpu()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(got).all())
    if not finite:
        return {"passed": False, "finite": False, "coordinates": ref.numel()}
    error = got - ref
    ref_norm, error_norm = torch.linalg.vector_norm(ref).item(), torch.linalg.vector_norm(error).item()
    mismatches = int((error.abs() > atol + rtol * ref.abs()).sum())
    return {"passed": mismatches == 0, "finite": True, "coordinates": ref.numel(),
            "mismatched_coordinates": mismatches, "reference_l2": ref_norm,
            "actual_l2": torch.linalg.vector_norm(got).item(), "error_l2": error_norm,
            "relative_l2": error_norm / max(ref_norm, 1e-12),
            "max_absolute_error": error.abs().max().item() if ref.numel() else 0.0,
            "atol": atol, "rtol": rtol}


def timing_summary(seconds, *, words, length, overhead_fraction=0.15):
    if len(seconds) < 2 or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError("Timing needs at least two finite positive measurements")
    if overhead_fraction < 0 or not math.isfinite(overhead_fraction):
        raise ValueError("Invalid overhead allowance")
    mean = statistics.mean(seconds)
    split = len(seconds) // 2
    first, last = statistics.median(seconds[:split]), statistics.median(seconds[split:])
    ratio = max(first, last) / min(first, last)
    return {"timed_updates": len(seconds), "mean_seconds": mean,
            "median_seconds": statistics.median(seconds), "min_seconds": min(seconds),
            "max_seconds": max(seconds), "words_per_second": words / mean,
            "tokens_per_second": words * length / mean,
            "first_half_median_seconds": first, "last_half_median_seconds": last,
            "half_window_ratio": ratio, "needs_more_warmup_or_timing": ratio > 1.2,
            "overhead_fraction_assumed": overhead_fraction,
            "estimated_hours": {str(n): {"training_only": n * mean / 3600,
                "with_assumed_eval_checkpoint_overhead": n * mean * (1 + overhead_fraction) / 3600}
                for n in (10_000, 400_000)},
            "scope": "Synchronized update timing includes forward/backward/clip/Adam; excludes fixture transfer, finite-state audit, metrics, W&B and JSON writes. Overhead allowance is an assumption, not measured evaluation cadence."}


def finite_state(model, optimizer=None, *, require_gradients=False):
    missing, wrong_dtype, nonfinite = [], [], []
    for name, parameter in model.named_parameters():
        for kind, value in (("parameter", parameter), ("gradient", parameter.grad)):
            if value is None:
                if kind == "gradient" and require_gradients and parameter.requires_grad:
                    missing.append(name)
                continue
            if value.dtype != torch.float32:
                wrong_dtype.append(f"{kind}:{name}")
            if not bool(torch.isfinite(value).all()):
                nonfinite.append(f"{kind}:{name}")
    steps = []
    if optimizer is not None:
        for name, parameter in model.named_parameters():
            state = optimizer.state.get(parameter, {})
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if not bool(torch.isfinite(value).all()):
                        nonfinite.append(f"optimizer:{name}:{key}")
                    if key in ("exp_avg", "exp_avg_sq") and value.dtype != torch.float32:
                        wrong_dtype.append(f"optimizer:{name}:{key}")
            if "step" in state:
                steps.append(int(state["step"]))
    return {"passed": not (missing or wrong_dtype or nonfinite),
            "missing_gradients": missing, "wrong_dtype": wrong_dtype, "nonfinite": nonfinite,
            "optimizer_steps": sorted(set(steps)), "optimizer_state_count": len(steps)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_snapshot(output_dir):
    paths = [ROOT / "scripts" / name for name in
             ("rt_a5_validate.py", "rt_a5_common.py", "rt_a5_data.py", "experiment_tracking.py", "stage_a_common.py")]
    paths += sorted((ROOT / "recurrent-transformer/olmo").rglob("*.py"))
    paths += sorted(path for path in (ROOT / "configs/rt_a5").rglob("*") if path.is_file())
    hashes = {}
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(ROOT)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        target = output_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return hashes


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[0]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def fixture(batch, length, seed):
    import rt_a5_data as data
    words = data.generate_unique_words(batch, length, seed=seed)
    labels = data.prefix_labels(words)
    return torch.as_tensor(words, dtype=torch.long, device="cuda"), torch.as_tensor(labels, dtype=torch.long, device="cuda")


def save_fixture(output_dir, name, inputs, labels):
    path = output_dir / f"{name}.npz"
    np.savez(path, inputs=inputs.cpu().numpy(), labels=labels.cpu().numpy())
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(inputs.shape), "label_alignment": "same-position prefix state; no BOS/PAD/shift"}


def logits_for(model, inputs):
    with common.fp32_context(device="cuda"):
        return model(inputs).logits


def metrics_for(logits, labels):
    metrics = common.A5Metrics()
    metrics.update(logits, labels)
    return metrics.compute()


def checked_result(report, key, value):
    report[key] = value
    if not value["passed"]:
        raise AssertionError(f"A5 validation failed: {key}")


def run_checks(args, report, tracker):
    counts = {}
    for width in (128, 256, 512):
        pair = {arm: common.build_model(arm, width=width, seed=args.seed, device="cpu") for arm in ("seq", "rt")}
        counts[str(width)] = {arm: sum(p.numel() for p in model.parameters()) for arm, model in pair.items()}
        expected = 24 * width * width + 129 * width
        counts[str(width)].update(expected=expected, mapped_initialization_equal=
            common.canonical_parameter_sha256(pair["seq"]) == common.canonical_parameter_sha256(pair["rt"]))
        counts[str(width)]["passed"] = counts[str(width)]["seq"] == counts[str(width)]["rt"] == expected and counts[str(width)]["mapped_initialization_equal"]
        del pair
    report["paired_counts"] = counts
    if not all(row["passed"] for row in counts.values()):
        raise AssertionError("Paired model count or weight mapping failed")
    inputs, labels = fixture(2, 36, args.seed + 10)
    report["fixture"] = save_fixture(args.output_dir, "checks-b2-t36", inputs, labels)
    report["causality"] = {}
    for arm in ("seq", "rt"):
        model = common.build_model(arm, width=128, seed=args.seed, device="cuda").eval()
        case = report["causality"][arm] = {}
        with torch.no_grad():
            full = logits_for(model, inputs)
            changed = inputs.clone()
            changed[:, 12:] = (changed[:, 12:] + 1) % 60
            checked_result(case, "changed_future", compare_tensors(full[:, :12], logits_for(model, changed)[:, :12]))
            permutation = torch.tensor([1, 0], device="cuda")
            checked_result(case, "batch_permutation", compare_tensors(full[permutation], logits_for(model, inputs[permutation])))
            for length in (1, 12, 24):
                checked_result(case, f"truncated_prefix_{length}", compare_tensors(full[:, :length], logits_for(model, inputs[:, :length])))
            checked_result(case, "finite_fp32", finite_state(model))
        del model
    report["backward"] = {}
    for length in (12, 36):
        case = report["backward"][str(length)] = {"shape": [2, length], "width": 128, "seed": args.seed}
        packets = {}
        for backend in ("naive", "tiled"):
            model = common.build_model("rt", width=128, seed=args.seed, device="cuda", backend=backend).train()
            with common.fp32_context(device="cuda"):
                output = model(inputs[:, :length]).logits
                loss = common.task_loss(output, labels[:, :length])
                loss.backward()
            checked_result(case, f"{backend}_finite_fp32", finite_state(model, require_gradients=True))
            packets[backend] = {"logits": output.detach().cpu(), "loss": loss.detach().cpu(),
                "gradients": {name: parameter.grad.detach().cpu() for name, parameter in model.named_parameters()},
                "initialization": common.canonical_parameter_sha256(model)}
            del output, loss, model
        reference, actual = packets["naive"], packets["tiled"]
        if reference["initialization"] != actual["initialization"] or reference["gradients"].keys() != actual["gradients"].keys():
            raise AssertionError("Backward comparison initialization or parameter coverage differs")
        checked_result(case, "logits", compare_tensors(reference["logits"], actual["logits"]))
        checked_result(case, "unshifted_loss", compare_tensors(reference["loss"], actual["loss"]))
        case["gradients"] = {name: compare_tensors(reference["gradients"][name], actual["gradients"][name])
                              for name in reference["gradients"]}
        case["gradient_tensor_count"] = len(case["gradients"])
        case["gradient_coordinate_count"] = sum(row["coordinates"] for row in case["gradients"].values())
        case["failed_gradient_tensors"] = [name for name, row in case["gradients"].items() if not row["passed"]]
        case["global_gradient_relative_l2"] = math.sqrt(sum(row["error_l2"] ** 2 for row in case["gradients"].values())) / max(math.sqrt(sum(row["reference_l2"] ** 2 for row in case["gradients"].values())), 1e-12)
        if case["failed_gradient_tensors"]:
            raise AssertionError(f"T{length} FP32 backward comparison failed")
        tracker.log({"check/length": length, "check/gradient_relative_l2": case["global_gradient_relative_l2"]})
        write_json(args.output_dir / "progress.json", report)


def update(model, optimizer, inputs, labels):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with common.fp32_context(device="cuda"):
        output = model(inputs).logits
        loss = common.task_loss(output, labels)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return {"seconds": seconds, "loss": loss.item(), "raw_gradient_norm": norm.item()}, output.detach()


def run_overfit(args, report, tracker):
    inputs, labels = fixture(32, args.overfit_length, args.seed + 20)
    report["fixture"] = save_fixture(args.output_dir, "easy-set", inputs, labels)
    report["scope"] = "Memorization/plumbing smoke on 32 fixed short words, LR1e-3; no task-generalization claim"
    report["arms"] = {}
    for arm in args.arms:
        model = common.build_model(arm, width=128, seed=args.seed, device="cuda").train()
        optimizer = common.make_optimizer(model, lr=1e-3)
        case = report["arms"][arm] = {"width": 128, "length": args.overfit_length,
            "batch": 32, "lr": 1e-3, "max_updates": args.overfit_updates, "rows": [],
            "initialization_sha256": common.canonical_parameter_sha256(model)}
        with torch.no_grad():
            case["initial_metrics"] = metrics_for(logits_for(model, inputs), labels)
        reached = False
        for step in range(1, args.overfit_updates + 1):
            row, output = update(model, optimizer, inputs, labels)
            row["update"] = step
            if step == 1 or step % 10 == 0 or step == args.overfit_updates:
                checked_result(case, "finite_fp32", finite_state(model, optimizer, require_gradients=True))
                with torch.no_grad():
                    predictions = logits_for(model, inputs)
                    correct = predictions.argmax(-1).eq(labels)
                    row["post_update_metrics"] = metrics_for(predictions, labels)
                    reached = bool((correct.float().mean() >= .99) & (correct.all(-1).float().mean() >= .99))
                case["rows"].append(row)
                tracker.log({"update": step, **scalar_metrics(row, f"train/{arm}")})
                write_json(args.output_dir / "progress.json", report)
                print(json.dumps({"arm": arm, "update": step, "loss": row["loss"], "easy_set_solved": reached}), flush=True)
            if reached:
                break
        case.update(completed_updates=step, reached_99_percent_token_and_word_accuracy=reached,
                    final_parameter_sha256=common.canonical_parameter_sha256(model))
        case["parameters_changed"] = case["initialization_sha256"] != case["final_parameter_sha256"]
        if not reached or not case["parameters_changed"]:
            raise AssertionError(f"{arm} bounded easy-set overfit did not pass")
        del output, predictions, optimizer, model
        torch.cuda.empty_cache()


def run_profile(args, report, tracker):
    inputs, labels = fixture(1024, 12, args.seed + 30)
    eval_inputs, eval_labels = fixture(1024, 36, args.seed + 31)
    report["fixtures"] = [save_fixture(args.output_dir, "profile-b1024-t12", inputs, labels),
                          save_fixture(args.output_dir, "profile-eval-b1024-t36", eval_inputs, eval_labels)]
    report["scope"] = "Physical B1024 FP32 operational profile; repeated random task batches and discarded updates; no task-performance claim"
    report["arms"] = {}
    for arm in args.arms:
        model = common.build_model(arm, width=512, seed=args.seed, device="cuda").train()
        optimizer = common.make_optimizer(model, lr=1e-4)
        case = report["arms"][arm] = {"width": 512, "blocks": 2, "batch": 1024, "length": 12,
            "physical_batch": True, "gradient_accumulation": 1, "rows": [],
            "resolved_model_config": dataclasses.asdict(model.config),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "initialization_sha256": common.canonical_parameter_sha256(model)}
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, args.warmup + args.timed_updates + 1):
            if step == args.warmup + 1:
                case["warmup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            row, output = update(model, optimizer, inputs, labels)
            row.update(update=step, warmup=step <= args.warmup)
            state = finite_state(model, optimizer, require_gradients=True)
            checked_result(case, "finite_fp32", state)
            if state["optimizer_steps"] != [step] or state["optimizer_state_count"] != len(list(model.parameters())):
                raise AssertionError("Each participating parameter must receive one Adam step per update")
            case["rows"].append(row)
            tracker.log({"update": step, **scalar_metrics(row, f"benchmark/{arm}"),
                         f"benchmark/{arm}/peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            write_json(args.output_dir / "progress.json", report)
            print(json.dumps({"arm": arm, **row, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
        case["training_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        case["training_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        case["timing"] = timing_summary([row["seconds"] for row in case["rows"] if not row["warmup"]],
            words=1024, length=12, overhead_fraction=args.overhead_fraction)
        case["parameters_changed"] = case["initialization_sha256"] != common.canonical_parameter_sha256(model)
        if not case["parameters_changed"]:
            raise AssertionError("Profile optimizer did not change parameters")
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            torch.cuda.synchronize()
            started = time.perf_counter()
            evaluation = logits_for(model, eval_inputs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if evaluation.dtype != torch.float32 or not bool(torch.isfinite(evaluation).all()):
                raise AssertionError("Length36 evaluation must produce finite FP32 logits")
            case["evaluation_length36"] = {"shape": [1024, 36], "seconds": seconds,
                "metrics": metrics_for(evaluation, eval_labels),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        tracker.summary(scalar_metrics(case["timing"], f"profile/{arm}"))
        del evaluation, output, optimizer, model
        torch.cuda.empty_cache()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("checks", "overfit", "profile"), required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=20260911)
    result.add_argument("--arms", nargs="+", choices=("seq", "rt"), default=["seq", "rt"])
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--timed-updates", type=int, default=10)
    result.add_argument("--overhead-fraction", type=float, default=.15)
    result.add_argument("--overfit-length", type=int, choices=(3, 4), default=3)
    result.add_argument("--overfit-updates", type=int, default=1000)
    add_wandb_arguments(result)
    result.set_defaults(wandb_project="rt-a5-state-tracking")
    return result


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    if not 1 <= args.warmup <= 50 or not 2 <= args.timed_updates <= 50 or not 1 <= args.overfit_updates <= 1000:
        argument_parser.error("Bounded limits: warmup1–50, timed2–50, overfit1–1000")
    if not args.wandb_project or len(set(args.arms)) != len(args.arms):
        argument_parser.error("Online W&B and distinct arms are required")
    if not math.isfinite(args.overhead_fraction) or args.overhead_fraction < 0:
        argument_parser.error("Overhead fraction must be finite and nonnegative")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh validation output directory")
    hardware = require_cuda_container()
    runtime_flags = common.configure_fp32_runtime()
    args.output_dir.mkdir(parents=True)
    report = {"schema": "rt-a5-validation-v1", "status": "running", "mode": args.mode,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "runtime": {"hardware": hardware, "python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "precision_flags": runtime_flags},
        "execution": {"dtype": "float32", "autocast": False, "tf32": False,
                      "compile": False, "cuda_graphs": False, "ordinary_sdpa": "math"},
        "source_sha256": source_snapshot(args.output_dir)}
    tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
        group=args.wandb_group, name=args.wandb_run_name or f"a5-{args.mode}",
        output_dir=args.output_dir, preserve_state=preserve_rng)
    report["wandb"] = tracker.record
    started = time.monotonic()
    try:
        tracker.start({"evidence": "bounded-a5-validation", "mode": args.mode,
                       "seed": args.seed, "execution": report["execution"]})
        {"checks": run_checks, "overfit": run_overfit, "profile": run_profile}[args.mode](args, report, tracker)
        if torch.cuda.is_current_stream_capturing():
            raise AssertionError("Unexpected CUDA graph capture")
        for relative, expected in report["source_sha256"].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise AssertionError(f"Source changed during validation: {relative}")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            tracker.finish(succeeded=report["status"] == "complete")
        except BaseException as error:
            report.update(status="failed", tracking_error_type=type(error).__name__)
            raise
        finally:
            write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
