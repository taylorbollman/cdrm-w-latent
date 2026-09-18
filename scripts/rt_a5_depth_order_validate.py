"""Bounded FP32 plumbing check for the fresh depth/order pair; GPU container only.

All synthetic updates are discarded. The reordered RT is checked against the
existing independent two-record oracle, followed by brief actual-shape updates.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time
from types import MethodType

import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_data import generate_unique_words, prefix_labels
from scripts.rt_a5_depth_order import build_model
from scripts.rt_a5_depth_order_train import source_manifest, train_step
from scripts.rt_a5_nextlat import nextlat_objective
from scripts.rt_a5_train import atomic_json, file_sha256, json_sha256, preserve_rng
from scripts.rt_a5_validate import compare_tensors, finite_state
from scripts.rt_a5_window_reference import direct_window2
from scripts.stage_a_common import require_cuda_container

ROOT = Path(__file__).resolve().parents[1]


def fixture(batch, length, seed):
    words = generate_unique_words(batch, length, seed=seed)
    return (torch.tensor(words, dtype=torch.long, device="cuda"),
            torch.tensor(prefix_labels(words), dtype=torch.long, device="cuda"))


def compare_first_window(length):
    x, y = fixture(2, length, 2234 + length)
    packets = {}
    for route in ("independent", "tiled"):
        model = build_model("rt_window2_first", width=128, device="cuda",
                            backend="naive" if route == "independent" else "tiled")
        if route == "independent":
            block = model.backbone.transformer.blocks[0]
            block._real_forward = MethodType(direct_window2, block)
        with fp32_context("cuda"):
            result = nextlat_objective(model, x, y)
            result["loss"].backward()
        state = finite_state(model, require_gradients=True)
        if not state["passed"]:
            raise AssertionError(f"Missing/nonfinite gradients in {route} at T{length}")
        packets[route] = {key: result[key].detach().cpu() for key in
                          ("logits", "loss", "state_loss", "latent_loss")}
        packets[route]["gradients"] = {name: p.grad.detach().cpu().clone()
                                       for name, p in model.named_parameters()}
        del model, result
    reference, actual = packets["independent"], packets["tiled"]
    assert reference["gradients"].keys() == actual["gradients"].keys()
    checks = {key: compare_tensors(reference[key], actual[key]) for key in
              ("logits", "loss", "state_loss", "latent_loss")}
    gradients = {key: compare_tensors(reference["gradients"][key], actual["gradients"][key])
                 for key in reference["gradients"]}
    passed = all(v["passed"] for v in [*checks.values(), *gradients.values()])
    if not passed:
        raise AssertionError(f"First-window FP32 oracle comparison failed at T{length}: "
                             + json.dumps({"outputs": checks, "gradients": gradients}))
    return {"passed": passed, "batch": 2, "length": length, "width": 128,
            "scope": "Existing FP32 atol2e-6/rtol2e-5; independent first-window scan and ordinary-autograd full second layer",
            "outputs": checks, "gradients": gradients}


def actual_shape(variant, tracker):
    model = build_model(variant, device="cuda")
    initialization = model.nextlat_initialization
    optimizer = make_optimizer(model)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for update in range(1, 11):
        x, y = fixture(1024, 12, 2250 + update)
        start = time.perf_counter()
        values = train_step(model, optimizer, x, y, diagnostics=update in (1, 10))
        torch.cuda.synchronize()
        rows.append({"update": update, "seconds": time.perf_counter() - start, **values})
        tracker.log({"update": update, **{f"train/{variant}/{key}": value
                                         for key, value in values.items()}})
    state = finite_state(model, optimizer, require_gradients=True)
    if not state["passed"] or state["optimizer_steps"] != [10]:
        raise AssertionError(f"Actual-shape optimizer/gradient check failed for {variant}")
    x, _ = fixture(2, 36, 2270)
    changed = x.clone()
    changed[:, 18:] = (changed[:, 18:] + 1) % 60
    with torch.no_grad(), fp32_context("cuda"):
        original = model(x).logits
        alternate = model(changed).logits
    causal = compare_tensors(original[:, :18], alternate[:, :18])
    if original.shape != (2, 36, 60) or not torch.isfinite(original).all() or not causal["passed"]:
        raise AssertionError(f"Length36 output/causality check failed for {variant}")
    return {"passed": True, "variant": variant, "discarded_updates": 10,
            "batch": 1024, "length": 12, "width": 512,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_tensors": len(list(model.parameters())),
            "initialization": initialization, "experiment_config": model.experiment_config,
            "finite_state": state, "causal_length36": causal, "rows": rows,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--wandb-group", required=True)
    args = parser.parse_args()
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = source_manifest()
    for path in (Path(__file__), ROOT / "scripts/rt_a5_validate.py",
                 ROOT / "scripts/rt_a5_window_reference.py"):
        sources[str(path.resolve().relative_to(ROOT))] = file_sha256(path)
    for relative in sources:
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    report = {"schema": "rt-a5-depth-order-validation-v1", "status": "running",
              "scope": "Discarded synthetic fixtures; bounded FP32 checks only",
              "hardware": hardware, "runtime": runtime, "source_files": sources,
              "source_sha256": json_sha256(sources), "confirmation_evaluated": False,
              "latent_rollout_evaluated": False}
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman",
                            output_dir=out, group=args.wandb_group,
                            name="nextlat-depth-order-bounded-preflight", preserve_state=preserve_rng)
    try:
        tracker.start({"scope": report["scope"], **runtime})
        report["first_window_oracle"] = {}
        for length in (12, 36):
            report["first_window_oracle"][str(length)] = compare_first_window(length)
            atomic_json(out / "report.json", report)
            gc.collect()
            torch.cuda.empty_cache()
        report["actual_shape"] = {}
        for variant in ("seq4_alibi", "rt_window2_first"):
            report["actual_shape"][variant] = actual_shape(variant, tracker)
            atomic_json(out / "report.json", report)
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "passed"
        tracker.summary({"passed": True})
        tracker.finish(succeeded=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        tracker.finish(succeeded=False)
        raise
    finally:
        report["wandb"] = tracker.record
        atomic_json(out / "report.json", report)
    print(json.dumps({"status": report["status"], "wandb": report["wandb"]["run_url"]}), flush=True)


if __name__ == "__main__":
    main()
