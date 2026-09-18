"""Bounded FP32 correctness and actual-shape smoke for the combined A5 variant.

Run only in the GPU container. Updates use discarded synthetic fixtures.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_data import generate_unique_words, prefix_labels
from scripts.rt_a5_nextlat import nextlat_objective
from scripts.rt_a5_nextlat_variant import build_variant_model
from scripts.rt_a5_nextlat_variant_train import source_manifest, train_step
from scripts.rt_a5_train import atomic_json, file_sha256, json_sha256, preserve_rng
from scripts.stage_a_common import require_cuda_container
from rt_a5_validate import compare_tensors, finite_state

ROOT = Path(__file__).resolve().parents[1]


def fixture(batch, length, seed):
    words = generate_unique_words(batch, length, seed=seed)
    return (torch.tensor(words, dtype=torch.long, device="cuda"),
            torch.tensor(prefix_labels(words), dtype=torch.long, device="cuda"))


def gradient_packet(model):
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
            if p.grad is not None}


def check_small(length):
    x, y = fixture(2, length, 1254 + length)
    packets, checks = {}, {}
    for backend in ("naive", "tiled"):
        model = build_variant_model(width=128, backend=backend, device="cuda")
        with fp32_context("cuda"):
            result = nextlat_objective(model, x, y)
            result["loss"].backward()
        state = finite_state(model, require_gradients=True)
        if not state["passed"]:
            raise AssertionError(f"Nonfinite or missing gradients: {backend}/T{length}")
        packets[backend] = {k: result[k].detach().cpu() for k in
                            ("logits", "loss", "state_loss", "latent_loss")}
        packets[backend]["gradients"] = gradient_packet(model)
        # Change only the suffix, holding shape constant. This exercises the
        # causal no-ALiBi tiled path without length-dependent GEMM comparisons.
        suffix = x.clone()
        cut = length // 2
        suffix[:, cut:] = (suffix[:, cut:] + 1) % 60
        with torch.no_grad(), fp32_context("cuda"):
            a = model(x).logits[:, :cut]
            b = model(suffix).logits[:, :cut]
        checks[backend] = {"finite": state, "causal_prefix": compare_tensors(a, b)}
        if not checks[backend]["causal_prefix"]["passed"]:
            raise AssertionError("Future operations affected an earlier prediction")
        del model, result
    a, b = packets["naive"], packets["tiled"]
    if a["gradients"].keys() != b["gradients"].keys():
        raise AssertionError("Naive/tiled gradient coverage differs")
    comparison = {k: compare_tensors(a[k], b[k]) for k in
                  ("logits", "loss", "state_loss", "latent_loss")}
    comparison["gradients"] = {n: compare_tensors(a["gradients"][n], b["gradients"][n])
                               for n in a["gradients"]}
    passed = (all(comparison[k]["passed"] for k in
                  ("logits", "loss", "state_loss", "latent_loss"))
              and all(v["passed"] for v in comparison["gradients"].values()))
    return {"passed": passed, "batch": 2, "width": 128, "length": length,
            "scope": "same-state combined loss; inherited atol2e-6 rtol2e-5",
            "backends": checks, "comparison": comparison}


def actual_shape(tracker):
    x, y = fixture(1024, 12, 1264)
    model = build_variant_model(device="cuda")
    optimizer = make_optimizer(model)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for update in range(1, 26):
        # Distinct valid words across updates; labels are recomputed consistently.
        if update in (1, 10, 20):
            x, y = fixture(1024, 12, 1264 + update)
        start = time.perf_counter()
        values = train_step(model, optimizer, x, y, diagnostics=update % 5 == 0)
        torch.cuda.synchronize()
        rows.append({"update": update, "seconds": time.perf_counter() - start, **values})
        if update % 5 == 0:
            tracker.log({"update": update, **{f"train/{k}": v for k, v in values.items()}})
    state = finite_state(model, optimizer, require_gradients=True)
    if not state["passed"]:
        raise AssertionError("Actual B1024/D512 joint-update state invalid")
    return {"passed": True, "batch": 1024, "length": 12, "width": 512,
            "discarded_updates": 25, "rows": rows, "finite_state": state,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--wandb-group", required=True)
    args = p.parse_args()
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    torch.manual_seed(1234)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = source_manifest()
    for path in (Path(__file__), ROOT / "scripts/rt_a5_validate.py"):
        sources[str(path.resolve().relative_to(ROOT))] = file_sha256(path)
    for relative in sources:
        dest = out / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    report = {"schema": "rt-a5-nextlat-variant-validation-v1", "status": "running",
              "hardware": hardware, "runtime": runtime, "source_files": sources,
              "source_sha256": json_sha256(sources), "confirmation_evaluated": False,
              "scope": "Discarded FP32 fixtures; no numerical qualification expansion"}
    tracker = OnlineTracker(project="rt-a5-state-tracking", output_dir=out,
                            group=args.wandb_group, name="identity-sinusoidal-preflight",
                            preserve_state=preserve_rng)
    try:
        tracker.start({"scope": report["scope"], **runtime})
        report["small"] = {}
        for length in (12, 36):
            result = check_small(length)
            report["small"][str(length)] = result
            atomic_json(out / "report.json", report)
            if not result["passed"]:
                raise AssertionError(f"T{length} combined FP32 comparison failed")
            print(json.dumps({"event": "small_check", "length": length, "passed": True}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        report["actual_shape"] = actual_shape(tracker)
        report["status"] = "passed"
        tracker.summary({"passed": True})
        tracker.finish(succeeded=True)
    except BaseException:
        report["status"] = "failed"
        tracker.finish(succeeded=False)
        raise
    finally:
        report["wandb"] = tracker.record
        atomic_json(out / "report.json", report)
    print(json.dumps({"event": "preflight_complete", "status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()
