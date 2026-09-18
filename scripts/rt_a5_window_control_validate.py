"""Bounded check of removing NextLat from the already validated first-window RT."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer, task_loss
from scripts.rt_a5_data import generate_unique_words, prefix_labels
from scripts.rt_a5_depth_order import build_model as build_reference
from scripts.rt_a5_window_control import build_model
from scripts.rt_a5_window_control_train import source_manifest, train_step
from scripts.rt_a5_train import atomic_json, file_sha256, json_sha256, preserve_rng
from scripts.rt_a5_validate import compare_tensors, finite_state
from scripts.stage_a_common import require_cuda_container

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / ".runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/train-window-first"


def fixture(batch, length, seed):
    words = generate_unique_words(batch, length, seed=seed)
    return (torch.tensor(words, dtype=torch.long, device="cuda"),
            torch.tensor(prefix_labels(words), dtype=torch.long, device="cuda"))


def compare_ce(length):
    """Check bare-model CE against backbone-only CE through the frozen wrapper."""
    x, y = fixture(2, length, 3210 + length)
    packets = {}
    for route in ("reference", "control"):
        model = (build_reference("rt_window2_first", width=128, device="cuda")
                 if route == "reference" else build_model(width=128, device="cuda"))
        with fp32_context("cuda"):
            logits = model(x).logits
            loss = task_loss(logits, y)
            loss.backward()
        backbone = model.backbone if route == "reference" else model
        state = finite_state(backbone, require_gradients=True)
        assert state["passed"], (route, length)
        if route == "reference":
            assert all(p.grad is None for p in model.predictor.parameters())
        packets[route] = {"logits": logits.detach().cpu(), "loss": loss.detach().cpu(),
                          "gradients": {n: p.grad.detach().cpu().clone() for n, p in backbone.named_parameters()}}
        del backbone, model, logits, loss
    ref, actual = packets["reference"], packets["control"]
    assert ref["gradients"].keys() == actual["gradients"].keys()
    outputs = {k: compare_tensors(ref[k], actual[k]) for k in ("logits", "loss")}
    gradients = {k: compare_tensors(ref["gradients"][k], actual["gradients"][k]) for k in ref["gradients"]}
    assert all(r["passed"] for r in [*outputs.values(), *gradients.values()])
    return {"passed": True, "batch": 2, "width": 128, "length": length,
            "scope": "Same tiled backbone and pure CE through frozen NextLat wrapper versus predictor-free model; inherited FP32 tolerance",
            "outputs": outputs, "gradients": gradients}


def primary_shape(tracker):
    model = build_model(device="cuda")
    ref_path = REFERENCE / "checkpoints/step-000000.pt"
    ref_report = json.loads((REFERENCE / "report.json").read_text())
    ref_record = next(r for r in ref_report["checkpoints"] if r["completed_updates"] == 0)
    assert file_sha256(ref_path) == ref_record["sha256"]
    saved = torch.load(ref_path, map_location="cpu", weights_only=False)
    expected = {k.removeprefix("backbone."): v for k, v in saved["model"].items() if k.startswith("backbone.")}
    assert model.state_dict().keys() == expected.keys()
    assert all(torch.equal(v.detach().cpu(), expected[k]) for k, v in model.state_dict().items())
    assert len(list(model.parameters())) == 21 and sum(p.numel() for p in model.parameters()) == 6357504
    assert not hasattr(model, "predictor")
    initialization = model.a5_initialization
    optimizer = make_optimizer(model)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters()}
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for update in range(1, 11):
        x, y = fixture(1024, 12, 3250 + update)
        start = time.perf_counter()
        values = train_step(model, optimizer, x, y)
        torch.cuda.synchronize()
        rows.append({"update": update, "seconds": time.perf_counter() - start, **values})
        tracker.log({"update": update, **{f"train/{k}": v for k, v in values.items()}})
    state = finite_state(model, optimizer, require_gradients=True)
    assert state["passed"] and state["optimizer_steps"] == [10]
    x, _ = fixture(2, 36, 3290)
    changed = x.clone()
    changed[:, 18:] = (changed[:, 18:] + 1) % 60
    with torch.no_grad(), fp32_context("cuda"):
        original = model(x).logits
        alternate = model(changed).logits
    causal = compare_tensors(original[:, :18], alternate[:, :18])
    assert original.shape == (2, 36, 60) and torch.isfinite(original).all() and causal["passed"]
    return {"passed": True, "discarded_updates": 10, "batch": 1024, "length": 12, "width": 512,
            "parameter_count": 6357504, "parameter_tensors": 21,
            "initialization": initialization, "experiment_config": model.experiment_config,
            "reference_step0_sha256": ref_record["sha256"], "all_initial_backbone_tensors_exact": True,
            "predictor_absent": True, "finite_state": state, "causal_length36": causal,
            "rows": rows, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-group", required=True)
    args = parser.parse_args()
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = source_manifest()
    sources[str(Path(__file__).relative_to(ROOT))] = file_sha256(Path(__file__))
    sources["scripts/rt_a5_validate.py"] = file_sha256(ROOT / "scripts/rt_a5_validate.py")
    for relative in sources:
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    report = {"schema": "rt-a5-window-control-validation-v1", "status": "running",
              "scope": "Predictor removal and pure-CE equivalence; ten discarded actual-shape updates; no new precision study",
              "hardware": hardware, "runtime": runtime, "source_files": sources,
              "source_sha256": json_sha256(sources), "confirmation_evaluated": False,
              "latent_rollout_evaluated": False}
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=out,
                            group=args.wandb_group, name="rt-window-first-no-nextlat-bounded-preflight",
                            preserve_state=preserve_rng)
    try:
        tracker.start({"scope": report["scope"], **runtime})
        report["pure_ce_equivalence"] = {}
        for length in (12, 36):
            report["pure_ce_equivalence"][str(length)] = compare_ce(length)
            gc.collect()
            torch.cuda.empty_cache()
        report["actual_shape"] = primary_shape(tracker)
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
