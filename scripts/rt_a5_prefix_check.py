#!/usr/bin/env python3
"""Bounded forward-only check of trained A5 causal prefixes; RT then SEQ."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import (
    A5Metrics, build_model, canonical_parameter_sha256,
    configure_fp32_runtime, fp32_context,
)
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_train import (
    ROOT, SCHEMA, atomic_json, batch_tensors, file_sha256, json_sha256,
    json_value, preserve_rng, source_manifest, validate_model_state,
)
from scripts.stage_a_common import require_cuda_container

# Reuse the established A5 FP32 screen, not a new precision qualification.
ATOL, RTOL = 2e-6, 2e-5


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def metrics_for(logits, labels):
    metrics = A5Metrics()
    metrics.update(logits, labels)
    result = metrics.compute()
    result["state_correct_counts"] = metrics.correct.tolist()
    result["prefix_correct_counts"] = metrics.prefix_correct.tolist()
    return result


def compare_prefix(reference, actual, labels):
    """Check logits and decisions; return evidence even when a screen fails."""
    if reference.shape != actual.shape or labels.shape != actual.shape[:2]:
        raise ValueError("Mismatched prefix comparison shapes")
    if reference.dtype != torch.float32 or actual.dtype != torch.float32:
        raise TypeError("Prefix check requires FP32 logits")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise ValueError("Nonfinite prefix logits")
    error = actual - reference
    mismatch = error.abs() > ATOL + RTOL * reference.abs()
    ref_prediction, prediction = reference.argmax(-1), actual.argmax(-1)
    changed = prediction.ne(ref_prediction)
    ref_metrics, metrics = metrics_for(reference, labels), metrics_for(actual, labels)
    counts_equal = all(ref_metrics[key] == metrics[key] for key in
                       ("state_correct_counts", "prefix_correct_counts"))
    return {
        "passed": not bool(mismatch.any() or changed.any()) and counts_equal,
        "atol": ATOL, "rtol": RTOL, "coordinates": reference.numel(),
        "mismatched_logit_coordinates": int(mismatch.sum()),
        "max_absolute_logit_error": float(error.abs().max()),
        "relative_logit_l2": float(torch.linalg.vector_norm(error) /
                                   torch.linalg.vector_norm(reference).clamp_min(1e-12)),
        "prediction_disagreements": int(changed.sum()),
        "words_with_prediction_disagreements": int(changed.any(dim=1).sum()),
        "accuracy_counts_equal": counts_equal,
        "reference_metrics": ref_metrics, "truncated_metrics": metrics,
    }


def validate_contract(contract, runtime, hardware, model):
    for key in ("precision", "tf32", "compile", "cuda_graphs", "torch", "cuda"):
        if contract.get(key) != runtime[key]:
            raise ValueError(f"Historical execution contract differs: {key}")
    if contract.get("device_capability") != hardware["capability"]:
        raise ValueError("Device capability differs from historical contract")
    if contract["model_config"] != json_value(model.config):
        raise ValueError("Model configuration differs from retained checkpoint")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Prefix checks require a fresh output directory")
    if args.rows < 1:
        raise ValueError("Rows must be positive")
    run_dir, data_dir = Path(args.run_dir).resolve(), Path(args.data_dir).resolve()
    original = json.loads((run_dir / "report.json").read_text())
    if (original.get("schema") != SCHEMA or original.get("status") != "complete"
            or original.get("confirmation_evaluated") is not False
            or original["contract"]["architecture"] != args.architecture):
        raise ValueError("Expected a completed development-only A5 run for this architecture")
    endpoint = original["completed_updates"]
    checkpoint = run_dir / "checkpoints" / f"step-{endpoint:06d}.pt"
    recorded = [item for item in original["checkpoints"] if item["completed_updates"] == endpoint]
    if len(recorded) != 1 or file_sha256(checkpoint) != recorded[0]["sha256"]:
        raise ValueError("Endpoint checkpoint hash does not match original run")
    sources = source_manifest()
    contract = original["contract"]
    if sources != original["source_files"] or json_sha256(sources) != contract["source_sha256"]:
        raise ValueError("Current historical source differs from the recorded run")
    validate_manifest(data_dir)
    if file_sha256(data_dir / "manifest.json") != contract["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from recorded run")
    x, y = load_split(data_dir, "ood_dev")
    if args.rows > len(x):
        raise ValueError("Requested rows exceed the frozen OOD development set")

    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    # These are our own hash-verified retained checkpoints.
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (packet.get("schema") != SCHEMA or packet.get("contract") != contract
            or packet.get("completed_updates") != endpoint
            or packet.get("examples_seen") != endpoint * contract["batch_size"]):
        raise ValueError("Checkpoint packet differs from original report")
    model = build_model(args.architecture, width=contract["width"], seed=contract["seed"], device="cuda")
    validate_contract(contract, runtime, hardware, model)
    validate_model_state(packet["model"], model.state_dict())
    model.load_state_dict(packet["model"], strict=True)
    del packet
    model.eval()
    before_hash = canonical_parameter_sha256(model)
    lengths = [36, 12, 13, 14, 16] if args.architecture == "rt" else [14, 12, 13]
    reference_length = lengths[0]
    output.mkdir(parents=True)
    script_copy = output / Path(__file__).name
    shutil.copy2(__file__, script_copy)
    atomic_json(output / "historical-source-manifest.json", sources)
    fixture = output / "fixture.npz"
    np.savez_compressed(fixture, inputs=np.array(x[:args.rows], copy=True),
                        labels=np.array(y[:args.rows], copy=True),
                        row_indices=np.arange(args.rows, dtype=np.int64))
    inputs, labels = batch_tensors(x, y, slice(0, args.rows), "cuda")
    report = {
        "schema": "rt-a5-trained-prefix-v1", "status": "running",
        "started_at": utc_now(), "architecture": args.architecture,
        "completed_updates": endpoint, "contract": contract,
        "source_sha256": contract["source_sha256"],
        "data_manifest_sha256": contract["data_manifest_sha256"],
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "original_report": {"path": str(run_dir / "report.json"),
                            "sha256": file_sha256(run_dir / "report.json")},
        "new_source": {"path": script_copy.name, "sha256": file_sha256(script_copy)},
        "hardware": hardware, "runtime": runtime, "rows": args.rows, "role": "ood_dev",
        "fixture": {"path": fixture.name, "sha256": file_sha256(fixture),
                    "row_selection": f"first {args.rows} rows of frozen ood_dev"},
        "reference_length": reference_length, "forward_lengths": lengths,
        "confirmation_evaluated": False, "training_updates_performed": 0,
        "parameter_sha256_before": before_hash, "checks": {}, "forwards": {},
        "scope": "Same-word trained-checkpoint prefix check; subset accuracy is not the full development estimate.",
    }
    tracker = OnlineTracker(project="rt-a5-state-tracking", output_dir=output,
                            group=args.wandb_group, name=f"{args.architecture}-trained-prefix-{endpoint}",
                            preserve_state=preserve_rng)
    succeeded = False
    logits = {}
    try:
        tracker.start({"contract": contract, "lengths": lengths, "rows": args.rows,
                       "kind": "forward-only-trained-prefix-check"})
        report["wandb"] = tracker.record
        atomic_json(output / "report.json", report)
        print(json.dumps({"event": "started", "architecture": args.architecture,
                          "wandb": tracker.record["run_url"]}), flush=True)
        with torch.no_grad(), fp32_context("cuda"):
            for length in lengths:
                torch.cuda.synchronize()
                started = time.perf_counter()
                out = model(inputs[:, :length].contiguous()).logits
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                logits[length] = out.cpu()
                if out.dtype != torch.float32 or not bool(torch.isfinite(out).all()):
                    raise ValueError("Expected finite FP32 model output")
                report["forwards"][str(length)] = {"seconds": elapsed, "dtype": str(out.dtype)}
                del out
        labels_cpu = labels.cpu()
        report["reference_metrics"] = metrics_for(logits[reference_length], labels_cpu[:, :reference_length])
        for length in lengths[1:]:
            check = compare_prefix(logits[reference_length][:, :length], logits[length], labels_cpu[:, :length])
            report["checks"][str(length)] = check
            tracker.log({"length": length, "check/prediction_disagreements": check["prediction_disagreements"],
                         "check/max_absolute_logit_error": check["max_absolute_logit_error"],
                         "check/passed": check["passed"],
                         "check/whole_word_exact_match": check["truncated_metrics"]["whole_word_exact_match"]})
        report["parameter_sha256_after"] = canonical_parameter_sha256(model)
        report["parameters_unchanged"] = report["parameter_sha256_after"] == before_hash
        report["no_gradients_created"] = all(p.grad is None for p in model.parameters())
        artifact = output / "logits.npz"
        np.savez_compressed(artifact, **{f"length_{k}": v.numpy() for k, v in logits.items()})
        report["logits_artifact"] = {"path": artifact.name, "sha256": file_sha256(artifact)}
        report["passed"] = (all(c["passed"] for c in report["checks"].values())
                            and report["parameters_unchanged"] and report["no_gradients_created"])
        tracker.summary({"passed": report["passed"], "forward_count": len(lengths),
                         "parameters_unchanged": report["parameters_unchanged"]})
        if not report["passed"]:
            raise AssertionError("Trained prefix check failed; inspect saved logits and report")
        succeeded = True
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        try:
            tracker.finish(succeeded=succeeded)
            if succeeded:
                report["status"] = "complete"
        finally:
            report["finished_at"] = utc_now()
            report["wandb"] = tracker.record
            atomic_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "passed": report["passed"],
                      "architecture": args.architecture, "report": str(output / "report.json"),
                      "wandb": tracker.record["run_url"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("rt", "seq"), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--wandb-group")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
