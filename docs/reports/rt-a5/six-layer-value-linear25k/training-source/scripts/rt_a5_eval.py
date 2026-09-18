#!/usr/bin/env python3
"""Evaluate retained A5 checkpoints on development data; confirmation stays sealed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import build_model, configure_fp32_runtime
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_train import (
    SCHEMA, atomic_json, evaluate_arrays, file_sha256, flatten_eval,
    json_sha256, preserve_rng, source_manifest, validate_model_state,
)
from scripts.stage_a_common import require_cuda_container


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--role", choices=("dev", "ood_dev"), default="ood_dev")
    parser.add_argument("--rows", type=int, default=102400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--wandb-project", default="rt-a5-state-tracking")
    parser.add_argument("--wandb-group")
    args = parser.parse_args()
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Evaluation output must be fresh")
    packet = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if packet.get("schema") != SCHEMA:
        raise ValueError("Not an A5 training checkpoint")
    contract = packet["contract"]
    validate_manifest(args.data_dir)
    if file_sha256(Path(args.data_dir) / "manifest.json") != contract["data_manifest_sha256"]:
        raise ValueError("Evaluation data differs from the recorded task lineage")
    if json_sha256(source_manifest()) != contract["source_sha256"]:
        raise ValueError("Evaluation source differs from the checkpoint source snapshot")
    model = build_model(contract["architecture"], width=contract["width"], seed=contract["seed"], device="cuda")
    validate_model_state(packet["model"], model.state_dict())
    model.load_state_dict(packet["model"], strict=True)
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Expected finite FP32 checkpoint parameters")
    x, y = load_split(args.data_dir, args.role)
    output.mkdir(parents=True)
    tracker = OnlineTracker(project=args.wandb_project, output_dir=output, group=args.wandb_group,
                            name=f"eval-{contract['architecture']}-{packet['completed_updates']}-{args.role}",
                            preserve_state=preserve_rng)
    succeeded = False
    report = {"status": "running", "checkpoint_sha256": file_sha256(args.checkpoint),
              "completed_updates": packet["completed_updates"], "contract": contract,
              "role": args.role, "hardware": hardware, "runtime": runtime,
              "confirmation_evaluated": False}
    try:
        tracker.start(contract)
        result = evaluate_arrays(model, x, y, batch_size=args.batch_size, limit=args.rows)
        report.update(metrics=result)
        tracker.log({"update": packet["completed_updates"], **flatten_eval(result, args.role)})
        tracker.summary({"role": args.role, "rows": result["rows"], "ce": result["ce"]})
        succeeded = True
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        try:
            tracker.finish(succeeded=succeeded)
            if succeeded:
                report["status"] = "complete"
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            report["wandb"] = tracker.record
            atomic_json(output / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
