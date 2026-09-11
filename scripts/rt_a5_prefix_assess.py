#!/usr/bin/env python3
"""Assess saved prefix decisions on CPU, retaining the original logit screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_prefix_check import compare_prefix, utc_now
from scripts.rt_a5_train import atomic_json, file_sha256


def run(args):
    if not Path("/.dockerenv").exists() or torch.cuda.is_available():
        raise RuntimeError("Run saved-output assessment in the explicitly GPU-disabled container")
    output, prefix = Path(args.output_dir).resolve(), Path(args.prefix_report).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh assessment output directory")
    original = json.loads(prefix.read_text())
    if original.get("schema") != "rt-a5-trained-prefix-v1":
        raise ValueError("Unexpected prefix report schema")
    if original["status"] not in ("complete", "failed"):
        raise ValueError("Prefix evaluation did not finish")
    if original["status"] == "failed" and original.get("error_type") != "AssertionError":
        raise ValueError("Only completed strict-screen outcomes can be assessed")
    if (original.get("parameters_unchanged") is not True or original.get("no_gradients_created") is not True
            or original.get("confirmation_evaluated") is not False):
        raise ValueError("Unexpected model mutation or evaluation scope")
    for key in ("fixture", "logits_artifact"):
        record = original[key]
        if file_sha256(prefix.parent / record["path"]) != record["sha256"]:
            raise ValueError(f"Saved {key} changed")
    labels = torch.from_numpy(np.load(prefix.parent / original["fixture"]["path"])["labels"].astype(np.int64))
    arrays = np.load(prefix.parent / original["logits_artifact"]["path"])
    reference = torch.from_numpy(arrays[f"length_{original['reference_length']}"])
    checks = {}
    for length, old in original["checks"].items():
        n = int(length)
        ref, actual = reference[:, :n], torch.from_numpy(arrays[f"length_{n}"])
        comparison = compare_prefix(ref, actual, labels[:, :n])
        if comparison != old:
            raise ValueError("Saved-output replay differs from the original comparison")
        ref_prob, actual_prob = ref.softmax(-1), actual.softmax(-1)
        top = ref.topk(2, dim=-1).values
        gap = top[..., 0] - top[..., 1]
        error = (ref - actual).abs().amax(-1)
        ref_prediction, prediction = ref.argmax(-1), actual.argmax(-1)
        target = labels[:, :n]
        correctness_disagreements = int((ref_prediction.eq(target) != prediction.eq(target)).sum())
        changed = prediction.ne(ref_prediction).nonzero()
        changed_details = [{"row_index": int(row), "position": int(pos) + 1,
                            "label": int(target[row, pos]),
                            "reference_prediction": int(ref_prediction[row, pos]),
                            "truncated_prediction": int(prediction[row, pos]),
                            "reference_top_two_margin": float(gap[row, pos])}
                           for row, pos in changed[:20]]
        # A margin exceeding twice the per-position infinity error certifies
        # that the observed logit perturbation cannot change the winner.
        certified = gap > 2 * error
        checks[length] = {
            "strict_logit_screen_passed": comparison["passed"],
            "prediction_disagreements": comparison["prediction_disagreements"],
            "accuracy_counts_equal": comparison["accuracy_counts_equal"],
            "correctness_disagreements": correctness_disagreements,
            "changed_prediction_details_first20": changed_details,
            "max_probability_error": float((ref_prob - actual_prob).abs().max()),
            "max_total_variation": float((ref_prob - actual_prob).abs().sum(-1).max() / 2),
            "min_top_two_margin": float(gap.min()),
            "max_2error_over_margin": float((2 * error / gap.clamp_min(1e-30)).max()),
            "uncertified_argmax_positions": int((~certified).sum()),
            "ce_difference": comparison["truncated_metrics"]["ce"] - comparison["reference_metrics"]["ce"],
        }
    # E/A/M depend only on the complete per-word correctness mask. Two
    # different wrong classes can change argmax while leaving every requested
    # metric unchanged. Aggregate equal counts alone are insufficient.
    usable = bool(checks) and all(c["correctness_disagreements"] == 0 and c["accuracy_counts_equal"]
                                for c in checks.values())
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    result = {
        "schema": "rt-a5-prefix-assessment-v1", "status": "running", "architecture": original["architecture"],
        "completed_updates": original["completed_updates"], "source_sha256": original["source_sha256"],
        "data_manifest_sha256": original["data_manifest_sha256"],
        "original_prefix_report_sha256": file_sha256(prefix), "original_prefix_report": str(prefix),
        "original_logit_screen_passed": original["passed"], "usable_for_length_metrics": usable,
        "checks": checks, "assessment_time": utc_now(), "confirmation_evaluated": False,
        "new_model_forwards": 0, "new_training_updates": 0,
        "source_files": {name: file_sha256(Path(__file__).parent / name) for name in
                         ("rt_a5_prefix_assess.py", "rt_a5_prefix_check.py")},
        "qualification": "This post-run assessment accepts only the observed E/A/M length metrics: every per-word, per-position correct/incorrect indicator and every integer accuracy count agrees. Wrong-class predictions may differ; all such changes and their margins are reported. Original strict logit-screen and prediction-agreement failures remain failures. Probability/loss changes and margin certificates are descriptive; no numerical tolerance was changed. No claim about all inputs, backward gradients or mixed precision.",
    }
    tracker = OnlineTracker(project="rt-a5-state-tracking", output_dir=output, group=args.wandb_group,
                            name=f"{original['architecture']}-prefix-metric-assessment")
    succeeded = False
    try:
        tracker.start({"kind": "saved-output-assessment", "architecture": original["architecture"],
                       "original_prefix_report_sha256": result["original_prefix_report_sha256"],
                       "qualification": result["qualification"]})
        for key, check in checks.items():
            tracker.log({"length": int(key), **{f"assessment/{k}": v for k, v in check.items()}})
        tracker.summary({"usable_for_length_metrics": usable,
                         "original_logit_screen_passed": original["passed"]})
        if not usable:
            raise AssertionError("Saved outputs do not justify accepting length metrics")
        succeeded = True
    finally:
        try:
            tracker.finish(succeeded=succeeded)
        finally:
            result["status"] = "complete" if succeeded and tracker.record["status"] == "synced" else "failed"
            result["wandb"] = tracker.record
            assert not torch.cuda.is_initialized()
            atomic_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "usable_for_length_metrics": usable,
                      "wandb": tracker.record["run_url"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
