#!/usr/bin/env python3
"""Checkpoint-bound development report for the first D128 Fuzzy-only pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    report = json.loads((args.train / "report.json").read_text())
    if report["status"] not in ("complete", "stopped"):
        raise ValueError("Require a completed or explicitly stopped pilot")
    evaluations = report["evaluations"]
    if not evaluations or evaluations[-1]["update"] != report["completed_updates"]:
        raise ValueError("Endpoint evaluation is missing")
    checkpoints = {p["completed_updates"]: p for p in report["checkpoints"]}
    endpoint = report["completed_updates"]
    if endpoint not in checkpoints:
        raise ValueError("Endpoint checkpoint is missing")
    cp = args.train / "checkpoints" / f"step-{endpoint:06d}.pt"
    if sha(cp) != checkpoints[endpoint]["sha256"]:
        raise ValueError("Endpoint checkpoint hash mismatch")
    data = json.loads((args.data / "manifest.json").read_text())
    if sha(args.data / "manifest.json") != report["contract"]["preparation_manifest_sha256"]:
        raise ValueError("Report data manifest differs from the training contract")
    baseline = data["splits"]["dev"]["baselines"]
    rows = [json.loads(line) for line in (args.train / "history.jsonl").read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.array([r["update"] for r in evaluations])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for key, label in (("answer_accuracy", "All native answer tokens"),
                       ("first_value_token_accuracy", "First value tokens"),
                       ("terminal_probe_accuracy", "Terminal probes")):
        axes[0, 0].plot(x, [100*r[key] for r in evaluations], marker="o", ms=3, label=label)
    axes[0, 0].axhline(100*baseline["query_ignoring_answer_prefix"]["answer_accuracy"],
                      color="0.5", ls="--", label="Query-ignoring prefix shortcut (all answers)")
    axes[0, 0].set(ylabel="Development accuracy (%)", ylim=(0, 102))
    axes[0, 0].legend(fontsize=8)
    for key, label in (("sequence_exact_match", "All answers in a sequence"),
                       ("answer_motif_exact_match", "Individual answer motifs")):
        axes[0, 1].plot(x, [100*r[key] for r in evaluations], marker="o", ms=3, label=label)
    axes[0, 1].set(ylabel="Teacher-forced exact match (%)", ylim=(0, 102))
    axes[0, 1].legend(fontsize=8)
    windows = [rows[i:i+50] for i in range(0, len(rows), 50)]
    tx = [w[-1]["update"] for w in windows]
    axes[1, 0].plot(tx, [np.mean([r["ce"] for r in w]) for w in windows], label="Train dense CE (50-update mean)")
    axes[1, 0].plot(x, [r["ce"] for r in evaluations], marker="o", ms=3, label="Dev masked answer CE")
    axes[1, 0].set(ylabel="Cross entropy (different supervision scopes)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(tx, [np.mean([r["latent"] for r in w]) for w in windows], label="Train NextLat")
    axes[1, 1].plot(x, [r["latent"] for r in evaluations], marker="o", ms=3, label="Dev NextLat diagnostic")
    axes[1, 1].set(ylabel="Mean SmoothL1 latent loss")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Optimizer updates (128 examples/update)")
        ax.grid(alpha=.2)
    fig.suptitle("Fuzzy Recall T400 — D128, two-layer restricted-first RT + NextLat\nOne seed; development results; backbone evaluation")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"learning-curves.{suffix}", dpi=160)
    plt.close(fig)
    final = evaluations[-1]
    distance = [key[:-9] for key in final if key.startswith("distance_") and key.endswith("_accuracy")
                and final[key] is not None]
    distance.sort(key=lambda name: int(name.split("_")[1]))
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.bar(np.arange(len(distance)), [100*final[key+"_accuracy"] for key in distance])
    ax.set_xticks(np.arange(len(distance)), [key.replace("distance_", "").replace("_", "–") for key in distance])
    ax.set(ylabel="Native answer-token accuracy (%)", xlabel="Distance to latest matching key start (tokens)",
           ylim=(0, 102), title=f"Length 400 retrieval at update {endpoint:,}")
    for i, key in enumerate(distance):
        ax.text(i, 2, f'n={final[key+"_tokens"]:,}', ha="center", va="bottom", fontsize=8, rotation=90)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"retrieval-distance.{suffix}", dpi=160)
    plt.close(fig)
    summary = {"schema": "rt-nextlat-fuzzy-report-v1", "status": "complete",
               "endpoint": endpoint, "checkpoint": checkpoints[endpoint], "final": final,
               "evaluations": evaluations, "baselines": baseline, "model": report["contract"]["model_config"],
               "parameters": report["parameter_count"], "training_wandb": report["wandb"],
               "training_report_sha256": sha(args.train / "report.json"),
               "data_manifest_sha256": sha(args.data / "manifest.json"),
               "train_seconds": report["train_seconds"], "elapsed_seconds": report["elapsed_seconds"],
               "confirmation_evaluated": False, "source_code": str(Path(__file__).resolve()),
               "source_sha256": sha(Path(__file__))}
    metrics = [("Native answer tokens", "answer_accuracy"), ("First value tokens", "first_value_token_accuracy"),
               ("Terminal probe tokens", "terminal_probe_accuracy"), ("Answer motifs exact", "answer_motif_exact_match"),
               ("All answers in a sequence exact", "sequence_exact_match")]
    lines = ["# Width-128 Fuzzy Recall pilot", "",
             f"Completed {endpoint:,} updates on native MAD Fuzzy Recall at sequence length 400.", "",
             "Two RT layers: first window two, second full recurrent. D128/H16/GELU FFN512; "
             "Mitchell, ALiBi, FP32 eager; NextLat predictor hidden128, weight one. "
             f"{report['parameter_count']:,} total training parameters. No embedding bypass.", "",
             f"Logical batch128, LR1e-4, {endpoint*128:,} examples; 12,800 train and 1,280 development examples. "
             "One seed; final confirmation remains unevaluated.", "", "| Development metric | Endpoint |", "| --- | ---: |"]
    lines += [f"| {label} | {100*final[key]:.3f}% |" for label, key in metrics]
    lines += ["", f"Query-ignoring answer-prefix shortcut: {100*baseline['query_ignoring_answer_prefix']['answer_accuracy']:.3f}% "
              "on the same native answer positions. Uniform value-token chance is12.5%.",
              f"Causal lookup availability: {100*final['oracle_coverage']:.4f}%; this is not a universal statistical ceiling.", "",
              "Training CE includes all native positions; development CE scores recall answers. "
              "These losses have different scopes. Exact match is teacher-forced, not autonomous generation.", "",
              f"Checkpoint: step-{endpoint:06d}.pt; SHA256 `{checkpoints[endpoint]['sha256']}`.", "",
              f"[Training W&B]({report['wandb'].get('run_url')})", "",
              "![Learning curves](learning-curves.png)", "", "![Retrieval distance](retrieval-distance.png)", "",
              "Stop for review before A5, joint training, other widths or additional sequence lengths."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    if args.wandb:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        from scripts.rt_a5_train import preserve_rng
        tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", output_dir=args.output,
                                name=f"d128-t400-report-{endpoint}", preserve_state=preserve_rng)
        try:
            tracker.start({"scope": "Report of completed development pilot", "endpoint": endpoint})
            tracker.log({"report/learning_curves": wandb.Image(str(args.output / "learning-curves.png")),
                         "report/retrieval_distance": wandb.Image(str(args.output / "retrieval-distance.png"))})
            tracker.summary({"answer_accuracy": final["answer_accuracy"], "sequence_exact_match": final["sequence_exact_match"]})
            tracker.finish(succeeded=True)
        except BaseException:
            tracker.finish(succeeded=False)
            raise
        summary["report_wandb"] = tracker.record
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "endpoint": endpoint, "answer_accuracy": final["answer_accuracy"],
                      "sequence_exact_match": final["sequence_exact_match"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
