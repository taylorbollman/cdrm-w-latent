#!/usr/bin/env python3
"""Compare the matched FP32/BF16 L1R A5 runs using saved CPU artifacts only."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_bf16 import PRECISION_CONTRACT
from scripts.rt_a5_depth_order_report import validate_contract as validate_original_contract
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, write_json


SCHEMA = "rt-a5-l1r-bf16-comparison-v1"
ROOT = Path(__file__).resolve().parents[1]
ADDITIONS = {"scripts/rt_a5_bf16.py", "scripts/rt_a5_bf16_train.py"}
STEPS = (1000, 5000, 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000)
QUALIFICATION = (
    "One paired development seed; precision is the intended change, not a replicated equivalence result. "
    "Both arms use the same D512 two-layer restricted-first RT + NextLat, initial tensors, optimizer, "
    "A5 corpus and word order. BF16 uses protected bf16_fp32_state with FP32 parameters/gradients/Adam. "
    "Evaluation uses each arm's native execution precision. Routine evaluations use 4096 words; "
    "full checkpoints use 102400. Compare full measurements at common optimizer updates. "
    "An unmatched early-stop endpoint is descriptive. Final confirmation and latent rollout remain unused."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_contracts(baseline, mixed):
    validate_original_contract(baseline, "rt_window2_first")
    require(mixed.get("schema") == "rt-a5-bf16-training-v1"
            and mixed.get("precision") == "bf16_mixed"
            and mixed.get("precision_contract") == PRECISION_CONTRACT,
            "Unexpected BF16 precision contract")
    functions = {
        "training_step": "scripts.rt_a5_bf16.train_step",
        "evaluation": "scripts.rt_a5_bf16.evaluate_arrays; unchanged A5Metrics on FP32 logits",
        "one_step_diagnostics": "scripts.rt_a5_bf16.evaluate_diagnostics",
    }
    require(all(mixed.get(key) == value for key, value in functions.items()),
            "Unexpected precision execution functions")
    a, b = copy.deepcopy(baseline), copy.deepcopy(mixed)
    for key in ("schema", "precision", "source_sha256", *functions):
        a.pop(key)
        b.pop(key)
    b.pop("precision_contract")
    require(a["model_config"]["recurrent_precision_policy"] == "legacy"
            and b["model_config"]["recurrent_precision_policy"] == "bf16_fp32_state",
            "Unexpected recurrent precision policy")
    a["model_config"].pop("recurrent_precision_policy")
    b["model_config"].pop("recurrent_precision_policy")
    require(a == b, "Non-precision training contract differs")
    return {"passed": True, "allowed_changes": ["precision and protected recurrent policy",
            "additive source closure", "checkpoint schema", "explicit precision execution function provenance"]}


def check_configuration(baseline, mixed):
    ignored = {"output_dir", "wandb_group", "wandb_run_name", "stop_file"}
    require({key: value for key, value in baseline.items() if key not in ignored}
            == {key: value for key, value in mixed.items() if key not in ignored},
            "CLI differs beyond output/tracking/stop controls")
    require(baseline.get("resume") is None and mixed.get("resume") is None,
            "This comparison requires fresh step-zero runs")


def read_run(directory, label):
    directory = local_path(directory).resolve()
    report = json.loads((directory / "report.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    endpoint = report["completed_updates"]
    require(report["status"] in ("complete", "stopped") and report["start_update"] == 0
            and report["parent_checkpoint"] is None and 0 < endpoint <= 80000,
            "Expected a fresh terminal run through at most80k")
    require(report["endpoint"] == 80000
            and (report["status"] != "complete" or endpoint == 80000), "Unexpected training budget")
    require(report.get("confirmation_evaluated") is False and report.get("latent_rollout_evaluated") is False,
            "Confirmation or latent rollout was evaluated")
    require(report.get("wandb", {}).get("status") == "synced", "Training W&B has not synced")
    require(report["schema"] == ("rt-a5-depth-order-training-v1" if label == "fp32" else "rt-a5-bf16-training-v1"),
            "Unexpected training report schema")
    sources = report["source_files"]
    require(digest(sources) == report["contract"]["source_sha256"], "Source manifest differs")
    for relative, expected in sources.items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "Invalid source snapshot path")
        require(hash_file(directory / "source" / relative)["sha256"] == expected, "Frozen source changed")
    require(hash_file(local_path(config["data_dir"]) / "manifest.json")["sha256"]
            == report["contract"]["data_manifest_sha256"], "Dataset identity differs")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines() if line.strip()]
    require([row["update"] for row in history] == list(range(1, endpoint + 1)), "History has gaps or extra updates")
    for row in history:
        require(row["examples_seen"] == row["update"] * 1024, "Word exposure differs")
        require(all(math.isfinite(value) for value in row.values() if isinstance(value, float)),
                "Nonfinite training history")
    require(history[-1]["order_chain"] == report["order_chain"], "Terminal data order differs")
    checkpoints = {}
    for record in report["checkpoints"]:
        update = record["completed_updates"]
        require(update not in checkpoints and 0 <= update <= endpoint, "Duplicate or out-of-range checkpoint")
        path = directory / "checkpoints" / f"step-{update:06d}.pt"
        actual = hash_file(path)
        require(actual["sha256"] == record["sha256"] and actual["bytes"] == record["bytes"]
                and record["examples_seen"] == update * 1024, "Checkpoint identity or exposure differs")
        checkpoints[update] = {**record, "local_path": str(path)}
    require(set(checkpoints) == {0, endpoint, *(step for step in STEPS if step <= endpoint)},
            "Saved checkpoint schedule differs")
    rows, metrics = [], {}
    for observation in report["evaluations"]:
        key = (observation["update"], observation["role"])
        require(key not in metrics and 0 < key[0] <= endpoint
                and observation["rows"] in (4096, 102400), "Duplicate or invalid development evaluation")
        rows.extend(metric_rows(observation, label))
        metrics[key] = observation
    expected_eval_steps = set(range(500, endpoint + 1, 500)) | {step for step in STEPS if step <= endpoint} | {endpoint}
    require(set(metrics) == {(step, role) for step in expected_eval_steps for role in ("dev", "ood_dev")},
            "Development evaluation cadence differs")
    require(all(metrics[(step, role)]["rows"] == (102400 if step in checkpoints else 4096)
                for step, role in metrics), "Full and routine development pool sizes differ")
    return {"directory": str(directory), "report": report, "config": config, "endpoint": endpoint,
            "sources": sources, "history": history, "checkpoints": checkpoints, "metrics": metrics,
            "metric_rows": rows,
            "input_hashes": {name: hash_file(directory / name) for name in ("report.json", "config.json", "history.jsonl")}}


def initial_pairing(baseline, mixed):
    a, b = [torch.load(run["checkpoints"][0]["local_path"], map_location="cpu", weights_only=False)
            for run in (baseline, mixed)]
    require(a["initialization"] == b["initialization"] == baseline["report"]["initialization"]
            == mixed["report"]["initialization"], "Initial provenance differs")
    require(set(a["model"]) == set(b["model"]) and all(
        a["model"][name].dtype == b["model"][name].dtype == torch.float32
        and torch.equal(a["model"][name], b["model"][name])
        and torch.isfinite(a["model"][name]).all().item() for name in a["model"]),
        "Learned initial tensors differ")
    require(a["optimizer"]["state"] == b["optimizer"]["state"] == {}
            and a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
            and a["optimizer_parameter_names"] == b["optimizer_parameter_names"]
            and a["order_chain"] == b["order_chain"], "Initial Adam or data-order state differs")
    return {"passed": True, "exact_model_tensors": len(a["model"]),
            "parameter_count": sum(value.numel() for value in a["model"].values()),
            "fresh_identical_adam_groups": True,
            "checkpoints": {"fp32": baseline["checkpoints"][0], "bf16": mixed["checkpoints"][0]}}


def compare(baseline_train, train):
    base, mixed = read_run(baseline_train, "fp32"), read_run(train, "bf16")
    require(base["endpoint"] == 80000 and base["report"]["status"] == "complete", "FP32 reference must be complete80k")
    pairing = check_contracts(base["report"]["contract"], mixed["report"]["contract"])
    check_configuration(base["config"], mixed["config"])
    require(set(mixed["sources"]) - set(base["sources"]) == ADDITIONS
            and all(mixed["sources"].get(name) == value for name, value in base["sources"].items()),
            "Original training source closure changed")
    initial = initial_pairing(base, mixed)
    for left, right in zip(base["history"], mixed["history"]):
        require(left["order_chain"] == right["order_chain"], f"Data order differs at update{right['update']}")
    common = sorted(step for step in base["checkpoints"] if step > 0 and step in mixed["checkpoints"])
    latest = common[-1] if common else None
    arms = {}
    for label, run in (("fp32", base), ("bf16", mixed)):
        end = run["endpoint"]
        arms[label] = {"directory": run["directory"], "endpoint": end,
                       "status": run["report"]["status"], "wandb": run["report"]["wandb"],
                       "endpoint_metrics": {role: run["metrics"][(end, role)] for role in ("dev", "ood_dev")},
                       "matched_metrics": {str(step): {role: run["metrics"][(step, role)] for role in ("dev", "ood_dev")}
                                           for step in common},
                       "development_metrics": [value for (step, role), value in sorted(run["metrics"].items())
                                               if step <= mixed["endpoint"]],
                       "training_curve": training_curve(run["history"][:mixed["endpoint"]], True),
                       "train_seconds": run["report"]["train_seconds"],
                       "elapsed_seconds": run["report"]["elapsed_seconds"],
                       "matched_exposure_train_seconds": sum(row["seconds"] for row in run["history"][:mixed["endpoint"]]),
                       "checkpoints": {str(key): value for key, value in run["checkpoints"].items()},
                       "input_hashes": run["input_hashes"], "contract": run["report"]["contract"]}
    rows = [row for run in (base, mixed) for row in run["metric_rows"] if row["update"] <= mixed["endpoint"]]
    return {"schema": SCHEMA, "status": "complete", "bf16_endpoint": mixed["endpoint"],
            "common_full_updates": common, "latest_matched_update": latest, "qualification": QUALIFICATION,
            "contract_pairing": pairing, "initial_pairing": initial, "matched_data_order_updates": mixed["endpoint"],
            "confirmation_evaluated": False, "latent_rollout_evaluated": False, "arms": arms, "metric_rows": rows}


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"fp32": "#222222", "bf16": "#D97706"}
    figures = []
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, role in zip(axes, ("dev", "ood_dev")):
        for label, arm in summary["arms"].items():
            metrics = [m for m in arm["development_metrics"] if m["role"] == role]
            axis.plot([m["update"] for m in metrics], [100*m["whole_word_exact_match"] for m in metrics],
                      color=colors[label], alpha=.65, label=label.upper())
            full = [m for m in metrics if m["rows"] == 102400]
            axis.scatter([m["update"] for m in full], [100*m["whole_word_exact_match"] for m in full],
                         color=colors[label], s=15)
        axis.set(title="Length12" if role == "dev" else "Length36", xlabel="Optimizer updates",
                 ylabel="Whole-word accuracy (%)", ylim=(-2, 102))
        axis.legend(); axis.grid(alpha=.2)
    fig.suptitle("L1R RT + NextLat: FP32 versus protected BF16\nLines: routine 4,096; markers: full 102,400 development words")
    save(fig, "whole-word-vs-updates")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    step = summary["latest_matched_update"]
    for label, arm in summary["arms"].items():
        if step is None and label == "fp32": continue
        metric = arm["matched_metrics"][str(step)]["ood_dev"] if step else arm["endpoint_metrics"]["ood_dev"]
        positions = range(1, 37)
        axes[0].plot(positions, [100*x for x in metric["cumulative_prefix_exactness"]], label=label.upper(), color=colors[label])
        axes[1].plot(positions, [100*x for x in metric["isolated_state_accuracy"]], label=label.upper(), color=colors[label])
    for axis, title in zip(axes, ("Every state through position correct", "State at position correct")):
        axis.set(title=title, xlabel="Position in length36 word", ylabel="Accuracy (%)", ylim=(-2, 102))
        axis.legend(); axis.grid(alpha=.2)
    fig.suptitle(f"Matched full checkpoint at update {step:,}" if step else "BF16 stopping checkpoint; no common full checkpoint yet")
    save(fig, "length36-prefix")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for label, arm in summary["arms"].items():
        curve = arm["training_curve"]
        for axis, field in zip(axes, ("state_ce", "latent_loss")):
            axis.plot([r["update"] for r in curve], [r[field] for r in curve], label=label.upper(), color=colors[label])
    for axis, title in zip(axes, ("State cross-entropy", "NextLat loss")):
        axis.set(title=title, xlabel="Optimizer updates", ylabel="Loss")
        axis.legend(); axis.grid(alpha=.2)
    save(fig, "training-losses")
    return figures


def markdown(summary):
    state = "completed" if summary["arms"]["bf16"]["status"] == "complete" else "stopped"
    lines = ["# A5 L1R RT + NextLat: BF16 precision repeat", "",
             f"BF16 {state} at **{summary['bf16_endpoint']:,} updates**; FP32 reference completed 80,000. "
             f"Latest common full checkpoint: **{summary['latest_matched_update']}**.", "",
             "Both models use D512, two RT layers (first window2), B1024 and identical initial tensors/data order.", "",
             "| Update | Arm | Length12 whole word | Length36 whole word | Length36 token |",
             "|---:|---|---:|---:|---:|"]
    for step in summary["common_full_updates"]:
        for label, arm in summary["arms"].items():
            values = arm["matched_metrics"][str(step)]
            lines.append(f"| {step:,} | {label.upper()} | {100*values['dev']['whole_word_exact_match']:.4f}% | "
                         f"{100*values['ood_dev']['whole_word_exact_match']:.4f}% | {100*values['ood_dev']['token_accuracy']:.4f}% |")
    actual = summary["arms"]["bf16"]["endpoint_metrics"]
    lines += ["", f"At the actual BF16 endpoint, length-12 whole-word accuracy is "
              f"**{100*actual['dev']['whole_word_exact_match']:.4f}%**, and length-36 whole-word accuracy is "
              f"**{100*actual['ood_dev']['whole_word_exact_match']:.4f}%** (102,400 words each).", "",
              "| Arm | Actual endpoint | Training-loop minutes | Total minutes |", "|---|---:|---:|---:|"]
    for label, arm in summary["arms"].items():
        lines.append(f"| {label.upper()} | {arm['endpoint']:,} | {arm['train_seconds']/60:.2f} | {arm['elapsed_seconds']/60:.2f} |")
    lines += ["", "Training-loop time includes per-update diagnostics and excludes evaluations/checkpointing/logging. "
              "Unequal terminal budgets are not throughput comparisons. Matched-exposure timing is retained in report.json.",
              "", summary["qualification"], "", "[Learning curves](whole-word-vs-updates.pdf) · "
              "[Length36 prefix](length36-prefix.pdf) · [Training losses](training-losses.pdf) · [Metrics](metrics.csv)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-train", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    require(Path("/.dockerenv").is_file() and not torch.cuda.is_initialized(), "Use CPU-only project container")
    require(not args.output.exists(), "Report output already exists")
    summary = compare(args.baseline_train, args.train)
    args.output.mkdir(parents=True)
    for label, path in (("fp32", args.baseline_train), ("bf16", args.train)):
        shutil.copy2(path / "report.json", args.output / f"{label}-training-report.json")
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader(); writer.writerows(summary["metric_rows"])
    figures = plots(summary, args.output)
    (args.output / "README.md").write_text(markdown(summary))
    summary["figures"] = figures
    summary["reporting_sources"] = {}
    for name in ("scripts/rt_a5_bf16_report.py", "scripts/rt_a5_nextlat_report.py",
                 "scripts/rt_a5_report.py", "scripts/rt_a5_depth_order_report.py"):
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        summary["reporting_sources"][name] = hash_file(target)
    if args.wandb:
        tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=args.output,
                                group=args.wandb_group, name="l1r-nextlat-protected-bf16-comparison")
        try:
            tracker.start({"schema": SCHEMA, "bf16_endpoint": summary["bf16_endpoint"], "qualification": QUALIFICATION})
            import wandb
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png")) for name in figures})
            tracker.summary({"bf16_endpoint": summary["bf16_endpoint"], "latest_matched_update": summary["latest_matched_update"],
                             "endpoint_metrics": {label: arm["endpoint_metrics"] for label, arm in summary["arms"].items()}})
            tracker.finish(succeeded=True)
        except BaseException:
            tracker.finish(succeeded=False)
            raise
        summary["wandb"] = tracker.record
    summary["artifacts"] = {p.name: hash_file(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output / "report.json", summary)
    print(json.dumps({"status": "complete", "output": str(args.output), "bf16_endpoint": summary["bf16_endpoint"]}))


if __name__ == "__main__":
    main()
