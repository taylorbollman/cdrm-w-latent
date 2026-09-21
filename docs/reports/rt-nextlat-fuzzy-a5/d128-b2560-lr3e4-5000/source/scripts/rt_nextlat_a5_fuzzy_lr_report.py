#!/usr/bin/env python3
"""CPU-only comparison of a 5k LR probe with the retained 15k mixed baseline."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from scripts import rt_nextlat_a5_fuzzy_report as saved
from scripts import rt_nextlat_a5_fuzzy_embedding_compare as paired


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-a5-fuzzy-lr-comparison-v1"
TRAIN_SCHEMA = "rt-nextlat-a5-fuzzy-lr-training-v1"
DRIVER = "scripts/rt_nextlat_a5_fuzzy_lr_train.py"
QUALIFICATION = (
    "One matched initialization and reused development pools. Only the learning-rate recipe changes: "
    "FP32, D128, two-layer restricted-first RT, NextLat, batch2560 per task, task weights, data order "
    "and gradient clipping remain fixed. The new arm is a 5000-update acceleration probe. "
    "The original 1e-4 baseline succeeded with more training: length36 A5 whole-word accuracy was "
    "92.853515625% at15000. A zero result at5000 does not establish architectural failure. "
    "Common-update comparisons match evaluation pool sizes; monitoring subsets and full evaluations "
    "are identified separately. First observed threshold crossings are not sustained convergence. "
    "Training time excludes evaluations/checkpointing/reporting and is not a repeated throughput benchmark. "
    "No new model inference, final confirmation or autonomous latent rollout is performed by this report."
)


def require(value, message):
    saved.require(value, message)


def expected_lr(update):
    require(type(update) is int and update >= 1, "Learning rate requires a positive update")
    return 1e-4 + (3e-4 - 1e-4) * min((update - 1) / 99, 1.0)


def read_fresh(directory):
    """Validate the new explicit LR schema without modifying the legacy loader."""
    directory = saved.local_path(directory)
    report = saved.read_json(directory / "report.json")
    endpoint = report.get("completed_updates")
    require(report.get("schema") == TRAIN_SCHEMA and report.get("status") in ("complete", "stopped")
            and report.get("start_update") == 0 and report.get("parent_checkpoint") is None
            and type(endpoint) is int and 0 < endpoint <= 5000
            and report.get("requested_endpoint") == 5000
            and (report["status"] != "complete" or endpoint == 5000), "Require a fresh terminal 5k LR probe")
    require(report.get("confirmation_evaluated") is False and report.get("latent_rollout_evaluated") is False,
            "Confirmation and latent rollout must remain unused")
    contract = report["contract"]
    require(contract.get("schema") == TRAIN_SCHEMA, "Checkpoint contract schema differs")
    configuration = directory / "model-config.json"
    identity = saved.read_json(directory / "data-identity.json")
    sources = saved.read_json(directory / "source-manifest.json")
    require(saved.read_json(configuration) == contract["model_config"]
            and saved.sha(configuration) == contract["configuration_file_sha256"], "Saved model configuration differs")
    require(saved.json_sha(identity) == contract["data_sha256"] and saved.json_sha(sources) == contract["source_sha256"],
            "Dataset or source manifest identity differs")
    require(report["initialization"] == contract["initialization"], "Initialization metadata differs")
    for relative, expected in sources.items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Invalid source path")
        require(saved.sha(directory / "source" / relative) == expected, "Frozen source changed: " + relative)
    history, history_audit = saved.read_history_prefix(directory / "history.jsonl", start=0, through=endpoint, allow_tail=False)
    checkpoints = {}
    for item in report["checkpoints"]:
        update = item["completed_updates"]
        require(type(update) is int and 0 <= update <= endpoint and update not in checkpoints, "Invalid checkpoint update")
        path = directory / "checkpoints" / f"step-{update:06d}.pt"
        require(path.stat().st_size == item["bytes"] and saved.sha(path) == item["sha256"], "Checkpoint bytes differ")
        checkpoints[update] = {**item, "verified_local_path": str(path)}
    require(0 in checkpoints and endpoint in checkpoints, "Initial or terminal checkpoint missing")
    evaluations = saved._evaluations(report)
    keys = set()
    for metric in evaluations:
        update = metric["update"]
        size = metric.get("rows", metric.get("examples"))
        key = (update, metric["task"], metric.get("role", "dev"), size)
        require(key not in keys and update in checkpoints, "Duplicate or unsaved development evaluation")
        keys.add(key)
        require(metric["checkpoint"]["sha256"] == checkpoints[update]["sha256"]
                and metric["checkpoint"]["completed_updates"] == update, "Metrics identify a different checkpoint")
    selected = saved.selected_evaluations(evaluations)
    terminal = {(m["task"], m.get("role", "dev")): m for m in selected if m["update"] == endpoint}
    require(set(terminal) == {("a5", "dev"), ("a5", "ood_dev"), ("fuzzy", "dev")}, "Terminal tasks differ")
    require(all(m.get("rows", m.get("examples")) == (102400 if task == "a5" else 1280)
                for (task, role), m in terminal.items()), "Endpoint requires full development pools")
    files = {name: saved.sha(directory / name) for name in
             ("report.json", "history.jsonl", "model-config.json", "data-identity.json", "source-manifest.json")}
    return {"directory": str(directory), "report": report, "endpoint": endpoint, "checkpoints": checkpoints,
            "evaluations": evaluations, "history": history, "data_identity": identity, "sources": sources,
            "tasks": ["a5", "fuzzy"], "input_hashes": files,
            "lineage": [{"directory": str(directory), "start_update": 0, "used_through_update": endpoint,
                         "history": history_audit, "training_wandb": report.get("wandb")} ]}


def check_contracts(base, candidate):
    """Remove only the declared LR/source/schema fields before exact equality."""
    a, b = copy.deepcopy(base), copy.deepcopy(candidate)
    require(a["mode"] == b["mode"] == "mixed" and a["batch_per_task"] == b["batch_per_task"] == 2560
            and a["model_config"]["backbone"]["d_model"] == 128
            and "embedding_injection" not in a["model_config"], "Expected the uninjected D128 mixed baseline")
    require(a["runtime"]["precision"] == b["runtime"]["precision"] == "fp32", "This probe must remain FP32")
    require(a["schema"] == "rt-nextlat-a5-fuzzy-training-v1" and b["schema"] == TRAIN_SCHEMA,
            "Unexpected training schemas")
    require(a["optimizer"]["lr"] == 1e-4 and b["optimizer"]["lr"] == 3e-4
            and b["optimizer"].pop("initial_lr") == 1e-4, "Unexpected optimizer LR endpoints")
    schedule = b.pop("learning_rate_schedule")
    from scripts.rt_nextlat_a5_fuzzy_lr_train import schedule_configuration
    require(schedule == schedule_configuration(), "Learning-rate schedule differs from the approved recipe")
    for contract in (a, b):
        contract.pop("schema")
        contract.pop("source_sha256")
        contract["optimizer"].pop("lr")
    require(a == b, "Training contract differs beyond the authorized LR recipe")
    return schedule


def compare(base, candidate):
    require(base["endpoint"] == 15000 and base["report"]["status"] == "complete", "Reference must include its successful15k endpoint")
    schedule = check_contracts(base["report"]["contract"], candidate["report"]["contract"])
    require(base["data_identity"] == candidate["data_identity"], "Dataset pools differ")
    require(base["report"]["initialization"] == candidate["report"]["initialization"]
            and base["report"]["parameter_count"] == candidate["report"]["parameter_count"] == 479616,
            "Model initialization or size differs")
    require(set(candidate["sources"]) - set(base["sources"]) == {DRIVER}
            and all(candidate["sources"].get(name) == expected for name, expected in base["sources"].items()),
            "Frozen baseline source closure changed")
    initial = paired.verify_initial_tensors(base, candidate, {"variant": "baseline"})
    for old, new in zip(base["history"], candidate["history"]):
        require(old["update"] == new["update"] and old["order_chains"] == new["order_chains"]
                and old["examples_seen"] == new["examples_seen"], "Task data order or exposure differs")
        require(math.isclose(new["learning_rate"], expected_lr(new["update"]), rel_tol=0, abs_tol=1e-15),
                "Applied learning-rate history differs")
        require(new["gradient_clipped"] == (new["grad_norm"] > 1.0), "Clipping flag differs from logged gradient norm")
        require(all(math.isfinite(x) for x in (new["loss"], new["grad_norm"], new["seconds"])), "Nonfinite training values")
    left, right = paired._packet(base, 0), paired._packet(candidate, 0)
    require(left["optimizer"]["param_groups"] == right["optimizer"]["param_groups"]
            and left["optimizer_parameter_names"] == right["optimizer_parameter_names"], "Initial Adam groups differ")
    check_saved_lr(right, 0, schedule)
    check_saved_lr(paired._packet(candidate, candidate["endpoint"]), candidate["endpoint"], schedule)
    arms = {"constant_1e4": paired.arm_summary("constant_1e4", base),
            "warmup_3e4": paired.arm_summary("warmup_3e4", candidate)}
    matching = paired.matched_observations(*arms.values())
    common = sorted({m["update"] for m in matching if m["task"] == "a5" and m["role"] == "ood_dev"
                     and m["evaluated_rows"] == 102400 and m["update"] > 0})
    for update in common:
        a, b = paired._packet(base, update), paired._packet(candidate, update)
        check_saved_lr(b, update, schedule)
        require(a["next_cursors"] == b["next_cursors"] and a["order_chains"] == b["order_chains"],
                "Matched checkpoint data cursors differ")
    first_positive = next((m for m in arms["constant_1e4"]["evaluations"]
                           if m["task"] == "a5" and m["role"] == "ood_dev"
                           and m["whole_word_correct"] > 0), None)
    return {"schema": SCHEMA, "status": "complete", "qualification": QUALIFICATION,
            "schedule": schedule, "arms": arms, "matched_observations": matching,
            "common_full_updates": common, "latest_common_full_update": common[-1] if common else None,
            "matched_data_order_updates": candidate["endpoint"], "initial_pairing": initial,
            "original_context": {"endpoint": 15000, "final": arms["constant_1e4"]["final"],
                                 "first_observed_positive_a5_l36": first_positive},
            "training_bins": {"constant_1e4": training_bins(base["history"], constant=True),
                              "warmup_3e4": training_bins(candidate["history"], constant=False)}}


def check_saved_lr(packet, update, schedule):
    from scripts.rt_nextlat_a5_fuzzy_lr_train import learning_rate_state, validate_optimizer_lr
    expected = learning_rate_state(update, schedule)
    require(packet.get("learning_rate_state") == expected, "Checkpoint LR schedule state differs")
    validate_optimizer_lr(packet["optimizer"], expected["optimizer_lr"])


def training_bins(history, *, constant, width=100):
    result, elapsed = [], 0.0
    for start in range(0, len(history), width):
        rows = history[start:start + width]
        elapsed += sum(row["seconds"] for row in rows)
        point = {"update": rows[-1]["update"], "training_seconds": elapsed,
                 "loss": sum(row["loss"] for row in rows) / len(rows),
                 "grad_norm": sum(row["grad_norm"] for row in rows) / len(rows),
                 "clip_fraction": sum(row["grad_norm"] > 1 for row in rows) / len(rows),
                 "learning_rate": 1e-4 if constant else rows[-1]["learning_rate"]}
        for task in ("a5", "fuzzy"):
            for field in ("ce", "latent"):
                point[f"{task}_{field}"] = sum(row["tasks"][task][field] for row in rows) / len(rows)
        result.append(point)
    return result


def figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, colors = [], {"constant_1e4": "#222222", "warmup_3e4": "#D97706"}
    labels = {"constant_1e4": "Original: constant 1e-4", "warmup_3e4": "Probe: warmup to 3e-4"}
    endpoint = summary["arms"]["warmup_3e4"]["endpoint"]
    def save(fig, name):
        for suffix in ("png", "pdf"): fig.savefig(output / f"{name}.{suffix}", dpi=150, bbox_inches="tight")
        plt.close(fig); names.append(name)
    panels = [("a5", "dev", "whole_word_exact_match", "A5 length12 whole word"),
              ("a5", "ood_dev", "whole_word_exact_match", "A5 length36 whole word"),
              ("a5", "ood_dev", "token_accuracy", "A5 length36 token"),
              ("fuzzy", "dev", "answer_accuracy", "Fuzzy answer tokens"),
              ("fuzzy", "dev", "first_value_token_accuracy", "Fuzzy first value"),
              ("fuzzy", "dev", "sequence_exact_match", "Fuzzy all-answer sequence")]
    for clock in ("updates", "training-time"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        for axis, (task, role, key, title) in zip(axes.flat, panels):
            for label, arm in summary["arms"].items():
                rows = [m for m in arm["evaluations"] if m["task"] == task and m.get("role", "dev") == role and m["update"] <= endpoint]
                x = [m["update"] if clock == "updates" else m["cumulative_training_seconds"]/3600 for m in rows]
                axis.plot(x, [100*m[key] for m in rows], color=colors[label], label=labels[label])
                full = [(index, m) for index, m in enumerate(rows) if task == "a5" and m["rows"] == 102400]
                axis.scatter([x[index] for index, m in full], [100*m[key] for index, m in full], color=colors[label], s=14)
            axis.set(title=title, xlabel="Optimizer updates" if clock == "updates" else "Cumulative training hours", ylabel="Accuracy (%)", ylim=(-2, 102))
            axis.grid(alpha=.2); axis.legend(fontsize=7)
        fig.suptitle(f"Matched training window through {endpoint:,} updates; same data exposure\nA5 lines include monitoring subsets; dots mark full102,400-word evaluations")
        save(fig, "matched-learning-" + clock)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, (task, role, field) in zip(axes, [("a5", "ood_dev", "whole_word_exact_match"), ("fuzzy", "dev", "answer_accuracy")]):
        for label, arm in summary["arms"].items():
            rows = [m for m in arm["evaluations"] if m["task"] == task and m.get("role", "dev") == role]
            axis.plot([m["update"] for m in rows], [100*m[field] for m in rows], label=labels[label], color=colors[label])
        axis.axvline(5000, ls=":", color=".6")
        axis.set(title="A5 length36 whole word" if task == "a5" else "Fuzzy answer accuracy", xlabel="Optimizer updates", ylabel="Accuracy (%)", ylim=(-2, 102))
        axis.legend(fontsize=8); axis.grid(alpha=.2)
    fig.suptitle("Extended reference context: original baseline continued to15k; probe ends at5k\nThis is not an equal-budget endpoint comparison")
    save(fig, "baseline-extended-context")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    common = summary["latest_common_full_update"]
    for label, arm in summary["arms"].items():
        if common is None and label == "constant_1e4": continue
        for axis, role in zip(axes, ("dev", "ood_dev")):
            candidates = [m for m in arm["evaluations"] if m["task"] == "a5" and m["role"] == role
                          and m["update"] == (common if common is not None else endpoint) and m["rows"] == 102400]
            require(len(candidates) == 1, "Prefix plot needs one full observation")
            metric = candidates[0]
            axis.plot(range(1, metric["length"] + 1), [100*x for x in metric["cumulative_prefix_exactness"]], label=labels[label], color=colors[label])
            axis.set(title=f"A5 length{metric['length']}", xlabel="Prefix length", ylabel="Whole prefix correct (%)", ylim=(-2, 102))
            axis.legend(fontsize=8); axis.grid(alpha=.2)
    fig.suptitle(f"Common full checkpoint at update{common}" if common else "Probe endpoint only: no common full checkpoint yet")
    save(fig, "matched-prefix")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    diagnostics = [("a5_ce", "A5 state CE"), ("fuzzy_ce", "Fuzzy dense-label CE"), ("learning_rate", "Learning rate applied"),
                   ("a5_latent", "A5 NextLat loss"), ("fuzzy_latent", "Fuzzy NextLat loss"), ("clip_fraction", "Fraction updates clipped at norm1")]
    for axis, (key, title) in zip(axes.flat, diagnostics):
        for label, rows in summary["training_bins"].items():
            points = [row for row in rows if row["update"] <= endpoint]
            if key == "learning_rate":
                updates = sorted({1, min(100, endpoint), endpoint})
                rates = [1e-4 if label == "constant_1e4" else expected_lr(update) for update in updates]
                axis.plot(updates, rates, label=labels[label], color=colors[label])
            else:
                axis.plot([r["update"] for r in points], [r[key] for r in points], label=labels[label], color=colors[label])
        axis.set(title=title, xlabel="Optimizer updates"); axis.legend(fontsize=7); axis.grid(alpha=.2)
    fig.suptitle("Training diagnostics: 100-update means; LR shows the exact applied schedule")
    save(fig, "training-diagnostics")
    return names


def markdown(summary):
    probe = summary["arms"]["warmup_3e4"]
    endpoint = probe["endpoint"]
    lines = ["# FP32 mixed A5/Fuzzy: learning-rate acceleration probe", "",
             f"The probe {probe['training_status']} at **{endpoint:,} optimizer updates**, with2,560 examples per task per update. "
             "LR rises linearly from1e-4 at update1 to3e-4 at update100, then remains constant.", "",
             "Only the LR recipe changes; model, NextLat, initialization, data order, task weights, FP32 and clipping remain matched.", "",
             "| Update | Arm | A5 L12 whole word | A5 L36 whole word | Fuzzy answer / first value / sequence |",
             "|---:|---|---:|---:|---:|"]
    selected = [step for step in summary["common_full_updates"] if step in (1000, 2500, 5000)]
    if summary["latest_common_full_update"] is not None and summary["latest_common_full_update"] not in selected:
        selected.append(summary["latest_common_full_update"])
    for step in sorted(selected):
        for label, arm in summary["arms"].items():
            observations = {f"{m['task']}/{m.get('role', 'dev')}": m for m in arm["evaluations"] if m["update"] == step}
            a, b, f = [observations[key] for key in ("a5/dev", "a5/ood_dev", "fuzzy/dev")]
            lines.append(f"| {step:,} | {label} | {100*a['whole_word_exact_match']:.4f}% | {100*b['whole_word_exact_match']:.4f}% | "
                         f"{100*f['answer_accuracy']:.4f}% / {100*f['first_value_token_accuracy']:.4f}% / {100*f['sequence_exact_match']:.4f}% |")
    final = probe["final"]
    lines += ["", f"At the probe's actual endpoint, full-development A5 length36 whole-word accuracy is "
              f"**{100*final['a5/ood_dev']['whole_word_exact_match']:.4f}%**; Fuzzy answer accuracy is "
              f"**{100*final['fuzzy/dev']['answer_accuracy']:.4f}%**. All endpoint tasks use the same saved checkpoint.", "",
              "The original baseline eventually learned A5: its first observed positive length36 result was at10,400 "
              "updates on4,096 words, and its full15k endpoint reached **92.8535% whole-word accuracy**, with **99.9112% Fuzzy answer accuracy**. "
              "A disappointing5k probe means this acceleration recipe did not achieve the desired early learning; it does not establish architectural failure.", "",
              "| Arm | Training updates shown | Cumulative training minutes |", "|---|---:|---:|"]
    for label, arm in summary["arms"].items():
        # The exact comparison time is retained even for a stop between baseline evaluations.
        seconds = summary["matched_training_seconds"][label]
        lines.append(f"| {label} | {endpoint:,} | {seconds/60:.2f} |")
    lines += ["", QUALIFICATION, "", "## Figures", ""]
    for name in summary["figures"]: lines.append(f"[{name}]({name}.pdf)")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-train", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    import torch
    require(Path("/.dockerenv").is_file() and not torch.cuda.is_initialized(), "Use the GPU-disabled project container")
    base, probe = saved.load_run(args.baseline_train), read_fresh(args.train)
    summary = compare(base, probe)
    summary["matched_training_seconds"] = {"constant_1e4": paired.cumulative_training_seconds(base["history"])[probe["endpoint"]],
                                           "warmup_3e4": paired.cumulative_training_seconds(probe["history"])[probe["endpoint"]]}
    args.output.mkdir(parents=True, exist_ok=False)
    summary["figures"] = figures(summary, args.output)
    (args.output / "report.md").write_text(markdown(summary))
    for name in ("scripts/rt_nextlat_a5_fuzzy_lr_report.py", "scripts/rt_nextlat_a5_fuzzy_report.py",
                 "scripts/rt_nextlat_a5_fuzzy_embedding_compare.py"):
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (args.output / "evidence.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.wandb:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", entity="taylorbollman", output_dir=args.output,
                                name="mixed-b2560-lr3e4-versus-original")
        try:
            tracker.start({"schema": SCHEMA, "qualification": QUALIFICATION, "schedule": summary["schedule"]})
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png")) for name in summary["figures"]})
            tracker.summary({"probe_endpoint": probe["endpoint"], "probe_final": summary["arms"]["warmup_3e4"]["final"],
                             "matched_training_seconds": summary["matched_training_seconds"]})
            tracker.finish(succeeded=True)
        except BaseException:
            tracker.finish(succeeded=False)
            raise
        summary["report_wandb"] = tracker.record
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "endpoint": probe["endpoint"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
