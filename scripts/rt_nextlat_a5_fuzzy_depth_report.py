#!/usr/bin/env python3
"""CPU-only comparison of fresh two- and three-layer mixed NextLat runs."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil

from scripts import rt_nextlat_a5_fuzzy_lr_report as lr
from scripts import rt_nextlat_a5_fuzzy_embedding_compare as paired
from scripts import rt_nextlat_a5_fuzzy_report as saved


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-a5-fuzzy-depth-comparison-v1"
EXTRA_SOURCES = {
    "cdrm/rt_nextlat_task_depth.py",
    "configs/rt_nextlat_tasks/fuzzy_d128_l3.json",
    "scripts/rt_nextlat_a5_fuzzy_depth_train.py",
}
LABELS = {"two_layers": "2 layers: restricted + full RT",
          "three_layers": "3 layers: restricted + two full RT"}
QUALIFICATION = (
    "One seed per architecture and reused development pools; this is a directional depth comparison, "
    "not a replicated effect or a parameter-matched comparison. Both use native Mitchell initialization "
    "with the same seeds; changing depth changes the backbone draw and depth-dependent scaling, so "
    "backbone tensor identity is neither required nor claimed. The independently initialized NextLat "
    "predictor is identical. Data order, effective batch, task weights, objective, FP32 runtime and LR "
    "schedule remain fixed. Any physical microbatch difference is disclosed. A5 monitoring subsets and "
    "full evaluations have different sample sizes; matched observations require equal sample sizes. "
    "Threshold crossings are first observed, not sustained convergence. Training time excludes "
    "evaluation, checkpointing and reporting and is not a repeated throughput benchmark. "
    "No new model inference, final confirmation or autonomous latent rollout is performed."
)
require = saved.require


def check_contracts(base, candidate):
    """Accept only depth, its recorded initialization/sources, and bounded accumulation."""
    a, b = copy.deepcopy(base), copy.deepcopy(candidate)
    require(a["schema"] == b["schema"] == lr.TRAIN_SCHEMA, "Unexpected training schema")
    require(a["mode"] == b["mode"] == "mixed" and a["batch_per_task"] == b["batch_per_task"] == 2560,
            "Both tasks require 2560 examples per optimizer update")
    require(a["microbatch"] == 2560 and b["microbatch"] in (1280, 2560), "Unapproved physical microbatch")
    require(a["runtime"]["precision"] == b["runtime"]["precision"] == "fp32", "Depth comparison must remain FP32")
    x, y = a["model_config"], b["model_config"]
    require(x["schema"] == "rt-nextlat-tasks-model-v1"
            and y["schema"] == "rt-nextlat-tasks-depth-model-v1", "Unexpected model schema")
    require(x["backbone"]["d_model"] == y["backbone"]["d_model"] == 128
            and x["backbone"]["n_layers"] == 2 and y["backbone"]["n_layers"] == 3,
            "Expected D128 two versus three layers")
    require(x.get("window_layer") == y.get("window_layer") == 0
            and x.get("window_size") == y.get("window_size") == 2
            and "embedding_injection" not in x and "embedding_injection" not in y,
            "Expected restricted first layer and no embedding injection")
    y["schema"] = x["schema"]
    y["backbone"]["n_layers"] = 2
    require(x == y, "Model changes beyond depth")
    from scripts.rt_nextlat_a5_fuzzy_lr_train import schedule_configuration
    require(a["learning_rate_schedule"] == b["learning_rate_schedule"] == schedule_configuration()
            and a["optimizer"]["lr"] == b["optimizer"]["lr"] == 3e-4
            and a["optimizer"]["initial_lr"] == b["optimizer"]["initial_lr"] == 1e-4,
            "Learning-rate recipe changed")
    for contract in (a, b):
        for key in ("model_config", "configuration_file_sha256", "source_sha256", "initialization", "microbatch"):
            contract.pop(key)
    require(a == b, "Training contract changes beyond the declared depth experiment")
    return base["learning_rate_schedule"]


def check_sources(base, candidate):
    require(set(candidate) - set(base) == EXTRA_SOURCES
            and all(candidate.get(name) == digest for name, digest in base.items()),
            "Historical source closure changed or depth source closure is incomplete")


def check_histories(base, candidate):
    require(len(candidate) <= len(base), "Candidate exceeds the comparison window")
    for old, new in zip(base, candidate):
        update = new["update"]
        require(old["update"] == update and old["order_chains"] == new["order_chains"]
                and old["examples_seen"] == new["examples_seen"]
                == {task: update * 2560 for task in ("a5", "fuzzy")}, "Task order or exposure differs")
        for row in (old, new):
            require(math.isclose(row["learning_rate"], lr.expected_lr(update), rel_tol=0, abs_tol=1e-15),
                    "Applied LR differs")
            require(row["gradient_clipped"] == (row["grad_norm"] > 1.0), "Clipping flag differs")
            values = [row["loss"], row["grad_norm"], row["seconds"]]
            values += [row["tasks"][task][field] for task in ("a5", "fuzzy") for field in ("ce", "latent")]
            require(all(math.isfinite(value) for value in values), "Nonfinite committed training values")


def check_initialization(base, candidate):
    """Audit same RNG policy and identical predictor; do not pair backbone tensors."""
    import torch
    a, b = base["report"]["initialization"], candidate["report"]["initialization"]
    require(a["schema"] == "rt-nextlat-tasks-initialization-v1"
            and b["schema"] == "rt-nextlat-tasks-depth-initialization-v1" and b.get("depth") == 3
            and b.get("depth_pairing") == "Fresh canonical initialization at actual depth; no exact shared-weight pairing across depths"
            and a["rule"] == "Canonical V60 CPU Mitchell SEQ; independently appended Mitchell Fuzzy rows; "
                             "exhaustive RT conversion; parameter-preserving first-layer window; isolated NextLat RNG"
            and b["rule"] == "Fresh actual-depth canonical V60 CPU Mitchell SEQ; independently appended Mitchell Fuzzy rows; "
                             "exhaustive RT conversion; parameter-preserving first-layer window; isolated NextLat RNG",
            "Unexpected native-depth initialization policy")
    for key in ("seed", "predictor_seed", "fuzzy_seed", "predictor_sha256", "predictor_parameter_count", "nextlat_config"):
        require(a[key] == b[key], "Initialization policy or predictor differs: " + key)
    require((a["seed"], a["predictor_seed"], a["fuzzy_seed"]) == (1234, 1235, 1236), "Unexpected initialization seeds")
    for run, total, backbone in ((base, 479616, 413824), (candidate, 676736, 610944)):
        metadata = run["report"]["initialization"]
        require(run["report"]["parameter_count"] == metadata["parameter_count"] == total
                and metadata["backbone_parameter_count"] == backbone
                and metadata["predictor_parameter_count"] == 65792
                and metadata["task_config"] == run["report"]["contract"]["model_config"],
                "Unexpected parameter count or initialization configuration")
    left, right = paired._packet(base, 0), paired._packet(candidate, 0)
    require(not left["optimizer"]["state"] and not right["optimizer"]["state"], "Fresh run inherited Adam state")
    require(left["order_chains"] == right["order_chains"], "Initial data chains differ")
    names = {key for key in left["model"] if key.startswith("predictor.")}
    require(names and names == {key for key in right["model"] if key.startswith("predictor.")}
            and all(torch.equal(left["model"][key], right["model"][key]) for key in names),
            "Independently seeded predictor tensors differ")
    for packet in (left, right):
        require(all(value.dtype == torch.float32 for value in packet["model"].values()), "Initial state is not FP32")
    # Parameter-index arrays necessarily grow with depth; all other Adam group settings must match.
    groups = lambda packet: [{key: value for key, value in group.items() if key != "params"}
                             for group in packet["optimizer"]["param_groups"]]
    require(groups(left) == groups(right), "Initial Adam group settings differ")
    return {"passed": True, "same_seeds": True, "native_depth_initialization": True,
            "backbone_tensor_identity_required": False, "predictor_tensors_exact": True,
            "predictor_tensor_count": len(names), "empty_initial_optimizer_states": True,
            "baseline_checkpoint": base["checkpoints"][0], "candidate_checkpoint": candidate["checkpoints"][0]}


def compare(base, candidate):
    require(base["endpoint"] == 5000 and base["report"]["status"] == "complete", "Reference must be the completed two-layer 5k run")
    schedule = check_contracts(base["report"]["contract"], candidate["report"]["contract"])
    require(base["data_identity"] == candidate["data_identity"], "Dataset identities differ")
    check_sources(base["sources"], candidate["sources"])
    check_histories(base["history"], candidate["history"])
    initialization = check_initialization(base, candidate)
    arms = {"two_layers": paired.arm_summary("two_layers", base),
            "three_layers": paired.arm_summary("three_layers", candidate)}
    matching = paired.matched_observations(*arms.values())
    common = sorted({row["update"] for row in matching if row["task"] == "a5" and row["role"] == "ood_dev"
                     and row["evaluated_rows"] == 102400 and row["update"] > 0})
    for run in (base, candidate):
        for update in sorted({0, run["endpoint"], *common}):
            lr.check_saved_lr(paired._packet(run, update), update, schedule)
    for update in common:
        left, right = paired._packet(base, update), paired._packet(candidate, update)
        require(left["next_cursors"] == right["next_cursors"] and left["order_chains"] == right["order_chains"],
                "Matched checkpoint data streams differ")
    return {"schema": SCHEMA, "status": "complete", "qualification": QUALIFICATION,
            "schedule": schedule, "arms": arms, "matched_observations": matching,
            "common_full_updates": common, "latest_common_full_update": common[-1] if common else None,
            "matched_data_order_updates": candidate["endpoint"], "initialization_audit": initialization,
            "parameter_difference": 197120,
            "microbatch": {name: arm["contract"]["microbatch"] for name, arm in arms.items()},
            "training_bins": {name: lr.training_bins(run["history"], constant=False)
                              for name, run in (("two_layers", base), ("three_layers", candidate))},
            "matched_training_seconds": {name: paired.cumulative_training_seconds(run["history"])[candidate["endpoint"]]
                                         for name, run in (("two_layers", base), ("three_layers", candidate))}}


def figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, colors = [], {"two_layers": "#222222", "three_layers": "#137c8b"}
    endpoint = summary["arms"]["three_layers"]["endpoint"]
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        names.append(name)
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
                rows = [row for row in arm["evaluations"] if row["task"] == task
                        and row.get("role", "dev") == role and row["update"] <= endpoint]
                x = [row["update"] if clock == "updates" else row["cumulative_training_seconds"]/3600 for row in rows]
                axis.plot(x, [100*row[key] for row in rows], color=colors[label], label=LABELS[label])
                full = [(i, row) for i, row in enumerate(rows) if task == "a5" and row["rows"] == 102400]
                axis.scatter([x[i] for i, row in full], [100*row[key] for i, row in full], color=colors[label], s=14)
            axis.set(title=title, xlabel="Optimizer updates" if clock == "updates" else "Cumulative training hours",
                     ylabel="Accuracy (%)", ylim=(-2, 102))
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        fig.suptitle(f"Depth comparison through {endpoint:,} updates; same effective batch and LR\n"
                     "A5 lines include monitoring subsets; dots mark full 102,400-word evaluations")
        save(fig, "matched-learning-" + clock)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    common = summary["latest_common_full_update"]
    for label, arm in summary["arms"].items():
        if common is None and label == "two_layers":
            continue
        for axis, role in zip(axes, ("dev", "ood_dev")):
            rows = [row for row in arm["evaluations"] if row["task"] == "a5" and row["role"] == role
                    and row["update"] == (common if common is not None else endpoint) and row["rows"] == 102400]
            require(len(rows) == 1, "Prefix plot needs one full observation")
            row = rows[0]
            axis.plot(range(1, row["length"] + 1), [100*x for x in row["cumulative_prefix_exactness"]],
                      label=LABELS[label], color=colors[label])
            axis.set(title=f"A5 length{row['length']}", xlabel="Prefix length", ylabel="Whole prefix correct (%)", ylim=(-2, 102))
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
    fig.suptitle(f"Common full checkpoint at update {common:,}" if common else
                 "Three-layer endpoint only: no common full checkpoint yet")
    save(fig, "matched-prefix")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    diagnostics = [("a5_ce", "A5 state CE"), ("fuzzy_ce", "Fuzzy dense-label CE"), ("learning_rate", "Learning rate applied"),
                   ("a5_latent", "A5 NextLat loss"), ("fuzzy_latent", "Fuzzy NextLat loss"),
                   ("clip_fraction", "Fraction of updates clipped at norm 1")]
    for axis, (key, title) in zip(axes.flat, diagnostics):
        for label, rows in summary["training_bins"].items():
            points = [row for row in rows if row["update"] <= endpoint]
            if key == "learning_rate":
                updates = sorted({1, min(100, endpoint), endpoint})
                axis.plot(updates, [lr.expected_lr(update) for update in updates],
                          label=LABELS[label], color=colors[label], ls="--" if label == "three_layers" else "-")
            else:
                axis.plot([row["update"] for row in points], [row[key] for row in points],
                          label=LABELS[label], color=colors[label])
        axis.set(title=title, xlabel="Optimizer updates")
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    fig.suptitle("Training diagnostics: 100-update means; both arms use the same exact LR schedule")
    save(fig, "training-diagnostics")
    return names


def markdown(summary):
    candidate = summary["arms"]["three_layers"]
    endpoint = candidate["endpoint"]
    lines = ["# Mixed A5/Fuzzy: two versus three RT layers", "",
             f"The three-layer run {candidate['training_status']} at **{endpoint:,} optimizer updates**. "
             "It uses a restricted first RT layer followed by two full RT layers, with NextLat.", "",
             "Both arms use 2,560 examples per task per update, FP32, and LR rising from 1e-4 at update 1 "
             "to 3e-4 at update 100, then constant. Parameters increase from 479,616 to 676,736.", "",
             f"Physical microbatch: two layers **{summary['microbatch']['two_layers']}**, three layers "
             f"**{summary['microbatch']['three_layers']}**. The effective batch and single global clipping/Adam step remain fixed.", "",
             "| Update | Arm | A5 L12 whole word | A5 L36 whole word | Fuzzy answer / first value / sequence |",
             "|---:|---|---:|---:|---:|"]
    selected = [step for step in summary["common_full_updates"] if step in (1000, 2500, 5000)]
    common = summary["latest_common_full_update"]
    if common is not None and common not in selected:
        selected.append(common)
    for step in sorted(selected):
        for label, arm in summary["arms"].items():
            rows = {f"{row['task']}/{row.get('role', 'dev')}": row for row in arm["evaluations"] if row["update"] == step}
            a, b, f = [rows[key] for key in ("a5/dev", "a5/ood_dev", "fuzzy/dev")]
            lines.append(f"| {step:,} | {LABELS[label]} | {100*a['whole_word_exact_match']:.4f}% | "
                         f"{100*b['whole_word_exact_match']:.4f}% | {100*f['answer_accuracy']:.4f}% / "
                         f"{100*f['first_value_token_accuracy']:.4f}% / {100*f['sequence_exact_match']:.4f}% |")
    final = candidate["final"]
    lines += ["", f"At the actual three-layer endpoint, full-development A5 length36 whole-word accuracy is "
              f"**{100*final['a5/ood_dev']['whole_word_exact_match']:.4f}%**; Fuzzy answer accuracy is "
              f"**{100*final['fuzzy/dev']['answer_accuracy']:.4f}%**. All endpoint tasks use the same saved checkpoint.", "",
              "| Arm | Training updates shown | Cumulative training minutes |", "|---|---:|---:|"]
    for label, seconds in summary["matched_training_seconds"].items():
        lines.append(f"| {LABELS[label]} | {endpoint:,} | {seconds/60:.2f} |")
    lines += ["", QUALIFICATION, "", "## Figures", ""]
    lines += [f"[{name}]({name}.pdf)" for name in summary["figures"]]
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
    base, candidate = lr.read_fresh(args.baseline_train), lr.read_fresh(args.train)
    summary = compare(base, candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    summary["figures"] = figures(summary, args.output)
    (args.output / "report.md").write_text(markdown(summary))
    for name in ("scripts/rt_nextlat_a5_fuzzy_depth_report.py", "scripts/rt_nextlat_a5_fuzzy_lr_report.py",
                 "scripts/rt_nextlat_a5_fuzzy_report.py", "scripts/rt_nextlat_a5_fuzzy_embedding_compare.py"):
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (args.output / "evidence.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.wandb:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", entity="taylorbollman", output_dir=args.output,
                                name="mixed-b2560-three-versus-two-rt-layers-lr3e4")
        try:
            tracker.start({"schema": SCHEMA, "qualification": QUALIFICATION, "schedule": summary["schedule"],
                           "microbatch": summary["microbatch"], "parameter_difference": summary["parameter_difference"]})
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png")) for name in summary["figures"]})
            tracker.summary({"candidate_endpoint": candidate["endpoint"],
                             "candidate_final": summary["arms"]["three_layers"]["final"],
                             "matched_training_seconds": summary["matched_training_seconds"]})
            tracker.finish(succeeded=True)
        except BaseException:
            tracker.finish(succeeded=False)
            raise
        summary["report_wandb"] = tracker.record
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "endpoint": candidate["endpoint"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
