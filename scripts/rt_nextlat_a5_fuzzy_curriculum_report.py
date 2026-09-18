#!/usr/bin/env python3
"""Report saved development evidence for paired A5 warm-start continuations.

This is an offline report, not a model evaluation. Phase zero is the restored
A5-trained checkpoint, not a fresh initialization. Training owns W&B logging.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.rt_nextlat_a5_fuzzy_report import (
    checked_a5_metric, json_sha, local_path, read_json, require,
    selected_evaluations, sha,
)


SCHEMA = "rt-nextlat-a5-fuzzy-curriculum-report-v1"
ROOT = Path(__file__).resolve().parents[1]
GLOBAL_START = 10000
PARENT_A5_EXAMPLES = 1280000
BATCH = 128
RAMP = 3000
ROLES = ("a5/dev", "a5/ood_dev", "fuzzy/dev")
LABELS = {"a5-control": "A5-only continuation", "mixed-curriculum": "A5 + Fuzzy curriculum"}
QUALIFICATION = (
    "One initialization and reused development data; final confirmation remains unused. "
    "Phase zero already has learned A5 state tracking. This measures retention and "
    "coexistence after an A5 warm start, not new early learning. Warm-start pretraining "
    "and the loss ramp are combined; an abrupt-mix branch would be needed to isolate "
    "their effects. The earlier fresh mixed 10k endpoint does not rule out later learning. "
    "All endpoint task metrics below are bound to one checkpoint. Fuzzy sequence "
    "exactness is teacher-forced all-answer exactness, not autonomous generation."
)


def counters(record, *, checkpoint=False):
    phase_key = "phase_updates" if checkpoint else "phase_update"
    global_key = "global_updates" if checkpoint else "global_update"
    update_key = "completed_updates" if checkpoint else "update"
    phase = record.get(phase_key)
    require(type(phase) is int and phase >= 0, "Invalid phase update counter")
    require(record.get(global_key) == record.get(update_key) == GLOBAL_START + phase,
            "Phase/global optimizer counters disagree")
    return phase


def expected_offsets(mode, phase):
    return {"a5": PARENT_A5_EXAMPLES + BATCH * phase,
            "fuzzy": BATCH * phase if mode == "mixed-curriculum" else 0}


def expected_weights(mode, phase):
    weight = .5 * min(phase / RAMP, 1) if mode == "mixed-curriculum" else 0.
    return {"a5": 1 - weight, "fuzzy": weight}


def checked_fuzzy_metric(metric):
    for key, numerator, denominator in (
            ("answer_accuracy", "answer_correct", "answer_tokens"),
            ("answer_motif_exact_match", "exact_answer_motifs", "answer_motifs"),
            ("sequence_exact_match", "exact_sequences", "examples")):
        n, count = metric.get(denominator), metric.get(numerator)
        require(type(n) is int and n > 0 and type(count) is int and 0 <= count <= n,
                f"Invalid Fuzzy {key} counts")
        require(math.isclose(metric[key], count / n, rel_tol=0, abs_tol=1e-12),
                f"Fuzzy {key} disagrees with its counts")
    require(metric["evaluated_rows"] == metric["examples"], "Fuzzy row count disagrees")


def metrics_at(run, phase):
    return {f"{m['task']}/{m['role']}": m
            for m in selected_evaluations(run["evaluations"]) if m["phase_update"] == phase}


def verify_checkpoint(record, directory):
    phase = counters(record, checkpoint=True)
    path = local_path(record["path"])
    require(path.parent == directory / "checkpoints", "Checkpoint belongs to another training directory")
    require(path.name == f"phase-{phase:06d}.pt", "Checkpoint filename differs from its phase")
    require(path.is_file() and sha(path) == record["sha256"], "Checkpoint hash mismatch")
    require(record["bytes"] == path.stat().st_size, "Checkpoint size mismatch")
    return phase, {**record, "verified_local_path": str(path)}


def load_run(directory):
    directory = local_path(directory)
    report = read_json(directory / "report.json")
    require(report.get("status") in ("complete", "stopped"),
            "Require a completed or explicitly stopped training report")
    require(report.get("confirmation_evaluated") is False
            and report.get("latent_rollout_evaluated") is False,
            "Final confirmation and autonomous latent rollout must remain unused")
    contract = report["contract"]
    require(report.get("schema") == contract.get("schema") == "rt-nextlat-a5-fuzzy-curriculum-v1",
            "Expected the curriculum training schema")
    mode = contract["mode"]
    require(mode in LABELS, "Expected an A5 warm-start control or curriculum")
    require(contract["batch_per_task"] == BATCH, "Expected 128 examples per task")
    require(contract["global_start_update"] == GLOBAL_START and contract["ramp_updates"] == RAMP
            and contract["initial_task_offsets"] == {"a5": PARENT_A5_EXAMPLES, "fuzzy": 0},
            "Warm-start counter origins or ramp duration differ")
    config = read_json(directory / "model-config.json")
    identity = read_json(directory / "data-identity.json")
    sources = read_json(directory / "source-manifest.json")
    require(config == contract["model_config"] and sha(directory / "model-config.json")
            == contract["configuration_file_sha256"], "Model configuration differs")
    require(json_sha(identity) == contract["data_sha256"], "Dataset identity differs")
    require(json_sha(sources) == contract["source_sha256"], "Frozen source manifest differs")
    for name, digest in sources.items():
        require(sha(directory / "source" / name) == digest, f"Frozen source differs: {name}")
    require(report["initialization"] == contract["initialization"], "Initialization lineage differs")

    parent = report["parent_training_checkpoint"]
    parent_path = local_path(parent["path"])
    require(parent["completed_updates"] == GLOBAL_START and parent_path.is_file()
            and sha(parent_path) == parent["sha256"], "A5 warm-start parent checkpoint differs")
    parent_report_path = parent_path.parent.parent / "report.json"
    parent_report = read_json(parent_report_path)
    require(parent_report["contract"]["mode"] == "a5-only", "Warm-start parent must be A5-only")
    parent_checkpoint = next((c for c in parent_report["checkpoints"]
                              if c["completed_updates"] == GLOBAL_START), None)
    require(parent_checkpoint is not None and parent_checkpoint["sha256"] == parent["sha256"],
            "Parent report identifies different checkpoint bytes")
    require(contract["parent_checkpoint"]["sha256"] == parent["sha256"]
            and contract["parent_checkpoint"]["completed_updates"] == GLOBAL_START
            and contract["parent_contract_sha256"] == json_sha(parent_report["contract"]),
            "Contract and reported warm-start parent identity differ")
    for key in ("model_config", "initialization", "optimizer", "runtime"):
        require(parent_report["contract"][key] == contract[key], f"Warm-start parent {key} differs")
    parent_identity = read_json(parent_path.parent.parent / "data-identity.json")
    require(parent_identity["a5"] == identity["a5"], "Warm-start A5 dataset identity differs")
    require(parent_report["contract"]["streams"]["a5"] == contract["streams"]["a5"],
            "Warm-start A5 data order settings differ")

    checkpoints = {}
    for record in report["checkpoints"]:
        phase, verified = verify_checkpoint(record, directory)
        require(phase not in checkpoints, "Duplicate checkpoint phase")
        require(record["task_offsets"] == expected_offsets(mode, phase), "Checkpoint task exposure differs")
        checkpoints[phase] = verified
    require(0 in checkpoints, "Require an actual restored phase-zero checkpoint")
    endpoint = max(checkpoints)
    require(endpoint > 0 and counters(report, checkpoint=True) == endpoint
            and report["task_offsets"] == expected_offsets(mode, endpoint),
            "Actual endpoint checkpoint and report disagree")
    require(endpoint <= report["requested_phase_endpoint"]
            and report["requested_endpoint"] == GLOBAL_START + report["requested_phase_endpoint"]
            and (report["status"] != "complete" or endpoint == report["requested_phase_endpoint"]),
            "Completed status or requested endpoint disagrees with the actual budget")
    # A new objective has a new contract. Do not use exact-continuation merging
    # from the old reporter to silently join these histories.
    require(report.get("start_phase_update") == 0 and report.get("start_update") == GLOBAL_START
            and report.get("parent_checkpoint") is None,
            "Stage-resumed reporting needs an explicit audited phase-history lineage")

    history_path = directory / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    require(len(history) == endpoint, "History does not span every additional optimizer update")
    for expected_phase, row in enumerate(history, 1):
        require(counters(row) == expected_phase, "History has a phase gap or duplicate")
        require(row["task_weights"] == expected_weights(mode, expected_phase), "Fuzzy weight schedule differs")
        require(row["task_offsets"] == expected_offsets(mode, expected_phase), "Task exposure offsets differ")
        require(set(row["tasks"]) == ({"a5", "fuzzy"} if mode == "mixed-curriculum" else {"a5"}),
                "Training tasks differ from the arm")
        for task, metric in row["tasks"].items():
            require(all(math.isfinite(metric[key]) and metric[key] >= 0 for key in ("ce", "latent")),
                    f"Invalid {task} training losses")
        require(math.isfinite(row["loss"]), "Invalid total loss")
        require(math.isclose(row["loss"], sum(row["task_weights"][task] * (m["ce"] + m["latent"])
                                            for task, m in row["tasks"].items()), rel_tol=1e-10, abs_tol=1e-12),
                "Weighted training objective disagrees with per-task losses")
        require(set(row["order_chains"]) == {"a5", "fuzzy"}, "Missing task order chains")
        if mode == "a5-control":
            require(row["order_chains"]["fuzzy"] == contract["initial_order_chains"]["fuzzy"],
                    "A5 control unexpectedly consumed the Fuzzy stream")

    evaluations, seen = report["evaluations"], {}
    for metric in evaluations:
        phase = counters(metric)
        key = (phase, metric["task"], metric["role"], metric["evaluated_rows"])
        require(phase in checkpoints, "Uncheckpointed evaluation")
        semantic_metric = {k: v for k, v in metric.items() if k != "evaluation_seconds"}
        require(key not in seen or seen[key] == semantic_metric, "Conflicting repeated evaluation")
        seen[key] = semantic_metric
        bound, checkpoint = metric["checkpoint"], checkpoints[phase]
        require(bound["sha256"] == checkpoint["sha256"]
                and bound["completed_updates"] == GLOBAL_START + phase
                and local_path(bound["path"]) == local_path(checkpoint["path"]),
                "Evaluation identifies a different checkpoint")
        if metric["task"] == "a5":
            curves = checked_a5_metric(metric)
            require(metric["whole_word_exact_count"] == metric["whole_word_correct"]
                    and metric["per_position_prefix_correct"] == [r["prefix_exact_count"] for r in curves]
                    and metric["per_position_state_correct"] == [r["state_correct_count"] for r in curves],
                    "A5 integer curve counts disagree")
        else:
            require(metric["task"] == "fuzzy" and metric["role"] == "dev", "Unknown evaluation task")
            checked_fuzzy_metric(metric)
    run = {"directory": str(directory), "report": report, "mode": mode,
           "endpoint": endpoint, "checkpoints": checkpoints, "history": history,
           "evaluations": evaluations, "data_identity": identity, "sources": sources,
           "input_hashes": {name: sha(directory / name) for name in
                ("report.json", "history.jsonl", "model-config.json", "data-identity.json", "source-manifest.json")},
           "parent": {**parent, "verified_local_path": str(parent_path),
                      "report_path": str(parent_report_path), "report_sha256": sha(parent_report_path)}}
    for phase in (0, endpoint):
        metrics = metrics_at(run, phase)
        require(set(metrics) == set(ROLES), "Baseline and endpoint require both A5 roles and Fuzzy at one checkpoint")
        require(all(m["evaluated_rows"] == (102400 if m["task"] == "a5" else 1280)
                    and m["scope"] == "full" for m in metrics.values()),
                "Baseline and endpoint require full development evaluations")
    previous = {m["role"]: m for m in parent_report["evaluations"]
                if m.get("task") == "a5" and m["update"] == GLOBAL_START and m["rows"] == 102400}
    require(set(previous) == {"dev", "ood_dev"}, "Parent full A5 endpoint metrics are missing")
    run["parent_a5_metrics"] = previous
    run["baseline_remeasurement"] = {
        role: {"parent": previous[role]["whole_word_exact_match"],
               "phase_zero": metrics_at(run, 0)[f"a5/{role}"]["whole_word_exact_match"],
               "delta_percentage_points": 100 * (metrics_at(run, 0)[f"a5/{role}"]["whole_word_exact_match"]
                                                  - previous[role]["whole_word_exact_match"])}
        for role in previous}
    return run


def compare_control(run, control):
    require(run["mode"] == "mixed-curriculum" and control["mode"] == "a5-control",
            "Pair the mixed curriculum with an A5-only continuation")
    require(run["parent"]["sha256"] == control["parent"]["sha256"], "Paired arms use different parents")
    for key in ("model_config", "initialization", "optimizer", "runtime", "streams", "batch_per_task"):
        require(run["report"]["contract"][key] == control["report"]["contract"][key],
                f"Paired control {key} differs")
    require(run["data_identity"] == control["data_identity"], "Paired control dataset identity differs")
    require(run["sources"] == control["sources"], "Paired control frozen sources differ")
    for left, right in zip(run["history"], control["history"]):
        require(left["phase_update"] == right["phase_update"]
                and left["order_chains"]["a5"] == right["order_chains"]["a5"],
                "Paired A5 data order differs")
    comparisons = []
    for phase in sorted(run["checkpoints"].keys() & control["checkpoints"].keys()):
        a, b = metrics_at(run, phase), metrics_at(control, phase)
        for role in sorted(a.keys() & b.keys()):
            if a[role]["evaluated_rows"] != b[role]["evaluated_rows"]:
                continue
            comparisons.append({"phase_update": phase, "global_update": GLOBAL_START + phase,
                                "role": role, "a5_examples_seen": PARENT_A5_EXAMPLES + BATCH * phase,
                                "run": a[role], "control": b[role]})
    return {"compatible": True, "matched_a5_order_updates": min(run["endpoint"], control["endpoint"]),
            "comparisons": comparisons,
            "qualification": "Same parent, A5 data order and A5 exposure; Fuzzy exposure, task weights and total compute differ. "
                             "Only equal-phase, equal-evaluation-size comparisons are paired; curves end at actual observations."}


def make_plots(run, control, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.right": False})
    series = [(run, "#C55A36")]
    if control is not None:
        series.append((control, "#3478AF"))
    figures = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=160)
        plt.close(fig)
        figures.append(name)

    def axes_labels(axis, ylabel, *, accuracy=False):
        axis.set(xlabel="Additional optimizer updates (phase)", ylabel=ylabel)
        axis.set_xlim(0, max(current["endpoint"] for current, _ in series))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        if accuracy:
            axis.set_ylim(-2, 102)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        top = axis.secondary_xaxis("top", functions=(lambda x: x + GLOBAL_START, lambda x: x - GLOBAL_START))
        top.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        top.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        top.set_xlabel("Global Adam updates", fontsize=9)

    def trajectory(axis, current, color, task, role, key):
        metrics = [m for m in selected_evaluations(current["evaluations"])
                   if m["task"] == task and m["role"] == role]
        axis.plot([m["phase_update"] for m in metrics], [100*m[key] for m in metrics],
                  color=color, label=LABELS[current["mode"]], lw=1.6)
        for full, marker in ((True, "o"), (False, "x")):
            selected = [m for m in metrics if (m["scope"] == "full") == full]
            axis.scatter([m["phase_update"] for m in selected], [100*m[key] for m in selected],
                         marker=marker, s=20, color=color)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for row, (role, length) in enumerate((("dev", 12), ("ood_dev", 36))):
        for axis, key, title in zip(axes[row], ("token_accuracy", "final_state_accuracy", "whole_word_exact_match"),
                                    ("Mean token accuracy", "Final-state accuracy", "Whole-word exactness")):
            for current, color in series:
                trajectory(axis, current, color, "a5", role, key)
            axis.set_title(f"A5 length {length}: {title}", pad=45)
            axes_labels(axis, "Accuracy (%)", accuracy=True)
    fig.suptitle("A5 retention from a trained 10k parent — one-seed development\n"
                 "Phase zero is measured after restoration; circles: 102,400 words; crosses: monitoring subset")
    save(fig, "a5-retention")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for axis, key, title in zip(axes, ("answer_accuracy", "answer_motif_exact_match", "sequence_exact_match"),
                                ("Answer tokens", "Answer-motif exactness", "All-answers sequence exactness")):
        for current, color in series:
            trajectory(axis, current, color, "fuzzy", "dev", key)
        axis.set_title(title, pad=45)
        axes_labels(axis, "Accuracy / teacher-forced exactness (%)", accuracy=True)
    fig.suptitle("Fuzzy Recall development — 1,280 sequences; A5-only arm receives no Fuzzy training")
    save(fig, "fuzzy-learning")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for current, color in series:
        history = current["history"]
        name = LABELS[current["mode"]]
        axes[0, 0].plot([0] + [r["phase_update"] for r in history],
                        [0] + [r["task_weights"]["fuzzy"] for r in history], color=color, label=name)
        windows = [history[i:i+50] for i in range(0, len(history), 50)]
        x = [w[-1]["phase_update"] for w in windows]
        axes[0, 1].plot(x, [sum(r["loss"] for r in w)/len(w) for w in windows],
                        color=color, label=name, marker="o", ms=2)
        for task in history[0]["tasks"]:
            for axis, key in ((axes[1, 0], "ce"), (axes[1, 1], "latent")):
                axis.plot(x, [sum(r["tasks"][task][key] for r in w)/len(w) for w in windows],
                          color=color, ls="--" if task == "fuzzy" else "-", marker="o", ms=2,
                          label=f"{name}: {task.upper()}")
    for axis, title, ylabel in zip(axes.flat,
            ("Fuzzy loss coefficient", "Weighted total training objective", "Unweighted task CE", "Unweighted task NextLat SmoothL1"),
            ("Weight", "50-update mean", "50-update task mean", "50-update task mean")):
        axis.set_title(title, pad=45)
        axes_labels(axis, ylabel)
    axes[0, 0].set_ylim(-.02, .52)
    fig.suptitle("Loss ramp uses phase updates; Fuzzy CE is dense, A5 CE scores every state\n"
                 "Task objectives are averaged separately before weighting; NextLat weight remains one")
    save(fig, "curriculum-and-losses")
    return figures


def compact_run(run):
    return {"directory": run["directory"], "mode": run["mode"], "training_status": run["report"]["status"],
            "phase_endpoint": run["endpoint"], "global_endpoint": GLOBAL_START + run["endpoint"],
            "checkpoint": run["checkpoints"][run["endpoint"]], "phase_zero_checkpoint": run["checkpoints"][0],
            "baseline": metrics_at(run, 0), "final": metrics_at(run, run["endpoint"]),
            "parent": run["parent"], "parent_a5_metrics": run["parent_a5_metrics"],
            "baseline_remeasurement": run["baseline_remeasurement"], "contract": run["report"]["contract"],
            "input_hashes": run["input_hashes"], "training_wandb": run["report"].get("wandb"),
            "task_offsets": expected_offsets(run["mode"], run["endpoint"]),
            "additional_task_presentations": {"a5": BATCH * run["endpoint"],
                "fuzzy": BATCH * run["endpoint"] if run["mode"] == "mixed-curriculum" else 0},
            "evaluations": run["evaluations"]}


def write_markdown(summary, output):
    lines = ["# A5 warm-start continuation and Fuzzy curriculum", "",
        f"**{LABELS[summary['mode']]}: phase {summary['phase_endpoint']:,}, "
        f"global Adam update {summary['global_endpoint']:,}.** Training status: {summary['training_status']}.", "",
        "The phase starts from A5-only training at global update 10,000, including its Adam state. "
        f"The parent's saved L36 whole-word accuracy was {100*summary['parent_a5_metrics']['ood_dev']['whole_word_exact_match']:.4f}%. "
        "The baseline column below is a new full evaluation of the restored checkpoint.", "",
        "| Metric | Restored phase 0 | Same-checkpoint endpoint |", "| --- | ---: | ---: |"]
    for role, length in (("dev", 12), ("ood_dev", 36)):
        a, b = summary["baseline"][f"a5/{role}"], summary["final"][f"a5/{role}"]
        for label, key in (("mean token", "token_accuracy"), ("final state", "final_state_accuracy"),
                           ("whole word", "whole_word_exact_match")):
            counts = f" ({b['whole_word_correct']:,}/{b['rows']:,})" if key == "whole_word_exact_match" else ""
            lines.append(f"| A5 L{length} {label} | {100*a[key]:.4f}% | {100*b[key]:.4f}%{counts} |")
    a, b = summary["baseline"]["fuzzy/dev"], summary["final"]["fuzzy/dev"]
    for label, key in (("answer token", "answer_accuracy"), ("answer motif", "answer_motif_exact_match"),
                       ("all-answer sequence", "sequence_exact_match")):
        lines.append(f"| Fuzzy {label} | {100*a[key]:.4f}% | {100*b[key]:.4f}% |")
    offsets = summary["task_offsets"]
    lines += ["", f"Cumulative A5 presentations: {offsets['a5']:,}; additional A5 presentations: "
              f"{summary['additional_task_presentations']['a5']:,}; Fuzzy presentations: {offsets['fuzzy']:,}. "
              "Global updates include 10,000 A5-only parent updates. Each active task contributes 128 examples per phase update.", "",
              "The mixed objective uses `w_FR(s) = 0.5 * min(s/3000, 1)` and "
              "`L = (1-w_FR)*L_A5 + w_FR*L_FR`; the first optimizer update uses `s=1`. "
              "The control uses only A5. Both retain task-local CE plus NextLat weight one.", "",
              QUALIFICATION, ""]
    if summary["control"] is not None:
        control = summary["control"]
        lines += [f"The paired A5-only control ends at phase {control['phase_endpoint']:,}, "
                  f"global {control['global_endpoint']:,}. {summary['paired_control']['qualification']}", "",
                  "| Endpoint comparison at equal phase | Curriculum | A5-only continuation |", "| --- | ---: | ---: |"]
        endpoint_pairs = [p for p in summary["paired_control"]["comparisons"]
                          if p["phase_update"] == summary["phase_endpoint"]]
        for pair in endpoint_pairs:
            key = "whole_word_exact_match" if pair["role"].startswith("a5/") else "answer_accuracy"
            lines.append(f"| {pair['role']} {key} | {100*pair['run'][key]:.4f}% | {100*pair['control'][key]:.4f}% |")
        if not endpoint_pairs:
            lines += ["No equal-phase full endpoint comparison exists; no different-checkpoint metrics are combined."]
        lines += [""]
    lines += [f"Parent checkpoint: `{summary['parent']['path']}`", "",
              f"Parent SHA256: `{summary['parent']['sha256']}`", "",
              f"Endpoint checkpoint: `{summary['checkpoint']['path']}`", "",
              f"Endpoint SHA256: `{summary['checkpoint']['sha256']}`", "",
              "Exact input, source, checkpoint and figure hashes are recorded in `evidence.json`.", ""]
    if summary["training_wandb"]:
        url = summary["training_wandb"].get("run_url")
        if url:
            lines += [f"[Training curves in W&B]({url})", ""]
    for name in summary["figures"]:
        lines += [f"![{name.replace('-', ' ')}]({name}.png)", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def build_report(args):
    run = load_run(args.train)
    control = load_run(args.control) if args.control is not None else None
    comparison = compare_control(run, control) if control is not None else None
    summary = {"schema": SCHEMA, "status": "complete", **compact_run(run),
               "control": compact_run(control) if control is not None else None,
               "paired_control": comparison, "qualification": QUALIFICATION,
               "confirmation_evaluated": False, "latent_rollout_evaluated": False,
               "report_source_sha256": sha(Path(__file__))}
    args.output.mkdir(parents=True, exist_ok=False)
    summary["figures"] = make_plots(run, control, args.output)
    write_markdown(summary, args.output)
    summary["artifact_sha256"] = {path.name: sha(path) for path in sorted(args.output.iterdir())}
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (args.output / "evidence.json").write_text(text)
    (args.output / "report.json").write_text(text)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train", type=Path, required=True, help="Completed control or curriculum training directory")
    result.add_argument("--control", type=Path, help="Paired A5-only continuation directory")
    result.add_argument("--output", type=Path, required=True, help="New report directory; never overwritten")
    return result


def main():
    args = parser().parse_args()
    summary = build_report(args)
    print(json.dumps({"status": "complete", "phase_endpoint": summary["phase_endpoint"],
                      "global_endpoint": summary["global_endpoint"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
