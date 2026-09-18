#!/usr/bin/env python3
"""CPU reports of saved six-layer fixed-input evaluations; no model inference.

Default mode requires completed/stopped training and its saved-state receipt.
An explicit --through-update selects an already retained checkpoint and labels
the report partial. Every output directory must be new.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json


ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / ".runtime/rt-a5/20260915T171500Z-six-layer-input001-10k"
SCHEMA = "rt-a5-six-layer-input-report-v1"
TRAIN_SCHEMA = "rt-a5-six-layer-input-training-v1"
SOURCE_SHA = "cb73ced068d1d01557251f2a7f5d43840fb9bfbcc0e05c6f7821e6ce35b7d877"
REPORTING_SOURCES = ("scripts/rt_a5_six_layer_input_report.py", "scripts/rt_a5_nextlat_report.py",
    "scripts/rt_a5_length_report.py", "scripts/rt_a5_report.py", "scripts/experiment_tracking.py")
COLUMNS = (*CSV_COLUMNS, "injection_coefficient")


def coefficient(update):
    return .01  # Fixed at initialization and every training/evaluation checkpoint.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(record, expected=None):
    path = local_path(record["path"]).resolve()
    if expected is not None:
        require(path == Path(expected).resolve(), "Bound input is outside the expected lineage")
    actual = hash_file(path)
    require(all(record[k] == actual[k] for k in ("sha256", "bytes")), "Bound input hash/size differs")
    return actual


def validate_endpoint(report, protocol, through_update):
    require(report.get("schema") == TRAIN_SCHEMA and report.get("start_update") == 0
            and report.get("parent_checkpoint") is None and report.get("endpoint") == 10000,
            "Require the fresh six-layer fixed-input lineage")
    require(report.get("confirmation_evaluated") is False and report.get("latent_rollout_evaluated") is False,
            "Confirmation and autonomous rollout must remain unevaluated")
    if through_update is not None:
        require(type(through_update) is int and 0 < through_update <= 10000,
                "Partial report update must be a positive retained checkpoint")
        require(report.get("status") in ("running", "complete", "stopped"),
                "A partial report cannot clear a failed training job")
        require(through_update in [row["completed_updates"] for row in report["checkpoints"]],
                "Partial reports require an already retained checkpoint")
        return through_update, True
    end = report.get("completed_updates")
    require(type(end) is int and 0 < end <= 10000 and report.get("wandb", {}).get("status") == "synced",
            "Endpoint reports require a finished synced training job")
    if report.get("status") == "complete":
        require(end == 10000 and report.get("requested_endpoint_reached") is True,
                "Complete means the requested 10k pilot endpoint was reached")
    else:
        stop = report.get("stop_request") or {}
        require(report.get("status") == "stopped" and end < 10000
                and report.get("requested_endpoint_reached") is False
                and stop.get("reason") == "user_stop_file" and stop.get("observed_after_update") == end
                and local_path(stop["path"]).resolve() == local_path(protocol["resolved_args"]["stop_file"]).resolve(),
                "Earlier endpoint must be an explicit graceful user stop")
        require(hash_file(local_path(stop["path"]))["sha256"] == stop["sha256"], "Retained stop sentinel changed")
    require(report.get("injection_coefficient") == coefficient(end), "Terminal fixed coefficient differs")
    return end, False


def validate_history(rows, end, reference, *, exact):
    require(len(rows) >= end and (not exact or len(rows) == end), "Missing or excess terminal history rows")
    selected = rows[:end]
    require(len(reference) >= end, "Missing original minibatch-order reference")
    for update, (row, old) in enumerate(zip(selected, reference), 1):
        require(row["update"] == old["update"] == update and row["order_chain"] == old["order_chain"]
                and row["examples_seen"] == update * 1024 and row["injection_coefficient"] == coefficient(update),
                "History update, fixed coefficient, exposure or original minibatch order differs")
        for key in ("loss", "state_loss", "latent_loss", "weighted_latent_loss", "grad_norm", "seconds", "token_accuracy"):
            require(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]), "Nonfinite history metric")
        require(math.isclose(row["loss"], row["state_loss"] + row["weighted_latent_loss"], rel_tol=2e-6, abs_tol=1e-7)
                and row["weighted_latent_loss"] == row["latent_loss"], "Original weight-one NextLat objective differs")
    return selected


def baseline_evidence(capture, current, curves, order_reference, reference_binding):
    """Read the closed six-layer baseline and compare only shared budgets."""
    lineage = ROOT / ".runtime/rt-a5/20260915T141414Z-l1r-six-layer-nextlat10k"
    directory = lineage / "train-depth"
    protocol = json.loads(capture("baseline_protocol", lineage / "protocol.json"))
    old = json.loads(capture("baseline_report", directory / "report.json"))
    config = json.loads(capture("baseline_config", directory / "config.json"))
    require(protocol["schema"] == "rt-a5-l1r-depth-protocol-v1"
            and local_path(protocol["training_directory"]).resolve() == directory
            and protocol["n_layers"] == 6 and config == protocol["resolved_args"],
            "Expected the original successful six-layer baseline")
    require(old["schema"] == "rt-a5-l1r-depth-training-v1" and old["status"] == "complete"
            and old["completed_updates"] == old["endpoint"] == 10000 and old["start_update"] == 0
            and old["parent_checkpoint"] is None and old["wandb"]["status"] == "synced"
            and old["contract"] == protocol["strict_contract"] and old["initialization"] == protocol["initialization"]
            and old["confirmation_evaluated"] is False and old["latent_rollout_evaluated"] is False,
            "Require the completed saved 10k six-layer baseline")
    for key in ("model_config", "batch_size", "length", "train_rows", "data_manifest_sha256", "data_order_seed",
                "seed", "predictor_seed", "objective", "optimizer", "nextlat_config", "precision"):
        require(old["contract"][key] == current["contract"][key], "Matched baseline core/data/objective contract differs")
    require(old["initialization"]["parameter_count"] == 19998208 and old["initialization"]["parameter_tensors"] == 61
            and old["initialization"] == current["initialization"]["reference_six_layer_initialization"]
            and old["initialization"]["model_parameter_sha256"] == current["initialization"]["shared_six_layer_model_sha256"],
            "Baseline must share the exact 61 initial backbone/predictor tensors")
    require(len(old["source_files"]) == 58 and old["source_files"] == protocol["source_files"]
            and all(current["source_files"].get(name) == digest for name, digest in old["source_files"].items()),
            "Baseline historical 58 source identity differs")
    for name, digest in old["source_files"].items():
        require(hash_file(directory / "source" / name)["sha256"] == digest, "Baseline source snapshot changed")
    for key in ("report", "protocol", "config", "history"):
        expected = lineage / "protocol.json" if key == "protocol" else directory / (key + (".jsonl" if key == "history" else ".json"))
        bound(reference_binding[key], expected)
    state = json.loads(capture("baseline_state", lineage / "final-state-validation.json"))
    require(state["schema"] == "rt-a5-l1r-depth-final-state-validation-v1" and state["passed"] is True
            and state["completed_updates"] == 10000 and state["model_parameter_tensors"] == 61,
            "Baseline final saved-state proof is required")
    require(bound(state["report"])["sha256"] == hash_file(directory / "report.json")["sha256"]
            and bound(state["protocol"])["sha256"] == hash_file(lineage / "protocol.json")["sha256"],
            "Baseline proof binds different report/protocol files")
    history = [json.loads(line) for line in capture("baseline_history", directory / "history.jsonl").splitlines()]
    require(len(history) == len(order_reference) == 10000 and history[-1]["order_chain"] == old["order_chain"],
            "Baseline minibatch-order history is incomplete")
    validate_history([{**row, "injection_coefficient": .01} for row in history], 10000, order_reference, exact=True)
    checkpoints = {}
    require([c["completed_updates"] for c in old["checkpoints"]] == [0, 1000, 5000, 10000], "Baseline checkpoints differ")
    for record in old["checkpoints"]:
        step = record["completed_updates"]
        checkpoints[str(step)] = bound(record, directory / f"checkpoints/step-{step:06d}.pt")
        require(checkpoints[str(step)]["sha256"] == state["checkpoint_inputs"][str(step)]["sha256"]
                == reference_binding["checkpoints"][str(step)]["sha256"], "Baseline checkpoint binding differs")
    baseline_curves, rows, seen = {}, [], set()
    for evaluation in old["evaluations"]:
        step, role = evaluation["update"], evaluation["role"]
        require((step, role) not in seen and role in ROLES and evaluation["route"] == "backbone_only"
                and evaluation["rows"] == (102400 if str(step) in checkpoints else 4096),
                "Baseline evaluation pool or route differs")
        seen.add((step, role))
        converted = metric_rows(evaluation, "baseline")
        for row in converted:
            row["injection_coefficient"] = 0.
        rows.extend(converted)
        if str(step) in checkpoints:
            baseline_curves.setdefault(str(step), {})[role] = converted
    require(seen == {(step, role) for step in range(500, 10001, 500) for role in ROLES}, "Missing baseline evaluation")
    common_steps = sorted(set(map(int, curves)) & set(map(int, baseline_curves)))
    comparisons = []
    for step in common_steps:
        prior, actual = baseline_curves[str(step)]["ood_dev"][-1], curves[str(step)]["ood_dev"][-1]
        comparisons.append({"update": step, "words": 102400,
            "baseline": {"coefficient": 0., **{key: prior[key] for key in ("E", "A", "M")}},
            "injection": {"coefficient": .01, **{key: actual[key] for key in ("E", "A", "M")}}})
    return {"completed_updates": 10000, "checkpoint_updates": [0,1000,5000,10000],
        "checkpoints": checkpoints, "curves": baseline_curves, "metric_rows": rows,
        "training_curve": training_curve(history, augmented=True), "saved_state": state,
        "parameter_count": 19998208, "parameter_tensors": 61, "wandb": old["wandb"],
        "common_checkpoint_updates": common_steps, "comparisons": comparisons}


def load_evidence(lineage=LINEAGE, through_update=None):
    lineage = local_path(lineage).resolve()
    inputs, payloads = {}, {}
    def capture(name, path):
        raw, record = read_input(path)
        inputs[name], payloads[name] = record, raw
        return raw
    protocol = json.loads(capture("protocol", lineage / "protocol.json"))
    require(protocol.get("schema") == "rt-a5-six-layer-input-protocol-v1"
            and protocol["start_update"] == 0 and protocol["endpoint"] == 10000
            and protocol["n_layers"] == 6 and protocol["variant"] == "input" and protocol["injection_layer"] == 1
            and protocol["coefficient"] == .01 and protocol["coefficient_learned"] is False,
            "Prospective six-layer fixed input configuration differs")
    directory = local_path(protocol["training_directory"]).resolve()
    require(directory == lineage / "train-injection", "Training directory differs")
    report = json.loads(capture("training_report", directory / "report.json"))
    config = json.loads(capture("training_config", directory / "config.json"))
    require(config == protocol["resolved_args"] and config["resume"] is None, "Resolved fresh training config differs")
    require(report["contract"] == protocol["strict_contract"] and report["initialization"] == protocol["initialization"]
            and report["source_files"] == protocol["source_files"], "Training record differs from frozen protocol")
    contract, initial = report["contract"], report["initialization"]
    require(len(protocol["source_files"]) == 65
            and _digest_dict(protocol["source_files"]) == SOURCE_SHA == protocol["source_sha256"] == contract["source_sha256"],
            "Frozen 65-source identity differs")
    require(contract["injection_coefficient"] == report["injection_coefficient"] == initial["coefficient"] == .01
            and contract["coefficient_learned"] is report["coefficient_learned"] is initial["coefficient_learned"] is False
            and contract["n_layers"] == initial["n_layers"] == 6
            and initial["baseline_parameter_tensors_changed"] == [], "Fixed input semantics differ")
    require(initial["parameter_count"] == 20260352 and initial["parameter_tensors"] == 62
            and contract["precision"] == "fp32" and contract["batch_size"] == 1024
            and contract["objective"]["latent_weight"] == 1 and contract["evaluation_route"] == "backbone_only",
            "Unexpected model count, precision, objective or evaluation route")
    for name, digest in protocol["source_files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid source path")
        require(hash_file(ROOT / name)["sha256"] == hash_file(directory / "source" / name)["sha256"] == digest,
                f"Frozen source/current snapshot changed: {name}")
    bound(protocol["data_manifest"], local_path(config["data_dir"]) / "manifest.json")
    manifest = capture("data_manifest", local_path(config["data_dir"]) / "manifest.json")
    require(hashlib.sha256(manifest).hexdigest() == contract["data_manifest_sha256"], "Data manifest differs")
    bound(protocol["preflight"])
    require(json.loads(capture("preflight", local_path(protocol["preflight"]["path"])))["passed"] is True,
            "Actual-shape preflight must have passed")
    end, partial = validate_endpoint(report, protocol, through_update)
    if not partial:
        require(capture("training_exit_code", lineage / "training-exit-code.txt").strip() == b"0", "Require successful terminal process exit")
        if report["status"] == "stopped":
            capture("stop_request", local_path(report["stop_request"]["path"]))
    reference_binding = protocol["reference"]
    bound(reference_binding["report"])
    reference_report = json.loads(capture("order_reference_report", local_path(reference_binding["report"]["path"])))
    for key in ("batch_size", "train_rows", "length", "data_order_seed", "data_manifest_sha256"):
        require(reference_report["contract"][key] == contract[key], "Original minibatch-order reference contract differs")
    reference_path = local_path(reference_binding["training_directory"]) / "history.jsonl"
    reference = [json.loads(line) for line in capture("order_reference_history", reference_path).splitlines()]
    raw_history = capture("training_history", directory / "history.jsonl")
    lines = raw_history.splitlines()
    if partial and lines and not raw_history.endswith(b"\n"):
        lines = lines[:-1]  # The last append may be in flight; the retained prefix must still be complete.
    history = [json.loads(line) for line in lines]
    selected = validate_history(history, end, reference, exact=not partial)
    if not partial:
        require(selected[-1]["order_chain"] == report["order_chain"], "History/report endpoint order differs")
    steps = sorted(set(step for step in protocol["checkpoint_steps"] if step <= end) | {end})
    recorded = [c for c in report["checkpoints"] if c["completed_updates"] <= end]
    require([c["completed_updates"] for c in recorded] == steps, "Retained checkpoint sequence differs")
    checkpoints = {}
    for row in recorded:
        step = row["completed_updates"]
        require(row["injection_coefficient"] == coefficient(step) and row["examples_seen"] == step * 1024,
                "Checkpoint coefficient or exposure differs")
        checkpoints[str(step)] = bound(row, directory / f"checkpoints/step-{step:06d}.pt")
    metrics, curves, rows = {}, {}, []
    seen = set()
    for evaluation in report["evaluations"]:
        step, role = evaluation["update"], evaluation["role"]
        if step > end:
            continue
        require((step, role) not in seen and step > 0 and role in ROLES
                and evaluation["route"] == "backbone_only" and evaluation["injection_coefficient"] == coefficient(step),
                "Duplicate or incompatible saved evaluation")
        seen.add((step, role))
        full = step in steps
        require(evaluation["rows"] == (102400 if full else 4096), "Saved evaluation pool size differs")
        converted = metric_rows(evaluation, "six_layer_input")
        for row in converted:
            row["injection_coefficient"] = coefficient(step)
        rows.extend(converted)
        if full:
            metrics.setdefault(str(step), {})[role] = evaluation
            curves.setdefault(str(step), {})[role] = converted
    evaluation_steps = sorted(set(range(protocol["evaluation_every"], end + 1, protocol["evaluation_every"])) | set(steps[1:]))
    require(seen == {(step, role) for step in evaluation_steps for role in ROLES}, "Missing saved development evaluation")
    state = None
    if not partial:
        state = json.loads(capture("saved_state_validation", lineage / "final-state-validation.json"))
        require(state.get("schema") == "rt-a5-six-layer-input-final-state-validation-v1" and state.get("passed") is True
                and state["completed_updates"] == end and state["model_parameter_tensors"] == state["final_active_adam_states"] == 62
                and state["protocol"]["sha256"] == inputs["protocol"]["sha256"]
                and state["report"]["sha256"] == inputs["training_report"]["sha256"]
                and state["source_sha256"] == SOURCE_SHA
                and state["model_parameters"] == 20260352
                and state["all61_baseline_initial_tensors_bitwise_equal"] is True
                and state["shared_six_layer_model_sha256"] == initial["shared_six_layer_model_sha256"],
                "Require the bound terminal saved-state proof")
        require(all(state["checkpoint_inputs"][step]["sha256"] == record["sha256"] for step, record in checkpoints.items()),
                "Saved-state proof checked different checkpoints")
    binned = training_curve(selected, augmented=True)
    for row in binned:
        row["injection_coefficient"] = coefficient(row["update"])
    baseline = baseline_evidence(capture, report, curves, reference, reference_binding)
    rows.extend(baseline["metric_rows"])
    summary = {"schema": SCHEMA, "report_scope": "partial retained checkpoint" if partial else "terminal saved endpoint",
        "partial": partial, "training_status_at_snapshot": report["status"], "through_update": end,
        "requested_endpoint": 10000, "requested_endpoint_reached": not partial and report["requested_endpoint_reached"],
        "injection_coefficient": .01, "coefficient_learned": False, "checkpoint_updates": steps,
        "checkpoints": checkpoints, "metrics": metrics, "curves": curves, "metric_rows": rows,
        "training_curve": binned, "matched_minibatch_order_hashes": end,
        "observed_complete_history_rows": len(history), "excluded_later_history_rows": len(history) - end,
        "protocol": protocol, "inputs": inputs, "training_wandb": report["wandb"], "saved_state": state,
        "baseline": baseline, "common_checkpoint_updates": baseline["common_checkpoint_updates"],
        "matched_update": max(baseline["common_checkpoint_updates"], default=None),
        "actual_endpoints": {"injection": end, "baseline": 10000},
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
        "qualification": "One seed and repeatedly inspected development pools. This fresh six-layer run shares all 61 baseline backbone/predictor initial tensors and the same data order; its only added parameter is Pe (262,144 weights), injected before block 1 with fixed coefficient 0.01. The six-layer baseline has 19,998,208 parameters; the injected model has 20,260,352. The experiment was chosen after previous development results. Matched comparisons use only common retained checkpoint budgets; an earlier stopped or partial endpoint is retrospective and its unequal terminal is shown separately. No confirmation or autonomous latent rollout was evaluated."}
    return summary, payloads


def make_summary(lineage=LINEAGE, through_update=None):
    return load_evidence(lineage, through_update)[0]


def plot_rows(summary, step, arm="injection"):
    view = summary if arm == "injection" else summary["baseline"]
    rows = view["curves"][str(step)]["ood_dev"]
    require([(row["update"], row["role"], row["length"]) for row in rows]
            == [(step, "ood_dev", length) for length in range(1, 37)],
            "Full and boundary figures require the identical 36-prefix rows")
    return rows


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    figures = []
    labels = {"baseline": "Original six-layer baseline", "injection": "Six layers + input injection λ=0.01"}
    colors = {"baseline": "#3574B2", "injection": "#D65B35"}
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        figures.append(name)
        plt.close(fig)
    def prefix_figure(name, limits, selected, title):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
        fig.subplots_adjust(left=.08, right=.99, bottom=.18, top=.78, wspace=.4)
        for axis, key, label in zip(axes, ("E", "A", "M"), ("All states through t correct", "Only state t correct", "Mean token accuracy through t")):
            for arm, step in selected:
                rows = plot_rows(summary, step, arm)
                axis.plot([r["length"] for r in rows], [r[key] for r in rows], color=colors[arm],
                          label=f"{labels[arm]} ({step:,})")
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix t of the same 36-token words", title=f"{key}(t): {label}")
            axis.axvline(12, color="gray", linestyle=":")
            axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend(fontsize=7)
        fig.suptitle(title + " · 102,400 words", y=.97)
        save(fig, name)
    end, matched = summary["through_update"], summary["matched_update"]
    selected = [("baseline", matched), ("injection", matched)] if matched is not None else [("injection", end)]
    title = f"Same training budget: {matched:,} updates" if matched is not None else "Injection endpoint; no shared retained checkpoint yet"
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        prefix_figure(name, limits, selected, title)
    if end != 10000:
        prefix_figure("length-unequal-terminals", (1, 36), [("baseline", 10000), ("injection", end)],
                      f"Unequal terminal budgets: baseline 10,000 versus injection {end:,}; descriptive only")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), layout="constrained")
    for axis, key in zip(axes, ("E", "A", "M")):
        for arm in labels:
            view = summary if arm == "injection" else summary["baseline"]
            steps = view["checkpoint_updates"][1:]
            axis.plot(steps, [plot_rows(summary, step, arm)[-1][key] for step in steps], marker="o", color=colors[arm], label=labels[arm])
        axis.set(xlabel="Completed training updates", ylabel=f"{key}(36)", ylim=(-.025, 1.025))
        axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.suptitle("Full 102,400-word development evaluations at each arm's retained checkpoints")
    save(fig, "length36-vs-updates")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
    for axis, key, title in zip(axes, ("state_ce", "latent_loss"), ("State cross entropy", "NextLat SmoothL1")):
        for arm in labels:
            view = summary if arm == "injection" else summary["baseline"]
            bins = view["training_curve"]
            axis.plot([b["update"] for b in bins], [b[key] for b in bins], color=colors[arm], label=labels[arm])
        axis.set(xlabel="Completed training updates", ylabel="Loss", title=title + ", 100-update means")
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    save(fig, "training-losses")
    return figures


def markdown(summary):
    end = summary["through_update"]
    lines = [f"# Six-layer input injection through {end:,} updates", "",
        f"{summary['report_scope'].capitalize()}; λ is fixed at 0.01, including initialization. "
        f"The requested pilot endpoint is 10,000 updates; training status when read: {summary['training_status_at_snapshot']}.", "",
        "The first block uses window-2 attention and the remaining five use full recurrent attention. Only the input to block 1 receives 0.01 × Pe(raw token embedding), before its existing normalization. NextLat training and backbone-only evaluation are unchanged.", "",
        "E(t) is the fraction of words with every state through t correct; A(t) is accuracy at state t alone; "
        "M(t) is mean token accuracy through t. Full and boundary figures use identical prefixes of the same 102,400 length-36 words.", "",
        "| Injection update | E(36) | A(36) | M(36) |", "|---:|---:|---:|---:|"]
    for step in summary["checkpoint_updates"][1:]:
        row = plot_rows(summary, step)[-1]
        lines.append(f"| {step:,} | {row['E']:.4%} | {row['A']:.4%} | {row['M']:.4%} |")
    if summary["baseline"]["comparisons"]:
        lines += ["", "Original six-layer baseline versus injection at common saved budgets:", "",
            "| Update | Arm | λ | E(36) | A(36) | M(36) |", "|---:|---|---:|---:|---:|---:|"]
        for point in summary["baseline"]["comparisons"]:
            for arm, label in (("baseline", "Original six-layer baseline"), ("injection", "Six-layer input injection")):
                values = point[arm]
                lines.append(f"| {point['update']:,} | {label} | {values['coefficient']:.3g} | {values['E']:.4%} | {values['A']:.4%} | {values['M']:.4%} |")
    if end != 10000:
        lines += ["", f"Actual endpoints differ: injection {end:,}, baseline 10,000. Terminal performance is descriptive, not an equal-budget comparison. "
                  "[Unequal terminal curves](length-unequal-terminals.pdf)"]
    lines += ["", summary["qualification"], "",
        "[Matched full length curve](length-full.pdf) · [Matched boundary](length-boundary.pdf) · "
        "[Length36 trajectory](length36-vs-updates.pdf) · [Training losses](training-losses.pdf)", "",
        f"[Injection training W&B]({summary['training_wandb']['run_url']}) · [Baseline training W&B]({summary['baseline']['wandb']['run_url']})", ""]
    return "\n".join(lines)


def run(args):
    summary, payloads = load_evidence(args.lineage, args.through_update)
    output = local_path(args.output_dir).resolve()
    require(not output.exists(), "Use a new report output directory")
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for name, raw in payloads.items():
        suffix = Path(summary["inputs"][name]["path"]).suffix
        (output / "inputs" / f"{name}{suffix}").write_bytes(raw)
    training_directory = local_path(summary["protocol"]["training_directory"])
    for relative in summary["protocol"]["source_files"]:
        destination = output / "training-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(training_directory / "source" / relative, destination)
    for relative in REPORTING_SOURCES:
        destination = output / "reporting-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    write_json(output / "summary.json", summary)
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS); writer.writeheader(); writer.writerows(summary["metric_rows"])
    with (output / "training-curve.csv").open("w") as stream:
        bins = [{**row, "arm": "injection", "injection_coefficient": .01} for row in summary["training_curve"]]
        bins += [{**row, "arm": "baseline", "injection_coefficient": 0.} for row in summary["baseline"]["training_curve"]]
        columns = sorted({key for row in bins for key in row})
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(bins)
    figures = plots(summary, output)
    (output / "README.md").write_text(markdown(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
        group=args.wandb_group or local_path(args.lineage).name,
        name=f"six-layer-input-report-through-{summary['through_update']:06d}")
    result = {"schema": SCHEMA, "status": "running", "through_update": summary["through_update"],
              "partial": summary["partial"], "matched_update": summary["matched_update"],
              "actual_endpoints": summary["actual_endpoints"], "figures": figures}
    try:
        tracker.start({"schema": SCHEMA, "injection_coefficient": .01, "through_update": summary["through_update"],
                       "partial": summary["partial"], "qualification": summary["qualification"]})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(COLUMNS), data=[[row[k] for k in COLUMNS] for row in summary["metric_rows"]]),
                     **{f"report/{name}": wandb.Image(str(output / f"{name}.png")) for name in figures}})
        for row in summary["training_curve"]:
            tracker.log({"update": row["update"], "injection/coefficient": row["injection_coefficient"],
                         **{f"train/{key}": row[key] for key in ("state_ce", "latent_loss", "loss")}})
        for step in summary["checkpoint_updates"][1:]:
            row = plot_rows(summary, step)[-1]
            tracker.log({"update": step, **{f"dev/ood_dev/{key}36": row[key] for key in ("E", "A", "M")}})
        tracker.summary({"through_update": summary["through_update"], "partial": summary["partial"],
                         "injection_coefficient": summary["injection_coefficient"]})
        tracker.finish(succeeded=True); result["status"] = "complete"
        (output / "README.md").write_text(markdown(summary) + f"\n[Report W&B]({tracker.record['run_url']})\n")
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        try:
            tracker.finish(succeeded=False)
        except Exception:
            pass
        raise
    finally:
        result["wandb"] = tracker.record
        result["artifacts"] = {str(path.relative_to(output)): hash_file(path) for path in sorted(output.rglob("*"))
                               if path.is_file() and "wandb" not in path.relative_to(output).parts and path != output / "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": tracker.record["run_url"]}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", default=str(LINEAGE))
    parser.add_argument("--output-dir", default=str(ROOT / "docs/reports/rt-a5/six-layer-input001-10k"))
    parser.add_argument("--through-update", type=int, help="Explicit partial report at a retained checkpoint")
    parser.add_argument("--wandb-group")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
