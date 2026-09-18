#!/usr/bin/env python3
"""CPU reports of saved quadratic-input evaluations; no model inference.

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
LINEAGE = ROOT / ".runtime/rt-a5/20260915T160000Z-input-quadratic50k"
SCHEMA = "rt-a5-quadratic-input-report-v1"
TRAIN_SCHEMA = "rt-a5-quadratic-input-training-v1"
SOURCE_SHA = "aea19cef549b65563fc9816ce68128520ca5f3a5eb3b829c0688753935b5c942"
REPORTING_SOURCES = ("scripts/rt_a5_quadratic_input_report.py", "scripts/rt_a5_nextlat_report.py",
    "scripts/rt_a5_length_report.py", "scripts/rt_a5_report.py", "scripts/experiment_tracking.py")
COLUMNS = (*CSV_COLUMNS, "injection_coefficient")


def coefficient(update):
    return .05 * min(max(update, 0) / 50000, 1) ** 2


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
            and report.get("parent_checkpoint") is None and report.get("endpoint") == 50000,
            "Require the fresh quadratic-input lineage")
    require(report.get("confirmation_evaluated") is False and report.get("latent_rollout_evaluated") is False,
            "Confirmation and autonomous rollout must remain unevaluated")
    if through_update is not None:
        require(type(through_update) is int and 0 < through_update <= 50000,
                "Partial report update must be a positive retained checkpoint")
        require(report.get("status") in ("running", "complete", "stopped"),
                "A partial report cannot clear a failed training job")
        require(through_update in [row["completed_updates"] for row in report["checkpoints"]],
                "Partial reports require an already retained checkpoint")
        return through_update, True
    end = report.get("completed_updates")
    require(type(end) is int and 0 < end <= 50000 and report.get("wandb", {}).get("status") == "synced",
            "Endpoint reports require a finished synced training job")
    if report.get("status") == "complete":
        require(end == 50000 and report.get("requested_endpoint_reached") is True,
                "Complete means the requested 50k endpoint was reached")
    else:
        stop = report.get("stop_request") or {}
        require(report.get("status") == "stopped" and end < 50000
                and report.get("requested_endpoint_reached") is False
                and stop.get("reason") == "user_stop_file" and stop.get("observed_after_update") == end
                and local_path(stop["path"]).resolve() == local_path(protocol["resolved_args"]["stop_file"]).resolve(),
                "Earlier endpoint must be an explicit graceful user stop")
        require(hash_file(local_path(stop["path"]))["sha256"] == stop["sha256"], "Retained stop sentinel changed")
    require(report.get("injection_coefficient") == coefficient(end), "Terminal scheduled coefficient differs")
    return end, False


def validate_history(rows, end, reference, *, exact):
    require(len(rows) >= end and (not exact or len(rows) == end), "Missing or excess terminal history rows")
    selected = rows[:end]
    require(len(reference) >= end, "Missing original minibatch-order reference")
    for update, (row, old) in enumerate(zip(selected, reference), 1):
        require(row["update"] == old["update"] == update and row["order_chain"] == old["order_chain"]
                and row["examples_seen"] == update * 1024 and row["injection_coefficient"] == coefficient(update),
                "History update, schedule, exposure or original minibatch order differs")
        for key in ("loss", "state_loss", "latent_loss", "weighted_latent_loss", "grad_norm", "seconds", "token_accuracy"):
            require(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]), "Nonfinite history metric")
        require(math.isclose(row["loss"], row["state_loss"] + row["weighted_latent_loss"], rel_tol=2e-6, abs_tol=1e-7)
                and row["weighted_latent_loss"] == row["latent_loss"], "Original weight-one NextLat objective differs")
    return selected


def load_evidence(lineage=LINEAGE, through_update=None):
    lineage = local_path(lineage).resolve()
    inputs, payloads = {}, {}
    def capture(name, path):
        raw, record = read_input(path)
        inputs[name], payloads[name] = record, raw
        return raw
    protocol = json.loads(capture("protocol", lineage / "protocol.json"))
    require(protocol.get("schema") == "rt-a5-input-quadratic-protocol-v1"
            and protocol["start_update"] == 0 and protocol["endpoint"] == 50000
            and protocol["n_layers"] == 4 and protocol["variant"] == "input" and protocol["injection_layer"] == 1
            and protocol["initial_coefficient"] == 0 and protocol["maximum_coefficient"] == .05
            and protocol["warmup_updates"] == 50000, "Prospective quadratic schedule differs")
    directory = local_path(protocol["training_directory"]).resolve()
    require(directory == lineage / "train-quadratic", "Training directory differs")
    report = json.loads(capture("training_report", directory / "report.json"))
    config = json.loads(capture("training_config", directory / "config.json"))
    require(config == protocol["resolved_args"] and config["resume"] is None, "Resolved fresh training config differs")
    require(report["contract"] == protocol["strict_contract"] and report["initialization"] == protocol["initialization"]
            and report["source_files"] == protocol["source_files"], "Training record differs from frozen protocol")
    contract, initial = report["contract"], report["initialization"]
    require(len(protocol["source_files"]) == 65
            and _digest_dict(protocol["source_files"]) == SOURCE_SHA == protocol["source_sha256"] == contract["source_sha256"],
            "Frozen 65-source identity differs")
    schedule = contract["injection_schedule"]
    require(schedule == report["schedule"] == initial["schedule"]
            and schedule["maximum_coefficient"] == .05 and schedule["warmup_updates"] == 50000
            and schedule["power"] == 2 and schedule["initial_coefficient"] == 0 and schedule["learned"] is False,
            "Saved schedule semantics differ")
    require(initial["parameter_count"] == 13964800 and initial["parameter_tensors"] == 44
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
        require((lineage / "training-exit-code.txt").read_text().strip() == "0", "Require successful terminal process exit")
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
        converted = metric_rows(evaluation, "quadratic_input")
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
        require(state.get("schema") == "rt-a5-quadratic-input-final-state-validation-v1" and state.get("passed") is True
                and state["completed_updates"] == end and state["model_parameter_tensors"] == state["final_active_adam_states"] == 44
                and state["protocol"]["sha256"] == inputs["protocol"]["sha256"]
                and state["report"]["sha256"] == inputs["training_report"]["sha256"]
                and state["source_sha256"] == SOURCE_SHA, "Require the bound terminal saved-state proof")
        require(all(state["checkpoint_inputs"][step]["sha256"] == record["sha256"] for step, record in checkpoints.items()),
                "Saved-state proof checked different checkpoints")
    binned = training_curve(selected, augmented=True)
    for row in binned:
        row["injection_coefficient"] = coefficient(row["update"])
    summary = {"schema": SCHEMA, "report_scope": "partial retained checkpoint" if partial else "terminal saved endpoint",
        "partial": partial, "training_status_at_snapshot": report["status"], "through_update": end,
        "requested_endpoint": 50000, "requested_endpoint_reached": not partial and report["requested_endpoint_reached"],
        "schedule": schedule, "injection_coefficient": coefficient(end), "checkpoint_updates": steps,
        "checkpoints": checkpoints, "metrics": metrics, "curves": curves, "metric_rows": rows,
        "training_curve": binned, "matched_minibatch_order_hashes": end,
        "observed_complete_history_rows": len(history), "excluded_later_history_rows": len(history) - end,
        "protocol": protocol, "inputs": inputs, "training_wandb": report["wandb"], "saved_state": state,
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
        "qualification": "One seed and repeatedly inspected development pools. This fresh run shares learned initialization with fixed input injection; only the nonlearned schedule changes. A stopped or partial endpoint is selected after reviewing development evidence. No unmodified four-layer control or confirmation result is supplied."}
    return summary, payloads


def make_summary(lineage=LINEAGE, through_update=None):
    return load_evidence(lineage, through_update)[0]


def plot_rows(summary, step):
    rows = summary["curves"][str(step)]["ood_dev"]
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
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        figures.append(name)
        plt.close(fig)
    steps = summary["checkpoint_updates"][1:]
    shown = sorted({steps[0], steps[-1], *[step for step in (10000, 20000) if step in steps]})
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
        fig.subplots_adjust(left=.08, right=.99, bottom=.17, top=.78, wspace=.4)
        for axis, key, title in zip(axes, ("E", "A", "M"), ("All states through t correct", "Only state t correct", "Mean token accuracy through t")):
            for step in shown:
                rows = plot_rows(summary, step)
                axis.plot([r["length"] for r in rows], [r[key] for r in rows],
                          label=f"{step:,} updates; λ={coefficient(step):.4g}")
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix t of the same 36-token words", title=f"{key}(t): {title}")
            axis.axvline(12, color="gray", linestyle=":")
            axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend(fontsize=7)
        fig.suptitle(f"Quadratic input injection · {summary['report_scope']} through {summary['through_update']:,} · 102,400 words", y=.97)
        save(fig, name)
    fig, axis = plt.subplots(figsize=(10, 4.7), layout="constrained")
    for key in ("E", "A", "M"):
        axis.plot(steps, [plot_rows(summary, step)[-1][key] for step in steps], marker="o", label=f"{key}(36)")
    axis.set(xlabel="Completed training updates", ylabel="Accuracy", ylim=(-.025, 1.025),
             title="Full 102,400-word development evaluations at retained checkpoints")
    axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend()
    save(fig, "length36-vs-updates")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    bins = summary["training_curve"]
    for key, label in (("state_ce", "State CE"), ("latent_loss", "Latent SmoothL1")):
        axes[0].plot([b["update"] for b in bins], [b[key] for b in bins], label=label)
    axes[0].set(xlabel="Completed training updates", ylabel="Loss", title="Training loss, 100-update means")
    axes[0].grid(alpha=.2); axes[0].legend()
    end = summary["through_update"]
    x = sorted(set(range(0, end + 1, 100)) | {end})
    axes[1].plot(x, [coefficient(step) for step in x], label="Observed schedule")
    if end < 50000:
        future = sorted(set(range(end, 50001, 100)) | {50000})
        axes[1].plot(future, [coefficient(step) for step in future], linestyle="--", color="gray", label="Planned schedule beyond this report")
    axes[1].set(xlabel="Completed training updates", ylabel="Scheduled coefficient λ", title="λ(s) = 0.05 × min(s / 50,000, 1)²")
    axes[1].grid(alpha=.2); axes[1].legend(fontsize=8)
    save(fig, "training-and-schedule")
    return figures


def markdown(summary):
    end = summary["through_update"]
    lines = [f"# Quadratic input injection through {end:,} updates", "",
        f"{summary['report_scope'].capitalize()}; λ={coefficient(end):.6g}. Planned endpoint: 50,000 updates. "
        f"Training status when read: {summary['training_status_at_snapshot']}.", "",
        "E(t) is the fraction of words with every state through t correct; A(t) is accuracy at state t alone; "
        "M(t) is mean token accuracy through t. Full and boundary figures use identical prefixes of the same 102,400 length-36 words.", "",
        "| Update | λ | E(36) | A(36) | M(36) |", "|---:|---:|---:|---:|---:|"]
    for step in summary["checkpoint_updates"][1:]:
        row = plot_rows(summary, step)[-1]
        lines.append(f"| {step:,} | {coefficient(step):.6g} | {row['E']:.4%} | {row['A']:.4%} | {row['M']:.4%} |")
    lines += ["", summary["qualification"], "",
        "[Full length curve](length-full.pdf) · [Boundary](length-boundary.pdf) · "
        "[Length36 trajectory](length36-vs-updates.pdf) · [Training and schedule](training-and-schedule.pdf)", "",
        f"[Training W&B]({summary['training_wandb']['run_url']})", ""]
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
        columns = sorted({key for row in summary["training_curve"] for key in row})
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(summary["training_curve"])
    figures = plots(summary, output)
    (output / "README.md").write_text(markdown(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
        group=args.wandb_group or local_path(args.lineage).name,
        name=f"quadratic-input-report-through-{summary['through_update']:06d}")
    result = {"schema": SCHEMA, "status": "running", "through_update": summary["through_update"],
              "partial": summary["partial"], "figures": figures}
    try:
        tracker.start({"schema": SCHEMA, "schedule": summary["schedule"], "through_update": summary["through_update"],
                       "partial": summary["partial"], "qualification": summary["qualification"]})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(COLUMNS), data=[[row[k] for k in COLUMNS] for row in summary["metric_rows"]]),
                     **{f"report/{name}": wandb.Image(str(output / f"{name}.png")) for name in figures}})
        for row in summary["training_curve"]:
            tracker.log({"update": row["update"], "schedule/injection_coefficient": row["injection_coefficient"],
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--through-update", type=int, help="Explicit partial report at a retained checkpoint")
    parser.add_argument("--wandb-group")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
