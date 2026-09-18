#!/usr/bin/env python3
"""Saved common-checkpoint and unequal-endpoint comparison of the four-layer uninjected control and stopped conservative input run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_l1r_depth_report import validate_history as validate_full_history
from scripts.rt_a5_conservative_input_report import bound, coefficient, require, validate_endpoint, validate_history as validate_scheduled_history
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, local_path, metric_rows, training_curve
from scripts.rt_a5_report import hash_file, read_input, write_json


ROOT = Path(__file__).resolve().parents[1]
LINEAGES = {
    "control": ROOT / ".runtime/rt-a5/20260915T164000Z-l1r-four-layer-control10k",
    "conservative": ROOT / ".runtime/rt-a5/20260915T163300Z-input-conservative10k",
}
OUTPUT = ROOT / "docs/reports/rt-a5/control-vs-stopped-conservative"
SCHEMA = "rt-a5-stopped-conservative-control-comparison-v1"
STEPS = (1000, 5000, 10000)
LABELS = {"control": "No embedding injection", "conservative": "Conservative input injection"}
COLORS = {"control": "#286E9D", "conservative": "#B25335"}
REPORTING_SOURCES = ("scripts/rt_a5_stopped_conservative_control_report.py", "scripts/rt_a5_conservative_input_report.py",
    "scripts/rt_a5_l1r_depth_report.py", "scripts/rt_a5_nextlat_report.py", "scripts/rt_a5_length_report.py",
    "scripts/rt_a5_report.py", "scripts/experiment_tracking.py")


def validate_pair(control, conservative):
    shared = ("architecture", "model_config", "width", "seed", "predictor_seed", "data_order_seed", "batch_size",
        "length", "train_rows", "data_manifest_sha256", "precision", "tf32", "compile", "cuda_graphs",
        "torch", "cuda", "device_capability", "optimizer", "word_order", "evaluation_route",
        "latent_rollout_evaluated", "objective", "nextlat_config")
    require(all(control["contract"][key] == conservative["contract"][key] for key in shared),
            "Control and conservative core/data/loss/optimizer/runtime contracts differ")
    require(control["contract"]["model_config"]["n_layers"] == 4
            and control["contract"]["objective"]["latent_weight"] == 1,
            "Both arms must use four layers and the original NextLat objective")
    left, right = control["initialization"], conservative["initialization"]
    require(left["parameter_count"] == 13702656 and left["parameter_tensors"] == 43
            and right["parameter_count"] == 13964800 and right["parameter_tensors"] == 44
            and left["model_parameter_sha256"] == right["shared_four_layer_model_sha256"]
            and left["predictor_sha256"] == right["predictor_sha256"],
            "Exactly43 initial backbone/predictor tensors must match; only the injection arm adds Pe")
    require(all(conservative["source_files"].get(name) == digest for name, digest in control["source_files"].items()),
            "Historical shared58 sources differ")
    for arm in (control, conservative):
        experiment = arm["contract"]["experiment_config"]
        require(experiment["n_layers"] == 4 and experiment["window_layer"] == 0 and experiment["window_length"] == 2,
                "Both arms require the same first-window four-layer architecture")
    require(control["contract"]["experiment_config"]["attention"] == conservative["contract"]["experiment_config"]["attention"],
            "Layer attention patterns differ")
    count = min(len(control["history"]), len(conservative["history"]))
    require(count == conservative["completed_updates"] and count > 0
            and [r["order_chain"] for r in control["history"][:count]]
                == [r["order_chain"] for r in conservative["history"]],
            "All common minibatch orders must match before the conservative stop")


def load_evidence():
    arms, payloads = {}, {}
    for name, lineage in LINEAGES.items():
        inputs = {}
        def capture(key, path):
            raw, record = read_input(path)
            inputs[key] = record; payloads[f"{name}_{key}"] = raw
            return raw
        protocol = json.loads(capture("protocol", lineage / "protocol.json"))
        require(protocol["schema"] == ("rt-a5-l1r-four-layer-control-protocol-v2" if name == "control"
                                       else "rt-a5-input-conservative-protocol-v1"), "Unexpected pilot protocol schema")
        directory = local_path(protocol["training_directory"]).resolve()
        require(directory == lineage / ("train-control" if name == "control" else "train-conservative"),
                "Declared training directory differs")
        require(protocol["endpoint"] == 10000 and protocol["start_update"] == 0 and protocol["n_layers"] == 4,
                "Require the separate fresh four-layer10k protocols")
        report = json.loads(capture("report", directory / "report.json"))
        expected_schema = "rt-a5-l1r-depth-training-v1" if name == "control" else "rt-a5-conservative-input-training-v1"
        end = report["completed_updates"]
        require(report["schema"] == expected_schema and report["start_update"] == 0
                and report["endpoint"] == 10000 and report["parent_checkpoint"] is None
                and report["wandb"]["status"] == "synced"
                and report["confirmation_evaluated"] is False and report["latent_rollout_evaluated"] is False
                and (lineage / "training-exit-code.txt").read_text().strip() == "0",
                "Require successful, fresh, synced terminal reports")
        if name == "control":
            require(report["status"] == "complete" and end == 10000, "Control must complete its actual10k endpoint")
        else:
            require(validate_endpoint(report, protocol, None) == (end, False)
                    and report["status"] == "stopped" and 0 < end < 10000,
                    "Conservative counterpart must be the explicit earlier graceful stop")
        config = json.loads(capture("config", directory / "config.json"))
        require(config == protocol["resolved_args"] and config["resume"] is None
                and report["contract"] == protocol["strict_contract"] and report["initialization"] == protocol["initialization"]
                and report["source_files"] == protocol["source_files"], "Saved training and frozen protocol differ")
        sources = report["source_files"]
        require(len(sources) == (58 if name == "control" else 65)
                and _digest_dict(sources) == protocol["source_sha256"] == report["contract"]["source_sha256"],
                "Source identity differs")
        for relative, digest in sources.items():
            require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Invalid source path")
            require(hash_file(ROOT / relative)["sha256"] == hash_file(directory / "source" / relative)["sha256"] == digest,
                    "Frozen source or retained snapshot changed")
        bound(protocol["preflight"])
        require(json.loads(capture("preflight", local_path(protocol["preflight"]["path"])))["passed"] is True,
                "Actual-shape preflight did not pass")
        bound(protocol["data_manifest"], local_path(config["data_dir"]) / "manifest.json")
        capture("data_manifest", local_path(config["data_dir"]) / "manifest.json")
        require(inputs["data_manifest"]["sha256"] == report["contract"]["data_manifest_sha256"], "Data manifest differs")
        history = [json.loads(line) for line in capture("history", directory / "history.jsonl").splitlines()]
        if name == "control":
            validate_full_history(history)
        else:
            validate_scheduled_history(history, end, arms["control"]["history"], exact=True)
        require(history[-1]["order_chain"] == report["order_chain"], "History and terminal order counters differ")
        if name == "conservative":
            require(report["injection_coefficient"] == coefficient(end)
                    and all(row["injection_coefficient"] == coefficient(row["update"]) for row in history),
                    "Conservative schedule differs")
        else:
            require(all("injection_coefficient" not in row for row in history), "Uninjected control unexpectedly has injection instrumentation")
        steps = sorted(set(s for s in protocol["checkpoint_steps"] if s <= end) | {end})
        require([c["completed_updates"] for c in report["checkpoints"]] == steps, "Retained scheduled/actual terminal checkpoints differ")
        checkpoints = {str(c["completed_updates"]): bound(c, directory / f"checkpoints/step-{c['completed_updates']:06d}.pt")
                       for c in report["checkpoints"]}
        state = json.loads(capture("saved_state", lineage / "final-state-validation.json"))
        require(state.get("schema") == ("rt-a5-l1r-four-layer-control-final-state-validation-v2" if name == "control"
                                        else "rt-a5-conservative-input-final-state-validation-v1")
                and state.get("passed") is True and state["completed_updates"] == end
                and state["model_parameter_tensors"] == state["final_active_adam_states"] == (43 if name == "control" else 44)
                and state["protocol"]["sha256"] == inputs["protocol"]["sha256"]
                and state["report"]["sha256"] == inputs["report"]["sha256"]
                and state["source_sha256"] == protocol["source_sha256"], "Bound final model/Adam proof is missing")
        require(all(state["checkpoint_inputs"][step]["sha256"] == row["sha256"] for step, row in checkpoints.items()),
                "Final state proof checked different checkpoints")
        curves = {}
        for step in steps[1:]:
            curves[str(step)] = {}
            for role in ROLES:
                selected = [e for e in report["evaluations"] if e["update"] == step and e["role"] == role]
                require(len(selected) == 1 and selected[0]["rows"] == 102400 and selected[0]["route"] == "backbone_only",
                        "Matched checkpoints require the same full development pools")
                curves[str(step)][role] = metric_rows(selected[0], name)
        arms[name] = {"protocol": protocol, "contract": report["contract"], "initialization": report["initialization"],
            "source_files": sources, "inputs": inputs, "checkpoints": checkpoints, "curves": curves,
            "history": history, "wandb": report["wandb"], "saved_state": state,
            "completed_updates": end, "training_status": report["status"], "checkpoint_updates": steps[1:]}
    validate_pair(arms["control"], arms["conservative"])
    gate = arms["control"]["protocol"]["authorization"]
    require(gate["schema"] == "rt-a5-l1r-four-layer-control-authorization-v2"
            and gate["passed"] is True and gate["authorization_kind"] == "explicit_unconditional_user_request"
            and gate["accuracy_condition"] is None, "Require the superseding unconditional control authorization")
    bound(gate["request"])
    for key, source in (("conservative_protocol", "protocol"), ("conservative_report", "report"), ("conservative_history", "history")):
        require(bound(gate[key])["sha256"] == arms["conservative"]["inputs"][source]["sha256"],
                "Control authorization refers to different stopped conservative evidence")
    for step, record in gate["conservative_checkpoints"].items():
        require(bound(record)["sha256"] == arms["conservative"]["checkpoints"][step]["sha256"],
                "Control authorization checkpoint differs")
    stopped_at = arms["conservative"]["completed_updates"]
    require(gate["conservative_completed_updates"] == stopped_at and gate["conservative_training_status"] == "stopped",
            "Control authorization uses a different conservative terminal update")
    bound(gate["conservative_stop_request"])
    for role in ROLES:
        require(metric_rows(gate["evaluations"][role], "conservative") == arms["conservative"]["curves"][str(stopped_at)][role],
                "Authorization counterpart evaluation differs from saved terminal rows")
    common = sorted(set(arms["control"]["checkpoint_updates"]) & set(arms["conservative"]["checkpoint_updates"]))
    require(common, "Require at least one common retained full-evaluation checkpoint")
    for arm in arms.values():
        arm["training_curve"] = training_curve(arm.pop("history"), augmented=True)
    return {"schema": SCHEMA, "primary_update": common[-1], "matched_update": common[-1],
        "common_checkpoint_updates": common, "actual_endpoints": {name: arm["completed_updates"] for name, arm in arms.items()},
        "arms": arms, "shared_initial_parameter_tensors": 43, "additional_projection_parameters": 262144,
        "matched_minibatch_order_hashes": stopped_at, "confirmation_evaluated": False, "latent_rollout_evaluated": False,
        "qualification": "One seed and reused development data. The user stopped the conservative run early and then unconditionally authorized the fresh uninjected control to 10k. Matched comparisons use only common retained checkpoints; actual terminal results have unequal training budgets and are displayed separately. Both arms share 43 initial backbone/predictor tensors and the same data order; only the injection arm adds Pe and its scheduled contribution. Conservative warmup remains incomplete. No confirmation or other-depth comparison is included."}, payloads


def make_summary():
    return load_evidence()[0]


def plot_rows(summary, arm, step=None):
    step = summary["matched_update"] if step is None else step
    rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
    require([(r["arm"], r["update"], r["role"], r["length"]) for r in rows]
            == [(arm, step, "ood_dev", t) for t in range(1, 37)], "Full/boundary plots need identical36-prefix rows")
    return rows


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    figures = []
    def save(fig, name):
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        figures.append(name); plt.close(fig)
    cases = (("length-full", (1, 36), False), ("length-boundary", (10, 18), False),
             ("length-unequal-terminals", (1, 36), True))
    for name, limits, terminals in cases:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
        fig.subplots_adjust(left=.08, right=.99, bottom=.17, top=.79, wspace=.4)
        for axis, metric in zip(axes, ("E", "A", "M")):
            for arm in LINEAGES:
                step = summary["actual_endpoints"][arm] if terminals else summary["matched_update"]
                rows = plot_rows(summary, arm, step)
                axis.plot([r["length"] for r in rows], [r[metric] for r in rows],
                          label=f"{LABELS[arm]} at {step:,}", color=COLORS[arm])
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix t of the same 36-token words", title=f"{metric}(t)")
            axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.axvline(12, color="gray", linestyle=":")
            axis.grid(alpha=.2); axis.legend(fontsize=8)
        title = "Actual terminal checkpoints: UNEQUAL training budgets" if terminals else f"Matched {summary['matched_update']:,}-update checkpoint"
        fig.suptitle(f"Four-layer L1R + NextLat · {title} · 102,400 words", y=.97)
        save(fig, name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for arm in LINEAGES:
        steps = summary["arms"][arm]["checkpoint_updates"]
        axes[0].plot(steps, [plot_rows(summary, arm, step)[-1]["E"] for step in steps], marker="o", color=COLORS[arm], label=LABELS[arm])
        bins = summary["arms"][arm]["training_curve"]
        for key, linestyle, lossname in (("state_ce", "-", "state CE"), ("latent_loss", "--", "latent loss")):
            axes[1].plot([r["update"] for r in bins], [r[key] for r in bins], linestyle=linestyle, color=COLORS[arm], label=f"{LABELS[arm]}: {lossname}")
    axes[0].set(xlabel="Completed updates", ylabel="Whole-word E(36)", ylim=(-.025, 1.025), title="Retained full evaluations; different stopping updates")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(xlabel="Completed updates", ylabel="Loss", title="Training loss, 100-update means")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=7)
    save(fig, "learning-curves")
    return figures


def markdown(summary):
    lines = ["# Four-layer control versus stopped conservative input injection", "",
        f"The matched comparison uses {summary['matched_update']:,} updates. Actual endpoints are "
        f"control {summary['actual_endpoints']['control']:,} and conservative {summary['actual_endpoints']['conservative']:,}; these terminal budgets are unequal.", "",
        "E(t): every state through t correct; A(t): state t alone correct; M(t): mean token accuracy through t. "
        "Full and boundary curves use the same saved length-36 prefix rows.", "",
        "| Matched checkpoint | Model | E(36) | A(36) | M(36) |", "|---:|---|---:|---:|---:|"]
    for step in summary["common_checkpoint_updates"]:
        for arm in LINEAGES:
            row = plot_rows(summary, arm, step)[-1]
            lines.append(f"| {step:,} | {LABELS[arm]} | {row['E']:.4%} | {row['A']:.4%} | {row['M']:.4%} |")
    lines += ["", "Actual terminal checkpoints (unequal budgets):", "",
              "| Model | Completed updates | E(36) | A(36) | M(36) |", "|---|---:|---:|---:|---:|"]
    for arm in LINEAGES:
        step = summary["actual_endpoints"][arm]; row = plot_rows(summary, arm, step)[-1]
        lines.append(f"| {LABELS[arm]} | {step:,} | {row['E']:.4%} | {row['A']:.4%} | {row['M']:.4%} |")
    lines += ["", summary["qualification"], "",
        "[Matched full curve](length-full.pdf) · [Matched boundary](length-boundary.pdf) · "
        "[Unequal terminals](length-unequal-terminals.pdf) · [Learning curves](learning-curves.pdf)", ""]
    return "\n".join(lines)


def run(args):
    summary, payloads = load_evidence()
    output = local_path(args.output_dir).resolve()
    require(not output.exists(), "Use a new comparison output directory")
    output.mkdir(parents=True); (output / "inputs").mkdir()
    for name, raw in payloads.items():
        (output / "inputs" / f"{name}.json{'l' if name.endswith('_history') else ''}").write_bytes(raw)
    for arm_name, arm in summary["arms"].items():
        directory = local_path(arm["protocol"]["training_directory"])
        for relative in arm["source_files"]:
            dest = output / "training-source" / arm_name / relative
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(directory / "source" / relative, dest)
    for relative in REPORTING_SOURCES:
        dest = output / "reporting-source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / relative, dest)
    write_json(output / "summary.json", summary)
    rows = [row for arm in summary["arms"].values() for step in arm["checkpoint_updates"] for role in ROLES for row in arm["curves"][str(step)][role]]
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS); writer.writeheader(); writer.writerows(rows)
    training_rows = [{"arm": name, **row} for name, arm in summary["arms"].items() for row in arm["training_curve"]]
    with (output / "training-curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in training_rows for key in row}))
        writer.writeheader(); writer.writerows(training_rows)
    figures = plots(summary, output)
    (output / "README.md").write_text(markdown(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
        group=args.wandb_group or LINEAGES["control"].name, name="four-layer-control-vs-stopped-conservative")
    result = {"schema": SCHEMA, "status": "running", "primary_update": summary["matched_update"], "matched_update": summary["matched_update"],
              "actual_endpoints": summary["actual_endpoints"], "figures": figures}
    try:
        tracker.start({"schema": SCHEMA, "qualification": summary["qualification"]})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(CSV_COLUMNS), data=[[r[k] for k in CSV_COLUMNS] for r in rows]),
                     **{f"report/{name}": wandb.Image(str(output / f"{name}.png")) for name in figures}})
        tracker.summary({"matched_update": summary["matched_update"],
            **{f"{arm}/matched_E36": plot_rows(summary, arm)[-1]["E"] for arm in LINEAGES},
            **{f"{arm}/terminal_E36": plot_rows(summary, arm, summary["actual_endpoints"][arm])[-1]["E"] for arm in LINEAGES}})
        tracker.finish(succeeded=True); result["status"] = "complete"
        (output / "README.md").write_text(markdown(summary) + f"\n[Report W&B]({tracker.record['run_url']})\n")
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        try: tracker.finish(succeeded=False)
        except Exception: pass
        raise
    finally:
        result["wandb"] = tracker.record
        result["artifacts"] = {str(p.relative_to(output)): hash_file(p) for p in sorted(output.rglob("*"))
            if p.is_file() and "wandb" not in p.relative_to(output).parts and p != output / "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": tracker.record["run_url"]}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--wandb-group")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
