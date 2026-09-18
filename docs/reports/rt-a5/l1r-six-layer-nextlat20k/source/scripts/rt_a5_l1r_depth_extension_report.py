#!/usr/bin/env python3
"""Saved-evidence 20k extension of the six-versus-two-layer first-window diagnostic."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil

from scripts import rt_a5_l1r_depth_report as base
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, _sha, local_path, metric_rows, training_curve
from scripts.rt_a5_report import finite_number, hash_file, read_input, write_json

ROOT = base.ROOT
LINEAGE = ROOT / ".runtime/rt-a5/20260915T144415Z-l1r-six-layer-nextlat20k"
OUTPUT = ROOT / "docs/reports/rt-a5/l1r-six-layer-nextlat20k"
SCHEMA = "rt-a5-l1r-depth-extension-comparison-v1"
ENDPOINT, STEPS = 20000, (1000, 5000, 10000, 20000)
ARMS, LABELS, COLORS = base.ARMS, base.LABELS, base.COLORS
REPORTING_SOURCES = ("scripts/rt_a5_l1r_depth_extension_report.py", *base.REPORTING_SOURCES)


def validate_segment(rows, start, stop):
    if len(rows) != stop - start or [row.get("update") for row in rows] != list(range(start + 1, stop + 1)):
        raise ValueError("Missing, duplicated, reset or out-of-order continuation updates")
    for row in rows:
        if row.get("examples_seen") != row["update"] * 1024 or not _sha(row.get("order_chain")):
            raise ValueError("Global training exposure/order counter differs")
        for key in ("seconds", "loss", "state_loss", "latent_loss", "weighted_latent_loss", "grad_norm",
                    "token_accuracy", "whole_word_exact"):
            if not finite_number(row.get(key)) or row[key] < 0:
                raise ValueError(f"Invalid continuation training metric:{key}")
        if row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError("Invalid continuation training accuracy")
        if (not math.isclose(row["weighted_latent_loss"], row["latent_loss"], rel_tol=1e-7, abs_tol=1e-9)
                or not math.isclose(row["loss"], row["state_loss"] + row["weighted_latent_loss"], rel_tol=2e-7, abs_tol=2e-7)):
            raise ValueError("Continuation objective differs from unchanged CE plus NextLat")


def read_history(path, start, stop, *, exact_file):
    rows = []
    with local_path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) == stop - start:
                break
        if exact_file and any(line.strip() for line in stream):
            raise ValueError("History exceeds the authorized completed segment")
    validate_segment(rows, start, stop)
    return rows


def stitch_histories(parent, child, reference):
    validate_segment(parent, 0, 10000)
    validate_segment(child, 10000, ENDPOINT)
    validate_segment(reference, 0, ENDPOINT)
    combined = parent + child
    if [row["order_chain"] for row in combined] != [row["order_chain"] for row in reference]:
        raise ValueError("Every corresponding minibatch order through20k must match")
    return combined


def select_checkpoint(report, directory, step):
    entries = [item for item in report["checkpoints"] if item["completed_updates"] == step]
    if len(entries) != 1 or local_path(entries[0]["path"]).resolve() != directory / "checkpoints" / f"step-{step:06d}.pt":
        raise ValueError(f"Need exactly one correctly located retained checkpoint at{step}")
    return base.bound_file(entries[0])


def selected_evaluations(report, arm, step):
    if step not in STEPS:
        raise ValueError("Only matched1k/5k/10k/20k checkpoints enter the primary comparison")
    curves, metrics = {}, {}
    for role in ROLES:
        selected = [item for item in report["evaluations"] if item["update"] == step and item["role"] == role]
        if len(selected) != 1 or selected[0].get("rows") != 102400 or selected[0].get("route") != "backbone_only":
            raise ValueError("Primary comparison requires matching102400-word backbone evaluations")
        metrics[role] = selected[0]
        curves[role] = metric_rows(selected[0], arm)
    return curves, metrics


def validate_child(report, config, protocol, parent):
    if (report.get("schema") != "rt-a5-l1r-depth-training-v1" or report.get("status") != "complete"
            or report.get("start_update") != 10000 or report.get("completed_updates") != ENDPOINT
            or report.get("endpoint") != ENDPOINT or report.get("wandb", {}).get("status") != "synced"
            or report.get("confirmation_evaluated") is not False or report.get("latent_rollout_evaluated") is not False):
        raise ValueError("Need the completed, synced10k-to20k continuation")
    for key in ("contract", "source_files", "initialization"):
        expected = protocol["strict_contract" if key == "contract" else key]
        if report.get(key) != parent[key] or report[key] != expected:
            raise ValueError(f"Continuation changed parent{key}")
    checkpoint = protocol["parent_checkpoint"]
    actual_parent = report.get("parent_checkpoint") or {}
    if (actual_parent.get("sha256") != checkpoint["sha256"]
            or local_path(actual_parent.get("path", "")).resolve() != local_path(checkpoint["path"]).resolve()
            or config.get("resume") is None
            or local_path(config["resume"]).resolve() != local_path(checkpoint["path"]).resolve()):
        raise ValueError("Continuation did not resume the exact approved parent checkpoint")
    if sorted(item["completed_updates"] for item in report["checkpoints"]) != [15000, ENDPOINT]:
        raise ValueError("Continuation must retain exactly15k and20k checkpoints")
    if _digest_dict(report["source_files"]) != protocol["source_sha256"] or report["contract"]["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("Continuation source digest differs")


def make_summary(protocol_path):
    raw, protocol_file = read_input(local_path(protocol_path))
    protocol = json.loads(raw)
    if (protocol.get("schema") != "rt-a5-l1r-depth-extension-protocol-v1" or protocol.get("start_update") != 10000
            or protocol.get("endpoint") != ENDPOINT or protocol.get("checkpoint_steps") != [15000, ENDPOINT]
            or protocol.get("primary_checkpoint_steps") != list(STEPS) or not protocol.get("qualification")):
        raise ValueError("Require the authorized user-extended20k protocol")
    for name in ("parent_protocol", "parent_report", "parent_checkpoint"):
        base.bound_file(protocol[name])
    if protocol["parent_checkpoint"].get("completed_updates") != 10000:
        raise ValueError("Continuation parent must be the10k checkpoint")
    parent_summary = base.make_summary(protocol["parent_protocol"]["path"])
    arms = copy.deepcopy(parent_summary["arms"])
    new, reference = (arms[arm] for arm in ARMS)
    if (new["inputs"]["report"]["sha256"] != protocol["parent_report"]["sha256"]
            or new["checkpoints"]["10000"]["sha256"] != protocol["parent_checkpoint"]["sha256"]):
        raise ValueError("Bound parent report/checkpoint differs from the completed10k run")
    ref_binding = protocol["reference"]
    if local_path(ref_binding["training_directory"]).resolve() != base.REFERENCE:
        raise ValueError("Use the same closed two-layer reference")
    if (ref_binding["source_files"] != reference["source_files"]
            or ref_binding["source_sha256"] != reference["contract"]["source_sha256"]):
        raise ValueError("Reference source binding differs")
    for name in ("report", "config"):
        if base.bound_file(ref_binding[name])["sha256"] != reference["inputs"][name]["sha256"]:
            raise ValueError("Reference report/config binding differs")
    child_dir = local_path(protocol["training_directory"]).resolve()
    child_raw, child_file = read_input(child_dir / "report.json")
    child = json.loads(child_raw)
    config_raw, config_file = read_input(child_dir / "config.json")
    config = json.loads(config_raw)
    validate_child(child, config, protocol, new)
    for name, wanted in child["source_files"].items():
        for path in (child_dir / "source" / name, ROOT / name):
            if hash_file(path)["sha256"] != wanted:
                raise ValueError(f"Continuation frozen source changed:{name}")
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != new["contract"]["data_manifest_sha256"]:
        raise ValueError("Continuation training data changed")
    child_history = read_history(child_dir / "history.jsonl", 10000, ENDPOINT, exact_file=True)
    parent_history = read_history(new["inputs"]["history"]["path"], 0, 10000, exact_file=True)
    reference_history = read_history(reference["inputs"]["history"]["path"], 0, ENDPOINT, exact_file=False)
    combined = stitch_histories(parent_history, child_history, reference_history)
    if child_history[-1]["order_chain"] != child["order_chain"]:
        raise ValueError("Continuation report/history endpoint order differs")
    for step in (15000, ENDPOINT):
        new["checkpoints"][str(step)] = select_checkpoint(child, child_dir, step)
    ref_raw = json.loads(local_path(reference["inputs"]["report"]["path"]).read_text())
    reference["checkpoints"][str(ENDPOINT)] = select_checkpoint(ref_raw, base.REFERENCE, ENDPOINT)
    for step in (0, *STEPS):
        if base.bound_file(ref_binding["checkpoints"][str(step)])["sha256"] != reference["checkpoints"][str(step)]["sha256"]:
            raise ValueError("Reference retained checkpoint binding differs")
    new["curves"][str(ENDPOINT)], new["metrics"][str(ENDPOINT)] = selected_evaluations(child, "six_layers", ENDPOINT)
    reference["curves"][str(ENDPOINT)], reference["metrics"][str(ENDPOINT)] = selected_evaluations(ref_raw, "two_layers", ENDPOINT)
    new["training_curve"] = training_curve(combined, augmented=True)
    reference["training_curve"] = training_curve(reference_history, augmented=True)
    new["training_runs"] = {"parent": new["wandb"], "continuation": child["wandb"]}
    new["wandb"] = child["wandb"]
    new["inputs"].update(continuation_report=child_file, continuation_config=config_file,
                         continuation_history=hash_file(child_dir / "history.jsonl"))
    return {"schema": SCHEMA, "primary_update": ENDPOINT, "checkpoint_updates": list(STEPS), "arms": arms,
            "protocol": protocol, "protocol_input": protocol_file, "parent_protocol_input": parent_summary["protocol_input"],
            "scope": "Six-layer 10k-to20k user extension versus matched two-layer checkpoints and first 20k history",
            "matched_minibatch_order_hashes": ENDPOINT, "same_initial_backbone_tensors": False,
            "history_segments": [[1, 10000], [10001, ENDPOINT]],
            "secondary_checkpoint_excluded_from_primary": 15000,
            "budget_selection": "User extended around 8.7k after development review; 20k was selected during training",
            "confirmation_evaluated": False, "latent_rollout_evaluated": False,
            "metric_definitions": parent_summary["metric_definitions"]}


def plot_rows(summary, arm, step=ENDPOINT):
    if arm not in ARMS or step not in STEPS:
        raise ValueError("Primary plots use only the matched1k/5k/10k/20k checkpoints")
    rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
    if [(row["arm"], row["update"], row["role"], row["length"]) for row in rows] != [
            (arm, step, "ood_dev", length) for length in range(1, 37)]:
        raise ValueError("Full/boundary plots must use the same length36 rows")
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
        figures.append(name); plt.close(fig)
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.80, wspace=.35)
        for axis, key, title in zip(axes, ("E", "A", "M"), ("Every state through t correct", "Only state t correct", "Mean token accuracy through t")):
            for arm in ARMS:
                rows = plot_rows(summary, arm)
                x = [row["length"] for row in rows]
                axis.plot(x, [row[key] for row in rows], color=COLORS[arm], label=LABELS[arm])
                if key != "M":
                    axis.fill_between(x, [row[f"{key}_low95"] for row in rows], [row[f"{key}_high95"] for row in rows], color=COLORS[arm], alpha=.12)
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix t of same length-36 words", title=f"{key}(t): {title}")
            axis.axvline(12, color="gray", linestyle=":"); axis.grid(alpha=.2)
            axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.legend(fontsize=8)
        fig.suptitle("First-window RT + NextLat · matched 20,000 updates · 102,400 development words", y=.97)
        save(fig, name)
    fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for arm in ARMS:
        axis.plot(STEPS, [plot_rows(summary, arm, step)[-1]["E"] for step in STEPS], marker="o", color=COLORS[arm], label=LABELS[arm])
    axis.set(xlabel="Optimizer updates", ylabel="E(36): whole-word accuracy", ylim=(-.025, 1.025), xticks=STEPS)
    axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend()
    fig.suptitle("Matched length-36 whole-word accuracy · 15k excluded from primary comparison")
    save(fig, "whole-word-vs-updates")
    fig, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for arm in ARMS:
        rows = summary["arms"][arm]["training_curve"]
        axis.plot([row["update"] for row in rows], [row["state_ce"] for row in rows], color=COLORS[arm], label=LABELS[arm])
    axis.axvline(10000, color="gray", linestyle=":", label="Six-layer continuation starts")
    axis.set(xlabel="Optimizer updates", ylabel="Training state CE", xlim=(0, ENDPOINT)); axis.grid(alpha=.2); axis.legend()
    fig.suptitle("First 20k state CE · stitched six-layer history · 100-update means")
    save(fig, "training-state-ce")
    return figures


def markdown(summary, *, brief=False):
    new, old = (plot_rows(summary, arm)[-1] for arm in ARMS)
    lines = ["# First-window RT + NextLat: six-layer extension to 20k", "",
             f"At **20,000 updates**, E(36) is **{new['E']:.4%}** for six layers and **{old['E']:.4%}** for two layers "
             f"({100*(new['E']-old['E']):+.2f} percentage points).", "",
             "The six-layer run resumes its exact 10k checkpoint. The 20k budget was selected around 8.7k after reviewing development results. "
             "The original 10k report remains separate; this extension is an adaptive follow-up, using one seed.", "",
             "| Model | L12 whole word | E(13) | E(14) | E(36) | A(36) | M(36) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        rows = plot_rows(summary, arm)
        values = [summary["arms"][arm]["metrics"][str(ENDPOINT)]["dev"]["whole_word_exact_match"], rows[12]["E"], rows[13]["E"],
                  rows[-1]["E"], rows[-1]["A"], rows[-1]["M"]]
        lines.append(f"| {LABELS[arm]} | " + " | ".join(f"{value:.4%}" for value in values) + " |")
    lines += ["", "E(t) requires every state through t correct; A(t) checks only state t; M(t) averages correctness through t. "
              "Primary checkpoints use the same 102,400 length-36 development words. The saved six-layer 15k checkpoint is excluded from "
              "the primary comparison because the existing two-layer 15k evaluation used only 4,096 words.", "",
              "The six-layer first 10k history and continuation updates 10,001–20,000 are stitched with every corresponding minibatch "
              "hash checked against the two-layer first 20k history. Architecture, source code, joint objective, precision and predictor seed are unchanged "
              "across the six-layer continuation. Saved-state resume checks are separate from this report, which deserializes no checkpoint tensors.", "",
              "Depth changes Mitchell initialization draws/scaling and compute. Poor learning at 20k concerns learning/optimization at this depth "
              "and budget; it does not establish that tracking is impossible. Final confirmation and autonomous latent rollout remain unevaluated."]
    if not brief:
        lines += ["", "| Updates | Six-layer E(36) | Two-layer E(36) |", "| --- | ---: | ---: |"]
        for step in STEPS:
            lines.append(f"| {step:,} | " + " | ".join(f"{plot_rows(summary, arm, step)[-1]['E']:.4%}" for arm in ARMS) + " |")
        for name in ("whole-word-vs-updates", "length-full", "length-boundary", "training-state-ce"):
            lines += ["", f"![{name}]({name}.png)"]
        lines += ["", "Full and boundary views use identical 20k rows. Pointwise Wilson 95% E/A intervals describe variation over words, "
                  "not seed uncertainty. Model widths 512, batch 1024, all FP32 and optimizer hyperparameters match; layer counts and compute differ.", "",
                  "[Exact metric counts](metrics.csv) · [Plot data and provenance](summary.json) · [Run/artifact record](report.json)"]
    return "\n".join(lines) + "\n"


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh20k extension report directory")
    summary = make_summary(args.protocol)
    output.mkdir(parents=True)
    rows = [row for arm in ARMS for step in STEPS for role in ROLES for row in summary["arms"][arm]["curves"][str(step)][role]]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS); writer.writeheader(); writer.writerows(rows)
    for name in REPORTING_SOURCES:
        target = output / "source" / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
    summary["reporting_sources"] = {name: hash_file(output / "source" / name) for name in REPORTING_SOURCES}
    write_json(output / "summary.json", summary)
    figures = plots(summary, output)
    (output / "report.md").write_text(markdown(summary)); (output / "outcome.md").write_text(markdown(summary, brief=True))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name="first-window-rt-nextlat-six-vs-two-layers-20k-extension")
    result = {"schema": SCHEMA, "status": "running", "primary_update": ENDPOINT, "figures": figures}
    try:
        tracker.start({"schema": SCHEMA, "scope": summary["scope"], "budget_selection": summary["budget_selection"],
                       "protocol_sha256": summary["protocol_input"]["sha256"]})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(CSV_COLUMNS), data=[[row[key] for key in CSV_COLUMNS] for row in rows]),
                     **{f"report/{name}": wandb.Image(str(output / f"{name}.png")) for name in figures}})
        for index in range(ENDPOINT // 100):
            update = (index + 1) * 100
            values = {"update": update, **{f"train/{arm}/state_ce": summary["arms"][arm]["training_curve"][index]["state_ce"] for arm in ARMS}}
            if update in STEPS:
                values.update({f"dev/{arm}/E36": plot_rows(summary, arm, update)[-1]["E"] for arm in ARMS})
            tracker.log(values)
        tracker.summary({"primary_update": ENDPOINT, "confirmation_evaluated": False,
                         **{f"{arm}/E36": plot_rows(summary, arm)[-1]["E"] for arm in ARMS}})
        tracker.finish(succeeded=True); result["status"] = "complete"
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
    parser.add_argument("--protocol", default=str(LINEAGE / "protocol.json"))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    parser.add_argument("--wandb-group", default=LINEAGE.name)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
