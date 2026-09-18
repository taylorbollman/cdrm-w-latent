#!/usr/bin/env python3
"""Compare saved six- and two-layer first-window RT+NextLat evidence through10k.

No model construction, checkpoint deserialization, inference, or training.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import CSV_COLUMNS, ROLES, _digest_dict, _sha, local_path, metric_rows, training_curve
from scripts.rt_a5_report import finite_number, hash_file, read_input, write_json

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / ".runtime/rt-a5/20260915T141414Z-l1r-six-layer-nextlat10k"
REFERENCE = ROOT / ".runtime/rt-a5/20260914T212935Z-nextlat-depth-order80k/train-window-first"
OUTPUT = ROOT / "docs/reports/rt-a5/l1r-six-layer-nextlat10k"
SCHEMA = "rt-a5-l1r-depth-comparison-v1"
ENDPOINT, STEPS = 10000, (1000, 5000, 10000)
ARMS = ("six_layers", "two_layers")
LABELS = {"six_layers": "6 layers: first window-2 + 5 full RT", "two_layers": "2 layers: first window-2 + 1 full RT"}
COLORS = {"six_layers": "#B85031", "two_layers": "#24938C"}
REPORTING_SOURCES = ("scripts/rt_a5_l1r_depth_report.py", "scripts/rt_a5_nextlat_report.py",
                     "scripts/rt_a5_length_report.py", "scripts/rt_a5_report.py", "scripts/experiment_tracking.py")


def bound_file(record):
    actual = hash_file(local_path(record["path"]))
    if any(actual[key] != record[key] for key in ("sha256", "bytes") if key in record):
        raise ValueError(f"Bound evidence changed: {record['path']}")
    return actual


def validate_history(history):
    if len(history) != ENDPOINT or [row.get("update") for row in history] != list(range(1, ENDPOINT + 1)):
        raise ValueError("Need every ordered update from1 through10k")
    for row in history:
        if row.get("examples_seen") != row["update"] * 1024 or not _sha(row.get("order_chain")):
            raise ValueError("Training word count or order chain is invalid")
        for key in ("seconds", "loss", "state_loss", "latent_loss", "weighted_latent_loss", "grad_norm",
                    "token_accuracy", "whole_word_exact"):
            if not finite_number(row.get(key)) or row[key] < 0:
                raise ValueError(f"Invalid training metric:{key}")
        if row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError("Invalid training accuracy")
        if (not math.isclose(row["weighted_latent_loss"], row["latent_loss"], rel_tol=1e-7, abs_tol=1e-9)
                or not math.isclose(row["loss"], row["state_loss"] + row["weighted_latent_loss"], rel_tol=2e-7, abs_tol=2e-7)):
            raise ValueError("Training loss is not unchanged CE plus weight-one NextLat")


def validate_shared_contracts(new, reference):
    shared = ("architecture", "width", "seed", "predictor_seed", "data_order_seed", "batch_size", "length",
              "train_rows", "data_manifest_sha256", "precision", "tf32", "compile", "cuda_graphs", "torch",
              "cuda", "device_capability", "optimizer", "word_order", "evaluation_route",
              "latent_rollout_evaluated", "objective", "nextlat_config")
    if any(new.get(key) != reference.get(key) for key in shared):
        raise ValueError("Shared objective, data, optimizer, predictor, or runtime contract differs")
    expected = {"architecture": "rt", "width": 512, "batch_size": 1024, "length": 12,
                "train_rows": 800000, "seed": 1234, "predictor_seed": 1235, "data_order_seed": 1234,
                "precision": "fp32", "tf32": False, "compile": False, "cuda_graphs": False,
                "evaluation_route": "backbone_only", "latent_rollout_evaluated": False}
    if any(new.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected shared first-window diagnostic recipe")
    if new["objective"].get("latent_weight") != 1 or new["objective"].get("target_detached") is not True:
        raise ValueError("Require the original NextLat objective")
    left, right = (copy.deepcopy(contract["model_config"]) for contract in (new, reference))
    if left.pop("n_layers") != 6 or right.pop("n_layers") != 2 or left != right:
        raise ValueError("Backbone configuration may differ only in layer count")
    for key, value in {"init_fn": "mitchell", "block_type": "recurrent", "alibi": True, "rope": False,
                       "reference_eager": True, "recurrent_write_rho": 1.0}.items():
        if left.get(key) != value:
            raise ValueError(f"Unexpected backbone recipe:{key}")
    reference_layers = reference["experiment_config"]["attention"]["layers"]
    for contract, layers in ((new, 6), (reference, 2)):
        experiment = contract["experiment_config"]
        if (experiment.get("n_layers") != layers or experiment.get("window_layer") != 0
                or experiment.get("window_length") != 2 or experiment.get("position_encoding") != "alibi"
                or experiment["attention"].get("gradient_truncation") is not False
                or experiment["attention"]["layers"] != [reference_layers[0]] + [reference_layers[1]] * (layers - 1)):
            raise ValueError("Require first-layer window2 and full attached recurrence in every following layer")


def compare_minibatch_orders(new, reference):
    if len(new) != ENDPOINT or len(reference) != ENDPOINT or [row["order_chain"] for row in new] != [row["order_chain"] for row in reference]:
        raise ValueError("Every one of the first10000 minibatch orders must match")


def read_arm(directory, arm):
    directory = local_path(directory).resolve()
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    schema = "rt-a5-l1r-depth-training-v1" if arm == "six_layers" else "rt-a5-depth-order-training-v1"
    complete_at = ENDPOINT if arm == "six_layers" else 80000
    if (report.get("schema") != schema or report.get("status") != "complete"
            or report.get("start_update") != 0 or report.get("parent_checkpoint") is not None
            or report.get("endpoint") != complete_at or report.get("completed_updates") != complete_at
            or report.get("wandb", {}).get("status") != "synced"
            or report.get("confirmation_evaluated") is not False or report.get("latent_rollout_evaluated") is not False):
        raise ValueError(f"{arm}: require the complete fresh, synced, approved training record")
    sources = report["source_files"]
    if _digest_dict(sources) != report["contract"]["source_sha256"]:
        raise ValueError("Training-source manifest digest differs")
    for name, wanted in sources.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not _sha(wanted):
            raise ValueError("Invalid training source path/hash")
        for path in (directory / "source" / relative, ROOT / relative):
            if hash_file(path)["sha256"] != wanted:
                raise ValueError(f"Frozen training source changed:{name}")
    config_raw, config_file = read_input(directory / "config.json")
    config = json.loads(config_raw)
    if config.get("resume") is not None:
        raise ValueError("Require a fresh run configuration")
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != report["contract"]["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from the recorded contract")
    history_path = directory / "history.jsonl"
    history = []
    with history_path.open() as stream:
        for line in stream:
            if line.strip():
                history.append(json.loads(line))
                if len(history) == ENDPOINT:
                    break
        if arm == "six_layers" and any(line.strip() for line in stream):
            raise ValueError("Six-layer history extends past the approved10k endpoint")
    validate_history(history)
    if arm == "six_layers" and history[-1]["order_chain"] != report["order_chain"]:
        raise ValueError("New report and history endpoint orders differ")
    checkpoints = {}
    if arm == "six_layers" and sorted(item["completed_updates"] for item in report["checkpoints"]) != [0, *STEPS]:
        raise ValueError("New pilot must retain exactly0/1k/5k/10k checkpoints")
    for step in (0, *STEPS):
        selected = [item for item in report["checkpoints"] if item["completed_updates"] == step]
        if len(selected) != 1 or local_path(selected[0]["path"]).resolve() != directory / "checkpoints" / f"step-{step:06d}.pt":
            raise ValueError(f"Need one correctly located checkpoint at{step}")
        checkpoints[str(step)] = bound_file(selected[0])
    curves, metrics = {}, {}
    for step in STEPS:
        curves[str(step)], metrics[str(step)] = {}, {}
        for role in ROLES:
            selected = [item for item in report["evaluations"] if item["update"] == step and item["role"] == role]
            if len(selected) != 1 or selected[0].get("rows") != 102400 or selected[0].get("route") != "backbone_only":
                raise ValueError(f"Need one102400-word backbone evaluation at{step}/{role}")
            metrics[str(step)][role] = selected[0]
            curves[str(step)][role] = metric_rows(selected[0], arm)
    return {"directory": str(directory), "contract": report["contract"], "initialization": report["initialization"],
            "source_files": sources, "history": history, "curves": curves, "metrics": metrics,
            "checkpoints": checkpoints, "wandb": report["wandb"], "config": config,
            "inputs": {"report": report_file, "config": config_file, "history": hash_file(history_path), "data_manifest": manifest}}


def make_summary(protocol_path):
    raw, protocol_file = read_input(local_path(protocol_path))
    protocol = json.loads(raw)
    expected = {"schema": "rt-a5-l1r-depth-protocol-v1", "start_update": 0, "endpoint": ENDPOINT,
                "n_layers": 6, "window_layer": 0, "window_length": 2, "width": 512,
                "checkpoint_steps": [0, *STEPS], "full_evaluation_rows": 102400}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Protocol differs from the bounded six-layer10k diagnostic")
    reference = protocol["reference"]
    if local_path(reference["training_directory"]).resolve() != REFERENCE:
        raise ValueError("Use the declared historical two-layer first-window RT+NextLat reference")
    arms = {"six_layers": read_arm(protocol["training_directory"], "six_layers"),
            "two_layers": read_arm(reference["training_directory"], "two_layers")}
    new, old = (arms[arm] for arm in ARMS)
    if (new["contract"] != protocol["strict_contract"] or new["source_files"] != protocol["source_files"]
            or _digest_dict(protocol["source_files"]) != protocol["source_sha256"]
            or new["initialization"] != protocol["initialization"]):
        raise ValueError("New training record differs from its frozen prospective protocol")
    if old["source_files"] != reference["source_files"] or _digest_dict(old["source_files"]) != reference["source_sha256"]:
        raise ValueError("Reference source provenance differs")
    for name, digest in old["source_files"].items():
        if new["source_files"].get(name) != digest:
            raise ValueError("Shared historical training source changed")
    for kind in ("report", "config"):
        if bound_file(reference[kind])["sha256"] != old["inputs"][kind]["sha256"]:
            raise ValueError("Bound reference report/config differs")
    for step in (0, *STEPS):
        if bound_file(reference["checkpoints"][str(step)])["sha256"] != old["checkpoints"][str(step)]["sha256"]:
            raise ValueError("Bound reference checkpoint differs")
    bound_file(protocol["preflight"])
    if bound_file(protocol["data_manifest"])["sha256"] != new["inputs"]["data_manifest"]["sha256"]:
        raise ValueError("Prospective data manifest differs from the actual training data")
    validate_shared_contracts(new["contract"], old["contract"])
    compare_minibatch_orders(new["history"], old["history"])
    if (new["initialization"].get("predictor_seed") != 1235
            or new["initialization"].get("predictor_sha256") != old["initialization"]["predictor_sha256"]):
        raise ValueError("Independent predictor initialization differs")
    summary = {"schema": SCHEMA, "primary_update": ENDPOINT, "checkpoint_updates": list(STEPS),
               "scope": "Matched 1k/5k/10k saved checkpoints; first 10k training only; development depth diagnostic with one seed",
               "protocol": protocol, "protocol_input": protocol_file, "arms": arms,
               "matched_minibatch_order_hashes": ENDPOINT, "same_initial_backbone_tensors": False,
               "initialization_note": "Depth changes Mitchell draws and scaling; independent predictor seed 1235 is preserved",
               "confirmation_evaluated": False, "latent_rollout_evaluated": False,
               "metric_definitions": {"E": "Every state through t correct", "A": "Only state t correct", "M": "Mean token accuracy through t"}}
    for arm in ARMS:
        packet = arms[arm]
        packet["training_curve"] = training_curve(packet.pop("history"), augmented=True)
    return summary


def plot_rows(summary, arm, step=ENDPOINT):
    if arm not in ARMS or step not in STEPS:
        raise ValueError("Only declared arms and matched1k/5k/10k checkpoints may be plotted")
    rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
    if [(row["arm"], row["update"], row["role"], row["length"]) for row in rows] != [
            (arm, step, "ood_dev", length) for length in range(1, 37)]:
        raise ValueError("Plots must use the shared length36 metric rows")
    return rows


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    names = []
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        names.append(name)
        plt.close(fig)
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for axis, key, title in zip(axes, ("E", "A", "M"), ("Every state through t correct", "Only state t correct", "Mean token accuracy through t")):
            for arm in ARMS:
                rows = plot_rows(summary, arm)
                axis.plot([row["length"] for row in rows], [row[key] for row in rows], color=COLORS[arm], label=LABELS[arm])
                if key != "M":
                    axis.fill_between([row["length"] for row in rows], [row[f"{key}_low95"] for row in rows],
                                      [row[f"{key}_high95"] for row in rows], color=COLORS[arm], alpha=.12)
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix t of same length-36 words", title=f"{key}(t): {title}")
            axis.axvline(12, color="gray", linestyle=":"); axis.grid(alpha=.2)
            axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.legend(fontsize=8)
        fig.suptitle("First-window RT + NextLat · matched 10,000 updates · 102,400 development words")
        save(fig, name)
    fig, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for arm in ARMS:
        axis.plot(STEPS, [plot_rows(summary, arm, step)[-1]["E"] for step in STEPS], marker="o", color=COLORS[arm], label=LABELS[arm])
    axis.set(xlabel="Optimizer updates", ylabel="E(36): whole-word accuracy", ylim=(-.025, 1.025), xticks=STEPS)
    axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend()
    fig.suptitle("Length-36 whole-word accuracy at matched retained checkpoints")
    save(fig, "whole-word-vs-updates")
    fig, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for arm in ARMS:
        rows = summary["arms"][arm]["training_curve"]
        axis.plot([row["update"] for row in rows], [row["state_ce"] for row in rows], color=COLORS[arm], label=LABELS[arm])
    axis.set(xlabel="Optimizer updates", ylabel="Training state CE", xlim=(0, ENDPOINT))
    axis.grid(alpha=.2); axis.legend(); fig.suptitle("Shared state CE · first 10k updates · nonoverlapping 100-update means")
    save(fig, "training-state-ce")
    return names


def markdown(summary, *, brief=False):
    new, old = (plot_rows(summary, arm)[-1] for arm in ARMS)
    lines = ["# First-window RT + NextLat: six versus two layers at 10k", "",
             f"At **10,000 updates**, length-36 whole-word accuracy E(36) is **{new['E']:.4%}** for six layers "
             f"and **{old['E']:.4%}** for two layers ({100*(new['E']-old['E']):+.2f} percentage points).", "",
             "| Model | L12 whole word | E(13) | E(14) | E(36) | A(36) | M(36) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        rows = plot_rows(summary, arm)
        values = [summary["arms"][arm]["metrics"][str(ENDPOINT)]["dev"]["whole_word_exact_match"], rows[12]["E"], rows[13]["E"],
                  rows[-1]["E"], rows[-1]["A"], rows[-1]["M"]]
        lines.append(f"| {LABELS[arm]} | " + " | ".join(f"{value:.4%}" for value in values) + " |")
    lines += ["", "E(t) requires every state through t correct; A(t) checks only state t; M(t) averages correctness through t. "
              "E/A/M curves use the same 102,400 length-36 development words. L12 whole-word accuracy uses separate short development words.", "",
              "This is a development diagnostic of early learning at a matched update budget, using one seed. "
              "If deeper training loses early learning, that concerns optimization at this depth/budget; it does not prove state tracking impossible. "
              "Neither equal compute nor identical backbone initialization is claimed: added layers change Mitchell draws and scaling. "
              "The independently seeded NextLat predictor, joint objective, minibatch order, width 512/batch 1024, and FP32 runtime settings are preserved. "
              "Final confirmation and autonomous latent rollout remain unevaluated."]
    if not brief:
        lines += ["", "| Model | Backbone parameters | Total parameters |", "| --- | ---: | ---: |"]
        for arm in ARMS:
            initial = summary["arms"][arm]["initialization"]
            lines.append(f"| {LABELS[arm]} | {initial['backbone_parameter_count']:,} | {initial['parameter_count']:,} |")
    if not brief:
        lines += ["", "| Updates | Six-layer E(36) | Two-layer E(36) |", "| --- | ---: | ---: |"]
        for step in STEPS:
            lines.append(f"| {step:,} | " + " | ".join(f"{plot_rows(summary, arm, step)[-1]['E']:.4%}" for arm in ARMS) + " |")
        for name in ("whole-word-vs-updates", "length-full", "length-boundary", "training-state-ce"):
            lines += ["", f"![{name}]({name}.png)"]
        lines += ["", "The full and boundary plots are views of identical 10k rows. Only 1k/5k/10k checkpoints and the first 10k "
                  "history from the longer reference run are compared. All 10k corresponding minibatch hashes match. "
                  "The reporter verifies the frozen protocol, contracts, source snapshots, selected checkpoint hashes and integer metric counts; "
                  "it loads no model or checkpoint tensors. Pointwise Wilson 95% E/A intervals describe sampling over words, not seed variability.", "",
                  "[Exact counts](metrics.csv) · [Plot data and provenance](summary.json) · [Run and artifact record](report.json)"]
    return "\n".join(lines) + "\n"


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh report output directory")
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
                            group=args.wandb_group, name="first-window-rt-nextlat-six-vs-two-layers-10k")
    result = {"schema": SCHEMA, "status": "running", "primary_update": ENDPOINT, "figures": figures}
    try:
        tracker.start({"schema": SCHEMA, "scope": summary["scope"], "protocol_sha256": summary["protocol_input"]["sha256"]})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(CSV_COLUMNS), data=[[row[key] for key in CSV_COLUMNS] for row in rows]),
                     **{f"report/{name}": wandb.Image(str(output / f"{name}.png")) for name in figures}})
        for index in range(100):
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
