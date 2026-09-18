#!/usr/bin/env python3
"""Compare completed RT+NextLat 10k pilots from saved evidence, without inference."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import (
    CSV_COLUMNS, DIAGNOSTICS, ROLES, STEPS, _digest_dict, _sha,
    local_path, metric_rows, read_training, training_curve,
)
from scripts.rt_a5_report import finite_number, hash_file, read_input, write_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-nextlat-variant-comparison-v1"
TRAIN_SCHEMA = "rt-a5-nextlat-variant-training-v1"
ARMS = ("rt_nextlat", "rt_nextlat_variant")
LABELS = {"rt_nextlat": "RT + NextLat (ALiBi baseline)",
          "rt_nextlat_variant": "RT + NextLat (identity-centered V/O + sinusoids)"}
COLORS = {"rt_nextlat": "#C68423", "rt_nextlat_variant": "#3465A4"}
NEW_SOURCES = {"scripts/rt_a5_nextlat_variant.py", "scripts/rt_a5_nextlat_variant_train.py",
               "configs/rt_a5_nextlat_variant/base.json"}
REPORTING_SOURCES = ("scripts/rt_a5_nextlat_variant_report.py", "scripts/rt_a5_nextlat_report.py",
                     "scripts/rt_a5_report.py", "scripts/rt_a5_length_report.py",
                     "scripts/experiment_tracking.py")
NEW_CONTRACT_FIELDS = {
    "training_step": "scripts.rt_a5_nextlat_train.train_step (same function object)",
    "evaluation": "scripts.rt_a5_train.evaluate_arrays (same function object)",
    "one_step_diagnostics": "scripts.rt_a5_nextlat_train.evaluate_diagnostics (same function object)",
}
RUN_SETTINGS = ("architecture", "batch_size", "checkpoint_steps", "data_order_seed",
                "diagnostic_rows", "eval_every", "eval_rows", "full_eval_rows",
                "latent_weight", "log_every", "predictor_hidden_width", "predictor_seed",
                "resume", "seed", "updates", "width")


def validate_variant_contract(contract):
    if contract.get("schema") != TRAIN_SCHEMA:
        raise ValueError("Unexpected variant training schema")
    for key, value in NEW_CONTRACT_FIELDS.items():
        if contract.get(key) != value:
            raise ValueError(f"Variant must reuse the original function: {key}")
    config = contract["variant_config"]
    if (config.get("schema") != "rt-a5-nextlat-variant-config-v1"
            or config.get("name") != "value-identity-sinusoidal"
            or config.get("architecture") != "rt" or config.get("width") != 512):
        raise ValueError("Unexpected variant identity or width")
    initialization = config["initialization"]
    for key, expected in {"kind": "value_path_identity", "noise_multiplier": 1.0,
                          "noise_seed": 1236,
                          "matrices": ["value_projection", "attention_output_projection"],
                          "layers": "both recurrent blocks"}.items():
        if initialization.get(key) != expected:
            raise ValueError(f"Unexpected identity-centered initialization: {key}")
    if not math.isclose(initialization.get("noise_std", -1), 1 / math.sqrt(512), rel_tol=0, abs_tol=1e-15):
        raise ValueError("Identity-centered noise must have per-entry variance 1/D")
    position = config["position_encoding"]
    for key, expected in {"kind": "fixed_sinusoidal", "base": 10000.0, "amplitude": 1.0,
                          "position_origin": 0, "token_embedding_scale": 1.0,
                          "alibi": False, "rope": False, "learned_position_parameters": False,
                          "persistent_position_buffers": False,
                          "nextlat_conditioning": "raw next-operation token embedding without positional addition"}.items():
        if position.get(key) != expected:
            raise ValueError(f"Unexpected sinusoidal position setting: {key}")
    if contract["model_config"].get("alibi") is not False or contract["model_config"].get("rope") is not False:
        raise ValueError("Both ALiBi and RoPE must be disabled")


def compare_contracts(baseline, variant):
    """Reject all unplanned changes, including loss, optimizer, data and runtime."""
    validate_variant_contract(variant)
    left, right = copy.deepcopy(baseline), copy.deepcopy(variant)
    for value in (left, right):
        value.pop("source_sha256")
        value.pop("schema")
    right.pop("variant_config")
    for key in NEW_CONTRACT_FIELDS:
        right.pop(key)
    right["model_config"]["alibi"] = True
    if left != right:
        differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise ValueError(f"Unplanned shared contract differences: {differing}")
    return left


def validate_history(history, report):
    if [row["update"] for row in history] != list(range(1, 10001)):
        raise ValueError("Variant history must contain each update 1..10000 exactly once")
    for row in history:
        if row["examples_seen"] != row["update"] * 1024 or not _sha(row["order_chain"]):
            raise ValueError("Invalid variant word count or data-order chain")
        for key in ("seconds", "loss", "state_loss", "latent_loss", "weighted_latent_loss",
                    "token_accuracy", "whole_word_exact", "grad_norm"):
            if not finite_number(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid variant history metric: {key}")
        if row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError("Invalid variant history accuracy")
        if (not math.isclose(row["weighted_latent_loss"], row["latent_loss"], rel_tol=1e-7, abs_tol=1e-9)
                or not math.isclose(row["loss"], row["state_loss"] + row["weighted_latent_loss"],
                                    rel_tol=2e-7, abs_tol=2e-7)):
            raise ValueError("Variant training loss is not state CE plus weight-one latent loss")
        for key in DIAGNOSTICS:
            if key in row and not finite_number(row[key]):
                raise ValueError(f"Nonfinite variant one-step diagnostic: {key}")
    if report["order_chain"] != history[-1]["order_chain"]:
        raise ValueError("Variant endpoint order differs from its history")
    seconds = sum(row["seconds"] for row in history)
    if (not finite_number(report["train_seconds"])
            or not math.isclose(seconds, report["train_seconds"], rel_tol=1e-9, abs_tol=1e-5)
            or not finite_number(report["elapsed_seconds"]) or report["elapsed_seconds"] < seconds):
        raise ValueError("Variant timing disagrees with training history")


def read_variant(directory):
    directory = Path(directory).resolve()
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    if report.get("schema") != TRAIN_SCHEMA or report.get("status") != "complete":
        raise ValueError("Variant must be a completed training report")
    if (report.get("start_update"), report.get("completed_updates"), report.get("endpoint")) != (0, 10000, 10000):
        raise ValueError("Variant must be a complete fresh 0->10000 pilot")
    if report.get("parent_checkpoint") is not None:
        raise ValueError("Variant must start from initialization, not resume a trained model")
    if report.get("confirmation_evaluated") is not False or report.get("latent_rollout_evaluated") is not False:
        raise ValueError("Confirmation and autonomous latent rollout must remain unevaluated")
    contract = report["contract"]
    validate_variant_contract(contract)
    sources = report["source_files"]
    if _digest_dict(sources) != contract["source_sha256"]:
        raise ValueError("Variant source manifest differs from its training contract")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(expected):
            raise ValueError("Invalid variant source path/hash")
        if hash_file(directory / "source" / path)["sha256"] != expected:
            raise ValueError(f"Variant source snapshot changed: {relative}")
    saved_config = json.loads((directory / "source/configs/rt_a5_nextlat_variant/base.json").read_text())
    saved_config["width"] = 512
    saved_config["initialization"]["noise_std"] = 1 / math.sqrt(512)
    if saved_config != contract["variant_config"]:
        raise ValueError("Saved variant configuration differs from the resolved contract")
    raw, config_file = read_input(directory / "config.json")
    args = json.loads(raw)
    manifest_file = hash_file(local_path(args["data_dir"]) / "manifest.json")
    if manifest_file["sha256"] != contract["data_manifest_sha256"]:
        raise ValueError("Variant data manifest differs from the execution contract")
    raw, history_file = read_input(directory / "history.jsonl")
    history = [json.loads(line) for line in raw.splitlines() if line.strip()]
    validate_history(history, report)
    checkpoints = {}
    for step in (0, *STEPS):
        selected = [item for item in report["checkpoints"] if item["completed_updates"] == step]
        if len(selected) != 1 or selected[0].get("examples_seen") != step * 1024:
            raise ValueError(f"Expected one variant checkpoint with matching exposure at {step}")
        actual = hash_file(local_path(selected[0]["path"]))
        if any(actual[key] != selected[0][key] for key in ("sha256", "bytes")):
            raise ValueError(f"Variant checkpoint changed at {step}")
        checkpoints[str(step)] = actual
    curves, metrics = {}, {}
    for step in STEPS:
        curves[str(step)], metrics[str(step)] = {}, {}
        for role in ROLES:
            matches = [row for row in report["evaluations"] if row["update"] == step and row["role"] == role]
            if len(matches) != 1 or matches[0]["rows"] != 102400:
                raise ValueError(f"Expected one full 102400-word evaluation at {step}/{role}")
            metric = matches[0]
            if metric.get("route") != "backbone_only":
                raise ValueError("Variant accuracy must use backbone predictions only")
            curves[str(step)][role] = metric_rows(metric, ARMS[1])
            metrics[str(step)][role] = metric
    for metric in report["evaluations"]:
        if metric.get("role") not in ROLES or metric.get("route") != "backbone_only":
            raise ValueError("Unexpected variant evaluation role/route")
        metric_rows(metric, ARMS[1])
    for diagnostic in report["one_step_diagnostics"]:
        if diagnostic.get("role") not in ROLES or diagnostic.get("route") != "teacher_conditioned_one_step_diagnostics":
            raise ValueError("Unexpected variant auxiliary diagnostic route")
        if any(key in diagnostic and not finite_number(diagnostic[key]) for key in DIAGNOSTICS):
            raise ValueError("Nonfinite auxiliary evaluation diagnostic")
    return {"report": report, "history": history, "curves": curves, "metrics": metrics,
            "checkpoints": checkpoints, "input_files": {"report": report_file, "history": history_file,
             "run_config": config_file, "data_manifest": manifest_file}}


def compare_runs(baseline_dir, variant_dir):
    arms = {ARMS[0]: read_training(baseline_dir, ARMS[0]), ARMS[1]: read_variant(variant_dir)}
    left, right = (arms[name]["report"] for name in ARMS)
    shared_contract = compare_contracts(left["contract"], right["contract"])
    configs = [json.loads(Path(arms[name]["input_files"]["run_config"]["path"]).read_text()) for name in ARMS]
    for key in RUN_SETTINGS:
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"Training argument differs: {key}")
    if set(right["source_files"]) != set(left["source_files"]) | NEW_SOURCES:
        raise ValueError("Unexpected variant source dependency additions or omissions")
    if any(right["source_files"].get(key) != value for key, value in left["source_files"].items()):
        raise ValueError("A frozen shared execution source changed")
    if [row["order_chain"] for row in arms[ARMS[0]]["history"]] != [row["order_chain"] for row in arms[ARMS[1]]["history"]]:
        raise ValueError("The two pilots consumed different per-update minibatch orders")
    initial = right["initialization"]
    if initial.get("baseline_initialization") != left["initialization"]:
        raise ValueError("Variant did not start from the baseline's paired initialization")
    if (initial.get("schema") != "rt-a5-nextlat-variant-initialization-v1"
            or initial.get("variant_config") != right["contract"]["variant_config"]
            or initial.get("predictor_sha256") != left["initialization"]["predictor_sha256"]
            or not _sha(initial.get("canonical_sha256"))
            or not _sha(initial.get("model_parameter_sha256"))
            or initial["canonical_sha256"] == left["initialization"]["canonical_sha256"]):
        raise ValueError("Unexpected variant initialization metadata")
    expected_slices = [{"parameter": f"backbone.transformer.blocks.{layer}.{name}", "rows": rows}
                       for layer in range(2)
                       for name, rows in (("kv_proj.weight", [512, 1024]), ("attn_out.weight", [0, 512]))]
    if initial.get("changed_parameter_slices") != expected_slices:
        raise ValueError("Unexpected identity-centered parameter slices")
    for report in (left, right):
        for key, value in {"parameter_count": 7407104, "backbone_parameter_count": 6357504,
                           "predictor_parameter_count": 1049600}.items():
            if report["initialization"].get(key) != value:
                raise ValueError(f"Unexpected model parameter count: {key}")
    eval_scope = lambda report: [(r["update"], r["role"], r["rows"], r["length"], r["route"])
                                 for r in report["evaluations"]]
    if eval_scope(left) != eval_scope(right):
        raise ValueError("Evaluation schedules, samples or routes differ")
    result = {
        "schema": SCHEMA, "primary_update": 10000, "diagnostic_updates": [1000, 5000],
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
        "evaluation_route": "backbone_only", "shared_contract": shared_contract,
        "variant_config": right["contract"]["variant_config"], "order_chain": right["order_chain"],
        "scope": "One paired development seed; fixed 10000 updates; combined initialization and positional changes",
        "intervals": "Pointwise Wilson 95% over words for E/A; no training-seed, paired-difference or M interval claim",
        "change_scope": "Both blocks: W_V/W_O=I+independent Normal(0,1/D); fixed unscaled additive sinusoids replace ALiBi; predictor unchanged",
        "causal_attribution": "Two simultaneous changes; neither individual effect is isolated",
        "plot_consistency": "Full and boundary figures plot the same 36 OOD rows at update10000; only x-axis limits differ",
        "arms": {},
    }
    for name, arm in arms.items():
        report = arm["report"]
        result["arms"][name] = {
            "label": LABELS[name], "contract": report["contract"], "initialization": report["initialization"],
            "input_files": arm["input_files"], "source_files": report["source_files"],
            "checkpoints": arm["checkpoints"], "curves": arm["curves"], "checkpoint_metrics": arm["metrics"],
            "endpoint_metrics": arm["metrics"]["10000"], "training_curve": training_curve(arm["history"], True),
            "one_step_diagnostics": report["one_step_diagnostics"], "training_wandb": report.get("wandb"),
            "train_seconds": report["train_seconds"], "elapsed_seconds": report["elapsed_seconds"],
        }
    return result


def metric_table(summary):
    rows = [row for arm in ARMS for step in map(str, STEPS) for role in ROLES
            for row in summary["arms"][arm]["curves"][step][role]]
    expected_keys = [(arm, step, role, length) for arm in ARMS for step in STEPS
                     for role in ROLES for length in range(1, {"dev": 12, "ood_dev": 36}[role] + 1)]
    if [(r["arm"], r["update"], r["role"], r["length"]) for r in rows] != expected_keys:
        raise ValueError("Metric table contains missing, duplicated or reordered curve rows")
    if any(r["words"] != 102400 for r in rows):
        raise ValueError("All checkpoint comparisons require the same 102400-word denominators")
    return rows


def endpoint_plot_rows(summary, arm):
    """Single source used by full, zoomed and exactness-only endpoint figures."""
    rows = summary["arms"][arm]["curves"]["10000"]["ood_dev"]
    if [row["length"] for row in rows] != list(range(1, 37)):
        raise ValueError("Endpoint plot needs exactly positions1..36")
    return rows


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}

    def save(figure, name):
        figures[name] = {}
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            figure.savefig(output / filename, dpi=180)
            figures[name][suffix] = filename
        plt.close(figure)

    def draw_length(axis, metric, limits):
        for arm in ARMS:
            rows = endpoint_plot_rows(summary, arm)
            x = [row["length"] for row in rows]
            axis.plot(x, [r[metric] for r in rows], color=COLORS[arm], label=LABELS[arm], linewidth=1.9)
            if metric in ("E", "A"):
                axis.fill_between(x, [r[metric+"_low95"] for r in rows],
                                  [r[metric+"_high95"] for r in rows], color=COLORS[arm], alpha=.13)
        axis.axvline(12, color="#777777", linewidth=1, linestyle=":")
        if metric == "A":
            axis.axhline(1/60, color="#777777", linewidth=.8, linestyle=":")
        axis.set(xlabel="Operation position / prefix length", ylim=(-.025, 1.025), xlim=limits)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.2)

    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), constrained_layout=True)
        for axis, metric, title in zip(axes, ("E", "A", "M"),
                ("Every state through t correct: E(t)", "State at t correct: A(t)", "Mean token accuracy through t: M(t)")):
            draw_length(axis, metric, limits)
            axis.set_title(title)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
        figure.suptitle("RT + NextLat · 10,000 updates · 102,400 OOD development words · one paired seed\n"
                       "Backbone-only inference; both views use the same rows; pointwise 95% intervals for E/A")
        save(figure, name)
    figure, axis = plt.subplots(figsize=(9.0, 4.9), constrained_layout=True)
    draw_length(axis, "E", (1, 36))
    axis.set(ylabel="All states through t correct", title="Cumulative exactness E(t) · fixed 10k endpoint")
    axis.legend(fontsize=8)
    save(figure, "length-exactness")
    figure, axes = plt.subplots(1, 2, figsize=(11.9, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("state_ce", "latent_loss"),
                                  ("Training state CE", "Training latent SmoothL1 loss")):
        for arm in ARMS:
            rows = summary["arms"][arm]["training_curve"]
            axis.plot([r["update"] for r in rows], [r[metric] for r in rows],
                      color=COLORS[arm], label=LABELS[arm], linewidth=1.8)
        axis.set(xlabel="Optimizer updates", ylabel=f"{title} (100-update means)")
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    figure.suptitle("Matched data order, loss and optimizer · state CE and latent loss shown separately")
    save(figure, "training-losses")
    return figures


def checkpoint_summary(summary):
    records = []
    for step in STEPS:
        for arm in ARMS:
            rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
            dev = summary["arms"][arm]["checkpoint_metrics"][str(step)]["dev"]
            record = {"arm": arm, "update": step, "selection": "primary" if step == 10000 else "diagnostic",
                      "dev_ce": dev["ce"], "dev_token_accuracy": dev["token_accuracy"],
                      "dev_whole_word_exact_match": dev["whole_word_exact_match"],
                      "E": {str(t): rows[t-1]["E"] for t in (12, 13, 14, 16, 18, 36)},
                      "A36": rows[-1]["A"], "M36": rows[-1]["M"]}
            records.append(record)
    return records


def markdown_report(summary):
    lines = ["# A5: RT + NextLat combined initialization/position pilot", "",
             "This is a fixed 10,000-update, single-seed development comparison. The new model changes "
             "both W_V/W_O initialization and positional encoding. Individual effects are not isolated.", "",
             "Both recurrent blocks use identity-centered V/O matrices with independent entry variance 1/512. "
             "Fixed unit-amplitude sinusoidal positions replace ALiBi; raw token embeddings remain unscaled. "
             "This does not guarantee a near-identity recurrent state update. The NextLat predictor, objective, "
             "optimizer, data, seed and minibatch order match the original pilot. All accuracy uses the backbone "
             "alone; confirmation and autonomous predictor rollout remain unevaluated.", "",
             "| Checkpoint | Model | L12 dev token accuracy | L12 dev whole word | OOD E(13) | OOD E(14) | OOD E(16) | OOD M(36) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["checkpoint_summary"]:
        values = [r["dev_token_accuracy"], r["dev_whole_word_exact_match"],
                  r["E"]["13"], r["E"]["14"], r["E"]["16"], r["M36"]]
        lines.append(f"| {r['update']:,} ({r['selection']}) | {LABELS[r['arm']]} | "
                     + " | ".join(f"{100*v:.4f}%" for v in values) + " |")
    lines += ["", "Each saved-checkpoint evaluation uses the same first 102,400 frozen words per development role. "
              "The 1k and 5k checkpoints are diagnostic; 10k remains the primary endpoint.", "",
              "E(t) requires every state through t to be correct; A(t) measures only state t; M(t) averages "
              "token accuracy through t. Length curves are prefixes of the same length-36 outputs. Full and "
              "boundary plots use exactly the same rows and differ only in horizontal display limits.", "",
              "![Full length curves](length-full.png)", "", "![Boundary view](length-boundary.png)", "",
              "![Exactness only](length-exactness.png)", "", "![Training state CE and latent loss](training-losses.png)", "",
              "Bands show pointwise Wilson 95% intervals over words for E/A. They do not describe training-seed "
              "uncertainty or paired-difference confidence; M has no interval based on independent token positions. "
              "Zero observed exactness is not proof of zero population success.", "",
              "The report verifies complete histories, each minibatch order hash, retained checkpoint hashes, "
              "source snapshots, matching shared contracts, initialization provenance, and evaluation tallies. "
              "No model inference or additional training is performed by this reporter.", "",
              "[Metrics CSV](metrics.csv) · [Plot data](plot-data.json) · [Machine-readable summary](summary.json) · "
              "[Reporting provenance](report.json)", ""]
    return "\n".join(lines)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Reporting requires a fresh output directory")
    summary = compare_runs(args.baseline_dir, args.variant_dir)
    rows = metric_table(summary)
    summary["checkpoint_summary"] = checkpoint_summary(summary)
    output.mkdir(parents=True)
    for arm in ARMS:
        for name, record in summary["arms"][arm]["input_files"].items():
            destination = output / "inputs" / arm / Path(record["path"]).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record["path"], destination)
            actual = hash_file(destination)
            if any(actual[key] != record[key] for key in ("sha256", "bytes")):
                raise ValueError(f"Input changed while copying: {arm}/{name}")
    summary["reporting_sources"] = {}
    for relative in REPORTING_SOURCES:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        summary["reporting_sources"][relative] = hash_file(destination)
    summary["reporting_source_sha256"] = _digest_dict(
        {name: record["sha256"] for name, record in summary["reporting_sources"].items()})
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "metrics.csv").open() as stream:
        recovered = list(csv.DictReader(stream))
    if recovered != [{key: str(row[key]) for key in CSV_COLUMNS} for row in rows]:
        raise ValueError("Written CSV does not reproduce the graph data exactly")
    write_json(output / "summary.json", summary)
    write_json(output / "plot-data.json", summary)
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    result = {**summary, "status": "running", "figures": figures}
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name="nextlat-identity-sinusoidal-comparison-step10000")
    try:
        tracker.start({key: summary[key] for key in ("schema", "primary_update", "shared_contract",
                                                    "variant_config", "scope", "evaluation_route")})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(CSV_COLUMNS), data=[[r[k] for k in CSV_COLUMNS] for r in rows]),
                     **{f"report/{name}": wandb.Image(str(output / paths["png"])) for name, paths in figures.items()}})
        tracker.summary({"confirmation_evaluated": False, "latent_rollout_evaluated": False,
                         "primary_update": 10000, "scope": summary["scope"],
                         "endpoint_metrics": {arm: summary["arms"][arm]["endpoint_metrics"] for arm in ARMS}})
        tracker.finish(succeeded=True)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        try:
            tracker.finish(succeeded=False)
        except Exception:
            pass
        raise
    finally:
        result["wandb"] = tracker.record
        result["artifacts"] = {str(path.relative_to(output)): hash_file(path)
                               for path in sorted(output.rglob("*")) if path.is_file()
                               and "wandb" not in path.relative_to(output).parts and path != output / "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output),
                      "wandb": tracker.record["run_url"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", default=str(ROOT / ".runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat"))
    parser.add_argument("--variant-dir", default=str(ROOT / ".runtime/rt-a5/20260914T173148Z-nextlat-identity-sinusoidal/train-rt-nextlat-variant"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group", default="20260914T173148Z-nextlat-identity-sinusoidal")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
