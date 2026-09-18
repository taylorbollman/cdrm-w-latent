#!/usr/bin/env python3
"""Compare Mitchell position and layer-2 window pilots from saved evidence only."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_nextlat_report import (
    CSV_COLUMNS, DIAGNOSTICS, ROLES, STEPS, _digest_dict, _sha,
    local_path, metric_rows, read_training, training_curve,
)
from scripts.rt_a5_report import finite_number, hash_file, read_input, write_json
from scripts.rt_a5_nextlat_variant_report import validate_history


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-window-comparison-v1"
TRAIN_SCHEMA = "rt-a5-window-training-v1"
ARMS = ("rt_nextlat", "rt_nextlat_sinusoidal", "rt_nextlat_window2")
LABELS = {"rt_nextlat": "Mitchell + ALiBi, full RT",
          "rt_nextlat_sinusoidal": "Mitchell + sinusoids, full RT",
          "rt_nextlat_window2": "Mitchell + selected positions, layer-2 window 2"}
COLORS = {ARMS[0]: "#C68423", ARMS[1]: "#3465A4", ARMS[2]: "#24938C"}
PRIOR_VARIANT_SOURCES = {"scripts/rt_a5_nextlat_variant.py", "scripts/rt_a5_nextlat_variant_train.py",
                         "configs/rt_a5_nextlat_variant/base.json"}
NEW_SOURCES = {"scripts/rt_a5_window.py", "scripts/rt_a5_window_train.py",
               "configs/rt_a5_window/base.json"}
FROZEN_SOURCES = {
    46: "1e6d0c63289f01525bc0c19bba6b2646d61df10ddb815bc74f5c111b46f9f961",
    49: "6a53fe8acd9291417ac9e8708095fa7dab945694fb7dc4f2d1dcd6bd11af8a69",
    52: "9d12d61e994bdad57567cb1d30fd36f1ea3296c94f7aa401d574ecbb34d339c2",
}
REPORTING_SOURCES = ("scripts/rt_a5_window_report.py", "scripts/rt_a5_nextlat_variant_report.py",
                     "scripts/rt_a5_nextlat_report.py", "scripts/rt_a5_report.py",
                     "scripts/rt_a5_length_report.py", "scripts/experiment_tracking.py")
NEW_CONTRACT_FIELDS = {
    "training_step": "scripts.rt_a5_nextlat_train.train_step (same function object)",
    "evaluation": "scripts.rt_a5_train.evaluate_arrays (same function object)",
    "one_step_diagnostics": "scripts.rt_a5_nextlat_train.evaluate_diagnostics (same function object)",
}
RUN_SETTINGS = ("architecture", "batch_size", "checkpoint_steps", "data_order_seed",
                "diagnostic_rows", "eval_every", "eval_rows", "full_eval_rows",
                "latent_weight", "log_every", "predictor_hidden_width", "predictor_seed",
                "resume", "seed", "updates", "width")


def validate_window_contract(contract):
    if contract.get("schema") != TRAIN_SCHEMA:
        raise ValueError("Unexpected window training schema")
    for key, value in NEW_CONTRACT_FIELDS.items():
        if contract.get(key) != value:
            raise ValueError(f"Window pilots must reuse the original function: {key}")
    config = contract["experiment_config"]
    if (config.get("schema") != "rt-a5-window-config-v1"
            or config.get("architecture") != "rt" or config.get("width") != 512
            or config.get("position_encoding") not in ("alibi", "sinusoidal")
            or config.get("second_layer_window") not in (None, 2)
            or isinstance(config.get("second_layer_window"), bool)):
        raise ValueError("Unexpected window experiment configuration")
    initialization = config["initialization"]
    if (initialization.get("kind") != "original_mitchell"
            or initialization.get("identity_centered_overrides") is not False
            or initialization.get("learned_tensor_changes") != 0):
        raise ValueError("Both pilots must retain original Mitchell initialization")
    position, details = config["position_encoding"], config["position_details"]
    if (details.get("alibi") is not (position == "alibi") or details.get("rope") is not False
            or details.get("learned_position_parameters") is not False
            or contract["model_config"].get("alibi") is not (position == "alibi")
            or contract["model_config"].get("rope") is not False):
        raise ValueError("Unexpected positional encoding or learned positional parameters")
    if position == "sinusoidal":
        for key, expected in {"kind": "fixed_sinusoidal", "base": 10000.0, "amplitude": 1.0,
                              "position_origin": 0, "token_embedding_scale": 1.0,
                              "persistent_position_buffers": False,
                              "nextlat_conditioning": "raw next-operation token embedding without positional addition"}.items():
            if details.get(key) != expected:
                raise ValueError(f"Unexpected sinusoidal setting: {key}")
    window = config["second_layer_window"]
    expected_attention = {
        "first_layer": "full causal recurrent attention",
        "second_layer": ("full causal recurrent attention" if window is None else
                         "self provisional K/V and immediately previous permanent output K/V"),
        "second_layer_allowed_keys": ("0 <= key <= query" if window is None else
                                      "max(0, query - 1) <= key <= query"),
        "first_token": "self provisional K/V only", "recurrent_write_rho": 1.0,
        "gradient_truncation": False,
        "implementation": ("original recurrent block" if window is None else
                           "parameter-free block subclass supplies additive mask to original forward/backward"),
    }
    if config.get("attention") != expected_attention:
        raise ValueError("Unexpected window read/write/gradient semantics")


def compare_contracts(baseline, candidate):
    """Allow only positions and direct attention reads in the second RT layer."""
    validate_window_contract(candidate)
    left, right = copy.deepcopy(baseline), copy.deepcopy(candidate)
    for value in (left, right):
        value.pop("source_sha256")
        value.pop("schema")
    right.pop("experiment_config")
    for key in NEW_CONTRACT_FIELDS:
        right.pop(key)
    right["model_config"]["alibi"] = True
    if left != right:
        differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise ValueError(f"Unplanned shared contract differences: {differing}")
    return left


def validate_source_lineage(baseline, candidate):
    if len(baseline) != 46 or _digest_dict(baseline) != FROZEN_SOURCES[46]:
        raise ValueError("Original 46-source NextLat lineage differs")
    if set(candidate) != set(baseline) | PRIOR_VARIANT_SOURCES | NEW_SOURCES:
        raise ValueError("Unexpected window source dependency additions or omissions")
    if any(candidate.get(key) != value for key, value in baseline.items()):
        raise ValueError("A frozen shared execution source changed")
    previous = {key: value for key, value in candidate.items() if key not in NEW_SOURCES}
    if (_digest_dict(previous) != FROZEN_SOURCES[49]
            or _digest_dict(candidate) != FROZEN_SOURCES[52]):
        raise ValueError("Frozen 49/52-source lineage differs")


def read_window(directory, arm):
    directory = Path(directory).resolve()
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    if report.get("schema") != TRAIN_SCHEMA or report.get("status") != "complete":
        raise ValueError("Window pilot must be a completed training report")
    if (report.get("start_update"), report.get("completed_updates"), report.get("endpoint")) != (0, 10000, 10000):
        raise ValueError("Window pilot must be a complete fresh 0->10000 pilot")
    if report.get("parent_checkpoint") is not None:
        raise ValueError("Window pilot must start from initialization, not resume a trained model")
    if report.get("confirmation_evaluated") is not False or report.get("latent_rollout_evaluated") is not False:
        raise ValueError("Confirmation and autonomous latent rollout must remain unevaluated")
    contract = report["contract"]
    validate_window_contract(contract)
    sources = report["source_files"]
    if _digest_dict(sources) != contract["source_sha256"]:
        raise ValueError("Window pilot source manifest differs from its training contract")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(expected):
            raise ValueError("Invalid window source path/hash")
        if hash_file(directory / "source" / path)["sha256"] != expected:
            raise ValueError(f"Window pilot source snapshot changed: {relative}")
    saved_config = json.loads((directory / "source/configs/rt_a5_window/base.json").read_text())
    experiment = contract["experiment_config"]
    saved_config.update(width=512, position_encoding=experiment["position_encoding"],
                        second_layer_window=experiment["second_layer_window"])
    saved_config["position_details"] = saved_config.pop("position_options")[experiment["position_encoding"]]
    # Attention semantics are checked independently above; the rest must resolve from the saved JSON.
    saved_config["attention"] = experiment["attention"]
    if saved_config != experiment:
        raise ValueError("Saved experiment configuration differs from its resolved contract")
    raw, config_file = read_input(directory / "config.json")
    args = json.loads(raw)
    if (args.get("position_encoding") != experiment["position_encoding"]
            or args.get("second_layer_window") != experiment["second_layer_window"]):
        raise ValueError("Saved CLI position/window differs from the resolved contract")
    manifest_file = hash_file(local_path(args["data_dir"]) / "manifest.json")
    if manifest_file["sha256"] != contract["data_manifest_sha256"]:
        raise ValueError("Window pilot data manifest differs from the execution contract")
    raw, history_file = read_input(directory / "history.jsonl")
    history = [json.loads(line) for line in raw.splitlines() if line.strip()]
    validate_history(history, report)
    checkpoints = {}
    for step in (0, *STEPS):
        selected = [item for item in report["checkpoints"] if item["completed_updates"] == step]
        if len(selected) != 1 or selected[0].get("examples_seen") != step * 1024:
            raise ValueError(f"Expected one window checkpoint with matching exposure at {step}")
        actual = hash_file(local_path(selected[0]["path"]))
        if any(actual[key] != selected[0][key] for key in ("sha256", "bytes")):
            raise ValueError(f"Window pilot checkpoint changed at {step}")
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
                raise ValueError("Window pilot accuracy must use backbone predictions only")
            curves[str(step)][role] = metric_rows(metric, arm)
            metrics[str(step)][role] = metric
    for metric in report["evaluations"]:
        if metric.get("role") not in ROLES or metric.get("route") != "backbone_only":
            raise ValueError("Unexpected window evaluation role/route")
        metric_rows(metric, arm)
    for diagnostic in report["one_step_diagnostics"]:
        if diagnostic.get("role") not in ROLES or diagnostic.get("route") != "teacher_conditioned_one_step_diagnostics":
            raise ValueError("Unexpected variant auxiliary diagnostic route")
        if any(key in diagnostic and not finite_number(diagnostic[key]) for key in DIAGNOSTICS):
            raise ValueError("Nonfinite auxiliary evaluation diagnostic")
    return {"report": report, "history": history, "curves": curves, "metrics": metrics,
            "checkpoints": checkpoints, "input_files": {"report": report_file, "history": history_file,
             "run_config": config_file, "data_manifest": manifest_file}}


def verify_selection(selection, arms, protocol):
    """Independently recover the prospective winner and bind it to exact input evidence."""
    expected_criterion = protocol["position_selection"]
    if (selection.get("schema") != "rt-a5-window-position-selection-v1"
            or selection.get("status") != "selected"
            or selection.get("criterion") != expected_criterion
            or expected_criterion.get("checkpoint") != 10000
            or expected_criterion.get("role") != "ood_dev"
            or expected_criterion.get("rows") != 102400
            or expected_criterion.get("exact_tie") != "alibi"
            or expected_criterion.get("primary") != "mean cumulative-prefix exactness E(t) over t=13..36 inclusive"
            or selection.get("confirmation_evaluated") is not False):
        raise ValueError("Selection protocol or scope differs")
    scores = {}
    for position, name in (("alibi", ARMS[0]), ("sinusoidal", ARMS[1])):
        arm, recorded = arms[name], selection["arms"][position]
        metric = arm["metrics"]["10000"]["ood_dev"]
        count_sum = sum(row["prefix_exact_count"] for row in metric_rows(metric, name)[12:36])
        scores[position] = count_sum
        if (recorded.get("exact_prefix_count_sum_13_36") != count_sum
                or recorded.get("denominator") != 102400 * 24
                or recorded.get("score") != count_sum / (102400 * 24)):
            raise ValueError("Recorded selection score differs from integer development tallies")
        report_path = Path(arm["input_files"]["report"]["path"])
        if (local_path(recorded["directory"]).resolve() != report_path.parent.resolve()
                or recorded.get("report_sha256") != arm["input_files"]["report"]["sha256"]):
            raise ValueError("Selection source report identity differs")
        checkpoint = arm["checkpoints"]["10000"]
        if any(recorded["checkpoint"].get(key) != checkpoint[key] for key in ("sha256", "bytes")):
            raise ValueError("Selection checkpoint identity differs")
        if (recorded["checkpoint"].get("completed_updates") != 10000
                or recorded["checkpoint"].get("examples_seen") != 10240000
                or local_path(recorded["checkpoint"]["path"]).resolve() != Path(checkpoint["path"]).resolve()
                or recorded.get("endpoint_metrics") != arm["metrics"]["10000"]):
            raise ValueError("Selection endpoint evidence differs")
    winner = "sinusoidal" if scores["sinusoidal"] > scores["alibi"] else "alibi"
    if (selection.get("selected_position") != winner
            or selection.get("selected_full_attention_directory") != selection["arms"][winner]["directory"]
            or selection.get("matched_order_chain") != arms[ARMS[0]]["report"]["order_chain"]
            or selection.get("next_run") != {"position_encoding": winner, "second_layer_window": 2,
                                             "initialization": "fresh original Mitchell", "updates": 10000}):
        raise ValueError("Recorded winner or subsequent window experiment differs")
    return winner


def compare_runs(baseline_dir, sinusoidal_dir, window_dir, selection_path, protocol_path):
    arms = {ARMS[0]: read_training(baseline_dir, ARMS[0]),
            ARMS[1]: read_window(sinusoidal_dir, ARMS[1]),
            ARMS[2]: read_window(window_dir, ARMS[2])}
    baseline = arms[ARMS[0]]["report"]
    baseline_args = json.loads(Path(arms[ARMS[0]]["input_files"]["run_config"]["path"]).read_text())
    baseline_order = [row["order_chain"] for row in arms[ARMS[0]]["history"]]
    full_parameter_hash = None
    for name in ARMS[1:]:
        arm, report = arms[name], arms[name]["report"]
        shared_contract = compare_contracts(baseline["contract"], report["contract"])
        args = json.loads(Path(arm["input_files"]["run_config"]["path"]).read_text())
        for key in RUN_SETTINGS:
            if baseline_args.get(key) != args.get(key):
                raise ValueError(f"Training argument differs: {name}/{key}")
        validate_source_lineage(baseline["source_files"], report["source_files"])
        if [row["order_chain"] for row in arm["history"]] != baseline_order:
            raise ValueError("The pilots consumed different per-update minibatch orders")
        initial = report["initialization"]
        if initial.get("baseline_initialization") != baseline["initialization"]:
            raise ValueError("A pilot did not start from the original paired initialization")
        if (initial.get("schema") != "rt-a5-window-initialization-v1"
                or initial.get("experiment_config") != report["contract"]["experiment_config"]
                or initial.get("changed_parameter_slices") != []
                or initial.get("predictor_sha256") != baseline["initialization"]["predictor_sha256"]
                or initial.get("canonical_sha256") != baseline["initialization"]["canonical_sha256"]
                or not _sha(initial.get("model_parameter_sha256"))):
            raise ValueError("Unexpected original-Mitchell initialization metadata")
        if full_parameter_hash is not None and initial["model_parameter_sha256"] != full_parameter_hash:
            raise ValueError("The new pilots used different initial learned tensors")
        full_parameter_hash = initial["model_parameter_sha256"]
        scope = lambda r: [(m["update"], m["role"], m["rows"], m["length"], m["route"]) for m in r["evaluations"]]
        if scope(baseline) != scope(report):
            raise ValueError("Evaluation schedules, samples or routes differ")
    for name, arm in arms.items():
        if arm["report"].get("wandb", {}).get("status") != "synced":
            raise ValueError(f"Online training evidence is not synced: {name}")
        for key, value in {"parameter_count": 7407104, "backbone_parameter_count": 6357504,
                           "predictor_parameter_count": 1049600}.items():
            if arm["report"]["initialization"].get(key) != value:
                raise ValueError(f"Unexpected model parameter count: {name}/{key}")
    raw, selection_file = read_input(selection_path)
    selection = json.loads(raw)
    raw, protocol_file = read_input(protocol_path)
    protocol = json.loads(raw)
    selector_file = hash_file(Path(selection_path).resolve().parent / "selector-source/rt_a5_window_select.py")
    if (selection.get("protocol_sha256") != protocol_file["sha256"]
            or selection.get("selector_sha256") != selector_file["sha256"]):
        raise ValueError("Prospective selector/protocol identity differs")
    winner = verify_selection(selection, arms, protocol)
    sinus_config = arms[ARMS[1]]["report"]["contract"]["experiment_config"]
    window_config = arms[ARMS[2]]["report"]["contract"]["experiment_config"]
    if (sinus_config["position_encoding"], sinus_config["second_layer_window"]) != ("sinusoidal", None):
        raise ValueError("The first new pilot must retain full attention and fixed sinusoids")
    if (window_config["position_encoding"], window_config["second_layer_window"]) != (winner, 2):
        raise ValueError("The window pilot did not use the prospectively selected positions")
    parent_arm = ARMS[0] if winner == "alibi" else ARMS[1]
    result = {
        "schema": SCHEMA, "primary_update": 10000, "diagnostic_updates": [1000, 5000],
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
        "evaluation_route": "backbone_only", "shared_contract": shared_contract,
        "position_selection": selection, "selected_position": winner,
        "window_parent_arm": parent_arm,
        "selection_input_files": {"selection": selection_file, "protocol": protocol_file, "selector": selector_file},
        "order_chain": baseline["order_chain"],
        "scope": "Two sequential single-seed development pilots; each fixed at 10000 updates; original Mitchell initialization",
        "intervals": "Pointwise Wilson 95% over words for E/A; no training-seed, paired-difference or M interval claim",
        "change_scope": "Stage1 sinusoids replace ALiBi; stage2 restricts only second-block direct reads to temporary self and previous permanent output",
        "causal_attribution": "Window comparison uses its selected full-attention parent; position selection is provisional one-seed development evidence",
        "plot_consistency": "Full and boundary figures plot the same 36 OOD rows at update10000; only x-axis limits differ",
        "arms": {},
    }
    for name, arm in arms.items():
        report = arm["report"]
        position_label = "ALiBi" if winner == "alibi" else "sinusoids"
        label = LABELS[name] if name != ARMS[2] else f"Mitchell + {position_label}, layer-2 window 2"
        result["arms"][name] = {
            "label": label, "contract": report["contract"], "initialization": report["initialization"],
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
            axis.plot(x, [r[metric] for r in rows], color=COLORS[arm], label=summary["arms"][arm]["label"], linewidth=1.9)
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
        figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
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
                      color=COLORS[arm], label=summary["arms"][arm]["label"], linewidth=1.8)
        axis.set(xlabel="Optimizer updates", ylabel=f"{title} (100-update means)")
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
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
    winner = summary["selected_position"]
    parent = summary["arms"][summary["window_parent_arm"]]["label"]
    lines = ["# A5: Mitchell positions and second-layer attention window pilots", "",
             "All three models use the original Mitchell initialization and the original NextLat predictor initialization. "
             "Each was trained for 10,000 updates with matched data order, objective, optimizer and FP32 runtime.", "",
             f"The first new pilot changes only ALiBi to fixed sinusoids. The frozen development criterion selected **{winner}** "
             "using mean cumulative-prefix exactness over positions 13–36 at update 10k. Exact ties select ALiBi. "
             "This is a provisional one-seed development choice, not a confirmed population ranking.", "",
             f"The second pilot compares against **{parent}** and restricts only RT layer 2 to its temporary self K/V "
             "and the preceding token's permanent output K/V. Layer 1 retains full causal recurrent attention. "
             "The recurrent graph remains attached across all earlier positions; window length 2 does not mean two-token memory. "
             "Both new pilots start fresh. Backbone-only inference, latent objective and predictor are unchanged.", "",
             "| Position candidate | Mean E(13–36), 10k | Selected |",
             "| --- | ---: | --- |"]
    for position in ("alibi", "sinusoidal"):
        value = summary["position_selection"]["arms"][position]
        lines.append(f"| {position} | {100 * value['score']:.4f}% | {'yes' if position == winner else 'no'} |")
    lines += ["", "| Checkpoint | Model | L12 dev token accuracy | L12 dev whole word | OOD E(13) | OOD E(14) | OOD E(16) | OOD M(36) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["checkpoint_summary"]:
        values = [r["dev_token_accuracy"], r["dev_whole_word_exact_match"],
                  r["E"]["13"], r["E"]["14"], r["E"]["16"], r["M36"]]
        label = summary["arms"][r["arm"]]["label"]
        lines.append(f"| {r['update']:,} ({r['selection']}) | {label} | "
                     + " | ".join(f"{100*v:.4f}%" for v in values) + " |")
    lines += ["", "Each checkpoint evaluation uses the same first 102,400 frozen words per development role. "
              "The 1k and 5k checkpoints are diagnostic; 10k remains the primary endpoint. Final confirmation "
              "and autonomous predictor rollout remain unevaluated.", "",
              "E(t) requires every state through t to be correct; A(t) measures only state t; M(t) averages "
              "token accuracy through t. Length curves are prefixes of the same length-36 outputs. Full and "
              "boundary plots use exactly the same rows and differ only in horizontal display limits.", "",
              "![Full length curves](length-full.png)", "", "![Boundary view](length-boundary.png)", "",
              "![Exactness only](length-exactness.png)", "", "![Training state CE and latent loss](training-losses.png)", "",
              "Bands show pointwise Wilson 95% intervals over words for E/A. They do not describe training-seed "
              "uncertainty or paired-difference confidence; M has no interval based on independent token positions. "
              "Zero observed exactness is not proof of zero population success.", "",
              "The report verifies complete histories, every minibatch order hash, retained checkpoint hashes, "
              "frozen source snapshots, shared contracts, initialization provenance, evaluation tallies and the recorded "
              "position selection. It performs no model inference or additional training.", "",
              "[Metrics CSV](metrics.csv) · [Plot data](plot-data.json) · [Machine-readable summary](summary.json) · "
              "[Reporting provenance](report.json)", ""]
    return "\n".join(lines)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Reporting requires a fresh output directory")
    summary = compare_runs(args.baseline_dir, args.sinusoidal_dir, args.window_dir,
                           args.selection, args.protocol)
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
    for name, record in summary["selection_input_files"].items():
        destination = output / "inputs/selection" / Path(record["path"]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["path"], destination)
        if hash_file(destination)["sha256"] != record["sha256"]:
            raise ValueError(f"Selection input changed while copying: {name}")
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
                            group=args.wandb_group, name="nextlat-mitchell-position-window-comparison-step10000")
    try:
        tracker.start({key: summary[key] for key in ("schema", "primary_update", "shared_contract",
                                                    "selected_position", "scope", "evaluation_route")})
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
    lineage = ROOT / ".runtime/rt-a5/20260914T175404Z-nextlat-mitchell-window"
    parser.add_argument("--sinusoidal-dir", default=str(lineage / "train-sinusoidal"))
    parser.add_argument("--window-dir", default=str(lineage / "train-window2"))
    parser.add_argument("--selection", default=str(lineage / "position-selection.json"))
    parser.add_argument("--protocol", default=str(lineage / "protocol.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group", default="20260914T175404Z-nextlat-mitchell-window")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
