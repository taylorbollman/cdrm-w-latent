#!/usr/bin/env python3
"""CPU-only saved-evidence report for the original RT + NextLat 10k->100k resume."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_budget_report import verify_continuation as verify_pure_continuation
from scripts.rt_a5_nextlat_report import (
    CSV_COLUMNS, DIAGNOSTICS, ROLES, STEPS, _digest_dict, _sha,
    local_path, metric_rows, normalized_contract, read_training, training_curve,
)
from scripts.rt_a5_report import finite_number, hash_file, read_arm, read_input, write_json

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / ".runtime/rt-a5/20260914T185736Z-rt-nextlat100k"
SCHEMA = "rt-a5-nextlat-budget-comparison-v1"
TRAIN_SCHEMA = "rt-a5-nextlat-training-v1"
SOURCE_SHA = "1e6d0c63289f01525bc0c19bba6b2646d61df10ddb815bc74f5c111b46f9f961"
PARENT_SHA = "c5e425eb0e3321ae87d74a83208bcc944827dfba7ae64af2206abf32ee7c7f5b"
PARENT_REPORT_SHA = "9db3a65484467e2e02a3a33eea680821e5abbff52c940abda3c54d3d92b98656"
NEW_STEPS = (20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000)
ENDPOINT = 100000
ROWS = 102400
KEY_LENGTHS = (12, 13, 14, 15, 16, 36)
REPORTING_SOURCES = ("scripts/rt_a5_nextlat_budget_report.py", "scripts/rt_a5_nextlat_report.py",
                     "scripts/rt_a5_budget_report.py", "scripts/rt_a5_length_report.py",
                     "scripts/rt_a5_report.py", "scripts/experiment_tracking.py")


def validate_protocol(protocol):
    for key, expected in {"schema": "rt-a5-nextlat-budget-protocol-v1", "start_update": 10000,
                          "endpoint": ENDPOINT, "additional_updates": 90000, "batch_size": 1024,
                          "checkpoint_steps": list(NEW_STEPS), "full_evaluation_rows": ROWS,
                          "confirmation_evaluated": False, "latent_rollout_evaluated": False,
                          "parent_checkpoint_sha256": PARENT_SHA,
                          "parent_report_sha256": PARENT_REPORT_SHA}.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Frozen continuation protocol differs: {key}")
    sources, contract = protocol["source_files"], protocol["strict_contract"]
    if len(sources) != 46 or _digest_dict(sources) != SOURCE_SHA or contract.get("source_sha256") != SOURCE_SHA:
        raise ValueError("Original frozen 46-source lineage differs")
    if contract.get("schema") != TRAIN_SCHEMA:
        raise ValueError("Protocol must describe original RT + NextLat, not pure RT")


def validate_history(history, report, start, stop):
    if len(history) != stop - start or any(row.get("update") != step
            for step, row in enumerate(history, start + 1)):
        raise ValueError("History must contain each requested update exactly once, in order")
    for row in history:
        if row.get("examples_seen") != row["update"] * 1024 or not _sha(row.get("order_chain")):
            raise ValueError("History word exposure or order chain differs")
        for key in ("seconds", "loss", "state_loss", "latent_loss", "weighted_latent_loss",
                    "token_accuracy", "whole_word_exact", "grad_norm"):
            if not finite_number(row.get(key)) or row[key] < 0:
                raise ValueError(f"Nonfinite or negative history metric: {key}")
        if row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError("History accuracy outside [0,1]")
        validate_loss(row)
        if any(key in row and not finite_number(row[key]) for key in DIAGNOSTICS):
            raise ValueError("Nonfinite training one-step diagnostic")
    if history[-1]["order_chain"] != report.get("order_chain"):
        raise ValueError("History endpoint order chain differs from report")
    seconds = sum(row["seconds"] for row in history)
    if (not finite_number(report.get("train_seconds"))
            or not math.isclose(seconds, report["train_seconds"], rel_tol=1e-9, abs_tol=1e-5)):
        raise ValueError("Training-loop time differs from complete history")
    if not finite_number(report.get("elapsed_seconds")) or report["elapsed_seconds"] < seconds:
        raise ValueError("Job elapsed time must include training-loop time")


def validate_loss(row):
    if (not math.isclose(row["weighted_latent_loss"], row["latent_loss"], rel_tol=1e-7, abs_tol=1e-9)
            or not math.isclose(row["loss"], row["state_loss"] + row["weighted_latent_loss"],
                                rel_tol=2e-7, abs_tol=2e-7)):
        raise ValueError("Combined loss differs from CE plus weight-one latent loss")


def verify_evaluations(report, start, stop, checkpoint_steps):
    """Verify every routine/full development record and teacher-conditioned diagnostic."""
    expected_keys = {(step, role) for step in range(start + 500, stop + 1, 500) for role in ROLES}
    actual_keys = [(m.get("update"), m.get("role")) for m in report["evaluations"]]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise ValueError("Missing, duplicate or unexpected development evaluation")
    for metric in report["evaluations"]:
        expected_rows = ROWS if metric["update"] in checkpoint_steps else 4096
        if metric.get("route") != "backbone_only" or metric.get("rows") != expected_rows:
            raise ValueError("Development evaluation route or expected row count differs")
        metric_rows(metric, "rt_nextlat")
    diagnostics = report["one_step_diagnostics"]
    keys = [(m.get("update"), m.get("role")) for m in diagnostics]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("Missing, duplicate or unexpected one-step diagnostic")
    for item in diagnostics:
        if (item.get("route") != "teacher_conditioned_one_step_diagnostics" or item.get("rows") != 1024
                or item.get("length") != {"dev": 12, "ood_dev": 36}[item["role"]]):
            raise ValueError("One-step diagnostic scope differs")
        for key in ("loss", "state_loss", "latent_loss", "weighted_latent_loss"):
            if not finite_number(item.get(key)) or item[key] < 0:
                raise ValueError("Nonfinite or negative diagnostic loss")
        validate_loss(item)
        if any(not finite_number(item.get("diagnostics", {}).get(key)) for key in DIAGNOSTICS):
            raise ValueError("Nonfinite or missing latent diagnostic")


def verify_run_config(config, protocol, *, pilot):
    expected = {"architecture": "rt", "width": 512, "seed": 1234, "predictor_seed": 1235,
                "predictor_hidden_width": None, "latent_weight": 1.0, "data_order_seed": 1234,
                "batch_size": 1024, "eval_every": 500, "eval_rows": 4096, "full_eval_rows": ROWS,
                "diagnostic_rows": 1024, "log_every": 25, "wandb_project": "rt-a5-state-tracking",
                "updates": 10000 if pilot else ENDPOINT,
                "checkpoint_steps": list(STEPS if pilot else NEW_STEPS)}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Run configuration differs from approved recipe: {key}")
    if pilot and config.get("resume") is not None:
        raise ValueError("Original pilot must be fresh")
    if not pilot and local_path(config["resume"]).resolve() != local_path(protocol["parent_checkpoint"]).resolve():
        raise ValueError("Continuation resume CLI differs from approved parent")
    if local_path(config["data_dir"]).resolve() != local_path(protocol["data_dir"]).resolve():
        raise ValueError("Run data directory differs from frozen protocol")


def verify_join(pilot, continuation, protocol, through_update):
    """No restart, reset, overlapping tail, or changed recipe may masquerade as a resume."""
    left, right = pilot["report"], continuation["report"]
    if (left["start_update"], left["completed_updates"], right["start_update"], right["endpoint"]) != (0, 10000, 10000, ENDPOINT):
        raise ValueError("Expected original 0->10000 and exact 10000->100000 continuation")
    if left["contract"] != right["contract"] or right["contract"] != protocol["strict_contract"]:
        raise ValueError("Strict source/data/model/runtime/objective/optimizer contracts differ")
    if left["initialization"] != right["initialization"]:
        raise ValueError("Original Mitchell and predictor initialization lineage differs")
    if (left["source_files"] != right["source_files"] or right["source_files"] != protocol["source_files"]):
        raise ValueError("Frozen training sources differ")
    expected = pilot["checkpoints"]["10000"]
    parent = right.get("parent_checkpoint") or {}
    if (parent.get("sha256") != expected["sha256"] or expected["sha256"] != protocol["parent_checkpoint_sha256"]
            or local_path(parent["path"]).resolve() != local_path(expected["path"]).resolve()):
        raise ValueError("Continuation parent checkpoint identity differs")
    if pilot["input_files"]["report"]["sha256"] != protocol["parent_report_sha256"]:
        raise ValueError("Original parent report identity differs")
    combined = pilot["history"] + [row for row in continuation["history"] if row["update"] <= through_update]
    if len(combined) != through_update or any(row["update"] != i for i, row in enumerate(combined, 1)):
        raise ValueError("Joined history must contain every update exactly once")
    return combined


def read_continuation(directory, protocol, through_update):
    directory = local_path(directory).resolve()
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    partial = through_update != ENDPOINT
    if (report.get("schema") != TRAIN_SCHEMA or report.get("status") not in
            (("running", "complete") if partial else ("complete",))):
        raise ValueError("Expected completed RT + NextLat report (running allowed only for explicit interim snapshot)")
    observed = report.get("completed_updates", 0)
    if (type(observed) is not int or observed < through_update or observed > ENDPOINT
            or report.get("endpoint") != ENDPOINT or report.get("start_update") != 10000):
        raise ValueError("Requested checkpoint not yet completed or continuation scope differs")
    if not partial and observed != ENDPOINT:
        raise ValueError("Fixed 100k endpoint has not completed")
    if not partial and report.get("wandb", {}).get("status") != "synced":
        raise ValueError("Completed continuation must have synced online W&B evidence")
    if report.get("confirmation_evaluated") is not False or report.get("latent_rollout_evaluated") is not False:
        raise ValueError("Confirmation and latent rollout must remain unevaluated")
    contract, sources = report["contract"], report["source_files"]
    if contract != protocol["strict_contract"] or sources != protocol["source_files"]:
        raise ValueError("Continuation strict contract or frozen sources differ")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(expected):
            raise ValueError("Invalid source snapshot path/hash")
        if hash_file(directory / "source" / path)["sha256"] != expected:
            raise ValueError(f"Frozen source snapshot changed: {relative}")
    config_raw, config_file = read_input(directory / "config.json")
    config = json.loads(config_raw)
    verify_run_config(config, protocol, pilot=False)
    manifest = hash_file(local_path(config["data_dir"]) / "manifest.json")
    if manifest["sha256"] != contract["data_manifest_sha256"]:
        raise ValueError("Frozen data manifest differs")
    # A live report is atomically written at saved checkpoints. Read exactly that
    # many history lines, never include its later in-progress tail in the snapshot.
    if partial:
        with (directory / "history.jsonl").open("rb") as stream:
            history_raw = b"".join(stream.readline() for _ in range(observed - 10000))
        history_file = {"path": str(directory / "history.jsonl"), "bytes": len(history_raw),
                        "sha256": hashlib.sha256(history_raw).hexdigest(),
                        "snapshot": f"exact immutable prefix through observed saved update {observed}"}
    else:
        history_raw, history_file = read_input(directory / "history.jsonl")
    history = [json.loads(line) for line in history_raw.splitlines() if line.strip()]
    validate_history(history, report, 10000, observed)
    selected_steps = [step for step in NEW_STEPS if step <= observed]
    if [item["completed_updates"] for item in report["checkpoints"]] != selected_steps:
        raise ValueError("Unexpected retained continuation checkpoint schedule")
    checkpoints = {}
    for item in report["checkpoints"]:
        step = item["completed_updates"]
        actual = hash_file(local_path(item["path"]))
        if item.get("examples_seen") != step * 1024 or any(actual[k] != item[k] for k in ("sha256", "bytes")):
            raise ValueError(f"Retained checkpoint identity or exposure differs: {step}")
        checkpoints[str(step)] = actual
    verify_evaluations(report, 10000, observed, NEW_STEPS)
    return {"report": report, "history": history, "checkpoints": checkpoints,
            "input_files": {"report": report_file, "history": history_file, "run_config": config_file,
                            "data_manifest": manifest},
            "input_bytes": {"report": raw, "history": history_raw, "run_config": config_raw}}


def read_pure_references(pilot_dir, run_dir, nextlat_contract, nextlat_initialization, nextlat_history):
    pilot, continuation = read_arm(local_path(pilot_dir), "rt"), read_arm(local_path(run_dir), "rt")
    history = verify_pure_continuation(pilot, continuation, ENDPOINT)
    if normalized_contract(pilot["report"]["contract"]) != normalized_contract(nextlat_contract):
        raise ValueError("Pure RT reference shared architecture/data/runtime contract differs")
    if pilot["report"]["contract"].get("optimizer") + "-all-hybrid" != nextlat_contract.get("optimizer"):
        raise ValueError("Pure RT and NextLat reference optimizer recipes differ")
    if pilot["report"]["initialization"] != nextlat_initialization["backbone"]:
        raise ValueError("Pure RT reference canonical backbone initialization differs")
    if [row["order_chain"] for row in history[:len(nextlat_history)]] != [row["order_chain"] for row in nextlat_history]:
        raise ValueError("Per-update data order differs from independently retained pure RT history")
    sources = pilot["report"]["source_files"]
    if len(sources) != 43 or _digest_dict(sources) != "6f9d55a957bf505aefa1a351c33f0bc76291ff63736bc28b1b618393f46ef6ea":
        raise ValueError("Original pure RT 43-source lineage differs")
    result = {"label": "Pure RT (without NextLat), descriptive reference", "metrics": {}, "curves": {},
              "inputs": {}, "checkpoints": {}, "wandb": {}, "source_files": sources,
              "contract": pilot["report"]["contract"], "order_chain": history[-1]["order_chain"]}
    for step, phase, record in ((10000, "pilot", pilot), (ENDPOINT, "continuation", continuation)):
        result["metrics"][str(step)] = record["endpoint_metrics"]
        result["curves"][str(step)] = {}
        result["inputs"][phase] = record["input_files"]
        result["checkpoints"][str(step)] = record["input_files"]["endpoint_checkpoint"]
        result["wandb"][phase] = record["report"].get("wandb")
        for role, metric in record["endpoint_metrics"].items():
            if metric["rows"] != ROWS:
                raise ValueError("Pure RT reference must use identical full development row counts")
            result["curves"][str(step)][role] = metric_rows(metric, "rt")
    return result


def make_summary(pilot_dir, run_dir, protocol_path, *, through_update=ENDPOINT,
                 pure_pilot_dir=None, pure_run_dir=None):
    if through_update not in (20000, ENDPOINT) or isinstance(through_update, bool):
        raise ValueError("Use fixed 100k endpoint or explicitly bounded interim 20k snapshot")
    raw, protocol_file = read_input(local_path(protocol_path))
    protocol = json.loads(raw)
    validate_protocol(protocol)
    pilot = read_training(local_path(pilot_dir), "rt_nextlat")
    verify_run_config(json.loads(Path(pilot["input_files"]["run_config"]["path"]).read_text()), protocol, pilot=True)
    validate_history(pilot["history"], pilot["report"], 0, 10000)
    verify_evaluations(pilot["report"], 0, 10000, STEPS)
    continuation = read_continuation(run_dir, protocol, through_update)
    history = verify_join(pilot, continuation, protocol, through_update)
    steps = [*STEPS, *(step for step in NEW_STEPS if step <= through_update)]
    curves, metrics, checkpoints = {}, {}, {}
    for step in steps:
        source = pilot if step <= 10000 else continuation
        checkpoints[str(step)] = source["checkpoints"][str(step)]
        curves[str(step)], metrics[str(step)] = {}, {}
        for role in ROLES:
            selected = [m for m in source["report"]["evaluations"] if (m["update"], m["role"]) == (step, role)]
            if len(selected) != 1 or selected[0]["rows"] != ROWS:
                raise ValueError("Missing full development checkpoint evaluation")
            metrics[str(step)][role] = selected[0]
            curves[str(step)][role] = metric_rows(selected[0], "rt_nextlat")
    checkpoints["0"] = pilot["checkpoints"]["0"]
    if pure_pilot_dir is None or pure_run_dir is None:
        raise ValueError("Existing pure RT references are required for independent data-order verification")
    pure = read_pure_references(pure_pilot_dir, pure_run_dir, protocol["strict_contract"],
                                pilot["report"]["initialization"], history)
    partial = through_update != ENDPOINT
    result = {"schema": SCHEMA, "primary_update": ENDPOINT, "reported_through_update": through_update,
              "endpoint_completed": not partial, "checkpoint_updates": steps,
              "scope": ("Interim20k diagnostic snapshot; training continues to fixed 100k, with no endpoint selection"
                        if partial else "Fixed 100k endpoint of one original RT + NextLat development seed; intermediate checkpoints diagnostic"),
              "confirmation_evaluated": False, "latent_rollout_evaluated": False,
              "evaluation_route": "backbone_only", "contract": protocol["strict_contract"],
              "initialization": pilot["report"]["initialization"], "source_files": protocol["source_files"],
              "protocol": protocol, "protocol_input": protocol_file,
              "inputs": {"pilot": pilot["input_files"], "continuation": continuation["input_files"]},
              "parent_checkpoint": continuation["report"]["parent_checkpoint"],
              "order_chain": history[-1]["order_chain"],
              "observed_continuation_saved_update": continuation["report"]["completed_updates"],
              "historical_wandb": {"pilot": pilot["report"]["wandb"], "continuation": continuation["report"]["wandb"]},
              "checkpoints": checkpoints, "historical_metrics": metrics, "curves": curves,
              "pure_rt_reference": pure, "training_curve": training_curve(history, True),
              "training_bin_updates": 100, "key_lengths": list(KEY_LENGTHS),
              "budget": {"total_updates": through_update, "authorized_endpoint": ENDPOINT,
                         "additional_updates": through_update - 10000, "unique_training_words": 800000,
                         "total_word_presentations": through_update * 1024,
                         "total_token_presentations": through_update * 1024 * 12,
                         "nominal_training_passes": through_update * 1024 / 800000},
              "timing": {"training_seconds": sum(r["seconds"] for r in history),
                         "continuation_training_seconds": sum(r["seconds"] for r in history[10000:]),
                         "definition": "Summed update durations exclude evaluation, checkpointing and logging; partial snapshot excludes later updates."},
              "metric_definitions": {"E": "Every state through t correct", "A": "Only state at t correct",
                                     "M": "Mean token correctness through t"},
              "intervals": "Pointwise Wilson 95% across words for E/A; not seed or paired-difference uncertainty; no M interval",
              "input_bytes": {"protocol": raw, "continuation": continuation["input_bytes"]}}
    result["checkpoint_summary"] = checkpoint_summary(result)
    return result


def metric_table(summary):
    rows = [row for step in summary["checkpoint_updates"] for role in ROLES
            for row in summary["curves"][str(step)][role]]
    rows += [row for step in (10000, ENDPOINT) for role in ROLES
             for row in summary["pure_rt_reference"]["curves"][str(step)][role]]
    expected = (len(summary["checkpoint_updates"]) + 2) * 48
    if len(rows) != expected or len({(r["arm"], r["update"], r["role"], r["length"]) for r in rows}) != expected:
        raise ValueError("Metric table contains missing or duplicate positions")
    return rows


def endpoint_plot_rows(summary, step):
    """Both full and zoom figures consume this identical set of 36 rows."""
    if step not in (10000, summary["reported_through_update"]):
        raise ValueError("Endpoint display must use baseline 10k or declared report checkpoint")
    return summary["curves"][str(step)]["ood_dev"]


def checkpoint_summary(summary):
    result = []
    for arm, label, curves, metrics, steps in (
            ("rt_nextlat", "Original RT + NextLat (Mitchell, ALiBi, both layers full RT)",
             summary["curves"], summary["historical_metrics"], summary["checkpoint_updates"]),
            ("rt", "Pure RT, without NextLat (descriptive equal-budget reference)",
             summary["pure_rt_reference"]["curves"], summary["pure_rt_reference"]["metrics"], (10000, ENDPOINT))):
        for step in steps:
            dev, ood = metrics[str(step)]["dev"], curves[str(step)]["ood_dev"]
            row = {"arm": arm, "label": label, "update": step,
                   "selection": "fixed primary" if arm == "rt_nextlat" and step == ENDPOINT else "diagnostic/reference",
                   "dev_token_accuracy": dev["token_accuracy"], "dev_whole_word_exact_match": dev["whole_word_exact_match"],
                   "ood_ce": metrics[str(step)]["ood_dev"]["ce"],
                   "A36": ood[-1]["A"], "M36": ood[-1]["M"]}
            row.update({f"E{length}": ood[length - 1]["E"] for length in KEY_LENGTHS})
            result.append(row)
    return result


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    output = Path(output)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}

    def save(figure, name):
        figures[name] = {}
        for suffix in ("png", "pdf"):
            figures[name][suffix] = f"{name}.{suffix}"
            figure.savefig(output / figures[name][suffix], dpi=200)
        plt.close(figure)

    endpoint = summary["reported_through_update"]
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
        for step, color in ((10000, "#C68423"), (endpoint, "#3465A4")):
            rows = endpoint_plot_rows(summary, step)
            for axis, key, title in zip(axes, ("E", "A", "M"),
                    ("E: every state through t correct", "A: only state t correct", "M: mean token accuracy through t")):
                axis.plot([r["length"] for r in rows], [r[key] for r in rows], label=f"RT + NextLat, {step//1000}k", color=color)
                if key != "M":
                    axis.fill_between([r["length"] for r in rows], [r[f"{key}_low95"] for r in rows],
                                      [r[f"{key}_high95"] for r in rows], color=color, alpha=.15)
                axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Prefix length t of same length-36 words", title=title)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.axvline(12, color="black", linestyle=":", alpha=.45)
                axis.grid(alpha=.18)
                axis.legend(fontsize=8)
        axes[1].axhline(1/60, color="gray", linestyle="--", linewidth=.8)
        figure.suptitle(f"Original RT + NextLat · 10k versus {endpoint//1000}k · same 102,400 OOD development words")
        save(figure, name)

    figure, axis = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    steps = summary["checkpoint_updates"]
    for length, color in ((13, "#C68423"), (14, "#3465A4"), (16, "#24938C")):
        axis.plot([s/1000 for s in steps], [summary["curves"][str(s)]["ood_dev"][length-1]["E"] for s in steps],
                  marker="o", markersize=3, color=color, label=f"E({length})")
    axis.set(xlabel="Total optimizer updates (thousands)", ylabel="Cumulative prefix exactness",
             ylim=(-.025, 1.025), title="RT + NextLat · fixed development prefixes versus training budget")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(alpha=.18)
    axis.legend()
    save(figure, "exactness-vs-updates")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, key, label in zip(axes, ("state_ce", "latent_loss"), ("Training state CE", "Training latent SmoothL1")):
        points = summary["training_curve"]
        axis.plot([p["update"]/1000 for p in points], [p[key] for p in points], color="#3465A4", linewidth=1)
        axis.axvline(10, color="black", linestyle=":", alpha=.45, label="Exact resume at 10k")
        axis.set(xlabel="Total optimizer updates (thousands)", ylabel=label)
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    figure.suptitle("Original RT + NextLat · complete pilot + continuation · nonoverlapping 100-update means")
    save(figure, "training-losses")
    return figures


def markdown_report(summary):
    endpoint = summary["reported_through_update"]
    title = f"Original RT + NextLat: {'completed 100k continuation' if summary['endpoint_completed'] else 'interim 20k diagnostic'}"
    lines = [f"# {title}", "", summary["scope"] + ".", "",
             "This continues the original **RT + NextLat** 10k checkpoint, including Adam, RNG and absolute data-order state. "
             "It is distinct from the previously completed pure RT 100k run. Both recurrent layers retain full-prefix attention, "
             "Mitchell initialization and ALiBi; no sinusoidal, identity-init or sliding-window changes are present.", "",
             "Two tiled RT blocks, D512/H8/GELU-FFN2048, LayerNorm/full-width QK normalization, rho1. "
             "The backbone has 6,357,504 parameters and the auxiliary predictor 1,049,600 (total 7,407,104). "
             "Full FP32, math attention, no autocast, TF32, compilation or CUDA graphs. The unchanged objective is "
             "same-position state CE plus weight-one SmoothL1 latent prediction, with only the target latent detached. "
             "All accuracy evaluations use the backbone; the auxiliary predictor is never rolled out.", "",
             f"Reported budget: **{endpoint:,} total updates**, including {endpoint-10000:,} new updates, "
             f"{endpoint*1024:,} training-word presentations over 800,000 unique length-12 words "
             f"({endpoint*1024/800000:g} nominal passes). Fixed primary endpoint: **100,000**, with no best-checkpoint selection.", "",
             "| Model | Updates | L12 token | L12 whole word | E(13) | E(14) | E(16) | M(36) | E(36) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["checkpoint_summary"]:
        if row["arm"] == "rt_nextlat" and row["update"] not in (10000, endpoint):
            continue
        label = "RT + NextLat" if row["arm"] == "rt_nextlat" else "Pure RT, without NextLat (reference)"
        values = [row[k] for k in ("dev_token_accuracy", "dev_whole_word_exact_match", "E13", "E14", "E16", "M36", "E36")]
        lines.append(f"| {label} | {row['update']:,} | " + " | ".join(f"{100*v:.4f}%" for v in values) + " |")
    lines += ["", "The pure RT rows are already-existing descriptive references, without the NextLat objective or predictor. "
              "Only comparisons at the same update count have matched training budgets. Their shared backbone initialization, "
              "data/runtime contract and complete minibatch order were verified. No reference was retrained or re-evaluated.", "",
              "E(t) requires **every** state through t to be correct. A(t) measures only state t; M(t) averages correctness "
              "through t. All OOD curves come from prefixes of the same 102,400 frozen length-36 words, and full/boundary plots "
              "consume identical rows. L12 development is a separate short-word set. A 1/60 line applies only to isolated A(t).", "",
              "![Full curves](length-full.png)", "", "![Boundary view](length-boundary.png)", "",
              "![Exactness versus updates](exactness-vs-updates.png)", "", "![Training losses](training-losses.png)", "",
              "| RT + NextLat updates | E(13) | E(14) | E(16) | M(36) | OOD CE |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["checkpoint_summary"]:
        if row["arm"] == "rt_nextlat":
            lines.append(f"| {row['update']:,} | " + " | ".join(f"{100*row[k]:.4f}%" for k in ("E13", "E14", "E16", "M36"))
                         + f" | {row['ood_ce']:.6f} |")
    lines += ["", "Every retained checkpoint above uses 102,400 words per development role. Intermediate outcomes are diagnostic, "
              "and fluctuations do not imply a monotonic trend. Bands are pointwise Wilson 95% across words for E/A, not seed "
              "variation, simultaneous coverage or paired-difference intervals. Zero observed E means zero successes in this sample.", "",
              "This is one development seed and our existing generated corpus, not a multi-seed convergence claim or exact paper "
              "reproduction. Final confirmation and autonomous latent rollout remain **unevaluated**. The reporter only reads saved "
              "evidence, validates source/checkpoint hashes, exact continuation history, finite metrics and integer counts, and creates plots.", "",
              f"Training-loop time represented: {summary['timing']['training_seconds']/3600:.3f}h; "
              f"continuation portion {summary['timing']['continuation_training_seconds']/3600:.3f}h. These sums exclude evaluation/checkpoint/logging overhead.", "",
              "[All metrics/counts](metrics.csv) · [Checkpoint summary](checkpoint-summary.csv) · [Training bins](training-curves.csv) · "
              "[Plot data](plot-data.json) · [Provenance](report.json)", ""]
    return "\n".join(lines)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    output = local_path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh continuation report directory")
    summary = make_summary(args.pilot_dir, args.run_dir, args.protocol, through_update=args.through_update,
                           pure_pilot_dir=args.pure_pilot_dir, pure_run_dir=args.pure_run_dir)
    snapshots = summary.pop("input_bytes")
    rows = metric_table(summary)
    output.mkdir(parents=True)
    # Copy exactly the bytes already validated; live continuation inputs can advance.
    for phase, inputs in summary["inputs"].items():
        for key, record in inputs.items():
            target = output / "inputs" / phase / Path(record["path"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if phase == "continuation" and key in snapshots["continuation"]:
                target.write_bytes(snapshots["continuation"][key])
            else:
                shutil.copyfile(record["path"], target)
            if any(hash_file(target)[field] != record[field] for field in ("sha256", "bytes")):
                raise ValueError(f"Input changed during snapshot: {phase}/{key}")
    for phase, inputs in summary["pure_rt_reference"]["inputs"].items():
        for key, record in inputs.items():
            if key == "endpoint_checkpoint":
                continue
            target = output / "inputs/pure-rt" / phase / Path(record["path"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record["path"], target)
            if hash_file(target)["sha256"] != record["sha256"]:
                raise ValueError("Pure RT reference changed during evidence copy")
    (output / "inputs/protocol.json").write_bytes(snapshots["protocol"])
    summary["reporting_sources"] = {}
    for relative in REPORTING_SOURCES:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        summary["reporting_sources"][relative] = hash_file(target)
    summary["reporting_source_sha256"] = _digest_dict({k: v["sha256"] for k, v in summary["reporting_sources"].items()})
    write_csv(output / "metrics.csv", rows, CSV_COLUMNS)
    with (output / "metrics.csv").open() as stream:
        if list(csv.DictReader(stream)) != [{k: str(r[k]) for k in CSV_COLUMNS} for r in rows]:
            raise ValueError("CSV values differ from exact plot rows")
    write_csv(output / "checkpoint-summary.csv", summary["checkpoint_summary"], summary["checkpoint_summary"][0].keys())
    write_csv(output / "training-curves.csv", summary["training_curve"], summary["training_curve"][0].keys())
    write_json(output / "summary.json", summary)
    write_json(output / "plot-data.json", summary)
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name=f"rt-nextlat-original-budget-report-through{args.through_update}")
    result = {**summary, "status": "running", "figures": figures}
    try:
        tracker.start({key: summary[key] for key in ("schema", "scope", "contract", "budget", "primary_update", "reported_through_update")})
        import wandb
        tracker.log({"report/metrics": wandb.Table(columns=list(CSV_COLUMNS), data=[[r[k] for k in CSV_COLUMNS] for r in rows]),
                     **{f"report/{name}": wandb.Image(str(output / files["png"])) for name, files in figures.items()}})
        for point in summary["training_curve"]:
            values = {"update": point["update"], **{f"train/{k}": v for k, v in point.items()
                      if k not in ("update", "first_update", "updates_in_bin")}}
            if str(point["update"]) in summary["curves"]:
                for length in KEY_LENGTHS:
                    row = summary["curves"][str(point["update"])]["ood_dev"][length-1]
                    values.update({f"dev/ood_prefix_{length}/{key}": row[key] for key in ("E", "A", "M")})
            tracker.log(values)
        tracker.summary({"primary_update": ENDPOINT, "reported_through_update": args.through_update,
                         "endpoint_completed": summary["endpoint_completed"], "confirmation_evaluated": False,
                         "latent_rollout_evaluated": False, "checkpoint_summary": summary["checkpoint_summary"]})
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
        result["artifacts"] = {str(p.relative_to(output)): hash_file(p) for p in sorted(output.rglob("*"))
                               if p.is_file() and "wandb" not in p.relative_to(output).parts and p != output / "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": tracker.record["run_url"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", default=str(ROOT / ".runtime/rt-a5/20260911T191702Z-nextlat/train-rt-nextlat"))
    parser.add_argument("--run-dir", default=str(LINEAGE / "train-rt-nextlat"))
    parser.add_argument("--protocol", default=str(LINEAGE / "protocol.json"))
    parser.add_argument("--pure-pilot-dir", default=str(ROOT / ".runtime/rt-a5/20260911T154748Z/train-rt"))
    parser.add_argument("--pure-run-dir", default=str(ROOT / ".runtime/rt-a5/20260911T171239Z-rt100k/train-rt"))
    parser.add_argument("--through-update", "--endpoint", dest="through_update", type=int,
                        choices=(20000, ENDPOINT), default=ENDPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group", default=LINEAGE.name)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
