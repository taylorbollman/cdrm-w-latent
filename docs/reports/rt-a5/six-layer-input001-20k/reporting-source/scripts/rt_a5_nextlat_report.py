#!/usr/bin/env python3
"""Verify and report four completed A5 pilots without model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_length_report import recover_count, wilson95
from scripts.rt_a5_report import checkpoint_path, finite_number, hash_file, read_input, write_json


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-nextlat-comparison-v1"
STEPS = (1000, 5000, 10000)
ROLES = ("dev", "ood_dev")
ARMS = ("rt", "rt_nextlat", "seq", "seq_nextlat")
LABELS = {"rt": "RT", "rt_nextlat": "RT + NextLat", "seq": "Transformer",
          "seq_nextlat": "Transformer + NextLat"}
COLORS = {"rt": "#B85031", "rt_nextlat": "#E5A035", "seq": "#3465A4",
          "seq_nextlat": "#24938C"}
NEW_SOURCES = {"scripts/rt_a5_nextlat.py", "scripts/rt_a5_nextlat_train.py",
               "configs/rt_a5_nextlat/base.json"}
DIAGNOSTICS = ("latent_rms", "predicted_latent_rms", "latent_variation_rms",
               "latent_relative_l2", "latent_cosine", "predicted_state_ce",
               "teacher_predicted_kl")
OPTIMIZER = "AdamW-lr1e-4-betas0.9,0.95-eps1e-8-wd0.01-matrices-clip1"
CSV_COLUMNS = ("arm", "update", "role", "length", "words", "prefix_exact_count",
               "state_correct_count", "token_correct_count", "token_denominator",
               "E", "E_low95", "E_high95", "A", "A_low95", "A_high95", "M", "state_ce")


def local_path(value):
    path = checkpoint_path(value)
    return path if path.is_absolute() else ROOT / path


def _digest_dict(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def metric_rows(metric, arm):
    """Recover integer E/A/M tallies; finite-sample zeros stay literal zeros."""
    n, length, role = metric["rows"], metric["length"], metric["role"]
    if (type(n) is not int or n < 1 or role not in ROLES
            or length != {"dev": 12, "ood_dev": 36}[role]
            or metric["tokens"] != n * length or metric.get("evaluated_rows", n) != n):
        raise ValueError("Invalid development evaluation size or role")
    a, e, ce = (metric[key] for key in
                ("isolated_state_accuracy", "cumulative_prefix_exactness", "per_position_ce"))
    if len(a) != length or len(e) != length or len(ce) != length:
        raise ValueError("Evaluation must include every operation position")
    ac, ec = [recover_count(v, n) for v in a], [recover_count(v, n) for v in e]
    if (ac[0] != ec[0] or any(x > y for x, y in zip(ec, ac))
            or any(after > before or before - after > n - state
                   for before, after, state in zip(ec, ec[1:], ac[1:]))):
        raise ValueError("Counts violate cumulative prefix correctness semantics")
    if any(not finite_number(v) or v < 0 for v in ce):
        raise ValueError("Invalid per-position CE")
    for key, expected in (("token_accuracy", sum(ac) / (n * length)),
                          ("whole_word_exact_match", ec[-1] / n),
                          ("final_state_accuracy", ac[-1] / n), ("ce", sum(ce) / length)):
        if not finite_number(metric[key]) or not math.isclose(metric[key], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Evaluation scalar {key} disagrees with its curve")
    rows, token_count = [], 0
    for position, (state, exact, state_ce) in enumerate(zip(ac, ec, ce), 1):
        token_count += state
        elo, ehi = wilson95(exact, n)
        alo, ahi = wilson95(state, n)
        rows.append(dict(zip(CSV_COLUMNS, (arm, metric["update"], role, position, n, exact,
                    state, token_count, n * position, exact / n, elo, ehi,
                    state / n, alo, ahi, token_count / (n * position), state_ce))))
    return rows


def read_training(directory, arm):
    directory = Path(directory).resolve()
    augmented = arm.endswith("_nextlat")
    architecture = arm.split("_")[0]
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    expected_schema = "rt-a5-nextlat-training-v1" if augmented else "rt-a5-training-v1"
    if report.get("schema") != expected_schema or report.get("status") != "complete":
        raise ValueError(f"{arm}: expected a completed training report")
    if (report.get("start_update") != 0 or report.get("completed_updates") != 10000
            or report.get("endpoint") != 10000 or report.get("parent_checkpoint") is not None):
        raise ValueError(f"{arm}: require a fresh complete 10000-update pilot")
    if report.get("confirmation_evaluated") is not False:
        raise ValueError("Final confirmation must remain unevaluated")
    contract = report["contract"]
    expected = {"architecture": architecture, "schema": expected_schema,
                "width": 512, "seed": 1234, "data_order_seed": 1234,
                "batch_size": 1024, "length": 12, "train_rows": 800000,
                "precision": "fp32", "tf32": False, "compile": False, "cuda_graphs": False,
                "optimizer": OPTIMIZER + ("-all-hybrid" if augmented else "")}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"{arm}: incompatible contract setting {key}")
    config = contract["model_config"]
    for key, value in {"block_type": "recurrent" if architecture == "rt" else "sequential",
                       "n_layers": 2, "d_model": 512, "n_heads": 8, "mlp_hidden_size": 2048,
                       "activation_type": "gelu", "alibi": True, "recurrent_layers": None,
                       "cdrm_enabled": False, "reference_eager": True,
                       "recurrent_write_rho": 1.0, "vocab_size": 60, "weight_tying": False}.items():
        if config.get(key) != value:
            raise ValueError(f"{arm}: incompatible backbone setting {key}")
    if augmented:
        if (report.get("latent_rollout_evaluated") is not False
                or contract.get("latent_rollout_evaluated") is not False
                or contract.get("evaluation_route") != "backbone_only"):
            raise ValueError("NextLat evaluations must use only the backbone")
        objective = contract["objective"]
        for key, value in {"horizon": 1, "latent_weight": 1.0, "target_detached": True,
                           "source_and_embedding_attached": True, "kl_weight": 0.0,
                           "predicted_state_ce_weight": 0.0}.items():
            if objective.get(key) != value:
                raise ValueError(f"Unexpected NextLat objective {key}")
        predictor = contract["nextlat_config"]["predictor"]
        for key, value in {"hidden_width": 512, "input_width": 1024, "output_width": 512,
                           "normalization": "rms_norm", "normalization_epsilon": 1e-5,
                           "linear_layers": 3, "activation": "gelu", "bias": False,
                           "residual": True, "output_normalization": False}.items():
            if predictor.get(key) != value:
                raise ValueError(f"Unexpected NextLat predictor {key}")
        if (contract["nextlat_config"].get("latent") != "post_final_layer_norm"
                or contract.get("predictor_seed") != 1235):
            raise ValueError("Unexpected latent definition or predictor seed")
    sources = report["source_files"]
    if _digest_dict(sources) != contract["source_sha256"]:
        raise ValueError("Source manifest digest differs from training contract")
    for relative, expected_hash in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not _sha(expected_hash):
            raise ValueError("Invalid source snapshot path or SHA256")
        if hash_file(directory / "source" / path)["sha256"] != expected_hash:
            raise ValueError(f"Source snapshot differs: {relative}")
    run_config_raw, run_config_file = read_input(directory / "config.json")
    run_config = json.loads(run_config_raw)
    manifest_file = hash_file(local_path(run_config["data_dir"]) / "manifest.json")
    if manifest_file["sha256"] != contract["data_manifest_sha256"]:
        raise ValueError("Actual data manifest differs from training contract")
    history_raw, history_file = read_input(directory / "history.jsonl")
    history = [json.loads(line) for line in history_raw.splitlines() if line.strip()]
    if [row["update"] for row in history] != list(range(1, 10001)):
        raise ValueError("History has missing, duplicate or unordered updates")
    for row in history:
        if row["examples_seen"] != row["update"] * 1024 or not _sha(row["order_chain"]):
            raise ValueError("History word count/order chain invalid")
        for key in ("seconds", "loss", "token_accuracy", "whole_word_exact", "grad_norm"):
            if not finite_number(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid history metric {key}")
        if row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError("Invalid history accuracy")
        if augmented:
            for key in ("state_loss", "latent_loss", "weighted_latent_loss"):
                if not finite_number(row[key]) or row[key] < 0:
                    raise ValueError(f"Invalid auxiliary history metric {key}")
            if (not math.isclose(row["weighted_latent_loss"], row["latent_loss"], rel_tol=1e-7, abs_tol=1e-9)
                    or not math.isclose(row["loss"], row["state_loss"] + row["weighted_latent_loss"],
                                        rel_tol=2e-7, abs_tol=2e-7)):
                raise ValueError("Combined training loss disagrees with CE plus latent objective")
            for key in DIAGNOSTICS:
                if key in row and not finite_number(row[key]):
                    raise ValueError(f"Nonfinite one-step diagnostic {key}")
    if history[-1]["order_chain"] != report["order_chain"]:
        raise ValueError("History endpoint order differs from report")
    if not math.isclose(sum(r["seconds"] for r in history), report["train_seconds"], rel_tol=1e-9, abs_tol=1e-5):
        raise ValueError("Reported training-loop time differs from history")
    checkpoints = {}
    for step in (0, *STEPS):
        selected = [r for r in report["checkpoints"] if r["completed_updates"] == step]
        if len(selected) != 1:
            raise ValueError(f"Need exactly one retained checkpoint at {step}")
        actual = hash_file(local_path(selected[0]["path"]))
        if any(actual[k] != selected[0][k] for k in ("bytes", "sha256")):
            raise ValueError(f"Retained checkpoint differs at {step}")
        checkpoints[str(step)] = actual
    curves, metrics = {}, {}
    for step in STEPS:
        curves[str(step)], metrics[str(step)] = {}, {}
        for role in ROLES:
            selected = [r for r in report["evaluations"] if r["update"] == step and r["role"] == role]
            if len(selected) != 1:
                raise ValueError(f"Need one evaluation at {step}/{role}")
            metric = selected[0]
            if augmented and metric.get("route") != "backbone_only":
                raise ValueError("Only backbone predictions belong in accuracy comparisons")
            expected_rows = 4096 if step == 1000 and not augmented else 102400
            if metric["rows"] != expected_rows:
                raise ValueError(f"Expected {expected_rows} evaluation words at {step}/{role}")
            curves[str(step)][role] = metric_rows(metric, arm)
            metrics[str(step)][role] = metric
    for metric in report["evaluations"]:
        if metric.get("role") not in ROLES or (augmented and metric.get("route") != "backbone_only"):
            raise ValueError("Unexpected evaluation role or inference route")
    if augmented:
        for diagnostic in report["one_step_diagnostics"]:
            if diagnostic.get("role") not in ROLES or diagnostic.get("route") != "teacher_conditioned_one_step_diagnostics":
                raise ValueError("Unexpected auxiliary diagnostic route")
    return {"report": report, "history": history, "curves": curves, "metrics": metrics,
            "checkpoints": checkpoints, "input_files": {"report": report_file, "history": history_file,
             "run_config": run_config_file, "data_manifest": manifest_file}}


def training_curve(history, augmented, window=100):
    result, elapsed = [], 0.0
    for start in range(0, len(history), window):
        rows = history[start:start + window]
        elapsed += sum(r["seconds"] for r in rows)
        point = {"first_update": rows[0]["update"], "update": rows[-1]["update"],
                 "updates_in_bin": len(rows), "training_seconds": elapsed,
                 "state_ce": sum(r["state_loss"] if augmented else r["loss"] for r in rows) / len(rows),
                 "token_accuracy": sum(r["token_accuracy"] for r in rows) / len(rows)}
        if augmented:
            for key in ("loss", "latent_loss", "weighted_latent_loss", *DIAGNOSTICS):
                values = [r[key] for r in rows if key in r]
                if values:
                    point[key] = sum(values) / len(values)
        result.append(point)
    return result


def normalized_contract(contract):
    value = dict(contract)
    for key in ("architecture", "schema", "source_sha256", "optimizer", "predictor_seed",
                "nextlat_config", "objective", "evaluation_route", "latent_rollout_evaluated"):
        value.pop(key, None)
    value["model_config"] = dict(value["model_config"])
    value["model_config"].pop("block_type")
    return value


def compare_runs(directories):
    arms = {name: read_training(directories[name], name) for name in ARMS}
    reference = arms["rt"]["report"]
    shared_contract = normalized_contract(reference["contract"])
    canonical = reference["initialization"]["canonical_sha256"]
    if not _sha(canonical):
        raise ValueError("Invalid canonical initialization SHA256")
    historical_sources = reference["source_files"]
    for name, arm in arms.items():
        report = arm["report"]
        augmented = name.endswith("_nextlat")
        if normalized_contract(report["contract"]) != shared_contract:
            raise ValueError(f"{name}: shared data/model/seed/runtime/optimization contracts differ")
        if report["initialization"]["canonical_sha256"] != canonical:
            raise ValueError("Canonical backbone initialization differs")
        expected_sources = set(historical_sources) | (NEW_SOURCES if augmented else set())
        if set(report["source_files"]) != expected_sources:
            raise ValueError("Unexpected source dependency additions or omissions")
        if any(report["source_files"][k] != v for k, v in historical_sources.items()):
            raise ValueError("Shared historical execution source changed")
        if [r["order_chain"] for r in arm["history"]] != [r["order_chain"] for r in arms["rt"]["history"]]:
            raise ValueError("Per-update training data order differs")
        if augmented:
            old = arms[name.split("_")[0]]["report"]
            initial = report["initialization"]
            if initial["backbone"] != old["initialization"]:
                raise ValueError("Paired original backbone initialization metadata differs")
            if (initial["backbone_parameter_count"] != 6357504
                    or initial["predictor_parameter_count"] != 1049600
                    or initial["parameter_count"] != 7407104
                    or initial["predictor_config"] != report["contract"]["nextlat_config"]["predictor"]):
                raise ValueError("Unexpected NextLat parameter count or predictor initialization")
        elif report["initialization"]["parameter_count"] != 6357504:
            raise ValueError("Unexpected pure backbone parameter count")
    left, right = (arms[name]["report"] for name in ("rt_nextlat", "seq_nextlat"))
    for key in ("nextlat_config", "objective", "source_sha256", "predictor_seed"):
        if left["contract"][key] != right["contract"][key]:
            raise ValueError(f"NextLat arms differ in {key}")
    if (not _sha(left["initialization"]["predictor_sha256"])
            or left["initialization"]["predictor_sha256"] != right["initialization"]["predictor_sha256"]):
        raise ValueError("Predictor initial weights differ")
    return {
        "schema": SCHEMA, "primary_update": 10000, "diagnostic_updates": [1000, 5000],
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
        "evaluation_route": "backbone_only", "shared_contract": shared_contract,
        "canonical_backbone_initialization_sha256": canonical,
        "predictor_initialization_sha256": left["initialization"]["predictor_sha256"],
        "order_chain": reference["order_chain"], "shared_historical_sources": historical_sources,
        "scope": "One paired development seed at fixed 10000 updates, 2.5% of the 400000-update reference; backbone predictions only",
        "intervals": "Pointwise Wilson 95% over words for E/A; no simultaneous, paired-difference, M or training-seed uncertainty claim",
        "checkpoint_sample_sizes": {"1000": "Historical arms 4096 words; new arms 102400, from the same frozen development pools",
                                    "5000": "All arms use the same first 102400 words per role",
                                    "10000": "All arms use the same first 102400 words per role"},
        "arms": {name: {"label": LABELS[name], "contract": arm["report"]["contract"],
                        "initialization": arm["report"]["initialization"],
                        "input_files": arm["input_files"], "checkpoints": arm["checkpoints"],
                        "source_files": arm["report"]["source_files"],
                        "training_wandb": arm["report"].get("wandb"),
                        "train_seconds": arm["report"]["train_seconds"],
                        "elapsed_seconds": arm["report"]["elapsed_seconds"],
                        "curves": arm["curves"], "checkpoint_metrics": arm["metrics"],
                        "endpoint_metrics": arm["metrics"]["10000"],
                        "training_curve": training_curve(arm["history"], name.endswith("_nextlat")),
                        "one_step_diagnostics": arm["report"].get("one_step_diagnostics", [])}
                 for name, arm in arms.items()},
    }


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
    for name, limits in (("length-full", (1, 36)), ("length-boundary", (10, 18))):
        figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
        for axis, metric, title in zip(axes, ("E", "A", "M"),
                ("All states through t correct: E(t)", "State at t correct: A(t)", "Mean token accuracy through t: M(t)")):
            for arm in ARMS:
                rows = summary["arms"][arm]["curves"]["10000"]["ood_dev"]
                x = [r["length"] for r in rows]
                axis.plot(x, [r[metric] for r in rows], color=COLORS[arm], label=LABELS[arm],
                          linewidth=1.8, linestyle="-" if arm.endswith("_nextlat") else "--")
                if metric in ("E", "A"):
                    axis.fill_between(x, [r[metric + "_low95"] for r in rows],
                                      [r[metric + "_high95"] for r in rows], color=COLORS[arm], alpha=.12)
            axis.axvline(12, color="#777777", linewidth=1, linestyle=":")
            if metric == "A":
                axis.axhline(1 / 60, color="#777777", linewidth=.8, linestyle=":")
            axis.set(xlabel="Operation position / prefix length", title=title, ylim=(-.025, 1.025), xlim=limits)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.suptitle("A5 backbone predictions · 10,000 updates · 102,400 OOD development words · one paired seed\n"
                       "NextLat predictor excluded from inference; shaded pointwise 95% intervals for E/A")
        save(figure, name)
    figure, axes = plt.subplots(1, 2, figsize=(11.7, 4.3), constrained_layout=True)
    for axis, xkey, xlabel in zip(axes, ("update", "training_seconds"),
                                ("Optimizer updates", "Accumulated training-loop minutes")):
        for arm in ARMS:
            rows = summary["arms"][arm]["training_curve"]
            scale = 60 if xkey == "training_seconds" else 1
            axis.plot([r[xkey] / scale for r in rows], [r["state_ce"] for r in rows],
                      label=LABELS[arm], color=COLORS[arm], linewidth=1.8)
        axis.set(xlabel=xlabel, ylabel="Training state CE (100-update means)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Compare task CE across all arms; auxiliary loss is separate\n"
                   "Training-loop time excludes evaluation, checkpointing and W&B logging")
    save(figure, "task-ce-learning")
    figure, axes = plt.subplots(3, 3, figsize=(13.8, 10.2), constrained_layout=True)
    diagnostic_keys = ("latent_loss", "loss", *DIAGNOSTICS)
    for axis, key in zip(axes.flat, diagnostic_keys):
        for arm in ("rt_nextlat", "seq_nextlat"):
            rows = [r for r in summary["arms"][arm]["training_curve"] if key in r]
            axis.plot([r["update"] for r in rows], [r[key] for r in rows],
                      label=LABELS[arm], color=COLORS[arm], linewidth=1.6)
        axis.set(xlabel="Optimizer updates", ylabel=key.replace("_", " "))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Auxiliary training diagnostics · one-step predictions conditioned on true backbone latents\n"
                   "100-update bins; sampled diagnostics average the recorded observations in each bin")
    save(figure, "auxiliary-diagnostics")
    return figures


def markdown_report(summary):
    lines = ["# A5: NextLat training with recurrent and ordinary Transformers", "",
             "**All accuracy results use the backbone alone.** Each new model was trained jointly with a "
             "one-step NextLat predictor; that predictor is excluded from inference. No latent-RNN rollout was run.", "",
             "Four matched 10,000-update runs: two blocks, D512/H8/GELU-FFN2048, full FP32 eager execution and ALiBi. "
             "Corresponding backbone initialization and every training minibatch are identical. "
             "Both RT blocks are recurrent at rho=1. Each backbone has 6,357,504 parameters; NextLat adds "
             "1,049,600 training parameters, for 7,407,104 total. The new predictor initialization is also paired.", "",
             "The objective is ordinary same-position state CE plus weight-one SmoothL1 over 11 latent transitions per "
             "length-12 word. Targets alone are detached; the source latent and next-operation embedding receive gradients. "
             "Auxiliary KL and predicted-state CE have zero training weight. The predictor follows the released A5 "
             "width-512 implementation; the paper's Table 5 lists width 1,024. Runtime remains our baseline FP32/eager "
             "policy rather than the release's BF16/compile settings.", "",
             "| Development set | Model | State CE | Mean token accuracy | Whole-word exactness | Final-state accuracy |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for role in ROLES:
        for arm in ARMS:
            r = summary["arms"][arm]["endpoint_metrics"][role]
            lines.append(f"| {role}, L{r['length']} | {LABELS[arm]} | {r['ce']:.6f} | "
                         f"{100*r['token_accuracy']:.4f}% | {100*r['whole_word_exact_match']:.4f}% | "
                         f"{100*r['final_state_accuracy']:.4f}% |")
    lines += ["", "Each endpoint evaluation uses the same first 102,400 frozen words for its development role.", "",
              "**E(t)** means every state through t is correct; **A(t)** means only the state at t is correct; "
              "**M(t)** means mean token accuracy through t. The following values come from length-36 outputs. "
              "They are not new independent evaluations at every shorter length.", "",
              "| Model | E(12) | E(13) | E(14) | E(16) | E(36) | A(36) | M(36) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        rows = summary["arms"][arm]["curves"]["10000"]["ood_dev"]
        values = [rows[t - 1]["E"] for t in (12, 13, 14, 16, 36)] + [rows[-1]["A"], rows[-1]["M"]]
        lines.append(f"| {LABELS[arm]} | " + " | ".join(f"{100*v:.4f}%" for v in values) + " |")
    lines += ["", "| Comparison at 10k | Δ E(12) | Δ E(13) | Δ E(14) | Δ E(16) | Δ M(36) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    comparisons = (("rt", "rt_nextlat", "RT + NextLat minus RT"),
                   ("seq", "seq_nextlat", "Transformer + NextLat minus Transformer"),
                   ("seq_nextlat", "rt_nextlat", "RT + NextLat minus Transformer + NextLat"))
    for baseline_name, augmented_name, label in comparisons:
        baseline = summary["arms"][baseline_name]["curves"]["10000"]["ood_dev"]
        augmented = summary["arms"][augmented_name]["curves"]["10000"]["ood_dev"]
        differences = [augmented[t-1]["E"] - baseline[t-1]["E"] for t in (12, 13, 14, 16)]
        differences += [augmented[-1]["M"] - baseline[-1]["M"]]
        lines.append(f"| {label} | "
                     + " | ".join(f"{100*v:+.4f} pp" for v in differences) + " |")
    lines += ["", "![All length curves](length-full.png)", "", "![Training-length boundary](length-boundary.png)", "",
              "The shaded bands are pointwise Wilson 95% intervals over words for E and A. They are not "
              "training-seed uncertainty or confidence intervals for paired differences. Observed zero exactness "
              "does not establish zero population success. M has no interval that assumes independent positions.", "",
              "![Task learning curves](task-ce-learning.png)", "", "![Auxiliary diagnostics](auxiliary-diagnostics.png)", "",
              "Task CE is plotted separately from the combined training objective. Auxiliary metrics are one-step "
              "training diagnostics, not another model-evaluation route. Increasing or decreasing latent regression "
              "loss alone does not establish improvement or failure; the actual backbone task results are primary.", "",
              "| Model | Training-loop minutes | Complete run minutes |",
              "| --- | ---: | ---: |"]
    for arm in ARMS:
        r = summary["arms"][arm]
        lines.append(f"| {LABELS[arm]} | {r['train_seconds']/60:.2f} | {r['elapsed_seconds']/60:.2f} |")
    lines += ["", "Training-loop time includes per-update diagnostics but excludes evaluation, checkpointing and W&B logging. "
              "Complete run time includes those costs. These are measured elapsed times, not hardware-normalized benchmarks.", "",
              "The CSV includes 1k, 5k and 10k checkpoints. At 1k, historical RT/Transformer evaluations used 4,096 words "
              "and the new models used 102,400 from the same frozen pools; that comparison has different sample sizes. "
              "At 5k and 10k all four arms use the same 102,400 words. Every row carries its own denominator. "
              "The endpoint was fixed at 10k; intermediate checkpoints are diagnostics, not selected replacements.", "",
              "This is one paired development seed and 2.5% of the paper's 400,000-update reference budget. "
              "It does not establish convergence, multi-seed reliability, paper replication or a general RT×NextLat interaction. "
              "The prior pure-RT 100k continuation is a different training budget and is not included as an equal-budget arm. "
              "Final confirmation remains unevaluated.", "",
              "Verification checks complete histories, checkpoint hashes, actual source snapshots, dataset-manifest identity, "
              "shared historical execution sources, model/runtime contracts, backbone/predictor initialization and every "
              "minibatch order hash. It also recovers integer correctness counts and checks all curve/scalar identities.", "",
              "[Metrics CSV](metrics.csv) · [Plot data](plot-data.json) · [Machine-readable report](report.json)", ""]
    return "\n".join(lines)


def run(args):
    output = Path(args.output_dir).resolve()
    mirror = Path(args.mirror_dir).resolve() if getattr(args, "mirror_dir", None) else None
    if output.exists() or (mirror is not None and mirror.exists()):
        raise FileExistsError("Report and optional mirror require fresh output directories")
    directories = {"rt": args.rt_dir, "seq": args.seq_dir,
                   "rt_nextlat": args.rt_nextlat_dir, "seq_nextlat": args.seq_nextlat_dir}
    summary = compare_runs(directories)
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for name, directory in directories.items():
        shutil.copyfile(Path(directory) / "report.json", output / "inputs" / f"{name}-training-report.json")
    summary["reporting_sources"] = {}
    for relative in ("scripts/rt_a5_nextlat_report.py", "scripts/rt_a5_report.py",
                     "scripts/rt_a5_length_report.py", "scripts/experiment_tracking.py"):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        summary["reporting_sources"][relative] = hash_file(destination)
    rows = [row for arm in ARMS for step in map(str, STEPS) for role in ROLES
            for row in summary["arms"][arm]["curves"][step][role]]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "plot-data.json", summary)
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    result = {**summary, "status": "running", "figures": figures}
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name="nextlat-backbone-comparison-step10000")
    try:
        tracker.start({k: summary[k] for k in ("schema", "primary_update", "shared_contract", "scope", "evaluation_route")})
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
        result["artifacts"] = {str(p.relative_to(output)): hash_file(p)
                               for p in sorted(output.rglob("*")) if p.is_file()
                               and "wandb" not in p.relative_to(output).parts and p.name != "report.json"}
        write_json(output / "report.json", result)
    if mirror is not None:
        shutil.copytree(output, mirror, ignore=shutil.ignore_patterns("wandb"))
    print(json.dumps({"status": result["status"], "output_dir": str(output),
                      "mirror_dir": str(mirror) if mirror else None, "wandb": tracker.record["run_url"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rt-dir", required=True)
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--rt-nextlat-dir", required=True)
    parser.add_argument("--seq-nextlat-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mirror-dir")
    parser.add_argument("--wandb-group", default="20260911T191702Z-nextlat")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
