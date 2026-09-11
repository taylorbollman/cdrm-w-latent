#!/usr/bin/env python3
"""CPU-only, matched-50k report of replacing ALiBi with RoPE in our SEQ.

Reads saved reports and checkpoints as opaque files. No model is imported or
executed, and the completed 100k architecture comparison remains unchanged.
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
from scripts.rt_a5_budget_report import evaluation_rows
from scripts.rt_a5_length_report import COLUMNS, horizons
from scripts.rt_a5_reference_report import (
    COMMON_FIELDS, NEW_SOURCES as REFERENCE_SOURCES, assemble_arm,
    dictionary_sha, local_path, retained_checkpoint, valid_sha,
)
from scripts.rt_a5_report import finite_number, hash_file, read_input, training_bins, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-rope-comparison-v1"
TRAIN_SCHEMA = "rt-a5-rope-training-v1"
ENDPOINT = 50000
STEPS = (10000, 25000, 50000)
ARMS = ("seq", "seq_rope", "reference_gpt", "rt")
PRIMARY_ARMS = ("seq", "seq_rope")
NEW_SOURCES = {"scripts/rt_a5_rope.py", "scripts/rt_a5_rope_train.py",
               "configs/rt_a5_rope/base.json"}
LABELS = {"seq": "Our Transformer (ALiBi)", "seq_rope": "Our Transformer (RoPE)",
          "reference_gpt": "Authors' GPT architecture (context)", "rt": "Our RT (context)"}
COLORS = {"seq": "#3465A4", "seq_rope": "#8449A8", "reference_gpt": "#24938C", "rt": "#B85031"}
CSV_FIELDS = ("arm", "comparison_role", "role", *COLUMNS, "state_ce")
DIFF_FIELDS = ("update", "role", "length", "words", "E_seq", "E_seq_rope", "E_delta_pp",
               "A_seq", "A_seq_rope", "A_delta_pp", "M_seq", "M_seq_rope", "M_delta_pp")


def role_of(arm):
    return "primary" if arm in PRIMARY_ARMS else "context"


def read_rope_part(directory):
    """Verify the new completed 0->50k job without loading a checkpoint."""
    directory = Path(directory).resolve()
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    if report.get("schema") != TRAIN_SCHEMA or report.get("status") != "complete":
        raise ValueError("RoPE requires a completed rt-a5-rope-training-v1 report")
    if (report.get("start_update"), report.get("completed_updates"), report.get("endpoint")) != (0, ENDPOINT, ENDPOINT):
        raise ValueError("RoPE requires the exact 0->50000 update window")
    if report.get("parent_checkpoint") is not None:
        raise ValueError("RoPE must start from its paired initialization, not a trained checkpoint")
    if report.get("confirmation_evaluated") is not False:
        raise ValueError("Final confirmation must remain unevaluated")
    contract = report["contract"]
    for key, expected in {"schema": TRAIN_SCHEMA, "architecture": "seq_rope",
                          "training_step": "scripts.rt_a5_train.train_step (same function object)",
                          "evaluation": "scripts.rt_a5_train.evaluate_arrays (same function object)",
                          "objective": "scripts.rt_a5_common.task_loss: unshifted CE mean over B*T"}.items():
        if contract.get(key) != expected:
            raise ValueError(f"RoPE has an unexpected contract field: {key}")
    if not valid_sha(report["initialization"]["canonical_sha256"]):
        raise ValueError("Invalid RoPE initialization hash")
    sources = report["source_files"]
    if dictionary_sha(sources) != contract["source_sha256"]:
        raise ValueError("RoPE source manifest differs from the execution contract")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not valid_sha(expected):
            raise ValueError("Invalid RoPE source snapshot path/hash")
        if hash_file(directory / "source" / path)["sha256"] != expected:
            raise ValueError(f"RoPE source snapshot changed: {relative}")
    raw, config_file = read_input(directory / "config.json")
    args = json.loads(raw)
    data_manifest = hash_file(local_path(args["data_dir"]) / "manifest.json")
    if data_manifest["sha256"] != contract["data_manifest_sha256"]:
        raise ValueError("RoPE data manifest differs from the execution contract")
    raw, history_file = read_input(directory / "history.jsonl")
    history = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if [row["update"] for row in history] != list(range(1, ENDPOINT + 1)):
        raise ValueError("RoPE history must contain each update 1..50000 exactly once")
    for row in history:
        if row["examples_seen"] != row["update"] * 1024 or not valid_sha(row["order_chain"]):
            raise ValueError("Invalid RoPE exposure or data-order identity")
        for key in ("seconds", "loss", "token_accuracy", "whole_word_exact", "grad_norm"):
            if not finite_number(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid RoPE training metric {key}")
        if row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError("Invalid RoPE training accuracy")
    if history[-1]["order_chain"] != report["order_chain"]:
        raise ValueError("RoPE history endpoint order differs from the report")
    seconds = sum(row["seconds"] for row in history)
    if not finite_number(report["train_seconds"]) or not math.isclose(seconds, report["train_seconds"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("RoPE training duration disagrees with history")
    if not finite_number(report["elapsed_seconds"]) or report["elapsed_seconds"] < seconds:
        raise ValueError("RoPE elapsed duration must include training time")
    return {"directory": directory, "report": report, "history": history,
            "inputs": {"report": report_file, "config": config_file,
                       "history": history_file, "data_manifest": data_manifest}}


def assemble_rope(directory):
    part = read_rope_part(directory)
    report, history = part["report"], part["history"]
    checkpoints = {str(step): retained_checkpoint(part, step) for step in (0, *STEPS)}
    curves, metrics = {}, {}
    for step in STEPS:
        curves[str(step)], metrics[str(step)] = {}, {}
        for role in ("dev", "ood_dev"):
            matches = [m for m in report["evaluations"] if m["update"] == step and m["role"] == role]
            if len(matches) != 1:
                raise ValueError(f"Expected one full RoPE evaluation at {step}/{role}")
            metric = matches[0]
            metrics[str(step)][role] = metric
            curves[str(step)][role] = [{"arm": "seq_rope", **row} for row in evaluation_rows(metric)]
    return {"parts": [part], "history": history, "record": {
        "label": LABELS["seq_rope"], "contract": report["contract"],
        "initialization": report["initialization"], "source_files": report["source_files"],
        "parent_checkpoint": None, "order_chain": report["order_chain"],
        "input_files": [part["inputs"]], "checkpoints": checkpoints,
        "checkpoint_metrics": metrics, "curves": curves,
        "training_wandb": [report.get("wandb")]}}


def select_50k(item, arm):
    """Project longer completed histories onto saved checkpoints through 50k.

Raw input reports retain their true 100k source endpoint. No 100k metric,
training duration or data-order endpoint enters the matched comparison.
"""
    history = [row for row in item["history"] if row["update"] <= ENDPOINT]
    if [row["update"] for row in history] != list(range(1, ENDPOINT + 1)):
        raise ValueError("Selected history must contain exactly updates 1..50000")
    record = copy.deepcopy(item["record"])
    for key in ("curves", "checkpoint_metrics"):
        record[key] = {str(step): record[key][str(step)] for step in STEPS}
    record["checkpoints"] = {str(step): record["checkpoints"][str(step)] for step in (0, *STEPS)}
    record["horizons"] = {step: horizons(values["ood_dev"]) for step, values in record["curves"].items()}
    record["label"], record["comparison_role"] = LABELS[arm], role_of(arm)
    record["order_chain"] = history[-1]["order_chain"]
    record["training_curve"] = training_bins(history, 100)
    record["selected_checkpoint_update"] = ENDPOINT
    record["source_completed_updates"] = item["parts"][-1]["report"]["completed_updates"]
    record["history_selection"] = {"first_update": 1, "last_update": ENDPOINT, "rows": len(history)}
    record["timing"] = {
        "training_seconds_through_50k": sum(row["seconds"] for row in history),
        "scope": "Sum of update durations 1..50000; excludes evaluation, checkpointing, logging, and later updates. No 100k job duration is attributed to 50k.",
    }
    return {"parts": item["parts"], "history": history, "record": record}


def primary_differences(records):
    rows = []
    for step in STEPS:
        for role in ("dev", "ood_dev"):
            left = records["seq"]["curves"][str(step)][role]
            right = records["seq_rope"]["curves"][str(step)][role]
            for a, b in zip(left, right):
                if (a["length"], a["words"]) != (b["length"], b["words"]):
                    raise ValueError("Primary metric positions or denominators differ")
                row = {"update": step, "role": role, "length": a["length"], "words": a["words"]}
                for metric in ("E", "A", "M"):
                    row.update({metric+"_seq": a[metric], metric+"_seq_rope": b[metric],
                                metric+"_delta_pp": 100*(b[metric]-a[metric])})
                rows.append(row)
    return rows


def make_summary(args):
    loaded = {
        "seq": select_50k(assemble_arm("seq", args.seq_pilot_dir, args.seq_dir), "seq"),
        "seq_rope": select_50k(assemble_rope(args.rope_dir), "seq_rope"),
        "reference_gpt": select_50k(assemble_arm("reference_gpt", None, args.reference_dir), "reference_gpt"),
        "rt": select_50k(assemble_arm("rt", args.rt_pilot_dir, args.rt_dir), "rt"),
    }
    seq = loaded["seq"]["record"]
    historical = seq["source_files"]
    common = {key: seq["contract"][key] for key in COMMON_FIELDS}
    order = [row["order_chain"] for row in loaded["seq"]["history"]]
    for arm, item in loaded.items():
        contract, sources = item["record"]["contract"], item["record"]["source_files"]
        if {key: contract[key] for key in COMMON_FIELDS} != common:
            raise ValueError(f"{arm}: shared data, optimizer, seed or runtime differs")
        if [row["order_chain"] for row in item["history"]] != order:
            raise ValueError(f"{arm}: per-update training data order differs before 50k")
        additions = NEW_SOURCES if arm == "seq_rope" else REFERENCE_SOURCES if arm == "reference_gpt" else set()
        if set(sources) != set(historical) | additions:
            raise ValueError(f"{arm}: unexpected source additions or omissions")
        if any(sources[path] != value for path, value in historical.items()):
            raise ValueError(f"{arm}: historical shared execution sources changed")
    rope = loaded["seq_rope"]["record"]
    initial = rope["initialization"]
    expected_delta = {"alibi": {"from": True, "to": False}, "rope": {"from": False, "to": True}}
    if (initial.get("schema") != "rt-a5-seq-rope-initialization-v1"
            or initial.get("architecture") != "seq_rope"
            or initial.get("paired_base_architecture") != "seq"
            or initial.get("paired_base_canonical_sha256") != seq["initialization"]["canonical_sha256"]
            or initial.get("paired_base_initialization") != seq["initialization"]
            or initial.get("config_delta") != expected_delta):
        raise ValueError("RoPE initialization metadata does not preserve the original SEQ lineage")
    expected_config = {**seq["contract"]["model_config"], "alibi": False, "rope": True}
    if rope["contract"]["model_config"] != expected_config:
        raise ValueError("Primary architectures must differ only by alibi=False and rope=True")
    for arm in ("seq_rope", "rt"):
        initial = loaded[arm]["record"]["initialization"]
        if any(initial[key] != seq["initialization"][key] for key in ("canonical_sha256", "parameter_count")):
            raise ValueError(f"{arm}: paired initialization or parameter count differs")
    records = {arm: loaded[arm]["record"] for arm in ARMS}
    return {
        "schema": SCHEMA, "status": "verified", "primary_update": ENDPOINT,
        "checkpoint_updates": list(STEPS), "primary_arms": list(PRIMARY_ARMS),
        "context_arms": ["reference_gpt", "rt"], "evaluation_route": "backbone_only",
        "confirmation_evaluated": False, "nextlat_enabled": False,
        "common_contract": common, "historical_source_files": historical,
        "canonical_primary_initialization_sha256": seq["initialization"]["canonical_sha256"],
        "order_chain": order[-1],
        "primary_model_config_delta": {"alibi": {"seq": True, "seq_rope": False},
                                       "rope": {"seq": False, "seq_rope": True}},
        "budget": {"total_updates_per_arm": ENDPOINT, "reference_paper_updates": 400000,
                   "fraction_of_reference_budget": .125, "word_presentations_per_arm": 51200000,
                   "training_token_presentations_per_arm": 614400000,
                   "unique_training_words": 800000, "nominal_training_passes": 64,
                   "seq_rope_new_updates": ENDPOINT},
        "scope": "One paired development seed at fixed 50k: change only ALiBi to RoPE in our SEQ; reference GPT and RT are context at the same 50k checkpoint. No comparison to SEQ100k is used for the positional-encoding claim.",
        "initialization_qualification": "Primary SEQ/SEQ+RoPE have the same recorded canonical initialization and parameter count. Reference GPT has its own architecture and initialization; its common seed does not imply paired weights.",
        "intervals": "Pointwise Wilson 95% over words for E/A; no M interval. Primary percentage-point differences are descriptive: marginal counts do not supply paired-difference or training-seed confidence intervals.",
        "primary_differences": primary_differences(records), "arms": records,
    }


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}

    def style(arm):
        return {"color": COLORS[arm], "linewidth": 2.0 if arm in PRIMARY_ARMS else 1.5,
                "linestyle": "-" if arm in PRIMARY_ARMS else "--",
                "alpha": 1.0 if arm in PRIMARY_ARMS else .8, "label": LABELS[arm]}

    def save(fig, name):
        figures[name] = {}
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=190)
            figures[name][suffix] = filename
        plt.close(fig)

    titles = ("Every state through t correct: E(t)", "State at t correct: A(t)", "Mean token accuracy through t: M(t)")
    for name, limits in (("length-full", (1, 36)), ("length-zoom-10-18", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
        for axis, metric, title in zip(axes, ("E", "A", "M"), titles):
            for arm in ARMS:
                rows = summary["arms"][arm]["curves"]["50000"]["ood_dev"]
                x = [row["length"] for row in rows]
                axis.plot(x, [row[metric] for row in rows], **style(arm))
                if metric in ("E", "A") and arm in PRIMARY_ARMS:
                    axis.fill_between(x, [r[metric+"_low95"] for r in rows], [r[metric+"_high95"] for r in rows], color=COLORS[arm], alpha=.12)
            axis.axvline(12, color=".45", linestyle=":", linewidth=1)
            if metric == "A":
                axis.axhline(1/60, color=".45", linestyle=":", linewidth=1)
            axis.set(title=title, xlabel="Operation position / prefix length t", xlim=limits, ylim=(-.025, 1.025))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
            axis.grid(alpha=.2); axis.legend(fontsize=7.4)
        fig.suptitle("ALiBi → RoPE in our Transformer · every curve at 50k · same 102,400 OOD development words\n" +
                     ("Full range 1–36; solid lines are the primary pair, dashed lines are context" if limits[0] == 1 else "Explicit crop 10–18 of the same full curves; earlier failures are outside this view"))
        save(fig, name)
    fig, axis = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    for arm in ARMS:
        rows = summary["arms"][arm]["curves"]["50000"]["ood_dev"]
        axis.plot([r["length"] for r in rows], [r["E"] for r in rows], **style(arm))
    axis.axvline(12, color=".45", linestyle=":", label="Training length 12")
    axis.set(xlabel="Prefix length t", ylabel="Every state through t correct: E(t)", xlim=(1,36), ylim=(-.025,1.025))
    axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend(fontsize=9)
    axis.set_title("Figure 10-style cumulative accuracy · every curve at 50k\nE(t) only; solid lines are the paired ALiBi/RoPE comparison")
    save(fig, "figure10-style-prefix-exactness")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, metric, title in zip(axes, ("loss", "token_accuracy"), ("Training state CE", "Training token accuracy")):
        for arm in ARMS:
            rows = summary["arms"][arm]["training_curve"]
            axis.plot([r["update"]/1000 for r in rows], [r[metric] for r in rows], **style(arm))
        axis.set(xlabel="Optimizer updates (thousands)", ylabel=title, xlim=(0,50))
        axis.grid(alpha=.2); axis.legend(fontsize=8)
        if metric == "token_accuracy":
            axis.set_ylim(-.025,1.025); axis.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Same-budget learning curves through 50k · nonoverlapping 100-update means\nLater updates from the completed 100k context jobs are excluded")
    save(fig, "learning-curves")
    return figures


def markdown_report(summary):
    lines = ["# A5: ALiBi versus RoPE at 50,000 updates", "",
             "**Primary comparison: our SEQ Transformer versus the same Transformer with ALiBi replaced by RoPE, both at 50,000 updates.** The recorded canonical initialization, parameter count, data order, optimizer, and runtime match. Every other model configuration field is identical.", "",
             "The primary pair retains two D512/H8/GELU-FFN2048 blocks with LayerNorm and learned QK normalization (6,357,504 parameters). The authors' GPT architecture and our RT are context at the same 50k budget. The GPT architecture changes multiple components and has its own initialization; it is not part of the isolated positional-encoding comparison.", "",
             "Each arm has 51.2 million word presentations, 614.4 million training-token presentations, and 64 nominal passes over 800,000 training words. This is 12.5% of the paper's 400k update budget. The earlier 10k/25k checkpoints are diagnostics. Although the source SEQ, GPT-reference and RT jobs completed 100k, this report selects their retained 50k checkpoints and uses no later metrics or training updates.", "",
             "| Backbone at 50k | Comparison role | Set | State CE | Mean token accuracy M | Whole-word exactness E | Final-state accuracy A |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for role in ("dev", "ood_dev"):
            m = summary["arms"][arm]["checkpoint_metrics"]["50000"][role]
            lines.append(f"| {LABELS[arm]} | {role_of(arm)} | {role}, L{m['length']} | {m['ce']:.7f} | {100*m['token_accuracy']:.4f}% | {100*m['whole_word_exact_match']:.4f}% | {100*m['final_state_accuracy']:.4f}% |")
    lines += ["", "All checkpoint evaluations use the same first 102,400 frozen words for each development role. E(t) requires every state through t to be correct; A(t) checks only position t; M(t) averages correctness through t. Short development words and long-word prefixes are different samples. Length curves use prefixes of saved length-36 outputs, not separate model evaluations at each length.", "",
              "![Full 1–36 curves](length-full.png)", "", "![Explicit 10–18 crop](length-zoom-10-18.png)", "",
              "The zoom uses exactly the full plot's data and panels; failures before position 10 are outside the crop. M(t) can remain high because earlier states were correct even when A(t) is near chance and E(t) is zero. Only A has a flat 1/60 state-guessing line. Solid lines are the primary pair; dashed lines provide context.", "",
              "![Cumulative exactness only](figure10-style-prefix-exactness.png)", "",
              "The Figure 10-style plot shows E(t), following the cumulative-correctness convention supported by the released A5 evaluator. These are our results at 50k, not digitized paper measurements or a claimed reproduction of its 400k run.", "",
              "| OOD prefix t, at 50k | SEQ E(t) | SEQ+RoPE E(t) | Δ E, percentage points | Δ A, percentage points | Δ M, percentage points |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["primary_differences"]:
        if row["update"] == ENDPOINT and row["role"] == "ood_dev" and row["length"] in (2,5,6,8,10,12,13,14,16,18,36):
            lines.append(f"| {row['length']} | {100*row['E_seq']:.4f}% | {100*row['E_seq_rope']:.4f}% | {row['E_delta_pp']:+.4f} | {row['A_delta_pp']:+.4f} | {row['M_delta_pp']:+.4f} |")
    lines += ["", "Differences are RoPE minus ALiBi at the same update. They are descriptive; saved marginal counts do not provide paired-difference confidence intervals. This is one matched training seed, not evidence of seed robustness.", "",
              "| Backbone | Updates | Last E≥95% | Last E≥50% | Last E≥10% | Last E≥1% | First zero-success length |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for step in STEPS:
            h = summary["arms"][arm]["horizons"][str(step)]
            values = [h["last_length_at_or_above"][key] for key in ("95%", "50%", "10%", "1%")]
            lines.append(f"| {LABELS[arm]} | {step:,} | " + " | ".join("None" if x is None else str(x) for x in values) + f" | {h['first_observed_zero_length']} |")
    lines += ["", "Thresholds are inclusive and bounded by positions 1–36. Zero means zero successes in this sample, not population impossibility. CSV intervals are pointwise Wilson 95% over words for E/A; M has no interval assuming independent positions.", "",
              "| Backbone | Updates | OOD E(12) | E(13) | E(14) | E(36) | A(36) | M(36) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for step in STEPS:
            rows = summary["arms"][arm]["curves"][str(step)]["ood_dev"]
            values = [rows[t-1]["E"] for t in (12,13,14,36)] + [rows[-1]["A"],rows[-1]["M"]]
            lines.append(f"| {LABELS[arm]} | {step:,} | " + " | ".join(f"{100*x:.4f}%" for x in values) + " |")
    lines += ["", "![Learning curves through 50k](learning-curves.png)", "",
              "| Backbone | Training-loop minutes for updates 1–50,000 |", "| --- | ---: |"]
    for arm in ARMS:
        lines.append(f"| {LABELS[arm]} | {summary['arms'][arm]['timing']['training_seconds_through_50k']/60:.2f} |")
    lines += ["", "Timing sums only updates 1–50,000, including the original pilot portions for SEQ/RT. It excludes evaluation, checkpointing and logging. Full 100k job elapsed time is not reported as 50k cost, and this is not a controlled hardware-normalized benchmark.", "",
              "Both primary attention-only models lack a BOS token or positional contribution to their values/residual. An initial repeated nonidentity operation `(g, g)` therefore gives identical representations in exact arithmetic while the required states `g` and `g²` differ. RoPE and ALiBi share this limitation; a learned-through-12 pattern need not reach literally 100% E(12). This structural statement does not attribute individual model errors or assume exact arithmetic in an FP32 run.", "",
              "This bounded comparison uses FP32 eager execution and the existing same-position A5 loss. It introduces no NextLat objective or recurrent latent inference. Final confirmation remains unevaluated. Stop after this 50k experiment before further work.", "",
              "[All counts and denominators](metrics.csv) · [Primary differences](primary-differences.csv) · [Plot data](plot-data.json) · [Machine-readable report](report.json)", ""]
    return "\n".join(lines)


def run(args):
    output = Path(args.output_dir).resolve()
    mirror = Path(args.mirror_dir).resolve() if args.mirror_dir else None
    if output.exists() or (mirror is not None and mirror.exists()):
        raise FileExistsError("Report and optional mirror must be fresh directories")
    summary = make_summary(args)
    output.mkdir(parents=True); (output / "inputs").mkdir()
    for arm, record in summary["arms"].items():
        for i, files in enumerate(record["input_files"]):
            shutil.copyfile(local_path(files["report"]["path"]), output / "inputs" / f"{arm}-job{i}-report.json")
    summary["reporting_sources"] = {}
    for relative in ("scripts/rt_a5_rope_report.py", "scripts/rt_a5_reference_report.py",
                     "scripts/rt_a5_budget_report.py", "scripts/rt_a5_length_report.py",
                     "scripts/rt_a5_report.py", "scripts/experiment_tracking.py"):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        summary["reporting_sources"][relative] = hash_file(destination)
    rows = [{"comparison_role": role_of(arm), **row} for arm in ARMS for step in STEPS
            for role in ("dev", "ood_dev") for row in summary["arms"][arm]["curves"][str(step)][role]]
    for name, fields, values in (("metrics.csv", CSV_FIELDS, rows),
                                 ("primary-differences.csv", DIFF_FIELDS, summary["primary_differences"])):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    write_json(output / "plot-data.json", summary)
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    result = {**summary, "status": "running", "figures": figures}
    tracker = None
    try:
        if not args.no_wandb:
            tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                                    group=args.wandb_group, name="rope-only-comparison-50k")
            tracker.start({key: summary[key] for key in ("schema", "scope", "budget", "common_contract", "primary_model_config_delta")})
            import wandb
            tracker.log({"report/metrics": wandb.Table(columns=list(CSV_FIELDS), data=[[r[k] for k in CSV_FIELDS] for r in rows]),
                         "report/primary_differences": wandb.Table(columns=list(DIFF_FIELDS), data=[[r[k] for k in DIFF_FIELDS] for r in summary["primary_differences"]]),
                         **{f"report/{name}": wandb.Image(str(output / files["png"])) for name, files in figures.items()}})
            tracker.summary({"primary_update": ENDPOINT, "confirmation_evaluated": False,
                             "scope": summary["scope"], "budget": summary["budget"]})
            tracker.finish(succeeded=True)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__)
        if tracker is not None:
            try:
                tracker.finish(succeeded=False)
            except Exception:
                pass
        raise
    finally:
        result["wandb"] = tracker.record if tracker is not None else {"enabled": False, "reason": "Explicit --no-wandb artifact-only invocation"}
        result["artifacts"] = {str(path.relative_to(output)): hash_file(path) for path in sorted(output.rglob("*"))
                               if path.is_file() and "wandb" not in path.relative_to(output).parts and path.name != "report.json"}
        write_json(output / "report.json", result)
    if mirror is not None:
        shutil.copytree(output, mirror, ignore=shutil.ignore_patterns("wandb"))
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": result["wandb"]}), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    pilot = ".runtime/rt-a5/20260911T154748Z"
    reference = ".runtime/rt-a5/20260911T201820Z-transformer-reference"
    p.add_argument("--seq-pilot-dir", default=pilot + "/train-seq")
    p.add_argument("--rt-pilot-dir", default=pilot + "/train-rt")
    p.add_argument("--seq-dir", default=reference + "/train-seq")
    p.add_argument("--reference-dir", default=reference + "/train-reference-gpt")
    p.add_argument("--rt-dir", default=".runtime/rt-a5/20260911T171239Z-rt100k/train-rt")
    p.add_argument("--rope-dir", default=".runtime/rt-a5/20260911T210811Z-rope-only/train-seq-rope")
    p.add_argument("--output-dir", default="docs/reports/rt-a5/rope-only-50k")
    p.add_argument("--mirror-dir")
    p.add_argument("--wandb-group", default="20260911T210811Z-rope-only")
    p.add_argument("--no-wandb", action="store_true", help="Explicit local-only report; normal graphable runs log online")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
