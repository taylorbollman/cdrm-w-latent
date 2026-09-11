#!/usr/bin/env python3
"""CPU-only comparison of fixed-100k SEQ, RT and released-GPT architectures.

Reads retained reports, hashes checkpoints as opaque files, and never imports
or executes a model. The reference is an architecture port on our harness,
not a reproduction of the paper's 400k training run.
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
from scripts.rt_a5_budget_report import evaluation_rows
from scripts.rt_a5_length_report import COLUMNS, horizons
from scripts.rt_a5_report import (
    checkpoint_path, finite_number, hash_file, read_input, training_bins, write_json,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-a5-reference-comparison-v1"
ENDPOINT = 100000
STEPS = (10000, 25000, 50000, 100000)
ARMS = ("seq", "reference_gpt", "rt")
LABELS = {"seq": "Our Transformer (ALiBi)",
          "reference_gpt": "Authors' GPT architecture (our harness)",
          "rt": "Our RT (ALiBi)"}
COLORS = {"seq": "#3465A4", "reference_gpt": "#24938C", "rt": "#B85031"}
COMMON_FIELDS = ("width", "seed", "data_order_seed", "batch_size", "train_rows",
                 "length", "data_manifest_sha256", "precision", "tf32", "compile",
                 "cuda_graphs", "optimizer", "torch", "cuda", "device_capability",
                 "word_order")
NEW_SOURCES = {"scripts/rt_a5_reference_gpt.py", "scripts/rt_a5_reference_train.py",
               "configs/rt_a5_reference/base.json"}
CSV_FIELDS = ("arm", "role", *COLUMNS, "state_ce")


def local_path(value):
    path = checkpoint_path(value)
    return path if path.is_absolute() else ROOT / path


def dictionary_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_part(directory, architecture, start, endpoint):
    """Verify one finished job, without assuming histories start at zero."""
    directory = Path(directory).resolve()
    raw, report_file = read_input(directory / "report.json")
    report = json.loads(raw)
    schema = ("rt-a5-reference-gpt-training-v1" if architecture == "reference_gpt"
              else "rt-a5-training-v1")
    if report.get("schema") != schema or report.get("status") != "complete":
        raise ValueError(f"{architecture}: expected a completed {schema} report")
    if (report.get("start_update"), report.get("completed_updates"), report.get("endpoint")) != (start, endpoint, endpoint):
        raise ValueError(f"{architecture}: expected exact update window {start}->{endpoint}")
    if report.get("confirmation_evaluated") is not False:
        raise ValueError("Final confirmation must remain unevaluated")
    contract = report["contract"]
    required = {"schema": schema, "architecture": architecture, "width": 512,
                "seed": 1234, "data_order_seed": 1234, "batch_size": 1024,
                "train_rows": 800000, "length": 12, "precision": "fp32",
                "tf32": False, "compile": False, "cuda_graphs": False,
                "optimizer": "AdamW-lr1e-4-betas0.9,0.95-eps1e-8-wd0.01-matrices-clip1"}
    for key, value in required.items():
        if contract.get(key) != value:
            raise ValueError(f"{architecture}: unexpected execution setting {key}")
    config = contract["model_config"]
    if architecture == "reference_gpt":
        model_settings = {"n_layer": 2, "n_embd": 512, "n_head": 8,
                          "mlp_hidden_width": 1408, "activation": "swiglu",
                          "normalization": "rms_norm", "qk_normalization": False,
                          "rope": True, "alibi": False, "weight_tying": False,
                          "vocab_size": 60, "initialization": "normal",
                          "initialization_std": .02, "recurrent": False, "nextlat": False}
        if (contract.get("training_step") != "scripts.rt_a5_train.train_step (same function object)"
                or contract.get("evaluation") != "scripts.rt_a5_train.evaluate_arrays (same function object)"):
            raise ValueError("Reference GPT must use the existing training/evaluation functions")
        count = 6486528
    else:
        model_settings = {"block_type": "recurrent" if architecture == "rt" else "sequential",
                          "n_layers": 2, "d_model": 512, "n_heads": 8,
                          "mlp_hidden_size": 2048, "activation_type": "gelu",
                          "alibi": True, "rope": False, "attention_layer_norm": True,
                          "cdrm_enabled": False, "recurrent_layers": None,
                          "recurrent_write_rho": 1.0, "weight_tying": False,
                          "vocab_size": 60, "reference_eager": True}
        count = 6357504
    if any(config.get(key) != value for key, value in model_settings.items()):
        raise ValueError(f"{architecture}: unexpected model architecture")
    if report["initialization"]["parameter_count"] != count:
        raise ValueError(f"{architecture}: parameter count differs from the frozen architecture")
    if not valid_sha(report["initialization"]["canonical_sha256"]):
        raise ValueError("Invalid recorded initialization hash")

    sources = report["source_files"]
    if dictionary_sha(sources) != contract["source_sha256"]:
        raise ValueError("Source manifest differs from the execution contract")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not valid_sha(expected):
            raise ValueError("Invalid source snapshot path/hash")
        if hash_file(directory / "source" / path)["sha256"] != expected:
            raise ValueError(f"Source snapshot changed: {relative}")
    raw, config_file = read_input(directory / "config.json")
    args = json.loads(raw)
    data_manifest = hash_file(local_path(args["data_dir"]) / "manifest.json")
    if data_manifest["sha256"] != contract["data_manifest_sha256"]:
        raise ValueError("Actual data manifest differs from the run")
    raw, history_file = read_input(directory / "history.jsonl")
    history = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if [row["update"] for row in history] != list(range(start + 1, endpoint + 1)):
        raise ValueError("Missing, duplicate or unordered history updates")
    for row in history:
        if row["examples_seen"] != row["update"] * 1024 or not valid_sha(row["order_chain"]):
            raise ValueError("Invalid training exposure or data-order identity")
        for key in ("seconds", "loss", "token_accuracy", "whole_word_exact", "grad_norm"):
            if not finite_number(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid training metric {key}")
        if row["token_accuracy"] > 1 or row["whole_word_exact"] > 1:
            raise ValueError("Invalid training accuracy")
    if history[-1]["order_chain"] != report["order_chain"]:
        raise ValueError("History endpoint order differs from report")
    seconds = sum(row["seconds"] for row in history)
    if not finite_number(report["train_seconds"]) or not math.isclose(seconds, report["train_seconds"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Training-loop duration disagrees with update history")
    if not finite_number(report["elapsed_seconds"]) or report["elapsed_seconds"] < seconds:
        raise ValueError("Elapsed duration must include training-loop duration")
    return {"directory": directory, "report": report, "history": history,
            "inputs": {"report": report_file, "config": config_file,
                       "history": history_file, "data_manifest": data_manifest}}


def retained_checkpoint(part, update):
    matches = [r for r in part["report"]["checkpoints"] if r["completed_updates"] == update]
    if len(matches) != 1:
        raise ValueError(f"Expected one retained checkpoint at {update}")
    recorded = matches[0]
    actual = hash_file(local_path(recorded["path"]))
    if any(actual[key] != recorded[key] for key in ("bytes", "sha256")):
        raise ValueError(f"Checkpoint hash or size changed at {update}")
    return actual


def assemble_arm(architecture, pilot_dir, continuation_dir):
    if architecture == "reference_gpt":
        parts = [read_part(continuation_dir, architecture, 0, ENDPOINT)]
        if parts[0]["report"].get("parent_checkpoint") is not None:
            raise ValueError("Reference architecture must start from its fresh initialization")
    else:
        parts = [read_part(pilot_dir, architecture, 0, 10000),
                 read_part(continuation_dir, architecture, 10000, ENDPOINT)]
        first, last = (p["report"] for p in parts)
        if first.get("parent_checkpoint") is not None or first["contract"] != last["contract"]:
            raise ValueError("Continuation must preserve the original complete pilot contract")
        if first["initialization"] != last["initialization"]:
            raise ValueError("Continuation initialization lineage differs")
        expected = retained_checkpoint(parts[0], 10000)
        parent = last.get("parent_checkpoint") or {}
        if parent.get("sha256") != expected["sha256"]:
            raise ValueError("Continuation parent differs from the original 10k checkpoint")
        actual = hash_file(local_path(parent["path"]))
        if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError("Continuation parent checkpoint file changed")
    history = [row for part in parts for row in part["history"]]
    if len(history) != ENDPOINT or any(row["update"] != i for i, row in enumerate(history, 1)):
        raise ValueError("Joined history must contain exactly updates 1..100000")
    retained = {"0": retained_checkpoint(parts[0], 0)}
    curves, metrics = {}, {}
    for update in STEPS:
        part = parts[0] if len(parts) == 1 or update == 10000 else parts[1]
        retained[str(update)] = retained_checkpoint(part, update)
        curves[str(update)], metrics[str(update)] = {}, {}
        for role in ("dev", "ood_dev"):
            matches = [m for m in part["report"]["evaluations"] if m["update"] == update and m["role"] == role]
            if len(matches) != 1:
                raise ValueError(f"Expected one full evaluation at {architecture}/{update}/{role}")
            metric = matches[0]
            metrics[str(update)][role] = metric
            curves[str(update)][role] = [{"arm": architecture, **r} for r in evaluation_rows(metric)]
    final = parts[-1]["report"]
    return {"parts": parts, "history": history, "record": {
        "label": LABELS[architecture], "contract": final["contract"],
        "initialization": final["initialization"], "source_files": final["source_files"],
        "parent_checkpoint": final.get("parent_checkpoint"), "order_chain": final["order_chain"],
        "input_files": [p["inputs"] for p in parts], "checkpoints": retained,
        "checkpoint_metrics": metrics, "curves": curves,
        "horizons": {step: horizons(values["ood_dev"]) for step, values in curves.items()},
        "training_curve": training_bins(history, 100),
        "timing": {"training_seconds": sum(p["report"]["train_seconds"] for p in parts),
                   "sum_job_elapsed_seconds": sum(p["report"]["elapsed_seconds"] for p in parts),
                   "jobs": [{"start_update": p["report"]["start_update"],
                             "end_update": p["report"]["completed_updates"],
                             "train_seconds": p["report"]["train_seconds"],
                             "elapsed_seconds": p["report"]["elapsed_seconds"]} for p in parts]},
        "training_wandb": [p["report"].get("wandb") for p in parts]}}


def make_summary(args):
    loaded = {"seq": assemble_arm("seq", args.seq_pilot_dir, args.seq_dir),
              "rt": assemble_arm("rt", args.rt_pilot_dir, args.rt_dir),
              "reference_gpt": assemble_arm("reference_gpt", None, args.reference_dir)}
    historical = loaded["seq"]["record"]["source_files"]
    common = {key: loaded["seq"]["record"]["contract"][key] for key in COMMON_FIELDS}
    reference_order = [r["order_chain"] for r in loaded["seq"]["history"]]
    for arm, item in loaded.items():
        contract = item["record"]["contract"]
        if {key: contract[key] for key in COMMON_FIELDS} != common:
            raise ValueError(f"{arm}: shared data, optimizer, seed or runtime differs")
        if [r["order_chain"] for r in item["history"]] != reference_order:
            raise ValueError(f"{arm}: per-update training data order differs")
        sources = item["record"]["source_files"]
        if set(sources) != set(historical) | (NEW_SOURCES if arm == "reference_gpt" else set()):
            raise ValueError("Unexpected execution-source additions or omissions")
        if any(sources[path] != value for path, value in historical.items()):
            raise ValueError("Historical shared execution sources changed")
    if loaded["seq"]["record"]["initialization"]["canonical_sha256"] != loaded["rt"]["record"]["initialization"]["canonical_sha256"]:
        raise ValueError("Original SEQ/RT canonical initialization differs")
    return {"schema": SCHEMA, "primary_update": ENDPOINT, "checkpoint_updates": list(STEPS),
            "status": "verified", "evaluation_route": "backbone_only",
            "confirmation_evaluated": False, "nextlat_enabled": False,
            "rope_only_experiment_performed": False, "common_contract": common,
            "historical_source_files": historical, "order_chain": reference_order[-1],
            "budget": {"total_updates_per_arm": ENDPOINT, "reference_paper_updates": 400000,
                       "fraction_of_reference_budget": .25, "word_presentations_per_arm": 102400000,
                       "training_token_presentations_per_arm": 1228800000,
                       "unique_training_words": 800000, "nominal_training_passes": 128,
                       "seq_new_updates": 90000, "reference_gpt_new_updates": 100000},
            "scope": "One development seed at fixed 100k; authors' architecture on our data/loss/optimizer/FP32 harness, not exact 400k paper reproduction; no RoPE-only ablation",
            "initialization_qualification": "Our SEQ/RT use the original paired Mitchell initialization. Reference GPT uses its own released Normal(0,.02) construction; common seed does not imply paired weights across architecture families.",
            "intervals": "Pointwise Wilson 95% over words for E/A; no simultaneous, seed or paired-difference uncertainty claim; no M interval",
            "arms": {arm: loaded[arm]["record"] for arm in ARMS}}


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}

    def save(figure, name):
        figures[name] = {}
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            figure.savefig(output / filename, dpi=190)
            figures[name][suffix] = filename
        plt.close(figure)

    titles = ("Every state through t correct: E(t)", "State at t correct: A(t)", "Mean token accuracy through t: M(t)")
    for name, limits in (("length-full", (1, 36)), ("length-zoom-10-18", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
        for axis, metric, title in zip(axes, ("E", "A", "M"), titles):
            for arm in ARMS:
                rows = summary["arms"][arm]["curves"]["100000"]["ood_dev"]
                x = [r["length"] for r in rows]
                axis.plot(x, [r[metric] for r in rows], label=LABELS[arm], color=COLORS[arm], linewidth=1.8)
                if metric in ("E", "A"):
                    axis.fill_between(x, [r[metric+"_low95"] for r in rows], [r[metric+"_high95"] for r in rows], color=COLORS[arm], alpha=.13)
            axis.axvline(12, color=".45", linestyle=":", linewidth=1)
            if metric == "A":
                axis.axhline(1/60, color=".45", linestyle=":", linewidth=1)
            axis.set(title=title, xlabel="Operation position / prefix length t", xlim=limits, ylim=(-.025, 1.025))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
            axis.grid(alpha=.2)
            axis.legend(fontsize=7.4)
        fig.suptitle("A5 backbones at fixed 100k · same 102,400 OOD development words · one seed\n" +
                     ("Full range 1–36; training boundary at 12" if limits[0] == 1 else "Explicit crop 10–18 of the same full curves; earlier failures are outside this view"))
        save(fig, name)
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for arm in ARMS:
        rows = summary["arms"][arm]["curves"]["100000"]["ood_dev"]
        axis.plot([r["length"] for r in rows], [r["E"] for r in rows], label=LABELS[arm], color=COLORS[arm], linewidth=2)
    axis.axvline(12, color=".45", linestyle=":", label="Training length 12")
    axis.set(xlabel="Prefix length t", ylabel="Every state through t correct: E(t)", xlim=(1,36), ylim=(-.025,1.025))
    axis.yaxis.set_major_formatter(PercentFormatter(1)); axis.grid(alpha=.2); axis.legend(fontsize=9)
    axis.set_title("Figure 10-style cumulative accuracy · our fixed 100k experiment\nE(t) only; our results, not a digitization or reproduction of the paper")
    save(fig, "figure10-style-prefix-exactness")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, metric, title in zip(axes, ("loss", "token_accuracy"), ("Training state CE", "Training token accuracy")):
        for arm in ARMS:
            rows = summary["arms"][arm]["training_curve"]
            axis.plot([r["update"]/1000 for r in rows], [r[metric] for r in rows], label=LABELS[arm], color=COLORS[arm])
        axis.set(xlabel="Optimizer updates (thousands)", ylabel=title); axis.grid(alpha=.2); axis.legend(fontsize=8)
        if metric == "token_accuracy":
            axis.set_ylim(-.025, 1.025); axis.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Complete joined histories · nonoverlapping 100-update means\nSEQ/RT pilot 0–10k joined to continuation 10–100k; reference trained 0–100k")
    save(fig, "learning-curves")
    return figures


def markdown_report(summary):
    lines = ["# A5: Transformer architecture comparison at 100,000 updates", "",
             "**Fixed primary endpoint: 100,000 updates for every backbone.** This is 102.4 million word presentations, 128 nominal passes over 800,000 training words, and 25% of the paper's 400k update budget. Earlier 10k/25k/50k checkpoints are diagnostics.", "",
             "Our Transformer and RT retain two D512/H8/GELU-FFN2048 blocks with ALiBi, LayerNorm and learned QK normalization (6,357,504 parameters). The authors' GPT architecture uses two D512/H8/RMSNorm/RoPE blocks with SwiGLU hidden width 1408, no QK normalization, and Normal(0,.02) initialization (6,486,528 parameters).",
             "",
             "All runs use our frozen data, same-position A5 CE, AdamW update function, FP32 eager runtime and identical per-update word order. SEQ and RT continue their original 10k checkpoints; the reference starts from its own architecture initialization. A common seed does not make its weights paired with our models. This comparison changes several architecture components together and does not isolate an ALiBi or RoPE effect.", "",
             "| Backbone at 100k | Set | State CE | Mean token accuracy M | Whole-word exactness E | Final-state accuracy A |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for role in ("dev", "ood_dev"):
            m = summary["arms"][arm]["checkpoint_metrics"]["100000"][role]
            lines.append(f"| {LABELS[arm]} | {role}, L{m['length']} | {m['ce']:.7f} | {100*m['token_accuracy']:.4f}% | {100*m['whole_word_exact_match']:.4f}% | {100*m['final_state_accuracy']:.4f}% |")
    lines += ["", "Every reported checkpoint uses the same first 102,400 frozen words for each development role. E(t) requires every state through t to be correct; A(t) checks only position t; M(t) averages token correctness through t. Short development words and long-word prefixes are different samples. All length curves below are prefixes of saved length-36 outputs, not independently generated evaluations at every length.", "",
              "![Full 1–36 curves](length-full.png)", "", "![Explicit 10–18 crop](length-zoom-10-18.png)", "",
              "The zoom uses exactly the full plot's data and metric panels; failures before position 10 are cropped out. The right panel M(t) can stay high because of correct earlier states even when A(t) is near chance and E(t) is zero. Only A has a flat 1/60 state-guessing line.", "",
              "![Cumulative exactness only](figure10-style-prefix-exactness.png)", "",
              "The Figure 10-style plot uses E(t), the cumulative-correctness convention supported by the released A5 evaluator. Its title identifies this as our experiment; no paper measurements are overlaid or claimed reproduced.", "",
              "Both attention-only architectures also share a small structural limitation: without a BOS token, an initial repeated nonidentity operation `(g, g)` yields identical position representations in exact arithmetic although the required states `g` and `g²` differ. RoPE/ALiBi alter scores but cannot distinguish identical values. This limits perfect-prefix accuracy even after learning; it does not explain the large later-position gap. The [handoff](../../../rt-a5-transformer-reference-handoff.md#shared-repeated-prefix-limitation) records the bounded count and CPU checks, with the roundoff and per-word attribution qualifications.", "",
              "| Backbone | Updates | Last E≥95% | Last E≥50% | Last E≥10% | Last E≥1% | First zero-success length |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        for step in STEPS:
            h = summary["arms"][arm]["horizons"][str(step)]
            values = [h["last_length_at_or_above"][key] for key in ("95%", "50%", "10%", "1%")]
            lines.append(f"| {LABELS[arm]} | {step:,} | " + " | ".join("None" if x is None else str(x) for x in values) + f" | {h['first_observed_zero_length']} |")
    lines += ["", "Thresholds are inclusive and bounded by the tested range 1–36. Zero is an actual zero success count in this sample, not a proof of population impossibility. CSV intervals are pointwise Wilson 95% over words for E/A, not paired-difference or training-seed uncertainty. M has no interval assuming independent positions.", "",
              "| Backbone at 100k | E(12) | E(13) | E(14) | E(16) | E(36) | A(36) | M(36) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        rows = summary["arms"][arm]["curves"]["100000"]["ood_dev"]
        values = [rows[t-1]["E"] for t in (12,13,14,16,36)] + [rows[-1]["A"],rows[-1]["M"]]
        lines.append(f"| {LABELS[arm]} | " + " | ".join(f"{100*x:.4f}%" for x in values) + " |")
    lines += ["", "![Learning curves by update](learning-curves.png)", "",
              "| Backbone | Total training-loop minutes | Sum of job elapsed minutes |",
              "| --- | ---: | ---: |"]
    for arm in ARMS:
        t = summary["arms"][arm]["timing"]
        lines.append(f"| {LABELS[arm]} | {t['training_seconds']/60:.2f} | {t['sum_job_elapsed_seconds']/60:.2f} |")
    lines += ["", "Joined timing includes the original pilot for SEQ/RT; job-time sums exclude gaps between jobs. It is measured elapsed time, not a controlled hardware-normalized benchmark.", "",
              "This is the authors' architecture on our harness, not exact paper reproduction: we use our generated corpus, optimizer/data-order conventions, FP32/eager runtime and 100k rather than 400k updates. One development seed does not establish convergence or seed robustness. No NextLat objective, autonomous latent recurrence, or RoPE-only ablation is included in this 100k comparison. Final confirmation remains unevaluated. A separately authorized RoPE-only control uses a matched 50k endpoint and its own report.", "",
              "[All counts and denominators](metrics.csv) · [Plot data](plot-data.json) · [Machine-readable report](report.json)", ""]
    return "\n".join(lines)


def run(args):
    output = Path(args.output_dir).resolve()
    mirror = Path(args.mirror_dir).resolve() if args.mirror_dir else None
    if output.exists() or (mirror is not None and mirror.exists()):
        raise FileExistsError("Report and optional mirror must be fresh directories")
    summary = make_summary(args)
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for arm, record in summary["arms"].items():
        for i, files in enumerate(record["input_files"]):
            shutil.copyfile(local_path(files["report"]["path"]), output / "inputs" / f"{arm}-job{i}-report.json")
    summary["reporting_sources"] = {}
    for relative in ("scripts/rt_a5_reference_report.py", "scripts/rt_a5_budget_report.py",
                     "scripts/rt_a5_length_report.py", "scripts/rt_a5_report.py", "scripts/experiment_tracking.py"):
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        summary["reporting_sources"][relative] = hash_file(destination)
    rows = [r for arm in ARMS for step in STEPS for role in ("dev", "ood_dev") for r in summary["arms"][arm]["curves"][str(step)][role]]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    write_json(output / "plot-data.json", summary)
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    result = {**summary, "status": "running", "figures": figures}
    tracker = None
    try:
        if not args.no_wandb:
            tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                                    group=args.wandb_group, name="transformer-reference-comparison-100k")
            tracker.start({key: summary[key] for key in ("schema", "scope", "budget", "common_contract")})
            import wandb
            tracker.log({"report/metrics": wandb.Table(columns=list(CSV_FIELDS), data=[[r[k] for k in CSV_FIELDS] for r in rows]),
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
        result["artifacts"] = {str(p.relative_to(output)): hash_file(p) for p in sorted(output.rglob("*"))
                               if p.is_file() and "wandb" not in p.relative_to(output).parts and p.name != "report.json"}
        write_json(output / "report.json", result)
    if mirror is not None:
        shutil.copytree(output, mirror, ignore=shutil.ignore_patterns("wandb"))
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": result["wandb"]}), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    pilot = ".runtime/rt-a5/20260911T154748Z"
    lineage = ".runtime/rt-a5/20260911T201820Z-transformer-reference"
    p.add_argument("--seq-pilot-dir", default=pilot + "/train-seq")
    p.add_argument("--rt-pilot-dir", default=pilot + "/train-rt")
    p.add_argument("--seq-dir", default=lineage + "/train-seq")
    p.add_argument("--rt-dir", default=".runtime/rt-a5/20260911T171239Z-rt100k/train-rt")
    p.add_argument("--reference-dir", default=lineage + "/train-reference-gpt")
    p.add_argument("--output-dir", default="docs/reports/rt-a5/transformer-reference-100k")
    p.add_argument("--mirror-dir")
    p.add_argument("--wandb-group", default="20260911T201820Z-transformer-reference")
    p.add_argument("--no-wandb", action="store_true", help="Explicit local-only report; normal graphable runs log online")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
