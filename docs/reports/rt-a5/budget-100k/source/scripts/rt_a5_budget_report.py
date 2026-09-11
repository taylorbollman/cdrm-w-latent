#!/usr/bin/env python3
"""CPU-only report of the completed RT A5 budget continuation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_length_report import COLUMNS, horizons, recover_count, wilson95
from scripts.rt_a5_report import checkpoint_path, finite_number, hash_file, read_arm, training_bins, write_json

SCHEMA = "rt-a5-budget-report-v1"
ROOT = Path(__file__).resolve().parents[1]
PILOT_STEPS = (5000, 10000)
DEFAULT_ENDPOINT = 100000
REFERENCE_UPDATES = 400000
ROWS = 102400
KEY_LENGTHS = (12, 13, 14, 16, 36)
BIN_UPDATES = 100


def checkpoint_steps(endpoint):
    if type(endpoint) is not int or endpoint < 30000 or endpoint % 10000:
        raise ValueError("Endpoint must be a multiple of 10000, at least 30000")
    return PILOT_STEPS + (20000, 25000, *range(30000, endpoint + 1, 10000))


def evaluation_rows(metric):
    """Exact per-position tallies with word-level E/A intervals; no M interval."""
    n, length, role = metric["rows"], metric["length"], metric["role"]
    if (type(n) is not int or n != ROWS or metric.get("evaluated_rows", n) != n
            or role not in ("dev", "ood_dev") or length != (12 if role == "dev" else 36)
            or metric["tokens"] != n * length):
        raise ValueError("Expected a full 102400-word development evaluation with the declared role length")
    if type(metric["update"]) is not int or metric["update"] < 1:
        raise ValueError("Evaluation update must be a positive integer")
    a, e, ce = (metric[k] for k in ("isolated_state_accuracy", "cumulative_prefix_exactness", "per_position_ce"))
    if any(len(values) != length for values in (a, e, ce)):
        raise ValueError("Evaluation curves must contain every position")
    ac, ec = [recover_count(x, n) for x in a], [recover_count(x, n) for x in e]
    if (ac[0] != ec[0] or any(exact > state for exact, state in zip(ec, ac))
            or any(after > before or before - after > n - state
                   for before, after, state in zip(ec, ec[1:], ac[1:]))):
        raise ValueError("Recovered counts violate cumulative correctness semantics")
    if any(not finite_number(x) or x < 0 for x in ce):
        raise ValueError("Invalid per-position CE")
    for key, expected in (("token_accuracy", sum(ac) / (n * length)),
                          ("whole_word_exact_match", ec[-1] / n),
                          ("final_state_accuracy", ac[-1] / n), ("ce", sum(ce) / length)):
        if not finite_number(metric[key]) or not math.isclose(metric[key], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Recorded scalar {key} disagrees with its curve")
    rows, token_count = [], 0
    for t, (state, exact) in enumerate(zip(ac, ec), 1):
        token_count += state
        elo, ehi = wilson95(exact, n)
        alo, ahi = wilson95(state, n)
        rows.append({"role": role, **dict(zip(COLUMNS, (metric["update"], t, n, exact, state,
                    token_count, n * t, exact / n, elo, ehi, state / n, alo, ahi, token_count / (n * t)))),
                     "state_ce": ce[t - 1]})
    return rows


def verify_continuation(pilot, continuation, endpoint=DEFAULT_ENDPOINT):
    """Join only one exact uninterrupted continuation of the original 10k pilot."""
    left, right = pilot["report"], continuation["report"]
    if (left["start_update"], left["completed_updates"], right["start_update"], right["completed_updates"]) != (0, 10000, 10000, endpoint):
        raise ValueError(f"Expected original 0->10000 pilot and 10000->{endpoint} continuation")
    if left["contract"] != right["contract"]:
        raise ValueError("Pilot and continuation source/data/model/runtime/optimization contracts differ")
    if left["initialization"] != right["initialization"]:
        raise ValueError("Continuation initialization lineage differs")
    contract = right["contract"]
    if any(contract.get(key) != value for key, value in {"architecture": "rt", "batch_size": 1024,
           "train_rows": 800000, "length": 12, "width": 512, "precision": "fp32",
           "tf32": False, "compile": False, "cuda_graphs": False}.items()):
        raise ValueError("The completed run does not match the approved RT paper-budget experiment")
    parent = right.get("parent_checkpoint") or {}
    expected = pilot["input_files"]["endpoint_checkpoint"]
    if parent.get("sha256") != expected["sha256"]:
        raise ValueError("Continuation parent SHA256 differs from retained pilot endpoint")
    actual = hash_file(checkpoint_path(parent["path"]))
    if any(actual[k] != expected[k] for k in ("sha256", "bytes")):
        raise ValueError("Continuation parent file differs from retained pilot endpoint")
    combined = pilot["history"] + continuation["history"]
    if len(combined) != endpoint or any(row["update"] != i for i, row in enumerate(combined, 1)):
        raise ValueError(f"Combined history must contain each update 1..{endpoint} exactly once")
    return combined


def make_summary(pilot_dir, run_dir, endpoint=DEFAULT_ENDPOINT):
    steps = checkpoint_steps(endpoint)
    pilot, continuation = read_arm(pilot_dir, "rt"), read_arm(run_dir, "rt")
    history = verify_continuation(pilot, continuation, endpoint)
    reports = {"pilot": pilot["report"], "continuation": continuation["report"]}
    curves, evaluated, retained = {}, {}, {}
    for phase, selected_steps in (("pilot", PILOT_STEPS), ("continuation", steps[len(PILOT_STEPS):])):
        source = reports[phase]
        for step in selected_steps:
            key = str(step)
            curves[key], evaluated[key] = {}, {}
            for role in ("dev", "ood_dev"):
                matches = [row for row in source["evaluations"] if row["update"] == step and row["role"] == role]
                if len(matches) != 1:
                    raise ValueError(f"Need exactly one full {role} evaluation at update {step}")
                evaluated[key][role] = matches[0]
                curves[key][role] = evaluation_rows(matches[0])
            checkpoints = [row for row in source["checkpoints"] if row["completed_updates"] == step]
            if len(checkpoints) != 1:
                raise ValueError(f"Need one retained checkpoint at update {step}")
            actual = hash_file(checkpoint_path(checkpoints[0]["path"]))
            if any(actual[k] != checkpoints[0][k] for k in ("sha256", "bytes")):
                raise ValueError(f"Checkpoint hash/size differs at update {step}")
            retained[key] = {**actual, "phase": phase}
    bins = training_bins(history, BIN_UPDATES)
    loop_seconds = sum(row["seconds"] for row in history)
    timing = {}
    for name, record in (("pilot", pilot), ("continuation", continuation)):
        seconds = sum(row["seconds"] for row in record["history"])
        stated = record["report"].get("train_seconds")
        if not finite_number(stated) or not math.isclose(stated, seconds, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError(f"{name} training time disagrees with its update history")
        elapsed = record["report"].get("elapsed_seconds")
        if not finite_number(elapsed) or elapsed < seconds:
            raise ValueError(f"{name} elapsed time must include its training loop")
        timing[name] = {"training_seconds": seconds, "elapsed_seconds": elapsed}
    return {"schema": SCHEMA, "architecture": "rt", "primary_update": endpoint,
            "confirmation_evaluated": False,
            "scope": f"One RT development seed at fixed {endpoint} updates; earlier checkpoints are diagnostics, not selected endpoints; no new SEQ training",
            "checkpoint_updates": list(steps),
            "contract": reports["continuation"]["contract"],
            "initialization": reports["continuation"]["initialization"],
            "source_files": reports["continuation"]["source_files"],
            "parent_checkpoint": reports["continuation"]["parent_checkpoint"],
            "order_chain": reports["continuation"]["order_chain"],
            "inputs": {"pilot": pilot["input_files"], "continuation": continuation["input_files"]},
            "historical_wandb": {phase: packet.get("wandb") for phase, packet in reports.items()},
            "checkpoints": retained, "historical_metrics": evaluated, "curves": curves,
            "horizons": {step: horizons(roles["ood_dev"]) for step, roles in curves.items()},
            "key_lengths": list(KEY_LENGTHS), "training_bin_updates": BIN_UPDATES, "training_curve": bins,
            "budget": {"total_updates": endpoint, "additional_updates": endpoint - 10000,
                       "reference_updates": REFERENCE_UPDATES, "fraction_of_reference_budget": endpoint / REFERENCE_UPDATES,
                       "words_per_update": 1024, "unique_training_words": 800000,
                       "total_word_presentations": endpoint * 1024,
                       "additional_word_presentations": (endpoint - 10000) * 1024,
                       "total_training_token_presentations": endpoint * 1024 * 12,
                       "nominal_training_passes": endpoint * 1024 / 800000},
            "timing": {**timing, "total_training_seconds": loop_seconds,
                       "sum_job_elapsed_seconds": sum(row["elapsed_seconds"] for row in timing.values()),
                       "definition": "Training times sum update durations; job times include evaluation/checkpoint/logging. Sum of job times excludes the gap between jobs."},
            "intervals": "Pointwise Wilson 95% over sampled words for E/A, not seed or simultaneous uncertainty; no M interval",
            "metrics": {"E": "All states through t correct", "A": "State at t correct",
                        "M": "Mean correctness across positions 1..t"}}


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    output = Path(output)
    endpoint, steps = summary["primary_update"], summary["checkpoint_updates"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}

    def save(figure, name):
        figures[name] = {}
        for suffix in ("png", "pdf"):
            figures[name][suffix] = f"{name}.{suffix}"
            figure.savefig(output / figures[name][suffix], dpi=200)
        plt.close(figure)

    labels = ("Every state through t correct: E(t)", "State at t correct: A(t)", "Mean token accuracy through t: M(t)")
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("E", "A", "M"), labels):
        for step, color in (("10000", "#688CAF"), (str(endpoint), "#C44E34")):
            rows = summary["curves"][step]["ood_dev"]
            x = [r["length"] for r in rows]
            axis.plot(x, [r[metric] for r in rows], color=color, linewidth=1.8, marker="o", markersize=3,
                      label=f"{int(step):,} updates" + (" (primary)" if int(step) == endpoint else ""))
            if metric != "M":
                axis.fill_between(x, [r[metric + "_low95"] for r in rows], [r[metric + "_high95"] for r in rows], color=color, alpha=.2)
        axis.axvline(12, color="#777777", linestyle="--", linewidth=1, label="Training boundary")
        if metric == "A":
            axis.axhline(1 / 60, color="#777777", linestyle=":", linewidth=1, label="1/60 state chance")
        axis.set(xlim=(1, 36), ylim=(-.025, 1.025), xlabel="Operation position / prefix length t", title=title)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    figure.suptitle(f"A5 RT · fixed {endpoint:,}-update endpoint versus 10,000-update pilot\n102,400 OOD development words · one seed · pointwise Wilson intervals for E/A only", fontsize=12)
    save(figure, "final-length-curves")

    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), constrained_layout=True)
    colors = ("#356FA8", "#D16A35", "#35995A", "#9B5AA6", "#A78333")
    for axis, metric, title in zip(axes, ("E", "A", "M"), labels):
        for t, color in zip(KEY_LENGTHS, colors):
            axis.plot([step / 1000 for step in steps],
                      [summary["curves"][str(step)]["ood_dev"][t - 1][metric] for step in steps],
                      marker="o", markersize=2.5, linewidth=1.5, color=color, label=f"t={t}")
        if metric == "A":
            axis.axhline(1 / 60, color="#777777", linestyle=":", linewidth=1)
        axis.set(xlabel="Optimizer updates (thousands)", title=title, ylim=(-.025, 1.025))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    figure.suptitle("A5 RT · fixed OOD development prefixes versus training budget\nAll points use 102,400 words; earlier checkpoints are diagnostics", fontsize=12)
    save(figure, "length-accuracy-vs-updates")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, xkey in enumerate(("update", "training_seconds")):
        for row, (ykey, title) in enumerate((("loss", "Training cross-entropy"), ("token_accuracy", "Training token accuracy"))):
            axis = axes[row, col]
            scale = 1000 if xkey == "update" else 3600
            axis.plot([point[xkey] / scale for point in summary["training_curve"]],
                      [point[ykey] for point in summary["training_curve"]], color="#C44E34", linewidth=1.4)
            axis.set(xlabel="Optimizer updates (thousands)" if col == 0 else "Accumulated training-loop time (hours)", ylabel=title)
            axis.grid(alpha=.18)
            if ykey == "token_accuracy":
                axis.set_ylim(-.025, 1.025)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
    figure.suptitle("A5 RT · complete pilot + continuation history\nNonoverlapping 100-update means; training time excludes evaluation/checkpointing/logging", fontsize=12)
    save(figure, "training-curves")
    return figures


def markdown_report(summary):
    budget, timing = summary["budget"], summary["timing"]
    endpoint, steps = summary["primary_update"], summary["checkpoint_updates"]
    lines = [f"# A5 RT: completed {endpoint:,}-update budget", "",
             f"Two recurrent blocks, width 512, {summary['initialization']['parameter_count']:,} parameters, "
             f"seed {summary['contract']['seed']}; full FP32, ALiBi and GELU. Autocast, TF32, compilation and CUDA graphs remain disabled.", "",
             "The original 10,000-update RT checkpoint was resumed with its optimizer, RNG and data-order state. "
             "Source/data/model/runtime contracts and initialization lineage match, the parent checkpoint hash is verified, "
             f"and the joined histories contain each update 1–{endpoint:,} exactly once.", "",
             f"**Fixed primary endpoint: {endpoint:,} updates.** This is {budget['total_word_presentations']:,} training-word presentations "
             f"over {budget['unique_training_words']:,} unique words: **{budget['nominal_training_passes']:g} nominal passes**. "
             f"The continuation added {budget['additional_updates']:,} updates. This is {100*budget['fraction_of_reference_budget']:g}% of the "
             "400,000-update reference budget. Earlier checkpoints are diagnostics and do not replace the primary endpoint.", "",
             "| Checkpoint | Development set | CE | Mean token accuracy M | Whole-word exactness E | Final-state accuracy A |",
             "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for step in ("10000", str(endpoint)):
        for role in ("dev", "ood_dev"):
            m = summary["historical_metrics"][step][role]
            lines.append(f"| {int(step):,} | {role}, L{m['length']}, {m['rows']:,} words | {m['ce']:.8f} | "
                         f"{100*m['token_accuracy']:.4f}% | {100*m['whole_word_exact_match']:.4f}% | {100*m['final_state_accuracy']:.4f}% |")
    lines += ["", "E(t) requires every prediction through t to be correct; A(t) checks only the state at t; "
              "M(t) averages token correctness through t. Only A has a flat 1/60 guessing reference. "
              "The OOD curves reuse the same frozen 102,400 length-36 development words and show their prefixes. "
              "They are not separate newly sampled sets for each length. Short development words are a distinct set.", "",
              "![Final length curves](final-length-curves.png)", "",
              "![Accuracy versus update budget](length-accuracy-vs-updates.png)", "",
              "![Training curves](training-curves.png)", "",
              "| Updates | Last E ≥95% | Last E ≥50% | Last E ≥10% | Last E ≥1% | First zero-success length |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for step in steps:
        h = summary["horizons"][str(step)]
        values = [h["last_length_at_or_above"][threshold] for threshold in ("95%", "50%", "10%", "1%")]
        lines.append(f"| {step:,} | " + " | ".join("None" if x is None else str(x) for x in values)
                     + " | " + str(h["first_observed_zero_length"]) + " |")
    lines += ["", "Thresholds use inclusive ≥ comparisons. First zero means an actual integer zero in this sample, "
              "not a percentage rounded to zero or proof of population impossibility. A threshold reaching position 36 "
              "is bounded by the evaluated range; positions beyond 36 remain untested.", "",
              "| Final prefix t | All-correct words / 102,400 | E(t), % [95% interval] | State-correct words / 102,400 | A(t), % [95% interval] | M(t), % |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["curves"][str(endpoint)]["ood_dev"]:
        lines.append(f"| {r['length']} | {r['prefix_exact_count']:,} | {100*r['E']:.4f} "
                     f"[{100*r['E_low95']:.4f}, {100*r['E_high95']:.4f}] | {r['state_correct_count']:,} | "
                     f"{100*r['A']:.4f} [{100*r['A_low95']:.4f}, {100*r['A_high95']:.4f}] | {100*r['M']:.4f} |")
    lines += ["", "Intervals are pointwise Wilson 95% across words at one checkpoint; they do not describe seed variation "
              "or simultaneous coverage across positions/checkpoints. M has no interval assuming independent positions.", "",
              f"Measured training-loop time: {timing['total_training_seconds']/3600:.3f} hours across both jobs "
              f"({timing['continuation']['training_seconds']/3600:.3f} hours in the continuation). "
              f"Summed job elapsed time: {timing['sum_job_elapsed_seconds']/3600:.3f} hours; this includes evaluation, checkpointing "
              "and logging, and excludes the gap between the jobs.", "",
              "[All checkpoint/length counts (CSV)](length-curves.csv) · [Checkpoint summary (CSV)](checkpoint-metrics.csv) · "
              "[Training bins (CSV)](training-curves.csv) · [Plot data and provenance](plot-data.json) · [Report](report.json)", "",
              f"This is one RT development seed at {endpoint:,} updates, using our existing architecture and generated corpus. "
              "It does not establish an exact paper reproduction, final convergence or multi-seed reliability. "
              "No additional Transformer training is included; comparing this endpoint to SEQ-10k would have unequal budgets. "
              "Final confirmation remains unevaluated. This reporter performs no model inference or numerical analysis.", ""]
    return "\n".join(lines)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh budget-report output directory")
    summary = make_summary(args.pilot_dir, args.run_dir, args.endpoint)
    endpoint, steps = summary["primary_update"], summary["checkpoint_updates"]
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    for name, directory in (("pilot", args.pilot_dir), ("continuation", args.run_dir)):
        shutil.copyfile(Path(directory) / "report.json", output / "inputs" / f"{name}-report.json")
    summary["reporting_sources"] = {}
    for relative in ("scripts/rt_a5_budget_report.py", "scripts/rt_a5_length_report.py",
                     "scripts/rt_a5_report.py", "scripts/experiment_tracking.py"):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        summary["reporting_sources"][relative] = hash_file(target)
    rows = [row for step in steps for role in ("dev", "ood_dev") for row in summary["curves"][str(step)][role]]
    checkpoints = [{"update": step, "role": role, **{key: summary["historical_metrics"][str(step)][role][key]
                   for key in ("rows", "length", "ce", "token_accuracy", "whole_word_exact_match", "final_state_accuracy")}}
                   for step in steps for role in ("dev", "ood_dev")]
    write_csv(output / "length-curves.csv", rows, ("role", *COLUMNS, "state_ce"))
    write_csv(output / "checkpoint-metrics.csv", checkpoints, checkpoints[0].keys())
    write_csv(output / "training-curves.csv", summary["training_curve"], summary["training_curve"][0].keys())
    write_json(output / "plot-data.json", summary)
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name=f"rt-budget-report-step{endpoint}")
    result = {**summary, "status": "running", "figures": figures}
    try:
        tracker.start({k: summary[k] for k in ("schema", "contract", "budget", "scope", "intervals")})
        import wandb
        tracker.log({**{f"report/{name}": wandb.Image(str(output / files["png"])) for name, files in figures.items()},
                     "report/length_table": wandb.Table(columns=["role", *COLUMNS, "state_ce"],
                              data=[[r[k] for k in ("role", *COLUMNS, "state_ce")] for r in rows]),
                     "report/checkpoints": wandb.Table(columns=list(checkpoints[0]),
                              data=[list(r.values()) for r in checkpoints])})
        for point in summary["training_curve"]:
            update = point["update"]
            values = {"update": update, **{f"train/{k}": point[k] for k in ("loss", "token_accuracy", "whole_word_exact", "grad_norm", "training_seconds")}}
            if str(update) in summary["curves"]:
                for role in ("dev", "ood_dev"):
                    m = summary["historical_metrics"][str(update)][role]
                    values.update({f"dev/{role}/{key}": m[key] for key in ("ce", "token_accuracy", "whole_word_exact_match", "final_state_accuracy")})
                for length in KEY_LENGTHS:
                    row = summary["curves"][str(update)]["ood_dev"][length - 1]
                    values.update({f"dev/ood_prefix_{length}/{key}": row[key] for key in ("E", "A", "M")})
            tracker.log(values)
        tracker.summary({"confirmation_evaluated": False, "primary_update": endpoint,
                         "budget": summary["budget"], "primary_horizons": summary["horizons"][str(endpoint)],
                         "scope": summary["scope"]})
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
                               if p.is_file() and "wandb" not in p.relative_to(output).parts and p.name != "report.json"}
        write_json(output / "report.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output), "wandb": tracker.record["run_url"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-group")
    parser.add_argument("--endpoint", type=int, default=DEFAULT_ENDPOINT, help="Required fixed completed update count (default: 100000)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
