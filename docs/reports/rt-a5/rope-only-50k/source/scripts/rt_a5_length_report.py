#!/usr/bin/env python3
"""Report one A5 arm's saved length curves without model inference or training."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_report import checkpoint_path, hash_file, read_arm, write_json

SCHEMA = "rt-a5-length-report-v1"
STEPS = (5000, 10000)
THRESHOLDS = (.95, .50, .10, .01)
EXPECTED_ROWS = 102400
ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("update", "length", "words", "prefix_exact_count", "state_correct_count",
           "token_correct_count", "token_denominator", "E", "E_low95", "E_high95",
           "A", "A_low95", "A_high95", "M")


def recover_count(fraction, denominator):
    """Recover a recorded integer tally, rejecting rounded/incompatible fractions."""
    if type(denominator) is not int or denominator < 1:
        raise ValueError("Count denominator must be a positive integer")
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 <= fraction <= 1):
        raise ValueError("Accuracy must be a finite fraction in [0, 1]")
    scaled = fraction * denominator
    count = round(scaled)
    if not math.isclose(scaled, count, rel_tol=0, abs_tol=1e-7):
        raise ValueError("Reported fraction does not recover an integer count")
    return count


def wilson95(count, total):
    """Pointwise binomial Wilson interval over words, not over token positions."""
    if type(total) is not int or total < 1 or type(count) is not int or not 0 <= count <= total:
        raise ValueError("Wilson interval requires valid integer word counts")
    z = 1.959963984540054
    p = count / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def curve_rows(metric):
    n = metric["rows"]
    if type(n) is not int or n != EXPECTED_ROWS or metric.get("evaluated_rows", n) != n:
        raise ValueError("Length curves require the full 102400-word development evaluation")
    if metric["length"] != 36 or metric["tokens"] != n * 36 or metric["role"] != "ood_dev":
        raise ValueError("Expected a complete length-36 OOD development curve")
    if metric["update"] not in STEPS:
        raise ValueError("Expected the fixed 5000 or 10000 update checkpoint")
    a, e = metric["isolated_state_accuracy"], metric["cumulative_prefix_exactness"]
    if len(a) != 36 or len(e) != 36:
        raise ValueError("Each curve must have all 36 positions")
    ac, ec = [recover_count(x, n) for x in a], [recover_count(x, n) for x in e]
    if (ac[0] != ec[0] or any(x > y for x, y in zip(ec, ac))
            or any(after > before or before - after > n - state
                   for before, after, state in zip(ec, ec[1:], ac[1:]))):
        raise ValueError("Counts violate cumulative prefix correctness semantics")
    for key, expected in (("token_accuracy", sum(ac) / (n * 36)),
                          ("whole_word_exact_match", ec[-1] / n),
                          ("final_state_accuracy", ac[-1] / n)):
        if not math.isclose(metric[key], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Recorded scalar {key} disagrees with recovered counts")
    rows, token_count = [], 0
    for t, (state, exact) in enumerate(zip(ac, ec), 1):
        token_count += state
        elo, ehi = wilson95(exact, n)
        alo, ahi = wilson95(state, n)
        rows.append(dict(zip(COLUMNS, (metric["update"], t, n, exact, state, token_count, n * t,
                                      exact / n, elo, ehi, state / n, alo, ahi, token_count / (n * t)))))
    return rows


def horizons(rows):
    """Inclusive >= thresholds and literal zero counts for the fixed sample."""
    return {"last_length_at_or_above": {
                f"{threshold:.0%}": max((r["length"] for r in rows if r["E"] >= threshold), default=None)
                for threshold in THRESHOLDS},
            "first_observed_zero_length": next((r["length"] for r in rows if r["prefix_exact_count"] == 0), None),
            "first_zero_upper95": next((r["E_high95"] for r in rows if r["prefix_exact_count"] == 0), None)}


def make_summary(run_dir, architecture, prefix_report=None, prefix_assessment=None):
    arm = read_arm(run_dir, architecture)
    report = arm["report"]
    if report["start_update"] != 0 or report["completed_updates"] != 10000 or report["contract"]["length"] != 12:
        raise ValueError("This follow-up requires the original complete 10000-update, length-12 pilot")
    curves, checkpoints, metrics = {}, {}, {}
    for step in STEPS:
        selected = [m for m in report["evaluations"] if m["role"] == "ood_dev" and m["update"] == step]
        if len(selected) != 1:
            raise ValueError(f"Need exactly one OOD development evaluation at {step}")
        curves[str(step)] = curve_rows(selected[0])
        metrics[str(step)] = selected[0]
        selected_cp = [c for c in report["checkpoints"] if c["completed_updates"] == step]
        if len(selected_cp) != 1:
            raise ValueError(f"Need one retained checkpoint at {step}")
        actual = hash_file(checkpoint_path(selected_cp[0]["path"]))
        if any(actual[k] != selected_cp[0][k] for k in ("sha256", "bytes")):
            raise ValueError(f"Checkpoint hash/size differs at {step}")
        checkpoints[str(step)] = actual
    prefix = None
    if prefix_assessment is not None and prefix_report is None:
        raise ValueError("A prefix assessment must accompany its original report")
    if prefix_report is not None:
        prefix = {"input": hash_file(prefix_report), "report": json.loads(Path(prefix_report).read_text())}
        check = prefix["report"]
        if (check.get("architecture") != architecture
                or check.get("confirmation_evaluated") is not False or check.get("completed_updates") != 10000
                or check.get("schema") != "rt-a5-trained-prefix-v1"
                or check.get("role") != "ood_dev" or check.get("rows") != 1024):
            raise ValueError("Prefix check must match arm/endpoint and have confirmation unevaluated")
        expected_lengths = {"12", "13", "14", "16"} if architecture == "rt" else {"12", "13"}
        if (set(check.get("checks", {})) != expected_lengths
                or any(type(c.get("prediction_disagreements")) is not int or c["prediction_disagreements"] < 0
                       or c.get("accuracy_counts_equal") is not True for c in check["checks"].values())
                or check.get("parameters_unchanged") is not True or check.get("no_gradients_created") is not True
                or check.get("training_updates_performed") != 0):
            raise ValueError("Prefix check has missing or unequal accuracy counts, or modified model state")
        strict_passed = (check.get("status") == "complete" and check.get("passed") is True
                         and all(c.get("passed") is True and c["prediction_disagreements"] == 0 for c in check["checks"].values()))
        prefix["strict_logit_screen_passed"] = strict_passed
        prefix["prediction_disagreements_by_length"] = {length: c["prediction_disagreements"] for length, c in check["checks"].items()}
        if prefix_assessment is not None:
            assessment = json.loads(Path(prefix_assessment).read_text())
            prefix["assessment"] = {"input": hash_file(prefix_assessment), "report": assessment}
            if (assessment.get("schema") != "rt-a5-prefix-assessment-v1"
                    or assessment.get("status") != "complete" or assessment.get("usable_for_length_metrics") is not True
                    or assessment.get("original_prefix_report_sha256") != prefix["input"]["sha256"]
                    or assessment.get("architecture") != architecture or assessment.get("completed_updates") != 10000
                    or not isinstance(assessment.get("qualification"), str) or not assessment["qualification"].strip()
                    or any(assessment.get(k) != report["contract"][k]
                           for k in ("source_sha256", "data_manifest_sha256"))):
                raise ValueError("Prefix assessment must explicitly qualify this exact report and source/data identity")
            if (set(assessment.get("checks", {})) != expected_lengths
                    or any(c.get("correctness_disagreements", 0 if c.get("prediction_disagreements") == 0
                                 and check["checks"][length]["prediction_disagreements"] == 0 else None) != 0
                           or c.get("accuracy_counts_equal") is not True
                           or c.get("prediction_disagreements") != check["checks"][length]["prediction_disagreements"]
                           for length, c in assessment["checks"].items())):
                raise ValueError("Prefix assessment must verify identical per-word correctness masks; aggregate counts are insufficient")
        if not strict_passed and not (prefix_assessment is not None and check.get("status") == "failed"
                                     and check.get("passed") is False and check.get("error_type") == "AssertionError"):
            raise ValueError("A failed strict prefix screen requires a qualified assessment")
        if check["checkpoint"]["sha256"] != checkpoints["10000"]["sha256"]:
            raise ValueError("Prefix check checkpoint differs from historical endpoint")
        identity = check.get("contract", check)
        for key in ("source_sha256", "data_manifest_sha256"):
            if identity.get(key) != report["contract"][key]:
                raise ValueError(f"Prefix check {key} differs from training contract")
    return {"schema": SCHEMA, "architecture": architecture, "primary_update": 10000,
            "training_length": 12, "confirmation_evaluated": False,
            "scope": "One development seed; fixed 10000-update endpoint, 5000-update diagnostic; no new model inference in this reporter",
            "metrics": {"E": "Fraction of words with every state through t correct",
                        "A": "Fraction of words with the state at t correct",
                        "M": "Mean token accuracy through t: sum of state counts / (words * t)"},
            "intervals": "Pointwise Wilson 95% intervals over words for E and A; no seed uncertainty, simultaneous coverage or M intervals",
            "contract": report["contract"], "parameter_count": report["initialization"]["parameter_count"],
            "canonical_initialization_sha256": report["initialization"]["canonical_sha256"],
            "source_files": report["source_files"], "historical_wandb": report.get("wandb"),
            "inputs": arm["input_files"], "checkpoints": checkpoints, "prefix_check": prefix,
            "historical_metrics": metrics, "curves": curves,
            "horizons": {step: horizons(rows) for step, rows in curves.items()},
            "delta_10000_minus_5000": [{"length": late["length"], **{k: late[k] - early[k] for k in ("E", "A", "M")}}
                                        for early, late in zip(curves["5000"], curves["10000"])]}


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    figures = {}
    for view, limits in (("full", (1, 36)), ("zoom", (10, 18) if summary["architecture"] == "rt" else (4, 14))):
        figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), constrained_layout=True)
        for axis, key, title in zip(axes, ("E", "A", "M"),
                                   ("Every state through t correct: E(t)", "State at t correct: A(t)", "Mean token accuracy through t: M(t)")):
            for step, color in (("5000", "#5B83B4"), ("10000", "#C44E34")):
                rows = summary["curves"][step]
                x = [r["length"] for r in rows]
                axis.plot(x, [r[key] for r in rows], color=color, linewidth=1.8, marker="o", markersize=3,
                          label=f"{int(step):,} updates" + (" (primary)" if step == "10000" else ""))
                if key != "M":
                    axis.fill_between(x, [r[key + "_low95"] for r in rows], [r[key + "_high95"] for r in rows], color=color, alpha=.20)
            axis.axvline(12, color="#6E6E6E", linestyle="--", linewidth=1, label="Training boundary")
            if key == "A":
                axis.axhline(1 / 60, color="#777777", linestyle=":", linewidth=1.2, label="1/60 state chance")
            axis.set(xlim=limits, ylim=(-.025, 1.025), xlabel="Operation position / prefix length t", title=title)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=.18)
            axis.legend(fontsize=8, loc="upper right")
        figure.suptitle(f"A5 {summary['architecture'].upper()} · 102,400 OOD development words · one seed\n"
                       "Saved length-36 prefixes; shaded pointwise 95% Wilson intervals for E/A only", fontsize=12)
        name = f"length-{view}"
        figures[name] = {}
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            figure.savefig(Path(output) / filename, dpi=200)
            figures[name][suffix] = filename
        plt.close(figure)
    return figures


def markdown_report(summary):
    arm = summary["architecture"].upper()
    lines = [f"# A5 {arm}: length-generalization boundary", "",
             f"Two blocks, width {summary['contract']['width']}, {summary['parameter_count']:,} parameters; "
             f"full FP32, ALiBi, seed {summary['contract']['seed']}. The primary endpoint remains 10,000 updates.", "",
             "The curves below reuse the same first 102,400 frozen OOD development words at 5,000 and 10,000 updates. "
             "They are prefixes of historical length-36 model outputs, not newly inferred independent datasets at every length.", "",
             "**E(t)** requires every state through t to be correct. **A(t)** requires only the state at t. "
             "**M(t)** averages token correctness through t. At the end of a word, E is whole-word exactness, "
             "A is final-state accuracy and M is mean token accuracy. Only A has a flat 1/60 state-guessing reference.", "",
             "| Updates | Last E ≥95% | Last E ≥50% | Last E ≥10% | Last E ≥1% | First zero-success length |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for step in map(str, STEPS):
        h = summary["horizons"][step]
        values = [h["last_length_at_or_above"][f"{threshold:.0%}"] for threshold in THRESHOLDS]
        lines.append(f"| {int(step):,} | " + " | ".join(str(v) if v is not None else "None" for v in values)
                     + f" | {h['first_observed_zero_length']} |")
    lines += ["", "Thresholds are inclusive and are measured on this finite sample. Observed zeros come from actual "
              "zero integer success counts, not rounded percentages; they do not establish zero population success. "
              "Intervals below are pointwise Wilson 95% intervals across sampled words at a fixed checkpoint; "
              "they do not measure variation across training seeds. No mean-token interval assumes independent positions.", "",
              "![Full length curves](length-full.png)", "", "![Boundary detail](length-zoom.png)", ""]
    if summary["prefix_check"]:
        differences = summary["prefix_check"]["prediction_disagreements_by_length"]
        changed = sum(differences.values())
        agreement = ("All checked shapes produced identical predicted states and E/A/M counts. " if changed == 0 else
                     f"There were **{changed} changed predicted class(es)** across the prefix comparisons "
                     f"({', '.join(f'T{length}: {count}' for length, count in differences.items())}). "
                     "The saved-output assessment verified that each word/position retained the same correct-or-wrong "
                     "indicator: changed predictions remained wrong, and every E/A/M count was identical. ")
        lines += ["The new forward-only truncation check uses **1,024 words**, distinct from the **102,400-word "
                  "historical curves** above. " + agreement + "Weights were unchanged, with no gradients or optimizer updates. "
                  "[Original prefix-check snapshot](inputs/prefix-check.json).", ""]
        if not summary["prefix_check"]["strict_logit_screen_passed"]:
            lines += ["**Qualification:** the original elementwise FP32 logit screen failed and remains recorded as failed. "
                      "The separate saved-output assessment supports using the exactly matching accuracy metrics "
                      "for this bounded length check; it does not retroactively make that strict screen pass.", "",
                      summary["prefix_check"]["assessment"]["report"]["qualification"], "",
                      "[Separate assessment snapshot](inputs/prefix-assessment.json)", ""]
        else:
            lines += ["The original elementwise FP32 logit screen also passed.", ""]
    else:
        lines += ["No new trained-model truncation check is attached. These remain historical length-36 prefix measurements.", ""]
    for step in map(str, STEPS):
        lines += [f"## {int(step):,} updates: all lengths", "",
                  "| t | All-correct words / 102,400 | E(t), % [95% interval] | State-correct words / 102,400 | A(t), % [95% interval] | M(t), % |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in summary["curves"][step]:
            lines.append(f"| {r['length']} | {r['prefix_exact_count']:,} | {100*r['E']:.4f} "
                         f"[{100*r['E_low95']:.4f}, {100*r['E_high95']:.4f}] | {r['state_correct_count']:,} | "
                         f"{100*r['A']:.4f} [{100*r['A_low95']:.4f}, {100*r['A_high95']:.4f}] | {100*r['M']:.4f} |")
        lines += [""]
    lines += ["[CSV with exact counts and denominators](length-curves.csv) · [Plot data and provenance](plot-data.json) · "
              "[Machine-readable report](report.json)", "",
              "The 10,000-update endpoint is only 2.5% of the reference 400,000-update budget. "
              "This single-seed development result is not a convergence or architecture-impossibility claim. "
              "The 5,000-update curve is a diagnostic comparison, not a newly selected endpoint. "
              "Final confirmation remains unevaluated. No training or gradient analysis was performed for this report.", ""]
    return "\n".join(lines)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh length-report output directory")
    summary = make_summary(args.run_dir, args.architecture, args.prefix_report, args.prefix_assessment)
    output.mkdir(parents=True)
    (output / "inputs").mkdir()
    shutil.copyfile(Path(args.run_dir) / "report.json", output / "inputs/training-report.json")
    if args.prefix_report:
        shutil.copyfile(args.prefix_report, output / "inputs/prefix-check.json")
    if args.prefix_assessment:
        shutil.copyfile(args.prefix_assessment, output / "inputs/prefix-assessment.json")
    reporting_sources = {}
    for relative in ("scripts/rt_a5_length_report.py", "scripts/rt_a5_report.py", "scripts/experiment_tracking.py"):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        reporting_sources[relative] = hash_file(target)
    summary["reporting_sources"] = reporting_sources
    write_json(output / "plot-data.json", summary)
    with (output / "length-curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(row for step in map(str, STEPS) for row in summary["curves"][step])
    figures = plot_results(summary, output)
    (output / "report.md").write_text(markdown_report(summary))
    tracker = OnlineTracker(project="rt-a5-state-tracking", entity="taylorbollman", output_dir=output,
                            group=args.wandb_group, name=f"{args.architecture}-length-boundary-step10000")
    result = {**summary, "status": "running", "figures": figures}
    try:
        tracker.start({k: summary[k] for k in ("schema", "architecture", "primary_update", "contract", "scope", "intervals")})
        import wandb
        tracker.log({"length/table": wandb.Table(columns=list(COLUMNS),
                     data=[[row[k] for k in COLUMNS] for step in map(str, STEPS) for row in summary["curves"][step]]),
                     **{f"length/{name}": wandb.Image(str(output / files["png"])) for name, files in figures.items()}})
        for step in map(str, STEPS):
            tracker.log({"update": int(step), **{f"dev/length_{r['length']}/{key}": r[key]
                         for r in summary["curves"][step] for key in ("E", "A", "M")}})
        tracker.summary({"confirmation_evaluated": False, "horizons": summary["horizons"],
                         "primary_update": 10000, "scope": summary["scope"]})
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
    parser.add_argument("--architecture", choices=("rt", "seq"), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix-report")
    parser.add_argument("--prefix-assessment")
    parser.add_argument("--wandb-group")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
