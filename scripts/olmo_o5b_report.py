#!/usr/bin/env python3
"""Compare completed O5b arms; keep finite passes and short online tests separate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.lm_schedule import alpha_for_update
from scripts import olmo_o4_report as paired

ARMS = ("ordinary", "fbt")
SPLITS = ("dev", "retention_dev")
SCHEMA = "olmo-o5b-final-comparison-v1"
FIGURES = ("learning-curves.pdf", "learning-curves.png", "online-comparison.pdf", "online-comparison.png")
require, sha = paired.require, paired.sha


def _numeric(metric):
    # The older helper owns only the common numerical/document contract.
    # Never mutate, relabel or drop the original FBT execution metadata.
    return {**metric, "schema": "olmo-lm-evaluation-v1"}


def _mode(beta, execution="finite"):
    result = {"beta": beta, "rt_mode": {"selected_layers": [], "alpha": 1.0}}
    if execution == "finite":
        result.update(enabled=True, num_passes=2)
    return result


def validate_metric(metric, *, beta, index, rows, documents, execution="finite", max_length=512):
    require(metric.get("schema") == "olmo-fbt-pass-evaluation-v1", "Unexpected FBT per-pass evaluation schema")
    require(metric.get("execution") == execution and metric.get("pass_index") == index,
            "Evaluation execution/pass identity differs")
    require(metric.get("mode") == _mode(beta, execution) and metric.get("precision") == "bf16_mixed",
            "Evaluation mode/precision differs from frozen FBT-only protocol")
    paired.validate_metric(_numeric(metric), documents=documents)
    require(metric["documents"] == rows, "Evaluation window count differs from fixed selection")
    require(metric.get("input_tokens") == metric["ce_count"] + rows,
            "Evaluation is not all-target, one-document-per-row CE")
    require(paired._integer(metric.get("batches"), 1) and paired._integer(metric.get("positions_per_chunk"), 1),
            "Evaluation batching metadata is missing")
    require(metric["input_tokens"] <= rows * max_length, "Evaluation exceeds its context-length bound")
    if documents:
        require(all(row["ce_count"] < max_length for row in metric["document_records"]),
                "An evaluation window exceeds its context-length bound")
    return metric


def validate_container(container, *, beta, rows, documents):
    require(container.get("schema") == "olmo-fbt-evaluation-v1", "Unexpected FBT evaluation container schema")
    passes = container.get("passes")
    require(isinstance(passes, list) and len(passes) == 2, "Exactly two separately scored finite passes are required")
    metadata = ("precision", "mode", "execution", "positions_per_chunk", "batches", "documents", "input_tokens")
    for index, metric in enumerate(passes):
        validate_metric(metric, beta=beta, index=index, rows=rows, documents=documents)
        require(all(container.get(key) == metric.get(key) for key in metadata), "Container/per-pass metadata differs")
        require(metric["ce_count"] == passes[0]["ce_count"], "Per-pass target counts differ")
        if documents:
            require(paired._window_signature(metric) == paired._window_signature(passes[0]), "Per-pass window/document alignment differs")
    if beta == 0:
        for key in ("ce_sum", "mean_nll", "next_token_correct"):
            require(paired._close(passes[0][key], passes[1][key]), "Beta-zero ordinary pass metrics differ")
    return container


def contrast(left, right):
    return paired.paired_document_bootstrap(_numeric(left), _numeric(right))


def _checkpoint(report, configuration):
    # Reuse O4's receipt checks with an ephemeral view of the arm namespace.
    prefix = report.get("storage_prefix", configuration.get("storage_prefix", ""))
    return paired._checkpoint({**report, "configuration": {**configuration, "storage_prefix": prefix}}, configuration)


def _event_boundary(event, schedule, arm):
    update, tokens = event.get("update"), event.get("input_tokens")
    scheduled = alpha_for_update(schedule, update, tokens)
    beta = scheduled if arm == "fbt" else 0.0
    require(event.get("beta") == beta, "Evaluation beta differs from frozen schedule")
    return beta


def _online_event(event, schedule, arm):
    beta = _event_boundary(event, schedule, arm)
    require(event.get("rows_requested") == 32 and event.get("max_length") == 64,
            "Online diagnostic must use the frozen 32-window, maximum-64-token prefixes")
    require(set(event.get("metrics", {})) == set(SPLITS), "Online splits differ")
    for values in event["metrics"].values():
        require(set(values) == {"pass0", "finite", "online"}, "Online diagnostic needs matching pass0/finite/online results")
        for name, index in (("pass0", 0), ("finite", 1), ("online", None)):
            metric = values[name]
            validate_metric(metric, beta=beta, index=index, rows=32, documents=True,
                            execution="online" if name == "online" else "finite", max_length=64)
            require(paired._window_signature(metric) == paired._window_signature(values["pass0"]),
                    "Online and finite short-prefix window/target alignment differs")
    return event


def build_comparison(preflight, configuration, arm_reports):
    require(preflight.get("schema") == "olmo-o5b-preflight-v1" and preflight.get("status") == "passed"
            and preflight.get("finished_utc"), "Require completed passing O5b preflight")
    require(configuration.get("schema") == "olmo-o5b-pilot-config-v1", "Unexpected frozen O5b configuration")
    require(set(arm_reports) == set(ARMS), "Require both completed O5b arms")
    sources = configuration.get("source_hashes", {})
    require(sources and sources == preflight.get("source_hashes") and all(isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values()), "Preflight/training source identity differs")
    require(configuration.get("precision") == "bf16_mixed" and configuration.get("eval_rows") == 128
            and configuration.get("final_eval_rows") == 512 and configuration.get("online_eval_rows") == 32
            and configuration.get("online_eval_length") == 64, "Frozen precision/evaluation sizes differ")
    require(configuration.get("num_passes") == 2 and configuration.get("gamma") == 1.0
            and configuration.get("rt_layers") == [] and configuration.get("nextlat_enabled") is False
            and configuration.get("prefix_mixin") is False and configuration.get("hidden_jitter") == 0.0,
            "O5b requires matched K2/gamma1 FBT-only objectives without sampling or jitter")
    schedule = configuration["schedule"]
    alpha_for_update(schedule, schedule["total_updates"], schedule["total_tokens"])
    require(preflight.get("schedule") == schedule and preflight.get("selected_batch_size") == schedule["batch_size"],
            "Preflight schedule/effective batch differs")
    physical = configuration.get("physical_batch_size")
    require(paired._integer(physical, 1) and schedule["batch_size"] % physical == 0, "Physical batch must divide effective batch")
    require(preflight.get("data_manifest_sha256") == configuration.get("data_manifest_sha256")
            and preflight.get("checkpoint", {}).get("sha256") == configuration.get("checkpoint_sha256"),
            "Preflight data/native checkpoint identity differs")
    baseline = preflight.get("baseline", {})
    require(set(baseline) == set(SPLITS), "Missing finite preflight baselines")
    for metric in baseline.values():
        validate_container(metric, beta=0.0, rows=512, documents=True)
    cold = preflight.get("cold_feedback")
    if cold is not None:
        require(set(cold) == set(SPLITS), "Cold feedback splits differ")
        for metric in cold.values():
            validate_container(metric, beta=1.0, rows=128, documents=False)
    initial_online = preflight.get("initial_online")
    if initial_online is not None:
        _online_event({"update": 0, "input_tokens": 0, "beta": 0.0, "rows_requested": 32,
                       "max_length": 64, "metrics": initial_online}, schedule, "ordinary")
    summaries, curves, shared = {}, {}, None
    for arm in ARMS:
        report = arm_reports[arm]
        require(report.get("schema") == "olmo-o5b-arm-v1" and report.get("status") == "completed"
                and report.get("arm") == arm and report.get("finished_utc"), f"Arm {arm} is not completed")
        require(report.get("configuration") == configuration, f"Arm {arm} frozen configuration differs")
        source = report.get("source_fingerprint", {})
        require(source.get("checkpoint_sha256") == configuration["checkpoint_sha256"]
                and source.get("code") == sources and source.get("data_manifest_sha256") == configuration["data_manifest_sha256"]
                and source.get("runtime") == preflight.get("runtime"), f"Arm {arm} source/data/runtime identity differs")
        counters = report.get("counters", {})
        expected = {"optimizer_updates": schedule["total_updates"],
                    "microbatches": schedule["total_updates"] * (schedule["batch_size"] // physical),
                    "documents": schedule["used_windows"], "input_tokens": schedule["total_tokens"],
                    "ce_positions": schedule["total_tokens"] - schedule["used_windows"], "latent_pairs": 0, "kl_triples": 0}
        require(all(type(counters.get(key)) is int and counters[key] == value for key, value in expected.items())
                and report.get("data_cursor") == schedule["used_windows"], f"Arm {arm} exposure/cursor/auxiliary counters differ")
        require(shared is None or shared == counters, "Arm counters differ")
        shared = counters
        full = [event for event in report.get("evaluations", []) if event.get("full") is True]
        require(len(full) == 1, "Exactly one authoritative full final evaluation is required")
        final = full[0]
        beta = _event_boundary(final, schedule, arm)
        require(final.get("update") == schedule["total_updates"] and final.get("rows_requested") == 512,
                "Final finite evaluation differs from endpoint")
        require(set(final.get("metrics", {})) == set(SPLITS), "Final evaluation splits differ")
        for split, value in final["metrics"].items():
            validate_container(value, beta=beta, rows=512, documents=True)
            require(paired._window_signature(value["passes"][0]) == paired._window_signature(baseline[split]["passes"][0]),
                    "Final windows/document identities differ from preflight")
        by_update = {}
        for event in report["evaluations"]:
            if event.get("full") is True:
                continue
            require(event.get("full") is False and event.get("rows_requested") == 128, "Curve mixes evaluation sample sizes")
            beta = _event_boundary(event, schedule, arm)
            require(set(event.get("metrics", {})) == set(SPLITS), "Curve evaluation splits differ")
            for value in event["metrics"].values():
                validate_container(value, beta=beta, rows=128, documents=False)
            update = event["update"]
            require(update not in by_update, "Duplicate authoritative curve evaluation after resume")
            by_update[update] = event
        require({0, schedule["warmup_updates"], schedule["ramp_end_update"], schedule["total_updates"]} <= set(by_update),
                "Curves omit initial/phase endpoints")
        curves[arm] = [by_update[index] for index in sorted(by_update)]
        online_by_update = {}
        for event in report.get("online_evaluations", []):
            _online_event(event, schedule, arm)
            for split in SPLITS:
                short = event["metrics"][split]["pass0"]
                expected_signature = [(b, r, d, min(count, 63)) for b, r, d, count in
                                      paired._window_signature(baseline[split]["passes"][0])[:32]]
                require(paired._window_signature(short) == expected_signature,
                        "Online diagnostic differs from the first 32 truncated preflight windows")
                if initial_online is not None:
                    require(paired._window_signature(short) == paired._window_signature(initial_online[split]["pass0"]),
                            "Online diagnostic differs from initial short-prefix selection")
            require(event["update"] not in online_by_update, "Duplicate authoritative online evaluation")
            online_by_update[event["update"]] = event
        require(schedule["total_updates"] in online_by_update, "Missing final online diagnostic")
        link = report.get("wandb", {}).get("run_url")
        require(isinstance(link, str) and link.startswith("https://wandb.ai/"), "Missing online run link")
        summaries[arm] = {"metrics": final["metrics"], "online_evaluations": list(online_by_update.values()),
                          "online_metrics": online_by_update[schedule["total_updates"]]["metrics"],
                          "checkpoint": _checkpoint(report, configuration), "counters": counters, "wandb_url": link,
                          "started_utc": report.get("started_utc"), "finished_utc": report["finished_utc"]}
    for arm in ARMS:
        require([row["update"] for row in curves[arm]] == [row["update"] for row in curves["ordinary"]],
                "Learning curve boundaries differ between arms")
        for event in curves[arm]:
            for split in SPLITS:
                first = curves["ordinary"][0]["metrics"][split]["passes"][0]
                require(event["metrics"][split]["passes"][0]["ce_count"] == first["ce_count"], "Curve target counts differ")
    comparisons, online_comparisons = {}, {}
    for split in SPLITS:
        ordinary, fbt = (summaries[arm]["metrics"][split]["passes"] for arm in ARMS)
        comparisons[split] = {"fbt_pass1_minus_ordinary_pass1": contrast(fbt[1], ordinary[1]),
                              "fbt_pass0_minus_ordinary_pass0": contrast(fbt[0], ordinary[0]),
                              "fbt_pass1_minus_fbt_pass0": contrast(fbt[1], fbt[0]),
                              "versus_preflight": {f"{arm}_pass{p}": contrast(summaries[arm]["metrics"][split]["passes"][p], baseline[split]["passes"][0])
                                                    for arm in ARMS for p in (0, 1)}}
        ordinary, fbt = (summaries[arm]["online_metrics"][split] for arm in ARMS)
        online_comparisons[split] = {"fbt_online_minus_ordinary_online": contrast(fbt["online"], ordinary["online"]),
                                     "fbt_online_minus_fbt_finite": contrast(fbt["online"], fbt["finite"]),
                                     "fbt_finite_minus_ordinary_finite": contrast(fbt["finite"], ordinary["finite"])}
    return {"schema": SCHEMA, "status": "completed", "configuration": configuration, "exposure": shared,
            "preflight_baseline": baseline, "preflight_cold_feedback_128": cold, "preflight_initial_online": initial_online,
            "arms": summaries, "curves": curves, "comparisons": comparisons,
            "online_comparisons": online_comparisons, "difference_convention": "left minus right token NLL; negative favors left",
            "qualification": "One seed and development subsets only; reserved tests untouched. Equal data exposure and the same "
            "two-pass CE objective (pass0 + pass1), not a claim of equal measured compute. No RT or NextLat. Context resets per "
            "window; documents counter counts windows, not unique documents. Paired intervals resample original document clusters "
            "and do not quantify seed variability. Unknown OLMo pretraining overlap. Token NLL does not establish programming-task "
            "success. Exact-online diagnostics are teacher-forced on separate 32-window, maximum-64-token prefixes; they are not "
            "free-running generation and must not be equated with 512-token finite-pass evaluations."}


def _read_inputs(preflight, runs):
    path = Path(preflight)
    path = path / "report.json" if path.is_dir() else path
    paths = {"preflight": path, "configuration": path.parent / "configuration.json",
             **{arm: Path(runs) / arm / "report.json" for arm in ARMS}}
    require(all(path.is_file() for path in paths.values()), "Preflight and both completed arm files are required")
    values = {key: json.loads(path.read_text()) for key, path in paths.items()}
    return paths, values


def validate_runs(preflight, runs):
    """Path-based strict validation, also used by final evidence retention."""
    _, values = _read_inputs(preflight, runs)
    return build_comparison(values["preflight"], values["configuration"], {arm: values[arm] for arm in ARMS})


def write_figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    schedule = summary["configuration"]["schedule"]
    styles = (("ordinary", 1, "Ordinary (beta 0)", "#4b5563", "-"),
              ("fbt", 0, "FBT pass 0", "#2878b5", "--"), ("fbt", 1, "FBT pass 1", "#2878b5", "-"))
    titles = ("CodeSearchNet Python development", "WikiText-2 language retention")
    for axis, split, title in zip(axes, SPLITS, titles):
        for arm, index, label, color, style in styles:
            rows = summary["curves"][arm]
            axis.plot([row["input_tokens"] / 1e6 for row in rows],
                      [row["metrics"][split]["passes"][index]["mean_nll"] for row in rows],
                      label=label, color=color, linestyle=style, marker="o", markersize=3)
        axis.axvspan(schedule["ramp_start_tokens"] / 1e6, schedule["ramp_end_tokens"] / 1e6, alpha=.08, color="gray", label="Beta ramp")
        axis.set(title=title, xlabel="Valid input tokens seen (millions)", ylabel="Token-weighted NLL (nats)")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fixed 128-window finite-pass curves · one seed", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"learning-curves.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, split, title in zip(axes, SPLITS, titles):
        for arm, offset, color in (("ordinary", -.12, "#4b5563"), ("fbt", .12, "#2878b5")):
            values = summary["arms"][arm]["online_metrics"][split]
            axis.plot([index + offset for index in range(3)], [values[key]["mean_nll"] for key in ("pass0", "finite", "online")],
                      "o", label=arm, color=color)
        axis.set(xticks=range(3), xticklabels=("Pass 0", "Finite K=2", "Exact online"), title=title, ylabel="Token-weighted NLL (nats)")
        axis.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Endpoint short-prefix diagnostic · same 32 windows · maximum 64 tokens · teacher-forced", fontsize=10)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"online-comparison.{suffix}", dpi=180)
    plt.close(fig)


def markdown(summary):
    exposure = summary["exposure"]
    lines = ["# OLMo O5b: ordinary versus FBT continuation", "",
             f"Both arms completed **{exposure['optimizer_updates']:,} updates** and **{exposure['input_tokens']:,} valid input tokens** "
             f"({exposure['ce_positions']:,} CE targets per pass) from the same original checkpoint and document stream.", "",
             "Both optimize pass0 CE + pass1 CE (K=2, gamma=1). Ordinary keeps beta zero; FBT ramps feedback. "
             "Final results use fixed 512-window development selections. Separate 128-window curves preserve their original sample size.", "",
             "| Model / inference pass | Code NLL | Code perplexity | Code accuracy | Retention NLL | Retention perplexity | Retention accuracy |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows = [("Original checkpoint", {split: summary["preflight_baseline"][split]["passes"][0] for split in SPLITS})]
    rows += [(f"{arm} pass {p}", {split: summary["arms"][arm]["metrics"][split]["passes"][p] for split in SPLITS}) for arm in ARMS for p in (0, 1)]
    for label, metrics in rows:
        cells = [f"{metrics[split]['mean_nll']:.6f} | {metrics[split]['perplexity']:.4f} | {100*metrics[split]['next_token_accuracy']:.3f}%" for split in SPLITS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Differences are left minus right NLL; negative favors left. 95% intervals use 1,000 paired resamples of "
              "original-document clusters, combining their windows before calculating token-weighted NLL (seed 20260922).", "",
              "| Finite-pass comparison (512 windows) | Code difference [95% interval] | Retention difference [95% interval] |",
              "| --- | ---: | ---: |"]
    for key in ("fbt_pass1_minus_ordinary_pass1", "fbt_pass0_minus_ordinary_pass0", "fbt_pass1_minus_fbt_pass0"):
        cells = [summary["comparisons"][split][key] for split in SPLITS]
        lines.append("| " + key + " | " + " | ".join(f"{v['estimate']:+.6f} [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]" for v in cells) + " |")
    lines += ["", "![Finite-pass learning curves](learning-curves.png)", "", "[Curves PDF](learning-curves.pdf)", "",
              "The following **32-window, maximum-64-token** results use the same short prefixes for finite and exact-online "
              "execution. They are separate from the 512-window comparison above.", "",
              "| Arm | Code pass0 / finite / online NLL | Retention pass0 / finite / online NLL |", "| --- | ---: | ---: |"]
    for arm in ARMS:
        cells = [" / ".join(f"{summary['arms'][arm]['online_metrics'][split][key]['mean_nll']:.6f}" for key in ("pass0", "finite", "online")) for split in SPLITS]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "![Short-prefix online diagnostic](online-comparison.png)", "", "[Online PDF](online-comparison.pdf) · "
              "[All results and paired intervals](final-comparison.json)", "", summary["qualification"], "", "Run and endpoint records:", ""]
    for arm in ARMS:
        value = summary["arms"][arm]
        checkpoint = value["checkpoint"]["storage"]
        lines.append(f"- [{arm} on W&B]({value['wandb_url']}); [checkpoint]({checkpoint['uri']}), generation `{checkpoint['generation']}`, SHA256 `{checkpoint['sha256']}`.")
    return "\n".join(lines) + "\n"


def build_report(args):
    paths, values = _read_inputs(args.preflight, args.runs)
    summary = build_comparison(values["preflight"], values["configuration"], {arm: values[arm] for arm in ARMS})
    summary.update(created_utc=datetime.now(timezone.utc).isoformat(), report_source_sha256=sha(__file__),
                   helper_source_sha256={"scripts/olmo_o4_report.py": sha(paired.__file__),
                                         "cdrm/pretrained/lm_schedule.py": sha(ROOT / "cdrm/pretrained/lm_schedule.py")},
                   input_sha256={key: sha(path) for key, path in paths.items()})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_figures(summary, output)
    summary["figure_sha256"] = {name: sha(output / name) for name in FIGURES}
    (output / "final-comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "results.md").write_text(markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = build_report(parser.parse_args())
    print(json.dumps({"status": result["status"], "input_tokens": result["exposure"]["input_tokens"]}))


if __name__ == "__main__":
    main()
