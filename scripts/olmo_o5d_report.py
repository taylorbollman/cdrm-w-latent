#!/usr/bin/env python3
"""Summarize unchanged-checkpoint finite/online O5d development evaluations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import olmo_o4_report as paired

SCHEMA = "olmo-o5d-online-diagnostic-v1"
ENDPOINTS = ("source", "mixed")
SPLITS = ("dev", "retention_dev")
SECTIONS = {"short_prefix": (32, 64), "full_context": (512, 512)}
EXECUTIONS = ("K2", "K3", "K4", "online")
FUSION_NAMES = {"backbone.fusion.state_proj.weight", "backbone.fusion.token_gate.weight"}
FIGURES = ("finite-online.pdf", "finite-online.png", "adaptation-transfer.pdf", "adaptation-transfer.png")
QUALIFICATION = (
    "Evaluation only, with unchanged checkpoints, fixed beta 1, RT/NextLat off and "
    "BF16 mixed precision. Every execution within a selection scores identical "
    "teacher-forced targets; exact online means sequential feedback, not free-running "
    "generation or a high-precision numerical oracle. Short prefixes and full contexts "
    "are different selections and their NLL levels must not be compared as matched "
    "samples. The ordinary reference is the shared frozen pass, not an equally "
    "additionally trained control. Development data, one training seed, and no "
    "reserved-test evaluation; this is a transfer diagnostic, not an FBT efficacy claim."
)
require = paired.require


def execution(case):
    return "online" if case["passes"] is None else f"K{case['passes']}"


def _hashes(value, label):
    require(isinstance(value, dict) and value and all(isinstance(k, str) and k and
            isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in value.items()),
            f"Missing or malformed {label} hashes")
    return value


def _numeric(metric):
    return {**metric, "schema": "olmo-lm-evaluation-v1"}


def _contrast(left, right):
    return paired.paired_document_bootstrap(_numeric(left), _numeric(right))


def _validate_container(container, case, batch_size):
    online = case["passes"] is None
    mode = {"beta": 1., "rt_mode": {"selected_layers": [], "alpha": 1.}}
    if not online:
        mode.update(enabled=True, num_passes=case["passes"])
    require(container.get("schema") == "olmo-fbt-evaluation-v1", "Unexpected FBT evaluation container")
    values = container.get("passes")
    require(isinstance(values, list) and len(values) == (1 if online else case["passes"]),
            "Evaluation omits or adds finite/online passes")
    metadata = ("precision", "mode", "execution", "positions_per_chunk", "batches", "documents", "input_tokens")
    for index, metric in enumerate(values):
        require(metric.get("schema") == "olmo-fbt-pass-evaluation-v1", "Unexpected FBT pass schema")
        require(metric.get("execution") == ("online" if online else "finite") and
                metric.get("pass_index") == (None if online else index), "Execution/pass identity differs")
        require(metric.get("mode") == mode and metric.get("precision") == "bf16_mixed",
                "Evaluation mode/precision differs from fixed diagnostic")
        require(all(container.get(k) == metric.get(k) for k in metadata), "Container/pass metadata differs")
        paired.validate_metric(_numeric(metric), documents=True)
        require(metric["documents"] == case["rows"] and metric["batches"] == math.ceil(case["rows"] / batch_size),
                "Evaluation window/batch count differs from fixed selection")
        require(metric.get("positions_per_chunk") == 128, "Full-vocabulary projection chunk differs")
        require(metric["input_tokens"] == metric["ce_count"] + case["rows"] and
                metric["input_tokens"] <= case["rows"] * case["max_length"],
                "Evaluation is not bounded all-target one-document-per-row CE")
        for occurrence, row in enumerate(metric["document_records"]):
            require(row["batch_index"] == occurrence // batch_size and row["row_index"] == occurrence % batch_size,
                    "Evaluation window order/batching differs")
            require(row["ce_count"] < case["max_length"], "Evaluation window exceeds length bound")
        require(paired._window_signature(metric) == paired._window_signature(values[0]),
                "Per-pass document/target alignment differs")
    return container


def validate_report(report, *, require_complete=True):
    require(report.get("schema") == SCHEMA, "Unexpected O5d diagnostic schema")
    if require_complete:
        require(report.get("status") == "passed" and report.get("finished_utc"), "Diagnostic is not completed and passing")
    else:
        require(report.get("status") in ("running", "passed"), "Diagnostic failed or has unknown status")
    configuration = report.get("configuration", {})
    require(configuration.get("precision") == "bf16_mixed" and configuration.get("batch_size") == 8,
            "Frozen BF16 evaluation batch differs")
    require(configuration.get("bootstrap_repetitions") == paired.BOOTSTRAP_RESAMPLES and
            configuration.get("bootstrap_seed") == paired.BOOTSTRAP_SEED,
            "Declared bootstrap settings differ from the fixed reporter settings")
    _hashes(report.get("source_hashes"), "model source")
    _hashes(report.get("diagnostic_source_hashes"), "diagnostic source")
    endpoints = report.get("endpoints", {})
    require(set(endpoints) == set(ENDPOINTS), "Require source and repaired mixed endpoints")
    frozen = None
    for name, endpoint in endpoints.items():
        before = _hashes(endpoint.get("state_before"), f"{name} initial state")
        after = _hashes(endpoint.get("state_after"), f"{name} final state")
        require(endpoint.get("weights_unchanged") is True and before == after,
                "Evaluation changed model or buffer bytes")
        require(FUSION_NAMES <= set(before) and "backbone.fusion.output_scale" in before,
                "Missing fusion matrices or fixed output scale")
        current_frozen = {k: v for k, v in before.items() if k not in FUSION_NAMES}
        require(frozen is None or current_frozen == frozen, "Native backbone or fixed scale differs between endpoints")
        frozen = current_frozen
        checkpoint = endpoint.get("checkpoint", {})
        require(isinstance(checkpoint.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"])
                and paired._integer(checkpoint.get("size_bytes"), 1), "Endpoint checkpoint identity is missing")
    cases = report.get("cases", [])
    expected = {(endpoint, section, passes) for endpoint in ENDPOINTS for section in SECTIONS for passes in (2, 3, 4, None)}
    seen, by_selection, ordinary = set(), {}, {}
    for row in cases:
        case = row.get("case", {})
        identity = (row.get("endpoint"), case.get("section"), case.get("passes"))
        require(identity in expected and identity not in seen, "Unexpected or duplicate endpoint/selection/execution")
        seen.add(identity)
        require(case == dict(section=identity[1], beta=1., passes=identity[2], rows=SECTIONS[identity[1]][0],
                             max_length=SECTIONS[identity[1]][1]), "Case differs from fixed selection/grid")
        require(set(row.get("metrics", {})) == set(SPLITS), "Evaluation must use only both development splits")
        require(paired._finite(row.get("elapsed_seconds")) and row["elapsed_seconds"] >= 0,
                "Missing finite case duration")
        durations = row.get("split_elapsed_seconds", {})
        require(set(durations) == set(SPLITS) and all(paired._finite(v) and v >= 0 for v in durations.values()),
                "Missing finite split durations")
        for split, container in row["metrics"].items():
            _validate_container(container, case, configuration["batch_size"])
            key = (case["section"], split)
            signature = paired._window_signature(container["passes"][0])
            require(key not in by_selection or by_selection[key] == signature,
                    "Endpoints/executions have mismatched document/target selections")
            by_selection[key] = signature
            if case["passes"] is not None:
                metric = container["passes"][0]
                if key in ordinary:
                    prior = ordinary[key]
                    require(metric["ce_count"] == prior["ce_count"] and metric["next_token_correct"] == prior["next_token_correct"]
                            and abs(metric["mean_nll"] - prior["mean_nll"]) <= 2e-6,
                            "Shared frozen ordinary-pass scores changed")
                ordinary[key] = metric
    require(seen == expected, "Diagnostic omits fixed endpoint/selection/execution cases")
    for split in SPLITS:
        full = by_selection[("full_context", split)][:32]
        expected_short = [(b, r, doc, min(count, 63)) for b, r, doc, count in full]
        require(by_selection[("short_prefix", split)] == expected_short,
                "Short prefixes differ from truncated first32 full-context windows")
    return report


def summarize_cases(report):
    validate_report(report, require_complete=False)
    result = {}
    for section, (rows, length) in SECTIONS.items():
        section_rows = [r for r in report["cases"] if r["case"]["section"] == section]
        result[section] = {"rows": rows, "max_length": length, "splits": {}}
        for split in SPLITS:
            by_endpoint = {endpoint: {execution(r["case"]): r["metrics"][split]
                for r in section_rows if r["endpoint"] == endpoint} for endpoint in ENDPOINTS}
            ordinary = by_endpoint["source"]["K2"]["passes"][0]
            final = {endpoint: {key: value["passes"][-1] for key, value in cases.items()}
                     for endpoint, cases in by_endpoint.items()}
            select_keys = ("mean_nll", "perplexity", "next_token_accuracy", "ce_count", "next_token_correct")
            item = {"ordinary": {k: ordinary[k] for k in select_keys},
                "document_clusters": len(paired._clusters(ordinary)), "ce_count": ordinary["ce_count"],
                "input_tokens": ordinary["input_tokens"], "endpoints": {}, "mixed_minus_source": {},
                "execution_gap_interaction": {}}
            for endpoint in ENDPOINTS:
                metrics = final[endpoint]
                item["endpoints"][endpoint] = {"final_metrics": {name: {k: metric[k] for k in select_keys}
                        for name, metric in metrics.items()},
                    "versus_ordinary": {name: _contrast(metric, ordinary) for name, metric in metrics.items()},
                    "versus_K2": {name: _contrast(metrics[name], metrics["K2"]) for name in EXECUTIONS[1:]}}
            for name in EXECUTIONS:
                item["mixed_minus_source"][name] = _contrast(final["mixed"][name], final["source"][name])
            for name in EXECUTIONS[1:]:
                metrics = {"mixed_execution": final["mixed"][name], "mixed_K2": final["mixed"]["K2"],
                           "source_execution": final["source"][name], "source_K2": final["source"]["K2"]}
                item["execution_gap_interaction"][name] = paired.document_cluster_contrast(
                    {k: _numeric(v) for k, v in metrics.items()},
                    {"mixed_execution": 1., "mixed_K2": -1., "source_execution": -1., "source_K2": 1.})
            result[section]["splits"][split] = item
    return result


def _interval(value):
    low, high = value["ci95"]
    return f"{value['estimate']:+.6f} [{low:+.6f}, {high:+.6f}]"


def markdown(report):
    summary = report.get("summary") or summarize_cases(report)
    lines = ["# O5d: does fusion adaptation transfer to sequential feedback?", "", QUALIFICATION, "",
             "All NLL values score an individual final pass or the exact online state; the summed training objective is not reported.", ""]
    for section, value in summary.items():
        lines += [f"## {'Short-prefix diagnostic' if section == 'short_prefix' else 'Full-context extension'}", "",
            f"First {value['rows']} development windows per domain, maximum {value['max_length']} input tokens per window; batch 8.", "",
            "| Endpoint / execution | Code NLL | WikiText NLL | Code accuracy | WikiText accuracy |",
            "| --- | ---: | ---: | ---: | ---: |"]
        ordinary = [value["splits"][split]["ordinary"] for split in SPLITS]
        lines.append(f"| Shared frozen ordinary | {ordinary[0]['mean_nll']:.6f} | {ordinary[1]['mean_nll']:.6f} | {ordinary[0]['next_token_accuracy']:.3%} | {ordinary[1]['next_token_accuracy']:.3%} |")
        for endpoint in ENDPOINTS:
            for name in EXECUTIONS:
                metrics = [value["splits"][split]["endpoints"][endpoint]["final_metrics"][name] for split in SPLITS]
                lines.append(f"| {'O5b source' if endpoint == 'source' else 'O5c mixed'} / {name} | {metrics[0]['mean_nll']:.6f} | {metrics[1]['mean_nll']:.6f} | {metrics[0]['next_token_accuracy']:.3%} | {metrics[1]['next_token_accuracy']:.3%} |")
        lines += ["", "Paired NLL differences with original-document bootstrap 95% intervals:", "",
            "| Contrast | Code | WikiText |", "| --- | ---: | ---: |"]
        contrasts = [(f"Mixed minus source, {name}", lambda s, n=name: s["mixed_minus_source"][n]) for name in EXECUTIONS]
        contrasts += [(f"{endpoint.capitalize()} online minus K2", lambda s, e=endpoint: s["endpoints"][e]["versus_K2"]["online"]) for endpoint in ENDPOINTS]
        contrasts += [("Mixed online minus ordinary", lambda s: s["endpoints"]["mixed"]["versus_ordinary"]["online"]),
                      ("(Online − K2) mixed minus source", lambda s: s["execution_gap_interaction"]["online"])]
        for label, obtain in contrasts:
            lines.append(f"| {label} | {_interval(obtain(value['splits']['dev']))} | {_interval(obtain(value['splits']['retention_dev']))} |")
        counts = "; ".join(f"{'code' if split == 'dev' else 'WikiText'}: {v['document_clusters']} original documents, {v['ce_count']:,} CE targets"
                           for split, v in value["splits"].items())
        lines += ["", f"Selection: {counts}. Intervals quantify evaluation-document variability only, not training-seed uncertainty. Short selections contain few original documents.", ""]
    lines += ["## Execution and provenance", "", "All model and buffer hashes were unchanged at both endpoints. Their native backbone and fixed fusion scale are identical; only the fusion matrices differ.", "",
              "| Endpoint | Checkpoint SHA256 |", "| --- | --- |"]
    for name in ENDPOINTS:
        lines.append(f"| {name} | `{report['endpoints'][name]['checkpoint']['sha256']}` |")
    lines += ["", "Case durations are recorded in report.json. They are eager evaluation wall times, not an optimized throughput comparison.", "",
              "[Finite/online figure](finite-online.pdf) · [Adaptation transfer figure](adaptation-transfer.pdf) · [Full report](report.json)", ""]
    url = report.get("wandb", {}).get("run_url")
    if url:
        lines += [f"[W&B evaluation run]({url})", ""]
    return "\n".join(lines)


def write_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = report.get("summary") or summarize_cases(report)
    output = Path(output)
    labels = {"dev": "Code", "retention_dev": "WikiText"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    for row, (section, value) in enumerate(summary.items()):
        for column, split in enumerate(SPLITS):
            axis = axes[row, column]
            item = value["splits"][split]
            for endpoint, label, color in (("source", "O5b source", "#a25b24"), ("mixed", "O5c mixed", "#187b9f")):
                metrics = item["endpoints"][endpoint]["final_metrics"]
                axis.plot(range(4), [metrics[name]["mean_nll"] for name in EXECUTIONS], "o-", label=label, color=color)
            axis.axhline(item["ordinary"]["mean_nll"], ls="--", color="#555555", label="Frozen ordinary")
            axis.set(xticks=range(4), xticklabels=("K2", "K3", "K4", "Online"), ylabel="Token NLL (nats)",
                title=f"{labels[split]} · {value['rows']} windows · max {value['max_length']} tokens")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle("Fixed beta 1 · identical targets within each panel · no weight updates")
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"finite-online.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    for row, (section, value) in enumerate(summary.items()):
        for column, split in enumerate(SPLITS):
            axis = axes[row, column]
            contrasts = value["splits"][split]["mixed_minus_source"]
            estimates = [contrasts[name]["estimate"] for name in EXECUTIONS]
            lower = [max(0., estimates[i] - contrasts[name]["ci95"][0]) for i, name in enumerate(EXECUTIONS)]
            upper = [max(0., contrasts[name]["ci95"][1] - estimates[i]) for i, name in enumerate(EXECUTIONS)]
            axis.errorbar(range(4), estimates, yerr=[lower, upper], fmt="o-", color="#187b9f", capsize=4)
            axis.axhline(0, ls="--", color="#555555")
            axis.set(xticks=range(4), xticklabels=("K2", "K3", "K4", "Online"), ylabel="Mixed − source NLL (nats)",
                title=f"{labels[split]} · {value['rows']} windows · max {value['max_length']} tokens")
            axis.grid(alpha=.2)
    fig.suptitle("Fusion adaptation benefit across executions · paired document 95% intervals\nNegative favors mixed; intervals exclude training-seed uncertainty", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"adaptation-transfer.{suffix}", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text())
    validate_report(report)
    report["summary"] = summarize_cases(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_figures(report, args.output_dir)
    (args.output_dir / "results.md").write_text(markdown(report))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
