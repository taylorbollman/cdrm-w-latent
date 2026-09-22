#!/usr/bin/env python3
"""Summarize bounded F2 health, complete-step capacity and graph diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import sha256_file, write_json

HEALTH_SCHEMA = "olmo-f2-health-capacity-v1"
GRAPH_SCHEMA = "olmo-f2-graph-probe-v1"
LOCALIZATION_SCHEMA = "olmo-f2-graph-localization-v1"
GRAPH_CASES = {"original_tokens_weights", "changed_tokens", "changed_tokens_and_weights",
               "two_replays_do_not_accumulate"}


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Require a finite {'positive ' if positive else ''}{name}")
    return value


def _sources(report, root):
    hashes = report.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Report lacks runtime source provenance")
    changed = []
    for name, digest in hashes.items():
        path = root/name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe runtime source path")
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            changed.append(name)
    if report["status"] in ("passed", "completed") and changed:
        raise ValueError("Passing report runtime source changed: "+", ".join(changed))
    return {"recorded": hashes, "matches_current": not changed, "changed_since_run": changed}


def _capacity(row, config):
    base = {key: row[key] for key in ("case", "checkpointing", "physical_batch", "status", "passed")}
    base["comfortable_limit_gib"] = config["comfortable_gib"]
    if row["status"] == "oom_capacity_limit":
        return base
    if row["status"] != "measured" or row.get("post_timing_state_health", {}).get("passed") is not True:
        raise ValueError("Capacity cell lacks a finite measured complete update")
    times = row["wall_seconds"]
    if len(times) != row["timed_updates"] or len(row["records"]) != row["warmup_updates"]+row["timed_updates"]:
        raise ValueError("Capacity timing/update counts differ")
    for value in times: _finite(value, "wall time", positive=True)
    median = statistics.median(times)
    if not math.isclose(median, row["median_wall_seconds"], rel_tol=1e-12):
        raise ValueError("Capacity median differs from raw timing samples")
    expected = row["valid_input_tokens_per_update"]/median
    if not math.isclose(expected, row["valid_input_tokens_per_second"], rel_tol=1e-10):
        raise ValueError("Capacity throughput differs from valid-token count and time")
    if not math.isclose(row["records"][-1]["counts"]["ce"]/median, row["ce_targets_per_second"], rel_tol=1e-10):
        raise ValueError("CE throughput differs from actual target count")
    for value in row["records"]:
        _finite(value["objective"], "capacity objective")
        _finite(value["gradient_norm_before_clip"], "capacity gradient norm")
    for name in ("peak_allocated_gib", "peak_reserved_gib"):
        _finite(row[name], name, positive=True)
    if row["within_comfortable_memory"] != (row["peak_allocated_gib"] <= config["comfortable_gib"]):
        raise ValueError("Comfortable-memory label differs from the configured bound")
    if row["physical_batch"] != row["case"]["batch_size"] or row["gradient_accumulation"] != 1 or row["cuda_graphs"] is not False:
        raise ValueError("Capacity batch/execution metadata differs")
    return base | {key: row[key] for key in ("warmup_updates", "timed_updates", "median_wall_seconds",
        "valid_input_tokens_per_second", "ce_targets_per_second", "valid_input_tokens_per_update",
        "peak_allocated_gib", "peak_reserved_gib", "within_comfortable_memory", "scope")} | {
        "total_preclip_norms": [record["gradient_norm_before_clip"] for record in row["records"]],
        "last_loss_means": row["records"][-1].get("loss_means", {})}


def _activation(row):
    attention = row["attention"]
    result = {key: row[key] for key in ("pass", "layer", "kind", "backend")}
    for name in ("input", "output", "query_post_rope", "temporary_key_post_rope", "permanent_key_post_rope"):
        result[name+"_rms"] = row[name]["rms"]
    result["attention"] = {key: attention[key] for key in ("scaled_logits", "absolute_scaled_logits",
        "maximum_probability", "entropy_fraction_of_uniform", "self_probability",
        "temporary_self_minus_best_history_logit", "eligible_head_queries", "finite")}
    if attention["finite"] is not True:
        raise ValueError("Nonfinite observed attention")
    return result


def _health(row):
    if (row.get("activation_finite") is not True or row.get("weights_rng_grad_buffers_unchanged") is not True
            or row["gradients"].get("passed") is not True):
        raise ValueError("Health row failed its numerical/neutrality contract")
    if row.get("neutrality") is not None and row["neutrality"].get("passed") is not True:
        raise ValueError("Observer changed the model result")
    gradients = row["gradients"]
    _finite(gradients["component_sum_relative_l2"], "component closure")
    _finite(row["losses"]["objective"], "health objective")
    for groups in [gradients["total"], *(component["groups"] for component in gradients["components"])]:
        if any(group.get("finite") is not True for group in groups.values()):
            raise ValueError("Nonfinite health gradient group")
    return {"case": row["case"], "losses": row["losses"], "gradients": gradients,
        "neutrality": row.get("neutrality"), "observed_layers": row["activations"]["observed_layers"],
        "activations": [_activation(value) for value in row["activations"]["records"]],
        "stack_pass_rms": [{"pass": value["pass"], "input": value["input"]["rms"],
            "final_normalized_output": value["final_normalized_output"]["rms"]}
            for value in row["activations"]["passes"]],
        "attention_reconstruction": row["activations"]["reconstruction"]}


def build_summary(paths, *, project_root=ROOT):
    output = {"schema": "olmo-f2-summary-v1", "status": "passed", "inputs": [],
        "health": [], "capacity": [], "checkpoint": [], "graphs": [], "graph_localization": [],
        "limitations": ["Bounded synthetic/text fixtures, not language-quality comparisons",
            "Only the recorded RT layer selection is covered; current runs select layer 0",
            "BF16 component-gradient closure is descriptive, not a newly widened acceptance budget",
            "Attention concentrations are detached FP32 reconstruction, not private fused or exact dyadic probabilities",
            "Complete-step capacity and stack-only CUDA graph timing are distinct measurements",
            "No combined FBT/NextLat graph, all-layer RT, or multi-GPU clearance"]}
    seen = set()
    checkpoint = None
    for path in paths:
        path = Path(path)
        if path.is_dir(): path = path/"report.json"
        if path.resolve() in seen: raise ValueError("Duplicate report input")
        seen.add(path.resolve())
        report = json.loads(path.read_text())
        schema, status = report.get("schema"), report.get("status")
        if schema not in (HEALTH_SCHEMA, GRAPH_SCHEMA, LOCALIZATION_SCHEMA) or not report.get("finished_utc"):
            raise ValueError("Require completed F2 reports")
        if schema == HEALTH_SCHEMA and status != "passed":
            raise ValueError("Core health/capacity/checkpoint report failed")
        if schema == GRAPH_SCHEMA and status not in ("passed", "failed", "capture_blocked"):
            raise ValueError("Graph diagnostic is incomplete")
        if schema == LOCALIZATION_SCHEMA and status not in ("completed", "failed", "capture_blocked"):
            raise ValueError("Graph localization is incomplete")
        if not report.get("checkpoint", {}).get("sha256"):
            raise ValueError("Missing native checkpoint provenance")
        if checkpoint is None: checkpoint = report["checkpoint"]
        elif checkpoint != report["checkpoint"]: raise ValueError("Reports used different native checkpoints")
        provenance = {"path": str(path), "sha256": sha256_file(path), "schema": schema, "status": status,
            "runtime": report.get("runtime"), "wandb": report.get("wandb"), "source": _sources(report, project_root)}
        if schema == HEALTH_SCHEMA:
            if not report.get("rows") or any(row.get("passed") is not True for row in report["rows"]):
                raise ValueError("Core report contains a failed or absent row")
            stage = report["config"]["stage"]
            protocol = project_root/"docs/reports/olmo1b-f2/protocol.md"
            if report.get("protocol_sha256") != sha256_file(protocol):
                raise ValueError("F2 protocol changed after the core run")
            provenance.update(stage=stage, configuration=report["config"], protocol_sha256=report["protocol_sha256"])
            if stage == "health":
                output["health"].extend(_health(row) | {"source_report": str(path)} for row in report["rows"])
            elif stage == "capacity":
                output["capacity"].extend(_capacity(row, report["config"]) | {"source_report": str(path)} for row in report["rows"])
            elif stage == "checkpoint":
                for row in report["rows"]:
                    if row.get("metrics_exact") is not True or row.get("boundary_exact") is not True:
                        raise ValueError("Activation-checkpoint complete-update parity failed")
                    output["checkpoint"].append({"case": row["case"], "metrics_exact": True,
                        "boundary_exact": True, "source_report": str(path)})
            else: raise ValueError("Unknown F2 core stage")
        elif schema == GRAPH_SCHEMA:
            graph = {"source_report": str(path), "status": status, "configuration": report["configuration"],
                "capture_succeeded": report.get("capture_succeeded", False), "stage": report.get("stage"),
                "comparisons": [{"name": row["name"], "passed": row["passed"],
                    "all_bitwise_equal": row["all_bitwise_equal"], "hidden": row["hidden"],
                    "failed_gradient_tensors": [name for name, value in row["gradients"].items() if not value["passed"]],
                    "max_gradient_relative_l2": max(value["relative_l2"] for value in row["gradients"].values())}
                    for row in report.get("comparisons", [])]}
            if status == "passed":
                if (report.get("capture_succeeded") is not True
                        or len(report["comparisons"]) != len(GRAPH_CASES)
                        or {row["name"] for row in report["comparisons"]} != GRAPH_CASES
                        or any(row["passed"] is not True for row in report["comparisons"])):
                    raise ValueError("Passing graph lacks complete equivalence checks")
                graph.update(timings=report["timings"], replay_speedup_wall=report["replay_speedup_wall"],
                    peak_allocated_gib_timing=report["peak_allocated_gib_timing"],
                    reserved_gib_after_capture=report["reserved_gib_after_capture"])
            else:
                graph.update(error_type=report.get("error_type"), error_message=report.get("error_message"),
                    disposition="historical capture blocker" if status == "capture_blocked" else "failed original graph check")
                if status == "failed": output["status"] = "core_passed_with_graph_qualifications"
            output["graphs"].append(graph)
        else:
            localization = {"source_report": str(path), "status": status,
                "configuration": report["configuration"], "capture_succeeded": report.get("capture_succeeded", False),
                "stage": report.get("stage"), "structural_invariants_passed": report.get("structural_invariants_passed"),
                "all_comparisons_within_original_budget": report.get("all_comparisons_within_original_budget"),
                "all_comparisons_bitwise_equal": report.get("all_comparisons_bitwise_equal"),
                "comparisons": [{key: value for key, value in row.items() if key != "gradients"}
                    for row in report.get("comparisons", [])], "dispatch": report.get("dispatch"),
                "limitations": report.get("limitations", []), "error_message": report.get("error_message")}
            if status == "completed":
                count = report["configuration"]["repeats"]
                expected_names = {f"eager_default_repeat_{i+1}" for i in range(count)} | {
                    f"graph_repeat_{i+1}" for i in range(count)} | {
                    "eager_capture_stream_vs_default_stream", "graph_vs_eager_default",
                    "graph_vs_eager_capture_stream", "eager_default_after_capture_vs_before",
                    "graph_vs_eager_default_after_capture"}
                if ({row["name"] for row in report["comparisons"]} != expected_names
                        or len(report["comparisons"]) != len(expected_names)):
                    raise ValueError("Completed localization omitted requested control comparisons")
                expected_pass = report["structural_invariants_passed"] and all(row["passed"] for row in report["comparisons"])
                if report["all_comparisons_within_original_budget"] != expected_pass:
                    raise ValueError("Localization outcome differs from individual comparisons")
            if status != "completed" or localization["all_comparisons_within_original_budget"] is not True:
                output["status"] = "core_passed_with_graph_qualifications"
            output["graph_localization"].append(localization)
        output["inputs"].append(provenance)
    if not output["inputs"]: raise ValueError("No F2 input reports")
    output["checkpoint_source"] = checkpoint
    return output


def _f(value, digits=4):
    return "—" if value is None else f"{value:.{digits}g}"


def markdown(summary):
    lines = ["# F2 bounded numerical health and execution", "", f"Status: **{summary['status']}**.", "",
        "These measurements test operation and resource use on one GPU. They do not measure language-model quality. "
        "Native Q/K normalization remains off; only the explicitly recorded RT layers and shapes are covered.", "",
        "## Objective and gradient scale", "",
        "FP32 parameters, gradients and Adam moments with BF16 autocast. FBT uses pass 0 plus the mean of extra-pass "
        "losses; each CE/latent/KL term keeps its own denominator. Component norms below include their objective weights. "
        "Repeated-VJP closure in BF16 is descriptive; no tolerance was widened for this report.", "",
        "| Case | B / T | Total objective | Native / fusion / predictor gradient norm | Component-sum relative L2 |",
        "|---|---:|---:|---|---:|"]
    for row in summary["health"]:
        case, gradient = row["case"], row["gradients"]
        lines.append(f"| {case['name']} | {case['batch_size']} / {case['length']} | {_f(row['losses']['objective'])} | "
            + " / ".join(_f(gradient["total"][group]["norm"]) for group in ("native", "fusion", "predictor"))
            + f" | {_f(gradient['component_sum_relative_l2'])} |")
    lines += ["", "| Case / pass | CE / latent / KL means | Weighted CE / latent / KL gradient norms |", "|---|---|---|"]
    for row in summary["health"]:
        for index, means in enumerate(row["losses"]["pass_means"]):
            components = {component["term"]: component["groups"]["all"]["norm"]
                for component in row["gradients"]["components"] if component["pass"] == index}
            lines.append(f"| {row['case']['name']} B{row['case']['batch_size']}/T{row['case']['length']} / {index} | "+" / ".join(_f(means.get(term)) for term in ("ce", "latent", "kl"))
                +" | "+" / ".join(_f(components.get(term)) for term in ("ce", "latent", "kl"))+" |")
    lines += ["", "## Observed attention scale", "",
        "Q/K RMS uses actual post-RoPE operands. Logits and concentrations are detached FP32 reconstructions, "
        "not the fused backend's private probabilities or exact mixed dyadic scores. Padding/future entries are "
        "excluded; concentration excludes queries with only one allowed key. Permanent K is used below; "
        "temporary-K RMS is also retained in summary.json.", "",
        "| Case / pass / layer (kind) | Q / permanent K RMS | Absolute logit p99 / max | Max probability p95 | Entropy / uniform mean |",
        "|---|---:|---:|---:|---:|"]
    for row in summary["health"]:
        for item in row["activations"]:
            attention = item["attention"]
            lines.append(f"| {row['case']['name']} B{row['case']['batch_size']}/T{row['case']['length']} / {item['pass']} / {item['layer']} ({item['kind']}) | "
                f"{_f(item['query_post_rope_rms'])} / {_f(item['permanent_key_post_rope_rms'])} | "
                f"{_f(attention['absolute_scaled_logits']['p99'])} / {_f(attention['absolute_scaled_logits']['max'])} | "
                f"{_f(attention['maximum_probability']['p95'])} | {_f(attention['entropy_fraction_of_uniform']['mean'])} |")
    lines += ["", "## Complete-step physical-batch measurements", "",
        "Includes forward, losses, backward, gradient clipping, AdamW and scheduler. Excludes fixture preparation, "
        "W&B and reporting. These short measurements use changing tokens and weights; physical batch and gradient "
        "accumulation must not be conflated. All measured cells here use accumulation 1. 'Comfortable' means only "
        "the declared allocated-memory bound, not a universal maximum batch or stability clearance.", "",
        "| Case / T | Physical B | Ordinary checkpointing | Median step s | Valid input tok/s | CE targets/s | Peak allocated / reserved GiB | Comfortable |",
        "|---|---:|---|---:|---:|---:|---:|---|"]
    for row in summary["capacity"]:
        label = f"{row['case']['name']} / {row['case']['length']}"
        if row["status"] != "measured":
            lines.append(f"| {label} | {row['physical_batch']} | {row['checkpointing']} | OOM | — | — | — | No |")
        else:
            lines.append(f"| {label} | {row['physical_batch']} | {row['checkpointing']} | {_f(row['median_wall_seconds'])} | "
                f"{_f(row['valid_input_tokens_per_second'], 6)} | {_f(row['ce_targets_per_second'], 6)} | "
                f"{_f(row['peak_allocated_gib'])} / {_f(row['peak_reserved_gib'])} | "
                f"{'Yes' if row['within_comfortable_memory'] else 'No'} (≤{row['comfortable_limit_gib']:g} GiB) |")
    if summary["capacity"]: lines += ["", "![Complete-step throughput](throughput.png)"]
    for row in summary["checkpoint"]:
        lines += ["", f"Activation checkpointing: **{row['case']['name']} B{row['case']['batch_size']}/T{row['case']['length']}** "
            "matched the complete update exactly, including model/optimizer/scheduler/counter boundary digests."]
    lines += ["", "## CUDA graph microbenchmark", "",
        "This captures only the unpadded native stack and a fixed hidden-state cotangent backward, with gradient "
        "buffers zeroed inside the graph. It excludes CE, readout projection, FBT, NextLat, input copies and optimizer "
        "work. A separate SGD update checks changed-weight replay. It is not end-to-end training throughput.", ""]
    for graph in summary["graphs"]:
        cfg = graph["configuration"]
        backend = cfg.get("backend", "math" if cfg.get("attention_backend") == "math" else "auto")
        deterministic = cfg.get("deterministic_algorithms", cfg.get("deterministic", "not explicitly set"))
        label = f"B{cfg['batch_size']}/T{cfg['length']}, RT layers {cfg['rt_layers']}, ordinary backend {backend}, deterministic {deterministic}"
        if graph["status"] == "passed":
            timing = graph["timings"]
            exact = all(row["all_bitwise_equal"] for row in graph["comparisons"])
            lines.append(f"- {label}: {_f(timing['eager']['median_wall_seconds'])} → "
                f"{_f(timing['graph_replay']['median_wall_seconds'])} s, **{graph['replay_speedup_wall']:.2f}×**. "
                f"Changed-input/weight and repeated-replay checks passed; all compared tensors bitwise equal: {exact}.")
        else:
            message = str(graph.get("error_message", "")).replace("\n", " ")
            lines.append(f"- {label}: **{graph['disposition']}** at {graph['stage']}. {message} No efficiency claim from this attempt.")
    if summary["graph_localization"]:
        lines += ["", "### Fixed-input localization controls", "",
            "Completed localization means the controls ran, not that they agreed. These controls retain the original "
            "budgets and hold tokens/weights fixed; they cannot substitute for changed-token/weight replay or erase "
            "the failed T512 attempt. They make no throughput claim.", "",
            "| B / T / RT layers / backend | Comparison | Within original budget | Bitwise equal | Max gradient relative L2 | Highest layer outside budget |",
            "|---|---|---|---|---:|---:|"]
        for item in summary["graph_localization"]:
            cfg = item["configuration"]
            label = f"{cfg['batch_size']} / {cfg['length']} / {cfg['rt_layers']} / {cfg['backend']}"
            for row in item["comparisons"]:
                lines.append(f"| {label} | {row['name']} | {row['passed']} | {row['all_bitwise_equal']} | "
                    f"{_f(row['max_gradient_relative_l2'])} | {row.get('highest_layer_outside_original_budget')} |")
            if item.get("dispatch"):
                observed = [name for name, value in item["dispatch"]["attention_dispatch"].items() if value["observed"]]
                lines += ["", f"{label}: observed dispatch categories `{', '.join(observed) or 'none established'}`; "
                    f"structural invariants passed: {item['structural_invariants_passed']}.", ""]
            if item["status"] != "completed":
                lines += ["", f"Localization **{item['status']}** at {item['stage']}: {item['error_message']}", ""]
    lines += ["", "## Evidence and scope", ""]
    for item in summary["inputs"]:
        link = item.get("wandb", {}).get("run_url")
        lines.append(f"- `{item['path']}` — {item['status']}"+(f"; [W&B]({link})" if link else "")
            +(f"; {len(item['source']['changed_since_run'])} recorded historical source differences" if not item["source"]["matches_current"] else "; runtime sources match"))
    lines += ["", "The JSON summary retains input hashes, configurations, gradient groups, selected-layer statistics and "
        "the full stated limitations. Raw reports remain authoritative. More layers, larger/different masks, combined "
        "CUDA graphs, and multi-GPU execution require their own checks.", ""]
    return "\n".join(lines)


def plot_capacity(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 5))
    groups = {}
    for row in summary["capacity"]:
        if row["status"] == "measured":
            key = (row["case"]["name"], row["case"]["length"], row["checkpointing"])
            groups.setdefault(key, []).append(row)
    for (name, length, checkpointing), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["physical_batch"])
        display = {"combined": "RT + FBT(K2) + NextLat", "rt": "RT"}.get(name, name)
        axis.plot([row["physical_batch"] for row in rows], [row["valid_input_tokens_per_second"] for row in rows],
            marker="o", linestyle="--" if checkpointing else "-",
            label=display+(" + ordinary checkpointing" if checkpointing else ""))
    if groups:
        axis.set_xscale("log", base=2)
        ticks = sorted({row["physical_batch"] for rows in groups.values() for row in rows})
        axis.set_xticks(ticks, [str(value) for value in ticks])
        outside = [row for rows in groups.values() for row in rows if not row["within_comfortable_memory"]]
        if outside:
            axis.scatter([row["physical_batch"] for row in outside],
                         [row["valid_input_tokens_per_second"] for row in outside],
                         marker="x", color="black", s=70, zorder=4,
                         label="Above configured memory cutoff")
        axis.legend(fontsize=8)
    else:
        axis.text(.5, .5, "No complete-step capacity cells supplied", ha="center", transform=axis.transAxes)
    lengths = sorted({row["case"]["length"] for row in summary["capacity"]})
    layers = sorted({index for row in summary["capacity"] for index in row["case"].get("rt_layers", [])})
    title = "OLMo-1B" + (" · RT layers " + ",".join(map(str, layers)) if layers else "")
    title += " · T" + ",".join(map(str, lengths)) if lengths else ""
    axis.set(xlabel="Physical batch size (accumulation 1)", ylabel="Valid input tokens / second",
        title=title+"\nComplete optimizer steps · no CUDA graphs")
    axis.grid(alpha=.25)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir/("throughput."+suffix), dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--historical-runtime-dir", "--diagnostic-runtime-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=ROOT/"docs/reports/olmo1b-f2")
    args = parser.parse_args(argv)
    summary = build_summary([*args.runtime_dir, *args.historical_runtime_dir])
    summary.update(generated_utc=datetime.now(timezone.utc).isoformat(), report_source_sha256=sha256_file(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_capacity(summary, args.output_dir)
    summary["figure_sha256"] = {name: sha256_file(args.output_dir/name) for name in ("throughput.pdf", "throughput.png")}
    write_json(args.output_dir/"summary.json", summary)
    (args.output_dir/"results.md").write_text(markdown(summary))
    print({"status": summary["status"], "output_dir": str(args.output_dir)}, flush=True)


if __name__ == "__main__":
    main()
