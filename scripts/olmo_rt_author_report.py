#!/usr/bin/env python3
"""CPU-only report selection for the native/author-derived RT block comparison."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_rt_efficiency_report import require, digest, regular, safe_source, worst
from scripts.olmo_rt_efficiency_retain import verify_operator_trace

RUNTIME = ".runtime/olmo-rt-author-comparison"
DOCS = "docs/reports/olmo-rt-author-comparison"
PROTOCOL = DOCS + "/protocol.md"
AUDIT = DOCS + "/author-port-audit.md"
SCHEMA = "olmo-rt-author-comparison-v1"
ESSENTIAL_SOURCES = {
    "scripts/olmo_rt_author_compare.py", "scripts/olmo_rt_author_event_probe.py",
    "cdrm/pretrained/olmo_author.py", "cdrm/pretrained/rt_block_resources.py",
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_rope.py", "cdrm/pretrained/olmo_tiled.py",
    "cdrm/pretrained/olmo_recurrent.py", "cdrm/pretrained/olmo_rt_kernels.py",
    "cdrm/pretrained/olmo_rt_backward_kernels.py", "cdrm/pretrained/olmo_rt_recompute_kernels.py",
}
OPERATIONAL = {
    "verify": {"candidate_initial_graph", "candidate_changed_inputs_overwrite",
               "complete_adamw_update_parity", "candidate_changed_weights"},
    "capacity": {"capacity_initial_graph", "capacity_changed_inputs_overwrite",
                 "capacity_changed_weights", "finite_complete_updates"},
}


def resolve_commit(root, revision):
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{7,40}", revision), "Require a Git hex commit")
    return subprocess.check_output(["git", "rev-parse", revision + "^{commit}"], cwd=root, text=True).strip()


@lru_cache(maxsize=1024)
def frozen_digest(root, revision, path):
    return hashlib.sha256(subprocess.check_output(["git", "show", revision + ":" + path], cwd=root)).hexdigest()


def gate_summary(raw):
    expected = OPERATIONAL[raw["configuration"]["stage"]]
    grouped = {"operational": [], "compatibility": [], "diagnostic": []}
    for check in raw["checks"]:
        category = ("operational" if check["name"] in expected else
                    "compatibility" if check.get("gate", True) else "diagnostic")
        small = {k: v for k, v in check.items() if not isinstance(v, (dict, list))}
        small["gate"] = check.get("gate", True)
        for field in ("output", "input_gradient", "mse", "budgets", "reference_execution", "candidate_execution"):
            if field in check:
                small[field] = check[field]
        if check.get("gradients"):
            small["gradients"] = {"count": len(check["gradients"]),
                "worst_relative_l2": worst(check["gradients"], "relative_l2"),
                "worst_max_relative": worst(check["gradients"], "max_relative")}
        grouped[category].append(small)
    for category in tuple(grouped):
        checks = grouped[category]
        missing = sorted(expected - {c["name"] for c in checks}) if category == "operational" else []
        grouped[category] = {"checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks),
            "missing": missing, "all_observed_passed": all(c["passed"] for c in checks) if checks else None,
            "complete_and_passed": bool(checks) and not missing and all(c["passed"] for c in checks)}
    return grouped


def validate_finished(raw):
    require(raw.get("schema") == SCHEMA and raw.get("finished_utc")
            and raw.get("status") in {"passed", "failed", "oom"}, "Require a finished author-comparison report")
    require(type(raw.get("physical_optimizer_updates")) is int and raw["physical_optimizer_updates"] >= 0,
            "Invalid physical optimizer-update count")
    config, checks = raw.get("configuration", {}), raw.get("checks")
    require(config.get("stage") in OPERATIONAL and config.get("backend") in {"native", "author"},
            "Unknown comparison stage/backend")
    require(isinstance(checks, list) and all(isinstance(c.get("name"), str)
        and type(c.get("passed")) is bool and type(c.get("gate", True)) is bool for c in checks), "Invalid checks")
    require(len({c["name"] for c in checks}) == len(checks), "Duplicate checks")
    if raw["status"] != "passed":
        require(raw.get("error", {}).get("type") and isinstance(raw["error"].get("message"), str),
                "Failed/OOM report lacks its explicit error")
        return
    require(raw.get("stage") == "complete" and checks and all(c["passed"] for c in checks if c.get("gate", True)),
            "Passing report contains a failed gate or incomplete stage")
    require(raw.get("checkpoint", {}).get("sha256"), "Passing report lacks checkpoint provenance")
    require(gate_summary(raw)["operational"]["complete_and_passed"], "Passing report lacks operational checks")
    require(raw["physical_optimizer_updates"] == (6 if config["stage"] == "verify" else 8),
            "Passing report has incomplete optimizer updates")
    if config["stage"] == "verify":
        expected = {"author_vs_native_" + config["precision"]}
        if config["length"] == 32:
            expected |= {name + "_fp32_vs_native_scan" for name in ("author_scan", "author", "native")}
            if config["precision"] == "bf16_mixed":
                expected |= {name + "_bf16_vs_fp32_diagnostic" for name in ("author", "native")}
        require(expected <= {check["name"] for check in checks}, "Passing verify lacks planned raw comparisons")
    if config["stage"] == "capacity":
        require(len(raw.get("preparation_records", [])) == 3 and len(raw.get("timed_records", [])) == 5,
                "Passing capacity report lacks complete update records")
        for key, count in (("full_update", 5), ("forward_loss_backward", 3)):
            timing = raw[key]
            for clock in ("wall", "cuda"):
                samples = timing[clock + "_seconds"]
                require(len(samples) == count and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples),
                        "Invalid timing sample count/value")
                require(math.isclose(timing["median_" + clock + "_seconds"], statistics.median(samples), rel_tol=1e-12),
                        "Timing median differs from samples")
        if config.get("region_events"):
            require(len(raw.get("region_timings", [])) == 5 and all(math.isfinite(row[key]) and row[key] > 0
                for row in raw["region_timings"] for key in ("forward_objective_seconds", "backward_seconds")),
                "Missing or invalid forward/backward event samples")
        require(math.isclose(raw["input_tokens_per_second"],
            config["batch_size"] * config["length"] / raw["full_update"]["median_wall_seconds"], rel_tol=1e-12),
            "Throughput differs from complete-update wall time")


def load_run(root, name, revision):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name), "Unsafe run name")
    revision = resolve_commit(root, revision)
    path = regular(root / RUNTIME / name / "report.json", root)
    report_bytes = path.read_bytes()
    raw = json.loads(report_bytes)
    validate_finished(raw)
    require(resolve_commit(root, raw.get("runtime_commit")) == revision, "Selected and recorded runtime commits differ")
    hashes = raw.get("source_hashes")
    require(isinstance(hashes, dict) and ESSENTIAL_SOURCES <= hashes.keys(), "Missing essential runtime sources")
    differences = {}
    for source, expected in hashes.items():
        safe_source(source)
        snapshot = regular(path.parent / "source-snapshot" / source, root)
        require(digest(snapshot) == expected == frozen_digest(str(root), revision, source),
                "Source snapshot differs from report or frozen commit: " + source)
        current = digest(root / source) if (root / source).is_file() else None
        if current != expected:
            differences[source] = {"reported_sha256": expected, "current_sha256": current}
    for local, project, key in (("protocol.md", PROTOCOL, "protocol_sha256"),
                                ("author-port-audit.md", AUDIT, "audit_sha256")):
        require(digest(regular(path.parent / local, root)) == raw.get(key) == frozen_digest(str(root), revision, project),
                "Frozen protocol/audit snapshot mismatch: " + local)
    row = {"name": name, "runtime_commit": revision, "report_path": path.relative_to(root).as_posix(),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(), "status": raw["status"],
        "physical_optimizer_updates": raw["physical_optimizer_updates"], "source_pairs_checked": len(hashes),
        "stage": raw["configuration"]["stage"], "backend": raw["configuration"]["backend"],
        "gate_groups": gate_summary(raw), "current_source_differences": differences,
        "wandb_url": raw.get("wandb", {}).get("run_url"),
        "compiler_final": raw.get("compiler_final"), "compiler_on_exit": raw.get("compiler_on_exit")}
    if "error" in raw:
        row["error"] = raw["error"]
    trace = verify_operator_trace(raw, path.parent)
    if trace:
        row["profile"] = trace
    return row, raw


def event_preflight(root, *, required=False):
    path = root / RUNTIME / "event-preflight.json"
    if not path.exists():
        require(not required, "Selected event timings require event-preflight evidence")
        return None
    regular(path, root)
    raw = json.loads(path.read_text())
    require(raw.get("schema") == "olmo-author-event-preflight-v1" and raw.get("status") in {"passed", "failed"}
            and isinstance(raw.get("checks"), list), "Invalid event preflight")
    passed = bool(raw["checks"]) and all(c.get("passed") is True for c in raw["checks"])
    require(raw["status"] != "passed" or passed, "Passing preflight contains failed checks")
    require(not required or (passed and raw["status"] == "passed"), "Selected event timings require a passing preflight")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path), **raw,
        "provenance_scope": "Standalone instrumentation evidence; raw preflight has no independent runtime commit. Probe source is retained in selected run snapshots."}


def comparison_signature(raw):
    c = raw["configuration"]
    # Backend, backward memory and author arithmetic are intentional labeled
    # differences. Fixture generation, geometry and measurement boundaries are
    # held fixed within each shape/precision/event-instrumentation group.
    shared = {key: c.get(key) for key in ("model", "layers", "batch_size", "length", "precision", "seed",
        "checkpoint_blocks", "input_distribution", "training_objective", "raw_gradient_cotangent",
        "capture_boundary", "optimizer_boundary", "ordinary_layers", "fbt", "nextlat", "embedding_readout",
        "physical_batch", "accumulation", "world_size", "tf32", "region_events", "preparation_updates",
        "capture_warmup_backwards", "timed_updates", "graph_timing_samples")}
    payload = {"shared_configuration": shared, "checkpoint": raw.get("checkpoint"),
        "runtime": raw.get("runtime"), "determinism": raw.get("determinism"),
        "fixtures": raw.get("fixtures"), "fixture_order": raw.get("fixture_order")}
    if raw["status"] == "passed":
        require(isinstance(payload["fixtures"], list) and len(payload["fixtures"]) == 2
                and isinstance(payload["fixture_order"], str), "Successful capacity lacks recorded fixture provenance")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def comparison_groups(capacity):
    groups = defaultdict(list)
    for row in capacity:
        if row["status"] == "passed":
            groups[(row["layers"], row["batch_size"], row["length"], row["precision"], row["region_events"])].append(row)
    result = []
    for (layers, batch, length, precision, events), rows in sorted(groups.items()):
        require(len({row["comparison_signature"] for row in rows}) == 1,
                "Capacity comparison fixtures, geometry, runtime or measurement boundaries differ")
        result.append({"layers": layers, "batch_size": batch, "length": length, "precision": precision,
            "region_events": events, "runs": [row["name"] for row in rows],
            "comparison_signature": rows[0]["comparison_signature"],
            "fixture_scope": "Same recorded generation settings, shapes, order and norm statistics; raw reports do not independently hash fixture tensor bytes."})
    return result


def summarize(names, revision, *, overrides=None, root=ROOT):
    require(names and len(set(names)) == len(names), "Require explicit unique run names")
    overrides = overrides or {}
    require(set(overrides) <= set(names), "A runtime override names an unselected run")
    revision = resolve_commit(root, revision)
    rows, raws = zip(*(load_run(root, name, overrides.get(name, revision)) for name in names))
    checkpoints = {json.dumps(raw["checkpoint"], sort_keys=True) for raw in raws if "checkpoint" in raw}
    require(len(checkpoints) <= 1, "Selected runs use different checkpoint provenance")
    summary = {"schema": "olmo-rt-author-summary-v1", "status": "completed", "runtime_commit": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(), "runs": list(rows),
        "physical_optimizer_updates": sum(row["physical_optimizer_updates"] for row in rows),
        "source_pairs_checked": sum(row["source_pairs_checked"] for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)), "capacity": [], "resource_cards": [],
        "checkpoint": json.loads(next(iter(checkpoints))) if checkpoints else None,
        "qualification": "Explicit finished selection, not an all-passed assertion. Compatibility and operational checks are separate. Block-stack throughput excludes embeddings/readout and is not language-model training throughput."}
    for row, raw in zip(rows, raws):
        c = raw["configuration"]
        if raw.get("resources"):
            summary["resource_cards"].append({"name": row["name"], "status": row["status"], **raw["resources"]})
        if c["stage"] != "capacity":
            continue
        regions = raw.get("region_timings", [])
        summary["capacity"].append({"name": row["name"], "status": row["status"],
            "runtime_commit": row["runtime_commit"],
            "comparison_signature": comparison_signature(raw),
            "fixtures": raw.get("fixtures"), "fixture_order": raw.get("fixture_order"),
            **{key: c.get(key) for key in ("backend", "layers", "batch_size", "length", "precision", "native_arm",
                "native_backward", "author_precision", "bwd_mlp_chunks", "region_events")},
            **{key: raw.get(key) for key in ("input_tokens_per_second", "layer_token_work_per_second",
                "estimated_matrix_tflops_per_second", "forward_loss_backward_tokens_per_second",
                "full_update", "forward_loss_backward", "setup_memory", "steady_memory", "resources", "parameters")},
            "region_timings": regions,
            "event_region_medians": {key: statistics.median(r[key] for r in regions)
                for key in ("forward_objective_seconds", "backward_seconds")} if regions else None,
            "event_scope": "Forward includes scalar FP32 MSE; backward is inside the same graph. Complete-update wall time also includes copy, clipping and optimizer."})
    summary["comparison_groups"] = comparison_groups(summary["capacity"])
    summary["event_preflight"] = event_preflight(root, required=any(raw["status"] == "passed"
        and raw["configuration"].get("region_events") for raw in raws))
    return summary


def plot_capacity(summary, directory):
    rows = [row for row in summary["capacity"] if row["status"] == "passed"]
    if not rows:
        return []
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    result = []
    primary = [r for r in rows if r["region_events"] and (
        r["backend"] == "native" and r["native_arm"] == "both" and r["native_backward"] == "recompute"
        or r["backend"] == "author" and r["author_precision"] == "author_legacy" and r["bwd_mlp_chunks"] == 4)]
    depths = sorted({r["layers"] for r in primary})

    def save(fig, stem):
        for extension in ("png", "pdf"):
            name = stem + "." + extension
            fig.savefig(directory / name, dpi=180, bbox_inches="tight")
            result.append(name)
        plt.close(fig)

    def values(samples):
        center = statistics.median(samples)
        return center, center - min(samples), max(samples) - center

    if depths:
        fig, axes = plt.subplots(1, len(depths), figsize=(4.5 * len(depths), 4.5), squeeze=False)
        for axis, depth in zip(axes[0], depths):
            batches = sorted({r["batch_size"] for r in primary if r["layers"] == depth})
            for backend, shift, color in (("native", -.2, "#3881b7"), ("author", .2, "#249b73")):
                positions, medians, lower, upper = [], [], [], []
                for index, batch in enumerate(batches):
                    samples = [r["input_tokens_per_second"] / 1000 for r in primary
                               if (r["layers"], r["batch_size"], r["backend"]) == (depth, batch, backend)]
                    if samples:
                        center, lo, hi = values(samples)
                        positions.append(index + shift)
                        medians.append(center)
                        lower.append(lo)
                        upper.append(hi)
                if positions:
                    bars = axis.bar(positions, medians, width=.36, color=color, label=backend,
                                    yerr=[lower, upper], capsize=3)
                    axis.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
            axis.set_xticks(range(len(batches)), [str(b) for b in batches])
            axis.set_xlim(-.6, len(batches) - .4)
            axis.set_xlabel("Physical batch")
            axis.set_ylabel("Block-stack input tokens/s (thousands)")
            axis.set_title(f"{depth} RT block" + ("s" if depth != 1 else ""))
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            axis.margins(y=.16)
            axis.legend(loc="upper left")
        fig.suptitle("Native both/recompute vs author legacy · T512 · H10080GB")
        fig.text(.5, .01, "Median of run medians; whiskers span repeats where present. Five timed updates per run.\n"
                 "Isolated Gaussian/MSE fixture; no embeddings, CE, FBT or NextLat. Not language-model throughput.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .10, 1, .94))
        save(fig, "throughput")

    bridge = [("Native\ncontrol", "native", "control", "recompute"),
              ("Native\nRoPE reuse", "native", "rope", "recompute"),
              ("Native\nboth", "native", "both", "recompute"),
              ("Native both\nmaterialized", "native", "both", "materialized"),
              ("Author\nlegacy", "author", None, None)]
    labels, centers, lower, upper, colors = [], [], [], [], []
    for label, backend, arm, backward in bridge:
        samples = [r["input_tokens_per_second"] / 1000 for r in rows
                   if r["layers"] == 1 and r["batch_size"] == 32 and r["region_events"] and r["backend"] == backend
                   and ((backend == "native" and r["native_arm"] == arm and r["native_backward"] == backward)
                        or (backend == "author" and r["author_precision"] == "author_legacy" and r["bwd_mlp_chunks"] == 4))]
        if samples:
            center, lo, hi = values(samples)
            labels.append(label)
            centers.append(center)
            lower.append(lo)
            upper.append(hi)
            colors.append("#249b73" if backend == "author" else "#3881b7")
    if labels:
        fig, axis = plt.subplots(figsize=(9, 4.7))
        bars = axis.bar(range(len(labels)), centers, color=colors, yerr=[lower, upper], capsize=3)
        axis.bar_label(bars, fmt="%.2f", padding=4)
        axis.set_xticks(range(len(labels)), labels)
        axis.set_ylabel("Block-stack input tokens/s (thousands)")
        axis.set_title("One block · B32/T512 · native optimization bridge")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.margins(y=.16)
        fig.text(.5, .015, "Median of run medians; whiskers span repeats where present. Isolated block fixture, not LM throughput.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, 1))
        save(fig, "native-bridge")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--run-commit", action="append", default=[], metavar="NAME=COMMIT")
    parser.add_argument("--output-dir", type=Path, default=ROOT / DOCS)
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.run_commit:
        require("=" in item, "Per-run override must be NAME=COMMIT")
        name, revision = item.split("=", 1)
        require(name not in overrides, "Duplicate per-run runtime override")
        overrides[name] = revision
    summary = summarize(args.runs, args.runtime_commit, overrides=overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary["plots"] = plot_capacity(summary, args.output_dir)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "statuses", "physical_optimizer_updates", "source_pairs_checked")}))


if __name__ == "__main__":
    main()
