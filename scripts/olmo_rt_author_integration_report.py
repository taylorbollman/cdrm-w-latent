#!/usr/bin/env python3
"""CPU-only explicit evidence selection for bounded native/author LM integration."""
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
from scripts.olmo_rt_efficiency_report import require, digest, regular, safe_source, check_summary

RUNTIME = ".runtime/olmo-rt-author-integration"
DOCS = "docs/reports/olmo-rt-author-integration"
PROTOCOL = DOCS + "/protocol.md"
SCHEMA = "olmo-rt-author-integration-v1"
ESSENTIAL_SOURCES = {
    "scripts/olmo_rt_author_integration.py", "scripts/olmo_rt_efficiency.py",
    "scripts/olmo_rt_author_compare.py", "scripts/olmo_f1_common.py", "scripts/olmo_lm_common.py",
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_author.py", "cdrm/pretrained/olmo_rope.py",
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_static.py", "cdrm/pretrained/olmo_fbt.py",
    "cdrm/pretrained/static_training.py", "cdrm/pretrained/static_nextlat.py",
    "cdrm/pretrained/nextlat.py", "cdrm/pretrained/fbt_training.py",
    "cdrm/pretrained/resource_estimates.py", "cdrm/pretrained/rt_block_resources.py",
    "cdrm/pretrained/olmo_rt_kernels.py", "cdrm/pretrained/olmo_rt_backward_kernels.py",
    "cdrm/pretrained/olmo_rt_recompute_kernels.py",
}
OPERATIONAL = {
    "verify": {"ordinary_flash_dispatch", "candidate_initial_graph", "candidate_changed_tokens_overwrite",
               "complete_adamw_update_parity", "candidate_changed_weights"},
    "capacity": {"capacity_initial_graph", "capacity_changed_tokens_overwrite",
                 "capacity_changed_weights", "finite_complete_updates"},
}
BATCH_FIELDS = {"input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"}


def resolve_commit(root, revision):
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{7,40}", revision), "Require a Git hex commit")
    return subprocess.check_output(["git", "rev-parse", revision + "^{commit}"], cwd=root, text=True).strip()


@lru_cache(maxsize=1024)
def frozen_digest(root, revision, path):
    return hashlib.sha256(subprocess.check_output(["git", "show", revision + ":" + path], cwd=root)).hexdigest()


def gate_summary(raw):
    expected = OPERATIONAL[raw["configuration"]["stage"]]
    groups = {"operational": [], "compatibility": [], "diagnostic": []}
    for check in raw["checks"]:
        category = "operational" if check["name"] in expected else "compatibility" if check.get("gate", True) else "diagnostic"
        groups[category].append(check_summary(check))
    for category, checks in tuple(groups.items()):
        required = expected if category == "operational" else (
            {"author_vs_native_bf16_mixed"} if category == "compatibility" and raw["configuration"]["stage"] == "verify" else set())
        missing = sorted(required - {check["name"] for check in checks})
        groups[category] = {"checks": checks, "passed": sum(check["passed"] for check in checks),
            "total": len(checks), "missing": missing,
            "all_observed_passed": all(check["passed"] for check in checks) if checks else None,
            "complete_and_passed": bool(checks) and not missing and all(check["passed"] for check in checks)}
    return groups


def tensor_manifest(value, *, shape=None, dtype=None):
    require(isinstance(value, dict) and isinstance(value.get("shape"), list)
        and all(type(n) is int and n > 0 for n in value["shape"])
        and isinstance(value.get("dtype"), str)
        and isinstance(value.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "Invalid tensor hash manifest")
    require(shape is None or value["shape"] == shape, "Fixture tensor shape differs")
    require(dtype is None or value["dtype"] == dtype, "Fixture tensor dtype differs")


def validate_fixtures(raw):
    c = raw["configuration"]
    shape = [c["batch_size"], c["length"]]
    batches = [raw.get("initial_batch")]
    if c["stage"] == "capacity":
        require(isinstance(raw.get("timed_batches"), list) and len(raw["timed_batches"]) == 5,
                "Capacity lacks five timed fixture hashes")
        require(isinstance(raw.get("preparation_batches"), list) and len(raw["preparation_batches"]) == 3,
                "Capacity lacks three preparation fixture hashes")
        require(raw["preparation_batches"][0] == raw["initial_batch"], "Preparation starts from another fixture")
        batches += raw["preparation_batches"] + raw["timed_batches"]
    for batch in batches:
        require(isinstance(batch, dict) and set(batch) == BATCH_FIELDS, "Missing full fixture hash manifest")
        for name, value in batch.items():
            tensor_manifest(value, shape=shape, dtype="torch.int64" if name in {"input_ids", "document_ids"} else "torch.bool")
        require(batch["ce_mask"] == batch["valid_mask"], "Full CE requires every valid target selected")
        require(batch["valid_mask"]["sha256"] == hashlib.sha256(bytes([1]) * math.prod(shape)).hexdigest(),
                "Integration fixtures must be entirely unpadded")
    require(all({k: b[k] for k in BATCH_FIELDS - {"input_ids"}} ==
                {k: batches[0][k] for k in BATCH_FIELDS - {"input_ids"}} for b in batches),
            "Fixture masks/documents changed across timed inputs")
    auxiliary = raw.get("initial_auxiliary_parameters")
    require(isinstance(auxiliary, dict) and auxiliary and any(name.startswith("backbone.fusion.") for name in auxiliary),
            "Missing initial auxiliary parameter hashes")
    require(c["case"] != "combined" or any(name.startswith("predictor.") for name in auxiliary),
            "Combined run lacks initial predictor hashes")
    for name, value in auxiliary.items():
        require(name.startswith(("backbone.fusion.", "predictor.")), "Unexpected auxiliary parameter name")
        tensor_manifest(value, dtype="torch.float32")
    expected_order = {"comparison_and_initial_capture": 0,
        "changed_input": 1 if c["stage"] == "verify" else 3,
        "parity_per_arm": [5, 6, 7] if c["stage"] == "verify" else [],
        "preparation": list(range(3)) if c["stage"] == "capacity" else [],
        "timed": list(range(3, 8)) if c["stage"] == "capacity" else []}
    require(raw.get("fixture_order") == expected_order, "Recorded fixture order differs from protocol")


def validate_compiler(raw):
    if raw["configuration"]["backend"] != "author":
        return
    for field in ("compiler_after_warmup", "compiler_final"):
        item = raw.get(field, {})
        counters = item.get("counters", {})
        require(item.get("required") is True and item.get("fail_on_recompile_limit_hit") is True
            and counters.get("stats", {}).get("unique_graphs", 0) > 0
            and not any(value for group in ("unimplemented", "graph_break") for value in counters.get(group, {}).values()),
            "Author compiler evidence missing or contains fallback: " + field)


def validate_resources(raw):
    card = raw.get("resources", {})
    matrix, observed, parameters = card.get("analytic_matrix_work", {}), card.get("observed_parameters", {}), raw.get("parameters", {})
    components = matrix.get("components")
    require(isinstance(components, list) and components and len({row["name"] for row in components}) == len(components),
            "Missing/duplicate matrix resource components")
    for row in components:
        require(all(type(row.get(key)) is int and row[key] >= 0 for key in ("minimum", "maximum"))
            and row["minimum"] <= row["maximum"], "Invalid matrix resource component")
    for key in ("minimum", "maximum"):
        require(matrix.get("matrix_flops_" + key) == sum(row[key] for row in components), "Matrix ledger sum differs")
    require(all(type(parameters.get(key)) is int and parameters[key] > 0 for key in
                ("active", "resident", "owned_parameter_tensors")) and parameters["active"] <= parameters["resident"],
            "Invalid observed active/resident parameter counts")
    require(observed.get("registered_unique") == parameters.get("resident")
        and observed.get("trainable") == parameters.get("active")
        and observed.get("gradient_participating") == parameters.get("active")
        and observed.get("executed_declared") == parameters.get("active"), "Observed parameter counts differ")
    c = raw["configuration"]
    if c["stage"] == "capacity":
        require(observed.get("optimizer_owned") == parameters.get("active"), "Optimizer parameter ownership differs")
    architecture = matrix.get("parameter_counts", {})
    require(architecture.get("training_architecture") == parameters.get("active")
        and architecture.get("deployable_inference") == observed.get("deployable_inference_declared"),
        "Architecture and active/deployable parameter counts differ")
    passes = 2 if c["case"] == "combined" else 1
    require(matrix.get("input_tokens_per_update") == raw["input_tokens"]
        and matrix.get("pass_token_work_per_update") == raw["input_tokens"] * passes
        and matrix.get("rt_block_calls_per_microbatch") == 2
        and matrix.get("ordinary_block_calls_per_microbatch") == 16 * passes - 2,
        "Resource token/pass/block accounting differs")
    require(matrix.get("rt_implementation") == c["backend"] and matrix.get("rt_backward_memory") == (
        "materialized" if c["backend"] == "author" else "recompute"), "Resource backend differs")
    counts = raw["counts"]
    require(card.get("loss_work", {}).get("ce_targets") == counts["ce"]
        and card["loss_work"].get("latent_pairs") == counts["latent"]
        and card["loss_work"].get("kl_triples") == counts["kl"], "Resource objective counts differ")
    if c["backend"] == "author":
        replacement = matrix.get("rt_component_replacement", {})
        ledger = replacement.get("per_call_author_ledger", {})
        require(replacement.get("calls") == 2, "Author replacement call count differs")
        require(not any(row["name"].startswith("rt_") for row in components), "Author ledger retains native RT components")
        actual = {row["name"]: row for row in components if row["name"].startswith("author_rt_")}
        require(len(actual) == 4, "Author resource ledger lacks four matrix components")
        for phase in ("forward", "backward"):
            for kind in ("dense", "attention"):
                row = actual.get(f"author_rt_{kind}_{phase}", {})
                expected = 2 * ledger.get("matrix_breakdown", {}).get(phase, {}).get(kind, -1)
                require(row.get("minimum") == row.get("maximum") == expected, "Author RT component differs from block ledger")


def validate_finished(raw):
    require(raw.get("schema") == SCHEMA and raw.get("finished_utc")
        and raw.get("status") in {"passed", "failed", "oom"}, "Require a finished integration report")
    require(type(raw.get("physical_optimizer_updates")) is int and raw["physical_optimizer_updates"] >= 0,
            "Invalid physical optimizer-update count")
    c, checks = raw.get("configuration", {}), raw.get("checks")
    require(c.get("stage") in OPERATIONAL and c.get("backend") in {"native", "author"}
        and c.get("case") in {"rt", "combined"}, "Unknown integration stage/backend/case")
    require(isinstance(checks, list) and all(isinstance(check, dict) and isinstance(check.get("name"), str)
        and type(check.get("passed")) is bool and type(check.get("gate", True)) is bool for check in checks), "Invalid checks")
    require(len({check["name"] for check in checks}) == len(checks), "Duplicate checks")
    if raw["status"] != "passed":
        require(raw.get("error", {}).get("type") and isinstance(raw["error"].get("message"), str), "Missing retained failure reason")
        return
    groups = gate_summary(raw)
    require(raw.get("stage") == "complete" and checks and all(check["passed"] for check in checks if check.get("gate", True)),
            "Passing report contains a failed gate or incomplete stage")
    require(groups["operational"]["complete_and_passed"], "Passing report lacks operational checks")
    require(c["stage"] != "verify" or groups["compatibility"]["complete_and_passed"], "Passing verify lacks raw numerical comparison")
    require(raw.get("checkpoint", {}).get("sha256") and raw.get("parameter_identity_preserved") is True,
            "Passing report lacks checkpoint/parameter-identity evidence")
    require(c.get("length") == 512 and c.get("batch_size") in ({8} if c["stage"] == "verify" else {32, 64})
        and c.get("supervision") == "full" and c.get("precision") == "bf16_mixed"
        and c.get("ce_chunk_size") == 2048 and c.get("kl_chunk_size") == 128
        and c.get("selected_rt_layers") == [0, 15], "Passing run differs from fixed integration protocol")
    require(raw["physical_optimizer_updates"] == (6 if c["stage"] == "verify" else 8), "Passing report has incomplete optimizer updates")
    require(raw.get("input_tokens") == c["batch_size"] * c["length"]
        and raw.get("counts", {}).get("ce") == c["batch_size"] * (c["length"] - 1)
        and raw.get("pass_input_token_work") == raw["input_tokens"] * (2 if c["case"] == "combined" else 1),
        "Full CE/input/pass counts differ")
    validate_fixtures(raw)
    validate_compiler(raw)
    validate_resources(raw)
    layout = raw.get("prepared_layout", {})
    require(layout.get("all_tokens_valid") is True and layout.get("reuse_rope") is True
        and layout.get("rt_implementation") == c["backend"]
        and layout.get("valid_mask_sha256") == raw["initial_batch"]["valid_mask"]["sha256"],
        "Prepared layout differs from the unpadded backend/fixture contract")
    if c["stage"] == "capacity":
        require(len(raw.get("preparation_records", [])) == 3 and len(raw.get("timed_records", [])) == 5, "Missing complete update records")
        for key, count in (("full_update", 5), ("forward_loss_backward", 3)):
            timing = raw.get(key, {})
            for clock in ("wall", "cuda"):
                samples = timing.get(clock + "_seconds", [])
                require(len(samples) == count and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples), "Invalid timing samples")
                require(math.isclose(timing.get("median_" + clock + "_seconds", -1), statistics.median(samples), rel_tol=1e-12), "Timing median differs")
        for key, numerator, timing in (("input_tokens_per_second", raw["input_tokens"], "full_update"),
                ("ce_targets_per_second", raw["counts"]["ce"], "full_update"),
                ("forward_loss_backward_tokens_per_second", raw["input_tokens"], "forward_loss_backward")):
            require(math.isclose(raw.get(key, -1), numerator / raw[timing]["median_wall_seconds"], rel_tol=1e-12), "Throughput differs from complete timing")


def load_run(root, name, revision):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name), "Unsafe run name")
    revision = resolve_commit(root, revision)
    path = regular(root / RUNTIME / name / "report.json", root)
    raw = json.loads(path.read_text())
    validate_finished(raw)
    require(resolve_commit(root, raw.get("runtime_commit")) == revision, "Selected/recorded runtime commits differ")
    hashes = raw.get("source_hashes")
    require(isinstance(hashes, dict) and ESSENTIAL_SOURCES <= hashes.keys(), "Missing essential runtime sources")
    differences = {}
    for source, expected in hashes.items():
        safe_source(source)
        require(digest(regular(path.parent / "source-snapshot" / source, root)) == expected == frozen_digest(str(root), revision, source),
                "Source snapshot differs from report/frozen commit: " + source)
        current = digest(root / source) if (root / source).is_file() else None
        if current != expected:
            differences[source] = {"reported_sha256": expected, "current_sha256": current}
    require(digest(regular(path.parent / "protocol.md", root)) == raw.get("protocol_sha256") == frozen_digest(str(root), revision, PROTOCOL), "Frozen protocol snapshot differs")
    return {"name": name, "runtime_commit": revision, "report_path": path.relative_to(root).as_posix(),
        "report_sha256": digest(path), "status": raw["status"], "physical_optimizer_updates": raw["physical_optimizer_updates"],
        "source_pairs_checked": len(hashes), "stage": raw["configuration"]["stage"],
        "case": raw["configuration"]["case"], "backend": raw["configuration"]["backend"],
        "gate_groups": gate_summary(raw), "current_source_differences": differences,
        "wandb_url": raw.get("wandb", {}).get("run_url"), "error": raw.get("error"),
        "compiler_final": raw.get("compiler_final"), "compiler_on_exit": raw.get("compiler_on_exit")}, raw


def comparison_signature(raw):
    c = raw["configuration"]
    ignored = {"backend", "output_dir", "artifacts"}
    payload = {"configuration": {k: v for k, v in c.items() if k not in ignored},
        **{key: raw.get(key) for key in ("checkpoint", "runtime", "determinism", "nextlat_config",
            "initial_auxiliary_parameters", "initial_batch", "preparation_batches", "timed_batches", "fixture_order", "counts", "parameters")}}
    payload["prepared_layout"] = {key: value for key, value in raw.get("prepared_layout", {}).items()
                                  if key != "rt_implementation"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def summarize(names, revision, *, overrides=None, root=ROOT):
    require(names and len(set(names)) == len(names), "Require explicit unique run names")
    overrides = overrides or {}
    require(set(overrides) <= set(names), "Runtime override names an unselected run")
    revision = resolve_commit(root, revision)
    rows, raws = zip(*(load_run(root, name, overrides.get(name, revision)) for name in names))
    checkpoints = {json.dumps(raw["checkpoint"], sort_keys=True) for raw in raws if "checkpoint" in raw}
    require(len(checkpoints) <= 1, "Selected runs use different checkpoint provenance")
    summary = {"schema": "olmo-rt-author-integration-summary-v1", "status": "completed", "runtime_commit": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(), "runs": list(rows),
        "physical_optimizer_updates": sum(row["physical_optimizer_updates"] for row in rows),
        "source_pairs_checked": sum(row["source_pairs_checked"] for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)), "capacity": [], "resource_cards": [],
        "comparison_groups": [], "checkpoint": json.loads(next(iter(checkpoints))) if checkpoints else None,
        "qualification": "Explicit finished selection, not an all-passed assertion. Numerical compatibility and operational gates are separate. Five-update full-LM benchmarks are not learning runs; K2 throughput counts input tokens only."}
    groups = defaultdict(list)
    for row, raw in zip(rows, raws):
        c = raw["configuration"]
        if raw.get("resources"):
            summary["resource_cards"].append({"name": row["name"], "status": row["status"], **raw["resources"]})
        if c["stage"] != "capacity":
            continue
        capacity = {"name": row["name"], "status": row["status"], "runtime_commit": row["runtime_commit"],
            **{key: c.get(key) for key in ("case", "backend", "batch_size", "length", "precision")},
            "comparison_signature": comparison_signature(raw),
            **{key: raw.get(key) for key in ("input_tokens_per_second", "ce_targets_per_second", "forward_loss_backward_tokens_per_second",
                "full_update", "forward_loss_backward", "setup_memory", "steady_memory", "resources", "parameters")}}
        summary["capacity"].append(capacity)
        if row["status"] == "passed":
            groups[(c["case"], c["batch_size"], c["length"], c["precision"])].append(capacity)
    for (case, batch, length, precision), members in sorted(groups.items()):
        require(len({row["comparison_signature"] for row in members}) == 1,
                "Capacity comparison fixtures, auxiliary weights, runtime or configuration differ")
        medians = {backend: statistics.median(row["input_tokens_per_second"] for row in members if row["backend"] == backend)
                   for backend in {row["backend"] for row in members}}
        summary["comparison_groups"].append({"case": case, "batch_size": batch, "length": length, "precision": precision,
            "runs": [row["name"] for row in members], "comparison_signature": members[0]["comparison_signature"],
            "median_run_input_tokens_per_second": medians,
            "author_gain_fraction": medians["author"] / medians["native"] - 1 if {"native", "author"} <= medians.keys() else None,
            "fixture_scope": "Identical recorded initial/timed tensor hashes, auxiliary weight hashes, checkpoint and execution settings. No fixture data or weights retained."})
    return summary


def plot_capacity(summary, directory):
    rows = [row for row in summary["capacity"] if row["status"] == "passed"]
    if not rows:
        return []
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    cases = sorted({row["case"] for row in rows})
    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4.5), squeeze=False)
    for axis, case in zip(axes[0], cases):
        batches = sorted({row["batch_size"] for row in rows if row["case"] == case})
        for backend, shift, color in (("native", -.2, "#3881b7"), ("author", .2, "#249b73")):
            x, y, lower, upper = [], [], [], []
            for index, batch in enumerate(batches):
                values = [r["input_tokens_per_second"] / 1000 for r in rows if (r["case"], r["batch_size"], r["backend"]) == (case, batch, backend)]
                if values:
                    center = statistics.median(values)
                    x.append(index + shift); y.append(center); lower.append(center - min(values)); upper.append(max(values) - center)
            if x:
                bars = axis.bar(x, y, width=.36, color=color, label=backend, yerr=[lower, upper], capsize=3)
                axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set_xticks(range(len(batches)), batches)
        axis.set_xlabel("Physical batch, T512")
        axis.set_ylabel("Input tokens/s (thousands)")
        axis.set_title("RT layers 0/15" if case == "rt" else "K2 FBT + RT + NextLat")
        axis.legend(); axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True); axis.margins(y=.2)
    fig.suptitle("OLMo-1B full-CE complete updates · native vs author RT")
    fig.text(.5, .01, "Five-update short benchmarks, not learning runs. Median of run medians; whiskers span repeats. K2 counts inputs once.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95))
    files = ["throughput.png", "throughput.pdf"]
    for name in files:
        fig.savefig(directory / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return files


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
        require(name not in overrides, "Duplicate per-run override")
        overrides[name] = revision
    summary = summarize(args.runs, args.runtime_commit, overrides=overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary["plots"] = plot_capacity(summary, args.output_dir)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "statuses", "physical_optimizer_updates", "source_pairs_checked")}))


if __name__ == "__main__":
    main()
