#!/usr/bin/env python3
"""Validate explicitly selected ordinary-efficiency evidence without loading models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_rt_efficiency_report import require, digest, regular, safe_source, check_summary, verify_profile
from scripts.olmo_rt_author_integration_report import resolve_commit, frozen_digest, tensor_manifest

RUNTIME = ".runtime/olmo-ordinary-efficiency"
DOCS = "docs/reports/olmo-ordinary-efficiency"
PROTOCOL = DOCS + "/protocol.md"
SCHEMA = "olmo-ordinary-efficiency-v1"
HARNESS_SOURCE = "scripts/olmo_ordinary_efficiency.py"
# Reviewed diff changes compatibility-failure continuation/reporting only; it
# leaves capacity computation, fixtures, warmup and timed boundaries unchanged.
AUDITED_HARNESS_REVISIONS = ("ed26653a5606546ab0581049be8a8dd8e52df5a5",
                            "5445f2561089a76c6938b66a2f2d4bb53343fe67")
ARMS = {
    "control": ("sdpa", "all", "eager"), "fa4": ("fa4", "all", "eager"),
    "checkpoint-none": ("sdpa", "none", "eager"),
    "checkpoint-alternating": ("sdpa", "alternating", "eager"),
    "compiled": ("sdpa", "all", "compiled"),
    "compiled-checkpoint-alternating": ("sdpa", "alternating", "compiled"),
    "compiled-checkpoint-none": ("sdpa", "none", "compiled"),
    "fa4-compiled": ("fa4", "all", "compiled"),
    "fa4-compiled-checkpoint-alternating": ("fa4", "alternating", "compiled"),
    "fa4-compiled-checkpoint-none": ("fa4", "none", "compiled"),
}
ESSENTIAL_SOURCES = {
    "scripts/olmo_ordinary_efficiency.py", "scripts/docker_shell.sh", "scripts/olmo_rt_efficiency.py",
    "scripts/olmo_f1_common.py", "scripts/olmo_lm_common.py", "scripts/olmo_f3d_validate.py",
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_ordinary.py", "cdrm/pretrained/olmo_rope.py",
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_static.py", "cdrm/pretrained/olmo_fbt.py",
    "cdrm/pretrained/static_training.py", "cdrm/pretrained/static_nextlat.py",
    "cdrm/pretrained/nextlat.py", "cdrm/pretrained/fbt_training.py", "cdrm/pretrained/resource_estimates.py",
}
EXPECTED_CHECKS = {
    "correctness": {"same_state_candidate_vs_reference", "ordinary_dispatch_no_fallback",
        "candidate_initial_graph", "candidate_changed_tokens_overwrite",
        "complete_adamw_update_parity", "candidate_changed_weights"},
    "capacity": {"ordinary_dispatch_no_fallback", "capacity_initial_graph",
        "capacity_changed_tokens_overwrite", "capacity_changed_weights", "finite_complete_updates"},
}
BATCH_FIELDS = {"input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"}


def check_groups(raw):
    groups = {"compatibility": [], "operational": []}
    for check in raw["checks"]:
        group = "compatibility" if check["name"] == "same_state_candidate_vs_reference" else "operational"
        groups[group].append(check_summary(check))
    result = {}
    for key, rows in groups.items():
        expected = EXPECTED_CHECKS[raw["configuration"]["stage"]]
        expected = ({"same_state_candidate_vs_reference"} & expected if key == "compatibility"
                    else expected - {"same_state_candidate_vs_reference"})
        missing = sorted(expected - {row["name"] for row in rows})
        result[key] = {"checks": rows, "passed": sum(row["passed"] for row in rows), "total": len(rows),
            "missing": missing, "all_observed_passed": all(row["passed"] for row in rows) if rows else None,
            "complete_and_passed": bool(rows) and not missing and all(row["passed"] for row in rows)}
    return result


def validate_fixtures(raw):
    """Hash-only evidence verifies same full-CE layout and declared input order."""
    config = raw["configuration"]
    shape = [config["batch_size"], config["length"]]
    batches = [raw.get("initial_batch")]
    if config["stage"] == "correctness":
        comparisons = raw.get("comparison_batches")
        require(isinstance(comparisons, dict) and set(comparisons) == {"1", "2", "5", "6", "7"},
                "Missing changed-input/parity fixture inventory")
        batches.extend(comparisons.values())
    if config["stage"] == "capacity":
        for key, count in (("preparation_batches", 3), ("timed_batches", 5)):
            rows = raw.get(key)
            require(isinstance(rows, list) and len(rows) == count, "Missing fixture inventory: " + key)
            batches.extend(rows)
        require(raw["preparation_batches"][0] == raw["initial_batch"], "Preparation initial fixture differs")
    for batch in batches:
        require(isinstance(batch, dict) and set(batch) == BATCH_FIELDS, "Missing batch tensor hashes")
        for name, value in batch.items():
            tensor_manifest(value, shape=shape, dtype="torch.int64" if name in {"input_ids", "document_ids"} else "torch.bool")
        require(batch["ce_mask"] == batch["valid_mask"], "Full CE mask differs from validity")
        require(batch["valid_mask"]["sha256"] == hashlib.sha256(bytes([1]) * math.prod(shape)).hexdigest(),
                "Ordinary efficiency fixture is padded")
    for batch in batches[1:]:
        require(all(batch[key] == batches[0][key] for key in BATCH_FIELDS - {"input_ids"}),
                "Static fixture masks/documents changed")


def validate_resources(raw):
    card = raw.get("resources", {})
    ledger, observed = card.get("analytic_matrix_work", {}), card.get("observed_parameters", {})
    components = ledger.get("components")
    require(isinstance(components, list) and components and len({r["name"] for r in components}) == len(components),
            "Missing or duplicate matrix components")
    for row in components:
        require(all(type(row.get(key)) is int and row[key] >= 0 for key in ("minimum", "maximum"))
                and row["minimum"] <= row["maximum"], "Invalid matrix component bounds")
    for key in ("minimum", "maximum"):
        require(ledger.get("matrix_flops_" + key) == sum(row[key] for row in components), "Matrix component sum differs")
    require(ledger.get("input_tokens_per_update") == ledger.get("pass_token_work_per_update") == raw["input_tokens"]
            and ledger.get("ordinary_block_calls_per_microbatch") == 16
            and ledger.get("rt_block_calls_per_microbatch") == 0, "Ordinary block/token accounting differs")
    for key in ("registered_unique", "trainable", "executed_declared", "deployable_inference_declared"):
        require(type(observed.get(key)) is int and observed[key] > 0
                and observed[key] == raw.get("parameters", {}).get(key), "Parameter inventory differs: " + key)
    require(observed.get("gradient_participating") == observed["trainable"], "Gradient parameter ownership differs")
    if raw["configuration"]["stage"] == "capacity":
        require(observed.get("optimizer_owned") == observed["trainable"], "Optimizer parameter ownership differs")
    architecture = ledger.get("parameter_counts", {})
    require(architecture.get("training_architecture") == observed["trainable"]
            and architecture.get("deployable_inference") == observed["deployable_inference_declared"],
            "Architecture parameter count differs")
    policy = ARMS[raw["configuration"]["arm"]][1]
    count = {"all": 16, "alternating": 8, "none": 0}[policy]
    require(card.get("checkpointed_ordinary_layer_count") == count, "Checkpoint selection count differs")
    require(card.get("loss_work", {}).get("ce_targets") == raw["counts"]["ce"], "CE resource count differs")


def validate_finished(raw):
    require(raw.get("schema") == SCHEMA and raw.get("finished_utc")
            and raw.get("status") in {"passed", "failed", "oom"}, "Only finished ordinary-efficiency reports may be selected")
    require(type(raw.get("physical_optimizer_updates")) is int and raw["physical_optimizer_updates"] >= 0,
            "Invalid physical optimizer-update count")
    config, checks = raw.get("configuration", {}), raw.get("checks")
    require(config.get("stage") in EXPECTED_CHECKS and config.get("arm") in ARMS
            and config.get("reference_arm") in ARMS, "Unknown stage or arm")
    require(isinstance(checks, list) and all(isinstance(c, dict) and isinstance(c.get("name"), str)
            and type(c.get("passed")) is bool for c in checks), "Invalid check inventory")
    require(len({c["name"] for c in checks}) == len(checks), "Duplicate checks")
    if raw["status"] != "passed":
        require(raw.get("error", {}).get("type") and isinstance(raw["error"].get("message"), str),
                "Failed report lacks an explicit failure reason")
        if raw.get("stage") != "complete":
            return  # Explicit partial failure: no completed-operational claim.
        require(raw["status"] == "failed" and config["stage"] == "correctness"
                and raw.get("compatibility_miss_continued") is True
                and config.get("continue_after_compatibility_miss") is True
                and raw.get("numerical_compatibility_passed") is False
                and raw.get("operational_checks_passed") is True,
                "Inconsistent completed compatibility-miss diagnostic")
        require({c["name"] for c in checks} == EXPECTED_CHECKS["correctness"]
                and [c["name"] for c in checks if not c["passed"]] == ["same_state_candidate_vs_reference"],
                "Completed diagnostic must retain one numeric failure and five passing operational gates")
        numeric = next(c for c in checks if c["name"] == "same_state_candidate_vs_reference")
        require(all(numeric.get(key) is True for key in ("finite", "ownership_matches", "counts_equal"))
                and all(numeric.get(key) for key in ("losses", "outputs", "gradients")),
                "Completed diagnostic contains a structural or nonfinite comparison failure")
    else:
        require(raw.get("stage") == "complete" and all(c["passed"] for c in checks)
                and EXPECTED_CHECKS[config["stage"]] <= {c["name"] for c in checks}, "Passing report lacks required gates")
    require(raw.get("checkpoint", {}).get("sha256"), "Passing report lacks checkpoint provenance")
    require(config.get("precision") == "bf16_mixed" and config.get("parameter_optimizer_dtype") == "float32"
            and config.get("supervision") == "all_valid_ce" and config.get("ce_chunk_size") == 2048
            and config.get("kl_chunk_size") == 128 and config.get("reuse_rope") is True
            and config.get("active_rt_layer_count") == 0 and config.get("fbt") is False
            and config.get("nextlat") is False, "Passing report differs from ordinary-only full-CE protocol")
    require(raw["physical_optimizer_updates"] == (6 if config["stage"] == "correctness" else 8),
            "Incomplete optimizer-update protocol")
    require(raw.get("input_tokens") == config["batch_size"] * config["length"]
            and raw.get("counts") == {"ce": config["batch_size"] * (config["length"] - 1), "latent": 0, "kl": 0},
            "Input or objective counts differ")
    validate_fixtures(raw)
    validate_resources(raw)
    attention, policy, pointwise = ARMS[config["arm"]]
    layout = raw.get("prepared_layout", {})
    layers = {"all": None, "none": [], "alternating": list(range(0, 16, 2))}[policy]
    require(layout.get("all_tokens_valid") is True and layout.get("reuse_rope") is True
            and layout.get("ordinary_attention_backend") == attention
            and layout.get("ordinary_pointwise_backend") == pointwise
            and layout.get("ordinary_checkpoint_layers") == layers, "Prepared ordinary execution metadata differs")
    if pointwise == "compiled":
        compiler = raw.get("compiler_observations", {})
        options = raw.get("compiler_configuration", {})
        require(options.get("fullgraph") is True and options.get("suppress_errors") is False
                and options.get("fail_on_recompile_limit_hit") is True, "Compiled pointwise failure policy differs")
        require(compiler.get("stats", {}).get("unique_graphs", 0) > 0
                and not any(value for group in ("graph_break", "unimplemented") for value in compiler.get(group, {}).values()),
                "Compiled pointwise evidence is missing or contains fallback")
    if config["stage"] == "capacity":
        require(len(raw.get("preparation_records", [])) == 3 and len(raw.get("timed_records", [])) == 5,
                "Missing complete optimizer-update records")
        for key, count in (("full_update", 5), ("forward_loss_backward", 3)):
            timing = raw.get(key, {})
            for clock in ("wall", "cuda"):
                values = timing.get(clock + "_seconds", [])
                require(len(values) == count and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values),
                        "Invalid timing samples")
                require(math.isclose(timing.get("median_" + clock + "_seconds", -1), statistics.median(values), rel_tol=1e-12),
                        "Timing median differs from samples")
        for key, count, field in (("input_tokens_per_second", raw["input_tokens"], "full_update"),
            ("ce_targets_per_second", raw["counts"]["ce"], "full_update"),
            ("forward_loss_backward_tokens_per_second", raw["input_tokens"], "forward_loss_backward")):
            require(math.isclose(raw.get(key, -1), count / raw[field]["median_wall_seconds"], rel_tol=1e-12),
                    "Throughput differs from synchronized complete timing")


def verify_dependencies(raw, directory, root):
    dependencies = raw.get("dependencies", {})
    sources = dependencies.get("fa4_sources", {})
    require(isinstance(sources, dict), "Invalid FA4 dependency inventory")
    config = raw["configuration"]
    needs_fa4 = any(ARMS[config[key]][0] == "fa4" for key in ("arm", "reference_arm"))
    require(raw["status"] != "passed" or not needs_fa4 or ("interface.py" in sources and dependencies.get("fa4_interface")),
            "FA4 run lacks imported interface snapshot")
    for name, item in sources.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and path.suffix == ".py", "Unsafe FA4 dependency path")
        actual = regular(directory / "dependency-snapshot/flash_attn/cute" / name, root)
        require(digest(actual) == item.get("sha256"), "FA4 dependency snapshot differs: " + name)
    if sources:
        require(sources["interface.py"].get("source") == dependencies.get("fa4_interface"), "Imported FA4 interface path differs")
    return {"files_checked": len(sources), "packages": dependencies.get("packages"),
            "fa4_interface": dependencies.get("fa4_interface"),
            "scope": "Installed dependency bytes match recorded snapshots; project sources additionally match frozen Git."}


def kernel_category(name):
    lower = name.lower()
    if "memcpy" in lower or "memset" in lower:
        return "memory_operations"
    if any(word in lower for word in ("flash", "fmha", "attention")):
        return "attention_named"
    if lower.startswith("nvjet_sm") or any(word in lower for word in ("gemm", "gemv", "matmul", "cublas")):
        return "matrix_multiply_named"
    if "triton_poi_" in lower or "triton_red_" in lower:
        return "compiled_pointwise_named"
    if any(word in lower for word in ("layernorm", "layer_norm", "rmsnorm", "rms_norm")):
        return "normalization_named"
    if "logsoftmax" in lower or "nll_loss" in lower:
        return "vocabulary_loss_named"
    if "copy_kernel" in lower:
        return "copy_or_cast_named"
    if "fillfunctor" in lower:
        return "fill_named"
    if "silu" in lower:
        return "silu_named"
    if "cudafunctor_add<float>" in lower:
        return "fp32_add_named"
    if "mulfunctor" in lower:
        if "binaryfunctor<c10::bfloat16" in lower:
            return "bf16_multiply_named"
        if "binaryfunctor<float" in lower:
            return "fp32_multiply_named"
    if "catarray" in lower:
        return "concatenation_named"
    if any(word in lower for word in ("elementwise", "pointwise", "reduce", "reduction")):
        return "pointwise_or_reduction_named"
    return "other_or_unclassified"


def profile_summary(raw, directory, root):
    profile = raw.get("profile")
    if profile is not None:
        require(isinstance(profile.get("device_kernels"), dict), "Missing device kernel inventory")
        for name, row in profile["device_kernels"].items():
            require(isinstance(name, str) and type(row.get("calls")) is int and row["calls"] > 0
                    and type(row.get("self_device_us")) in (int, float)
                    and math.isfinite(row["self_device_us"]) and row["self_device_us"] >= 0,
                    "Invalid device-event statistics")
    verified, details = verify_profile(raw, directory, root)
    if verified is None:
        return None, None
    groups = defaultdict(lambda: {"calls": 0, "self_device_us": 0., "distinct_names": 0})
    for name, row in profile["device_kernels"].items():
        group = groups[kernel_category(name)]
        group["calls"] += row["calls"]
        group["self_device_us"] += row["self_device_us"]
        group["distinct_names"] += 1
    total = sum(row["self_device_us"] for row in groups.values())
    details["kernel_name_categories"] = {name: {**row, "share_of_summed_device_time": row["self_device_us"] / total if total else 0.}
                                           for name, row in groups.items()}
    details["classification_scope"] = "Kernel-name inferences, not measured model-layer attribution. nvjet_sm names are classified as NVIDIA matrix multiplication. Copy kernels may include casts; BF16/FP32 arithmetic names do not identify their layer. Unrecognized kernels remain unclassified; summed CUDA events are not optimizer wall time."
    return verified, details


def load_run(root, name, revision):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name), "Unsafe run name")
    revision = resolve_commit(root, revision)
    path = regular(root / RUNTIME / name / "report.json", root)
    raw = json.loads(path.read_text())
    validate_finished(raw)
    require(resolve_commit(root, raw.get("runtime_commit")) == revision, "Selected runtime revision differs from report")
    hashes = raw.get("source_hashes")
    require(isinstance(hashes, dict) and ESSENTIAL_SOURCES <= hashes.keys(), "Missing essential runtime source snapshots")
    differences = {}
    for source, expected in hashes.items():
        if source != "scripts/docker_shell.sh":
            safe_source(source)
        require(digest(regular(path.parent / "source-snapshot" / source, root)) == expected == frozen_digest(str(root), revision, source),
                "Source snapshot/report/frozen commit differs: " + source)
        current = digest(root / source) if (root / source).is_file() else None
        if current != expected:
            differences[source] = {"reported_sha256": expected, "current_sha256": current}
    require(digest(regular(path.parent / "protocol.md", root)) == raw.get("protocol_sha256") == frozen_digest(str(root), revision, PROTOCOL),
            "Protocol snapshot/report/frozen commit differs")
    dependencies = verify_dependencies(raw, path.parent, root)
    trace, profile = profile_summary(raw, path.parent, root)
    row = {"name": name, "runtime_commit": revision, "report_path": path.relative_to(root).as_posix(),
        "report_sha256": digest(path), "status": raw["status"], "stage": raw["configuration"]["stage"],
        "arm": raw["configuration"]["arm"], "reference_arm": raw["configuration"]["reference_arm"],
        "physical_optimizer_updates": raw["physical_optimizer_updates"], "source_pairs_checked": len(hashes),
        "gate_groups": check_groups(raw), "checks": [check_summary(c) for c in raw["checks"]],
        "current_source_differences": differences, "dependency_verification": dependencies,
        "wandb_url": raw.get("wandb", {}).get("run_url"), "error": raw.get("error"),
        "compatibility_miss_continued": raw.get("compatibility_miss_continued", False),
        "numerical_compatibility_passed": raw.get("numerical_compatibility_passed"),
        "operational_checks_passed": raw.get("operational_checks_passed")}
    if trace:
        row["profile"] = trace
    return row, raw, profile


def comparison_sources(raw):
    return {key: value for key, value in raw["source_hashes"].items() if key != HARNESS_SOURCE}


def comparison_signature(raw):
    config = {key: value for key, value in raw["configuration"].items()
              if key not in {"arm", "reference_arm", "output_dir", "artifacts", "profile",
                             "continue_after_compatibility_miss"}}
    payload = {"configuration": config, **{key: raw.get(key) for key in (
        "checkpoint", "runtime", "determinism", "nextlat_config", "initial_batch",
        "preparation_batches", "timed_batches", "fixture_order", "counts", "parameters", "protocol_sha256")}}
    payload["dependency_packages"] = raw.get("dependencies", {}).get("packages")
    payload["runtime_source_hashes"] = comparison_sources(raw)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def summarize(names, revision, *, overrides=None, root=ROOT):
    require(names and len(names) == len(set(names)), "Require explicit unique run names")
    overrides = overrides or {}
    require(set(overrides) <= set(names), "Runtime override names an unselected run")
    revision = resolve_commit(root, revision)
    selected = [load_run(root, name, overrides.get(name, revision)) for name in names]
    rows, raws = [item[0] for item in selected], [item[1] for item in selected]
    checkpoints = {json.dumps(raw["checkpoint"], sort_keys=True) for raw in raws if raw.get("checkpoint")}
    require(len(checkpoints) <= 1, "Selected runs use different checkpoint provenance")
    summary = {"schema": "olmo-ordinary-efficiency-summary-v1", "status": "completed", "runtime_commit": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(), "runs": rows,
        "physical_optimizer_updates": sum(row["physical_optimizer_updates"] for row in rows),
        "source_pairs_checked": sum(row["source_pairs_checked"] for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)), "all_selected_runs_passed": all(row["status"] == "passed" for row in rows),
        "checkpoint": json.loads(next(iter(checkpoints))) if checkpoints else None,
        "capacity": [], "profiles": [], "resource_cards": [], "comparison_groups": [],
        "qualification": "Explicit finished selection, not a complete-queue or all-passed assertion. Five-update measurements are directional, not learning results. Matrix FLOPs exclude pointwise work and are not measured hardware FLOPs."}
    groups = defaultdict(list)
    for row, raw, profile in selected:
        config = raw["configuration"]
        if profile:
            summary["profiles"].append({"name": row["name"], "arm": row["arm"], **profile})
        if raw.get("resources"):
            summary["resource_cards"].append({"name": row["name"], "status": row["status"], **raw["resources"]})
        if config["stage"] != "capacity":
            continue
        capacity = {"name": row["name"], "status": row["status"], "runtime_commit": row["runtime_commit"],
            **{key: config.get(key) for key in ("arm", "batch_size", "length", "precision")},
            "comparison_signature": comparison_signature(raw),
            "runtime_source_fingerprint": hashlib.sha256(json.dumps(comparison_sources(raw), sort_keys=True).encode()).hexdigest(),
            "harness_source_sha256": raw["source_hashes"][HARNESS_SOURCE],
            **{key: raw.get(key) for key in ("input_tokens_per_second", "ce_targets_per_second",
                "forward_loss_backward_tokens_per_second", "full_update", "forward_loss_backward",
                "setup_memory", "steady_memory", "capture_seconds", "resources", "parameters", "counts")}}
        summary["capacity"].append(capacity)
        if row["status"] == "passed":
            groups[(config["batch_size"], config["length"], config["precision"], capacity["comparison_signature"])].append(capacity)
    for (batch, length, precision, signature), members in sorted(groups.items()):
        require(len({row["comparison_signature"] for row in members}) == 1, "Capacity fixtures/runtime/protocol differ within comparison cohort")
        harness_hashes = {row["harness_source_sha256"] for row in members}
        audit = None
        if len(harness_hashes) > 1:
            try:
                audited = {frozen_digest(str(root), revision, HARNESS_SOURCE) for revision in AUDITED_HARNESS_REVISIONS}
            except subprocess.CalledProcessError as error:
                raise ValueError("Different capacity harness sources lack recorded audit revisions") from error
            require(harness_hashes <= audited, "Different capacity harness sources lack a matching explicit audit")
            audit = {"revisions": list(AUDITED_HARNESS_REVISIONS),
                "source": HARNESS_SOURCE, "sha256": sorted(harness_hashes),
                "reason": "Reviewed diff only enables retained correctness-failure continuation and final diagnostic reporting; capacity computation and measurement boundaries are unchanged."}
        arms = {arm: [row["input_tokens_per_second"] for row in members if row["arm"] == arm]
                for arm in {row["arm"] for row in members}}
        medians = {arm: statistics.median(values) for arm, values in arms.items()}
        summary["comparison_groups"].append({"batch_size": batch, "length": length, "precision": precision,
            "runs": [row["name"] for row in members], "comparison_signature": signature,
            "runtime_commits": sorted({row["runtime_commit"] for row in members}),
            "runtime_source_fingerprint": members[0]["runtime_source_fingerprint"],
            "source_scope": "All recorded runtime source hashes except the independently audited diagnostic harness. Different source, fixture or protocol signatures form separate cohorts, even at the same batch/length.",
            "harness_difference_audit": audit,
            "arms": {arm: {"runs": len(values), "median_input_tokens_per_second": medians[arm],
                           "minimum_input_tokens_per_second": min(values), "maximum_input_tokens_per_second": max(values)}
                     for arm, values in arms.items()},
            "gain_fraction_vs_control": {arm: value / medians["control"] - 1 for arm, value in medians.items() if arm != "control"}
                if "control" in medians else {}})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--run-commit", action="append", default=[], metavar="NAME=COMMIT")
    parser.add_argument("--output-dir", type=Path, default=ROOT / DOCS)
    args = parser.parse_args(argv)
    overrides = {}
    for value in args.run_commit:
        name, separator, revision = value.partition("=")
        require(separator and name and revision and name not in overrides, "Invalid/duplicate --run-commit")
        overrides[name] = revision
    summary = summarize(args.runs, args.runtime_commit, overrides=overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"statuses": summary["statuses"], "runs": len(summary["runs"]),
                      "physical_optimizer_updates": summary["physical_optimizer_updates"]}))


if __name__ == "__main__":
    main()
