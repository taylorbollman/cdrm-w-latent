#!/usr/bin/env python3
"""Validate explicitly selected ordinary-fusions evidence without loading models."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_rt_efficiency_report import require, digest, regular, safe_source, check_summary
from scripts.olmo_rt_author_integration_report import resolve_commit, frozen_digest, tensor_manifest

RUNTIME = ".runtime/olmo-ordinary-fusions"
DOCS = "docs/reports/olmo-ordinary-fusions"
PROTOCOL = DOCS + "/protocol.md"
SCHEMA = "olmo-ordinary-fusions-v1"
HARNESS_SOURCE = "scripts/olmo_ordinary_fusions.py"
ARMS = {
    "control": ("native", None), "dao-rope": ("dao", None),
    "fused-adam": ("native", True), "dao-rope-fused-adam": ("dao", True),
}
ESSENTIAL_SOURCES = {
    "scripts/olmo_ordinary_fusions.py", "scripts/docker_shell.sh", "scripts/olmo_rt_efficiency.py",
    "scripts/olmo_ordinary_efficiency.py", "scripts/olmo_ordinary_optimizer_probe.py",
    "scripts/olmo_f1_common.py", "scripts/olmo_lm_common.py", "scripts/olmo_f3d_validate.py",
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_ordinary.py", "cdrm/pretrained/olmo_rope.py",
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_static.py", "cdrm/pretrained/olmo_fbt.py",
    "cdrm/pretrained/static_training.py", "cdrm/pretrained/static_nextlat.py",
    "cdrm/pretrained/nextlat.py", "cdrm/pretrained/fbt_training.py", "cdrm/pretrained/resource_estimates.py",
    "cdrm/pretrained/lm_training.py",
}
OPTIMIZER_PROBE = "fixed_gradient_scalar_vs_fused_adamw"
DAO_SOURCES = {"layers/rotary.py", "ops/triton/rotary.py"}
EXPECTED_CHECKS = {
    "correctness": {"same_state_candidate_vs_reference", "ordinary_dispatch_no_fallback",
        "candidate_initial_graph", "candidate_changed_tokens_overwrite",
        "complete_adamw_update_parity", "candidate_changed_weights"},
    "capacity": {"ordinary_dispatch_no_fallback", "capacity_initial_graph",
        "capacity_changed_tokens_overwrite", "capacity_changed_weights", "finite_complete_updates"},
}
BATCH_FIELDS = {"input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"}


def expected_checks(raw):
    config = raw["configuration"]
    return EXPECTED_CHECKS[config["stage"]] | ({OPTIMIZER_PROBE} if config.get("optimizer_probe") else set())


def check_groups(raw):
    groups = {"compatibility": [], "operational": []}
    for check in raw["checks"]:
        group = "compatibility" if check["name"] == "same_state_candidate_vs_reference" else "operational"
        groups[group].append(check_summary(check))
    result = {}
    for key, rows in groups.items():
        expected = expected_checks(raw)
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
        if config.get("profile"):
            batches.append(raw.get("profile_batch"))
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
    require(card.get("checkpointed_ordinary_layer_count") == 16, "Checkpoint selection count differs")
    require(card.get("loss_work", {}).get("ce_targets") == raw["counts"]["ce"], "CE resource count differs")


def validate_optimizer_flags(flags, fused):
    require(isinstance(flags, list) and flags, "Missing actual optimizer flags")
    for group in flags:
        require(group.get("fused") is fused and group.get("foreach") is False
                and group.get("capturable") is False and not group.get("differentiable", False)
                and not group.get("amsgrad", False), "Actual optimizer implementation differs")


def validate_optimizer_probe(raw):
    if not raw["configuration"].get("optimizer_probe"):
        require(not any(c["name"] == OPTIMIZER_PROBE for c in raw["checks"]), "Undeclared optimizer probe")
        return
    config = raw["configuration"]
    require(config["stage"] == "correctness" and config["arm"] == "fused-adam"
            and config["batch_size"] == 8 and config["length"] == 512, "Unexpected optimizer probe scope")
    check = next(c for c in raw["checks"] if c["name"] == OPTIMIZER_PROBE)
    require(check.get("passed") is True and check.get("semantic_state_matches") is True
            and check.get("optimizer_groups_match_except_fused") is True
            and check.get("initial_state_restored") is True and raw.get("optimizer_probe_state_restored") is True
            and check.get("physical_optimizer_updates") == 6 and check.get("updates_per_arm") == 3,
            "Incomplete or unrestored optimizer comparison")
    arms, progress = check.get("arms"), raw.get("optimizer_probe_progress")
    require(isinstance(arms, list) and len(arms) == 2 and isinstance(progress, list) and len(progress) == 6,
            "Optimizer comparison arm/update inventory differs")
    require([row.get("fused") for row in progress] == [False] * 3 + [True] * 3
            and [row.get("optimizer_update") for row in progress] == [1, 2, 3] * 2,
            "Optimizer comparison physical update order differs")
    for index, outcome in enumerate(arms):
        require(outcome.get("fused") is bool(index) and outcome.get("moments_fp32") is True
                and outcome.get("state_keys_complete") is True and len(outcome.get("records", [])) == 3
                and outcome.get("step_values") and all(v == 3 for v in outcome["step_values"].values()),
                "Optimizer state/step semantics differ")
        validate_optimizer_flags(outcome.get("optimizer_groups"), True if index else None)
    require([{key: value for key, value in group.items() if key != "fused"}
             for group in arms[0]["optimizer_groups"]] ==
            [{key: value for key, value in group.items() if key != "fused"}
             for group in arms[1]["optimizer_groups"]],
            "Scalar and fused optimizer hyperparameters differ")
    require(all(arms[0].get(key) == arms[1].get(key) for key in
                ("records", "ownership", "step_values", "scheduler", "counters")),
            "Scalar and fused optimizer semantic state differs")


def validate_update_record(record, index, raw):
    require(isinstance(record, dict) and record.get("update_completed") is True
            and record.get("counts") == raw["counts"], "Incomplete canonical optimizer-update record")
    counters, config = record.get("counters", {}), raw["configuration"]
    expected = {"optimizer_updates": index, "microbatches": index,
        "input_tokens": index * raw["input_tokens"], "documents": index * config["batch_size"],
        "ce_positions": index * raw["counts"]["ce"], "latent_pairs": 0, "kl_triples": 0}
    require(counters == expected, "Canonical update counters differ")


def validate_finished(raw):
    require(raw.get("schema") == SCHEMA and raw.get("finished_utc")
            and raw.get("status") in {"passed", "failed", "oom"}, "Only finished ordinary-fusions reports may be selected")
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
        require({c["name"] for c in checks} == expected_checks(raw)
                and [c["name"] for c in checks if not c["passed"]] == ["same_state_candidate_vs_reference"],
                "Completed diagnostic must retain one numeric failure and all passing operational gates")
        numeric = next(c for c in checks if c["name"] == "same_state_candidate_vs_reference")
        require(all(numeric.get(key) is True for key in ("finite", "ownership_matches", "counts_equal"))
                and all(numeric.get(key) for key in ("losses", "outputs", "gradients")),
                "Completed diagnostic contains a structural or nonfinite comparison failure")
    else:
        require(raw.get("stage") == "complete" and all(c["passed"] for c in checks)
                and expected_checks(raw) == {c["name"] for c in checks}, "Passing report lacks required gates")
    require(raw.get("checkpoint", {}).get("sha256"), "Passing report lacks checkpoint provenance")
    require(config.get("precision") == "bf16_mixed" and config.get("parameter_optimizer_dtype") == "float32"
            and config.get("supervision") == "all_valid_ce" and config.get("ce_chunk_size") == 2048
            and config.get("kl_chunk_size") == 128 and config.get("reuse_rope") is True
            and config.get("active_rt_layer_count") == 0 and config.get("fbt") is False
            and config.get("nextlat") is False, "Passing report differs from ordinary-only full-CE protocol")
    rope, fused = ARMS[config["arm"]]
    require(config.get("ordinary_attention") == "sdpa" and config.get("ordinary_pointwise") == "compiled"
            and config.get("ordinary_checkpointing") == "all"
            and config.get("ordinary_rope_backend") == rope and config.get("fused_adam") is fused,
            "Declared arm differs from ordinary fusion protocol")
    require(raw["physical_optimizer_updates"] == (6 + 6 * bool(config.get("optimizer_probe"))
            if config["stage"] == "correctness" else 8 + int(bool(config.get("profile")))),
            "Incomplete optimizer-update protocol")
    if config["stage"] == "correctness":
        parity = next(c for c in checks if c["name"] == "complete_adamw_update_parity")
        validate_optimizer_flags(parity.get("optimizer_flags"), fused)
    else:
        validate_optimizer_flags(raw.get("optimizer_flags"), fused)
    validate_optimizer_probe(raw)
    require(raw.get("input_tokens") == config["batch_size"] * config["length"]
            and raw.get("counts") == {"ce": config["batch_size"] * (config["length"] - 1), "latent": 0, "kl": 0},
            "Input or objective counts differ")
    validate_fixtures(raw)
    validate_resources(raw)
    layout = raw.get("prepared_layout", {})
    require(layout.get("all_tokens_valid") is True and layout.get("reuse_rope") is True
            and layout.get("ordinary_attention_backend") == "sdpa"
            and layout.get("ordinary_pointwise_backend") == "compiled"
            and layout.get("ordinary_rope_backend") == rope
            and layout.get("ordinary_checkpoint_layers") is None, "Prepared ordinary execution metadata differs")
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
        for index, record in enumerate(raw["preparation_records"] + raw["timed_records"], 1):
            validate_update_record(record, index, raw)
        require(raw.get("health", {}).get("passed") is True, "Nonfinite final training state")
        if config.get("profile"):
            require(isinstance(raw.get("profile"), dict) and isinstance(raw.get("full_step_profile"), dict)
                    and raw.get("post_profile_health", {}).get("passed") is True,
                    "Missing full-step profile or post-profile health")
            validate_update_record(raw["full_step_profile"].get("update_record"), 9, raw)
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
    sources = dependencies.get("dao_sources", {})
    require(isinstance(sources, dict), "Invalid Dao dependency inventory")
    config = raw["configuration"]
    needs_dao = any(ARMS[config[key]][0] == "dao" for key in ("arm", "reference_arm"))
    require(raw["status"] != "passed" or not needs_dao or set(sources) == DAO_SOURCES,
            "Dao run lacks imported dependency snapshots")
    require(not sources or set(sources) == DAO_SOURCES, "Unexpected Dao dependency files")
    for name, item in sources.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts and path.suffix == ".py", "Unsafe Dao dependency path")
        require(isinstance(item.get("source"), str) and Path(item["source"]).is_absolute(), "Missing installed Dao source path")
        actual = regular(directory / "dependency-snapshot/flash_attn" / name, root)
        require(digest(actual) == item.get("sha256"), "Dao dependency snapshot differs: " + name)
    return {"files_checked": len(sources), "packages": dependencies.get("packages"),
            "scope": "Installed dependency bytes match recorded snapshots; project sources additionally match frozen Git."}


def kernel_category(name):
    lower = name.lower()
    if "memcpy" in lower or "memset" in lower:
        return "memory_operations"
    if "adam" in lower:
        return "optimizer_named"
    if "rotary" in lower:
        return "rotary_named"
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


def profile_summary(raw, directory, root, key="profile"):
    profile = raw.get(key)
    if profile is None:
        require(not raw["configuration"].get("profile") or raw["status"] != "passed",
                "Successful requested profile is missing")
        return None, None
    filename = {"profile": "operator-trace.json.gz", "full_step_profile": "full-step-trace.json.gz"}[key]
    require(raw["configuration"].get("profile") is True and profile.get("trace_file") == filename,
            "Unexpected trace declaration")
    path = regular(directory / filename, root)
    require(path.stat().st_size == profile.get("trace_bytes") and digest(path) == profile.get("trace_sha256"),
            "Operator trace bytes differ from report")
    with path.open("rb") as stream:
        require(stream.read(2) == b"\x1f\x8b", "Trace is not gzip data")
    with gzip.open(path, "rt") as stream:
        trace = json.load(stream)
    require(isinstance(trace.get("traceEvents"), list), "Trace event inventory is missing")
    gpu_annotations = {event.get("name") for event in trace["traceEvents"]
                       if event.get("cat") == "gpu_user_annotation" and isinstance(event.get("name"), str)}
    cpu_scopes = defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("cat") != "user_annotation" or event.get("ph") != "X" \
                or not event.get("name", "").startswith("ordinary_step/"):
            continue
        duration = event.get("dur")
        require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                "Invalid CPU annotation duration")
        cpu_scopes[event["name"]].append(duration)
    require(isinstance(profile.get("device_kernels"), dict), "Missing device kernel inventory")
    groups = defaultdict(list)
    raw_kernels = profile["device_kernels"]
    kernels, excluded = {}, {}
    for name, row in raw_kernels.items():
        require(isinstance(name, str) and type(row.get("calls")) is int and row["calls"] > 0
                and type(row.get("self_device_us")) in (int, float)
                and math.isfinite(row["self_device_us"]) and row["self_device_us"] >= 0,
                "Invalid device-event statistics")
        # Kineto emits CUDA user-annotation intervals in addition to actual
        # kernels. Their durations overlap the events they describe, and CPU
        # key_averages can share/overwrite the same annotation key. Preserve
        # the raw report while deriving kernel totals without those intervals.
        if name.startswith("ordinary_step/") or name in gpu_annotations:
            excluded[name] = row
        else:
            kernels[name] = row
            groups[kernel_category(name)].append(row)
    require(profile.get("device_event_count") == sum(row["calls"] for row in raw_kernels.values()),
            "Device-event count differs from kernel inventory")
    total = math.fsum(row["self_device_us"] for row in kernels.values())
    verified = {field: profile[field] for field in ("trace_file", "trace_bytes", "trace_sha256")}
    top = sorted(kernels.items(), key=lambda item: (-item[1]["self_device_us"], item[0]))[:30]
    details = {**verified, "scope": profile.get("scope"), "profile_kind": key,
        "device_event_count": sum(row["calls"] for row in kernels.values()),
        "raw_device_event_count": profile["device_event_count"], "summed_device_ms": total / 1000,
        "raw_summed_device_ms": math.fsum(row["self_device_us"] for row in raw_kernels.values()) / 1000,
        "excluded_gpu_annotations": {"events": excluded,
            "calls": sum(row["calls"] for row in excluded.values()),
            "summed_device_ms": math.fsum(row["self_device_us"] for row in excluded.values()) / 1000,
            "scope": "Exact ordinary_step/ markers plus names identified as gpu_user_annotation in the verified trace. These overlapping annotation intervals are excluded from kernel totals, categories and rankings."},
        "top_device_events": [{"name": name, **row,
            "share_of_summed_device_time": row["self_device_us"] / total if total else 0.} for name, row in top],
        "cpu_phase_scopes": {name: {"calls": len(values), "cpu_us": math.fsum(values)}
                             for name, values in sorted(cpu_scopes.items())},
        "raw_cpu_phase_scopes": profile.get("cpu_phase_scopes", {}),
        "cpu_phase_scope_qualification": "Recovered from CPU user_annotation complete events in the verified trace, not the raw key_averages dictionary whose CPU/CUDA keys can collide. Inclusive CPU scopes are nested and nonadditive; CPU dispatch duration is not GPU execution time. No device attribution is inferred from these CPU durations."}
    details["kernel_name_categories"] = {name: {"calls": sum(row["calls"] for row in rows),
        "self_device_us": math.fsum(row["self_device_us"] for row in rows), "distinct_names": len(rows),
        "share_of_summed_device_time": math.fsum(row["self_device_us"] for row in rows) / total if total else 0.}
        for name, rows in sorted(groups.items())}
    details["classification_scope"] = "Kernel-name inferences after GPU user-annotation exclusion, not measured model-layer attribution. nvjet_sm names are classified as NVIDIA matrix multiplication. Copy kernels may include casts; BF16/FP32 arithmetic names do not identify their layer. Unrecognized kernels remain unclassified; summed CUDA events are not optimizer wall time."
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
    profiles = {}
    traces = {}
    for key in ("profile", "full_step_profile"):
        trace, profile = profile_summary(raw, path.parent, root, key)
        if trace:
            traces[key], profiles[key] = trace, profile
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
    row.update(traces)
    return row, raw, profiles


def comparison_sources(raw):
    return dict(raw["source_hashes"])


def comparison_signature(raw):
    config = {key: value for key, value in raw["configuration"].items()
              if key not in {"arm", "reference_arm", "output_dir", "artifacts", "profile",
                             "continue_after_compatibility_miss", "ordinary_rope_backend", "fused_adam"}}
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
    summary = {"schema": "olmo-ordinary-fusions-summary-v1", "status": "completed", "runtime_commit": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(), "runs": rows,
        "physical_optimizer_updates": sum(row["physical_optimizer_updates"] for row in rows),
        "source_pairs_checked": sum(row["source_pairs_checked"] for row in rows),
        "statuses": dict(Counter(row["status"] for row in rows)), "all_selected_runs_passed": all(row["status"] == "passed" for row in rows),
        "checkpoint": json.loads(next(iter(checkpoints))) if checkpoints else None,
        "capacity": [], "profiles": [], "resource_cards": [], "comparison_groups": [],
        "qualification": "Explicit finished selection, not a complete-queue or all-passed assertion. Five-update measurements are directional, not learning results. Matrix FLOPs exclude pointwise work and are not measured hardware FLOPs."}
    groups = defaultdict(list)
    for row, raw, profiles in selected:
        config = raw["configuration"]
        for profile in profiles.values():
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
        arms = {arm: [row["input_tokens_per_second"] for row in members if row["arm"] == arm]
                for arm in {row["arm"] for row in members}}
        medians = {arm: statistics.median(values) for arm, values in arms.items()}
        summary["comparison_groups"].append({"batch_size": batch, "length": length, "precision": precision,
            "runs": [row["name"] for row in members], "comparison_signature": signature,
            "runtime_commits": sorted({row["runtime_commit"] for row in members}),
            "runtime_source_fingerprint": members[0]["runtime_source_fingerprint"],
            "source_scope": "All recorded runtime source hashes. Different source, fixture or protocol signatures form separate cohorts, even at the same batch/length.",
            "arms": {arm: {"runs": len(values), "median_input_tokens_per_second": medians[arm],
                           "minimum_input_tokens_per_second": min(values), "maximum_input_tokens_per_second": max(values)}
                     for arm, values in arms.items()},
            "gain_fraction_vs_control": {arm: value / medians["control"] - 1 for arm, value in medians.items() if arm != "control"}
                if "control" in medians else {}})
    return summary


def render_plot(summary, directory):
    """Plot only verified comparable run medians; preserve numerical qualification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [group for group in summary["comparison_groups"] if len(group["arms"]) > 1]
    require(groups, "No verified timing cohorts")
    labels = {"control": "Native RoPE / scalar AdamW", "dao-rope": "Dao RoPE",
              "fused-adam": "Fused AdamW", "dao-rope-fused-adam": "Dao RoPE + fused AdamW"}
    figure, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)
    for axis, group in zip(axes[0], groups):
        arms = [arm for arm in labels if arm in group["arms"]]
        rows = [group["arms"][arm] for arm in arms]
        values = [row["median_input_tokens_per_second"] / 1000 for row in rows]
        lower = [value - row["minimum_input_tokens_per_second"] / 1000 for value, row in zip(values, rows)]
        upper = [row["maximum_input_tokens_per_second"] / 1000 - value for value, row in zip(values, rows)]
        bars = axis.barh(range(len(arms)), values, xerr=[lower, upper], capsize=3,
                         color=["#536d82" if arm == "control" else "#438f83" for arm in arms])
        axis.set_yticks(range(len(arms)), [labels[arm] for arm in arms])
        axis.invert_yaxis()
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5)
        axis.set_xlim(0, max(value + delta for value, delta in zip(values, upper)) * 1.2)
        axis.set_xlabel("Input tokens / second (thousands)")
        axis.set_title(f"B{group['batch_size']} / T{group['length']}")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Pretrained OLMo-1B: complete optimizer-step throughput")
    figure.text(.02, .03, "H100 80GB · BF16 mixed · CUDA graphs · full CE · five timed updates/run\n"
        "Whiskers: range of run medians, not confidence intervals. See retained numerical qualifications; no quality claim.", fontsize=9)
    figure.tight_layout(rect=(0, .13, 1, .93))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(directory / ("throughput." + suffix), dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--run-commit", action="append", default=[], metavar="NAME=COMMIT")
    parser.add_argument("--output-dir", type=Path, default=ROOT / DOCS)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    overrides = {}
    for value in args.run_commit:
        name, separator, revision = value.partition("=")
        require(separator and name and revision and name not in overrides, "Invalid/duplicate --run-commit")
        overrides[name] = revision
    summary = summarize(args.runs, args.runtime_commit, overrides=overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot:
        render_plot(summary, args.output_dir)
    print(json.dumps({"statuses": summary["statuses"], "runs": len(summary["runs"]),
                      "physical_optimizer_updates": summary["physical_optimizer_updates"]}))


if __name__ == "__main__":
    main()
