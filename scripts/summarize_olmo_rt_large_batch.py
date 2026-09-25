#!/usr/bin/env python3
"""Verify, plot and retain an explicit native-RT batch-scaling report selection."""
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
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_rt_efficiency_report import require, digest, regular, safe_source, check_summary
from scripts.olmo_rt_author_integration_report import resolve_commit, frozen_digest, tensor_manifest
from scripts.olmo_ordinary_fusions_report import (
    kernel_category, profile_summary as verified_profile_summary, validate_optimizer_flags,
)
from scripts.olmo_rt_efficiency_retain import safe_relative
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference

SCHEMA = "olmo-rt-large-batch-v1"
SUMMARY_SCHEMA = "olmo-rt-large-batch-summary-v1"
RETENTION_SCHEMA = "olmo-rt-large-batch-retention-v1"
RUNTIME = ".runtime/olmo-rt-large-batch"
DOCS = "docs/reports/olmo-rt-large-batch"
PROTOCOL = DOCS + "/protocol.md"
VALIDATION_SOURCE = "scripts/olmo_large_batch_validation.py"
ARTIFACT = ".runtime/olmo1b-step60000/artifacts/artifact-manifest.json"
RECEIPT = "docs/reports/olmo1b-o1/storage-receipt.json"
MAX_BYTES = 64 * 1024**2
ARMS = {"control": ("sdpa", "eager", "native", None),
        "optimized": ("sdpa", "compiled", "dao", True),
        "fa4": ("fa4", "compiled", "dao", True),
        "compiled-native": ("sdpa", "compiled", "native", True),
        "fa4-native": ("fa4", "compiled", "native", True)}
EXPECTED_CHECKS = {
    "correctness": {"same_state_candidate_vs_reference", "ordinary_dispatch_no_fallback",
        "candidate_initial_graph", "candidate_changed_tokens_overwrite",
        "complete_adamw_update_parity", "candidate_changed_weights"},
    "capacity": {"ordinary_dispatch_no_fallback", "capacity_initial_graph",
        "capacity_changed_tokens_overwrite", "capacity_changed_weights", "finite_complete_updates"},
}
ESSENTIAL_SOURCES = {
    "scripts/olmo_rt_large_batch.py", "scripts/olmo_rt_efficiency.py", "scripts/docker_shell.sh",
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_ordinary.py", "cdrm/pretrained/olmo_rope.py",
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_static.py", "cdrm/pretrained/olmo_fbt.py",
    "cdrm/pretrained/static_training.py", "cdrm/pretrained/static_nextlat.py",
    "cdrm/pretrained/nextlat.py", "cdrm/pretrained/lm_training.py", "cdrm/pretrained/resource_estimates.py",
}
PROJECT_FILES = (
    "scripts/summarize_olmo_rt_large_batch.py", "tests/test_summarize_olmo_rt_large_batch.py",
    "scripts/olmo_ordinary_fusions_report.py", "scripts/olmo_rt_efficiency_report.py",
    "scripts/olmo_rt_author_integration_report.py", "scripts/olmo_rt_efficiency_retain.py",
    "scripts/openelm_retain.py", "scripts/olmo_tiled_retain.py", "scripts/docker_shell.sh",
    VALIDATION_SOURCE, "tests/test_olmo_large_batch_validation.py",
    "docker/requirements-docker.txt", "docs/native-rt-large-batch-plan.md",
    "docs/native-rt-single-to-two-gpu-plan.md",
    "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v4.md", RECEIPT, "AGENTS.md",
)
BATCH_FIELDS = {"input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"}
NATIVE_RT_NAMED_KERNELS = {"historical_backward_kernel", "historical_recomputed_backward_kernel"}


def finite_number(value, *, nonnegative=True):
    return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)


def profile_summary(raw, directory, root, key="profile"):
    """Keep verified trace accounting while distinguishing names from RT attribution."""
    verified, details = verified_profile_summary(raw, directory, root, key)
    if details is None:
        return verified, details
    excluded = details["excluded_gpu_annotations"]["events"]
    kernels = {name: row for name, row in raw[key]["device_kernels"].items() if name not in excluded}
    groups = defaultdict(list)
    for name, row in kernels.items():
        category = "native_rt_historical_backward_named" if name in NATIVE_RT_NAMED_KERNELS else kernel_category(name)
        groups[category].append(row)
    total = math.fsum(row["self_device_us"] for row in kernels.values())
    details["kernel_name_categories"] = {name: {"calls": sum(row["calls"] for row in rows),
        "self_device_us": math.fsum(row["self_device_us"] for row in rows), "distinct_names": len(rows),
        "share_of_summed_device_time": math.fsum(row["self_device_us"] for row in rows) / total if total else 0.}
        for name, rows in sorted(groups.items())}
    named = {name: row for name, row in kernels.items() if name in NATIVE_RT_NAMED_KERNELS}
    named_us = math.fsum(row["self_device_us"] for row in named.values())
    details["native_rt_named_kernels"] = {"events": named,
        "calls": sum(row["calls"] for row in named.values()), "self_device_us": named_us,
        "share_of_summed_device_time": named_us / total if total else 0.,
        "scope": "Exact historical backward kernel names from the native RT source. This is only the named subset, not total RT time or finish/writer time. The forward tile is named kernel and remains unclassified because that generic name does not establish provenance."}
    details["classification_scope"] = (
        "Kernel-name inferences after GPU user-annotation exclusion, not measured model-layer attribution. "
        "Generic GEMMs, pointwise kernels, normalization, casts and rotary kernels may serve RT, ordinary blocks "
        "or losses; they are not assigned to RT finish/writer or to ordinary-only work. Named historical RT "
        "backward kernels identify only a subset of RT execution. Unrecognized names remain unclassified. "
        "Summed device events are not optimizer wall time or hardware utilization.")
    details["cpu_phase_scope_qualification"] += (
        " The inherited ordinary_step/ marker is also used by this RT/combined harness for the complete "
        "canonical update; its prefix does not mean ordinary-only execution.")
    return verified, details


def gate_groups(raw):
    expected = EXPECTED_CHECKS[raw["configuration"]["stage"]]
    groups = {}
    for name, names in (("compatibility", expected & {"same_state_candidate_vs_reference"}),
                        ("operational", expected - {"same_state_candidate_vs_reference"})):
        rows = [check_summary(c) for c in raw["checks"] if c["name"] in names]
        missing = sorted(names - {row["name"] for row in rows})
        groups[name] = {"checks": rows, "passed": sum(row["passed"] for row in rows),
            "total": len(rows), "missing": missing,
            "complete_and_passed": bool(names) and not missing and all(row["passed"] for row in rows)}
    return groups


def validate_memory(raw):
    phases = raw.get("memory_phases", {})
    require(isinstance(phases, dict), "Invalid memory phase inventory")
    snapshot_keys = {"allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib",
                     "device_free_gib", "device_total_gib", "device_used_gib"}
    for name, phase in phases.items():
        require(isinstance(name, str) and isinstance(phase, dict), "Invalid memory phase")
        require(phase.get("reset_peaks") is True and isinstance(phase.get("measurement_scope"), str)
                and phase["measurement_scope"], "Memory reset/scope is undocumented")
        for boundary in ("start", "end"):
            snapshot = phase.get(boundary)
            if snapshot is None and raw["status"] != "passed":
                continue
            require(isinstance(snapshot, dict) and snapshot_keys <= snapshot.keys()
                    and all(finite_number(snapshot[key]) for key in snapshot_keys),
                    "Missing/nonfinite phase memory snapshot")
        for key in ("peak_allocated_gib", "peak_reserved_gib"):
            if phase.get(key) is None and raw["status"] != "passed":
                continue
            require(finite_number(phase.get(key)), "Missing/nonfinite phase memory peak")
        require(raw["status"] != "passed" or not phase.get("error"), "Passing run contains a failed memory phase")
    if raw["status"] == "passed" and raw["configuration"]["stage"] == "capacity":
        require({"load_model", "prepare_plan", "dispatch", "preparation_updates", "capture_warmup",
                 "capture_capture", "validation_initial", "timing", "backward_timing", "health"} <= phases.keys(),
                "Passing capacity report lacks required measured phases")
        config = raw["configuration"]
        if config.get("validation_order", "live-graph") == "before-capture":
            require({"validation_references", "validation_changed_tokens", "validation_changed_weights"}
                    <= phases.keys(), "Before-capture validation lacks measured phases")
        if "validation_order" in config and config.get("release_transient_cache"):
            require({"capture_pre_warmup_transient_cleanup", "capture_transient_cleanup"} <= phases.keys(),
                    "New cleanup policy lacks measured pre/post-warmup phases")
        for key in ("setup_memory", "steady_memory"):
            snapshot = raw.get(key)
            require(isinstance(snapshot, dict) and snapshot_keys <= snapshot.keys()
                    and all(finite_number(snapshot[field]) for field in snapshot_keys),
                    "Missing/nonfinite aggregate memory snapshot")
        excluded = {"timing", "backward_timing", "validation_changed_weights", "health", "profile"}
        for key in ("peak_allocated_gib", "peak_reserved_gib"):
            require(raw["setup_memory"][key] >= max(phase[key] for name, phase in phases.items() if name not in excluded),
                    "Setup peak omits a measured preparation phase")
            require(raw["steady_memory"][key] == phases["timing"]["end"][key],
                    "Steady peak differs from timed phase")


def validate_fixtures(raw):
    config = raw["configuration"]
    shape = [config["batch_size"], config["length"]]
    batches = [raw.get("initial_batch")]
    if config["stage"] == "correctness":
        comparisons = raw.get("comparison_batches")
        require(isinstance(comparisons, dict) and set(comparisons) == {"1", "2", "5", "6", "7"},
                "Missing correctness fixture inventory")
        batches.extend(comparisons.values())
    else:
        for key, count in (("preparation_batches", 3), ("timed_batches", 5)):
            rows = raw.get(key)
            require(isinstance(rows, list) and len(rows) == count, "Missing capacity fixture inventory: " + key)
            batches.extend(rows)
        require(raw["preparation_batches"][0] == raw["initial_batch"], "Preparation fixture differs")
        if config.get("validation_order", "live-graph") == "before-capture":
            references = raw.get("validation_reference_batches")
            require(isinstance(references, dict) and set(references) == {"initial", "changed"},
                    "Missing before-capture reference fixture hashes")
            require(references["initial"] == raw["preparation_batches"][2]
                    and references["changed"] == raw["preparation_batches"][1],
                    "Before-capture reference fixtures differ from preparation batches 2/1")
            batches.extend(references.values())
        if config.get("profile"):
            batches.append(raw.get("profile_batch"))
    for batch in batches:
        require(isinstance(batch, dict) and set(batch) == BATCH_FIELDS, "Missing fixture tensor hashes")
        for name, value in batch.items():
            tensor_manifest(value, shape=shape, dtype="torch.int64" if name in {"input_ids", "document_ids"} else "torch.bool")
        require(batch["ce_mask"] == batch["valid_mask"], "Fixture is not full CE")
        require(batch["valid_mask"]["sha256"] == hashlib.sha256(bytes([1]) * math.prod(shape)).hexdigest(),
                "Fixture contains padding")
    require(all(all(batch[key] == batches[0][key] for key in BATCH_FIELDS - {"input_ids"})
                for batch in batches[1:]), "Static fixture masks/documents changed")


def validate_record(record, index, raw):
    config, counts = raw["configuration"], raw["counts"]
    require(record.get("update_completed") is True and record.get("counts") == counts,
            "Incomplete canonical optimizer update")
    expected = {"optimizer_updates": index, "microbatches": index,
        "input_tokens": index * raw["input_tokens"], "documents": index * config["batch_size"],
        "ce_positions": index * counts["ce"], "latent_pairs": index * counts["latent"],
        "kl_triples": index * counts["kl"]}
    require(record.get("counters") == expected, "Canonical update counters differ")


def validate_before_capture(raw):
    checks = {check["name"]: check for check in raw["checks"]}
    for name, replays in (("capacity_initial_graph", 1), ("capacity_changed_tokens_overwrite", 2),
                          ("capacity_changed_weights", None)):
        check = checks[name]
        require(all(check.get(key) is True for key in (
            "storage_matches", "ownership_matches", "loss_names_match", "all_bitwise_equal")),
            "Before-capture validation lacks exact storage/ownership checks: " + name)
        for field in ("losses", "gradients"):
            rows = check.get(field)
            require(isinstance(rows, dict) and rows
                    and all(row.get("bitwise_equal") is True for row in rows.values()),
                    "Before-capture validation lacks exact tensor evidence: " + name)
        if replays is not None:
            require(check.get("replays_checked") == replays,
                    "Before-capture validation replay count differs: " + name)
    require(checks["capacity_changed_weights"].get("graph_released_before_eager") is True,
            "Terminal eager validation did not record graph release")


def validate_resources(raw):
    card = raw.get("resources", {})
    matrix, observed = card.get("analytic_matrix_work", {}), card.get("observed_parameters", {})
    components = matrix.get("components", [])
    require(components and len({row["name"] for row in components}) == len(components),
            "Missing or duplicate resource components")
    for row in components:
        require(all(type(row.get(k)) is int and row[k] >= 0 for k in ("minimum", "maximum"))
                and row["minimum"] <= row["maximum"], "Invalid matrix bounds")
    for key in ("minimum", "maximum"):
        require(matrix.get("matrix_flops_" + key) == sum(row[key] for row in components), "Matrix totals differ")
    passes = 2 if raw["configuration"]["case"] == "combined" else 1
    require(matrix.get("input_tokens_per_update") == raw["input_tokens"]
            and matrix.get("pass_token_work_per_update") == passes * raw["input_tokens"]
            and matrix.get("rt_block_calls_per_microbatch") == 2
            and matrix.get("ordinary_block_calls_per_microbatch") == 16 * passes - 2,
            "Matrix token/pass/block accounting differs")
    for key in ("registered_unique", "trainable", "executed_declared", "deployable_inference_declared"):
        require(type(observed.get(key)) is int and observed[key] > 0
                and observed[key] == raw.get("parameters", {}).get(key), "Parameter inventory differs: " + key)
    require(observed.get("gradient_participating") == observed["trainable"], "Gradient ownership differs")
    if raw["configuration"]["stage"] == "capacity":
        require(observed.get("optimizer_owned") == observed["trainable"], "Optimizer ownership differs")
    for key, target in (("ce_targets", "ce"), ("latent_pairs", "latent"), ("kl_triples", "kl")):
        require(card.get("loss_work", {}).get(key) == raw["counts"][target], "Resource loss counts differ")


def validate_finished(raw):
    require(raw.get("schema") == SCHEMA and raw.get("finished_utc")
            and raw.get("status") in {"passed", "failed", "oom"}, "Require a finished large-batch report")
    require(type(raw.get("physical_optimizer_updates")) is int and raw["physical_optimizer_updates"] >= 0,
            "Invalid physical optimizer-update count")
    config, checks = raw.get("configuration", {}), raw.get("checks")
    require(config.get("case") in {"rt", "combined"} and config.get("arm") in ARMS
            and config.get("reference_arm") in ARMS and config.get("stage") in EXPECTED_CHECKS,
            "Unknown case, arm or stage")
    order = config.get("validation_order", "live-graph")
    require(order in {"live-graph", "before-capture"}
            and (order == "live-graph" or config["stage"] == "capacity"), "Invalid validation order")
    require(isinstance(checks, list) and all(isinstance(c, dict) and isinstance(c.get("name"), str)
            and type(c.get("passed")) is bool for c in checks), "Invalid check inventory")
    require(len({c["name"] for c in checks}) == len(checks), "Duplicate checks")
    validate_memory(raw)
    if raw["status"] != "passed":
        require(raw.get("error", {}).get("type") and isinstance(raw["error"].get("message"), str),
                "Failed report lacks an explicit reason")
        if raw.get("stage") != "complete":
            return
        require(raw["status"] == "failed" and config["stage"] == "correctness"
                and raw.get("compatibility_miss_continued") is True
                and config.get("continue_after_compatibility_miss") is True
                and raw.get("numerical_compatibility_passed") is False
                and raw.get("operational_checks_passed") is True, "Inconsistent completed numerical failure")
        require([c["name"] for c in checks if not c["passed"]] == ["same_state_candidate_vs_reference"],
                "Completed diagnostic has another failed gate")
        numeric = next(c for c in checks if c["name"] == "same_state_candidate_vs_reference")
        require(all(numeric.get(key) is True for key in ("finite", "ownership_matches", "counts_equal"))
                and all(numeric.get(key) for key in ("losses", "outputs", "gradients")),
                "Completed numerical failure contains nonfinite or structural failure")
    else:
        require(raw.get("stage") == "complete" and all(c["passed"] for c in checks), "Passing report is incomplete")
    require({c["name"] for c in checks} == EXPECTED_CHECKS[config["stage"]], "Missing required checks")
    require(config.get("length") in ((512,) if config["stage"] == "capacity" else (32, 512))
            and type(config.get("batch_size")) is int and config["batch_size"] > 0,
            "Unexpected shape")
    require(config.get("precision") == "bf16_mixed" and config.get("parameter_optimizer_dtype") == "float32"
            and config.get("supervision") == "all_valid_ce" and config.get("ce_chunk_size") == 2048
            and config.get("kl_chunk_size") == 128 and config.get("reuse_rope") is True,
            "Precision/full-CE protocol differs")
    require(config.get("selected_rt_layers") == [0, 15], "Selected native RT layers differ")
    require(config.get("world_size") == 1 and config.get("accumulation") == 1,
            "Physical batch scope differs")
    combined = config["case"] == "combined"
    require(config.get("fbt") is combined and config.get("nextlat") is combined,
            "Declared feature composition differs")
    mode = config.get("mode", {})
    require(mode.get("enabled") is combined and mode.get("num_passes") == (2 if combined else 1)
            and mode.get("rt_mode", {}).get("selected_layers") == [0, 15],
            "Actual RT/FBT mode differs")
    require(config.get("capture_warmup_backwards") == 10 and config.get("kv_only_writes") is True,
            "Warmup/native RT execution options differ")
    attention, pointwise, rope, fused = ARMS[config["arm"]]
    require(config.get("ordinary_attention") == attention and config.get("ordinary_pointwise") == pointwise
            and config.get("ordinary_rope_backend") == rope and config.get("fused_adam") is fused
            and config.get("ordinary_checkpointing") == "all", "Arm execution flags differ")
    layout = raw.get("prepared_layout", {})
    require(layout.get("all_tokens_valid") is True and layout.get("rt_implementation") == "native"
            and layout.get("ordinary_attention_backend") == attention
            and layout.get("ordinary_pointwise_backend") == pointwise
            and layout.get("ordinary_rope_backend") == rope and layout.get("reuse_rope") is True
            and layout.get("ordinary_checkpoint_layers") is None, "Prepared execution flags differ")
    if pointwise == "compiled":
        compiler = raw.get("compiler_observations", {})
        require(compiler.get("stats", {}).get("unique_graphs", 0) > 0
                and not any(value for key in ("graph_break", "unimplemented") for value in compiler.get(key, {}).values()),
                "Compiled pointwise execution is unverified or fell back")
    expected_updates = 6 if config["stage"] == "correctness" else 8 + int(bool(config.get("profile")))
    require(raw["physical_optimizer_updates"] == expected_updates, "Physical optimizer-update count differs")
    flags = (next(c for c in checks if c["name"] == "complete_adamw_update_parity").get("optimizer_flags")
             if config["stage"] == "correctness" else raw.get("optimizer_flags"))
    validate_optimizer_flags(flags, fused)
    require(raw.get("checkpoint", {}).get("sha256"), "Missing checkpoint provenance")
    require(raw.get("input_tokens") == config["batch_size"] * config["length"]
            and raw.get("counts", {}).get("ce") == config["batch_size"] * (config["length"] - 1),
            "Input/CE counts differ")
    counts = raw.get("counts", {})
    require(set(counts) == {"ce", "latent", "kl"}
            and all(type(value) is int and value >= 0 for value in counts.values()), "Invalid loss counts")
    require((counts["latent"] > 0 and counts["kl"] > 0) if config["case"] == "combined"
            else counts["latent"] == counts["kl"] == 0, "NextLat scope differs")
    validate_fixtures(raw)
    validate_resources(raw)
    if config["stage"] == "capacity":
        if order == "before-capture":
            validate_before_capture(raw)
        require(len(raw.get("preparation_records", [])) == 3 and len(raw.get("timed_records", [])) == 5,
                "Missing optimizer records")
        for index, record in enumerate(raw["preparation_records"] + raw["timed_records"], 1):
            validate_record(record, index, raw)
        require(raw.get("health", {}).get("passed") is True, "Nonfinite final state")
        if config.get("profile"):
            require(raw.get("post_profile_health", {}).get("passed") is True, "Missing post-profile health")
            validate_record(raw.get("full_step_profile", {}).get("update_record", {}), 9, raw)
        for key, count in (("full_update", 5), ("forward_loss_backward", 3)):
            timing = raw.get(key, {})
            for clock in ("wall", "cuda"):
                values = timing.get(clock + "_seconds", [])
                require(len(values) == count and all(finite_number(x) and x > 0 for x in values), "Invalid timing samples")
                require(math.isclose(timing.get("median_" + clock + "_seconds", -1), statistics.median(values), rel_tol=1e-12),
                        "Timing median differs")
        for key, tokens, field in (("input_tokens_per_second", raw["input_tokens"], "full_update"),
            ("ce_targets_per_second", counts["ce"], "full_update"),
            ("forward_loss_backward_tokens_per_second", raw["input_tokens"], "forward_loss_backward")):
            require(math.isclose(raw.get(key, -1), tokens / raw[field]["median_wall_seconds"], rel_tol=1e-12),
                    "Throughput differs from full-update timing")


def verify_dependencies(raw, directory, root):
    dependencies = raw.get("dependencies", {})
    count = 0
    for key, prefix in (("dao_sources", "flash_attn"), ("fa4_sources", "flash_attn/cute")):
        sources = dependencies.get(key, {})
        require(isinstance(sources, dict), "Invalid dependency inventory")
        if raw.get("stage") == "complete":
            arms = (raw["configuration"]["arm"], raw["configuration"]["reference_arm"])
            if key == "dao_sources" and any(ARMS[arm][2] == "dao" for arm in arms):
                require(set(sources) == {"layers/rotary.py", "ops/triton/rotary.py"}, "Missing Dao snapshots")
            if key == "fa4_sources" and any(ARMS[arm][0] == "fa4" for arm in arms):
                require("interface.py" in sources, "Missing FA4 interface snapshot")
        for name, record in sources.items():
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and path.suffix == ".py", "Unsafe dependency path")
            snapshot = regular(directory / "dependency-snapshot" / prefix / name, root)
            require(digest(snapshot) == record.get("sha256"), "Dependency snapshot differs: " + name)
            count += 1
    return {"files_checked": count, "packages": dependencies.get("packages"),
            "scope": "Installed source snapshots match recorded hashes; project sources additionally match frozen Git."}


def load_run(root, name, revision):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name), "Unsafe run name")
    revision = resolve_commit(root, revision)
    path = regular(root / RUNTIME / name / "report.json", root)
    raw = json.loads(path.read_text())
    validate_finished(raw)
    require(resolve_commit(root, raw.get("runtime_commit")) == revision, "Report runtime revision differs")
    hashes = raw.get("source_hashes")
    require(isinstance(hashes, dict) and ESSENTIAL_SOURCES <= hashes.keys(), "Missing essential source snapshots")
    if raw["configuration"].get("validation_order", "live-graph") == "before-capture":
        require(VALIDATION_SOURCE in hashes, "Missing before-capture validation source snapshot")
    differences = {}
    for source, expected in hashes.items():
        if source != "scripts/docker_shell.sh":
            safe_source(source)
        require(digest(regular(path.parent / "source-snapshot" / source, root)) == expected
                == frozen_digest(str(root), revision, source), "Source snapshot/report/frozen commit differs: " + source)
        current = digest(root / source) if (root / source).is_file() else None
        if current != expected:
            differences[source] = {"reported_sha256": expected, "current_sha256": current}
    require(digest(regular(path.parent / "protocol.md", root)) == raw.get("protocol_sha256")
            == frozen_digest(str(root), revision, PROTOCOL), "Protocol snapshot differs from frozen commit")
    dependencies = verify_dependencies(raw, path.parent, root)
    traces, profiles = {}, {}
    for key in ("profile", "full_step_profile"):
        trace, derived = profile_summary(raw, path.parent, root, key)
        if trace:
            traces[key], profiles[key] = trace, derived
    config = raw["configuration"]
    row = {"name": name, "runtime_commit": revision, "report_path": path.relative_to(root).as_posix(),
        "report_sha256": digest(path), "status": raw["status"],
        **{key: config[key] for key in ("case", "arm", "reference_arm", "stage")},
        "validation_order": config.get("validation_order", "live-graph"),
        "validation_reference_batches": raw.get("validation_reference_batches"),
        "physical_optimizer_updates": raw["physical_optimizer_updates"], "source_pairs_checked": len(hashes),
        "gate_groups": gate_groups(raw), "checks": [check_summary(c) for c in raw["checks"]],
        "current_source_differences": differences, "dependency_verification": dependencies,
        "wandb_url": raw.get("wandb", {}).get("run_url"), "error": raw.get("error"),
        "failure_stage": raw.get("stage") if raw["status"] != "passed" else None,
        "failed_memory_phases": sorted(name for name, phase in raw.get("memory_phases", {}).items()
                                       if phase.get("error")),
        "memory_phases": raw.get("memory_phases", {}),
        "numerical_compatibility_passed": raw.get("numerical_compatibility_passed"),
        "operational_checks_passed": raw.get("operational_checks_passed"), **traces}
    return row, raw, profiles


def signature(raw):
    ignored = {"arm", "reference_arm", "output_dir", "artifacts", "profile", "continue_after_compatibility_miss",
               "ordinary_attention", "ordinary_pointwise", "ordinary_rope_backend", "fused_adam"}
    config = {"validation_order": "live-graph", **raw["configuration"]}
    payload = {"configuration": {key: value for key, value in config.items() if key not in ignored},
        **{key: raw.get(key) for key in ("checkpoint", "runtime", "determinism", "nextlat_config",
            "initial_batch", "preparation_batches", "timed_batches", "counts", "parameters",
            "protocol_sha256", "source_hashes")},
        "dependency_packages": raw.get("dependencies", {}).get("packages")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def runtime_fingerprint(raw):
    """Identify verified runtime content independently of reporting-only commits."""
    dependencies = raw.get("dependencies", {})
    payload = {key: raw.get(key) for key in ("source_hashes", "protocol_sha256", "checkpoint",
        "runtime", "determinism", "nextlat_config")}
    payload["dependencies"] = {"packages": dependencies.get("packages"), **{
        key: {name: record["sha256"] for name, record in dependencies.get(key, {}).items()}
        for key in ("dao_sources", "fa4_sources")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def summarize(names, revision, *, overrides=None, root=ROOT):
    require(names and len(names) == len(set(names)), "Require explicit unique run names")
    overrides = overrides or {}
    require(set(overrides) <= set(names), "Override names an unselected run")
    revision = resolve_commit(root, revision)
    selected = [load_run(root, name, overrides.get(name, revision)) for name in names]
    rows = [row for row, _, _ in selected]
    checkpoints = {json.dumps(raw["checkpoint"], sort_keys=True) for _, raw, _ in selected if raw.get("checkpoint")}
    require(len(checkpoints) <= 1, "Selected checkpoints differ")
    summary = {"schema": SUMMARY_SCHEMA, "status": "completed", "runtime_commit": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(), "runs": rows,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "physical_optimizer_updates": sum(row["physical_optimizer_updates"] for row in rows),
        "source_pairs_checked": sum(row["source_pairs_checked"] for row in rows),
        "all_selected_runs_passed": all(row["status"] == "passed" for row in rows),
        "checkpoint": json.loads(next(iter(checkpoints))) if checkpoints else None,
        "capacity": [], "comparison_groups": [], "profiles": [], "resource_cards": [],
        "qualification": "Explicit finished selection including failures; not a claim of complete queue or cleared historical numerics. Physical per-device batch, no accumulation. Timings are directional execution measurements, not learning results. Allocated and reserved setup/steady memory are distinct; OOM phase is not an intrinsic model limit."}
    groups = defaultdict(list)
    for row, raw, profiles in selected:
        for profile in profiles.values():
            summary["profiles"].append({"name": row["name"], "case": row["case"], "arm": row["arm"], **profile})
        if raw.get("resources"):
            summary["resource_cards"].append({"name": row["name"], "status": row["status"], **raw["resources"]})
        if row["stage"] != "capacity":
            continue
        config = raw["configuration"]
        capacity = {"name": row["name"], "status": row["status"], "runtime_commit": row["runtime_commit"],
            "validation_order": row["validation_order"],
            **{key: config.get(key) for key in ("case", "arm", "batch_size", "length", "release_transient_cache")},
            "runtime_fingerprint": runtime_fingerprint(raw),
            "selected_same_case_arm_integration": [{"name": candidate["name"],
                "runtime_commit": candidate["runtime_commit"], "status": candidate["status"],
                "compatibility_complete_and_passed": candidate["gate_groups"]["compatibility"]["complete_and_passed"],
                "operational_complete_and_passed": candidate["gate_groups"]["operational"]["complete_and_passed"]}
                for candidate in rows if candidate["stage"] == "correctness"
                and candidate["case"] == row["case"] and candidate["arm"] == row["arm"]],
            "comparison_signature": signature(raw), "failure_stage": row["failure_stage"],
            "failed_memory_phases": row["failed_memory_phases"],
            **{key: raw.get(key) for key in ("input_tokens_per_second", "ce_targets_per_second",
                "forward_loss_backward_tokens_per_second", "full_update", "forward_loss_backward",
                "setup_memory", "steady_memory", "memory_phases", "parameters", "counts", "resources")}}
        summary["capacity"].append(capacity)
        if row["status"] == "passed":
            groups[capacity["comparison_signature"]].append(capacity)
    for cohort, members in sorted(groups.items()):
        arms = {arm: [row["input_tokens_per_second"] for row in members if row["arm"] == arm]
                for arm in sorted({row["arm"] for row in members})}
        medians = {arm: statistics.median(values) for arm, values in arms.items()}
        summary["comparison_groups"].append({"comparison_signature": cohort,
            **{key: members[0][key] for key in ("case", "batch_size", "length", "release_transient_cache", "validation_order")},
            "runs": [row["name"] for row in members],
            "arms": {arm: {"runs": len(values), "median_input_tokens_per_second": medians[arm],
                "minimum_input_tokens_per_second": min(values), "maximum_input_tokens_per_second": max(values)}
                for arm, values in arms.items()},
            "gain_fraction_vs_control": {arm: value / medians["control"] - 1 for arm, value in medians.items()
                if arm != "control"} if "control" in medians else {},
            "scope": "Same shape, fixtures, all frozen runtime sources, precision, protocol, validation order and capture-cleanup policy; only declared arm choices differ."})
    return summary


def render_plot(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in summary["capacity"] if row["status"] == "passed"]
    require(rows, "No successful capacity points to plot")
    cases = sorted({row["case"] for row in rows})
    figure, axes = plt.subplots(2, len(cases), figsize=(6 * len(cases), 8), squeeze=False)
    for column, case in enumerate(cases):
        selected = [row for row in rows if row["case"] == case]
        # Reporting-only commits may share a curve; runtime bytes, dependencies,
        # protocol and cleanup policy must still agree.
        groups = defaultdict(list)
        for row in selected:
            groups[(row["arm"], row["runtime_fingerprint"], row["release_transient_cache"], row["validation_order"])].append(row)
        for (arm, fingerprint, cleanup, order), group in sorted(groups.items(), key=lambda item: str(item[0])):
            by_batch = defaultdict(list)
            for row in group:
                by_batch[row["batch_size"]].append(row)
            batches = sorted(by_batch)
            label = f"{arm} / content {fingerprint[:7]}" + (" / cleanup" if cleanup else "")
            if order != "live-graph":
                label += " / " + order
            if any(not check["compatibility_complete_and_passed"] for row in group
                   for check in row["selected_same_case_arm_integration"]):
                label += " / integration not cleared"
            line, = axes[0, column].plot(batches, [statistics.median(r["input_tokens_per_second"] for r in by_batch[b]) / 1000
                for b in batches], "o-", label=label)
            for field, style in (("peak_allocated_gib", "o-"), ("peak_reserved_gib", "s--")):
                axes[1, column].plot(batches, [max(r["setup_memory"][field] for r in by_batch[b]) for b in batches],
                                     style, color=line.get_color(),
                                     label=label + " / " + field.replace("peak_", "").replace("_gib", ""))
        axes[0, column].set_title(case + ": complete updates, T512")
        axes[0, column].set_ylabel("Input tokens/s (thousands)")
        axes[1, column].set_ylabel("Setup peak memory (GiB)")
        for index, axis in enumerate(axes[:, column]):
            axis.set_xlabel("Physical batch per GPU")
            axis.legend(fontsize=7, **({"loc": "upper left", "bbox_to_anchor": (0, -.2)} if index else {}))
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Native OLMo RT: bounded throughput and setup memory")
    figure.text(.02, .015, "H100 · BF16 mixed · CUDA graphs · 5 timed updates/run.\n"
                "Failed attempts remain in summary. Setup peaks include preparation/validation.\n"
                "Operational capacity passes do not clear numerical integration failures.\n"
                "Run medians are directional; no quality claim.", fontsize=8)
    figure.tight_layout(rect=(0, .085, 1, .96))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(directory / ("capacity." + suffix), dpi=160)
    plt.close(figure)


def collect_evidence(root=ROOT):
    root = Path(root)
    docs, runtime = root / DOCS, root / RUNTIME
    summary = json.loads(regular(docs / "summary.json", root).read_text())
    require(summary.get("schema") == SUMMARY_SCHEMA and summary.get("status") == "completed" and summary.get("runs"),
            "Require a completed explicit selection")
    refreshed = summarize([row["name"] for row in summary["runs"]], summary["runtime_commit"], root=root,
                          overrides={row["name"]: row["runtime_commit"] for row in summary["runs"]})
    for key in set(refreshed) | set(summary):
        if key not in {"created_utc", "runs"}:
            require(refreshed.get(key) == summary.get(key), "Derived summary changed: " + key)
    for actual, selected in zip(refreshed["runs"], summary["runs"]):
        require({k: v for k, v in actual.items() if k != "current_source_differences"}
                == {k: v for k, v in selected.items() if k != "current_source_differences"}, "Selected report summary changed")
    artifact = json.loads(regular(root / ARTIFACT, root).read_text())
    receipt = json.loads(regular(root / RECEIPT, root).read_text())
    checkpoint = checkpoint_reference(receipt, artifact["checkpoint"])
    members = {}
    def add(path, name):
        regular(path, root)
        name = safe_relative(name).as_posix()
        require(name not in members, "Duplicate retained member")
        members[name] = path
    for row in refreshed["runs"]:
        path = root / row["report_path"]
        raw = json.loads(path.read_text())
        if raw.get("checkpoint"):
            require(checkpoint_reference(receipt, raw["checkpoint"]) == checkpoint, "Checkpoint reference differs")
        prefix = "runtime/" + row["name"] + "/"
        add(path, prefix + "report.json")
        add(path.parent / "protocol.md", prefix + "protocol.md")
        log_name = row["name"] + ".log"
        run_logs = [candidate for candidate in (runtime / "logs" / log_name, runtime / log_name)
                    if candidate.exists() or candidate.is_symlink()]
        require(len(run_logs) == 1, "Require exactly one retained run log: " + row["name"])
        add(run_logs[0], "logs/" + log_name)
        for source in raw["source_hashes"]:
            add(path.parent / "source-snapshot" / source, prefix + "source-snapshot/" + source)
        for key, base in (("dao_sources", "flash_attn"), ("fa4_sources", "flash_attn/cute")):
            for source in raw.get("dependencies", {}).get(key, {}):
                name = "dependency-snapshot/" + base + "/" + source
                add(path.parent / name, prefix + name)
        for key in ("profile", "full_step_profile"):
            if row.get(key):
                add(path.parent / row[key]["trace_file"], prefix + row[key]["trace_file"])
    for required in ("protocol.md", "results.md", "usage.md", "test-results.txt"):
        regular(docs / required, root)
    for path in sorted(docs.iterdir()):
        if path.suffix in {".md", ".json", ".txt", ".png", ".pdf", ".svg", ".csv"} and path.name != "storage-receipt.json":
            add(path, "report/" + path.name)
    for name in PROJECT_FILES:
        add(root / name, "project/" + name)
    for path in sorted(runtime.iterdir()):
        if re.fullmatch(r"(?:run_large_batch_queue\.py|large-batch-queue-[A-Za-z0-9_-]+\.(?:json|log))", path.name):
            add(path, "queue/" + path.name)
    add(root / ARTIFACT, "native-reference/artifact-manifest.json")
    add(runtime / "final-gpu.log", "logs/final-gpu.log")
    require(sum(path.stat().st_size for name, path in members.items() if not name.endswith("trace.json.gz")) < MAX_BYTES,
            "Non-trace evidence exceeds 64 MiB")
    return refreshed, checkpoint, members


def upload_verified(bucket, path, prefix):
    expected = file_digest(path)
    blob = bucket.blob(prefix + "/" + path.name)
    blob.metadata = {"sha256": expected["sha256"], "artifact_schema": RETENTION_SCHEMA}
    blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
    blob.reload()
    _check_remote(blob, expected)
    require(hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation)).hexdigest() == expected["sha256"],
            "Downloaded retained object differs")
    return {"uri": "gs://fast-chunks/" + blob.name, "generation": str(blob.generation), **expected}


def retain(*, dry_run=False):
    summary, checkpoint, members = collect_evidence()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / RUNTIME / ("retention-" + stamp)
    output.mkdir(exist_ok=False)
    archive = output / "evidence.tar.gz"
    inventory = build_evidence_archive(archive, [(path, name) for name, path in sorted(members.items())],
        "# Native RT physical-batch evidence\n\nExplicit selected successes/failures, frozen sources, dependency bytes, "
        "phase memory, raw traces and W&B links. No model/optimizer weights, datasets or credentials uploaded. "
        "Original native checkpoint reused by verified reference. This project overlay requires the full Git checkout "
        "and container. Historical numerical qualifications remain open.\n")
    require(archive.stat().st_size < MAX_BYTES, "Compressed evidence exceeds 64 MiB")
    manifest = {"schema": RETENTION_SCHEMA, "members": inventory, "weights_uploaded": False,
        "checkpoint_reference": checkpoint, "runtime_commit": summary["runtime_commit"],
        "runs": summary["runs"], "statuses": summary["statuses"],
        "physical_optimizer_updates": summary["physical_optimizer_updates"], "evidence": file_digest(archive)}
    manifest_path = output / "retention-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if dry_run:
        print(json.dumps({"status": "dry_run", "members": len(inventory), "archive": str(archive)}))
        return
    from google.cloud import storage
    bucket = storage.Client().bucket("fast-chunks")
    prefix = "cdrm-w-latent/fbt-rt-nextlat/olmo-rt-large-batch/" + stamp
    receipt = {"schema": RETENTION_SCHEMA, "status": "verified", "weights_uploaded": False,
        "members": len(inventory), "runs": len(summary["runs"]), "statuses": summary["statuses"],
        "physical_optimizer_updates": summary["physical_optimizer_updates"],
        "checkpoint_reference": verify_checkpoint_reference(bucket, checkpoint),
        "objects": [upload_verified(bucket, path, prefix) for path in (archive, manifest_path)]}
    receipt_path = output / "storage-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_object"] = upload_verified(bucket, receipt_path, prefix)
    (ROOT / DOCS / "storage-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "verified", "receipt": receipt["receipt_object"]["uri"]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--runtime-commit")
    parser.add_argument("--run-commit", action="append", default=[], metavar="NAME=COMMIT")
    parser.add_argument("--output-dir", type=Path, default=ROOT / DOCS)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--retain", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.retain:
        if args.runs or args.runtime_commit or args.run_commit or args.plot:
            parser.error("Retention reads the existing explicit summary; no new selection")
        retain(dry_run=args.dry_run)
        return
    if not args.runs or not args.runtime_commit or args.dry_run:
        parser.error("Summarization requires --runs and --runtime-commit; --dry-run requires --retain")
    overrides = {}
    for value in args.run_commit:
        name, separator, revision = value.partition("=")
        require(separator and name and revision and name not in overrides, "Invalid/duplicate runtime override")
        overrides[name] = revision
    summary = summarize(args.runs, args.runtime_commit, overrides=overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.plot:
        render_plot(summary, args.output_dir)
    print(json.dumps({"statuses": summary["statuses"], "runs": len(summary["runs"]),
                      "physical_optimizer_updates": summary["physical_optimizer_updates"]}))


if __name__ == "__main__":
    main()
