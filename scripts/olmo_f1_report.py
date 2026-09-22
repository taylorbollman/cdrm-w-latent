#!/usr/bin/env python3
"""Validate completed F1 evidence and render a scope-preserving capability ledger."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_f1_common import IntegrationCase

SCHEMA = "olmo-f1-capability-ledger-v1"
CHECKPOINT_SHA256 = "ccd2f952be7e68fdb602a486eb3eef64a81f9afc97901c640f5480867a1d750c"
MODEL_REVISION = "81b71efbce6f4dada57c94860301af4298bcd351"
NATIVE_PARAMETERS = 1176764416
FUSION_PARAMETERS = 8388608
PREDICTOR_PARAMETERS = 82726912
TERMS = ("ce", "latent", "kl")
SCOPES = ("registered", "requires_grad", "active", "optimizer_owned", "inference")
LIMITATIONS = [
    "Operational fixtures from two prompt strings with deterministic token rotations, EOS, response masks and right padding; no language-quality or learning-efficiency claim.",
    "BF16 mixed compute with FP32 parameters, gradients and moments; historical independent tiny math checks remain separate evidence. This is not a new broad FP32 comparison.",
    "Selected native RT uses eager dyadic tiling and explicit custom VJP. Fused ordinary SDPA events do not establish Flash/CuTE RT execution; current RT backward has quadratic attention intermediates.",
    "FBT K counts total shared-stack passes including ordinary pass0; RT applies to extra FBT passes. Standalone RT is a single recurrent pass. NextLat is training-only.",
    "Directional timings use complete nonzero-LR updates after warmup, excluding fixture construction, hooks, hashes, checkpoint/W&B I/O and the separate profiler update. They are not optimized capacity, final FLOP or scaling measurements.",
    "All-three changes pass count, recurrence and auxiliary training together. Throughput differences do not isolate any one feature.",
    "Profiler key averages can include both CPU-attributed device durations and raw CUDA kernels for the same work. Do not sum those mixed rows or convert them into fractions of total GPU time.",
    "No compile, CUDA graphs, Q/K normalization or distributed execution. All-16-layer RT and full-length online combined execution remain untested.",
    "Resume replays reproduce an already counted update after restoring an earlier boundary; count those physical executions separately from retained training counters.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _number(value, *, minimum=None):
    return type(value) in (int, float) and math.isfinite(value) and (minimum is None or value >= minimum)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _close(actual, expected, message):
    require(_number(actual) and math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-10), message)


def _walk_health(value, path="report"):
    if isinstance(value, dict):
        for name, child in value.items():
            if name in ("passed", "finite", "endpoint_losses_finite", "update_completed"):
                require(child is True, f"Failed or invalid {path}.{name}")
            _walk_health(child, f"{path}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_health(child, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite value in {path}")


def _json(value):
    return json.loads(json.dumps(value))


def _fixture_counts(fixture, case):
    fields = ("input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask")
    require(set(fixture) == set(fields), "Fixture fields differ")
    for name in fields:
        require(isinstance(fixture[name], list) and len(fixture[name]) == case.batch_size,
                "Fixture batch differs")
        require(all(isinstance(row, list) and len(row) == case.length for row in fixture[name]),
                "Fixture length differs")
    counts = dict.fromkeys(TERMS, 0)
    for row in range(case.batch_size):
        valid_count = case.length - (5 if row % 2 else 0)
        valid = [True] * valid_count + [False] * (case.length - valid_count)
        response = [False] * (valid_count // 2) + [True] * (valid_count - valid_count // 2) + [False] * (case.length - valid_count)
        require(fixture["valid_mask"][row] == valid and all(type(x) is bool for x in fixture["valid_mask"][row]),
                "Fixture padding contract differs")
        require(fixture["document_ids"][row] == [row] * valid_count + [-1] * (case.length - valid_count),
                "Fixture document boundaries differ")
        require(fixture["ce_mask"][row] == response and fixture["kl_mask"][row] == response
                and fixture["latent_mask"][row] == valid, "Fixture objective masks differ")
        require(all(_integer(token) and token < 50304 for token in fixture["input_ids"][row])
                and fixture["input_ids"][row][valid_count-1] == 50279,
                "Fixture vocabulary/EOS differs")
        counts["ce"] += valid_count - valid_count // 2
        if case.nextlat:
            counts["latent"] += valid_count - 1
            counts["kl"] += valid_count - valid_count // 2
    return counts


def _counters(case, updates, counts):
    return {"optimizer_updates": updates,
            "microbatches": updates + (case.batch_size - 1 if updates >= 2 and case.updates >= 2 else 0),
            "documents": updates * case.batch_size,
            "input_tokens": updates * (case.batch_size * case.length - 5 * (case.batch_size // 2)),
            "ce_positions": updates * counts["ce"], "latent_pairs": updates * counts["latent"],
            "kl_triples": updates * counts["kl"]}


def _metrics(metrics, case, updates, counts):
    require(metrics.get("schema") == "olmo-lm-optimizer-step-v1" and metrics.get("update_completed") is True,
            "Missing completed canonical optimizer step")
    require(metrics.get("counts") == counts and metrics.get("counters") == _counters(case, updates, counts),
            "Optimizer counters or target counts differ")
    weights = {"ce": 1.0, "latent": float(case.nextlat), "kl": float(case.nextlat)}
    require(metrics.get("objective_weights") == weights, "Objective weights differ")
    for key in ("loss_sums", "loss_means"):
        require(set(metrics.get(key, {})) == set(TERMS), "Missing objective terms")
    for name in TERMS:
        require(_number(metrics["loss_sums"][name]) and _number(metrics["loss_means"][name]), "Nonfinite loss")
        _close(metrics["loss_means"][name], metrics["loss_sums"][name] / counts[name] if counts[name] else 0,
               "Loss normalization differs")
    _close(metrics.get("objective"), sum(weights[name] * metrics["loss_means"][name] for name in TERMS),
           "Aggregate objective differs")
    require(_number(metrics.get("gradient_norm_before_clip"), minimum=0)
            and metrics["gradient_norm_before_clip"] > 0 and metrics.get("max_grad_norm") == 1,
            "Missing finite nonzero preclip gradient or changed clip norm")
    require(isinstance(metrics.get("lr_used"), list) and metrics["lr_used"]
            and all(_number(x, minimum=0) and x > 0 for x in metrics["lr_used"]), "Invalid learning rates")
    require(isinstance(metrics.get("lr_next"), list) and len(metrics["lr_next"]) == len(metrics["lr_used"])
            and all(_number(x, minimum=0) for x in metrics["lr_next"]), "Invalid next learning rates")


def _parameters(parameters, case):
    require(parameters.get("schema") == "olmo-f1-parameter-accounting-v1", "Wrong parameter schema")
    rows = parameters.get("parameter_records")
    require(isinstance(rows, list) and rows, "Missing parameter records")
    by_name, aliases = {}, set()
    for row in rows:
        name = row.get("name")
        require(isinstance(name, str) and name not in by_name, "Duplicate parameter object/name")
        require(isinstance(row.get("aliases"), list) and name in row["aliases"]
                and len(set(row["aliases"])) == len(row["aliases"])
                and not aliases.intersection(row["aliases"]), "Duplicate/missing parameter aliases")
        require(row.get("dtype") == "torch.float32" and str(row.get("device", "")).startswith("cuda")
                and type(row.get("requires_grad")) is bool, "Parameter dtype/device/flag differs")
        shape = row.get("shape")
        require(isinstance(shape, list) and all(_integer(x, minimum=1) for x in shape), "Invalid parameter shape")
        require(row.get("parameter_count") == math.prod(shape) and row.get("parameter_bytes") == math.prod(shape)*4,
                "Parameter size/bytes differ")
        by_name[name] = row; aliases.update(row["aliases"])
    for label in SCOPES:
        scope = parameters.get(label)
        require(isinstance(scope, dict) and isinstance(scope.get("names"), list), "Missing measured parameter scope")
        names = scope["names"]
        require(len(set(names)) == len(names) and set(names) <= set(by_name), "Parameter scope ownership differs")
        require(scope.get("tensor_count") == len(names)
                and scope.get("parameter_count") == sum(by_name[name]["parameter_count"] for name in names)
                and scope.get("parameter_bytes") == sum(by_name[name]["parameter_bytes"] for name in names),
                "Parameter scope totals differ")
    registered = set(by_name)
    native = {name for name in registered if name.startswith("backbone.backbone.")}
    fusion = {name for name in registered if name.startswith("backbone.fusion.")}
    predictor = {name for name in registered if name.startswith("predictor.")}
    require(native | fusion | predictor == registered and len(native) == 65 and len(fusion) == 2
            and len(predictor) == (4 if case.nextlat else 0), "Native/fusion/predictor ownership differs")
    require(sum(by_name[name]["parameter_count"] for name in native) == NATIVE_PARAMETERS
            and sum(by_name[name]["parameter_count"] for name in fusion) == FUSION_PARAMETERS
            and sum(by_name[name]["parameter_count"] for name in predictor) == (PREDICTOR_PARAMETERS if case.nextlat else 0),
            "Native/fusion/predictor parameter count differs")
    active = native | predictor | (fusion if case.fbt and (case.beta > 0 or case.transition) else set())
    trainable = native | predictor | (fusion if case.fbt else set())
    expected = {"registered": registered, "active": active, "requires_grad": trainable,
                "optimizer_owned": trainable, "inference": native | (fusion if case.fbt else set())}
    for label, names in expected.items():
        require(set(parameters[label]["names"]) == names, f"Wrong {label} participation")
    require({name for name, row in by_name.items() if row["requires_grad"]} == trainable,
            "Trainability flags differ from parameter scope")
    groups = parameters.get("optimizer_groups")
    require(isinstance(groups, list) and all(isinstance(group, list) for group in groups), "Missing optimizer ownership")
    flat = [name for group in groups for name in group]
    require(len(flat) == len(set(flat)) and set(flat) == trainable, "Duplicate/missing optimizer ownership")
    return by_name, active


def _gradients(gradients, expected):
    require(gradients.get("passed") is True and gradients.get("nonzero_accumulated_gradient") is True,
            "Gradient check is missing or failed")
    require(gradients.get("missing") == [] and gradients.get("unexpected") == []
            and set(gradients.get("tensors", {})) == expected, "Gradient ownership differs")
    for row in gradients["tensors"].values():
        require(row.get("finite") is True and _integer(row.get("backward_contributions"), minimum=1)
                and _number(row.get("contribution_norm_sum"), minimum=0)
                and _number(row.get("max_abs"), minimum=0), "Invalid gradient observation")


def _health(row):
    require(row.get("passed") is True and row.get("nonfinite_parameters") == []
            and row.get("nonfinite_optimizer_tensors") == [], "State health failed or incomplete")


def _profile(profile, case, config, counts):
    warmup, repeats = config["profile_warmup"], config["profile_repeats"]
    require(profile.get("warmup_updates") == warmup and profile.get("timed_updates") == repeats
            and profile.get("profiler_updates") == 1, "Profile update scope differs")
    for name in ("wall_seconds", "cuda_event_seconds"):
        values = profile.get(name)
        require(isinstance(values, list) and len(values) == repeats
                and all(_number(x) and x > 0 for x in values), "Profile timing observations missing or invalid")
    _close(profile.get("median_wall_seconds"), statistics.median(profile["wall_seconds"]), "Wall median differs")
    _close(profile.get("median_cuda_seconds"), statistics.median(profile["cuda_event_seconds"]), "CUDA median differs")
    seconds = profile["median_wall_seconds"]
    tokens = case.batch_size * case.length - 5 * (case.batch_size // 2)
    require(profile.get("input_tokens_per_update") == tokens and profile.get("counts_per_update") == counts,
            "Profile target/input count differs")
    _close(profile.get("valid_input_tokens_per_second"), tokens / seconds, "Token throughput differs")
    _close(profile.get("ce_targets_per_second"), counts["ce"] / seconds, "CE throughput differs")
    require(profile.get("physical_batch") == profile.get("logical_batch") == case.batch_size
            and profile.get("effective_passes") == (case.passes if case.fbt else 1)
            and profile.get("capacity_optimized") is False, "Profile batch/pass/capacity scope differs")
    for name in ("baseline_allocated_gib", "peak_allocated_gib", "peak_reserved_gib"):
        require(_number(profile.get(name), minimum=0), "Invalid memory observation")
    require(profile["peak_reserved_gib"] >= profile["peak_allocated_gib"] >= profile["baseline_allocated_gib"],
            "Memory peaks contradict allocation baseline")
    steps = profile.get("timed_steps")
    require(isinstance(steps, list) and len(steps) == repeats, "Missing timed complete-step records")
    for index, step in enumerate(steps):
        _metrics(step, case, case.updates + warmup + index + 1, counts)
    operators = profile.get("operators", {})
    require(operators.get("schema") == "olmo-f1-profiler-summary-v1"
            and operators.get("rt_selected_layers") == list(case.rt_layers), "Profiler scope differs")
    dispatch = operators.get("attention_dispatch")
    require(isinstance(dispatch, dict) and dispatch, "Missing observed backend evidence")
    for category in dispatch.values():
        require(type(category.get("observed")) is bool and isinstance(category.get("evidence"), list)
                and category["observed"] == bool(category["evidence"]), "Backend evidence flag differs")
        require(category.get("event_count") == sum(row["count"] for row in category["evidence"]),
                "Backend event counts differ")
    for key in ("top_cpu_operators", "top_device_operators"):
        require(isinstance(operators.get(key), list), "Missing operator timing list")
    all_rows = operators["top_cpu_operators"] + operators["top_device_operators"]
    all_rows += [row for category in dispatch.values() for row in category["evidence"]]
    for row in all_rows:
        require(isinstance(row.get("name"), str) and _integer(row.get("count")), "Invalid profiler operator")
        require(all(_number(row.get(key), minimum=0) for key in
                    ("self_cpu_time_us", "cpu_time_us", "self_device_time_us", "device_time_us")),
                "Invalid profiler duration")


def _validate(report):
    require(report.get("schema") == "olmo-f1-integration-v1" and report.get("status") == "passed"
            and report.get("started_utc") and report.get("finished_utc"), "Require completed passing F1 report")
    _walk_health(report)
    require(report.get("precision") == "bf16_mixed" and report.get("fp32_parameters_gradients_moments") is True,
            "Runtime precision differs")
    require(all(report.get(key) is False for key in ("compile", "cuda_graphs", "distributed", "qk_normalization")),
            "Unsupported runtime flags changed")
    require(_number(report.get("elapsed_seconds"), minimum=0), "Missing run duration")
    checkpoint = report.get("checkpoint", {})
    require(checkpoint.get("sha256") == CHECKPOINT_SHA256 and checkpoint.get("size_bytes") == 4707065440
            and MODEL_REVISION in checkpoint.get("url", ""), "Native checkpoint pin differs")
    require(_sha(report.get("config_sha256")) and _sha(report.get("protocol_sha256")), "Missing config/protocol hashes")
    require(isinstance(report.get("source_hashes"), dict) and report["source_hashes"]
            and all(isinstance(name, str) and _sha(value) for name, value in report["source_hashes"].items()),
            "Invalid runtime source inventory")
    for key in ("native_reference", "nextlat_reference", "fbt_reference"):
        require(isinstance(report.get(key), dict) and isinstance(report[key].get("revision"), str),
                "Missing pinned reference version")
    require(isinstance(report.get("runtime"), dict) and all(report["runtime"].get(key) for key in ("torch", "cuda", "gpu")),
            "Missing actual runtime")
    wandb = report.get("wandb", {})
    require(wandb.get("status") == "synced" and wandb.get("enabled") is True and wandb.get("mode") == "online"
            and str(wandb.get("run_url", "")).startswith("https://wandb.ai/taylorbollman/"), "W&B run is not synced")
    config = report.get("config", {})
    require(config.get("schema") == "olmo-f1-configuration-v1", "Wrong integration configuration")
    for key in ("seed", "vocab_chunk_size", "profile_warmup", "profile_repeats"):
        require(_integer(config.get(key), minimum=1), "Invalid configuration count")
    require(_number(config.get("learning_rate")) and 0 < config["learning_rate"] <= 1e-4, "Invalid learning rate")
    configured = [IntegrationCase(**row) for row in config["cases"]]
    by_name = {case.name: case for case in configured}
    require(configured and len(by_name) == len(configured), "Duplicate/empty configured cases")
    requested = report.get("requested_cases")
    require(isinstance(requested, list) and requested and len(set(requested)) == len(requested)
            and set(requested) <= set(by_name), "Invalid requested scope")
    require(requested == [case.name for case in configured if case.name in requested], "Requested order differs")
    cases = report.get("cases")
    require(isinstance(cases, list) and [row.get("name") for row in cases] == requested,
            "Completed cases must exactly cover requested cases in order")
    for row in cases:
        case = by_name[row["name"]]
        require(row.get("passed") is True and row.get("configuration") == _json(asdict(case)),
                "Case configuration or pass result differs")
        require(_number(row.get("elapsed_seconds_including_checks"), minimum=0), "Missing case duration")
        params, active = _parameters(row["parameters"], case)
        updates, fixtures = row.get("updates"), row.get("fixtures")
        require(isinstance(updates, list) and len(updates) == case.updates
                and isinstance(fixtures, list) and len(fixtures) == case.updates, "Missing original observed updates/fixtures")
        counts = _fixture_counts(fixtures[0], case)
        observed = set()
        for index, (update, fixture) in enumerate(zip(updates, fixtures)):
            require(update.get("update") == index + 1 and update.get("mode") == _json(asdict(case.mode(index))),
                    "Recorded update/mode differs")
            require(_fixture_counts(fixture, case) == counts, "Target counts changed unexpectedly")
            _metrics(update["metrics"], case, index+1, counts)
            expected = active - {name for name in active if name.startswith("backbone.fusion.")
                                 and case.mode(index).beta == 0}
            _gradients(update["gradients"], expected); observed.update(update["gradients"]["tensors"])
        require(observed == active, "Recorded active parameter union differs")
        change = row.get("state_change", {})
        require(change.get("passed") is True and change.get("unchanged_active") == []
                and change.get("unexpected_changes") == [] and set(change.get("changed", [])) == active
                and len(change["changed"]) == len(active), "Active/inactive state change check differs")
        _health(row["state_health"])
        require(("resume" in row) is case.resume and ("profile" in row) is case.profile
                and ("cache" in row) is (case.name == "rt-fbt-nextlat"), "Optional check coverage differs")
        if case.resume:
            resume = row["resume"]
            flags = ("passed", "loaded_boundary_exact", "cursor_exact", "next_fixture_exact", "next_update_state_exact",
                     "next_rng_exact", "next_update_metrics_exact", "disposable_checkpoint_deleted")
            require(all(resume.get(key) is True for key in flags), "Exact replay or checkpoint cleanup failed")
            saved = resume.get("checkpoint", {})
            require(saved.get("schema") == "olmo-lm-training-checkpoint-v1" and saved.get("optimizer_updates") == 2
                    and _sha(saved.get("sha256")) and _integer(saved.get("size_bytes"), minimum=1), "Missing replay checkpoint identity")
            _gradients(resume["gradients"], active)
        if "cache" in row:
            cache = row["cache"]
            require(cache.get("passed") is True and cache.get("incompatible_mode_rejected") is True, "Cache rejection failed")
            compared = cache.get("online_chunk_comparison", {})
            require(compared.get("passed") is True and compared.get("finite") is True
                    and all(compared.get(key) == 0 for key in ("max_abs", "difference_l2", "atol", "rtol")),
                    "Exact online/cache comparison failed")
        extra = config["profile_warmup"] + config["profile_repeats"] + 1 if case.profile else 0
        if case.profile:
            _profile(row["profile"], case, config, counts); _health(row["post_profile_health"])
        require(row.get("counters") == _counters(case, case.updates + extra, counts), "Final counters differ")
        endpoint = row.get("endpoint_losses", {})
        passes = case.passes if case.fbt else 1
        coefficients = [1.] + ([1. / (passes-1)] * (passes-1) if passes > 1 else [])
        require(row.get("endpoint_losses_finite") is True and endpoint.get("counts") == counts
                and endpoint.get("pass_coefficients") == coefficients
                and endpoint.get("objective_weights") == {"ce": 1., "latent": float(case.nextlat), "kl": float(case.nextlat)},
                "Endpoint objective semantics differ")
        losses = endpoint.get("per_pass_means")
        require(isinstance(losses, list) and len(losses) == passes and set(endpoint.get("aggregate_means", {})) == set(TERMS),
                "Missing endpoint pass losses")
        for loss in [*losses, endpoint["aggregate_means"]]:
            require(set(loss) == set(TERMS) and all(_number(value) for value in loss.values()), "Endpoint loss invalid")
    return report


def validate_report(report):
    """Reject incomplete, failed or internally inconsistent completed evidence."""
    try:
        return _validate(report)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError(f"Malformed F1 report: {error}") from error


def build_ledger(report):
    validate_report(report)
    config = report["config"]
    ledger = {"schema": SCHEMA, "status": "passed", "scope": report["scope"],
              "configured_cases": [row["name"] for row in config["cases"]],
              "requested_cases": list(report["requested_cases"]),
              "untested_configured_cases": [row["name"] for row in config["cases"] if row["name"] not in report["requested_cases"]],
              "checkpoint": report["checkpoint"], "model_revision": MODEL_REVISION,
              "native_reference": report["native_reference"], "nextlat_reference": report["nextlat_reference"],
              "fbt_reference": report["fbt_reference"], "runtime": report["runtime"], "wandb_url": report["wandb"]["run_url"],
              "config_sha256": report["config_sha256"], "protocol_sha256": report["protocol_sha256"],
              "source_hashes": report["source_hashes"], "started_utc": report["started_utc"],
              "finished_utc": report["finished_utc"], "elapsed_seconds": report["elapsed_seconds"],
              "precision": report["precision"], "learning_rate": config["learning_rate"],
              "cases": [], "limitations": list(LIMITATIONS)}
    totals = dict.fromkeys(("original_observed_updates", "warmup_updates", "timed_updates", "profiler_updates",
                           "resume_replay_updates", "retained_counter_updates", "physical_optimizer_executions"), 0)
    for row in report["cases"]:
        case, profile = row["configuration"], row.get("profile")
        counts = {"original_observed_updates": len(row["updates"]),
                  "warmup_updates": profile["warmup_updates"] if profile else 0,
                  "timed_updates": profile["timed_updates"] if profile else 0,
                  "profiler_updates": profile["profiler_updates"] if profile else 0,
                  "resume_replay_updates": int("resume" in row),
                  "retained_counter_updates": row["counters"]["optimizer_updates"]}
        counts["physical_optimizer_executions"] = counts["retained_counter_updates"] + counts["resume_replay_updates"]
        for key in totals: totals[key] += counts[key]
        gradients = [update["metrics"]["gradient_norm_before_clip"] for update in row["updates"]]
        ledger["cases"].append({"name": row["name"], "passed": True, "configuration": case,
            "observed_modes": [update["mode"] for update in row["updates"]], "update_accounting": counts,
            "parameters": {key: row["parameters"][key] for key in (*SCOPES, "resident_parameter_storage")},
            "checks": {"gradient_ownership_finite": True, "active_changed_inactive_unchanged": True,
                       "parameters_optimizer_finite": True, "endpoint_losses_finite": True,
                       "exact_resume": "passed" if "resume" in row else "untested",
                       "online_cache_B1_T8": "passed" if "cache" in row else "untested",
                       "post_profile_health": "passed" if profile else "untested"},
            "observed_preclip_gradient_norm": {"median": statistics.median(gradients), "max": max(gradients)},
            "resume": row.get("resume"), "cache": row.get("cache"), "profile": profile,
            "counters": row["counters"], "endpoint_losses": row["endpoint_losses"]})
    ledger["update_accounting"] = totals
    ledger["coverage"] = "full_configured_matrix" if not ledger["untested_configured_cases"] else "explicit_subset"
    return ledger


def _backend(profile):
    categories = profile["operators"]["attention_dispatch"]
    labels = {"pytorch_flash_sdpa": "PyTorch Flash SDPA", "cudnn_sdpa": "cuDNN SDPA",
              "efficient_sdpa": "efficient SDPA", "math_sdpa": "math SDPA", "cpu_flash_attention": "CPU Flash"}
    actual = [label for name, label in labels.items() if categories.get(name, {}).get("observed")]
    return ", ".join(actual) or "backend unestablished; see operator evidence"


def markdown(ledger):
    totals = ledger["update_accounting"]
    lines = ["# F1: bounded integration and early execution profile", "",
        f"**{len(ledger['cases'])}/{len(ledger['configured_cases'])} configured cases passed** within the explicitly requested scope ({ledger['coverage']}). "
        "This checks functionality and execution; it does not establish language-model quality.", "",
        f"Original OLMo-1B step60,000 (~252B tokens), native model revision `{MODEL_REVISION}`; checkpoint SHA256 `{CHECKPOINT_SHA256}`. "
        "16 layers, width2048, 16 heads, SwiGLU8192, tied50304-row embedding/readout, native RoPE and non-affine LayerNorm; no Q/K normalization.", "",
        f"BF16 mixed compute / FP32 parameters, gradients and AdamW moments; LR{ledger['learning_rate']:g}. "
        f"Runtime: {ledger['runtime']['torch']}, CUDA{ledger['runtime']['cuda']}, {ledger['runtime']['gpu']}. "
        f"[W&B run]({ledger['wandb_url']}).", "",
        f"{totals['original_observed_updates']} original observed updates + {totals['warmup_updates']} warmup + "
        f"{totals['timed_updates']} timed + {totals['profiler_updates']} separately profiled = "
        f"**{totals['retained_counter_updates']} retained-counter updates**. "
        f"{totals['resume_replay_updates']} additional physical replays reproduce already counted updates, for "
        f"**{totals['physical_optimizer_executions']} optimizer executions** overall.", "",
        "| Case | B / T | RT layers | FBT passes | NextLat | Observed updates | Resume / online cache |",
        "| --- | ---: | --- | ---: | --- | ---: | --- |"]
    for row in ledger["cases"]:
        case, checks = row["configuration"], row["checks"]
        lines.append(f"| {row['name']} | {case['batch_size']} / {case['length']} | {case['rt_layers'] or 'off'} | "
                     f"{case['passes'] if case['fbt'] else 'off'} | {'on' if case['nextlat'] else 'off'} | "
                     f"{row['update_accounting']['original_observed_updates']} | {checks['exact_resume']} / {checks['online_cache_B1_T8']} |")
    lines += ["", "All listed cases passed expected gradient participation/finite checks, active-weight changes, inactive-state invariance, "
              "finite parameters/optimizer state and endpoint losses. Exact resume includes loaded state, cursor, next fixture, next update, "
              "metrics and CPU/CUDA RNG. The online cache check is B1/T8 whole-versus-split plus incompatible-mode rejection.", "",
              "K includes the ordinary bootstrap; RT applies to extra FBT passes. Alpha/beta are1 unless the case is fractional "
              "(.37/.35) or transitioning through(0/0),(.37/.35),(1/1). Objective: pass0 + mean(extra-pass losses), "
              "with unit CE and, when enabled, latent/KL weights. CE and KL use response positions; latent regression uses valid pairs."]
    if ledger["untested_configured_cases"]:
        lines += ["", "**Not requested / untested:** " + ", ".join(ledger["untested_configured_cases"]) + "."]
    lines += ["", "| Parameter scope (unique tensors, tying counted once) | Registered/resident | Requires grad | Observed active | Optimizer owned | Inference |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    grouped = {}
    for row in ledger["cases"]:
        signature = tuple(row["parameters"][scope]["parameter_count"] for scope in SCOPES)
        grouped.setdefault(signature, []).append(row["name"])
    for signature, names in grouped.items():
        lines.append("| " + ", ".join(names) + " | " + " | ".join(f"{count:,}" for count in signature) + " |")
    lines += ["", "FBT-off models retain frozen fusion matrices in resident memory. NextLat-off models omit the predictor; "
              "NextLat inference excludes its training-only predictor. RT adds no parameters. Activity is the union of backward participation "
              "across observed updates, including intentionally zero contributions; transition-case fusion is inactive at beta0."]
    profiles = [row for row in ledger["cases"] if row["profile"]]
    if profiles:
        lines += ["", "| Early complete-update profile | B / T | Wall median (s) | CUDA median (s) | Input tokens/s | CE targets/s | Peak allocated / reserved GiB | Actual ordinary dispatch |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in profiles:
            p, case = row["profile"], row["configuration"]
            lines.append(f"| {row['name']} | {case['batch_size']} / {case['length']} | {p['median_wall_seconds']:.4f} | "
                         f"{p['median_cuda_seconds']:.4f} | {p['valid_input_tokens_per_second']:,.0f} | "
                         f"{p['ce_targets_per_second']:,.0f} | {p['peak_allocated_gib']:.2f} / {p['peak_reserved_gib']:.2f} | {_backend(p)} |")
        lines += ["", "Largest recorded exclusive CPU/device operators in the separate profiler update:", ""]
        for row in profiles:
            operators = row["profile"]["operators"]
            def top(key, field):
                return "; ".join(f"`{op['name']}` {op[field]/1000:.1f}ms" for op in operators[key][:2]) or "none recorded"
            lines.append(f"- {row['name']}: CPU {top('top_cpu_operators', 'self_cpu_time_us')}; device "
                         f"{top('top_device_operators', 'self_device_time_us')}.")
        lines += ["", "Profiler durations include instrumentation overhead and are not synchronized wall-time attribution. "
                  "Device lists can mix CPU-attributed scopes and raw CUDA kernels representing the same work; do not add mixed rows "
                  "or interpret their sum as a fraction of total GPU time. "
                  "Generic SDPA is not a backend identity; cuDNN fused attention does not mean use of the flash-attn package/FA4. "
                  "Native RT still uses eager tiles/custom backward; ordinary fused operators do not imply fused RT."]
    lines += ["", "Scope and remaining limitations:", ""]
    lines += ["- " + line for line in ledger["limitations"]]
    lines += ["", "Machine-readable case scopes, ownership, profiler evidence, exact replay details, counters and frozen source/version hashes "
              "are in [capability-ledger.json](capability-ledger.json).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/reports/olmo1b-f1")
    args = parser.parse_args()
    raw = args.report.read_bytes()
    ledger = build_ledger(json.loads(raw))
    ledger["input_report"] = {"path": str(args.report), "sha256": hashlib.sha256(raw).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "capability-ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output_dir / "results.md").write_text(markdown(ledger))
    print(json.dumps({"status": "passed", "coverage": ledger["coverage"], "cases": len(ledger["cases"]),
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
