#!/usr/bin/env python3
"""Validate and summarize bounded F3 graph-training evidence without scope promotion."""
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
from cdrm.pretrained.artifacts import write_json
from scripts.olmo_f1_retain import safe_evidence
from scripts.olmo_f2_retain import require, _sha, _source_name
from scripts.olmo_f3_retain import ESSENTIAL_SOURCES, REPORT_SCHEMA
from scripts.olmo_tiled_retain import CHECKPOINT_SHA256, CHECKPOINT_SIZE
from scripts.openelm_retain import file_digest

SCHEMA = "olmo-f3-summary-v1"
TENSOR_CHECKS = ("original_tokens", "changed_tokens", "repeated_replay_overwrites_gradients",
                 "changed_weights_and_tokens")
UPDATE_CHECK = "complete_adamw_update_parity"
TERMS = ("ce", "latent", "kl")
CAPTURE_SETUP_SCOPE = ("capture_seconds measures the entire plan.capture call: backward warmup plus graph capture, "
    "synchronization and validation; correctness setup also includes initial gradient discovery when not already initialized.")


def number(value, name, *, positive=False, signed=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else signed or value >= 0), f"Invalid finite numeric value: {name}")
    return value


def integer(value, name, *, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid integer: {name}")
    return value


def close(actual, expected, name):
    number(actual, name, signed=True)
    require(math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-10), f"Inconsistent {name}")


def healthy(value):
    if isinstance(value, dict):
        for name, child in value.items():
            if name in ("passed", "finite", "update_completed"):
                require(child is True, f"Passing report contains failed {name}")
            healthy(child)
    elif isinstance(value, list):
        for child in value:
            healthy(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Passing report contains nonfinite values")


def source_lineage(report, directory, project_root):
    hashes = report.get("source_hashes")
    require(isinstance(hashes, dict) and ESSENTIAL_SOURCES <= hashes.keys(), "Missing essential F3 source inventory")
    lineage = {}
    for name, expected in hashes.items():
        _source_name(name); require(_sha(expected), "Invalid source SHA256")
        current = project_root/name
        current_sha = file_digest(safe_evidence(project_root, name))["sha256"] if current.exists() or current.is_symlink() else None
        snapshot = f"source-snapshot/{name}"
        saved = directory/snapshot
        snapshot_sha = file_digest(safe_evidence(directory, snapshot))["sha256"] if saved.exists() or saved.is_symlink() else None
        require(snapshot_sha is None or snapshot_sha == expected, f"Corrupt run source snapshot: {name}")
        require(current_sha == expected or snapshot_sha == expected, f"Unreconstructable runtime source: {name}")
        lineage[name] = {"recorded_sha256": expected, "current_matches": current_sha == expected,
            "exact_snapshot": snapshot if snapshot_sha == expected else None}
    protocol_sha = report.get("protocol_sha256")
    require(_sha(protocol_sha), "Missing frozen protocol hash")
    protocol = "docs/reports/olmo1b-f3/protocol.md"
    require(file_digest(safe_evidence(project_root, protocol))["sha256"] == protocol_sha,
            "F3 protocol changed after execution")
    return lineage


def tensor_check(row):
    for name in ("relative_l2", "max_abs", "reference_max_abs", "relative_l2_limit", "max_abs_limit"):
        number(row.get(name), name)
    require(type(row.get("bitwise_equal")) is bool and row.get("finite") is True, "Missing finite tensor comparison")
    require(row["relative_l2_limit"] == 1e-5, "Original relative-L2 budget changed")
    close(row["max_abs_limit"], 1e-6 + 1e-5*row["reference_max_abs"], "original maximum-error budget")
    require(row.get("passed") is True and row["relative_l2"] <= 1e-5
            and row["max_abs"] <= row["max_abs_limit"], "Tensor comparison exceeds original budget")
    if row["bitwise_equal"]:
        require(row["relative_l2"] == row["max_abs"] == 0, "Bitwise equality contradicts tensor errors")


def metrics_check(metrics, config, update):
    require(metrics.get("schema") == "olmo-lm-optimizer-step-v1" and metrics.get("update_completed") is True,
            "Missing completed canonical optimizer update")
    counts = metrics.get("counts", {})
    require(set(counts) == set(TERMS), "Missing objective counts")
    for term in TERMS:
        integer(counts[term], term+" count")
        total = number(metrics.get("loss_sums", {}).get(term), term+" loss sum", signed=True)
        close(metrics.get("loss_means", {}).get(term), total/counts[term] if counts[term] else 0, term+" denominator")
    require(counts["ce"] > 0, "Missing CE supervision")
    enabled = config["case"] != "rt"
    expected_weights = {"ce": 1., "latent": float(enabled), "kl": float(enabled)}
    require(metrics.get("objective_weights") == expected_weights, "Canonical objective weights changed")
    close(metrics.get("objective"), sum(expected_weights[t]*metrics["loss_means"][t] for t in TERMS), "objective")
    number(metrics.get("gradient_norm_before_clip"), "gradient norm", positive=True)
    require(metrics.get("max_grad_norm") == 1., "Canonical clipping changed")
    for key in ("lr_used", "lr_next"):
        require(isinstance(metrics.get(key), list) and metrics[key], "Missing learning rates")
        for value in metrics[key]:
            number(value, key, positive=key == "lr_used")
    expected_counters = {"optimizer_updates": update, "microbatches": update,
        "documents": update*config["batch_size"], "input_tokens": update*config["batch_size"]*config["length"],
        "ce_positions": update*counts["ce"], "latent_pairs": update*counts["latent"], "kl_triples": update*counts["kl"]}
    require(metrics.get("counters") == expected_counters, "Optimizer counters differ from physical updates")
    healthy(metrics)


def _state_health(value):
    require(isinstance(value, dict) and value.get("passed") is True
            and value.get("nonfinite_parameters") == [] and value.get("nonfinite_optimizer_tensors") == [],
            "Missing finite model/optimizer state")


def correctness(report):
    config, checks = report["configuration"], report["checks"]
    named = {row["name"]: row for row in checks}
    require(len(checks) == len(named) and set(named) == {*TENSOR_CHECKS, UPDATE_CHECK},
            "Passed correctness run omitted/duplicated a required check")
    require(report.get("capture_succeeded") is True, "Correctness capture did not succeed")
    active = report.get("parameters", {}).get("active", {}).get("names")
    require(isinstance(active, list) and active and len(set(active)) == len(active), "Missing active parameter inventory")
    comparisons = []
    for name in TENSOR_CHECKS:
        check = named[name]
        require(check.get("ownership_matches") is True and set(check.get("gradients", {})) == set(active),
                "Gradient ownership does not cover all active parameters")
        losses = check.get("losses", {})
        passes = 1 if config["case"] == "rt" else 3 if config["case"] == "combined-k3" else 2
        require(set(losses) == {f"pass{p}/{term}" for p in range(passes) for term in TERMS},
                "Missing per-pass objective comparisons")
        values = [*losses.values(), *check["gradients"].values()]
        for value in values:
            tensor_check(value)
        bitwise = all(value["bitwise_equal"] for value in values)
        require(check.get("all_bitwise_equal") is bitwise, "Aggregate bitwise flag contradicts tensors")
        comparisons.append({"name": name, "gradient_tensor_count": len(active), "all_bitwise_equal": bitwise,
            "all_gradients_bitwise_equal": all(value["bitwise_equal"] for value in check["gradients"].values()),
            "max_gradient_relative_l2": max(value["relative_l2"] for value in check["gradients"].values()),
            "max_gradient_absolute_error": max(value["max_abs"] for value in check["gradients"].values())})
    update = named[UPDATE_CHECK]
    count = integer(update.get("updates_per_arm"), "updates per arm", minimum=1)
    require(count == config["updates"] and update.get("physical_optimizer_updates") == 2*count,
            "Complete-update execution count differs")
    require(all(update.get(name) is True for name in ("metrics_exact", "model_optimizer_scheduler_counters_exact", "weights_changed")),
            "Missing exact changed-weight complete-update parity")
    arms = update.get("arms")
    require(isinstance(arms, list) and len(arms) == 2 and [arm.get("replay") for arm in arms] == [False, True],
            "Missing eager/graph optimizer arms")
    for arm in arms:
        require(len(arm.get("metrics", [])) == count, "Wrong number of optimizer metrics")
        for index, metrics in enumerate(arm["metrics"], 1):
            metrics_check(metrics, config, index)
        boundary = arm.get("boundary")
        require(isinstance(boundary, dict) and set(boundary) == {"model", "optimizer", "scheduler", "counters"}
                and boundary["model"] and boundary["optimizer"], "Missing full-state comparison digests")
        require(boundary["counters"] == arm["metrics"][-1]["counters"], "Boundary counters disagree")
        _state_health(arm.get("health"))
    require(arms[0]["metrics"] == arms[1]["metrics"] and arms[0]["boundary"] == arms[1]["boundary"],
            "Claimed exact update differs between recorded arms")
    preparation = report.get("backward_preparation", {})
    require(preparation.get("warmup") == config["warmup"]+1 and preparation.get("capture") == 1
            and preparation.get("replay") == 5+count, "Backward preparation/replay counters differ")
    for key in ("peak_allocated_gib", "peak_reserved_gib"):
        number(report.get("memory", {}).get(key), key, positive=True)
    require(report["memory"]["peak_reserved_gib"] >= report["memory"]["peak_allocated_gib"], "Reserved memory below allocated peak")
    number(report.get("capture_seconds"), "capture seconds", positive=True)
    operators = report.get("observed_attention_operators")
    require(isinstance(operators, list) and operators and all(isinstance(name, str) for name in operators), "Missing observed backend operators")
    return {"configuration": config, "comparisons": comparisons, "active_gradient_tensors": len(active),
        "all_gradients_bitwise_equal": all(row["all_gradients_bitwise_equal"] for row in comparisons),
        "all_losses_and_gradients_bitwise_equal": all(row["all_bitwise_equal"] for row in comparisons),
        "complete_update_exact": True, "updates_per_arm": count, "physical_optimizer_updates": 2*count,
        "backward_preparation": preparation, "capture_seconds": report["capture_seconds"], "memory": report["memory"],
        "observed_attention_operators": operators,
        "parameters": {name: report["parameters"].get(name) for name in ("registered", "requires_grad", "active", "inference")}}


def timing(value, repeats, name):
    require(isinstance(value, dict), f"Missing {name} timing")
    for kind in ("wall", "cuda"):
        samples = value.get(kind+"_seconds")
        require(isinstance(samples, list) and len(samples) == repeats, f"Incomplete {name} timing samples")
        for sample in samples:
            number(sample, name+" sample", positive=True)
        close(value.get("median_"+kind+"_seconds"), statistics.median(samples), name+" median")


def capacity(report):
    config, rows = report["configuration"], report["rows"]
    require(isinstance(rows, list) and 1 <= len(rows) <= 2
            and [row.get("replay") for row in rows] == ([False] if len(rows) == 1 else [False, True]),
            "Capacity must retain ordered unique eager/graph arms")
    require([row.get("name") for row in report["checks"]] ==
            ["graph_complete_updates_finite" if row["replay"] else "eager_complete_updates_finite" for row in rows],
            "Capacity completion checks differ from measured arms")
    for row in rows:
        require(row.get("case") == config["case"] and row.get("batch_size") == config["batch_size"]
                and row.get("length") == config["length"] and row.get("checkpointing") is (config["checkpointing"] == "on"),
                "Capacity row does not match requested configuration")
        require(row.get("warmup_updates") == 3 and row.get("timed_updates") == config["repeats"]
                and row.get("gradient_initialization_backward") == 1
                and row.get("backward_only_warmup") == config["warmup"]
                and row.get("graph_capture_backward") == int(row["replay"]), "Capacity update/backward counts differ")
        for name in ("full_step", "forward_loss_backward"):
            timing(row.get(name), config["repeats"], name)
        require(row.get("input_tokens_per_update") == config["batch_size"]*config["length"], "Input token count differs")
        close(row["full_step"].get("valid_input_tokens_per_second"),
              row["input_tokens_per_update"]/row["full_step"]["median_wall_seconds"], "input throughput")
        if row["replay"]:
            number(row.get("capture_seconds"), "capacity capture seconds", positive=True)
            require(report.get("capture_succeeded") is True, "Graph arm lacks successful capture")
        else:
            require(row.get("capture_seconds") is None, "Eager arm claims graph capture")
        memory = row.get("capture_memory", {})
        for name in ("allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib"):
            number(memory.get(name), "capture "+name, positive=True)
        require(memory["reserved_gib"] >= memory["allocated_gib"] and
                memory["peak_reserved_gib"] >= memory["peak_allocated_gib"] >= memory["allocated_gib"], "Capture memory is contradictory")
        number(row.get("peak_allocated_gib"), "timing allocated peak", positive=True)
        number(row.get("peak_reserved_gib"), "timing reserved peak", positive=True)
        require(row["peak_reserved_gib"] >= row["peak_allocated_gib"] >= memory["peak_allocated_gib"], "Timing memory is contradictory")
        require(row.get("within_comfortable_budget") is (row["peak_allocated_gib"] <= config["comfortable_gib"]),
                "Capacity comfort flag contradicts peak memory")
        records = row.get("records", [])
        require(len(records) == config["repeats"], "Missing timed optimizer records")
        for index, metrics in enumerate(records, 4):
            metrics_check(metrics, config, index)
            require(metrics["counts"] == row.get("counts"), "Capacity counts differ from update records")
        _state_health(row.get("post_timing_health"))
        require(row.get("full_step_scope") and row.get("region_scope"), "Missing timing scope descriptions")
    stopped = any(not row["within_comfortable_budget"] for row in rows)
    require(bool(report.get("stopped_at_memory_budget", False)) == stopped,
            "Capacity memory-stop state contradicts measured arms")
    require(len(rows) == 2 or stopped, "Missing graph arm without explicit memory stop")
    return rows


def summarize(runtime_dirs, *, project_root=ROOT):
    summary = {"schema": SCHEMA, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "runs": [], "correctness": [], "capacity": [], "failed_diagnostics": [],
        "physical_optimizer_updates": {"prepared_eager": 0, "graph": 0},
        "optimizer_updates_by_comparison_arm": {"prepared_eager": 0, "graph": 0},
        "capture_seconds_scope": CAPTURE_SETUP_SCOPE,
        "memory_scopes": {
            "capture_memory.allocated_gib/reserved_gib": "Current memory after setup; postcapture for graph, after warmup for eager.",
            "capture_memory.peak_allocated_gib/peak_reserved_gib": "High-water marks through optimizer warmup and backward setup/capture, before timing.",
            "peak_allocated_gib/peak_reserved_gib": "High-water marks through setup, complete updates and region timing; not the continuing graph-pool footprint."},
        "scope": "Only supplied completed runs; capacity finite-update checks are separate from full-gradient/full-state equivalence."}
    names = set()
    for directory in runtime_dirs:
        directory = Path(directory).absolute()
        require(directory.name not in names, "Duplicate runtime name")
        names.add(directory.name)
        path = safe_evidence(directory, "report.json")
        report = json.loads(path.read_text())
        status, config = report.get("status"), report.get("configuration")
        require(report.get("schema") == REPORT_SCHEMA and report.get("finished_utc")
                and status in ("passed", "failed", "capture_blocked"), "Require a completed F3 report")
        require(isinstance(config, dict) and config.get("stage") in ("correctness", "capacity")
                and config.get("case") in ("rt", "combined", "combined-k3")
                and config.get("checkpointing") in ("off", "on"), "Invalid requested F3 case/stage")
        for key in ("batch_size", "length", "updates", "warmup", "repeats"):
            integer(config.get(key), key, minimum=1)
        require(config["batch_size"] <= 128 and config["length"] in (32, 128, 512) and config["updates"] <= 3,
                "Unexpected unbounded F3 shape/update scope")
        require(config["warmup"] >= 10 and config["repeats"] >= 3, "Insufficient warmup/timing scope")
        number(config.get("comfortable_gib"), "memory budget", positive=True)
        require(20 <= config["comfortable_gib"] <= 65, "Memory comfort budget is outside F3 scope")
        require(config.get("precision") == "bf16_mixed" and config.get("ordinary_sdpa_backend") == "flash"
                and config.get("deterministic_algorithms") is True and config.get("autocast_weight_cache") is False
                and config.get("gradient_accumulation") == 1 and config.get("rt_layers") == [0],
                "Unexpected precision/backend/RT/capture configuration")
        checkpoint = report.get("checkpoint", {})
        require(checkpoint.get("sha256") == CHECKPOINT_SHA256 and checkpoint.get("size_bytes") == CHECKPOINT_SIZE,
                "Unexpected native checkpoint")
        wandb = report.get("wandb", {})
        require(wandb.get("status") == ("synced" if status == "passed" else "synced_failed_experiment")
                and str(wandb.get("run_url", "")).startswith("https://wandb.ai/taylorbollman/"), "Missing synchronized W&B")
        checks = report.get("checks")
        require(isinstance(checks, list) and all(isinstance(row, dict) and isinstance(row.get("name"), str)
                and type(row.get("passed")) is bool for row in checks), "Malformed check inventory")
        lineage = source_lineage(report, directory, project_root)
        run = {"name": directory.name, "status": status, "stage": report.get("stage"), "configuration": config,
            "report_path": str(path), "report_sha256": file_digest(path)["sha256"], "source_lineage": lineage,
            "protocol_sha256": report["protocol_sha256"], "checkpoint": checkpoint, "runtime": report.get("runtime"),
            "wandb_url": wandb["run_url"], "capture_succeeded": report.get("capture_succeeded"),
            "limitations": report.get("limitations", []), "used_for_performance": False}
        if status != "passed":
            require(report.get("stage") and report["stage"] != "complete" and report.get("error_type")
                    and isinstance(report.get("error_message"), str), "Failed diagnostic lacks stage/error")
            if status == "capture_blocked":
                require(report["stage"] in ("capture", "capacity_capture") and report.get("capture_succeeded") is False,
                        "Capture blocker contradicts capture outcome")
            failure = {"run": directory.name, "configuration": config, "status": status, "stage": report["stage"],
                "capture_succeeded": report.get("capture_succeeded"), "error_type": report["error_type"],
                "error_message": report["error_message"], "checks": [{"name": row["name"], "passed": row["passed"]} for row in checks],
                "excluded_partial_capacity_rows": len(report.get("rows", [])), "used_for_performance": False}
            summary["failed_diagnostics"].append(failure)
            summary["status"] = "completed_with_failed_diagnostics"
        else:
            require(report.get("stage") == "complete" and checks and all(row["passed"] for row in checks), "Passing run lacks completed checks")
            healthy(report)
            if config["stage"] == "correctness":
                row = correctness(report) | {"run": directory.name, "wandb_url": wandb["run_url"]}
                summary["correctness"].append(row)
                for arm in ("prepared_eager", "graph"):
                    summary["physical_optimizer_updates"][arm] += row["updates_per_arm"]
                    summary["optimizer_updates_by_comparison_arm"][arm] += row["updates_per_arm"]
            else:
                rows = capacity(report)
                paired = len(rows) == 2
                summary["capacity"].append({"run": directory.name, "configuration": config,
                    "wandb_url": wandb["run_url"], "paired": paired,
                    "full_step_speedup": rows[0]["full_step"]["median_wall_seconds"]/rows[1]["full_step"]["median_wall_seconds"] if paired else None,
                    "region_speedup": rows[0]["forward_loss_backward"]["median_wall_seconds"]/rows[1]["forward_loss_backward"]["median_wall_seconds"] if paired else None,
                    "stopped_at_memory_budget": report.get("stopped_at_memory_budget", False), "rows": rows,
                    "validation_scope": "Finite complete updates only at this batch; not all-gradient/full-state parity."})
                run["used_for_performance"] = paired
                for row in rows:
                    arm = "graph" if row["replay"] else "prepared_eager"
                    summary["optimizer_updates_by_comparison_arm"][arm] += row["warmup_updates"]+row["timed_updates"]
                    # Both capacity arms prepare Adam with eager updates before
                    # the graph arm captures and starts its timed replays.
                    summary["physical_optimizer_updates"]["prepared_eager"] += row["warmup_updates"]
                    summary["physical_optimizer_updates"][arm] += row["timed_updates"]
        summary["runs"].append(run)
    require(summary["runs"], "No F3 reports supplied")
    summary["physical_optimizer_updates"]["total"] = sum(summary["physical_optimizer_updates"].values())
    summary["optimizer_updates_by_comparison_arm"]["total"] = sum(summary["optimizer_updates_by_comparison_arm"].values())
    require(summary["physical_optimizer_updates"]["total"] == summary["optimizer_updates_by_comparison_arm"]["total"],
            "Execution-mode and comparison-arm update totals differ")
    return summary


def markdown(summary):
    lines = ["# F3 canonical static-layout graph training", "", f"Status: **{summary['status']}**.", "",
        "Original OLMo-1B step60000 (~252B tokens), RT at layer index0, native RoPE/LayerNorm and tied readout; "
        "no Q/K normalization change. BF16 mixed compute, FP32 parameters/gradients/Adam state, deterministic "
        "ordinary Flash SDPA. Native RT remains eager tiled math executed inside the captured graph.", "",
        "These are bounded functionality and execution measurements, not quality or learning-speed results. "
        "Only the supplied configurations are covered; all-layer RT, cache continuation, distributed execution "
        "and graph serialization/checkpoint-resume are not cleared.", "",
        "## Correctness", "",
        "Graph versus prepared-eager uses the original relative-L2 <=1e-5 and maximum-error "
        "<=1e-6+1e-5×reference-max budgets. Bitwise equality is reported separately. Tiny FP32 canonical-versus-static "
        "tests are separate evidence. Each row includes changed tokens, repeated gradient overwrite, changed weights "
        "and complete AdamW/model/scheduler/counter comparisons.", "",
        "| Run | Case | B/T | Ordinary checkpoint | Active gradient tensors | All gradients bitwise | Full update exact | Updates eager + graph |",
        "|---|---|---:|---|---:|---|---|---:|"]
    for row in summary["correctness"]:
        c = row["configuration"]
        lines.append(f"| [{row['run']}]({row['wandb_url']}) | {c['case']} | {c['batch_size']}/{c['length']} | {c['checkpointing']} | "
            f"{row['active_gradient_tensors']} | {row['all_gradients_bitwise_equal']} | {row['complete_update_exact']} | "
            f"{row['updates_per_arm']} + {row['updates_per_arm']} |")
    if not summary["correctness"]:
        lines += ["", "No passing correctness report was supplied."]
    lines += ["", "## Complete-update capacity", "",
        "Capacity rows establish finite changing-input/changing-weight updates at their recorded shape; they do not "
        "substitute for all-gradient/full-state parity there. Three real preparation updates per arm initialize Adam; "
        "then ten or more backward warmups and three or more timed updates. Full-step wall times include validated input "
        "copies, forward/loss/backward or replay, clipping, AdamW and scheduler. Independently measured region timings "
        "exclude input copies and optimizer work. Input tokens/s counts one physical batch once, not internal FBT passes "
        "or CE-only targets.", "",
        "| Case | B/T | CP | Eager / graph input tokens/s | Full-step speedup | Region speedup | Eager / graph peak allocated GiB | Eager / graph peak reserved GiB |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for cell in summary["capacity"]:
        if not cell["paired"]:
            continue
        c, rows = cell["configuration"], cell["rows"]
        tps = " / ".join(f"{row['full_step']['valid_input_tokens_per_second']:,.0f}" for row in rows)
        allocated = " / ".join(f"{row['peak_allocated_gib']:.2f}" for row in rows)
        reserved = " / ".join(f"{row['peak_reserved_gib']:.2f}" for row in rows)
        lines.append(f"| [{c['case']}]({cell['wandb_url']}) | {c['batch_size']}/{c['length']} | {c['checkpointing']} | {tps} | "
            f"{cell['full_step_speedup']:.2f}× | {cell['region_speedup']:.2f}× | {allocated} | {reserved} |")
    if not any(cell["paired"] for cell in summary["capacity"]):
        lines += ["", "No completed paired capacity measurement was supplied."]
    lines += ["", "The raw capture_seconds field measures setup/warmup + capture, not just the CUDA graph-capture "
        "context. It wraps plan.capture, including backward warmup, synchronization and validation. Correctness setup "
        "also includes initial gradient discovery when not already initialized. This setup time is outside timed complete updates.", "",
        "| Run | Arm | Setup/warmup + capture seconds | Current postcapture allocated / reserved GiB (eager: after warmup) | Setup/capture peak allocated / reserved GiB | Within comfort budget |",
              "|---|---|---:|---:|---:|---|"]
    for cell in summary["capacity"]:
        for row in cell["rows"]:
            memory = row["capture_memory"]
            seconds = "—" if row["capture_seconds"] is None else f"{row['capture_seconds']:.3f}"
            lines.append(f"| {cell['run']} | {'graph' if row['replay'] else 'eager'} | {seconds} | "
                f"{memory['allocated_gib']:.2f} / {memory['reserved_gib']:.2f} | "
                f"{memory['peak_allocated_gib']:.2f} / {memory['peak_reserved_gib']:.2f} | {row['within_comfortable_budget']} |")
    lines += ["", "Current postcapture memory and reserved-memory peaks are different measurements. Warmup cache can be "
        "released on graph entry, so peak reserved memory is not the continuing graph-pool footprint. The first table's "
        "peaks cover setup, complete updates and region timing; the second table separates current memory after setup "
        "from setup/capture high-water marks. None of these totals isolates graph-private memory from model/optimizer memory.", "",
        "Memory-stop boundaries and incomplete pairs remain recorded; they do not become successful graph capacity points. "
        "Failed runs contribute no throughput values or plots. All timing samples, target counts, post-update health and memory fields remain in summary.json.", "",
        "## Diagnostics and provenance", ""]
    for failure in summary["failed_diagnostics"]:
        message = failure["error_message"].replace("\n", " ")
        lines.append(f"- **{failure['run']}: {failure['status']}**, stage `{failure['stage']}`; "
            f"{failure['error_type']}: {message}. Excluded partial capacity rows: {failure['excluded_partial_capacity_rows']}.")
    if not summary["failed_diagnostics"]:
        lines.append("No failed diagnostic was supplied.")
    updates = summary["physical_optimizer_updates"]
    arms = summary["optimizer_updates_by_comparison_arm"]
    lines += ["", f"Reported successful-run physical optimizer executions: {updates['total']} "
        f"({updates['prepared_eager']} prepared-eager updates + {updates['graph']} graph-replay updates). "
        f"Grouping by comparison arm instead gives {arms['prepared_eager']} eager-arm and {arms['graph']} graph-arm updates. "
        "Each graph capacity arm begins with three eager preparation updates, followed by its timed graph updates; "
        "those preparation updates count as eager execution even though they belong to the graph comparison arm. "
        "Correctness arms replay the same short trajectory. Backward warmup, capture and region-only calls "
        "do not advance optimizer counters. Partial executions from failed runs are excluded from this successful-run total.", "",
        "| Run | Status | Capture succeeded | Source version | W&B |", "|---|---|---|---|---|"]
    for run in summary["runs"]:
        current = all(value["current_matches"] for value in run["source_lineage"].values())
        lines.append(f"| {run['name']} | {run['status']} | {run['capture_succeeded']} | "
            f"{'current hashes match' if current else 'exact run snapshot required'} | [run]({run['wandb_url']}) |")
    lines += ["", "Actual ordinary attention operator traces from correctness runs:", ""]
    for row in summary["correctness"]:
        lines.append(f"- {row['run']}: " + ", ".join(f"`{name}`" for name in row["observed_attention_operators"]))
    lines += ["", f"Native checkpoint SHA256: `{CHECKPOINT_SHA256}` ({CHECKPOINT_SIZE:,} bytes). "
        "Per-run source/report/protocol hashes and exact snapshot lineage are retained in summary.json. "
        "The results use no widened numerical budgets and make no fused RT-kernel claim.", ""]
    return "\n".join(lines)


def plot(summary, output_dir):
    cells = [cell for cell in summary["capacity"] if cell["paired"]]
    if not cells:
        for suffix in (".png", ".pdf"):
            (output_dir/("throughput"+suffix)).unlink(missing_ok=True)
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(max(8., len(cells)*1.8), 4.6), constrained_layout=True)
    common = {(cell["configuration"]["length"], cell["configuration"]["checkpointing"],
               tuple(cell["configuration"]["rt_layers"])) for cell in cells}
    labels = []
    for cell in cells:
        config = cell["configuration"]
        case = {"rt": "RT", "combined": "Combined", "combined-k3": "Combined K3"}[config["case"]]
        label = f"{case}\nB{config['batch_size']}"
        if len(common) != 1:
            layers = ",".join(map(str, config["rt_layers"]))
            label += f" / T{config['length']}\nCP {config['checkpointing']}; RT {layers}"
        labels.append(label)
    for index, label in enumerate(("Prepared eager", "Graph complete step")):
        x = [position + (index-.5)*.36 for position in range(len(cells))]
        axes[0].bar(x, [cell["rows"][index]["full_step"]["valid_input_tokens_per_second"] for cell in cells], .36, label=label)
        axes[1].bar(x, [cell["rows"][index]["peak_allocated_gib"] for cell in cells], .36, label=label)
    axes[0].set_ylabel("Valid input tokens/s (complete update)")
    axes[1].set_ylabel("Peak allocated GiB")
    for axis in axes:
        axis.set_xticks(range(len(cells)), labels, fontsize=8)
        axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    maximum = max(row["full_step"]["valid_input_tokens_per_second"] for cell in cells for row in cell["rows"])
    axes[0].set_ylim(0, maximum*1.2)
    axes[0].legend(fontsize=8, loc="upper left", ncol=2)
    title = "F3 complete-update capacity"
    if len(common) == 1:
        length, checkpointing, layers = next(iter(common))
        title += f" — T{length}, ordinary checkpointing {checkpointing}, RT layer " + ",".join(map(str, layers))
    figure.suptitle(title+"\nCombined = RT + FBT (K2) + NextLat; K3 is named separately if present", fontsize=10)
    figure.supxlabel("Finite complete updates; full-gradient/state correctness is a separate scope.", fontsize=8)
    paths = [output_dir/("throughput"+suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        figure.savefig(path, dpi=170)
    plt.close(figure)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"docs/reports/olmo1b-f3")
    args = parser.parse_args(argv)
    summary = summarize(args.runtime_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/"summary.json", summary)
    (args.output_dir/"results.md").write_text(markdown(summary))
    paths = plot(summary, args.output_dir)
    print(json.dumps({"status": summary["status"], "runs": len(summary["runs"]), "plots": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
