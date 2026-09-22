#!/usr/bin/env python3
"""Summarize explicit F3b evidence and analytic resource cards without scope promotion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo_fbt import FBTMode
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.resource_estimates import LossWork, estimate_training_resources
from scripts.olmo_tiled_retain import CHECKPOINT_SHA256

SCHEMAS = {"olmo-f3b-profile-v1": "profile", "olmo-f3b-native-v1": "native",
           "olmo-f3b-tile-probe-v1": "tile", "olmo-fa4-smoke-v1": "fa4"}
NATIVE_CHECKS = {"same_state_variant_vs_reference", "candidate_initial_graph",
    "candidate_changed_tokens_and_overwrite", "complete_adamw_update_parity", "candidate_changed_weights"}
PROTOCOL = "docs/reports/olmo1b-f3b/protocol.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), f"Invalid {name}")
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(base, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and path.parts, "Unsafe evidence source path")
    resolved = (base/path).resolve()
    require(resolved.is_relative_to(base.resolve()), "Evidence source escapes its root")
    return resolved


def source_lineage(report, directory, project_root):
    hashes = report.get("source_hashes")
    if report["schema"] == "olmo-fa4-smoke-v1":
        hashes = {"scripts/olmo_fa4_smoke.py": report.get("script_sha256"),
                  "scripts/docker_shell.sh": report.get("launcher_sha256")}
    require(isinstance(hashes, dict) and hashes, "Missing source inventory")
    result = {}
    for name, expected in hashes.items():
        require(isinstance(expected, str) and len(expected) == 64 and
                all(c in "0123456789abcdef" for c in expected), "Malformed source SHA256")
        current = source_path(project_root, name)
        snapshot = source_path(directory, "source-snapshot/"+name)
        current_matches = current.is_file() and digest(current) == expected
        snapshot_matches = snapshot.is_file() and digest(snapshot) == expected
        require(not snapshot.is_file() or snapshot_matches, f"Corrupt source snapshot: {name}")
        require(current_matches or snapshot_matches, f"Unreconstructable source: {name}")
        result[name] = {"sha256": expected, "current_matches": current_matches,
            "snapshot": "source-snapshot/"+name if snapshot_matches else None}
    expected = report.get("protocol_sha256")
    protocol = None
    if expected is not None:
        candidates = ((directory/"protocol.md", "protocol.md"),
            (directory/"source-snapshot"/PROTOCOL, "source-snapshot/"+PROTOCOL),
            (project_root/PROTOCOL, "current:"+PROTOCOL))
        available = [(path, name) for path, name in candidates if path.is_file()]
        for path, name in available:
            if not name.startswith("current:"):
                require(digest(path) == expected, "Corrupt frozen protocol snapshot")
        matches = [name for path, name in available if digest(path) == expected]
        require(matches, "Unreconstructable frozen protocol")
        protocol = {"sha256": expected, "verified_at": matches}
    return {"sources": result, "protocol": protocol,
            "external_interface_record": report.get("interface")}


def timed(value, name):
    require(isinstance(value, dict), f"Missing {name} timing")
    for clock in ("wall", "cuda"):
        samples = value.get(clock+"_seconds")
        require(isinstance(samples, list) and len(samples) >= 3, f"Insufficient {name} samples")
        for sample in samples:
            number(sample, name+" sample", positive=True)
        median = number(value.get("median_"+clock+"_seconds"), name+" median", positive=True)
        require(math.isclose(median, statistics.median(samples), rel_tol=1e-10, abs_tol=1e-12),
                f"Inconsistent {name} median")
    return value


def failed_paths(value, prefix=""):
    """Keep diagnostic failures visible without replacing declared acceptance gates."""
    result = []
    if isinstance(value, dict):
        for name, child in value.items():
            path = f"{prefix}.{name}" if prefix else name
            if name in ("passed", "finite") and child is False:
                result.append(path)
            else:
                result.extend(failed_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(failed_paths(child, f"{prefix}[{index}]"))
    return result


def declared_checks(report):
    checks = report.get("checks", [])
    require(isinstance(checks, list), "Malformed check list")
    names = []
    rows = []
    for check in checks:
        require(isinstance(check, dict) and isinstance(check.get("name"), str)
                and type(check.get("passed")) is bool, "Malformed declared check")
        names.append(check["name"])
        rows.append({"name": check["name"], "passed": check["passed"],
            "kind": check.get("kind"), "all_bitwise_equal": check.get("all_bitwise_equal"),
            "acceptance_policy": check.get("acceptance_policy"),
            "nested_diagnostic_failures": failed_paths({k:v for k,v in check.items() if k != "passed"}),
            "record": check})
    require(len(set(names)) == len(names), "Duplicate declared checks")
    if report["status"] == "passed" and SCHEMAS[report["schema"]] in ("native", "tile"):
        require(checks and all(c["passed"] for c in checks), "Passed run contradicts declared top-level check gates")
    return rows


def native_correctness(report):
    named = {c["name"]: c for c in report["checks"]}
    require(set(named) == NATIVE_CHECKS, "Passed native correctness lacks required checks")
    variant = report["configuration"]["variant"]
    from scripts.olmo_f3_report import tensor_check
    expected_gradients = set(named["same_state_variant_vs_reference"].get("gradients", {}))
    expected_losses = set(named["same_state_variant_vs_reference"].get("losses", {}))
    for name in NATIVE_CHECKS - {"complete_adamw_update_parity"}:
        check = named[name]
        require(check.get("gradients") and check.get("losses"), "Missing gradient/loss comparisons")
        require(set(check["gradients"]) == expected_gradients and set(check["losses"]) == expected_losses,
                "Changed gradient ownership or loss inventory across candidate checks")
        require(check.get("ownership_matches", True) is True, "Declared gradient ownership mismatch")
        # The initial mixed-candidate comparison deliberately retains stricter
        # same-arithmetic diagnostics. Its top-level engineering screen is the gate.
        if name == "same_state_variant_vs_reference" and variant == "triton":
            require(check.get("mixed_gradient_screen") and check.get("mixed_loss_screen"),
                    "Missing candidate mixed-precision screens")
            number(check.get("global_gradient_relative_l2"), "candidate global gradient error")
            require(check["global_gradient_relative_l2"] <= 1/64, "Candidate global gradient screen failed")
            for row in check["mixed_gradient_screen"].values():
                require(number(row["relative_l2"], "gradient error") <= 1/32 and
                        number(row["max_relative"], "gradient maximum error") <= 1/16,
                        "Candidate per-tensor gradient screen failed")
            for row in check["mixed_loss_screen"].values():
                require(number(row["relative_l2"], "loss error") <= 1/64, "Candidate loss screen failed")
        else:
            for row in (*check["losses"].values(), *check["gradients"].values()):
                tensor_check(row)
    update = named["complete_adamw_update_parity"]
    arms = update.get("arms")
    require(isinstance(arms, list) and len(arms) == 2 and
            [a.get("replay") for a in arms] == [False, True], "Missing complete-update arms")
    require(all(update.get(k) is True for k in ("metrics_exact", "model_optimizer_scheduler_counters_exact", "weights_changed")),
            "Incomplete exact update evidence")
    require(arms[0]["metrics"] == arms[1]["metrics"] and arms[0]["boundary"] == arms[1]["boundary"],
            "Recorded optimizer arms contradict exact parity")
    updates = update.get("updates_per_arm")
    require(type(updates) is int and updates == 3 and update.get("physical_optimizer_updates") == 2*updates,
            "Wrong complete-update count")
    require(all(len(a.get("metrics", [])) == updates and a.get("health", {}).get("passed") is True for a in arms),
            "Incomplete complete-update records")
    return {"physical_optimizer_updates": 2*updates,
            "variant_vs_reference_bitwise": named["same_state_variant_vs_reference"].get("all_bitwise_equal"),
            "candidate_graph_checks_bitwise": all(named[n].get("all_bitwise_equal") is True for n in
                ("candidate_initial_graph", "candidate_changed_tokens_and_overwrite", "candidate_changed_weights")),
            "complete_updates_exact": True}


def full_step_record(report, run_name, *, profile=False):
    config = report["configuration"]
    data = report if profile else report["capacity"]
    timing = timed(data.get("full_step"), "complete update")
    batch, length = config.get("batch_size"), config.get("length")
    require(type(batch) is int and batch > 0 and type(length) is int and length > 0, "Bad measured shape")
    tokens = batch*length
    tps = number(data.get("input_tokens_per_second"), "input tokens/s", positive=True)
    require(math.isclose(tps, tokens/timing["median_wall_seconds"], rel_tol=1e-10), "Inconsistent input throughput")
    records = data.get("records", [])
    require(len(records) == 3 and all(r.get("update_completed") is True for r in records), "Missing timed full updates")
    for step, record in enumerate(records, 4):
        counters = record.get("counters", {})
        require(counters.get("optimizer_updates") == step and counters.get("input_tokens") == step*tokens,
                "Full-step exposure/optimizer counters differ")
    require(data.get("health", {}).get("passed") is True, "Missing post-update health")
    allocated = number(data.get("peak_allocated_gib"), "peak allocated memory", positive=True)
    reserved = data.get("peak_reserved_gib")
    current = data.get("current_reserved_gib")
    if reserved is not None:
        require(number(reserved, "peak reserved memory", positive=True) >= allocated, "Reserved peak below allocated")
    if current is not None:
        require(number(current, "current reserved memory", positive=True) <= reserved, "Current reserved exceeds peak")
    return {"run": run_name, "case": config["case"], "variant": config.get("variant", "reference") if profile else config["variant"],
        "batch_size": batch, "length": length, "input_tokens_per_update": tokens,
        "input_tokens_per_second": tps, "ce_targets_per_second": records[0]["counts"]["ce"]/timing["median_wall_seconds"],
        "counts": records[0]["counts"], "full_step": timing,
        "peak_allocated_gib": allocated, "peak_reserved_gib": reserved, "current_reserved_gib": current,
        "physical_optimizer_updates": 6, "provenance": "pre-profile uninstrumented timing" if profile else "native capacity",
        "wandb_url": report.get("wandb", {}).get("run_url"), "historical": False,
        "scope": "Input validation/copy + graph forward/loss/backward + clipping/AdamW/scheduler; three eager preparation and three timed graph updates.",
        "memory_scope": "Peak allocated covers setup and timed updates. Reserved peak and current reserved are distinct; not isolated graph-private memory."}


def profile_record(report, run_name):
    graph = report.get("graph_profile")
    annotated = report.get("annotated_eager_profile")
    require(isinstance(graph, list) and graph and isinstance(annotated, list) and annotated, "Missing device profile")
    # CUDA annotation ranges can appear as device events too. They are not
    # kernels and cannot be added to the kernel total a second time.
    kernels = [r for r in graph if "CUDA" in r.get("device_type", "") and not r.get("name", "").startswith("RT/")]
    for row in kernels:
        number(row.get("self_device_us"), "kernel device time")
        require(type(row.get("count")) is int and row["count"] > 0, "Invalid kernel invocation count")
    total = sum(r["self_device_us"] for r in kernels)
    phases = [r for r in annotated if r.get("name", "").startswith("RT/")]
    return {"run": run_name, "configuration": report["configuration"],
        "graph_kernel_invocations": sum(r["count"] for r in kernels),
        "graph_kernel_self_device_us": total,
        "graph_kernel_rows": sorted(kernels, key=lambda r:r["self_device_us"], reverse=True),
        "annotated_eager_ranges": phases,
        "scope": "One observed graph backward and a separate annotated eager backward. Kernel self-time sums are not wall throughput; RT annotation ranges are not additional kernels."}


def resource_cards():
    config = OLMoConfig.native_1b()
    predictor = NextLatConfig(config.model_dim)
    rows = []
    for fbt in (False, True):
        for rt in (False, True):
            for nextlat in (False, True):
                names = ["RT layer0" if rt else "Ordinary"]
                if fbt: names.append("FBT K2")
                if nextlat: names.append("NextLat")
                estimate = estimate_training_resources(config, batch_size=64, sequence_length=512,
                    mode=FBTMode(enabled=fbt, rt_mode=RTMode((0,) if rt else ())),
                    nextlat=predictor if nextlat else None,
                    loss_work=LossWork(16384, 32704, 16384, 32704) if nextlat else LossWork(16384),
                    ordinary_checkpointing=True)
                rows.append({"name": " + ".join(names), "rt": rt, "fbt": fbt, "nextlat": nextlat,
                             "estimate": estimate.to_dict()})
    return rows


def summarize(runtime_dirs, *, project_root=ROOT, allow_incomplete=False, f3_baselines=()):
    project_root = Path(project_root)
    summary = {"schema": "olmo-f3b-summary-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "runs": [], "full_steps": [], "helper_timings": [], "profiles": [],
        "correctness": [], "fa4": [], "failed_diagnostics": [], "incomplete": [],
        "resource_cards": resource_cards(), "successful_f3b_optimizer_updates": 0,
        "scope": "Explicit supplied F3b reports only. No quality, all-layer RT, fused-backward or multi-GPU clearance."}
    seen = set()
    for item in runtime_dirs:
        directory = Path(item).resolve()
        require(directory.name not in seen, "Duplicate runtime name")
        seen.add(directory.name)
        path = directory/"report.json"
        if not path.is_file():
            require(allow_incomplete and directory.is_dir(), "Runtime report does not exist")
            summary["incomplete"].append({"run": directory.name, "status": "no_report_yet"})
            continue
        report = json.loads(path.read_text())
        require(report.get("schema") in SCHEMAS, "Unsupported F3b report schema")
        kind, status = SCHEMAS[report["schema"]], report.get("status")
        if status == "running":
            require(allow_incomplete, "A run is still active; use --allow-incomplete only for preview")
            summary["incomplete"].append({"run": directory.name, "status": status,
                "stage": report.get("stage"), "reported_checks": len(report.get("checks", []))})
            continue
        require(status in ("passed", "failed", "capture_blocked") and report.get("finished_utc"), "Require completed report status")
        lineage = source_lineage(report, directory, project_root)
        checks = declared_checks(report)
        run = {"name": directory.name, "kind": kind, "status": status,
            "configuration": report.get("configuration", {}), "runtime": report.get("runtime"),
            "source_lineage": lineage, "report_sha256": digest(path), "report_path": str(path),
            "checks": checks, "wandb_url": report.get("wandb", {}).get("run_url"),
            "used_for_performance": False}
        summary["runs"].append(run)
        if status != "passed":
            summary["failed_diagnostics"].append({"run": directory.name, "status": status,
                "stage": report.get("stage"), "error_type": report.get("error_type"),
                "error_message": report.get("error_message"), "declared_failed_checks": [r["name"] for r in checks if not r["passed"]],
                "excluded_partial_performance": bool(report.get("capacity") or report.get("full_step") or any("timing" in r["record"] for r in checks))})
            continue
        if kind in ("native", "profile"):
            require(report.get("checkpoint", {}).get("sha256") == CHECKPOINT_SHA256, "Wrong native checkpoint")
        if kind == "profile":
            full = full_step_record(report, directory.name, profile=True)
            summary["full_steps"].append(full)
            summary["profiles"].append(profile_record(report, directory.name))
            summary["successful_f3b_optimizer_updates"] += 6
            run["used_for_performance"] = True
        elif kind == "native":
            if report["configuration"].get("stage") == "correctness":
                row = native_correctness(report)
                summary["correctness"].append({"run": directory.name, "configuration": report["configuration"], **row})
                summary["successful_f3b_optimizer_updates"] += row["physical_optimizer_updates"]
            elif report["configuration"].get("stage") == "capacity":
                require({c["name"] for c in checks} == {"finite_complete_updates"}, "Missing capacity completion gate")
                summary["full_steps"].append(full_step_record(report, directory.name))
                summary["successful_f3b_optimizer_updates"] += 6
                run["used_for_performance"] = True
            else:
                raise ValueError("Unknown native stage")
        elif kind == "tile":
            selected = report["configuration"].get("stage")
            expected = {"all": 60, "tiles": 48, "blocks": 12}.get(selected)
            require(len(checks) == expected, "Passing tile probe omitted requested cases")
            for row in checks:
                record = row["record"]
                if "timing" in record:
                    for arm in ("eager", "triton"):
                        timed(record["timing"][arm], "helper "+arm)
                    summary["helper_timings"].append({"run": directory.name, "check": row["name"],
                        "configuration": record["configuration"], "timing": record["timing"],
                        "measurement_scope": "Uncaptured historical-tile call timed with CUDA events; includes host submission gaps between kernels, not isolated kernel latency."})
                    run["used_for_performance"] = True
        else:
            require(report.get("capture_output_exact") is True and set(report.get("comparisons", {})) == {"output", "dq", "dk", "dv"},
                    "Missing FA4 forward/backward/capture evidence")
            for row in report["comparisons"].values():
                number(row["relative_l2"], "FA4 relative error")
                number(row["max_relative"], "FA4 maximum error")
            summary["fa4"].append({"run": directory.name, "versions": report.get("versions"),
                "comparisons": report["comparisons"], "capture_output_exact": True,
                "scope": "Standalone standard FA4 attention smoke; does not validate RT temporary-diagonal or permanent-write semantics."})
    if f3_baselines:
        from scripts.olmo_f3_report import summarize as summarize_f3
        historical = summarize_f3(f3_baselines, project_root=project_root)
        summary["historical_f3_lineage"] = historical["runs"]
        for cell in historical["capacity"]:
            for row in cell["rows"]:
                if not row["replay"]:
                    continue
                config = cell["configuration"]
                summary["full_steps"].append({"run": cell["run"], "case": config["case"], "variant": "F3 reference",
                    "batch_size": config["batch_size"], "length": config["length"],
                    "input_tokens_per_update": row["input_tokens_per_update"],
                    "input_tokens_per_second": row["full_step"]["valid_input_tokens_per_second"],
                    "ce_targets_per_second": row["counts"]["ce"]/row["full_step"]["median_wall_seconds"],
                    "counts": row["counts"], "full_step": row["full_step"], "peak_allocated_gib": row["peak_allocated_gib"],
                    "peak_reserved_gib": row["peak_reserved_gib"], "current_reserved_gib": row["capture_memory"]["reserved_gib"],
                    "historical": True, "provenance": "separately validated historical F3 capacity", "wandb_url": cell["wandb_url"]})
    require(summary["runs"] or summary["incomplete"], "No F3b runs supplied")
    if summary["incomplete"]:
        summary["status"] = "partial_preview"
    elif summary["failed_diagnostics"]:
        summary["status"] = "completed_with_failed_diagnostics"
    summary["declared_check_count"] = sum(len(r["checks"]) for r in summary["runs"])
    summary["declared_checks_passed"] = sum(c["passed"] for r in summary["runs"] for c in r["checks"])
    return summary


def markdown(summary):
    lines = ["# F3b native RT execution evidence", "", f"Status: **{summary['status']}**.", "",
        "Bounded functionality and execution measurements on original OLMo-1B step60000. Native-model rows use "
        "RT at layer index0, BF16 mixed compute with FP32 weights/gradients/Adam, ordinary Flash and activation "
        "checkpointing. No quality, all-layer RT, fused RT backward or multi-GPU result is implied.", "",
        "## Evidence and correctness", "",
        "| Run | Scope | Status | Declared checks passed / total |",
        "| --- | --- | --- | ---: |"]
    for run in summary["runs"]:
        name = f"[{run['name']}]({run['wandb_url']})" if run.get("wandb_url") else run["name"]
        lines.append(f"| {name} | {run['kind']} | {run['status']} | {sum(c['passed'] for c in run['checks'])} / {len(run['checks'])} |")
    lines += ["", f"Successful F3b reports record {summary['successful_f3b_optimizer_updates']} physical optimizer updates. "
        "Backward-only warmup/capture/profile calls and historical F3 measurements are excluded from that count.", "",
        "Declared top-level checks determine acceptance. Nested stricter or independent-oracle diagnostics remain "
        "visible in summary.json, including false flags; they are not silently promoted to passes or recursively "
        "substituted for the run's stated gate. Candidate/eager graph parity retains the stricter F3 tolerance. "
        "FP64 reduction can land on a different BF16 rounding tie, so independent-oracle state diagnostics are "
        "reported separately from the candidate/eager FP32-state gate."]
    for row in summary["correctness"]:
        c = row["configuration"]
        lines += ["", f"- {row['run']}: {c['variant']}, {c['case']}, B{c['batch_size']}/T{c['length']}; "
            f"reference comparison bitwise={row['variant_vs_reference_bitwise']}, candidate graph comparisons "
            f"bitwise={row['candidate_graph_checks_bitwise']}; three complete updates per arm match exactly."]
    for row in summary["fa4"]:
        worst = max(c["relative_l2"] for c in row["comparisons"].values())
        lines += ["", f"FA4 smoke {row['run']}: forward/dQ/dK/dV recorded; maximum relative-L2 error {worst:.4g}; "
            "capture output exact. This confirms standalone FA4 availability, not native RT kernel equivalence."]
    lines += ["", "## Measured complete updates", "",
        "Full-step input tokens/s includes input validation/copy, graph forward/loss/backward, clipping, AdamW "
        "and scheduler. It counts each input once; FBT passes do not multiply data exposure. Profile-run timing "
        "was collected before instrumentation. Capacity checks establish finite updates, not every-gradient parity "
        "at the larger batch. Reserved setup peaks and current reserved memory are distinct.", "",
        "| Run / variant | Case | B/T | Input tokens/s | CE targets/s | Seconds/update | Peak allocated / reserved GiB | Current reserved GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    for r in summary["full_steps"]:
        lines.append(f"| {r['run']} / {r['variant']} | {r['case']} | {r['batch_size']}/{r['length']} | "
            f"{r['input_tokens_per_second']:,.0f} | {r['ce_targets_per_second']:,.0f} | {r['full_step']['median_wall_seconds']:.4f} | "
            f"{fmt(r['peak_allocated_gib'])} / {fmt(r['peak_reserved_gib'])} | {fmt(r['current_reserved_gib'])} |")
    if not summary["full_steps"]:
        lines += ["", "No passing complete-update timing was supplied."]
    lines += ["", "## Device attribution and helper timings", "",
        "RT/* entries are observer annotation ranges, including CUDA-attributed ranges; they are not additional "
        "kernels. CPU and CUDA annotations must not be added together. Graph kernel self-time sums can overlap "
        "and are not wall-clock update times."]
    for profile in summary["profiles"]:
        lines += ["", f"{profile['run']}: {profile['graph_kernel_invocations']:,} graph kernel invocations; "
            f"aggregate recorded kernel self-device time {profile['graph_kernel_self_device_us']/1e6:.3f}s.", "",
            "| Graph kernel (top 8 by recorded self-device time) | Count | Self-device ms |", "| --- | ---: | ---: |"]
        for row in profile["graph_kernel_rows"][:8]:
            label = row["name"].replace("|", "\\|")
            lines.append(f"| `{label}` | {row['count']:,} | {row['self_device_us']/1000:.3f} |")
        lines += ["", "| Observer annotation | Device attribution | Count | Inclusive CPU ms | Attributed device ms |",
            "| --- | --- | ---: | ---: | ---: |"]
        for row in profile["annotated_eager_ranges"]:
            lines.append(f"| {row['name']} | {row['device_type']} | {row['count']:,} | "
                f"{row['cpu_total_us']/1000:.3f} | {row['device_total_us']/1000:.3f} |")
    lines += ["", "Helper CUDA-event elapsed time covers an uncaptured historical-tile forward call. "
        "The interval includes host submission gaps while Python dispatches kernels, so it is not isolated kernel "
        "latency or device compute time. It excludes RT projection/MLP/backward, full loss and optimizer work, "
        "and cannot be presented as full-model throughput.", "",
        "| Helper case | Source / target / head width | Eager / fused CUDA-event microseconds, including host gaps |", "| --- | ---: | ---: |"]
    for row in summary["helper_timings"]:
        c, t = row["configuration"], row["timing"]
        lines.append(f"| {row['check']} | {c['width']} / {c['target_width']} / {c['head_dim']} | "
            f"{t['eager']['median_cuda_seconds']*1e6:.1f} / {t['triton']['median_cuda_seconds']*1e6:.1f} |")
    lines += ["", "## Common analytic resource cards", "",
        "All eight rows use B64/T512, ordinary checkpointing, one unpadded document per row, CE=16,384; "
        "enabled NextLat has 32,704 latent/predictor positions and 16,384 KL triples. These are analytic matrix "
        "FLOP estimates, not eight throughput experiments. Counts include permanent full-QKV writes, custom "
        "backward reconstruction, pass work, fusion, predictor and checkpointed full-vocabulary readouts. "
        "Multiply-add=2; optimizer, pointwise/RoPE/softmax, launch overhead and hardware padding are excluded. "
        "The range covers ordinary attention and checkpoint early-stop assumptions. Shared/tied weights count "
        "once; inactive resident modules require the actual ownership inventory. See [derivation](../../olmo-resource-accounting.md).", "",
        "| Architecture | Training parameters | Deployable inference parameters | Matrix TFLOPs/update |",
        "| --- | ---: | ---: | ---: |"]
    for row in summary["resource_cards"]:
        e = row["estimate"]; p = e["parameter_counts"]
        lines.append(f"| {row['name']} | {p['training_architecture']:,} | {p['deployable_inference']:,} | "
            f"{e['matrix_flops_minimum']/1e12:.1f}–{e['matrix_flops_maximum']/1e12:.1f} |")
    lines += ["", "## Failures, incomplete scope and provenance", ""]
    if not summary["failed_diagnostics"]:
        lines.append("No failed diagnostic was supplied.")
    for f in summary["failed_diagnostics"]:
        lines.append(f"- **{f['run']}: {f['status']}**, stage {f['stage']}; {f['error_type']}: {f['error_message']}. "
            "Its partial timings are excluded from performance tables and plots.")
    for item in summary["incomplete"]:
        lines.append(f"- **{item['run']}: {item['status']}**; preview only, no performance promotion.")
    lines += ["", "summary.json pins each input report and verifies recorded source versions against exact snapshots "
        "or matching current files. Historical source versions remain distinct. FA4's external installed interface "
        "hash is recorded provenance; it is not revalidated against a different host installation. "
        "No run is discovered automatically and missing experiments remain untested.", ""]
    return "\n".join(lines)


def plot(summary, output_dir):
    output_dir = Path(output_dir)
    if not summary["full_steps"] and not summary["helper_timings"]:
        for suffix in ("png", "pdf"):
            (output_dir/f"throughput.{suffix}").unlink(missing_ok=True)
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, helpers = summary["full_steps"], summary["helper_timings"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    if rows:
        labels = [f"{r['case']} B{r['batch_size']} T{r['length']}\n{r['variant']}"
            + (" · profile run" if r["provenance"] == "pre-profile uninstrumented timing" else "") for r in rows]
        axes[0].barh(range(len(rows)), [r["input_tokens_per_second"]/1000 for r in rows], color="#297b99")
        axes[0].set_yticks(range(len(rows)), labels, fontsize=8); axes[0].invert_yaxis()
        axes[0].set_xlabel("Input tokens/s (thousands)")
        axes[0].set_title("Complete updates · uninstrumented")
        axes[0].grid(axis="x", alpha=.2); axes[0].set_axisbelow(True)
    else:
        axes[0].text(.5, .5, "No passing full-step timing", ha="center", transform=axes[0].transAxes)
        axes[0].axis("off")
    if helpers:
        y = list(range(len(helpers)))
        for offset, arm, color in ((-.18, "eager", "#8996a0"), (.18, "triton", "#cf793d")):
            axes[1].barh([i+offset for i in y], [r["timing"][arm]["median_cuda_seconds"]*1e6 for r in helpers],
                height=.33, label=arm, color=color)
        axes[1].set_yticks(y, [f"W{r['configuration']['width']} / D{r['configuration']['head_dim']}" for r in helpers], fontsize=8)
        axes[1].invert_yaxis(); axes[1].set_xlabel("CUDA-event elapsed time (microseconds)")
        axes[1].set_title("Uncaptured tile forward call\nHost submission gaps included", fontsize=10)
        axes[1].legend(); axes[1].grid(axis="x", alpha=.2); axes[1].set_axisbelow(True)
    else:
        axes[1].text(.5, .5, "No passing helper timing", ha="center", transform=axes[1].transAxes)
        axes[1].axis("off")
    fig.suptitle("F3b native RT execution · measured scopes kept separate", fontsize=12)
    fig.text(.5, .01, "Failed/incomplete runs excluded. Combined = FBT K2 + RT layer0 + NextLat. No quality or all-layer RT claim.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "pdf"):
        path = output_dir/f"throughput.{suffix}"
        fig.savefig(path, dpi=160, bbox_inches="tight"); files.append(str(path))
    plt.close(fig)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--f3-baseline-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=ROOT/"docs/reports/olmo1b-f3b")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    summary = summarize(args.runtime_dir, allow_incomplete=args.allow_incomplete, f3_baselines=args.f3_baseline_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/"summary.json", summary)
    (args.output_dir/"results.md").write_text(markdown(summary))
    files = plot(summary, args.output_dir)
    print({"status": summary["status"], "runs": len(summary["runs"]), "plots": files}, flush=True)
    return summary


if __name__ == "__main__":
    main()
