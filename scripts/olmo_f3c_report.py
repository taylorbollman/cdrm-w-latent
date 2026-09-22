#!/usr/bin/env python3
"""Summarize explicitly supplied F3c backward-fusion evidence and scope."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from scripts import olmo_f3b_report as prior
from scripts.olmo_f3_report import tensor_check

SCHEMAS = {"olmo-f3c-native-v1": "native", "olmo-f3c-profile-v1": "profile", "olmo-f3c-tile-probe-v1": "tile"}
PROTOCOL = "docs/reports/olmo1b-f3c/protocol.md"
require, number, digest = prior.require, prior.number, prior.digest


def source_lineage(report, directory, project_root):
    # Reuse source validation without replacing its immutable F3b protocol
    # constant or modifying any global state in the earlier reporter.
    without_protocol = {k:v for k,v in report.items() if k != "protocol_sha256"}
    lineage = prior.source_lineage(without_protocol, directory, project_root)
    expected = report.get("protocol_sha256")
    require(isinstance(expected, str) and len(expected) == 64, "Missing frozen F3c protocol hash")
    candidates = ((directory/"protocol.md", "protocol.md"),
        (directory/"source-snapshot"/PROTOCOL, "source-snapshot/"+PROTOCOL),
        (project_root/PROTOCOL, "current:"+PROTOCOL))
    found = []
    for path, label in candidates:
        if not path.is_file():
            continue
        matches = digest(path) == expected
        if not label.startswith("current:"):
            require(matches, "Corrupt F3c protocol snapshot")
        if matches:
            found.append(label)
    require(found, "Unreconstructable F3c protocol")
    lineage["protocol"] = {"sha256": expected, "verified_at": found}
    return lineage


def declared_checks(report):
    kind = SCHEMAS[report["schema"]]
    adapted = {**report, "schema": {"native": "olmo-f3b-native-v1", "profile": "olmo-f3b-profile-v1",
                                    "tile": "olmo-f3b-tile-probe-v1"}[kind]}
    checks = prior.declared_checks(adapted)
    if report["status"] == "passed":
        require(checks and all(c["passed"] for c in checks), "Passed report contradicts declared gates")
    return checks


def engineering_gradients(record, names=None):
    require(record.get("passed") is True, "Primary gradient comparison did not pass")
    global_error = number(record.get("global_gradient_relative_l2"), "global gradient error")
    require(record.get("global_gradient_relative_l2_limit") == 1/64 and global_error <= 1/64,
            "Global gradient budget changed or exceeded")
    gradients = record.get("gradients")
    require(isinstance(gradients, dict) and gradients, "Missing gradient tensor inventory")
    if names is not None:
        require(set(gradients) == set(names), "Wrong gradient tensor inventory")
    for row in gradients.values():
        require(row.get("passed") is True and row.get("finite") is True and row.get("shape_matches") is True,
                "Primary gradient contains failed tensor flags")
        require(row.get("relative_l2_limit") == 1/32 and row.get("max_error_reference_max_limit") == 1/16,
                "Per-tensor gradient budget changed")
        require(number(row.get("relative_l2"), "tensor relative error") <= 1/32 and
                number(row.get("max_error_reference_max"), "tensor maximum error") <= 1/16,
                "Per-tensor gradient budget exceeded")
        if number(row.get("reference_max_abs"), "reference maximum") == 0:
            require(number(row.get("max_abs"), "absolute error") == 0, "Zero reference requires exact zero")


def tile_checks(report):
    checks = report["checks"]
    stage = report["configuration"].get("stage")
    expected = {"all": (48, 12), "tiles": (48, 0), "blocks": (0, 12)}.get(stage)
    require(expected is not None, "Unknown tile probe stage")
    tiles = [r for r in checks if r.get("kind") == "frozen_backward_tile"]
    blocks = [r for r in checks if r.get("kind") == "native_tiny_block"]
    require((len(tiles), len(blocks)) == expected and len(tiles)+len(blocks) == len(checks),
            "Passed probe omitted/changed requested cases")
    if blocks:
        require({(r["configuration"]["length"], r["configuration"]["prefix"], r["configuration"]["alpha"]) for r in blocks} ==
                {(t,p,a) for t in (9,17) for p in (0,3) for a in (0.,.37,1.)}, "Missing requested block geometry/alpha coverage")
    if tiles:
        variants = {"nonzero", "strided_masked", "zero_probability", "zero_adjoint", "large_adjoint", "small_probability_large_dot"}
        require({(r["configuration"]["width"],r["configuration"]["variant"]) for r in tiles} ==
                {(w,v) for w in (1,3,8,17,32,64,128,256) for v in variants}, "Missing requested frozen-tile geometry/variant coverage")
    helpers = []
    for row in tiles:
        require(row.get("candidate_fused_backward_calls") == 1, "Tile candidate did not execute exactly one fused call")
        for key in ("candidate_vs_eager_bf16", "candidate_vs_fp64_boundary_oracle"):
            engineering_gradients(row[key], names=("dkey", "dvalue"))
        if "timing" in row:
            for arm in ("eager", "triton"):
                prior.timed(row["timing"][arm], "backward helper "+arm)
            helpers.append({"check": row["name"], "configuration": row["configuration"], "timing": row["timing"]})
    max_fp32, max_primary = [], []
    for row in blocks:
        config = row["configuration"]
        length, prefix = config["length"], config["prefix"]
        require(row.get("primary_forward_and_cache_bitwise_equal") is True, "Backward candidate changed primary forward/cache")
        expected_forward, expected_backward = length-1+int(prefix > 0), length-1
        require(row.get("candidate_fused_forward_calls") == row.get("control_fused_forward_calls") ==
                row.get("expected_fused_forward_calls") == expected_forward and
                row.get("candidate_fused_backward_calls") == row.get("expected_fused_backward_calls") == expected_backward
                and row.get("control_fused_backward_calls") == 0, "Primary fused-call count mismatch")
        require(config.get("primary_forward_backend") == "triton" and config.get("cast_weights_once") is True and
                config.get("candidate_backward_backend") == "triton" and config.get("control_backward_backend") == "eager",
                "Primary block comparison changed more than the backward backend")
        gradient_names = {"input", "att_proj.weight", "attn_out.weight", "ff_proj.weight", "ff_out.weight"}
        if prefix:
            gradient_names |= {"prefix_key", "prefix_value"}
        primary = row["candidate_vs_f3b_control"]
        engineering_gradients(primary, gradient_names)
        require(primary.get("gradient_ownership_matches") is True and set(primary.get("outputs", {})) == {"hidden", "key", "value"},
                "Primary block ownership/output inventory differs")
        require(all(r.get("bitwise_equal") is True and r.get("finite") is True for r in primary["outputs"].values()),
                "Primary block forward/cache records contradict exactness")
        max_primary.append(primary["global_gradient_relative_l2"])
        fp32 = row.get("candidate_vs_full_fp32", {}).get("global_gradient_relative_l2")
        if isinstance(fp32, (int, float)) and math.isfinite(fp32):
            max_fp32.append(fp32)
    return {"tile_count": len(tiles), "block_count": len(blocks),
        "max_primary_block_global_gradient_relative_l2": max(max_primary, default=None),
        "max_fp32_diagnostic_block_global_gradient_relative_l2": max(max_fp32, default=None)}, helpers


def native_checks(report):
    result = prior.native_correctness(report)
    first = next(c for c in report["checks"] if c["name"] == "same_state_variant_vs_reference")
    require(first.get("forward_losses_bitwise_equal") is True, "Backward-only candidate changed initial forward losses")
    for row in first["losses"].values():
        tensor_check(row)
        require(row.get("bitwise_equal") is True, "Initial loss records contradict forward exactness")
    require(first.get("ownership_matches") is True, "Native gradient participation differs")
    expected_calls = report["configuration"]["length"]-1 if report["configuration"]["variant"] == "triton" else 0
    require(first.get("dispatch_matches") is True and first.get("reference_fused_backward_calls") == 0 and
            first.get("candidate_fused_backward_calls") == first.get("expected_candidate_fused_backward_calls") == expected_calls,
            "Native primary comparison did not execute expected fused backward tiles")
    mixed = first.get("mixed_gradient_screen")
    if mixed:
        require(set(mixed) == set(first["gradients"]), "Mixed and strict native gradient inventories differ")
        for name, row in mixed.items():
            diagnostic = first["gradients"][name]
            require(diagnostic.get("finite") is True, "Native gradient contains nonfinite values")
            if number(diagnostic.get("reference_max_abs"), "native reference maximum") == 0:
                require(number(diagnostic.get("max_abs"), "native absolute error") == 0,
                        "Zero native tensor reference requires exact zero")
            if number(row.get("reference_sq"), "tensor squared reference") == 0:
                require(number(row.get("delta_sq"), "tensor squared error") == 0,
                        "Zero native tensor norm requires exact zero")
        error = sum(number(r.get("delta_sq"), "global squared error") for r in mixed.values())
        reference = sum(number(r.get("reference_sq"), "global squared reference") for r in mixed.values())
        if reference == 0:
            require(error == 0, "Zero global gradient reference requires exact zero")
        expected_global = math.sqrt(error/reference) if reference else 0.
        require(math.isclose(first["global_gradient_relative_l2"], expected_global, rel_tol=1e-10, abs_tol=1e-14),
                "Recorded global gradient error contradicts tensor sums")
    return {**result, "forward_losses_bitwise_equal": True,
        "global_gradient_relative_l2": first.get("global_gradient_relative_l2"),
        "max_tensor_gradient_relative_l2": max((r["relative_l2"] for r in (first.get("mixed_gradient_screen") or {}).values()), default=0.),
        "max_tensor_gradient_error_reference_max": max((r["max_relative"] for r in (first.get("mixed_gradient_screen") or {}).values()), default=0.),
        "observed_fused_backward_calls": first["candidate_fused_backward_calls"]}


def profile_checks(report):
    named = {c["name"]: c for c in report["checks"]}
    require(set(named) == {"observer_neutrality", "expected_backward_annotations", "finite_complete_updates"},
            "Missing profile observer/count/health gates")
    neutral = named["observer_neutrality"]
    require(neutral.get("ownership_matches") is True and neutral.get("loss_keys_match") is True and
            neutral.get("nonidentical_losses") == [] and neutral.get("nonidentical_gradients") == [],
            "Profiler changed native losses/gradients")
    length = report["configuration"]["length"]
    expected = {"RT/local_writer_vjp": length, "RT/local_finish_vjp": length,
        "RT/batched_parameter_vjp": 1, "RT/attention_reconstruction": 1,
        "RT/reverse_historical_tile": length-1, "RT/final_attention_matmul": 4}
    counts = named["expected_backward_annotations"]
    require(counts.get("expected") == counts.get("actual") == expected,
            "Backward annotation count differs from one selected RT invocation")
    require(all(report.get("annotation_calls", {}).get(k) == v for k,v in expected.items()),
            "Recorded annotation inventory contradicts gate")
    require(report.get("completed_optimizer_updates") == 6 and
            {k:report.get("optimizer_update_scope", {}).get(k) for k in ("eager", "graph", "total")} ==
            {"eager": 3, "graph": 3, "total": 6}, "Profile optimizer update count differs")


def configuration_check(report, kind):
    c = report["configuration"]
    if kind == "tile":
        require(c.get("primary_forward_backend") == "triton" and c.get("primary_cast_weights_once") is True and
                c.get("control_backward_backend") == "eager" and c.get("candidate_backward_backend") == "triton",
                "Probe primary configuration differs")
        return
    require(c.get("variant") in ("reference", "triton") and c.get("cast_weights_once") is True,
            "Unknown primary execution variant")
    require(c.get("forward_tile_backend", c.get("tile_backend")) == "triton", "Primary forward backend must remain F3b Triton")
    require(c.get("backward_tile_backend") == ("triton" if c["variant"] == "triton" else "eager"),
            "Backward execution flag contradicts variant")


def summarize(runtime_dirs, *, project_root=ROOT, allow_incomplete=False, f3b_baselines=()):
    project_root = Path(project_root)
    summary = {"schema": "olmo-f3c-summary-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "runs": [], "full_steps": [], "profiles": [], "correctness": [], "tile_probes": [],
        "helper_timings": [], "failed_diagnostics": [], "incomplete": [],
        "successful_f3c_optimizer_updates": {"eager": 0, "graph": 0, "total": 0},
        "primary_reference": "Validated F3b forward fusion and cast reuse; eager historical backward tiles",
        "scope": "Backward tile fusion only; global probabilities/error remain, with no quadratic-memory removal claim."}
    seen = set()
    for item in runtime_dirs:
        directory = Path(item).resolve()
        require(directory.name not in seen, "Duplicate runtime name")
        seen.add(directory.name)
        path = directory/"report.json"
        if not path.is_file():
            require(allow_incomplete and directory.is_dir(), "Missing runtime report")
            summary["incomplete"].append({"run": directory.name, "status": "no_report_yet"})
            continue
        report = json.loads(path.read_text())
        require(report.get("schema") in SCHEMAS, "Unknown F3c schema")
        kind, status = SCHEMAS[report["schema"]], report.get("status")
        if status == "running":
            require(allow_incomplete, "Active report requires explicit preview")
            summary["incomplete"].append({"run": directory.name, "status": status, "stage": report.get("stage")})
            continue
        require(status in ("passed", "failed", "capture_blocked") and report.get("finished_utc"), "Report is not completed")
        checks = declared_checks(report)
        run = {"name": directory.name, "kind": kind, "status": status,
            "configuration": report.get("configuration", {}), "runtime": report.get("runtime"),
            "source_lineage": source_lineage(report, directory, project_root),
            "report_path": str(path), "report_sha256": digest(path), "checks": checks,
            "wandb_url": report.get("wandb", {}).get("run_url"), "used_for_performance": False}
        summary["runs"].append(run)
        if status != "passed":
            summary["failed_diagnostics"].append({"run": directory.name, "status": status,
                "stage": report.get("stage"), "error_type": report.get("error_type"), "error_message": report.get("error_message"),
                "declared_failed_checks": [r["name"] for r in checks if not r["passed"]],
                "excluded_partial_performance": bool(report.get("full_step") or report.get("capacity") or any("timing" in r["record"] for r in checks))})
            continue
        configuration_check(report, kind)
        updates = 0
        if kind in ("native", "profile"):
            require(report.get("checkpoint", {}).get("sha256") == prior.CHECKPOINT_SHA256, "Wrong native checkpoint")
        if kind == "profile":
            profile_checks(report)
            summary["full_steps"].append(prior.full_step_record(report, directory.name, profile=True))
            profile = prior.profile_record(report, directory.name)
            profile.update(profile_scope=report.get("profile_scope"), annotation_calls=report["annotation_calls"], observer_neutral=True)
            summary["profiles"].append(profile)
            updates = 6; run["used_for_performance"] = True
        elif kind == "native":
            if report["configuration"].get("stage") == "correctness":
                row = native_checks(report)
                summary["correctness"].append({"run": directory.name, "configuration": report["configuration"], **row})
                updates = row["physical_optimizer_updates"]
            elif report["configuration"].get("stage") == "capacity":
                require({c["name"] for c in checks} == {"finite_complete_updates"}, "Missing finite-update gate")
                summary["full_steps"].append(prior.full_step_record(report, directory.name))
                updates = 6; run["used_for_performance"] = True
            else:
                raise ValueError("Unknown native stage")
        else:
            row, helpers = tile_checks(report)
            summary["tile_probes"].append({"run": directory.name, **row})
            summary["helper_timings"].extend({"run": directory.name, **r} for r in helpers)
            run["used_for_performance"] = bool(helpers)
        if updates:
            # Three eager versus three graph updates for correctness; three
            # eager prep plus three graph timed updates for capacity/profiles.
            require(updates == 6, "Unexpected bounded optimizer-update count")
            summary["successful_f3c_optimizer_updates"]["eager"] += 3
            summary["successful_f3c_optimizer_updates"]["graph"] += 3
            summary["successful_f3c_optimizer_updates"]["total"] += 6
    if f3b_baselines:
        historical = prior.summarize(f3b_baselines, project_root=project_root)
        require(not historical["failed_diagnostics"] and not historical["incomplete"], "Historical F3b baseline must be completed and passed")
        summary["historical_f3b_lineage"] = historical["runs"]
        require(historical["full_steps"], "Historical baseline has no measured updates")
        for original in historical["full_steps"]:
            require(original["case"] == "rt" and original["batch_size"] == 128 and original["length"] == 512 and
                    original["variant"] == "triton", "Historical comparison must use F3b fused-forward RT B128/T512")
            row = copy.deepcopy(original)
            row.update(historical=True, variant="F3b reference", provenance="separately validated F3b fused-forward RT B128 baseline")
            summary["full_steps"].append(row)
    require(summary["runs"] or summary["incomplete"], "No F3c runs supplied")
    if summary["incomplete"]:
        summary["status"] = "partial_preview"
    elif summary["failed_diagnostics"]:
        summary["status"] = "completed_with_failed_diagnostics"
    summary["declared_check_count"] = sum(len(r["checks"]) for r in summary["runs"])
    summary["declared_checks_passed"] = sum(c["passed"] for r in summary["runs"] for c in r["checks"])
    return summary


def markdown(summary):
    lines = ["# F3c historical RT backward fusion", "", f"Status: **{summary['status']}**.", "",
        "Primary reference: F3b fused historical forward tiles and local BF16 weight-cast reuse, with the existing "
        "eager backward. The candidate changes only historical dK/dV tile execution. Native checkpoint, forward "
        "recurrence, parameters, RoPE, Q/K treatment and canonical FBT/NextLat losses stay fixed.", "",
        "This is a bounded functionality/execution result, not a quality or all-layer RT result. Full-model checks "
        "select RT layer0 only. **Global probability/error matrices remain materialized: this milestone does not "
        "remove quadratic backward storage.** Multi-GPU, longer/padded graphs and graph resume remain separate.", "",
        "## Evidence and numerical checks", "", "| Run | Scope | Status | Declared passed / total |",
        "| --- | --- | --- | ---: |"]
    for run in summary["runs"]:
        name = f"[{run['name']}]({run['wandb_url']})" if run.get("wandb_url") else run["name"]
        lines.append(f"| {name} | {run['kind']} | {run['status']} | {sum(c['passed'] for c in run['checks'])} / {len(run['checks'])} |")
    updates = summary["successful_f3c_optimizer_updates"]
    lines += ["", f"Completed successful F3c runs contain {updates['total']} physical optimizer updates: "
        f"{updates['eager']} eager and {updates['graph']} graph. Backward-only warmup/capture/profiles and historical "
        "F3b updates are excluded. Profile observer neutrality is checked bitwise for losses and all active gradients.", "",
        "Primary gradient screens are global relative L2<=1/64, per tensor<=1/32, maximum error/reference maximum<=1/16. "
        "An exact-zero reference requires exact zero. Same-candidate eager/graph uses stricter F3 limits; three full "
        "AdamW updates per arm must match exactly. Stricter coordinate flags and original-eager/full-FP32 controls "
        "remain visible diagnostics rather than silently replacing the declared primary gate.", "",
        "| Native case | B/T | Initial losses bitwise | Global gradient relative L2 | Max tensor relative L2 | Max error/reference max | Graph comparisons bitwise | Full updates exact |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |"]
    for row in summary["correctness"]:
        c = row["configuration"]; value = row["global_gradient_relative_l2"]
        global_text = "—" if value is None else f"{value:.6g}"
        lines.append(f"| {row['run']} | {c['batch_size']}/{c['length']} | {row['forward_losses_bitwise_equal']} | "
            f"{global_text} | {row['max_tensor_gradient_relative_l2']:.6g} | {row['max_tensor_gradient_error_reference_max']:.6g} | "
            f"{row['candidate_graph_checks_bitwise']} | {row['complete_updates_exact']} |")
    for row in summary["tile_probes"]:
        lines += ["", f"{row['run']}: {row['tile_count']} frozen backward rectangles and {row['block_count']} tiny native "
            "blocks passed their requested gates. Primary block forward/cache are bitwise equal, every expected fused "
            "call is observed, and raw hidden/exported-KV cotangents include masked attached prefixes and irregular lengths."]
    lines += ["", "## Measured complete updates", "",
        "Full-step wall timing includes validated input copy, graph forward/loss/backward, clipping, AdamW and scheduler. "
        "Input tokens count a physical batch once, not K times for FBT passes. Profile-run timings were collected before "
        "instrumentation. Peak allocated/reserved and current reserved memory have different meanings; none isolates "
        "graph-private memory or establishes a quadratic-storage improvement.", "",
        "| Run | Backward variant | Case | B/T | Input tokens/s | Seconds/update | Peak allocated / reserved GiB | Current reserved GiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    for row in summary["full_steps"]:
        variant = {"reference": "eager", "triton": "fused", "F3b reference": "historical eager"}[row["variant"]]
        lines.append(f"| {row['run']} | {variant} | {row['case']} | {row['batch_size']}/{row['length']} | "
            f"{row['input_tokens_per_second']:,.0f} | {row['full_step']['median_wall_seconds']:.4f} | "
            f"{fmt(row['peak_allocated_gib'])} / {fmt(row['peak_reserved_gib'])} | {fmt(row['current_reserved_gib'])} |")
    if not summary["full_steps"]:
        lines += ["", "No passing complete-update measurement was supplied."]
    lines += ["", "## Backward attribution", "",
        "Observer RT/* spans are inclusive and can overlap primals, VJPs, kernels and host gaps. CPU and CUDA "
        "annotation views cannot be added together. Actual graph-kernel self-time sums are separate from wall time "
        "and can include overlapping execution. Local writer/finish VJPs, batched parameter VJP, historical "
        "backward tiles, complete attention reconstruction and four final attention matmuls are labeled separately. "
        "CPU observer child-device time can undercount directly launched Triton kernels; use actual CUDA kernel "
        "events and full-update timing for performance comparisons, not ratios of those observer values."]
    for profile in summary["profiles"]:
        lines += ["", f"{profile['run']}: {profile['graph_kernel_invocations']:,} graph kernel invocations; "
            f"summed kernel self-device time {profile['graph_kernel_self_device_us']/1e6:.4f}s. Observer neutrality passed.", "",
            "| Observer span | Attribution | Calls | Inclusive CPU ms | Attributed device ms |", "| --- | --- | ---: | ---: | ---: |"]
        for row in profile["annotated_eager_ranges"]:
            lines.append(f"| {row['name']} | {row['device_type']} | {row['count']:,} | "
                f"{row['cpu_total_us']/1000:.3f} | {row['device_total_us']/1000:.3f} |")
        lines += ["", "| Actual graph kernel, top 8 | Calls | Self-device ms |", "| --- | ---: | ---: |"]
        for row in profile["graph_kernel_rows"][:8]:
            name = row["name"].replace("|", "\\|")
            lines.append(f"| `{name}` | {row['count']:,} | {row['self_device_us']/1000:.3f} |")
    lines += ["", "Backward helper timings are uncaptured CUDA-event intervals, including host submission gaps. "
        "They are not isolated kernel latency, full backward time or model throughput.", "",
        "| Rectangle | Source / target / head width | Eager / fused event microseconds |", "| --- | ---: | ---: |"]
    for row in summary["helper_timings"]:
        c, t = row["configuration"], row["timing"]
        lines.append(f"| {row['check']} | {c['width']} / {c['target_width']} / {c['head_dim']} | "
            f"{t['eager']['median_cuda_seconds']*1e6:.1f} / {t['triton']['median_cuda_seconds']*1e6:.1f} |")
    lines += ["", "## Accounting, failures and provenance", "",
        "Shared parameter counts and matrix FLOP formulas remain those in [resource accounting](../../olmo-resource-accounting.md). "
        "Fusion changes scheduling and intermediate storage, not the three complete historical dK/dV matrix products. "
        "The existing analytic range excludes pointwise work, optimizer arithmetic, host launch gaps and hardware padding.", ""]
    if not summary["failed_diagnostics"]:
        lines.append("No failed diagnostic was supplied.")
    for row in summary["failed_diagnostics"]:
        lines.append(f"- **{row['run']}: {row['status']}**, stage {row['stage']}; {row['error_type']}: {row['error_message']}. "
            "Partial timings are excluded from tables and plots.")
    for row in summary["incomplete"]:
        lines.append(f"- **{row['run']}: {row['status']}**; incomplete preview only, not performance evidence.")
    lines += ["", "summary.json retains declared checks, nested stricter diagnostic flags, exact per-run source/protocol "
        "lineage, timing samples and target counts. Failed variants are not promoted and historical F3b source versions "
        "remain distinct. Only explicitly supplied runtime directories are summarized.", ""]
    return "\n".join(lines)


def plot(summary, output_dir):
    output_dir = Path(output_dir)
    rows, helpers = summary["full_steps"], summary["helper_timings"]
    if not rows and not helpers:
        for suffix in ("png", "pdf"):
            (output_dir/f"throughput.{suffix}").unlink(missing_ok=True)
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    if rows:
        variants = {"reference": "eager backward", "triton": "fused backward", "F3b reference": "F3b historical backward"}
        labels = [f"{r['case']} B{r['batch_size']} T{r['length']}\n{variants[r['variant']]}" +
            (" · profile run" if r["provenance"] == "pre-profile uninstrumented timing" else "") for r in rows]
        axes[0].barh(range(len(rows)), [r["input_tokens_per_second"]/1000 for r in rows], color="#287a96")
        axes[0].set_yticks(range(len(rows)), labels, fontsize=8); axes[0].invert_yaxis()
        axes[0].set_xlabel("Input tokens/s (thousands)"); axes[0].set_title("Complete updates · uninstrumented")
        axes[0].grid(axis="x", alpha=.2); axes[0].set_axisbelow(True)
    else:
        axes[0].text(.5, .5, "No passing full-step timing", ha="center", transform=axes[0].transAxes); axes[0].axis("off")
    if helpers:
        for offset, arm, color in ((-.18, "eager", "#8996a0"), (.18, "triton", "#cf793d")):
            axes[1].barh([i+offset for i in range(len(helpers))], [r["timing"][arm]["median_cuda_seconds"]*1e6 for r in helpers],
                height=.33, label=arm, color=color)
        axes[1].set_yticks(range(len(helpers)), [f"W{r['configuration']['width']} / D{r['configuration']['head_dim']}" for r in helpers], fontsize=8)
        axes[1].invert_yaxis(); axes[1].set_xlabel("CUDA-event elapsed time (microseconds)")
        axes[1].set_title("Uncaptured backward-tile call\nHost submission gaps included", fontsize=10)
        axes[1].legend(); axes[1].grid(axis="x", alpha=.2); axes[1].set_axisbelow(True)
    else:
        axes[1].text(.5, .5, "No passing backward helper timing", ha="center", transform=axes[1].transAxes); axes[1].axis("off")
    fig.suptitle("F3c historical RT backward fusion · fixed F3b forward", fontsize=12)
    fig.text(.5, .01, "Full-model RT at layer0 only. Global probability/error arrays remain; no quadratic-memory removal or quality claim.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94)); output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "pdf"):
        path = output_dir/f"throughput.{suffix}"
        fig.savefig(path, dpi=160, bbox_inches="tight"); files.append(str(path))
    plt.close(fig)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--f3b-baseline-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=ROOT/"docs/reports/olmo1b-f3c")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    summary = summarize(args.runtime_dir, allow_incomplete=args.allow_incomplete, f3b_baselines=args.f3b_baseline_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/"summary.json", summary)
    (args.output_dir/"results.md").write_text(markdown(summary))
    files = plot(summary, args.output_dir)
    print({"status": summary["status"], "runs": len(summary["runs"]), "plots": files}, flush=True)
    return summary


if __name__ == "__main__":
    main()
