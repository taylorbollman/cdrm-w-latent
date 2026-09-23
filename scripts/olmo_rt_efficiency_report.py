#!/usr/bin/env python3
"""Validate explicit Stage A reports and summarize bounded RT efficiency tests.

This is CPU-only reporting: no model construction, CUDA calls, or training.
An explicit subset is allowed, but every selected report must have finished.
"""
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


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ".runtime/olmo-rt-efficiency"
DOCS = "docs/reports/olmo-rt-efficiency"
PROTOCOL = DOCS + "/protocol.md"
ESSENTIAL_SOURCES = {
    "scripts/olmo_rt_efficiency.py", "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_rope.py",
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_static.py",
    "cdrm/pretrained/static_training.py", "cdrm/pretrained/nextlat.py",
    "cdrm/pretrained/static_nextlat.py", "cdrm/pretrained/resource_estimates.py",
    "cdrm/pretrained/olmo_rt_kernels.py", "cdrm/pretrained/olmo_rt_backward_kernels.py",
    "cdrm/pretrained/olmo_rt_recompute_kernels.py",
}
EXPECTED_CORRECTNESS = {("ordinary", "control-rope"), ("rt", "control-rope"),
    ("combined", "control-rope"), ("rt", "rope-both"), ("combined", "rope-both")}
EXPECTED_CAPACITY = {(case, arm) for case in ("ordinary", "rt", "combined")
                     for arm in (("control", "rope") if case == "ordinary" else ("control", "rope", "both"))}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular(path, root):
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            "Expected project-local regular file: " + str(path))
    return path


def safe_source(source):
    path = PurePosixPath(source)
    require(not path.is_absolute() and ".." not in path.parts
            and (source.startswith("cdrm/pretrained/") or source.startswith("scripts/"))
            and (path.suffix == ".py" or source == "cdrm/pretrained/_fbt_reference/manifest.json"),
            "Unexpected runtime source path: " + source)
    return source


def worst(rows, key):
    if not rows:
        return None
    name, row = max(rows.items(), key=lambda item: item[1].get(key, -math.inf))
    return {"name": name, key: row.get(key)}


def check_summary(check):
    result = {key: value for key, value in check.items() if not isinstance(value, (dict, list))}
    for field in ("gradients", "outputs", "losses"):
        rows = check.get(field, {})
        if rows:
            result[field] = {"tensor_count": len(rows),
                "worst_relative_l2": worst(rows, "relative_l2"),
                "worst_max_relative": worst(rows, "max_relative"),
                "all_bitwise_equal": all(row.get("bitwise_equal") is True for row in rows.values())}
    if "budgets" in check:
        result["budgets"] = check["budgets"]
    return result


def verify_profile(report, directory, root):
    profile = report.get("profile")
    if profile is None:
        require(not report["configuration"].get("profile") or report["status"] != "passed",
                "Successful requested profile is missing")
        return None, None
    require(report["configuration"].get("profile") is True
            and profile.get("trace_file") == "operator-trace.json.gz", "Unexpected trace declaration")
    path = regular(directory / profile["trace_file"], root)
    require(path.stat().st_size == profile.get("trace_bytes")
            and digest(path) == profile.get("trace_sha256"), "Operator trace bytes differ from report")
    with path.open("rb") as stream:
        require(stream.read(2) == b"\x1f\x8b", "Trace is not gzip data")
    verified = {key: profile[key] for key in ("trace_file", "trace_bytes", "trace_sha256")}
    kernels = profile.get("device_kernels", {})
    total_us = sum(row["self_device_us"] for row in kernels.values())
    top = sorted(kernels.items(), key=lambda item: item[1]["self_device_us"], reverse=True)[:30]
    details = {**verified, "scope": profile.get("scope"),
        "device_event_count": profile.get("device_event_count"),
        "summed_device_ms": total_us / 1000,
        "top_device_events": [{"name": name, **row,
            "share_of_summed_device_time": row["self_device_us"] / total_us if total_us else 0}
            for name, row in top],
        "qualification": "Includes kernels and other reported CUDA events; summed device time is not complete-update wall time. Shared GEMMs cannot be attributed to permanent writes from names alone."}
    return verified, details


def validate_finished(report):
    status = report.get("status")
    require(report.get("schema") == "olmo-rt-efficiency-v1" and report.get("finished_utc")
            and status in {"passed", "failed", "oom"}, "Only completed reports may be selected")
    require(type(report.get("physical_optimizer_updates")) is int
            and report["physical_optimizer_updates"] >= 0, "Invalid physical optimizer-update count")
    checks = report.get("checks")
    require(isinstance(checks, list) and all(type(check.get("passed")) is bool for check in checks),
            "Invalid check inventory")
    require(len({check["name"] for check in checks}) == len(checks), "Duplicate check names")
    config = report.get("configuration", {})
    require(config.get("stage") in {"correctness", "capacity"}
            and config.get("case") in {"ordinary", "rt", "combined"}, "Unknown report configuration")
    if status != "passed":
        require(report.get("error", {}).get("type") and isinstance(report["error"].get("message"), str),
                "Unsuccessful report lacks an explicit error")
        return
    require(report.get("stage") == "complete" and checks and all(check["passed"] for check in checks),
            "Passing report is incomplete or contains failed checks")
    require(report.get("checkpoint", {}).get("sha256"), "Passing report lacks checkpoint provenance")
    if config["stage"] == "correctness":
        expected = {config["comparison"].replace("-", "_vs_"), "candidate_initial_graph",
            "candidate_changed_tokens_overwrite", "complete_adamw_update_parity", "candidate_changed_weights"}
        require({check["name"] for check in checks} == expected
                and report["physical_optimizer_updates"] == 6, "Incomplete correctness/update protocol")
    else:
        require({check["name"] for check in checks} == {"finite_complete_updates"}
                and report["physical_optimizer_updates"] == 8
                and len(report.get("preparation_records", [])) == 3
                and len(report.get("timed_records", [])) == 5, "Incomplete timing/update protocol")
        for name, count in (("full_update", 5), ("forward_loss_backward", 3)):
            record = report[name]
            for clock in ("wall", "cuda"):
                samples = record[clock + "_seconds"]
                require(len(samples) == count and all(math.isfinite(x) and x > 0 for x in samples),
                        "Invalid timing samples")
                require(math.isclose(record["median_" + clock + "_seconds"], statistics.median(samples), rel_tol=1e-12),
                        "Timing median disagrees with samples")
        require(math.isclose(report["input_tokens_per_second"],
            report["input_tokens"] / report["full_update"]["median_wall_seconds"], rel_tol=1e-12),
            "Throughput disagrees with synchronized wall time")


def paired_gains(capacity):
    groups = defaultdict(lambda: defaultdict(list))
    for row in capacity:
        if row["status"] == "passed":
            groups[(row["case"], row["batch_size"], row["length"], row["supervision"])][row["arm"]].append(row)
    result = []
    for (case, batch, length, supervision), arms in sorted(groups.items()):
        signatures = {row["comparison_signature"] for runs in arms.values() for row in runs}
        require(len(signatures) == 1, "Paired timing configurations/runtime/checkpoints differ")
        aggregates = {arm: {"runs": [row["name"] for row in runs], "run_count": len(runs),
            "median_run_input_tokens_per_second": statistics.median(row["input_tokens_per_second"] for row in runs),
            "minimum_run_input_tokens_per_second": min(row["input_tokens_per_second"] for row in runs),
            "maximum_run_input_tokens_per_second": max(row["input_tokens_per_second"] for row in runs)}
            for arm, runs in arms.items()}
        comparisons = []
        for reference, candidate in (("control", "rope"), ("rope", "both"), ("control", "both")):
            if reference in aggregates and candidate in aggregates:
                ratio = (aggregates[candidate]["median_run_input_tokens_per_second"]
                         / aggregates[reference]["median_run_input_tokens_per_second"])
                comparisons.append({"reference_arm": reference, "candidate_arm": candidate,
                    "throughput_ratio": ratio, "throughput_gain_percent": 100 * (ratio - 1),
                    "step_time_reduction_percent": 100 * (1 - 1 / ratio)})
        result.append({"case": case, "batch_size": batch, "length": length, "supervision": supervision,
            "arms": aggregates, "comparisons": comparisons,
            "qualification": "Five timed optimizer updates per report; median of run medians if repeated. Directional benchmark, not a training or significance result."})
    return result


def summarize(names, revision, root=ROOT):
    require(names and len(set(names)) == len(names), "Require a nonempty, unique explicit run list")
    require(re.fullmatch(r"[0-9a-f]{7,40}", revision), "Require a Git hex runtime commit")
    revision = subprocess.check_output(["git", "rev-parse", revision + "^{commit}"], cwd=root, text=True).strip()
    frozen = {}

    def git_digest(name):
        if name not in frozen:
            frozen[name] = hashlib.sha256(subprocess.check_output(["git", "show", revision + ":" + name], cwd=root)).hexdigest()
        return frozen[name]

    summary = {"schema": "olmo-rt-efficiency-summary-v1", "status": "completed", "runtime_commit": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(), "runs": [], "correctness": [], "capacity": [],
        "profiles": [], "resource_cards": [], "physical_optimizer_updates": 0, "source_pairs_checked": 0,
        "qualification": "Completed means every explicitly selected report finished; it does not mean every report passed or the whole prospective queue completed. Prior numerical qualifications remain open."}
    checkpoints, statuses = set(), Counter()
    correctness_coverage, capacity_coverage = set(), set()
    for name in names:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name), "Unsafe run name")
        path = regular(root / RUNTIME / name / "report.json", root)
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        validate_finished(raw)
        config, status = raw["configuration"], raw["status"]
        hashes = raw.get("source_hashes")
        require(isinstance(hashes, dict) and ESSENTIAL_SOURCES <= hashes.keys(), "Missing essential source inventory")
        differences = {}
        for source, expected in hashes.items():
            safe_source(source)
            snapshot = regular(path.parent / "source-snapshot" / source, root)
            require(digest(snapshot) == expected == git_digest(source), "Source snapshot/commit mismatch: " + source)
            summary["source_pairs_checked"] += 1
            if (root / source).is_file() and digest(root / source) != expected:
                differences[source] = {"reported_sha256": expected, "current_sha256": digest(root / source)}
        protocol = regular(path.parent / "protocol.md", root)
        require(digest(protocol) == raw.get("protocol_sha256") == git_digest(PROTOCOL), "Protocol snapshot/commit mismatch")
        if raw.get("checkpoint"):
            checkpoints.add(json.dumps(raw["checkpoint"], sort_keys=True))
        row = {"name": name, "report_path": path.relative_to(root).as_posix(),
            "report_sha256": hashlib.sha256(raw_bytes).hexdigest(), "status": status,
            "stage": config["stage"], "case": config["case"], "arm": config.get("arm"),
            "comparison": config.get("comparison"), "physical_optimizer_updates": raw["physical_optimizer_updates"],
            "wandb_url": raw.get("wandb", {}).get("run_url"), "checks": [check_summary(c) for c in raw["checks"]],
            "current_source_differences": differences}
        if status != "passed":
            row["error"] = raw["error"]
        verified, profile = verify_profile(raw, path.parent, root)
        if verified:
            row["profile"] = verified
            summary["profiles"].append({"name": name, "case": config["case"], "arm": config.get("arm"), **profile})
        summary["runs"].append(row)
        summary["physical_optimizer_updates"] += raw["physical_optimizer_updates"]
        statuses[status] += 1
        if raw.get("resources"):
            summary["resource_cards"].append({"name": name, "case": config["case"],
                "arm": config.get("arm") or ("rope" if config.get("comparison") == "control-rope" else "both"),
                "batch_size": config["batch_size"], "stage": config["stage"], "status": status,
                **raw["resources"]})
        if config["stage"] == "correctness":
            correctness_coverage.add((config["case"], config["comparison"]))
            summary["correctness"].append({"name": name, "case": config["case"], "comparison": config["comparison"],
                "status": status, "checks": row["checks"], "memory": raw.get("memory")})
        else:
            if config["batch_size"] == 64 and config["length"] == 512 and config["supervision"] == "full":
                capacity_coverage.add((config["case"], config["arm"]))
            comparable = {key: value for key, value in config.items()
                if key not in {"output_dir", "arm", "profile", "artifacts"}}
            signature = hashlib.sha256(json.dumps({"configuration": comparable, "runtime": raw.get("runtime"),
                "checkpoint": raw.get("checkpoint"), "protocol_sha256": raw["protocol_sha256"]}, sort_keys=True).encode()).hexdigest()
            summary["capacity"].append({"name": name, "case": config["case"], "arm": config["arm"],
                "status": status, "batch_size": config["batch_size"], "length": config["length"],
                "supervision": config["supervision"], "comparison_signature": signature,
                **{key: raw.get(key) for key in ("input_tokens_per_second", "ce_targets_per_second",
                    "forward_loss_backward_tokens_per_second", "full_update", "forward_loss_backward",
                    "setup_memory", "steady_memory", "capture_seconds", "parameters", "counts")}})
    require(len(checkpoints) <= 1, "Selected runs use different native checkpoints")
    summary["checkpoint"] = json.loads(next(iter(checkpoints))) if checkpoints else None
    summary["statuses"] = dict(statuses)
    summary["all_selected_runs_passed"] = statuses["passed"] == len(names)
    checks = [check for row in summary["runs"] for check in row["checks"]]
    summary["checks"] = {"passed": sum(check["passed"] for check in checks), "total": len(checks),
        "failed": [{"run": row["name"], "check": check["name"]} for row in summary["runs"]
                   for check in row["checks"] if not check["passed"]]}
    missing = {"correctness": sorted(EXPECTED_CORRECTNESS - correctness_coverage),
               "capacity": sorted(EXPECTED_CAPACITY - capacity_coverage)}
    summary["coverage"] = {"primary_queue_complete": not any(missing.values()), "missing": missing,
        "scope": "Five correctness cases and eight B64/full-CE capacity arms; failed attempts count as attempted, not passed."}
    summary["paired_gains"] = paired_gains(summary["capacity"])
    return summary


def plot_primary(summary, destination):
    rows = {row["case"]: row for row in summary["paired_gains"]
            if row["batch_size"] == 64 and row["length"] == 512 and row["supervision"] == "full"}
    if not rows:
        return []
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    cases = [case for case in ("ordinary", "rt", "combined") if case in rows]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    arms = [("control", "Original", "#8996a5"), ("rope", "RoPE reuse", "#3881b7"),
            ("both", "RoPE + K/V writes", "#249b73")]
    for offset, (arm, label, color) in enumerate(arms):
        x, heights = [], []
        for index, case in enumerate(cases):
            if arm in rows[case]["arms"]:
                x.append(index + (offset - 1) * .24)
                heights.append(rows[case]["arms"][arm]["median_run_input_tokens_per_second"] / 1000)
        bars = axis.bar(x, heights, width=.23, color=color, label=label)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_xticks(range(len(cases)), [dict(ordinary="Ordinary OLMo", rt="RT layers 0/15",
        combined="RT + FBT K2 + NextLat")[case] for case in cases])
    axis.set_ylabel("Input tokens/sec (thousands)")
    axis.set_title("Native OLMo-1B · H100 · B64 × T512 · full CE")
    axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
    axis.legend(loc="upper right", frameon=False)
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
    fig.text(.5, .015, "Five timed optimizer updates per report; short benchmark, not a training result.\n"
        "BF16 mixed · CUDA graphs · ordinary checkpointing · CE2048 / KL128", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    outputs = []
    for extension in ("png", "pdf"):
        path = destination / ("throughput." + extension)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Explicit names; no runtime-directory discovery")
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / DOCS)
    args = parser.parse_args(argv)
    summary = summarize(args.runs, args.runtime_commit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary["plots"] = plot_primary(summary, args.output_dir)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(path), "statuses": summary["statuses"], "checks": summary["checks"],
        "physical_optimizer_updates": summary["physical_optimizer_updates"],
        "source_pairs_checked": summary["source_pairs_checked"], "coverage": summary["coverage"]}))


if __name__ == "__main__":
    main()
