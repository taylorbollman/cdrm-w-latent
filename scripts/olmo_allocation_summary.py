#!/usr/bin/env python3
"""CPU-only, source-matched comparison of pair versus concurrent-single cells.

Aggregate throughput uses total useful inputs divided by the joint measured
makespan, including idle tails and gaps. It never sums disjoint job rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "olmo-allocation-summary-v1"
BENCHMARK_SCHEMA = "olmo-allocation-benchmark-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Invalid finite " + name)
    return value


def _close(actual, expected, name):
    number(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-7):
        raise ValueError("Recorded summary differs from update evidence: " + name)


def analyze_job(report):
    if report.get("schema") != BENCHMARK_SCHEMA or report.get("status") != "completed":
        raise ValueError("Require a completed allocation benchmark")
    if report.get("scale") != "native" or report.get("original_checkpoint_unchanged") is not True:
        raise ValueError("Require native benchmark with unchanged source checkpoint")
    allocation = report["allocation"]
    ranks = allocation["world_size"]
    batch = allocation["physical_batch_per_rank"]
    rows, length = allocation["real_rows"], report["length"]
    if type(ranks) is not int or ranks not in (1, 2, 8) or any(type(v) is not int or v <= 0 for v in (batch, rows, length)):
        raise ValueError("Invalid allocation dimensions")
    if (rows, length) != (512, 1024):
        raise ValueError("This native comparison requires T1024 and 512 real rows/update")
    slots = math.ceil(rows / (batch * ranks))
    for name, expected in (("microsteps_per_rank", slots), ("physical_rows", slots * batch * ranks),
                           ("dummy_rows", slots * batch * ranks - rows)):
        if allocation[name] != expected:
            raise ValueError("Allocation accounting differs: " + name)
    warmup, measured = report["warmup_updates"], report["measured_updates"]
    if type(warmup) is not int or type(measured) is not int or not 1 <= warmup <= 8 or not 1 <= measured <= 12:
        raise ValueError("Invalid bounded warmup/measurement counts")
    updates = report["updates"]
    if len(updates) != warmup + measured:
        raise ValueError("Incomplete update inventory")
    selected, signatures = [], []
    prior_end = 0
    for index, row in enumerate(updates):
        phase = "warmup" if index < warmup else "measured"
        if row["update"] != index + 1 or row["phase"] != phase:
            raise ValueError("Update ordering or phase differs")
        start = number(row["started_unix"], "update start", positive=True)
        end = number(row["finished_unix"], "update end", positive=True)
        if end <= start or start < prior_end:
            raise ValueError("Nonpositive or overlapping within-job update interval")
        prior_end = end
        metrics = row["metrics"]
        if metrics["input_tokens"] != rows * length:
            raise ValueError("Useful tokens do not match the fixed effective batch")
        number(metrics["gradient_norm_before_clip"], "gradient norm")
        # Objectives may contain signed terms; finiteness, rather than positivity,
        # is the relevant invariant here.
        objective = metrics["objective"]
        if type(objective) not in (int, float) or not math.isfinite(objective):
            raise ValueError("Nonfinite objective")
        timing = row["timing_by_rank"]
        if len(timing) != ranks or len(row["memory_by_rank"]) != ranks:
            raise ValueError("Missing rank timing or memory evidence")
        for rank in timing:
            for key in ("materialization", "backward", "optimizer"):
                number(rank[key], "rank " + key)
        keys = row["row_key_sha256_by_rank"]
        if len(keys) != ranks or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in keys):
            raise ValueError("Missing rank row-key digests")
        signatures.append({"update": row["update"], "phase": phase, "input_tokens": metrics["input_tokens"],
                           "counts": metrics["counts"], "lr_used": metrics["lr_used"],
                           "lr_next": metrics["lr_next"]})
        if phase == "measured":
            selected.append(row)
    tokens = measured * rows * length
    compute = sum(max(t["backward"] + t["optimizer"] for t in row["timing_by_rank"]) for row in selected)
    region = sum(max(t["materialization"] + t["backward"] + t["optimizer"]
                     for t in row["timing_by_rank"]) for row in selected)
    start, end = selected[0]["started_unix"], selected[-1]["finished_unix"]
    window = end - start
    number(compute, "compute seconds", positive=True)
    number(region, "selected seconds", positive=True)
    if region > window * 1.01:
        raise ValueError("Selected timing exceeds measured wall interval")
    summary = report["summary"]
    expected = {"measured_updates": measured, "real_input_tokens": tokens,
                "selected_compute_materialization_seconds": region,
                "selected_compute_materialization_tokens_per_second": tokens / region,
                "measured_window_seconds": window, "measured_window_tokens_per_second": tokens / window,
                "measured_started_unix": start, "measured_finished_unix": end}
    for key, value in expected.items():
        _close(summary[key], value, key)
    if summary.get("finite_updates") is not True:
        raise ValueError("Finite update gate failed")
    counters = report["final_counters"]
    if counters["optimizer_updates"] != warmup + measured or counters["input_tokens"] != (warmup + measured) * rows * length:
        raise ValueError("Final update/token clocks differ")
    memory_sets = [report["memory_after_capture_by_rank"]] + [row["memory_by_rank"] for row in updates]
    if any(len(values) != ranks for values in memory_sets):
        raise ValueError("Memory rank inventory differs")
    memories = []
    for rank in range(ranks):
        rows_memory = [values[rank] for values in memory_sets]
        for value in rows_memory:
            for key in ("peak_allocated_gib", "peak_reserved_gib", "sampled_free_gib", "total_gib"):
                number(value[key], "memory " + key)
        memories.append({"rank": rank,
            "peak_allocated_gib": max(v["peak_allocated_gib"] for v in rows_memory),
            "peak_reserved_gib": max(v["peak_reserved_gib"] for v in rows_memory),
            "minimum_sampled_free_gib": min(v["sampled_free_gib"] for v in rows_memory)})
    origin = report["origin"]
    contract = {key: report[key] for key in ("arm", "scale", "length", "recipe", "objective_weights",
        "parameters", "sources", "runtime", "determinism", "scheduler", "data_start_update", "warmup_updates", "measured_updates")}
    contract["origin"] = {key: origin[key] for key in ("manifest_sha256", "state_sha256", "original_counters",
        "learning_rates", "execution", "configuration")}
    contract["data"] = {key: report["data"][key] for key in ("index_sha256", "start_cursor")}
    contract["physical_batch_per_rank"] = batch
    contract["effective_rows"] = rows
    contract["updates"] = signatures
    if "scripts/olmo_allocation_benchmark.py" not in contract["sources"]:
        raise ValueError("Missing benchmark source pin")
    result = {**expected, "world_size": ranks, "allocation": allocation,
              "effective_input_tokens_per_update": rows * length,
              "physical_input_positions_per_update": allocation["physical_rows"] * length,
              "compute_seconds": compute, "compute_tokens_per_second": tokens / compute,
              "memory_by_rank": memories, "parameters": report["parameters"],
              "peak_allocated_gib_max_rank": max(v["peak_allocated_gib"] for v in memories),
              "peak_reserved_gib_max_rank": max(v["peak_reserved_gib"] for v in memories),
              "row_key_sha256_by_update_and_rank": [row["row_key_sha256_by_rank"] for row in updates],
              "wandb": report.get("wandb"), "contract": contract}
    return result


def joint_timing(jobs, *, minimum_overlap=0.95):
    starts = [job["measured_started_unix"] for job in jobs]
    ends = [job["measured_finished_unix"] for job in jobs]
    makespan = max(ends) - min(starts)
    overlap = max(0., min(ends) - max(starts))
    shortest = min(end - start for start, end in zip(starts, ends))
    if makespan <= 0 or shortest <= 0:
        raise ValueError("Invalid joint measurement intervals")
    tokens = sum(job["real_input_tokens"] for job in jobs)
    gpu_count = sum(job["world_size"] for job in jobs)
    gpu_seconds = sum(job["world_size"] * (end - start) for job, start, end in zip(jobs, starts, ends))
    start_gap = max(starts) - min(starts)
    qualified = len(jobs) == 1 or (overlap / shortest >= minimum_overlap and start_gap <= max(2., .05 * shortest))
    return {"real_input_tokens": tokens, "allocated_gpus": gpu_count,
        "measured_started_unix": min(starts), "measured_finished_unix": max(ends),
        "joint_makespan_seconds": makespan, "common_overlap_seconds": overlap,
        "overlap_percent_of_shorter_job": 100 * overlap / shortest,
        "overlap_percent_of_makespan": 100 * overlap / makespan,
        "start_gap_seconds": start_gap, "concurrency_qualified": qualified,
        "aggregate_window_tokens_per_second": tokens / makespan,
        "allocated_gpu_seconds": gpu_count * makespan,
        "tokens_per_allocated_gpu_second": tokens / (gpu_count * makespan),
        "active_measured_gpu_seconds": gpu_seconds,
        "tokens_per_active_measured_gpu_second": tokens / gpu_seconds,
        "qualification": "matched concurrent intervals" if qualified else "insufficient overlap or excessive start gap; no concurrency-efficiency conclusion"}


def load_cell(path):
    path = Path(path).resolve()
    supervisor = read_json(path / "supervisor.json")
    if supervisor.get("schema") != "olmo-allocation-supervisor-v1" or supervisor.get("status") != "completed":
        raise ValueError("Require every requested cell to be completed")
    layout = supervisor["layout"]
    expected_jobs = {"pair": 2} if layout == "pair" else {"single0": 1, "single1": 1} if layout == "singles" else None
    if expected_jobs is None or len(supervisor["jobs"]) != len(expected_jobs):
        raise ValueError("Unsupported cell layout")
    jobs, contracts, gates, seen = [], [], [], set()
    for record in supervisor["jobs"]:
        name = record["name"]
        if name not in expected_jobs or name in seen or record.get("status") != "completed" or record.get("exit_code") != 0:
            raise ValueError("Invalid completed job inventory")
        expected_devices = {"pair": "0,1", "single0": "0", "single1": "1"}
        if record.get("devices") != expected_devices[name] or record.get("ranks") != expected_jobs[name]:
            raise ValueError("Supervisor device isolation differs from declared layout")
        seen.add(name)
        report_path = path / name / "report.json"
        if digest(report_path) != record["report_sha256"]:
            raise ValueError("Job report bytes differ from supervisor receipt")
        report = read_json(report_path)
        job = analyze_job(report)
        if report["arm"] != supervisor["arm"] or job["world_size"] != expected_jobs[name]:
            raise ValueError("Job arm/rank allocation differs from supervisor")
        contracts.append(job.pop("contract"))
        job.update(name=name, report_path=str(report_path), report_sha256=digest(report_path))
        jobs.append(job)
        if layout == "singles":
            gate = report.get("start_gate", {})
            release = gate.get("release", {})
            if release.get("schema") != BENCHMARK_SCHEMA or release.get("gate_id") != path.name:
                raise ValueError("Missing matching concurrent start gate")
            if release["released_unix"] > job["measured_started_unix"]:
                raise ValueError("Measurement precedes gate release")
            if gate.get("sha256") != digest(path / "release.json") or release != read_json(path / "release.json"):
                raise ValueError("Start gate bytes differ")
            gates.append(gate["sha256"])
    if any(value != contracts[0] for value in contracts[1:]):
        raise ValueError("Concurrent jobs differ in origin, data, objective, runtime or sources")
    if layout == "singles" and jobs[0]["row_key_sha256_by_update_and_rank"] != jobs[1]["row_key_sha256_by_update_and_rank"]:
        raise ValueError("Single jobs processed different ordered row keys")
    return {"arm": supervisor["arm"], "layout": layout, "cell_path": str(path),
            "supervisor_sha256": digest(path / "supervisor.json"), "jobs": jobs,
            "timing": joint_timing(jobs), "contract": contracts[0]}


def build_summary(paths):
    cells = [load_cell(path) for path in paths]
    grouped = {}
    for cell in cells:
        pair = grouped.setdefault(cell["arm"], {})
        if cell["layout"] in pair:
            raise ValueError("Duplicate arm/layout; repeats need an explicit separate comparison")
        pair[cell["layout"]] = cell
    comparisons = []
    for arm, pair in sorted(grouped.items()):
        if set(pair) != {"pair", "singles"}:
            raise ValueError("Require both layouts for every requested arm")
        if pair["pair"]["contract"] != pair["singles"]["contract"]:
            raise ValueError("Unmatched checkpoint, implementation, actual objective, physical batch or ordered data prefix: " + arm)
        one, two = pair["singles"]["timing"], pair["pair"]["timing"]
        ratio = one["aggregate_window_tokens_per_second"] / two["aggregate_window_tokens_per_second"]
        comparisons.append({"arm": arm, "qualified": one["concurrency_qualified"],
            "singles_over_pair_aggregate_window_rate": ratio,
            "singles_aggregate_advantage_percent": (ratio - 1) * 100,
            "pair_over_mean_single_job_window_rate": two["aggregate_window_tokens_per_second"] /
                (sum(j["measured_window_tokens_per_second"] for j in pair["singles"]["jobs"]) / 2),
            "data_match_basis": "same authenticated ordered index, start cursor, data offset, update counts, objective counts, real tokens and unchanged deterministic source; rank key hashes are not joined"})
    for cell in cells:
        cell["matched_contract_sha256"] = hashlib.sha256(json.dumps(cell.pop("contract"), sort_keys=True).encode()).hexdigest()
    return {"schema": SCHEMA, "status": "completed", "scope": "Short disposable allocation benchmark; no production continuation, quality, H200 or eight-rank scaling claim",
            "cells": cells, "comparisons": comparisons,
            "timing_scope": "Original real input tokens once, over joint measured makespan including gaps/idle tails; excludes setup, warmup, final teardown and checkpoint I/O. Selected compute/materialization rates are shown separately, not substituted for node throughput.",
            "memory_scope": "Maximum allocated/reserved per rank over capture and updates, minimum sampled free; not pooled VRAM or continuous free-memory measurement."}


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [row["arm"] for row in summary["comparisons"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for index, layout in enumerate(("pair", "singles")):
        rows = [next(c for c in summary["cells"] if c["arm"] == arm and c["layout"] == layout) for arm in arms]
        x = [i + (index - .5) * .35 for i in range(len(arms))]
        rates = [c["timing"]["aggregate_window_tokens_per_second"] for c in rows]
        label = "One two-GPU job" if layout == "pair" else "Two single-GPU jobs"
        bars = axes[0].bar(x, rates, .35, label=label)
        for bar, row in zip(bars, rows):
            if not row["timing"]["concurrency_qualified"]:
                bar.set_hatch("//")
        axes[1].bar(x, [max(j["peak_reserved_gib_max_rank"] for j in c["jobs"]) for c in rows], .35, label=label)
    for axis in axes:
        axis.set_xticks(range(len(arms)), arms)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Useful input tokens/s, joint measured interval")
    axes[1].set_ylabel("Maximum reserved GiB per GPU")
    axes[0].legend(fontsize=8)
    fig.suptitle("Two H100 allocation comparison; same effective batch per experiment")
    if any(not c["qualified"] for c in summary["comparisons"]):
        axes[0].set_title("Hatched bars: concurrency comparison unqualified", fontsize=9)
    fig.savefig(output / "allocation.pdf")
    fig.savefig(output / "allocation.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--wandb-project", default="pretrained-fbt-rt-nextlat")
    parser.add_argument("--no-wandb", action="store_true", help="Local validation only; final experiment summary should log online")
    args = parser.parse_args(argv)
    report = build_summary(args.cell)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report["producer_sha256"] = digest(__file__)
    tracker = None
    def persist():
        temporary = output / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output / "report.json")
    try:
        persist()
        plot(report, output)
        if not args.no_wandb:
            from scripts.experiment_tracking import OnlineTracker
            import wandb
            tracker = OnlineTracker(project=args.wandb_project, output_dir=output,
                group="olmo-allocation-readiness", name=output.name)
            tracker.start({"scope": report["scope"], "cells": [str(p) for p in args.cell], "schema": SCHEMA})
            columns = ["arm", "layout", "aggregate_tokens_per_second", "tokens_per_gpu_second", "joint_seconds", "overlap_percent", "qualified", "peak_reserved_gib_per_gpu"]
            rows = [[c["arm"], c["layout"], c["timing"]["aggregate_window_tokens_per_second"],
                     c["timing"]["tokens_per_allocated_gpu_second"], c["timing"]["joint_makespan_seconds"],
                     c["timing"]["overlap_percent_of_shorter_job"], c["timing"]["concurrency_qualified"],
                     max(j["peak_reserved_gib_max_rank"] for j in c["jobs"])] for c in report["cells"]]
            table = wandb.Table(columns=columns, data=rows)
            tracker.log({"allocation/table": table, "allocation/comparison": wandb.Image(str(output / "allocation.png"))})
            tracker.summary({"comparisons": report["comparisons"], "all_concurrency_qualified": all(c["qualified"] for c in report["comparisons"])})
        report["figures"] = {name: digest(output / name) for name in ("allocation.pdf", "allocation.png")}
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        try:
            if tracker:
                tracker.finish(succeeded=report["status"] == "completed")
        except BaseException:
            report["status"] = "failed"
            raise
        finally:
            if tracker:
                report["wandb"] = tracker.record
            persist()
    return report


if __name__ == "__main__":
    main()
