"""Guard allocation claims against disjoint windows and mismatched experiments."""
import copy
import hashlib
import json

import pytest

from scripts.olmo_allocation_summary import analyze_job, build_summary, joint_timing, load_cell, plot


def job_report(*, ranks=1, start=100., seconds=10., arm="B"):
    tokens = 512 * 1024
    memory = {"peak_allocated_gib": 30., "peak_reserved_gib": 40., "sampled_free_gib": 35., "total_gib": 80.}
    updates = []
    for index in range(3):
        updates.append({"update": index + 1, "phase": "warmup" if index == 0 else "measured",
            "started_unix": start + (index - 1) * seconds,
            "finished_unix": start + index * seconds,
            "metrics": {"input_tokens": tokens, "objective": 3., "gradient_norm_before_clip": 2.,
                        "counts": {"ce": tokens - 512, "latent": 0, "kl": 0},
                        "lr_used": [.0002], "lr_next": [.0002]},
            "timing_by_rank": [{"materialization": 1., "backward": seconds - 3., "optimizer": 1.} for _ in range(ranks)],
            "memory_by_rank": [dict(memory) for _ in range(ranks)],
            "row_key_sha256_by_rank": [str(index) * 64 for _ in range(ranks)]})
    return {"schema": "olmo-allocation-benchmark-v1", "status": "completed", "scale": "native",
        "arm": arm, "original_checkpoint_unchanged": True,
        "allocation": {"world_size": ranks, "physical_batch_per_rank": 32, "real_rows": 512,
                       "microsteps_per_rank": 16 // ranks, "physical_rows": 512, "dummy_rows": 0},
        "length": 1024, "warmup_updates": 1, "measured_updates": 2, "updates": updates,
        "summary": {"measured_updates": 2, "real_input_tokens": 2 * tokens,
            "selected_compute_materialization_seconds": 2 * (seconds - 1.),
            "selected_compute_materialization_tokens_per_second": tokens / (seconds - 1.),
            "measured_window_seconds": 2 * seconds, "measured_window_tokens_per_second": tokens / seconds,
            "measured_started_unix": start, "measured_finished_unix": start + 2 * seconds, "finite_updates": True},
        "final_counters": {"optimizer_updates": 3, "input_tokens": 3 * tokens},
        "memory_after_capture_by_rank": [dict(memory) for _ in range(ranks)],
        "origin": {"manifest_sha256": "a" * 64, "state_sha256": "b" * 64, "original_counters": {"optimizer_updates": 128},
                   "learning_rates": [.0002], "execution": {"precision": "bf16_mixed"}, "configuration": {"model": arm}},
        "data": {"index_sha256": "c" * 64, "start_cursor": {"next_update": 0, "chunk": 0}},
        "data_start_update": 0, "recipe": {"arm": arm}, "objective_weights": {"ce": 1.},
        "parameters": {"resident": 1200, "trainable": 1100},
        "sources": {"scripts/olmo_allocation_benchmark.py": "d" * 64, "cdrm/pretrained/model.py": "e" * 64},
        "runtime": {"torch": "test", "gpu": "H100"}, "determinism": {"enabled": True}, "scheduler": "fixed"}


def write_json(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(tmp_path, layout, reports):
    output = tmp_path / (reports[0]["arm"].lower() + "-" + layout)
    output.mkdir()
    jobs = []
    names = ["pair"] if layout == "pair" else ["single0", "single1"]
    if layout == "singles":
        release = {"schema": "olmo-allocation-benchmark-v1", "gate_id": output.name, "released_unix": 99.}
        gate_sha = write_json(output / "release.json", release)
    for name, original in zip(names, reports):
        report = copy.deepcopy(original)
        directory = output / name
        directory.mkdir()
        if layout == "singles":
            report["start_gate"] = {"sha256": gate_sha, "release": release}
        sha = write_json(directory / "report.json", report)
        jobs.append({"name": name, "status": "completed", "exit_code": 0, "report_sha256": sha,
                     "devices": {"pair": "0,1", "single0": "0", "single1": "1"}[name],
                     "ranks": 2 if layout == "pair" else 1})
    write_json(output / "supervisor.json", {"schema": "olmo-allocation-supervisor-v1", "status": "completed",
               "arm": reports[0]["arm"], "layout": layout, "jobs": jobs})
    return output


def test_disjoint_windows_cannot_be_called_concurrent():
    jobs = [{"measured_started_unix": start, "measured_finished_unix": start + 10,
             "real_input_tokens": 100, "world_size": 1} for start in (100, 120)]
    result = joint_timing(jobs)
    assert result["aggregate_window_tokens_per_second"] == pytest.approx(200 / 30)
    assert result["aggregate_window_tokens_per_second"] != 20  # Incorrect sum of individual rates.
    assert result["tokens_per_allocated_gpu_second"] == pytest.approx(200 / 60)
    assert result["common_overlap_seconds"] == 0
    assert result["concurrency_qualified"] is False


def test_joint_interval_penalizes_start_gap_and_idle_tail():
    jobs = [{"measured_started_unix": 100., "measured_finished_unix": 120., "real_input_tokens": 100, "world_size": 1},
            {"measured_started_unix": 102., "measured_finished_unix": 130., "real_input_tokens": 100, "world_size": 1}]
    result = joint_timing(jobs)
    assert result["joint_makespan_seconds"] == 30
    assert result["overlap_percent_of_shorter_job"] == 90
    assert result["overlap_percent_of_makespan"] == 60
    assert result["concurrency_qualified"] is False


def test_job_timing_uses_slowest_rank_and_reports_per_rank_memory():
    report = job_report(ranks=2)
    for row in report["updates"]:
        row["timing_by_rank"][0]["backward"] -= 1
        row["memory_by_rank"][1]["peak_reserved_gib"] = 45.
    result = analyze_job(report)
    assert result["selected_compute_materialization_seconds"] == 18
    assert result["compute_seconds"] == 16
    assert result["peak_reserved_gib_max_rank"] == 45
    assert result["memory_by_rank"][0]["peak_reserved_gib"] == 40


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="running"),
    lambda r: r["updates"][1]["metrics"].update(objective=float("nan")),
    lambda r: r["updates"][1]["metrics"].update(gradient_norm_before_clip=float("inf")),
    lambda r: r["updates"][1]["metrics"].update(input_tokens=1),
    lambda r: r["summary"].update(measured_window_tokens_per_second=99.),
    lambda r: r["summary"].update(finite_updates=False),
    lambda r: r["updates"].pop(),
    lambda r: r["updates"][2].update(started_unix=100.),
])
def test_incomplete_nonfinite_or_inconsistent_report_is_rejected(change):
    report = job_report()
    change(report)
    with pytest.raises(ValueError):
        analyze_job(report)


def test_matched_layout_summary_ignores_unlisted_cpu_reports(tmp_path):
    pair = cell(tmp_path, "pair", [job_report(ranks=2)])
    singles = cell(tmp_path, "singles", [job_report(seconds=18.), job_report(seconds=18., start=100.1)])
    write_json(pair / "cpu-report.json", {"status": "failed", "not_a_gpu_job": True})
    result = build_summary([pair, singles])
    assert result["comparisons"][0]["qualified"] is True
    assert result["comparisons"][0]["singles_over_pair_aggregate_window_rate"] == pytest.approx(40 / 36.1)
    assert result["cells"][0]["matched_contract_sha256"] == result["cells"][1]["matched_contract_sha256"]


@pytest.mark.parametrize("field", ["sources", "origin", "objective_weights", "data"])
def test_source_checkpoint_objective_and_data_changes_are_rejected(tmp_path, field):
    pair = cell(tmp_path, "pair", [job_report(ranks=2)])
    changed = job_report()
    changed[field][next(iter(changed[field]))] = "different"
    singles = cell(tmp_path, "singles", [changed, changed])
    with pytest.raises(ValueError, match="Unmatched"):
        build_summary([pair, singles])


def test_unfinished_requested_cell_and_missing_layout_are_rejected(tmp_path):
    pair = cell(tmp_path, "pair", [job_report(ranks=2)])
    with pytest.raises(ValueError, match="both layouts"):
        build_summary([pair])
    path = pair / "supervisor.json"
    value = json.loads(path.read_text())
    value["status"] = "running"
    write_json(path, value)
    with pytest.raises(ValueError, match="completed"):
        load_cell(pair)


def test_report_hash_tampering_and_single_job_data_divergence_are_rejected(tmp_path):
    changed = job_report()
    changed["updates"][1]["row_key_sha256_by_rank"] = ["f" * 64]
    singles = cell(tmp_path, "singles", [job_report(), changed])
    with pytest.raises(ValueError, match="different ordered row keys"):
        load_cell(singles)
    path = singles / "single0" / "report.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="bytes differ"):
        load_cell(singles)


def test_static_artifacts_export_and_misassigned_gpu_rejection(tmp_path):
    pair = cell(tmp_path, "pair", [job_report(ranks=2)])
    singles = cell(tmp_path, "singles", [job_report(), job_report()])
    summary = build_summary([pair, singles])
    plot(summary, tmp_path)
    assert (tmp_path / "allocation.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "allocation.png").read_bytes().startswith(b"\x89PNG")
    path = singles / "supervisor.json"
    value = json.loads(path.read_text())
    value["jobs"][1]["devices"] = "0"
    write_json(path, value)
    with pytest.raises(ValueError, match="isolation"):
        load_cell(singles)
