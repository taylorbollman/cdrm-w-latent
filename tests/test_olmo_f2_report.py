"""Independent F2 report acceptance, throughput arithmetic and scope checks."""
import copy
import json
from pathlib import Path

import pytest

from scripts import olmo_f2_report as reporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def data(tmp_path):
    root = tmp_path/"project"
    source = root/"runtime.py"
    source.parent.mkdir(); source.write_text("runtime")
    protocol = root/"docs/reports/olmo1b-f2/protocol.md"
    protocol.parent.mkdir(parents=True); protocol.write_text("frozen protocol")
    base = {"schema": reporter.HEALTH_SCHEMA, "status": "passed", "finished_utc": "now",
        "source_hashes": {"runtime.py": reporter.sha256_file(source)},
        "protocol_sha256": reporter.sha256_file(protocol), "checkpoint": {"sha256": "a"*64},
        "runtime": {"gpu": "fixture"}, "wandb": {"run_url": "https://wandb.ai/fixture"},
        "config": {"stage": "capacity", "comfortable_gib": 65}}
    records = [{"objective": 3., "gradient_norm_before_clip": 2., "counts": {"ce": 40},
        "loss_means": {"ce": 3.}} for _ in range(4)]
    measured = {"case": {"name": "combined", "batch_size": 2, "length": 32},
        "checkpointing": False, "physical_batch": 2, "status": "measured", "passed": True,
        "warmup_updates": 2, "timed_updates": 2, "wall_seconds": [.2, .4], "median_wall_seconds": .3,
        "valid_input_tokens_per_update": 60, "valid_input_tokens_per_second": 200., "ce_targets_per_second": 40/.3,
        "peak_allocated_gib": 50., "peak_reserved_gib": 55., "within_comfortable_memory": True,
        "post_timing_state_health": {"passed": True}, "records": records,
        "gradient_accumulation": 1, "cuda_graphs": False, "scope": "complete optimizer step"}
    capacity = copy.deepcopy(base); capacity["rows"] = [measured]
    checkpoint = copy.deepcopy(base)
    checkpoint["config"]["stage"] = "checkpoint"
    checkpoint["rows"] = [{"case": measured["case"], "passed": True, "metrics_exact": True, "boundary_exact": True}]
    graph = {key: copy.deepcopy(value) for key, value in base.items() if key not in ("config", "protocol_sha256")}
    graph.update(schema=reporter.GRAPH_SCHEMA, configuration={"batch_size": 1, "length": 32, "rt_layers": [0]},
        capture_succeeded=True, stage="complete", comparisons=[{"name": name, "passed": True,
            "all_bitwise_equal": True, "hidden": {"passed": True},
            "gradients": {"native": {"passed": True, "relative_l2": 0.}}} for name in sorted(reporter.GRAPH_CASES)],
        timings={"eager": {"median_wall_seconds": .3}, "graph_replay": {"median_wall_seconds": .1}},
        replay_speedup_wall=3., peak_allocated_gib_timing=20., reserved_gib_after_capture=25.)
    return root, base, capacity, checkpoint, graph


def summarize(tmp_path, data, *reports):
    paths = [write(tmp_path/f"run-{index}/report.json", report) for index, report in enumerate(reports)]
    return reporter.build_summary(paths, project_root=data[0])


def test_capacity_uses_actual_valid_tokens_and_keeps_graph_timing_separate(tmp_path, data):
    summary = summarize(tmp_path, data, data[2], data[3], data[4])
    assert summary["capacity"][0]["valid_input_tokens_per_second"] == 200
    assert summary["capacity"][0]["physical_batch"] == 2
    assert summary["capacity"][0]["last_loss_means"] == {"ce": 3.}
    assert summary["graphs"][0]["replay_speedup_wall"] == 3
    text = reporter.markdown(summary)
    assert "CE targets/s" in text and "fixed hidden-state cotangent" in text
    assert "not end-to-end training throughput" in text
    assert summary["checkpoint"][0]["boundary_exact"]


def test_graph_report_explicitly_labels_forced_backend_and_determinism(tmp_path, data):
    data[4]["configuration"].update(backend="flash", deterministic_algorithms=True)
    text = reporter.markdown(summarize(tmp_path, data, data[4]))
    assert "ordinary backend flash, deterministic True" in text


@pytest.mark.parametrize("mutation", ["failed_report", "failed_row", "median", "throughput", "ce_throughput",
    "comfortable", "nonfinite", "counts", "accumulation", "state_health", "protocol"])
def test_core_failures_or_wrong_capacity_arithmetic_are_rejected(tmp_path, data, mutation):
    row = data[2]["rows"][0]
    if mutation == "failed_report": data[2]["status"] = "failed"
    elif mutation == "failed_row": row["passed"] = False
    elif mutation == "median": row["median_wall_seconds"] = .2
    elif mutation == "throughput": row["valid_input_tokens_per_second"] = 999
    elif mutation == "ce_throughput": row["ce_targets_per_second"] = 999
    elif mutation == "comfortable": row["within_comfortable_memory"] = False
    elif mutation == "nonfinite": row["records"][0]["gradient_norm_before_clip"] = float("inf")
    elif mutation == "counts": row["timed_updates"] = 1
    elif mutation == "accumulation": row["gradient_accumulation"] = 2
    elif mutation == "state_health": row["post_timing_state_health"]["passed"] = False
    elif mutation == "protocol": data[2]["protocol_sha256"] = "changed"
    with pytest.raises(ValueError): summarize(tmp_path, data, data[2])


def test_oom_is_a_capacity_boundary_not_a_successful_measured_cell(tmp_path, data):
    data[2]["rows"] = [{"case": {"name": "rt", "batch_size": 64, "length": 512},
        "checkpointing": False, "physical_batch": 64, "status": "oom_capacity_limit", "passed": True}]
    summary = summarize(tmp_path, data, data[2])
    assert "valid_input_tokens_per_second" not in summary["capacity"][0]
    assert "OOM" in reporter.markdown(summary)


def test_failed_graph_is_explicitly_unresolved_without_timing_claim(tmp_path, data):
    failed = copy.deepcopy(data[4])
    failed.update(status="failed", stage="equivalence", error_message="replay differs", error_type="AssertionError")
    failed["comparisons"][0]["passed"] = False
    # Even if a malformed producer left timings, failure cannot become a speed claim.
    summary = summarize(tmp_path, data, data[2], failed)
    assert summary["status"] == "core_passed_with_graph_qualifications"
    assert "timings" not in summary["graphs"][0]
    assert "failed original graph check" in reporter.markdown(summary)


def test_historical_capture_blocker_keeps_recorded_source_drift(tmp_path, data):
    blocked = copy.deepcopy(data[4]); blocked.update(status="capture_blocked", capture_succeeded=False,
        stage="capture", comparisons=[], error_message="CPU scalar copy", error_type="RuntimeError")
    blocked["source_hashes"]["runtime.py"] = "old"
    summary = summarize(tmp_path, data, data[4], blocked)
    assert summary["status"] == "passed"
    assert summary["inputs"][1]["source"]["changed_since_run"] == ["runtime.py"]
    assert summary["graphs"][1]["disposition"] == "historical capture blocker"


def test_successful_report_cannot_silently_use_changed_sources(tmp_path, data):
    (data[0]/"runtime.py").write_text("modified")
    with pytest.raises(ValueError, match="source changed"):
        summarize(tmp_path, data, data[4])


def test_checkpoint_parity_failure_cannot_be_hidden_by_passed_top_level(tmp_path, data):
    data[3]["rows"][0]["boundary_exact"] = False
    with pytest.raises(ValueError): summarize(tmp_path, data, data[3])


def test_passed_graph_requires_all_four_changed_state_and_overwrite_checks(tmp_path, data):
    data[4]["comparisons"].pop()
    with pytest.raises(ValueError, match="equivalence"):
        summarize(tmp_path, data, data[4])


def test_figures_are_standalone_and_use_only_capacity_cells(tmp_path, data):
    summary = summarize(tmp_path, data, data[2], data[4])
    reporter.plot_capacity(summary, tmp_path)
    assert (tmp_path/"throughput.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path/"throughput.png").read_bytes().startswith(b"\x89PNG")


def test_completed_localization_does_not_claim_equality_or_resolve_changed_weight_failure(tmp_path, data):
    local = copy.deepcopy(data[4])
    names = ["eager_default_repeat_1", "graph_repeat_1", "eager_capture_stream_vs_default_stream",
        "graph_vs_eager_default", "graph_vs_eager_capture_stream", "eager_default_after_capture_vs_before",
        "graph_vs_eager_default_after_capture"]
    local.update(schema=reporter.LOCALIZATION_SCHEMA, status="completed", structural_invariants_passed=True,
        all_comparisons_within_original_budget=False, all_comparisons_bitwise_equal=False,
        comparisons=[{"name": name, "passed": "graph_vs" not in name, "all_bitwise_equal": False,
            "max_gradient_relative_l2": .0001, "highest_layer_outside_original_budget": 15}
            for name in names])
    local["configuration"].update(backend="auto", repeats=1)
    summary = summarize(tmp_path, data, data[2], local)
    assert summary["status"] == "core_passed_with_graph_qualifications"
    assert "timings" not in summary["graph_localization"][0]
    text = reporter.markdown(summary)
    assert "Completed localization means the controls ran" in text
    assert "eager_capture_stream_vs_default_stream" in text
    local["all_comparisons_within_original_budget"] = True
    with pytest.raises(ValueError, match="outcome"):
        summarize(tmp_path, data, local)
