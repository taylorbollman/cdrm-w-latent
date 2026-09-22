"""F3b summaries keep source versions, acceptance gates and measured scopes separate."""
import copy
import json
from pathlib import Path

import pytest

from scripts import olmo_f3b_report as reporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)


def timing(seconds):
    return {"wall_seconds": [seconds]*3, "cuda_seconds": [seconds*.9]*3,
        "median_wall_seconds": seconds, "median_cuda_seconds": seconds*.9}


def comparison():
    return {"passed": True, "finite": True, "bitwise_equal": True,
        "relative_l2": 0., "max_abs": 0., "reference_max_abs": 1.,
        "relative_l2_limit": 1e-5, "max_abs_limit": 1.1e-5}


def capacity():
    return {"full_step": timing(2.), "input_tokens_per_second": 64*512/2,
        "peak_allocated_gib": 40., "peak_reserved_gib": 64., "current_reserved_gib": 43.,
        "health": {"passed": True},
        "records": [{"update_completed": True, "counts": {"ce": 16384, "latent": 32704, "kl": 16384},
            "counters": {"optimizer_updates": step, "input_tokens": step*64*512}} for step in (4, 5, 6)]}


def profile_event(name, device="CUDA", count=2, microseconds=100):
    return {"name": name, "count": count, "device_type": "DeviceType."+device,
        "self_device_us": microseconds, "device_total_us": microseconds,
        "cpu_total_us": 0 if device == "CUDA" else microseconds, "self_cpu_us": 0}


@pytest.fixture
def evidence(tmp_path):
    project = tmp_path/"project"
    source = project/"scripts/runtime.py"
    write(source, "frozen implementation")
    protocol = project/reporter.PROTOCOL
    write(protocol, "frozen protocol")
    base = {"schema": "olmo-f3b-native-v1", "status": "passed", "finished_utc": "now",
        "stage": "capacity", "checkpoint": {"sha256": reporter.CHECKPOINT_SHA256},
        "configuration": {"case": "combined", "batch_size": 64, "length": 512,
            "variant": "cast_once", "stage": "capacity"},
        "source_hashes": {"scripts/runtime.py": reporter.digest(source)},
        "protocol_sha256": reporter.digest(protocol),
        "checks": [{"name": "finite_complete_updates", "passed": True}],
        "capacity": capacity(), "wandb": {"run_url": "https://wandb.ai/taylorbollman/test/runs/id"}}
    directory = tmp_path/"runs"/"capacity"
    write(directory/"report.json", base)
    return project, directory, base


def summarize(evidence, **kwargs):
    project, directory, _ = evidence
    return reporter.summarize([directory], project_root=project, **kwargs)


def test_all_eight_cards_count_shared_weights_and_exposure_separately():
    cards = reporter.resource_cards()
    assert len(cards) == 8
    assert len({(r["rt"], r["fbt"], r["nextlat"]) for r in cards}) == 8
    for row in cards:
        e = row["estimate"]
        assert e["input_tokens_per_update"] == 32768
        assert e["pass_token_work_per_update"] == 32768*(2 if row["fbt"] else 1)
        assert e["objective_positions_per_update"]["ce"] == 16384
        assert e["objective_positions_per_update"]["latent"] == (32704 if row["nextlat"] else 0)
        assert e["parameter_counts"]["training_architecture"] == 1176764416 + 8388608*row["fbt"] + 82726912*row["nextlat"]
        assert e["parameter_counts"]["deployable_inference"] == 1176764416 + 8388608*row["fbt"]
    combined = cards[-1]["estimate"]
    assert combined["matrix_flops_minimum"] == 644544689340416
    assert combined["matrix_flops_maximum"] == 689264425697280


def test_measured_full_steps_preserve_counts_and_memory_scopes(evidence):
    summary = summarize(evidence)
    assert summary["status"] == "passed"
    assert summary["successful_f3b_optimizer_updates"] == 6
    row = summary["full_steps"][0]
    assert row["input_tokens_per_second"] == 16384
    assert row["ce_targets_per_second"] == 8192
    assert row["peak_reserved_gib"] == 64 and row["current_reserved_gib"] == 43
    assert len(summary["resource_cards"]) == 8 and len(summary["full_steps"]) == 1
    markdown = reporter.markdown(summary)
    assert "not eight throughput experiments" in markdown
    assert "No quality, all-layer RT" in markdown
    assert "reserved setup peaks" in markdown.lower()


@pytest.mark.parametrize("change", ["false_gate", "missing_capacity_gate", "wrong_tps", "median", "nan", "counter", "reserved", "checkpoint"])
def test_passing_report_cannot_override_contradictory_evidence(evidence, change):
    project, directory, data = evidence
    if change == "false_gate": data["checks"][0]["passed"] = False
    elif change == "missing_capacity_gate": data["checks"][0]["name"] = "other"
    elif change == "wrong_tps": data["capacity"]["input_tokens_per_second"] = 999
    elif change == "median": data["capacity"]["full_step"]["median_wall_seconds"] = 9
    elif change == "nan": data["capacity"]["full_step"]["wall_seconds"][0] = float("nan")
    elif change == "counter": data["capacity"]["records"][0]["counters"]["input_tokens"] *= 2
    elif change == "reserved": data["capacity"]["peak_reserved_gib"] = 30
    elif change == "checkpoint": data["checkpoint"]["sha256"] = "0"*64
    write(directory/"report.json", data)
    with pytest.raises(ValueError): summarize(evidence)


def test_failed_capacity_never_enters_performance_or_update_count(evidence, tmp_path):
    _, directory, data = evidence
    data.update(status="failed", error_type="RuntimeError", error_message="failed candidate")
    data["capacity"]["input_tokens_per_second"] = float("nan")
    write(directory/"report.json", data)
    summary = summarize(evidence)
    assert summary["status"] == "completed_with_failed_diagnostics"
    assert not summary["full_steps"] and summary["successful_f3b_optimizer_updates"] == 0
    assert summary["failed_diagnostics"][0]["excluded_partial_performance"]
    write(tmp_path/"throughput.png", "stale")
    assert reporter.plot(summary, tmp_path) == []
    assert not (tmp_path/"throughput.png").exists()


def test_old_exact_source_and_protocol_snapshots_remain_valid(evidence):
    project, directory, _ = evidence
    write(directory/"source-snapshot/scripts/runtime.py", "frozen implementation")
    write(directory/"protocol.md", "frozen protocol")
    write(project/"scripts/runtime.py", "new implementation")
    write(project/reporter.PROTOCOL, "new protocol")
    summary = summarize(evidence)
    lineage = summary["runs"][0]["source_lineage"]
    assert not lineage["sources"]["scripts/runtime.py"]["current_matches"]
    assert lineage["protocol"]["verified_at"] == ["protocol.md"]
    write(directory/"source-snapshot/scripts/runtime.py", "corrupt")
    with pytest.raises(ValueError, match="snapshot"):
        summarize(evidence)


def test_unreconstructable_source_and_path_escape_reject(evidence):
    project, directory, data = evidence
    write(project/"scripts/runtime.py", "changed")
    with pytest.raises(ValueError, match="Unreconstructable"):
        summarize(evidence)
    data["source_hashes"] = {"../escape.py": "0"*64}
    write(directory/"report.json", data)
    with pytest.raises(ValueError, match="Unsafe"):
        summarize(evidence)


def tile_report(data):
    result = copy.deepcopy(data)
    result.update(schema="olmo-f3b-tile-probe-v1", configuration={"stage": "tiles"}, stage="complete")
    result.pop("capacity")
    result["checks"] = [{"name": f"tile_{i}", "kind": "frozen_tile", "passed": True,
        "candidate_vs_eager_bf16": {"passed": True},
        "candidate_vs_fp64_boundary_oracle": {"denominator": {"passed": False}, "passed": False},
        "acceptance_policy": "Oracle state diagnostic; eager state and oracle output gate."} for i in range(48)]
    result["checks"][0].update(configuration={"width": 32, "target_width": 31, "head_dim": 64},
        timing={"scope": "helper forward only", "eager": timing(.0004), "triton": timing(.0001)})
    return result


def test_nested_oracle_diagnostics_do_not_override_declared_gate(evidence):
    _, directory, data = evidence
    write(directory/"report.json", tile_report(data))
    summary = summarize(evidence)
    assert summary["status"] == "passed" and summary["declared_checks_passed"] == 48
    assert len(summary["helper_timings"]) == 1 and not summary["full_steps"]
    assert summary["runs"][0]["checks"][0]["nested_diagnostic_failures"]
    assert not summary["runs"][0]["checks"][0]["record"]["candidate_vs_fp64_boundary_oracle"]["passed"]
    text = reporter.markdown(summary)
    assert "360.0 / 90.0" in text
    assert "cannot be presented as full-model throughput" in text
    assert "host submission gaps" in text
    assert "not isolated kernel latency" in text
    assert "host submission gaps" in summary["helper_timings"][0]["measurement_scope"]


def test_failed_tile_run_excludes_even_individually_passing_helper_timings(evidence):
    _, directory, data = evidence
    data = tile_report(data)
    data["checks"][-1]["passed"] = False
    data.update(status="failed", error_type="AssertionError", error_message="last screen failed")
    write(directory/"report.json", data)
    summary = summarize(evidence)
    assert not summary["helper_timings"]
    assert summary["declared_checks_passed"] == 47
    assert summary["failed_diagnostics"][0]["declared_failed_checks"] == ["tile_47"]


def test_missing_requested_tile_cases_cannot_pass(evidence):
    _, directory, data = evidence
    data = tile_report(data); data["checks"].pop()
    write(directory/"report.json", data)
    with pytest.raises(ValueError, match="omitted"):
        summarize(evidence)


@pytest.mark.parametrize("variant", [None, "triton"])
def test_profile_annotations_are_not_counted_as_graph_kernels(evidence, variant):
    _, directory, data = evidence
    data.update(schema="olmo-f3b-profile-v1", **data.pop("capacity"))
    data["configuration"].pop("variant")
    if variant is not None:
        data["configuration"]["variant"] = variant
    data["graph_profile"] = [profile_event("gemm", count=7, microseconds=800),
        profile_event("RT/_finish", count=100, microseconds=9000),
        profile_event("aten::mm", device="CPU", count=7, microseconds=10000)]
    data["annotated_eager_profile"] = [profile_event("RT/_finish"), profile_event("RT/_finish", device="CPU")]
    write(directory/"report.json", data)
    summary = summarize(evidence)
    profile = summary["profiles"][0]
    assert profile["graph_kernel_invocations"] == 7
    assert profile["graph_kernel_self_device_us"] == 800
    assert len(profile["annotated_eager_ranges"]) == 2
    assert summary["full_steps"][0]["provenance"] == "pre-profile uninstrumented timing"
    assert summary["full_steps"][0]["variant"] == (variant or "reference")


@pytest.mark.parametrize("exists", [False, True])
def test_running_or_missing_report_requires_explicit_preview(evidence, exists):
    _, directory, data = evidence
    if exists:
        data.update(status="running"); data.pop("finished_utc")
        write(directory/"report.json", data)
    else:
        (directory/"report.json").unlink()
    with pytest.raises(ValueError): summarize(evidence)
    summary = summarize(evidence, allow_incomplete=True)
    assert summary["status"] == "partial_preview" and not summary["full_steps"]
    assert len(summary["incomplete"]) == 1


def test_native_exact_update_evidence_is_verified_not_merely_labeled(evidence):
    _, directory, data = evidence
    data["configuration"].update(stage="correctness")
    data.pop("capacity")
    comparisons = {"name": "", "passed": True, "all_bitwise_equal": True,
        "gradients": {"weight": comparison()}, "losses": {"pass0/ce": comparison()}}
    data["checks"] = [{**copy.deepcopy(comparisons), "name": name}
        for name in sorted(reporter.NATIVE_CHECKS - {"complete_adamw_update_parity"})]
    metrics = [{"update_completed": True} for _ in range(3)]
    update = {"name": "complete_adamw_update_parity", "passed": True,
        "metrics_exact": True, "model_optimizer_scheduler_counters_exact": True,
        "weights_changed": True, "updates_per_arm": 3, "physical_optimizer_updates": 6,
        "arms": [{"replay": replay, "metrics": copy.deepcopy(metrics),
            "boundary": {"model": "a", "optimizer": "b"}, "health": {"passed": True}}
            for replay in (False, True)]}
    data["checks"].append(update)
    write(directory/"report.json", data)
    summary = summarize(evidence)
    assert summary["correctness"][0]["complete_updates_exact"]
    assert summary["successful_f3b_optimizer_updates"] == 6
    update["arms"][1]["boundary"]["optimizer"] = "changed"
    write(directory/"report.json", data)
    with pytest.raises(ValueError, match="contradict"):
        summarize(evidence)


def test_fa4_smoke_keeps_runtime_versions_and_standalone_scope(evidence):
    project, directory, _ = evidence
    for name in ("scripts/olmo_fa4_smoke.py", "scripts/docker_shell.sh"):
        write(directory/"source-snapshot"/name, name)
    data = {"schema": "olmo-fa4-smoke-v1", "status": "passed", "finished_utc": "now",
        "script_sha256": reporter.digest(directory/"source-snapshot/scripts/olmo_fa4_smoke.py"),
        "launcher_sha256": reporter.digest(directory/"source-snapshot/scripts/docker_shell.sh"),
        "capture_output_exact": True, "versions": {"flash-attn-4": "test"},
        "comparisons": {n: {"relative_l2": .001, "max_relative": .002} for n in ("output", "dq", "dk", "dv")}}
    write(directory/"report.json", data)
    summary = summarize(evidence)
    assert summary["fa4"][0]["versions"]["flash-attn-4"] == "test"
    assert "Standalone" in summary["fa4"][0]["scope"]
    assert not summary["full_steps"] and not summary["correctness"]


def test_plot_exports_separate_measured_axes(evidence, tmp_path):
    summary = summarize(evidence)
    paths = reporter.plot(summary, tmp_path/"output")
    assert len(paths) == 2
    assert all(Path(path).stat().st_size > 1000 for path in paths)
