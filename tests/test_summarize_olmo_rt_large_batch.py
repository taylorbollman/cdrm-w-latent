"""Frozen evidence, phase-memory failures and optimizer counts without GPU work."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import summarize_olmo_rt_large_batch as report


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-c", "user.name=Evidence test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    for name in report.ESSENTIAL_SOURCES | {report.VALIDATION_SOURCE}:
        write(root, name, "frozen " + name)
    write(root, report.PROTOCOL, "frozen protocol")
    git(root, "add", ".")
    git(root, "commit", "-m", "first")
    first = git(root, "rev-parse", "HEAD")
    write(root, "cdrm/pretrained/olmo.py", "new source")
    git(root, "add", ".")
    git(root, "commit", "-m", "second")
    return SimpleNamespace(root=root, first=first, second=git(root, "rev-parse", "HEAD"))


def fixture_batch(batch, step):
    rows = {key: {"shape": [batch, 512], "dtype": "torch.int64" if key in {"input_ids", "document_ids"} else "torch.bool",
        "sha256": hashlib.sha256((key + str(step if key == "input_ids" else 0)).encode()).hexdigest()}
        for key in report.BATCH_FIELDS}
    rows["valid_mask"]["sha256"] = hashlib.sha256(bytes([1]) * batch * 512).hexdigest()
    rows["ce_mask"] = deepcopy(rows["valid_mask"])
    return rows


def phase():
    snapshot = {"allocated_gib": 20., "reserved_gib": 40., "peak_allocated_gib": 30.,
        "peak_reserved_gib": 45., "device_free_gib": 30., "device_total_gib": 80., "device_used_gib": 50.}
    return {"start": deepcopy(snapshot), "end": deepcopy(snapshot), "reset_peaks": True,
        "peak_allocated_gib": 30., "peak_reserved_gib": 45., "measurement_scope": "absolute phase peaks"}


def add_run(evidence, name, *, arm="optimized", case="rt", stage="capacity", status="passed", revision=None,
            batch=None, cleanup=False, profile=False, validation_order=None):
    root, revision = evidence.root, revision or evidence.second
    directory = root / report.RUNTIME / name
    hashes = {}
    for source in report.ESSENTIAL_SOURCES | {report.VALIDATION_SOURCE}:
        path = write(directory, "source-snapshot/" + source, git(root, "show", revision + ":" + source))
        hashes[source] = report.digest(path)
    protocol = write(directory, "protocol.md", git(root, "show", revision + ":" + report.PROTOCOL))
    batch = batch or (64 if stage == "capacity" else 8)
    attention, pointwise, rope, fused = report.ARMS[arm]
    config = {"case": case, "arm": arm, "reference_arm": "control", "stage": stage,
        "batch_size": batch, "length": 512, "precision": "bf16_mixed", "parameter_optimizer_dtype": "float32",
        "supervision": "all_valid_ce", "ce_chunk_size": 2048, "kl_chunk_size": 128, "reuse_rope": True,
        "selected_rt_layers": [0, 15], "world_size": 1, "accumulation": 1,
        "ordinary_attention": attention, "ordinary_pointwise": pointwise, "ordinary_rope_backend": rope,
        "ordinary_checkpointing": "all", "fused_adam": fused, "profile": profile, "release_transient_cache": cleanup,
        "fbt": case == "combined", "nextlat": case == "combined", "capture_warmup_backwards": 10,
        "kv_only_writes": True, "mode": {"enabled": case == "combined", "num_passes": 2 if case == "combined" else 1,
                                          "rt_mode": {"selected_layers": [0, 15]}}}
    if validation_order is not None:
        config["validation_order"] = validation_order
    counts = {"ce": batch * 511, "latent": batch * 511 if case == "combined" else 0,
              "kl": batch * 510 if case == "combined" else 0}
    parameters = {"registered_unique": 110, "trainable": 100, "executed_declared": 100,
                  "deployable_inference_declared": 100}
    observed = {**parameters, "gradient_participating": 100, "optimizer_owned": 100 if stage == "capacity" else None}
    flags = [{"fused": fused, "foreach": False, "capturable": False}]
    raw = {"schema": report.SCHEMA, "configuration": config, "status": status, "stage": "complete",
        "finished_utc": "2026-09-24T00:00:00Z", "runtime_commit": revision,
        "source_hashes": hashes, "protocol_sha256": report.digest(protocol),
        "physical_optimizer_updates": 6 if stage == "correctness" else 8 + int(profile),
        "checks": [{"name": name, "passed": True} for name in sorted(report.EXPECTED_CHECKS[stage])],
        "checkpoint": {"sha256": "a" * 64, "uri": "gs://fast-chunks/fixture"},
        "dependencies": {"packages": {"torch": "fixed", "flash-attn": "fixed"}, "dao_sources": {}, "fa4_sources": {}},
        "runtime": {"gpu": "H100"}, "determinism": {"enabled": True}, "parameters": parameters,
        "input_tokens": batch * 512, "counts": counts, "initial_batch": fixture_batch(batch, 0),
        "resources": {"analytic_matrix_work": {"components": [{"name": "work", "minimum": 100, "maximum": 200}],
            "matrix_flops_minimum": 100, "matrix_flops_maximum": 200,
            "input_tokens_per_update": batch * 512, "pass_token_work_per_update": batch * 512 * (2 if case == "combined" else 1),
            "rt_block_calls_per_microbatch": 2, "ordinary_block_calls_per_microbatch": 30 if case == "combined" else 14},
            "observed_parameters": observed, "loss_work": {"ce_targets": counts["ce"], "latent_pairs": counts["latent"], "kl_triples": counts["kl"]}},
        "wandb": {"run_url": "https://wandb.ai/taylorbollman/test/runs/fixture"},
        "prepared_layout": {"all_tokens_valid": True, "rt_implementation": "native", "ordinary_attention_backend": attention,
            "ordinary_pointwise_backend": pointwise, "ordinary_rope_backend": rope, "reuse_rope": True,
            "ordinary_checkpoint_layers": None},
        "compiler_observations": {"stats": {"unique_graphs": 2}},
        "memory_phases": {key: phase() for key in ("load_model", "prepare_plan", "dispatch", "preparation_updates",
            "capture_warmup", "capture_capture", "validation_initial", "timing", "backward_timing", "health")}}
    if rope == "dao":
        for source in ("layers/rotary.py", "ops/triton/rotary.py"):
            path = write(directory, "dependency-snapshot/flash_attn/" + source, "installed " + source)
            raw["dependencies"]["dao_sources"][source] = {"source": "/installed/" + source, "sha256": report.digest(path)}
    if attention == "fa4":
        path = write(directory, "dependency-snapshot/flash_attn/cute/interface.py", "installed FA4")
        raw["dependencies"]["fa4_sources"]["interface.py"] = {"source": "/installed/interface.py", "sha256": report.digest(path)}
    if stage == "correctness":
        raw["comparison_batches"] = {str(index): fixture_batch(batch, index) for index in (1, 2, 5, 6, 7)}
        next(c for c in raw["checks"] if c["name"] == "complete_adamw_update_parity")["optimizer_flags"] = flags
        next(c for c in raw["checks"] if c["name"] == "same_state_candidate_vs_reference").update(
            finite=True, ownership_matches=True, counts_equal=True,
            **{key: {"x": {"relative_l2": 0., "max_relative": 0.}} for key in ("losses", "outputs", "gradients")})
    def record(index):
        return {"update_completed": True, "counts": deepcopy(counts), "counters": {
            "optimizer_updates": index, "microbatches": index, "input_tokens": index * batch * 512,
            "documents": index * batch, "ce_positions": index * counts["ce"],
            "latent_pairs": index * counts["latent"], "kl_triples": index * counts["kl"]}}
    if stage == "capacity":
        def timing(count):
            return {"wall_seconds": [2.] * count, "cuda_seconds": [1.9] * count,
                    "median_wall_seconds": 2., "median_cuda_seconds": 1.9}
        raw.update(preparation_records=[record(i) for i in range(1, 4)], timed_records=[record(i) for i in range(4, 9)],
            preparation_batches=[fixture_batch(batch, i) for i in range(3)], timed_batches=[fixture_batch(batch, i) for i in range(3, 8)],
            full_update=timing(5), forward_loss_backward=timing(3), input_tokens_per_second=batch * 256.,
            ce_targets_per_second=counts["ce"] / 2, forward_loss_backward_tokens_per_second=batch * 256.,
            optimizer_flags=flags, health={"passed": True}, setup_memory=phase()["end"], steady_memory=phase()["end"])
    if profile:
        for key, filename in (("profile", "operator-trace.json.gz"), ("full_step_profile", "full-step-trace.json.gz")):
            path = write(directory, filename, gzip.compress(b'{"traceEvents":[]}'))
            raw[key] = {"trace_file": filename, "trace_bytes": path.stat().st_size, "trace_sha256": report.digest(path),
                        "device_event_count": 1, "device_kernels": {"matmul": {"calls": 1, "self_device_us": 2.}}}
        raw["full_step_profile"]["update_record"] = record(9)
        raw["profile_batch"] = fixture_batch(batch, 9)
        raw["post_profile_health"] = {"passed": True}
    if validation_order == "before-capture":
        raw["validation_reference_batches"] = {"initial": deepcopy(raw["preparation_batches"][2]),
                                               "changed": deepcopy(raw["preparation_batches"][1])}
        raw["memory_phases"].update({name: phase() for name in (
            "validation_references", "validation_changed_tokens", "validation_changed_weights")})
        for name, replays in (("capacity_initial_graph", 1), ("capacity_changed_tokens_overwrite", 2),
                              ("capacity_changed_weights", None)):
            check = next(check for check in raw["checks"] if check["name"] == name)
            check.update(storage_matches=True, ownership_matches=True, loss_names_match=True, all_bitwise_equal=True,
                losses={"loss": {"bitwise_equal": True}}, gradients={"weight": {"bitwise_equal": True}})
            if replays is None:
                check["graph_released_before_eager"] = True
            else:
                check["replays_checked"] = replays
    if validation_order is not None and cleanup:
        raw["memory_phases"].update({name: phase() for name in (
            "capture_pre_warmup_transient_cleanup", "capture_transient_cleanup")})
    if status != "passed":
        raw.update(stage="capture", checks=[], physical_optimizer_updates=3,
                   error={"type": "OutOfMemoryError", "message": "retained setup failure"})
        raw["memory_phases"]["capture_capture"].pop("end")
        raw["memory_phases"]["capture_capture"]["error"] = True
    path = write(directory, "report.json", raw)
    return path, raw


def test_runtime_source_and_arm_contract():
    from scripts import olmo_rt_large_batch as harness
    assert report.ESSENTIAL_SOURCES <= set(harness.SOURCES)
    assert report.VALIDATION_SOURCE in harness.SOURCES
    assert report.ARMS == {key: tuple(value[name] for name in ("attention", "pointwise", "rope", "fused_adam"))
                           for key, value in harness.ARMS.items()}


def test_failed_shapes_updates_and_wandb_remain_visible(evidence):
    path, raw = add_run(evidence, "old-oom", revision=evidence.first, status="oom", batch=256)
    # The old harness kept stage=capture when validation failed after capture.
    raw["memory_phases"]["capture_capture"] = phase()
    raw["memory_phases"]["validation_initial"]["error"] = True
    write(path.parent, "report.json", raw)
    add_run(evidence, "control", arm="control")
    add_run(evidence, "optimized")
    result = report.summarize(["old-oom", "control", "optimized"], evidence.second,
        root=evidence.root, overrides={"old-oom": evidence.first})
    assert result["statuses"] == {"oom": 1, "passed": 2}
    assert result["physical_optimizer_updates"] == 19
    assert result["capacity"][0]["failure_stage"] == "capture"
    assert result["capacity"][0]["failed_memory_phases"] == ["validation_initial"]
    assert result["runs"][0]["failed_memory_phases"] == ["validation_initial"]
    assert not result["runs"][0]["memory_phases"]["capture_capture"].get("error")
    assert result["runs"][1]["wandb_url"].startswith("https://wandb.ai/")
    assert result["comparison_groups"][0]["gain_fraction_vs_control"] == {"optimized": 0.}


@pytest.mark.parametrize("profile,cleanup", [(False, False), (True, True)])
def test_before_capture_preserves_full_updates_and_terminal_evidence(evidence, profile, cleanup):
    add_run(evidence, "before", case="combined", validation_order="before-capture", profile=profile, cleanup=cleanup)
    result = report.summarize(["before"], evidence.second, root=evidence.root)
    row = result["runs"][0]
    assert result["physical_optimizer_updates"] == 8 + int(profile)
    assert row["validation_order"] == "before-capture"
    assert set(row["validation_reference_batches"]) == {"initial", "changed"}
    terminal = next(check for check in row["checks"] if check["name"] == "capacity_changed_weights")
    assert terminal["graph_released_before_eager"] and terminal["storage_matches"]
    assert result["comparison_groups"][0]["validation_order"] == "before-capture"


@pytest.mark.parametrize("damage", ["references", "initial_batch", "changed_batch", "storage", "release",
    "replays", "gradients", "source", "phase", "cleanup", "order"])
def test_before_capture_rejects_missing_or_inconsistent_validation_evidence(evidence, damage):
    path, raw = add_run(evidence, "before", validation_order="before-capture", cleanup=True)
    checks = {check["name"]: check for check in raw["checks"]}
    if damage == "references": raw.pop("validation_reference_batches")
    elif damage == "initial_batch": raw["validation_reference_batches"]["initial"] = raw["preparation_batches"][0]
    elif damage == "changed_batch": raw["validation_reference_batches"]["changed"] = raw["preparation_batches"][2]
    elif damage == "storage": checks["capacity_initial_graph"]["storage_matches"] = False
    elif damage == "release": checks["capacity_changed_weights"]["graph_released_before_eager"] = False
    elif damage == "replays": checks["capacity_changed_tokens_overwrite"]["replays_checked"] = 1
    elif damage == "gradients": checks["capacity_changed_weights"]["gradients"]["weight"]["bitwise_equal"] = False
    elif damage == "source": raw["source_hashes"].pop(report.VALIDATION_SOURCE)
    elif damage == "phase": raw["memory_phases"].pop("validation_references")
    elif damage == "cleanup": raw["memory_phases"].pop("capture_pre_warmup_transient_cleanup")
    elif damage == "order": raw["configuration"]["validation_order"] = "unknown"
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        report.summarize(["before"], evidence.second, root=evidence.root)


def test_validation_order_normalizes_legacy_default_and_separates_curves(evidence, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    add_run(evidence, "legacy")
    add_run(evidence, "explicit", validation_order="live-graph")
    add_run(evidence, "before", validation_order="before-capture")
    result = report.summarize(["legacy", "explicit", "before"], evidence.second, root=evidence.root)
    assert len({row["runtime_fingerprint"] for row in result["capacity"]}) == 1
    assert sorted(len(group["runs"]) for group in result["comparison_groups"]) == [1, 2]
    assert {group["validation_order"] for group in result["comparison_groups"]} == {"live-graph", "before-capture"}
    figures, close = [], plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    report.render_plot(result, tmp_path / "plots")
    figure = figures[-1]
    assert len(figure.axes[0].lines) == 2
    assert sum("before-capture" in line.get_label() for line in figure.axes[0].lines) == 1
    close(figure)


@pytest.mark.parametrize("case", ["rt", "combined"])
@pytest.mark.parametrize("arm", ["control", "optimized", "fa4", "compiled-native", "fa4-native"])
def test_all_cases_arms_validate_real_update_and_loss_scope(evidence, case, arm):
    add_run(evidence, "case", case=case, arm=arm, stage="correctness")
    result = report.summarize(["case"], evidence.second, root=evidence.root)
    assert result["physical_optimizer_updates"] == 6 and result["statuses"] == {"passed": 1}
    assert result["runs"][0]["dependency_verification"]["files_checked"] == {
        "control": 0, "optimized": 2, "fa4": 3, "compiled-native": 0, "fa4-native": 1}[arm]


@pytest.mark.parametrize("damage", ["source", "protocol", "dependency", "updates", "counter", "gate", "memory",
    "phase", "phase_error", "shape", "ce", "tokens", "median", "throughput", "resource", "owner", "optimizer", "scope",
    "setup_peak", "steady_peak", "mode", "layout", "compiler"])
def test_damaged_completed_evidence_is_rejected(evidence, damage):
    path, raw = add_run(evidence, "candidate", case="combined")
    if damage == "source":
        write(path.parent, "source-snapshot/cdrm/pretrained/olmo.py", "changed")
    elif damage == "protocol":
        write(path.parent, "protocol.md", "changed")
    elif damage == "dependency":
        write(path.parent, "dependency-snapshot/flash_attn/layers/rotary.py", "changed")
    elif damage == "updates": raw["physical_optimizer_updates"] = 7
    elif damage == "counter": raw["timed_records"][4]["counters"]["optimizer_updates"] = 7
    elif damage == "gate": raw["checks"].pop()
    elif damage == "memory": raw["memory_phases"]["capture_capture"]["end"]["device_used_gib"] = float("nan")
    elif damage == "phase": raw["memory_phases"].pop("capture_capture")
    elif damage == "phase_error": raw["memory_phases"]["capture_capture"]["error"] = True
    elif damage == "shape": raw["configuration"]["batch_size"] = 128
    elif damage == "ce": raw["initial_batch"]["ce_mask"]["sha256"] = "b" * 64
    elif damage == "tokens": raw["input_tokens"] *= 2
    elif damage == "median": raw["full_update"]["median_wall_seconds"] = 3.
    elif damage == "throughput": raw["input_tokens_per_second"] *= 2
    elif damage == "resource": raw["resources"]["analytic_matrix_work"]["rt_block_calls_per_microbatch"] = 1
    elif damage == "owner": raw["resources"]["observed_parameters"]["optimizer_owned"] -= 1
    elif damage == "optimizer": raw["optimizer_flags"][0]["fused"] = None
    elif damage == "scope": raw["configuration"]["accumulation"] = 2
    elif damage == "setup_peak": raw["setup_memory"]["peak_reserved_gib"] = 40.
    elif damage == "steady_peak": raw["steady_memory"]["peak_reserved_gib"] = 50.
    elif damage == "mode": raw["configuration"]["mode"]["num_passes"] = 3
    elif damage == "layout": raw["prepared_layout"]["rt_implementation"] = "author"
    elif damage == "compiler": raw["compiler_observations"]["graph_break"] = {"fallback": 1}
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        report.summarize(["candidate"], evidence.second, root=evidence.root)


def test_completed_numeric_failure_stays_failed_with_successful_operational_checks(evidence):
    path, raw = add_run(evidence, "qualified", stage="correctness", arm="fa4")
    next(c for c in raw["checks"] if c["name"] == "same_state_candidate_vs_reference")["passed"] = False
    raw.update(status="failed", compatibility_miss_continued=True, numerical_compatibility_passed=False,
        operational_checks_passed=True, error={"type": "AssertionError", "message": "retained numeric miss"})
    raw["configuration"]["continue_after_compatibility_miss"] = True
    write(path.parent, "report.json", raw)
    result = report.summarize(["qualified"], evidence.second, root=evidence.root)
    assert result["statuses"] == {"failed": 1} and result["physical_optimizer_updates"] == 6
    assert result["runs"][0]["gate_groups"]["operational"]["complete_and_passed"]
    assert not result["runs"][0]["gate_groups"]["compatibility"]["complete_and_passed"]



def test_capacity_and_plot_preserve_failed_integration_qualification(evidence, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    path, raw = add_run(evidence, "failed-integration", stage="correctness", arm="fa4-native")
    next(c for c in raw["checks"] if c["name"] == "same_state_candidate_vs_reference")["passed"] = False
    raw.update(status="failed", compatibility_miss_continued=True, numerical_compatibility_passed=False,
        operational_checks_passed=True, error={"type": "AssertionError", "message": "retained numeric miss"})
    raw["configuration"]["continue_after_compatibility_miss"] = True
    write(path.parent, "report.json", raw)
    add_run(evidence, "operational-capacity", arm="fa4-native")
    add_run(evidence, "different-case", arm="fa4-native", case="combined")
    result = report.summarize(["failed-integration", "operational-capacity", "different-case"],
        evidence.second, root=evidence.root)
    capacity = {row["name"]: row for row in result["capacity"]}
    assert capacity["operational-capacity"]["status"] == "passed"
    checks = capacity["operational-capacity"]["selected_same_case_arm_integration"]
    assert len(checks) == 1 and checks[0]["status"] == "failed"
    assert not checks[0]["compatibility_complete_and_passed"]
    assert checks[0]["operational_complete_and_passed"]
    assert capacity["different-case"]["selected_same_case_arm_integration"] == []
    figures, close = [], plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    report.render_plot(result, tmp_path / "plots")
    figure = figures[-1]
    assert sum("integration not cleared" in line.get_label() for axis in figure.axes for line in axis.lines) == 3
    assert any("Operational capacity passes do not clear" in text.get_text() for text in figure.texts)
    close(figure)


@pytest.mark.parametrize("missing", ["dao_sources", "fa4_sources"])
def test_completed_numeric_failure_requires_dependency_evidence(evidence, missing):
    path, raw = add_run(evidence, "qualified", stage="correctness", arm="fa4")
    next(c for c in raw["checks"] if c["name"] == "same_state_candidate_vs_reference")["passed"] = False
    raw.update(status="failed", compatibility_miss_continued=True, numerical_compatibility_passed=False,
        operational_checks_passed=True, error={"type": "AssertionError", "message": "retained numeric miss"})
    raw["configuration"]["continue_after_compatibility_miss"] = True
    raw["dependencies"][missing] = {}
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError, match="Missing (Dao|FA4)"):
        report.summarize(["qualified"], evidence.second, root=evidence.root)


@pytest.mark.parametrize("difference", ["source", "cleanup", "fixture"])
def test_comparison_cohorts_do_not_hide_source_cleanup_or_fixture_differences(evidence, difference):
    add_run(evidence, "control", arm="control", revision=evidence.first if difference == "source" else evidence.second)
    path, raw = add_run(evidence, "optimized", cleanup=difference == "cleanup")
    if difference == "fixture":
        raw["timed_batches"][2]["input_ids"]["sha256"] = "b" * 64
        write(path.parent, "report.json", raw)
    result = report.summarize(["control", "optimized"], evidence.second, root=evidence.root,
        overrides={"control": evidence.first} if difference == "source" else None)
    assert len(result["comparison_groups"]) == 2
    assert all(not group["gain_fraction_vs_control"] for group in result["comparison_groups"])


def test_profile_counts_ninth_update_and_retains_both_traces(evidence):
    path, raw = add_run(evidence, "profile", profile=True)
    result = report.summarize(["profile"], evidence.second, root=evidence.root)
    assert result["physical_optimizer_updates"] == 9 and len(result["profiles"]) == 2
    raw["physical_optimizer_updates"] = 8
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError, match="update count"):
        report.summarize(["profile"], evidence.second, root=evidence.root)


def test_native_rt_profile_excludes_annotations_and_limits_kernel_attribution(evidence):
    path, raw = add_run(evidence, "profile", profile=True)
    profile = raw["full_step_profile"]
    trace = {"traceEvents": [
        {"cat": "user_annotation", "ph": "X", "name": "ordinary_step/complete", "dur": 123.},
        {"cat": "user_annotation", "ph": "X", "name": "ordinary_step/forward_loss_backward", "dur": 100.},
        {"cat": "gpu_user_annotation", "ph": "X", "name": "captured_rt_region", "dur": 1000.},
    ]}
    trace_path = write(path.parent, profile["trace_file"], gzip.compress(json.dumps(trace).encode()))
    profile.update(trace_bytes=trace_path.stat().st_size, trace_sha256=report.digest(trace_path),
        cpu_phase_scopes={"ordinary_step/complete": {"cpu_us": 0., "device_us": 9999., "calls": 1}},
        device_kernels={name: {"calls": 1, "self_device_us": duration} for name, duration in {
            "ordinary_step/complete": 5000., "captured_rt_region": 1000.,
            "historical_recomputed_backward_kernel": 10., "historical_backward_kernel": 5.,
            "kernel": 20., "nvjet_sm90_gemm": 40., "triton_poi_fused_silu_mul_0": 25.,
        }.items()}, device_event_count=7)
    write(path.parent, "report.json", raw)
    result = report.summarize(["profile"], evidence.second, root=evidence.root)
    derived = next(p for p in result["profiles"] if p["profile_kind"] == "full_step_profile")
    assert derived["summed_device_ms"] == .1 and derived["device_event_count"] == 5
    assert derived["excluded_gpu_annotations"]["calls"] == 2
    assert derived["cpu_phase_scopes"]["ordinary_step/complete"]["cpu_us"] == 123.
    assert derived["kernel_name_categories"]["native_rt_historical_backward_named"]["self_device_us"] == 15.
    assert derived["kernel_name_categories"]["matrix_multiply_named"]["self_device_us"] == 40.
    assert derived["kernel_name_categories"]["other_or_unclassified"]["self_device_us"] == 20.
    assert derived["native_rt_named_kernels"]["share_of_summed_device_time"] == .15
    assert set(derived["native_rt_named_kernels"]["events"]) == report.NATIVE_RT_NAMED_KERNELS
    assert "not assigned to RT finish/writer" in derived["classification_scope"]
    assert "prefix does not mean ordinary-only" in derived["cpu_phase_scope_qualification"]


def test_plot_capacity_and_setup_memory(evidence, tmp_path):
    add_run(evidence, "b64")
    add_run(evidence, "b128", batch=128)
    result = report.summarize(["b64", "b128"], evidence.second, root=evidence.root)
    report.render_plot(result, tmp_path / "plots")
    assert (tmp_path / "plots/capacity.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "plots/capacity.png").read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("change", ["reporting", "source", "protocol", "dependency"])
def test_capacity_curves_follow_verified_content_across_commits(evidence, tmp_path, monkeypatch, change):
    import matplotlib.pyplot as plt

    add_run(evidence, "b64", batch=64)
    changed_path = {"source": "cdrm/pretrained/olmo.py", "protocol": report.PROTOCOL}.get(
        change, "docs/reporting-note.md")
    write(evidence.root, changed_path, "capacity follow-up " + change)
    git(evidence.root, "add", changed_path)
    git(evidence.root, "commit", "-m", change)
    revision = git(evidence.root, "rev-parse", "HEAD")
    path, raw = add_run(evidence, "b128", batch=128, revision=revision)
    if change == "dependency":
        source = "layers/rotary.py"
        dependency = write(path.parent, "dependency-snapshot/flash_attn/" + source, "new installed dependency")
        raw["dependencies"]["dao_sources"][source]["sha256"] = report.digest(dependency)
        write(path.parent, "report.json", raw)
    result = report.summarize(["b64", "b128"], revision, root=evidence.root,
        overrides={"b64": evidence.second})
    rows = result["capacity"]
    assert [row["runtime_commit"] for row in rows] == [evidence.second, revision]
    assert (rows[0]["runtime_fingerprint"] == rows[1]["runtime_fingerprint"]) == (change == "reporting")
    figures, close = [], plt.close
    monkeypatch.setattr(plt, "close", figures.append)
    report.render_plot(result, tmp_path / "plots")
    figure = figures[-1]
    curves = [list(line.get_xdata()) for line in figure.axes[0].lines]
    close(figure)
    assert sorted(curves) == ([[64, 128]] if change == "reporting" else [[64], [128]])


@pytest.fixture
def retained(evidence, monkeypatch):
    add_run(evidence, "complete", profile=True)
    add_run(evidence, "oom", status="oom", batch=256)
    for name in report.PROJECT_FILES:
        if not (evidence.root / name).exists(): write(evidence.root, name, "support " + name)
    for name in ("results.md", "usage.md", "test-results.txt"):
        write(evidence.root, report.DOCS + "/" + name, "reviewed evidence")
    for name in ("complete.log", "oom.log", "final-gpu.log"):
        write(evidence.root, report.RUNTIME + "/" + name, "retained log")
    checkpoint = {"sha256": "a" * 64, "uri": "gs://fast-chunks/fixture"}
    write(evidence.root, report.ARTIFACT, {"checkpoint": checkpoint})
    write(evidence.root, report.RECEIPT, {"checkpoint": checkpoint})
    monkeypatch.setattr(report, "checkpoint_reference", lambda _receipt, checkpoint: checkpoint)
    evidence.summary = report.summarize(["complete", "oom"], evidence.second, root=evidence.root)
    write(evidence.root, report.DOCS + "/summary.json", evidence.summary)
    return evidence


def test_retention_contains_only_selected_evidence(retained):
    root = retained.root
    write(root, report.RUNTIME + "/complete/weights.safetensors", "unselected")
    write(root, report.RUNTIME + "/complete/.env", "unselected")
    summary, checkpoint, members = report.collect_evidence(root)
    assert summary["physical_optimizer_updates"] == 12
    assert "runtime/complete/full-step-trace.json.gz" in members
    assert "runtime/oom/report.json" in members
    assert "runtime/complete/dependency-snapshot/flash_attn/layers/rotary.py" in members
    assert not any(name.endswith((".env", "weights.safetensors")) for name in members)


def test_retention_supports_actual_run_log_subdirectory(retained):
    runtime = retained.root / report.RUNTIME
    (runtime / "logs").mkdir()
    for name in ("complete.log", "oom.log"):
        (runtime / name).rename(runtime / "logs" / name)
    _, _, members = report.collect_evidence(retained.root)
    assert members["logs/complete.log"] == runtime / "logs/complete.log"
    assert members["logs/oom.log"] == runtime / "logs/oom.log"


def test_retention_rejects_ambiguous_run_log_locations(retained):
    write(retained.root, report.RUNTIME + "/logs/complete.log", "conflicting duplicate log")
    with pytest.raises(ValueError, match="exactly one retained run log"):
        report.collect_evidence(retained.root)


@pytest.mark.parametrize("damage", ["summary", "report", "trace", "log", "symlink", "checkpoint"])
def test_retention_rejects_changed_evidence(retained, damage):
    root = retained.root
    if damage == "summary":
        retained.summary["physical_optimizer_updates"] += 1
        write(root, report.DOCS + "/summary.json", retained.summary)
    elif damage == "report":
        path = root / report.RUNTIME / "complete/report.json"
        raw = json.loads(path.read_text()); raw["wandb"]["run_url"] = "changed"; write(path.parent, path.name, raw)
    elif damage == "trace": write(root, report.RUNTIME + "/complete/full-step-trace.json.gz", b"changed")
    elif damage == "log": (root / report.RUNTIME / "oom.log").unlink()
    elif damage == "symlink": (root / report.DOCS / "extra.md").symlink_to(root / report.DOCS / "results.md")
    elif damage == "checkpoint": write(root, report.ARTIFACT, {"checkpoint": {"sha256": "b" * 64}})
    with pytest.raises(ValueError): report.collect_evidence(root)


@pytest.mark.parametrize("names,overrides", [([], {}), (["x", "x"], {}), (["x"], {"y": "abc1234"}), (["../x"], {})])
def test_selection_is_explicit_unique_and_local(evidence, names, overrides):
    with pytest.raises(ValueError): report.summarize(names, evidence.second, root=evidence.root, overrides=overrides)


@pytest.mark.parametrize("corruption", [None, "download", "server_md5", "server_sha"])
def test_upload_checks_create_only_generation_and_downloaded_content(tmp_path, corruption):
    path = write(tmp_path, "evidence.tar.gz", b"retained archive")
    expected = report.file_digest(path)
    class Blob:
        name = "evidence/evidence.tar.gz"
        generation = 42
        size = expected["size_bytes"]
        md5_hash = expected["md5_base64"]
        metadata = None
        def upload_from_filename(self, filename, *, if_generation_match, checksum):
            assert Path(filename) == path and if_generation_match == 0 and checksum == "md5"
        def reload(self):
            if corruption == "server_md5": self.md5_hash = "corrupted"
            if corruption == "server_sha": self.metadata["sha256"] = "b" * 64
        def download_as_bytes(self, *, if_generation_match):
            assert if_generation_match == self.generation
            return b"corrupted" if corruption == "download" else path.read_bytes()
    class Bucket:
        def blob(self, name):
            assert name == Blob.name
            return Blob()
    if corruption:
        with pytest.raises(ValueError): report.upload_verified(Bucket(), path, "evidence")
    else:
        result = report.upload_verified(Bucket(), path, "evidence")
        assert result["sha256"] == expected["sha256"] and result["generation"] == "42"
