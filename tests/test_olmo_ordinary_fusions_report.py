"""CPU evidence fixtures exercise exact commits, failures and bounded summaries."""
from copy import deepcopy
import gzip
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import olmo_ordinary_fusions_report as report


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-c", "user.name=Evidence test", "-c",
        "user.email=evidence@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=root, stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    for name in report.ESSENTIAL_SOURCES:
        write(root, name, "initial " + name)
    write(root, report.PROTOCOL, "fixed protocol")
    git(root, "add", ".")
    git(root, "commit", "-m", "first")
    first = git(root, "rev-parse", "HEAD")
    write(root, "cdrm/pretrained/olmo_ordinary.py", "repaired implementation")
    git(root, "add", ".")
    git(root, "commit", "-m", "repair")
    return SimpleNamespace(root=root, first=first, second=git(root, "rev-parse", "HEAD"))


def tensor(tag, shape, dtype="torch.bool"):
    return {"shape": shape, "dtype": dtype, "sha256": hashlib.sha256(tag.encode()).hexdigest()}


def batch(batch_size, update):
    shape = [batch_size, 512]
    result = {key: tensor(key + str(update if key == "input_ids" else 0), shape,
        "torch.int64" if key in {"input_ids", "document_ids"} else "torch.bool") for key in report.BATCH_FIELDS}
    result["valid_mask"]["sha256"] = hashlib.sha256(bytes([1]) * (batch_size * 512)).hexdigest()
    result["ce_mask"] = deepcopy(result["valid_mask"])
    return result


def add_run(evidence, name, *, revision=None, stage="capacity", arm="control", status="passed", profile=False,
            optimizer_probe=False):
    root, revision = evidence.root, revision or evidence.second
    directory = root / report.RUNTIME / name
    hashes = {}
    for source in report.ESSENTIAL_SOURCES:
        path = write(directory, "source-snapshot/" + source, git(root, "show", revision + ":" + source))
        hashes[source] = report.digest(path)
    protocol = write(directory, "protocol.md", git(root, "show", revision + ":" + report.PROTOCOL))
    size = 32 if stage == "capacity" else 8
    rope, fused = report.ARMS[arm]
    config = {"stage": stage, "arm": arm, "reference_arm": "control", "batch_size": size,
        "length": 512, "precision": "bf16_mixed", "parameter_optimizer_dtype": "float32",
        "supervision": "all_valid_ce", "ce_chunk_size": 2048, "kl_chunk_size": 128,
        "reuse_rope": True, "active_rt_layer_count": 0, "fbt": False, "nextlat": False, "profile": profile,
        "ordinary_attention": "sdpa", "ordinary_pointwise": "compiled", "ordinary_checkpointing": "all",
        "ordinary_rope_backend": rope, "fused_adam": fused, "optimizer_probe": optimizer_probe}
    flags = [{"fused": fused, "foreach": False, "capturable": False}]
    inventory = {"registered_unique": 104, "trainable": 100, "executed_declared": 100,
        "deployable_inference_declared": 100, "gradient_participating": 100,
        "optimizer_owned": 100 if stage == "capacity" else None}
    raw = {"schema": report.SCHEMA, "status": status, "stage": "complete",
        "finished_utc": "2026-09-24T00:00:00Z", "runtime_commit": revision,
        "configuration": config, "source_hashes": hashes, "protocol_sha256": report.digest(protocol),
        "physical_optimizer_updates": (8 + int(profile)) if stage == "capacity" else (6 + 6 * optimizer_probe),
        "checks": [{"name": key, "passed": True} for key in sorted(report.EXPECTED_CHECKS[stage])],
        "checkpoint": {"sha256": "a" * 64, "uri": "gs://fast-chunks/fixture"},
        "runtime": {"gpu": "H100 fixture"}, "determinism": {"deterministic_algorithms": True},
        "dependencies": {"packages": {"flash-attn": "fixture"}, "dao_sources": {}},
        "initial_batch": batch(size, 0), "input_tokens": size * 512,
        "counts": {"ce": size * 511, "latent": 0, "kl": 0}, "parameters": deepcopy(inventory),
        "prepared_layout": {"all_tokens_valid": True, "reuse_rope": True,
            "ordinary_attention_backend": "sdpa", "ordinary_pointwise_backend": "compiled",
            "ordinary_rope_backend": rope, "ordinary_checkpoint_layers": None},
        "resources": {"analytic_matrix_work": {"components": [
                {"name": "ordinary_dense_forward", "minimum": 1000, "maximum": 1100}],
            "matrix_flops_minimum": 1000, "matrix_flops_maximum": 1100,
            "input_tokens_per_update": size * 512, "pass_token_work_per_update": size * 512,
            "ordinary_block_calls_per_microbatch": 16, "rt_block_calls_per_microbatch": 0,
            "parameter_counts": {"training_architecture": 100, "deployable_inference": 100}},
            "observed_parameters": inventory, "checkpointed_ordinary_layer_count": 16,
            "loss_work": {"ce_targets": size * 511}},
        "compiler_observations": {"stats": {"unique_graphs": 2}},
        "compiler_configuration": {"fullgraph": True, "suppress_errors": False, "fail_on_recompile_limit_hit": True},
        "wandb": {"run_url": "https://wandb.ai/taylorbollman/test/runs/fixture"}}
    if stage == "correctness":
        next(c for c in raw["checks"] if c["name"] == "complete_adamw_update_parity")["optimizer_flags"] = flags
        raw["comparison_batches"] = {str(i): batch(size, i) for i in (1, 2, 5, 6, 7)}
        numeric = next(c for c in raw["checks"] if c["name"] == "same_state_candidate_vs_reference")
        numeric.update(finite=True, ownership_matches=True, counts_equal=True,
            **{key: {"example": {"relative_l2": 0., "max_relative": 0.}} for key in ("losses", "outputs", "gradients")})
    if rope == "dao":
        for source in report.DAO_SOURCES:
            path = write(directory, "dependency-snapshot/flash_attn/" + source, "frozen imported " + source)
            raw["dependencies"]["dao_sources"][source] = {
                "source": "/installed/flash_attn/" + source, "sha256": report.digest(path)}
    def record(index):
        return {"update_completed": True, "counts": deepcopy(raw["counts"]), "counters": {
            "optimizer_updates": index, "microbatches": index, "input_tokens": index * size * 512,
            "documents": index * size, "ce_positions": index * size * 511, "latent_pairs": 0, "kl_triples": 0}}
    if stage == "capacity":
        timing = lambda count: {"wall_seconds": [2.] * count, "cuda_seconds": [1.9] * count,
            "median_wall_seconds": 2., "median_cuda_seconds": 1.9}
        raw.update(preparation_records=[record(i) for i in range(1, 4)], timed_records=[record(i) for i in range(4, 9)],
            preparation_batches=[batch(size, i) for i in range(3)], timed_batches=[batch(size, i) for i in range(3, 8)],
            full_update=timing(5), forward_loss_backward=timing(3), input_tokens_per_second=size * 512 / 2,
            ce_targets_per_second=size * 511 / 2, forward_loss_backward_tokens_per_second=size * 512 / 2,
            setup_memory={"peak_reserved_gib": 30.}, steady_memory={"peak_allocated_gib": 25.},
            optimizer_flags=flags, health={"passed": True})
    if profile:
        path = write(directory, "operator-trace.json.gz", gzip.compress(b'{"traceEvents":[]}'))
        raw["profile"] = {"trace_file": path.name, "trace_bytes": path.stat().st_size, "trace_sha256": report.digest(path),
            "device_event_count": 6, "device_kernels": {
                "FlashAttentionForwardSm90": {"calls": 2, "self_device_us": 60.},
                "ampere_bf16_gemm": {"calls": 3, "self_device_us": 30.},
                "uncertain_kernel": {"calls": 1, "self_device_us": 10.}}}
        path = write(directory, "full-step-trace.json.gz", gzip.compress(b'{"traceEvents":[]}'))
        raw["full_step_profile"] = {**deepcopy(raw["profile"]), "trace_file": path.name,
            "trace_bytes": path.stat().st_size, "trace_sha256": report.digest(path), "update_record": record(9)}
        raw["profile_batch"] = batch(size, 9)
        raw["post_profile_health"] = {"passed": True}
    if optimizer_probe:
        outcomes = []
        for selected in (False, True):
            outcomes.append({"fused": selected, "moments_fp32": True, "state_keys_complete": True,
                "records": [{"optimizer_update": i} for i in range(1, 4)], "step_values": {"weight": 3},
                "optimizer_groups": [{"fused": True if selected else None, "foreach": False, "capturable": False}],
                "ownership": {"trainable": 100}, "scheduler": {"last_epoch": 3}, "counters": {"optimizer_updates": 3}})
        raw["checks"].append({"name": report.OPTIMIZER_PROBE, "passed": True,
            "semantic_state_matches": True, "initial_state_restored": True,
            "optimizer_groups_match_except_fused": True,
            "physical_optimizer_updates": 6, "updates_per_arm": 3, "arms": outcomes})
        raw["optimizer_probe_state_restored"] = True
        raw["optimizer_probe_progress"] = [{"fused": selected, "optimizer_update": i}
            for selected in (False, True) for i in range(1, 4)]
    if status != "passed":
        raw.update(stage="load", checks=[], physical_optimizer_updates=0,
            error={"type": "RuntimeError", "message": "retained failed attempt"})
    path = write(directory, "report.json", raw)
    return path, raw


def test_sources_and_arm_names_match_the_runtime_harness():
    from scripts import olmo_ordinary_fusions as harness
    assert report.ESSENTIAL_SOURCES <= set(harness.SOURCES)
    assert report.ARMS == harness.ARMS


def test_selection_retains_failures_and_exact_prior_revision(evidence):
    add_run(evidence, "failure", revision=evidence.first, status="failed")
    add_run(evidence, "base")
    add_run(evidence, "candidate", arm="dao-rope")
    result = report.summarize(["failure", "base", "candidate"], evidence.second,
        overrides={"failure": evidence.first}, root=evidence.root)
    assert result["statuses"] == {"failed": 1, "passed": 2}
    assert result["physical_optimizer_updates"] == 16
    assert result["runs"][0]["runtime_commit"] == evidence.first
    assert "cdrm/pretrained/olmo_ordinary.py" in result["runs"][0]["current_source_differences"]
    assert result["comparison_groups"][0]["gain_fraction_vs_control"] == {"dao-rope": 0.}
    assert result["capacity"][1]["setup_memory"]["peak_reserved_gib"] == 30.


@pytest.mark.parametrize("arm", ["dao-rope", "fused-adam", "dao-rope-fused-adam"])
def test_fusion_arms_share_only_the_expected_changed_options(evidence, arm):
    add_run(evidence, "control")
    add_run(evidence, "combined", arm=arm)
    result = report.summarize(["control", "combined"], evidence.second, root=evidence.root)
    assert result["statuses"] == {"passed": 2}
    assert result["capacity"][1]["arm"] == arm
    assert result["resource_cards"][1]["checkpointed_ordinary_layer_count"] == 16
    assert result["comparison_groups"][0]["gain_fraction_vs_control"] == {arm: 0.}


@pytest.mark.parametrize("damage", ["source", "protocol", "commit", "running", "gate", "updates", "median",
    "throughput", "full_ce", "resources", "optimizer", "compiled", "layout"])
def test_invalid_or_inconsistent_completed_evidence_is_rejected(evidence, damage):
    path, raw = add_run(evidence, "candidate", arm="dao-rope")
    if damage == "source":
        write(path.parent, "source-snapshot/cdrm/pretrained/olmo_ordinary.py", "changed snapshot")
    elif damage == "protocol":
        write(path.parent, "protocol.md", "changed protocol")
    elif damage == "commit":
        raw["runtime_commit"] = evidence.first
    elif damage == "running":
        raw["status"] = "running"
    elif damage == "gate":
        raw["checks"].pop()
    elif damage == "updates":
        raw["physical_optimizer_updates"] = 7
    elif damage == "median":
        raw["full_update"]["median_wall_seconds"] = 3
    elif damage == "throughput":
        raw["input_tokens_per_second"] *= 2
    elif damage == "full_ce":
        raw["initial_batch"]["ce_mask"]["sha256"] = "b" * 64
    elif damage == "resources":
        raw["resources"]["analytic_matrix_work"]["matrix_flops_minimum"] += 1
    elif damage == "optimizer":
        raw["resources"]["observed_parameters"]["optimizer_owned"] -= 1
    elif damage == "compiled":
        raw["compiler_observations"]["graph_break"] = {"fallback": 1}
    elif damage == "layout":
        raw["prepared_layout"]["ordinary_pointwise_backend"] = "eager"
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        report.summarize(["candidate"], evidence.second, root=evidence.root)


def test_correctness_and_dependency_snapshot_are_verified(evidence):
    path, _ = add_run(evidence, "dao", stage="correctness", arm="dao-rope")
    result = report.summarize(["dao"], evidence.second, root=evidence.root)
    assert result["runs"][0]["dependency_verification"]["files_checked"] == 2
    assert result["runs"][0]["gate_groups"]["compatibility"]["total"] == 1
    write(path.parent, "dependency-snapshot/flash_attn/layers/rotary.py", "different imported bytes")
    with pytest.raises(ValueError, match="dependency snapshot"):
        report.summarize(["dao"], evidence.second, root=evidence.root)


def test_failed_compatibility_can_retain_completed_operational_checks(evidence):
    path, raw = add_run(evidence, "qualified", stage="correctness", arm="dao-rope")
    for check in raw["checks"]:
        if check["name"] == "same_state_candidate_vs_reference":
            check["passed"] = False
    raw.update(status="failed", error={"type": "AssertionError", "message": "retained CE screen miss"},
        compatibility_miss_continued=True, numerical_compatibility_passed=False, operational_checks_passed=True)
    raw["configuration"]["continue_after_compatibility_miss"] = True
    write(path.parent, "report.json", raw)
    result = report.summarize(["qualified"], evidence.second, root=evidence.root)
    row = result["runs"][0]
    assert row["status"] == "failed" and result["physical_optimizer_updates"] == 6
    assert row["gate_groups"]["operational"]["complete_and_passed"]
    assert not row["gate_groups"]["compatibility"]["complete_and_passed"]
    assert row["compatibility_miss_continued"] and not row["numerical_compatibility_passed"]


@pytest.mark.parametrize("damage", ["updates", "full_ce", "resources", "layout", "compiler", "gate",
                                     "flag", "operational", "numeric", "finite"])
def test_completed_qualified_failures_must_still_validate_full_operational_evidence(evidence, damage):
    path, raw = add_run(evidence, "qualified", stage="correctness", arm="dao-rope")
    numeric = next(c for c in raw["checks"] if c["name"] == "same_state_candidate_vs_reference")
    numeric["passed"] = False
    raw.update(status="failed", error={"type": "AssertionError", "message": "retained numeric miss"},
        compatibility_miss_continued=True, numerical_compatibility_passed=False, operational_checks_passed=True)
    raw["configuration"]["continue_after_compatibility_miss"] = True
    if damage == "updates":
        raw["physical_optimizer_updates"] = 5
    elif damage == "full_ce":
        raw["comparison_batches"]["5"]["ce_mask"]["sha256"] = "c" * 64
    elif damage == "resources":
        raw["resources"]["analytic_matrix_work"]["matrix_flops_minimum"] += 1
    elif damage == "layout":
        raw["prepared_layout"]["ordinary_pointwise_backend"] = "eager"
    elif damage == "compiler":
        raw["compiler_configuration"]["fail_on_recompile_limit_hit"] = False
    elif damage == "gate":
        raw["checks"] = [c for c in raw["checks"] if c["name"] != "candidate_changed_weights"]
    elif damage == "flag":
        raw["compatibility_miss_continued"] = False
    elif damage == "operational":
        raw["operational_checks_passed"] = False
    elif damage == "numeric":
        raw["numerical_compatibility_passed"] = True
    elif damage == "finite":
        numeric["finite"] = False
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        report.summarize(["qualified"], evidence.second, root=evidence.root)


def test_profile_hash_and_conservative_device_categories(evidence):
    path, _ = add_run(evidence, "profile", profile=True)
    result = report.summarize(["profile"], evidence.second, root=evidence.root)
    profile = result["profiles"][0]
    categories = profile["kernel_name_categories"]
    assert categories["attention_named"]["share_of_summed_device_time"] == .6
    assert categories["matrix_multiply_named"]["calls"] == 3
    assert categories["other_or_unclassified"]["self_device_us"] == 10
    assert result["runs"][0]["profile"]["trace_file"] == "operator-trace.json.gz"
    write(path.parent, "operator-trace.json.gz", gzip.compress(b'changed trace'))
    with pytest.raises(ValueError, match="trace bytes"):
        report.summarize(["profile"], evidence.second, root=evidence.root)


@pytest.mark.parametrize("damage", [None, "count", "profile_fixture", "profile_counter", "second_trace", "health"])
def test_full_step_profile_has_separate_update_and_verified_trace(evidence, damage):
    path, raw = add_run(evidence, "profile", profile=True)
    if damage == "count":
        raw["physical_optimizer_updates"] = 8
    elif damage == "profile_fixture":
        raw.pop("profile_batch")
    elif damage == "profile_counter":
        raw["full_step_profile"]["update_record"]["counters"]["optimizer_updates"] = 8
    elif damage == "second_trace":
        write(path.parent, "full-step-trace.json.gz", gzip.compress(b'changed full step'))
    elif damage == "health":
        raw["post_profile_health"]["passed"] = False
    write(path.parent, "report.json", raw)
    if damage:
        with pytest.raises(ValueError):
            report.summarize(["profile"], evidence.second, root=evidence.root)
    else:
        summary = report.summarize(["profile"], evidence.second, root=evidence.root)
        assert summary["physical_optimizer_updates"] == 9 and len(summary["profiles"]) == 2
        assert summary["profiles"][1]["profile_kind"] == "full_step_profile"


@pytest.mark.parametrize("damage", [None, "count", "restore", "order", "flags", "steps", "records",
                                   "group_check", "groups"])
def test_fixed_gradient_probe_preserves_physical_updates_and_semantics(evidence, damage):
    path, raw = add_run(evidence, "optimizer", stage="correctness", arm="fused-adam", optimizer_probe=True)
    check = next(c for c in raw["checks"] if c["name"] == report.OPTIMIZER_PROBE)
    if damage == "count":
        raw["physical_optimizer_updates"] = 6
    elif damage == "restore":
        raw["optimizer_probe_state_restored"] = False
    elif damage == "order":
        raw["optimizer_probe_progress"][0]["fused"] = True
    elif damage == "flags":
        check["arms"][1]["optimizer_groups"][0]["fused"] = None
    elif damage == "steps":
        check["arms"][1]["step_values"]["weight"] = 2
    elif damage == "records":
        check["arms"][1]["records"][0]["optimizer_update"] = 2
    elif damage == "group_check":
        check["optimizer_groups_match_except_fused"] = False
    elif damage == "groups":
        check["arms"][1]["optimizer_groups"][0]["eps"] = 1e-7
    write(path.parent, "report.json", raw)
    if damage:
        with pytest.raises(ValueError):
            report.summarize(["optimizer"], evidence.second, root=evidence.root)
    else:
        summary = report.summarize(["optimizer"], evidence.second, root=evidence.root)
        assert summary["physical_optimizer_updates"] == 12


def test_profile_reductions_use_correctly_rounded_summation(evidence):
    path, raw = add_run(evidence, "profile", profile=True)
    raw["profile"]["device_kernels"] = {str(i): {"calls": 1, "self_device_us": value}
        for i, value in enumerate([1e16, 1., 1.])}
    raw["profile"]["device_event_count"] = 3
    write(path.parent, "report.json", raw)
    summary = report.summarize(["profile"], evidence.second, root=evidence.root)
    assert summary["profiles"][0]["summed_device_ms"] == (1e16 + 2.) / 1000


def test_gpu_annotations_do_not_doublecount_kernels_and_cpu_scopes_come_from_trace(evidence):
    path, raw = add_run(evidence, "profile", profile=True)
    profile = raw["full_step_profile"]
    profile["device_kernels"].update({
        "ordinary_step/forward_loss_backward": {"calls": 1, "self_device_us": 100.},
        "Optimizer.step#AdamW.step": {"calls": 1, "self_device_us": 30.}})
    profile["device_event_count"] += 2
    profile["cpu_phase_scopes"] = {"ordinary_step/forward_loss_backward":
        {"calls": 1, "cpu_us": 0., "device_us": 100.}}
    events = [
        {"ph": "X", "cat": "user_annotation", "name": "ordinary_step/complete", "dur": 150.},
        {"ph": "X", "cat": "user_annotation", "name": "ordinary_step/forward_loss_backward", "dur": 50.},
        {"ph": "X", "cat": "user_annotation", "name": "ordinary_step/optimizer", "dur": 20.},
        {"ph": "X", "cat": "gpu_user_annotation", "name": "ordinary_step/forward_loss_backward", "dur": 100.},
        {"ph": "X", "cat": "gpu_user_annotation", "name": "Optimizer.step#AdamW.step", "dur": 30.},
    ]
    trace = write(path.parent, "full-step-trace.json.gz", gzip.compress(json.dumps({"traceEvents": events}).encode()))
    profile.update(trace_bytes=trace.stat().st_size, trace_sha256=report.digest(trace))
    write(path.parent, "report.json", raw)
    summary = report.summarize(["profile"], evidence.second, root=evidence.root)
    derived = summary["profiles"][1]
    assert derived["raw_device_event_count"] == 8 and derived["device_event_count"] == 6
    assert derived["raw_summed_device_ms"] == .23 and derived["summed_device_ms"] == .1
    assert derived["excluded_gpu_annotations"]["summed_device_ms"] == .13
    assert set(derived["excluded_gpu_annotations"]["events"]) == {
        "ordinary_step/forward_loss_backward", "Optimizer.step#AdamW.step"}
    assert derived["kernel_name_categories"]["attention_named"]["share_of_summed_device_time"] == .6
    assert all(not row["name"].startswith(("ordinary_step/", "Optimizer.step#")) for row in derived["top_device_events"])
    assert derived["cpu_phase_scopes"]["ordinary_step/forward_loss_backward"] == {"calls": 1, "cpu_us": 50.}
    assert derived["raw_cpu_phase_scopes"]["ordinary_step/forward_loss_backward"]["cpu_us"] == 0.
    assert json.loads(path.read_text()) == raw  # The source report remains immutable.


def test_plot_generated_from_verified_matching_cohort(evidence, tmp_path):
    add_run(evidence, "control")
    add_run(evidence, "candidate", arm="fused-adam")
    summary = report.summarize(["control", "candidate"], evidence.second, root=evidence.root)
    report.render_plot(summary, tmp_path / "plots")
    assert (tmp_path / "plots/throughput.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "plots/throughput.png").read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("name", ["triton_poi_fused_mul_silu_silu_backward_0",
                                 "triton_red_fused_silu_backward_sum_1"])
def test_compiled_fused_kernels_are_not_attributed_to_standalone_silu(name):
    assert report.kernel_category(name) == "compiled_pointwise_named"
    assert report.kernel_category("at::native::silu_backward_kernel") == "silu_named"


def test_mismatched_fixture_cohorts_do_not_produce_paired_gains(evidence):
    add_run(evidence, "base")
    path, raw = add_run(evidence, "candidate", arm="fused-adam")
    raw["timed_batches"][1]["input_ids"]["sha256"] = "f" * 64
    write(path.parent, "report.json", raw)
    result = report.summarize(["base", "candidate"], evidence.second, root=evidence.root)
    assert len(result["comparison_groups"]) == 2
    assert all(not group["gain_fraction_vs_control"] for group in result["comparison_groups"])
    assert len({group["runtime_source_fingerprint"] for group in result["comparison_groups"]}) == 1


def test_changed_model_sources_do_not_produce_paired_gains(evidence):
    add_run(evidence, "base", revision=evidence.first)
    add_run(evidence, "new-base", revision=evidence.second)
    add_run(evidence, "candidate", revision=evidence.second, arm="fused-adam")
    result = report.summarize(["base", "new-base", "candidate"], evidence.second, root=evidence.root,
                             overrides={"base": evidence.first})
    groups = result["comparison_groups"]
    assert len(groups) == 2 and len({group["runtime_source_fingerprint"] for group in groups}) == 2
    old = next(group for group in groups if group["runtime_commits"] == [evidence.first])
    current = next(group for group in groups if group["runtime_commits"] == [evidence.second])
    assert old["runs"] == ["base"] and old["gain_fraction_vs_control"] == {}
    assert current["runs"] == ["new-base", "candidate"]
    assert current["gain_fraction_vs_control"] == {"fused-adam": 0.}


def test_changed_harness_sources_form_separate_cohorts(evidence):
    write(evidence.root, report.HARNESS_SOURCE, "diagnostic continuation only")
    git(evidence.root, "add", report.HARNESS_SOURCE)
    git(evidence.root, "commit", "-m", "diagnostic report extension")
    third = git(evidence.root, "rev-parse", "HEAD")
    add_run(evidence, "base", revision=evidence.second)
    add_run(evidence, "candidate", revision=third, arm="dao-rope")
    result = report.summarize(["base", "candidate"], third, root=evidence.root, overrides={"base": evidence.second})
    assert len(result["comparison_groups"]) == 2
    assert all(not group["gain_fraction_vs_control"] for group in result["comparison_groups"])


@pytest.mark.parametrize("names,overrides", [([], {}), (["x", "x"], {}), (["x"], {"y": "abc1234"}), (["../x"], {})])
def test_selection_is_explicit_unique_and_local(evidence, names, overrides):
    with pytest.raises(ValueError):
        report.summarize(names, evidence.second, root=evidence.root, overrides=overrides)
