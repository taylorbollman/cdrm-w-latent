"""Integration evidence contracts: exact cohorts, failed attempts and retention.

All repositories, reports and hashes are synthetic CPU fixtures. No GPU, model
allocation, credential access or cloud calls are needed.
"""
from copy import deepcopy
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import olmo_rt_author_integration_report as report
from scripts import olmo_rt_author_integration_retain as retain


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if isinstance(data, (dict, list)) else data)
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-c", "user.name=Evidence test",
        "-c", "user.email=evidence@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=root, stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    for path in set(retain.PROJECT_FILES) | report.ESSENTIAL_SOURCES:
        write(root, path, "initial " + path)
    write(root, report.PROTOCOL, "frozen protocol")
    write(root, report.DOCS + "/test-results.txt", "CPU tests passed")
    write(root, report.RUNTIME + "/final-gpu.log", "GPU idle")
    checkpoint = {"sha256": "a" * 64, "uri": "gs://fast-chunks/native", "generation": "42"}
    write(root, retain.ARTIFACT, {"checkpoint": checkpoint})
    write(root, retain.RECEIPT, {"checkpoint": checkpoint})
    monkeypatch.setattr(retain, "checkpoint_reference", lambda receipt, checkpoint: checkpoint)
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")
    first = git(root, "rev-parse", "HEAD")
    write(root, "cdrm/pretrained/olmo_tiled.py", "repaired integration")
    write(root, report.PROTOCOL, "repaired frozen protocol")
    git(root, "add", ".")
    git(root, "commit", "-m", "repaired")
    return SimpleNamespace(root=root, first=first, second=git(root, "rev-parse", "HEAD"), checkpoint=checkpoint)


def tensor_record(tag, shape, dtype="torch.float32"):
    return {"shape": shape, "dtype": dtype, "sha256": hashlib.sha256(tag.encode()).hexdigest()}


def batch_record(batch_size, update):
    shape = [batch_size, 512]
    result = {name: tensor_record(name + str(update if name == "input_ids" else 0), shape,
        "torch.int64" if name in ("input_ids", "document_ids") else "torch.bool")
        for name in ("input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask")}
    result["valid_mask"]["sha256"] = hashlib.sha256(bytes([1]) * (batch_size * 512)).hexdigest()
    result["ce_mask"] = deepcopy(result["valid_mask"])
    return result


def resource_record(case, batch_size, backend, stage):
    # Simple self-consistent records isolate evidence validation from the
    # separately tested model arithmetic estimator.
    active, resident, deployable = (100, 104, 100) if case == "rt" else (120, 120, 104)
    components = [{"name": "ordinary_dense_forward", "minimum": 1000, "maximum": 1200}]
    ledger = {"forward": {"dense": 10, "attention": 2}, "backward": {"dense": 20, "attention": 4}}
    if backend == "author":
        for phase in ("forward", "backward"):
            for kind in ("dense", "attention"):
                value = 2 * ledger[phase][kind]
                components.append({"name": f"author_rt_{kind}_{phase}", "minimum": value, "maximum": value})
    return {"analytic_matrix_work": {
        "components": components,
        "matrix_flops_minimum": sum(c["minimum"] for c in components),
        "matrix_flops_maximum": sum(c["maximum"] for c in components),
        "parameter_counts": {"training_architecture": active, "deployable_inference": deployable},
        "rt_implementation": backend,
        "rt_backward_memory": "materialized" if backend == "author" else "recompute",
        "rt_block_calls_per_microbatch": 2,
        "ordinary_block_calls_per_microbatch": 30 if case == "combined" else 14,
        "rt_component_replacement": {"calls": 2, "per_call_author_ledger": {"matrix_breakdown": ledger}},
        "input_tokens_per_update": batch_size * 512,
        "pass_token_work_per_update": batch_size * 512 * (2 if case == "combined" else 1)},
        "observed_parameters": {"registered_unique": resident, "trainable": active,
            "gradient_participating": active, "optimizer_owned": active if stage == "capacity" else None,
            "executed_declared": active, "deployable_inference_declared": deployable,
            "resident_parameter_bytes": resident * 4},
        "loss_work": {"ce_targets": batch_size * 511, "latent_pairs": batch_size * 511 if case == "combined" else 0,
            "kl_triples": batch_size * 256 if case == "combined" else 0,
            "predictor_positions": batch_size * 511 if case == "combined" else 0}}


def add_run(evidence, name, revision, *, status="passed", stage="verify", case="rt", backend="author",
            compatibility=True, preload=False):
    root = evidence.root
    directory = root / report.RUNTIME / name
    sources = {}
    for source in sorted(report.ESSENTIAL_SOURCES):
        snapshot = write(directory, "source-snapshot/" + source, git(root, "show", revision + ":" + source))
        sources[source] = report.digest(snapshot)
    protocol = write(directory, "protocol.md", git(root, "show", revision + ":" + report.PROTOCOL))
    batch_size = 32 if stage == "capacity" else 8
    checks = [{"name": name, "passed": True, "gate": True} for name in sorted(report.OPERATIONAL[stage])]
    if stage == "verify":
        checks.append({"name": "author_vs_native_bf16_mixed", "gate": True, "passed": compatibility})
    config = {"stage": stage, "case": case, "backend": backend, "batch_size": batch_size,
        "physical_batch": batch_size, "length": 512, "precision": "bf16_mixed", "supervision": "full",
        "selected_rt_layers": [0, 15], "ce_chunk_size": 2048, "kl_chunk_size": 128,
        "seed": 20260922, "accumulation": 1, "world_size": 1, "tf32": False,
        "capture_warmup_backwards": 10, "preparation_updates": 3 if stage == "capacity" else 0,
        "timed_updates": 5 if stage == "capacity" else 0, "graph_timing_samples": 3 if stage == "capacity" else 0}
    raw = {"schema": report.SCHEMA, "status": status, "stage": "load" if preload else "complete",
        "finished_utc": "2026-09-23T00:00:00Z", "runtime_commit": revision, "configuration": config,
        "checks": [] if preload else checks, "physical_optimizer_updates": 0 if preload else 6 if stage == "verify" else 8,
        "source_hashes": sources, "protocol_sha256": report.digest(protocol),
        "runtime": {"torch": "fixture", "cuda": "fixture", "gpu": "H100 fixture"},
        "determinism": {"deterministic_algorithms": True}}
    if not preload:
        raw["checkpoint"] = evidence.checkpoint
        raw["initial_batch"] = batch_record(batch_size, 0)
        raw["initial_auxiliary_parameters"] = {"backbone.fusion.state_proj.weight": tensor_record("state", [2048, 2048]),
            "backbone.fusion.token_gate.weight": tensor_record("gate", [2048, 2048])}
        if case == "combined":
            raw["initial_auxiliary_parameters"]["predictor.mlp.0.weight"] = tensor_record("predictor", [128, 4096])
        raw["counts"] = {"ce": batch_size * 511, "latent": batch_size * 511 if case == "combined" else 0,
                         "kl": batch_size * 256 if case == "combined" else 0}
        raw["input_tokens"] = batch_size * 512
        raw["pass_input_token_work"] = batch_size * 512 * (2 if case == "combined" else 1)
        raw["parameter_identity_preserved"] = True
        raw["resources"] = resource_record(case, batch_size, backend, stage)
        inventory = raw["resources"]["observed_parameters"]
        raw["parameters"] = {"active": inventory["trainable"], "resident": inventory["registered_unique"],
                             "owned_parameter_tensors": 8}
        raw["prepared_layout"] = {"all_tokens_valid": True, "rt_implementation": backend,
            "reuse_rope": True, "valid_mask_sha256": raw["initial_batch"]["valid_mask"]["sha256"]}
        raw["nextlat_config"] = {"model_dim": 2048, "ce_chunk_size": 2048, "vocab_chunk_size": 128}
        raw["fixture_order"] = {"comparison_and_initial_capture": 0,
            "changed_input": 1 if stage == "verify" else 3,
            "parity_per_arm": [5, 6, 7] if stage == "verify" else [],
            "preparation": list(range(3)) if stage == "capacity" else [],
            "timed": list(range(3, 8)) if stage == "capacity" else []}
        compiler = {"required": backend == "author", "fail_on_recompile_limit_hit": True,
            "counters": {"stats": {"unique_graphs": 5 if backend == "author" else 0}}}
        raw["compiler_final"] = deepcopy(compiler)
        raw["compiler_after_warmup"] = deepcopy(compiler)
    if status != "passed":
        raw["error"] = {"type": "RuntimeError", "message": "explicit retained failure"}
    if stage == "capacity" and not preload:
        timing = lambda count: {"wall_seconds": [2.] * count, "cuda_seconds": [1.9] * count,
            "median_wall_seconds": 2., "median_cuda_seconds": 1.9}
        raw.update(full_update=timing(5), forward_loss_backward=timing(3),
            preparation_records=[{}] * 3, timed_records=[{}] * 5,
            preparation_batches=[batch_record(batch_size, i) for i in range(3)],
            timed_batches=[batch_record(batch_size, i) for i in range(3, 8)],
            input_tokens_per_second=batch_size * 512 / 2,
            ce_targets_per_second=batch_size * 511 / 2,
            forward_loss_backward_tokens_per_second=batch_size * 512 / 2)
    path = write(directory, "report.json", raw)
    write(root, report.RUNTIME + "/" + name + ".log", "selected run")
    return path, raw


def select(evidence, names, **kwargs):
    summary = report.summarize(names, evidence.second, root=evidence.root, **kwargs)
    write(evidence.root, report.DOCS + "/summary.json", summary)
    return summary


def test_harness_snapshots_every_source_required_by_report_validation():
    from scripts import olmo_rt_author_integration as harness
    assert report.ESSENTIAL_SOURCES <= set(harness.SOURCES)


def test_failed_load_and_compatibility_screen_remain_distinct_from_operational_success(evidence):
    add_run(evidence, "load-failed", evidence.first, status="failed", preload=True)
    add_run(evidence, "compatibility-failed", evidence.second, status="failed", compatibility=False)
    add_run(evidence, "passed", evidence.second)
    summary = select(evidence, ["load-failed", "compatibility-failed", "passed"],
                     overrides={"load-failed": evidence.first})
    assert summary["statuses"] == {"failed": 2, "passed": 1}
    assert summary["physical_optimizer_updates"] == 12
    first, qualified, passed = summary["runs"]
    assert not first["gate_groups"]["operational"]["complete_and_passed"]
    assert qualified["gate_groups"]["operational"]["complete_and_passed"]
    assert not qualified["gate_groups"]["compatibility"]["complete_and_passed"]
    assert passed["gate_groups"]["compatibility"]["complete_and_passed"]
    _, _, members, checked = retain.collect_evidence(evidence.root)
    assert "runtime/load-failed/report.json" in members
    assert checked["run_revisions"]["load-failed"] == evidence.first
    assert checked["physical_optimizer_updates"] == 12


@pytest.mark.parametrize("damage", ["source", "protocol", "revision", "unfinished", "passed_failed_gate"])
def test_selected_source_commit_protocol_and_status_cannot_disagree(evidence, damage):
    path, raw = add_run(evidence, "candidate", evidence.second)
    if damage == "source":
        source = "cdrm/pretrained/olmo_tiled.py"
        snapshot = write(path.parent, "source-snapshot/" + source, "tampered")
        raw["source_hashes"][source] = report.digest(snapshot)
    elif damage == "protocol":
        changed = write(path.parent, "protocol.md", "tampered")
        raw["protocol_sha256"] = report.digest(changed)
    elif damage == "revision":
        raw["runtime_commit"] = evidence.first
    elif damage == "unfinished":
        raw["status"] = "running"
    else:
        raw["checks"][0]["passed"] = False
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        select(evidence, ["candidate"])


@pytest.mark.parametrize("damage", ["ce_count", "half_ce_hash", "missing_mask", "invalid_hash", "pass_work",
                                    "missing_graph_gate", "missing_raw_comparison", "identity"])
def test_passed_report_requires_full_ce_layout_counts_and_complete_gates(evidence, damage):
    path, raw = add_run(evidence, "candidate", evidence.second, case="combined")
    if damage == "ce_count":
        raw["counts"]["ce"] //= 2
    elif damage == "half_ce_hash":
        raw["initial_batch"]["ce_mask"]["sha256"] = "b" * 64
    elif damage == "missing_mask":
        del raw["initial_batch"]["latent_mask"]
    elif damage == "invalid_hash":
        raw["initial_batch"]["input_ids"]["sha256"] = "not-a-digest"
    elif damage == "pass_work":
        raw["pass_input_token_work"] = raw["input_tokens"]
    elif damage == "identity":
        raw["parameter_identity_preserved"] = False
    else:
        name = "candidate_changed_weights" if damage == "missing_graph_gate" else "author_vs_native_bf16_mixed"
        raw["checks"] = [check for check in raw["checks"] if check["name"] != name]
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        select(evidence, ["candidate"])


@pytest.mark.parametrize("damage", ["zero_graphs", "graph_break", "unimplemented", "limit_disabled", "audit_missing"])
def test_author_success_requires_actual_compilation_without_fallback(evidence, damage):
    path, raw = add_run(evidence, "candidate", evidence.second, stage="capacity")
    if damage == "zero_graphs":
        raw["compiler_final"]["counters"]["stats"]["unique_graphs"] = 0
    elif damage in ("graph_break", "unimplemented"):
        raw["compiler_final"]["counters"][damage] = {"failure": 1}
    elif damage == "limit_disabled":
        raw["compiler_final"]["fail_on_recompile_limit_hit"] = False
    else:
        del raw["compiler_after_warmup"]
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        select(evidence, ["candidate"])


@pytest.mark.parametrize("damage", ["initial_tokens", "preparation_tokens", "timed_tokens", "auxiliary", "runtime"])
def test_same_shape_capacity_arms_require_exact_batch_and_auxiliary_hashes(evidence, damage):
    add_run(evidence, "native", evidence.second, stage="capacity", case="combined", backend="native")
    path, raw = add_run(evidence, "author", evidence.second, stage="capacity", case="combined")
    if damage == "initial_tokens":
        raw["initial_batch"]["input_ids"]["sha256"] = "b" * 64
    elif damage == "preparation_tokens":
        raw["preparation_batches"][1]["input_ids"]["sha256"] = "b" * 64
    elif damage == "timed_tokens":
        raw["timed_batches"][2]["input_ids"]["sha256"] = "b" * 64
    elif damage == "auxiliary":
        raw["initial_auxiliary_parameters"]["predictor.mlp.0.weight"]["sha256"] = "b" * 64
    else:
        raw["runtime"]["torch"] = "other-runtime"
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        select(evidence, ["native", "author"])


def test_matching_native_author_capacity_selection_preserves_work_and_unique_parameters(evidence):
    add_run(evidence, "native", evidence.second, stage="capacity", backend="native")
    add_run(evidence, "author", evidence.second, stage="capacity")
    summary = select(evidence, ["native", "author"])
    assert summary["physical_optimizer_updates"] == 16
    assert len(summary["comparison_groups"]) == 1
    assert all(row["input_tokens_per_second"] == 8192 for row in summary["capacity"])
    _, _, members, checked = retain.collect_evidence(evidence.root)
    assert checked["statuses"] == {"passed": 2}
    assert "runtime/native/report.json" in members and "runtime/author/report.json" in members


@pytest.mark.parametrize("damage", ["report", "summary_tps", "summary_count", "summary_gate", "summary_resource"])
def test_retainer_recomputes_selection_and_rejects_stale_summary(evidence, damage):
    path, raw = add_run(evidence, "candidate", evidence.second, stage="capacity")
    summary = select(evidence, ["candidate"])
    if damage == "report":
        raw["extra"] = "changed bytes"
        write(path.parent, "report.json", raw)
    elif damage == "summary_tps":
        summary["capacity"][0]["input_tokens_per_second"] *= 2
    elif damage == "summary_count":
        summary["physical_optimizer_updates"] += 1
    elif damage == "summary_gate":
        summary["runs"][0]["gate_groups"]["operational"]["complete_and_passed"] = False
    else:
        summary["resource_cards"][0]["analytic_matrix_work"]["matrix_flops_minimum"] *= 2
    write(evidence.root, report.DOCS + "/summary.json", summary)
    with pytest.raises(ValueError):
        retain.collect_evidence(evidence.root)


def test_oom_before_load_has_no_throughput_or_successful_comparison_claim(evidence):
    add_run(evidence, "oom", evidence.second, status="oom", stage="capacity", preload=True)
    summary = select(evidence, ["oom"])
    assert summary["physical_optimizer_updates"] == 0
    assert summary["capacity"][0]["input_tokens_per_second"] is None
    assert not summary["comparison_groups"]
    assert report.plot_capacity(summary, evidence.root / "unused-plots") == []


def test_retention_allowlist_excludes_weights_credentials_and_other_runs(evidence):
    path, _ = add_run(evidence, "selected", evidence.second)
    add_run(evidence, "unselected", evidence.second)
    for name in ("weights.safetensors", ".env", "wandb/private.json"):
        write(path.parent, name, "excluded")
    write(evidence.root, report.DOCS + "/unapproved.json", "excluded")
    select(evidence, ["selected"])
    _, _, members, _ = retain.collect_evidence(evidence.root)
    assert not any(any(part in name for part in ("unselected", "safetensors", ".env", "wandb", "unapproved"))
                   for name in members)


@pytest.mark.parametrize("damage", ["sum", "gradient_ownership", "optimizer_ownership", "reused_parameters", "author_component"])
def test_resources_cannot_overcount_shared_weights_or_misreport_executed_work(evidence, damage):
    path, raw = add_run(evidence, "candidate", evidence.second, stage="capacity", case="combined")
    card = raw["resources"]
    if damage == "sum":
        card["analytic_matrix_work"]["matrix_flops_minimum"] += 1
    elif damage == "gradient_ownership":
        card["observed_parameters"]["gradient_participating"] -= 1
    elif damage == "optimizer_ownership":
        card["observed_parameters"]["optimizer_owned"] *= 2
    elif damage == "reused_parameters":
        card["analytic_matrix_work"]["parameter_counts"]["training_architecture"] *= 2
    else:
        card["analytic_matrix_work"]["rt_component_replacement"]["per_call_author_ledger"]["matrix_breakdown"]["forward"]["dense"] += 1
    write(path.parent, "report.json", raw)
    with pytest.raises(ValueError):
        select(evidence, ["candidate"])


def test_queue_artifacts_are_safe_top_level_files_and_logs_are_not_duplicated(evidence):
    add_run(evidence, "integration-selected", evidence.second)
    runtime = evidence.root / report.RUNTIME
    for name in ("run_integration_queue.py", "integration-stage1.json", "integration-stage1.log", "capacity-pair.json"):
        write(runtime, name, "queue evidence")
    for name in ("unrelated.json", ".env", "integration-secret.env.json", "integration-stage1.pt"):
        write(runtime, name, "excluded")
    write(runtime, "integration-directory.json/nested.json", "excluded")
    select(evidence, ["integration-selected"])
    _, _, members, checked = retain.collect_evidence(evidence.root)
    assert set(checked["queue_artifacts"]) == {"run_integration_queue.py", "integration-stage1.json",
        "integration-stage1.log", "capacity-pair.json", "integration-selected.log"}
    assert "logs/integration-selected.log" in members
    assert "queue/integration-selected.log" not in members
    assert not any("nested" in name or "unrelated" in name or ".env" in name or name.endswith(".pt") for name in members)


def test_queue_artifact_matching_symlink_is_rejected(tmp_path):
    target = write(tmp_path, "original.json", "data")
    (tmp_path / "integration-stage1.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        list(retain.queue_artifacts(tmp_path))
