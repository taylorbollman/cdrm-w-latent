"""CPU provenance, gate accounting and create-only retention regression tests."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import summarize_olmo_single_gpu_readiness as report
from scripts.olmo_tiled_retain import (
    CHECKPOINT_FILENAME, CHECKPOINT_SHA256, CHECKPOINT_SIZE,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) if isinstance(value, (dict, list)) else value)


def gradient_gate(name, *, exact=True):
    row = {"passed": True, "finite": True, "bitwise_equal": True, "relative_l2": 0.,
           "max_relative": 0., "max_absolute": 0., "delta_sq": 0., "reference_sq": 1.}
    return {"name": name, "passed": True, "ownership_matches": True, "finite": True,
            "expected_participation_matches": True, "losses_finite": True, "losses_and_counts_exact": True,
            "missing_expected_gradients": [], "unexpected_gradients": [], "exact_required": exact,
            "all_bitwise_equal": True, "reference_losses": [{"test": "same"}], "candidate_losses": [{"test": "same"}],
            "global_gradient_relative_l2": 0., "gradients": {"p": row},
            "budgets": None if exact else report.ACCUMULATION_BUDGETS}


def adapter_raw():
    counts = {"ce": 1200, "latent": 600, "kl": 500}
    locals_ = [{"ce": 700, "latent": 600, "kl": 500}, {"ce": 500, "latent": 0, "kl": 0}]
    checks = [gradient_gate("world1_adapter_exact"), gradient_gate("same_order_accumulation_exact"),
              gradient_gate("independent_cpu_vjp_sum", exact=False)]
    health = {"passed": True, "nonfinite_parameters": [], "nonfinite_optimizer_tensors": []}
    records = []
    for i in (1, 2):
        checks.extend([{"name": f"adapter_update_{i}_gradient_ownership", "passed": True,
                        "expected_participation_matches": True, "missing_expected_gradients": [], "unexpected_gradients": []},
                       {"name": f"adapter_update_{i}_health", "weights_changed": True, **health}])
        records.append({"update": i, "microbatches": 2, "global_counts": counts, "input_tokens": 2048,
                        "weights_changed": True, "state_health": health, "gradient_norm_before_clip": 2.})
    return {"schema": report.GROUPS["distributed"]["schema"], "physical_optimizer_updates": 2,
            "configuration": {"case": "combined", "case_specification": {"name": "combined", "batch_size": 2, "length": 512},
                "batch_size": 2, "length": 512, "updates": 2, "world_size": 1, "real_distributed_execution": False,
                "cuda_graphs": False, "arm": "compiled-native", "precision": "bf16_mixed",
                "accumulation_budgets": report.ACCUMULATION_BUDGETS},
            "checks": checks, "expected_active_parameter_names": ["p"],
            "batches": {"microbatch_counts": locals_, "global_counts": counts},
            "independent_reference_microbatches_completed": 2, "update_records": records}


def recovery_raw(name):
    checks = []
    for gate in sorted(report.RECOVERY_GATES):
        row = {"name": gate, "passed": True}
        if gate.endswith("_eager_graph") or gate.endswith("_released_graph_eager"):
            row.update(all_bitwise_equal=True, loss_names_match=True, ownership_matches=True, storage_matches=True,
                       gradients={"p": {"bitwise_equal": True}}, losses={"ce": {"bitwise_equal": True}})
            if gate.endswith("_eager_graph"): row["replays_checked"] = 2
            else: row["graph_released_before_eager"] = True
        elif gate not in {"reference_state_health", "restored_state_health", "frozen_sources_dependencies"}:
            row["bitwise_manifest_equal"] = {"state": True}
        checks.append(row)
    def boundary(update):
        return {"state": {"counters": {"optimizer_updates": update}, "model": {"p": "digest"},
                          "optimizer": {"state": "digest"}, "scheduler": {"last_epoch": update}},
                "cursor": {"next_update": update}, "rng": {"cpu": "digest"}}
    def records(offset):
        return [{"metrics": {"update_completed": True, "counters": {"optimizer_updates": i}},
                 "cursor": {"next_update": i}} for i in (offset+1, offset+2)]
    return {"schema": report.GROUPS["recovery"]["schema"], "physical_optimizer_updates": 6,
            "branch_physical_optimizer_updates": {"preparation": 2, "reference": 2, "restored": 2},
            "configuration": {"case": {"name": "combined", "batch_size": 2, "length": 512},
                "world_size": 1, "accumulation": 1, "reference_graph_rebuilt": True, "precision": "bf16_mixed"},
            "checks": checks, "boundary": boundary(2), "reference_final": boundary(4), "restored_final": boundary(4),
            "preparation_records": records(0), "reference_records": records(2), "restored_records": records(2),
            "recovery_checkpoint": {"schema": "olmo-lm-training-checkpoint-v1", "optimizer_updates": 2,
                "path": f"/workspace/cdrm-w-latent/.runtime/olmo-graph-recovery/{name}/diagnostic-boundary.pt",
                "size_bytes": 123, "sha256": "a"*64},
            "checkpoint_disposal": {"deleted_after_success": True, "size_bytes": 123, "sha256": "a"*64}}


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    receipt = json.loads((report.ROOT/report.RECEIPT).read_text())
    checkpoint = {"path": "native/"+CHECKPOINT_FILENAME, "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT_SIZE}
    sources = report.ESSENTIAL_SOURCES | {g["script"] for g in report.GROUPS.values()} | {"cdrm/pretrained/distributed_training.py"}
    for source in sources: write(root/source, "# frozen fixture: "+source+"\n")
    for group, spec in report.GROUPS.items():
        for name in ("protocol.md", "usage.md", "test-results.txt"):
            write(root/f"docs/reports/{spec['name']}"/name, group+" "+name)
        for test in report.GROUP_TESTS[group]: write(root/test, "# unit fixture\n")
    for name in report.PROJECT_FILES:
        if not (root/name).exists(): write(root/name, "fixture\n")
    write(root/report.RECEIPT, receipt)
    write(root/report.ARTIFACT, {"checkpoint": checkpoint})
    for name in ("results.md", "test-results.txt", "usage.md"): write(root/report.DOCS/name, "fixture")
    write(root/report.RUNTIME/"final-gpu.log", "GPU idle\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.test",
                    "commit", "-qm", "frozen fixture"], cwd=root, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    raws, paths = {}, {}
    for group, spec in report.GROUPS.items():
        name = "combined-01"
        selection = group+"/"+name
        directory = root/".runtime"/spec["name"]/name
        directory.mkdir(parents=True)
        raw = adapter_raw() if group == "distributed" else recovery_raw(name)
        raw.update(status="passed", stage="complete", finished_utc="2026-09-25T00:00:00Z",
                   runtime_commit=revision, checkpoint=checkpoint,
                   source_hashes={source: report.digest(root/source) for source in sources},
                   protocol_sha256=report.digest(root/f"docs/reports/{spec['name']}/protocol.md"),
                   dependencies={"packages": {"torch": "test", "triton": "test"}, "dao_sources": {}, "fa4_sources": {}},
                   wandb={"status": "synced", "mode": "online", "entity": "taylorbollman",
                          "project": "pretrained-fbt-rt-nextlat", "group": spec["name"], "run_url": "https://wandb.ai/test"})
        for source in sources: write(directory/"source-snapshot"/source, (root/source).read_text())
        write(directory/"protocol.md", (root/f"docs/reports/{spec['name']}/protocol.md").read_text())
        write(directory/"report.json", raw)
        write(directory.parent/"logs"/(name+".log"), "complete\n")
        raws[selection], paths[selection] = raw, directory/"report.json"
    return root, revision, raws, paths


def selected(evidence, group="distributed"):
    root, revision, raws, paths = evidence
    name = group+"/combined-01"
    return root, revision, name, raws[name], paths[name]


def test_verified_selection_counts_actual_work_and_qualifies_scope(evidence):
    root, revision, raws, _ = evidence
    summary = report.summarize(list(raws), revision, root=root)
    assert summary["statuses"] == {"passed": 2}
    assert summary["physical_optimizer_updates"] == 8
    assert [row["logical_endpoint_updates"] for row in summary["runs"]] == [2, 4]
    assert not summary["primary_scope_complete"]  # RT-only cases not selected.
    assert "no two-GPU" in summary["scope"]


@pytest.mark.parametrize("mutation", ["snapshot", "report_hash", "self_consistent_source_forgery", "protocol", "runtime_commit", "omit_source", "checkpoint"])
def test_rejects_provenance_changes(evidence, mutation):
    root, revision, selection, raw, path = selected(evidence)
    source = "cdrm/pretrained/olmo_tiled.py"
    if mutation == "snapshot": write(path.parent/"source-snapshot"/source, "changed")
    elif mutation == "report_hash": raw["source_hashes"][source] = "0"*64
    elif mutation == "self_consistent_source_forgery":
        write(path.parent/"source-snapshot"/source, "changed")
        write(root/source, "changed")
        raw["source_hashes"][source] = report.digest(root/source)
    elif mutation == "protocol": write(path.parent/"protocol.md", "changed")
    elif mutation == "runtime_commit": raw["runtime_commit"] = "0"*40
    elif mutation == "omit_source": del raw["source_hashes"][source]
    elif mutation == "checkpoint": raw["checkpoint"]["sha256"] = "0"*64
    write(path, raw)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        report.load_run(root, selection, revision)


def test_current_code_drift_is_recorded_without_rejecting_frozen_evidence(evidence):
    root, revision, selection, _, _ = selected(evidence)
    write(root/"cdrm/pretrained/olmo_tiled.py", "later work")
    row, _ = report.load_run(root, selection, revision)
    assert "cdrm/pretrained/olmo_tiled.py" in row["current_source_differences"]


@pytest.mark.parametrize("mutation", ["count", "missing_gate", "failed_gate", "denominators", "zero_scope",
                                     "gradient_names", "budget", "exactness", "health", "wandb"])
def test_passing_adapter_cannot_hide_missing_or_failed_evidence(evidence, mutation):
    root, revision, selection, raw, path = selected(evidence)
    if mutation == "count": raw["physical_optimizer_updates"] = 3
    elif mutation == "missing_gate": raw["checks"].pop()
    elif mutation == "failed_gate": raw["checks"][0]["passed"] = False
    elif mutation == "denominators": raw["batches"]["global_counts"]["ce"] += 1
    elif mutation == "zero_scope": raw["batches"]["microbatch_counts"][1]["latent"] = 1
    elif mutation == "gradient_names": raw["expected_active_parameter_names"].append("missing")
    elif mutation == "budget": raw["checks"][2]["budgets"] = {**report.ACCUMULATION_BUDGETS, "global_relative_l2": .1}
    elif mutation == "exactness": raw["checks"][0]["gradients"]["p"]["bitwise_equal"] = False
    elif mutation == "health": raw["update_records"][0]["weights_changed"] = False
    elif mutation == "wandb": raw["wandb"]["status"] = "running"
    write(path, raw)
    with pytest.raises(ValueError): report.load_run(root, selection, revision)


@pytest.mark.parametrize("mutation", ["physical", "logical", "branch", "records", "rng", "disposal", "checkpoint_sha", "file_remains", "live_reference", "replay"])
def test_passing_recovery_requires_exact_branches_accounting_and_verified_disposal(evidence, mutation):
    root, revision, selection, raw, path = selected(evidence, "recovery")
    if mutation == "physical": raw["physical_optimizer_updates"] = 4
    elif mutation == "logical": raw["restored_final"]["state"]["counters"]["optimizer_updates"] = 6
    elif mutation == "branch": raw["branch_physical_optimizer_updates"]["reference"] = 0
    elif mutation == "records": raw["restored_records"][0]["cursor"]["next_update"] = 8
    elif mutation == "rng": raw["restored_final"]["rng"]["cpu"] = "different"
    elif mutation == "disposal": raw["checkpoint_disposal"]["deleted_after_success"] = False
    elif mutation == "checkpoint_sha": raw["checkpoint_disposal"]["sha256"] = "0"*64
    elif mutation == "file_remains": write(path.parent/"diagnostic-boundary.pt", "weights")
    elif mutation == "live_reference": raw["configuration"]["reference_graph_rebuilt"] = False
    elif mutation == "replay": next(c for c in raw["checks"] if c["name"] == "restored_eager_graph")["replays_checked"] = 1
    write(path, raw)
    with pytest.raises(ValueError): report.load_run(root, selection, revision)


@pytest.mark.parametrize("group", ["distributed", "recovery"])
def test_failed_report_is_retained_and_not_promoted_even_if_observed_checks_pass(evidence, group):
    root, revision, selection, raw, path = selected(evidence, group)
    raw["status"] = "failed"
    if group == "distributed": raw["tracking_finish_error"] = {"type": "RuntimeError", "message": "sync failed"}
    else: raw.update(tracking_error_type="RuntimeError", tracking_error="sync failed")
    write(path, raw)
    summary = report.summarize([selection], revision, root=root)
    assert summary["statuses"] == {"failed": 1}
    assert summary["physical_optimizer_updates"] == (2 if group == "distributed" else 6)
    assert summary["runs"][0]["logical_endpoint_updates"] is None
    assert not summary["primary_scope_complete"]


def test_dependency_snapshot_hash_is_checked(evidence):
    root, revision, selection, raw, path = selected(evidence)
    raw["dependencies"]["fa4_sources"] = {"interface.py": {"sha256": "0"*64, "source": "/irrelevant/interface.py"}}
    write(path.parent/"dependency-snapshot/flash_attn/cute/interface.py", "wrong")
    write(path, raw)
    with pytest.raises(ValueError, match="Dependency snapshot"):
        report.load_run(root, selection, revision)


@pytest.mark.parametrize("selection", ["../outside", "distributed/../bad", "unknown/name", "recovery/.env", "distributed//name"])
def test_unsafe_selection_rejected(selection):
    with pytest.raises(ValueError): report.parse_selection(selection)


def save_summary(evidence):
    root, revision, raws, _ = evidence
    summary = report.summarize(list(raws), revision, root=root)
    write(root/report.DOCS/"summary.json", summary)
    return root, summary


def test_collect_evidence_is_allowlisted_and_omits_failed_checkpoint_and_credentials(evidence):
    root, revision, selection, raw, path = selected(evidence, "recovery")
    raw.update(status="failed", error="interrupted", error_type="RuntimeError")
    raw.pop("checkpoint_disposal")
    write(path, raw)
    write(path.parent/"diagnostic-boundary.pt", "large diagnostic weights")
    write(path.parent/"model.pt", "unrelated weights")
    write(path.parent/".env", "TEST_SECRET=not-real")
    save_summary(evidence)
    refreshed, _, members = report.collect_evidence(root)
    assert refreshed["statuses"] == {"passed": 1, "failed": 1}
    assert any(name.endswith("recovery/combined-01/report.json") for name in members)
    assert all(Path(name).suffix not in {".pt", ".safetensors"} and ".env" not in name for name in members)
    assert not any(path.name in {"model.pt", "diagnostic-boundary.pt", ".env"} for path in members.values())


def test_retention_rechecks_summary_and_raw_reports(evidence):
    root, _ = save_summary(evidence)
    _, _, _, raw, path = selected(evidence)
    raw["wandb"]["run_url"] = "https://wandb.ai/changed"
    write(path, raw)
    with pytest.raises(ValueError, match="changed since selection"):
        report.collect_evidence(root)


@pytest.mark.parametrize("change", ["missing", "ambiguous", "symlink"])
def test_retention_requires_one_project_local_unambiguous_run_log(evidence, change):
    root, _ = save_summary(evidence)
    path = root/".runtime/olmo-distributed-prepare/logs/combined-01.log"
    if change == "missing": path.unlink()
    elif change == "ambiguous": write(path.parent.parent/"combined-01.log", "duplicate")
    else:
        path.unlink()
        path.symlink_to(root/"AGENTS.md")
    with pytest.raises(ValueError): report.collect_evidence(root)


def test_dry_run_builds_archive_without_cloud_or_weights(evidence):
    root, _ = save_summary(evidence)
    result = report.retain(root=root, dry_run=True)
    assert result["status"] == "dry_run" and Path(result["archive"]).is_file()
    assert not (root/report.DOCS/"storage-receipt.json").exists()


def test_duplicate_and_unselected_runtime_override_rejected(evidence):
    root, revision, raws, _ = evidence
    selection = next(iter(raws))
    with pytest.raises(ValueError, match="unique"):
        report.summarize([selection, selection], revision, root=root)
    with pytest.raises(ValueError, match="unselected"):
        report.summarize([selection], revision, overrides={"recovery/not-selected": revision}, root=root)


@pytest.mark.parametrize("corrupt", [False, True])
def test_upload_is_create_only_and_checks_downloaded_content(tmp_path, corrupt):
    path = tmp_path/"evidence.tar.gz"
    path.write_bytes(b"example retained bytes")
    expected = report.file_digest(path)
    class Blob:
        name = "prefix/evidence.tar.gz"
        generation = 123
        size = expected["size_bytes"]
        md5_hash = expected["md5_base64"]
        def upload_from_filename(self, source, **kwargs):
            assert source == str(path)
            assert kwargs == {"if_generation_match": 0, "checksum": "md5"}
        def reload(self): pass
        def download_as_bytes(self, **kwargs):
            assert kwargs == {"if_generation_match": 123}
            return b"corrupt" if corrupt else path.read_bytes()
    class Bucket:
        def blob(self, name):
            assert name == "prefix/evidence.tar.gz"
            return Blob()
    if corrupt:
        with pytest.raises(ValueError, match="Downloaded"):
            report.upload_verified(Bucket(), path, "prefix")
    else:
        result = report.upload_verified(Bucket(), path, "prefix")
        assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
