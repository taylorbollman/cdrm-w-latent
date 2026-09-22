"""F3 evidence keeps provenance, bounded sources and explicit failure outcomes."""
import base64
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts import olmo_f3_retain as retain
from scripts.olmo_tiled_retain import (O1_CHECKPOINT_URI, O1_CHECKPOINT_GENERATION,
    O1_CHECKPOINT_MD5, CHECKPOINT_SIZE, CHECKPOINT_SHA256)


def write(root, name, value="evidence"):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


@pytest.fixture
def evidence(tmp_path):
    project, docs, runtime = (tmp_path/name for name in ("project", "docs", "runs"))
    for name in retain.EXTRA_PROJECT_FILES | retain.ESSENTIAL_SOURCES:
        write(project, name)
    references = {}
    for name, files in retain.SNAPSHOT_FILES.items():
        references[name] = {"files": [{"file": filename} for filename in sorted(files)]}
        for filename in files | {"README.md", "manifest.json"}:
            write(project, f"cdrm/pretrained/{name}/{filename}")
    for name in ("protocol.md", "results.md", "assessment.md", "test-results.txt"):
        write(docs, name)
    reference = {"uri": O1_CHECKPOINT_URI, "generation": O1_CHECKPOINT_GENERATION,
        "size_bytes": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256, "md5_base64": O1_CHECKPOINT_MD5}
    receipt = write(tmp_path, "o1-storage-receipt.json", {"schema": "olmo-o1-storage-receipt-v1",
        "status": "verified", "objects": [reference]})
    report = {"schema": retain.REPORT_SCHEMA, "status": "passed", "stage": "complete", "finished_utc": "now",
        "capture_succeeded": True,
        "configuration": {"case": "combined", "length": 32, "checkpointing": True},
        "checks": [{"name": "changed_tokens", "passed": True, "gradients": {"finite": True}}],
        "timings": [{"above_memory_budget": False}],
        "wandb": {"status": "synced", "run_url": "https://wandb.ai/taylorbollman/test/runs/id"},
        "protocol_sha256": retain.file_digest(docs/"protocol.md")["sha256"],
        "source_hashes": {name: retain.file_digest(project/name)["sha256"] for name in retain.ESSENTIAL_SOURCES},
        "checkpoint": {"path": "native/model.safetensors", "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT_SIZE}}
    directory = runtime/"f3-small-01"
    write(directory, "report.json", report); write(directory, "configuration.json", report["configuration"])
    return SimpleNamespace(project=project, docs=docs, runtime=runtime, directory=directory, directories=[directory],
        report=report, receipt=receipt, reference=reference, references=references)


def validate(e, **kwargs):
    return retain.validate_runs(e.directories, project_root=e.project, checkpoint_receipt=e.receipt,
                                report_dir=e.docs, **kwargs)


def collect(e, **kwargs):
    runs, _, sources = validate(e, **kwargs)
    return retain.collect_evidence(runs, sources, e.references, project_root=e.project,
                                    checkpoint_receipt=e.receipt, report_dir=e.docs)


def fail(e, status="failed"):
    e.report.update(status=status, stage="capture" if status == "capture_blocked" else "update_comparison",
                    error_type="RuntimeError", error_message="diagnostic error")
    e.report["wandb"]["status"] = "synced_failed_experiment"
    if status == "capture_blocked":
        e.report["capture_succeeded"] = False
    e.report["checks"] = [{"name": "original_budget", "passed": False}]
    write(e.directory, "report.json", e.report)


def test_passed_run_validates_sources_config_checkpoint_and_scope(evidence):
    runs, reference, sources = validate(evidence)
    assert reference == evidence.reference
    assert sources == evidence.report["source_hashes"]
    assert runs[0]["report"] == evidence.report
    p = runs[0]["provenance"]
    assert p["counts_as_success"] and p["current_source_hashes_match"]
    assert p["check_count"] == p["passed_check_count"] == 1
    # Capacity limits are observations, not mislabeled failed numerical checks.
    evidence.report["timings"][0]["above_memory_budget"] = True
    write(evidence.directory, "report.json", evidence.report)
    validate(evidence)


@pytest.mark.parametrize("status", ["failed", "capture_blocked"])
def test_explicit_failed_diagnostics_preserve_outcomes_without_promotion(evidence, status):
    fail(evidence, status)
    with pytest.raises(ValueError, match="allow-failed-diagnostics"):
        validate(evidence)
    runs, _, sources = validate(evidence, allow_failed_diagnostics=True)
    assert not sources
    assert runs[0]["provenance"]["diagnostic_failure_only"]
    assert not runs[0]["provenance"]["counts_as_success"]
    assert runs[0]["report"]["checks"][0]["passed"] is False
    members = collect(evidence, allow_failed_diagnostics=True)
    assert (evidence.directory/"report.json", "runtime/f3-small-01/report.json") in members


def test_historical_failure_snapshot_matches_original_without_promoting_old_source(evidence):
    fail(evidence)
    name = "cdrm/pretrained/olmo_static.py"
    original = write(evidence.directory, "source-snapshot/"+name, "original capture failure source")
    evidence.report["source_hashes"][name] = retain.file_digest(original)["sha256"]
    evidence.report["protocol_sha256"] = "a"*64
    write(evidence.directory, "report.json", evidence.report)
    runs, _, sources = validate(evidence, allow_failed_diagnostics=True)
    p = runs[0]["provenance"]
    assert not sources and not p["current_source_hashes_match"]
    assert not p["current_protocol_hash_matches"]
    assert p["historical_runtime_sources_complete"]
    assert (original, "runtime/f3-small-01/source-snapshot/"+name) in collect(evidence, allow_failed_diagnostics=True)
    original.write_text("wrong snapshot")
    with pytest.raises(ValueError, match="snapshot differs"):
        validate(evidence, allow_failed_diagnostics=True)


def test_missing_historical_bytes_are_reported_and_not_claimed_reconstructed(evidence):
    fail(evidence)
    evidence.report["source_hashes"]["scripts/removed_f3.py"] = "a"*64
    write(evidence.directory, "report.json", evidence.report)
    runs, _, _ = validate(evidence, allow_failed_diagnostics=True)
    p = runs[0]["provenance"]
    assert not p["historical_runtime_sources_complete"]
    assert p["historical_source_mismatches"]["scripts/removed_f3.py"]["current_sha256"] is None


def test_capacity_capture_blocker_retains_explicit_unsuccessful_capture(evidence):
    fail(evidence, "capture_blocked")
    evidence.report["stage"] = "capacity_capture"
    write(evidence.directory, "report.json", evidence.report)
    runs, _, _ = validate(evidence, allow_failed_diagnostics=True)
    assert runs[0]["provenance"]["stage"] == "capacity_capture"
    evidence.report["capture_succeeded"] = True
    write(evidence.directory, "report.json", evidence.report)
    with pytest.raises(ValueError, match="unsuccessful capture"):
        validate(evidence, allow_failed_diagnostics=True)


@pytest.mark.parametrize("change", ["running", "missing_checks", "failed_check", "nested_finite", "stage",
    "missing_inventory", "source_changed", "protocol_changed", "standalone_config", "wandb", "checkpoint"])
def test_success_requires_complete_consistent_evidence(evidence, change):
    report = evidence.report
    if change == "running": report["status"] = "running"
    elif change == "missing_checks": report["checks"] = []
    elif change == "failed_check": report["checks"][0]["passed"] = False
    elif change == "nested_finite": report["checks"][0]["gradients"]["finite"] = False
    elif change == "stage": report["stage"] = "warmup"
    elif change == "missing_inventory": del report["source_hashes"]["cdrm/pretrained/static_training.py"]
    elif change == "source_changed": write(evidence.project, "cdrm/pretrained/static_training.py", "changed")
    elif change == "protocol_changed": write(evidence.docs, "protocol.md", "changed")
    elif change == "standalone_config": write(evidence.directory, "configuration.json", {"other": True})
    elif change == "wandb": report["wandb"]["status"] = "running"
    elif change == "checkpoint": report["checkpoint"]["sha256"] = "a"*64
    write(evidence.directory, "report.json", report)
    with pytest.raises(ValueError): validate(evidence)


def test_duplicate_names_and_empty_inventory_rejected(evidence):
    evidence.directories.append(evidence.directory)
    with pytest.raises(ValueError, match="unique"): validate(evidence)
    evidence.directories.clear()
    with pytest.raises(ValueError, match="At least one"): validate(evidence)


def test_positive_archive_inventory_is_bounded_and_keeps_scoped_tests_logs(evidence, tmp_path):
    selected = [write(evidence.project, "tests/test_static_training.py"),
                write(evidence.project, "tests/test_olmo_static.py"),
                write(evidence.project, "scripts/olmo_f3_report.py")]
    for root in (evidence.project, evidence.docs, evidence.directory):
        for name in (".env", "model.safetensors", "checkpoint.pt", "credentials.json", "wandb/config.json", "other.json"):
            write(root, name, "NEVER INCLUDE")
    log = write(evidence.runtime, evidence.directory.name+".log", "selected log")
    write(evidence.runtime, "unrelated.log", "NEVER INCLUDE")
    write(evidence.docs, "throughput.png", "plot")
    members = collect(evidence)
    assert all(any(path == selected_path for path, _ in members) for selected_path in selected)
    assert (log, "runtime-logs/"+log.name) in members
    assert not any("NEVER INCLUDE" in path.read_text() for path, _ in members)
    archive = tmp_path/"evidence.tar.gz"
    retain.build_evidence_archive(archive, members, "restore")
    with tarfile.open(archive) as stream:
        assert set(stream.getnames()) == {name for _, name in members} | {"RESTORE.md", "evidence-members.json"}
    (evidence.docs/"assessment.md").unlink()
    with pytest.raises(ValueError, match="assessment"): collect(evidence)


@pytest.mark.parametrize("name", [".env", "credentials.json", "model.safetensors", "../outside.py"])
def test_even_failed_source_inventories_cannot_include_unsafe_members(evidence, name):
    fail(evidence)
    evidence.report["source_hashes"][name] = "a"*64
    write(evidence.directory, "report.json", evidence.report)
    with pytest.raises(ValueError): validate(evidence, allow_failed_diagnostics=True)


def test_symlink_source_rejected(evidence):
    target = write(evidence.project, "target.py")
    source = evidence.project/"cdrm/pretrained/olmo_static.py"
    source.unlink(); source.symlink_to(target)
    with pytest.raises(ValueError):
        validate(evidence)


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self.generation, self.metadata, self.uploads = bucket, name, "123", {}, 0
    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "md5"
        data = Path(filename).read_bytes()
        self.size = len(data); self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()
        self.bucket.objects[self.name] = self; self.uploads += 1
    def reload(self): pass


class Bucket:
    name = "fast-chunks"
    def __init__(self): self.objects = {}
    def get_blob(self, name): return self.objects.get(name)
    def blob(self, name): return Blob(self, name)


def test_transport_is_create_only_f3_scoped_and_rechecks_server_identity(tmp_path):
    path = write(tmp_path, "evidence.tar.gz"); bucket = Bucket()
    _, prefix = retain.parse_prefix(retain.PREFIX_ROOT+"20260922T010000Z")
    key, digest = prefix+"/evidence.tar.gz", retain.file_digest(path)
    first = retain.upload_verified(bucket, key, path, digest)
    assert retain.upload_verified(bucket, key, path, digest) == first
    blob = bucket.objects[key]
    assert blob.uploads == 1 and blob.metadata["artifact_schema"] == retain.SCHEMA
    blob.md5_hash = "wrong"
    with pytest.raises(ValueError): retain.upload_verified(bucket, key, path, digest)
    with pytest.raises(ValueError): retain.upload_verified(bucket, "other/evidence.tar.gz", path, digest)


def test_checkpoint_identity_verification_reuses_existing_native_weights(evidence):
    bucket = Bucket(); key = O1_CHECKPOINT_URI.split("/", 3)[3]
    blob = Blob(bucket, key); bucket.objects[key] = blob
    blob.size, blob.md5_hash, blob.generation = CHECKPOINT_SIZE, O1_CHECKPOINT_MD5, O1_CHECKPOINT_GENERATION
    blob.metadata = {"sha256": CHECKPOINT_SHA256}
    assert retain.verify_checkpoint_reference(bucket, evidence.reference)["reused_without_upload"]
    assert blob.uploads == 0
    blob.generation = "changed"
    with pytest.raises(ValueError): retain.verify_checkpoint_reference(bucket, evidence.reference)
