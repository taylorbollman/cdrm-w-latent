"""F1 evidence selection, scoped completion, immutable uploads and source pins."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts import olmo_f1_retain as retain
from scripts.olmo_tiled_retain import (O1_CHECKPOINT_URI, O1_CHECKPOINT_GENERATION,
    O1_CHECKPOINT_MD5, CHECKPOINT_SIZE, CHECKPOINT_SHA256)


def write(root, name, value="evidence"):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


@pytest.fixture
def evidence(tmp_path):
    project, runtime, docs = (tmp_path/name for name in ("project", "runtime", "docs"))
    for name in retain.EXTRA_PROJECT_FILES | {"scripts/olmo_f1_validate.py", "cdrm/pretrained/olmo.py"}:
        write(project, name)
    references = {}
    for name, files in retain.SNAPSHOT_FILES.items():
        references[name] = {"files": [{"file": filename} for filename in sorted(files)]}
        for filename in files | {"README.md", "manifest.json"}:
            write(project, f"cdrm/pretrained/{name}/{filename}")
    for filename in ("protocol.md", "results.md", "test-results.txt"):
        write(docs, filename)
    record = {"uri": O1_CHECKPOINT_URI, "generation": O1_CHECKPOINT_GENERATION,
        "md5_base64": O1_CHECKPOINT_MD5, "size_bytes": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256}
    receipt = write(tmp_path, "o1-storage-receipt.json", {"schema": "olmo-o1-storage-receipt-v1",
        "status": "verified", "objects": [record]})
    config = {"cases": ["ordinary", "all_three"], "precision": "bf16_mixed"}
    report = {"schema": "olmo-f1-integration-v1", "status": "passed", "finished_utc": "now",
        "cases": [{"name": "ordinary", "passed": True, "trace_summary": {"sdpa": 2}}],
        "requested_cases": ["ordinary"], "config": config, "wandb": {"run_id": "example"},
        "checkpoint": {"path": "native/model.safetensors", "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT_SIZE},
        "source_hashes": {name: retain.file_digest(project/name)["sha256"]
            for name in ("scripts/olmo_f1_validate.py", "cdrm/pretrained/olmo.py")},
        "protocol_sha256": retain.file_digest(docs/"protocol.md")["sha256"]}
    write(runtime, "report.json", report)
    write(runtime, "configuration.json", config)
    write(runtime, "ordinary.json", report["cases"][0])
    return SimpleNamespace(project=project, runtime=runtime, docs=docs, receipt=receipt,
        references=references, report=report, reference=record, output=tmp_path/"output")


def validate(e):
    return retain.validate_report(e.runtime, project_root=e.project, checkpoint_receipt=e.receipt)


def members(e):
    return retain.collect_evidence(e.runtime, e.report, e.references, project_root=e.project,
        checkpoint_receipt=e.receipt, report_dir=e.docs)


def test_passing_selected_subset_retains_its_actual_scope(evidence):
    report, reference = validate(evidence)
    assert report == evidence.report
    assert report["requested_cases"] == ["ordinary"]
    assert len(report["config"]["cases"]) == 2  # A subset does not imply the full configured matrix passed.
    assert reference == evidence.reference


@pytest.mark.parametrize("field,value", [("status", "running"), ("finished_utc", ""),
    ("cases", []), ("cases", [{"name": "ordinary", "passed": False}]),
    ("cases", [{"name": "ordinary", "passed": True}]*2),
    ("requested_cases", []), ("requested_cases", ["ordinary", "all_three"]),
    ("requested_cases", ["../ordinary"]), ("config", {}), ("wandb", {}),
    ("checkpoint", {}), ("source_hashes", {}), ("protocol_sha256", "short")])
def test_partial_or_mismatched_completion_is_rejected(evidence, field, value):
    evidence.report[field] = value
    write(evidence.runtime, "report.json", evidence.report)
    with pytest.raises(ValueError): validate(evidence)


@pytest.mark.parametrize("changed", ["source", "configuration", "case", "receipt"])
def test_changed_runtime_lineage_is_rejected(evidence, changed):
    if changed == "source": write(evidence.project, "cdrm/pretrained/olmo.py", "changed")
    elif changed == "configuration": write(evidence.runtime, "configuration.json", {"changed": True})
    elif changed == "case": write(evidence.runtime, "ordinary.json", {"name": "ordinary", "passed": True})
    elif changed == "receipt":
        receipt = json.loads(evidence.receipt.read_text())
        receipt["objects"][0]["generation"] = "changed"
        evidence.receipt.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): validate(evidence)


def test_positive_inventory_omits_disposable_checkpoints_secrets_and_unrequested_cases(evidence):
    for root in (evidence.project, evidence.runtime, evidence.docs):
        for filename in ("resume-boundary.pt", "ordinary/resume-boundary.pt", "ordinary/resume-boundary.pt.tmp",
                "model.safetensors", ".env", "credentials.json", "wandb/config.json", "unlisted.md", "all_three.json"):
            write(root, filename, "NEVER RETAIN")
    rows = members(evidence)
    names = [name for _, name in rows]
    assert "runtime/ordinary.json" in names
    assert "report/protocol.md" in names
    assert "project/cdrm/pretrained/_fbt_reference/LICENSE" in names
    assert "provenance/o1-storage-receipt.json" in names
    assert not any("NEVER RETAIN" in path.read_text() for path, _ in rows)
    evidence.output.mkdir()
    archive = evidence.output/"evidence.tar.gz"
    retain.build_evidence_archive(archive, rows, "restore")
    with tarfile.open(archive) as stream:
        assert set(stream.getnames()) == set(names) | {"RESTORE.md", "evidence-members.json"}


@pytest.mark.parametrize("filename", ["checkpoint.pt", "checkpoint.pt.tmp", "model.safetensors", ".env", "credentials.json"])
def test_forbidden_file_cannot_be_smuggled_through_runtime_source_inventory(evidence, filename):
    path = write(evidence.project, filename)
    evidence.report["source_hashes"][filename] = retain.file_digest(path)["sha256"]
    write(evidence.runtime, "report.json", evidence.report)
    with pytest.raises(ValueError): validate(evidence)


def test_protocol_changed_and_missing_cpu_record_are_rejected(evidence):
    write(evidence.docs, "protocol.md", "changed")
    with pytest.raises(ValueError, match="protocol"): members(evidence)
    write(evidence.docs, "protocol.md")
    (evidence.docs/"test-results.txt").unlink()
    with pytest.raises(ValueError, match="test-results"): members(evidence)


def test_final_and_ancestor_symlinks_are_rejected(evidence):
    target = write(evidence.project, "actual.py")
    (evidence.project/"alias.py").symlink_to(target)
    with pytest.raises(ValueError): retain.safe_evidence(evidence.project, "alias.py")
    (evidence.project/"alias").symlink_to(evidence.project/"scripts", target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor"):
        retain.safe_evidence(evidence.project, "alias/olmo_f1_validate.py")


def test_large_payload_budget_is_enforced(evidence, monkeypatch):
    monkeypatch.setattr(retain, "MAX_MEMBER_BYTES", 2)
    with pytest.raises(ValueError, match="small summaries"):
        retain.safe_evidence(evidence.runtime, "report.json")


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self.generation, self.metadata = bucket, name, "123", {}
        self.uploads = 0
    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "md5"
        payload = Path(filename).read_bytes()
        self.size = len(payload); self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode()
        self.bucket.objects[self.name] = self
        self.uploads += 1
    def reload(self): pass


class Bucket:
    name = "fast-chunks"
    def __init__(self): self.objects = {}
    def get_blob(self, name): return self.objects.get(name)
    def blob(self, name): return Blob(self, name)


def test_upload_create_only_reuses_identical_bytes_and_rejects_mismatch(tmp_path):
    path = write(tmp_path, "evidence.tar.gz")
    bucket = Bucket(); digest = retain.file_digest(path)
    first = retain.upload_verified(bucket, "prefix/evidence.tar.gz", path, digest)
    assert retain.upload_verified(bucket, "prefix/evidence.tar.gz", path, digest) == first
    blob = bucket.objects["prefix/evidence.tar.gz"]
    assert blob.metadata["artifact_schema"] == retain.SCHEMA
    blob.md5_hash = "wrong"
    with pytest.raises(ValueError): retain.upload_verified(bucket, "prefix/evidence.tar.gz", path, digest)
    assert blob.uploads == 1


@pytest.mark.parametrize("filename", ["checkpoint.pt", "model.safetensors", "unlisted.json"])
def test_model_upload_is_never_permitted(tmp_path, filename):
    path = write(tmp_path, filename)
    with pytest.raises(ValueError, match="whitelisted"):
        retain.upload_verified(Bucket(), "prefix/"+filename, path, retain.file_digest(path))


def test_reused_checkpoint_is_generation_and_hash_pinned_without_upload(evidence):
    bucket = Bucket(); key = O1_CHECKPOINT_URI.split("/", 3)[3]
    blob = Blob(bucket, key); bucket.objects[key] = blob
    blob.generation, blob.size, blob.md5_hash = O1_CHECKPOINT_GENERATION, CHECKPOINT_SIZE, O1_CHECKPOINT_MD5
    blob.metadata = {"sha256": CHECKPOINT_SHA256}
    assert retain.verify_checkpoint_reference(bucket, evidence.reference)["reused_without_upload"]
    assert blob.uploads == 0
    blob.generation = "changed"
    with pytest.raises(ValueError, match="generation"):
        retain.verify_checkpoint_reference(bucket, evidence.reference)


@pytest.mark.parametrize("prefix", [retain.PREFIX_ROOT, retain.PREFIX_ROOT+"../escape",
    retain.PREFIX_ROOT+"20260922T010000Z/extra", "gs://other/20260922T010000Z"])
def test_retention_stays_in_own_timestamped_prefix(prefix):
    with pytest.raises(ValueError): retain.parse_prefix(prefix)


def test_retention_prefix_accepts_exact_timestamp_only():
    bucket, key = retain.parse_prefix(retain.PREFIX_ROOT+"20260922T010000Z/")
    assert bucket == "fast-chunks"
    assert key.endswith("/20260922T010000Z")
