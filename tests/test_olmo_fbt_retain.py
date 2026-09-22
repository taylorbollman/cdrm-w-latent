"""Small O5a retention checks: pinned provenance and positive archive selection."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from scripts import olmo_fbt_retain as retain
from scripts.olmo_tiled_retain import (O1_CHECKPOINT_URI, O1_CHECKPOINT_GENERATION, O1_CHECKPOINT_MD5,
                                      CHECKPOINT_SIZE, CHECKPOINT_SHA256)


def write(root, name, content="evidence"):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
    return path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    project, validation, artifacts, docs = (tmp_path / name for name in ("project", "validation", "artifacts", "docs"))
    references = {}
    for name, names in retain.SNAPSHOT_FILES.items():
        references[name] = {"files": [{"file": file} for file in sorted(names)]}
        for file in names | {"manifest.json", "README.md"}:
            write(project, f"cdrm/pretrained/{name}/{file}")
    for file in retain.SOURCE_FILES | retain.EXTRA_PROJECT_FILES:
        write(project, file)
    for file in (retain.MANIFEST_FILENAME, "checkpoint-inspection.json"):
        write(artifacts, file)
    for file in retain.FILE_SPECS:
        write(artifacts, "native/" + file)
    for file in ("protocol.md", "results.md", "test-results.txt"):
        write(docs, file)
    checkpoint = {"path": "native/model.safetensors", "size_bytes": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256}
    native = {"checkpoint": checkpoint}
    report = {"schema": "olmo-fbt-validation-v1", "status": "passed", "finished_utc": "2026-09-22T01:00:00Z",
              "cases": [{"passed": True}], "checkpoint": checkpoint,
              "runtime": {"gpu": "mock H100", "torch": "test", "cuda": "test"},
              "nextlat_reference": references["_nextlat_reference"], "fbt_reference": references["_fbt_reference"],
              "source_hashes": {name: retain.file_digest(project / name)["sha256"] for name in retain.SOURCE_FILES}}
    write(validation, "report.json", json.dumps(report))
    receipt = write(tmp_path, "o1-receipt.json", json.dumps({"schema": "olmo-o1-storage-receipt-v1", "status": "verified",
        "objects": [{"uri": O1_CHECKPOINT_URI, "generation": O1_CHECKPOINT_GENERATION,
                     "size_bytes": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256, "md5_base64": O1_CHECKPOINT_MD5}]}))
    monkeypatch.setattr(retain, "validate_artifact_metadata", lambda _: native)
    monkeypatch.setattr(retain, "verify_snapshots", lambda _: references)
    return SimpleNamespace(project=project, validation=validation, artifacts=artifacts, docs=docs,
                           references=references, native=native, report=report, receipt=receipt, output=tmp_path / "out")


def validate(e):
    return retain.validate_report(e.validation, project_root=e.project, artifacts=e.artifacts)


def test_real_snapshot_inventory_and_hashes_are_exact():
    manifests = retain.verify_snapshots(retain.ROOT)
    assert set(manifests) == set(retain.SNAPSHOT_FILES)


def test_passing_report_accepts_exact_checkpoint_and_source(evidence):
    assert validate(evidence) == (evidence.report, evidence.native, evidence.references)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("finished_utc", None),
    ("cases", [{"passed": False}]), ("cases", []), ("runtime", {}), ("checkpoint", {}),
    ("fbt_reference", {}), ("nextlat_reference", {})])
def test_nonpassing_or_unpinned_report_is_rejected(evidence, field, value):
    evidence.report[field] = value
    write(evidence.validation, "report.json", json.dumps(evidence.report))
    with pytest.raises(ValueError): validate(evidence)


def test_changed_source_hash_rejected(evidence):
    write(evidence.project, "cdrm/pretrained/fbt_training.py", "changed")
    with pytest.raises(ValueError, match="changed"): validate(evidence)


def test_unlisted_report_source_is_rejected(evidence):
    path = write(evidence.project, "checkpoint.pt", "model")
    evidence.report["source_hashes"]["checkpoint.pt"] = retain.file_digest(path)["sha256"]
    write(evidence.validation, "report.json", json.dumps(evidence.report))
    with pytest.raises(ValueError, match="inventory"): validate(evidence)


def test_positive_archive_inventory_excludes_models_secrets_and_extra_files(evidence):
    e = evidence
    for root in (e.project, e.validation, e.docs, e.artifacts):
        for file in ("model.safetensors", "checkpoint.pt", ".env", "credentials.json", "wandb/private.json", "unlisted.md"):
            write(root, file, "not evidence")
    members = retain.collect_evidence(e.validation, e.report, e.references, project_root=e.project,
                artifacts=e.artifacts, checkpoint_receipt=e.receipt, report_dir=e.docs)
    names = [name for _, name in members]
    assert "project/tests/test_olmo_fbt_training.py" in names
    assert "project/cdrm/pretrained/_fbt_reference/LICENSE" in names
    assert "provenance/o1-storage-receipt.json" in names
    assert not any(name.endswith((".pt", ".safetensors", ".env", "credentials.json", "unlisted.md")) for name in names)
    archive = e.output / "evidence.tar.gz"; e.output.mkdir()
    retain.build_evidence_archive(archive, members, "restore evidence")
    with tarfile.open(archive) as stream:
        assert set(stream.getnames()) == set(names) | {"RESTORE.md", "evidence-members.json"}


def test_extra_snapshot_member_cannot_enter_archive(evidence):
    evidence.references["_fbt_reference"]["files"].append({"file": "credentials.json"})
    with pytest.raises(ValueError, match="whitelist"):
        retain.collect_evidence(evidence.validation, evidence.report, evidence.references,
            project_root=evidence.project, artifacts=evidence.artifacts, checkpoint_receipt=evidence.receipt, report_dir=evidence.docs)


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self.generation, self.metadata = bucket, name, "123", {}
        self.uploads = 0
    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "md5"
        payload = Path(filename).read_bytes()
        self.size = len(payload); self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode()
        self.bucket.objects[self.name] = self; self.uploads += 1
    def reload(self): pass


class Bucket:
    name = "fast-chunks"
    def __init__(self):
        self.objects = {}
    def get_blob(self, name): return self.objects.get(name)
    def blob(self, name): return Blob(self, name)


@pytest.mark.parametrize("name", ["model.safetensors", "checkpoint.pt", "arbitrary.json"])
def test_upload_refuses_model_and_unlisted_object_names(tmp_path, name):
    path = write(tmp_path, name)
    with pytest.raises(ValueError, match="no model objects"):
        retain.upload_verified(Bucket(), "prefix/"+name, path, retain.file_digest(path))


def test_upload_is_immutable_and_carries_fbt_schema(tmp_path):
    path = write(tmp_path, "evidence.tar.gz")
    bucket = Bucket(); digest = retain.file_digest(path)
    first = retain.upload_verified(bucket, "prefix/evidence.tar.gz", path, digest)
    assert retain.upload_verified(bucket, "prefix/evidence.tar.gz", path, digest) == first
    blob = bucket.get_blob("prefix/evidence.tar.gz")
    assert blob.metadata["artifact_schema"] == "olmo-fbt-reference-v1"
    blob.metadata["sha256"] = "wrong"
    with pytest.raises(ValueError): retain.upload_verified(bucket, "prefix/evidence.tar.gz", path, digest)
    assert blob.uploads == 1


def test_checkpoint_reference_is_read_only_and_generation_pinned():
    bucket = Bucket(); key = O1_CHECKPOINT_URI.split("/",3)[3]
    blob = Blob(bucket,key); blob.generation = O1_CHECKPOINT_GENERATION
    blob.size, blob.md5_hash = CHECKPOINT_SIZE, O1_CHECKPOINT_MD5
    blob.metadata = {"sha256": CHECKPOINT_SHA256}; bucket.objects[key] = blob
    record = {"uri": O1_CHECKPOINT_URI, "generation": O1_CHECKPOINT_GENERATION,
              "size_bytes": CHECKPOINT_SIZE, "md5_base64": O1_CHECKPOINT_MD5, "sha256": CHECKPOINT_SHA256}
    assert retain.verify_checkpoint_reference(bucket, record)["reused_without_upload"]
    assert blob.uploads == 0 and len(bucket.objects) == 1
    blob.generation = "unexpected"
    with pytest.raises(ValueError, match="generation"): retain.verify_checkpoint_reference(bucket, record)


@pytest.mark.parametrize("prefix", [retain.PREFIX_ROOT, retain.PREFIX_ROOT+"../escape",
    retain.PREFIX_ROOT+"20260922T010000Z/extra", "gs://other/20260922T010000Z"])
def test_timestamp_prefix_is_bounded(prefix):
    with pytest.raises(ValueError): retain.parse_prefix(prefix)
