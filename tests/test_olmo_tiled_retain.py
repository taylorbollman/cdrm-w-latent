"""O2 evidence retention reuses verified O1 weights without uploading them."""
import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from cdrm.pretrained import olmo_reference
from scripts import olmo_tiled_retain as retain


def write(root, name, text="evidence"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def checkpoint():
    return {"uri": retain.O1_CHECKPOINT_URI, "generation": retain.O1_CHECKPOINT_GENERATION,
            "sha256": retain.CHECKPOINT_SHA256, "size_bytes": retain.CHECKPOINT_SIZE,
            "md5_base64": retain.O1_CHECKPOINT_MD5}


def receipt():
    return {"schema": "olmo-o1-storage-receipt-v1", "status": "verified", "objects": [checkpoint()]}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    artifacts, validation, profile, project, docs = [tmp_path / name for name in
                                                   ("artifacts", "validation", "profile", "project", "docs")]
    model = {"path": "native/model.safetensors", "sha256": retain.CHECKPOINT_SHA256,
             "size_bytes": retain.CHECKPOINT_SIZE}
    manifest = {"checkpoint": model}
    reference = {"revision": "pinned", "files": [{"file": "model.py"}, {"file": "LICENSE"}]}
    for name in (retain.MANIFEST_FILENAME, "checkpoint-inspection.json"):
        write(artifacts, name, json.dumps(manifest))
    for name in retain.FILE_SPECS:
        write(artifacts, "native/" + name)
    for name in set(retain.EXTRA_PROJECT_FILES) | retain.SOURCE_FILES["profile"] | retain.SOURCE_FILES["validation"]:
        write(project, name)
    for name in ("manifest.json", "README.md", "model.py", "LICENSE"):
        write(project, "cdrm/pretrained/_olmo_reference/" + name)
    write(project, "tests/test_olmo_tiled_contract.py")
    reports, dirs = {}, {"validation": validation, "profile": profile}
    for kind, directory in dirs.items():
        reports[kind] = {"schema": retain.REPORT_SCHEMAS[kind], "status": "passed",
                         "finished_utc": "2026-09-21T22:00:00Z", "checkpoint": model,
                         "artifacts_manifest": manifest, "native_reference_sources": reference,
                         "runtime": {"gpu": "test GPU", "torch": "test", "cuda": "test"},
                         "source_hashes": {name: retain.file_digest(project / name)["sha256"]
                                           for name in retain.SOURCE_FILES[kind]}}
        write(directory, "report.json", json.dumps(reports[kind]))
    for name in ("results.md", "protocol.md", "test-results.txt"):
        write(docs, name)
    original_receipt = write(tmp_path, "o1-storage-receipt.json", json.dumps(receipt()))
    monkeypatch.setattr(retain, "validate_prepared_manifest", lambda _: manifest)
    monkeypatch.setattr(olmo_reference, "verify_olmo_reference_sources", lambda: reference)
    return SimpleNamespace(artifacts=artifacts, validation=validation, profile=profile,
                           project=project, docs=docs, reports=reports, dirs=dirs,
                           manifest=manifest, reference=reference, receipt=original_receipt,
                           output=tmp_path / "retained")


def validate(e):
    return retain.validate_reports(e.artifacts, e.validation, e.profile, project_root=e.project)


def save_report(e, kind):
    write(e.dirs[kind], "report.json", json.dumps(e.reports[kind]))


def test_exact_original_receipt_matches_local_model(evidence):
    assert retain.checkpoint_reference(receipt(), evidence.manifest["checkpoint"]) == checkpoint()


@pytest.mark.parametrize("field,value", [
    ("uri", "gs://other/model.safetensors"), ("generation", "42"),
    ("sha256", "bad"), ("size_bytes", 4), ("md5_base64", "bad"),
])
def test_rejects_changed_original_checkpoint_identity(evidence, field, value):
    changed = receipt()
    changed["objects"][0][field] = value
    with pytest.raises(ValueError):
        retain.checkpoint_reference(changed, evidence.manifest["checkpoint"])


@pytest.mark.parametrize("case", ["status", "schema", "missing", "duplicate", "local_hash", "local_path"])
def test_rejects_ambiguous_or_unverified_original_receipt(evidence, case):
    original, local = receipt(), dict(evidence.manifest["checkpoint"])
    if case == "status": original["status"] = "pending"
    elif case == "schema": original["schema"] = "wrong"
    elif case == "missing": original["objects"] = []
    elif case == "duplicate": original["objects"].append(checkpoint())
    elif case == "local_hash": local["sha256"] = "wrong"
    elif case == "local_path": local["path"] = "wrong.safetensors"
    with pytest.raises(ValueError):
        retain.checkpoint_reference(original, local)


def test_both_completed_reports_are_tied_to_sources_and_artifacts(evidence):
    manifest, reports, reference = validate(evidence)
    assert (manifest, reports, reference) == (evidence.manifest, evidence.reports, evidence.reference)


@pytest.mark.parametrize("kind", ["validation", "profile"])
@pytest.mark.parametrize("field,value", [
    ("schema", "wrong"), ("status", "failed"), ("finished_utc", None),
    ("checkpoint", {}), ("artifacts_manifest", {}), ("native_reference_sources", {}), ("runtime", {}),
])
def test_rejects_missing_or_mismatched_report_provenance(evidence, kind, field, value):
    evidence.reports[kind][field] = value
    save_report(evidence, kind)
    with pytest.raises(ValueError):
        validate(evidence)


@pytest.mark.parametrize("kind", ["validation", "profile"])
def test_rejects_stale_validated_source(evidence, kind):
    path = "scripts/olmo_tiled_validate.py" if kind == "validation" else "scripts/olmo_tiled_profile.py"
    (evidence.project / path).write_text("changed after run")
    with pytest.raises(ValueError, match="source changed"):
        validate(evidence)


def test_source_inventory_is_positive_whitelist(evidence):
    path = write(evidence.project, ".runtime/private/model.safetensors", "unapproved")
    evidence.reports["profile"]["source_hashes"][".runtime/private/model.safetensors"] = retain.file_digest(path)["sha256"]
    save_report(evidence, "profile")
    with pytest.raises(ValueError, match="inventory"):
        validate(evidence)


def test_archive_selection_never_includes_model_caches_or_secrets(evidence):
    e = evidence
    for name in (".env", "native/.cache/huggingface/token", "history/old.json"):
        write(e.artifacts, name, "unapproved")
    for name in ("credentials.json", "secrets.txt", ".env", "wandb/private.json"):
        write(e.docs, name, "unapproved")
    for directory in (e.validation, e.profile):
        write(directory, "wandb/files/config.json", "unapproved")
    members = retain.collect_evidence(e.artifacts, e.validation, e.profile, e.manifest, e.reports, e.reference,
                                      checkpoint_receipt=e.receipt, project_root=e.project, report_dir=e.docs)
    names = {name for _, name in members}
    assert {"validation/report.json", "profile/report.json", "provenance/o1-storage-receipt.json",
            "artifacts/checkpoint-inspection.json", "report/protocol.md", "report/results.md"}.issubset(names)
    assert "project/cdrm/pretrained/_olmo_reference/LICENSE" in names
    assert not any(any(part in name for part in ("model.safetensors", ".cache", "credential", "secrets", ".env", "wandb", "history")) for name in names)


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.generation, self.metadata, self.upload_count = "123", {}, 0
    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "md5"
        data = Path(filename).read_bytes()
        self.size = len(data)
        self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()
        self.bucket.objects[self.name] = self
        self.upload_count += 1
    def reload(self):
        pass


class FakeBucket:
    name = "fast-chunks"
    def __init__(self):
        self.objects = {}
        key = retain.O1_CHECKPOINT_URI.split("/", 3)[3]
        source = FakeBlob(self, key)
        source.generation = retain.O1_CHECKPOINT_GENERATION
        source.metadata = {"sha256": retain.CHECKPOINT_SHA256}
        source.size, source.md5_hash = retain.CHECKPOINT_SIZE, retain.O1_CHECKPOINT_MD5
        self.objects[key] = source
    def get_blob(self, key):
        return self.objects.get(key)
    def blob(self, key):
        return FakeBlob(self, key)


@pytest.mark.parametrize("case", ["generation", "sha256", "size", "md5", "missing", "bucket"])
def test_remote_original_checkpoint_is_reverified_read_only(case):
    bucket = FakeBucket()
    source = next(iter(bucket.objects.values()))
    if case == "generation": source.generation = "other"
    elif case == "sha256": source.metadata["sha256"] = "wrong"
    elif case == "size": source.size += 1
    elif case == "md5": source.md5_hash = "wrong"
    elif case == "missing": bucket.objects.clear()
    elif case == "bucket": bucket.name = "other"
    with pytest.raises(ValueError):
        retain.verify_checkpoint_reference(bucket, checkpoint())
    assert source.upload_count == 0


def test_uploads_reuse_exact_evidence_and_never_overwrite_or_upload_checkpoint(tmp_path):
    path = write(tmp_path, "evidence", "fixed")
    digest, bucket = retain.file_digest(path), FakeBucket()
    first = retain.upload_verified(bucket, "scope/evidence.tar.gz", path, digest)
    assert retain.upload_verified(bucket, "scope/evidence.tar.gz", path, digest) == first
    blob = bucket.objects["scope/evidence.tar.gz"]
    blob.metadata["sha256"] = "wrong"
    with pytest.raises(ValueError, match="differs"):
        retain.upload_verified(bucket, "scope/evidence.tar.gz", path, digest)
    assert blob.upload_count == 1
    with pytest.raises(ValueError, match="must not upload"):
        retain.upload_verified(bucket, "scope/model.safetensors", path, digest)


def test_end_to_end_mock_retention_creates_only_three_evidence_objects(evidence, monkeypatch):
    from google.cloud import storage
    e, bucket = evidence, FakeBucket()
    original_validate, original_collect = retain.validate_reports, retain.collect_evidence
    monkeypatch.setattr(retain, "validate_reports", lambda *args: original_validate(*args, project_root=e.project))
    monkeypatch.setattr(retain, "collect_evidence", lambda *args, **kw: original_collect(*args, **kw, project_root=e.project))
    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=lambda name: bucket))
    args = SimpleNamespace(artifacts=e.artifacts, validation=e.validation, profile=e.profile,
                           checkpoint_receipt=e.receipt, output_dir=e.output, report_dir=e.docs,
                           prefix=retain.PREFIX_ROOT + "20260921T220000Z")
    result = retain.retain(args)
    assert result["status"] == "verified"
    assert result["checkpoint_reference"]["reused_without_upload"] is True
    assert len(bucket.objects) == 4
    assert sum(blob.upload_count for blob in bucket.objects.values()) == 3
    assert bucket.objects[retain.O1_CHECKPOINT_URI.split("/", 3)[3]].upload_count == 0
    with tarfile.open(e.output / "evidence.tar.gz") as archive:
        names = archive.getnames()
        assert "provenance/o1-storage-receipt.json" in names
        assert not any(name.endswith("model.safetensors") for name in names)
    saved = json.loads((e.output / "retention-manifest.json").read_text())
    assert saved["checkpoint_uploaded"] is False
    assert saved["checkpoint_compressed_in_evidence"] is False


@pytest.mark.parametrize("prefix", [retain.PREFIX_ROOT, retain.PREFIX_ROOT + "../bad",
                                      retain.PREFIX_ROOT + "20260921T220000Z/nested", "gs://other/20260921T220000Z"])
def test_prefix_scope(prefix):
    with pytest.raises(ValueError):
        retain.parse_prefix(prefix)
