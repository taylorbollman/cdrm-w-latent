"""Stage B evidence retention must reuse, not copy or overwrite, Stage A weights."""

import base64
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts import openelm_rt_retain as retain


def _write(root: Path, relative: str, content: str = "evidence") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _checkpoint_record():
    data = b"native checkpoint fixture"
    return {
        "uri": f"{retain.stage_a.PREFIX_ROOT}20260921T182701Z/checkpoint/{retain.CHECKPOINT_FILENAME}",
        "generation": "100", "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "md5_base64": base64.b64encode(hashlib.md5(data).digest()).decode(),
    }


def _receipt(record):
    return {"schema": "openelm-import-storage-receipt-v1", "status": "verified", "objects": [record]}


@pytest.mark.parametrize("suffix", ["", "../bad", "20260921T120000Z/nested", "not-a-timestamp"])
def test_rejects_unscoped_retention_destination(suffix):
    with pytest.raises(ValueError):
        retain.parse_prefix(retain.PREFIX_ROOT + suffix)
    with pytest.raises(ValueError):
        retain.parse_prefix("gs://other/openelm-rt-reference/20260921T120000Z/")


def test_requires_unique_matching_verified_checkpoint_receipt():
    record = _checkpoint_record()
    checkpoint = {key: record[key] for key in ("size_bytes", "sha256")}
    assert retain.checkpoint_reference(_receipt(record), checkpoint) == record
    for receipt in (
        {**_receipt(record), "status": "failed"},
        {**_receipt(record), "objects": [record, record]},
        _receipt({**record, "generation": ""}),
        _receipt({**record, "sha256": "different"}),
        _receipt({**record, "uri": record["uri"].replace("fast-chunks", "other")}),
    ):
        with pytest.raises(ValueError):
            retain.checkpoint_reference(receipt, checkpoint)


class _FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = None
        self.generation = "100"
        self.upload_count = 0

    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0
        assert checksum == "md5"
        data = Path(filename).read_bytes()
        self.size = len(data)
        self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()
        self.bucket.objects[self.name] = self
        self.upload_count += 1

    def reload(self):
        pass


class _FakeBucket:
    name = "fast-chunks"

    def __init__(self):
        self.objects = {}

    def get_blob(self, name):
        return self.objects.get(name)

    def blob(self, name):
        return _FakeBlob(self, name)

    def add_checkpoint(self, record):
        key = record["uri"][5:].split("/", 1)[1]
        blob = self.blob(key)
        blob.size = record["size_bytes"]
        blob.md5_hash = record["md5_base64"]
        blob.metadata = {"sha256": record["sha256"]}
        blob.generation = record["generation"]
        self.objects[key] = blob
        return blob


def test_checkpoint_reuse_reads_existing_generation_and_never_uploads():
    bucket, record = _FakeBucket(), _checkpoint_record()
    with pytest.raises(ValueError, match="missing"):
        retain.verify_checkpoint_reference(bucket, record)
    blob = bucket.add_checkpoint(record)
    result = retain.verify_checkpoint_reference(bucket, record)
    assert result["reused_without_upload"] is True
    assert blob.upload_count == 0
    blob.generation = "101"
    with pytest.raises(ValueError, match="generation changed"):
        retain.verify_checkpoint_reference(bucket, record)
    blob.generation = "100"
    blob.md5_hash = "changed"
    with pytest.raises(ValueError, match="differs"):
        retain.verify_checkpoint_reference(bucket, record)
    assert blob.upload_count == 0


@pytest.mark.parametrize("status,finished", [("running", None), ("failed", "done"), ("passed", None)])
def test_rejects_incomplete_or_failed_stage_b_before_writes(tmp_path, monkeypatch, status, finished):
    validation = tmp_path / "validation"
    _write(validation, "report.json", json.dumps({"schema": "openelm-rt-reference-validation-v1",
                                                 "status": status, "finished_utc": finished}))
    monkeypatch.setattr(retain, "validate_prepared_manifest", lambda _root: {})
    with pytest.raises(ValueError, match="completed, passing Stage B"):
        retain.validate_report(tmp_path / "artifacts", validation)


def test_validated_source_changes_and_checkpoint_mismatch_are_rejected(tmp_path, monkeypatch):
    artifacts, validation, project = [tmp_path / name for name in ("artifacts", "validation", "project")]
    manifest = {"checkpoint": {"sha256": "checkpoint"}, "sources": {}, "tokenizer": {}}
    for key, relative in (("sources", "source_manifest.json"), ("tokenizer", "tokenizer_manifest.json")):
        _write(artifacts, relative, json.dumps(manifest[key]))
    native_manifest = {"files": []}
    _write(project, "cdrm/pretrained/_corenet_reference/manifest.json", json.dumps(native_manifest))
    hashes = {relative: retain.stage_a.file_digest(_write(project, relative))["sha256"]
              for relative in retain._VALIDATED_SOURCE_FILES}
    report = {"schema": "openelm-rt-reference-validation-v1", "status": "passed", "finished_utc": "done",
              "checkpoint": manifest["checkpoint"], "artifacts_manifest": manifest,
              "source_hashes": hashes, "native_reference_sources": native_manifest}
    report_path = _write(validation, "report.json", json.dumps(report))
    monkeypatch.setattr(retain, "validate_prepared_manifest", lambda _root: manifest)
    assert retain.validate_report(artifacts, validation, project_root=project) == (manifest, report)
    changed_source = project / "cdrm/pretrained/recurrent.py"
    changed_source.write_text("changed after validation")
    with pytest.raises(ValueError, match="Validated source changed"):
        retain.validate_report(artifacts, validation, project_root=project)
    report_path.write_text(json.dumps({**report, "checkpoint": {"sha256": "different"}}))
    with pytest.raises(ValueError, match="different checkpoint"):
        retain.validate_report(artifacts, validation, project_root=project)


def test_collection_adds_rt_sources_and_receipt_without_broad_runtime_capture(tmp_path, monkeypatch):
    project = tmp_path / "project"
    for relative in retain._PROJECT_FILES:
        _write(project, relative)
    receipt = _write(tmp_path, "stage-a-receipt.json", json.dumps(_receipt(_checkpoint_record())))
    _write(tmp_path, ".env", "not retained")
    _write(tmp_path, "unrelated-checkpoint.pt", "not retained")
    monkeypatch.setattr(retain.stage_a, "collect_evidence", lambda *args, **kwargs: [])
    members = retain.collect_evidence(tmp_path, tmp_path, {}, checkpoint_receipt=receipt,
                                      project_root=project)
    names = {name for _, name in members}
    assert names == {"provenance/stage-a-storage-receipt.json", *(f"project/{p}" for p in retain._PROJECT_FILES)}


def test_complete_retention_uploads_only_new_evidence_and_reuses_checkpoint(tmp_path, monkeypatch):
    from google.cloud import storage

    record, bucket = _checkpoint_record(), _FakeBucket()
    checkpoint_blob = bucket.add_checkpoint(record)
    checkpoint_receipt = _write(tmp_path, "stage-a-receipt.json", json.dumps(_receipt(record)))
    evidence = _write(tmp_path, "report.json", '{"completed":true}\n')
    manifest, report = {"checkpoint": record}, {"status": "passed", "finished_utc": "done"}
    monkeypatch.setattr(retain, "validate_report", lambda *args: (manifest, report))
    monkeypatch.setattr(retain, "collect_evidence", lambda *args, **kwargs: [(evidence, "validation/report.json")])
    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=lambda name: bucket))
    args = SimpleNamespace(artifacts=tmp_path / "artifacts", validation=tmp_path / "validation",
                           output_dir=tmp_path / "retained", checkpoint_receipt=checkpoint_receipt,
                           report_dir=None, prefix=retain.PREFIX_ROOT + "20260921T220000Z")
    result = retain.retain(args)
    assert result["checkpoint_reference"]["uri"] == record["uri"]
    assert result["checkpoint_reference"]["reused_without_upload"] is True
    assert checkpoint_blob.upload_count == 0
    assert len(bucket.objects) == 4  # existing checkpoint, archive, manifest, receipt
    assert {Path(obj["uri"]).name for obj in result["objects"]} == {"evidence.tar.gz", "retention-manifest.json"}
    assert all(blob.metadata["artifact_schema"] == "openelm-rt-reference-retention-v1"
               for blob in bucket.objects.values() if blob is not checkpoint_blob)
    with tarfile.open(args.output_dir / "evidence.tar.gz") as archive:
        assert set(archive.getnames()) == {"validation/report.json", "RESTORE.md", "evidence-members.json"}
    saved_manifest = json.loads((args.output_dir / "retention-manifest.json").read_text())
    assert saved_manifest["checkpoint_uploaded"] is False
    assert saved_manifest["checkpoint_compressed_in_evidence"] is False
    assert retain.retain(args) == result
    assert all(blob.upload_count == 1 for blob in bucket.objects.values() if blob is not checkpoint_blob)


def test_evidence_upload_refuses_to_overwrite_changed_object(tmp_path):
    path = _write(tmp_path, "evidence", "fixed bytes")
    bucket, digest = _FakeBucket(), retain.stage_a.file_digest(path)
    retain.upload_verified(bucket, "scope/object", path, digest)
    blob = bucket.objects["scope/object"]
    blob.metadata["sha256"] = "different"
    with pytest.raises(ValueError, match="differs"):
        retain.upload_verified(bucket, "scope/object", path, digest)
    assert blob.upload_count == 1
