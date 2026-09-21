"""O3 immutable evidence provenance, positive archive selection and no model upload."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from cdrm.pretrained import olmo_reference
from scripts import olmo_lm_retain as retain
from scripts.olmo_lm_common import verify_nextlat_sources


def write(root, name, text="evidence"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def checkpoint():
    return {"uri": retain.O1_CHECKPOINT_URI, "generation": retain.O1_CHECKPOINT_GENERATION,
            "sha256": retain.CHECKPOINT_SHA256, "size_bytes": retain.CHECKPOINT_SIZE,
            "md5_base64": retain.O1_CHECKPOINT_MD5}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    artifacts, validation, profile, project, docs = [tmp_path / name for name in
                                                   ("artifacts", "validation", "profile", "project", "docs")]
    model = {"path": "native/model.safetensors", "sha256": retain.CHECKPOINT_SHA256,
             "size_bytes": retain.CHECKPOINT_SIZE}
    manifest = {"checkpoint": model}
    reference = {"revision": "pinned", "files": [{"file": "model.py"}, {"file": "LICENSE"}]}
    nextlat = {"revision": "nextlat-pinned", "files": [{"file": name} for name in sorted(retain.NEXTLAT_SOURCE_FILES)]}
    for name in (retain.MANIFEST_FILENAME, "checkpoint-inspection.json"):
        write(artifacts, name, json.dumps(manifest))
    for name in retain.FILE_SPECS:
        write(artifacts, "native/" + name)
    for name in set(retain.EXTRA_PROJECT_FILES) | retain.SOURCE_FILES["profile"] | retain.SOURCE_FILES["validation"]:
        write(project, name)
    for name in ("manifest.json", "README.md", "model.py", "LICENSE"):
        write(project, "cdrm/pretrained/_olmo_reference/" + name)
    for name in {"manifest.json", "README.md"} | retain.NEXTLAT_SOURCE_FILES:
        write(project, "cdrm/pretrained/_nextlat_reference/" + name)
    for name in ("test_olmo_lm_training.py", "test_olmo_nextlat.py", "test_nextlat_memory.py"):
        write(project, "tests/" + name)
    reports, dirs = {}, {"validation": validation, "profile": profile}
    for kind, directory in dirs.items():
        reports[kind] = {"schema": retain.REPORT_SCHEMAS[kind], "status": "passed",
                         "finished_utc": "2026-09-22T00:00:00Z", "checkpoint": model,
                         "artifacts_manifest": manifest, "native_reference_sources": reference,
                         "nextlat_reference_sources": nextlat,
                         "runtime": {"gpu": "test GPU", "torch": "test", "cuda": "test"},
                         "source_hashes": {name: retain.file_digest(project / name)["sha256"]
                                           for name in retain.SOURCE_FILES[kind]}}
        write(directory, "report.json", json.dumps(reports[kind]))
    for name in ("results.md", "protocol.md", "test-results.txt", "validation-summary.json"):
        write(docs, name)
    receipt = write(tmp_path, "o1-storage-receipt.json", json.dumps({
        "schema": "olmo-o1-storage-receipt-v1", "status": "verified", "objects": [checkpoint()]}))
    monkeypatch.setattr(retain, "validate_prepared_manifest", lambda _: manifest)
    monkeypatch.setattr(olmo_reference, "verify_olmo_reference_sources", lambda: reference)
    monkeypatch.setattr(retain, "verify_nextlat_sources", lambda: nextlat)
    return SimpleNamespace(artifacts=artifacts, validation=validation, profile=profile,
                           project=project, docs=docs, reports=reports, dirs=dirs,
                           manifest=manifest, reference=reference, nextlat=nextlat, receipt=receipt,
                           output=tmp_path / "retained")


def validate(e):
    return retain.validate_reports(e.artifacts, e.validation, e.profile, project_root=e.project)


def save_report(e, kind):
    write(e.dirs[kind], "report.json", json.dumps(e.reports[kind]))


def test_real_nextlat_source_manifest_hashes_and_exact_inventory():
    manifest = verify_nextlat_sources()
    assert {item["file"] for item in manifest["files"]} == retain.NEXTLAT_SOURCE_FILES


def test_complete_reports_match_both_source_pins_and_checkpoint(evidence):
    assert validate(evidence) == (evidence.manifest, evidence.reports, evidence.reference, evidence.nextlat)
    receipt = json.loads(evidence.receipt.read_text())
    assert retain.checkpoint_reference(receipt, evidence.manifest["checkpoint"]) == checkpoint()


@pytest.mark.parametrize("kind", ["validation", "profile"])
@pytest.mark.parametrize("field,value", [
    ("schema", "wrong"), ("status", "failed"), ("finished_utc", None),
    ("checkpoint", {}), ("artifacts_manifest", {}), ("native_reference_sources", {}),
    ("nextlat_reference_sources", {}), ("runtime", {}),
])
def test_incomplete_or_stale_provenance_is_rejected(evidence, kind, field, value):
    evidence.reports[kind][field] = value
    save_report(evidence, kind)
    with pytest.raises(ValueError):
        validate(evidence)


@pytest.mark.parametrize("kind", ["validation", "profile"])
def test_runtime_requires_actual_gpu_cuda_and_torch(evidence, kind):
    del evidence.reports[kind]["runtime"]["cuda"]
    save_report(evidence, kind)
    with pytest.raises(ValueError, match="runtime"):
        validate(evidence)


@pytest.mark.parametrize("source", ["cdrm/pretrained/nextlat.py", "cdrm/pretrained/lm_training.py",
                                   "scripts/olmo_lm_common.py", "scripts/olmo_lm_profile.py"])
def test_changed_source_cannot_inherit_gpu_clearance(evidence, source):
    write(evidence.project, source, "changed after successful report")
    with pytest.raises(ValueError, match="source changed"):
        validate(evidence)


def test_report_inventory_cannot_include_arbitrary_files(evidence):
    path = write(evidence.project, "recovery.pt", "unapproved")
    evidence.reports["profile"]["source_hashes"]["recovery.pt"] = retain.file_digest(path)["sha256"]
    save_report(evidence, "profile")
    with pytest.raises(ValueError, match="inventory"):
        validate(evidence)


@pytest.mark.parametrize("names", [[".env"], ["../secret"], ["LICENSE", "LICENSE"]])
def test_nextlat_snapshot_cannot_add_arbitrary_or_duplicate_members(evidence, names):
    evidence.nextlat["files"].extend({"file": name} for name in names)
    with pytest.raises(ValueError, match="whitelist"):
        validate(evidence)


def test_evidence_selects_sources_and_docs_not_models_secrets_or_logs(evidence):
    e = evidence
    for root in (e.validation, e.profile, e.docs, e.project):
        for name in ("recovery.pt", "model.safetensors", ".env", "credentials.json", "logs.txt", "wandb/private.json"):
            write(root, name, "unapproved")
    members = retain.collect_evidence(e.artifacts, e.validation, e.profile, e.manifest,
                                      e.reports, e.reference, e.nextlat,
                                      checkpoint_receipt=e.receipt, project_root=e.project, report_dir=e.docs)
    names = {name for _, name in members}
    assert {"validation/report.json", "profile/report.json", "provenance/o1-storage-receipt.json",
            "project/tests/test_nextlat_memory.py", "project/tests/test_olmo_lm_training.py",
            "project/cdrm/pretrained/_nextlat_reference/LICENSE",
            "project/cdrm/pretrained/_nextlat_reference/fineweb_1b_horizon1.yaml",
            "report/protocol.md", "report/results.md"}.issubset(names)
    assert not any(any(part in name for part in (".pt", "model.safetensors", "credentials", ".env", "logs.txt", "wandb"))
                   for name in names)


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


@pytest.mark.parametrize("name", ["model.safetensors", "recovery.pt", "arbitrary.tar.gz"])
def test_upload_interface_refuses_model_or_unapproved_objects(tmp_path, name):
    path = write(tmp_path, name)
    with pytest.raises(ValueError, match="only whitelisted"):
        retain.upload_verified(FakeBucket(), "scope/" + name, path, retain.file_digest(path))


def test_upload_reuses_identical_bytes_and_refuses_conflicting_object(tmp_path):
    path = write(tmp_path, "evidence.tar.gz")
    digest, bucket = retain.file_digest(path), FakeBucket()
    first = retain.upload_verified(bucket, "scope/evidence.tar.gz", path, digest)
    assert retain.upload_verified(bucket, "scope/evidence.tar.gz", path, digest) == first
    blob = bucket.objects["scope/evidence.tar.gz"]
    blob.metadata["sha256"] = "different"
    with pytest.raises(ValueError):
        retain.upload_verified(bucket, "scope/evidence.tar.gz", path, digest)
    assert blob.upload_count == 1


def test_end_to_end_mock_keeps_original_checkpoint_read_only(evidence, monkeypatch):
    from google.cloud import storage
    e, bucket = evidence, FakeBucket()
    original_validate, original_collect = retain.validate_reports, retain.collect_evidence
    monkeypatch.setattr(retain, "validate_reports", lambda *args: original_validate(*args, project_root=e.project))
    monkeypatch.setattr(retain, "collect_evidence", lambda *args, **kw: original_collect(*args, **kw, project_root=e.project))
    monkeypatch.setattr(storage, "Client", lambda: SimpleNamespace(bucket=lambda name: bucket))
    args = SimpleNamespace(artifacts=e.artifacts, validation=e.validation, profile=e.profile,
                           checkpoint_receipt=e.receipt, output_dir=e.output, report_dir=e.docs,
                           prefix=retain.PREFIX_ROOT + "20260922T000000Z")
    result = retain.retain(args)
    assert result["status"] == "verified"
    assert result["checkpoint_reference"]["reused_without_upload"] is True
    assert len(bucket.objects) == 4
    assert sum(blob.upload_count for blob in bucket.objects.values()) == 3
    assert bucket.objects[retain.O1_CHECKPOINT_URI.split("/", 3)[3]].upload_count == 0
    with tarfile.open(e.output / "evidence.tar.gz") as archive:
        names = archive.getnames()
        assert "provenance/o1-storage-receipt.json" in names
        assert not any(name.endswith((".pt", ".safetensors")) for name in names)
    saved = json.loads((e.output / "retention-manifest.json").read_text())
    assert saved["checkpoint_uploaded"] is saved["checkpoint_compressed_in_evidence"] is False
    assert saved["disposable_recovery_checkpoint_retained"] is False


@pytest.mark.parametrize("prefix", [retain.PREFIX_ROOT, retain.PREFIX_ROOT + "../bad",
                                      retain.PREFIX_ROOT + "20260922T000000Z/nested", "gs://other/20260922T000000Z"])
def test_prefix_stays_in_selected_scope(prefix):
    with pytest.raises(ValueError):
        retain.parse_prefix(prefix)
