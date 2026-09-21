"""O1 retention provenance, positive whitelists and immutable uploads."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cdrm.pretrained import olmo_reference
from scripts import olmo_retain as retain

COMMON = {
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_reference.py",
    "cdrm/pretrained/olmo_artifacts.py", "cdrm/pretrained/artifacts.py",
    "scripts/olmo_validation.py", "scripts/experiment_tracking.py",
}
SOURCE_PATHS = {
    "ordinary": COMMON | {"scripts/olmo_validate.py"},
    "rt": COMMON | {"scripts/olmo_rt_validate.py", "cdrm/pretrained/recurrent.py",
                    "cdrm/pretrained/openelm.py", "cdrm/pretrained/olmo_recurrent.py",
                    "cdrm/pretrained/olmo_recurrent_oracle.py"},
}


def _write(root, relative, content="evidence"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    artifacts, ordinary, recurrent, project, docs = [tmp_path / x for x in
                                                   ("artifacts", "ordinary", "rt", "project", "docs")]
    checkpoint = _write(artifacts, "native/model.safetensors", "checkpoint bytes excluded from archive")
    manifest = {"checkpoint": {"path": "native/model.safetensors", **retain.file_digest(checkpoint)}}
    reference = {"revision": "pinned", "files": [{"file": "model.py"}, {"file": "LICENSE"}]}
    _write(artifacts, retain.MANIFEST_FILENAME, json.dumps(manifest))
    _write(artifacts, "checkpoint-inspection.json", '{"parameters": 1176764416}')
    for name in retain.FILE_SPECS:
        if name != retain.CHECKPOINT_FILENAME:
            _write(artifacts, "native/" + name)
    for name in set(retain.EXTRA_PROJECT_FILES) | SOURCE_PATHS["ordinary"] | SOURCE_PATHS["rt"]:
        _write(project, name)
    for name in ("manifest.json", "README.md", "model.py", "LICENSE"):
        _write(project, "cdrm/pretrained/_olmo_reference/" + name)
    _write(project, "tests/test_olmo_contract.py")
    reports = {}
    directories = {"ordinary": ordinary, "rt": recurrent}
    for kind, directory in directories.items():
        report = {"schema": retain.REPORT_SCHEMAS[kind], "status": "passed",
                  "finished_utc": "2026-09-21T20:00:00Z", "artifacts_manifest": manifest,
                  "checkpoint": manifest["checkpoint"], "native_reference_sources": reference,
                  "source_hashes": {name: retain.file_digest(project / name)["sha256"]
                                    for name in SOURCE_PATHS[kind]}}
        reports[kind] = report
        _write(directory, "report.json", json.dumps(report))
    for name in ("results.md", "test-results.txt", "protocol.md"):
        _write(docs, name)
    monkeypatch.setattr(retain, "validate_prepared_manifest", lambda _: manifest)
    monkeypatch.setattr(olmo_reference, "verify_olmo_reference_sources", lambda: reference)
    return SimpleNamespace(artifacts=artifacts, ordinary=ordinary, recurrent=recurrent, project=project,
                           docs=docs, reports=reports, reference=reference, manifest=manifest,
                           directories=directories)


def _validate(e):
    return retain.validate_reports(e.artifacts, e.ordinary, e.recurrent, project_root=e.project)


def _save_report(e, kind):
    _write(e.directories[kind], "report.json", json.dumps(e.reports[kind]))


def test_both_reports_have_same_pinned_artifacts_and_native_reference(evidence):
    manifest, reports, reference = _validate(evidence)
    assert manifest == evidence.manifest
    assert reports == evidence.reports
    assert reference == evidence.reference


@pytest.mark.parametrize("kind", ["ordinary", "rt"])
@pytest.mark.parametrize("field,value", [("schema", "wrong"), ("status", "failed"), ("finished_utc", None)])
def test_rejects_incomplete_wrong_schema_or_failed_report(evidence, kind, field, value):
    evidence.reports[kind][field] = value
    _save_report(evidence, kind)
    with pytest.raises(ValueError, match="completed, passing"):
        _validate(evidence)


@pytest.mark.parametrize("kind", ["ordinary", "rt"])
@pytest.mark.parametrize("field", ["checkpoint", "artifacts_manifest", "native_reference_sources"])
def test_rejects_artifact_or_native_reference_provenance_mismatch(evidence, kind, field):
    evidence.reports[kind][field] = {"different": "pin"}
    _save_report(evidence, kind)
    with pytest.raises(ValueError, match="provenance|reference sources"):
        _validate(evidence)


def test_rejects_source_changed_after_gpu_validation(evidence):
    (evidence.project / "cdrm/pretrained/olmo.py").write_text("changed since run")
    with pytest.raises(ValueError, match="source changed"):
        _validate(evidence)


def test_rejects_missing_helper_hash(evidence):
    evidence.reports["ordinary"]["source_hashes"].pop("cdrm/pretrained/artifacts.py")
    _save_report(evidence, "ordinary")
    with pytest.raises(ValueError, match="source|hash"):
        _validate(evidence)


@pytest.mark.parametrize("extra", [".runtime/private/model.safetensors", ".cache/token", "credentials.json"])
def test_rejects_non_source_member_smuggled_through_hash_dictionary(evidence, extra):
    path = _write(evidence.project, extra, "must not be archived")
    evidence.reports["rt"]["source_hashes"][extra] = retain.file_digest(path)["sha256"]
    _save_report(evidence, "rt")
    with pytest.raises(ValueError, match="source|whitelist|allowed"):
        _validate(evidence)


def test_positive_collection_whitelist_excludes_all_unselected_runtime_and_secrets(evidence):
    e = evidence
    for name in (".env", "native/.cache/huggingface/token", "hf-cache/model.safetensors", "history/old.json"):
        _write(e.artifacts, name, "do not retain")
    for directory in (e.ordinary, e.recurrent):
        _write(directory, "wandb/files/config.json")
        _write(directory, "checkpoints/latest.pt")
    for name in (".env.json", "credentials.json", "secrets.txt", "wandb/private.json", "weights.safetensors"):
        _write(e.docs, name, "do not retain")
    manifest, reports, reference = _validate(e)
    selected = retain.collect_evidence(e.artifacts, e.ordinary, e.recurrent, manifest, reports, reference,
                                       project_root=e.project, report_dir=e.docs)
    names = {name for _, name in selected}
    assert "artifacts/checkpoint-inspection.json" in names
    assert "artifacts/native/tokenizer.json" in names
    assert "validation/ordinary/report.json" in names
    assert "validation/rt/report.json" in names
    assert "project/cdrm/pretrained/_olmo_reference/LICENSE" in names
    assert "project/tests/test_olmo_contract.py" in names
    assert "report/results.md" in names
    assert "report/test-results.txt" in names
    assert "artifacts/native/model.safetensors" not in names
    assert not any(any(x in name for x in (".env", "credential", "secrets", "wandb", "hf-cache", ".cache", "history", "latest.pt")) for name in names)


def test_collection_rejects_symlink_for_explicit_source_member(evidence):
    e = evidence
    target = e.project / "scripts/olmo_prepare.py"
    target.unlink()
    target.symlink_to(e.project / "scripts/olmo_retain.py")
    with pytest.raises(ValueError, match="regular file"):
        retain.collect_evidence(e.artifacts, e.ordinary, e.recurrent, e.manifest, e.reports, e.reference,
                                project_root=e.project)


@pytest.mark.parametrize("prefix", [
    "gs://other/cdrm-w-latent/fbt-rt-nextlat/olmo1b-step60000/20260921T120000Z",
    retain.PREFIX_ROOT, retain.PREFIX_ROOT + "../wrong", retain.PREFIX_ROOT + "20260921T120000Z/nested",
])
def test_prefix_rejects_wrong_bucket_scope_and_nested_paths(prefix):
    with pytest.raises(ValueError):
        retain.parse_prefix(prefix)


def test_prefix_accepts_exact_scope():
    bucket, key = retain.parse_prefix(retain.PREFIX_ROOT + "20260921T120000Z/")
    assert bucket == "fast-chunks"
    assert key.endswith("/20260921T120000Z")


class FakeBlob:
    def __init__(self, bucket, key):
        self.bucket, self.key, self.name = bucket, key, key
        self.generation = "321"
        self.metadata = None
        self.upload_count = 0

    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0
        assert checksum == "md5"
        data = Path(filename).read_bytes()
        self.size = len(data)
        self.md5_hash = base64.b64encode(hashlib.md5(data).digest()).decode()
        self.upload_count += 1
        self.bucket.objects[self.key] = self

    def reload(self):
        pass


class FakeBucket:
    name = "fast-chunks"
    def __init__(self):
        self.objects = {}
    def get_blob(self, key):
        return self.objects.get(key)
    def blob(self, key):
        return FakeBlob(self, key)


@pytest.mark.parametrize("corruption", ["sha256", "size", "md5"])
def test_upload_only_creates_once_and_rejects_existing_remote_mismatch(tmp_path, corruption):
    path = _write(tmp_path, "evidence", "fixed evidence")
    expected = retain.file_digest(path)
    bucket = FakeBucket()
    first = retain.upload_verified(bucket, "scope/object", path, expected)
    assert retain.upload_verified(bucket, "scope/object", path, expected) == first
    assert first["generation"] == "321"
    blob = bucket.objects["scope/object"]
    assert blob.upload_count == 1
    if corruption == "sha256": blob.metadata["sha256"] = "wrong"
    elif corruption == "size": blob.size += 1
    elif corruption == "md5": blob.md5_hash = "wrong"
    with pytest.raises(ValueError, match="differs"):
        retain.upload_verified(bucket, "scope/object", path, expected)
    assert blob.upload_count == 1
