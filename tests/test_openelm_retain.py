"""Retention scope, deterministic evidence, and no-overwrite object checks."""

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from scripts import openelm_retain as retain


def _write(root: Path, relative: str, content: str = "evidence") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_collection_whitelist_excludes_checkpoint_caches_secrets_and_history(tmp_path):
    artifacts, validation, project, reports = [tmp_path / name for name in ("artifacts", "validation", "project", "reports")]
    manifest = {"checkpoint": {"path": f"checkpoint/{retain.CHECKPOINT_FILENAME}"},
                "sources": {"artifacts": {"sources/corenet/reference.py": {}}},
                "tokenizer": {"path": "tokenizer/tokenizer.model"}}
    for relative in (*retain._ARTIFACT_JSON, "sources/corenet/reference.py", "sources/corenet/reference.py.artifact.json",
                     "tokenizer/tokenizer.model", f"checkpoint/{retain.CHECKPOINT_FILENAME}.artifact.json"):
        _write(artifacts, relative)
    for relative in (f"checkpoint/{retain.CHECKPOINT_FILENAME}", "hf-cache/large.bin", ".env", "history/old-run.json"):
        _write(artifacts, relative)
    _write(validation, "report.json")
    _write(validation, "wandb/files/config.json")
    for relative in retain._PROJECT_FILES:
        _write(project, relative)
    _write(project, "cdrm/pretrained/_corenet_reference/manifest.json", json.dumps({"files": [{"file": "LICENSE"}]}))
    _write(project, "cdrm/pretrained/_corenet_reference/README.md")
    _write(project, "cdrm/pretrained/_corenet_reference/LICENSE")
    _write(reports, "summary.md")
    _write(reports, "test-results.txt")
    _write(reports, ".env.json")
    _write(reports, "wandb/private.json")
    selected = retain.collect_evidence(artifacts, validation, manifest, project_root=project, report_dir=reports)
    names = {name for _, name in selected}
    assert "artifacts/tokenizer/tokenizer.model" in names
    assert "validation/report.json" in names
    assert "report/summary.md" in names
    assert "report/test-results.txt" in names
    assert f"artifacts/checkpoint/{retain.CHECKPOINT_FILENAME}" not in names
    assert not any("hf-cache" in name or "wandb" in name or ".env" in name or "history" in name for name in names)


def test_path_validation_rejects_escape_and_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _write(tmp_path, "outside.json")
    (root / "alias.json").symlink_to(tmp_path / "outside.json")
    for relative in ("../outside.json", "/tmp/outside.json", ".env", "alias.json"):
        with pytest.raises(ValueError):
            retain._safe_member(root, relative)


def test_archive_is_deterministic_and_refuses_changed_idempotent_output(tmp_path):
    source = _write(tmp_path, "source.json", '{"fixed": true}\n')
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    members = [(source, "validation/report.json")]
    inventory = retain.build_evidence_archive(first, members, "restore instructions")
    retain.build_evidence_archive(second, members, "restore instructions")
    assert first.read_bytes() == second.read_bytes()
    retain.build_evidence_archive(first, members, "restore instructions")
    with tarfile.open(first) as archive:
        assert set(archive.getnames()) == {"validation/report.json", "RESTORE.md", "evidence-members.json"}
        assert json.load(archive.extractfile("evidence-members.json")) == inventory
    source.write_text("changed")
    with pytest.raises(FileExistsError, match="differs"):
        retain.build_evidence_archive(first, members, "restore instructions")


@pytest.mark.parametrize("prefix", [
    "gs://other/cdrm-w-latent/fbt-rt-nextlat/openelm-import/20260921T120000Z/",
    retain.PREFIX_ROOT, retain.PREFIX_ROOT + "../wrong", retain.PREFIX_ROOT + "20260921T120000Z/nested",
])
def test_prefix_is_scoped_to_one_timestamp(prefix):
    with pytest.raises(ValueError):
        retain.parse_prefix(prefix)


@pytest.mark.parametrize("status,finished", [("running", None), ("failed", "2026-09-21T12:00:00Z"), ("passed", None)])
def test_retention_rejects_incomplete_or_failed_validation_before_upload(tmp_path, monkeypatch, status, finished):
    artifacts, validation = tmp_path / "artifacts", tmp_path / "validation"
    manifest = {"sources": {"artifacts": {}}, "tokenizer": {}}
    _write(artifacts, "source_manifest.json", json.dumps(manifest["sources"]))
    _write(artifacts, "tokenizer_manifest.json", json.dumps(manifest["tokenizer"]))
    _write(validation, "report.json", json.dumps({"schema": "openelm-import-validation-v1", "status": status, "finished_utc": finished}))
    monkeypatch.setattr(retain, "validate_prepared_manifest", lambda _root: manifest)
    args = SimpleNamespace(artifacts=artifacts, validation=validation, output_dir=tmp_path / "retained",
                           report_dir=None, prefix=retain.PREFIX_ROOT + "20260921T120000Z")
    with pytest.raises(ValueError, match="completed, passing"):
        retain.retain(args)
    assert not args.output_dir.exists()


class _FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = None
        self.generation = "123"
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


def test_upload_verifies_remote_identity_and_reuses_only_exact_objects(tmp_path):
    path = _write(tmp_path, "evidence", "fixed bytes")
    digest = retain.file_digest(path)
    bucket = _FakeBucket()
    first = retain.upload_verified(bucket, "scope/object", path, digest)
    second = retain.upload_verified(bucket, "scope/object", path, digest)
    assert first == second
    assert first["generation"] == "123"
    assert bucket.objects["scope/object"].upload_count == 1
    bucket.objects["scope/object"].metadata["sha256"] = "incorrect"
    with pytest.raises(ValueError, match="differs"):
        retain.upload_verified(bucket, "scope/object", path, digest)
    assert bucket.objects["scope/object"].upload_count == 1
