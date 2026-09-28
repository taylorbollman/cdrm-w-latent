"""Create-only document publication, commit ordering, and bounded readback."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import olmo_document_retain as retain


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, sort_keys=True))
    elif isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value)
    return path


def make_shard(root, config_sha="a" * 64):
    write(root, "tokens.bin", bytes([1, 0, 2, 0, 255, 255]))
    write(root, "documents.jsonl", '{"document_id":"x","offset":0,"length":3}\n')
    manifest = {"schema": "olmo-tokenized-documents-v1", "config_sha256": config_sha,
                "files": {name: retain.file_digest(root / name)
                          for name in ("tokens.bin", "documents.jsonl")}}
    write(root, "manifest.json", manifest)
    return root


def args(tmp_path, source=None):
    return SimpleNamespace(input_dir=source or make_shard(tmp_path / "shard-000000"),
        prefix=retain.PREFIX_ROOT + "20260928T220000Z/shard-000000",
        receipt=tmp_path / "receipts" / "shard-000000.json", dry_run=False)


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = {}
        self.generation = "123"
        self.uploads = 0
        self.chunk_size = None

    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0
        assert checksum == "md5"
        assert self.name not in self.bucket.objects
        self.data = Path(filename).read_bytes()
        self.size = len(self.data)
        self.md5_hash = base64.b64encode(hashlib.md5(self.data).digest()).decode()
        self.uploads += 1
        self.bucket.objects[self.name] = self
        self.bucket.upload_order.append(self.name)

    def reload(self):
        pass

    def download_to_file(self, stream, *, if_generation_match, raw_download, checksum):
        assert if_generation_match == self.generation
        assert raw_download is True
        assert checksum is None
        assert self.chunk_size == retain.CHUNK_BYTES
        payload = getattr(self, "corrupt_readback", self.data)
        for offset in range(0, len(payload), 3):
            stream.write(payload[offset:offset + 3])
        self.bucket.downloads.append(self.name)


class Bucket:
    name = "fast-chunks"

    def __init__(self):
        self.objects = {}
        self.upload_order = []
        self.downloads = []

    def get_blob(self, key):
        return self.objects.get(key)

    def blob(self, key):
        return Blob(self, key)


def test_shard_publication_is_create_only_stream_verified_and_manifest_last(tmp_path):
    options, bucket = args(tmp_path), Bucket()
    first = retain.retain(options, bucket=bucket)
    assert first["status"] == "verified"
    assert bucket.upload_order[-1].endswith("/manifest.json")
    assert len(bucket.objects) == len(bucket.downloads) == 3
    assert json.loads(options.receipt.read_text()) == first
    assert all(row["verification"]["download_sha256"] for row in first["objects"])
    assert retain.retain(options, bucket=bucket) == first
    assert len(bucket.upload_order) == 3
    assert len(bucket.downloads) == 7  # existing commit marker is also preflighted
    assert all(blob.uploads == 1 for blob in bucket.objects.values())
    assert (options.input_dir / "tokens.bin").exists()


@pytest.mark.parametrize("field,value", [("size", 900), ("md5_hash", "wrong"),
    ("sha256", "a" * 64), ("artifact_schema", "wrong"), ("corrupt_readback", b"corrupt")])
def test_remote_corruption_never_yields_success_receipt(tmp_path, field, value):
    options, bucket = args(tmp_path), Bucket()
    retain.retain(options, bucket=bucket)
    options.receipt.unlink()
    name = next(key for key in bucket.objects if key.endswith("tokens.bin"))
    blob = bucket.objects[name]
    if field in ("sha256", "artifact_schema"):
        blob.metadata[field] = value
    else:
        setattr(blob, field, value)
    with pytest.raises(ValueError, match="differs"):
        retain.retain(options, bucket=bucket)
    assert not options.receipt.exists()
    assert len(bucket.upload_order) == 3


def test_interrupted_upload_retries_prior_verified_objects_without_manifest(tmp_path):
    options, bucket = args(tmp_path), Bucket()
    original_blob = bucket.blob

    def interrupt(key):
        if key.endswith("tokens.bin"):
            raise RuntimeError("network interrupted")
        return original_blob(key)

    bucket.blob = interrupt
    with pytest.raises(RuntimeError, match="interrupted"):
        retain.retain(options, bucket=bucket)
    assert len(bucket.objects) == 1
    assert not any(name.endswith("manifest.json") for name in bucket.objects)
    assert not options.receipt.exists()
    bucket.blob = original_blob
    retain.retain(options, bucket=bucket)
    assert len(bucket.objects) == 3
    assert all(blob.uploads == 1 for blob in bucket.objects.values())


def test_manifest_digest_rejects_changed_local_bytes_before_remote_access(tmp_path):
    options, bucket = args(tmp_path), Bucket()
    write(options.input_dir, "tokens.bin", b"different")
    with pytest.raises(ValueError, match="manifest file differs"):
        retain.retain(options, bucket=bucket)
    assert not bucket.objects


@pytest.mark.parametrize("name", ["credentials.json", ".env", "secret.txt", "model.pt",
    "tokens.bin.partial", "subtree/file.tmp", "uncommitted.json", "service_account.json"])
def test_unsafe_or_uncommitted_content_rejected_before_cloud_access(tmp_path, name):
    options, bucket = args(tmp_path), Bucket()
    write(options.input_dir, name, "do not publish")
    with pytest.raises(ValueError):
        retain.retain(options, bucket=bucket)
    assert not bucket.objects


@pytest.mark.parametrize("kind", ["file", "directory", "input"])
def test_symlinks_rejected(tmp_path, kind):
    options, bucket = args(tmp_path), Bucket()
    outside = write(tmp_path, "outside.txt", "do not publish")
    if kind == "file":
        (options.input_dir / "linked.txt").symlink_to(outside)
    elif kind == "directory":
        (options.input_dir / "linked").symlink_to(tmp_path, target_is_directory=True)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(options.input_dir, target_is_directory=True)
        options.input_dir = alias
    with pytest.raises(ValueError):
        retain.retain(options, bucket=bucket)
    assert not bucket.objects


def test_missing_manifest_and_partial_directory_rejected(tmp_path):
    options, bucket = args(tmp_path), Bucket()
    (options.input_dir / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest.json"):
        retain.retain(options, bucket=bucket)
    options.input_dir = make_shard(tmp_path / "shard.partial")
    with pytest.raises(ValueError, match="uncommitted"):
        retain.retain(options, bucket=bucket)


def test_receipt_outside_input_and_dry_run_no_cloud_access(tmp_path):
    options = args(tmp_path)
    options.dry_run = True
    result = retain.retain(options)
    assert result["status"] == "dry_run" and not result["uploaded"]
    assert not options.receipt.exists()
    options.receipt = options.input_dir / "receipt.json"
    with pytest.raises(ValueError, match="outside"):
        retain.retain(options)


@pytest.mark.parametrize("prefix", ["gs://other/something", retain.PREFIX_ROOT + "run",
    retain.PREFIX_ROOT + "../stage", retain.PREFIX_ROOT + "run/stage/extra",
    retain.PREFIX_ROOT + "run/sta.ge"])
def test_scoped_prefix_rejects_unsafe_locations(prefix):
    with pytest.raises(ValueError):
        retain.parse_prefix(prefix)


def test_bounded_size_and_entry_limits(tmp_path, monkeypatch):
    options = args(tmp_path)
    monkeypatch.setattr(retain, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="GiB"):
        retain.collect_committed(options.input_dir)
    monkeypatch.setattr(retain, "MAX_BYTES", 999999)
    monkeypatch.setattr(retain, "MAX_FILES", 1)
    with pytest.raises(ValueError, match="Too many"):
        retain.collect_committed(options.input_dir)


def test_complete_root_validates_config_and_nested_shards(tmp_path):
    root = tmp_path / "complete"
    config = write(root, "config.json", {"schema": "olmo-tokenized-documents-v1"})
    config_sha = retain.file_digest(config)["sha256"]
    shard = make_shard(root / "shard-000000", config_sha)
    write(root, "manifest.json", {"schema": "olmo-tokenized-documents-v1", "completed": True,
        "config_sha256": config_sha, "shards": [{"path": "shard-000000",
        "manifest_sha256": retain.file_digest(shard / "manifest.json")["sha256"]}]})
    options, bucket = args(tmp_path, root), Bucket()
    result = retain.retain(options, bucket=bucket)
    assert len(result["objects"]) == 5
    assert bucket.upload_order[-1].endswith("/shard-000000/manifest.json")  # stage also shard-000000
    assert bucket.upload_order[-2].endswith("/shard-000000/shard-000000/manifest.json")
    write(root, "config.json", {"changed": True})
    with pytest.raises(ValueError, match="config differs"):
        retain.collect_committed(root)


def test_metadata_stage_uses_explicit_file_inventory(tmp_path):
    root = tmp_path / "metadata"
    report = write(root, "report.json", {"source": "bounded fixture"})
    write(root, "manifest.json", {"schema": "olmo-document-preparation-metadata-v1",
        "files": {"report.json": retain.file_digest(report)}})
    assert len(retain.collect_committed(root)) == 2


def test_source_file_change_during_upload_prevents_commit(tmp_path):
    options, bucket = args(tmp_path), Bucket()
    blob_factory = bucket.blob

    def tampering(key):
        blob = blob_factory(key)
        upload = blob.upload_from_filename

        def alter(filename, **kwargs):
            upload(filename, **kwargs)
            Path(filename).write_bytes(b"changed during publication")
        blob.upload_from_filename = alter
        return blob

    bucket.blob = tampering
    with pytest.raises(ValueError, match="changed during upload"):
        retain.retain(options, bucket=bucket)
    assert not any(name.endswith("manifest.json") for name in bucket.objects)
    assert not options.receipt.exists()


def test_empty_operational_lock_is_ignored_but_uncommitted_writer_state_is_not(tmp_path):
    options = args(tmp_path)
    write(options.input_dir, ".prepare.lock", "")
    assert len(retain.collect_committed(options.input_dir)) == 3
    write(options.input_dir, ".pending-123/tokens.bin", b"partial")
    with pytest.raises(ValueError, match="uncommitted"):
        retain.collect_committed(options.input_dir)


def test_different_published_manifest_fails_before_extra_object_upload(tmp_path):
    options, bucket = args(tmp_path), Bucket()
    retain.retain(options, bucket=bucket)
    manifest_path = options.input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = write(options.input_dir, "additional.txt", "new version")
    manifest["files"][path.name] = retain.file_digest(path)
    write(options.input_dir, "manifest.json", manifest)
    with pytest.raises(ValueError, match="differs"):
        retain.retain(options, bucket=bucket)
    assert len(bucket.upload_order) == 3


@pytest.mark.parametrize("state", [{"status": "running"}, {"completed": False}])
def test_explicitly_incomplete_manifest_rejected(tmp_path, state):
    options = args(tmp_path)
    path = options.input_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(state)
    write(options.input_dir, path.name, manifest)
    with pytest.raises(ValueError, match="uncommitted"):
        retain.collect_committed(options.input_dir)
