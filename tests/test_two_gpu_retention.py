"""Bounded stage selection, checkpoint identity and create-only GCS protocol."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from scripts import olmo_two_gpu_retain as retain


PREFIX = retain.PREFIX_ROOT+"20260925T120000Z/eager-tiny"


def write(root, name, value="evidence"):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def fixture(root):
    source = write(root, "source-snapshot/cdrm/pretrained/example.py", "pass\n")
    write(root, "report.json", json.dumps({"passed": False, "error": {"message": "retained failure"},
        "sources": {"cdrm/pretrained/example.py": retain.file_digest(source)["sha256"]}}))
    write(root, "logs/attempt.log", "failed\n")


def checkpoint(root):
    state = write(root, "state.pt", "stand-in binary state for identity checks")
    digest = retain.file_digest(state)
    write(root, "manifest.json", json.dumps({"schema": retain.CHECKPOINT_SCHEMA, "world_size": 2,
        "counters": {"optimizer_updates": 2},
        "state": {"filename": "state.pt", "size_bytes": digest["size_bytes"], "sha256": digest["sha256"]}}))
    return root


def args(tmp_path, *, with_checkpoint=False):
    source = tmp_path/"stage"
    fixture(source)
    return SimpleNamespace(input_dir=source, prefix=PREFIX, receipt=tmp_path/"receipts/stage.json",
        checkpoint_dir=checkpoint(tmp_path/"saved") if with_checkpoint else None, dry_run=False)


def test_selection_preserves_failed_evidence_but_omits_weights_secrets_and_wandb(tmp_path):
    fixture(tmp_path)
    for name in ("wandb/run.json", "checkpoint/state.pt", "checkpoint/manifest.json", ".env.json",
                 "checkpoints/step-2.json", "loose-state.pt", ".git/config.json", "__pycache__/some.py"):
        write(tmp_path, name)
    members, verified = retain.collect_evidence(tmp_path)
    assert [name for _, name in members] == ["evidence/logs/attempt.log", "evidence/report.json",
                                           "evidence/source-snapshot/cdrm/pretrained/example.py"]
    assert verified == [{"report": "report.json", "count": 1}]


def test_hardware_or_interrupted_stage_without_main_report_is_retainable(tmp_path):
    write(tmp_path, "hardware.json", '{"count":2}')
    write(tmp_path, "startup.log", "terminated before final report")
    members, verified = retain.collect_evidence(tmp_path)
    assert len(members) == 2 and verified == []


@pytest.mark.parametrize("name", ["report.json", "source-snapshot"])
def test_symlink_evidence_is_rejected(tmp_path, name):
    stage = tmp_path/"stage"; stage.mkdir()
    outside = tmp_path/"outside"; outside.mkdir()
    target = outside if name == "source-snapshot" else write(outside, "report.json", "{}")
    (stage/name).symlink_to(target)
    with pytest.raises(ValueError, match="regular|Symlink"):
        retain.collect_evidence(stage)


def test_source_snapshot_hash_mismatch_is_rejected(tmp_path):
    fixture(tmp_path)
    write(tmp_path, "source-snapshot/cdrm/pretrained/example.py", "changed\n")
    with pytest.raises(ValueError, match="snapshot differs"):
        retain.collect_evidence(tmp_path)


def test_source_inventory_cannot_escape_stage(tmp_path):
    write(tmp_path, "report.json", json.dumps({"sources": {"../secret.py": "0"*64}}))
    with pytest.raises(ValueError, match="Unsafe"):
        retain.collect_evidence(tmp_path)


@pytest.mark.parametrize("prefix", [retain.PREFIX_ROOT, PREFIX+"/nested", PREFIX.replace("fast-chunks", "other"),
    retain.PREFIX_ROOT+"20260925T120000Z/../secret", retain.PREFIX_ROOT+"20260230T120000Z/eager",
    retain.PREFIX_ROOT+"not-a-date/eager"])
def test_prefix_requires_timestamp_and_one_scoped_stage(prefix):
    with pytest.raises(ValueError):
        retain.parse_prefix(prefix)


def test_valid_prefix_allows_trailing_slash():
    assert retain.parse_prefix(PREFIX+"/") == ("fast-chunks", PREFIX[5:].split("/", 1)[1])


def test_small_evidence_limit_is_enforced(tmp_path, monkeypatch):
    write(tmp_path, "log.txt", "123456")
    monkeypatch.setattr(retain, "MAX_EVIDENCE_BYTES", 5)
    with pytest.raises(ValueError, match="128 MiB"):
        retain.collect_evidence(tmp_path)


@pytest.mark.parametrize("failure", ["missing_manifest", "wrong_state", "wrong_filename", "wrong_schema"])
def test_checkpoint_requires_committed_matching_state(tmp_path, failure):
    checkpoint(tmp_path)
    if failure == "missing_manifest":
        (tmp_path/"manifest.json").unlink()
    elif failure == "wrong_state":
        write(tmp_path, "state.pt", "changed bytes")
    else:
        manifest = json.loads((tmp_path/"manifest.json").read_text())
        if failure == "wrong_filename": manifest["state"]["filename"] = "../state.pt"
        else: manifest["schema"] = "other"
        write(tmp_path, "manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError):
        retain.checkpoint_inventory(tmp_path)


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = None
        self.generation = "42"
        self.uploads = 0

    def upload_from_filename(self, filename, *, if_generation_match, checksum):
        assert if_generation_match == 0 and checksum == "md5"
        assert self.name not in self.bucket.objects
        self.payload = Path(filename).read_bytes()
        self.size = len(self.payload)
        self.md5_hash = base64.b64encode(hashlib.md5(self.payload).digest()).decode()
        self.bucket.objects[self.name] = self
        self.bucket.order.append(self.name)
        self.uploads += 1

    def reload(self): pass

    def download_as_bytes(self, *, if_generation_match):
        assert str(if_generation_match) == self.generation
        return self.payload


class Bucket:
    name = "fast-chunks"
    def __init__(self): self.objects, self.order = {}, []
    def get_blob(self, name): return self.objects.get(name)
    def blob(self, name): return Blob(self, name)


def test_upload_reuses_only_exact_existing_objects(tmp_path):
    path = write(tmp_path, "evidence.json", "fixed bytes")
    digest = retain.file_digest(path)
    bucket = Bucket()
    first = retain.upload_verified(bucket, "object", path, digest, download_sha256=True)
    assert retain.upload_verified(bucket, "object", path, digest, download_sha256=True) == first
    assert bucket.objects["object"].uploads == 1
    bucket.objects["object"].metadata["sha256"] = "bad"
    with pytest.raises(ValueError, match="differs"):
        retain.upload_verified(bucket, "object", path, digest)


def test_download_check_detects_wrong_payload_despite_matching_metadata(tmp_path):
    path = write(tmp_path, "evidence.json", "fixed bytes")
    digest = retain.file_digest(path)
    bucket = Bucket()
    retain.upload_verified(bucket, "object", path, digest)
    bucket.objects["object"].payload = b"changed"
    with pytest.raises(ValueError, match="Downloaded"):
        retain.upload_verified(bucket, "object", path, digest, download_sha256=True)


def test_retention_archive_and_checkpoint_receipt_are_deterministic_idempotent(tmp_path):
    options = args(tmp_path, with_checkpoint=True)
    bucket = Bucket()
    result = retain.retain(options, bucket=bucket)
    assert result["status"] == "verified" and result["checkpoint_uploaded"]
    assert result["local_files_deleted"] is False
    assert options.checkpoint_dir.joinpath("state.pt").is_file()
    assert bucket.order[0].endswith("checkpoint/state.pt")
    assert bucket.order[1].endswith("checkpoint/manifest.json")
    assert bucket.order[-1].endswith("storage-receipt.json")
    assert len(result["objects"]) == 4
    assert result["objects"][0]["verification"]["download_sha256"] is False
    assert result["objects"][2]["verification"]["download_sha256"] is True
    assert retain.retain(options, bucket=bucket) == result
    assert all(blob.uploads == 1 for blob in bucket.objects.values())
    archive = options.receipt.parent/(options.receipt.stem+".artifacts")/"evidence.tar.gz"
    with tarfile.open(archive) as stream:
        assert "evidence/report.json" in stream.getnames()
        assert not any(name.endswith("state.pt") for name in stream.getnames())
    write(options.input_dir, "logs/attempt.log", "changed evidence")
    with pytest.raises(FileExistsError):
        retain.retain(options, bucket=bucket)


def test_dry_run_makes_no_verified_receipt_or_cloud_request(tmp_path):
    options = args(tmp_path)
    options.dry_run = True
    result = retain.retain(options)
    assert result["status"] == "dry_run"
    assert Path(result["archive"]).is_file()
    assert not options.receipt.exists()


def test_receipt_cannot_be_in_input_stage(tmp_path):
    options = args(tmp_path)
    options.receipt = options.input_dir/"receipt.json"
    with pytest.raises(ValueError, match="outside"):
        retain.retain(options)


def test_failed_payload_upload_never_creates_success_receipt(tmp_path, monkeypatch):
    options = args(tmp_path, with_checkpoint=True)
    def fail(*args, **kwargs): raise RuntimeError("network interrupted")
    monkeypatch.setattr(retain, "upload_verified", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        retain.retain(options, bucket=Bucket())
    assert not options.receipt.exists()
    assert options.checkpoint_dir.joinpath("state.pt").exists()
