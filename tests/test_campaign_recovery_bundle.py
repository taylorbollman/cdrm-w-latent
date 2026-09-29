"""Operator recovery with real archive bytes and generation-addressed fake storage."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import tarfile

import pytest

from scripts import olmo_campaign_recovery_bundle as bundle


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def pin(value):
    return {"size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


@pytest.fixture
def authority(tmp_path):
    root, corpus = tmp_path / "checkout", tmp_path / "corpus"
    root.mkdir(); corpus.mkdir()
    source_bytes = {"scripts/olmo_campaign_loop_guarded.py": b"# guarded runner\n",
                    "scripts/olmo_campaign_loop_run.py": b"# frozen runner\n"}
    for name, value in {**source_bytes, "scripts/docker_shell.sh": b"# container launcher\n"}.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    corpus_bytes = {"manifest.json": encoded({"corpus": "immutable"}), "shard/tokens.bin": b"actual token bytes"}
    for name, value in corpus_bytes.items():
        path = corpus / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    database = b"opaque SQLite file authenticated without tensor or database execution"
    index = {"schema": "olmo-packed-campaign-data-v1", "length": 16, "split": "train",
             "policy": {"document_policy": "continuous-stream-v1"},
             "index": {"path": "documents.sqlite", **pin(database)},
             "corpus_files": {name: pin(value) for name, value in corpus_bytes.items()},
             "corpus_manifest_sha256": pin(corpus_bytes["manifest.json"])["sha256"]}
    index_bytes = encoded(index); index_sha = pin(index_bytes)["sha256"]
    objects, calls = {}, []
    def remote(name, data):
        record = {"uri": bundle.REMOTE_ROOT + "olmo-fusion-startup/20260929T075900Z/fake/" + name,
                  "generation": str(100 + len(objects)), **pin(data)}
        objects[(record["uri"], record["generation"])] = data
        return record
    config = {"scale": "tiny_native_vocabulary", "world_size": 2, "targets": [64, 128, 192],
              "recipe": {"arm": "NFR"}, "runtime": {"gpu": "NVIDIA H100", "torch": "pinned"},
              "determinism": {"enabled": True}, "data_manifest_sha256": index_sha}
    sources = {name: pin(value)["sha256"] for name, value in source_bytes.items()}
    fingerprint = {"sha256": "a" * 64, "sources": sources}
    counters = {"optimizer_updates": 1}
    cursors = [{"schema": "olmo-campaign-loop-acceptance-v1", "rank": rank, "world_size": 2,
                "physical_batch_per_rank": 2, "cursor": {"manifest_sha256": index_sha,
                    "split": "train", "next_update": 1, "next_chunk": 4}} for rank in (0, 1)]
    state = b"opaque checkpoint tensor archive bytes; never deserialized"
    checkpoint = {"schema": "olmo-replicated-ddp-checkpoint-v1", "metadata": {
        "configuration": config, "source_fingerprint": fingerprint}, "world_size": 2,
        "counters": counters, "rank_cursors": cursors,
        "state": {"filename": "state.pt", **pin(state)}}
    checkpoint_bytes = encoded(checkpoint)
    selected = {"counters": counters, "metadata": checkpoint["metadata"], "rank_cursors": cursors,
                "manifest_sha256": pin(checkpoint_bytes)["sha256"],
                "retention": {"objects": [remote("update-000001/state.pt", state),
                                           remote("update-000001/manifest.json", checkpoint_bytes)]}}
    report = {"schema": "olmo-campaign-loop-acceptance-v1", "status": "passed",
              "lifecycle_adapter": {"version": "olmo-campaign-loop-guarded-v1"},
              "configuration": config, "fingerprint": fingerprint, "sources": sources,
              "published_checkpoints": [selected]}
    members = {"evidence/report.json": encoded(report), **{
        "evidence/source-snapshot/" + name: value for name, value in source_bytes.items()}}
    inventory = [{"path": name, **pin(value)} for name, value in members.items()]
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name, data in {**members, "RESTORE.md": b"Read authority first\n",
                           "evidence-members.json": encoded(inventory)}.items():
            member = tarfile.TarInfo(name); member.size = len(data); tar.addfile(member, io.BytesIO(data))
    archive_bytes = archive.getvalue()
    retention = {"members": inventory, "evidence": pin(archive_bytes)}
    stage = {"schema": "olmo-two-gpu-stage-retention-v1", "status": "verified", "objects": [
        remote("evidence.tar.gz", archive_bytes), remote("retention-manifest.json", encoded(retention))]}
    index_receipt = {"schema": "olmo-t16-index-retention-envelope-v1", "status": "retained", "objects": [
        remote("index/documents.sqlite", database), remote("index/manifest.json", index_bytes)]}
    stage_path, index_path = tmp_path / "stage-receipt.json", tmp_path / "index-receipt.json"
    stage_path.write_bytes(encoded(stage)); index_path.write_bytes(encoded(index_receipt))
    def fetch(record, destination):
        calls.append((record["uri"], record["generation"]))
        destination.write_bytes(objects[(record["uri"], record["generation"])])
    kwargs = dict(stage_receipt=stage_path, stage_receipt_sha256=bundle.sha(stage_path),
                  index_receipt=index_path, index_receipt_sha256=bundle.sha(index_path), checkpoint_update=1,
                  corpus_root=corpus, checkout_root=root, output_dir=root / "restored",
                  host_checkout_root=Path("/home/operator/project with spaces"),
                  resume_storage_prefix=bundle.REMOTE_ROOT + "olmo-fusion-startup/20260929T075900Z/fresh-resume",
                  fetch=fetch)
    return {**locals(), "kwargs": kwargs}


def test_complete_bundle_authenticates_bytes_sources_data_and_only_renders_command(authority):
    result = bundle.recover(**authority["kwargs"])
    output = authority["kwargs"]["output_dir"]
    assert result["status"] == "assets_verified_launch_pending" and not result["gpu_executed"]
    assert len(authority["calls"]) == 6 and result["source_snapshot_count"] == 2
    assert result["corpus_file_count"] == 2 and result["launch_blockers"]
    assert (output / "checkpoint/state.pt").read_bytes() == authority["state"]
    assert (output / "index/documents.sqlite").read_bytes() == authority["database"]
    command = (output / "resume-command.txt").read_text()
    assert "--nproc_per_node=2" in command and "--resume-manifest-sha256" in command
    assert authority["selected"]["manifest_sha256"] in command
    assert authority["index_sha"] in command and "--storage-prefix" in command
    assert shlex.split(command.splitlines()[1]) == ["cd", "/home/operator/project with spaces"]
    assert result["verified_checkout_path"] != result["operator_host_checkout_path"]
    assert "/workspace/cdrm-w-latent/restored/checkpoint" in command
    assert not (output / "resumed-stage").exists()
    for name, value in authority["source_bytes"].items():
        assert (authority["root"] / name).read_bytes() == value
    with pytest.raises(ValueError, match="fresh"):
        bundle.recover(**authority["kwargs"])


@pytest.mark.parametrize("failure", ["interrupted", "corrupt", "missing_generation"])
def test_incomplete_download_never_publishes_success_or_launch_command(authority, failure):
    original = authority["kwargs"]["fetch"]
    def fetch(record, destination):
        if record["uri"].endswith("/state.pt"):
            if failure == "missing_generation":
                raise FileNotFoundError("exact remote generation unavailable")
            destination.write_bytes(b"partial or corrupt")
            if failure == "interrupted":
                raise OSError("interrupted download")
        else:
            original(record, destination)
    with pytest.raises((ValueError, OSError)):
        bundle.recover(**{**authority["kwargs"], "fetch": fetch})
    output = authority["kwargs"]["output_dir"]
    assert (output / "recovery-failure.json").is_file()
    assert not (output / "recovery-manifest.json").exists()
    assert not (output / "resume-command.txt").exists()
    assert not (output / "checkpoint/state.pt").exists()


@pytest.mark.parametrize("mutation", ["checkout", "missing_corpus", "changed_corpus", "receipt_pin"])
def test_unresolved_authority_is_a_blocker_not_silently_repaired(authority, mutation):
    if mutation == "checkout":
        (authority["root"] / "scripts/olmo_campaign_loop_run.py").write_text("changed source")
    elif mutation == "missing_corpus":
        (authority["corpus"] / "shard/tokens.bin").unlink()
    elif mutation == "changed_corpus":
        (authority["corpus"] / "shard/tokens.bin").write_bytes(b"changed corpus")
    else:
        authority["kwargs"]["stage_receipt_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        bundle.recover(**authority["kwargs"])
    output = authority["kwargs"]["output_dir"]
    assert not (output / "recovery-manifest.json").exists()
    assert not (output / "resume-command.txt").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "evidence/../../escape", "evidence\\escape"])
def test_archive_path_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        bundle.extract_evidence(tmp_path / "unused.tar.gz", {"members": [{"path": name, "size_bytes": 1}]}, tmp_path)


def test_tar_symlink_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    inventory = [{"path": "evidence/report.json", **pin(b"x")}]
    with tarfile.open(archive, "w:gz") as tar:
        for name, value in (("RESTORE.md", b"r"), ("evidence-members.json", encoded(inventory))):
            info = tarfile.TarInfo(name); info.size = len(value); tar.addfile(info, io.BytesIO(value))
        info = tarfile.TarInfo("evidence/report.json"); info.type = tarfile.SYMTYPE; info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ValueError, match="link"):
        bundle.extract_evidence(archive, {"members": inventory}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_remote_generation_and_project_scope_are_required(authority):
    record = copy.deepcopy(authority["selected"]["retention"]["objects"][0])
    assert bundle.object_record(record)[2] == int(record["generation"])
    for bad in ({**record, "generation": "latest"}, {**record, "size_bytes": bundle.MAX_OBJECT_BYTES + 1},
                {**record, "uri": "gs://other-bucket/state.pt"}, {**record, "uri": record["uri"] + "?generation=0"}):
        with pytest.raises(ValueError):
            bundle.object_record(bad)
