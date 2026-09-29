#!/usr/bin/env python3
"""Restore one guarded tiny distributed checkpoint and its packed-index authority.

CPU/cloud only: no torch import, tensor deserialization, CUDA or model execution.
The matching checked-out runner remains responsible for runtime and full restore.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import tarfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "olmo-guarded-loop-recovery-bundle-v1"
MAX_OBJECT_BYTES = 256 * 1024**2
MAX_ARCHIVE_BYTES = 128 * 1024**2
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
REMOTE_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_name(name):
    if not isinstance(name, str) or not name:
        raise ValueError("A relative artifact name is required")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("..", ".", "") for part in name.split("/")) or "\\" in name:
        raise ValueError("Unsafe relative artifact name")
    return path


def regular(path, root=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular artifact: {path}")
    if root is not None:
        root = Path(root).resolve()
        if not path.resolve().is_relative_to(root):
            raise ValueError("Artifact escapes its declared directory")
        for parent in path.parents:
            if parent.resolve() == root:
                break
            if parent.is_symlink():
                raise ValueError("Artifact has a symlink parent")
    return path


def verify_file(path, record, *, root=None):
    regular(path, root)
    if (type(record.get("size_bytes")) is not int or record["size_bytes"] < 0
            or not isinstance(record.get("sha256"), str) or not SHA_PATTERN.fullmatch(record["sha256"])
            or Path(path).stat().st_size != record["size_bytes"] or sha(path) != record["sha256"]):
        raise ValueError(f"Artifact SHA256/size differs: {path}")


def pinned_json(path, expected_sha256):
    regular(path)
    if not SHA_PATTERN.fullmatch(expected_sha256) or Path(path).stat().st_size > MAX_ARCHIVE_BYTES or sha(path) != expected_sha256:
        raise ValueError("Pinned JSON authority differs")
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or sha(path) != expected_sha256:
        raise ValueError("JSON authority changed or has invalid schema")
    return value


def write_manifest_once(path, value):
    """Publish the completion marker only after its full bytes are durable."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def object_record(record):
    uri = record.get("uri", "")
    parsed = urlparse(uri)
    if (not uri.startswith(REMOTE_ROOT) or parsed.scheme != "gs" or parsed.netloc != "fast-chunks"
            or parsed.query or parsed.fragment or not parsed.path):
        raise ValueError("Recovery reads only the explicit fast-chunks project lineage")
    relative_name(parsed.path.lstrip("/"))
    if (not isinstance(record.get("generation"), str) or not record["generation"].isdigit()
            or int(record["generation"]) < 1 or type(record.get("size_bytes")) is not int
            or not 0 <= record["size_bytes"] <= MAX_OBJECT_BYTES
            or not isinstance(record.get("sha256"), str) or not SHA_PATTERN.fullmatch(record["sha256"])):
        raise ValueError("Remote object needs bounded bytes, exact generation and SHA256")
    return parsed.netloc, parsed.path.lstrip("/"), int(record["generation"])


def select_object(receipt, suffix):
    matches = [value for value in receipt.get("objects", []) if value.get("uri", "").endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Receipt must identify exactly one {suffix}")
    object_record(matches[0])
    return matches[0]


def google_fetch(record, destination):
    """Read one specified generation; no publication or latest-object fallback."""
    from google.cloud import storage
    bucket, key, generation = object_record(record)
    blob = storage.Client().bucket(bucket).blob(key, generation=generation)
    blob.download_to_filename(str(destination), if_generation_match=generation, checksum="md5")


def download(record, destination, fetch):
    object_record(record)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("A new recovery directory is required")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError("Unexpected partial recovery object")
    # Interrupted/corrupt bytes remain visibly partial. They are never promoted
    # to an authority filename or treated as a completed recovery.
    fetch(record, temporary)
    verify_file(temporary, record)
    temporary.rename(destination)
    return {"uri": record["uri"], "generation": record["generation"],
            "sha256": record["sha256"], "size_bytes": record["size_bytes"],
            "destination": str(destination)}


def extract_evidence(archive, manifest, destination):
    """Verify every member and extract only regular files without tar links."""
    inventory = manifest.get("members")
    if not isinstance(inventory, list) or not inventory or len(inventory) > 4096:
        raise ValueError("Invalid bounded archive member inventory")
    expected = {}
    for item in inventory:
        name = str(relative_name(item["path"]))
        if not name.startswith("evidence/") or name in expected:
            raise ValueError("Unexpected or duplicate evidence member")
        expected[name] = item
    if sum(row["size_bytes"] for row in inventory) > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive expanded bytes exceed the bound")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected) | {"RESTORE.md", "evidence-members.json"}:
            raise ValueError("Archive members differ from the pinned inventory")
        for member in members:
            relative_name(member.name)
            if not member.isfile() or member.size < 0 or member.size > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive contains a link, special file or oversized member")
        if json.load(tar.extractfile("evidence-members.json")) != inventory:
            raise ValueError("Archive internal/external inventories differ")
        for name, pin in expected.items():
            member = tar.getmember(name)
            if member.size != pin["size_bytes"]:
                raise ValueError("Archive member size differs")
            target = Path(destination) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            verify_file(target, pin, root=destination)
    return len(expected)


def source_inventory(evidence, sources, checkout):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Source inventory is missing")
    for name, digest in sources.items():
        relative_name(name)
        if not isinstance(digest, str) or not SHA_PATTERN.fullmatch(digest):
            raise ValueError("Source authority must contain exact SHA256 strings")
        snapshot, live = Path(evidence) / "source-snapshot" / name, Path(checkout) / name
        if sha(regular(snapshot, evidence)) != digest or sha(regular(live, checkout)) != digest:
            raise ValueError(f"Matching checkout/source snapshot required: {name}")
    return len(sources)


def index_and_corpus(index_dir, corpus_root, manifest_sha256):
    index_dir, corpus_root = Path(index_dir), Path(corpus_root)
    manifest_path = regular(index_dir / "manifest.json", index_dir)
    if sha(manifest_path) != manifest_sha256:
        raise ValueError("Index manifest differs from checkpoint authority")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "olmo-packed-campaign-data-v1" or manifest.get("length") != 16
            or manifest.get("policy", {}).get("document_policy") != "continuous-stream-v1"
            or manifest.get("index", {}).get("path") != "documents.sqlite"):
        raise ValueError("This bundle supports the tested continuous-stream T16 index")
    verify_file(index_dir / "documents.sqlite", manifest["index"], root=index_dir)
    pins = manifest.get("corpus_files")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Index has no immutable corpus dependencies")
    for name, pin in pins.items():
        relative_name(name)
        verify_file(corpus_root / name, pin, root=corpus_root)
    if sha(corpus_root / "manifest.json") != manifest["corpus_manifest_sha256"]:
        raise ValueError("Prepared corpus manifest authority differs")
    return manifest


def container_path(path, checkout):
    path, checkout = Path(path).resolve(), Path(checkout).resolve()
    return str(Path("/workspace/cdrm-w-latent") / path.relative_to(checkout)) if path.is_relative_to(checkout) else str(path)


def resume_command(checkout, output, corpus, report, checkpoint_sha, index_sha, storage_prefix, host_checkout_root):
    """Render a conditional operator command; this helper never executes it."""
    cv = lambda path: container_path(path, checkout)
    args = ["torchrun", "--standalone", "--nproc_per_node=2", "scripts/olmo_campaign_loop_guarded.py",
            "--corpus", cv(corpus), "--index", cv(Path(output) / "index"), "--index-sha256", index_sha,
            "--resume", cv(Path(output) / "checkpoint"), "--resume-manifest-sha256", checkpoint_sha,
            "--arm", report["configuration"]["recipe"]["arm"], "--max-updates", "3",
            "--output-dir", cv(Path(output) / "resumed-stage"),
            "--checkpoint-root", cv(Path(output) / "resumed-checkpoints"),
            "--storage-prefix", storage_prefix]
    inner = "\n".join(("set -euo pipefail", "test -f /.dockerenv",
        'test "$PWD" = /workspace/cdrm-w-latent', "nvidia-smi",
        "env -u GOOGLE_APPLICATION_CREDENTIALS NCCL_ASYNC_ERROR_HANDLING=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=0 "
        "timeout --signal=TERM --kill-after=30s 180s " + shlex.join(args)))
    return "cd " + shlex.quote(str(host_checkout_root)) + "\n" + (
        "CDRM_FLASH_ATTENTION_SOURCE=installed bash scripts/docker_shell.sh bash -lc " + shlex.quote(inner))


def recover(*, stage_receipt, stage_receipt_sha256, index_receipt, index_receipt_sha256,
            checkpoint_update, corpus_root, checkout_root, output_dir, resume_storage_prefix,
            host_checkout_root, fetch=google_fetch):
    if type(checkpoint_update) is not int or checkpoint_update not in (1, 2):
        raise ValueError("Select completed guarded update 1 or 2 before the final limit")
    output, checkout = Path(output_dir).resolve(), Path(checkout_root).resolve()
    host_checkout_root = Path(host_checkout_root)
    if not host_checkout_root.is_absolute() or ".." in host_checkout_root.parts:
        raise ValueError("Declare an absolute host checkout path for the Docker launcher")
    if output.exists() or not output.is_relative_to(checkout):
        raise ValueError("Use a fresh persistent directory under the matching project checkout")
    stage = pinned_json(stage_receipt, stage_receipt_sha256)
    index = pinned_json(index_receipt, index_receipt_sha256)
    if not re.fullmatch(re.escape(REMOTE_ROOT) + r"olmo-fusion-startup/[0-9]{8}T[0-9]{6}Z/[A-Za-z0-9][A-Za-z0-9_.-]*", resume_storage_prefix):
        raise ValueError("Specify a distinct immutable fusion-startup prefix for later resumed checkpoints")
    if (stage.get("schema") != "olmo-two-gpu-stage-retention-v1" or stage.get("status") != "verified"
            or index.get("schema") != "olmo-t16-index-retention-envelope-v1" or index.get("status") != "retained"):
        raise ValueError("Use the tested distributed-stage and explicit T16 index receipts")
    output.mkdir(parents=True)
    downloads = []
    try:
        (output / "authorities").mkdir()
        for name, path, digest in (("stage-receipt.json", stage_receipt, stage_receipt_sha256),
                                   ("index-receipt.json", index_receipt, index_receipt_sha256)):
            target = output / "authorities" / name
            shutil.copyfile(path, target)
            if sha(target) != digest:
                raise ValueError("Input receipt changed while preserving its authority")
        for suffix in ("retention-manifest.json", "evidence.tar.gz"):
            downloads.append(download(select_object(stage, suffix), output / "stage" / suffix, fetch))
        retention = json.loads((output / "stage/retention-manifest.json").read_text())
        verify_file(output / "stage/evidence.tar.gz", retention["evidence"])
        members = extract_evidence(output / "stage/evidence.tar.gz", retention, output / "stage")
        evidence = output / "stage/evidence"
        report = json.loads((evidence / "report.json").read_text())
        if (report.get("schema") != "olmo-campaign-loop-acceptance-v1"
                or report.get("lifecycle_adapter", {}).get("version") != "olmo-campaign-loop-guarded-v1"
                or report.get("status") not in ("passed", "failed")
                or report.get("configuration", {}).get("scale") != "tiny_native_vocabulary"
                or report["configuration"].get("world_size") != 2
                or report["configuration"].get("targets") != [64, 128, 192]):
            raise ValueError("This recovery bundle supports only the guarded tiny acceptance lineage")
        selected = [value for value in report.get("published_checkpoints", [])
                    if value.get("counters", {}).get("optimizer_updates") == checkpoint_update]
        if len(selected) != 1:
            raise ValueError("Selected update has no unique completely published checkpoint")
        selected = selected[0]
        if any(record["uri"].startswith(resume_storage_prefix + "/")
               for entry in report.get("published_checkpoints", [])
               for record in entry.get("retention", {}).get("objects", [])):
            raise ValueError("Prospective resume storage prefix must differ from the original stage")
        if report["status"] == "failed" and report.get("failure_scope") != "coordinated ordinary host failure":
            raise ValueError("Unknown failed-update state is not a recovery authority")
        snapshots = source_inventory(evidence, report["sources"], checkout)
        for required in ("scripts/olmo_campaign_loop_guarded.py", "scripts/olmo_campaign_loop_run.py"):
            if required not in report["sources"]:
                raise ValueError("Guarded runner source authority is missing")
        launcher = regular(checkout / "scripts/docker_shell.sh", checkout)
        for suffix in ("state.pt", "manifest.json"):
            downloads.append(download(select_object(selected["retention"], suffix), output / "checkpoint" / suffix, fetch))
        checkpoint_path = output / "checkpoint/manifest.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        if (sha(checkpoint_path) != selected["manifest_sha256"]
                or checkpoint.get("schema") != "olmo-replicated-ddp-checkpoint-v1"
                or checkpoint.get("metadata") != selected["metadata"]
                or checkpoint["metadata"].get("configuration") != report["configuration"]
                or checkpoint["metadata"].get("source_fingerprint") != report["fingerprint"]
                or checkpoint.get("counters") != selected["counters"]
                or checkpoint.get("rank_cursors") != selected["rank_cursors"]
                or checkpoint.get("world_size") != 2 or checkpoint.get("state", {}).get("filename") != "state.pt"):
            raise ValueError("Checkpoint metadata differs from retained stage authority")
        verify_file(output / "checkpoint/state.pt", checkpoint["state"])
        for suffix in ("index/documents.sqlite", "index/manifest.json"):
            downloads.append(download(select_object(index, suffix), output / suffix, fetch))
        index_sha = report["configuration"]["data_manifest_sha256"]
        manifest = index_and_corpus(output / "index", corpus_root, index_sha)
        expected_chunk = sum(report["configuration"]["targets"][:checkpoint_update]) // manifest["length"]
        for rank, cursor in enumerate(checkpoint["rank_cursors"]):
            expected = {"schema": report["schema"], "rank": rank, "world_size": 2, "physical_batch_per_rank": 2,
                        "cursor": {"manifest_sha256": index_sha, "split": manifest["split"],
                                   "next_update": checkpoint_update, "next_chunk": expected_chunk}}
            if cursor != expected:
                raise ValueError("Restored cursor/index/rank authority differs")
        if len(checkpoint["rank_cursors"]) != 2:
            raise ValueError("Checkpoint rank cursor topology differs")
        source_record = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
        for name in ("tests/test_campaign_recovery_bundle.py", "docs/reports/olmo-campaign-lifecycle/recovery-bundle-protocol.md"):
            source_record[name] = sha(ROOT / name)
        result = {"schema": SCHEMA, "status": "assets_verified_launch_pending", "gpu_executed": False,
            "checkpoint_update": checkpoint_update, "world_size": 2, "downloads": downloads,
            "stage_receipt_sha256": stage_receipt_sha256, "index_receipt_sha256": index_receipt_sha256,
            "source_snapshot_count": snapshots, "archive_members": members,
            "checkpoint_manifest_sha256": sha(checkpoint_path), "index_manifest_sha256": index_sha,
            "resume_storage_prefix": resume_storage_prefix,
            "verified_checkout_path": str(checkout), "operator_host_checkout_path": str(host_checkout_root),
            "operator_launcher_sha256": sha(launcher),
            "corpus_root": str(Path(corpus_root).resolve()), "corpus_manifest_sha256": manifest["corpus_manifest_sha256"],
            "corpus_file_count": len(manifest["corpus_files"]), "sources": source_record,
            "required_runtime": report["configuration"]["runtime"], "required_determinism": report["configuration"]["determinism"],
            "required_configuration": report["configuration"], "source_fingerprint": report["fingerprint"],
            "launch_blockers": ["Target GPU/runtime/topology has not been checked by this CPU helper; unchanged guarded runner must match the saved configuration exactly.",
                                "The declared host checkout path must be the host location of this verified checkout; CPU-container preparation cannot verify that host mapping.",
                                "Confirm the existing corpus path is mounted at the same path inside the project container."],
            "qualification": "Verified operator assets only; no tensor deserialization, GPU launch, cross-runtime/H200 or pretrained recovery clearance."}
        command = resume_command(checkout, output, corpus_root, report, sha(checkpoint_path), index_sha,
                                 resume_storage_prefix, host_checkout_root)
        (output / "resume-command.txt").write_text("# Conditional operator command; read recovery-manifest.json blockers first.\n" + command + "\n")
        for name, digest in source_record.items():
            target = output / "source-snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, target)
            if sha(target) != digest:
                raise ValueError("Recovery helper source changed during execution")
        # This marker is last. A partial directory is never a successful bundle.
        write_manifest_once(output / "recovery-manifest.json", result)
        return result
    except BaseException as error:
        (output / "recovery-failure.json").write_text(json.dumps({"schema": SCHEMA, "status": "failed",
            "type": type(error).__name__, "message": str(error), "completed_downloads": downloads}, indent=2, sort_keys=True) + "\n")
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage-receipt", "index-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--checkpoint-update", type=int, choices=(1, 2), default=1)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, default=ROOT)
    parser.add_argument("--host-checkout-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-storage-prefix", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    result = recover(**vars(parse_args(argv)))
    print(json.dumps({key: result[key] for key in ("status", "checkpoint_update", "world_size", "source_snapshot_count", "corpus_file_count", "gpu_executed")}, sort_keys=True))


if __name__ == "__main__":
    main()
