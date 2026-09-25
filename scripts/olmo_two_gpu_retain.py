#!/usr/bin/env python3
"""Retain one explicit two-GPU stage and optional committed checkpoint.

Successful, failed and interrupted stages are all valid evidence. This helper
does not certify numerical success, modify training state, or delete local
files. Run after writers stop; changed evidence must use a new stage prefix.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote
from cdrm.pretrained.distributed_checkpoint import CHECKPOINT_SCHEMA

SCHEMA = "olmo-two-gpu-stage-retention-v1"
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-two-gpu/"
MAX_EVIDENCE_BYTES = 128 * 1024**2
FORBIDDEN = {"wandb", ".git", ".docker-home", "__pycache__", "hf-cache", "checkpoints", "checkpoint"}
SUFFIXES = {".json", ".log", ".txt", ".md", ".py", ".sh", ".csv", ".yaml", ".yml",
            ".toml", ".png", ".pdf", ".svg", ".html"}


def parse_prefix(prefix: str) -> tuple[str, str]:
    prefix = prefix.rstrip("/")
    if not prefix.startswith(PREFIX_ROOT):
        raise ValueError(f"Prefix must be under {PREFIX_ROOT}")
    suffix = prefix[len(PREFIX_ROOT):]
    match = re.fullmatch(r"(\d{8}T\d{6}Z)/([A-Za-z0-9][A-Za-z0-9_.-]*)", suffix)
    if match is None:
        raise ValueError("Prefix must end in YYYYMMDDTHHMMSSZ/<stage>")
    datetime.strptime(match[1], "%Y%m%dT%H%M%SZ")
    bucket, key = prefix[5:].split("/", 1)
    return bucket, key


def _forbidden(name):
    return name in FORBIDDEN or name.startswith((".env", "checkpoint-", ".retention-"))


def _relative_path(value):
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute() or ".." in path.parts
            or any(_forbidden(p) for p in path.parts)):
        raise ValueError(f"Unsafe evidence path: {value!r}")
    return path


def _regular(path: Path, root: Path):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Evidence must be a regular file inside its root: {path}")
    return path


def collect_evidence(input_dir: Path):
    """Bounded recursive selection within only the explicitly requested stage."""
    input_dir = Path(input_dir)
    if input_dir.is_symlink() or not input_dir.is_dir():
        raise ValueError("input-dir must be a real stage directory")
    members = []
    size = 0
    for directory, subdirs, files in os.walk(input_dir, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if not _forbidden(name))
        for name in subdirs:
            if (Path(directory)/name).is_symlink():
                raise ValueError("Symlink directories are not retention evidence")
        for name in sorted(files):
            if _forbidden(name) or Path(name).suffix not in SUFFIXES:
                continue
            path = _regular(Path(directory)/name, input_dir)
            relative = path.relative_to(input_dir).as_posix()
            _relative_path(relative)
            size += path.stat().st_size
            if size > MAX_EVIDENCE_BYTES:
                raise ValueError("Small stage evidence exceeds 128 MiB; select a narrower input-dir")
            members.append((path, "evidence/"+relative))
    if not members:
        raise ValueError("No report, log or source evidence found")
    available = {str(path.relative_to(input_dir)): path for path, _ in members}
    source_records = []
    for path, _ in members:
        if path.name not in ("report.json", "progress.json"):
            continue
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or "sources" not in report:
            continue
        sources = report["sources"]
        if not isinstance(sources, dict) or not sources:
            raise ValueError("Declared source inventory must be a nonempty mapping")
        for relative, expected in sources.items():
            _relative_path(relative)
            if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise ValueError("Declared source inventory must contain SHA256 strings")
            source = path.parent/"source-snapshot"/relative
            _regular(source, input_dir)
            if str(source.relative_to(input_dir)) not in available:
                raise ValueError("Declared source is excluded from evidence selection")
            if file_digest(source)["sha256"] != expected:
                raise ValueError(f"Declared source snapshot differs: {relative}")
        source_records.append({"report": path.relative_to(input_dir).as_posix(), "count": len(sources)})
    return sorted(members, key=lambda item: item[1]), source_records


def checkpoint_inventory(directory: Path):
    """Verify bytes against the manifest without loading model/optimizer tensors."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint-dir must be a real committed checkpoint directory")
    manifest_path = _regular(directory/"manifest.json", directory)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint is not a committed replicated-DDP checkpoint")
    state = manifest.get("state", {})
    if state.get("filename") != "state.pt":
        raise ValueError("Checkpoint state filename must be state.pt")
    state_path = _regular(directory/"state.pt", directory)
    state_digest = file_digest(state_path)
    if any(state_digest[key] != state.get(key) for key in ("size_bytes", "sha256")):
        raise ValueError("Checkpoint state differs from its committed manifest")
    metadata = {"schema": manifest["schema"], "world_size": manifest.get("world_size"),
                "counters": manifest.get("counters"),
                "state": {"object": "checkpoint/state.pt", **state_digest},
                "manifest": {"object": "checkpoint/manifest.json", **file_digest(manifest_path)}}
    return metadata, [(state_path, "checkpoint/state.pt", state_digest),
                      (manifest_path, "checkpoint/manifest.json", metadata["manifest"])]


def _write_once(path: Path, value):
    data = (json.dumps(value, indent=2, sort_keys=True)+"\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise FileExistsError(f"Existing retention record differs: {path}")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="."+path.name+".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def upload_verified(bucket, key: str, path: Path, expected: dict, *, download_sha256=False):
    """Create-only publication; exact existing objects support interrupted retries."""
    from google.api_core.exceptions import PreconditionFailed
    blob = bucket.get_blob(key)
    if blob is None:
        blob = bucket.blob(key)
        blob.metadata = {"sha256": expected["sha256"], "artifact_schema": SCHEMA}
        try:
            blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
        except PreconditionFailed:
            blob = bucket.get_blob(key)
            if blob is None:
                raise
        blob.reload()
    _check_remote(blob, expected)
    if (blob.metadata or {}).get("artifact_schema") != SCHEMA:
        raise ValueError("Existing object has a different artifact schema")
    if download_sha256:
        payload = blob.download_as_bytes(if_generation_match=blob.generation)
        if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise ValueError("Downloaded evidence SHA256 differs")
    return {"uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation),
        **{key: expected[key] for key in ("size_bytes", "sha256", "md5_base64")},
        "verification": {"server_size": True, "server_md5": True, "sha256_metadata": True,
                         "download_sha256": download_sha256}}


def retain(args, *, bucket=None):
    bucket_name, key = parse_prefix(args.prefix)
    prefix = f"gs://{bucket_name}/{key}"
    input_dir = Path(args.input_dir)
    receipt_path = Path(args.receipt)
    output = receipt_path.parent/(receipt_path.stem+".artifacts")
    if (receipt_path.resolve().is_relative_to(input_dir.resolve()) or
            output.resolve().is_relative_to(input_dir.resolve())):
        raise ValueError("Receipt and retention outputs must be outside input-dir")
    members, source_records = collect_evidence(input_dir)
    checkpoint, checkpoint_objects = None, []
    if args.checkpoint_dir is not None:
        checkpoint, checkpoint_objects = checkpoint_inventory(Path(args.checkpoint_dir))
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output/"evidence.tar.gz"
    inventory = build_evidence_archive(archive_path, members,
        "# Two-GPU stage evidence\n\n"
        "This archive preserves the explicitly selected stage, including failed or interrupted attempts. "
        "Retention does not establish numerical or training success. Check reports before interpreting results.\n\n"
        "Extract evidence/ and verify evidence-members.json SHA256 and sizes. Source snapshots belong to "
        "their original reports; this is a bounded source overlay, not a full repository checkout.\n\n"
        f"Remote prefix: {prefix}\n\n"
        "Optional checkpoint/state.pt and checkpoint/manifest.json are separate objects. Download both "
        "into one directory, verify their receipt hashes, and validate the committed manifest before resume. "
        "Only the same world size and recorded training configuration are supported.\n")
    archive_digest = file_digest(archive_path)
    if archive_digest["size_bytes"] > MAX_EVIDENCE_BYTES:
        raise ValueError("Compressed evidence exceeds the size bound")
    manifest = {"schema": SCHEMA, "prefix": prefix, "evidence": {"object": "evidence.tar.gz", **archive_digest},
                "members": inventory, "source_inventories_verified": source_records,
                "checkpoint": checkpoint, "local_files_deleted": False}
    manifest_path = output/"retention-manifest.json"
    _write_once(manifest_path, manifest)
    if args.dry_run:
        return {"schema": SCHEMA, "status": "dry_run", "prefix": prefix,
                "members": len(inventory), "checkpoint": checkpoint,
                "archive": str(archive_path), "manifest": str(manifest_path)}
    if bucket is None:
        from google.cloud import storage
        bucket = storage.Client().bucket(bucket_name)
    if bucket.name != bucket_name:
        raise ValueError("Provided bucket differs from the scoped prefix")
    # Publish the checkpoint state before its commit marker. A remote manifest
    # therefore never deliberately points at an upload that has not completed.
    objects = [upload_verified(bucket, key+"/"+name, path, digest)
               for path, name, digest in checkpoint_objects]
    objects.extend([upload_verified(bucket, key+"/evidence.tar.gz", archive_path, archive_digest,
                                   download_sha256=True),
                    upload_verified(bucket, key+"/retention-manifest.json", manifest_path,
                                    file_digest(manifest_path), download_sha256=True)])
    receipt = {"schema": SCHEMA, "status": "verified", "prefix": prefix, "objects": objects,
               "checkpoint_uploaded": bool(checkpoint_objects), "members": len(inventory),
               "local_files_deleted": False}
    # This is the final local commit marker; failed uploads leave all local
    # checkpoint/evidence bytes available and never produce a success receipt.
    _write_once(receipt_path, receipt)
    receipt_object = upload_verified(bucket, key+"/storage-receipt.json", receipt_path,
                                     file_digest(receipt_path), download_sha256=True)
    return {**receipt, "receipt_object": receipt_object, "receipt_path": str(receipt_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(retain(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
