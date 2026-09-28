#!/usr/bin/env python3
"""Publish one committed tokenized-document shard or metadata stage to GCS.

Individual objects are create-only and verified by a streaming readback. The
root manifest is the final remote publication marker. No local input is deleted.
Only explicit committed directories are accepted, so resumable preparation can
retain each shard before the complete corpus is ready.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

SCHEMA = "olmo-document-retention-v1"
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/data/olmo-dolma-v1_5/"
SUFFIXES = {".json", ".jsonl", ".u16", ".bin", ".md", ".txt", ".py", ".log"}
FORBIDDEN = {"wandb", "hf-cache", "__pycache__", "checkpoint", "checkpoints"}
MAX_FILES = 10000
MAX_BYTES = 64 * 1024**3
CHUNK_BYTES = 8 * 1024**2


def parse_prefix(prefix: str) -> tuple[str, str]:
    prefix = prefix.rstrip("/")
    if not prefix.startswith(PREFIX_ROOT):
        raise ValueError(f"Prefix must be under {PREFIX_ROOT}")
    suffix = prefix[len(PREFIX_ROOT):]
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*", suffix) is None:
        raise ValueError("Prefix must end in one safe <run>/<stage> pair")
    return tuple(prefix[5:].split("/", 1))


def file_digest(path: Path) -> dict:
    sha256, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return {"size_bytes": size, "sha256": sha256.hexdigest(),
            "md5_base64": base64.b64encode(md5.digest()).decode("ascii")}


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("A retention path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"Unsafe retention path: {value!r}")
    for part in path.parts:
        lower = part.lower()
        if (part.startswith(".") or lower in FORBIDDEN or
                any(word in lower for word in ("credential", "secret", "service-account", "service_account")) or
                lower.endswith((".partial", ".tmp", ".pending"))):
            raise ValueError(f"Unsafe or uncommitted retention path: {value!r}")
    return path


def _regular(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Retention input must be a regular file within input-dir: {path}")
    return path


def _verify_declared_files(directory: Path, manifest: dict, inventory: dict) -> set[str]:
    """Verify committed manifest digests; never trust an arbitrary directory list."""
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Committed shard/metadata manifest requires a nonempty files mapping")
    referenced = set()
    for name, expected in declared.items():
        relative = _relative(name)
        if not isinstance(expected, dict):
            raise ValueError("Manifest file entries require size_bytes and sha256")
        full_name = (directory / relative).as_posix()
        actual = inventory.get(full_name)
        if actual is None or any(actual[key] != expected.get(key) for key in ("size_bytes", "sha256")):
            raise ValueError(f"Committed manifest file differs: {full_name}")
        referenced.add(full_name)
    return referenced


def collect_committed(input_dir: Path) -> list[tuple[Path, str, dict]]:
    input_dir = Path(input_dir)
    if input_dir.is_symlink() or not input_dir.is_dir() or input_dir.name.startswith("."):
        raise ValueError("input-dir must be a real committed directory")
    _relative(input_dir.name)
    members, size, entries = [], 0, 0
    for directory, subdirs, files in os.walk(input_dir, followlinks=False):
        if Path(directory) == input_dir and ".prepare.lock" in files:
            lock = _regular(input_dir / ".prepare.lock", input_dir)
            if lock.stat().st_size != 0:
                raise ValueError("Operational preparation lock must be empty")
            files.remove(".prepare.lock")
        for name in sorted(subdirs + files):
            entries += 1
            if entries > MAX_FILES:
                raise ValueError("Too many retention entries; select one smaller committed stage")
            path = Path(directory) / name
            relative = path.relative_to(input_dir).as_posix()
            _relative(relative)
            if path.is_symlink():
                raise ValueError("Symlinks are never retained")
        subdirs.sort()
        for name in sorted(files):
            path = _regular(Path(directory) / name, input_dir)
            if path.suffix not in SUFFIXES:
                raise ValueError(f"Unsupported retention suffix: {path.name}")
            size += path.stat().st_size
            if size > MAX_BYTES:
                raise ValueError("Retention stage exceeds 64 GiB; retain individual shards")
            members.append((path, path.relative_to(input_dir).as_posix(), file_digest(path)))
    inventory = {name: digest for _, name, digest in members}
    if "manifest.json" not in inventory:
        raise ValueError("Require manifest.json from an explicitly committed directory")
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not str(manifest.get("schema", "")).startswith("olmo-"):
        raise ValueError("Unsupported committed manifest schema")
    if manifest.get("completed") is False or manifest.get("status") in {"running", "partial", "pending", "interrupted"}:
        raise ValueError("Input manifest explicitly describes an uncommitted stage")
    referenced = {"manifest.json"}
    if manifest.get("completed") is True:
        config = inventory.get("config.json")
        if config is None or config["sha256"] != manifest.get("config_sha256"):
            raise ValueError("Completed root config differs from its manifest")
        referenced.add("config.json")
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("Completed root requires committed shards")
        for shard in shards:
            if not isinstance(shard, dict):
                raise ValueError("Invalid committed shard entry")
            relative = _relative(shard.get("path"))
            name = (relative / "manifest.json").as_posix()
            if name not in inventory or inventory[name]["sha256"] != shard.get("manifest_sha256"):
                raise ValueError("Completed root shard manifest differs")
            child = json.loads((input_dir / name).read_text())
            if child.get("config_sha256") != manifest["config_sha256"]:
                raise ValueError("Shard config differs from completed root")
            referenced.add(name)
            referenced.update(_verify_declared_files(relative, child, inventory))
    else:
        referenced.update(_verify_declared_files(Path(), manifest, inventory))
    if referenced != inventory.keys():
        raise ValueError("Input contains files not committed by its manifest")
    # Nested shard commit markers also follow their data. Root marker is LAST.
    return sorted(members, key=lambda row: (row[1] == "manifest.json", Path(row[1]).name == "manifest.json", row[1]))


class _DigestSink(io.RawIOBase):
    """Discard streamed readback bytes after hashing; memory is bounded by a chunk."""
    def __init__(self):
        self.sha256 = hashlib.sha256()
        self.size = 0

    def writable(self):
        return True

    def write(self, data):
        self.sha256.update(data)
        self.size += len(data)
        return len(data)

    def tell(self):
        return self.size


def upload_verified(bucket, key: str, path: Path, expected: dict) -> dict:
    from google.api_core.exceptions import PreconditionFailed

    _, name = key.rsplit("/", 1)
    # validate the namespace and every member, including direct callers
    root_key = PREFIX_ROOT[5:].split("/", 1)[1]
    if not key.startswith(root_key):
        raise ValueError("Object is outside the document retention namespace")
    parts = key[len(root_key):].split("/", 2)
    if len(parts) != 3:
        raise ValueError("Object requires run/stage/member")
    parse_prefix(PREFIX_ROOT + "/".join(parts[:2]))
    _relative(parts[2])
    if Path(name).suffix not in SUFFIXES or bucket.name != "fast-chunks":
        raise ValueError("Disallowed retention object")
    if file_digest(path) != expected:
        raise ValueError("Local file changed before upload")
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
    if (int(blob.size) != expected["size_bytes"] or blob.md5_hash != expected["md5_base64"] or
            (blob.metadata or {}).get("sha256") != expected["sha256"] or
            (blob.metadata or {}).get("artifact_schema") != SCHEMA):
        raise ValueError(f"Existing/uploaded GCS object differs: {key}")
    # Setting chunk_size makes download_to_file use streamed ranged requests.
    # Each is pinned to the same generation; SHA256 and size cover the full file.
    blob.chunk_size = CHUNK_BYTES
    sink = _DigestSink()
    blob.download_to_file(sink, if_generation_match=blob.generation, raw_download=True, checksum=None)
    if sink.size != expected["size_bytes"] or sink.sha256.hexdigest() != expected["sha256"]:
        raise ValueError(f"Downloaded GCS object differs: {key}")
    if file_digest(path) != expected:
        raise ValueError("Local file changed during upload")
    return {"uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation), **expected,
            "verification": {"server_size": True, "server_md5": True,
                             "sha256_metadata": True, "download_sha256": True}}


def _write_once(path: Path, value: dict) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Receipt must not be a symlink")
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError("Existing retention receipt differs")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".retention-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def retain(args, *, bucket=None) -> dict:
    bucket_name, key = parse_prefix(args.prefix)
    input_dir, receipt = Path(args.input_dir), Path(args.receipt)
    if receipt.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("Receipt must be outside input-dir")
    members = collect_committed(input_dir)
    prefix = f"gs://{bucket_name}/{key}"
    if args.dry_run:
        return {"schema": SCHEMA, "status": "dry_run", "prefix": prefix,
                "files": [{"path": name, **digest} for _, name, digest in members], "uploaded": False}
    if bucket is None:
        from google.cloud import storage
        bucket = storage.Client().bucket(bucket_name)
    if bucket.name != bucket_name:
        raise ValueError("Provided bucket differs from prefix")
    # An already published, different manifest is a different dataset version;
    # reject it before adding any new objects under its immutable prefix.
    manifest_path, manifest_name, manifest_digest = members[-1]
    if bucket.get_blob(key + "/" + manifest_name) is not None:
        upload_verified(bucket, key + "/" + manifest_name, manifest_path, manifest_digest)
    # Preflight EVERY local file before uploading any bytes. The root commit
    # marker only becomes discoverable after all of its dependencies verify.
    objects = [upload_verified(bucket, key + "/" + name, path, digest) for path, name, digest in members]
    result = {"schema": SCHEMA, "status": "verified", "prefix": prefix,
              "objects": objects, "publication_marker": prefix + "/manifest.json",
              "local_files_deleted": False}
    _write_once(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    print(json.dumps(retain(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
