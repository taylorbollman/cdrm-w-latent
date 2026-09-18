#!/usr/bin/env python3
"""Retain the A5 BF16 repeat and its explicit report artifacts on GCS."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--label", default="final")
    args = parser.parse_args()
    if not args.prefix.startswith("gs://fast-chunks/cdrm-w-latent/rt-a5/"):
        raise ValueError("Require the declared experiment retention prefix")
    runtime = args.runtime.resolve()
    retention = runtime / "retention"
    retention.mkdir(exist_ok=True)
    archive = retention / f"{args.label}-evidence.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    files = []
    for role, root in (("runtime", runtime), ("report", args.report)):
        if root is None:
            continue
        root = root.resolve()
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if (not path.is_file() or path.is_symlink() or
                set(rel.parts) & {"wandb", "retention", "__pycache__", ".git"} or
                path.name.startswith(".env") or path.suffix in (".tmp", ".pyc")):
                continue
            files.append((path, str(Path(role) / rel)))
    if not files:
        raise ValueError("No artifacts to retain")
    members = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path).hexdigest()}
               for path, name in files]
    with tarfile.open(archive, "w:gz") as stream:
        for path, name in files:
            stream.add(path, arcname=name, recursive=False)
    # Ensure no source/checkpoint changed while building the evidence archive.
    for (path, _), record in zip(files, members):
        if digest(path).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Artifact changed during retention: {record['path']}")
    sha = digest(archive).hexdigest()
    md5 = base64.b64encode(digest(archive, "md5").digest()).decode()
    from google.cloud import storage
    client = storage.Client()
    bucket_name, prefix = args.prefix[5:].split("/", 1)
    blob = client.bucket(bucket_name).blob(prefix.rstrip("/") + "/" + archive.name)
    blob.metadata = {"sha256": sha, "artifact_schema": "rt-a5-bf16-evidence-v1"}
    blob.upload_from_filename(str(archive), if_generation_match=0, checksum="auto")
    blob.reload()
    if int(blob.size) != archive.stat().st_size or blob.md5_hash != md5 or blob.metadata.get("sha256") != sha:
        raise AssertionError("GCS size/checksum/metadata verification failed")
    receipt = {"schema": "rt-a5-bf16-retention-v1", "status": "verified",
               "uri": f"gs://{bucket_name}/{blob.name}", "generation": blob.generation,
               "bytes": blob.size, "sha256": sha, "md5_base64": md5,
               "verification": "remote size and MD5 equal local content; SHA256 recorded in remote metadata",
               "members": members}
    path = retention / f"{args.label}-receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_blob = client.bucket(bucket_name).blob(prefix.rstrip("/") + "/" + path.name)
    receipt_blob.upload_from_filename(str(path), if_generation_match=0, checksum="auto")
    print(json.dumps({k: v for k, v in receipt.items() if k != "members"}), flush=True)


if __name__ == "__main__":
    main()
