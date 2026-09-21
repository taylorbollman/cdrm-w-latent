#!/usr/bin/env python3
"""Retain verified Stage A evidence and its standalone native checkpoint on GCS."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import CHECKPOINT_FILENAME, validate_prepared_manifest, write_json


PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-import/"
_FORBIDDEN_PARTS = {"hf-cache", "wandb", "__pycache__", ".git", ".docker-home"}
_ARTIFACT_JSON = (
    "manifest.json", "source_manifest.json", "tokenizer_manifest.json", "checkpoint_inspection.json",
)
_VALIDATED_SOURCE_FILES = {
    "cdrm/pretrained/openelm.py", "cdrm/pretrained/reference.py",
    "cdrm/pretrained/artifacts.py", "scripts/openelm_validate.py",
}
_PROJECT_FILES = (
    "cdrm/__init__.py", "cdrm/pretrained/__init__.py",
    "cdrm/pretrained/openelm.py", "cdrm/pretrained/reference.py", "cdrm/pretrained/artifacts.py",
    "scripts/openelm_prepare.py", "scripts/openelm_validate.py", "scripts/openelm_retain.py",
    "scripts/experiment_tracking.py", "docker/requirements-docker.txt",
    "tests/test_openelm_model.py", "tests/test_openelm_reference.py", "tests/test_openelm_artifacts.py",
    "tests/test_openelm_retain.py",
)
_REPORT_SUFFIXES = {".md", ".json", ".txt", ".csv", ".pdf", ".png", ".svg"}


def parse_prefix(prefix: str) -> tuple[str, str]:
    prefix = prefix.rstrip("/") + "/"
    if not prefix.startswith(PREFIX_ROOT):
        raise ValueError(f"Retention prefix must begin with {PREFIX_ROOT}")
    timestamp = prefix[len(PREFIX_ROOT):].rstrip("/")
    if re.fullmatch(r"\d{8}T\d{6}Z", timestamp) is None:
        raise ValueError("Retention prefix must end in one YYYYMMDDTHHMMSSZ timestamp")
    bucket, path = prefix[5:].split("/", 1)
    return bucket, path.rstrip("/")


def file_digest(path: Path) -> dict[str, Any]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            sha256.update(block)
            md5.update(block)
    return {"size_bytes": size, "sha256": sha256.hexdigest(),
            "md5_base64": base64.b64encode(md5.digest()).decode("ascii")}


def _safe_member(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if (
        not parts or PurePosixPath(relative).is_absolute() or ".." in parts
        or any(part in _FORBIDDEN_PARTS or part.startswith(".env") for part in parts)
    ):
        raise ValueError(f"Disallowed retention member: {relative}")
    path = root.joinpath(*parts)
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Retention member must be a regular file within its root: {relative}")
    return path


def collect_evidence(
    artifacts: Path, validation: Path, manifest: dict[str, Any], *,
    project_root: Path = ROOT, report_dir: Path | None = None,
) -> list[tuple[Path, str]]:
    """Select known evidence only; never recursively archive an experiment root."""
    members: dict[str, Path] = {}

    def add(root: Path, relative: str, archive_root: str) -> None:
        path = _safe_member(root, relative)
        members[f"{archive_root}/{relative}"] = path

    for relative in _ARTIFACT_JSON:
        if (artifacts / relative).is_file():
            add(artifacts, relative, "artifacts")
    checkpoint_relative = manifest["checkpoint"]["path"]
    if checkpoint_relative != f"checkpoint/{CHECKPOINT_FILENAME}":
        raise ValueError("Unexpected native checkpoint path")
    add(artifacts, checkpoint_relative + ".artifact.json", "artifacts")
    for relative in sorted(manifest["sources"]["artifacts"]):
        if not relative.startswith("sources/"):
            raise ValueError("Source record is outside the source whitelist")
        add(artifacts, relative, "artifacts")
        add(artifacts, relative + ".artifact.json", "artifacts")
    if manifest["tokenizer"]["path"] != "tokenizer/tokenizer.model":
        raise ValueError("Unexpected tokenizer path")
    add(artifacts, "tokenizer/tokenizer.model", "artifacts")
    add(validation, "report.json", "validation")
    for relative in _PROJECT_FILES:
        add(project_root, relative, "project")
    source_snapshot = project_root / "cdrm/pretrained/_corenet_reference"
    snapshot = json.loads((source_snapshot / "manifest.json").read_text())
    for relative in ["manifest.json", "README.md", *(item["file"] for item in snapshot["files"])]:
        add(project_root, f"cdrm/pretrained/_corenet_reference/{relative}", "project")
    if report_dir is not None:
        for path in sorted(report_dir.rglob("*")):
            relative = path.relative_to(report_dir)
            if (
                not path.is_file() or path.suffix not in _REPORT_SUFFIXES
                or any(part in _FORBIDDEN_PARTS or part.startswith(".env") for part in relative.parts)
            ):
                continue
            add(report_dir, relative.as_posix(), "report")
    return [(path, name) for name, path in sorted(members.items())]


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def build_evidence_archive(
    destination: Path, members: list[tuple[Path, str]], restore_text: str,
) -> list[dict[str, Any]]:
    """Create a deterministic small archive, checking concurrent source changes."""
    inventory = [{"path": name, **file_digest(path)} for path, name in members]
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for (path, name), record in zip(members, inventory):
                data = path.read_bytes()
                if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise RuntimeError(f"Evidence changed while archiving: {name}")
                _add_bytes(archive, name, data)
            _add_bytes(archive, "RESTORE.md", restore_text.encode("utf-8"))
            _add_bytes(archive, "evidence-members.json", (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for (path, name), record in zip(members, inventory):
        if file_digest(path) != {key: record[key] for key in ("size_bytes", "sha256", "md5_base64")}:
            raise RuntimeError(f"Evidence changed during retention: {name}")
    if destination.exists():
        if file_digest(destination) != file_digest(temporary):
            temporary.unlink()
            raise FileExistsError("Existing evidence archive differs; choose a new output directory and timestamp")
        temporary.unlink()
    else:
        temporary.replace(destination)
    return inventory


def _check_remote(blob: Any, expected: dict[str, Any]) -> None:
    if (
        int(blob.size) != expected["size_bytes"]
        or blob.md5_hash != expected["md5_base64"]
        or (blob.metadata or {}).get("sha256") != expected["sha256"]
    ):
        raise ValueError(f"Existing/uploaded GCS object differs from expected bytes: {blob.name}")


def upload_verified(bucket: Any, key: str, path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Create once, or reuse an object whose server MD5/size and SHA metadata match."""
    from google.api_core.exceptions import PreconditionFailed

    blob = bucket.get_blob(key)
    if blob is None:
        blob = bucket.blob(key)
        blob.metadata = {"sha256": expected["sha256"], "artifact_schema": "openelm-import-retention-v1"}
        try:
            blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
        except PreconditionFailed:
            # A competing exact upload is safe; a different object is rejected.
            blob = bucket.get_blob(key)
            if blob is None:
                raise
        blob.reload()
    _check_remote(blob, expected)
    return {
        "uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation),
        **expected,
        "verification": "GCS size and server MD5 match local bytes; SHA256 matches object metadata",
    }


def _restore_text(prefix: str, checkpoint_record: dict[str, Any]) -> str:
    return f"""# Restore the OpenELM Stage A evidence

The 4.3GB native checkpoint is a standalone object, intentionally absent from
this compressed archive. The source, tokenizer, manifests and validation report
are retained here with relative paths. `project/` contains a bounded source
overlay for the main repository; it is not a complete repository checkout.

1. Download `{prefix}/evidence.tar.gz`, `retention-manifest.json`, and
   `storage-receipt.json` from the same prefix. Check their recorded checksums.
2. Extract this archive into a new directory. Verify the SHA256 and byte counts
   in `evidence-members.json` before using the files.
3. Download `{prefix}/checkpoint/{CHECKPOINT_FILENAME}` into
   `artifacts/checkpoint/{CHECKPOINT_FILENAME}`. Expected SHA256:
   `{checkpoint_record['sha256']}`; expected bytes: {checkpoint_record['size_bytes']}.
4. Use the recorded source revisions and the project source overlay in the
   project Docker environment. Run `validate_prepared_manifest("artifacts")`
   from `cdrm.pretrained.artifacts` to verify the preparation offline.
5. `validation/report.json` contains the frozen native configuration, token
   fixtures, runtime, W&B identity, source hashes and completed fidelity checks.
   To repeat validation, use `scripts/openelm_validate.py --help` in the project
   container with a GPU; do not run GPU checks in the host shell.

No HF cache, W&B local cache, credentials, unrelated runtime history, optimizer
states or synthetic-training checkpoints are part of this archive.
"""


def retain(args: argparse.Namespace) -> dict[str, Any]:
    bucket_name, prefix_key = parse_prefix(args.prefix)
    prefix = f"gs://{bucket_name}/{prefix_key}"
    artifacts, validation, output = args.artifacts.resolve(), args.validation.resolve(), args.output_dir.resolve()
    report_dir = args.report_dir.resolve() if args.report_dir is not None else None
    if output.is_relative_to(validation) or (report_dir is not None and output.is_relative_to(report_dir)):
        raise ValueError("Retention output must be outside validation and report directories")
    manifest = validate_prepared_manifest(artifacts)
    for key, name in (("sources", "source_manifest.json"), ("tokenizer", "tokenizer_manifest.json")):
        if json.loads(_safe_member(artifacts, name).read_text()) != manifest[key]:
            raise ValueError(f"Component manifest differs from the verified preparation: {name}")
    report = json.loads(_safe_member(validation, "report.json").read_text())
    if (report.get("schema") != "openelm-import-validation-v1" or report.get("status") != "passed"
            or not report.get("finished_utc")):
        raise ValueError("Require a completed, passing Stage A validation report")
    if report.get("checkpoint") != manifest["checkpoint"]:
        raise ValueError("Validation used a different checkpoint provenance record")
    for key in ("sources", "tokenizer"):
        if report.get("artifacts_manifest", {}).get(key) != manifest[key]:
            raise ValueError(f"Validation used different prepared {key}")
    current_reference = json.loads(_safe_member(ROOT, "cdrm/pretrained/_corenet_reference/manifest.json").read_text())
    if report.get("native_reference_sources") != current_reference:
        raise ValueError("Native oracle snapshot differs from the validated source manifest")
    if set(report.get("source_hashes", {})) != _VALIDATED_SOURCE_FILES:
        raise ValueError("Validation source inventory differs from the Stage A contract")
    for relative, expected in report["source_hashes"].items():
        if file_digest(_safe_member(ROOT, relative))["sha256"] != expected:
            raise ValueError(f"Validated source changed after the reported checks: {relative}")
    checkpoint = _safe_member(artifacts, manifest["checkpoint"]["path"])
    checkpoint_digest = file_digest(checkpoint)
    if any(checkpoint_digest[key] != manifest["checkpoint"][key] for key in ("size_bytes", "sha256")):
        raise ValueError("Checkpoint changed after preparation verification")
    output.mkdir(parents=True, exist_ok=True)
    members = collect_evidence(artifacts, validation, manifest, report_dir=report_dir)
    archive = output / "evidence.tar.gz"
    inventory = build_evidence_archive(archive, members, _restore_text(prefix, manifest["checkpoint"]))
    archive_digest = file_digest(archive)
    retention_manifest = {
        "schema": "openelm-import-retention-v1", "prefix": prefix,
        "checkpoint": {"object": f"checkpoint/{CHECKPOINT_FILENAME}", **checkpoint_digest},
        "evidence": {"object": archive.name, **archive_digest, "members": inventory},
        "validation_status": report["status"], "validation_finished_utc": report.get("finished_utc"),
        "checkpoint_compressed_in_evidence": False,
    }
    manifest_path = output / "retention-manifest.json"
    write_json(manifest_path, retention_manifest)
    from google.cloud import storage

    bucket = storage.Client().bucket(bucket_name)
    objects = []
    for name, path, digest in (
        (f"checkpoint/{CHECKPOINT_FILENAME}", checkpoint, checkpoint_digest),
        (archive.name, archive, archive_digest),
        (manifest_path.name, manifest_path, file_digest(manifest_path)),
    ):
        objects.append(upload_verified(bucket, f"{prefix_key}/{name}", path, digest))
    receipt = {"schema": "openelm-import-storage-receipt-v1", "status": "verified", "objects": objects}
    receipt_path = output / "storage-receipt.json"
    write_json(receipt_path, receipt)
    receipt_object = upload_verified(bucket, f"{prefix_key}/{receipt_path.name}", receipt_path, file_digest(receipt_path))
    result = {**receipt, "receipt_object": receipt_object}
    write_json(output / "upload-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    result = retain(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
