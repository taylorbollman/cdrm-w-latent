#!/usr/bin/env python3
"""Retain Stage B RT evidence while reusing the verified Stage A checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import CHECKPOINT_FILENAME, validate_prepared_manifest, write_json
from scripts import openelm_retain as stage_a


PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/openelm-rt-reference/"
_VALIDATED_SOURCE_FILES = {
    "cdrm/pretrained/openelm.py", "cdrm/pretrained/reference.py",
    "cdrm/pretrained/artifacts.py", "cdrm/pretrained/recurrent.py",
    "cdrm/pretrained/recurrent_oracle.py", "scripts/openelm_validate.py",
    "scripts/openelm_rt_validate.py",
}
_PROJECT_FILES = (
    "cdrm/pretrained/recurrent.py", "cdrm/pretrained/recurrent_oracle.py",
    "scripts/openelm_rt_validate.py", "scripts/openelm_rt_retain.py",
    "tests/test_openelm_recurrent.py", "tests/test_openelm_recurrent_oracle.py",
    "tests/test_openelm_rt_retain.py", "tests/test_openelm_rt_validation.py",
)


def parse_prefix(prefix: str) -> tuple[str, str]:
    prefix = prefix.rstrip("/") + "/"
    if not prefix.startswith(PREFIX_ROOT):
        raise ValueError(f"Retention prefix must begin with {PREFIX_ROOT}")
    if re.fullmatch(r"\d{8}T\d{6}Z/", prefix[len(PREFIX_ROOT):]) is None:
        raise ValueError("Retention prefix must end in one YYYYMMDDTHHMMSSZ timestamp")
    bucket, key = prefix[5:].split("/", 1)
    return bucket, key.rstrip("/")


def checkpoint_reference(receipt: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Select the exact retained source; reject an unrelated or ambiguous receipt."""
    if receipt.get("schema") != "openelm-import-storage-receipt-v1" or receipt.get("status") != "verified":
        raise ValueError("Require a verified Stage A checkpoint storage receipt")
    candidates = [record for record in receipt.get("objects", [])
                  if record.get("uri", "").endswith(f"/checkpoint/{CHECKPOINT_FILENAME}")]
    if len(candidates) != 1:
        raise ValueError("Stage A receipt must contain exactly one native checkpoint")
    record = candidates[0]
    # This validates both the named bucket and the complete Stage A timestamp.
    stage_a.parse_prefix(record["uri"].removesuffix(f"/checkpoint/{CHECKPOINT_FILENAME}"))
    if any(record.get(key) != checkpoint.get(key) for key in ("sha256", "size_bytes")):
        raise ValueError("Retained checkpoint differs from the validated checkpoint")
    if (re.fullmatch(r"[0-9]+", str(record.get("generation", ""))) is None
            or not record.get("md5_base64")):
        raise ValueError("Checkpoint receipt lacks a generation or server MD5")
    return {key: record[key] for key in ("uri", "generation", "size_bytes", "sha256", "md5_base64")}


def verify_checkpoint_reference(bucket: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Read and verify the existing checkpoint object without uploading it again."""
    bucket_name, key = record["uri"][5:].split("/", 1)
    if bucket.name != bucket_name:
        raise ValueError("Checkpoint bucket differs from the retained reference")
    blob = bucket.get_blob(key)
    if blob is None:
        raise ValueError("Retained Stage A checkpoint object is missing")
    stage_a._check_remote(blob, record)
    if str(blob.generation) != str(record["generation"]):
        raise ValueError("Retained checkpoint generation changed since Stage A")
    return {**record, "reused_without_upload": True,
            "verification": "Existing GCS generation, size, server MD5 and SHA256 metadata match the Stage A receipt"}


def upload_verified(bucket: Any, key: str, path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Create Stage B evidence once, verifying any existing exact object."""
    from google.api_core.exceptions import PreconditionFailed

    blob = bucket.get_blob(key)
    if blob is None:
        blob = bucket.blob(key)
        blob.metadata = {"sha256": expected["sha256"], "artifact_schema": "openelm-rt-reference-retention-v1"}
        try:
            blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
        except PreconditionFailed:
            blob = bucket.get_blob(key)
            if blob is None:
                raise
        blob.reload()
    stage_a._check_remote(blob, expected)
    return {"uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation), **expected,
            "verification": "GCS size and server MD5 match local bytes; SHA256 matches object metadata"}


def validate_report(artifacts: Path, validation: Path, *, project_root: Path = ROOT) -> tuple[dict, dict]:
    """Verify the finished evidence and its source snapshot before network writes."""
    manifest = validate_prepared_manifest(artifacts)
    report = json.loads(stage_a._safe_member(validation, "report.json").read_text())
    if (report.get("schema") != "openelm-rt-reference-validation-v1"
            or report.get("status") != "passed" or not report.get("finished_utc")):
        raise ValueError("Require a completed, passing Stage B validation report")
    if report.get("checkpoint") != manifest["checkpoint"]:
        raise ValueError("Validation used a different checkpoint provenance record")
    if report.get("artifacts_manifest") != manifest:
        raise ValueError("Validation used different prepared artifacts")
    for key, name in (("sources", "source_manifest.json"), ("tokenizer", "tokenizer_manifest.json")):
        if json.loads(stage_a._safe_member(artifacts, name).read_text()) != manifest[key]:
            raise ValueError(f"Component manifest differs from the verified preparation: {name}")
    native_manifest = json.loads(stage_a._safe_member(project_root, "cdrm/pretrained/_corenet_reference/manifest.json").read_text())
    if report.get("native_reference_sources") != native_manifest:
        raise ValueError("Native oracle snapshot differs from the validated source manifest")
    if set(report.get("source_hashes", {})) != _VALIDATED_SOURCE_FILES:
        raise ValueError("Validation source inventory differs from the Stage B contract")
    for relative, digest in report["source_hashes"].items():
        if stage_a.file_digest(stage_a._safe_member(project_root, relative))["sha256"] != digest:
            raise ValueError(f"Validated source changed after the reported checks: {relative}")
    for item in native_manifest["files"]:
        source = stage_a._safe_member(project_root, f"cdrm/pretrained/_corenet_reference/{item['file']}")
        digest = stage_a.file_digest(source)
        if digest["sha256"] != item["sha256"] or digest["size_bytes"] != item["bytes"]:
            raise ValueError(f"Native oracle source changed after validation: {item['file']}")
    return manifest, report


def collect_evidence(
    artifacts: Path, validation: Path, manifest: dict[str, Any], *,
    checkpoint_receipt: Path, project_root: Path = ROOT, report_dir: Path | None = None,
) -> list[tuple[Path, str]]:
    members = stage_a.collect_evidence(artifacts, validation, manifest,
                                      project_root=project_root, report_dir=report_dir)
    for relative in _PROJECT_FILES:
        members.append((stage_a._safe_member(project_root, relative), f"project/{relative}"))
    receipt = stage_a._safe_member(checkpoint_receipt.parent, checkpoint_receipt.name)
    members.append((receipt, "provenance/stage-a-storage-receipt.json"))
    return sorted(members, key=lambda item: item[1])


def _restore_text(prefix: str, checkpoint: dict[str, Any]) -> str:
    return f"""# Restore OpenELM Stage B RT reference evidence

This evidence archive contains a bounded project source overlay, pinned native
reference sources, tokenizer, manifests and the actual-checkpoint RT report.
It does not contain or duplicate the 4.3GB Stage A source checkpoint.

1. Download `{prefix}/evidence.tar.gz`, `retention-manifest.json` and
   `storage-receipt.json`; verify their recorded checksums before extraction.
2. Check extracted files against `evidence-members.json`.
3. Download the existing object `{checkpoint['uri']}` at generation
   `{checkpoint['generation']}` to `artifacts/checkpoint/{CHECKPOINT_FILENAME}`.
   Expected SHA256: `{checkpoint['sha256']}`; bytes: {checkpoint['size_bytes']}.
   `provenance/stage-a-storage-receipt.json` records its original retention.
4. Overlay `project/` on the matching repository revision in the project Docker
   environment. Run `validate_prepared_manifest("artifacts")` to verify prepared
   source/model/tokenizer bytes. Use `scripts/openelm_rt_validate.py --help` for
   GPU validation; do not execute CUDA work in the host shell.
5. `validation/report.json` records checkpoint, configuration, token fixtures,
   source hashes, runtime, W&B identity and the bounded scope. Passing this
   reference milestone does not clear tiled execution, long contexts, optimizer
   adaptation, FBT, NextLat or training quality.

No credentials, runtime cache, unrelated histories or trained checkpoints are
part of this archive. The original native checkpoint is reused without upload.
"""


def retain(args: argparse.Namespace) -> dict[str, Any]:
    bucket_name, prefix_key = parse_prefix(args.prefix)
    prefix = f"gs://{bucket_name}/{prefix_key}"
    artifacts, validation, output = args.artifacts.resolve(), args.validation.resolve(), args.output_dir.resolve()
    report_dir = args.report_dir.resolve() if args.report_dir is not None else None
    if output.is_relative_to(validation) or (report_dir is not None and output.is_relative_to(report_dir)):
        raise ValueError("Retention output must be outside validation and report directories")
    manifest, report = validate_report(artifacts, validation)
    checkpoint_receipt = args.checkpoint_receipt.resolve()
    receipt_path = stage_a._safe_member(checkpoint_receipt.parent, checkpoint_receipt.name)
    checkpoint = checkpoint_reference(json.loads(receipt_path.read_text()), manifest["checkpoint"])
    members = collect_evidence(artifacts, validation, manifest, checkpoint_receipt=receipt_path,
                               report_dir=report_dir)
    from google.cloud import storage

    bucket = storage.Client().bucket(bucket_name)
    checkpoint = verify_checkpoint_reference(bucket, checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "evidence.tar.gz"
    inventory = stage_a.build_evidence_archive(archive, members, _restore_text(prefix, checkpoint))
    archive_digest = stage_a.file_digest(archive)
    retention_manifest = {
        "schema": "openelm-rt-reference-retention-v1", "prefix": prefix,
        "checkpoint_reference": checkpoint,
        "evidence": {"object": archive.name, **archive_digest, "members": inventory},
        "checkpoint_compressed_in_evidence": False, "checkpoint_uploaded": False,
        "validation_status": report["status"], "validation_finished_utc": report["finished_utc"],
    }
    manifest_path = output / "retention-manifest.json"
    write_json(manifest_path, retention_manifest)
    objects = [upload_verified(bucket, f"{prefix_key}/{path.name}", path, digest)
               for path, digest in ((archive, archive_digest), (manifest_path, stage_a.file_digest(manifest_path)))]
    receipt = {"schema": "openelm-rt-reference-storage-receipt-v1", "status": "verified",
               "checkpoint_reference": checkpoint, "objects": objects}
    receipt_path = output / "storage-receipt.json"
    write_json(receipt_path, receipt)
    receipt_object = upload_verified(bucket, f"{prefix_key}/{receipt_path.name}", receipt_path,
                                     stage_a.file_digest(receipt_path))
    result = {**receipt, "receipt_object": receipt_object}
    write_json(output / "upload-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--report-dir", type=Path)
    print(json.dumps(retain(parser.parse_args()), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
