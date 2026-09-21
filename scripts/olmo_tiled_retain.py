#!/usr/bin/env python3
"""Retain OLMo O2 tiled evidence, reusing the immutable O1 source checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.olmo_artifacts import (
    CHECKPOINT_FILENAME, CHECKPOINT_SHA256, CHECKPOINT_SIZE, FILE_SPECS,
    MANIFEST_FILENAME, validate_prepared_manifest,
)
from scripts import olmo_retain as o1
from scripts.openelm_retain import _safe_member, _check_remote, file_digest, build_evidence_archive

PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-tiled-rt/"
O1_CHECKPOINT_URI = (
    "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-step60000/"
    "20260921T202457Z/checkpoint/model.safetensors"
)
O1_CHECKPOINT_GENERATION = "1790022395832236"
O1_CHECKPOINT_MD5 = "Ewj7L/fkQ1sk59gBObku4Q=="
REPORT_SCHEMAS = {"validation": "olmo-tiled-rt-validation-v1", "profile": "olmo-tiled-rt-profile-v1"}
VALIDATION_SOURCE_FILES = o1.SOURCE_FILES["rt"] | {
    "cdrm/pretrained/olmo_tiled.py", "scripts/olmo_tiled_validate.py",
    "scripts/olmo_tiled_roundoff.py",
}
SOURCE_FILES = {"validation": VALIDATION_SOURCE_FILES,
                "profile": o1.SOURCE_FILES["rt"] | {"cdrm/pretrained/olmo_tiled.py", "scripts/olmo_tiled_profile.py"}}
EXTRA_PROJECT_FILES = (*o1.EXTRA_PROJECT_FILES, "scripts/olmo_tiled_retain.py",
                       "docs/fbt-rt-nextlat-handoff.md", "docs/olmo1b-tiled-rt-usage.md", "AGENTS.md")
REPORT_FILES = {"results.md", "protocol.md", "test-results.txt", "validation-summary.json", "storage-receipt.json", "initial-validation-01.json", "initial-validation-01-source.patch"}


def parse_prefix(prefix: str) -> tuple[str, str]:
    prefix = prefix.rstrip("/")
    if not prefix.startswith(PREFIX_ROOT) or re.fullmatch(r"\d{8}T\d{6}Z", prefix[len(PREFIX_ROOT):]) is None:
        raise ValueError(f"Retention requires one timestamp under {PREFIX_ROOT}")
    return tuple(prefix[5:].split("/", 1))


def checkpoint_reference(receipt: dict, checkpoint: dict) -> dict:
    """Accept only the selected O1 checkpoint object and its recorded generation."""
    if receipt.get("schema") != "olmo-o1-storage-receipt-v1" or receipt.get("status") != "verified":
        raise ValueError("Require a verified O1 checkpoint storage receipt")
    candidates = [item for item in receipt.get("objects", [])
                  if item.get("uri", "").endswith("/checkpoint/" + CHECKPOINT_FILENAME)]
    if len(candidates) != 1:
        raise ValueError("O1 receipt must identify exactly one source checkpoint")
    record = candidates[0]
    required = {"uri": O1_CHECKPOINT_URI, "generation": O1_CHECKPOINT_GENERATION,
                "size_bytes": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256, "md5_base64": O1_CHECKPOINT_MD5}
    if any(record.get(k) != v for k, v in required.items()):
        raise ValueError("O1 checkpoint reference differs from immutable selected source")
    if checkpoint.get("path") != "native/" + CHECKPOINT_FILENAME or any(
        checkpoint.get(k) != required[k] for k in ("size_bytes", "sha256")
    ):
        raise ValueError("Validated local checkpoint differs from the reused O1 checkpoint")
    return required


def verify_checkpoint_reference(bucket: Any, record: dict) -> dict:
    """Verify the retained checkpoint read-only; never create/upload a model object."""
    bucket_name, key = record["uri"][5:].split("/", 1)
    if bucket.name != bucket_name:
        raise ValueError("Checkpoint bucket differs from selected O1 reference")
    blob = bucket.get_blob(key)
    if blob is None:
        raise ValueError("Retained O1 checkpoint object is missing")
    _check_remote(blob, record)
    if str(blob.generation) != record["generation"]:
        raise ValueError("Retained O1 checkpoint generation changed")
    return {**record, "reused_without_upload": True,
            "verification": "Existing GCS generation, size, server MD5 and SHA256 metadata match selected O1 receipt"}


def validate_reports(artifacts: Path, validation: Path, profile: Path, *, project_root=ROOT):
    """Require both completed GPU reports and the exact validated source inventory."""
    from cdrm.pretrained.olmo_reference import verify_olmo_reference_sources
    manifest = validate_prepared_manifest(artifacts)
    reference = verify_olmo_reference_sources()
    reports = {}
    for kind, directory in (("validation", validation), ("profile", profile)):
        report = json.loads(_safe_member(directory, "report.json").read_text())
        if report.get("schema") != REPORT_SCHEMAS[kind] or report.get("status") != "passed" or not report.get("finished_utc"):
            raise ValueError(f"Require a completed, passing O2 {kind} report")
        if report.get("checkpoint") != manifest["checkpoint"] or report.get("artifacts_manifest") != manifest:
            raise ValueError(f"O2 {kind} checkpoint/artifact provenance differs")
        if report.get("native_reference_sources") != reference:
            raise ValueError(f"O2 {kind} native source provenance differs")
        runtime = report.get("runtime", {})
        if not all(runtime.get(key) for key in ("gpu", "cuda", "torch")):
            raise ValueError(f"O2 {kind} report lacks its actual GPU runtime")
        hashes = report.get("source_hashes", {})
        if set(hashes) != SOURCE_FILES[kind]:
            raise ValueError(f"O2 {kind} source inventory differs from the exact whitelist")
        for relative, digest in hashes.items():
            if file_digest(_safe_member(project_root, relative))["sha256"] != digest:
                raise ValueError(f"Validated source changed after O2 {kind}: {relative}")
        reports[kind] = report
    return manifest, reports, reference


def collect_evidence(artifacts, validation, profile, manifest, reports, reference, *,
                     checkpoint_receipt: Path, project_root=ROOT, report_dir=None):
    members = {}
    def add(root, relative, category):
        members[f"{category}/{relative}"] = _safe_member(root, relative)
    for name in (MANIFEST_FILENAME, "checkpoint-inspection.json"):
        add(artifacts, name, "artifacts")
    for name in FILE_SPECS:
        if name != CHECKPOINT_FILENAME:
            add(artifacts, f"native/{name}", "artifacts")
    add(validation, "report.json", "validation")
    add(profile, "report.json", "profile")
    project_files = set(EXTRA_PROJECT_FILES)
    for kind, report in reports.items():
        if set(report.get("source_hashes", {})) != SOURCE_FILES[kind]:
            raise ValueError("Evidence source inventory differs from the exact whitelist")
        project_files.update(report["source_hashes"])
    project_files.update(str(p.relative_to(project_root)) for p in (project_root / "tests").glob("test_olmo_*.py"))
    for name in sorted(project_files):
        add(project_root, name, "project")
    for name in ("manifest.json", "README.md", *(item["file"] for item in reference["files"])):
        add(project_root, "cdrm/pretrained/_olmo_reference/" + name, "project")
    receipt = _safe_member(checkpoint_receipt.parent, checkpoint_receipt.name)
    members["provenance/o1-storage-receipt.json"] = receipt
    if report_dir is not None:
        for path in sorted(report_dir.iterdir()):
            if path.name in REPORT_FILES:
                add(report_dir, path.name, "report")
    return [(path, name) for name, path in sorted(members.items())]


def upload_verified(bucket: Any, key: str, path: Path, expected: dict) -> dict:
    """Create immutable evidence; a conflicting remote object is never overwritten."""
    from google.api_core.exceptions import PreconditionFailed
    if key.endswith("/" + CHECKPOINT_FILENAME):
        raise ValueError("O2 retention must not upload a checkpoint")
    blob = bucket.get_blob(key)
    if blob is None:
        blob = bucket.blob(key)
        blob.metadata = {"sha256": expected["sha256"], "artifact_schema": "olmo-tiled-rt-retention-v1"}
        try:
            blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
        except PreconditionFailed:
            blob = bucket.get_blob(key)
            if blob is None:
                raise
        blob.reload()
    _check_remote(blob, expected)
    return {"uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation), **expected,
            "verification": "GCS size/server MD5 and SHA256 metadata match local evidence bytes"}


def _restore_text(prefix: str, checkpoint: dict) -> str:
    return f"""# Restore OLMo O2 tiled RT evidence

Download `{prefix}/evidence.tar.gz`, `retention-manifest.json` and
`storage-receipt.json`. Verify checksums before extraction and compare
`evidence-members.json`. Restore `project/` onto the recorded repository.

The source checkpoint is reused, not embedded in the archive or reuploaded:
`{checkpoint['uri']}` at generation `{checkpoint['generation']}`.
Download it to `artifacts/native/{CHECKPOINT_FILENAME}`; expected SHA256
`{checkpoint['sha256']}`, {checkpoint['size_bytes']} bytes.
`provenance/o1-storage-receipt.json` records its original retention.

Use `validate_prepared_manifest('artifacts')` to rehash complete source bytes.
Run `scripts/olmo_tiled_validate.py` and `scripts/olmo_tiled_profile.py` only in
the project's GPU container. Reports at `validation/report.json` and
`profile/report.json` contain checkpoint identity, source hashes, runtime and
bounded fixtures. Saved reports describe the tested precision/scope; they do
not imply long-run training quality, arbitrary batch/length capacity or any
FBT/NextLat implementation. No optimizer step or adapted checkpoint is implied.
"""


def retain(args):
    bucket_name, prefix_key = parse_prefix(args.prefix)
    prefix = f"gs://{bucket_name}/{prefix_key}"
    artifacts, validation, profile, output = [Path(value).resolve() for value in
                                              (args.artifacts, args.validation, args.profile, args.output_dir)]
    report_dir = None if args.report_dir is None else Path(args.report_dir).resolve()
    if any(output.is_relative_to(path) for path in (artifacts, validation, profile)) or (
        report_dir is not None and output.is_relative_to(report_dir)
    ):
        raise ValueError("Retention output must be separate from artifact/evidence directories")
    manifest, reports, reference = validate_reports(artifacts, validation, profile)
    receipt_path = Path(args.checkpoint_receipt).resolve()
    receipt_path = _safe_member(receipt_path.parent, receipt_path.name)
    checkpoint = checkpoint_reference(json.loads(receipt_path.read_text()), manifest["checkpoint"])
    members = collect_evidence(artifacts, validation, profile, manifest, reports, reference,
                               checkpoint_receipt=receipt_path, report_dir=report_dir)
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    checkpoint = verify_checkpoint_reference(bucket, checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "evidence.tar.gz"
    inventory = build_evidence_archive(archive, members, _restore_text(prefix, checkpoint))
    retention = {"schema": "olmo-tiled-rt-retention-v1", "prefix": prefix,
                 "checkpoint_reference": checkpoint, "checkpoint_uploaded": False,
                 "checkpoint_compressed_in_evidence": False,
                 "evidence": {"object": archive.name, **file_digest(archive), "members": inventory},
                 "reports": {kind: {"status": report["status"], "finished_utc": report["finished_utc"]}
                             for kind, report in reports.items()}}
    manifest_path = output / "retention-manifest.json"
    write_json(manifest_path, retention)
    objects = [upload_verified(bucket, f"{prefix_key}/{path.name}", path, file_digest(path))
               for path in (archive, manifest_path)]
    receipt = {"schema": "olmo-tiled-rt-storage-receipt-v1", "status": "verified",
               "checkpoint_reference": checkpoint, "objects": objects}
    receipt_path = output / "storage-receipt.json"
    write_json(receipt_path, receipt)
    receipt_object = upload_verified(bucket, f"{prefix_key}/{receipt_path.name}", receipt_path, file_digest(receipt_path))
    result = {**receipt, "receipt_object": receipt_object}
    write_json(output / "upload-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--prefix", required=True)
    result = retain(parser.parse_args())
    print(json.dumps({"status": result["status"], "receipt_uri": result["receipt_object"]["uri"]}), flush=True)


if __name__ == "__main__":
    main()
