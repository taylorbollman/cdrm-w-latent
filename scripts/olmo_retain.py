#!/usr/bin/env python3
"""Retain verified OLMo O1 checkpoint and both bounded validation reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.olmo_artifacts import (
    CHECKPOINT_FILENAME, FILE_SPECS, MANIFEST_FILENAME, validate_prepared_manifest,
)
from scripts.openelm_retain import (
    _safe_member, _check_remote, build_evidence_archive, file_digest,
)

PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-step60000/"
REPORT_SCHEMAS = {"ordinary": "olmo-import-validation-v1", "rt": "olmo-rt-reference-validation-v1"}
COMMON_SOURCE_FILES = {
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_reference.py",
    "cdrm/pretrained/olmo_artifacts.py", "cdrm/pretrained/artifacts.py",
    "scripts/olmo_validation.py", "scripts/experiment_tracking.py",
}
SOURCE_FILES = {
    "ordinary": COMMON_SOURCE_FILES | {"scripts/olmo_validate.py"},
    "rt": COMMON_SOURCE_FILES | {
        "cdrm/pretrained/recurrent.py", "cdrm/pretrained/openelm.py",
        "cdrm/pretrained/olmo_recurrent.py", "cdrm/pretrained/olmo_recurrent_oracle.py",
        "scripts/olmo_rt_validate.py",
    },
}
REPORT_FILES = {"results.md", "protocol.md", "test-results.txt", "validation-summary.json", "storage-receipt.json"}
EXTRA_PROJECT_FILES = (
    "cdrm/__init__.py", "cdrm/pretrained/__init__.py",
    "scripts/olmo_prepare.py", "scripts/olmo_retain.py", "scripts/openelm_retain.py",
    "docker/requirements-docker.txt", "docs/olmo-1b-250b-checkpoint-selection.json",
    "docs/olmo1b-native-rt-usage.md", "docs/fbt-rt-nextlat-research-plan-v3.md",
)


def parse_prefix(prefix: str) -> tuple[str, str]:
    prefix = prefix.rstrip("/")
    if not prefix.startswith(PREFIX_ROOT) or re.fullmatch(
        r"\d{8}T\d{6}Z", prefix[len(PREFIX_ROOT):]
    ) is None:
        raise ValueError(f"Retention requires one timestamp under {PREFIX_ROOT}")
    return tuple(prefix[5:].split("/", 1))


def validate_reports(artifacts: Path, ordinary: Path, recurrent: Path, *, project_root=ROOT):
    """Reject stale source or incomplete evidence before uploading anything."""
    manifest = validate_prepared_manifest(artifacts)
    from cdrm.pretrained.olmo_reference import verify_olmo_reference_sources
    reference = verify_olmo_reference_sources()
    reports = {}
    for kind, directory in (("ordinary", ordinary), ("rt", recurrent)):
        report = json.loads(_safe_member(directory, "report.json").read_text())
        if (report.get("schema") != REPORT_SCHEMAS[kind] or report.get("status") != "passed"
                or not report.get("finished_utc")):
            raise ValueError(f"Require a completed, passing {kind} validation report")
        if report.get("artifacts_manifest") != manifest or report.get("checkpoint") != manifest["checkpoint"]:
            raise ValueError(f"{kind} report used different artifact provenance")
        if report.get("native_reference_sources") != reference:
            raise ValueError(f"{kind} report used different native reference sources")
        hashes = report.get("source_hashes", {})
        if set(hashes) != SOURCE_FILES[kind]:
            raise ValueError(f"{kind} report source hash inventory differs from the contract")
        for relative, digest in hashes.items():
            if file_digest(_safe_member(project_root, relative))["sha256"] != digest:
                raise ValueError(f"Validated source changed after {kind} run: {relative}")
        reports[kind] = report
    return manifest, reports, reference


def collect_evidence(artifacts, ordinary, recurrent, manifest, reports, reference, *, project_root=ROOT, report_dir=None):
    members = {}

    def add(root, relative, category):
        members[f"{category}/{relative}"] = _safe_member(root, relative)

    add(artifacts, MANIFEST_FILENAME, "artifacts")
    if (artifacts / "checkpoint-inspection.json").is_file():
        add(artifacts, "checkpoint-inspection.json", "artifacts")
    # Full checkpoint is uploaded separately, never compressed into evidence.
    for name in FILE_SPECS:
        if name != CHECKPOINT_FILENAME:
            add(artifacts, f"native/{name}", "artifacts")
    add(ordinary, "report.json", "validation/ordinary")
    add(recurrent, "report.json", "validation/rt")
    project_files = set(EXTRA_PROJECT_FILES)
    for report in reports.values():
        project_files.update(report["source_hashes"])
    project_files.update(str(p.relative_to(project_root)) for p in (project_root / "tests").glob("test_olmo_*.py"))
    for relative in sorted(project_files):
        add(project_root, relative, "project")
    reference_files = ["manifest.json", "README.md", *(r["file"] for r in reference["files"])]
    for name in reference_files:
        add(project_root, f"cdrm/pretrained/_olmo_reference/{name}", "project")
    if report_dir is not None:
        # One explicit report directory, excluding runtime logs, caches and credentials.
        for name in sorted(REPORT_FILES):
            if (report_dir / name).is_file():
                add(report_dir, name, "report")
    return [(path, name) for name, path in sorted(members.items())]


def upload_verified(bucket, key, path, expected):
    """Create an immutable object or verify exact existing bytes; no overwrite."""
    from google.api_core.exceptions import PreconditionFailed
    blob = bucket.get_blob(key)
    if blob is None:
        blob = bucket.blob(key)
        blob.metadata = {"sha256": expected["sha256"], "artifact_schema": "olmo-o1-retention-v1"}
        try:
            blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
        except PreconditionFailed:
            blob = bucket.get_blob(key)
            if blob is None:
                raise
        blob.reload()
    _check_remote(blob, expected)
    return {"uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation), **expected,
            "verification": "GCS size/server MD5 and SHA256 metadata match local bytes"}


def retain(args):
    bucket_name, key = parse_prefix(args.prefix)
    artifacts, ordinary, recurrent, output = [Path(p).resolve() for p in
                                              (args.artifacts, args.ordinary, args.recurrent, args.output_dir)]
    report_dir = None if args.report_dir is None else args.report_dir.resolve()
    if any(output.is_relative_to(p) for p in (artifacts, ordinary, recurrent)) or (
        report_dir is not None and output.is_relative_to(report_dir)
    ):
        raise ValueError("Retention output must be separate from artifacts/evidence")
    manifest, reports, reference = validate_reports(artifacts, ordinary, recurrent)
    checkpoint = _safe_member(artifacts, manifest["checkpoint"]["path"])
    checkpoint_digest = file_digest(checkpoint)
    if any(checkpoint_digest[k] != manifest["checkpoint"][k] for k in ("size_bytes", "sha256")):
        raise ValueError("Checkpoint changed during retention preparation")
    members = collect_evidence(artifacts, ordinary, recurrent, manifest, reports, reference, report_dir=report_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"gs://{bucket_name}/{key}"
    restore = f"""# Restore OLMo O1

Download evidence.tar.gz, retention-manifest.json and storage-receipt.json from
{prefix}. Verify checksums before extraction and compare evidence-members.json.
Download {prefix}/checkpoint/{CHECKPOINT_FILENAME} separately to
artifacts/native/{CHECKPOINT_FILENAME}; expected SHA256 {checkpoint_digest['sha256']}.
Overlay project/ onto the recorded repository in its Docker environment.
validate_prepared_manifest('artifacts') rehashes the full source checkpoint.
scripts/olmo_validate.py and scripts/olmo_rt_validate.py reproduce the bounded
GPU checks; run only inside the project container. The two validation/report.json
files record source hashes, fixtures, runtime and W&B URLs. No trained model,
tiled/FBT/NextLat correctness, large-batch capacity or training quality is implied.
"""
    archive = output / "evidence.tar.gz"
    inventory = build_evidence_archive(archive, members, restore)
    retention = {"schema": "olmo-o1-retention-v1", "prefix": prefix,
                 "checkpoint": {"object": f"checkpoint/{CHECKPOINT_FILENAME}", **checkpoint_digest},
                 "evidence": {"object": archive.name, **file_digest(archive), "members": inventory},
                 "checkpoint_compressed_in_evidence": False,
                 "validation_status": {k: r["status"] for k, r in reports.items()}}
    retention_path = output / "retention-manifest.json"
    write_json(retention_path, retention)
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    objects = [upload_verified(bucket, f"{key}/{relative}", path, digest) for relative, path, digest in (
        (f"checkpoint/{CHECKPOINT_FILENAME}", checkpoint, checkpoint_digest),
        (archive.name, archive, file_digest(archive)),
        (retention_path.name, retention_path, file_digest(retention_path)),
    )]
    receipt = {"schema": "olmo-o1-storage-receipt-v1", "status": "verified", "objects": objects}
    receipt_path = output / "storage-receipt.json"
    write_json(receipt_path, receipt)
    receipt_record = upload_verified(bucket, f"{key}/{receipt_path.name}", receipt_path, file_digest(receipt_path))
    result = {**receipt, "receipt_object": receipt_record}
    write_json(output / "upload-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    result = retain(args)
    print(json.dumps({"status": result["status"], "receipt_uri": result["receipt_object"]["uri"]}))


if __name__ == "__main__":
    main()
