#!/usr/bin/env python3
"""Retain bounded O5a FBT evidence without uploading any model object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.olmo_artifacts import (CHECKPOINT_FILENAME, FILE_SPECS, MANIFEST_FILENAME,
    MANIFEST_SCHEMA, REPO_ID, REVISION)
from cdrm.pretrained.olmo_reference import verify_olmo_reference_sources
from scripts.olmo_fbt_validate import SOURCE_FILES as VALIDATED_SOURCE_FILES
from scripts.olmo_lm_common import verify_nextlat_sources
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference
from scripts.openelm_retain import _safe_member, _check_remote, file_digest, build_evidence_archive

SCHEMA = "olmo-fbt-reference-v1"
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-fbt-reference/"
ARTIFACTS = ROOT / ".runtime/olmo1b-step60000/artifacts"
CHECKPOINT_RECEIPT = ROOT / "docs/reports/olmo1b-o1/storage-receipt.json"
SOURCE_FILES = frozenset(VALIDATED_SOURCE_FILES)
SNAPSHOT_FILES = {
    "_olmo_reference": {"model.py", "config.py", "util.py", "torch_util.py", "exceptions.py", "LICENSE",
                        "initialization.py", "aliases.py", "checkpoint_config.json"},
    "_nextlat_reference": {"model_nextlat.py", "model_base.py", "fineweb_1b_horizon1.yaml",
                           "fineweb_100m.yaml", "a5.yaml", "LICENSE"},
    "_fbt_reference": {"gpt.py", "LICENSE"},
}
EXTRA_PROJECT_FILES = {
    "cdrm/__init__.py", "cdrm/pretrained/__init__.py", "scripts/olmo_fbt_retain.py",
    "scripts/olmo_tiled_retain.py", "scripts/olmo_retain.py", "scripts/openelm_retain.py",
    "scripts/docker_shell.sh", "docker/requirements-docker.txt", "AGENTS.md",
    "docs/olmo1b-fbt-usage.md", "docs/fbt-rt-nextlat-handoff.md",
    "docs/fbt-rt-nextlat-research-plan-v3.md", "docs/olmo-1b-250b-checkpoint-selection.json",
    "tests/test_olmo_fbt.py", "tests/test_olmo_fbt_adversarial.py",
    "tests/test_olmo_fbt_training.py", "tests/test_olmo_fbt_retain.py",
}
REPORT_FILES = {"protocol.md", "results.md", "test-results.txt", "validation-summary.json", "storage-receipt.json"}
UPLOAD_FILES = {"evidence.tar.gz", "retention-manifest.json", "storage-receipt.json"}


def parse_prefix(prefix):
    prefix = prefix.rstrip("/")
    if not prefix.startswith(PREFIX_ROOT) or re.fullmatch(r"\d{8}T\d{6}Z", prefix[len(PREFIX_ROOT):]) is None:
        raise ValueError(f"Require one timestamp under {PREFIX_ROOT}")
    return tuple(prefix[5:].split("/", 1))


def verify_snapshots(project_root):
    references = {"_olmo_reference": verify_olmo_reference_sources(),
                  "_nextlat_reference": verify_nextlat_sources()}
    directory = project_root / "cdrm/pretrained/_fbt_reference"
    fbt = json.loads(_safe_member(directory, "manifest.json").read_text())
    if fbt.get("repository") != "xidulu/Full-bandwidth-transformer" or fbt.get("revision") != "7037c60924870aca6e30fac95212b0c7caee052d":
        raise ValueError("FBT source revision differs")
    references["_fbt_reference"] = fbt
    for name, manifest in references.items():
        rows = manifest.get("files", [])
        if len(rows) != len(SNAPSHOT_FILES[name]) or {row["file"] for row in rows} != SNAPSHOT_FILES[name]:
            raise ValueError("Snapshot inventory differs from explicit whitelist")
        for row in rows:
            digest = file_digest(_safe_member(project_root / "cdrm/pretrained" / name, row["file"]))
            if digest["sha256"] != row["sha256"] or digest["size_bytes"] != row["bytes"]:
                raise ValueError("Pinned snapshot bytes differ")
    return references


def validate_artifact_metadata(artifacts):
    """Recheck small pinned artifacts; the full model is verified remotely."""
    manifest = json.loads(_safe_member(artifacts, MANIFEST_FILENAME).read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("repo_id") != REPO_ID or manifest.get("revision") != REVISION:
        raise ValueError("Native artifact manifest source differs")
    for name, (size, sha) in FILE_SPECS.items():
        if name == CHECKPOINT_FILENAME:
            continue
        digest = file_digest(_safe_member(artifacts, "native/" + name))
        if digest["size_bytes"] != size or digest["sha256"] != sha:
            raise ValueError("Native small-artifact bytes differ")
    return manifest


def validate_report(validation, *, project_root=ROOT, artifacts=ARTIFACTS):
    report = json.loads(_safe_member(validation, "report.json").read_text())
    if report.get("schema") != "olmo-fbt-validation-v1" or report.get("status") != "passed" or not report.get("finished_utc"):
        raise ValueError("Require a completed passing O5a validation report")
    if not report.get("cases") or any(row.get("passed") is not True for row in report["cases"]):
        raise ValueError("O5a report contains missing or nonpassing cases")
    if not all(report.get("runtime", {}).get(key) for key in ("gpu", "torch", "cuda")):
        raise ValueError("O5a report lacks actual GPU runtime")
    hashes = report.get("source_hashes", {})
    if set(hashes) != SOURCE_FILES:
        raise ValueError("Validated source inventory differs from exact whitelist")
    for name, sha in hashes.items():
        if file_digest(_safe_member(project_root, name))["sha256"] != sha:
            raise ValueError(f"Validated source changed: {name}")
    native = validate_artifact_metadata(artifacts)
    if report.get("checkpoint") != native["checkpoint"]:
        raise ValueError("Validation checkpoint provenance differs")
    references = verify_snapshots(project_root)
    if report.get("nextlat_reference") != references["_nextlat_reference"] or report.get("fbt_reference") != references["_fbt_reference"]:
        raise ValueError("Validation source snapshot provenance differs")
    return report, native, references


def collect_evidence(validation, report, references, *, project_root=ROOT, artifacts=ARTIFACTS,
                     checkpoint_receipt=CHECKPOINT_RECEIPT, report_dir):
    if set(report.get("source_hashes", {})) != SOURCE_FILES:
        raise ValueError("Archive source inventory differs from exact whitelist")
    members = {}
    def add(root, name, category):
        members[f"{category}/{name}"] = _safe_member(root, name)
    for name in sorted(SOURCE_FILES | EXTRA_PROJECT_FILES):
        add(project_root, name, "project")
    for name, snapshot in references.items():
        rows = snapshot["files"]
        if len(rows) != len(SNAPSHOT_FILES[name]) or {row["file"] for row in rows} != SNAPSHOT_FILES[name]:
            raise ValueError("Archive snapshot inventory differs from exact whitelist")
        for file in sorted(SNAPSHOT_FILES[name] | {"README.md", "manifest.json"}):
            add(project_root, f"cdrm/pretrained/{name}/{file}", "project")
    add(validation, "report.json", "validation")
    for file in sorted(REPORT_FILES):
        if (report_dir / file).exists():
            add(report_dir, file, "report")
    for file in ("protocol.md", "results.md", "test-results.txt"):
        if f"report/{file}" not in members:
            raise ValueError(f"Missing required O5a report evidence: {file}")
    for name in (MANIFEST_FILENAME, "checkpoint-inspection.json"):
        add(artifacts, name, "artifacts")
    for name in FILE_SPECS:
        if name != CHECKPOINT_FILENAME:
            add(artifacts, "native/" + name, "artifacts")
    members["provenance/o1-storage-receipt.json"] = _safe_member(checkpoint_receipt.parent, checkpoint_receipt.name)
    return [(path, name) for name, path in sorted(members.items())]


def upload_verified(bucket, key, path, expected):
    """Create only known evidence objects with a create-only GCS precondition."""
    from google.api_core.exceptions import PreconditionFailed
    if Path(key).name not in UPLOAD_FILES:
        raise ValueError("Only whitelisted evidence objects may be uploaded; no model objects")
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
    return {"uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation), **expected}


def retain(args):
    bucket_name, key = parse_prefix(args.prefix)
    validation, output, docs = (Path(value).resolve() for value in (args.validation, args.output_dir, args.report_dir))
    if any(output.is_relative_to(path) for path in (validation, docs, ARTIFACTS.resolve())):
        raise ValueError("Retention output must be outside its evidence inputs")
    report, native, references = validate_report(validation)
    receipt = json.loads(_safe_member(CHECKPOINT_RECEIPT.parent, CHECKPOINT_RECEIPT.name).read_text())
    checkpoint = checkpoint_reference(receipt, native["checkpoint"])
    members = collect_evidence(validation, report, references, report_dir=docs)
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    checkpoint = verify_checkpoint_reference(bucket, checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "evidence.tar.gz"
    restore = (f"# O5a FBT reference evidence\n\nReuse {checkpoint['uri']} at generation {checkpoint['generation']}; "
               f"SHA256 {checkpoint['sha256']}, {checkpoint['size_bytes']} bytes. No model is embedded or uploaded.\n\n"
               "Verify evidence-members.json before restoring project/. validation/report.json records bounded "
               "native-checkpoint checks, not a learning run, general precision clearance or capacity benchmark. "
               "Native artifact metadata/tokenizer and source snapshots are included. Run GPU checks only through "
               "the project's Docker launcher. NextLat remains a training objective, not inference recurrence.\n")
    inventory = build_evidence_archive(archive, members, restore)
    # Revalidate the exact tested source after archiving to close a local race.
    validate_report(validation)
    manifest = {"schema": SCHEMA, "kind": "retention-manifest", "checkpoint_reference": checkpoint,
                "checkpoint_uploaded": False, "checkpoint_compressed_in_evidence": False,
                "source_hashes": report["source_hashes"], "validation_finished_utc": report["finished_utc"],
                "evidence": {"object": archive.name, **file_digest(archive), "members": inventory}}
    manifest_path = output / "retention-manifest.json"
    write_json(manifest_path, manifest)
    objects = [upload_verified(bucket, f"{key}/{path.name}", path, file_digest(path)) for path in (archive, manifest_path)]
    receipt = {"schema": SCHEMA, "kind": "storage-receipt", "status": "verified",
               "checkpoint_reference": checkpoint, "objects": objects}
    receipt_path = output / "storage-receipt.json"
    write_json(receipt_path, receipt)
    receipt_object = upload_verified(bucket, f"{key}/{receipt_path.name}", receipt_path, file_digest(receipt_path))
    result = {**receipt, "receipt_object": receipt_object}
    write_json(output / "upload-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    result = retain(parser.parse_args())
    print(json.dumps({"status": result["status"], "receipt_uri": result["receipt_object"]["uri"]}), flush=True)


if __name__ == "__main__":
    main()
