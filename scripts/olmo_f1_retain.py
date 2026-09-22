#!/usr/bin/env python3
"""Retain bounded F1 integration evidence, reusing the immutable O1 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import write_json
from scripts.olmo_fbt_retain import verify_snapshots, SNAPSHOT_FILES
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference
from scripts.openelm_retain import _safe_member, _check_remote, file_digest, build_evidence_archive

SCHEMA = "olmo-f1-retention-v1"
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f1-integration/"
CHECKPOINT_RECEIPT = ROOT/"docs/reports/olmo1b-o1/storage-receipt.json"
EXTRA_PROJECT_FILES = {
    "cdrm/__init__.py", "cdrm/pretrained/__init__.py", "scripts/olmo_f1_retain.py",
    "scripts/olmo_fbt_retain.py", "scripts/olmo_fbt_validate.py", "scripts/olmo_lm_common.py",
    "scripts/olmo_tiled_retain.py", "scripts/olmo_retain.py", "scripts/openelm_retain.py",
    "scripts/docker_shell.sh", "docker/requirements-docker.txt",
    "AGENTS.md", "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v4.md",
    "docs/olmo1b-f1-usage.md", "configs/olmo_f1_integration.json",
    "tests/test_olmo_f1_retain.py",
}
RUNTIME_FILES = {
    "report.json", "configuration.json", "config.json", "validation-summary.json",
    "integration-summary.json", "trace-summary.json", "trace-summary.txt",
    "backend-summary.json", "test-results.txt", "capability-ledger.json",
}
REPORT_FILES = {
    "protocol.md", "results.md", "assessment.md", "test-results.txt",
    "validation-summary.json", "integration-summary.json", "capability-ledger.json",
    "capability-ledger.md", "trace-summary.json", "trace-summary.txt", "backend-summary.json",
    "timings.csv", "timings.pdf", "timings.png", "parameters.csv",
}
UPLOAD_FILES = {"evidence.tar.gz", "retention-manifest.json", "storage-receipt.json"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".safetensors", ".bin", ".pkl", ".pickle", ".tmp"}
FORBIDDEN_NAMES = {"credentials.json", "secrets.json", "token", "token.json", "config.env"}
MAX_MEMBER_BYTES = 64 * 1024**2
MAX_EVIDENCE_BYTES = 256 * 1024**2


def parse_prefix(prefix):
    prefix = prefix.rstrip("/")
    if not prefix.startswith(PREFIX_ROOT) or re.fullmatch(r"\d{8}T\d{6}Z", prefix[len(PREFIX_ROOT):]) is None:
        raise ValueError(f"Require one timestamp under {PREFIX_ROOT}")
    return tuple(prefix[5:].split("/", 1))


def safe_evidence(root, relative):
    """Reject binary model payloads and every symlink component before reading."""
    parts = PurePosixPath(relative).parts
    if (PurePosixPath(relative).suffix.lower() in FORBIDDEN_SUFFIXES
            or any(part.lower() in FORBIDDEN_NAMES for part in parts)):
        raise ValueError("Model/checkpoint/secret files must never enter F1 evidence")
    path = _safe_member(root, relative)
    # _safe_member checks the final symlink and containment; reject internal aliases too.
    current = root
    if current.is_symlink():
        raise ValueError("Symlinked evidence root")
    for part in parts:
        current = current/part
        if current.is_symlink():
            raise ValueError("Symlinked evidence ancestor")
    if path.stat().st_size > MAX_MEMBER_BYTES:
        raise ValueError("F1 retains small summaries, not large trace/model payloads")
    return path


def validate_report(runtime_dir, *, project_root=ROOT, checkpoint_receipt=CHECKPOINT_RECEIPT):
    report = json.loads(safe_evidence(runtime_dir, "report.json").read_text())
    if (report.get("schema") != "olmo-f1-integration-v1" or report.get("status") != "passed"
            or not report.get("finished_utc")):
        raise ValueError("Require a completed passing F1 integration report")
    cases = report.get("cases")
    if (not isinstance(cases, list) or not cases or any(not isinstance(row, dict)
            or not isinstance(row.get("name"), str) or not row["name"] or row.get("passed") is not True for row in cases)
            or len({row["name"] for row in cases}) != len(cases)):
        raise ValueError("Require uniquely named passing F1 cases")
    requested = report.get("requested_cases")
    if (not isinstance(requested, list) or not requested
            or any(not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in requested)
            or len(set(requested)) != len(requested) or set(requested) != {row["name"] for row in cases}):
        raise ValueError("Completed cases must exactly cover the explicitly requested scope")
    if not isinstance(report.get("protocol_sha256"), str) or re.fullmatch(r"[a-f0-9]{64}", report["protocol_sha256"]) is None:
        raise ValueError("Require the frozen protocol SHA256")
    if (not isinstance(report.get("config"), dict) or not report["config"]
            or not isinstance(report.get("wandb"), dict) or not report["wandb"]):
        raise ValueError("F1 report must retain configuration and W&B provenance")
    hashes = report.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Require actual runtime source hashes")
    for name, digest in hashes.items():
        # Reporting/protocol documents have their own inventory, not runtime hashes.
        source = safe_evidence(project_root, name)
        if source.suffix not in {".py", ".json", ".yaml", ".yml", ".md"}:
            raise ValueError("Runtime source inventory contains a non-source file")
        if file_digest(source)["sha256"] != digest:
            raise ValueError(f"Validated source changed: {name}")
    receipt = json.loads(safe_evidence(checkpoint_receipt.parent, checkpoint_receipt.name).read_text())
    reference = checkpoint_reference(receipt, report.get("checkpoint", {}))
    for filename in ("configuration.json", "config.json"):
        if (runtime_dir/filename).exists():
            if json.loads(safe_evidence(runtime_dir, filename).read_text()) != report["config"]:
                raise ValueError("Standalone configuration differs from the completed report")
    for row in cases:
        if json.loads(safe_evidence(runtime_dir, row["name"]+".json").read_text()) != row:
            raise ValueError("Per-case evidence differs from the completed report")
    return report, reference


def collect_evidence(runtime_dir, report, references, *, project_root=ROOT,
                     checkpoint_receipt=CHECKPOINT_RECEIPT, report_dir):
    members = {}
    def add(root, name, category):
        path = safe_evidence(root, name)
        key = f"{category}/{name}"
        if key in members and members[key] != path:
            raise ValueError("Conflicting archive member")
        members[key] = path
    for name in sorted(set(report["source_hashes"]) | EXTRA_PROJECT_FILES):
        add(project_root, name, "project")
    # These are new milestone scripts/tests only, not all experiment history.
    for directory, pattern in (("scripts", "olmo_f1_*.py"), ("tests", "test_olmo_f1*.py")):
        for path in sorted((project_root/directory).glob(pattern)):
            add(project_root, path.relative_to(project_root).as_posix(), "project")
    for name, files in SNAPSHOT_FILES.items():
        rows = references[name]["files"]
        if len(rows) != len(files) or {row["file"] for row in rows} != files:
            raise ValueError("Pinned source snapshot inventory differs")
        for filename in sorted(files | {"README.md", "manifest.json"}):
            add(project_root, f"cdrm/pretrained/{name}/{filename}", "project")
    for filename in sorted(RUNTIME_FILES):
        if (runtime_dir/filename).exists():
            add(runtime_dir, filename, "runtime")
    for name in report["requested_cases"]:
        add(runtime_dir, name+".json", "runtime")
    for filename in sorted(REPORT_FILES):
        if (report_dir/filename).exists():
            add(report_dir, filename, "report")
    for filename in ("protocol.md", "results.md", "test-results.txt"):
        if f"report/{filename}" not in members:
            raise ValueError(f"Missing final F1 evidence: {filename}")
    if file_digest(members["report/protocol.md"])["sha256"] != report["protocol_sha256"]:
        raise ValueError("Frozen protocol changed after integration started")
    add(checkpoint_receipt.parent, checkpoint_receipt.name, "provenance")
    if sum(path.stat().st_size for path in members.values()) > MAX_EVIDENCE_BYTES:
        raise ValueError("F1 evidence exceeded the bounded small-archive budget")
    return [(path, name) for name, path in sorted(members.items())]


def upload_verified(bucket, key, path, expected):
    """Create only the three small evidence objects; never overwrite different bytes."""
    from google.api_core.exceptions import PreconditionFailed
    if Path(key).name not in UPLOAD_FILES or path.name != Path(key).name:
        raise ValueError("Only whitelisted F1 evidence objects may be uploaded")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Upload requires a regular evidence file")
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
    return {"uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation), **expected,
            "verification": "GCS size/server MD5 and SHA256 metadata match local evidence bytes"}


def retain(args):
    bucket_name, key = parse_prefix(args.storage_prefix)
    runtime, docs, output = (Path(path).absolute() for path in (args.runtime_dir, args.report_dir, args.output_dir))
    checkpoint_receipt = Path(args.checkpoint_receipt).absolute()
    if any(output.resolve().is_relative_to(path.resolve()) for path in (runtime, docs)):
        raise ValueError("Retention output must be outside its evidence inputs")
    report, reference = validate_report(runtime, checkpoint_receipt=checkpoint_receipt)
    references = verify_snapshots(ROOT)
    members = collect_evidence(runtime, report, references, checkpoint_receipt=checkpoint_receipt, report_dir=docs)
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    reference = verify_checkpoint_reference(bucket, reference)
    output.mkdir(parents=True, exist_ok=True)
    archive = output/"evidence.tar.gz"
    restore = ("# F1 integration evidence\n\n"
        f"Reuse the O1 native checkpoint {reference['uri']} at generation {reference['generation']}; "
        f"SHA256 {reference['sha256']}, {reference['size_bytes']} bytes. It is not uploaded again or included here.\n\n"
        "Verify evidence-members.json before restoring the bounded project/ source overlay. "
        "runtime/report.json includes the exact configuration, cases, source hashes and W&B identity. "
        "report/ contains the frozen protocol, CPU record and final interpretation. "
        "Pinned native/NextLat/FBT sources and licenses are retained under project/. "
        "The retained O1 receipt also identifies the parent evidence archive containing native "
        "artifact metadata and tokenizer files; reuse it to restore the original artifacts directory. "
        "Disposable optimizer-resume checkpoint files are excluded. Their recovery results are in the report. "
        "This is bounded functionality evidence, not quality-comparison or general numerical clearance. "
        "GPU commands must run through the project's Docker launcher.\n")
    inventory = build_evidence_archive(archive, members, restore)
    # Close the race between initial validation and reading source/report bytes.
    refreshed, _ = validate_report(runtime, checkpoint_receipt=checkpoint_receipt)
    if refreshed != report:
        raise ValueError("Integration report changed during retention")
    manifest = {"schema": SCHEMA, "kind": "retention-manifest", "checkpoint_reference": reference,
        "checkpoint_uploaded": False, "checkpoint_compressed_in_evidence": False,
        "disposable_resume_checkpoints_included": False, "source_hashes": report["source_hashes"],
        "requested_cases": report["requested_cases"],
        "scope": "Only the explicitly requested passing cases; a subset is not full-matrix clearance",
        "protocol_sha256": report["protocol_sha256"],
        "report_sha256": file_digest(runtime/"report.json")["sha256"],
        "evidence": {"object": archive.name, **file_digest(archive), "members": inventory}}
    manifest_path = output/"retention-manifest.json"
    write_json(manifest_path, manifest)
    objects = [upload_verified(bucket, f"{key}/{path.name}", path, file_digest(path)) for path in (archive, manifest_path)]
    receipt = {"schema": SCHEMA, "kind": "storage-receipt", "status": "verified",
        "checkpoint_reference": reference, "objects": objects}
    receipt_path = output/"storage-receipt.json"
    write_json(receipt_path, receipt)
    receipt_object = upload_verified(bucket, f"{key}/{receipt_path.name}", receipt_path, file_digest(receipt_path))
    result = {**receipt, "receipt_object": receipt_object}
    write_json(output/"upload-result.json", result)
    write_json(docs/"storage-receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT/".runtime/olmo1b-step60000/f1-integration-01")
    parser.add_argument("--report-dir", type=Path, default=ROOT/"docs/reports/olmo1b-f1")
    parser.add_argument("--checkpoint-receipt", type=Path, default=CHECKPOINT_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-prefix", required=True)
    result = retain(parser.parse_args(argv))
    print(json.dumps({"status": result["status"], "receipt_uri": result["receipt_object"]["uri"]}), flush=True)


if __name__ == "__main__":
    main()
