#!/usr/bin/env python3
"""Retain bounded F3 evidence, preserving failures and reusing native weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from scripts.olmo_f2_retain import require, _sha, _source_name, _passing_flags
from scripts.olmo_f1_retain import safe_evidence, MAX_EVIDENCE_BYTES, UPLOAD_FILES, CHECKPOINT_RECEIPT
from scripts.olmo_fbt_retain import verify_snapshots, SNAPSHOT_FILES
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote

SCHEMA = "olmo-f3-retention-v1"
REPORT_SCHEMA = "olmo-f3-graph-training-v1"
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3-graph-training/"
ESSENTIAL_SOURCES = {
    "scripts/olmo_f3_graph_training.py", "cdrm/pretrained/static_training.py",
    "cdrm/pretrained/static_nextlat.py", "cdrm/pretrained/olmo_static.py",
    "cdrm/pretrained/olmo_tiled.py",
}
EXTRA_PROJECT_FILES = {
    "cdrm/__init__.py", "cdrm/pretrained/__init__.py", "scripts/olmo_f3_retain.py",
    "scripts/olmo_f2_retain.py", "scripts/olmo_f1_retain.py", "scripts/olmo_fbt_retain.py",
    "scripts/olmo_tiled_retain.py", "scripts/olmo_retain.py", "scripts/openelm_retain.py",
    "scripts/docker_shell.sh", "docker/requirements-docker.txt", "AGENTS.md",
    "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v4.md",
    "docs/olmo1b-f3-usage.md", "tests/test_olmo_f3_retain.py",
}
RUNTIME_FILES = {"report.json", "config.json", "configuration.json", "run.log", "stdout.log",
                 "stderr.log", "test-results.txt", "capture-error.txt"}
REPORT_FILES = {"protocol.md", "results.md", "assessment.md", "test-results.txt", "summary.json",
                "capability-ledger.json", "results.json", "backend-summary.json", "retention-tests.txt"}


def parse_prefix(prefix):
    prefix = prefix.rstrip("/")
    require(prefix.startswith(PREFIX_ROOT) and
            re.fullmatch(r"\d{8}T\d{6}Z", prefix[len(PREFIX_ROOT):]) is not None,
            f"Require one timestamp under {PREFIX_ROOT}")
    return tuple(prefix[5:].split("/", 1))


def validate_runs(runtime_dirs, *, project_root=ROOT, checkpoint_receipt=CHECKPOINT_RECEIPT,
                  report_dir, allow_failed_diagnostics=False):
    """Validate actual run lineage; a retained failure never contributes success."""
    receipt = json.loads(safe_evidence(checkpoint_receipt.parent, checkpoint_receipt.name).read_text())
    protocol_sha = file_digest(safe_evidence(report_dir, "protocol.md"))["sha256"]
    runs, names, successful_sources, reference = [], set(), {}, None
    for directory in runtime_dirs:
        directory = Path(directory).absolute()
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", directory.name) is not None
                and directory.name not in names, "Runtime names must be safe and unique")
        names.add(directory.name)
        path = safe_evidence(directory, "report.json")
        report = json.loads(path.read_text())
        status = report.get("status")
        failed = status in ("failed", "capture_blocked")
        require(report.get("schema") == REPORT_SCHEMA and report.get("finished_utc")
                and (status == "passed" or failed), "Require a completed F3 report")
        require(not failed or allow_failed_diagnostics,
                "Failed diagnostics require explicit --allow-failed-diagnostics retention")
        configuration = report.get("configuration")
        require(isinstance(configuration, dict) and configuration, "Missing F3 configuration")
        checks = report.get("checks")
        require(isinstance(checks, list) and all(isinstance(row, dict) and type(row.get("passed")) is bool
                for row in checks), "F3 checks must retain explicit boolean outcomes")
        if failed:
            require(report.get("stage") and report["stage"] != "complete" and report.get("error_type")
                    and isinstance(report.get("error_message"), str), "Failed diagnostic lacks stage/error evidence")
            if status == "capture_blocked":
                require(report["stage"] in ("capture", "capacity_capture")
                        and report.get("capture_succeeded") is False,
                        "capture_blocked must identify an unsuccessful capture stage")
        else:
            require(report.get("stage") == "complete" and checks and all(row["passed"] for row in checks),
                    "Passing F3 report requires completed passing checks")
            _passing_flags(report)
        wandb = report.get("wandb", {})
        require(wandb.get("status") == ("synced_failed_experiment" if failed else "synced")
                and str(wandb.get("run_url", "")).startswith("https://wandb.ai/taylorbollman/"),
                "Missing synchronized W&B provenance")
        current_reference = checkpoint_reference(receipt, report.get("checkpoint", {}))
        require(reference is None or reference == current_reference, "F3 runs reference different native checkpoints")
        reference = current_reference
        hashes = report.get("source_hashes")
        require(isinstance(hashes, dict) and ESSENTIAL_SOURCES <= hashes.keys(), "Missing essential F3 source inventory")
        mismatches, matching_sources, snapshots = {}, [], {}
        for name, expected in hashes.items():
            _source_name(name)
            require(_sha(expected), "Invalid runtime source SHA256")
            source = project_root/name
            actual = file_digest(safe_evidence(project_root, name))["sha256"] if source.exists() or source.is_symlink() else None
            if actual == expected:
                matching_sources.append(name)
                if not failed:
                    require(name not in successful_sources or successful_sources[name] == expected,
                            "Successful F3 runs require incompatible source versions")
                    successful_sources[name] = expected
            else:
                require(failed, f"Successful source is absent or changed: {name}")
                mismatches[name] = {"reported_sha256": expected, "current_sha256": actual}
                snapshot = f"source-snapshot/{name}"
                if (directory/snapshot).exists() or (directory/snapshot).is_symlink():
                    require(file_digest(safe_evidence(directory, snapshot))["sha256"] == expected,
                            "Historical source snapshot differs from recorded bytes")
                    snapshots[name] = snapshot
                    mismatches[name]["exact_historical_snapshot"] = snapshot
        protocol = report.get("protocol_sha256")
        require(_sha(protocol), "Missing frozen F3 protocol SHA256")
        require(failed or protocol == protocol_sha, "Successful F3 protocol changed")
        for filename in ("config.json", "configuration.json"):
            if (directory/filename).exists():
                require(json.loads(safe_evidence(directory, filename).read_text()) == configuration,
                        "Standalone F3 configuration differs")
        runs.append({"directory": directory, "report": report, "matching_sources": matching_sources,
            "source_snapshots": snapshots, "provenance": {
                "name": directory.name, "status": status, "stage": report["stage"],
                "counts_as_success": not failed, "diagnostic_failure_only": failed,
                "check_count": len(checks), "passed_check_count": sum(row["passed"] for row in checks),
                "report_sha256": file_digest(path)["sha256"], "wandb_url": wandb["run_url"],
                "current_source_hashes_match": not mismatches, "historical_source_mismatches": mismatches,
                "historical_runtime_sources_complete": all(name in snapshots for name in mismatches) if failed else None,
                "reported_protocol_sha256": protocol, "current_protocol_sha256": protocol_sha,
                "current_protocol_hash_matches": protocol == protocol_sha}})
    require(runs, "At least one completed F3 report is required")
    return runs, reference, successful_sources


def collect_evidence(runs, successful_sources, references, *, project_root=ROOT,
                     checkpoint_receipt=CHECKPOINT_RECEIPT, report_dir):
    members = {}
    def add(root, name, category):
        path, key = safe_evidence(root, name), f"{category}/{name}"
        require(key not in members or members[key] == path, "Conflicting evidence member")
        members[key] = path
    for name in sorted(set(successful_sources) | EXTRA_PROJECT_FILES):
        add(project_root, name, "project")
    # Narrow F3/helper test inventories, never a recursive workspace archive.
    for directory, pattern in (("scripts", "olmo_f3_*.py"), ("tests", "test_olmo_f3*.py"),
                               ("tests", "test_olmo_static*.py"), ("tests", "test_static*.py")):
        for path in sorted((project_root/directory).glob(pattern)):
            add(project_root, path.relative_to(project_root).as_posix(), "project")
    for name in ("tests/test_olmo_ordinary_checkpointing.py", "docs/reports/olmo1b-f2/storage-receipt.json"):
        if (project_root/name).exists():
            add(project_root, name, "project")
    for name, files in SNAPSHOT_FILES.items():
        require({row["file"] for row in references[name]["files"]} == files, "Pinned source snapshot inventory differs")
        for filename in sorted(files | {"README.md", "manifest.json"}):
            add(project_root, f"cdrm/pretrained/{name}/{filename}", "project")
    for run in runs:
        directory = run["directory"]
        for name in run["matching_sources"]:
            add(project_root, name, "project")
        for snapshot in run["source_snapshots"].values():
            add(directory, snapshot, f"runtime/{directory.name}")
        for filename in sorted(RUNTIME_FILES):
            if (directory/filename).exists():
                add(directory, filename, f"runtime/{directory.name}")
        adjacent = directory.name+".log"
        if (directory.parent/adjacent).exists():
            add(directory.parent, adjacent, "runtime-logs")
    for filename in sorted(REPORT_FILES):
        if (report_dir/filename).exists():
            add(report_dir, filename, "report")
    for path in sorted(report_dir.iterdir()):
        if path.suffix in {".pdf", ".png", ".csv"}:
            add(report_dir, path.name, "report")
    for filename in ("protocol.md", "results.md", "assessment.md", "test-results.txt"):
        require(f"report/{filename}" in members, f"Missing final F3 evidence: {filename}")
    add(checkpoint_receipt.parent, checkpoint_receipt.name, "provenance")
    require(sum(path.stat().st_size for path in members.values()) <= MAX_EVIDENCE_BYTES, "F3 evidence exceeds small-archive budget")
    return [(path, name) for name, path in sorted(members.items())]


def upload_verified(bucket, key, path, expected):
    """Create-only generic evidence transport with F3 namespace and metadata."""
    from google.api_core.exceptions import PreconditionFailed
    require(Path(key).name in UPLOAD_FILES and path.name == Path(key).name,
            "Only whitelisted small F3 evidence objects may be uploaded")
    parse_prefix(f"gs://{bucket.name}/{key.rsplit('/', 1)[0]}")
    require(path.is_file() and not path.is_symlink(), "Upload requires a regular evidence file")
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
    directories = [Path(path).absolute() for path in args.runtime_dir]
    docs, output, receipt = (Path(path).absolute() for path in (args.report_dir, args.output_dir, args.checkpoint_receipt))
    require(not any(output.resolve().is_relative_to(path.resolve()) for path in [*directories, docs]),
            "Retention output must be outside its evidence inputs")
    options = {"checkpoint_receipt": receipt, "report_dir": docs,
               "allow_failed_diagnostics": args.allow_failed_diagnostics}
    runs, reference, sources = validate_runs(directories, **options)
    members = collect_evidence(runs, sources, verify_snapshots(ROOT), checkpoint_receipt=receipt, report_dir=docs)
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    reference = verify_checkpoint_reference(bucket, reference)
    output.mkdir(parents=True, exist_ok=True)
    archive = output/"evidence.tar.gz"
    restore = ("# F3 static-layout graph training evidence\n\n"
        f"Reuse {reference['uri']} at generation {reference['generation']}; SHA256 {reference['sha256']}, "
        f"{reference['size_bytes']} bytes. Weights are neither archived nor uploaded again.\n\n"
        "Verify evidence-members.json before restoring project/. Each runtime report preserves its own "
        "configuration, checks, source hashes and W&B identity. Failed/capture_blocked diagnostics never count "
        "as successful coverage. The manifest records historical source mismatches and exact run-local "
        "source snapshots where available; missing historical bytes are not claimed reconstructed. "
        "Current successful source hashes are verified independently. Final interpretation, scoped test logs "
        "and plots are under report/. Check full-step versus captured-region scope before comparing timing. "
        "These fixture updates do not establish quality, all-layer RT, distributed execution or a fused RT kernel. "
        "Native/reference sources and the original O1 receipt are retained. Model/optimizer checkpoints, "
        "W&B directories and secrets are excluded. GPU commands require the project container.\n")
    inventory = build_evidence_archive(archive, members, restore)
    refreshed, _, refreshed_sources = validate_runs(directories, **options)
    require(refreshed == runs and refreshed_sources == sources, "F3 evidence changed during retention")
    manifest = {"schema": SCHEMA, "kind": "retention-manifest", "checkpoint_reference": reference,
        "checkpoint_uploaded": False, "checkpoint_compressed_in_evidence": False,
        "runs": [run["provenance"] for run in runs], "current_successful_source_hashes": sources,
        "failed_diagnostics_count_as_success": False,
        "evidence": {"object": archive.name, **file_digest(archive), "members": inventory}}
    manifest_path = output/"retention-manifest.json"; write_json(manifest_path, manifest)
    objects = [upload_verified(bucket, f"{key}/{path.name}", path, file_digest(path)) for path in (archive, manifest_path)]
    result = {"schema": SCHEMA, "kind": "storage-receipt", "status": "verified", "checkpoint_reference": reference,
              "objects": objects, "runs": manifest["runs"]}
    receipt_path = output/"storage-receipt.json"; write_json(receipt_path, result)
    uploaded = upload_verified(bucket, f"{key}/{receipt_path.name}", receipt_path, file_digest(receipt_path))
    result = {**result, "receipt_object": uploaded}
    write_json(output/"upload-result.json", result); write_json(docs/"storage-receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--allow-failed-diagnostics", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=ROOT/"docs/reports/olmo1b-f3")
    parser.add_argument("--checkpoint-receipt", type=Path, default=CHECKPOINT_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-prefix", required=True)
    result = retain(parser.parse_args(argv))
    print(json.dumps({"status": result["status"], "receipt_uri": result["receipt_object"]["uri"]}), flush=True)


if __name__ == "__main__":
    main()
