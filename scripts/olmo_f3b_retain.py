#!/usr/bin/env python3
"""Retain bounded F3b evidence with exact per-run historical source snapshots."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from scripts.olmo_f2_retain import require, _sha
from scripts.olmo_f1_retain import safe_evidence, MAX_EVIDENCE_BYTES, UPLOAD_FILES, CHECKPOINT_RECEIPT, FORBIDDEN_NAMES
from scripts.olmo_fbt_retain import verify_snapshots, SNAPSHOT_FILES
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference, CHECKPOINT_SHA256, CHECKPOINT_SIZE
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote

SCHEMA = "olmo-f3b-retention-v1"
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f3b-rt-kernel/"
PROFILE, NATIVE, TILE, FA4 = ("olmo-f3b-profile-v1", "olmo-f3b-native-v1", "olmo-f3b-tile-probe-v1", "olmo-fa4-smoke-v1")
ESSENTIAL_SOURCES = {
    PROFILE: {"scripts/olmo_f3b_profile.py", "cdrm/pretrained/olmo_tiled.py"},
    NATIVE: {"scripts/olmo_f3b_validate.py", "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_rt_kernels.py"},
    TILE: {"scripts/olmo_f3b_tile_probe.py", "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_rt_kernels.py"},
    FA4: {"scripts/olmo_fa4_smoke.py", "scripts/docker_shell.sh"},
}
EXTRA_PROJECT_FILES = {
    "cdrm/__init__.py", "cdrm/pretrained/__init__.py", "cdrm/pretrained/resource_estimates.py",
    "scripts/olmo_f3b_retain.py", "scripts/olmo_f3_retain.py", "scripts/olmo_f2_retain.py",
    "scripts/olmo_f1_retain.py", "scripts/olmo_fbt_retain.py", "scripts/olmo_tiled_retain.py",
    "scripts/olmo_retain.py", "scripts/openelm_retain.py", "scripts/docker_shell.sh",
    "docker/requirements-docker.txt", "AGENTS.md", "docs/olmo-fa4-environment.md", "docs/olmo1b-f3b-usage.md",
    "docs/olmo-resource-accounting.md", "docs/fbt-rt-nextlat-handoff.md",
    "docs/reports/olmo1b-f3/storage-receipt.json", "docs/reports/olmo1b-f3/results.md",
    "docs/fbt-rt-nextlat-research-plan-v4.md", "tests/test_olmo_f3b_retain.py",
    "tests/test_docker_shell.py", "tests/test_resource_estimates.py",
    "tests/test_olmo_rt_kernels.py", "tests/test_olmo_rt_execution_options.py",
}
RUNTIME_FILES = {"report.json", "configuration.json", "config.json", "run.log", "stdout.log",
                 "stderr.log", "test-results.txt", "capture-error.txt"}
REPORT_FILES = {"protocol.md", "results.md", "assessment.md", "test-results.txt", "summary.json",
    "results.json", "capability-ledger.json", "resource-ledger.json", "backend-summary.json",
    "retention-tests.txt", "environment.json", "profile-summary.json"}


def parse_prefix(prefix):
    prefix = prefix.rstrip("/")
    require(prefix.startswith(PREFIX_ROOT) and re.fullmatch(r"\d{8}T\d{6}Z", prefix[len(PREFIX_ROOT):]),
            f"Require one timestamp under {PREFIX_ROOT}")
    return tuple(prefix[5:].split("/", 1))


def source_name(name):
    require(isinstance(name, str), "Source names must be strings")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.parts and ".." not in path.parts
        and not any(part.startswith(".") or part.lower() in FORBIDDEN_NAMES for part in path.parts)
        and path.suffix in {".py", ".sh", ".json", ".yaml", ".yml", ".md"}
        and (name.startswith("cdrm/pretrained/") or name.startswith("scripts/")
             or name == "docs/reports/olmo1b-f3b/protocol.md"), "Unsafe/unscoped runtime source name")


def _checks(report, schema, failed):
    checks = report.get("checks", [])
    require(isinstance(checks, list) and all(isinstance(row, dict) and isinstance(row.get("name"), str)
            and type(row.get("passed")) is bool for row in checks), "Invalid explicit check outcomes")
    if failed:
        require(report.get("error_type") and isinstance(report.get("error_message"), str),
                "Failed diagnostic lacks its recorded error")
    elif schema in (NATIVE, TILE):
        require(checks and all(row["passed"] for row in checks), "Passing run contains missing/failed aggregate checks")
        # Nested strict-coordinate comparisons and named controls may fail while
        # the explicitly selected mixed engineering screen passes. Preserve them.
        if schema == NATIVE and report.get("configuration", {}).get("stage") == "capacity":
            require(report.get("capacity", {}).get("health", {}).get("passed") is True,
                    "Native capacity lacks passing state health")
    elif schema == PROFILE:
        require(report.get("health", {}).get("passed") is True
                and report.get("annotated_eager_profile") and report.get("graph_profile")
                and report.get("records"), "Profile lacks completed device/health observations")
        seconds = report.get("full_step", {}).get("median_wall_seconds")
        require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0,
                "Profile lacks positive uninstrumented timing")
    elif schema == FA4:
        comparisons = report.get("comparisons", {})
        require(set(comparisons) == {"output", "dq", "dk", "dv"}
                and report.get("capture_output_exact") is True, "FA4 smoke lacks its bounded checks")
        for row in comparisons.values():
            for name, limit in (("relative_l2", 1/64), ("max_relative", 1/16)):
                value = row.get(name)
                require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value < limit,
                        "FA4 smoke contradicts its reported passing screen")
        require(isinstance(report.get("interface", {}).get("path"), str)
                and _sha(report.get("interface", {}).get("sha256")) and report.get("versions"),
                "FA4 smoke lacks external package source identity")
    return checks


def validate_runs(runtime_dirs, *, project_root=ROOT, checkpoint_receipt=CHECKPOINT_RECEIPT,
                  report_dir, allow_failed_diagnostics=False):
    receipt = json.loads(safe_evidence(checkpoint_receipt.parent, checkpoint_receipt.name).read_text())
    native = {"path": "native/model.safetensors", "sha256": CHECKPOINT_SHA256, "size_bytes": CHECKPOINT_SIZE}
    reference = checkpoint_reference(receipt, native)
    protocol_sha = file_digest(safe_evidence(report_dir, "protocol.md"))["sha256"]
    runs, names, current_sources = [], set(), {}
    for value in runtime_dirs:
        directory = Path(value).absolute()
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", directory.name) and directory.name not in names,
                "Runtime names must be safe and unique")
        names.add(directory.name)
        path = safe_evidence(directory, "report.json")
        report = json.loads(path.read_text())
        schema, status = report.get("schema"), report.get("status")
        failed = status in ("failed", "capture_blocked")
        require(schema in ESSENTIAL_SOURCES and report.get("finished_utc")
                and (status == "passed" or failed), "Require a completed supported F3b report")
        require(not failed or allow_failed_diagnostics, "Failed runs require --allow-failed-diagnostics")
        checks = _checks(report, schema, failed)
        configuration = report.get("configuration")
        if schema != FA4:
            require(isinstance(configuration, dict) and configuration, "Missing runtime configuration")
        wandb = report.get("wandb", {})
        require(wandb.get("status") == ("synced_failed_experiment" if failed else "synced")
                and str(wandb.get("run_url", "")).startswith("https://wandb.ai/taylorbollman/"),
                "Missing synchronized W&B provenance")
        used_checkpoint = schema in (PROFILE, NATIVE)
        if used_checkpoint:
            require(checkpoint_reference(receipt, report.get("checkpoint", {})) == reference,
                    "Native checkpoint identity differs")
        elif "checkpoint" in report:
            raise ValueError("Synthetic tile/FA4 schema must not claim native-checkpoint consumption")
        hashes = report.get("source_hashes")
        if schema == FA4:
            hashes = {"scripts/olmo_fa4_smoke.py": report.get("script_sha256"),
                      "scripts/docker_shell.sh": report.get("launcher_sha256")}
        require(isinstance(hashes, dict) and ESSENTIAL_SOURCES[schema] <= hashes.keys(),
                "Missing essential runtime source inventory")
        snapshots, differences = {}, {}
        for name, expected in hashes.items():
            source_name(name); require(_sha(expected), "Invalid runtime source SHA256")
            snapshot = "source-snapshot/"+name
            require(file_digest(safe_evidence(directory, snapshot))["sha256"] == expected,
                    f"Run-local source snapshot differs from recorded bytes: {name}")
            snapshots[name] = snapshot
            candidate = project_root/name
            actual = file_digest(safe_evidence(project_root, name))["sha256"] if candidate.exists() or candidate.is_symlink() else None
            if actual is not None:
                require(name not in current_sources or current_sources[name] == actual, "Current sources changed while reading")
                current_sources[name] = actual
            if actual != expected:
                differences[name] = {"reported_sha256": expected, "current_sha256": actual}
        protocol = report.get("protocol_sha256")
        protocol_snapshot = None
        require(schema not in (NATIVE, TILE) or _sha(protocol), "Missing native/tile frozen protocol hash")
        if protocol is not None:
            require(_sha(protocol), "Invalid frozen protocol hash")
            choices = ["protocol.md", "source-snapshot/docs/reports/olmo1b-f3b/protocol.md"]
            existing = [name for name in choices if (directory/name).exists() or (directory/name).is_symlink()]
            require(existing, "Missing run-local frozen protocol snapshot")
            for name in existing:
                require(file_digest(safe_evidence(directory, name))["sha256"] == protocol,
                        "Run-local protocol snapshot differs from recorded bytes")
            protocol_snapshot = existing[0]
        for name in ("config.json", "configuration.json"):
            if (directory/name).exists() or (directory/name).is_symlink():
                require(json.loads(safe_evidence(directory, name).read_text()) == configuration,
                        "Standalone configuration differs from report")
        runs.append({"directory": directory, "report": report, "source_snapshots": snapshots,
            "protocol_snapshot": protocol_snapshot, "provenance": {
                "name": directory.name, "schema": schema, "status": status, "stage": report.get("stage"),
                "configuration": configuration, "counts_as_success": not failed,
                "diagnostic_failure_only": failed, "uses_native_checkpoint": used_checkpoint,
                "check_count": len(checks), "passed_check_count": sum(row["passed"] for row in checks),
                "report_sha256": file_digest(path)["sha256"], "wandb_url": wandb["run_url"],
                "reported_source_hashes": hashes, "exact_runtime_sources_complete": True,
                "current_source_hashes_match": not differences, "historical_source_differences": differences,
                "reported_protocol_sha256": protocol, "current_protocol_sha256": protocol_sha,
                "current_protocol_hash_matches": None if protocol is None else protocol == protocol_sha,
                "protocol_scope": "exact run-local snapshot" if protocol is not None else "not recorded by this diagnostic schema",
                "external_package_identity": {key: report.get(key) for key in ("interface", "versions")} if schema == FA4 else None,
                "excluded_fixtures": report.get("fixtures"),
            }})
    require(runs, "At least one completed F3b report is required")
    return runs, reference, current_sources


def collect_evidence(runs, current_sources, references, *, project_root=ROOT,
                     checkpoint_receipt=CHECKPOINT_RECEIPT, report_dir):
    members = {}
    def add(root, name, category):
        path, key = safe_evidence(root, name), f"{category}/{name}"
        require(key not in members or members[key] == path, "Conflicting evidence member")
        members[key] = path
    for name in sorted(set(current_sources) | EXTRA_PROJECT_FILES):
        add(project_root, name, "project")
    for directory, pattern in (("scripts", "olmo_f3b_*.py"), ("tests", "test_olmo_f3b*.py")):
        for path in sorted((project_root/directory).glob(pattern)):
            add(project_root, path.relative_to(project_root).as_posix(), "project")
    for name, files in SNAPSHOT_FILES.items():
        require({row["file"] for row in references[name]["files"]} == files, "Pinned source inventory differs")
        for filename in sorted(files | {"README.md", "manifest.json"}):
            add(project_root, f"cdrm/pretrained/{name}/{filename}", "project")
    for run in runs:
        directory = run["directory"]
        for snapshot in run["source_snapshots"].values():
            add(directory, snapshot, f"runtime/{directory.name}")
        if run["protocol_snapshot"]:
            add(directory, run["protocol_snapshot"], f"runtime/{directory.name}")
        for name in sorted(RUNTIME_FILES):
            if (directory/name).exists() or (directory/name).is_symlink():
                add(directory, name, f"runtime/{directory.name}")
        adjacent = directory.name+".log"
        if (directory.parent/adjacent).exists() or (directory.parent/adjacent).is_symlink():
            add(directory.parent, adjacent, "runtime-logs")
    for name in sorted(REPORT_FILES):
        if (report_dir/name).exists() or (report_dir/name).is_symlink():
            add(report_dir, name, "report")
    for path in sorted(report_dir.iterdir()):
        if path.suffix in {".pdf", ".png", ".csv"}:
            add(report_dir, path.name, "report")
    for name in ("protocol.md", "results.md", "assessment.md", "test-results.txt"):
        require("report/"+name in members, f"Missing final F3b evidence: {name}")
    add(checkpoint_receipt.parent, checkpoint_receipt.name, "provenance")
    require(sum(path.stat().st_size for path in members.values()) <= MAX_EVIDENCE_BYTES, "F3b evidence exceeds small-archive budget")
    return [(path, name) for name, path in sorted(members.items())]


def upload_verified(bucket, key, path, expected):
    from google.api_core.exceptions import PreconditionFailed
    require(Path(key).name in UPLOAD_FILES and path.name == Path(key).name, "Only known small evidence objects may be uploaded")
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
    return {"uri": f"gs://{bucket.name}/{key}", "generation": str(blob.generation), **expected}


def retain(args, *, project_root=ROOT):
    bucket_name, key = parse_prefix(args.storage_prefix)
    directories = [Path(path).absolute() for path in args.runtime_dir]
    docs, output, receipt = (Path(path).absolute() for path in (args.report_dir, args.output_dir, args.checkpoint_receipt))
    require(not any(output.resolve().is_relative_to(path.resolve()) for path in [*directories, docs]),
            "Retention output must be outside its evidence inputs")
    options = {"project_root": project_root, "checkpoint_receipt": receipt, "report_dir": docs,
               "allow_failed_diagnostics": args.allow_failed_diagnostics}
    runs, reference, sources = validate_runs(directories, **options)
    members = collect_evidence(runs, sources, verify_snapshots(project_root), project_root=project_root,
                               checkpoint_receipt=receipt, report_dir=docs)
    if not args.dry_run:
        from google.cloud import storage
        bucket = storage.Client().bucket(bucket_name)
        reference = verify_checkpoint_reference(bucket, reference)
    output.mkdir(parents=True, exist_ok=True)
    archive = output/"evidence.tar.gz"
    restore = ("# F3b native RT execution evidence\n\n"
        f"Reuse {reference['uri']} at generation {reference['generation']}; SHA256 {reference['sha256']}. "
        "No native or optimizer weights are archived/uploaded. Verify evidence-members.json before extraction.\n\n"
        "Each runtime directory contains the exact source bytes recorded by its report. Successful earlier "
        "runs may use earlier source/protocol versions; project/ is a separately recorded current overlay, "
        "not a claim that every run validated those current bytes. Failures remain diagnostic failures. "
        "Tile/FA4 fixtures do not consume the native checkpoint. FA4 smoke covers standalone attention only. "
        "Profiler annotations may overlap kernels/include gaps; use uninstrumented full_step for throughput. "
        "No quality or multi-GPU claim follows. Binary fixtures/checkpoints, W&B directories and secrets "
        "are excluded; deterministic diagnostic fixtures can be regenerated from recorded scripts. "
        "GPU execution requires the project container.\n")
    inventory = build_evidence_archive(archive, members, restore)
    refreshed, _, refreshed_sources = validate_runs(directories, **options)
    require(refreshed == runs and refreshed_sources == sources, "Evidence changed during retention")
    manifest = {"schema": SCHEMA, "kind": "retention-manifest", "checkpoint_reference": reference,
        "remote_checkpoint_verified": not args.dry_run, "checkpoint_uploaded": False,
        "checkpoint_compressed_in_evidence": False, "failed_diagnostics_count_as_success": False,
        "runs": [run["provenance"] for run in runs], "current_runtime_source_hashes": sources,
        "evidence": {"object": archive.name, **file_digest(archive), "members": inventory}}
    manifest_path = output/"retention-manifest.json"; write_json(manifest_path, manifest)
    if args.dry_run:
        return {"status": "dry_run", "archive": str(archive), "manifest": str(manifest_path),
                "member_count": len(inventory), "remote_checkpoint_verified": False, "uploaded": False}
    objects = [upload_verified(bucket, f"{key}/{path.name}", path, file_digest(path)) for path in (archive, manifest_path)]
    result = {"schema": SCHEMA, "kind": "storage-receipt", "status": "verified", "checkpoint_reference": reference,
              "objects": objects, "runs": manifest["runs"]}
    receipt_path = output/"storage-receipt.json"; write_json(receipt_path, result)
    result = {**result, "receipt_object": upload_verified(bucket, f"{key}/{receipt_path.name}", receipt_path, file_digest(receipt_path))}
    write_json(output/"upload-result.json", result); write_json(docs/"storage-receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--report-dir", type=Path, default=ROOT/"docs/reports/olmo1b-f3b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-prefix", required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, default=CHECKPOINT_RECEIPT)
    parser.add_argument("--allow-failed-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build/check local evidence; no cloud API or upload")
    result = retain(parser.parse_args(argv))
    print(json.dumps({key: result[key] for key in ("status", "archive", "member_count") if key in result}
        | ({"receipt_uri": result["receipt_object"]["uri"]} if "receipt_object" in result else {})), flush=True)


if __name__ == "__main__":
    main()
