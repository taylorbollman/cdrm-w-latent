#!/usr/bin/env python3
"""Retain small F2 multi-run evidence and reuse the immutable O1 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from scripts.olmo_f1_retain import (safe_evidence, MAX_EVIDENCE_BYTES, UPLOAD_FILES,
                                   CHECKPOINT_RECEIPT, FORBIDDEN_NAMES)
from scripts.olmo_fbt_retain import verify_snapshots, SNAPSHOT_FILES
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote

SCHEMA = "olmo-f2-retention-v1"
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-f2-health-capacity/"
CORE_SCHEMA = "olmo-f2-health-capacity-v1"
GRAPH_SCHEMA = "olmo-f2-graph-probe-v1"
LOCALIZATION_SCHEMA = "olmo-f2-graph-localization-v1"
EXTRA_PROJECT_FILES = {
    "cdrm/__init__.py", "cdrm/pretrained/__init__.py", "scripts/olmo_f2_retain.py",
    "scripts/olmo_f1_retain.py", "scripts/olmo_fbt_retain.py", "scripts/olmo_tiled_retain.py",
    "scripts/olmo_retain.py", "scripts/openelm_retain.py", "scripts/docker_shell.sh",
    "docker/requirements-docker.txt", "tests/test_olmo_f2_retain.py",
    "tests/test_olmo_ordinary_checkpointing.py", "AGENTS.md",
    "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v4.md",
}
RUNTIME_FILES = {"report.json", "configuration.json", "config.json", "run.log", "stdout.log",
                 "stderr.log", "test-results.txt", "capture-error.txt"}
REPORT_FILES = {
    "protocol.md", "results.md", "assessment.md", "test-results.txt", "summary.json",
    "report.json", "results.json", "health-summary.json", "capacity-summary.json",
    "graph-summary.json", "checkpoint-parity.json", "capability-ledger.json",
    "graph-tests.txt", "activation-tests.txt", "checkpoint-tests.txt", "retention-tests.txt",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_prefix(prefix):
    prefix = prefix.rstrip("/")
    require(prefix.startswith(PREFIX_ROOT) and
            re.fullmatch(r"\d{8}T\d{6}Z", prefix[len(PREFIX_ROOT):]) is not None,
            f"Require one timestamp under {PREFIX_ROOT}")
    return tuple(prefix[5:].split("/", 1))


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _source_name(name):
    require(isinstance(name, str), "Source inventory names must be strings")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.parts and ".." not in path.parts
            and not any(part.startswith(".") or part.lower() in FORBIDDEN_NAMES for part in path.parts)
            and path.suffix in {".py", ".json", ".yaml", ".yml", ".md"}, "Unsafe/non-source inventory entry")


def _passing_flags(value):
    if isinstance(value, dict):
        for name, child in value.items():
            if name in ("passed", "finite"):
                require(child is True, "Passing run contains a failed nested check")
            _passing_flags(child)
    elif isinstance(value, list):
        for child in value:
            _passing_flags(child)


def validate_runs(runtime_dirs, *, project_root=ROOT, checkpoint_receipt=CHECKPOINT_RECEIPT,
                  report_dir=None, allow_failed_graph=False):
    """Keep successful current lineage distinct from historical blocked capture."""
    receipt = json.loads(safe_evidence(checkpoint_receipt.parent, checkpoint_receipt.name).read_text())
    runs, stages, names, successful_hashes = [], set(), set(), {}
    reference = None
    for directory in runtime_dirs:
        directory = Path(directory).absolute()
        require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", directory.name) is not None
                and directory.name not in names, "Runtime directory names must be safe and unique")
        names.add(directory.name)
        path = safe_evidence(directory, "report.json")
        report = json.loads(path.read_text())
        schema, status = report.get("schema"), report.get("status")
        graph_like = schema in (GRAPH_SCHEMA, LOCALIZATION_SCHEMA)
        localization = schema == LOCALIZATION_SCHEMA
        historical = graph_like and status in ("capture_blocked", "failed")
        require(not (graph_like and status == "failed") or allow_failed_graph,
                "Failed graph reports require explicit --allow-failed-graph diagnostic retention")
        require(schema in (CORE_SCHEMA, GRAPH_SCHEMA, LOCALIZATION_SCHEMA) and report.get("finished_utc")
                and (status == ("completed" if localization else "passed") or historical),
                "Require completed passing core/graph or explicitly retained graph diagnostic")
        config = report.get("config") if schema == CORE_SCHEMA else report.get("configuration")
        require(isinstance(config, dict) and config, "Missing run configuration")
        wandb = report.get("wandb", {})
        require(wandb.get("status") == ("synced_failed_experiment" if historical else "synced")
                and str(wandb.get("run_url", "")).startswith("https://wandb.ai/taylorbollman/"),
                "Missing synchronized run tracking")
        if schema == CORE_SCHEMA:
            stage = config.get("stage")
            require(stage in ("health", "capacity", "checkpoint"), "Unknown core run stage")
            require(isinstance(report.get("rows"), list) and report["rows"]
                    and all(row.get("passed") is True for row in report["rows"]), "Core run rows are missing or failed")
            stages.add(stage)
        else:
            stage = "graph_localization" if localization else "graph"
            if historical:
                require(report.get("error_type") and report.get("stage") != "complete",
                        "Historical graph failure lacks its failure stage/error")
                if status == "capture_blocked":
                    require(report.get("stage") == "capture" and report.get("capture_succeeded") is False,
                            "Historical capture_blocked report must identify an unsuccessful capture")
            elif localization:
                require(report.get("capture_succeeded") is True and report.get("stage") == "complete"
                        and isinstance(report.get("comparisons"), list) and report["comparisons"]
                        and all(type(row.get("passed")) is bool for row in report["comparisons"]),
                        "Completed localization lacks completed comparison observations")
                require(all(type(report.get(name)) is bool for name in (
                    "all_comparisons_within_original_budget", "all_comparisons_bitwise_equal", "structural_invariants_passed")),
                    "Localization must explicitly distinguish completion and numerical outcomes")
                require(report["all_comparisons_within_original_budget"] ==
                        (report["structural_invariants_passed"] and all(row["passed"] for row in report["comparisons"])),
                        "Localization aggregate budget flag contradicts its observations")
            else:
                require(report.get("capture_succeeded") is True and report.get("stage") == "complete"
                        and isinstance(report.get("comparisons"), list) and report["comparisons"]
                        and all(row.get("passed") is True for row in report["comparisons"]),
                        "Successful graph report lacks completed equivalence checks")
        if not historical and not localization:
            _passing_flags(report)
        current_reference = checkpoint_reference(receipt, report.get("checkpoint", {}))
        require(reference is None or reference == current_reference, "Runs reference different native checkpoints")
        reference = current_reference
        hashes = report.get("source_hashes")
        require(isinstance(hashes, dict) and hashes, "Missing run source inventory")
        essential = {"cdrm/pretrained/olmo_tiled.py", "scripts/olmo_f2_graph_probe.py"} if graph_like else {
            "cdrm/pretrained/olmo_tiled.py", "scripts/olmo_f2_health_capacity.py", "scripts/olmo_f2_observe.py"}
        if localization:
            essential.add("scripts/olmo_f2_graph_localize.py")
        require(essential <= hashes.keys(), "Missing essential current source inventory")
        comparison, matching_diagnostic_sources, snapshots = {}, [], {}
        for name, digest in hashes.items():
            _source_name(name); require(_sha(digest), "Invalid runtime source hash")
            source = project_root/name
            # A historical source may have been removed. Existing entries still
            # go through the common symlink/secret/member-size protections.
            actual = file_digest(safe_evidence(project_root, name))["sha256"] if source.exists() or source.is_symlink() else None
            if historical:
                if actual != digest:
                    comparison[name] = {"reported_sha256": digest, "current_sha256": actual}
                    snapshot = f"source-snapshot/{name}"
                    if (directory/snapshot).exists() or (directory/snapshot).is_symlink():
                        saved = safe_evidence(directory, snapshot)
                        require(file_digest(saved)["sha256"] == digest, "Historical source snapshot differs from recorded bytes")
                        snapshots[name] = snapshot
                        comparison[name]["exact_historical_snapshot"] = snapshot
                else:
                    matching_diagnostic_sources.append(name)
            else:
                require(actual == digest, f"Successful source is absent or changed: {name}")
                if localization:
                    matching_diagnostic_sources.append(name)
                else:
                    require(name not in successful_hashes or successful_hashes[name] == digest,
                            "Successful runs require incompatible source versions")
                    successful_hashes[name] = digest
        protocol = report.get("protocol_sha256")
        if protocol is not None:
            require(_sha(protocol), "Invalid protocol hash")
            if report_dir is not None:
                actual = file_digest(safe_evidence(report_dir, "protocol.md"))["sha256"]
                if historical and actual != protocol:
                    comparison["report/protocol.md"] = {"reported_sha256": protocol, "current_sha256": actual}
                else:
                    require(actual == protocol, "Successful run protocol changed")
        for filename in ("configuration.json", "config.json"):
            if (directory/filename).exists():
                require(json.loads(safe_evidence(directory, filename).read_text()) == config,
                        "Standalone run configuration differs")
        runs.append({"directory": directory, "report": report,
            "matching_diagnostic_sources": matching_diagnostic_sources, "source_snapshots": snapshots,
            "provenance": {"name": directory.name, "stage": stage, "status": status,
                "report_sha256": file_digest(path)["sha256"], "wandb_url": wandb["run_url"],
                "historical_failure_only": historical, "counts_as_success": not historical and not localization,
                "diagnostic_failure_only": status == "failed",
                "diagnostic_only": historical or localization,
                "execution_completed": status in ("passed", "completed"),
                "localization_outcomes": {name: report.get(name) for name in (
                    "all_comparisons_within_original_budget", "all_comparisons_bitwise_equal", "structural_invariants_passed")}
                    if localization else None,
                "current_source_hashes_match": not comparison,
                "historical_source_mismatches": comparison,
                "historical_runtime_sources_complete": (all(name in snapshots for name in comparison
                    if name != "report/protocol.md") if historical else None),
                "source_snapshot_scope": "Current bytes are checked for matching sources; changed historical bytes require an exact run-local source snapshot."}})
    require(stages == {"health", "capacity", "checkpoint"}, "Final F2 retention requires passing health, capacity and checkpoint stages")
    return runs, reference, successful_hashes


def collect_evidence(runs, successful_hashes, references, *, project_root=ROOT,
                     checkpoint_receipt=CHECKPOINT_RECEIPT, report_dir):
    members = {}
    def add(root, name, category):
        path = safe_evidence(root, name)
        key = f"{category}/{name}"
        require(key not in members or members[key] == path, "Conflicting archive member")
        members[key] = path
    for name in sorted(set(successful_hashes) | EXTRA_PROJECT_FILES):
        add(project_root, name, "project")
    for directory, pattern in (("scripts", "olmo_f2_*.py"), ("tests", "test_olmo_f2*.py")):
        for path in sorted((project_root/directory).glob(pattern)):
            add(project_root, path.relative_to(project_root).as_posix(), "project")
    usage = "docs/olmo1b-f2-usage.md"
    if (project_root/usage).exists():
        add(project_root, usage, "project")
    for name, files in SNAPSHOT_FILES.items():
        require({row["file"] for row in references[name]["files"]} == files, "Pinned source snapshot inventory differs")
        for filename in sorted(files | {"README.md", "manifest.json"}):
            add(project_root, f"cdrm/pretrained/{name}/{filename}", "project")
    for run in runs:
        directory = run["directory"]
        for name in run["matching_diagnostic_sources"]:
            add(project_root, name, "project")
        for snapshot in run["source_snapshots"].values():
            add(directory, snapshot, f"runtime/{directory.name}")
        for filename in sorted(RUNTIME_FILES):
            if (directory/filename).exists():
                add(directory, filename, f"runtime/{directory.name}")
        # Only the explicitly selected run's adjacent log, never all runtime logs.
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
        require(f"report/{filename}" in members, f"Missing final F2 evidence: {filename}")
    add(checkpoint_receipt.parent, checkpoint_receipt.name, "provenance")
    require(sum(path.stat().st_size for path in members.values()) <= MAX_EVIDENCE_BYTES,
            "F2 evidence exceeds the small-archive budget")
    return [(path, name) for name, path in sorted(members.items())]


def upload_verified(bucket, key, path, expected):
    """Same create-only verified transport as F1, with F2 object metadata."""
    from google.api_core.exceptions import PreconditionFailed
    require(Path(key).name in UPLOAD_FILES and path.name == Path(key).name,
            "Only whitelisted small F2 evidence objects may be uploaded")
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
    docs, output, receipt = (Path(path).absolute() for path in
                            (args.report_dir, args.output_dir, args.checkpoint_receipt))
    require(not any(output.resolve().is_relative_to(path.resolve()) for path in [*directories, docs]),
            "Retention output must be outside its evidence inputs")
    allow_failed = getattr(args, "allow_failed_graph", False)
    runs, reference, sources = validate_runs(directories, checkpoint_receipt=receipt, report_dir=docs,
                                            allow_failed_graph=allow_failed)
    members = collect_evidence(runs, sources, verify_snapshots(ROOT), checkpoint_receipt=receipt, report_dir=docs)
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    reference = verify_checkpoint_reference(bucket, reference)
    output.mkdir(parents=True, exist_ok=True)
    archive = output/"evidence.tar.gz"
    restore = ("# F2 functionality and execution evidence\n\n"
        f"Reuse {reference['uri']} at generation {reference['generation']}; SHA256 {reference['sha256']}, "
        f"{reference['size_bytes']} bytes. Native weights are not included or uploaded again.\n\n"
        "Verify evidence-members.json before restoring project/. Current successful source snapshots and pinned "
        "native/NextLat/FBT references are included. Each runtime/<run>/report.json keeps that run's original hashes, "
        "scope and W&B identity. Historical capture_blocked and explicitly retained failed graph reports are failure "
        "evidence only. The manifest lists source mismatches; any included run-local source-snapshot files match "
        "the original report hashes exactly. Missing historical bytes are explicitly not claimed reproduced. "
        "A completed graph-localization run only means its controls executed; numerical disagreements and structural "
        "flags remain in that report, and even passing fixed-input controls do not clear changed-token/weight replay. "
        "report/ contains final interpretation and scoped tests/plots. The O1 receipt locates original artifact/tokenizer "
        "metadata. This archive excludes model/optimizer checkpoints, W&B directories and secrets. GPU commands "
        "must use the project container. Eager capacity and graph microbenchmarks have different scopes.\n")
    inventory = build_evidence_archive(archive, members, restore)
    refreshed, _, refreshed_sources = validate_runs(directories, checkpoint_receipt=receipt, report_dir=docs,
                                                   allow_failed_graph=allow_failed)
    require(refreshed == runs and refreshed_sources == sources, "Run evidence changed during retention")
    manifest = {"schema": SCHEMA, "kind": "retention-manifest", "checkpoint_reference": reference,
        "checkpoint_uploaded": False, "checkpoint_compressed_in_evidence": False,
        "runs": [run["provenance"] for run in runs], "current_successful_source_hashes": sources,
        "historical_failures_count_as_success": False,
        "evidence": {"object": archive.name, **file_digest(archive), "members": inventory}}
    manifest_path = output/"retention-manifest.json"; write_json(manifest_path, manifest)
    objects = [upload_verified(bucket, f"{key}/{path.name}", path, file_digest(path)) for path in (archive, manifest_path)]
    result = {"schema": SCHEMA, "kind": "storage-receipt", "status": "verified",
              "checkpoint_reference": reference, "objects": objects, "runs": manifest["runs"]}
    receipt_path = output/"storage-receipt.json"; write_json(receipt_path, result)
    uploaded = upload_verified(bucket, f"{key}/{receipt_path.name}", receipt_path, file_digest(receipt_path))
    result = {**result, "receipt_object": uploaded}
    write_json(output/"upload-result.json", result); write_json(docs/"storage-receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--allow-failed-graph", action="store_true",
                        help="Retain graph status=failed as unresolved diagnostic evidence, never as successful coverage")
    parser.add_argument("--report-dir", type=Path, default=ROOT/"docs/reports/olmo1b-f2")
    parser.add_argument("--checkpoint-receipt", type=Path, default=CHECKPOINT_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-prefix", required=True)
    result = retain(parser.parse_args(argv))
    print(json.dumps({"status": result["status"], "receipt_uri": result["receipt_object"]["uri"]}), flush=True)


if __name__ == "__main__":
    main()
