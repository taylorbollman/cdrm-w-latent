#!/usr/bin/env python3
"""Retain explicit ordinary-fusions evidence without uploading model weights."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import olmo_ordinary_fusions_report as report
from scripts.olmo_rt_efficiency_retain import safe_relative, require
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference

SCHEMA = "olmo-ordinary-fusions-retention-v1"
MAX_BYTES = 64 * 1024**2
ARTIFACT = ".runtime/olmo1b-step60000/artifacts/artifact-manifest.json"
RECEIPT = "docs/reports/olmo1b-o1/storage-receipt.json"
REQUIRED_DOCS = ("protocol.md", "summary.json", "results.md", "usage.md",
                 "runtime-test-results.txt", "harness-test-results.txt", "evidence-test-results.txt")
PROJECT_FILES = (
    "scripts/olmo_ordinary_fusions_report.py", "scripts/olmo_ordinary_fusions_retain.py",
    "scripts/olmo_rt_efficiency_report.py", "scripts/olmo_rt_efficiency_retain.py",
    "scripts/olmo_rt_author_integration_report.py", "scripts/openelm_retain.py",
    "scripts/olmo_tiled_retain.py", "scripts/docker_shell.sh", "docker/requirements-docker.txt",
    "tests/test_olmo_ordinary_execution.py", "tests/test_olmo_ordinary_fusions.py",
    "tests/test_olmo_ordinary_rope.py", "tests/test_olmo_ordinary_optimizer_probe.py",
    "tests/test_olmo_ordinary_fusions_report.py", "tests/test_olmo_ordinary_fusions_retain.py",
    "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v4.md",
    "docs/olmo-resource-accounting.md", "docs/olmo-1b-250b-checkpoint-selection.json",
    RECEIPT, "AGENTS.md",
)


def queue_artifacts(runtime):
    """Allow only named queue descriptors/runners, never recurse over outputs."""
    for path in sorted(runtime.iterdir()):
        if path.name != "run_ordinary_queue.py" and re.fullmatch(
                r"ordinary-queue-[A-Za-z0-9][A-Za-z0-9_-]*\.(?:json|log)", path.name) is None:
            continue
        require(not path.is_symlink(), "Queue artifact cannot be a symlink: " + path.name)
        if path.is_file():
            yield path


def collect_evidence(root=ROOT):
    """Revalidate each frozen revision and every derived selected-summary field."""
    root = Path(root)
    docs, runtime = root / report.DOCS, root / report.RUNTIME
    summary_path = report.regular(docs / "summary.json", root)
    summary = json.loads(summary_path.read_text())
    require(summary.get("schema") == "olmo-ordinary-fusions-summary-v1"
            and summary.get("status") == "completed" and summary.get("runs"),
            "Require a completed explicit ordinary-fusions selection")
    primary = report.resolve_commit(root, summary.get("runtime_commit"))
    native = json.loads(report.regular(root / ARTIFACT, root).read_text())
    receipt = json.loads(report.regular(root / RECEIPT, root).read_text())
    checkpoint = checkpoint_reference(receipt, native["checkpoint"])
    members, rows, profiles, dependencies = {}, [], {}, {}

    def add(path, name):
        name = safe_relative(name).as_posix()
        report.regular(path, root)
        require(name not in members, "Duplicate evidence member: " + name)
        members[name] = path

    for selected in summary["runs"]:
        name = selected.get("name")
        require(name not in {row["name"] for row in rows}, "Duplicate selected run")
        row, raw, _ = report.load_run(root, name, selected.get("runtime_commit", primary))
        # Current-source differences are recomputed and recorded separately;
        # the immutable report/source/commit bytes remain independently pinned.
        for key in set(row) | set(selected):
            if key != "current_source_differences":
                require(selected.get(key) == row.get(key), "Selected run summary differs: " + key)
        if "checkpoint" in raw:
            require(checkpoint_reference(receipt, raw["checkpoint"]) == checkpoint,
                    "Run uses another native checkpoint")
        else:
            require(raw["status"] != "passed", "Successful run lacks checkpoint provenance")
        path = root / row["report_path"]
        prefix = "runtime/" + name + "/"
        add(path, prefix + "report.json")
        add(path.parent / "protocol.md", prefix + "protocol.md")
        add(runtime / (name + ".log"), "logs/" + name + ".log")
        for source in raw["source_hashes"]:
            add(path.parent / "source-snapshot" / source, prefix + "source-snapshot/" + source)
        sources = raw.get("dependencies", {}).get("dao_sources", {})
        dependencies[name] = len(sources)
        for source in sources:
            relative = "dependency-snapshot/flash_attn/" + source
            add(path.parent / relative, prefix + relative)
        for key in ("profile", "full_step_profile"):
            if row.get(key) is not None:
                profiles[name + "/" + key] = row[key]
                add(path.parent / row[key]["trace_file"], prefix + row[key]["trace_file"])
        rows.append(row)

    refreshed = report.summarize([row["name"] for row in rows], primary, root=root,
        overrides={row["name"]: row["runtime_commit"] for row in rows})
    for key in set(refreshed) | set(summary):
        if key not in {"created_utc", "runs"}:
            require(summary.get(key) == refreshed.get(key), "Derived summary differs from raw selection: " + key)
    for name in REQUIRED_DOCS:
        report.regular(docs / name, root)
    for path in sorted(docs.iterdir()):
        if path.suffix in {".md", ".json", ".txt", ".png", ".pdf", ".svg", ".csv"} and path.name != "storage-receipt.json":
            add(path, "report/" + path.name)
    for source in PROJECT_FILES:
        add(root / source, "project/" + source)
    add(root / ARTIFACT, "native-reference/artifact-manifest.json")
    add(runtime / "final-gpu.log", "logs/final-gpu.log")
    queued = {}
    for path in queue_artifacts(runtime):
        queued[path.name] = file_digest(path)
        if path not in members.values():
            add(path, "queue/" + path.name)
    require(sum(path.stat().st_size for name, path in members.items()
                if not name.endswith(("/operator-trace.json.gz", "/full-step-trace.json.gz"))) < MAX_BYTES,
            "Non-trace evidence exceeds 64 MiB")
    return summary, checkpoint, members, {
        "statuses": dict(Counter(row["status"] for row in rows)),
        "physical_optimizer_updates": sum(row["physical_optimizer_updates"] for row in rows),
        "source_pairs_checked": sum(row["source_pairs_checked"] for row in rows),
        "current_source_differences": {row["name"]: row["current_source_differences"] for row in rows},
        "run_revisions": {row["name"]: row["runtime_commit"] for row in rows},
        "dependency_files_checked": dependencies, "queue_artifacts": queued,
        "profiles": profiles, "profile_count": len(profiles),
        "profile_bytes": sum(item["trace_bytes"] for item in profiles.values()),
    }


def upload_verified(bucket, path, prefix):
    """Create a fresh object and verify downloaded bytes at that generation."""
    expected = file_digest(path)
    blob = bucket.blob(prefix + "/" + path.name)
    blob.metadata = {"sha256": expected["sha256"], "artifact_schema": SCHEMA}
    blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
    blob.reload()
    _check_remote(blob, expected)
    require(hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation)).hexdigest()
            == expected["sha256"], "Downloaded GCS bytes differ")
    return {"uri": "gs://fast-chunks/" + blob.name, "generation": str(blob.generation), **expected,
            "verification": "server size/MD5, SHA metadata and downloaded SHA256"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary, checkpoint, members, checked = collect_evidence()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / report.RUNTIME / ("retention-" + stamp)
    output.mkdir(exist_ok=False)
    prefix = "cdrm-w-latent/fbt-rt-nextlat/olmo-ordinary-fusions/" + stamp
    archive = output / "evidence.tar.gz"
    inventory = build_evidence_archive(archive, [(path, name) for name, path in sorted(members.items())],
        "# Ordinary OLMo fusion evidence\n\n"
        f"Primary runtime: {summary['runtime_commit']}. Each selected run records its exact source revision, "
        "source/protocol snapshots and imported Dao dependency bytes. Earlier failed attempts remain included "
        "and failed; completing operational diagnostics does not clear numerical compatibility. "
        "Timings are bounded full-CE optimizer-update measurements, not learning evidence. "
        f"Native checkpoint reused by reference: {checkpoint['uri']}, generation {checkpoint['generation']}, "
        f"SHA256 {checkpoint['sha256']}. No model or optimizer weights, datasets, credentials or W&B directories "
        "are uploaded. Exact selected logs, compressed traces and fixture hashes are retained. "
        "This partial project overlay requires the full Git checkout and project container for reproduction.\n")
    require(archive.stat().st_size < MAX_BYTES, "Compressed evidence exceeds 64 MiB")
    manifest = {"schema": SCHEMA, "runtime_commit": summary["runtime_commit"], "weights_uploaded": False,
        "checkpoint_reference": checkpoint, "runs": summary["runs"], "members": inventory,
        "evidence": file_digest(archive), **checked}
    manifest_path = output / "retention-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "archive": str(archive), "members": len(inventory),
            "runs": len(summary["runs"]), "statuses": checked["statuses"],
            "physical_optimizer_updates": checked["physical_optimizer_updates"]}))
        return
    from google.cloud import storage
    bucket = storage.Client().bucket("fast-chunks")
    verified_checkpoint = verify_checkpoint_reference(bucket, checkpoint)
    receipt = {"schema": SCHEMA, "status": "verified", "members": len(inventory), "weights_uploaded": False,
        "objects": [upload_verified(bucket, archive, prefix), upload_verified(bucket, manifest_path, prefix)],
        "checkpoint_reference": verified_checkpoint, "runs": len(summary["runs"]), **checked}
    receipt_path = output / "storage-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_object"] = upload_verified(bucket, receipt_path, prefix)
    (ROOT / report.DOCS / "storage-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "verified", "receipt": receipt["receipt_object"]["uri"]}))


if __name__ == "__main__":
    main()
