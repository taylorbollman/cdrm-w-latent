#!/usr/bin/env python3
"""Retain explicitly selected native/author LM integration evidence, without model weights."""
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
from scripts import olmo_rt_author_integration_report as report
from scripts.olmo_rt_efficiency_retain import safe_relative, require
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference

SCHEMA = "olmo-rt-author-integration-retention-v1"
MAX_BYTES = 64 * 1024**2
ARTIFACT = ".runtime/olmo1b-step60000/artifacts/artifact-manifest.json"
RECEIPT = "docs/reports/olmo1b-o1/storage-receipt.json"
DOC_FILES = ("protocol.md", "summary.json", "test-results.txt", "retention-test-results.txt",
    "combined-test-results.txt", "final-harness-test-results.txt",
    "localization-protocol.md", "localization-test-results.txt", "localization-selection.json",
    "results.md", "assessment.md", "usage.md", "numerical-assessment.md", "execution-notes.md",
    "throughput.png", "throughput.pdf")
PROJECT_FILES = (
    "scripts/olmo_rt_author_integration_report.py", "scripts/olmo_rt_author_integration_retain.py",
    "scripts/olmo_rt_efficiency_report.py", "scripts/olmo_rt_efficiency_retain.py",
    "scripts/openelm_retain.py", "scripts/olmo_tiled_retain.py", "scripts/docker_shell.sh",
    "docker/requirements-docker.txt", "tests/test_olmo_rt_author_integration_evidence.py",
    "tests/test_olmo_author_integration.py", "tests/test_olmo_rt_author_integration.py",
    "tests/test_olmo_rt_author_localize.py",
    "docs/reports/olmo-rt-author-comparison/author-port-audit.md",
    "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v4.md",
    "docs/olmo-rt-efficiency-and-author-comparison-plan.md", "docs/olmo-resource-accounting.md",
    "docs/olmo-1b-250b-checkpoint-selection.json", RECEIPT, "AGENTS.md",
)


def queue_artifacts(runtime):
    """Only the known runner and safe capacity JSON/log basenames, no recursion."""
    for path in sorted(runtime.iterdir()):
        if path.name not in {"run_capacity_queue.py", "run_integration_queue.py"} and re.fullmatch(
                r"(?:capacity|integration)-[A-Za-z0-9][A-Za-z0-9_-]*\.(?:json|log)", path.name) is None:
            continue
        require(not path.is_symlink(), "Queue artifact cannot be a symlink: " + path.name)
        if path.is_file():
            yield path


def collect_localizations(root, checkpoint, receipt, add):
    """Optional, explicitly selected zero-update diagnostic overlays."""
    selection = root / report.DOCS / "localization-selection.json"
    if not selection.exists():
        return []
    report.regular(selection, root)
    selected = json.loads(selection.read_text())
    require(isinstance(selected, list) and selected, "Require explicit localization selection")
    checked, seen = [], set()
    for item in selected:
        name = item.get("name")
        require(isinstance(name, str) and re.fullmatch(r"localize-[A-Za-z0-9_-]+", name),
                "Unsafe localization name")
        require(name not in seen, "Duplicate localization selection")
        seen.add(name)
        revision = report.resolve_commit(root, item.get("runtime_commit"))
        directory = root / report.RUNTIME / name
        path = directory / "report.json"
        report.regular(path, root)
        raw = json.loads(path.read_text())
        require(file_digest(path)["sha256"] == item.get("report_sha256"), "Localization report changed")
        require(raw.get("schema") == "olmo-rt-author-localization-v1"
                and raw.get("status") in {"passed", "failed", "oom", "completed"}
                and raw.get("runtime_commit") == revision
                and raw.get("physical_optimizer_updates") == 0, "Invalid localization report")
        if "checkpoint" in raw:
            require(checkpoint_reference(receipt, raw["checkpoint"]) == checkpoint,
                    "Localization uses another checkpoint")
        else:
            require(raw["status"] in {"failed", "oom"}, "Completed localization lacks checkpoint")
        hashes = raw.get("source_hashes", {})
        require(isinstance(hashes, dict)
                and "scripts/olmo_rt_author_localize.py" in hashes
                and report.ESSENTIAL_SOURCES <= set(hashes), "Incomplete localization source overlay")
        prefix = "localization/" + name + "/"
        add(path, prefix + "report.json")
        for source, expected in hashes.items():
            safe_relative(source)
            report.safe_source(source)
            snapshot = directory / "source-snapshot" / source
            report.regular(snapshot, root)
            require(file_digest(snapshot)["sha256"] == expected
                    == report.frozen_digest(str(root), revision, source), "Localization source differs: " + source)
            add(snapshot, prefix + "source-snapshot/" + source)
        protocol = directory / "protocol.md"
        report.regular(protocol, root)
        require(file_digest(protocol)["sha256"] == raw.get("protocol_sha256")
                == report.frozen_digest(str(root), revision, report.DOCS + "/localization-protocol.md"),
                "Localization protocol differs")
        add(protocol, prefix + "protocol.md")
        add(root / report.RUNTIME / (name + ".log"), "logs/" + name + ".log")
        checked.append({"name": name, "runtime_commit": revision, "status": raw["status"],
                        "report_sha256": item["report_sha256"], "source_pairs_checked": len(hashes),
                        "physical_optimizer_updates": 0})
    return checked


def collect_evidence(root=ROOT):
    """Revalidate recorded bytes/revisions; unsuccessful attempts remain included."""
    root = Path(root)
    docs, runtime = root / report.DOCS, root / report.RUNTIME
    summary = json.loads((docs / "summary.json").read_text())
    require(summary.get("status") == "completed" and summary.get("runs"), "Require a completed explicit selection")
    primary = report.resolve_commit(root, summary.get("runtime_commit"))
    native = json.loads((root / ARTIFACT).read_text())
    receipt = json.loads((root / RECEIPT).read_text())
    checkpoint = checkpoint_reference(receipt, native["checkpoint"])
    members, rows, raws = {}, [], []

    def add(path, name):
        name = safe_relative(name).as_posix()
        report.regular(path, root)
        require(name not in members, "Duplicate evidence member: " + name)
        members[name] = path

    for selected in summary["runs"]:
        name = selected.get("name")
        require(name not in {row["name"] for row in rows}, "Duplicate selected run")
        row, raw = report.load_run(root, name, selected.get("runtime_commit", primary))
        require(all(selected.get(key) == row[key] for key in ("report_path", "report_sha256", "status")),
                "Selected report bytes/path/status changed")
        if "gate_groups" in selected:
            require(selected["gate_groups"] == row["gate_groups"], "Summary gate groups differ from raw evidence")
        if "checkpoint" in raw:
            require(checkpoint_reference(receipt, raw["checkpoint"]) == checkpoint, "Run uses another native checkpoint")
        else:
            require(raw["status"] != "passed", "Successful run lacks a native checkpoint reference")
        path = root / row["report_path"]
        prefix = "runtime/" + name + "/"
        add(path, prefix + "report.json")
        add(path.parent / "protocol.md", prefix + "protocol.md")
        add(runtime / (name + ".log"), "logs/" + name + ".log")
        for source in raw["source_hashes"]:
            add(path.parent / "source-snapshot" / source, prefix + "source-snapshot/" + source)
        rows.append(row)
        raws.append(raw)
    localizations = collect_localizations(root, checkpoint, receipt, add)
    updates = sum(row["physical_optimizer_updates"] for row in rows)
    pairs = sum(row["source_pairs_checked"] for row in rows)
    require(updates == summary.get("physical_optimizer_updates"), "Summary physical update count differs")
    require(pairs == summary.get("source_pairs_checked"), "Summary source-pair count differs")
    refreshed = report.summarize([row["name"] for row in rows], primary, root=root,
        overrides={row["name"]: row["runtime_commit"] for row in rows})
    for field in ("statuses", "capacity", "resource_cards", "checkpoint", "comparison_groups"):
        require(summary.get(field) == refreshed[field], "Derived summary differs from raw selection: " + field)
    for required in ("protocol.md", "summary.json", "test-results.txt"):
        require((docs / required).is_file(), "Missing required report evidence: " + required)
    for name in DOC_FILES:
        if (docs / name).exists():
            add(docs / name, "report/" + name)
    for source in PROJECT_FILES:
        add(root / source, "project/" + source)
    add(root / ARTIFACT, "native-reference/artifact-manifest.json")
    add(runtime / "final-gpu.log", "logs/final-gpu.log")
    queued = {}
    for path in queue_artifacts(runtime):
        queued[path.name] = file_digest(path)
        # Selected run logs are already retained under logs/. Keep just one
        # copy; queue descriptors/runners and additional queue logs go here.
        if path not in members.values():
            add(path, "queue/" + path.name)
    require(sum(path.stat().st_size for path in members.values()) < MAX_BYTES, "Evidence exceeds 64 MiB")
    return summary, checkpoint, members, {
        "statuses": dict(Counter(row["status"] for row in rows)),
        "physical_optimizer_updates": updates, "source_pairs_checked": pairs,
        "current_source_differences": {row["name"]: row["current_source_differences"] for row in rows},
        "run_revisions": {row["name"]: row["runtime_commit"] for row in rows},
        "queue_artifacts": queued,
        "localizations": localizations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summary, checkpoint, members, checked = collect_evidence()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / report.RUNTIME / ("retention-" + stamp)
    output.mkdir(exist_ok=False)
    prefix = "cdrm-w-latent/fbt-rt-nextlat/olmo-rt-author-integration/" + stamp
    archive = output / "evidence.tar.gz"
    inventory = build_evidence_archive(archive, [(path, name) for name, path in sorted(members.items())],
        "# Native/author RT integration in OLMo-1B\n\n"
        f"Primary runtime: {summary['runtime_commit']}. Use each run's recorded runtime commit and exact source, "
        "protocol snapshots; failed and repaired attempts may use different revisions. "
        "Failures remain failures. Compatibility gates and same-candidate operational checks are distinct. "
        "Timings are five-update full-CE language-model benchmarks, not learning evidence; K2 counts input tokens once. "
        f"Native checkpoint reused by reference: {checkpoint['uri']}, generation {checkpoint['generation']}, "
        f"SHA256 {checkpoint['sha256']}. No model/optimizer weights, datasets, credentials or W&B files "
        "are uploaded. Exact selected logs, source snapshots and hashed fixture manifests are retained. "
        "This partial project overlay requires the full Git checkout/container to reproduce GPU execution.\n")
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

    def upload(path):
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

    receipt = {"schema": SCHEMA, "status": "verified", "members": len(inventory), "weights_uploaded": False,
        "objects": [upload(archive), upload(manifest_path)], "checkpoint_reference": verified_checkpoint,
        "runs": len(summary["runs"]), **checked}
    receipt_path = output / "storage-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_object"] = upload(receipt_path)
    (ROOT / report.DOCS / "storage-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "verified", "receipt": receipt["receipt_object"]["uri"]}))


if __name__ == "__main__":
    main()
