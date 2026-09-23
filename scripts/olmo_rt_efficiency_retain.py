#!/usr/bin/env python3
"""Retain selected RT-efficiency evidence; reuse the native checkpoint by reference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference

SCHEMA = "olmo-rt-efficiency-retention-v1"
REPORT_SCHEMA = "olmo-rt-efficiency-v1"
MAX_BYTES = 64 * 1024**2
RUNTIME_RELATIVE = ".runtime/olmo-rt-efficiency"
DOCS_RELATIVE = "docs/reports/olmo-rt-efficiency"
PROTOCOL_RELATIVE = DOCS_RELATIVE + "/protocol.md"
ARTIFACT_RELATIVE = ".runtime/olmo1b-step60000/artifacts/artifact-manifest.json"
RECEIPT_RELATIVE = "docs/reports/olmo1b-o1/storage-receipt.json"
PROJECT_FILES = (
    "scripts/olmo_rt_efficiency_retain.py", "scripts/olmo_rt_efficiency_report.py",
    "scripts/openelm_retain.py", "scripts/olmo_tiled_retain.py",
    "scripts/docker_shell.sh", "docker/requirements-docker.txt",
    "tests/test_olmo_rt_efficiency.py", "tests/test_olmo_rt_efficiency_retain.py",
    "tests/test_olmo_rope_reuse.py", "tests/test_olmo_kv_writes.py", "tests/test_resource_estimates.py",
    "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v4.md",
    "docs/olmo-rt-efficiency-and-author-comparison-plan.md",
    "docs/olmo-resource-accounting.md",
    "docs/olmo-1b-250b-checkpoint-selection.json", RECEIPT_RELATIVE, "AGENTS.md",
)
ESSENTIAL_SOURCES = {
    "scripts/olmo_rt_efficiency.py", "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_rope.py",
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_static.py",
    "cdrm/pretrained/static_training.py", "cdrm/pretrained/nextlat.py",
    "cdrm/pretrained/static_nextlat.py", "cdrm/pretrained/resource_estimates.py",
    "cdrm/pretrained/olmo_rt_kernels.py", "cdrm/pretrained/olmo_rt_backward_kernels.py",
    "cdrm/pretrained/olmo_rt_recompute_kernels.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(value):
    require(isinstance(value, str) and value, "Require a nonempty relative path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts
            and not any(p.startswith(".env") or p in {".git", ".docker-home", "wandb"} for p in path.parts),
            "Unsafe evidence path: " + value)
    return path


def git_bytes(root, revision, name):
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{7,40}", revision),
            "Runtime revision must be a Git hex commit")
    safe_relative(name)
    return subprocess.check_output(["git", "show", revision + ":" + name], cwd=root)


def verify_operator_trace(raw, directory):
    """Verify exact declared compressed trace bytes without loading its JSON."""
    requested = raw.get("configuration", {}).get("profile", False)
    require(type(requested) is bool, "Profile configuration must be boolean")
    profile = raw.get("profile")
    if profile is None:
        require(not requested or raw.get("status") != "passed", "Passing requested profile is missing")
        return None
    require(requested and isinstance(profile, dict), "Undeclared or invalid profile")
    require(profile.get("trace_file") == "operator-trace.json.gz", "Unexpected profile trace filename")
    expected_size, expected_sha = profile.get("trace_bytes"), profile.get("trace_sha256")
    require(type(expected_size) is int and expected_size > 0, "Profile trace size must be a positive integer")
    require(isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha),
            "Profile trace SHA256 is invalid")
    path = directory / profile["trace_file"]
    require(path.is_file() and not path.is_symlink()
            and path.resolve().parent == directory.resolve(), "Profile trace must be a regular run-local file")
    digest = file_digest(path)
    require(digest["size_bytes"] == expected_size and digest["sha256"] == expected_sha,
            "Profile trace differs from recorded bytes")
    with path.open("rb") as stream:
        require(stream.read(2) == b"\x1f\x8b", "Profile trace is not gzip data")
    return {"trace_file": profile["trace_file"], "trace_bytes": expected_size,
            "trace_sha256": expected_sha}


def collect_evidence(root=ROOT):
    """Validate the explicit final selection, including failed diagnostic reports."""
    docs, runtime = root / DOCS_RELATIVE, root / RUNTIME_RELATIVE
    summary = json.loads((docs / "summary.json").read_text())
    require(summary.get("status") == "completed" and summary.get("runs"),
            "Require the completed nonempty report selection")
    revision = summary.get("runtime_commit")
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{7,40}", revision),
            "Summary lacks its frozen runtime commit")
    native = json.loads((root / ARTIFACT_RELATIVE).read_text())
    receipt = json.loads((root / RECEIPT_RELATIVE).read_text())
    checkpoint = checkpoint_reference(receipt, native["checkpoint"])
    members, names, source_differences, profiles = {}, set(), {}, {}
    updates = source_pairs = 0
    statuses = {}

    def add(path, name):
        rel = safe_relative(name)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                "Evidence must be a regular file inside the project: " + str(path))
        require(rel.as_posix() not in members, "Duplicate archive member: " + name)
        members[rel.as_posix()] = path

    for row in summary["runs"]:
        name = row.get("name")
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                and name not in names, "Require unique safe run names")
        names.add(name)
        report_path = root / safe_relative(row["report_path"])
        require(report_path == runtime / name / "report.json", "Unexpected selected report path")
        require(file_digest(report_path)["sha256"] == row["report_sha256"], "Selected report changed")
        raw = json.loads(report_path.read_text())
        status = raw.get("status")
        require(raw.get("schema") == REPORT_SCHEMA and raw.get("finished_utc")
                and status in {"passed", "failed", "oom"} and row.get("status") == status,
                "Require a completed report with an explicit matching status")
        if status == "passed":
            require(raw.get("checks") and all(c.get("passed") is True for c in raw["checks"]),
                    "Passing report has missing or failed checks")
        else:
            require(raw.get("error", {}).get("type") and isinstance(raw["error"].get("message"), str),
                    "Failed report lacks its recorded error")
        if "checkpoint" in raw:
            require(checkpoint_reference(receipt, raw["checkpoint"]) == checkpoint,
                    "Run checkpoint differs from selected native checkpoint")
        else:
            require(status != "passed", "Successful run lacks checkpoint provenance")
        count = raw.get("physical_optimizer_updates")
        require(type(count) is int and count >= 0, "Invalid physical update count")
        updates += count
        statuses[status] = statuses.get(status, 0) + 1
        run_revision = row.get("runtime_commit", revision)
        prefix = "runtime/" + name + "/"
        hashes = raw.get("source_hashes")
        require(isinstance(hashes, dict) and ESSENTIAL_SOURCES <= hashes.keys(),
                "Missing essential runtime sources")
        differences = {}
        for source, digest in hashes.items():
            rel = safe_relative(source)
            require((source.startswith("cdrm/pretrained/") or source.startswith("scripts/"))
                    and (rel.suffix == ".py" or source == "cdrm/pretrained/_fbt_reference/manifest.json"),
                    "Unexpected runtime source: " + source)
            snapshot = report_path.parent / "source-snapshot" / source
            require(file_digest(snapshot)["sha256"] == digest
                    == hashlib.sha256(git_bytes(root, run_revision, source)).hexdigest(),
                    "Runtime snapshot differs from recorded bytes or frozen commit: " + source)
            current_digest = file_digest(root / source)["sha256"] if (root / source).is_file() else None
            if current_digest != digest:
                differences[source] = {"reported_sha256": digest, "current_sha256": current_digest}
            source_pairs += 1
            add(snapshot, prefix + "source-snapshot/" + source)
        source_differences[name] = differences
        protocol = report_path.parent / "protocol.md"
        require(file_digest(protocol)["sha256"] == raw["protocol_sha256"]
                == hashlib.sha256(git_bytes(root, run_revision, PROTOCOL_RELATIVE)).hexdigest(),
                "Protocol snapshot differs from its frozen commit")
        add(report_path, prefix + "report.json")
        add(protocol, prefix + "protocol.md")
        add(runtime / (name + ".log"), "logs/" + name + ".log")
        trace = verify_operator_trace(raw, report_path.parent)
        if "profile" in row:
            require(row["profile"] == trace, "Selected summary profile differs from its report")
        if trace is not None:
            profiles[name] = trace
            add(report_path.parent / trace["trace_file"], prefix + trace["trace_file"])
    require(updates == summary["physical_optimizer_updates"], "Summary update total differs from reports")
    require(source_pairs == summary["source_pairs_checked"], "Summary source-pair total differs from reports")
    require((docs / "test-results.txt").is_file(), "Missing CPU test evidence")
    for path in sorted(docs.iterdir()):
        if path.suffix in {".md", ".json", ".txt", ".png", ".pdf", ".csv"} and path.name != "storage-receipt.json":
            add(path, "report/" + path.name)
    for name in PROJECT_FILES:
        add(root / name, "project/" + name)
    add(root / ARTIFACT_RELATIVE, "native-reference/artifact-manifest.json")
    add(runtime / "final-gpu.log", "logs/final-gpu.log")
    # Chrome traces are already compressed. Keep their exact bytes even when
    # the raw source/report inventory exceeds the final compressed limit; the
    # final archive below is still strictly bounded to 64 MiB before upload.
    require(sum(path.stat().st_size for name, path in members.items()
                if not name.endswith("/operator-trace.json.gz")) < MAX_BYTES,
            "Non-trace evidence exceeds the 64 MiB uncompressed budget")
    return summary, checkpoint, members, {"statuses": statuses, "source_pairs_checked": source_pairs,
        "physical_optimizer_updates": updates, "current_source_differences": source_differences,
        "profiles": profiles, "profile_count": len(profiles),
        "profile_bytes": sum(record["trace_bytes"] for record in profiles.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary, checkpoint, members, validated = collect_evidence()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / RUNTIME_RELATIVE / ("retention-" + stamp)
    out.mkdir(exist_ok=False)
    prefix = f"cdrm-w-latent/fbt-rt-nextlat/olmo-rt-efficiency/{stamp}"
    archive = out / "evidence.tar.gz"
    inventory = build_evidence_archive(archive, [(p, n) for n, p in sorted(members.items())],
        "# RT efficiency and native OLMo evidence\n\n"
        f"Frozen primary runtime commit: {summary['runtime_commit']}. Per-run overrides, where present, "
        "are recorded in the selected summary. Use exact run-local source/protocol snapshots. "
        "The native OLMo-1B step60000 checkpoint is reused without upload: "
        f"{checkpoint['uri']} at generation {checkpoint['generation']}; SHA256 {checkpoint['sha256']}. "
        "Its manifest, selection audit and O1 storage receipt are included. No model/optimizer weights, "
        "dataset, credential or W&B directory is included. The project overlay is partial; restore the "
        "full Git repository/environment before use. GPU work requires the project container. "
        "Exact compressed operator traces, where requested and completed, are retained with recorded "
        "SHA256 and byte counts. Failed diagnostics remain failed in the selection. These bounded integration/performance "
        "checks do not establish long-training precision, paper reproduction or quality.\n")
    require(archive.stat().st_size < MAX_BYTES, "Compressed evidence exceeds 64 MiB")
    manifest = {"schema": SCHEMA, "runtime_commit": summary["runtime_commit"], "weights_uploaded": False,
        "checkpoint_reference": checkpoint, "runs": summary["runs"], "members": inventory,
        "evidence": file_digest(archive), **validated}
    manifest_path = out / "retention-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "archive": str(archive), "members": len(inventory),
            "runs": len(summary["runs"]), **validated}))
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
        return {"uri": f"gs://fast-chunks/{blob.name}", "generation": str(blob.generation), **expected,
                "verification": "server size/MD5, SHA metadata and downloaded SHA256"}

    receipt = {"schema": SCHEMA, "status": "verified", "members": len(inventory), "weights_uploaded": False,
        "objects": [upload(archive), upload(manifest_path)], "checkpoint_reference": verified_checkpoint,
        "runs": len(summary["runs"]), **validated}
    receipt_path = out / "storage-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_object"] = upload(receipt_path)
    (ROOT / DOCS_RELATIVE / "storage-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "verified", "receipt": receipt["receipt_object"]["uri"]}))


if __name__ == "__main__":
    main()
