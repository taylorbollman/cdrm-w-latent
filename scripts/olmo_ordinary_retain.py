#!/usr/bin/env python3
"""Retain the explicitly selected small ordinary-throughput evidence, no weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.openelm_retain import build_evidence_archive, file_digest, _check_remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    docs = ROOT / "docs/reports/olmo-ordinary-throughput"
    selected = json.loads((docs / "summary.json").read_text())
    assert selected["status"] == "completed" and len(selected["runs"]) == 7
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / ".runtime/olmo-ordinary-throughput" / ("retention-" + stamp)
    out.mkdir(exist_ok=False)
    prefix = f"cdrm-w-latent/fbt-rt-nextlat/olmo-ordinary-throughput/{stamp}"
    members = {}

    def add(path, name):
        assert path.is_file() and not path.is_symlink() and not PurePosixPath(name).is_absolute()
        assert ".." not in PurePosixPath(name).parts and name not in members
        members[name] = path

    for row in selected["runs"]:
        path = ROOT / row["report_path"]
        assert path.parent.parent == ROOT / ".runtime/olmo-ordinary-throughput"
        raw = json.loads(path.read_text())
        assert file_digest(path)["sha256"] == row["report_sha256"]
        assert raw["status"] == "passed" and raw["physical_optimizer_updates"] == 6
        category = "runtime/" + path.parent.name + "/"
        for name, digest in raw["source_hashes"].items():
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and ".." not in rel.parts
            assert rel.suffix == ".py" or name == "cdrm/pretrained/_fbt_reference/manifest.json"
            snapshot = path.parent / "source-snapshot" / name
            assert file_digest(snapshot)["sha256"] == digest == file_digest(ROOT/name)["sha256"]
            add(snapshot, category + "source-snapshot/" + name)
        assert file_digest(path.parent / "protocol.md")["sha256"] == raw["protocol_sha256"]
        add(path, category + "report.json")
        add(path.parent / "protocol.md", category + "protocol.md")
        add(path.parent.parent / (path.parent.name + ".log"), "logs/" + path.parent.name + ".log")
        if "profile" in raw:
            trace = path.parent / "operator-trace.json.gz"
            assert file_digest(trace)["sha256"] == raw["profile"]["trace_sha256"]
            add(trace, category + trace.name)
    for path in sorted(docs.iterdir()):
        if path.suffix in (".md", ".json", ".txt", ".png", ".pdf") and path.name != "storage-receipt.json":
            add(path, "report/" + path.name)
    for name in ("scripts/olmo_ordinary_retain.py", "scripts/openelm_retain.py",
                 "scripts/docker_shell.sh", "docker/requirements-docker.txt",
                 "tests/test_olmo_ordinary_throughput.py", "docs/fbt-rt-nextlat-handoff.md",
                 "docs/fbt-rt-nextlat-research-plan-v4.md", "AGENTS.md"):
        add(ROOT/name, "project/" + name)
    add(ROOT / ".runtime/olmo-ordinary-throughput/final-gpu.log", "logs/final-gpu.log")
    assert sum(p.stat().st_size for p in members.values()) < 64 * 1024**2
    archive = out / "evidence.tar.gz"
    inventory = build_evidence_archive(archive, [(p,n) for n,p in sorted(members.items())],
        "# Ordinary OLMo throughput evidence\n\nRandom disposable weights are omitted. "
        "Use recorded initialization/token seeds and exact run-local sources. Runtime commit71fbccd. "
        "All seven runs use six-layer native OLMo, no RT/FBT/NextLat. GPU work requires the project container. "
        "The project overlay is partial; use the Git repository for full environment restoration. "
        "No dataset, pretrained checkpoint, optimizer checkpoint, secret or W&B directory is included. "
        "These are bounded throughput diagnostics, not paper reproduction, convergence or full precision clearance.\n")
    manifest = {"schema": "olmo-ordinary-throughput-retention-v1", "runtime_commit": "71fbccd",
        "weights_uploaded": False, "runs": selected["runs"], "members": inventory,
        "evidence": file_digest(archive), "physical_optimizer_updates": 42}
    manifest_path = out / "retention-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "archive": str(archive), "members": len(inventory)}))
        return
    from google.cloud import storage
    bucket = storage.Client().bucket("fast-chunks")

    def upload(path):
        expected = file_digest(path)
        blob = bucket.blob(prefix + "/" + path.name)
        blob.metadata = {"sha256": expected["sha256"], "artifact_schema": manifest["schema"]}
        blob.upload_from_filename(str(path), if_generation_match=0, checksum="md5")
        blob.reload()
        _check_remote(blob, expected)
        assert hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation)).hexdigest() == expected["sha256"]
        return {"uri": f"gs://fast-chunks/{blob.name}", "generation": str(blob.generation), **expected,
                "verification": "server size/MD5, SHA metadata and downloaded SHA256"}

    receipt = {"schema": manifest["schema"], "status": "verified", "members": len(inventory),
        "weights_uploaded": False, "objects": [upload(archive), upload(manifest_path)],
        "runs": 7, "physical_optimizer_updates": 42, "source_pairs_checked": selected["source_pairs_checked"]}
    receipt_path = out / "storage-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt_object"] = upload(receipt_path)
    (docs / "storage-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "verified", "receipt": receipt["receipt_object"]["uri"]}))


if __name__ == "__main__":
    main()
