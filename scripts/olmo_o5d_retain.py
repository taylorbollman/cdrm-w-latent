#!/usr/bin/env python3
"""Retain explicit O5d evidence and verify reused source/mixed checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.openelm_retain import build_evidence_archive
from scripts.olmo_o4_common import retain_file as retain_verified_file
from scripts.olmo_o5c_common import endpoint_metadata as source_metadata, source_hashes
from scripts.olmo_o5d_common import endpoint_metadata
from scripts.olmo_o5d_diagnose import diagnostic_source_hashes
from scripts.olmo_o5d_report import validate_report

PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5d-online-diagnostic/"
DATA_MANIFEST = ROOT/".runtime/olmo1b-step60000/o4-data-01/prepared/manifest.json"
EXTRA_SOURCES = (
    "scripts/olmo_o5d_common.py", "scripts/olmo_o5d_diagnose.py", "scripts/olmo_o5d_report.py",
    "scripts/olmo_o5d_retain.py", "tests/test_olmo_o5d_common.py", "tests/test_olmo_o5d_diagnose.py",
    "tests/test_olmo_o5d_report.py", "scripts/openelm_retain.py", "scripts/olmo_o4_report.py",
    "AGENTS.md", "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v3.md",
    "docs/olmo1b-o5d-usage.md",
)
PARENT_RECEIPTS = (
    "olmo1b-o4/initial-storage-receipt.json", "olmo1b-o5b/final-storage-receipt.json",
    "olmo1b-o5c/final-storage-receipt.json",
)
REPORT_SUFFIXES = {".md", ".json", ".pdf", ".png", ".txt"}
FORBIDDEN_PARTS = {"hf-cache", "wandb", "__pycache__", ".git", ".docker-home"}


def checked_prefix(value):
    prefix = value.rstrip("/")+"/"
    if (not prefix.startswith(PREFIX_ROOT)
            or re.fullmatch(r"\d{8}T\d{6}Z/", prefix[len(PREFIX_ROOT):]) is None):
        raise ValueError("Use the O5d lineage followed by exactly one YYYYMMDDTHHMMSSZ timestamp")
    return prefix


def retain_file(path, uri):
    """Restrict the existing create-only byte verifier to O5d evidence objects."""
    parent, name = uri.rsplit("/", 1)
    checked_prefix(parent)
    if name not in ("evidence.tar.gz", "manifest.json", "storage-receipt.json"):
        raise ValueError("Only the explicit O5d evidence objects may be uploaded")
    return retain_verified_file(Path(path), uri)


def verify_checkpoint(storage_client, record):
    """Require the previously retained object generation, checksums and size."""
    uri = record.get("uri", "")
    if not uri.startswith("gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/"):
        raise ValueError("Endpoint checkpoint is outside the retained project lineage")
    bucket, key = uri[5:].split("/", 1)
    blob = storage_client.bucket(bucket).get_blob(key)
    if (blob is None or str(blob.generation) != record["generation"]
            or blob.size != record["size_bytes"] or blob.md5_hash != record["md5_base64"]
            or (blob.metadata or {}).get("sha256") != record["sha256"]):
        raise ValueError("Retained endpoint generation or bytes differ")
    return {**record, "reused_without_upload": True}


def collect_members(run_dir, report_dir, report):
    """Only allowlisted evidence; no recursive archive or model payloads."""
    members = {}

    def add(path, name):
        path = Path(path)
        archive = PurePosixPath(name)
        if (archive.is_absolute() or not archive.parts or ".." in archive.parts
                or any(p in FORBIDDEN_PARTS or p.startswith(".env") for p in archive.parts)
                or path.suffix == ".pt" or not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT.resolve())):
            raise ValueError(f"Explicit regular project evidence member required: {name}")
        for parent in path.absolute().parents:
            if parent == ROOT:
                break
            if parent.is_symlink():
                raise ValueError(f"Symlinked evidence ancestors are not allowed: {name}")
        if name in members and members[name] != path:
            raise ValueError("Duplicate archive member with different source")
        members[name] = path

    sources = {**report["source_hashes"], **report["diagnostic_source_hashes"]}
    if report["source_hashes"] != source_hashes() or report["diagnostic_source_hashes"] != diagnostic_source_hashes():
        raise ValueError("Frozen evaluation sources changed before retention")
    for name, digest in sources.items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError(f"Frozen source differs: {name}")
        add(ROOT/name, "project/"+name)
    for name in EXTRA_SOURCES:
        add(ROOT/name, "project/"+name)
    add(run_dir/"report.json", "run/report.json")
    for name in ("report.json", "results.md", "finite-online.pdf", "finite-online.png",
                 "adaptation-transfer.pdf", "adaptation-transfer.png"):
        add(report_dir/name, "report/"+name)
    for path in sorted(report_dir.iterdir()):
        if path.suffix in REPORT_SUFFIXES and path.name != "storage-receipt.json":
            add(path, "report/"+path.name)
    if sha256_file(DATA_MANIFEST) != report["configuration"]["base_data_manifest_sha256"]:
        raise ValueError("Retained evaluation data manifest differs")
    add(DATA_MANIFEST, "parents/evaluation-data-manifest.json")
    authorities = {"source": source_metadata(), "mixed": endpoint_metadata()}
    for name, authority in authorities.items():
        if (authority["checkpoint"] != report["endpoints"][name]["checkpoint"]
                or authority["checkpoint"] != report["checkpoint_identities"][name]
                or authority["report_sha256"] != report["input_file_hashes"][name+"_report"]):
            raise ValueError("Diagnostic endpoint authority differs from its retained parent")
        add(authority["report_path"], "parents/"+name+"-endpoint-report.json")
    if report["input_file_hashes"]["data_manifest"] != sha256_file(DATA_MANIFEST):
        raise ValueError("Diagnostic input data identity differs")
    for name in PARENT_RECEIPTS:
        path = ROOT/"docs/reports"/name
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "verified":
            raise ValueError("Parent evidence must have a verified retention receipt")
        add(path, "parents/"+name.replace("/", "-"))
    return [(path, name) for name, path in sorted(members.items())]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "report-dir", "output-dir"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--storage-prefix", required=True)
    args = parser.parse_args(argv)
    prefix = checked_prefix(args.storage_prefix)
    report = json.loads((args.run_dir/"report.json").read_text())
    validate_report(report)
    rendered = json.loads((args.report_dir/"report.json").read_text())
    validate_report(rendered)
    if {k: v for k, v in rendered.items() if k != "summary"} != report or "summary" not in rendered:
        raise ValueError("Rendered report differs from the completed diagnostic")
    members = collect_members(args.run_dir, args.report_dir, report)
    from google.cloud import storage
    client = storage.Client()
    references = {name: verify_checkpoint(client, endpoint["checkpoint"]["storage"])
                  for name, endpoint in report["endpoints"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    archive = args.output_dir/"evidence.tar.gz"
    inventory = build_evidence_archive(archive, members,
        "Restore project/ into the repository. The unchanged source and mixed model checkpoints are referenced "
        "in manifest.json and parent receipts; verify their generation, size, MD5 and SHA256 before reuse. "
        "The O4 initial evidence archive retains the prepared evaluation data identified by parents/evaluation-data-manifest.json. "
        "No weights or dataset payloads are duplicated here. Verify all archive member hashes before running. "
        "Read the O5d usage and frozen protocol; the retained diagnostic is complete, not a queued run.\n")
    manifest = {"schema": "olmo-o5d-evidence-v1", "status": "verified",
        "diagnostic_report_sha256": sha256_file(args.run_dir/"report.json"),
        "rendered_report_sha256": sha256_file(args.report_dir/"report.json"),
        "source_checkpoints": references, "members": inventory}
    write_json(args.output_dir/"manifest.json", manifest)
    objects = [retain_file(args.output_dir/name, prefix+name) for name in ("evidence.tar.gz", "manifest.json")]
    receipt = {"schema": "olmo-o5d-evidence-receipt-v1", "status": "verified",
               "source_checkpoints": references, "objects": objects}
    write_json(args.output_dir/"storage-receipt.json", receipt)
    retain_file(args.output_dir/"storage-receipt.json", prefix+"storage-receipt.json")
    write_json(args.report_dir/"storage-receipt.json", receipt)
    print({"status": "verified", "objects": [obj["uri"] for obj in objects],
           "reused_checkpoints": list(references)}, flush=True)


if __name__ == "__main__":
    main()
