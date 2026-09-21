#!/usr/bin/env python3
"""Retain bounded O4 source/data/evidence; model files have their own receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_data import load_lm_data
from scripts.olmo_o4_common import retain_file, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--phase", choices=("initial", "final"), required=True)
    args = parser.parse_args()
    config = json.loads((args.preflight / "configuration.json").read_text())
    if config["source_hashes"] != source_hashes():
        raise ValueError("Current sources differ from frozen learning configuration")
    if json.loads((args.preflight / "report.json").read_text())["status"] != "passed":
        raise ValueError("Preflight evidence did not pass")
    corpus = load_lm_data(args.data)
    if corpus.manifest_sha256 != config["data_manifest_sha256"]:
        raise ValueError("Prepared data differs from frozen learning configuration")
    if args.phase == "final":
        queue = json.loads((args.runs / "queue.json").read_text())
        if queue["status"] != "completed":
            raise ValueError("Do not label an unfinished queue as final retained evidence")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    members = {}
    def add(path, name):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or ".." in Path(name).parts:
            raise ValueError("Archive members must be regular explicit files")
        if path.suffix == ".pt" or path.name.startswith(".env"):
            raise ValueError("Model/secret files are not evidence archive members")
        members[name] = path
    for name in config["source_hashes"]:
        add(ROOT / name, "project/" + name)
    add(ROOT / "cdrm/pretrained/__init__.py", "project/cdrm/pretrained/__init__.py")
    for reference in ("_olmo_reference", "_nextlat_reference"):
        directory = ROOT / "cdrm/pretrained" / reference
        reference_manifest = json.loads((directory / "manifest.json").read_text())
        for name in ("manifest.json", "README.md", *(row["file"] for row in reference_manifest["files"])):
            path = directory / name
            add(path, "project/" + str(path.relative_to(ROOT)))
    for pattern in ("olmo_o4*.py", "olmo_lm_prepare_data.py"):
        for path in (ROOT / "scripts").glob(pattern):
            add(path, "project/" + str(path.relative_to(ROOT)))
    for pattern in ("test_olmo*.py", "test_nextlat*.py"):
        for path in (ROOT / "tests").glob(pattern):
            add(path, "project/" + str(path.relative_to(ROOT)))
    for name in ("AGENTS.md", "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v3.md",
                 "docs/olmo1b-nextlat-platform-usage.md", "docs/olmo1b-o4-usage.md",
                 "docs/reports/olmo1b-o1/storage-receipt.json", "docker/requirements-docker.txt"):
        path = ROOT / name
        if path.exists():
            add(path, "project/" + name)
    for path in (ROOT / "docs/reports/olmo1b-o4").glob("*"):
        if path.suffix in (".md", ".json", ".txt", ".pdf", ".png"):
            add(path, "project/" + str(path.relative_to(ROOT)))
    for name in ("report.json", "configuration.json", "configuration.pre-retention-retry.json",
                 "olmo_o4_train_before_retention_retry.py"):
        path = args.preflight / name
        if path.exists():
            add(path, "preflight/" + name)
    if args.phase == "initial":
        for name in ("manifest.json", *corpus.manifest["files"]):
            add(args.data / name, "prepared-data/" + name)
        for repo in ("code_search_net", "wikitext"):
            path = args.data.parent / "raw" / repo / "README.md"
            add(path, "dataset-cards/" + repo + ".md")
    if args.phase == "final":
        add(args.runs / "queue.json", "runs/queue.json")
        for arm in ("ordinary", "ordinary-nextlat", "rt", "rt-nextlat"):
            for path in (args.runs / arm).iterdir():
                if path.name in ("report.json", "events.jsonl") or path.name.endswith(".receipt.json"):
                    add(path, f"runs/{arm}/{path.name}")
    inventory = {name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for name, path in sorted(members.items())}
    archive = args.output_dir / "evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(members.items()):
            bundle.add(path, arcname=name, recursive=False)
    manifest = {"schema": "olmo-o4-evidence-v1", "phase": args.phase, "members": inventory,
                "configuration_sha256": sha256_file(args.preflight / "configuration.json"),
                "data_manifest_sha256": corpus.manifest_sha256,
                "native_checkpoint_reference": json.loads((ROOT / "docs/reports/olmo1b-o1/storage-receipt.json").read_text()),
                "model_checkpoints_in_archive": False,
                "scope": "Model optimizer checkpoints are separate immutable objects, referenced by per-arm receipts"}
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    receipts = [retain_file(path, args.prefix.rstrip("/") + "/" + args.phase + "/" + path.name)
                for path in (archive, manifest_path)]
    receipt_path = args.output_dir / "storage-receipt.json"
    receipt = {"schema": "olmo-o4-evidence-receipt-v1", "status": "verified", "phase": args.phase, "objects": receipts}
    write_json(receipt_path, receipt)
    receipt["receipt_object"] = retain_file(receipt_path, args.prefix.rstrip("/") + "/" + args.phase + "/storage-receipt.json")
    write_json(args.output_dir / "upload-result.json", receipt)
    print({"status": "verified", "phase": args.phase, "receipt": receipt["receipt_object"]["uri"]})


if __name__ == "__main__":
    main()
