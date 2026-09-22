#!/usr/bin/env python3
"""Retain O5b evidence while reusing the verified O4 data and O1 checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.lm_data import load_lm_data
from cdrm.pretrained.olmo_artifacts import CHECKPOINT_FILENAME, FILE_SPECS, MANIFEST_FILENAME
from scripts.olmo_fbt_retain import (SNAPSHOT_FILES, validate_artifact_metadata, verify_snapshots)
from scripts.olmo_tiled_retain import checkpoint_reference, verify_checkpoint_reference
from scripts.openelm_retain import _safe_member, _check_remote, file_digest, build_evidence_archive

SCHEMA = "olmo-o5b-evidence-v1"
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5b-code-pilot/"
ARTIFACTS = ROOT / ".runtime/olmo1b-step60000/artifacts"
O1_RECEIPT = ROOT / "docs/reports/olmo1b-o1/storage-receipt.json"
O4_RECEIPT = ROOT / "docs/reports/olmo1b-o4/initial-storage-receipt.json"
O4_MANIFEST = ROOT / ".runtime/olmo1b-step60000/o4-initial-retention-01/manifest.json"
O4_PREFIX = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o4-code-pilot/20260921T220500Z/initial/"
O4_OBJECTS = {
    "evidence.tar.gz": {"generation": "1790028911751263", "size_bytes": 68381709,
        "sha256": "ab0d8f4e0f39b9e9107f37dcc3da0d509beec760d828dc3a2c3b7c77b36888f4", "md5_base64": "DrQDiCwzpwoZW1cnnUHsqw=="},
    "manifest.json": {"generation": "1790028912626318", "size_bytes": 18385,
        "sha256": "51c39dcffa9f21ed85254058b3fb1c37520c583a2ec4024501c102704571cf35", "md5_base64": "BgskcfIaByszAnljzvXM9w=="},
}
REPORT_FILES = {"protocol.md", "results.md", "assessment.md", "test-results.txt", "preflight-summary.json",
                "final-comparison.json", "learning-curves.pdf", "learning-curves.png", "initial-storage-receipt.json",
                "online-comparison.pdf", "online-comparison.png"}
EXTRA_PROJECT_FILES = {
    "cdrm/__init__.py", "cdrm/pretrained/__init__.py", "scripts/openelm_retain.py",
    "scripts/olmo_retain.py", "scripts/olmo_tiled_retain.py", "scripts/olmo_fbt_retain.py", "scripts/olmo_o4_report.py",
    "scripts/docker_shell.sh", "docker/requirements-docker.txt", "AGENTS.md",
    "docs/fbt-rt-nextlat-handoff.md", "docs/fbt-rt-nextlat-research-plan-v3.md",
    "docs/olmo1b-fbt-usage.md", "docs/olmo1b-o5b-usage.md",
}
UPLOAD_FILES = {"evidence.tar.gz", "manifest.json", "storage-receipt.json"}


def parse_prefix(prefix):
    prefix = prefix.rstrip("/")
    if not prefix.startswith(PREFIX_ROOT) or re.fullmatch(r"\d{8}T\d{6}Z", prefix[len(PREFIX_ROOT):]) is None:
        raise ValueError(f"Require one timestamp under {PREFIX_ROOT}")
    return tuple(prefix[5:].split("/", 1))


def prepared_data_reference(receipt, manifest_path, corpus):
    """Bind the selected prepared files to their already retained O4 archive."""
    if receipt.get("schema") != "olmo-o4-evidence-receipt-v1" or receipt.get("status") != "verified" or receipt.get("phase") != "initial":
        raise ValueError("Require the verified initial O4 data receipt")
    references = []
    for name, expected in O4_OBJECTS.items():
        uri = O4_PREFIX + name
        candidates = [row for row in receipt.get("objects", []) if row.get("uri") == uri]
        if len(candidates) != 1 or any(candidates[0].get(key) != value for key, value in expected.items()):
            raise ValueError("O4 prepared-data object differs from its immutable pin")
        references.append({"uri": uri, **expected})
    if file_digest(manifest_path) != {key: O4_OBJECTS["manifest.json"][key] for key in ("sha256", "size_bytes", "md5_base64")}:
        raise ValueError("Prior O4 archive manifest bytes differ")
    prior = json.loads(manifest_path.read_text())
    if prior.get("schema") != "olmo-o4-evidence-v1" or prior.get("phase") != "initial" or prior.get("data_manifest_sha256") != corpus.manifest_sha256:
        raise ValueError("Prepared data does not match the retained O4 archive")
    current = {"manifest.json": {"sha256": corpus.manifest_sha256, "size_bytes": (corpus.root / "manifest.json").stat().st_size},
               **corpus.manifest["files"]}
    for name, record in current.items():
        member = prior.get("members", {}).get("prepared-data/" + name, {})
        if any(member.get(key) != record[key] for key in ("sha256", "size_bytes")):
            raise ValueError(f"Retained O4 prepared member differs: {name}")
    return references


def verify_reference(bucket, record):
    """Read-only remote generation and byte verification; no download/upload."""
    bucket_name, key = record["uri"][5:].split("/", 1)
    if bucket_name != bucket.name:
        raise ValueError("Reference bucket differs")
    blob = bucket.get_blob(key)
    if blob is None:
        raise ValueError("Previously retained reference is missing")
    _check_remote(blob, record)
    if str(blob.generation) != record["generation"]:
        raise ValueError("Previously retained reference generation differs")
    return {**record, "reused_without_upload": True}


def learning_source_files():
    from scripts.olmo_o5b_common import SOURCE_FILES
    return set(SOURCE_FILES)


def validate_learning(preflight, runs, phase, corpus, *, project_root=ROOT):
    config = json.loads(_safe_member(preflight, "configuration.json").read_text())
    report = json.loads(_safe_member(preflight, "report.json").read_text())
    if config.get("schema") != "olmo-o5b-pilot-config-v1" or report.get("schema") != "olmo-o5b-preflight-v1":
        raise ValueError("O5b configuration/preflight schema differs")
    hashes = config.get("source_hashes", {})
    if set(hashes) != learning_source_files():
        raise ValueError("Frozen O5b source inventory differs")
    for name, digest in hashes.items():
        if file_digest(_safe_member(project_root, name))["sha256"] != digest:
            raise ValueError(f"Frozen learning source changed: {name}")
    if report.get("status") != "passed" or not report.get("finished_utc") or report.get("source_hashes") != hashes:
        raise ValueError("Require completed passing preflight with identical sources")
    if corpus.manifest_sha256 != config.get("data_manifest_sha256") or report.get("data_manifest_sha256") != corpus.manifest_sha256:
        raise ValueError("Prepared data differs from frozen O5b configuration")
    if report.get("schedule") != config.get("schedule"):
        raise ValueError("Preflight schedule differs from frozen configuration")
    if phase == "final":
        queue = json.loads(_safe_member(runs, "queue.json").read_text())
        if queue.get("schema") != "olmo-o5b-queue-v1" or queue.get("status") != "completed":
            raise ValueError("Do not retain an unfinished queue as final evidence")
    elif phase != "initial":
        raise ValueError("Unknown retention phase")
    return config, report


def collect_evidence(preflight, runs, config, references, *, phase, data, report_dir,
                     project_root=ROOT, artifacts=ARTIFACTS, o1_receipt=O1_RECEIPT,
                     o4_receipt=O4_RECEIPT, o4_manifest=O4_MANIFEST, arms):
    members = {}
    def add(root, name, category):
        if Path(name).suffix in {".pt", ".safetensors", ".bin", ".npy"}:
            raise ValueError("Model and prepared-data bytes must not enter evidence archive")
        members[f"{category}/{name}"] = _safe_member(root, name)
    project_files = set(config["source_hashes"])
    project_files.update(name for name in EXTRA_PROJECT_FILES if (project_root / name).exists())
    for directory, patterns in (("scripts", ("olmo_o5b*.py",)), ("tests", ("test_olmo*.py", "test_nextlat*.py"))):
        for pattern in patterns:
            project_files.update(path.relative_to(project_root).as_posix() for path in (project_root / directory).glob(pattern))
    for name in sorted(project_files):
        add(project_root, name, "project")
    for directory, snapshot in references.items():
        if directory not in SNAPSHOT_FILES or len(snapshot["files"]) != len(SNAPSHOT_FILES[directory]) or {row["file"] for row in snapshot["files"]} != SNAPSHOT_FILES[directory]:
            raise ValueError("Reference snapshot differs from explicit whitelist")
        for name in sorted(SNAPSHOT_FILES[directory] | {"manifest.json", "README.md"}):
            add(project_root, f"cdrm/pretrained/{directory}/{name}", "project")
    for name in ("configuration.json", "report.json"):
        add(preflight, name, "preflight")
    for name in (MANIFEST_FILENAME, "checkpoint-inspection.json"):
        add(artifacts, name, "artifacts")
    for name in FILE_SPECS:
        if name != CHECKPOINT_FILENAME:
            add(artifacts, "native/" + name, "artifacts")
    add(data, "manifest.json", "prepared-data")
    for path, name in ((o1_receipt, "o1-storage-receipt.json"), (o4_receipt, "o4-initial-storage-receipt.json"), (o4_manifest, "o4-initial-manifest.json")):
        members["provenance/" + name] = _safe_member(path.parent, path.name)
    for name in sorted(REPORT_FILES):
        if (report_dir / name).exists():
            add(report_dir, name, "report")
    required = {"protocol.md"} | ({"results.md", "final-comparison.json", "learning-curves.pdf", "learning-curves.png",
                                  "online-comparison.pdf", "online-comparison.png"} if phase == "final" else set())
    if any("report/" + name not in members for name in required):
        raise ValueError("Required protocol/final comparison artifacts are missing")
    if phase == "final":
        add(runs, "queue.json", "runs")
        for arm in arms:
            for name in ("report.json", "events.jsonl"):
                add(runs, f"{arm}/{name}", "runs")
            for path in sorted((runs / arm).iterdir()):
                if re.fullmatch(r"update-\d+\.receipt\.json", path.name):
                    add(runs, f"{arm}/{path.name}", "runs")
    return [(path, name) for name, path in sorted(members.items())]


def upload_verified(bucket, key, path, expected):
    from google.api_core.exceptions import PreconditionFailed
    if Path(key).name not in UPLOAD_FILES:
        raise ValueError("Only bounded evidence objects may be uploaded")
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


def verify_training_checkpoints(bucket, runs, arms, prefix):
    references = {}
    for arm in arms:
        report = json.loads(_safe_member(runs, f"{arm}/report.json").read_text())
        records = report.get("checkpoints", [])
        if not records:
            raise ValueError("Completed arm has no retained checkpoint")
        references[arm] = []
        for record in records:
            name = Path(record["path"]).name
            storage = record.get("storage", {})
            if not re.fullmatch(r"update-\d+\.pt", name) or storage.get("uri") != prefix.rstrip("/") + f"/{arm}/{name}":
                raise ValueError("Training checkpoint URI differs from selected pilot")
            if any(storage.get(key) != record.get(key) for key in ("sha256", "size_bytes")):
                raise ValueError("Training checkpoint receipt bytes differ")
            disk = json.loads(_safe_member(runs, f"{arm}/{name.removesuffix('.pt')}.receipt.json").read_text())
            recorded = {key: value for key, value in record.items() if key != "local_file_removed_after_verified_successor"}
            if disk != recorded or record.get("local_file_removed_after_verified_successor", True) is not True:
                raise ValueError("Checkpoint report and local receipt differ")
            references[arm].append(verify_reference(bucket, storage))
    return references


def validate_final_comparison(preflight, runs, docs, *, project_root=ROOT):
    from scripts.olmo_o5b_report import validate_runs
    expected = validate_runs(preflight, runs)
    comparison = json.loads(_safe_member(docs, "final-comparison.json").read_text())
    if any(comparison.get(name) != value for name, value in expected.items()):
        raise ValueError("Final comparison differs from current strict run validation")
    if comparison.get("report_source_sha256") != file_digest(_safe_member(project_root, "scripts/olmo_o5b_report.py"))["sha256"]:
        raise ValueError("Final comparison reporter source changed")
    inputs = {"preflight": _safe_member(preflight, "report.json"), "configuration": _safe_member(preflight, "configuration.json"),
              **{arm: _safe_member(runs, f"{arm}/report.json") for arm in ("ordinary", "fbt")}}
    if comparison.get("input_sha256") != {name: file_digest(path)["sha256"] for name, path in inputs.items()}:
        raise ValueError("Final comparison input report bytes changed")
    figures = {"learning-curves.pdf", "learning-curves.png", "online-comparison.pdf", "online-comparison.png"}
    if set(comparison.get("figure_sha256", {})) != figures:
        raise ValueError("Final comparison figure inventory differs")
    for name, digest in comparison["figure_sha256"].items():
        if file_digest(_safe_member(docs, name))["sha256"] != digest:
            raise ValueError("Final comparison figure bytes changed")
    helpers = {"scripts/olmo_o4_report.py", "cdrm/pretrained/lm_schedule.py"}
    if set(comparison.get("helper_source_sha256", {})) != helpers:
        raise ValueError("Final comparison helper inventory differs")
    for name, digest in comparison["helper_source_sha256"].items():
        if file_digest(_safe_member(project_root, name))["sha256"] != digest:
            raise ValueError("Final comparison helper source changed")
    return comparison


def retain(args):
    from scripts.olmo_o5b_common import ARMS
    bucket_name, key = parse_prefix(args.prefix)
    preflight, runs, data, output, docs = (Path(value).resolve() for value in
        (args.preflight, args.runs, args.data, args.output_dir, args.report_dir))
    if any(output.is_relative_to(path) for path in (preflight, runs, data, docs, ARTIFACTS.resolve())):
        raise ValueError("Retention output must be outside evidence input directories")
    corpus = load_lm_data(data)
    config, report = validate_learning(preflight, runs, args.phase, corpus)
    native = validate_artifact_metadata(ARTIFACTS)
    if config.get("checkpoint_sha256") != native["checkpoint"]["sha256"] or report.get("checkpoint") != native["checkpoint"]:
        raise ValueError("O5b checkpoint differs from selected native artifact")
    snapshots = verify_snapshots(ROOT)
    checkpoint = checkpoint_reference(json.loads(O1_RECEIPT.read_text()), native["checkpoint"])
    prior_manifest = Path(args.prior_data_manifest)
    data_objects = prepared_data_reference(json.loads(O4_RECEIPT.read_text()), prior_manifest, corpus)
    if args.phase == "final":
        # The reporter is the authoritative strict matching/completion check.
        validate_final_comparison(preflight, runs, docs)
    members = collect_evidence(preflight, runs, config, snapshots, phase=args.phase,
        data=data, report_dir=docs, o4_manifest=prior_manifest, arms=ARMS)
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    checkpoint = verify_checkpoint_reference(bucket, checkpoint)
    data_objects = [verify_reference(bucket, record) for record in data_objects]
    trained = verify_training_checkpoints(bucket, runs, ARMS, args.prefix) if args.phase == "final" else {}
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "evidence.tar.gz"
    restore = ("# O5b recovery-pilot evidence\n\nThis is a bounded source overlay, not a full repository. "
        "Verify evidence-members.json before restoring project/.\n\n"
        f"Reuse native checkpoint {checkpoint['uri']} generation {checkpoint['generation']}. "
        "No original/adapted model or optimizer bytes are in this archive. Per-arm checkpoint receipts "
        "identify separate immutable objects.\n\n"
        f"Restore prepared-data/ from {data_objects[0]['uri']} generation {data_objects[0]['generation']}; "
        f"SHA256 {data_objects[0]['sha256']}. O4's archive contains the exact prepared dataset and cards. "
        "Compare the prepared manifest against this archive's prepared-data/manifest.json. "
        "Do not substitute O4 model weights or its frozen training source configuration. "
        "Run GPU commands only through the project Docker launcher.\n")
    inventory = build_evidence_archive(archive, members, restore)
    validate_learning(preflight, runs, args.phase, corpus)
    manifest = {"schema": SCHEMA, "phase": args.phase, "members": inventory,
        "configuration_sha256": file_digest(preflight / "configuration.json")["sha256"],
        "data_manifest_sha256": corpus.manifest_sha256, "checkpoint_reference": checkpoint,
        "prepared_data_references": data_objects, "model_checkpoints_in_archive": False,
        "prepared_dataset_uploaded": False, "source_hashes": config["source_hashes"],
        "training_checkpoint_references": trained}
    manifest_path = output / "manifest.json"; write_json(manifest_path, manifest)
    objects = [upload_verified(bucket, f"{key}/{args.phase}/{path.name}", path, file_digest(path))
               for path in (archive, manifest_path)]
    receipt = {"schema": "olmo-o5b-evidence-receipt-v1", "status": "verified", "phase": args.phase,
               "objects": objects, "checkpoint_reference": checkpoint, "prepared_data_references": data_objects,
               "training_checkpoint_references": trained}
    receipt_path = output / "storage-receipt.json"; write_json(receipt_path, receipt)
    receipt["receipt_object"] = upload_verified(bucket, f"{key}/{args.phase}/{receipt_path.name}", receipt_path, file_digest(receipt_path))
    write_json(output / "upload-result.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "preflight", "runs", "output-dir", "report-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-data-manifest", type=Path, default=O4_MANIFEST)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--phase", choices=("initial", "final"), required=True)
    result = retain(parser.parse_args())
    print({"status": result["status"], "phase": result["phase"], "receipt": result["receipt_object"]["uri"]}, flush=True)


if __name__ == "__main__":
    main()
