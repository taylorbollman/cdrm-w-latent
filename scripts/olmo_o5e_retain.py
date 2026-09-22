#!/usr/bin/env python3
"""Retain explicit ordinary-control evidence and verify reused parent artifacts."""
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
from scripts.olmo_o5c_common import endpoint_metadata as source_metadata
from scripts.olmo_o5d_common import endpoint_metadata as mixed_metadata
from scripts.olmo_o5e_common import PREFIX_ROOT, source_hashes, retain_file
from scripts.olmo_o5e_train import validate_configuration, validate_preflight

MIXED_MANIFEST = ROOT/".runtime/olmo1b-step60000/o5c-data-02/prepared/manifest.json"
BASE_MANIFEST = ROOT/".runtime/olmo1b-step60000/o4-data-01/prepared/manifest.json"
PARENT_RECEIPTS = (
    "olmo1b-o4/initial-storage-receipt.json", "olmo1b-o5b/final-storage-receipt.json",
    "olmo1b-o5c/final-storage-receipt.json", "olmo1b-o5d/storage-receipt.json",
)
EXTRA_SOURCES = (
    "scripts/olmo_o5e_common.py", "scripts/olmo_o5e_preflight.py", "scripts/olmo_o5e_train.py",
    "scripts/olmo_o5e_report.py", "scripts/olmo_o5e_retain.py",
    "tests/test_olmo_o5e_common.py", "tests/test_olmo_o5e_train.py", "tests/test_olmo_o5e_report.py",
    "tests/test_olmo_o5e_retain.py",
    "scripts/olmo_o4_report.py", "scripts/olmo_o5c_report.py", "scripts/olmo_o5d_report.py",
    "scripts/openelm_retain.py", "AGENTS.md", "docs/fbt-rt-nextlat-handoff.md",
    "docs/fbt-rt-nextlat-research-plan-v3.md",
)
FORBIDDEN = {"hf-cache", "wandb", "__pycache__", ".git", ".docker-home"}


def checked_prefix(value):
    prefix = value.rstrip("/")+"/"
    if (not prefix.startswith(PREFIX_ROOT)
            or re.fullmatch(r"\d{8}T\d{6}Z/", prefix[len(PREFIX_ROOT):]) is None):
        raise ValueError("Use the O5e lineage followed by one YYYYMMDDTHHMMSSZ timestamp")
    return prefix


def verify_reference(client, record):
    uri = record.get("uri", "")
    if not uri.startswith("gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/"):
        raise ValueError("Referenced object is outside the retained project lineage")
    bucket, key = uri[5:].split("/", 1)
    blob = client.bucket(bucket).get_blob(key)
    if (blob is None or str(blob.generation) != record["generation"]
            or blob.size != record["size_bytes"] or blob.md5_hash != record["md5_base64"]
            or (blob.metadata or {}).get("sha256") != record["sha256"]):
        raise ValueError("Retained object generation or checksums differ")
    return {**record, "reused_without_upload": True}


def comparison_inputs(preflight, run_dir):
    from scripts import olmo_o5e_report as reporter
    return {"preflight": preflight/"report.json", "configuration": preflight/"configuration.json",
        "ordinary": run_dir/"report.json", "online_reference": reporter.O5D_REPORT,
        "fusion_preflight": reporter.O5C_PREFLIGHT/"report.json",
        "fusion_config": reporter.O5C_PREFLIGHT/"configuration.json",
        "fusion_code": reporter.O5C_RUNS/"code/report.json", "fusion_mixed": reporter.O5C_RUNS/"mixed/report.json"}


def validate_final(preflight, run_dir, report_dir):
    """Rebuild the paired comparison from authoritative completed inputs."""
    from scripts import olmo_o5e_report as reporter
    paths = comparison_inputs(preflight, run_dir)
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    fusion = reporter.fusion_report.build_comparison(values["fusion_preflight"], values["fusion_config"],
        {"code": values["fusion_code"], "mixed": values["fusion_mixed"]})
    comparison = reporter.build_comparison(values["preflight"], values["configuration"], values["ordinary"],
                                           fusion, values["online_reference"])
    rendered = json.loads((report_dir/"final-comparison.json").read_text())
    if any(rendered.get(name) != value for name, value in comparison.items()):
        raise ValueError("Rendered final comparison differs from validated completed runs")
    if (rendered.get("input_sha256") != {name: sha256_file(path) for name, path in paths.items()}
            or rendered.get("report_source_sha256") != sha256_file(reporter.__file__)):
        raise ValueError("Rendered comparison input/source hashes differ")
    for name, digest in rendered["helper_source_sha256"].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError("Comparison helper source changed")
    if set(rendered.get("figure_sha256", {})) != set(reporter.FIGURES):
        raise ValueError("Final comparison figure inventory differs")
    for name, digest in rendered["figure_sha256"].items():
        if sha256_file(report_dir/name) != digest:
            raise ValueError("Final comparison figure changed")
    if (report_dir/"results.md").read_text() != reporter.markdown(rendered):
        raise ValueError("Final Markdown differs from validated comparison")
    return values["ordinary"], paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preflight", "run-dir", "report-dir", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--phase", choices=("initial", "final"), required=True)
    parser.add_argument("--storage-prefix", required=True)
    args = parser.parse_args(argv)
    prefix = checked_prefix(args.storage_prefix)
    config = json.loads((args.preflight/"configuration.json").read_text())
    preflight = json.loads((args.preflight/"report.json").read_text())
    validate_configuration(config)
    validate_preflight(config, args.preflight/"configuration.json", config["runtime"])
    if config["source_hashes"] != source_hashes() or preflight.get("weights_unchanged") is not True:
        raise ValueError("Require frozen passing ordinary-control preflight")
    if (sha256_file(MIXED_MANIFEST) != config["data_manifest_sha256"]
            or sha256_file(BASE_MANIFEST) != config["base_data_manifest_sha256"]):
        raise ValueError("Shared data manifest changed")
    authorities = {"source": source_metadata(), "mixed": mixed_metadata()}
    if (authorities["source"]["checkpoint"]["sha256"] != config["source_checkpoint_sha256"]
            or authorities["mixed"]["report_sha256"] != config["reference_o5c_mixed_report_sha256"]
            or authorities["mixed"]["checkpoint"]["sha256"] != config["reference_o5c_mixed_checkpoint_sha256"]):
        raise ValueError("Shared endpoint authority differs")
    members = {}
    def add(path, name):
        path = Path(path)
        archive = PurePosixPath(name)
        if (archive.is_absolute() or not archive.parts or ".." in archive.parts
                or any(p in FORBIDDEN or p.startswith(".env") for p in (*archive.parts, *path.parts))
                or path.suffix == ".pt" or not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT.resolve())):
            raise ValueError(f"Explicit regular project evidence member required: {name}")
        for parent in path.absolute().parents:
            if parent == ROOT:
                break
            if parent.is_symlink():
                raise ValueError(f"Symlinked evidence ancestor: {name}")
        if name in members and members[name] != path:
            raise ValueError("Duplicate archive member with different source")
        members[name] = path
    for name, digest in config["source_hashes"].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError(f"Frozen source differs: {name}")
        add(ROOT/name, "project/"+name)
    for name in EXTRA_SOURCES:
        add(ROOT/name, "project/"+name)
    usage = ROOT/"docs/olmo1b-o5e-usage.md"
    if usage.exists():
        add(usage, "project/docs/olmo1b-o5e-usage.md")
    for name in ("configuration.json", "report.json"):
        add(args.preflight/name, "preflight/"+name)
    add(MIXED_MANIFEST, "parents/o5c-mixed-data-manifest.json")
    add(BASE_MANIFEST, "parents/o4-evaluation-data-manifest.json")
    for name, authority in authorities.items():
        add(authority["report_path"], "parents/"+name+"-endpoint-report.json")
    references = {}
    for name in PARENT_RECEIPTS:
        path = ROOT/"docs/reports"/name
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "verified":
            raise ValueError("Parent artifact receipt is not verified")
        references[name] = receipt["objects"]
        add(path, "parents/"+name.replace("/", "-"))
    ordinary = None
    if args.phase == "final":
        ordinary, paths = validate_final(args.preflight, args.run_dir, args.report_dir)
        for name, path in paths.items():
            add(path, "comparison-inputs/"+name+".json")
        add(args.run_dir/"report.json", "run/report.json")
        add(args.run_dir/"events.jsonl", "run/events.jsonl")
        for record in ordinary["checkpoints"]:
            path = Path(record["path"]).with_suffix(".receipt.json")
            if not path.is_absolute(): path = ROOT/path
            add(path, "run/"+path.name)
    for path in sorted(args.report_dir.iterdir()):
        if (path.suffix in (".md", ".json", ".txt", ".pdf", ".png")
                and path.name != args.phase+"-storage-receipt.json"):
            add(path, "report/"+path.name)
    from google.cloud import storage
    client = storage.Client()
    verified_parents = {name: [verify_reference(client, record) for record in records]
                        for name, records in references.items()}
    verified_checkpoints = {name: verify_reference(client, authority["checkpoint"]["storage"])
                            for name, authority in authorities.items()}
    retained_control = []
    if ordinary is not None:
        for record in ordinary["checkpoints"]:
            storage_record = record.get("storage", {})
            if (storage_record.get("sha256") != record["sha256"]
                    or storage_record.get("size_bytes") != record["size_bytes"]
                    or storage_record.get("uri") != prefix+"ordinary/"+Path(record["path"]).name):
                raise ValueError("Control checkpoint storage identity differs")
            retained_control.append(verify_reference(client, storage_record))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inventory = build_evidence_archive(args.output_dir/"evidence.tar.gz",
        [(path, name) for name, path in sorted(members.items())],
        "Restore project/ into the repository. Reuse the unchanged O5b source and O5c mixed checkpoints and "
        "the O4/O5c prepared-data archives identified in parent receipts and manifest.json. "
        "All referenced object generations, sizes, MD5 and SHA metadata were verified. "
        "Control checkpoints are separately retained complete training boundaries; resume only the latest "
        "authorized incomplete boundary using matching config/sources/runtime. Final evidence describes a completed run. "
        "No model or prepared-data payloads are duplicated in this archive. Verify all member hashes and read usage/protocol.\n")
    manifest = {"schema": "olmo-o5e-evidence-v1", "phase": args.phase,
        "configuration_sha256": sha256_file(args.preflight/"configuration.json"),
        "source_checkpoints": verified_checkpoints, "parent_evidence": verified_parents,
        "control_checkpoints": retained_control, "members": inventory}
    write_json(args.output_dir/"manifest.json", manifest)
    destination = prefix+args.phase+"/"
    objects = [retain_file(args.output_dir/name, destination+name) for name in ("evidence.tar.gz", "manifest.json")]
    receipt = {"schema": "olmo-o5e-evidence-receipt-v1", "phase": args.phase, "status": "verified",
        "source_checkpoints": verified_checkpoints, "control_checkpoints": retained_control, "objects": objects}
    write_json(args.output_dir/"storage-receipt.json", receipt)
    retain_file(args.output_dir/"storage-receipt.json", destination+"storage-receipt.json")
    write_json(args.report_dir/(args.phase+"-storage-receipt.json"), receipt)
    print({"phase": args.phase, "status": "verified", "objects": [item["uri"] for item in objects]}, flush=True)


if __name__ == "__main__":
    main()
