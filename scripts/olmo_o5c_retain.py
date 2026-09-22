#!/usr/bin/env python3
"""Retain explicit O5c data/source/evidence files with parent artifact receipts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.openelm_retain import build_evidence_archive
from scripts.olmo_o5c_common import source_hashes, retain_file, endpoint_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "preflight", "runs", "report-dir", "output-dir"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--phase", choices=("initial", "final"), required=True)
    parser.add_argument("--storage-prefix", required=True)
    args = parser.parse_args()
    config = json.loads((args.preflight/"configuration.json").read_text())
    preflight = json.loads((args.preflight/"report.json").read_text())
    if config["source_hashes"] != source_hashes() or preflight["status"] != "passed":
        raise ValueError("Retain only frozen passing configuration")
    if sha256_file(args.data/"manifest.json") != config["data_manifest_sha256"]:
        raise ValueError("Prepared data manifest changed")
    if args.phase == "final":
        queue = json.loads((args.runs/"queue.json").read_text())
        if queue["status"] != "completed": raise ValueError("Final evidence requires completed queue")
        from scripts.olmo_o5c_report import validate_runs
        validate_runs(args.preflight,args.runs)
    members = {}
    def add(path, name):
        path = Path(path)
        if not path.is_file() or path.is_symlink() or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("Explicit regular safe archive member required")
        if name in members: raise ValueError("Duplicate archive member")
        members[name] = path
    for name in config["source_hashes"]: add(ROOT/name,"project/"+name)
    extra = [*ROOT.glob("scripts/olmo_o5c_*.py"), *ROOT.glob("tests/test_olmo_o5c_*.py"),
             ROOT/"scripts/olmo_o4_report.py", ROOT/"scripts/olmo_o5b_report.py", ROOT/"scripts/openelm_retain.py"]
    extra += [ROOT/"AGENTS.md", ROOT/"docs/fbt-rt-nextlat-handoff.md",
              ROOT/"docs/fbt-rt-nextlat-research-plan-v3.md"]
    usage = ROOT/"docs/olmo1b-o5c-usage.md"
    if usage.exists(): extra.append(usage)
    for path in extra:
        name = "project/"+path.relative_to(ROOT).as_posix()
        if name not in members: add(path,name)
    manifest = json.loads((args.data/"manifest.json").read_text())
    add(args.data/"manifest.json", "prepared/manifest.json")
    for name, metadata in manifest["files"].items():
        path = args.data/name
        if sha256_file(path) != metadata["sha256"]: raise ValueError("Prepared file changed")
        add(path,"prepared/"+name)
    for name in ("configuration.json", "report.json"): add(args.preflight/name,"preflight/"+name)
    for path in sorted(args.report_dir.iterdir()):
        if path.is_file() and path.suffix in (".md", ".json", ".txt", ".pdf", ".png"):
            if path.name != args.phase+"-storage-receipt.json": add(path,"report/"+path.name)
    endpoint = endpoint_metadata()
    add(endpoint["report_path"],"parents/o5b-fbt-report.json")
    for name in ("olmo1b-o4/initial-storage-receipt.json", "olmo1b-o5b/final-storage-receipt.json"):
        add(ROOT/"docs/reports"/name,"parents/"+name.replace("/","-"))
    # Verify the starting checkpoint is still retained with the exact pinned generation.
    from google.cloud import storage
    record = endpoint["checkpoint"]["storage"]
    bucket, key = record["uri"][5:].split("/",1)
    blob = storage.Client().bucket(bucket).get_blob(key)
    if (blob is None or str(blob.generation) != record["generation"] or blob.size != record["size_bytes"]
            or blob.md5_hash != record["md5_base64"] or (blob.metadata or {}).get("sha256") != record["sha256"]):
        raise ValueError("Retained source endpoint differs")
    if args.phase == "final":
        add(args.runs/"queue.json","runs/queue.json")
        for arm in ("code", "mixed"):
            for name in ("report.json", "events.jsonl"): add(args.runs/arm/name,"runs/"+arm+"/"+name)
            for path in sorted((args.runs/arm).glob("update-*.receipt.json")):
                add(path,"runs/"+arm+"/"+path.name)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    archive = args.output_dir/"evidence.tar.gz"
    inventory = build_evidence_archive(archive, [(path,name) for name,path in sorted(members.items())],
        "Restore project/ into the repository; prepared/ is the O5c data. Reuse the immutable O4 prepared-data archive and O5b endpoint named in parents/ receipts. Verify all member hashes before running. See the retained usage and frozen protocol.\n")
    record = {"schema":"olmo-o5c-evidence-v1","phase":args.phase,
        "configuration_sha256":sha256_file(args.preflight/"configuration.json"),
        "source_checkpoint":endpoint["checkpoint"]["storage"],"members":inventory}
    write_json(args.output_dir/"manifest.json",record)
    prefix = args.storage_prefix.rstrip("/")+"/"+args.phase
    objects = [retain_file(args.output_dir/name,prefix+"/"+name) for name in ("evidence.tar.gz","manifest.json")]
    receipt = {"schema":"olmo-o5c-evidence-receipt-v1","phase":args.phase,"status":"verified","objects":objects}
    write_json(args.output_dir/"storage-receipt.json",receipt)
    retain_file(args.output_dir/"storage-receipt.json",prefix+"/storage-receipt.json")
    write_json(args.report_dir/(args.phase+"-storage-receipt.json"),receipt)
    print({"phase":args.phase,"status":"verified","objects":[o["uri"] for o in objects]},flush=True)


if __name__ == "__main__": main()
