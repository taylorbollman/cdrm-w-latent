#!/usr/bin/env python3
"""Run the frozen O5c pair sequentially, resuming only retained boundaries."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.olmo_o5c_common import source_hashes, plain_metadata
from scripts.olmo_validation import require_container_gpu


def validate_completed_arm(report, config, arm, prefix, runtime):
    """Reject partial or differently sourced arms before the queue skips work."""
    try:
        prefix = prefix.rstrip("/")
        expected_source = {"checkpoint_sha256": config["source_checkpoint_sha256"],
            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "code": config["source_hashes"], "data_manifest_sha256": config["data_manifest_sha256"],
            "runtime": runtime}
        plan = config["schedule"]["arms"][arm]
        total = config["schedule"]["total_updates"]
        physical = config["physical_batch_size"]
        rows = plan["batch_row_prefix"]
        counters = {"optimizer_updates": total, "input_tokens": plan["batch_token_prefix"][total],
            "ce_positions": plan["batch_ce_prefix"][total], "documents": rows[total],
            "microbatches": sum((rows[u+1]-rows[u]+physical-1)//physical for u in range(total)),
            "latent_pairs": 0, "kl_triples": 0}
        domains = {domain: {key: values[total] for key, values in fields.items()}
                   for domain, fields in plan["domain_prefixes"].items()}
        if (report["schema"] != "olmo-o5c-arm-v1" or report["status"] != "completed"
                or report["arm"] != arm or not report["finished_utc"]
                or report["configuration"] != config or report["source_fingerprint"] != expected_source
                or report["storage_prefix"] != prefix or plan["total_updates"] != total
                or report["counters"] != counters or report["data_cursor"] != total
                or any(type(value) is not int for value in report["counters"].values())
                or report["domain_counts"] != domains or report["trainable_layout"] != config["trainable_layout"]
                or report["frozen_state_initial"] != config["frozen_state_initial"]
                or report["frozen_state_final"] != config["frozen_state_initial"]):
            raise ValueError("Completed arm lineage, ownership or exposure differs")
        records = report["checkpoints"]
        final = [record for record in records if record["optimizer_updates"] == total]
        if len(final) != 1 or final[0] != records[-1]:
            raise ValueError("Require exactly one final checkpoint as latest boundary")
        record = final[0]; storage = record["storage"]
        name = f"update-{total:06d}.pt"
        if (Path(record["path"]).name != name
                or not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
                or type(record["size_bytes"]) is not int or record["size_bytes"] <= 0
                or record["input_tokens"] != counters["input_tokens"]
                or record["ce_positions"] != counters["ce_positions"]
                or storage["uri"] != f"{prefix}/{arm}/{name}"
                or storage["sha256"] != record["sha256"] or storage["size_bytes"] != record["size_bytes"]
                or not storage["generation"] or not storage["md5_base64"]
                or storage["verification"] != "GCS generation, size, server MD5 and SHA256 metadata verified"):
            raise ValueError("Completed arm lacks matching verified final checkpoint receipt")
    except (KeyError, IndexError, TypeError, ZeroDivisionError) as error:
        raise ValueError("Completed arm has malformed completion evidence") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "base-data", "configuration", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--storage-prefix", required=True)
    args = parser.parse_args()
    args.storage_prefix = args.storage_prefix.rstrip("/")
    runtime = plain_metadata(require_container_gpu())
    config = json.loads(args.configuration.read_text())
    if config.get("schema") != "olmo-o5c-pilot-config-v1" or config["source_hashes"] != source_hashes():
        raise ValueError("Queue requires frozen O5c configuration/sources")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/"queue.json"
    digest = sha256_file(args.configuration)
    if path.exists():
        queue = json.loads(path.read_text())
        if (queue["schema"] != "olmo-o5c-queue-v1" or queue["order"] != ["code", "mixed"]
                or queue["configuration_sha256"] != digest or queue["runtime"] != runtime
                or queue["storage_prefix"] != args.storage_prefix):
            raise ValueError("Queue lineage/runtime changed")
    else:
        queue = {"schema": "olmo-o5c-queue-v1", "status": "running", "configuration_sha256": digest,
            "runtime": runtime, "storage_prefix": args.storage_prefix, "order": ["code", "mixed"],
            "started_utc": datetime.now(timezone.utc).isoformat(), "arms": {}}
    for arm in queue["order"]:
        directory = args.output_dir/arm; report_path = directory/"report.json"
        previous = json.loads(report_path.read_text()) if report_path.exists() else None
        if previous and previous.get("status") == "completed":
            validate_completed_arm(previous, config, arm, args.storage_prefix, runtime)
            queue["arms"][arm] = {"status": "completed", "report": str(report_path), "wandb": previous["wandb"]["run_url"]}
            continue
        if (args.output_dir/"STOP").exists():
            queue["status"] = "paused"; write_json(path, queue); return
        command = [sys.executable, "scripts/olmo_o5c_train.py", "--data", str(args.data),
            "--base-data", str(args.base_data), "--configuration", str(args.configuration),
            "--output-dir", str(directory), "--storage-prefix", args.storage_prefix, "--arm", arm]
        if previous:
            if previous["status"] == "health_gate_failed": raise RuntimeError("Diagnose health gate before restart")
            records = previous.get("checkpoints", [])
            if not records or not Path(records[-1]["path"]).is_file():
                raise RuntimeError("No recoverable local boundary; do not silently restart")
            command += ["--resume", records[-1]["path"]]
        queue.update(status="running", active_arm=arm); write_json(path, queue)
        with (args.output_dir/(arm+".log")).open("a") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        final = json.loads(report_path.read_text()) if report_path.exists() else {}
        queue["arms"][arm] = {"status": final.get("status", "failed_before_report"), "exit_code": result.returncode,
            "report": str(report_path), "wandb": final.get("wandb", {}).get("run_url")}
        if result.returncode or final.get("status") != "completed":
            queue.update(status="stopped", finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(path, queue); raise SystemExit(result.returncode or 1)
        try:
            validate_completed_arm(final, config, arm, args.storage_prefix, runtime)
        except ValueError:
            queue.update(status="stopped", finished_utc=datetime.now(timezone.utc).isoformat())
            queue["arms"][arm]["status"] = "completion_validation_failed"
            write_json(path, queue)
            raise
        write_json(path, queue)
    queue.update(status="completed", active_arm=None, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, queue)


if __name__ == "__main__": main()
