#!/usr/bin/env python3
"""Run the two frozen O5b arms sequentially; halt the queue on any failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.olmo_o5b_common import ARMS, source_hashes, validate_completed_arm
from scripts.olmo_validation import require_container_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-prefix", required=True)
    args = parser.parse_args()
    runtime = require_container_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = args.output_dir / "queue.json"
    config_sha = sha256_file(args.configuration)
    config = json.loads(args.configuration.read_text())
    if config.get("schema") != "olmo-o5b-pilot-config-v1" or config["source_hashes"] != source_hashes():
        raise ValueError("Queue requires unchanged frozen O5b configuration/sources")
    if queue_path.exists():
        queue = json.loads(queue_path.read_text())
        if queue["configuration_sha256"] != config_sha or queue["storage_prefix"] != args.storage_prefix:
            raise ValueError("Queue configuration changed")
    else:
        queue = {"schema": "olmo-o5b-queue-v1", "status": "running", "configuration_sha256": config_sha,
                 "runtime": runtime, "order": list(ARMS), "storage_prefix": args.storage_prefix,
                 "started_utc": datetime.now(timezone.utc).isoformat(), "arms": {}}
    for arm in queue["order"]:
        directory = args.output_dir / arm
        report_path = directory / "report.json"
        previous = json.loads(report_path.read_text()) if report_path.exists() else None
        if previous and previous["status"] == "completed":
            validate_completed_arm(previous,config,arm,args.storage_prefix,runtime)
            queue["arms"][arm] = {"status": "completed", "report": str(report_path),
                                   "wandb":previous.get("wandb",{}).get("run_url")}
            continue
        if (args.output_dir / "STOP").exists():
            queue["status"] = "paused"; write_json(queue_path, queue)
            return
        command = [sys.executable, "scripts/olmo_o5b_train.py", "--artifacts", str(args.artifacts),
                   "--data", str(args.data), "--configuration", str(args.configuration), "--arm", arm,
                   "--output-dir", str(directory), "--storage-prefix", args.storage_prefix]
        if previous:
            if previous["status"] == "health_gate_failed":
                raise RuntimeError("A health gate requires diagnosis before queue resumption")
            checkpoints = previous.get("checkpoints", [])
            if not checkpoints or not Path(checkpoints[-1]["path"]).is_file():
                raise RuntimeError("Interrupted arm has no local recoverable checkpoint; do not restart silently")
            command += ["--resume", checkpoints[-1]["path"]]
        queue.update(status="running", active_arm=arm)
        write_json(queue_path, queue)
        print({"starting_arm": arm, "log": str(args.output_dir / (arm + ".log"))}, flush=True)
        with (args.output_dir / (arm + ".log")).open("a") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        final = json.loads(report_path.read_text()) if report_path.exists() else {}
        queue["arms"][arm] = {"status": final.get("status", "failed_before_report"),
                               "exit_code": result.returncode, "report": str(report_path),
                               "wandb": final.get("wandb", {}).get("run_url")}
        if result.returncode or final.get("status") != "completed":
            queue.update(status="stopped", finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(queue_path, queue)
            raise SystemExit(result.returncode or 1)
        write_json(queue_path, queue)
    queue.update(status="completed", active_arm=None, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(queue_path, queue)
    print({"status": "completed", "queue": str(queue_path)}, flush=True)


if __name__ == "__main__":
    main()
