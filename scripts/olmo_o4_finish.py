#!/usr/bin/env python3
"""Finish local plots and verified evidence retention after the O4 queue ends."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--retention-output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    status_path = args.runs / "finish-status.json"
    record = {"schema": "olmo-o4-finish-v1", "status": "waiting_for_queue",
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(status_path, record)
    try:
        while True:
            queue = json.loads((args.runs / "queue.json").read_text())
            if queue["status"] == "completed":
                break
            if queue["status"] != "running":
                record.update(status="queue_stopped", queue_status=queue["status"])
                return
            time.sleep(20)
        record["status"] = "building_report"; write_json(status_path, record)
        subprocess.run([sys.executable, "scripts/olmo_o4_report.py", "--preflight", str(args.preflight),
                        "--runs", str(args.runs), "--output-dir", str(args.report_dir)], cwd=ROOT, check=True)
        record["status"] = "retaining_evidence"; write_json(status_path, record)
        subprocess.run([sys.executable, "scripts/olmo_o4_retain.py", "--data", str(args.data),
                        "--preflight", str(args.preflight), "--runs", str(args.runs),
                        "--output-dir", str(args.retention_output), "--prefix", args.prefix,
                        "--phase", "final"], cwd=ROOT, check=True)
        shutil.copyfile(args.retention_output / "upload-result.json", args.report_dir / "final-storage-receipt.json")
        record.update(status="completed", report=str(args.report_dir / "results.md"),
                      receipt=str(args.report_dir / "final-storage-receipt.json"))
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, record)


if __name__ == "__main__":
    main()
