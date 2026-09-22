#!/usr/bin/env python3
"""Build strict O5b results and retain evidence after both arms complete."""
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


def finish(args, *, sleep=time.sleep, run=subprocess.run, project_root=ROOT):
    status_path = args.runs / "finish-status.json"
    record = {"schema": "olmo-o5b-finish-v1", "status": "waiting_for_queue",
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(status_path, record)
    try:
        while True:
            try:
                queue = json.loads((args.runs / "queue.json").read_text())
            except FileNotFoundError:
                sleep(20)
                continue
            if queue.get("status") == "completed":
                break
            if queue.get("status") != "running":
                record.update(status="queue_stopped", queue_status=queue.get("status"))
                return record
            sleep(20)
        record["status"] = "building_report"; write_json(status_path, record)
        if (args.report_dir / "final-comparison.json").exists() and (args.retention_output / "evidence.tar.gz").exists():
            # Preserve the original report timestamp/plot bytes when retrying a
            # failed immutable upload. Stale evidence still fails strict checks.
            from scripts.olmo_o5b_retain import validate_final_comparison
            validate_final_comparison(args.preflight, args.runs, args.report_dir, project_root=project_root)
        else:
            run([sys.executable, "scripts/olmo_o5b_report.py", "--preflight", str(args.preflight),
                 "--runs", str(args.runs), "--output-dir", str(args.report_dir)], cwd=project_root, check=True)
        record["status"] = "retaining_evidence"; write_json(status_path, record)
        command = [sys.executable, "scripts/olmo_o5b_retain.py", "--data", str(args.data),
            "--preflight", str(args.preflight), "--runs", str(args.runs), "--output-dir", str(args.retention_output),
            "--report-dir", str(args.report_dir), "--prefix", args.prefix, "--phase", "final"]
        if args.prior_data_manifest is not None:
            command.extend(["--prior-data-manifest", str(args.prior_data_manifest)])
        run(command, cwd=project_root, check=True)
        receipt = json.loads((args.retention_output / "upload-result.json").read_text())
        if receipt.get("schema") != "olmo-o5b-evidence-receipt-v1" or receipt.get("status") != "verified" or receipt.get("phase") != "final" or not receipt.get("receipt_object"):
            raise ValueError("Final retention did not produce a verified O5b receipt")
        shutil.copyfile(args.retention_output / "upload-result.json", args.report_dir / "final-storage-receipt.json")
        record.update(status="completed", report=str(args.report_dir / "results.md"),
                      receipt=str(args.report_dir / "final-storage-receipt.json"))
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "preflight", "runs", "retention-output", "report-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-data-manifest", type=Path)
    parser.add_argument("--prefix", required=True)
    result = finish(parser.parse_args())
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
