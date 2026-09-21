#!/usr/bin/env python3
"""Host supervisor for one three-layer 5k mixed-task pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal

from scripts.rt_a5_bf16_queue import Queue as OriginalQueue, ROOT, read, sha


class Queue(OriginalQueue):
    def status(self, phase, **values):
        packet = {"schema": "rt-nextlat-mixed-depth-queue-v1", "phase": phase,
                  "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "endpoint": 5000, "batch_per_task": 2560, **values}
        temporary = self.runtime / "status.tmp"
        temporary.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.runtime / "status.json")

    def verify(self):
        ready = read(self.runtime / "ready.json")
        if ready.get("passed") is not True:
            raise RuntimeError("Depth pilot readiness has not passed")
        for name, expected in ready["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError("Frozen pilot source changed: " + name)
        profile = ROOT / self.config["profile"] / "report.json"
        evidence = read(profile)
        if (sha(profile) != ready["profile_sha256"] or evidence["status"] != "passed"
                or evidence["batch_per_task"] != 2560 or evidence["microbatch"] != self.config["microbatch"]):
            raise RuntimeError("Actual-batch GPU memory check has not passed")

    def run(self):
        self.verify()
        if self.config["endpoint"] != 5000:
            raise RuntimeError("Only the authorized 5,000-update pilot is scheduled")
        if (self.runtime / "STOP").exists():
            self.status("cancelled_before_training")
            return
        self.invoke("train", ["python", "-u", "-m", "scripts.rt_nextlat_a5_fuzzy_depth_train",
                              *self.config["training_cli"]], gpu=True)
        report = read(ROOT / self.config["training"] / "report.json")
        endpoint = report["completed_updates"]
        if (report["status"] not in ("complete", "stopped")
                or report["start_update"] != 0 or report["parent_checkpoint"] is not None
                or report["requested_endpoint"] != 5000 or not 0 <= endpoint <= 5000
                or (report["status"] == "complete" and endpoint != 5000)
                or report["contract"]["batch_per_task"] != 2560
                or report["contract"]["microbatch"] != self.config["microbatch"]
                or report["contract"]["model_config"]["backbone"]["n_layers"] != 3):
            raise RuntimeError("Fresh mixed run did not obey the authorized 5k/B2560 contract")
        report_args = []
        if endpoint > 0:
            self.invoke("compare", ["python", "-m", "scripts.rt_nextlat_a5_fuzzy_depth_report",
                                    "--baseline-train", self.config["baseline_training"],
                                    "--train", self.config["training"],
                                    "--output", self.config["report"], "--wandb"])
            if read(ROOT / self.config["report"] / "report.json").get("status") != "complete":
                raise RuntimeError("Depth comparison did not complete")
            report_args = ["--report", self.config["report"]]
        self.invoke("retain", ["python", "-m", "scripts.rt_nextlat_fuzzy_retain",
                               "--runtime", str(self.relative), *report_args,
                               "--prefix", self.config["retention_prefix"], "--label", "final"], adc=True)
        receipt = read(self.runtime / "retention/final-receipt.json")
        record = next(x for x in report["checkpoints"] if x["completed_updates"] == endpoint)
        name = f"train/checkpoints/step-{endpoint:06d}.pt"
        member = next(x for x in receipt["members"] if x["path"] == "runtime/" + name)
        if (receipt["status"] != "verified" or member["sha256"] != record["sha256"]
                or sha(self.runtime / name) != record["sha256"]):
            raise RuntimeError("Retained endpoint does not match completed training")
        self.status("complete" if report["status"] == "complete" else "stopped_and_retained",
                    completed_updates=endpoint, retained_uri=receipt["uri"],
                    checkpoint_sha256=record["sha256"], no_following_experiments=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    queue = Queue(args.runtime)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, queue.stop)
    try:
        queue.run()
    except BaseException as error:
        queue.status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
