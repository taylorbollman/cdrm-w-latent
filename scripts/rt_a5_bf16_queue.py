#!/usr/bin/env python3
"""Host supervisor: one authorized BF16 A5 repeat, report, then retain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


class Queue:
    def __init__(self, runtime):
        self.runtime = Path(runtime).resolve()
        self.relative = self.runtime.relative_to(ROOT)
        self.config = read(self.runtime / "execution-config.json")

    def status(self, phase, **values):
        packet = {"schema": "rt-a5-bf16-queue-v1", "phase": phase, "pid": os.getpid(),
                  "updated_at_utc": datetime.now(timezone.utc).isoformat(), **values}
        temporary = self.runtime / "status.tmp"
        temporary.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.runtime / "status.json")

    def stop(self, number, frame):
        (self.runtime / "STOP").touch(exist_ok=True)

    def verify(self):
        ready = read(self.runtime / "ready.json")
        if ready.get("passed") is not True:
            raise RuntimeError("BF16 repeat readiness has not passed")
        for name, expected in ready["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError("Frozen repeat source changed: " + name)
        validation = ROOT / self.config["validation"] / "report.json"
        if (sha(validation) != ready["validation_report_sha256"]
                or read(validation).get("status") != "passed"):
            raise RuntimeError("Bounded BF16 GPU preflight has not passed")

    def invoke(self, name, arguments, *, gpu=False, adc=False):
        self.verify()
        environment = os.environ.copy()
        environment["CDRM_DOCKER_GPUS"] = "all" if gpu else "none"
        command = ('test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && '
                   + ("nvidia-smi -L && " if gpu else "")
                   + ("env -u GOOGLE_APPLICATION_CREDENTIALS " if adc else "")
                   + shlex.join(arguments))
        with (self.runtime / (name + ".log")).open("x") as log:
            process = subprocess.Popen(["bash", "scripts/docker_shell.sh", "bash", "-lc", command],
                                       cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            self.status(name, child_pid=process.pid)
            code = process.wait()
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect its log")

    def run(self):
        self.verify()
        if (self.runtime / "STOP").exists():
            self.status("cancelled_before_training")
            return
        self.invoke("train", ["python", "-u", "-m", "scripts.rt_a5_bf16_train",
                              *self.config["training_cli"]], gpu=True)
        report = read(ROOT / self.config["training"] / "report.json")
        endpoint = report["completed_updates"]
        if (report["status"] not in ("complete", "stopped")
                or report["start_update"] != 0 or report["parent_checkpoint"] is not None
                or not 0 <= endpoint <= 80000
                or (report["status"] == "complete" and endpoint != 80000)):
            raise RuntimeError("BF16 repeat did not obey its fresh 80k endpoint")
        report_args = []
        if endpoint > 0:
            self.invoke("compare", ["python", "-m", "scripts.rt_a5_bf16_report",
                                    "--baseline-train", self.config["baseline_training"],
                                    "--train", self.config["training"],
                                    "--output", self.config["report"], "--wandb"])
            if read(ROOT / self.config["report"] / "report.json").get("status") != "complete":
                raise RuntimeError("BF16 paired report did not complete")
            report_args = ["--report", self.config["report"]]
        self.invoke("retain", ["python", "-m", "scripts.rt_a5_bf16_retain",
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
