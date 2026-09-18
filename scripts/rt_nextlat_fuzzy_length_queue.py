#!/usr/bin/env python3
"""Run the authorized four-model length evaluation after training releases GPU.

Host-side standard-library orchestration only. The existing training queue,
sources, contracts and its stop controls are never changed by this supervisor.
"""
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
import time


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, packet):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class LengthQueue:
    def __init__(self, runtime):
        self.runtime = Path(runtime).resolve()
        self.relative = self.runtime.relative_to(ROOT)
        self.config = read(self.runtime / "execution-config.json")

    def status(self, phase, **values):
        write(self.runtime / "status.json", {
            "schema": "rt-nextlat-fuzzy-length-queue-v1", "phase": phase,
            "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Evaluation only: all four 15k checkpoints, native Fuzzy lengths 512 and 1024",
            **values,
        })

    def request_stop(self, number, frame):
        (self.runtime / "STOP").touch(exist_ok=True)

    def frozen_sources(self):
        ready = read(self.runtime / "ready.json")
        if ready.get("passed") is not True:
            raise RuntimeError("Evaluation queue readiness has not passed")
        for relative, expected in ready["source_sha256"].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError("Frozen evaluation source changed: " + relative)
        if sha(ROOT / self.config["data"] / "manifest.json") != ready["data_manifest_sha256"]:
            raise RuntimeError("Prepared shared evaluation data identity changed")

    def wait_for_training(self):
        source = ROOT / self.config["training_queue"]
        while True:
            if (self.runtime / "STOP").exists():
                self.status("cancelled_before_evaluation")
                return False
            packet = read(source / "status.json")
            phase = packet["phase"]
            if phase == "complete":
                finished = packet["completed_variants"]
                if ([item["variant"] for item in finished] != ["input", "value", "head"]
                        or any(item["endpoint"] != 15000 or item["status"] != "complete" for item in finished)):
                    raise RuntimeError("Training queue did not complete all three matched 15k arms")
                return True
            if phase in ("failed", "stopped_and_retained", "stopped_before_next_variant",
                         "cancelled_before_variants", "stopped_after_preflight"):
                self.status("blocked_training_incomplete", training_phase=phase,
                            detail="All four matched 15k endpoints are required; adjust explicitly if training scope changes")
                return False
            try:
                os.kill(packet["pid"], 0)
            except ProcessLookupError as error:
                raise RuntimeError("Training supervisor exited without completing all arms") from error
            self.status("waiting_for_training_queue", training_phase=phase,
                        current_training_variant=packet.get("current_variant"))
            time.sleep(30)

    def retained_endpoints(self):
        identities = {}
        for arm in self.config["arms"]:
            report = read(ROOT / arm["training"] / "report.json")
            receipt = read(ROOT / arm["runtime"] / "retention/final-receipt.json")
            if report["status"] != "complete" or report["completed_updates"] != 15000:
                raise RuntimeError("Expected a completed 15k checkpoint for " + arm["label"])
            checkpoint = next(p for p in report["checkpoints"] if p["completed_updates"] == 15000)
            member = next(p for p in receipt["members"] if p["path"] == arm["archive_checkpoint"])
            local = ROOT / arm["training"] / "checkpoints/step-015000.pt"
            if (receipt["status"] != "verified" or checkpoint["sha256"] != member["sha256"]
                    or sha(local) != checkpoint["sha256"]):
                raise RuntimeError("Local/retained checkpoint identity differs for " + arm["label"])
            identities[arm["label"]] = {"checkpoint_sha256": checkpoint["sha256"],
                                       "retained_uri": receipt["uri"]}
        write(self.runtime / "retained-endpoints.json", identities)

    def invoke(self, label, arguments, *, gpu=False, adc=False):
        environment = os.environ.copy()
        environment["CDRM_DOCKER_GPUS"] = "all" if gpu else "none"
        command = ('test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && '
                   + ("nvidia-smi -L && " if gpu else "")
                   + ("env -u GOOGLE_APPLICATION_CREDENTIALS " if adc else "")
                   + shlex.join(arguments))
        with (self.runtime / f"{label}.log").open("x") as log:
            process = subprocess.Popen(
                ["bash", "scripts/docker_shell.sh", "bash", "-lc", command], cwd=ROOT,
                env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            )
            self.status(label, child_pid=process.pid)
            code = process.wait()
        if code:
            raise RuntimeError(f"{label} exited {code}; inspect its log")

    def run(self):
        self.frozen_sources()
        if not self.wait_for_training():
            return
        self.frozen_sources()
        self.retained_endpoints()
        if (self.runtime / "STOP").exists():
            self.status("cancelled_before_evaluation")
            return
        arguments = ["python", "-u", "scripts/rt_nextlat_fuzzy_length_eval.py"]
        for arm in self.config["arms"]:
            arguments.extend(["--run", arm["label"] + "=" + arm["training"]])
        arguments.extend(["--data", self.config["data"], "--output", self.config["report"], "--wandb"])
        self.invoke("evaluate-four-models", arguments, gpu=True)
        report = read(ROOT / self.config["report"] / "report.json")
        if report.get("status") != "complete":
            raise RuntimeError("Four-model evaluation report did not complete")
        self.invoke("retain-final", [
            "python", "scripts/rt_nextlat_fuzzy_retain.py", "--runtime", str(self.relative),
            "--report", self.config["report"], "--prefix", self.config["retention_prefix"],
            "--label", "final",
        ], adc=True)
        receipt = read(self.runtime / "retention/final-receipt.json")
        if receipt.get("status") != "verified":
            raise RuntimeError("Length evaluation retention was not verified")
        self.status("complete", report=self.config["report"], retained_uri=receipt["uri"],
                    no_following_experiments=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    queue = LengthQueue(args.runtime)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, queue.request_stop)
    try:
        queue.run()
    except BaseException as error:
        queue.status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
