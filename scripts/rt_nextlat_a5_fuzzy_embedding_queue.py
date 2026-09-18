#!/usr/bin/env python3
"""Host-side sequential supervisor for three matched mixed embedding pilots.

Only standard-library orchestration runs on the host. CUDA and model/report
work enter the required project container, with explicit CPU/GPU selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
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


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Queue:
    def __init__(self, runtime):
        self.runtime = Path(runtime).resolve()
        self.relative = self.runtime.relative_to(ROOT)
        self.config = read(self.runtime / "execution-config.json")
        self.finished = []
        self.current = None

    def status(self, phase, **values):
        write(self.runtime / "status.json", {
            "schema": "rt-nextlat-mixed-embedding-queue-v1", "phase": phase,
            "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "current_variant": self.current, "completed_variants": self.finished,
            "authorized_order": [arm["variant"] for arm in self.config["arms"]],
            "endpoint_per_arm": 15000, "batch_per_task": 2560, **values,
        })

    def stopped(self):
        return any((self.runtime / name).exists() for name in ("STOP", "STOP_QUEUE"))

    def request_stop(self, number, frame):
        # Training checks this same marker between optimizer updates.
        (self.runtime / "STOP").touch(exist_ok=True)

    def frozen_sources(self):
        ready = read(self.runtime / "ready.json")
        if ready.get("passed") is not True:
            raise RuntimeError("CPU readiness has not passed")
        for relative, expected in ready["source_sha256"].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError("Frozen source/configuration changed: " + relative)

    def invoke(self, label, arguments, *, gpu=False, adc=False):
        environment = os.environ.copy()
        environment["CDRM_DOCKER_GPUS"] = "all" if gpu else "none"
        command = ('test -f /.dockerenv && test "$PWD" = /workspace/cdrm-w-latent && '
                   + ("nvidia-smi -L && " if gpu else "")
                   + ("env -u GOOGLE_APPLICATION_CREDENTIALS " if adc else "")
                   + shlex.join(arguments))
        with (self.runtime / f"{label}.log").open("x") as log:
            process = subprocess.Popen(
                ["bash", "scripts/docker_shell.sh", "bash", "-lc", command],
                cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
            )
            self.status(label, child_pid=process.pid)
            code = process.wait()
        if code:
            raise RuntimeError(f"{label} exited {code}; inspect its log")

    def wait_for_baseline(self):
        parent = ROOT / self.config["baseline_runtime"]
        while True:
            if self.stopped():
                self.status("cancelled_before_variants")
                return False
            packet = read(parent / "status.json")
            phase = packet["phase"]
            if phase == "complete":
                break
            if phase in ("failed", "stopped_and_retained", "cancelled_before_training"):
                raise RuntimeError(f"Baseline did not complete the required 15k budget: {phase}")
            try:
                os.kill(packet["pid"], 0)
            except ProcessLookupError as error:
                raise RuntimeError("Baseline supervisor exited without completed summary/retention") from error
            self.status("waiting_for_baseline_summary_and_retention", baseline_phase=phase)
            time.sleep(30)
        training = read(ROOT / self.config["baseline_training"] / "report.json")
        summary = read(ROOT / self.config["baseline_report"] / "report.json")
        receipt = read(parent / "retention/final-receipt.json")
        if (training["status"] != "complete" or training["completed_updates"] != 15000
                or summary["endpoint"] != 15000 or receipt["status"] != "verified"):
            raise RuntimeError("Baseline completion/summary/retention is inconsistent")
        checkpoint = next(p for p in training["checkpoints"] if p["completed_updates"] == 15000)
        member = next(p for p in receipt["members"] if
                      p["path"] == "runtime/train-mixed-continue15000/checkpoints/step-015000.pt")
        if checkpoint["sha256"] != member["sha256"] or checkpoint["sha256"] != packet["checkpoint_sha256"]:
            raise RuntimeError("Baseline retained endpoint differs")
        write(self.runtime / "baseline-complete.json", {
            "baseline_report": self.config["baseline_report"], "endpoint": 15000,
            "checkpoint_sha256": checkpoint["sha256"], "retained_uri": receipt["uri"],
            "final": summary["final"], "wandb": training["wandb"],
        })
        return True

    def run(self):
        self.frozen_sources()
        if not self.wait_for_baseline():
            return
        compared = [("baseline", self.config["baseline_training"])]
        for arm in self.config["arms"]:
            if self.stopped():
                self.status("stopped_before_next_variant")
                return
            self.current = arm["variant"]
            self.frozen_sources()
            runtime = ROOT / arm["runtime"]
            if runtime.exists():
                raise FileExistsError(f"Refuse duplicate arm launch: {runtime}")
            runtime.mkdir()
            write(runtime / "execution-config.json", arm)
            write(runtime / "queue-readiness.json", read(self.runtime / "ready.json"))
            for name in ("scripts/rt_nextlat_a5_fuzzy_embedding_queue.py",
                         "scripts/rt_nextlat_a5_fuzzy_embedding_compare.py"):
                destination = runtime / "orchestration-source" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            self.invoke(self.current + "-preflight", [
                "python", "-u", "scripts/rt_nextlat_a5_fuzzy_embeddings_validate.py",
                *arm["validation_cli"],
            ], gpu=True)
            validation = read(runtime / "validation/report.json")
            if validation.get("status") != "passed":
                raise RuntimeError("Variant GPU preflight did not pass")
            if self.stopped():
                self.status("stopped_after_preflight")
                return
            self.invoke(self.current + "-train", [
                "python", "-u", "scripts/rt_nextlat_a5_fuzzy_embedding_train.py",
                *arm["training_cli"],
            ], gpu=True)
            training = read(ROOT / arm["training"] / "report.json")
            endpoint = training["completed_updates"]
            if (training["status"] not in ("complete", "stopped")
                    or training["start_update"] != 0 or not 0 <= endpoint <= 15000
                    or (training["status"] == "complete" and endpoint != 15000)
                    or training["parent_checkpoint"] is not None
                    or training["contract"]["batch_per_task"] != 2560
                    or training["contract"]["microbatch"] != 2560):
                raise RuntimeError("Variant did not obey fresh 15k B2560 contract")
            report_args = []
            if endpoint > 0:
                compared.append((self.current, arm["training"]))
                arguments = ["python", "-m", "scripts.rt_nextlat_a5_fuzzy_embedding_compare"]
                for label, directory in compared:
                    arguments.extend(["--run", f"{label}={directory}"])
                arguments.extend(["--output", arm["report"], "--wandb"])
                self.invoke(self.current + "-compare", arguments)
                if not (ROOT / arm["report"] / "report.json").exists():
                    raise RuntimeError("Comparative report did not complete")
                report_args = ["--report", arm["report"]]
            self.invoke(self.current + "-retain", [
                "python", "scripts/rt_nextlat_fuzzy_retain.py",
                "--runtime", arm["runtime"], *report_args,
                "--prefix", arm["retention_prefix"], "--label", "final",
            ], adc=True)
            receipt = read(runtime / "retention/final-receipt.json")
            checkpoint = next(p for p in training["checkpoints"] if p["completed_updates"] == endpoint)
            member = next(p for p in receipt["members"] if
                          p["path"] == f"runtime/train/checkpoints/step-{endpoint:06d}.pt")
            if receipt["status"] != "verified" or member["sha256"] != checkpoint["sha256"]:
                raise RuntimeError("Variant endpoint retention is inconsistent")
            completed = {"variant": self.current, "endpoint": endpoint,
                         "status": training["status"], "wandb": training["wandb"].get("run_url"),
                         "report": arm["report"] if report_args else None,
                         "checkpoint_sha256": checkpoint["sha256"], "retained_uri": receipt["uri"]}
            write(runtime / "completion.json", completed)
            self.finished.append(completed)
            if training["status"] == "stopped" or self.stopped():
                self.status("stopped_and_retained")
                return
        self.current = None
        self.status("complete", no_following_experiments=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    queue = Queue(args.runtime)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, queue.request_stop)
    try:
        queue.run()
    except BaseException as error:
        queue.status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
