#!/usr/bin/env python3
"""Run stored-value, embedding-head, then exact resumed-input mixed pilots.

This new supervisor leaves the original stopped queue and its frozen sources
unchanged. Only standard-library orchestration executes on the host.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import time

from scripts.rt_nextlat_a5_fuzzy_embedding_queue import Queue as OriginalQueue
from scripts.rt_nextlat_a5_fuzzy_embedding_queue import ROOT, read, sha, write


class Queue(OriginalQueue):
    def wait_for_previous(self):
        previous = ROOT / self.config["previous_queue"]
        while True:
            if self.stopped():
                self.status("cancelled_before_variants")
                return False
            packet = read(previous / "status.json")
            phase = packet["phase"]
            if phase == "stopped_and_retained":
                break
            if phase in ("complete", "failed", "cancelled_before_variants",
                         "stopped_before_next_variant", "stopped_after_preflight"):
                raise RuntimeError("Original input queue did not reach its approved retained stop: " + phase)
            try:
                os.kill(packet["pid"], 0)
            except ProcessLookupError as error:
                raise RuntimeError("Original input supervisor exited before retained stop") from error
            self.status("waiting_for_previous_input_retention", previous_phase=phase)
            time.sleep(30)
        self.verify_previous_input()
        return True

    def verify_previous_input(self):
        cfg = self.config
        update = cfg["previous_input_update"]
        expected_sha = cfg["previous_input_checkpoint_sha256"]
        report = read(ROOT / cfg["previous_input_training"] / "report.json")
        receipt = read(ROOT / cfg["previous_input_runtime"] / "retention/final-receipt.json")
        checkpoint = ROOT / cfg["previous_input_checkpoint"]
        record = next(p for p in report["checkpoints"] if p["completed_updates"] == update)
        member = next(p for p in receipt["members"] if
                      p["path"] == f"runtime/train/checkpoints/step-{update:06d}.pt")
        if (report["status"] != "stopped" or report["start_update"] != 0
                or report["completed_updates"] != update or not 0 < update < 15000
                or receipt["status"] != "verified"
                or record["sha256"] != expected_sha or member["sha256"] != expected_sha
                or sha(checkpoint) != expected_sha):
            raise RuntimeError("Original input checkpoint or verified retention differs")
        write(self.runtime / "previous-input-retained.json", {
            "checkpoint": cfg["previous_input_checkpoint"], "checkpoint_sha256": expected_sha,
            "completed_updates": update, "retained_uri": receipt["uri"],
            "scope": "Approved stopped input state will continue after stored-value and head pilots.",
        })

    def verify_arm_configuration(self):
        arms = self.config["arms"]
        if [arm["variant"] for arm in arms] != ["value", "head", "input"]:
            raise RuntimeError("Authorized arm order is value, head, input")
        for arm in arms:
            expected = self.config["previous_input_update"] if arm["variant"] == "input" else 0
            expected_sha = self.config["previous_input_checkpoint_sha256"] if expected else None
            if (arm["expected_start_update"] != expected
                    or arm.get("parent_checkpoint_sha256") != expected_sha):
                raise RuntimeError("Arm resume boundary does not match the approved order change")
            argv = arm["training_cli"]
            required = {"--output": arm["training"], "--updates": "15000", "--batch-per-task": "2560",
                        "--microbatch": "2560", "--stop-file": str(self.relative / "STOP")}
            for flag, value in required.items():
                if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
                    raise RuntimeError("Training CLI differs from approved contract: " + flag)
            if expected:
                if (argv.count("--resume") != 1
                        or argv[argv.index("--resume") + 1] != self.config["previous_input_checkpoint"]):
                    raise RuntimeError("Input arm must resume the approved checkpoint")
            elif "--resume" in argv:
                raise RuntimeError("Value and head arms must start fresh")

    def verify_training(self, arm, report):
        start = arm["expected_start_update"]
        endpoint = report["completed_updates"]
        if (report["status"] not in ("complete", "stopped") or report["start_update"] != start
                or not start <= endpoint <= 15000
                or (report["status"] == "complete" and endpoint != 15000)
                or report["contract"]["batch_per_task"] != 2560
                or report["contract"]["microbatch"] != 2560):
            raise RuntimeError("Variant did not obey the approved 15k B2560 contract")
        if start:
            parent = report.get("parent_checkpoint") or {}
            path = Path(parent.get("path", ""))
            expected = Path(self.config["previous_input_checkpoint"])
            if path.is_absolute():
                for prefix in (ROOT, Path("/workspace/cdrm-w-latent")):
                    if path.is_relative_to(prefix):
                        path = path.relative_to(prefix)
                        break
            original = read(ROOT / self.config["previous_input_training"] / "report.json")
            if (path != expected or parent.get("sha256") != arm["parent_checkpoint_sha256"]
                    or report["contract"] != original["contract"]
                    or report["initialization"] != original["initialization"]):
                raise RuntimeError("Input continuation changed its parent or exact training contract")
        elif report.get("parent_checkpoint") is not None:
            raise RuntimeError("Fresh value/head arm cannot inherit a checkpoint")
        return endpoint

    def run(self):
        self.verify_arm_configuration()
        self.frozen_sources()
        if not self.wait_for_previous() or not self.wait_for_baseline():
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
            for name in ("scripts/rt_nextlat_a5_fuzzy_embedding_reordered_queue.py",
                         "scripts/rt_nextlat_a5_fuzzy_embedding_queue.py",
                         "scripts/rt_nextlat_embedding_resume_audit.py",
                         "scripts/rt_nextlat_a5_fuzzy_embedding_compare.py"):
                destination = runtime / "orchestration-source" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            if not arm["expected_start_update"]:
                self.invoke(self.current + "-preflight", [
                    "python", "-u", "scripts/rt_nextlat_a5_fuzzy_embeddings_validate.py",
                    *arm["validation_cli"],
                ], gpu=True)
                if read(runtime / "validation/report.json").get("status") != "passed":
                    raise RuntimeError("Variant GPU preflight did not pass")
            if self.stopped():
                self.status("stopped_after_preflight")
                return
            self.invoke(self.current + "-train", [
                "python", "-u", "scripts/rt_nextlat_a5_fuzzy_embedding_train.py",
                *arm["training_cli"],
            ], gpu=True)
            training = read(ROOT / arm["training"] / "report.json")
            endpoint = self.verify_training(arm, training)
            if arm["expected_start_update"]:
                self.invoke("input-resume-boundary", [
                    "python", "-m", "scripts.rt_nextlat_embedding_resume_audit",
                    "--parent", self.config["previous_input_checkpoint"],
                    "--child", str(Path(arm["training"]) / "checkpoints"
                                   / f"step-{arm['expected_start_update']:06d}.pt"),
                    "--expected-update", str(arm["expected_start_update"]),
                    "--parent-sha256", arm["parent_checkpoint_sha256"],
                    "--output", str(Path(arm["runtime"]) / "recovery-boundary-state-audit.json"),
                ])
            report_args = []
            if endpoint > arm["expected_start_update"]:
                compared.append((self.current, arm["training"]))
                arguments = ["python", "-m", "scripts.rt_nextlat_a5_fuzzy_embedding_compare"]
                for label, directory in compared:
                    arguments.extend(["--run", f"{label}={directory}"])
                arguments.extend(["--output", arm["report"], "--wandb"])
                self.invoke(self.current + "-compare", arguments)
                if read(ROOT / arm["report"] / "report.json").get("status") != "complete":
                    raise RuntimeError("Comparative report did not complete")
                report_args = ["--report", arm["report"]]
            self.invoke(self.current + "-retain", [
                "python", "scripts/rt_nextlat_fuzzy_retain.py", "--runtime", arm["runtime"],
                *report_args, "--prefix", arm["retention_prefix"], "--label", "final",
            ], adc=True)
            receipt = read(runtime / "retention/final-receipt.json")
            checkpoint = next(p for p in training["checkpoints"] if p["completed_updates"] == endpoint)
            member = next(p for p in receipt["members"] if
                          p["path"] == f"runtime/train/checkpoints/step-{endpoint:06d}.pt")
            local = ROOT / arm["training"] / "checkpoints" / f"step-{endpoint:06d}.pt"
            if (receipt["status"] != "verified" or member["sha256"] != checkpoint["sha256"]
                    or sha(local) != checkpoint["sha256"]):
                raise RuntimeError("Variant endpoint retention is inconsistent")
            completed = {"variant": self.current, "endpoint": endpoint, "status": training["status"],
                         "wandb": training["wandb"].get("run_url"),
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
