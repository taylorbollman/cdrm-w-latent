#!/usr/bin/env python3
"""Observe the reordered value/head/input training queue, then evaluate lengths.

The original observer, frozen data and evaluator remain unchanged. Only the
required training completion order differs; all four arms still need matched,
retained 15,000-update checkpoints before this evaluation-only queue uses GPU.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rt_nextlat_fuzzy_length_queue import LengthQueue, read


class ReorderedLengthQueue(LengthQueue):
    def wait_for_training(self):
        expected = self.config.get("expected_training_order")
        if expected != ["value", "head", "input"]:
            raise RuntimeError("Reordered observer requires value/head/input training order")
        source = ROOT / self.config["training_queue"]
        incomplete_phases = {
            "failed", "stopped_and_retained", "stopped_before_next_variant",
            "cancelled_before_variants", "stopped_after_preflight",
            "cancelled_before_launch", "cancelled_before_training",
        }
        while True:
            if (self.runtime / "STOP").exists():
                self.status("cancelled_before_evaluation")
                return False
            packet = read(source / "status.json")
            phase = packet["phase"]
            if phase == "complete":
                finished = packet["completed_variants"]
                if ([item["variant"] for item in finished] != expected
                        or any(item["endpoint"] != 15000 or item["status"] != "complete"
                               for item in finished)):
                    raise RuntimeError("Reordered training queue did not complete value/head/input at 15k")
                return True
            if phase in incomplete_phases:
                self.status(
                    "blocked_training_incomplete", training_phase=phase,
                    expected_training_order=expected,
                    detail="All four matched 15k endpoints are required; adjust explicitly if training scope changes",
                )
                return False
            try:
                os.kill(packet["pid"], 0)
            except ProcessLookupError as error:
                raise RuntimeError("Training supervisor exited without completing all arms") from error
            self.status("waiting_for_training_queue", training_phase=phase,
                        current_training_variant=packet.get("current_variant"),
                        expected_training_order=expected)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    queue = ReorderedLengthQueue(args.runtime)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, queue.request_stop)
    try:
        queue.run()
    except BaseException as error:
        queue.status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
