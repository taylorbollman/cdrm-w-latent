#!/usr/bin/env python3
"""Opt-in abrupt-rank-exit acceptance around the frozen tiny guarded loop."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import sha256_file
from scripts import olmo_campaign_loop_guarded as guarded

VERSION = "olmo-campaign-abrupt-rank-exit-v1"
EXIT_CODE = 73


def source_hashes():
    result = guarded.source_hashes()
    for name in ("scripts/olmo_campaign_rank_failure.py", "tests/test_campaign_rank_failure.py",
                 "docs/reports/olmo-campaign-lifecycle/rank-failure-protocol.md"):
        result[name] = sha256_file(ROOT / name)
    return dict(sorted(result.items()))


def write_exit_marker(args, *, rank, call):
    """Authenticate a retained boundary before deliberately losing this rank."""
    latest = args.output_dir / "latest-checkpoint.json"
    receipt = json.loads(latest.read_text())
    checkpoint = args.checkpoint_root / "update-000001"
    if (rank != 1 or call != 2 or receipt["counters"]["optimizer_updates"] != 1
            or Path(receipt["directory"]) != checkpoint
            or sha256_file(checkpoint / "manifest.json") != receipt["manifest_sha256"]
            or not receipt.get("retention", {}).get("download_sha256_verified")):
        raise RuntimeError("Abrupt-exit test requires an authenticated retained update1")
    marker = {"schema": VERSION, "rank": rank, "backward_invocation": call,
              "exit_code": EXIT_CODE, "last_committed_update": 1,
              "checkpoint_manifest_sha256": receipt["manifest_sha256"],
              "location": "before rank1 enters the second update backward; rank0 may enter its collectives"}
    with (args.output_dir / "rank-1-deliberate-exit.json").open("x") as stream:
        json.dump(marker, stream, sort_keys=True, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return marker


@contextmanager
def inject_exit(args, coordinator, *, enabled, runner_class=None, exit_process=None):
    runner_class = guarded.legacy.CampaignDDPGraphTraining if runner_class is None else runner_class
    exit_process = os._exit if exit_process is None else exit_process
    if enabled and (args.resume is not None or args.inject_log_error_at is not None
                    or args.request_stop_after is not None or args.storage_prefix is None):
        raise ValueError("Inject only a fresh retained run without another failure or stop request")
    original = runner_class.backward
    calls = 0

    def backward(runner, *values, **keywords):
        nonlocal calls
        calls += 1
        if enabled and coordinator.rank == 1 and calls == 2:
            write_exit_marker(args, rank=coordinator.rank, call=calls)
            exit_process(EXIT_CODE)
            raise RuntimeError("The abrupt process-exit primitive unexpectedly returned")
        return original(runner, *values, **keywords)

    with patch.object(runner_class, "backward", backward):
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inject-rank-one-exit", action="store_true")
    control, remaining = parser.parse_known_args(argv)

    def stage(args, coordinator, device, runtime, determinism, report, tracker):
        report["rank_failure_acceptance"] = {
            "version": VERSION, "injected": control.inject_rank_one_exit,
            "scope": "Abrupt rank1 exit before update2 backward; no in-flight checkpoint or same-process recovery"}
        with inject_exit(args, coordinator, enabled=control.inject_rank_one_exit):
            return guarded.run_stage_releasing_failure(args, coordinator, device, runtime,
                                                       determinism, report, tracker)

    with guarded.guarded_driver_scope(), patch.object(guarded.legacy, "source_hashes", source_hashes), \
            patch.object(guarded.legacy, "run_stage", stage):
        return guarded.legacy.main(remaining)


if __name__ == "__main__":
    main()
