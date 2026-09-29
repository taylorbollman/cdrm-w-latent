#!/usr/bin/env python3
"""Bounded two-process CPU/Gloo fault and checkpoint-replay harness.

The toy checkpoint is not a campaign checkpoint implementation. It tests safe
control flow and recovery semantics without modifying frozen production code.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from scripts.olmo_campaign_lifecycle import BoundaryActionError, coordinated_boundary_action

CASES = ("success", "log_failure", "persist_failure", "data_failure",
         "phase_mismatch", "ownership_mismatch", "invalid_descriptor", "two_errors")
SECRET_MARKER = "fixture-private-message-must-not-be-reported"
SOURCES = ("scripts/olmo_campaign_lifecycle.py", "scripts/olmo_campaign_lifecycle_check.py",
           "tests/test_campaign_lifecycle.py", "docs/reports/olmo-campaign-lifecycle/protocol.md")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def same_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_tree(x, y) for x, y in zip(left, right))
    return left == right


def construct(rank):
    torch.manual_seed(9000 + rank)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
    return model, optimizer


def snapshot(model, optimizer, cursor):
    return copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "rng": torch.get_rng_state(), "cursor": cursor})


def update(model, optimizer, cursor):
    optimizer.zero_grad(set_to_none=True)
    inputs, targets = torch.randn(4, 3), torch.randn(4, 2)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return cursor + 1


def gather_states(state):
    states = [None, None]
    dist.all_gather_object(states, state)
    return states


def publish(path, states, *, fail=False):
    temporary = path.with_suffix(".unpublished.pt")
    with temporary.open("wb") as handle:
        torch.save({"schema": "cpu-lifecycle-toy-v1", "ranks": states}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if fail:
        raise OSError(SECRET_MARKER)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def recover(path, rank):
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload["schema"] != "cpu-lifecycle-toy-v1" or len(payload["ranks"]) != 2:
        raise ValueError("Invalid toy checkpoint")
    saved = payload["ranks"][rank]
    model, optimizer = construct(rank)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    return model, optimizer, saved["cursor"]


def run_case(output, rank, case):
    model, optimizer = construct(rank)
    cursor = update(model, optimizer, 0)
    checkpoint = output / f"{case}.pt"
    states = gather_states(snapshot(model, optimizer, cursor))
    coordinated_boundary_action("publish initial", lambda: publish(checkpoint, states), rank_zero_only=True)
    initial_hash = digest(checkpoint)
    first = snapshot(model, optimizer, cursor)
    callbacks = 0
    error = None

    def logging():
        nonlocal callbacks
        callbacks += 1
        if case == "log_failure":
            raise RuntimeError(SECRET_MARKER)
        return 100 + rank

    def data():
        nonlocal callbacks
        callbacks += 1
        if case == "data_failure" and rank == 1:
            raise ValueError(SECRET_MARKER)
        return rank

    def failure():
        nonlocal callbacks
        callbacks += 1
        raise (ValueError if rank == 0 else OSError)(SECRET_MARKER)

    expected_failure = case != "success"
    current = first
    try:
        if case in ("success", "log_failure", "persist_failure"):
            cursor = update(model, optimizer, cursor)
            current = snapshot(model, optimizer, cursor)
            if case == "persist_failure":
                states = gather_states(current)
                coordinated_boundary_action("publish update", lambda: publish(checkpoint, states, fail=True),
                                            rank_zero_only=True)
            else:
                result = coordinated_boundary_action("log update", logging, rank_zero_only=True)
                if result != (100 if rank == 0 else None):
                    raise AssertionError("Rank-zero callback returned to wrong rank")
                if not same_tree(current, snapshot(model, optimizer, cursor)):
                    raise AssertionError("Successful logging changed numerical state")
                states = gather_states(current)
                coordinated_boundary_action("publish update", lambda: publish(checkpoint, states), rank_zero_only=True)
        elif case == "data_failure":
            coordinated_boundary_action("prepare next data", data)
        elif case == "phase_mismatch":
            coordinated_boundary_action("publish" if rank == 0 else "prepare", data)
        elif case == "ownership_mismatch":
            coordinated_boundary_action("ownership", data, rank_zero_only=(rank == 0))
        elif case == "invalid_descriptor":
            coordinated_boundary_action("prepare" if rank == 0 else None, data)  # type: ignore[arg-type]
        elif case == "two_errors":
            coordinated_boundary_action("two callbacks", failure)
        else:
            raise AssertionError("Unknown scenario")
    except BoundaryActionError as exc:
        error = str(exc)
    # A failed boundary is handled as a failed run, never by proceeding to update3.
    checks = {"expected_failure_observed": (error is not None) == expected_failure,
              "in_memory_update_not_rolled_back": same_tree(current, snapshot(model, optimizer, cursor)),
              "no_third_update": cursor <= 2,
              "exception_text_redacted": error is None or SECRET_MARKER not in error}
    if case in ("phase_mismatch", "ownership_mismatch", "invalid_descriptor"):
        checks["callback_prevented"] = callbacks == 0
    if case in ("success", "log_failure"):
        checks["rank_zero_callback_only"] = callbacks == (1 if rank == 0 else 0)
    checks["published_checkpoint_preserved_or_advanced"] = (
        digest(checkpoint) != initial_hash if case == "success" else digest(checkpoint) == initial_hash)
    recovered, recovered_optimizer, recovered_cursor = recover(checkpoint, rank)
    checks["complete_checkpoint_readable"] = same_tree(
        snapshot(recovered, recovered_optimizer, recovered_cursor), current if case == "success" else first)
    replayed = False
    if cursor == 2 and expected_failure:
        recovered_cursor = update(recovered, recovered_optimizer, recovered_cursor)
        checks["lost_unsaved_update_replays_exactly"] = same_tree(
            snapshot(recovered, recovered_optimizer, recovered_cursor), current)
        replayed = True
    checks["cpu_only"] = not torch.cuda.is_initialized()
    return {"case": case, "rank": rank, "error": error, "callback_count": callbacks,
            "in_memory_completed_updates": cursor,
            "last_published_update": 2 if case == "success" else 1,
            "replayed_unsaved_update": replayed, "checks": checks}


def worker(rank, rendezvous, output, timeout_seconds):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=timeout_seconds))
    try:
        reports = []
        for case in CASES:
            reports.append(run_case(Path(output), rank, case))
        write_json(Path(output) / f"rank-{rank}.json", reports)
    finally:
        dist.destroy_process_group()


def run_harness(output, *, timeout_seconds=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.is_initialized():
        raise RuntimeError("This diagnostic is CPU-only")
    start = time.monotonic()
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, str(output.resolve()/"rendezvous"),
                                                     str(output.resolve()), timeout_seconds)) for rank in range(2)]
    for process in processes:
        process.start()
    deadline = start + timeout_seconds * 3
    try:
        for process in processes:
            process.join(max(0, deadline-time.monotonic()))
        if any(process.is_alive() or process.exitcode != 0 for process in processes):
            raise RuntimeError("CPU lifecycle worker failed or exceeded bounded deadline")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    ranks = [json.loads((output/f"rank-{rank}.json").read_text()) for rank in range(2)]
    checks = {"all_case_checks": all(all(row["checks"].values()) for rows in ranks for row in rows),
              "matching_rank_errors": all(a["error"] == b["error"] for a,b in zip(*ranks)),
              "all_eight_cases": all([r["case"] for r in rows] == list(CASES) for rows in ranks),
              "cpu_only_parent": not torch.cuda.is_initialized()}
    return {"schema": "olmo-campaign-lifecycle-v1", "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic()-start, "backend": "gloo", "world_size": 2,
            "checks": checks, "ranks": ranks,
            "limitations": ["New opt-in helper only; existing campaign runner unchanged.",
                            "Python callback exceptions, not process death or hung/in-flight collectives.",
                            "CPU/Gloo does not qualify NCCL/CUDA graph fault recovery.",
                            "Completed unsaved updates remain in memory and are replayed from last complete checkpoint."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_hashes = {path: digest(ROOT/path) for path in SOURCES}
    report = run_harness(args.output_dir)
    for path in SOURCES:
        destination = args.output_dir/"source-snapshot"/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, destination)
    report["sources"] = source_hashes
    report["checks"]["sources_unchanged"] = all(digest(ROOT/p) == h == digest(args.output_dir/"source-snapshot"/p)
                                                for p,h in source_hashes.items())
    write_json(args.output_dir/"report.json", report)
    print(json.dumps({"elapsed_seconds": report["elapsed_seconds"], "checks": report["checks"]}, sort_keys=True))
    if not all(report["checks"].values()):
        raise RuntimeError("CPU lifecycle acceptance failed")


if __name__ == "__main__":
    main()
