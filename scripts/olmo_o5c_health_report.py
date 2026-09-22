#!/usr/bin/env python3
"""Describe numerical execution health from completed O5c event records."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def finite_tree(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value, list): return all(finite_tree(v) for v in value)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {"schema": "olmo-o5c-optimization-health-v1", "arms": {},
        "scope": "Event scalars and recorded frozen-state checks; not an independent checkpoint replay",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for arm in ("code", "mixed"):
        path = args.runs/arm
        report = json.loads((path/"report.json").read_text())
        if report["status"] != "completed" or report["resumptions"]:
            raise ValueError("This descriptive helper requires complete unrestarted arms; adjudicate resumed events explicitly")
        events = [json.loads(line) for line in (path/"events.jsonl").read_text().splitlines()]
        updates = [r for r in events if r["kind"] == "update"]
        expected = report["configuration"]["schedule"]["total_updates"]
        if [r["counters"]["optimizer_updates"] for r in updates] != list(range(1,expected+1)):
            raise ValueError("Missing or duplicated optimizer update events")
        if not finite_tree(events) or report["frozen_state_initial"] != report["frozen_state_final"]:
            raise ValueError("Nonfinite recorded scalar or frozen-state mutation")
        norms = [r["gradient_norm_before_clip"] for r in updates]
        seconds = [r["seconds"] for r in updates]
        summary["arms"][arm] = {
            "input_hashes": {name: hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ("report.json","events.jsonl")},
            "updates": expected, "all_recorded_scalars_finite": True, "frozen_state_unchanged": True,
            "wall_seconds_including_evaluation_checkpoint": (datetime.fromisoformat(report["finished_utc"])-datetime.fromisoformat(report["started_utc"])).total_seconds(),
            "gradient_norm": {"min": min(norms), "median": statistics.median(norms), "max": max(norms)},
            "clipped_updates": sum(n > report["configuration"]["max_grad_norm"] for n in norms),
            "step_seconds": {"median": statistics.median(seconds), "sum_excluding_evaluation_checkpoint": sum(seconds)},
            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in updates),
            "counters": report["counters"], "domain_counts": report["domain_counts"],
            "checkpoints": [{"update":r["optimizer_updates"], "retained": bool(r.get("storage")),
                             "size_bytes":r["size_bytes"]} for r in report["checkpoints"]]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({arm:{k:v for k,v in row.items() if k in ("gradient_norm","clipped_updates","step_seconds")}
                      for arm,row in summary["arms"].items()},indent=2))


if __name__ == "__main__": main()
