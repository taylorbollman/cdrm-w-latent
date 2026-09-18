#!/usr/bin/env python3
"""Exact CPU checkpoint-boundary audit for an embedding-arm continuation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from scripts import rt_nextlat_a5_fuzzy_report as reporting


def require(condition, message):
    if not condition:
        raise ValueError(message)


def differences(left, right, prefix=""):
    result = []
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or left.shape != right.shape or not torch.equal(left, right)):
            result.append(prefix)
    elif isinstance(left, np.ndarray):
        if (not isinstance(right, np.ndarray) or left.dtype != right.dtype
                or left.shape != right.shape or not np.array_equal(left, right)):
            result.append(prefix)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            result.append(prefix + "/keys")
        else:
            for key in left:
                result.extend(differences(left[key], right[key], prefix + "/" + str(key)))
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            result.append(prefix + "/sequence")
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                result.extend(differences(a, b, prefix + "/" + str(index)))
    elif type(left) is not type(right) or left != right:
        result.append(prefix)
    return result


def audit(parent, child, *, expected_update, parent_sha256, output):
    parent, child, output = Path(parent), Path(child), Path(output)
    require(not output.exists(), "Boundary evidence output already exists")
    require(reporting.sha(parent) == parent_sha256, "Approved parent checkpoint changed")
    packets = {role: torch.load(path, map_location="cpu", weights_only=False)
               for role, path in (("parent", parent), ("child", child))}
    require(all(packet["completed_updates"] == expected_update for packet in packets.values()),
            "Checkpoint does not identify the expected continuation boundary")
    diff = differences(packets["parent"], packets["child"])
    fields = {key: not differences(value, packets["child"].get(key))
              for key, value in packets["parent"].items()}
    receipt = {
        "schema": "rt-nextlat-a5-fuzzy-recovery-boundary-state-audit-v1",
        "status": "failed" if diff else "passed", "passed": not diff,
        "completed_updates": expected_update, "entire_packet_exact": not diff,
        "differences": diff, "field_equality": fields,
        "checkpoints": {role: {"path": str(path.resolve()), "sha256": reporting.sha(path),
                                "bytes": path.stat().st_size}
                        for role, path in (("parent", parent), ("child", child))},
        "matching_examples_seen": packets["parent"]["examples_seen"],
        "matching_next_cursors": packets["parent"]["next_cursors"],
        "matching_order_chains": packets["parent"]["order_chains"],
        "scope": "Exact full saved-state comparison of approved parent and re-saved child boundary; no inference or updates.",
        "execution": "GPU-disabled project container, map_location=cpu.",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    require(not diff, "Resumed boundary differs from the approved parent")
    return receipt


def main():
    require(Path("/.dockerenv").is_file() and Path.cwd() == Path("/workspace/cdrm-w-latent"),
            "Use the required project container")
    require(not torch.cuda.is_initialized(), "This audit is CPU-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.parent, args.child, expected_update=args.expected_update,
                    parent_sha256=args.parent_sha256, output=args.output)
    print(json.dumps({"passed": receipt["passed"], "completed_updates": args.expected_update}), flush=True)


if __name__ == "__main__":
    main()
