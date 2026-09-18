#!/usr/bin/env python3
"""Check the user's exact A5 criterion before launching the mixed pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(value):
    path = str(value)
    for prefix in ("/workspace/cdrm-w-latent/", "/home/taylorbollman/cdrm-w-latent/"):
        if path.startswith(prefix):
            return ROOT / path[len(prefix):]
    return Path(path)


def select_gate(report_path, *, expected_endpoint=10000, expected_rows=102400):
    path = Path(report_path)
    report = json.loads(path.read_text())
    if (report["status"] != "complete" or report["completed_updates"] != expected_endpoint
            or report["contract"]["mode"] != "a5-only"):
        raise ValueError("Require the completed A5-only pilot at its declared endpoint")
    observations = list(report["evaluations"])
    inherited = report.get("a5_positive_gate", {}).get("inherited_from")
    visited, limit = {str(path.resolve())}, expected_endpoint
    while inherited:
        parent_path = local_path(inherited["report"]).resolve()
        if str(parent_path) in visited or sha(parent_path) != inherited["report_sha256"]:
            raise ValueError("Invalid inherited A5 evidence provenance")
        visited.add(str(parent_path))
        parent = json.loads(parent_path.read_text())
        limit = min(limit, inherited["restored_update"])
        if parent["contract"] != report["contract"] or type(limit) is not int or limit < 0:
            raise ValueError("Inherited A5 evidence belongs to a different training contract")
        observations.extend(row for row in parent["evaluations"] if row["update"] <= limit)
        inherited = parent.get("a5_positive_gate", {}).get("inherited_from")
    evidence, seen = [], set()
    for row in observations:
        if row.get("task") != "a5" or row.get("role") != "ood_dev" or row.get("scope") != "full":
            continue
        count = row.get("whole_word_correct", row.get("whole_word_exact_count"))
        rows = row["evaluated_rows"]
        step = row["update"]
        if (type(count) is not int or type(rows) is not int or rows != expected_rows
                or row["length"] != 36 or not 0 <= count <= rows
                or type(step) is not int or not 0 <= step <= expected_endpoint
                or not math.isfinite(row["whole_word_exact_match"])
                or abs(row["whole_word_exact_match"] - count / rows) > 1e-12):
            raise ValueError("Invalid full length-36 whole-word evidence")
        checkpoint = row["checkpoint"]
        if checkpoint["completed_updates"] != step or sha(local_path(checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError("A5 evaluation checkpoint identity failed verification")
        identity = (step, checkpoint["sha256"])
        if identity in seen:
            continue
        seen.add(identity)
        evidence.append({"update": step, "whole_word_correct": count, "rows": rows,
                         "whole_word_exact_match": count / rows, "checkpoint": checkpoint})
    if not any(row["update"] == expected_endpoint for row in evidence):
        raise ValueError("Missing full A5 endpoint evaluation")
    evidence.sort(key=lambda row: row["update"])
    positive = [row for row in evidence if row["whole_word_correct"] > 0]
    # The trainer carries checked parent evaluation records across a resume;
    # decisions always derive from explicit checkpoint-bound observations.
    return {"schema": "rt-nextlat-a5-mixed-eligibility-v1", "eligible": bool(positive),
            "criterion": "At least one complete correct length-36 word in a full 102400-word development evaluation during the completed 10k A5-only pilot",
            "source_report": str(path.resolve()), "source_report_sha256": sha(path),
            "a5_completed_updates": expected_endpoint, "evidence": evidence,
            "first_positive": positive[0] if positive else None,
            "positive_by_3000": any(row["update"] <= 3000 for row in positive),
            "positive_by_5000": any(row["update"] <= 5000 for row in positive),
            "endpoint": next(row for row in evidence if row["update"] == expected_endpoint),
            "next_action": "fresh mixed pilot to10000 updates" if positive else "stop after A5 report; mixed criterion not met",
            "confirmation_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_gate(args.report)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("eligible", "first_positive", "positive_by_3000", "next_action")}))


if __name__ == "__main__":
    main()
