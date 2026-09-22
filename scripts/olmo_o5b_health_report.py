#!/usr/bin/env python3
"""CPU-only optimization observations for two completed O5b arms.

Recorded finite scalars establish observed health, not tensor-level numerical
equivalence. Quantiles use linear interpolation. Warmup differences are reported
as measured; identical configurations are not treated as exact trajectory replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ("ordinary", "fbt")
SCHEMA = "olmo-o5b-optimization-summary-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite_values(value, path="root"):
    """Check every floating-point value, including superseded audit events."""
    if isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite recorded value at {path}")
        return 1
    if isinstance(value, dict):
        return sum(finite_values(child, f"{path}.{key}") for key, child in value.items())
    if isinstance(value, list):
        return sum(finite_values(child, f"{path}[{index}]") for index, child in enumerate(value))
    return 0


def quantile(values, fraction):
    ordered = sorted(values)
    require(bool(ordered), "Cannot summarize empty observations")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {"count": len(values), "min": min(values), "median": quantile(values, .5),
            "p95": quantile(values, .95), "p99": quantile(values, .99), "max": max(values),
            "mean": statistics.fmean(values)}


def numeric(value, name, minimum=0.):
    require(type(value) in (float, int) and math.isfinite(value) and value >= minimum,
            f"Invalid numerical observation: {name}")
    return value


def update_id(row):
    update = row.get("counters", {}).get("optimizer_updates")
    require(type(update) is int and update > 0, "Invalid update counter")
    return update


def canonical_updates(report, events):
    raw = [row for row in events if row.get("kind") == "update"]
    require(raw, "No optimizer update events")
    selected = {}
    for row in raw:
        require(row.get("schema") == "olmo-lm-optimizer-step-v1" and row.get("update_completed") is True,
                "Expected completed optimizer-step records")
        selected[update_id(row)] = row
    duplicates = len(raw) - len(selected)
    require(not duplicates or report.get("resumptions"), "Duplicate updates without recorded resumption")
    end = report.get("counters", {}).get("optimizer_updates")
    require(type(end) is int and end >= 100, "Completed O5b evidence must contain at least 100 updates")
    require(set(selected) == set(range(1, end + 1)), "Update events omit or exceed the completed trajectory")
    rows = [selected[index] for index in range(1, end + 1)]
    require(rows[-1]["counters"] == report["counters"], "Final event/report exposure counters differ")
    return rows, {"raw_update_events": len(raw), "selected_updates": len(rows),
                  "superseded_update_events": duplicates,
                  "selection": "latest recorded occurrence per update; duplicates require recorded resumptions"}


def summarize_arm(report, rows, selection, checked_floats, events):
    norms, limits, seconds, memory = [], [], [], []
    for row in rows:
        norms.append(numeric(row.get("gradient_norm_before_clip"), "gradient_norm_before_clip"))
        limits.append(numeric(row.get("max_grad_norm"), "max_grad_norm", minimum=1e-300))
        seconds.append(numeric(row.get("seconds"), "seconds", minimum=1e-300))
        memory.append(numeric(row.get("peak_allocated_gib"), "peak_allocated_gib"))
        require(set(row.get("group_gradient_norm_after_clip", {})) == {"backbone", "fusion"}, "Missing native/fusion gradient observations")
        for value in row["group_gradient_norm_after_clip"].values():
            numeric(value, "group norm")
        require(len(row.get("pass_ce_means", [])) == 2, "Expected separate pass0/pass1 CE observations")
        for value in row["pass_ce_means"]:
            numeric(value, "pass CE")
    recent = rows[-100:]
    fractions = []
    for row in recent:
        groups = row["group_gradient_norm_after_clip"]
        total = sum(value * value for value in groups.values())
        fractions.append(groups["fusion"] ** 2 / total if total else 0.0)
    largest = max(rows, key=lambda row: row["gradient_norm_before_clip"])
    highest_memory = max(rows, key=lambda row: row["peak_allocated_gib"])
    gate_events = [row for row in events if "gate" in str(row.get("kind", "")).lower()]
    return {"status": report["status"], "finished_utc": report["finished_utc"],
            "exposure": report["counters"], "data_cursor": report.get("data_cursor"),
            "resumptions": report.get("resumptions", []), "event_selection": selection,
            "finite_event_check": {"passed": True, "checked_float_values": checked_floats,
                                   "scope": "all JSON event floats, including superseded audit events; not a scan of model tensors"},
            "gate_observations": {"final_status": report["status"], "explicit_gate_events": gate_events,
                                  "qualification": "Completion does not imply satisfactory retention; the frozen gate requires severe loss in both domains."},
            "clipping": {"updates": len(rows), "above_limit": sum(value > limit for value, limit in zip(norms, limits)),
                         "fraction_above_limit": sum(value > limit for value, limit in zip(norms, limits)) / len(rows),
                         "limits": sorted(set(limits))},
            "gradient_norm_before_clip": {**distribution(norms), "maximum_update": update_id(largest),
                                          "maximum_beta": largest["beta"], "maximum_pass_ce_means": largest["pass_ce_means"]},
            "step_seconds": distribution(seconds), "peak_allocated_gib": {**distribution(memory), "maximum_update": update_id(highest_memory)},
            "last100": {"first_update": update_id(recent[0]), "last_update": update_id(recent[-1]),
                        "gradient_norm_before_clip": distribution([row["gradient_norm_before_clip"] for row in recent]),
                        "group_gradient_norm_after_clip": {name: distribution([row["group_gradient_norm_after_clip"][name] for row in recent])
                                                            for name in ("backbone", "fusion")},
                        "fusion_squared_norm_fraction": distribution(fractions),
                        "fraction_definition": "fusion_norm_squared / (backbone_norm_squared + fusion_norm_squared); zero when both norms are zero",
                        "pass_ce_mean_over_updates": [statistics.fmean(row["pass_ce_means"][index] for row in recent) for index in (0, 1)]}}


def differences(left, right, extract):
    a, b = [extract(row) for row in left], [extract(row) for row in right]
    delta = [y - x for x, y in zip(a, b)]
    absolute = [abs(value) for value in delta]
    worst = max(range(len(delta)), key=lambda index: absolute[index])
    return {"exact_matches": sum(x == y for x, y in zip(a, b)), "updates": len(delta),
            "mean_signed_difference": statistics.fmean(delta), "mean_absolute_difference": statistics.fmean(absolute),
            "p95_absolute_difference": quantile(absolute, .95), "max_absolute_difference": absolute[worst],
            "maximum_difference_update": update_id(left[worst]), "ordinary_at_maximum": a[worst], "fbt_at_maximum": b[worst]}


def warmup_comparison(updates):
    left, right = (updates[arm][:100] for arm in ARMS)
    structural = ("beta", "counters", "counts", "lr_used", "lr_next", "objective_weights", "max_grad_norm",
                  "optimizer_state_bytes_by_device", "stack_input_tokens", "update_completed")
    fields = {key: {"matching_updates": sum(a.get(key) == b.get(key) for a, b in zip(left, right)),
                    "mismatch_updates": [update_id(a) for a, b in zip(left, right) if a.get(key) != b.get(key)]} for key in structural}
    excluded = {"seconds", "peak_allocated_gib"}
    exact = [update_id(a) for a, b in zip(left, right) if {k: v for k, v in a.items() if k not in excluded}
             == {k: v for k, v in b.items() if k not in excluded}]
    return {"updates_compared": 100, "difference_convention": "FBT minus ordinary", "structural_fields": fields,
            "all_structural_fields_match": all(value["matching_updates"] == 100 for value in fields.values()),
            "both_beta_zero": all(row.get("beta") == 0.0 for row in [*left, *right]),
            "exact_non_timing_update_matches": len(exact), "exact_non_timing_matching_updates": exact,
            "objective": differences(left, right, lambda row: numeric(row.get("objective"), "objective")),
            "pass_ce": {str(index): differences(left, right, lambda row, i=index: row["pass_ce_means"][i]) for index in (0, 1)},
            "gradient_norm_before_clip": differences(left, right, lambda row: row["gradient_norm_before_clip"]),
            "group_gradient_norm_after_clip": {name: differences(left, right, lambda row, n=name: row["group_gradient_norm_after_clip"][n])
                                               for name in ("backbone", "fusion")},
            "qualification": "Matched metadata does not establish exact replay. Differences are observations; this report does not identify their numerical cause."}


def build_summary(runs):
    runs = Path(runs)
    reports, updates, summaries, hashes = {}, {}, {}, {}
    for arm in ARMS:
        report_bytes = (runs / arm / "report.json").read_bytes()
        event_bytes = (runs / arm / "events.jsonl").read_bytes()
        report = json.loads(report_bytes)
        require(report.get("schema") == "olmo-o5b-arm-v1" and report.get("arm") == arm
                and report.get("status") == "completed" and report.get("finished_utc"), f"Arm {arm} is not completed")
        events = [json.loads(line) for line in event_bytes.splitlines() if line.strip()]
        require(all(isinstance(event, dict) for event in events), "Expected JSON event objects")
        checked = finite_values(events, f"{arm}.events")
        finite_values(report, f"{arm}.report")
        rows, selection = canonical_updates(report, events)
        reports[arm], updates[arm] = report, rows
        summaries[arm] = summarize_arm(report, rows, selection, checked, events)
        hashes[arm] = {"report.json": sha(report_bytes), "events.jsonl": sha(event_bytes)}
    require(reports["ordinary"].get("configuration") == reports["fbt"].get("configuration"), "Arm configurations differ")
    require(reports["ordinary"].get("source_fingerprint") == reports["fbt"].get("source_fingerprint"), "Arm source/data/runtime identity differs")
    require(reports["ordinary"]["counters"] == reports["fbt"]["counters"], "Completed arm exposures differ")
    require(reports["ordinary"].get("data_cursor") == reports["fbt"].get("data_cursor"), "Completed arm cursors differ")
    return {"schema": SCHEMA, "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha(Path(__file__).read_bytes()), "input_sha256": hashes, "arms": summaries,
            "first100_warmup": warmup_comparison(updates),
            "qualification": "Recorded optimizer observations only. Finite loss/norms do not demonstrate acceptable task quality or validate all tensor numerics. "
            "The pass CE average here is an unweighted descriptive average over updates, not held-out token-weighted NLL. "
            "Timing excludes checkpoint/evaluation work. Norm fractions do not determine parameter-update magnitudes."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve() not in {(args.runs / arm / name).resolve() for arm in ARMS for name in ("report.json", "events.jsonl")},
            "Output must not overwrite input evidence")
    summary = build_summary(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "output": str(args.output),
                      "exact_warmup_updates": summary["first100_warmup"]["exact_non_timing_update_matches"]}))


if __name__ == "__main__":
    main()
