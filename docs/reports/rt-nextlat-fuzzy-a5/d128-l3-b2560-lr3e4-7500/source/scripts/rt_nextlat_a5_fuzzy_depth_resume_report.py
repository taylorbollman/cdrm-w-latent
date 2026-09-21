#!/usr/bin/env python3
"""CPU-only depth comparison with exact checkpoint lineage after interruption.

Historical trainers and reporters remain unchanged. An interrupted parent's raw
history is preserved, but only the restored checkpoint's prefix contributes to
curves or timing. A continuation's duplicate boundary evaluation is omitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts import rt_nextlat_a5_fuzzy_depth_report as depth
from scripts import rt_nextlat_a5_fuzzy_lr_report as lr
from scripts import rt_nextlat_a5_fuzzy_embedding_compare as paired
from scripts import rt_nextlat_a5_fuzzy_report as saved

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rt-nextlat-a5-fuzzy-depth-resume-comparison-v1"
RECOVERY_QUALIFICATION = (
    "The three-layer run resumed from a verified full-state checkpoint after VM shutdown. "
    "Only checkpoint-committed ancestor updates are included; post-checkpoint work is excluded "
    "and replayed from restored model, Adam, RNG and data-stream state. Original files are preserved. "
    "Timing sums canonical committed updates once and excludes lost work and shutdown downtime. "
    "The same initialization, objective, data order and global LR schedule continue across recovery."
)
require, read_json, sha, json_sha = saved.require, saved.read_json, saved.sha, saved.json_sha
local_path, read_history_prefix = saved.local_path, saved.read_history_prefix
_evaluations, selected_evaluations = saved._evaluations, saved.selected_evaluations


def exact_tree(left, right):
    """Exact CPU comparison of a checkpoint packet, including numpy RNG state."""
    import numpy as np
    import torch
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_tree(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact_tree(a, b) for a, b in zip(left, right))
    return bool(left == right)


def boundary_state_proof(directory, parent, child_record, child_path, update):
    """Read actual checkpoint state; serialized bytes may differ after resaving."""
    import torch
    original = paired._packet(parent, update)
    child = torch.load(child_path, map_location="cpu", weights_only=False)
    require(original.keys() == child.keys(), "Restored checkpoint fields differ")
    field_equality = {key: exact_tree(value, child[key]) for key, value in original.items()}
    require(all(field_equality.values()), "Restored boundary full state differs: "
            + ", ".join(key for key, value in field_equality.items() if not value))
    lr.check_saved_lr(original, update, parent["report"]["contract"]["learning_rate_schedule"])
    return {"passed": True, "completed_updates": update, "entire_packet_exact": True,
            "field_equality": field_equality, "parent_checkpoint": parent["checkpoints"][update],
            "child_checkpoint": {**child_record, "verified_local_path": str(child_path)},
            "matching_examples_seen": original["examples_seen"],
            "matching_next_cursors": original["next_cursors"],
            "matching_order_chains": original["order_chains"]}


def _load_lineage(directory, *, through=None, ancestors=()):
    directory = local_path(directory)
    require(directory not in ancestors and len(ancestors) < 16, "Cyclic or excessively deep checkpoint lineage")
    report_path = directory / "report.json"
    report = read_json(report_path)
    terminal = through is None
    require(report.get("schema") == lr.TRAIN_SCHEMA and report.get("requested_endpoint") == 5000,
            "Expected the approved LR-training schema and 5000-update endpoint")
    require(report["contract"].get("schema") == lr.TRAIN_SCHEMA
            and report["contract"].get("mode") == "mixed", "Expected a mixed LR-training contract")
    require(report.get("status") in (("complete", "stopped") if terminal else ("running", "complete", "stopped")),
            "Require a completed or explicitly stopped training report; running is allowed only for a bound ancestor")
    require(report.get("confirmation_evaluated") is False and report.get("latent_rollout_evaluated") is False,
            "Final confirmation and autonomous latent rollout must remain unused")
    reported_endpoint, start = report.get("completed_updates"), report.get("start_update", 0)
    endpoint = reported_endpoint if terminal else through
    require(type(reported_endpoint) is int and type(start) is int and type(endpoint) is int
            and 0 <= start <= endpoint <= reported_endpoint <= 5000 and endpoint > 0, "Invalid actual endpoint or restored boundary")
    require(not terminal or report["status"] != "complete" or endpoint == report["requested_endpoint"],
            "Complete status must reach the requested endpoint")
    contract = report["contract"]
    config_path, identity_path = directory / "model-config.json", directory / "data-identity.json"
    identity = read_json(identity_path)
    require(read_json(config_path) == contract["model_config"]
            and sha(config_path) == contract["configuration_file_sha256"], "Saved model configuration differs")
    require(json_sha(identity) == contract["data_sha256"], "Saved dataset identity differs")
    require(report["initialization"] == contract["initialization"], "Initialization record differs")
    sources = read_json(directory / "source-manifest.json")
    require(json_sha(sources) == contract["source_sha256"], "Source manifest differs")
    for relative, digest in sources.items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Invalid frozen source path")
        require(sha(directory / "source" / relative) == digest, f"Frozen source differs: {relative}")

    parent_record, parent = report.get("parent_checkpoint"), None
    if start:
        require(isinstance(parent_record, dict), "Continuation is missing its parent checkpoint identity")
        parent_path = local_path(parent_record["path"])
        require(parent_path.parent.name == "checkpoints" and parent_path.name == f"step-{start:06d}.pt",
                "Parent checkpoint path does not identify the restored boundary")
        require(parent_path.is_file() and sha(parent_path) == parent_record["sha256"], "Parent checkpoint hash mismatch")
        parent = _load_lineage(parent_path.parent.parent, through=start, ancestors=(*ancestors, directory))
        require(parent["checkpoints"][start]["sha256"] == parent_record["sha256"],
                "Parent report records a different boundary checkpoint")
        require(parent["report"]["contract"] == contract and parent["report"]["initialization"] == report["initialization"]
                and parent["data_identity"] == identity and parent["sources"] == sources,
                "Exact continuation contract, source, initialization or data identity differs")
    else:
        require(parent_record is None, "Fresh training cannot claim a parent checkpoint")

    checkpoints = dict(parent["checkpoints"]) if parent else {}
    seen, duplicate_boundary, exact_boundary_audit, discarded_checkpoints = set(), None, None, 0
    for record in report["checkpoints"]:
        update = record["completed_updates"]
        require(type(update) is int and start <= update <= reported_endpoint and update not in seen,
                "Invalid or duplicate checkpoint update")
        seen.add(update)
        if update > endpoint:
            discarded_checkpoints += 1
            continue
        path = directory / "checkpoints" / f"step-{update:06d}.pt"
        require(path.is_file() and sha(path) == record["sha256"], "Checkpoint hash mismatch")
        require(path.stat().st_size == record["bytes"], "Checkpoint byte count mismatch")
        if update in checkpoints:
            require(update == start, "Only the restored boundary checkpoint can be duplicated")
            byte_identical = record["sha256"] == checkpoints[update]["sha256"] and record["bytes"] == checkpoints[update]["bytes"]
            exact_boundary_audit = boundary_state_proof(directory, parent, record, path, update)
            duplicate_boundary = {**record, "verified_local_path": str(path), "byte_identical": byte_identical,
                                  "exact_state_verified": byte_identical or exact_boundary_audit is not None}
        else:
            checkpoints[update] = {**record, "verified_local_path": str(path)}
    require(0 in checkpoints and endpoint in checkpoints and start in seen, "Initial, restored or endpoint checkpoint is missing")
    require(not start or duplicate_boundary is not None, "Continuation did not save its restored boundary")
    own_evaluations = [m for m in report["evaluations"] if start < m["update"] <= endpoint]
    if not start:
        own_evaluations = [m for m in report["evaluations"] if 0 <= m["update"] <= endpoint]
    evaluations = list(parent["evaluations"]) if parent else []
    boundary_only_stop = bool(start and terminal and endpoint == start)
    if boundary_only_stop:
        # A stop immediately after restoration has no new updates, but its fresh
        # full endpoint evaluation must supersede the ancestor's monitoring subset.
        checkpoints[start] = duplicate_boundary
        evaluations = [metric for metric in evaluations if metric["update"] != start]
        own_evaluations = [metric for metric in report["evaluations"] if metric["update"] == start]
    evaluations += _evaluations({**report, "evaluations": own_evaluations})
    keys = [(m["update"], m["task"], m.get("role", "dev"), m.get("rows", m.get("examples"))) for m in evaluations]
    require(len(set(keys)) == len(keys), "Duplicate evaluation task/role/update")
    require(all(type(k[0]) is int and 0 <= k[0] <= endpoint for k in keys), "Evaluation beyond actual endpoint")
    tasks = {m["task"] for m in evaluations}
    expected_tasks = {"a5", "fuzzy"}
    require(tasks == expected_tasks, "Evaluation tasks differ from the training mode")
    for metric in evaluations:
        if report.get("schema") != "rt-nextlat-fuzzy-training-v1":
            cp = metric.get("checkpoint", {})
            require(metric["update"] in checkpoints and cp.get("completed_updates") == metric["update"]
                    and cp.get("sha256") == checkpoints[metric["update"]]["sha256"],
                    "Evaluation is bound to a different checkpoint")
    if terminal:
        required = {("a5", "dev"), ("a5", "ood_dev")} if "a5" in tasks else set()
        if "fuzzy" in tasks:
            required.add(("fuzzy", "dev"))
        actual = {(task, role) for update, task, role, _ in keys if update == endpoint}
        require(required and actual == required, "Endpoint must contain all task evaluations at the same update")
        for metric in selected_evaluations(evaluations):
            if metric["update"] == endpoint:
                require(metric.get("rows", metric.get("examples")) == (102400 if metric["task"] == "a5" else 1280),
                        "Endpoint requires the full development evaluation")
    history_path = directory / "history.jsonl"
    own_history, history_audit = read_history_prefix(history_path, start=start, through=endpoint, allow_tail=not terminal)
    history = (list(parent["history"]) if parent else []) + own_history
    require([r["update"] for r in history] == list(range(1, endpoint + 1)), "Stitched history has a gap or duplicate")
    input_hashes = {"report": sha(report_path), "history": sha(history_path),
                    "model_config": sha(config_path), "data_identity": sha(identity_path)}
    stage = {"directory": str(directory), "status": report["status"], "start_update": start,
             "reported_completed_updates": reported_endpoint, "used_through_update": endpoint,
             "parent_checkpoint": parent_record, "input_hashes": input_hashes,
             "history": history_audit, "discarded_post_boundary_checkpoints": discarded_checkpoints,
             "discarded_post_boundary_evaluations": sum(m["update"] > endpoint for m in report["evaluations"]),
             "omitted_child_boundary_evaluations": sum(m["update"] == start for m in report["evaluations"])
                if start and not boundary_only_stop else 0,
             "endpoint_boundary_reevaluation_used": boundary_only_stop,
             "boundary_checkpoint_copy": duplicate_boundary, "boundary_state_audit": exact_boundary_audit,
             "training_wandb": report.get("wandb")}
    lineage = (list(parent["lineage"]) if parent else []) + [stage]
    return {"directory": str(directory), "report": report, "endpoint": endpoint,
            "checkpoints": checkpoints, "evaluations": evaluations, "history": history,
            "data_identity": identity, "sources": sources,
            "input_hashes": input_hashes, "lineage": lineage, "tasks": sorted(tasks)}



def load_run(directory):
    """Require terminal metrics and stitch only verified committed ancestors."""
    return _load_lineage(directory)


def compare(base, candidate):
    summary = depth.compare(base, candidate)
    summary["schema"] = SCHEMA
    summary["qualification"] += " " + RECOVERY_QUALIFICATION
    summary["recovery"] = {"stages": len(candidate["lineage"]),
        "restored_updates": [stage["start_update"] for stage in candidate["lineage"] if stage["start_update"]],
        "discarded_tail_bytes": sum(stage["history"]["discarded_tail_bytes"] for stage in candidate["lineage"]),
        "boundary_full_state_exact": all(stage["boundary_state_audit"]["passed"]
            for stage in candidate["lineage"] if stage["start_update"])}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-train", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    import torch
    require(Path("/.dockerenv").is_file() and not torch.cuda.is_initialized(),
            "Use the GPU-disabled project container")
    base, candidate = lr.read_fresh(args.baseline_train), load_run(args.train)
    summary = compare(base, candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    summary["figures"] = depth.figures(summary, args.output)
    (args.output / "report.md").write_text(depth.markdown(summary) + "\n## Recovery lineage\n\n"
        + RECOVERY_QUALIFICATION + "\n\n" + json.dumps(summary["recovery"], indent=2) + "\n")
    for name in ("scripts/rt_nextlat_a5_fuzzy_depth_resume_report.py", "scripts/rt_nextlat_a5_fuzzy_depth_report.py",
                 "scripts/rt_nextlat_a5_fuzzy_lr_report.py", "scripts/rt_nextlat_a5_fuzzy_report.py",
                 "scripts/rt_nextlat_a5_fuzzy_embedding_compare.py"):
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (args.output / "evidence.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.wandb:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", entity="taylorbollman", output_dir=args.output,
                                name="mixed-b2560-three-versus-two-rt-layers-lr3e4-resumed")
        try:
            tracker.start({"schema": SCHEMA, "qualification": summary["qualification"],
                           "schedule": summary["schedule"], "recovery": summary["recovery"],
                           "microbatch": summary["microbatch"], "parameter_difference": summary["parameter_difference"]})
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png")) for name in summary["figures"]})
            tracker.summary({"candidate_endpoint": candidate["endpoint"], "recovery": summary["recovery"],
                             "candidate_final": summary["arms"]["three_layers"]["final"],
                             "matched_training_seconds": summary["matched_training_seconds"]})
            tracker.finish(succeeded=True)
        except BaseException:
            tracker.finish(succeeded=False)
            raise
        summary["report_wandb"] = tracker.record
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "endpoint": candidate["endpoint"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
