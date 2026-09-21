#!/usr/bin/env python3
"""Checkpoint-bound development reporting for the D128 A5 and joint pilots.

This consumes saved metrics and checkpoint bytes only; it never evaluates a
model. Control curves end at their actual observations, and joint endpoint
metrics always come from the same optimizer update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.rt_a5_nextlat_report import metric_rows


SCHEMA = "rt-nextlat-a5-fuzzy-report-v1"
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def checked_a5_metric(metric):
    """Validate scalar/curve identities before using an integer gate count."""
    curves = metric_rows(metric, "a5")
    count = metric.get("whole_word_correct")
    require(type(count) is int and count == curves[-1]["prefix_exact_count"],
            "A5 whole_word_correct must agree with the cumulative prefix curve")
    return curves


def a5_gate(evaluations, *, early_through_update=3000):
    """Report the first positive full L36 observation without selecting an endpoint."""
    eligible = []
    for metric in evaluations:
        if metric.get("task") != "a5" or metric.get("role") != "ood_dev":
            continue
        checked_a5_metric(metric)
        if metric["rows"] == 102400 and 0 < metric["update"] <= 10000:
            eligible.append(metric)
    eligible.sort(key=lambda metric: metric["update"])
    first = next((metric for metric in eligible if metric["whole_word_correct"] > 0), None)
    return {"rule": "Any positive integer whole-word count on a full 102,400-word L36 development evaluation by update 10,000",
            "passed": first is not None,
            "first_positive_update": None if first is None else first["update"],
            "first_positive_whole_word_correct": None if first is None else first["whole_word_correct"],
            "first_positive_rows": None if first is None else first["rows"],
            "first_positive_accuracy": None if first is None else first["whole_word_exact_match"],
            "early_through_update": early_through_update,
            "positive_within_first_few_thousand": first is not None and first["update"] <= early_through_update,
            "full_evaluation_updates": [metric["update"] for metric in eligible],
            "interpretation": "Observed finite-sample development success; the endpoint is reported independently, and the first positive observation is not the exact onset of learning."}


def _evaluations(report):
    """Normalize only the older Fuzzy-only report's implicit task field."""
    old_fuzzy = report.get("schema") == "rt-nextlat-fuzzy-training-v1"
    result = []
    for original in report["evaluations"]:
        metric = dict(original)
        if old_fuzzy:
            metric["task"] = "fuzzy"
        require(metric.get("task") in ("a5", "fuzzy"), "Unknown evaluation task")
        if metric["task"] == "a5":
            checked_a5_metric(metric)
        else:
            n, count = metric["answer_tokens"], metric["answer_correct"]
            require(type(n) is int and n > 0 and type(count) is int and 0 <= count <= n
                    and math.isclose(metric["answer_accuracy"], count / n, abs_tol=1e-12),
                    "Fuzzy answer counts and accuracy disagree")
        result.append(metric)
    return result


def local_path(path):
    path = Path(path)
    container_root = Path("/workspace/cdrm-w-latent")
    if path.is_relative_to(container_root):
        path = ROOT / path.relative_to(container_root)
    elif not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def boundary_state_proof(directory, parent, child_record, child_path, update):
    """Accept different serialization only with an exact full-state audit."""
    candidates = [directory / "recovery-boundary-state-audit.json",
                  directory.parent / "recovery-boundary-state-audit.json"]
    proof_path = next((path for path in candidates if path.is_file()), None)
    require(proof_path is not None, "Resaved boundary differs; exact full-state audit receipt is required")
    proof = read_json(proof_path)
    require(proof.get("schema") == "rt-nextlat-a5-fuzzy-recovery-boundary-state-audit-v1"
            and proof.get("passed") is True and proof.get("status") == "passed"
            and proof.get("completed_updates") == update and proof.get("entire_packet_exact") is True
            and proof.get("differences") == [], "Boundary full-state audit did not establish exact equality")
    fields = ("model", "optimizer", "rng", "contract", "initialization", "order_chains", "next_cursors",
              "completed_updates", "examples_seen", "optimizer_parameter_names", "schema", "finite_state")
    require(all(proof.get("field_equality", {}).get(key) is True for key in fields),
            "Boundary audit is missing an exact state field")
    for role, record, path in (("parent", parent["checkpoints"][update],
                               Path(parent["checkpoints"][update]["verified_local_path"])),
                              ("child", child_record, child_path)):
        bound = proof.get("checkpoints", {}).get(role, {})
        require(local_path(bound.get("path", "")) == path.resolve()
                and bound.get("sha256") == record["sha256"] and bound.get("bytes") == record["bytes"],
                "Boundary audit identifies different checkpoint bytes")
    contract = parent["report"]["contract"]
    expected_seen = {task: update * contract["batch_per_task"] for task in parent["tasks"]}
    expected_cursors = {task: {"absolute_example_offset": offset,
        "epoch": offset // contract["streams"][task]["train_rows"],
        "position": offset % contract["streams"][task]["train_rows"]} for task, offset in expected_seen.items()}
    require(proof.get("matching_examples_seen") == expected_seen
            and proof.get("matching_next_cursors") == expected_cursors
            and proof.get("matching_order_chains") == parent["history"][-1]["order_chains"],
            "Boundary audit counters or order chains disagree with the committed history")
    return {"path": str(proof_path), "sha256": sha(proof_path), "receipt": proof}


def read_history_prefix(path, *, start, through, allow_tail):
    """Use only committed updates; hash and describe any uncommitted raw tail."""
    history, prefix = [], bytearray()
    with Path(path).open("rb") as stream:
        for expected in range(start + 1, through + 1):
            while True:
                line = stream.readline()
                require(bool(line), "History ends before the restored checkpoint boundary")
                prefix.extend(line)
                if line.strip():
                    break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid committed history at update {expected}") from error
            require(row.get("update") == expected, "History does not contain exactly the completed optimizer updates")
            history.append(row)
        tail = stream.read()
    require(allow_tail or not tail.strip(), "Uncommitted history follows a terminal report")
    valid_tail, first_invalid = [], None
    offset = len(prefix)
    for line in tail.splitlines(keepends=True):
        if not line.strip():
            offset += len(line)
            continue
        try:
            row = json.loads(line)
            require(type(row.get("update")) is int and row["update"] > through,
                    "Tail contains a non-post-boundary update")
        except (ValueError, UnicodeDecodeError, AttributeError):
            first_invalid = offset
            break
        valid_tail.append(row["update"])
        offset += len(line)
    return history, {"start_update": start, "used_through_update": through,
        "used_records": len(history), "used_prefix_bytes": len(prefix),
        "used_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "raw_file_sha256": sha(path), "discarded_tail_bytes": len(tail),
        "discarded_tail_sha256": hashlib.sha256(tail).hexdigest(),
        "discarded_valid_records_before_first_invalid": len(valid_tail),
        "discarded_valid_update_range": [valid_tail[0], valid_tail[-1]] if valid_tail else None,
        "first_invalid_tail_byte_offset": first_invalid,
        "tail_policy": "Excluded from this exact-checkpoint continuation; original bytes are preserved in place."}


def _load_lineage(directory, *, through=None, ancestors=()):
    directory = local_path(directory)
    require(directory not in ancestors and len(ancestors) < 16, "Cyclic or excessively deep checkpoint lineage")
    report_path = directory / "report.json"
    report = read_json(report_path)
    terminal = through is None
    require(report.get("status") in (("complete", "stopped") if terminal else ("running", "complete", "stopped")),
            "Require a completed or explicitly stopped training report; running is allowed only for a bound ancestor")
    require(report.get("confirmation_evaluated") is False and report.get("latent_rollout_evaluated") is False,
            "Final confirmation and autonomous latent rollout must remain unused")
    reported_endpoint, start = report.get("completed_updates"), report.get("start_update", 0)
    endpoint = reported_endpoint if terminal else through
    require(type(reported_endpoint) is int and type(start) is int and type(endpoint) is int
            and 0 <= start < endpoint <= reported_endpoint, "Invalid actual endpoint or restored boundary")
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
            if not byte_identical:
                exact_boundary_audit = boundary_state_proof(directory, parent, record, path, update)
            duplicate_boundary = {**record, "verified_local_path": str(path), "byte_identical": byte_identical,
                                  "exact_state_verified": byte_identical or exact_boundary_audit is not None}
        else:
            checkpoints[update] = {**record, "verified_local_path": str(path)}
    require(endpoint in checkpoints, "Actual endpoint checkpoint is missing")
    own_evaluations = [m for m in report["evaluations"] if start < m["update"] <= endpoint]
    if not start:
        own_evaluations = [m for m in report["evaluations"] if 0 <= m["update"] <= endpoint]
    evaluations = list(parent["evaluations"]) if parent else []
    evaluations += _evaluations({**report, "evaluations": own_evaluations})
    keys = [(m["update"], m["task"], m.get("role", "dev"), m.get("rows", m.get("examples"))) for m in evaluations]
    require(len(set(keys)) == len(keys), "Duplicate evaluation task/role/update")
    require(all(type(k[0]) is int and 0 <= k[0] <= endpoint for k in keys), "Evaluation beyond actual endpoint")
    tasks = {m["task"] for m in evaluations}
    expected_tasks = {"rt-nextlat-fuzzy-training-v1": {"fuzzy"}}.get(report.get("schema"))
    if expected_tasks is None:
        require(report.get("schema") == "rt-nextlat-a5-fuzzy-training-v1", "Unknown training report schema")
        expected_tasks = {"a5-only": {"a5"}, "mixed": {"a5", "fuzzy"}}.get(contract.get("mode"))
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
             "omitted_child_boundary_evaluations": sum(m["update"] == start for m in report["evaluations"]) if start else 0,
             "boundary_checkpoint_copy": duplicate_boundary, "boundary_state_audit": exact_boundary_audit,
             "training_wandb": report.get("wandb")}
    lineage = (list(parent["lineage"]) if parent else []) + [stage]
    return {"directory": str(directory), "report": report, "endpoint": endpoint,
            "checkpoints": checkpoints, "evaluations": evaluations, "history": history,
            "data_identity": identity, "sources": sources,
            "input_hashes": input_hashes, "lineage": lineage, "tasks": sorted(tasks)}


def load_run(directory):
    """Load a terminal run, following only verified checkpoint ancestors."""
    return _load_lineage(directory)


def selected_evaluations(evaluations):
    """Prefer a full evaluation when a positive subset triggered confirmation."""
    selected = {}
    for metric in evaluations:
        key = (metric["update"], metric["task"], metric.get("role", "dev"))
        size = metric.get("rows", metric.get("examples"))
        if key not in selected or size > selected[key].get("rows", selected[key].get("examples")):
            selected[key] = metric
    return [selected[key] for key in sorted(selected)]


def endpoint_metrics(run):
    return {f"{m['task']}/{m.get('role', 'dev')}": m for m in selected_evaluations(run["evaluations"])
            if m["update"] == run["endpoint"]}


def task_identity(run, task):
    contract = run["report"]["contract"]
    if contract.get("schema") == "rt-nextlat-fuzzy-training-v1":
        require(task == "fuzzy", "Legacy control only contains Fuzzy data")
        stream = {key: contract[key] for key in ("train_rows", "length", "order_seed")}
        identity = {**run["data_identity"],
                    "preparation_manifest_sha256": contract["preparation_manifest_sha256"]}
        batch = contract["batch_size"]
    else:
        stream, identity = contract["streams"][task], run["data_identity"][task]
        batch = contract["batch_per_task"]
    return {"stream": stream, "data": identity, "batch_size": batch}


def order_chain(row, task):
    return row["order_chains"][task] if "order_chains" in row else row["order_chain"]


def compare_control(run, control, task):
    require(task in run["tasks"] and control["tasks"] == [task], "Control task differs")
    left, right = run["report"]["contract"], control["report"]["contract"]
    for key in ("model_config", "initialization", "optimizer", "runtime"):
        require(left[key] == right[key], f"Matched control {key} differs")
    require(task_identity(run, task) == task_identity(control, task),
            "Matched control task data, batch size or order seed differs")
    core = ("cdrm/rt_nextlat_tasks.py", "scripts/rt_a5_window.py",
            "scripts/rt_a5_nextlat.py", "scripts/rt_a5_nextlat_variant.py",
            "recurrent-transformer/olmo/model.py")
    for name in core:
        require(name in run["sources"] and name in control["sources"]
                and run["sources"][name] == control["sources"][name],
                f"Matched control model source differs: {name}")
    matched_history = min(run["endpoint"], control["endpoint"])
    for left_row, right_row in zip(run["history"], control["history"]):
        require(order_chain(left_row, task) == order_chain(right_row, task),
                f"Matched control {task} data order differs at update {left_row['update']}")
    left_metrics = {(m["update"], m.get("role", "dev")): m
                    for m in selected_evaluations(run["evaluations"]) if m["task"] == task}
    right_metrics = {(m["update"], m.get("role", "dev")): m
                     for m in selected_evaluations(control["evaluations"]) if m["task"] == task}
    comparisons = []
    for update in sorted(run["checkpoints"].keys() & control["checkpoints"].keys()):
        roles = ("dev", "ood_dev") if task == "a5" else ("dev",)
        for role in roles:
            a, b = left_metrics.get((update, role)), right_metrics.get((update, role))
            if a is None or b is None:
                continue
            if a.get("rows", a.get("examples")) != b.get("rows", b.get("examples")):
                continue
            comparisons.append({"update": update, "role": role,
                                "examples_per_task": update * task_identity(run, task)["batch_size"],
                                "run": a, "control": b,
                                "run_checkpoint": run["checkpoints"][update],
                                "control_checkpoint": control["checkpoints"][update]})
    return {"task": task, "matched": True, "matched_order_updates": matched_history,
            "control_endpoint": control["endpoint"], "comparisons": comparisons,
            "qualification": "Matched initialization, model, task data/order and per-task examples. Joint training includes the other task and halves each task's loss coefficient; total compute and total example count differ."}


def make_plots(run, controls, baselines, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = []
    label = "Joint A5 + Fuzzy" if run["tasks"] == ["a5", "fuzzy"] else "A5 only"
    colors = {"primary": "#D65B35", "a5": "#3574B2", "fuzzy": "#3574B2"}

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=160)
        plt.close(fig)
        figures.append(name)

    def points(current, task, role="dev"):
        return [m for m in selected_evaluations(current["evaluations"])
                if m["task"] == task and m.get("role", "dev") == role]

    def trajectory(axis, metrics, key, name, color):
        axis.plot([m["update"] for m in metrics], [100*m[key] for m in metrics], label=name, color=color)
        for full, marker in ((True, "o"), (False, "x")):
            selected = [m for m in metrics if (m.get("rows", m.get("examples")) >= 102400) == full]
            axis.scatter([m["update"] for m in selected], [100*m[key] for m in selected],
                         marker=marker, s=24, color=color)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for row, (role, length) in enumerate((("dev", 12), ("ood_dev", 36))):
        for axis, key, title in zip(axes[row], ("token_accuracy", "final_state_accuracy", "whole_word_exact_match"),
                                    ("Mean token accuracy", "Final-state accuracy", "Whole-word exactness")):
            trajectory(axis, points(run, "a5", role), key, label, colors["primary"])
            if "a5" in controls:
                trajectory(axis, points(controls["a5"], "a5", role), key, "A5-only control", colors["a5"])
            axis.set(title=f"A5 length {length}: {title}", xlabel="Optimizer updates", ylabel="Accuracy (%)", ylim=(-2, 102))
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle("A5 development — one seed; backbone evaluation\nCircles: 102,400 words; crosses: smaller monitoring subset")
    save(fig, "a5-learning")

    final = endpoint_metrics(run)["a5/ood_dev"]
    prefix_series = [(label + f" at {run['endpoint']:,}", final, colors["primary"])]
    if "a5" in controls:
        other = next((m for m in points(controls["a5"], "a5", "ood_dev")
                      if m["update"] == run["endpoint"] and m["rows"] == final["rows"]), None)
        if other is not None:
            prefix_series.append((f"A5-only at {run['endpoint']:,}", other, colors["a5"]))
    for name, limits in (("a5-prefix-full", (1, 36)), ("a5-prefix-boundary", (10, 18))):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for series_label, metric, color in prefix_series:
            state = metric["isolated_state_accuracy"]
            values = (metric["cumulative_prefix_exactness"], state,
                      [sum(state[:n])/n for n in range(1, len(state)+1)])
            for axis, curve in zip(axes, values):
                axis.plot(range(1, len(curve)+1), [100*v for v in curve], label=series_label, color=color)
        for axis, title in zip(axes, ("E(t): cumulative prefix exactness", "A(t): state at position t", "M(t): mean accuracy through t")):
            axis.set(title=title, xlim=limits, ylim=(-2, 102), xlabel="Prefix of the same length-36 words", ylabel="Accuracy (%)")
            axis.axvline(12, color=".5", ls=":")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        fig.suptitle(f"Identical full and boundary source rows — {final['rows']:,} development words")
        save(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    windows = [run["history"][i:i+50] for i in range(0, len(run["history"]), 50)]
    for task in run["tasks"]:
        for axis, key in zip(axes, ("ce", "latent")):
            axis.plot([w[-1]["update"] for w in windows],
                      [np.mean([r["tasks"][task][key] for r in w]) for w in windows], label=task.upper())
    for axis, title in zip(axes, ("Per-task training CE", "Per-task training NextLat SmoothL1")):
        axis.set(title=title, xlabel="Optimizer updates", ylabel="Task mean (50-update windows)")
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle("Task losses averaged separately; Fuzzy CE is dense, A5 CE scores every state")
    save(fig, "task-losses")

    if "fuzzy" in run["tasks"]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        fuzzy_series = [(run, label, colors["primary"])]
        if "fuzzy" in controls:
            fuzzy_series.append((controls["fuzzy"], "Fuzzy-only (stopped)", colors["fuzzy"]))
        for axis, key, title in zip(axes, ("answer_accuracy", "answer_motif_exact_match", "sequence_exact_match"),
                                   ("Answer tokens", "Individual answer motifs", "All answers in a sequence")):
            for current, name, color in fuzzy_series:
                metrics = points(current, "fuzzy")
                axis.plot([m["update"] for m in metrics], [100*m[key] for m in metrics],
                          marker="o", ms=3, label=name, color=color)
            if key == "answer_accuracy" and baselines:
                axis.axhline(100*baselines["query_ignoring_answer_prefix"]["answer_accuracy"],
                             color=".5", ls="--", label="Query-ignoring prefix shortcut")
            axis.set(title=title, xlabel="Optimizer updates", ylabel="Accuracy / teacher-forced exactness (%)", ylim=(-2, 102))
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        fig.suptitle("Fuzzy Recall development — controls end at their actual observations")
        save(fig, "fuzzy-learning")
    return figures


def build_report(args):
    run = load_run(args.train)
    require("a5" in run["tasks"], "The primary run must include A5")
    controls = {}
    for task, directory in (("a5", args.a5_control), ("fuzzy", args.fuzzy_control)):
        if directory is not None:
            controls[task] = load_run(directory)
            require(controls[task]["tasks"] == [task], "Expected a single-task control")
    comparisons = {task: compare_control(run, control, task)
                   for task, control in controls.items() if task in run["tasks"]}
    baselines, data_manifest_sha = None, None
    if args.fuzzy_data is not None:
        data_manifest_sha = sha(args.fuzzy_data / "manifest.json")
        baselines = read_json(args.fuzzy_data / "manifest.json")["splits"]["dev"]["baselines"]
        for current in [run, *controls.values()]:
            if "fuzzy" in current["tasks"]:
                require(task_identity(current, "fuzzy")["data"]["preparation_manifest_sha256"] == data_manifest_sha,
                        "Fuzzy baseline dataset differs from the evaluated dataset")
    if "fuzzy" in run["tasks"]:
        require(baselines is not None, "Mixed reporting requires --fuzzy-data for native baselines")
    endpoint = run["endpoint"]
    finals = endpoint_metrics(run)
    summary = {"schema": SCHEMA, "status": "complete", "mode": run["report"]["contract"]["mode"],
               "endpoint": endpoint, "requested_endpoint": run["report"]["requested_endpoint"],
               "training_status": run["report"]["status"], "checkpoint": run["checkpoints"][endpoint],
               "final": finals, "a5_positive_gate": a5_gate(run["evaluations"]),
               "evaluations": run["evaluations"], "model": run["report"]["contract"]["model_config"],
               "initialization": run["report"]["initialization"],
               "parameters": run["report"]["parameter_count"], "contract": run["report"]["contract"],
               "input_hashes": run["input_hashes"], "lineage": run["lineage"], "fuzzy_baselines": baselines,
               "fuzzy_preparation_manifest_sha256": data_manifest_sha,
               "training_wandb": run["report"]["wandb"],
               "matched_controls": comparisons, "controls": {},
               "confirmation_evaluated": False, "latent_rollout_evaluated": False,
               "qualification": "Single initialization and reused development data; no final confirmation. Joint endpoint metrics belong to one checkpoint. First-positive A5 development evaluation is a prospective continuation gate, not the selected endpoint. Only common retained checkpoints with equal task exposure and evaluation rows are matched control comparisons; unequal terminal budgets are descriptive.",
               "source_code": str(Path(__file__).resolve()), "source_sha256": sha(Path(__file__))}
    for task, control in controls.items():
        summary["controls"][task] = {"endpoint": control["endpoint"],
            "checkpoint": control["checkpoints"][control["endpoint"]],
            "input_hashes": control["input_hashes"], "lineage": control["lineage"], "final": endpoint_metrics(control),
            "training_wandb": control["report"]["wandb"],
            "role": "matched single-task control" if task in comparisons else "prior single-task reference; no primary-run task comparison"}
    args.output.mkdir(parents=True, exist_ok=False)
    summary["figures"] = make_plots(run, controls, baselines, args.output)
    write_markdown(summary, args.output)
    # This complete evidence record is immutable when a W&B artifact is created.
    (args.output / "evidence.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def write_markdown(summary, output):
    endpoint, mode = summary["endpoint"], summary["mode"]
    batch = summary["contract"]["batch_per_task"]
    model = summary["model"]["backbone"]
    lines = ["# D128 A5 and Fuzzy development pilot", "",
             f"**{mode}, actual endpoint {endpoint:,} updates.** Training status: {summary['training_status']}.", "",
             f"Two RT layers; first window two, second full recurrent. Width {model['d_model']}, "
             f"{model['n_heads']} heads, FFN {model['mlp_hidden_size']}; ALiBi, Mitchell, FP32 eager, NextLat retained. "
             f"{summary['parameters']:,} parameters; no embedding bypass or autonomous latent rollout.", "",
             f"{batch:,} examples per active task per optimizer update; {endpoint*batch:,} examples per task at the endpoint. "
             "Losses are means within each task; mixed training weights the two task objectives equally.", "",
             "| Same-checkpoint endpoint metric | Accuracy |", "| --- | ---: |"]
    for role, length in (("dev", 12), ("ood_dev", 36)):
        final = summary["final"][f"a5/{role}"]
        for label, key in (("mean token", "token_accuracy"), ("final state", "final_state_accuracy"),
                           ("whole word", "whole_word_exact_match")):
            suffix = f" ({final['whole_word_correct']:,} / {final['rows']:,})" if key == "whole_word_exact_match" else ""
            lines.append(f"| A5 L{length} {label} | {100*final[key]:.4f}%{suffix} |")
    if "fuzzy/dev" in summary["final"]:
        final = summary["final"]["fuzzy/dev"]
        for label, key in (("answer tokens", "answer_accuracy"), ("answer-motif exact", "answer_motif_exact_match"),
                           ("all-answers sequence exact", "sequence_exact_match"),
                           ("first value token", "first_value_token_accuracy"), ("terminal probe", "terminal_probe_accuracy")):
            lines.append(f"| Fuzzy {label} | {100*final[key]:.4f}% |")
    gate = summary["a5_positive_gate"]
    lines += ["", "## A5 continuation gate", ""]
    if gate["passed"]:
        lines += [f"The first observed positive full L36 evaluation was at **{gate['first_positive_update']:,} updates**: "
                  f"{gate['first_positive_whole_word_correct']:,} / {gate['first_positive_rows']:,} whole words "
                  f"({100*gate['first_positive_accuracy']:.6f}%). "
                  f"Positive within the first {gate['early_through_update']:,} updates: "
                  f"{'yes' if gate['positive_within_first_few_thousand'] else 'no'}. "
                  "This is the prospective continuation gate; the endpoint table remains independent."]
    else:
        lines += ["No full 102,400-word L36 evaluation observed a positive whole-word count by the completed budget, "
                  "within the 10k gate horizon. This pilot does not establish length-36 state tracking."]
    lines += ["", "## Controls and qualifications", ""]
    for task, control in summary["controls"].items():
        lines += [f"{task.upper()} control actual endpoint: **{control['endpoint']:,} updates**. {control['role']}."]
        matched = summary["matched_controls"].get(task)
        if matched:
            updates = sorted({point["update"] for point in matched["comparisons"]})
            lines += ["Shared retained comparison updates: " + (", ".join(f"{u:,}" for u in updates) or "none") + ". "
                      f"Matching per-task data order verified through {matched['matched_order_updates']:,} updates."]
            lines += ["", "| Update | Task metric | Joint/current | Single-task control |", "| ---: | --- | ---: | ---: |"]
            for point in matched["comparisons"]:
                key = "whole_word_exact_match" if task == "a5" else "answer_accuracy"
                name = f"A5 {point['role']} whole word" if task == "a5" else "Fuzzy answer tokens"
                lines += [f"| {point['update']:,} | {name} | {100*point['run'][key]:.4f}% | {100*point['control'][key]:.4f}% |"]
            lines += [""]
    if summary["fuzzy_baselines"]:
        baseline = summary["fuzzy_baselines"]
        lines += [f"Fuzzy query-ignoring answer-prefix shortcut: "
                  f"{100*baseline['query_ignoring_answer_prefix']['answer_accuracy']:.4f}%. "
                  "Dense Fuzzy training CE and masked answer evaluation CE have different scopes. "
                  "Answer exactness is teacher-forced. Causal lookup coverage is not a universal accuracy ceiling.", ""]
    lines += [summary["qualification"], "",
              f"Checkpoint: `step-{endpoint:06d}.pt`; SHA256 `{summary['checkpoint']['sha256']}`.", "",
              f"[Training W&B]({summary['training_wandb'].get('run_url')})", ""]
    if len(summary["lineage"]) > 1:
        lines += ["## Exact checkpoint recovery", "",
                  "History, evaluations and control comparisons use the committed parent prefix followed by the resumed updates. "
                  "Post-boundary parent observations are excluded; their original bytes remain preserved. "
                  "Matching state at the restored boundary does not establish bitwise equivalence of later updates across GPU instances.", "",
                  "| Stage | Updates used | Raw history SHA256 | Excluded tail bytes |",
                  "| --- | --- | --- | ---: |"]
        for stage in summary["lineage"]:
            lines += [f"| {Path(stage['directory']).name} | {stage['start_update']+1:,}–{stage['used_through_update']:,} "
                      f"| `{stage['history']['raw_file_sha256']}` | {stage['history']['discarded_tail_bytes']:,} |"]
        lines += [""]
    for name in summary["figures"]:
        lines += [f"![{name.replace('-', ' ')}]({name}.png)", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    args = parser().parse_args()
    summary = build_report(args)
    if args.wandb:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker = OnlineTracker(project="rt-nextlat-fuzzy-a5", output_dir=args.output,
                                name=f"d128-{summary['mode']}-report-{summary['endpoint']}")
        try:
            tracker.start({"scope": "Saved development evidence only", "mode": summary["mode"],
                           "endpoint": summary["endpoint"], "checkpoint_sha256": summary["checkpoint"]["sha256"]})
            tracker.log({f"report/{name}": wandb.Image(str(args.output / f"{name}.png")) for name in summary["figures"]})
            tracker.summary({"endpoint": summary["endpoint"], "a5_positive_gate": summary["a5_positive_gate"],
                             "endpoint_metrics": summary["final"], "training_run": summary["training_wandb"].get("run_url"),
                             "checkpoint_sha256": summary["checkpoint"]["sha256"]})
            artifact_name = f"d128-{summary['mode']}-development-{tracker.record['run_id']}"
            artifact = wandb.Artifact(artifact_name, type="development-report",
                                     metadata={"endpoint": summary["endpoint"], "evidence_sha256": sha(args.output / "evidence.json")})
            for name in ("evidence.json", "report.md", *[f"{stem}.{ext}" for stem in summary["figures"] for ext in ("png", "pdf")]):
                artifact.add_file(str(args.output / name), name=name)
            tracker._call("artifact logging", lambda: tracker._run.log_artifact(artifact))
            tracker.finish(succeeded=True)
            summary["report_wandb"] = tracker.record
            summary["wandb_artifact"] = {"name": artifact_name, "type": "development-report",
                                         "project": "taylorbollman/rt-nextlat-fuzzy-a5", "evidence_sha256": sha(args.output / "evidence.json")}
        except BaseException:
            tracker.finish(succeeded=False)
            raise
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "endpoint": summary["endpoint"],
                      "a5_positive_gate": summary["a5_positive_gate"], "output": str(args.output)}))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train", type=Path, required=True)
    result.add_argument("--a5-control", type=Path)
    result.add_argument("--fuzzy-control", type=Path)
    result.add_argument("--fuzzy-data", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--wandb", action="store_true")
    return result


if __name__ == "__main__":
    main()
