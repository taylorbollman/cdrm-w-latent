#!/usr/bin/env python3
"""Validate paired Stage B artifacts and summarize completed final evaluations.

This script reads saved predictions; it performs no model or GPU evaluation.
Bootstrap units are whole examples, paired between architectures. Intervals
describe held-out-example uncertainty for one training seed, not seed variation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 937


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    container_root = Path("/workspace/cdrm-w-latent")
    if path.is_relative_to(container_root):
        return PROJECT_ROOT / path.relative_to(container_root)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixture_arrays(entry: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    path = resolve_path(entry["path"])
    with np.load(path, allow_pickle=False) as data:
        inputs, labels = data["input_ids"], data["labels"]
    digest = hashlib.sha256()
    for array in (inputs, labels):
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"Fixture array hash mismatch: {path}")
    if len(labels) != entry["examples"] or inputs.shape != labels.shape:
        raise ValueError("Fixture shape/example count mismatch")
    metadata = read_json(path.with_suffix(".metadata.json"))
    if len(metadata) != len(labels):
        raise ValueError("Fixture metadata/example count mismatch")
    return inputs, labels, metadata


def prediction_statistics(rows: list[dict], labels: np.ndarray) -> dict[str, np.ndarray]:
    """Validate every answer against saved labels and reduce only within examples."""
    if len(rows) != len(labels):
        raise ValueError("Prediction and fixture example counts differ")
    correct, counts, nll, sequences = [], [], [], []
    for index, (row, gold_row) in enumerate(zip(rows, labels)):
        if row["example_index"] != index:
            raise ValueError("Prediction example order differs from the fixed fixture")
        positions = np.flatnonzero(gold_row != -100).tolist()
        if not positions or [answer["position"] for answer in row["answers"]] != positions:
            raise ValueError("Scored positions differ from the fixed fixture")
        answers = row["answers"]
        if [answer["gold"] for answer in answers] != gold_row[positions].tolist():
            raise ValueError("Gold answer tokens differ from the fixed fixture")
        if any(not isinstance(answer["prediction"], int) or answer["prediction"] < 0 for answer in answers):
            raise ValueError("Invalid predicted token ID")
        log_probs = [float(answer["gold_log_probability"]) for answer in answers]
        if any(not math.isfinite(value) or value > 1e-6 for value in log_probs):
            raise ValueError("Invalid gold log probability")
        matches = [answer["prediction"] == answer["gold"] for answer in answers]
        correct.append(sum(matches))
        counts.append(len(matches))
        nll.append(-sum(log_probs))
        sequences.append(all(matches))
    return {"correct": np.asarray(correct, dtype=np.float64),
            "counts": np.asarray(counts, dtype=np.float64),
            "nll": np.asarray(nll, dtype=np.float64),
            "sequences": np.asarray(sequences, dtype=np.float64)}


def aggregate(statistics: dict[str, np.ndarray]) -> dict[str, float | int]:
    targets = float(statistics["counts"].sum())
    if targets <= 0:
        raise ValueError("Cannot report a condition without scored answers")
    return {"answer_accuracy": float(statistics["correct"].sum() / targets),
            "sequence_accuracy": float(statistics["sequences"].mean()),
            "answer_ce": float(statistics["nll"].sum() / targets),
            "examples": len(statistics["counts"]), "supervised_targets": int(targets)}


def paired_bootstrap(seq: dict[str, np.ndarray], r3: dict[str, np.ndarray], *,
                     seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES,
                     chunk_size: int = 64) -> dict:
    """Paired ratio estimators correctly retain variable answers per example."""
    if resamples < 1 or chunk_size < 1:
        raise ValueError("Positive bootstrap resample and chunk counts required")
    if not np.array_equal(seq["counts"], r3["counts"]) or not len(seq["counts"]):
        raise ValueError("Paired bootstrap requires identical nonempty example target counts")
    length = len(seq["counts"])
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in ("answer_accuracy", "sequence_accuracy", "answer_ce")}
    for begin in range(0, resamples, chunk_size):
        indices = rng.integers(length, size=(min(chunk_size, resamples - begin), length))
        denominator = seq["counts"][indices].sum(axis=1)
        draws["answer_accuracy"].append(((r3["correct"] - seq["correct"])[indices]).sum(axis=1) / denominator)
        draws["sequence_accuracy"].append(((r3["sequences"] - seq["sequences"])[indices]).mean(axis=1))
        draws["answer_ce"].append(((r3["nll"] - seq["nll"])[indices]).sum(axis=1) / denominator)
    first, second = aggregate(seq), aggregate(r3)
    return {"method": "paired percentile bootstrap over whole examples",
            "direction": "R3 minus SEQ", "seed": seed, "resamples": resamples,
            "confidence": 0.95, "examples": length,
            "scope": "Held-out-example variation conditional on one trained pair; excludes training-seed uncertainty",
            "metrics": {key: {"difference": second[key] - first[key],
                "interval": np.quantile(np.concatenate(values), [0.025, 0.975]).tolist()}
                for key, values in draws.items()}}


def compare_summary(statistics: dict, stored: dict) -> None:
    actual = aggregate(statistics)
    for key in ("answer_accuracy", "sequence_accuracy", "answer_ce"):
        # Stored CE sums float32 minibatch reductions; saved per-answer log
        # probabilities are another float32 reduction of the same predictions.
        tolerance = 2e-6 if key == "answer_ce" else 1e-12
        if not math.isclose(actual[key], stored[key], rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(f"Saved aggregate {key} disagrees with per-example predictions")
    for key in ("examples", "supervised_targets"):
        if actual[key] != stored[key]:
            raise ValueError(f"Saved aggregate {key} disagrees with per-example predictions")


def artifact_location(path: Path, run_root: Path, storage_prefix: str) -> dict:
    path = path.resolve()
    local = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    cloud = storage_prefix.rstrip("/") + "/" + str(path.relative_to(run_root)) if path.is_relative_to(run_root) else None
    return {"local_path": local, "gcs_uri": cloud}


def validate_compiler_audit(audit: dict, topology: str) -> None:
    required = topology == "r3"
    if audit["required"] != required:
        raise ValueError("Compiler audit requirement disagrees with topology")
    if required:
        counters = audit.get("counters", {})
        failures = {group: values for group in ("unimplemented", "graph_break")
                    if any(values := list(counters.get(group, {}).values()))}
        if failures:
            raise ValueError(f"Compiled R3 artifact contains fallback or graph-break evidence: {failures}")
        if counters.get("stats", {}).get("unique_graphs", 0) < 1:
            raise ValueError("Compiled R3 artifact has no observed compiled graphs")
        if not audit.get("fail_on_recompile_limit_hit", False):
            raise ValueError("Compiled R3 artifact did not fail closed on cache-limit fallback")


def validate_development_curve(run_dir: Path, plan: dict, task: str,
                               manifest: dict, fixtures_manifest: dict) -> list[dict]:
    horizon = plan["training"]["updates"]
    primary = plan["tasks"][task]["primary_dev_conditions"]
    expected_conditions = set(plan["tasks"][task]["evaluation_conditions"])
    curves = []
    for path in sorted((run_dir / "evaluations").glob("dev-u*-metrics.json")):
        row = read_json(path)
        if row["split"] != "dev" or not 0 <= row["update"] <= horizon:
            raise ValueError("Development curve contains an invalid split or update")
        if set(row["conditions"]) != expected_conditions:
            raise ValueError("Development evaluation conditions differ from the frozen plan")
        for name, values in row["conditions"].items():
            if values["fixture_sha256"] != fixtures_manifest["conditions"][name]["dev"]["sha256"]:
                raise ValueError("Development curve used a different fixed fixture")
            if not math.isfinite(values["answer_ce"]) or values["answer_ce"] < 0:
                raise ValueError("Development answer CE must be finite and nonnegative")
            if not 0 <= values["answer_accuracy"] <= 1:
                raise ValueError("Development answer accuracy outside [0,1]")
        score = float(np.mean([row["conditions"][name]["answer_ce"] for name in primary]))
        if not math.isclose(score, row["primary_macro_answer_ce"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Stored primary-development score differs from its declared condition mean")
        curves.append({"update": row["update"], "primary_macro_answer_ce": score,
            "primary_macro_answer_accuracy": float(np.mean([row["conditions"][name]["answer_accuracy"] for name in primary]))})
    updates = [row["update"] for row in curves]
    expected = {0, horizon, *range(plan["training"]["eval_interval"], horizon + 1, plan["training"]["eval_interval"])}
    if len(set(updates)) != len(updates) or not expected.issubset(updates):
        raise ValueError("Development curve lacks required evaluations or repeats an update")
    selected = min(curves, key=lambda row: row["primary_macro_answer_ce"])
    recorded = manifest["best_development"]
    if selected["update"] != recorded["update"] or not math.isclose(
        selected["primary_macro_answer_ce"], recorded["score"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Retained best-development selection disagrees with the complete curve")
    return curves


def validate_run(plan: dict, plan_sha: str, task: str, topology: str, run_dir: Path,
                 fixtures_manifest: dict, fixtures_sha: str, run_root: Path) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    identity = manifest["identity"]
    horizon = plan["training"]["updates"]
    if manifest["status"] != "complete" or manifest["completed_updates"] != horizon:
        raise ValueError(f"Incomplete {task}/{topology} trajectory")
    if manifest["evidence_class"] != "SYN" or identity["fixed_batch"]:
        raise ValueError("NUM/OPS artifacts cannot enter the completed SYN pilot report")
    if identity["plan"] != plan or identity["plan_sha256"] != plan_sha:
        raise ValueError("Trajectory used a different frozen plan")
    if identity["task"] != task or identity["topology"] != topology:
        raise ValueError("Task/topology identity mismatch")
    if manifest["identity_sha256"] != digest_json(identity):
        raise ValueError("Trajectory identity digest mismatch")
    if identity["fixture_manifest_sha256"] != fixtures_sha:
        raise ValueError("Trajectory used a different fixture/audited-stream manifest")
    if manifest["settings"] != identity["settings"] or manifest["settings"] != plan["training"]:
        raise ValueError("Actual optimizer/schedule settings differ from the frozen plan")
    validate_compiler_audit(manifest["compiler_audit"], topology)
    history = read_jsonl(run_dir / "learning-curve.jsonl")
    if [row["update"] for row in history] != list(range(1, horizon + 1)):
        raise ValueError("Learning curve must contain each planned update exactly once")
    stream = hashlib.sha256()
    for row in history:
        stream.update(row["data_sha256"].encode())
    if stream.hexdigest() != fixtures_manifest["audit"]["training_stream_sha256"]:
        raise ValueError("Training trajectory did not consume the frozen audited batch stream")
    for key in ("examples", "input_tokens", "supervised_targets"):
        if sum(row[key] for row in history) != manifest["training_counters"][key]:
            raise ValueError(f"Training curve and manifest disagree on {key}")
    checkpoint_records = {}
    for role in ("init", "final", "bestdev", "latest"):
        record = manifest["checkpoint_records"][role]
        path = resolve_path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest_file(path) != record["sha256"]:
            raise ValueError(f"Retained {role} checkpoint missing or corrupted: {path}")
        expected = 0 if role == "init" else horizon if role in {"final", "latest"} else manifest["best_development"]["update"]
        if record["completed_updates"] != expected:
            raise ValueError(f"Retained {role} checkpoint has the wrong update count")
        checkpoint_records[role] = {**record, **artifact_location(path, run_root, plan["storage_prefix"])}
    evaluation = read_json(run_dir / "evaluation-final/evaluation-manifest.json")
    if evaluation["split"] != "test" or evaluation["update"] != horizon or evaluation["checkpoint_role"] != "final":
        raise ValueError("Report requires the frozen-horizon final checkpoint's separate test evaluation")
    if evaluation["checkpoint_sha256"] != checkpoint_records["final"]["sha256"]:
        raise ValueError("Final evaluation checkpoint digest does not match the retained final checkpoint")
    if evaluation["identity_sha256"] != manifest["identity_sha256"]:
        raise ValueError("Final evaluation identity differs from the training trajectory")
    validate_compiler_audit(evaluation["compiler_audit"], topology)
    if set(evaluation["conditions"]) != set(plan["tasks"][task]["evaluation_conditions"]):
        raise ValueError("Final evaluation has missing or unexpected conditions")
    dev_curves = validate_development_curve(run_dir, plan, task, manifest, fixtures_manifest)
    cumulative_seconds = np.concatenate(([0.0], np.cumsum([row["update_seconds"] for row in history])))
    for row in dev_curves:
        row["cumulative_update_seconds"] = float(cumulative_seconds[row["update"]])
    return {"manifest": manifest, "evaluation": evaluation, "history": history,
            "development_curve": dev_curves, "checkpoint_records": checkpoint_records,
            "run_artifacts": artifact_location(run_dir, run_root, plan["storage_prefix"]),
            "training_stream_sha256": stream.hexdigest()}


def collect_report(plan_path: Path, run_root: Path, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """All required artifacts are validated before a report is written."""
    plan, plan_sha = read_json(plan_path), digest_file(plan_path)
    run_root = run_root.resolve()
    report = {"schema": "stage-b-paired-report-v1", "plan_sha256": plan_sha,
        "plan": plan, "seed_count": 1, "training_seed": plan["seed"],
        "interpretation": "Single-seed exploratory synthetic pilot; no automatic architectural win claim",
        "bootstrap": {"resamples": resamples, "seed": BOOTSTRAP_SEED, "confidence": 0.95,
            "unit": "whole example, paired across SEQ/R3",
            "limitation": "Conditional test-example uncertainty; excludes training-seed variation and does not adjust for multiple conditions"},
        "tasks": {}}
    for task, task_plan in plan["tasks"].items():
        fixture_path = run_root / "fixtures" / task / "manifest.json"
        fixture_manifest = read_json(fixture_path)
        if fixture_manifest["plan_sha256"] != plan_sha or fixture_manifest["audit"]["status"] != "passed":
            raise ValueError("Fixtures require a successful audit under the same frozen plan")
        arms = {}
        for topology in ("seq", "r3"):
            run_dir = run_root / "runs" / f"SYN-{task}-{topology.upper()}-seed{plan['seed']}"
            arms[topology] = validate_run(plan, plan_sha, task, topology, run_dir,
                fixture_manifest, digest_file(fixture_path), run_root)
        identities = [{key: value for key, value in arm["manifest"]["identity"].items() if key != "topology"}
                      for arm in arms.values()]
        if identities[0] != identities[1]:
            raise ValueError("Paired arms differ in identity beyond topology (including source hashes or settings)")
        if arms["seq"]["manifest"]["parameter_count"] != arms["r3"]["manifest"]["parameter_count"]:
            raise ValueError("SEQ and R3 parameter counts differ")
        for seq_row, r3_row in zip(arms["seq"]["history"], arms["r3"]["history"]):
            for key in ("data_sha256", "learning_rate", "examples", "input_tokens", "supervised_targets"):
                if seq_row[key] != r3_row[key]:
                    raise ValueError(f"Paired training differs at update {seq_row['update']} in {key}")
        task_report = {"primary_dev_conditions": task_plan["primary_dev_conditions"], "conditions": {},
            "training_pairing": {"status": "identical_batches_and_lr_at_every_update",
                "updates": plan["training"]["updates"], "training_stream_sha256": arms["seq"]["training_stream_sha256"]},
            "fixture_audit": fixture_manifest["audit"], "runs": {}}
        for topology, arm in arms.items():
            task_report["runs"][topology] = {
                "identity_sha256": arm["manifest"]["identity_sha256"],
                "source_sha256": arm["manifest"]["identity"]["source_sha256"],
                "parameter_count": arm["manifest"]["parameter_count"],
                "training_counters": arm["manifest"]["training_counters"],
                "best_development": arm["manifest"]["best_development"],
                "checkpoint_records": arm["checkpoint_records"],
                "development_curve": arm["development_curve"],
                "run_artifacts": arm["run_artifacts"],
                "compiler_audit": arm["manifest"]["compiler_audit"],
                "gradient_accumulation": arm["manifest"].get("gradient_accumulation"),
                "training_curve": [{key: row[key] for key in ("update", "loss", "answer_accuracy", "sequence_accuracy")}
                                   for row in arm["history"]]}
        for condition in task_plan["evaluation_conditions"]:
            if fixture_manifest["conditions"][condition]["config"] != task_plan["evaluation_conditions"][condition]:
                raise ValueError("Final fixture condition differs from the frozen plan")
            entry = fixture_manifest["conditions"][condition]["test"]
            _, labels, fixture_metadata = fixture_arrays(entry)
            if len(labels) != plan["fixtures"]["test_examples"]:
                raise ValueError("Final test count differs from the frozen plan")
            rows, stats = {}, {}
            result = {"config": fixture_manifest["conditions"][condition]["config"],
                      "fixture_sha256": entry["sha256"], "examples": len(labels),
                      "baselines": entry["baselines"], "arms": {}}
            for topology in ("seq", "r3"):
                metrics = arms[topology]["evaluation"]["conditions"][condition]
                if metrics["fixture_sha256"] != entry["sha256"]:
                    raise ValueError("Evaluated fixture hash differs from the frozen fixture")
                path = resolve_path(metrics["predictions"])
                rows[topology] = read_jsonl(path)
                if [row.get("metadata") for row in rows[topology]] != fixture_metadata:
                    raise ValueError("Prediction metadata/example ordering differs from the fixed fixture")
                stats[topology] = prediction_statistics(rows[topology], labels)
                compare_summary(stats[topology], metrics)
                result["arms"][topology] = {**aggregate(stats[topology]),
                    "stored_aggregate": metrics,
                    "prediction_artifact": {**artifact_location(path, run_root, plan["storage_prefix"]),
                                             "sha256": digest_file(path)}}
            for seq_row, r3_row in zip(rows["seq"], rows["r3"]):
                if seq_row.get("metadata") != r3_row.get("metadata"):
                    raise ValueError("Paired prediction metadata/example ordering differs")
            condition_seed = BOOTSTRAP_SEED + int.from_bytes(hashlib.sha256(f"{task}/{condition}".encode()).digest()[:4], "little")
            result["paired_difference"] = paired_bootstrap(stats["seq"], stats["r3"], seed=condition_seed, resamples=resamples)
            task_report["conditions"][condition] = result
        report["tasks"][task] = task_report
    return report


def write_plots(report: dict, output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"seq": "#2463a0", "r3": "#d16a2b"}
    tasks = list(report["tasks"])
    artifacts = []
    for x_key, label, stem in (
        ("update", "Completed training updates", "development-learning-curves"),
        ("cumulative_update_seconds", "Cumulative recorded update seconds", "development-update-time-curves"),
    ):
        figure, axes = plt.subplots(len(tasks), 2, figsize=(11, 3.5 * len(tasks)), squeeze=False)
        for row, task in enumerate(tasks):
            for topology in ("seq", "r3"):
                values = report["tasks"][task]["runs"][topology]["development_curve"]
                for column, metric in enumerate(("primary_macro_answer_ce", "primary_macro_answer_accuracy")):
                    axes[row, column].plot([value[x_key] for value in values], [value[metric] for value in values],
                        marker="o", markersize=3, color=colors[topology], label=topology.upper())
                    axes[row, column].set_xlabel(label)
                    axes[row, column].set_ylabel("Answer CE" if column == 0 else "Answer accuracy")
                    axes[row, column].set_title(task.replace("_", " ") + " · primary development conditions")
                    axes[row, column].grid(alpha=0.2)
            axes[row, 0].legend()
            axes[row, 1].set_ylim(0, 1.02)
        if x_key == "cumulative_update_seconds":
            figure.suptitle("Includes first-update overhead; excludes evaluation/startup. Descriptive curves, not a matched-time experiment.", fontsize=10)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            name = f"{stem}.{suffix}"
            figure.savefig(output / name, dpi=180, bbox_inches="tight")
            artifacts.append(name)
        plt.close(figure)
    figure, axes = plt.subplots(len(tasks), 1, figsize=(11, 3.5 * len(tasks)), squeeze=False)
    for row, task in enumerate(tasks):
        conditions = report["tasks"][task]["conditions"]
        positions = np.arange(len(conditions))
        plotted_accuracies = []
        for topology, offset in (("seq", -0.18), ("r3", 0.18)):
            accuracies = [value["arms"][topology]["answer_accuracy"] for value in conditions.values()]
            plotted_accuracies.extend(accuracies)
            axes[row, 0].bar(positions + offset,
                accuracies,
                width=0.35, label=topology.upper(), color=colors[topology])
        chance = [value["baselines"]["chance_accuracy"] for value in conditions.values()]
        shortcut_key, shortcut_label = (
            ("restricted_last_two_accuracy", "Initial state + last two operations") if task == "state_tracking" else
            ("observed_values_uniform_expected_accuracy", "Uniform observed value")
        )
        shortcut = [value["baselines"][shortcut_key] for value in conditions.values()]
        axes[row, 0].scatter(positions, chance, marker="x", s=38, color="#666666",
                             linewidths=1.5, label="Uniform legal answer (chance)", zorder=4)
        axes[row, 0].scatter(positions, shortcut, marker="D", s=35, facecolors="white",
                             edgecolors="#151515", linewidths=1.4, label=shortcut_label, zorder=5)
        plotted_accuracies.extend(chance)
        plotted_accuracies.extend(shortcut)
        axes[row, 0].set_xticks(positions, list(conditions), rotation=20, ha="right")
        axes[row, 0].set_ylim(0, min(1.02, max(0.25, 1.1 * max(plotted_accuracies))))
        axes[row, 0].set_ylabel("Final test answer accuracy")
        axes[row, 0].set_title(task.replace("_", " "))
        axes[row, 0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
        axes[row, 0].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        name = f"final-condition-accuracy.{suffix}"
        figure.savefig(output / name, dpi=180, bbox_inches="tight")
        artifacts.append(name)
    plt.close(figure)
    return artifacts


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _interval(value: dict, *, percentage: bool) -> str:
    scale = 100 if percentage else 1
    unit = " pp" if percentage else ""
    return f"{scale * value['difference']:+.3f}{unit} [{scale * value['interval'][0]:+.3f}, {scale * value['interval'][1]:+.3f}]"


def render_markdown(report: dict, output: Path) -> str:
    plan = report["plan"]
    lines = ["# Stage B paired synthetic pilot", "",
        f"This is a single-seed exploratory comparison (training seed {plan['seed']}) of SEQ and R3 after "
        f"{plan['training']['updates']:,} updates per arm. Each final condition uses {plan['fixtures']['test_examples']:,} "
        "fixed held-out examples. Corresponding initialization and every training batch are paired; "
        "R3 uses rho=1 from initialization.", "",
        f"Intervals use {report['bootstrap']['resamples']:,} paired bootstrap resamples of whole examples. They describe test-example variation "
        "conditional on this trained pair, exclude training-seed uncertainty, and have no adjustment for the "
        "multiple reported conditions. No automatic architectural win is inferred. Answer accuracy and CE use "
        "only designated answers; sequence accuracy requires every answer in an example to be correct.", "",
        "When neither arm has any all-correct example, the empirical paired bootstrap produces a degenerate "
        "[0, 0] interval for the sequence-accuracy difference. This reflects zero observed successes and does "
        "not establish equality of population sequence accuracies.", "",
        f"Training uses {plan['training']['precision']} with the frozen plan's model and optimizer settings. "
        "These are small symbolic models, not pretrained language-model checkpoints. The state composition "
        "holdout tests an unseen ordered operation pattern; finite compositions can still implement a function "
        "seen in training.", "",
        "[Machine-readable report and complete artifact hashes](report.json) · "
        "[Development curves (PDF)](development-learning-curves.pdf) · "
        "[Development versus recorded update time (PDF)](development-update-time-curves.pdf) · "
        "[Final accuracy figure (PDF)](final-condition-accuracy.pdf)", "",
        "![Primary development learning curves](development-learning-curves.png)", "",
        "The time-axis curves below use cumulative recorded update seconds through each development evaluation. "
        "They include first-update overhead, exclude evaluation and startup, and are descriptive curves rather than "
        "an end-to-end matched-time experiment. No matched-time extrapolation is performed.", "",
        "![Primary development curves versus recorded update time](development-update-time-curves.png)", ""]
    for task, result in report["tasks"].items():
        lines += [f"## {task.replace('_', ' ')}", "",
            "| Condition | SEQ answer | R3 answer | SEQ sequence | R3 sequence | SEQ CE | R3 CE |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for condition, values in result["conditions"].items():
            seq, r3 = values["arms"]["seq"], values["arms"]["r3"]
            lines.append(f"| {condition} | {_percent(seq['answer_accuracy'])} | {_percent(r3['answer_accuracy'])} | "
                         f"{_percent(seq['sequence_accuracy'])} | {_percent(r3['sequence_accuracy'])} | "
                         f"{seq['answer_ce']:.4f} | {r3['answer_ce']:.4f} |")
        lines += ["", "Differences below are **R3 minus SEQ**, with paired 95% intervals. Accuracy differences "
                  "are percentage points; lower CE is better.", "",
                  "| Condition | Answer difference | Sequence difference | CE difference |",
                  "|---|---:|---:|---:|"]
        for condition, values in result["conditions"].items():
            metrics = values["paired_difference"]["metrics"]
            lines.append(f"| {condition} | {_interval(metrics['answer_accuracy'], percentage=True)} | "
                         f"{_interval(metrics['sequence_accuracy'], percentage=True)} | "
                         f"{_interval(metrics['answer_ce'], percentage=False)} |")
        lines += ["", "Measured or analytically defined shortcut baselines on these same test fixtures:", "",
                  "| Condition | Baseline | Answer accuracy |", "|---|---|---:|"]
        baseline_keys = {"chance_accuracy": "Uniform legal answer",
            "observed_values_uniform_expected_accuracy": "Uniform observed value",
            "last_record_value_accuracy": "Last record value",
            "restricted_history_expected_accuracy_uniform_fallback": "Restricted history, uniform fallback",
            "last_assignment_accuracy": "Initial state only",
            "last_record_accuracy": "Initial state plus last operation",
            "restricted_last_two_accuracy": "Initial state plus last two operations"}
        for condition, values in result["conditions"].items():
            for key, label in baseline_keys.items():
                if key in values["baselines"]:
                    lines.append(f"| {condition} | {label} | {_percent(values['baselines'][key])} |")
        lines += ["", f"Best-development selection uses the unweighted mean answer CE over "
                  f"{', '.join(result['primary_dev_conditions'])}. Final-checkpoint results above remain primary.", ""]
    lines += ["## Training cost and retained checkpoints", "",
        "Recorded elapsed time includes development evaluation and checkpoint work after startup; it is not "
        "pure GPU-kernel time and excludes model construction/fixture loading before the training timer. "
        "Update time includes transfer, answer loss, backward, clipping, and AdamW. Generator audits, calibration, "
        "separate final evaluation, and earlier NUM/OPS work are outside these counters.", "",
        "| Task | Arm | Examples | Input tokens | Targets | Update seconds | Recorded elapsed seconds | Best dev update |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task, result in report["tasks"].items():
        for topology, run in result["runs"].items():
            counts = run["training_counters"]
            lines.append(f"| {task} | {topology.upper()} | {counts['examples']:,} | {counts['input_tokens']:,} | "
                         f"{counts['supervised_targets']:,} | {counts['update_seconds']:.2f} | "
                         f"{counts['elapsed_seconds']:.2f} | {run['best_development']['update']} |")
    lines += ["", "Every pair was checked for matching source/plan identity, all per-update data hashes, learning rates, "
        "and token counts. Final prediction order, gold labels, positions, fixture hashes, and aggregates were checked "
        "against the retained arrays. Init, final, best-development, and latest checkpoint bytes were verified against "
        "their recorded SHA-256 digests. Full local paths, GCS URIs, hashes, and raw prediction references are in report.json.", "",
        "Compiled R3 recovery from update 0 is disabled because that cold-start resume path did not pass bitwise "
        "validation; fresh initialization and validated midpoint recovery remain available. The pilot uses one "
        "microbatch per global batch. A separate R3 accumulation diagnostic passed gradient and optimizer-tensor "
        "bounds but failed a parameter-equivalence bound for one embedding element. That unused-path failure "
        "is retained, and its cause is not established.", "",
        "| Task | Arm | Retained checkpoints |", "|---|---|---|"]
    for task, result in report["tasks"].items():
        for topology, run in result["runs"].items():
            links = []
            for role, artifact in run["checkpoint_records"].items():
                target = resolve_path(artifact["local_path"])
                links.append(f"[{role}]({Path(os.path.relpath(target, output)).as_posix()})")
            lines.append(f"| {task} | {topology.upper()} | {' · '.join(links)} |")
    lines += ["", f"Artifact storage prefix: `{plan['storage_prefix']}`. "
        "These URI mappings describe the requested archive location; this report does not independently verify cloud upload completion.", "",
        "The final accuracy panels start at zero and adapt their upper limits to the displayed values. "
        "Markers show uniform legal-answer chance and a task-specific shortcut: uniform over observed values "
        "for retrieval, or the true initial state followed by only the last two operations for state tracking. "
        "These shortcuts use the same final test fixtures as the model bars.", "",
        "![Final answer accuracy by condition with chance and shortcut markers](final-condition-accuracy.png)", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output directory to preserve prior reports")
    report = collect_report(args.plan, args.run_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["plot_artifacts"] = write_plots(report, args.output_dir)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "results.md").write_text(render_markdown(report, args.output_dir.resolve()))
    print(json.dumps({"status": "complete", "report": str(args.output_dir / "results.md"),
                      "tasks": list(report["tasks"])}))


if __name__ == "__main__":
    main()
