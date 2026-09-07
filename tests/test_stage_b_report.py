"""Report tests use fabricated artifacts only; no real final test results."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[1] / "scripts/stage_b_report.py"
_spec = importlib.util.spec_from_file_location("stage_b_report", _path)
reporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reporter)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_bootstrap_pairs_whole_examples_and_weights_variable_answer_counts():
    seq = {"counts": np.array([1., 9.]), "correct": np.array([0., 9.]),
           "nll": np.array([2., 9.]), "sequences": np.array([0., 1.])}
    r3 = {"counts": np.array([1., 9.]), "correct": np.array([1., 0.]),
          "nll": np.array([1., 18.]), "sequences": np.array([1., 0.])}
    result = reporter.paired_bootstrap(seq, r3, seed=2, resamples=2000)
    assert result["metrics"]["answer_accuracy"]["difference"] == pytest.approx(-0.8)
    assert result["metrics"]["answer_ce"]["difference"] == pytest.approx(0.8)
    assert result["metrics"]["sequence_accuracy"]["difference"] == 0
    assert result == reporter.paired_bootstrap(seq, r3, seed=2, resamples=2000, chunk_size=17)
    identical = reporter.paired_bootstrap(seq, seq, resamples=100)
    for metric in identical["metrics"].values():
        assert metric["difference"] == 0 and metric["interval"] == [0., 0.]


def test_predictions_must_match_each_fixture_answer_and_position():
    labels = np.array([[-100, 2, 3], [-100, -100, 1]])
    rows = [{"example_index": 0, "answers": [
        {"position": 1, "prediction": 2, "gold": 2, "gold_log_probability": -0.1},
        {"position": 2, "prediction": 1, "gold": 3, "gold_log_probability": -2.0}]},
        {"example_index": 1, "answers": [
        {"position": 2, "prediction": 1, "gold": 1, "gold_log_probability": -0.2}]}]
    stats = reporter.prediction_statistics(rows, labels)
    assert reporter.aggregate(stats)["answer_accuracy"] == pytest.approx(2 / 3)
    assert reporter.aggregate(stats)["sequence_accuracy"] == 0.5
    for key, value in (("position", 0), ("gold", 0), ("gold_log_probability", float("nan"))):
        changed = copy.deepcopy(rows)
        changed[0]["answers"][0][key] = value
        with pytest.raises(ValueError):
            reporter.prediction_statistics(changed, labels)
    with pytest.raises(ValueError, match="order"):
        reporter.prediction_statistics(list(reversed(rows)), labels)


def _fabricate_pilot(tmp_path):
    root = tmp_path / "runtime"
    plan = {"seed": 0, "storage_prefix": "gs://example-bucket/test-only",
            "training": {"updates": 2, "precision": "fp32", "eval_interval": 2}, "fixtures": {"test_examples": 4},
            "tasks": {"state_tracking": {"primary_dev_conditions": ["iid"],
                "evaluation_conditions": {"iid": {"sequence_length": 3}}}}}
    plan_path = tmp_path / "plan.json"
    _write(plan_path, plan)
    plan_sha = reporter.digest_file(plan_path)
    inputs = np.arange(12, dtype=np.int64).reshape(4, 3)
    labels = np.array([[-100, -100, 1], [-100, 1, 2], [-100, -100, 0], [-100, 1, 2]])
    fixture = root / "fixtures/state_tracking/iid-test.npz"
    fixture.parent.mkdir(parents=True)
    np.savez_compressed(fixture, input_ids=inputs, labels=labels)
    _write(fixture.with_suffix(".metadata.json"), [{"id": index} for index in range(4)])
    digest = hashlib.sha256()
    for array in (inputs, labels):
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    fixture_sha = digest.hexdigest()
    data_hashes = [hashlib.sha256(f"batch{update}".encode()).hexdigest() for update in range(2)]
    stream_sha = hashlib.sha256("".join(data_hashes).encode()).hexdigest()
    fixture_manifest = {"plan_sha256": plan_sha,
        "audit": {"status": "passed", "training_stream_sha256": stream_sha},
        "conditions": {"iid": {"config": {"sequence_length": 3}, "dev": {"sha256": fixture_sha}, "test": {
            "path": str(fixture), "sha256": fixture_sha, "examples": 4,
            "baselines": {"chance_accuracy": 1 / 6, "restricted_last_two_accuracy": 0.25}}}}}
    fixture_manifest_path = root / "fixtures/state_tracking/manifest.json"
    _write(fixture_manifest_path, fixture_manifest)
    for topology in ("seq", "r3"):
        run = root / "runs" / f"SYN-state_tracking-{topology.upper()}-seed0"
        run.mkdir(parents=True)
        records = {}
        for role, update in (("init", 0), ("final", 2), ("bestdev", 2), ("latest", 2)):
            path = run / f"{role}.pt"
            path.write_bytes(f"Fabricated test checkpoint {topology}/{role}".encode())
            records[role] = {"path": str(path), "sha256": reporter.digest_file(path),
                             "bytes": path.stat().st_size, "completed_updates": update}
        identity = {"plan": plan, "plan_sha256": plan_sha, "task": "state_tracking",
            "topology": topology, "fixed_batch": False, "source_sha256": {"test-source": "abc"},
            "settings": plan["training"],
            "fixture_manifest_sha256": reporter.digest_file(fixture_manifest_path)}
        compiler_audit = {"required": topology == "r3", "counters": {"stats": {"unique_graphs": 1}},
                          "fail_on_recompile_limit_hit": True}
        manifest = {"status": "complete", "completed_updates": 2, "evidence_class": "SYN",
            "identity": identity, "identity_sha256": reporter.digest_json(identity),
            "training_counters": {"examples": 8, "input_tokens": 24, "supervised_targets": 12,
                                  "update_seconds": 0.2, "elapsed_seconds": 0.5},
            "best_development": {"update": 2, "score": 0.5}, "checkpoint_records": records,
            "parameter_count": 32, "compiler_audit": compiler_audit, "settings": plan["training"]}
        _write(run / "manifest.json", manifest)
        _jsonl(run / "learning-curve.jsonl", [{"update": index + 1, "data_sha256": data_hashes[index],
            "examples": 4, "input_tokens": 12, "supervised_targets": 6, "learning_rate": 0.001,
            "loss": 0.5, "answer_accuracy": 0.5, "sequence_accuracy": 0.5, "update_seconds": 0.1} for index in range(2)])
        for update in (0, 2):
            _write(run / "evaluations" / f"dev-u{update:04d}-metrics.json", {
                "update": update, "split": "dev", "primary_macro_answer_ce": 1.0 if update == 0 else 0.5,
                "conditions": {"iid": {"answer_accuracy": 0.5, "answer_ce": 1.0 if update == 0 else 0.5,
                                       "fixture_sha256": fixture_sha}}})
        rows = []
        for index, label in enumerate(labels):
            answers = []
            for position in np.flatnonzero(label != -100):
                gold = int(label[position])
                correct = topology == "r3" or index % 2 == 0
                answers.append({"position": int(position), "gold": gold,
                    "prediction": gold if correct else (gold + 1) % 3,
                    "gold_log_probability": -0.1 if correct else -2.0})
            rows.append({"example_index": index, "metadata": {"id": index}, "answers": answers})
        predictions = run / "evaluation-final" / "predictions.jsonl"
        _jsonl(predictions, rows)
        stats = reporter.prediction_statistics(rows, labels)
        _write(run / "evaluation-final/evaluation-manifest.json", {
            "split": "test", "update": 2, "checkpoint_role": "final",
            "checkpoint_sha256": records["final"]["sha256"],
            "identity_sha256": manifest["identity_sha256"],
            "compiler_audit": compiler_audit,
            "conditions": {"iid": {**reporter.aggregate(stats), "fixture_sha256": fixture_sha,
                                    "predictions": str(predictions)}}})
    return plan_path, root


def test_collects_complete_pair_and_writes_standalone_plots(tmp_path):
    plan, root = _fabricate_pilot(tmp_path)
    report = reporter.collect_report(plan, root, resamples=64)
    condition = report["tasks"]["state_tracking"]["conditions"]["iid"]
    assert condition["arms"]["seq"]["answer_accuracy"] == pytest.approx(1 / 3)
    assert condition["arms"]["r3"]["answer_accuracy"] == 1.0
    assert condition["paired_difference"]["metrics"]["answer_accuracy"]["difference"] == pytest.approx(2 / 3)
    output = tmp_path / "report"
    output.mkdir()
    plots = reporter.write_plots(report, output)
    assert len(plots) == 6 and all((output / name).stat().st_size > 1000 for name in plots)
    curves = report["tasks"]["state_tracking"]["runs"]["seq"]["development_curve"]
    assert [row["cumulative_update_seconds"] for row in curves] == [0.0, 0.2]
    text = reporter.render_markdown(report, output)
    assert "training-seed uncertainty" in text and "64 paired bootstrap" in text
    assert "Initial state plus last two operations" in text


@pytest.mark.parametrize("corruption", ["incomplete", "different_data", "different_source", "checkpoint",
                                       "prediction_order", "compiler_failure", "best_selection", "metadata_order"])
def test_refuses_unpaired_or_corrupt_results(tmp_path, corruption):
    plan, root = _fabricate_pilot(tmp_path)
    run = root / "runs/SYN-state_tracking-R3-seed0"
    if corruption in {"incomplete", "different_source", "compiler_failure", "best_selection"}:
        path = run / "manifest.json"
        manifest = reporter.read_json(path)
        if corruption == "incomplete":
            manifest["status"] = "paused"
        elif corruption == "compiler_failure":
            manifest["compiler_audit"]["counters"]["unimplemented"] = {"fallback": 1}
        elif corruption == "best_selection":
            manifest["best_development"]["score"] = 0.1
        else:
            manifest["identity"]["source_sha256"]["test-source"] = "different"
            manifest["identity_sha256"] = reporter.digest_json(manifest["identity"])
            evaluation_path = run / "evaluation-final/evaluation-manifest.json"
            evaluation = reporter.read_json(evaluation_path)
            evaluation["identity_sha256"] = manifest["identity_sha256"]
            _write(evaluation_path, evaluation)
        _write(path, manifest)
    elif corruption == "different_data":
        path = run / "learning-curve.jsonl"
        rows = reporter.read_jsonl(path)
        rows[0]["data_sha256"] = "different"
        _jsonl(path, rows)
    elif corruption == "checkpoint":
        (run / "final.pt").write_bytes(b"corrupted")
    elif corruption == "metadata_order":
        path = root / "fixtures/state_tracking/iid-test.metadata.json"
        metadata = reporter.read_json(path)
        _write(path, list(reversed(metadata)))
    else:
        path = run / "evaluation-final/predictions.jsonl"
        rows = reporter.read_jsonl(path)
        _jsonl(path, list(reversed(rows)))
    with pytest.raises(ValueError):
        reporter.collect_report(plan, root, resamples=16)
