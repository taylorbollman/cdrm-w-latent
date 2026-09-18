"""Saved-evidence reporting tests: no training, checkpoint deserialization or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import rt_nextlat_a5_fuzzy_curriculum_report as report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def chain(name, phase):
    return hashlib.sha256(f"{name}:{phase}".encode()).hexdigest()


def a5_metric(update, role, correct=24255):
    n, length = 102400, 12 if role == "dev" else 36
    accuracy = correct / n
    return {"task": "a5", "role": role, "update": update, "rows": n,
            "evaluated_rows": n, "length": length, "tokens": n * length,
            "ce": 1., "latent": .1, "token_accuracy": accuracy,
            "final_state_accuracy": accuracy, "whole_word_exact_match": accuracy,
            "whole_word_correct": correct, "whole_word_exact_count": correct,
            "isolated_state_accuracy": [accuracy] * length,
            "cumulative_prefix_exactness": [accuracy] * length,
            "per_position_ce": [1.] * length,
            "per_position_prefix_correct": [correct] * length,
            "per_position_state_correct": [correct] * length, "scope": "full"}


def fuzzy_metric(update):
    return {"task": "fuzzy", "role": "dev", "update": update,
            "examples": 1280, "evaluated_rows": 1280,
            "answer_tokens": 100, "answer_correct": 40, "answer_accuracy": .4,
            "answer_motifs": 100, "exact_answer_motifs": 38, "answer_motif_exact_match": .38,
            "exact_sequences": 10, "sequence_exact_match": 10 / 1280, "scope": "full"}


@pytest.fixture
def parent(tmp_path):
    directory = tmp_path / "parent"
    checkpoint = directory / "checkpoints/step-010000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"retained A5 10k model/Adam/RNG fixture")
    identity = {"a5": {"manifest_sha256": "5" * 64}}
    config = {"backbone": {"d_model": 128, "n_heads": 16, "mlp_hidden_size": 512}}
    contract = {"mode": "a5-only", "model_config": config,
                "initialization": {"canonical_sha256": "1" * 64},
                "optimizer": {"type": "AdamW", "lr": 1e-4}, "runtime": {"precision": "fp32"},
                "streams": {"a5": {"train_rows": 800000, "order_seed": 5432, "length": 12}}}
    record = {"path": str(checkpoint), "sha256": report.sha(checkpoint), "completed_updates": 10000}
    write_json(directory / "data-identity.json", identity)
    write_json(directory / "report.json", {"status": "complete", "contract": contract,
               "checkpoints": [record], "evaluations": [a5_metric(10000, role) for role in ("dev", "ood_dev")]})
    return directory, record, contract, identity


def fixture(directory, parent, *, mode="mixed-curriculum", endpoint=3):
    _, parent_checkpoint, parent_contract, identity = parent
    identity = {**identity, "fuzzy": {"preparation_manifest_sha256": "6" * 64}}
    source = directory / "source/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# shared frozen model source fixture\n")
    sources = {"model.py": report.sha(source)}
    for filename, contents in (("source-manifest.json", sources), ("data-identity.json", identity),
                               ("model-config.json", parent_contract["model_config"])):
        write_json(directory / filename, contents)
    contract = {**copy.deepcopy(parent_contract), "schema": "rt-nextlat-a5-fuzzy-curriculum-v1", "mode": mode,
                "batch_per_task": 128, "global_start_update": 10000, "ramp_updates": 3000,
                "initial_task_offsets": {"a5": 1280000, "fuzzy": 0},
                "initial_order_chains": {"a5": chain("a5", 0), "fuzzy": chain("fuzzy", 0)},
                "parent_checkpoint": {"sha256": parent_checkpoint["sha256"], "completed_updates": 10000},
                "parent_contract_sha256": report.json_sha(parent_contract),
                "configuration_file_sha256": report.sha(directory / "model-config.json"),
                "source_sha256": report.json_sha(sources), "data_sha256": report.json_sha(identity)}
    contract["streams"]["fuzzy"] = {"train_rows": 12800, "order_seed": 2026091604, "length": 400}
    checkpoints, evaluations, history = [], [], []
    for phase in range(endpoint + 1):
        global_update = 10000 + phase
        offsets = {"a5": 1280000 + 128 * phase, "fuzzy": 128 * phase if mode == "mixed-curriculum" else 0}
        path = directory / "checkpoints" / f"phase-{phase:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{mode} checkpoint {phase}".encode())
        checkpoint = {"path": str(path), "sha256": report.sha(path), "bytes": path.stat().st_size,
                      "completed_updates": global_update, "global_updates": global_update,
                      "phase_updates": phase, "task_offsets": offsets}
        checkpoints.append(checkpoint)
        metrics = [a5_metric(global_update, role) for role in ("dev", "ood_dev")] + [fuzzy_metric(global_update)]
        for metric in metrics:
            metric.update(global_update=global_update, phase_update=phase, checkpoint={k: checkpoint[k]
                          for k in ("path", "sha256", "completed_updates", "global_updates", "phase_updates")})
        evaluations.extend(metrics)
        if phase:
            weight = phase / 6000 if mode == "mixed-curriculum" else 0.
            tasks = ("a5", "fuzzy") if mode == "mixed-curriculum" else ("a5",)
            history.append({"update": global_update, "global_update": global_update, "phase_update": phase,
                            "task_offsets": offsets, "task_weights": {"a5": 1 - weight, "fuzzy": weight},
                            "order_chains": {"a5": chain("a5", phase),
                                             "fuzzy": chain("fuzzy", phase if mode == "mixed-curriculum" else 0)},
                            "tasks": {t: {"ce": 1., "latent": .1} for t in tasks}, "loss": 1.1})
    raw = {"schema": contract["schema"], "status": "complete", "contract": contract,
           "parent_training_checkpoint": parent_checkpoint, "parent_checkpoint": None,
           "initialization": contract["initialization"], "start_phase_update": 0, "start_update": 10000,
           "completed_updates": 10000 + endpoint, "global_updates": 10000 + endpoint,
           "phase_updates": endpoint, "task_offsets": offsets,
           "requested_endpoint": 10000 + endpoint, "requested_phase_endpoint": endpoint,
           "checkpoints": checkpoints, "evaluations": evaluations,
           "confirmation_evaluated": False, "latent_rollout_evaluated": False,
           "wandb": {"run_url": "https://wandb.ai/taylorbollman/rt-nextlat-fuzzy-a5/runs/fixture"}}
    write_json(directory / "report.json", raw)
    (directory / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    return raw


def test_module_cli_help_before_input_access():
    completed = subprocess.run([sys.executable, "-m", "scripts.rt_nextlat_a5_fuzzy_curriculum_report", "--help"],
                               cwd=report.ROOT, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert all(option in completed.stdout for option in ("--train", "--control", "--output"))


def test_counter_schedule_distinguishes_phase_from_global():
    assert report.expected_weights("mixed-curriculum", 0) == {"a5": 1, "fuzzy": 0}
    assert report.expected_weights("mixed-curriculum", 1)["fuzzy"] == pytest.approx(1 / 6000)
    assert report.expected_weights("mixed-curriculum", 1500) == {"a5": .75, "fuzzy": .25}
    assert report.expected_weights("mixed-curriculum", 3000) == {"a5": .5, "fuzzy": .5}
    assert report.expected_weights("mixed-curriculum", 10000) == {"a5": .5, "fuzzy": .5}
    assert report.expected_offsets("a5-control", 10000) == {"a5": 2560000, "fuzzy": 0}
    assert report.expected_offsets("mixed-curriculum", 10000) == {"a5": 2560000, "fuzzy": 1280000}


def test_actual_baseline_is_remeasured_and_differences_are_reported(tmp_path, parent):
    path = tmp_path / "run"
    raw = fixture(path, parent)
    for i, metric in enumerate(raw["evaluations"]):
        if metric["phase_update"] == 0 and metric["task"] == "a5" and metric["role"] == "ood_dev":
            raw["evaluations"][i] = {**metric, **a5_metric(10000, "ood_dev", correct=24254)}
    write_json(path / "report.json", raw)
    loaded = report.load_run(path)
    assert loaded["baseline_remeasurement"]["ood_dev"]["phase_zero"] == 24254 / 102400
    assert loaded["baseline_remeasurement"]["ood_dev"]["parent"] == 24255 / 102400
    assert loaded["baseline_remeasurement"]["ood_dev"]["delta_percentage_points"] < 0


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(status="running"), "completed or explicitly stopped"),
    (lambda r: r.update(phase_updates=2), "counters disagree"),
    (lambda r: r["contract"].update(ramp_updates=10000), "ramp duration"),
    (lambda r: r["parent_training_checkpoint"].update(sha256="9" * 64), "parent checkpoint differs"),
    (lambda r: r["evaluations"][-1]["checkpoint"].update(sha256="9" * 64), "different checkpoint"),
    (lambda r: r["evaluations"].pop(2), "Baseline and endpoint require both"),
    (lambda r: r["evaluations"][-1].update(exact_sequences=0), "disagrees with its counts"),
    (lambda r: r["evaluations"][0]["per_position_prefix_correct"].__setitem__(0, 1), "integer curve counts"),
])
def test_rejects_unbound_or_inconsistent_evidence(tmp_path, parent, mutation, match):
    path = tmp_path / "run"
    raw = fixture(path, parent)
    mutation(raw)
    write_json(path / "report.json", raw)
    with pytest.raises(ValueError, match=match):
        report.load_run(path)


@pytest.mark.parametrize("field,value,match", [
    ("global_update", 10000, "counters disagree"),
    ("phase_update", 0, "counters disagree"),
    ("task_offsets", {"a5": 128, "fuzzy": 128}, "exposure offsets"),
    ("task_weights", {"a5": .5, "fuzzy": .5}, "weight schedule"),
    ("loss", .1, "Weighted training objective"),
])
def test_rejects_wrong_history_axis_exposure_or_weighting(tmp_path, parent, field, value, match):
    path = tmp_path / "run"
    fixture(path, parent)
    rows = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
    rows[0][field] = value
    (path / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=match):
        report.load_run(path)


def test_pair_uses_same_parent_and_actual_equal_phase_observations(tmp_path, parent):
    a, b = tmp_path / "mixed", tmp_path / "control"
    fixture(a, parent, endpoint=3)
    fixture(b, parent, mode="a5-control", endpoint=2)
    primary, control = report.load_run(a), report.load_run(b)
    paired = report.compare_control(primary, control)
    assert paired["matched_a5_order_updates"] == 2
    assert {r["phase_update"] for r in paired["comparisons"]} == {0, 1, 2}
    assert {r["global_update"] for r in paired["comparisons"]} == {10000, 10001, 10002}
    assert primary["report"]["task_offsets"]["a5"] == 1280384
    control["history"][0]["order_chains"]["a5"] = "9" * 64
    with pytest.raises(ValueError, match="data order differs"):
        report.compare_control(primary, control)
    control["parent"]["sha256"] = "8" * 64
    with pytest.raises(ValueError, match="different parents"):
        report.compare_control(primary, control)


def test_control_does_not_advance_fuzzy_and_stops_are_supported(tmp_path, parent):
    path = tmp_path / "control"
    raw = fixture(path, parent, mode="a5-control")
    raw.update(status="stopped", requested_phase_endpoint=10000, requested_endpoint=20000)
    write_json(path / "report.json", raw)
    loaded = report.load_run(path)
    assert loaded["report"]["task_offsets"]["fuzzy"] == 0
    assert len(report.metrics_at(loaded, 0)) == len(report.metrics_at(loaded, 3)) == 3


def test_repeated_identical_eval_is_safe_but_conflicts_fail(tmp_path, parent):
    path = tmp_path / "run"
    raw = fixture(path, parent)
    repeated = {**raw["evaluations"][-1], "evaluation_seconds": 999}
    raw["evaluations"].append(repeated)
    write_json(path / "report.json", raw)
    assert report.load_run(path)["endpoint"] == 3
    repeated["answer_accuracy"] = 0
    write_json(path / "report.json", raw)
    with pytest.raises(ValueError, match="Conflicting repeated"):
        report.load_run(path)


def test_offline_report_renders_both_axes_and_retains_exact_evidence(tmp_path, parent):
    primary, control = tmp_path / "mixed", tmp_path / "control"
    fixture(primary, parent)
    fixture(control, parent, mode="a5-control")
    output = tmp_path / "report"
    summary = report.build_report(SimpleNamespace(train=primary, control=control, output=output))
    assert summary["phase_endpoint"] == 3 and summary["global_endpoint"] == 10003
    assert summary["additional_task_presentations"] == {"a5": 384, "fuzzy": 384}
    assert len(summary["figures"]) == 3
    for name in summary["figures"]:
        for suffix in ("png", "pdf"):
            artifact = output / f"{name}.{suffix}"
            assert artifact.stat().st_size > 1000
            assert report.sha(artifact) == summary["artifact_sha256"][artifact.name]
    text = (output / "report.md").read_text()
    assert "23.6865%" in text and "not new early learning" in text
    assert "global Adam update 10,003" in text
    assert "phase 3" in text
    assert report.read_json(output / "evidence.json") == report.read_json(output / "report.json")
    with pytest.raises(FileExistsError):
        report.build_report(SimpleNamespace(train=primary, control=control, output=output))


def test_control_only_report_does_not_require_a_mixed_run(tmp_path, parent, monkeypatch):
    path = tmp_path / "control"
    fixture(path, parent, mode="a5-control")
    monkeypatch.setattr(report, "make_plots", lambda *args: [])
    summary = report.build_report(SimpleNamespace(train=path, control=None, output=tmp_path / "report"))
    assert summary["control"] is None and summary["paired_control"] is None
    assert summary["additional_task_presentations"] == {"a5": 384, "fuzzy": 0}
