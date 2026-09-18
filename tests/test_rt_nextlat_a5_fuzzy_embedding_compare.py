"""CPU-only retained-evidence checks for the four-arm mixed comparison."""
import copy
import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest
import torch

from scripts import rt_nextlat_a5_fuzzy_embedding_compare as compare
from scripts import rt_nextlat_a5_fuzzy_report as saved


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def a5_metric(update, role, checkpoint, *, rows=102400, correct=100):
    length = 12 if role == "dev" else 36
    accuracy = correct / rows
    return {"task": "a5", "role": role, "update": update, "rows": rows, "length": length,
            "evaluated_rows": rows, "tokens": rows*length, "ce": 1., "latent": .1,
            "token_accuracy": accuracy, "final_state_accuracy": accuracy,
            "whole_word_exact_match": accuracy, "whole_word_correct": correct,
            "isolated_state_accuracy": [accuracy]*length, "cumulative_prefix_exactness": [accuracy]*length,
            "per_position_ce": [1.]*length, "checkpoint": checkpoint}


def fixture(directory, *, variant=None, endpoint=3, parent=None):
    config = {"backbone": {"d_model": 128, "n_heads": 16, "n_layers": 2}}
    initialization = {"model_parameter_sha256": "shared", "predictor_sha256": "predictor", "seed": 1234}
    if variant:
        route = ({"variant": variant, "layer_index": 1, "coefficient": .01, "projection_seed": 1237}
                 if variant != "head" else {"variant": "head", "layer_index": 1, "head_index": 15, "enabled": True})
        config["embedding_injection"] = route
        initialization = {"baseline_initialization": initialization, "model_parameter_sha256": "variant"}
    sources = {}
    for name in (["scripts/rt_nextlat_a5_fuzzy_train.py"] + (["cdrm/rt_nextlat_task_embeddings.py"] if variant else [])):
        path = directory / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen fixture\n")
        sources[name] = saved.sha(path)
    identity = {"a5": {"sha256": "a5"}, "fuzzy": {"sha256": "fuzzy"}}
    write_json(directory / "source-manifest.json", sources)
    write_json(directory / "model-config.json", config)
    write_json(directory / "data-identity.json", identity)
    schema = "rt-nextlat-a5-fuzzy-training-v1"
    contract = {"schema": schema, "mode": "mixed", "batch_per_task": 2560, "microbatch": 2560,
                "model_config": config, "initialization": initialization,
                "configuration_file_sha256": saved.sha(directory / "model-config.json"),
                "source_sha256": saved.json_sha(sources), "data_sha256": saved.json_sha(identity),
                "streams": {"a5": {"train_rows": 800000, "length": 12, "order_seed": 5432},
                            "fuzzy": {"train_rows": 12800, "length": 400, "order_seed": 2026091604}},
                "optimizer": {"lr": 1e-4}, "runtime": {"precision": "fp32"}}
    start = saved.read_json(parent / "report.json")["completed_updates"] if parent else 0
    checkpoints, evaluations, history = [], [], []
    for update in range(start, endpoint+1):
        chains = {task: hashlib.sha256(f"{task}:{update}".encode()).hexdigest() for task in ("a5", "fuzzy")}
        seen = {task: update*2560 for task in ("a5", "fuzzy")}
        cursors = {task: {"absolute_example_offset": value, "epoch": value // contract["streams"][task]["train_rows"],
                          "position": value % contract["streams"][task]["train_rows"]} for task, value in seen.items()}
        model = {"backbone.original": torch.arange(8, dtype=torch.float32), "predictor.original": torch.ones(3)}
        if variant in ("input", "value"):
            model[compare.PROJECTION] = torch.eye(128)
        packet = {"schema": schema, "completed_updates": update, "contract": contract, "initialization": initialization,
                  "model": model, "optimizer": {"state": {}}, "examples_seen": seen,
                  "next_cursors": cursors, "order_chains": chains}
        path = directory / "checkpoints" / f"step-{update:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if parent and update == start:
            shutil.copy2(parent / "checkpoints" / path.name, path)
        else:
            torch.save(packet, path)
        record = {"path": str(path), "sha256": saved.sha(path), "bytes": path.stat().st_size, "completed_updates": update}
        checkpoints.append(record)
        if update == start:
            continue
        evaluations.extend(a5_metric(update, role, record) for role in ("dev", "ood_dev"))
        evaluations.append({"task": "fuzzy", "role": "dev", "update": update, "examples": 1280,
                            "answer_tokens": 100, "answer_correct": 90, "answer_accuracy": .9,
                            "answer_motif_exact_match": .8, "sequence_exact_match": .2,
                            "first_value_token_accuracy": .95, "terminal_probe_accuracy": .9, "checkpoint": record})
        history.append({"update": update, "seconds": 2. if not parent else 3., "order_chains": chains,
                        "examples_seen": seen, "tasks": {task: {"ce": 1., "latent": .1} for task in seen}})
    parent_record = None
    if parent:
        path = parent / "checkpoints" / f"step-{start:06d}.pt"
        parent_record = {"path": str(path), "sha256": saved.sha(path)}
    report = {"schema": schema, "status": "complete", "requested_endpoint": endpoint,
              "completed_updates": endpoint, "start_update": start, "parent_checkpoint": parent_record,
              "contract": contract, "initialization": initialization,
              "parameter_count": 479616 + (16384 if variant in ("input", "value") else 0),
              "evaluations": evaluations, "checkpoints": checkpoints,
              "confirmation_evaluated": False, "latent_rollout_evaluated": False,
              "train_seconds": sum(row["seconds"] for row in history)}
    write_json(directory / "report.json", report)
    (directory / "history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in history))
    return directory


def test_full_report_stitches_baseline_time_and_pairs_checkpoint_state(tmp_path):
    ancestor = fixture(tmp_path / "original", endpoint=2)
    baseline = fixture(tmp_path / "continued", endpoint=3, parent=ancestor)
    variant = fixture(tmp_path / "input", variant="input")
    output = tmp_path / "comparison"
    report = compare.build_report(SimpleNamespace(run=[("baseline", baseline), ("input", variant)], output=output))
    assert report["arms"]["baseline"]["cumulative_training_seconds"] == 7
    assert report["arms"]["input"]["cumulative_training_seconds"] == 6
    assert report["arms"]["baseline"]["evaluations"][-1]["cumulative_training_seconds"] == 7
    audit = report["compatibility"]["input"]
    assert audit["passed"] and audit["initialization"]["shared_tensors_exact"]
    assert audit["audited_full_milestone_updates"] == [1, 2, 3]
    assert audit["parameter_difference"] == 16384
    assert len(report["figures"]) == 8
    assert len(report["matched_observations"]["input"]) == 9
    assert all((output / f"{name}.pdf").stat().st_size > 1000 for name in report["figures"])
    text = (output / "report.md").read_text()
    assert "Upper input u + 0.01" in text and "No automatic winner" in text
    assert "no embedding bypass" not in text


@pytest.mark.parametrize("variant", ["value", "head"])
def test_other_mechanisms_match_shared_initial_state(tmp_path, variant):
    base = saved.load_run(fixture(tmp_path / "base"))
    other = saved.load_run(fixture(tmp_path / variant, variant=variant))
    audit = compare.compatibility(base, other)
    assert audit["passed"]
    assert audit["parameter_difference"] == (0 if variant == "head" else 16384)
    if variant == "head":
        assert audit["variant"]["head_index"] == 15


@pytest.mark.parametrize("problem", ["source", "order", "lr", "initialization", "head", "tensor"])
def test_rejects_unpaired_changes(tmp_path, problem):
    base = saved.load_run(fixture(tmp_path / "base"))
    other = saved.load_run(fixture(tmp_path / "variant", variant="head" if problem == "head" else "input"))
    if problem == "source":
        other["sources"]["scripts/rt_nextlat_a5_fuzzy_train.py"] = "changed"
    elif problem == "order":
        other["history"][1]["order_chains"]["a5"] = "changed"
    elif problem == "lr":
        other["report"]["contract"]["optimizer"]["lr"] = 1e-3
    elif problem == "initialization":
        other["report"]["initialization"]["baseline_initialization"]["seed"] = 42
    elif problem == "head":
        other["report"]["contract"]["model_config"]["embedding_injection"]["head_index"] = 0
    else:
        record = other["checkpoints"][0]
        path = record["verified_local_path"]
        packet = torch.load(path, map_location="cpu", weights_only=False)
        packet["model"]["backbone.original"][0] += 1
        torch.save(packet, path)
        record["sha256"] = saved.sha(path)
    with pytest.raises(ValueError):
        compare.compatibility(base, other)


def test_thresholds_use_full_pools_and_matching_requires_same_rows():
    cp = {"sha256": "checkpoint"}
    subset = a5_metric(1, "ood_dev", cp, rows=4096, correct=100)
    full = a5_metric(2, "ood_dev", cp, correct=1)
    observed = compare.observed_crossings([subset, full], {1: 2., 2: 4.})
    assert observed["a5_l36_whole_word"]["positive"]["update"] == 2
    assert observed["a5_l36_whole_word"]["0.1"] is None
    same_step = {**full, "update": 1}
    assert compare.matched_observations({"evaluations": [subset]}, {"evaluations": [same_step]}) == []


def test_timing_rejects_gaps_and_nonfinite_data():
    with pytest.raises(ValueError, match="gap"):
        compare.cumulative_training_seconds([{"update": 2, "seconds": 1}])
    with pytest.raises(ValueError, match="seconds"):
        compare.cumulative_training_seconds([{"update": 1, "seconds": float("nan")}])
