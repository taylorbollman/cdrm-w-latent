"""Prospective position selection uses unconditional exact-prefix counts."""
import copy
import hashlib
import json

import pytest

from scripts import rt_a5_window_select as selector


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def arms(tmp_path):
    initialization = {"canonical_sha256": "original-mitchell", "predictor_seed": 1235}
    contract = {"schema": "baseline-contract", "source_sha256": "old-sources",
                "model_config": {"alibi": True, "init_fn": "mitchell"},
                "data_manifest_sha256": "same-data", "optimizer": "same-optimizer",
                "torch": "same-runtime", "device_capability": [9, 0]}
    source_text = "# Historical immutable source fixture\n"
    source_digest = hashlib.sha256(source_text.encode()).hexdigest()
    history = [{"update": i, "examples_seen": 1024 * i, "order_chain": str(i)}
               for i in range(1, 10001)]
    locations = {}
    for position in ("alibi", "sinusoidal"):
        root = tmp_path / position
        root.mkdir()
        (root / "source").mkdir()
        (root / "source" / "model.py").write_text(source_text)
        checkpoint = root / "endpoint.pt"
        checkpoint.write_bytes(b"test fixture, not a serialized model")
        local_contract = copy.deepcopy(contract)
        local_init = copy.deepcopy(initialization)
        schema = "rt-a5-nextlat-training-v1"
        if position == "sinusoidal":
            schema = "rt-a5-window-training-v1"
            local_contract.update(schema="new-contract", source_sha256="new-sources",
                                  experiment_config={"position_encoding": "sinusoidal",
                                                     "second_layer_window": None},
                                  training_step="same-function", evaluation="same-function",
                                  one_step_diagnostics="same-function")
            local_contract["model_config"]["alibi"] = False
            local_init = {"baseline_initialization": copy.deepcopy(initialization),
                          "changed_parameter_slices": []}
        report = {"schema": schema, "status": "complete", "completed_updates": 10000,
                  "start_update": 0, "endpoint": 10000, "parent_checkpoint": None,
                  "confirmation_evaluated": False, "latent_rollout_evaluated": False,
                  "wandb": {"status": "synced"}, "order_chain": "10000",
                  "contract": local_contract, "initialization": local_init,
                  "evaluations": [{"role": role, "update": 10000, "length": length,
                                   "rows": 102400, "route": "backbone_only",
                                   "cumulative_prefix_exactness": [1.0] * 12 + [0.0] * (length - 12)}
                                  for role, length in (("dev", 12), ("ood_dev", 36))],
                  "checkpoints": [{"completed_updates": 10000, "path": str(checkpoint),
                                   "sha256": selector.sha(checkpoint)}],
                  "source_files": {"model.py": source_digest}}
        write_json(root / "report.json", report)
        (root / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        locations[position] = root
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"position_selection": {
        "checkpoint": 10000, "role": "ood_dev", "rows": 102400, "exact_tie": "alibi",
        "primary": "mean cumulative-prefix exactness E(t) over t=13..36 inclusive"}})
    return locations, protocol


def change_report(arms, position, fn):
    path = arms[0][position] / "report.json"
    report = json.loads(path.read_text())
    fn(report)
    write_json(path, report)


def select(arms):
    locations, protocol = arms
    return selector.select(locations["alibi"], locations["sinusoidal"], protocol)


def set_counts(arms, position, counts):
    assert len(counts) == 36
    change_report(arms, position, lambda r: r["evaluations"][1].update(
        cumulative_prefix_exactness=[c / 102400 for c in counts]))


def test_longer_exact_tail_can_win_despite_lower_length13_accuracy(arms):
    set_counts(arms, "alibi", [102400] * 12 + [70000] + [0] * 23)
    set_counts(arms, "sinusoidal", [102400] * 12 + [65000, 60000] + [0] * 22)
    result = select(arms)
    assert result["selected_position"] == "sinusoidal"
    assert result["arms"]["sinusoidal"]["exact_prefix_count_sum_13_36"] == 125000
    assert result["arms"]["sinusoidal"]["score"] == 125000 / (102400 * 24)
    assert result["next_run"] == {"position_encoding": "sinusoidal", "second_layer_window": 2,
                                  "initialization": "fresh original Mitchell", "updates": 10000}
    assert result["confirmation_evaluated"] is False


def test_score_is_unconditional_not_divided_by_training_length_survivors(arms):
    set_counts(arms, "alibi", [102400] * 12 + [60000] + [0] * 23)
    set_counts(arms, "sinusoidal", [102400] * 11 + [50000, 50000] + [0] * 23)
    result = select(arms)
    # A conditional E13/E12 would select sinusoids (100% versus 58.6%).
    assert result["selected_position"] == "alibi"
    assert result["arms"]["sinusoidal"]["denominator"] == 102400 * 24
    assert result["arms"]["sinusoidal"]["score"] == 50000 / (102400 * 24)


def test_exact_integer_tie_keeps_alibi_and_includes_both_endpoints(arms):
    set_counts(arms, "alibi", [102400] * 12 + [24] + [0] * 23)
    set_counts(arms, "sinusoidal", [102400] * 12 + [1] * 24)
    result = select(arms)
    assert result["selected_position"] == "alibi"
    assert result["arms"]["alibi"]["exact_prefix_count_sum_13_36"] == 24
    assert result["arms"]["sinusoidal"]["exact_prefix_count_sum_13_36"] == 24


@pytest.mark.parametrize("mutation", ["initialization", "window", "optimizer", "runtime", "source", "order"])
def test_unmatched_pilots_rejected(arms, mutation):
    if mutation == "order":
        path = arms[0]["sinusoidal"] / "history.jsonl"
        history = [json.loads(line) for line in path.read_text().splitlines()]
        history[10]["order_chain"] = "different-order"
        path.write_text("".join(json.dumps(r) + "\n" for r in history))
    else:
        def mutate(report):
            if mutation == "initialization":
                report["initialization"]["changed_parameter_slices"] = ["value-weight"]
            elif mutation == "window":
                report["contract"]["experiment_config"]["second_layer_window"] = 2
            elif mutation in ("optimizer", "runtime"):
                report["contract"]["optimizer" if mutation == "optimizer" else "torch"] = "different"
            elif mutation == "source":
                report["source_files"]["model.py"] = "changed-source"
        change_report(arms, "sinusoidal", mutate)
    with pytest.raises(ValueError):
        select(arms)


@pytest.mark.parametrize("mutation", ["fractional_count", "increasing_curve", "wrong_rows", "incomplete", "wrong_route"])
def test_invalid_evaluation_and_endpoint_rejected(arms, mutation):
    def mutate(report):
        metric = report["evaluations"][1]
        if mutation == "fractional_count":
            metric["cumulative_prefix_exactness"][12] = 0.1 / 102400
        elif mutation == "increasing_curve":
            metric["cumulative_prefix_exactness"][13] = 1 / 102400
        elif mutation == "wrong_rows":
            metric["rows"] = 4096
        elif mutation == "incomplete":
            report["completed_updates"] = 5000
        elif mutation == "wrong_route":
            metric["route"] = "latent-rollout"
    change_report(arms, "sinusoidal", mutate)
    with pytest.raises(ValueError):
        select(arms)
