"""Focused saved-evidence guards for the input-only20k supplement."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import rt_a5_embedding_injection_extension_report as reporter


@pytest.fixture
def child_fixture():
    parent_path = reporter.paired.LINEAGES["input"] / "train-injection/report.json"
    parent = json.loads(parent_path.read_text())
    protocol = json.loads((reporter.LINEAGE / "protocol.json").read_text())
    child = copy.deepcopy(parent)
    child.update(start_update=10000, endpoint=20000, completed_updates=20000,
                 parent_checkpoint=copy.deepcopy(protocol["parent_checkpoint"]),
                 checkpoints=[{"completed_updates": 15000}, {"completed_updates": 20000}])
    return child, copy.deepcopy(protocol["resolved_args"]), protocol, parent


def test_exact_input_continuation_contract_is_accepted(child_fixture):
    reporter.validate_child(*child_fixture)


@pytest.mark.parametrize("change", ["parent", "start", "sync", "initialization", "source", "config", "checkpoints"])
def test_changed_or_incomplete_continuation_is_rejected(child_fixture, change):
    child, config, protocol, parent = child_fixture
    if change == "parent":
        child["parent_checkpoint"]["sha256"] = "f" * 64
    elif change == "start":
        child["start_update"] = 0
    elif change == "sync":
        child["wandb"]["status"] = "running"
    elif change == "initialization":
        child["initialization"]["projection_seed"] = 999
    elif change == "source":
        child["source_files"].pop(next(iter(child["source_files"])))
    elif change == "config":
        config["variant"] = "value"
    else:
        child["checkpoints"].insert(0, {"completed_updates": 10000})
    with pytest.raises(ValueError):
        reporter.validate_child(child, config, protocol, parent)


def evaluation(step=15000):
    raw = json.loads((reporter.paired.LINEAGES["input"] / "train-injection/report.json").read_text())
    row = copy.deepcopy(next(item for item in raw["evaluations"]
                             if item["update"] == 10000 and item["role"] == "ood_dev"))
    row["update"] = step
    return row


def test_own_full15k_evaluation_is_valid_without_any_value_arm():
    item = evaluation()
    selected, rows = reporter.selected_evaluation({"evaluations": [item]}, 15000, "ood_dev")
    assert selected is item
    assert len(rows) == 36 and all(row["arm"] == "input" and row["words"] == 102400 for row in rows)
    assert rows[-1]["E"] == item["whole_word_exact_match"]


@pytest.mark.parametrize("change", ["routine_rows", "duplicate", "wrong_step", "confirmation"])
def test_mismatched_evaluation_scope_is_rejected(change):
    item = evaluation()
    packet = {"evaluations": [item]}
    step, role = 15000, "ood_dev"
    if change == "routine_rows":
        item["rows"] = 4096
    elif change == "duplicate":
        packet["evaluations"].append(copy.deepcopy(item))
    elif change == "wrong_step":
        step = 17500
    else:
        role = "confirmation"
    with pytest.raises(ValueError):
        reporter.selected_evaluation(packet, step, role)


def plot_summary():
    curves = {}
    for step in reporter.STEPS:
        rows = []
        for length in range(1, 37):
            value = 1 - length / (40 + step / 1000)
            row = {"arm": "input", "update": step, "role": "ood_dev", "length": length,
                   "E": value, "A": min(.99, value+.1), "M": .5 + .4*value}
            for key in ("E", "A"):
                row[key + "_low95"] = max(0., row[key]-.01)
                row[key + "_high95"] = min(1., row[key]+.01)
            rows.append(row)
        curves[str(step)] = {"ood_dev": rows}
    return {"curves": curves, "training_curve": [
        {"update": i*100, "state_ce": 1/i, "latent_loss": .5/i, "loss": 1.5/i}
        for i in range(1, 201)]}


def test_plot_accessor_preserves_identity_and_rejects_other_arm_or_missing_position():
    summary = plot_summary()
    rows = summary["curves"]["20000"]["ood_dev"]
    assert reporter.plot_rows(summary, 20000) is rows
    rows[4]["arm"] = "value"
    with pytest.raises(ValueError):
        reporter.plot_rows(summary, 20000)
    summary = plot_summary()
    summary["curves"]["10000"]["ood_dev"].pop(12)
    with pytest.raises(ValueError):
        reporter.plot_rows(summary, 10000)


def test_actual_own10k_and20k_curves_match_between_full_and_boundary(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    captured = {}

    def capture(fig, path, **_kwargs):
        name = Path(path).stem
        if name in ("length-full", "length-boundary"):
            captured[name] = [(axis.get_xlim(), [line.get_xydata().copy() for line in axis.lines]) for axis in fig.axes]

    monkeypatch.setattr(Figure, "savefig", capture)
    names = reporter.plots(plot_summary(), tmp_path)
    assert names == ["length-full", "length-boundary", "whole-word-vs-updates", "training-losses"]
    assert len(captured["length-full"]) == len(captured["length-boundary"]) == 3
    for full, boundary in zip(captured["length-full"], captured["length-boundary"]):
        assert full[0] == (1., 36.) and boundary[0] == (10., 18.)
        assert len(full[1]) == len(boundary[1]) == 3
        assert not np.array_equal(full[1][0], full[1][1])
        for left, right in zip(full[1], boundary[1]):
            np.testing.assert_array_equal(left, right)
