"""Focused saved-evidence guards for the bounded depth diagnostic reporter."""
import copy
import json

import pytest

from scripts import rt_a5_l1r_depth_report as reporter


@pytest.fixture
def contracts():
    reference = json.loads((reporter.REFERENCE / "report.json").read_text())["contract"]
    new = copy.deepcopy(reference)
    new["model_config"]["n_layers"] = 6
    new["experiment_config"]["n_layers"] = 6
    layers = reference["experiment_config"]["attention"]["layers"]
    new["experiment_config"]["attention"]["layers"] = [layers[0]] + [layers[1]] * 5
    return new, reference


def history():
    return [{"update": i, "examples_seen": i * 1024, "order_chain": f"{i:064x}", "seconds": .01,
             "loss": 1.25, "state_loss": 1., "latent_loss": .25, "weighted_latent_loss": .25,
             "grad_norm": 1., "token_accuracy": .5, "whole_word_exact": .1} for i in range(1, 10001)]


def test_depth_only_contract_change_is_accepted(contracts):
    reporter.validate_shared_contracts(*contracts)


@pytest.mark.parametrize("change", ["objective", "positions", "window", "depth", "tf32"])
def test_nonmatching_recipe_is_rejected(contracts, change):
    new, reference = contracts
    if change == "objective":
        new["objective"]["latent_weight"] = 0.
    elif change == "positions":
        new["model_config"]["rope"] = True
    elif change == "window":
        new["experiment_config"]["attention"]["layers"].reverse()
    elif change == "depth":
        new["model_config"]["n_layers"] = 5
    else:
        new["tf32"] = True
    with pytest.raises(ValueError):
        reporter.validate_shared_contracts(new, reference)


@pytest.mark.parametrize("change", [None, "missing_update", "loss", "nonfinite"])
def test_history_scope_and_loss_guards(change):
    rows = history()
    if change is None:
        reporter.validate_history(rows)
        return
    if change == "missing_update":
        rows.pop(100)
    elif change == "loss":
        rows[333]["loss"] += .1
    else:
        rows[9]["state_loss"] = float("nan")
    with pytest.raises(ValueError):
        reporter.validate_history(rows)


def test_every_minibatch_order_is_compared():
    left, right = history(), history()
    reporter.compare_minibatch_orders(left, right)
    right[5173]["order_chain"] = "f" * 64
    with pytest.raises(ValueError):
        reporter.compare_minibatch_orders(left, right)


def test_closed_80k_reference_is_limited_to_first10k():
    packet = reporter.read_arm(reporter.REFERENCE, "two_layers")
    assert len(packet["history"]) == 10000
    assert packet["history"][-1]["update"] == 10000
    assert set(packet["curves"]) == {"1000", "5000", "10000"}
    assert set(packet["checkpoints"]) == {"0", "1000", "5000", "10000"}
    summary = {"arms": {"two_layers": packet}}
    rows = reporter.plot_rows(summary, "two_layers")
    assert len(rows) == 36
    assert all(row["update"] == 10000 and row["role"] == "ood_dev" for row in rows)
    with pytest.raises(ValueError):
        reporter.plot_rows(summary, "two_layers", 80000)
    changed = copy.deepcopy(summary)
    changed["arms"]["two_layers"]["curves"]["10000"]["ood_dev"][0]["arm"] = "six_layers"
    with pytest.raises(ValueError):
        reporter.plot_rows(changed, "two_layers")
