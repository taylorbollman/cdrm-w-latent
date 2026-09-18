"""Bounded saved-evidence and shared-plot guards for the paired injection report."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import rt_a5_embedding_injection_report as reporter


def contract_pair(variant="input"):
    reference = json.loads((reporter.base.REFERENCE / "report.json").read_text())["contract"]
    new = copy.deepcopy(reference)
    new.update(n_layers=4, variant=variant, projection_seed=1236, injection_coefficient=.02)
    new["model_config"]["n_layers"] = 4
    experiment = new["experiment_config"]
    experiment.update(n_layers=4, variant=variant, injection_layer=1, coefficient=.02,
                      coefficient_learned=False, projection_seed=1236, learned_position_parameters=False)
    layers = reference["experiment_config"]["attention"]["layers"]
    experiment["attention"]["layers"] = [layers[0]] + [layers[1]] * 3
    return new, reference


@pytest.mark.parametrize("variant", reporter.ARMS)
def test_exact_four_layer_contract_is_accepted(variant):
    reporter.validate_shared_contracts(*contract_pair(variant), variant)


@pytest.mark.parametrize("change", ["objective", "window", "coefficient", "projection_seed", "depth", "tf32"])
def test_nonmatching_recipe_is_rejected(change):
    new, reference = contract_pair()
    if change == "objective":
        new["objective"]["latent_weight"] = 0
    elif change == "window":
        new["experiment_config"]["attention"]["layers"].reverse()
    elif change == "coefficient":
        new["experiment_config"]["coefficient"] = .01
    elif change == "projection_seed":
        new["projection_seed"] = 1237
    elif change == "depth":
        new["model_config"]["n_layers"] = 6
    else:
        new["tf32"] = True
    with pytest.raises(ValueError):
        reporter.validate_shared_contracts(new, reference, "input")


def initializations():
    shared = {"schema": "rt-a5-embedding-injection-initialization-v1",
              "parameter_count": 13964800, "parameter_tensors": 44,
              "projection_seed": 1236, "projection_parameter_count": 512 * 512,
              "predictor_seed": 1235, "coefficient": .02, "coefficient_learned": False,
              "baseline_parameter_tensors_changed": [], "predictor_initialization_paired": True}
    for index, key in enumerate(("model_parameter_sha256", "canonical_sha256", "predictor_sha256",
                                 "projection_sha256", "shared_four_layer_model_sha256", "shared_transformer_sha256"), 1):
        shared[key] = f"{index:064x}"
    return [{**copy.deepcopy(shared), "variant": variant} for variant in reporter.ARMS]


def test_identical_learned_initialization_is_paired_across_different_routes():
    reporter.validate_initializations(*initializations())


@pytest.mark.parametrize("key", ["model_parameter_sha256", "predictor_sha256", "projection_sha256", "shared_transformer_sha256"])
def test_source_identity_alone_cannot_substitute_for_exact_initial_tensor_hashes(key):
    first, second = initializations()
    second[key] = "f" * 64
    with pytest.raises(ValueError, match="identical learned tensor"):
        reporter.validate_initializations(first, second)


def history():
    return [{"update": update, "examples_seen": update * 1024, "order_chain": f"{update:064x}",
             "seconds": .01, "loss": 1.25, "state_loss": 1., "latent_loss": .25,
             "weighted_latent_loss": .25, "grad_norm": 1., "token_accuracy": .5, "whole_word_exact": .1}
            for update in range(1, 10001)]


def test_all_10000_minibatch_hashes_match_not_just_endpoints():
    reference = history()
    arms = {arm: {"history": history()} for arm in reporter.ARMS}
    reporter.compare_orders(arms, reference)
    arms["value"]["history"][4186]["order_chain"] = "f" * 64
    with pytest.raises(ValueError):
        reporter.compare_orders(arms, reference)


def plot_summary():
    arms = {}
    for arm in reporter.ARMS:
        curves = {}
        for step in reporter.STEPS:
            rows = []
            for length in range(1, 37):
                value = (37-length) / 36
                row = {"arm": arm, "update": step, "role": "ood_dev", "length": length,
                       "E": value, "A": .25 + .5 * value, "M": .5 + .25 * value}
                for key in ("E", "A"):
                    row[key + "_low95"] = max(0., row[key]-.01)
                    row[key + "_high95"] = min(1., row[key]+.01)
                rows.append(row)
            curves[str(step)] = {"ood_dev": rows}
        arms[arm] = {"curves": curves, "training_curve": [
            {"update": i * 100, "state_ce": 1 / i, "latent_loss": .5 / i, "loss": 1.5 / i}
            for i in range(1, 101)]}
    return {"arms": arms}


@pytest.mark.parametrize("arm", reporter.ARMS)
def test_plot_accessor_preserves_exact_saved_row_identity(arm):
    summary = plot_summary()
    expected = summary["arms"][arm]["curves"]["10000"]["ood_dev"]
    assert reporter.plot_rows(summary, arm) is expected


@pytest.mark.parametrize("case", ["wrong_arm", "wrong_step", "mixed_rows", "missing_position"])
def test_plot_scope_and_row_mixing_are_rejected(case):
    summary = plot_summary()
    if case == "wrong_arm":
        call = lambda: reporter.plot_rows(summary, "two_layers")
    elif case == "wrong_step":
        call = lambda: reporter.plot_rows(summary, "input", 15000)
    else:
        rows = summary["arms"]["input"]["curves"]["10000"]["ood_dev"]
        if case == "mixed_rows":
            rows[9]["arm"] = "value"
        else:
            rows.pop(12)
        call = lambda: reporter.plot_rows(summary, "input")
    with pytest.raises(ValueError):
        call()


def test_actual_full_and_boundary_figures_share_every_plotted_metric(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    captured = {}

    def capture(fig, path, **_kwargs):
        name = Path(path).stem
        if name in ("length-full", "length-boundary"):
            captured[name] = [(axis.get_xlim(), [line.get_xydata().copy() for line in axis.lines])
                              for axis in fig.axes]

    monkeypatch.setattr(Figure, "savefig", capture)
    assert reporter.plots(plot_summary(), tmp_path) == [
        "length-full", "length-boundary", "whole-word-vs-updates", "training-losses"]
    assert len(captured["length-full"]) == len(captured["length-boundary"]) == 3
    for full, boundary in zip(captured["length-full"], captured["length-boundary"]):
        assert full[0] == (1., 36.) and boundary[0] == (10., 18.)
        assert len(full[1]) == len(boundary[1]) == 3  # input, value, training-length marker
        for left, right in zip(full[1], boundary[1]):
            np.testing.assert_array_equal(left, right)
