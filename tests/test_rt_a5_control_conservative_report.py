"""Small saved-comparison guards and plotting tests; no live training inputs."""
import copy

import pytest

from scripts import rt_a5_control_conservative_report as report


def pair():
    keys = ("architecture", "width", "seed", "predictor_seed", "data_order_seed", "batch_size",
        "length", "train_rows", "data_manifest_sha256", "precision", "tf32", "compile", "cuda_graphs",
        "torch", "cuda", "device_capability", "optimizer", "word_order", "evaluation_route",
        "latent_rollout_evaluated", "nextlat_config")
    contract = {key: "same" for key in keys}
    contract.update(model_config={"n_layers": 4}, objective={"latent_weight": 1},
        experiment_config={"n_layers": 4, "window_layer": 0, "window_length": 2, "attention": ["window", "full", "full", "full"]})
    common = {"contract": contract, "source_files": {"shared": "a"},
        "history": [{"order_chain": str(step)} for step in range(10000)],
        "curves": {"10000": {"ood_dev": [{"prefix_exact_count": 0}]}}}
    control, conservative = copy.deepcopy(common), copy.deepcopy(common)
    control["initialization"] = {"parameter_count": 13702656, "parameter_tensors": 43,
        "model_parameter_sha256": "a" * 64, "predictor_sha256": "b" * 64}
    conservative["initialization"] = {"parameter_count": 13964800, "parameter_tensors": 44,
        "shared_four_layer_model_sha256": "a" * 64, "predictor_sha256": "b" * 64}
    return control, conservative


def test_pair_requires_exact_shared43_initial_tensors_and_common_optimizer():
    control, conservative = pair()
    report.validate_pair(control, conservative)
    bad = copy.deepcopy(control); bad["initialization"]["model_parameter_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Exactly43"):
        report.validate_pair(bad, conservative)
    bad = copy.deepcopy(control); bad["contract"]["optimizer"] = "other"
    with pytest.raises(ValueError, match="contracts differ"):
        report.validate_pair(bad, conservative)


def test_conditional_control_requires_zero_conservative_exact_count_and_same_order():
    control, conservative = pair()
    conservative["curves"]["10000"]["ood_dev"][-1]["prefix_exact_count"] = 1
    with pytest.raises(ValueError, match="zero whole-word"):
        report.validate_pair(control, conservative)
    conservative["curves"]["10000"]["ood_dev"][-1]["prefix_exact_count"] = 0
    conservative["history"][0]["order_chain"] = "different"
    with pytest.raises(ValueError, match="minibatch orders"):
        report.validate_pair(control, conservative)


def test_two_arm_full_boundary_and_learning_plots_share_saved_rows(tmp_path):
    summary = {"arms": {}}
    for arm in ("control", "conservative"):
        curves = {str(step): {"ood_dev": [{"arm": arm, "update": step, "role": "ood_dev", "length": t,
            "E": 1 / t, "A": .8, "M": .9} for t in range(1, 37)]} for step in report.STEPS}
        summary["arms"][arm] = {"curves": curves,
            "training_curve": [{"update": step, "state_ce": .1, "latent_loss": .2} for step in report.STEPS]}
        assert report.plot_rows(summary, arm) is curves["10000"]["ood_dev"]
    figures = report.plots(summary, tmp_path)
    assert figures == ["length-full", "length-boundary", "learning-curves"]
    assert all((tmp_path / f"{name}.{suffix}").stat().st_size > 1000 for name in figures for suffix in ("pdf", "png"))
