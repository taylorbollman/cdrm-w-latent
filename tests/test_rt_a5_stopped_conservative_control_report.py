"""Small saved-comparison guards and plotting tests; no live training inputs."""
import copy

import pytest

from scripts import rt_a5_stopped_conservative_control_report as report


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
    control["completed_updates"] = 10000
    conservative["completed_updates"] = 9665
    conservative["history"] = conservative["history"][:9665]
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


def test_stopped_reference_has_no_accuracy_gate_but_keeps_common_order_check():
    control, conservative = pair()
    conservative["curves"]["10000"]["ood_dev"][-1]["prefix_exact_count"] = 1
    report.validate_pair(control, conservative)
    conservative["history"][0]["order_chain"] = "different"
    with pytest.raises(ValueError, match="common minibatch"):
        report.validate_pair(control, conservative)


def test_matched_and_unequal_terminal_plots_use_the_declared_saved_checkpoints(tmp_path):
    summary = {"arms": {}, "matched_update": 5000, "common_checkpoint_updates": [1000, 5000],
               "actual_endpoints": {"control": 10000, "conservative": 9665}, "qualification": "fixture"}
    for arm in ("control", "conservative"):
        end = summary["actual_endpoints"][arm]
        steps = [1000, 5000, end]
        curves = {str(step): {"ood_dev": [{"arm": arm, "update": step, "role": "ood_dev", "length": t,
            "E": 1 / t, "A": .8, "M": .9} for t in range(1, 37)]} for step in steps}
        summary["arms"][arm] = {"curves": curves, "checkpoint_updates": steps,
            "training_curve": [{"update": step, "state_ce": .1, "latent_loss": .2} for step in steps]}
        assert report.plot_rows(summary, arm) is curves["5000"]["ood_dev"]
        assert report.plot_rows(summary, arm, end) is curves[str(end)]["ood_dev"]
    figures = report.plots(summary, tmp_path)
    assert figures == ["length-full", "length-boundary", "length-unequal-terminals", "learning-curves"]
    assert all((tmp_path / f"{name}.{suffix}").stat().st_size > 1000 for name in figures for suffix in ("pdf", "png"))
    text = report.markdown(summary)
    assert "matched comparison uses 5,000" in text and "conservative 9,665" in text
    assert "terminal budgets are unequal" in text
