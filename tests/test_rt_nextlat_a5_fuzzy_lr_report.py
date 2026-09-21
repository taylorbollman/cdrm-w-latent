import copy

import pytest

from scripts import rt_nextlat_a5_fuzzy_lr_report as report
from scripts.rt_nextlat_a5_fuzzy_lr_train import learning_rate_state, schedule_configuration


@pytest.fixture
def contracts():
    base = {"mode": "mixed", "batch_per_task": 2560, "microbatch": 2560,
            "model_config": {"backbone": {"d_model": 128, "n_layers": 2}},
            "runtime": {"precision": "fp32", "compile": False},
            "schema": "rt-nextlat-a5-fuzzy-training-v1", "source_sha256": "original",
            "optimizer": {"lr": 1e-4, "clip_norm": 1.0, "betas": [.9, .95]},
            "objective": {"a5_weight": .5, "fuzzy_weight": .5}, "data_sha256": "same"}
    candidate = copy.deepcopy(base)
    candidate.update(schema=report.TRAIN_SCHEMA, source_sha256="extended", learning_rate_schedule=schedule_configuration())
    candidate["optimizer"].update(lr=3e-4, initial_lr=1e-4)
    return base, candidate


def test_schedule_only_contract_changes_accepted(contracts):
    assert report.check_contracts(*contracts) == schedule_configuration()


@pytest.mark.parametrize("change", ["batch", "model", "precision", "task_weight", "clip", "warmup"])
def test_unapproved_changes_rejected(contracts, change):
    base, candidate = contracts
    if change == "batch": candidate["batch_per_task"] = 128
    if change == "model": candidate["model_config"]["backbone"]["n_layers"] = 4
    if change == "precision": candidate["runtime"]["precision"] = "bf16_mixed"
    if change == "task_weight": candidate["objective"]["a5_weight"] = .75
    if change == "clip": candidate["optimizer"]["clip_norm"] = 2.0
    if change == "warmup": candidate["learning_rate_schedule"] = schedule_configuration(warmup_updates=200)
    with pytest.raises(ValueError): report.check_contracts(base, candidate)


def test_actual_lr_check_includes_warmup_boundary():
    assert report.expected_lr(1) == 1e-4
    assert report.expected_lr(50) == pytest.approx(1e-4 + 2e-4*49/99)
    assert report.expected_lr(100) == pytest.approx(3e-4)
    assert report.expected_lr(5000) == pytest.approx(3e-4)
    with pytest.raises(ValueError): report.expected_lr(0)


def test_checkpoint_lr_rejects_metadata_and_actual_optimizer_disagreement():
    schedule = schedule_configuration()
    state = learning_rate_state(100, schedule)
    packet = {"learning_rate_state": state, "optimizer": {"param_groups": [{"lr": 3e-4}]}}
    report.check_saved_lr(packet, 100, schedule)
    packet["optimizer"]["param_groups"][0]["lr"] = 1e-4
    with pytest.raises(ValueError): report.check_saved_lr(packet, 100, schedule)


def test_training_bins_count_clipped_updates_without_averaging_threshold():
    rows = [{"update": i, "seconds": 2, "loss": 3, "grad_norm": grad,
             "learning_rate": report.expected_lr(i),
             "tasks": {task: {"ce": 2.0, "latent": .1} for task in ["a5", "fuzzy"]}}
            for i, grad in [(1, .1), (2, 1.1), (3, 2.0)]]
    point = report.training_bins(rows, constant=False)[0]
    assert point["clip_fraction"] == pytest.approx(2/3)
    assert point["training_seconds"] == 6
    assert point["learning_rate"] == report.expected_lr(3)
