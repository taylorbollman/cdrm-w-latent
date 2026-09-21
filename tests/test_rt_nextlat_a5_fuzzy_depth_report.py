import copy
import json

import pytest

from scripts import rt_nextlat_a5_fuzzy_depth_report as report
from scripts.rt_nextlat_a5_fuzzy_lr_train import schedule_configuration


@pytest.fixture
def contracts():
    config = json.loads((report.ROOT / "configs/rt_nextlat_tasks/fuzzy_d128.json").read_text())
    base = {"schema": report.lr.TRAIN_SCHEMA, "mode": "mixed", "batch_per_task": 2560,
            "microbatch": 2560, "runtime": {"precision": "fp32", "compile": False},
            "model_config": config, "initialization": {}, "source_sha256": "old",
            "configuration_file_sha256": "old", "learning_rate_schedule": schedule_configuration(),
            "optimizer": {"lr": 3e-4, "initial_lr": 1e-4, "clip_norm": 1., "betas": [.9, .95]},
            "data_sha256": "same", "objective": {"task_weights": {"a5": .5, "fuzzy": .5}},
            "streams": {"a5": {"order_seed": 5432}, "fuzzy": {"order_seed": 2026091604}}}
    candidate = copy.deepcopy(base)
    candidate["model_config"]["schema"] = "rt-nextlat-tasks-depth-model-v1"
    candidate["model_config"]["backbone"]["n_layers"] = 3
    candidate.update(initialization={"depth": 3}, source_sha256="new", configuration_file_sha256="new")
    return base, candidate


@pytest.mark.parametrize("microbatch", [1280, 2560])
def test_native_depth_and_declared_accumulation_allowed(contracts, microbatch):
    base, candidate = contracts
    candidate["microbatch"] = microbatch
    original = copy.deepcopy(contracts)
    assert report.check_contracts(base, candidate) == schedule_configuration()
    assert contracts == original


@pytest.mark.parametrize("field", ["effective_batch", "microbatch", "width", "layers", "window",
                                   "embedding", "lr", "warmup", "weights", "data", "order", "precision", "clip"])
def test_unapproved_experiment_changes_rejected(contracts, field):
    base, candidate = contracts
    if field == "effective_batch": candidate["batch_per_task"] = 1280
    if field == "microbatch": candidate["microbatch"] = 640
    if field == "width": candidate["model_config"]["backbone"]["d_model"] = 256
    if field == "layers": candidate["model_config"]["backbone"]["n_layers"] = 4
    if field == "window": candidate["model_config"]["window_layer"] = 1
    if field == "embedding": candidate["model_config"]["embedding_injection"] = {"variant": "value"}
    if field == "lr": candidate["optimizer"]["lr"] = 5e-4
    if field == "warmup": candidate["learning_rate_schedule"] = schedule_configuration(warmup_updates=200)
    if field == "weights": candidate["objective"]["task_weights"]["a5"] = .75
    if field == "data": candidate["data_sha256"] = "different"
    if field == "order": candidate["streams"]["a5"]["order_seed"] = 999
    if field == "precision": candidate["runtime"]["precision"] = "bf16_mixed"
    if field == "clip": candidate["optimizer"]["clip_norm"] = 2.
    with pytest.raises(ValueError): report.check_contracts(base, candidate)


def test_source_closure_requires_only_depth_additions():
    old = {"old.py": "sha"}
    new = {**old, **{name: "newsha" for name in report.EXTRA_SOURCES}}
    report.check_sources(old, new)
    for changed in ({**new, "old.py": "changed"}, {**new, "unapproved.py": "extra"}, old):
        with pytest.raises(ValueError): report.check_sources(old, changed)


def row(update):
    return {"update": update, "order_chains": {"a5": "a", "fuzzy": "f"},
            "examples_seen": {"a5": 2560*update, "fuzzy": 2560*update},
            "learning_rate": report.lr.expected_lr(update), "gradient_clipped": True, "grad_norm": 2.,
            "loss": 1., "seconds": .5, "tasks": {task: {"ce": 1., "latent": .1} for task in ("a5", "fuzzy")}}


def test_history_allows_clean_early_stop_and_different_gradients():
    old, new = [row(1), row(2)], [row(1)]
    new[0]["grad_norm"] = 3.
    new[0]["seconds"] = 1.
    report.check_histories(old, new)


@pytest.mark.parametrize("field", ["update", "order", "examples", "lr", "clipping", "nonfinite", "task_nonfinite"])
def test_history_rejects_unmatched_orders_or_unstable_values(field):
    old, new = [row(1)], [row(1)]
    if field == "update": new[0]["update"] = 2
    if field == "order": new[0]["order_chains"]["a5"] = "changed"
    if field == "examples": new[0]["examples_seen"]["fuzzy"] = 1280
    if field == "lr": new[0]["learning_rate"] = 3e-4
    if field == "clipping": new[0]["gradient_clipped"] = False
    if field == "nonfinite": new[0]["grad_norm"] = float("nan")
    if field == "task_nonfinite": new[0]["tasks"]["fuzzy"]["ce"] = float("inf")
    with pytest.raises(ValueError): report.check_histories(old, new)


def test_qualifications_disclose_native_initialization_and_sample_scope():
    assert "backbone tensor identity is neither required nor claimed" in report.QUALIFICATION
    assert "equal sample sizes" in report.QUALIFICATION
    assert "parameter-matched" in report.QUALIFICATION
    assert "physical microbatch" in report.QUALIFICATION
