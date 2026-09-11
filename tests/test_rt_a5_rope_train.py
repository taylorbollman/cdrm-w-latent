"""Bounded CPU checks for the paired RoPE arm and unchanged A5 harness."""
import hashlib

import numpy as np
import pytest
import torch

import scripts.rt_a5_common as common
import scripts.rt_a5_train as historical
import scripts.rt_a5_rope_train as driver
from scripts.rt_a5_rope import build_rope_model


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    common.configure_fp32_runtime()


def make_model():
    return build_rope_model(width=128, seed=17)


def batch(index=0):
    inputs = (torch.arange(20).reshape(5, 4) + 3 * index) % 60
    labels = (inputs.cumsum(dim=1) + index) % 60
    return inputs, labels


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def saved_fixture(tmp_path, completed=1):
    model = make_model()
    optimizer = driver.make_optimizer(model)
    for index in range(completed):
        driver.train_step(model, optimizer, *batch(index))
    contract = {"batch_size": 5, "architecture": driver.ARCHITECTURE,
                "source_sha256": "source", "model_config": driver.json_value(model.config)}
    path = tmp_path / "state.pt"
    driver.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                           completed=completed, order_chain=hashlib.sha256(b"ordered words").hexdigest(),
                           initialization=model.a5_initialization)
    return model, optimizer, contract, path


def test_historical_harness_functions_are_identical_objects():
    for name in ("WordOrder", "batch_tensors", "train_step", "evaluate_arrays",
                 "cpu_tree", "rng_state", "restore_rng", "preserve_rng", "optimizer_names",
                 "validate_model_state"):
        assert getattr(driver, name) is getattr(historical, name), name
    for name in ("make_optimizer", "task_loss", "fp32_context", "configure_fp32_runtime"):
        assert getattr(driver, name) is getattr(common, name), name
    assert driver.SCHEMA == "rt-a5-rope-training-v1"
    assert driver.SCHEMA != historical.SCHEMA


def test_initialization_is_exactly_paired_and_only_position_config_differs():
    model = make_model()
    baseline = common.build_model("seq", width=128, seed=17)
    assert_tree_equal(model.state_dict(), baseline.state_dict())
    assert common.canonical_parameter_sha256(model) == baseline.a5_initialization["canonical_sha256"]
    config, original = driver.json_value(model.config), driver.json_value(baseline.config)
    differences = {key: (original[key], config[key]) for key in original if original[key] != config[key]}
    assert differences == {"alibi": (True, False), "rope": (False, True)}


def test_rope_model_uses_historical_update_and_evaluation_without_adaptation():
    model, reference = make_model(), make_model()
    optimizer, reference_optimizer = driver.make_optimizer(model), common.make_optimizer(reference)
    assert driver.train_step(model, optimizer, *batch()) == historical.train_step(
        reference, reference_optimizer, *batch())
    assert_tree_equal(model.state_dict(), reference.state_dict())
    assert_tree_equal(optimizer.state_dict(), reference_optimizer.state_dict())
    inputs, labels = (value.numpy().astype(np.uint8) for value in batch(1))
    evaluated = driver.evaluate_arrays(model, inputs, labels, batch_size=3, device="cpu")
    assert evaluated["rows"] == 5 and evaluated["tokens"] == 20 and model.training


def test_resume_preserves_model_adam_rng_and_next_updates(tmp_path):
    model, optimizer, contract, path = saved_fixture(tmp_path)
    for index in (1, 2):
        driver.train_step(model, optimizer, *batch(index))
    expected_draw = torch.rand(7)
    replacement = make_model()
    replacement_optimizer = driver.make_optimizer(replacement)
    packet = driver.load_checkpoint(path, model=replacement, optimizer=replacement_optimizer,
                                    contract=contract)
    assert packet["completed_updates"] == 1 and packet["examples_seen"] == 5
    for index in (1, 2):
        driver.train_step(replacement, replacement_optimizer, *batch(index))
    assert_tree_equal(replacement.state_dict(), model.state_dict())
    assert_tree_equal(replacement_optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.rand(7), expected_draw)


def test_initial_checkpoint_roundtrip_has_no_adam_state(tmp_path):
    model, _, contract, path = saved_fixture(tmp_path, completed=0)
    restored = make_model()
    optimizer = driver.make_optimizer(restored)
    packet = driver.load_checkpoint(path, model=restored, optimizer=optimizer, contract=contract)
    assert not optimizer.state and packet["initialization"] == model.a5_initialization
    assert_tree_equal(restored.state_dict(), model.state_dict())


@pytest.mark.parametrize("corruption", ["family", "initialization", "source", "moment_nan", "counter"])
def test_invalid_resume_rejected_before_mutation(tmp_path, corruption):
    _, _, contract, path = saved_fixture(tmp_path)
    packet = torch.load(path, weights_only=False)
    first_id = next(iter(packet["optimizer"]["state"]))
    if corruption == "family":
        packet["schema"] = "rt-a5-reference-gpt-training-v1"
    elif corruption == "initialization":
        packet["initialization"]["seed"] = -1
    elif corruption == "source":
        packet["contract"]["source_sha256"] = "different"
    elif corruption == "moment_nan":
        packet["optimizer"]["state"][first_id]["exp_avg"].flatten()[0] = float("nan")
    elif corruption == "counter":
        packet["optimizer"]["state"][first_id]["step"].fill_(2)
    torch.save(packet, path)
    replacement = make_model()
    before = driver.cpu_tree(replacement.state_dict())
    optimizer = driver.make_optimizer(replacement)
    with pytest.raises(ValueError):
        driver.load_checkpoint(path, model=replacement, optimizer=optimizer, contract=contract)
    assert_tree_equal(replacement.state_dict(), before)
    assert not optimizer.state


def test_source_manifest_adds_only_new_rope_execution_files():
    old, new = historical.source_manifest(), driver.source_manifest()
    assert {key: new[key] for key in old} == old
    assert set(new) - set(old) == {"scripts/rt_a5_rope.py", "scripts/rt_a5_rope_train.py",
                                   "configs/rt_a5_rope/base.json"}


def test_defaults_are_the_authorized_50k_position_comparison():
    args = driver.parser().parse_args(["--data-dir", "data", "--output-dir", "fresh"])
    assert args.architecture == "seq_rope"
    assert (args.updates, args.batch_size, args.width, args.seed, args.data_order_seed) == (
        50000, 1024, 512, 1234, 1234)
    assert args.checkpoint_steps == [10000, 25000, 50000]
    assert args.eval_every == 5000 and args.eval_rows == 4096 and args.full_eval_rows == 102400
