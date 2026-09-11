"""CPU checks for unchanged A5 harness ownership and new-model exact resume."""
import copy
import hashlib

import numpy as np
import pytest
import torch

import scripts.rt_a5_common as common
import scripts.rt_a5_train as historical
import scripts.rt_a5_reference_train as driver
from scripts.rt_a5_reference_gpt import build_reference_gpt


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    common.configure_fp32_runtime()


def make_model():
    return build_reference_gpt(width=128, seed=17)


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
                "source_sha256": "source", "model_config": model.config}
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
    # Schema-sensitive serialization is deliberately distinct.
    assert driver.SCHEMA != historical.SCHEMA
    assert driver.save_checkpoint is not historical.save_checkpoint


def test_reference_model_uses_historical_update_and_evaluation_without_adaptation():
    model, reference = make_model(), make_model()
    optimizer, reference_optimizer = driver.make_optimizer(model), common.make_optimizer(reference)
    result = driver.train_step(model, optimizer, *batch())
    expected = historical.train_step(reference, reference_optimizer, *batch())
    assert result == expected
    assert_tree_equal(model.state_dict(), reference.state_dict())
    assert_tree_equal(optimizer.state_dict(), reference_optimizer.state_dict())
    inputs, labels = (value.numpy().astype(np.uint8) for value in batch(1))
    evaluated = driver.evaluate_arrays(model, inputs, labels, batch_size=3, device="cpu")
    assert evaluated["rows"] == 5 and evaluated["tokens"] == 20
    assert model.training
    with torch.no_grad(), driver.fp32_context("cpu"):
        logits = model(torch.from_numpy(inputs.astype(np.int64))).logits
        labels_tensor = torch.from_numpy(labels.astype(np.int64))
        correct = logits.argmax(-1).eq(labels_tensor)
    assert evaluated["token_accuracy"] == pytest.approx(correct.double().mean().item())
    assert evaluated["cumulative_prefix_exactness"] == pytest.approx(
        correct.long().cumprod(dim=1).double().mean(dim=0).tolist())


def test_fresh_resume_preserves_model_adam_rng_and_next_updates(tmp_path):
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


def test_initial_checkpoint_has_no_adam_state_and_preserves_initialization(tmp_path):
    model, _, contract, path = saved_fixture(tmp_path, completed=0)
    restored = make_model()
    optimizer = driver.make_optimizer(restored)
    packet = driver.load_checkpoint(path, model=restored, optimizer=optimizer, contract=contract)
    assert not optimizer.state and packet["initialization"] == model.a5_initialization
    assert_tree_equal(restored.state_dict(), model.state_dict())
    with pytest.raises(FileExistsError):
        driver.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                               completed=0, order_chain=packet["order_chain"],
                               initialization=model.a5_initialization)


@pytest.mark.parametrize("corruption", ["schema", "initialization", "model_dtype", "parameter_names",
                                       "moment_nan", "moment_shape", "counter", "missing_state",
                                       "order", "rng"])
def test_bad_resume_rejected_before_model_or_adam_mutation(tmp_path, corruption):
    _, _, contract, path = saved_fixture(tmp_path)
    packet = torch.load(path, weights_only=False)
    first_name = next(iter(packet["model"]))
    first_id = next(iter(packet["optimizer"]["state"]))
    if corruption == "schema":
        packet["schema"] = historical.SCHEMA
    elif corruption == "initialization":
        packet["initialization"]["seed"] = -1
    elif corruption == "model_dtype":
        packet["model"][first_name] = packet["model"][first_name].bfloat16()
    elif corruption == "parameter_names":
        packet["optimizer_parameter_names"][0][0] = "wrong.name"
    elif corruption == "moment_nan":
        packet["optimizer"]["state"][first_id]["exp_avg"].flatten()[0] = float("nan")
    elif corruption == "moment_shape":
        packet["optimizer"]["state"][first_id]["exp_avg"] = torch.zeros(1)
    elif corruption == "counter":
        packet["optimizer"]["state"][first_id]["step"].fill_(2)
    elif corruption == "missing_state":
        del packet["optimizer"]["state"][first_id]
    elif corruption == "order":
        packet["order_chain"] = "not a digest"
    elif corruption == "rng":
        packet["rng"].pop("torch_cpu")
    torch.save(packet, path)
    replacement = make_model()
    before = driver.cpu_tree(replacement.state_dict())
    optimizer = driver.make_optimizer(replacement)
    with pytest.raises(ValueError):
        driver.load_checkpoint(path, model=replacement, optimizer=optimizer, contract=contract)
    assert_tree_equal(replacement.state_dict(), before)
    assert not optimizer.state


def test_changed_model_or_source_contract_is_not_an_exact_resume(tmp_path):
    _, _, contract, path = saved_fixture(tmp_path)
    for key in ("architecture", "source_sha256", "model_config"):
        changed = copy.deepcopy(contract)
        changed[key] = "different"
        model = make_model()
        with pytest.raises(ValueError, match="contract differs"):
            driver.load_checkpoint(path, model=model, optimizer=driver.make_optimizer(model), contract=changed)


def test_new_source_manifest_covers_architecture_without_changing_old_membership():
    old = historical.source_manifest()
    new = driver.source_manifest()
    assert {key: new[key] for key in old} == old
    assert set(new) - set(old) == {"scripts/rt_a5_reference_gpt.py",
                                   "scripts/rt_a5_reference_train.py",
                                   "configs/rt_a5_reference/base.json"}
    assert driver.json_sha256(new) != driver.json_sha256(old)


def test_defaults_are_the_authorized_100k_architecture_comparison():
    args = driver.parser().parse_args(["--data-dir", "data", "--output-dir", "fresh"])
    assert args.architecture == "reference_gpt"
    assert (args.updates, args.batch_size, args.width, args.seed, args.data_order_seed) == (
        100000, 1024, 512, 1234, 1234)
    assert args.checkpoint_steps == [10000, 25000, 50000, 100000]
    assert args.eval_every == 5000 and args.eval_rows == 4096 and args.full_eval_rows == 102400
