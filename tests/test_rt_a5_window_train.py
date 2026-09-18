"""Bounded CPU checks for Mitchell/position/window pilot harness and strict resume."""
import copy
import hashlib

import numpy as np
import pytest
import torch

from scripts import rt_a5_common, rt_a5_nextlat_train, rt_a5_train
from scripts import rt_a5_window_train as variant
from scripts import rt_a5_nextlat_variant_train as previous_variant
from scripts.rt_a5_window import build_window_model


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    rt_a5_common.configure_fp32_runtime()


def build_fixture_model(*, position_encoding="sinusoidal", second_layer_window=None):
    return build_window_model("rt", width=128, seed=7, predictor_seed=8,
                              device="cpu", backend="naive",
                              position_encoding=position_encoding,
                              second_layer_window=second_layer_window)


def batch(index=0):
    inputs = (torch.arange(8).reshape(2, 4) + 3 * index) % 60
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


def checkpoint_fixture(tmp_path):
    model = build_fixture_model()
    optimizer = variant.make_optimizer(model)
    variant.train_step(model, optimizer, *batch())
    contract = {"batch_size": 2, "source_sha256": "original", "architecture": "rt",
                "objective": {"latent_weight": 1.0}, "predictor_seed": 8,
                "experiment_config": copy.deepcopy(model.experiment_config)}
    path = tmp_path / "checkpoint.pt"
    record = variant.save_checkpoint(
        path, model=model, optimizer=optimizer, contract=contract, completed=1,
        order_chain=hashlib.sha256(b"ordered rows").hexdigest(),
        initialization=rt_a5_train.json_value(model.nextlat_initialization))
    return model, optimizer, contract, path, record


def test_update_evaluation_order_and_optimizer_are_original_function_objects():
    assert variant.train_step is rt_a5_nextlat_train.train_step
    assert variant.evaluate_diagnostics is rt_a5_nextlat_train.evaluate_diagnostics
    assert variant.evaluate_arrays is rt_a5_train.evaluate_arrays
    assert variant.WordOrder is rt_a5_train.WordOrder
    assert variant.make_optimizer is rt_a5_common.make_optimizer
    assert variant._validate_optimizer is rt_a5_nextlat_train._validate_optimizer


def test_contract_records_experiment_and_preserves_baseline_recipe(tmp_path):
    model = build_fixture_model()
    (tmp_path / "manifest.json").write_text('{"fixture": true}\n')
    args = variant.parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", "new"])
    args.width, args.seed, args.predictor_seed = 128, 7, 8
    kwargs = dict(model=model, sources={"fixture.py": "abc"}, data_root=tmp_path,
                  train_x=np.zeros((8, 12), dtype=np.uint8), hardware={"capability": [9, 0]})
    original = rt_a5_nextlat_train.make_contract(args, **kwargs)
    actual = variant.make_contract(args, **kwargs)
    assert actual["schema"] == variant.SCHEMA != original["schema"]
    assert actual["experiment_config"] == rt_a5_train.json_value(model.experiment_config)
    for key in original.keys() - {"schema"}:
        assert actual[key] == original[key]
    assert actual["evaluation_route"] == "backbone_only"
    assert actual["latent_rollout_evaluated"] is False


def test_resume_exactly_recovers_model_adam_rng_and_fixed_positions(tmp_path):
    model, optimizer, contract, path, record = checkpoint_fixture(tmp_path)
    assert record["completed_updates"] == 1 and record["examples_seen"] == 2
    assert record["bytes"] > 0 and len(record["sha256"]) == 64
    variant.train_step(model, optimizer, *batch(1))
    expected_draw = torch.rand(5)
    restored = build_fixture_model()
    restored_optimizer = variant.make_optimizer(restored)
    packet = variant.load_checkpoint(path, model=restored, optimizer=restored_optimizer,
                                     contract=contract)
    assert packet["schema"] == variant.SCHEMA
    variant.train_step(restored, restored_optimizer, *batch(1))
    assert_tree_equal(restored.state_dict(), model.state_dict())
    assert_tree_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.rand(5), expected_draw)
    with pytest.raises(FileExistsError):
        variant.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                                completed=2, order_chain=packet["order_chain"],
                                initialization=model.nextlat_initialization)


@pytest.mark.parametrize("change", ["experiment", "source", "objective", "old_schema"])
def test_resume_rejects_changed_experiment_before_mutation(tmp_path, change):
    _, _, contract, path, _ = checkpoint_fixture(tmp_path)
    changed = copy.deepcopy(contract)
    if change == "experiment":
        changed["experiment_config"]["different_experiment"] = True
    elif change == "source":
        changed["source_sha256"] = "changed"
    elif change == "objective":
        changed["objective"]["latent_weight"] = 0.5
    else:
        packet = torch.load(path, weights_only=False)
        packet["schema"] = rt_a5_nextlat_train.SCHEMA
        torch.save(packet, path)
    replacement = build_fixture_model()
    optimizer = variant.make_optimizer(replacement)
    before = rt_a5_train.cpu_tree(replacement.state_dict())
    with pytest.raises(ValueError, match="contract differs"):
        variant.load_checkpoint(path, model=replacement, optimizer=optimizer, contract=changed)
    assert_tree_equal(replacement.state_dict(), before)
    assert not optimizer.state


@pytest.mark.parametrize("corruption", ["model_dtype", "moment_nan", "missing_moment",
                                       "counter", "initialization", "rng"])
def test_resume_rejects_corrupt_state_before_mutation(tmp_path, corruption):
    _, _, contract, path, _ = checkpoint_fixture(tmp_path)
    packet = torch.load(path, weights_only=False)
    first_name = next(iter(packet["model"]))
    first_id = next(iter(packet["optimizer"]["state"]))
    state = packet["optimizer"]["state"][first_id]
    if corruption == "model_dtype":
        packet["model"][first_name] = packet["model"][first_name].bfloat16()
    elif corruption == "moment_nan":
        state["exp_avg"].flatten()[0] = float("nan")
    elif corruption == "missing_moment":
        packet["optimizer"]["state"].pop(first_id)
    elif corruption == "counter":
        state["step"].fill_(2)
    elif corruption == "initialization":
        packet["initialization"]["predictor_seed"] = 100
    elif corruption == "rng":
        packet["rng"].pop("torch_cpu")
    torch.save(packet, path)
    replacement = build_fixture_model()
    optimizer = variant.make_optimizer(replacement)
    before = rt_a5_train.cpu_tree(replacement.state_dict())
    with pytest.raises(ValueError):
        variant.load_checkpoint(path, model=replacement, optimizer=optimizer, contract=contract)
    assert_tree_equal(replacement.state_dict(), before)
    assert not optimizer.state


def test_source_manifest_preserves_closed_nextlat_identity_and_covers_new_files():
    original = rt_a5_nextlat_train.source_manifest()
    expanded = variant.source_manifest()
    assert rt_a5_train.json_sha256(original) == (
        "1e6d0c63289f01525bc0c19bba6b2646d61df10ddb815bc74f5c111b46f9f961")
    assert {key: expanded[key] for key in original} == original
    previous = previous_variant.source_manifest()
    assert len(original) == 46 and len(previous) == 49 and len(expanded) == 52
    assert {key: expanded[key] for key in previous} == previous
    assert {"scripts/rt_a5_window.py", "scripts/rt_a5_window_train.py",
            "configs/rt_a5_window/base.json"}.issubset(expanded)
    assert rt_a5_train.json_sha256(expanded) != rt_a5_train.json_sha256(original)


def test_default_protocol_matches_rt_nextlat_pilot_and_has_no_rollout_mode():
    args = variant.parser().parse_args(["--data-dir", "data", "--output-dir", "new"])
    assert (args.architecture, args.width, args.updates, args.batch_size, args.seed,
            args.predictor_seed, args.data_order_seed, args.latent_weight) == (
                "rt", 512, 10000, 1024, 1234, 1235, 1234, 1.0)
    assert args.position_encoding == "sinusoidal" and args.second_layer_window is None
    assert args.checkpoint_steps == [1000, 5000, 10000]
    assert (args.eval_every, args.eval_rows, args.full_eval_rows) == (500, 4096, 102400)
    assert not any("rollout" in key for key in vars(args))
    with pytest.raises(SystemExit):
        variant.parser().parse_args(["--data-dir", "data", "--output-dir", "new",
                                     "--architecture", "seq"])


@pytest.mark.parametrize("position_encoding", ["alibi", "sinusoidal"])
def test_window_cli_and_checkpoint_contract_separate_model_choices(tmp_path, position_encoding):
    args = variant.parser().parse_args([
        "--data-dir", str(tmp_path), "--output-dir", "new",
        "--position-encoding", position_encoding, "--second-layer-window", "2"])
    assert args.position_encoding == position_encoding and args.second_layer_window == 2
    model = build_fixture_model(position_encoding=position_encoding, second_layer_window=2)
    optimizer = variant.make_optimizer(model)
    variant.train_step(model, optimizer, *batch())
    (tmp_path / "manifest.json").write_text('{"fixture": true}\n')
    args.width, args.seed, args.predictor_seed = 128, 7, 8
    kwargs = dict(sources={"fixture.py": "abc"}, data_root=tmp_path,
                  train_x=np.zeros((8, 12), dtype=np.uint8), hardware={"capability": [9, 0]})
    contract = variant.make_contract(args, model=model, **kwargs)
    path = tmp_path / "window.pt"
    variant.save_checkpoint(
        path, model=model, optimizer=optimizer, contract=contract, completed=1,
        order_chain=hashlib.sha256(b"ordered rows").hexdigest(),
        initialization=rt_a5_train.json_value(model.nextlat_initialization))
    replacement = build_fixture_model(position_encoding=position_encoding, second_layer_window=2)
    replacement_optimizer = variant.make_optimizer(replacement)
    variant.load_checkpoint(path, model=replacement, optimizer=replacement_optimizer,
                            contract=contract)
    assert_tree_equal(replacement.state_dict(), model.state_dict())
    assert_tree_equal(replacement_optimizer.state_dict(), optimizer.state_dict())
    for alternate in [
            build_fixture_model(position_encoding=position_encoding),
            build_fixture_model(position_encoding=("alibi" if position_encoding == "sinusoidal"
                                                    else "sinusoidal"), second_layer_window=2)]:
        alternate_contract = variant.make_contract(args, model=alternate, **kwargs)
        assert alternate_contract != contract
        alternate_optimizer = variant.make_optimizer(alternate)
        before = rt_a5_train.cpu_tree(alternate.state_dict())
        with pytest.raises(ValueError, match="contract differs"):
            variant.load_checkpoint(path, model=alternate, optimizer=alternate_optimizer,
                                    contract=alternate_contract)
        assert_tree_equal(alternate.state_dict(), before)
        assert not alternate_optimizer.state


@pytest.mark.parametrize("value", ["0", "1", "3", "-2"])
def test_cli_rejects_unapproved_window_lengths(value):
    with pytest.raises(SystemExit):
        variant.parser().parse_args(["--data-dir", "data", "--output-dir", "new",
                                     "--second-layer-window", value])
