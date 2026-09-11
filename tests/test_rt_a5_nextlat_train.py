"""CPU tests for the hybrid trainer's update, evaluation and resume contracts."""
import copy
import hashlib

import numpy as np
import pytest
import torch

from scripts.rt_a5_common import build_model, configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_nextlat import build_nextlat_model, nextlat_objective
from scripts.rt_a5_nextlat_train import (
    SCHEMA, TRAIN_METRICS, evaluate_diagnostics, load_checkpoint, parser,
    save_checkpoint, source_manifest, train_step,
)
from scripts.rt_a5_train import (
    cpu_tree, evaluate_arrays, json_sha256, source_manifest as base_source_manifest,
    train_step as original_train_step,
)


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    configure_fp32_runtime()


def model_pair():
    return tuple(build_nextlat_model("seq", width=128, seed=7, predictor_seed=8)
                 for _ in range(2))


def batch(index=0):
    # Fixed distinct words, including identity; target values need not be easy.
    inputs = (torch.arange(12).reshape(3, 4) + 3 * index) % 60
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
    model, _ = model_pair()
    optimizer = make_optimizer(model)
    train_step(model, optimizer, *batch())
    contract = {"batch_size": 3, "source_sha256": "original", "architecture": "seq",
                "objective": {"latent_weight": 1.0}, "predictor_seed": 8}
    path = tmp_path / "checkpoint.pt"
    record = save_checkpoint(
        path, model=model, optimizer=optimizer, contract=contract, completed=1,
        order_chain=hashlib.sha256(b"ordered rows").hexdigest(),
        initialization=model.nextlat_initialization)
    return model, optimizer, contract, path, record


def test_train_step_matches_one_combined_backward_clip_and_update():
    actual, reference = model_pair()
    actual_optimizer, reference_optimizer = make_optimizer(actual), make_optimizer(reference)
    x, y = batch()
    values = train_step(actual, actual_optimizer, x, y, diagnostics=True)
    reference_optimizer.zero_grad(set_to_none=True)
    with fp32_context("cpu"):
        objective = nextlat_objective(reference, x, y)
        objective["loss"].backward()
    norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    reference_optimizer.step()
    assert_tree_equal(actual.state_dict(), reference.state_dict())
    assert_tree_equal(actual_optimizer.state_dict(), reference_optimizer.state_dict())
    assert values["loss"] == pytest.approx(values["state_loss"] + values["latent_loss"])
    assert values["grad_norm"] == norm.item()
    assert set(TRAIN_METRICS).issubset(values)
    assert "predicted_state_ce" in values and "latent_rms" in values
    assert all(isinstance(value, float) for value in values.values())
    assert len(actual_optimizer.state) == len(list(actual.parameters()))
    # A second update must clear rather than retain previous parameter gradients.
    train_step(actual, actual_optimizer, *batch(1), diagnostics=False)
    train_step(reference, reference_optimizer, *batch(1), diagnostics=False)
    assert_tree_equal(actual.state_dict(), reference.state_dict())


def test_zero_auxiliary_recovers_original_update_without_predictor_adam_state():
    hybrid = build_nextlat_model("seq", width=128, seed=7, predictor_seed=8)
    plain = build_model("seq", width=128, seed=7)
    before_predictor = cpu_tree(hybrid.predictor.state_dict())
    hybrid_optimizer, plain_optimizer = make_optimizer(hybrid), make_optimizer(plain)
    values = train_step(hybrid, hybrid_optimizer, *batch(), latent_weight=0.0, diagnostics=True)
    original = original_train_step(plain, plain_optimizer, *batch())
    assert_tree_equal(hybrid.backbone.state_dict(), plain.state_dict())
    assert_tree_equal(hybrid.predictor.state_dict(), before_predictor)
    assert values["loss"] == original["loss"] and values["latent_loss"] == 0.0
    assert all(parameter not in hybrid_optimizer.state for parameter in hybrid.predictor.parameters())
    assert "latent_rms" not in values


def test_backbone_evaluation_never_calls_predictor_and_diagnostics_restore_mode():
    model, _ = model_pair()
    x, y = (value.numpy().astype(np.uint8) for value in batch())
    calls = []
    handle = model.predictor.register_forward_hook(lambda *_: calls.append(1))
    try:
        before = cpu_tree(model.state_dict())
        metrics = evaluate_arrays(model, x, y, batch_size=2, device="cpu")
        assert metrics["rows"] == 3 and metrics["tokens"] == 12 and calls == []
        assert model.training
        probe = evaluate_diagnostics(model, x, y, rows=2, device="cpu")
        assert calls == [1] and probe["rows"] == 2
        assert probe["route"] == "teacher_conditioned_one_step_diagnostics"
        assert "predicted_state_ce" in probe["diagnostics"] and model.training
        assert_tree_equal(before, model.state_dict())
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        handle.remove()


def test_resume_matches_uninterrupted_hybrid_model_adam_and_rng(tmp_path):
    model, optimizer, contract, path, record = checkpoint_fixture(tmp_path)
    assert record["bytes"] > 0 and len(record["sha256"]) == 64
    for index in (1, 2):
        train_step(model, optimizer, *batch(index))
    expected_draw = torch.rand(5)
    restored, _ = model_pair()
    restored_optimizer = make_optimizer(restored)
    packet = load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=contract)
    assert packet["schema"] == SCHEMA and packet["completed_updates"] == 1
    assert packet["examples_seen"] == 3
    for index in (1, 2):
        train_step(restored, restored_optimizer, *batch(index))
    assert_tree_equal(restored.state_dict(), model.state_dict())
    assert_tree_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.rand(5), expected_draw)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                        completed=3, order_chain=packet["order_chain"],
                        initialization=model.nextlat_initialization)


@pytest.mark.parametrize("corruption", ["model_dtype", "model_shape", "optimizer_names",
                                       "moment_nan", "moment_dtype", "moment_shape",
                                       "moment_negative", "counter", "missing_moment",
                                       "initialization", "order_chain", "rng"])
def test_resume_rejects_corruption_before_mutating_model(tmp_path, corruption):
    _, _, contract, path, _ = checkpoint_fixture(tmp_path)
    packet = torch.load(path, weights_only=False)
    first_name = next(iter(packet["model"]))
    first_id = next(iter(packet["optimizer"]["state"]))
    state = packet["optimizer"]["state"][first_id]
    if corruption == "model_dtype":
        packet["model"][first_name] = packet["model"][first_name].bfloat16()
    elif corruption == "model_shape":
        packet["model"][first_name] = packet["model"][first_name].reshape(-1)[:1]
    elif corruption == "optimizer_names":
        packet["optimizer_parameter_names"][0][0] = "predictor.wrong_parameter"
    elif corruption == "moment_nan":
        state["exp_avg"].flatten()[0] = float("nan")
    elif corruption == "moment_dtype":
        state["exp_avg"] = state["exp_avg"].bfloat16()
    elif corruption == "moment_shape":
        state["exp_avg"] = state["exp_avg"].reshape(-1)[:1]
    elif corruption == "moment_negative":
        state["exp_avg_sq"].flatten()[0] = -1
    elif corruption == "counter":
        state["step"].fill_(2)
    elif corruption == "missing_moment":
        del packet["optimizer"]["state"][first_id]
    elif corruption == "initialization":
        packet["initialization"]["predictor_seed"] += 1
    elif corruption == "order_chain":
        packet["order_chain"] = "not a digest"
    elif corruption == "rng":
        packet["rng"].pop("torch_cpu")
    torch.save(packet, path)
    replacement, _ = model_pair()
    replacement_optimizer = make_optimizer(replacement)
    before = cpu_tree(replacement.state_dict())
    with pytest.raises(ValueError):
        load_checkpoint(path, model=replacement, optimizer=replacement_optimizer, contract=contract)
    assert_tree_equal(replacement.state_dict(), before)
    assert not replacement_optimizer.state


def test_checkpoint_contract_rejects_objective_and_predictor_changes(tmp_path):
    _, _, contract, path, _ = checkpoint_fixture(tmp_path)
    replacement, _ = model_pair()
    for changed in (dict(contract, predictor_seed=9),
                    dict(contract, objective={"latent_weight": 0.5}),
                    dict(contract, source_sha256="different")):
        with pytest.raises(ValueError, match="contract differs"):
            load_checkpoint(path, model=replacement, optimizer=make_optimizer(replacement), contract=changed)


def test_expanded_source_manifest_keeps_historical_sources_and_new_coverage():
    historical = base_source_manifest()
    expanded = source_manifest()
    assert {key: expanded[key] for key in historical} == historical
    assert set(expanded) > set(historical)
    assert {"scripts/rt_a5_nextlat.py", "scripts/rt_a5_nextlat_train.py",
            "configs/rt_a5_nextlat/base.json"}.issubset(expanded)
    assert json_sha256(expanded) != json_sha256(historical)


def test_training_defaults_match_pilot_and_have_no_rollout_mode():
    args = parser().parse_args(["--data-dir", "data", "--output-dir", "new", "--architecture", "rt"])
    assert (args.updates, args.batch_size, args.seed, args.predictor_seed,
            args.data_order_seed, args.latent_weight) == (10000, 1024, 1234, 1235, 1234, 1.0)
    assert args.checkpoint_steps == [1000, 5000, 10000]
    assert args.eval_rows == 4096 and args.full_eval_rows == 102400
    assert not any("rollout" in key for key in vars(args))
