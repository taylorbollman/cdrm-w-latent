"""Pure-CE function reuse, exact CPU resume, and complete optimizer validation."""
import copy
import hashlib
import random

import numpy as np
import pytest
import torch

from scripts import rt_a5_common, rt_a5_depth_order_train, rt_a5_train
from scripts import rt_a5_window_control_train as driver
from scripts.rt_a5_window_control import build_model


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    rt_a5_common.configure_fp32_runtime()


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            tree_equal(a, b)
    else:
        assert left == right


def fixture_model(tmp_path):
    args = driver.parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", "unused"])
    args.width, args.seed, args.batch_size = 128, 7, 2
    (tmp_path / "manifest.json").write_text('{"fixture":true}\n')
    model = build_model(width=128, seed=7, backend="naive")
    optimizer = driver.make_optimizer(model)
    contract = driver.make_contract(args, model=model, sources={"fixture.py": "abc"},
        data_root=tmp_path, train_x=np.zeros((8, 12), dtype=np.uint8), hardware={"capability": [9, 0]})
    return model, optimizer, contract


def test_original_pure_ce_step_evaluator_optimizer_and_order_are_same_function_objects():
    assert driver.train_step is rt_a5_train.train_step
    assert driver.evaluate_arrays is rt_a5_train.evaluate_arrays
    assert driver.WordOrder is rt_a5_train.WordOrder
    assert driver.make_optimizer is rt_a5_common.make_optimizer
    assert not hasattr(driver, "evaluate_diagnostics")


def test_exact_cpu_resume_and_explicit_ce_only_contract(tmp_path):
    model, optimizer, contract = fixture_model(tmp_path)
    assert contract["objective"] == {"state_loss": "same-position CE mean over B*T", "latent_weight": 0.0,
                                    "kl_weight": 0.0, "predicted_state_ce_weight": 0.0}
    assert contract["variant"] == "rt_window2_first_ce" and contract["architecture"] == "rt"
    assert not contract["nextlat_enabled"] and not contract["predictor_registered"]
    assert contract["experiment_config"]["window_layer"] == 0
    tokens = torch.tensor([[0, 7, 2, 9], [3, 1, 8, 4]])
    targets = torch.tensor([[5, 9, 1, 2], [4, 6, 8, 0]])
    metrics = driver.train_step(model, optimizer, tokens, targets)
    assert set(metrics) == {"loss", "token_accuracy", "whole_word_exact", "grad_norm"}
    path = tmp_path / "step.pt"
    driver.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract, completed=1,
        order_chain=hashlib.sha256(b"fixture-order").hexdigest(), initialization=rt_a5_train.json_value(model.a5_initialization))
    driver.train_step(model, optimizer, tokens.flip(1), targets.flip(1))
    expected_random = (random.random(), np.random.rand(), torch.rand(4))
    restored = build_model(width=128, seed=7, backend="naive")
    restored_optimizer = driver.make_optimizer(restored)
    driver.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=contract)
    driver.train_step(restored, restored_optimizer, tokens.flip(1), targets.flip(1))
    tree_equal(model.state_dict(), restored.state_dict())
    tree_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    assert (random.random(), np.random.rand()) == expected_random[:2]
    assert torch.equal(torch.rand(4), expected_random[2])
    assert len(restored_optimizer.state) == 21
    assert all(state["step"].item() == 2 for state in restored_optimizer.state.values())
    changed = copy.deepcopy(contract)
    changed["objective"]["latent_weight"] = 1.0
    before = rt_a5_train.cpu_tree(restored.state_dict())
    with pytest.raises(ValueError, match="contract differs"):
        driver.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=changed)
    tree_equal(restored.state_dict(), before)


@pytest.mark.parametrize("corruption", ["missing_state", "stale_counter", "wrong_initialization"])
def test_bad_checkpoint_rejected_before_model_or_optimizer_mutation(tmp_path, corruption):
    model, optimizer, contract = fixture_model(tmp_path)
    tokens = torch.tensor([[0, 7, 2, 9], [3, 1, 8, 4]])
    for _ in range(2):
        driver.train_step(model, optimizer, tokens, tokens)
    path = tmp_path / "step.pt"
    driver.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract, completed=2,
        order_chain=hashlib.sha256(b"fixture-order").hexdigest(), initialization=rt_a5_train.json_value(model.a5_initialization))
    packet = torch.load(path, map_location="cpu", weights_only=False)
    key = next(iter(packet["optimizer"]["state"]))
    if corruption == "missing_state":
        packet["optimizer"]["state"].pop(key)
    elif corruption == "stale_counter":
        packet["optimizer"]["state"][key]["step"].fill_(1)
    else:
        packet["initialization"]["exact_reference_backbone_initialization"] = False
    corrupted = tmp_path / "corrupted.pt"
    torch.save(packet, corrupted)
    fresh = build_model(width=128, seed=7, backend="naive")
    fresh_optimizer = driver.make_optimizer(fresh)
    before_model, before_adam = rt_a5_train.cpu_tree(fresh.state_dict()), rt_a5_train.cpu_tree(fresh_optimizer.state_dict())
    with pytest.raises(ValueError, match="optimizer state/counter|initialization differs"):
        driver.load_checkpoint(corrupted, model=fresh, optimizer=fresh_optimizer, contract=contract)
    tree_equal(fresh.state_dict(), before_model)
    tree_equal(fresh_optimizer.state_dict(), before_adam)


def test_source58_preserves_frozen55_and_recipe_defaults():
    old, new = rt_a5_depth_order_train.source_manifest(), driver.source_manifest()
    assert len(old) == 55 and len(new) == 58
    assert rt_a5_train.json_sha256(old) == "cc9fbfa50ebd57387f4ad95803bb2db9d42b69a13875268c0400759858fe4176"
    assert {key: new[key] for key in old} == old
    assert set(new) - set(old) == {"scripts/rt_a5_window_control.py", "scripts/rt_a5_window_control_train.py",
                                 "configs/rt_a5_window_control/base.json"}
    args = driver.parser().parse_args(["--data-dir", "data", "--output-dir", "unused"])
    assert (args.variant, args.architecture, args.updates, args.width, args.batch_size, args.seed,
            args.data_order_seed) == ("rt_window2_first_ce", "rt", 80000, 512, 1024, 1234, 1234)
    assert (args.eval_every, args.eval_rows, args.full_eval_rows) == (500, 4096, 102400)
    assert args.checkpoint_steps == [1000, 5000, 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000]
    assert not any("latent" in key or "predictor" in key or "diagnostic" in key for key in vars(args))
