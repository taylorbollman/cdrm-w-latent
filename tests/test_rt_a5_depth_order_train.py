"""Bounded CPU integration checks for the new depth/order training identities."""
import copy
import hashlib

import numpy as np
import pytest
import torch

from scripts import rt_a5_common, rt_a5_nextlat_train, rt_a5_train, rt_a5_window_train
from scripts import rt_a5_depth_order_train as driver
from scripts.rt_a5_depth_order import build_model


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    rt_a5_common.configure_fp32_runtime()


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_original_update_evaluator_optimizer_and_order_are_reused():
    assert driver.train_step is rt_a5_nextlat_train.train_step
    assert driver.evaluate_diagnostics is rt_a5_nextlat_train.evaluate_diagnostics
    assert driver.evaluate_arrays is rt_a5_train.evaluate_arrays
    assert driver.WordOrder is rt_a5_train.WordOrder
    assert driver.make_optimizer is rt_a5_common.make_optimizer


@pytest.mark.parametrize("variant", ("seq4_alibi", "rt_window2_first"))
def test_changed_architecture_has_exact_resume_and_explicit_contract(tmp_path, variant):
    args = driver.parser().parse_args([
        "--variant", variant, "--data-dir", str(tmp_path), "--output-dir", "unused"])
    args.width, args.seed, args.predictor_seed, args.batch_size = 128, 7, 8, 2
    (tmp_path / "manifest.json").write_text('{"fixture":true}\n')
    model = build_model(variant, width=128, seed=7, predictor_seed=8, backend="naive")
    optimizer = driver.make_optimizer(model)
    contract = driver.make_contract(
        args, model=model, sources={"fixture.py": "abc"}, data_root=tmp_path,
        train_x=np.zeros((8, 12), dtype=np.uint8), hardware={"capability": [9, 0]})
    assert contract["variant"] == variant
    assert contract["architecture"] == driver.VARIANT_ARCHITECTURES[variant]
    assert contract["model_config"]["n_layers"] == (4 if variant == "seq4_alibi" else 2)
    assert contract["experiment_config"] == rt_a5_train.json_value(model.experiment_config)
    assert contract["objective"]["latent_weight"] == 1.0
    assert contract["precision"] == "fp32" and contract["evaluation_route"] == "backbone_only"
    tokens = torch.tensor([[0, 7, 2, 9], [3, 1, 8, 4]])
    targets = torch.tensor([[5, 9, 1, 2], [4, 6, 8, 0]])
    driver.train_step(model, optimizer, tokens, targets)
    path = tmp_path / "step.pt"
    driver.save_checkpoint(
        path, model=model, optimizer=optimizer, contract=contract, completed=1,
        order_chain=hashlib.sha256(b"fixture-order").hexdigest(),
        initialization=rt_a5_train.json_value(model.nextlat_initialization))
    driver.train_step(model, optimizer, tokens.flip(1), targets.flip(1))
    expected_random = torch.rand(4)
    restored = build_model(variant, width=128, seed=7, predictor_seed=8, backend="naive")
    restored_optimizer = driver.make_optimizer(restored)
    driver.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=contract)
    driver.train_step(restored, restored_optimizer, tokens.flip(1), targets.flip(1))
    assert_tree_equal(restored.state_dict(), model.state_dict())
    assert_tree_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.rand(4), expected_random)
    changed = copy.deepcopy(contract)
    changed["variant"] = "different-depth-or-window-order"
    before = rt_a5_train.cpu_tree(restored.state_dict())
    with pytest.raises(ValueError, match="contract differs"):
        driver.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=changed)
    assert_tree_equal(restored.state_dict(), before)


def test_shared_manifest_preserves_frozen_52_sources_and_covers_pair():
    old, new = rt_a5_window_train.source_manifest(), driver.source_manifest()
    assert len(old) == 52 and len(new) == 55
    assert rt_a5_train.json_sha256(old) == "9d12d61e994bdad57567cb1d30fd36f1ea3296c94f7aa401d574ecbb34d339c2"
    assert {key: new[key] for key in old} == old
    assert set(new) - set(old) == {
        "scripts/rt_a5_depth_order.py", "scripts/rt_a5_depth_order_train.py",
        "configs/rt_a5_depth_order/base.json"}


def test_default_budget_and_invalid_variant():
    args = driver.parser().parse_args([
        "--variant", "seq4_alibi", "--data-dir", "data", "--output-dir", "unused"])
    assert (args.updates, args.width, args.batch_size, args.seed, args.predictor_seed,
            args.data_order_seed, args.latent_weight) == (80000, 512, 1024, 1234, 1235, 1234, 1.0)
    assert (args.eval_every, args.eval_rows, args.full_eval_rows) == (500, 4096, 102400)
    assert args.checkpoint_steps[-1] == 80000
    assert not any("rollout" in key for key in vars(args))
    with pytest.raises(SystemExit):
        driver.parser().parse_args([
            "--variant", "rt_window2_second", "--data-dir", "data", "--output-dir", "unused"])
