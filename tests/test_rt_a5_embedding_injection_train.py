"""Bounded checkpoint/contract check for the new injection driver."""
import copy
import hashlib

import numpy as np
import pytest
import torch

from scripts import rt_a5_embedding_injection_train as driver
from scripts import rt_a5_nextlat_train, rt_a5_train, rt_a5_common


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


def test_input_checkpoint_restores_projection_adam_rng_and_rejects_other_variant(tmp_path):
    torch.set_num_threads(1)
    rt_a5_common.configure_fp32_runtime()
    args = driver.parser().parse_args([
        "--data-dir", str(tmp_path), "--output-dir", "unused", "--width", "64",
        "--batch-size", "2", "--seed", "7", "--predictor-seed", "8",
        "--projection-seed", "9"])
    (tmp_path / "manifest.json").write_text('{"fixture":true}\n')
    kwargs = dict(width=64, seed=7, predictor_seed=8, projection_seed=9, backend="naive",
                  predictor_hidden_width=64)
    model = driver.build_model(**kwargs)
    optimizer = driver.make_optimizer(model)
    contract = driver.make_contract(args, model=model, sources={"fixture": "a"},
        data_root=tmp_path, train_x=np.zeros((8, 12), dtype=np.uint8), hardware={"capability": [9, 0]})
    assert contract["n_layers"] == 4 and contract["injection_coefficient"] == .02
    assert contract["variant"] == "input" and contract["projection_seed"] == 9
    assert driver.train_step is rt_a5_nextlat_train.train_step
    assert driver.evaluate_arrays is rt_a5_train.evaluate_arrays
    assert driver.make_optimizer is rt_a5_common.make_optimizer
    x = torch.tensor([[0, 7, 2, 9], [3, 1, 8, 4]])
    y = torch.tensor([[5, 9, 1, 2], [4, 6, 8, 0]])
    driver.train_step(model, optimizer, x, y)
    projection = model.backbone.embedding_projection.weight
    assert projection.grad is not None and torch.count_nonzero(projection.grad)
    path = tmp_path / "step.pt"
    driver.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract, completed=1,
        order_chain=hashlib.sha256(b"fixture-order").hexdigest(),
        initialization=rt_a5_train.json_value(model.nextlat_initialization))
    driver.train_step(model, optimizer, x.flip(1), y.flip(1))
    expected_random = torch.rand(4)
    restored = driver.build_model(**kwargs)
    restored_optimizer = driver.make_optimizer(restored)
    driver.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=contract)
    driver.train_step(restored, restored_optimizer, x.flip(1), y.flip(1))
    assert_tree_equal(restored.state_dict(), model.state_dict())
    assert_tree_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.rand(4), expected_random)
    other = copy.deepcopy(contract)
    other["variant"] = "value"
    before = rt_a5_train.cpu_tree(restored.state_dict())
    with pytest.raises(ValueError, match="contract differs"):
        driver.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=other)
    assert_tree_equal(before, restored.state_dict())
