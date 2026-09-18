"""CPU conservative schedule, exact initialization, tiny first-step gradient and resume checks."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import rt_a5_common, rt_a5_nextlat_train, rt_a5_train
from scripts import rt_a5_conservative_input_train as driver
from scripts.rt_a5_embedding_injection import build_model as fixed_model
from scripts.rt_a5_nextlat import _parameter_sha256
from scripts.rt_a5_conservative_input import SCHEDULE, build_model, coefficient, set_update


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    rt_a5_common.configure_fp32_runtime()


def small():
    return build_model(width=64, backend="naive", predictor_hidden_width=64)


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


@pytest.mark.parametrize("step, expected", [(-1, 0.), (0, 0.), (1, 4e-13),
    (10000, .00004), (20000, .00016), (50000, .001), (60000, .001)])
def test_quadratic_boundaries(step, expected):
    assert coefficient(step) == pytest.approx(expected, rel=1e-15, abs=0)


def test_initial_weights_and_metadata_stay_unchanged_as_lambda_advances():
    fixed = fixed_model(width=64, backend="naive", predictor_hidden_width=64)
    model = small()
    assert _parameter_sha256(model) == _parameter_sha256(fixed)
    assert len(list(model.parameters())) == 44
    initial = copy.deepcopy(model.nextlat_initialization)
    config = copy.deepcopy(model.experiment_config)
    assert model.backbone.injection_coefficient == model.schedule_update == 0
    for update in (1, 10000, 20000, 50000):
        assert set_update(model, update) == coefficient(update)
        assert model.schedule_update == update
    assert model.nextlat_initialization == initial and model.experiment_config == config
    assert _parameter_sha256(model) == _parameter_sha256(fixed)
    assert model.nextlat_config == fixed.nextlat_config


def test_first_step_keeps_projection_attached_and_resume_is_exact(tmp_path):
    (tmp_path / "manifest.json").write_text('{"fixture":true}\n')
    args = driver.parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", "unused",
                                     "--width", "64", "--batch-size", "2"])
    model, restored = small(), small()
    optimizer, restored_optimizer = driver.make_optimizer(model), driver.make_optimizer(restored)
    contract = driver.make_contract(args, model=model, sources={}, data_root=tmp_path,
        train_x=np.zeros((8, 4), dtype=np.uint8), hardware={"capability": [0, 0]})
    assert contract["injection_schedule"] == SCHEDULE
    args.stop_file = "another-runtime-stop-path"
    assert driver.make_contract(args, model=model, sources={}, data_root=tmp_path,
        train_x=np.zeros((8, 4), dtype=np.uint8), hardware={"capability": [0, 0]}) == contract
    assert driver.train_step is rt_a5_nextlat_train.train_step
    assert driver.evaluate_arrays is rt_a5_train.evaluate_arrays
    x = torch.tensor([[0, 7, 2, 9], [3, 1, 8, 4]])
    y = x.flip(1)
    set_update(model, 1)
    driver.train_step(model, optimizer, x, y)
    projection = model.backbone.embedding_projection.weight
    assert projection.grad is not None and torch.isfinite(projection.grad).all()
    assert torch.count_nonzero(projection.grad) > 0
    assert len(optimizer.state) == 44 and all(s["step"].item() == 1 for s in optimizer.state.values())
    path = tmp_path / "step.pt"
    driver.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
        completed=1, order_chain=hashlib.sha256(b"fixture-order").hexdigest(),
        initialization=rt_a5_train.json_value(model.nextlat_initialization))
    set_update(model, 2)
    driver.train_step(model, optimizer, y, x)
    expected_random = torch.rand(4)
    packet = driver.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=contract)
    assert restored.schedule_update == 1 and restored.backbone.injection_coefficient == coefficient(1)
    set_update(restored, 2)
    driver.train_step(restored, restored_optimizer, y, x)
    tree_equal(restored.state_dict(), model.state_dict())
    tree_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    assert torch.equal(torch.rand(4), expected_random)
    before = copy.deepcopy(restored.state_dict())
    packet["injection_coefficient"] = .02
    invalid = tmp_path / "invalid.pt"
    torch.save(packet, invalid)
    with pytest.raises(ValueError, match="schedule or current coefficient"):
        driver.load_checkpoint(invalid, model=restored, optimizer=restored_optimizer, contract=contract)
    tree_equal(before, restored.state_dict())
    assert restored.schedule_update == 2


