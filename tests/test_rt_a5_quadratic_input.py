"""CPU schedule, first-step gradient, exact resume and graceful-stop checks."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import rt_a5_common, rt_a5_nextlat_train, rt_a5_train
from scripts import rt_a5_quadratic_input_train as driver
from scripts.rt_a5_embedding_injection import build_model as fixed_model
from scripts.rt_a5_nextlat import _parameter_sha256
from scripts.rt_a5_quadratic_input import SCHEDULE, build_model, coefficient, set_update


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


@pytest.mark.parametrize("step, expected", [(-1, 0.), (0, 0.), (1, 2e-11),
    (10000, .002), (20000, .008), (50000, .05), (60000, .05)])
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


@pytest.mark.parametrize("stop_at, checkpoints", [(1, [3]), (2, [2, 3]), (None, [3])])
def test_driver_stops_after_exact_completed_update_or_completes_endpoint(tmp_path, monkeypatch, stop_at, checkpoints):
    """Exercise the real CPU update/save/load loop with tiny data and local tracking."""
    x = np.arange(32, dtype=np.uint8).reshape(8, 4) % 60
    y = x[:, ::-1].copy()
    (tmp_path / "manifest.json").write_text('{"fixture":true}\n')
    stop_file = tmp_path / "USERSTOP"
    output = tmp_path / "run"
    args = driver.parser().parse_args(["--data-dir", str(tmp_path), "--output-dir", str(output),
        "--updates", "3", "--width", "64", "--batch-size", "2", "--predictor-hidden-width", "64",
        "--eval-every", "2", "--eval-rows", "2", "--full-eval-rows", "4",
        "--diagnostic-rows", "2", "--log-every", "3", "--stop-file", str(stop_file),
        "--checkpoint-steps", *map(str, checkpoints)])
    monkeypatch.setattr(driver, "require_cuda_container", lambda: {"capability": [0, 0]})
    monkeypatch.setattr(driver, "validate_manifest", lambda _root: None)
    monkeypatch.setattr(driver, "load_split", lambda _root, _role: (x, y))
    monkeypatch.setattr(driver, "source_manifest", lambda: {})
    monkeypatch.setattr(driver, "build_model", lambda **_kwargs: small())
    monkeypatch.setattr(driver, "batch_tensors", lambda tx, ty, indices, _device:
                        rt_a5_train.batch_tensors(tx, ty, indices, "cpu"))
    monkeypatch.setattr(driver, "evaluate_arrays", lambda model, dx, dy, **kw:
                        rt_a5_train.evaluate_arrays(model, dx, dy, device="cpu", **kw))
    monkeypatch.setattr(driver, "evaluate_diagnostics", lambda model, dx, dy, **kw:
                        rt_a5_nextlat_train.evaluate_diagnostics(model, dx, dy, device="cpu", **kw))
    calls = []
    def update(model, optimizer, tx, ty, **kwargs):
        step = len(calls) + 1
        assert model.schedule_update == step and model.backbone.injection_coefficient == coefficient(step)
        calls.append(step)
        values = rt_a5_nextlat_train.train_step(model, optimizer, tx, ty, **kwargs)
        if step == stop_at:
            stop_file.write_text("Requested stop for CPU fixture\n")
        return values
    monkeypatch.setattr(driver, "train_step", update)
    class LocalTracker:
        def __init__(self, **_kwargs):
            self.record = {"run_url": "local-test", "status": "running"}
        def start(self, _contract): pass
        def log(self, _values): pass
        def summary(self, _values): pass
        def finish(self, *, succeeded):
            assert succeeded
            self.record["status"] = "synced"
    monkeypatch.setattr(driver, "OnlineTracker", LocalTracker)
    report = driver.run(args)
    end = stop_at or 3
    assert calls == list(range(1, end + 1))
    assert report["endpoint"] == 3 and report["completed_updates"] == end
    assert report["status"] == ("stopped" if stop_at else "complete")
    assert report["requested_endpoint_reached"] is (stop_at is None)
    assert report["wandb"]["status"] == "synced"
    assert report["injection_coefficient"] == coefficient(end)
    saved = [row["completed_updates"] for row in report["checkpoints"]]
    assert saved == [0, end] and len(saved) == len(set(saved))
    evaluations = [r for r in report["evaluations"] if r["update"] == end]
    assert len(evaluations) == 2 and all(r["evaluated_rows"] == 4 for r in evaluations)
    assert all(r["injection_coefficient"] == coefficient(end) for r in evaluations)
    if stop_at:
        assert report["stop_request"]["observed_after_update"] == end
        assert stop_file.is_file()
    packet = torch.load(output / f"checkpoints/step-{end:06d}.pt", map_location="cpu", weights_only=False)
    assert packet["schedule_update"] == end and packet["injection_coefficient"] == coefficient(end)
    assert all(s["step"].item() == end for s in packet["optimizer"]["state"].values())
    rows = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert len(rows) == end and all(row["injection_coefficient"] == coefficient(row["update"]) for row in rows)
