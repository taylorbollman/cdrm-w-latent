"""Weighted objective and actual-data exact continuation of the new stage."""
import copy
import hashlib
import random

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from scripts import rt_nextlat_a5_fuzzy_curriculum as stage
from scripts.rt_a5_common import configure_fp32_runtime, make_optimizer
from scripts.rt_a5_train import WordOrder, batch_tensors, cpu_tree, rng_state


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 6)
        self.readout = nn.Linear(6, 8, bias=False)
        self.predictor = nn.Linear(6, 6, bias=False)
        self.initialization = {"fixture": "curriculum-actual-data-resume"}
        self.stochastic = False


def tiny_objective(model, x, y, *, task, latent_weight):
    hidden = model.embedding(x)
    if model.stochastic:
        hidden = hidden + torch.randn_like(hidden) * 0.001
    start = 0 if task == "a5" else 4
    logits = model.readout(hidden)[..., start:start + 4]
    ce = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100)
    predicted = hidden[:, :-1] + model.predictor(model.embedding(x[:, 1:]))
    latent = F.smooth_l1_loss(predicted, hidden[:, 1:].detach())
    return {"loss": ce + latent, "ce": ce, "latent": latent, "logits": logits}


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    torch.set_num_threads(1)
    configure_fp32_runtime()
    torch.manual_seed(82)
    monkeypatch.setattr(stage, "task_loss", tiny_objective)
    monkeypatch.setattr(stage.base, "task_loss", tiny_objective)


def batches():
    a5 = torch.arange(15).reshape(5, 3) % 8
    fuzzy = torch.arange(35).reshape(5, 7) % 8 + 8
    ay, fy = a5.cumsum(-1) % 4, fuzzy.cumsum(-1) % 4
    fy[0, :4] = -100
    fy[2, :2] = -100
    return {"a5": (a5, ay), "fuzzy": (fuzzy, fy)}


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("phase", [1, 1500, 3000, 8000])
def test_weighted_task_means_match_one_direct_objective_and_adam_step(phase):
    actual, reference = TinyModel(), TinyModel()
    reference.load_state_dict(actual.state_dict())
    opt, ref_opt = make_optimizer(actual), make_optimizer(reference)
    data = batches()
    measured = stage.train_step(actual, opt, data, phase_update=phase, microbatch=2)
    ref_opt.zero_grad(set_to_none=True)
    weights = {"fuzzy": 0.5 * min(phase / 3000, 1), "a5": 1 - 0.5 * min(phase / 3000, 1)}
    objective = sum(weights[task] * tiny_objective(reference, x, y, task=task, latent_weight=1)["loss"]
                    for task, (x, y) in data.items())
    objective.backward()
    norm = nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    ref_opt.step()
    assert measured["loss"] == pytest.approx(objective.item(), abs=4e-7)
    assert measured["grad_norm"] == pytest.approx(norm.item(), rel=2e-6)
    assert measured["task_weights"] == weights
    for a, b in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-6, atol=1e-7)
        torch.testing.assert_close(a, b, rtol=3e-6, atol=1e-7)
    assert all(state["step"].item() == 1 for state in opt.state.values())


@pytest.mark.parametrize("mode,phase", [("a5-control", 8000), ("mixed-curriculum", 0)])
def test_control_and_zero_weight_boundary_are_exact_frozen_a5_steps(mode, phase):
    actual = TinyModel()
    reference = copy.deepcopy(actual)
    opt, ref_opt = make_optimizer(actual), make_optimizer(reference)
    data = batches()
    measured = stage.train_step(actual, opt, {task: data[task] for task in stage.active_tasks(mode)},
                                mode=mode, phase_update=phase, microbatch=2)
    expected = stage.base.train_step(reference, ref_opt, {"a5": data["a5"]}, mode="a5-only", microbatch=2)
    assert measured.pop("task_weights") == {"a5": 1.0, "fuzzy": 0.0}
    assert measured == expected
    assert_tree_equal(actual.state_dict(), reference.state_dict())
    assert_tree_equal(opt.state_dict(), ref_opt.state_dict())


def continuation_fixture(mode):
    model = TinyModel()
    model.stochastic = True
    optimizer = make_optimizer(model)
    arrays = {}
    for task, rows, length, offset in (("a5", 7, 3, 0), ("fuzzy", 11, 7, 8)):
        x = np.arange(rows * length).reshape(rows, length) % 8 + offset
        y = np.cumsum(x, axis=-1) % 4
        arrays[task] = (x, y)
    orders = {"a5": WordOrder(7, 91), "fuzzy": WordOrder(11, 92)}
    chains = {task: stage.base.ORDER_INITIAL[task] for task in arrays}
    # A real two-update parent; each pretraining step consumes different rows.
    for update in range(2):
        indices = orders["a5"].indices(update * 5, 5)
        xy = batch_tensors(*arrays["a5"], indices, "cpu")
        stage.base.train_step(model, optimizer, {"a5": xy}, mode="a5-only", microbatch=2)
        chains["a5"] = hashlib.sha256(bytes.fromhex(chains["a5"]) + indices.astype("<i8").tobytes()).hexdigest()
    contract = {"mode": mode, "global_start_update": 2, "batch_per_task": 5, "ramp_updates": 3,
                "initial_task_offsets": {"a5": 10, "fuzzy": 0}, "initial_order_chains": dict(chains),
                "streams": {"a5": {"train_rows": 7}, "fuzzy": {"train_rows": 11}},
                "objective": {"latent_weight": 1.0}, "source_sha256": "fixture"}
    return model, optimizer, arrays, orders, chains, contract


@pytest.mark.parametrize("mode", ["a5-control", "mixed-curriculum"])
def test_exact_resume_uses_actual_ordered_rows_crossing_epochs_and_ramp(tmp_path, mode):
    model, optimizer, arrays, orders, chains, contract = continuation_fixture(mode)

    def advance(current_phase, active_model, active_optimizer, previous):
        offsets = stage.task_offsets(current_phase, contract)
        following, selected = dict(previous), {}
        for task in stage.active_tasks(mode):
            indices = orders[task].indices(offsets[task], 5)
            selected[task] = batch_tensors(*arrays[task], indices, "cpu")
            following[task] = hashlib.sha256(bytes.fromhex(previous[task]) + indices.astype("<i8").tobytes()).hexdigest()
        metrics = stage.train_step(active_model, active_optimizer, selected, mode=mode,
                                   phase_update=current_phase + 1, ramp_updates=3, microbatch=2)
        random.random()
        np.random.random()
        return metrics, following

    _, chains = advance(0, model, optimizer, chains)
    path = tmp_path / "phase-000001.pt"
    stage.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract, phase_updates=1,
                          order_chains=chains, initialization=model.initialization)
    expected = []
    for phase in (1, 2, 3):
        values, chains = advance(phase, model, optimizer, chains)
        expected.append(values)
    expected_model, expected_adam, expected_rng = cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()), rng_state()
    resumed = TinyModel()
    resumed.stochastic = True
    resumed_optimizer = make_optimizer(resumed)
    packet = stage.load_checkpoint(path, model=resumed, optimizer=resumed_optimizer, contract=contract)
    assert packet["phase_updates"] == 1
    assert packet["completed_updates"] == packet["global_updates"] == 3
    assert packet["next_cursors"]["a5"] == {"absolute_example_offset": 15, "epoch": 2, "position": 1}
    actual_chain = packet["order_chains"]
    for phase, metrics in zip((1, 2, 3), expected):
        actual, actual_chain = advance(phase, resumed, resumed_optimizer, actual_chain)
        assert actual == metrics
    assert actual_chain == chains
    assert_tree_equal(resumed.state_dict(), expected_model)
    assert_tree_equal(resumed_optimizer.state_dict(), expected_adam)
    assert_tree_equal(rng_state(), expected_rng)
    assert all(state["step"].item() == 6 for state in resumed_optimizer.state.values())


@pytest.mark.parametrize("field", ["phase_updates", "completed_updates", "fuzzy_offset", "weight", "parent"])
def test_resume_rejects_counter_schedule_or_parent_corruption(tmp_path, field):
    model, optimizer, _, _, chains, contract = continuation_fixture("mixed-curriculum")
    path = tmp_path / "phase-000000.pt"
    stage.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract, phase_updates=0,
                          order_chains=chains, initialization=model.initialization)
    packet = torch.load(path, weights_only=False)
    if field in ("phase_updates", "completed_updates"):
        packet[field] += 1
    elif field == "fuzzy_offset":
        packet["task_offsets"]["fuzzy"] += 5
    elif field == "weight":
        packet["task_weights"]["fuzzy"] = 0.25
    else:
        packet["contract"]["initial_task_offsets"]["a5"] = 0
    broken = tmp_path / "broken.pt"
    torch.save(packet, broken)
    with pytest.raises(ValueError):
        stage.load_checkpoint(broken, model=model, optimizer=optimizer, contract=contract)


def test_cli_fixed_batch_and_bounded_offline_guards():
    args = stage.parser().parse_args(["--mode", "mixed-curriculum", "--a5-data", "a5",
                                     "--fuzzy-data", "fuzzy", "--output", "new"])
    stage._validate_args(args)
    assert args.ramp_updates == 3000 and args.updates == 10000
    assert args.eval_every == 250 and args.early_eval_steps == [50, 100]
    assert args.wandb_mode == "online"
    args.batch_per_task = 256
    with pytest.raises(ValueError, match="128"):
        stage._validate_args(args)
    args.batch_per_task = 128
    args.wandb_mode = "disabled"
    with pytest.raises(ValueError, match="ten phase updates"):
        stage._validate_args(args)
    args.updates = 3
    stage._validate_args(args)
