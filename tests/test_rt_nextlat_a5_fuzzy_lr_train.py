"""The LR ablation preserves task math and resumes the global schedule exactly."""
import copy
import hashlib
import random

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from scripts import rt_nextlat_a5_fuzzy_lr_train as scheduled
from scripts import rt_nextlat_a5_fuzzy_train as original
from scripts.rt_a5_common import configure_fp32_runtime, make_optimizer
from scripts.rt_a5_train import WordOrder, cpu_tree, rng_state


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 6)
        self.readout = nn.Linear(6, 8, bias=False)
        self.predictor = nn.Linear(6, 6, bias=False)
        self.initialization = {"fixture": "unchanged-two-task-LR-continuation"}


def tiny_objective(model, x, y, *, task, latent_weight):
    h = model.embedding(x) + torch.randn((*x.shape, 6)) * 0.001
    start = 0 if task == "a5" else 4
    logits = model.readout(h)[..., start:start + 4]
    ce = F.cross_entropy(logits.transpose(1, 2), y)
    predicted = h[:, :-1] + model.predictor(model.embedding(x[:, 1:]))
    latent = F.smooth_l1_loss(predicted, h[:, 1:].detach())
    return {"loss": ce + latent_weight * latent, "ce": ce, "latent": latent, "logits": logits}


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    torch.set_num_threads(1)
    configure_fp32_runtime()
    torch.manual_seed(82)
    monkeypatch.setattr(original, "task_loss", tiny_objective)


def batches():
    a5 = torch.arange(6).reshape(2, 3) % 8
    fuzzy = torch.arange(10).reshape(2, 5) % 8 + 8
    return {"a5": (a5, a5.cumsum(-1) % 4), "fuzzy": (fuzzy, fuzzy.cumsum(-1) % 4)}


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def contract():
    return {"schema": scheduled.SCHEMA, "mode": "mixed", "batch_per_task": 2,
            "streams": {"a5": {"train_rows": 7}, "fuzzy": {"train_rows": 11}},
            "objective": {"latent_weight": 1.0}, "source_sha256": "fixture",
            "learning_rate_schedule": scheduled.schedule_configuration(),
            "optimizer": {"lr": 3e-4, "initial_lr": 1e-4}}


def step(model, optimizer, update, schedule, chains):
    orders = {task: WordOrder(rows, 96 + i) for i, (task, rows) in enumerate((('a5', 7), ('fuzzy', 11)))}
    following = {task: hashlib.sha256(bytes.fromhex(chains[task]) +
                 order.indices((update - 1) * 2, 2).astype("<i8").tobytes()).hexdigest()
                 for task, order in orders.items()}
    lr = scheduled.set_learning_rate(optimizer, update, schedule)
    values = scheduled.train_step(model, optimizer, batches(), mode="mixed", microbatch=2)
    random.random()
    np.random.random()
    return {"learning_rate": lr, **values}, following


def test_exact_warmup_endpoints_and_linear_increment():
    assert scheduled.learning_rate(1) == 1e-4
    assert scheduled.learning_rate(100) == 3e-4
    assert scheduled.learning_rate(101) == scheduled.learning_rate(5000) == 3e-4
    for update in range(2, 100):
        assert scheduled.learning_rate(update) - scheduled.learning_rate(update - 1) == pytest.approx(2e-4 / 99)
    schedule = scheduled.schedule_configuration()
    assert scheduled.learning_rate_state(0, schedule) == {
        "completed_updates": 0, "optimizer_lr": 1e-4, "next_update_lr": 1e-4}
    assert scheduled.learning_rate_state(99, schedule)["next_update_lr"] == 3e-4


@pytest.mark.parametrize("kwargs", [{"update": 0}, {"update": True}, {"update": 1, "start_lr": 0},
                                     {"update": 1, "peak_lr": float("nan")},
                                     {"update": 1, "warmup_updates": 1},
                                     {"update": 1, "start_lr": 4e-4}])
def test_invalid_schedule_rejected(kwargs):
    with pytest.raises(ValueError):
        scheduled.learning_rate(**kwargs)


@pytest.mark.parametrize("completed", [0, 2, 99, 100, 101])
def test_exact_resume_preserves_next_lr_adam_rng_and_both_streams(tmp_path, completed):
    initial = TinyModel()
    model, rules = copy.deepcopy(initial), contract()
    optimizer = make_optimizer(model, lr=rules["learning_rate_schedule"]["start_lr"])
    chains = dict(scheduled.ORDER_INITIAL)
    for update in range(1, completed + 1):
        _, chains = step(model, optimizer, update, rules["learning_rate_schedule"], chains)
    path = tmp_path / f"step-{completed:06d}.pt"
    scheduled.save_checkpoint(path, model=model, optimizer=optimizer, contract=rules,
                              completed=completed, order_chains=chains, initialization=model.initialization)
    expected, following = step(model, optimizer, completed + 1, rules["learning_rate_schedule"], chains)
    expected_model, expected_adam, expected_rng = cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()), rng_state()
    restored = copy.deepcopy(initial)
    restored_optimizer = make_optimizer(restored, lr=1e-4)
    packet = scheduled.load_checkpoint(path, model=restored, optimizer=restored_optimizer, contract=rules)
    assert packet["learning_rate_state"] == scheduled.learning_rate_state(completed, rules["learning_rate_schedule"])
    actual, actual_following = step(restored, restored_optimizer, completed + 1,
                                    rules["learning_rate_schedule"], packet["order_chains"])
    assert expected == actual and following == actual_following
    assert_tree_equal(expected_model, restored.state_dict())
    assert_tree_equal(expected_adam, restored_optimizer.state_dict())
    assert_tree_equal(expected_rng, rng_state())


@pytest.mark.parametrize("corruption", ["lr", "state", "schema", "betas", "decay", "adam", "group", "schedule"])
def test_reject_corrupted_lr_or_adam_before_mutating_caller(tmp_path, corruption):
    model, rules = TinyModel(), contract()
    optimizer = make_optimizer(model)
    chains = dict(scheduled.ORDER_INITIAL)
    for update in (1, 2):
        _, chains = step(model, optimizer, update, rules["learning_rate_schedule"], chains)
    path = tmp_path / "good.pt"
    scheduled.save_checkpoint(path, model=model, optimizer=optimizer, contract=rules,
                              completed=2, order_chains=chains, initialization=model.initialization)
    packet = torch.load(path, weights_only=False)
    if corruption == "lr":
        packet["optimizer"]["param_groups"][0]["lr"] = 3e-4
    elif corruption == "state":
        packet["learning_rate_state"]["completed_updates"] = 3
    elif corruption == "schema":
        packet["schema"] = original.SCHEMA
    elif corruption == "betas":
        packet["optimizer"]["param_groups"][0]["betas"] = (0.8, 0.95)
    elif corruption == "decay":
        packet["optimizer"]["param_groups"][0]["weight_decay"] = 0.0
    elif corruption == "adam":
        next(iter(packet["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
    elif corruption == "group":
        packet["optimizer"]["param_groups"].pop()
    else:
        packet["contract"]["learning_rate_schedule"]["warmup_updates"] = 200
    bad = tmp_path / "bad.pt"
    torch.save(packet, bad)
    fresh = TinyModel()
    fresh_optimizer = make_optimizer(fresh)
    before_model, before_adam, before_rng = cpu_tree(fresh.state_dict()), cpu_tree(fresh_optimizer.state_dict()), rng_state()
    with pytest.raises(ValueError):
        scheduled.load_checkpoint(bad, model=fresh, optimizer=fresh_optimizer, contract=rules)
    assert_tree_equal(before_model, fresh.state_dict())
    assert_tree_equal(before_adam, fresh_optimizer.state_dict())
    assert_tree_equal(before_rng, rng_state())


def test_checkpoint_rejects_stale_optimizer_lr(tmp_path):
    model, rules = TinyModel(), contract()
    optimizer = make_optimizer(model, lr=3e-4)
    with pytest.raises(ValueError, match="Optimizer LR differs"):
        scheduled.save_checkpoint(tmp_path / "wrong.pt", model=model, optimizer=optimizer,
                                  contract=rules, completed=0, order_chains=dict(scheduled.ORDER_INITIAL),
                                  initialization=model.initialization)


def test_frozen_math_factory_cli_and_source_closure_preserved():
    for name in ("build_model", "read_configuration", "train_step", "evaluate_a5", "evaluation_metrics", "cursors"):
        assert getattr(scheduled, name) is getattr(original, name)
    old, new = original.source_manifest(), scheduled.source_manifest()
    assert all(new[name] == sha for name, sha in old.items())
    assert set(new) - set(old) == {"scripts/rt_nextlat_a5_fuzzy_lr_train.py"}
    args = scheduled.parser().parse_args(["--mode", "mixed", "--a5-data", "a5", "--fuzzy-data", "fuzzy", "--output", "new"])
    assert (args.warmup_start_lr, args.learning_rate, args.warmup_updates) == (1e-4, 3e-4, 100)
    assert args.a5_order_seed == 5432 and args.fuzzy_order_seed == 2026091604
