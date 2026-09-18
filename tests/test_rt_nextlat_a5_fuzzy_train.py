"""Mixed task weighting, independent data streams, and full-state continuation."""
import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from scripts import rt_nextlat_a5_fuzzy_train as trainer
from scripts.rt_a5_common import configure_fp32_runtime, make_optimizer
from scripts.rt_a5_train import WordOrder, cpu_tree, rng_state


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 6)
        self.readout = nn.Linear(6, 8, bias=False)
        self.predictor = nn.Linear(6, 6, bias=False)
        self.initialization = {"fixture": "two-task-continuation"}
        self.stochastic = False


def tiny_objective(model, x, y, *, task, latent_weight):
    h = model.embedding(x)
    if model.stochastic:
        h = h + torch.randn_like(h) * 0.001
    start = 0 if task == "a5" else 4
    logits = model.readout(h)[..., start:start + 4]
    ce = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100)
    predicted = h[:, :-1] + model.predictor(model.embedding(x[:, 1:]))
    latent = F.smooth_l1_loss(predicted, h[:, 1:].detach())
    return {"loss": ce + latent_weight * latent, "ce": ce, "latent": latent, "logits": logits}


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    torch.set_num_threads(1)
    configure_fp32_runtime()
    torch.manual_seed(82)
    monkeypatch.setattr(trainer, "task_loss", tiny_objective)


def batches():
    a5 = torch.arange(15).reshape(5, 3) % 8
    fuzzy = torch.arange(35).reshape(5, 7) % 8 + 8
    ay, fy = a5.cumsum(-1) % 4, fuzzy.cumsum(-1) % 4
    # Unequal masks and lengths would expose pooled-token or microbatch means.
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


@pytest.mark.parametrize("mode", ["a5-only", "mixed"])
def test_task_normalized_accumulation_matches_direct_combined_objective(mode):
    model = TinyModel()
    direct = copy.deepcopy(model)
    optimizer, direct_optimizer = make_optimizer(model), make_optimizer(direct)
    data = {task: pair for task, pair in batches().items() if task in trainer.active_tasks(mode)}
    result = trainer.train_step(model, optimizer, data, mode=mode, microbatch=2)
    direct_optimizer.zero_grad(set_to_none=True)
    expected = {task: tiny_objective(direct, x, y, task=task, latent_weight=1.0)
                for task, (x, y) in data.items()}
    objective = sum(value["loss"] for value in expected.values()) / len(expected)
    objective.backward()
    norm = nn.utils.clip_grad_norm_(direct.parameters(), 1.0)
    direct_optimizer.step()
    assert result["loss"] == pytest.approx(objective.item(), abs=3e-7)
    assert result["grad_norm"] == pytest.approx(norm.item(), rel=1e-6)
    for task, value in expected.items():
        for key in ("ce", "latent"):
            assert result["tasks"][task][key] == pytest.approx(value[key].item(), abs=3e-7)
    for left, right in zip(model.parameters(), direct.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=3e-6, atol=1e-7)
        torch.testing.assert_close(left, right, rtol=3e-6, atol=1e-7)
    assert result["examples"] == 5 * len(data)
    assert all(state["step"].item() == 1 for state in optimizer.state.values())
    trainer.train_step(model, optimizer, data, mode=mode, microbatch=2)
    assert all(state["step"].item() == 2 for state in optimizer.state.values())


@pytest.mark.parametrize("mode", ["a5-only", "mixed"])
def test_exact_resume_preserves_all_task_cursors_rng_and_adam(tmp_path, mode):
    initial = TinyModel()
    initial.stochastic = True
    model = copy.deepcopy(initial)
    optimizer = make_optimizer(model)
    tasks = trainer.active_tasks(mode)
    rows = {"a5": 7, "fuzzy": 11}
    contract = {"mode": mode, "batch_per_task": 5,
                "streams": {task: {"train_rows": rows[task]} for task in tasks},
                "objective": {"latent_weight": 1.0}, "source_sha256": "fixture"}
    orders = {task: WordOrder(rows[task], 96 + i) for i, task in enumerate(tasks)}
    chains = {task: trainer.ORDER_INITIAL[task] for task in tasks}
    data = {task: batches()[task] for task in tasks}

    def step(completed, active_model, active_optimizer, previous):
        following = {task: hashlib.sha256(bytes.fromhex(previous[task]) +
                     orders[task].indices(completed * 5, 5).astype("<i8").tobytes()).hexdigest()
                     for task in tasks}
        values = trainer.train_step(active_model, active_optimizer, data, mode=mode, microbatch=2)
        random.random()
        np.random.random()
        return values, following

    for completed in range(2):
        _, chains = step(completed, model, optimizer, chains)
    path = tmp_path / "step-000002.pt"
    trainer.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                            completed=2, order_chains=chains, initialization=model.initialization)
    expected_metrics, expected_chains = step(2, model, optimizer, chains)
    expected_model, expected_adam, expected_rng = cpu_tree(model.state_dict()), cpu_tree(optimizer.state_dict()), rng_state()
    resumed = copy.deepcopy(initial)
    resumed_optimizer = make_optimizer(resumed)
    packet = trainer.load_checkpoint(path, model=resumed, optimizer=resumed_optimizer, contract=contract)
    assert packet["next_cursors"]["a5"] == {"absolute_example_offset": 10, "epoch": 1, "position": 3}
    if mode == "mixed":
        assert packet["next_cursors"]["fuzzy"] == {"absolute_example_offset": 10, "epoch": 0, "position": 10}
    actual_metrics, actual_chains = step(2, resumed, resumed_optimizer, packet["order_chains"])
    assert actual_metrics == expected_metrics
    assert actual_chains == expected_chains
    assert_tree_equal(resumed.state_dict(), expected_model)
    assert_tree_equal(resumed_optimizer.state_dict(), expected_adam)
    assert_tree_equal(rng_state(), expected_rng)


def test_reject_unequal_task_counts_before_optimizer_mutation():
    model = TinyModel()
    optimizer = make_optimizer(model)
    data = batches()
    data["fuzzy"] = tuple(value[:4] for value in data["fuzzy"])
    with pytest.raises(ValueError, match="equal example counts"):
        trainer.train_step(model, optimizer, data)
    assert not optimizer.state


@pytest.mark.parametrize("corruption", ["cursor", "chain", "mode", "adam"])
def test_checkpoint_rejects_independent_stream_or_state_corruption(tmp_path, corruption):
    model = TinyModel()
    optimizer = make_optimizer(model)
    trainer.train_step(model, optimizer, batches(), microbatch=2)
    contract = {"mode": "mixed", "batch_per_task": 5,
                "streams": {"a5": {"train_rows": 7}, "fuzzy": {"train_rows": 11}},
                "objective": {"latent_weight": 1.0}}
    path = tmp_path / "good.pt"
    trainer.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                            completed=1, order_chains=dict(trainer.ORDER_INITIAL),
                            initialization=model.initialization)
    packet = torch.load(path, weights_only=False)
    if corruption == "cursor":
        packet["next_cursors"]["fuzzy"]["position"] += 1
    elif corruption == "chain":
        del packet["order_chains"]["fuzzy"]
    elif corruption == "mode":
        packet["contract"]["mode"] = "a5-only"
    else:
        next(iter(packet["optimizer"]["state"].values()))["exp_avg"].flatten()[0] = float("nan")
    broken = tmp_path / "broken.pt"
    torch.save(packet, broken)
    with pytest.raises(ValueError):
        trainer.load_checkpoint(broken, model=model, optimizer=optimizer, contract=contract)


def test_a5_evaluation_exact_counts_and_position_zero_are_scored(monkeypatch):
    # Construct token accuracy 71/72 while E36=1/2 and final-state accuracy=1.
    # This detects substituting token/final-state accuracy for the gate.
    x = np.zeros((2, 36), dtype=np.int64)
    x[1, 0] = 1
    y = np.zeros_like(x)

    def objective(model, inputs, labels, **kwargs):
        prediction = torch.zeros_like(inputs)
        prediction[inputs[:, 0].eq(1), 0] = 1
        logits = F.one_hot(prediction, 60).float() * 8.0
        return {"logits": logits, "latent": logits.new_tensor(0.25)}

    monkeypatch.setattr(trainer, "task_loss", objective)
    model = TinyModel()
    before = rng_state()
    result = trainer.evaluate_a5(model, x, y, microbatch=1, device="cpu")
    assert_tree_equal(before, rng_state())
    assert model.training
    assert result["length"] == 36
    assert result["whole_word_correct"] == result["whole_word_exact_count"] == 1
    assert result["whole_word_exact_match"] == 0.5
    assert result["final_state_accuracy"] == 1.0
    assert result["per_position_prefix_correct"] == [1] * 36
    assert result["evaluated_rows"] == 2
    logged = trainer.evaluation_metrics({**result, "task": "a5", "role": "ood_dev"})
    assert logged["dev/a5/ood_dev/cumulative_prefix_exactness/position_36"] == 0.5


def test_fuzzy_order_matches_original_stream_across_epoch_boundaries():
    from scripts import rt_nextlat_fuzzy_train as original
    assert trainer.ORDER_INITIAL["fuzzy"] == original.ORDER_INITIAL
    standalone = WordOrder(12800, 2026091604)
    mixed = WordOrder(12800, 2026091604)
    for update in (0, 99, 100, 6520):
        assert np.array_equal(standalone.indices(update * 128, 128), mixed.indices(update * 128, 128))


def test_cli_scope_and_frozen_dependency_manifest():
    args = trainer.parser().parse_args(["--mode", "mixed", "--a5-data", "a5", "--fuzzy-data", "fuzzy",
                                       "--output", "new", "--checkpoint-steps", "0", "6521", "10000",
                                       "--full-eval-steps", "6521", "10000"])
    assert args.batch_per_task == args.microbatch == 128
    assert args.eval_microbatch == 1024
    assert args.full_eval_rows == 102400
    assert args.wandb_mode == "online"
    assert args.a5_order_seed == 5432
    assert args.fuzzy_order_seed == 2026091604
    assert args.checkpoint_steps == [0, 6521, 10000]
    sources = trainer.source_manifest()
    for path in ("scripts/rt_nextlat_a5_fuzzy_train.py", "scripts/rt_nextlat_fuzzy_train.py",
                 "cdrm/rt_nextlat_tasks.py", "scripts/rt_a5_data.py", "scripts/rt_a5_window.py"):
        assert len(sources[path]) == 64


def test_inherited_gate_respects_resume_cutoff_and_retained_checkpoint_hash(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    gate_checkpoint = checkpoint_dir / "step-001000.pt"
    gate_checkpoint.write_bytes(b"retained gate checkpoint fixture")
    restored = checkpoint_dir / "step-005000.pt"
    restored.write_bytes(b"restored checkpoint fixture")
    contract = {"source": "same", "mode": "a5-only"}
    parent_report = {"contract": contract, "a5_positive_gate": {
        "eligible": True, "first_positive_update": 1000, "observed_by_3000": True,
        "whole_word_exact_count": 3, "evaluated_rows": 102400, "length": 36,
        "checkpoint": {"path": str(gate_checkpoint), "sha256": trainer.file_sha256(gate_checkpoint),
                       "completed_updates": 1000}}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(parent_report))
    # A parent that later succeeds cannot leak that result into an earlier fork.
    assert trainer.inherited_positive_gate(restored, contract=contract, completed=500) is None
    inherited = trainer.inherited_positive_gate(restored, contract=contract, completed=5000)
    assert inherited["eligible"] and inherited["observed_by_3000"]
    assert inherited["inherited_from"]["report_sha256"] == trainer.file_sha256(report_path)
    assert inherited["first_positive_update"] == 1000
    gate_checkpoint.write_bytes(b"corrupted checkpoint")
    with pytest.raises(ValueError, match="checkpoint evidence"):
        trainer.inherited_positive_gate(restored, contract=contract, completed=5000)
