"""CPU-only tests for loss accumulation and exact full-state continuation."""
import copy
import hashlib
import random

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from scripts import rt_nextlat_fuzzy_train as trainer
from scripts.rt_a5_common import configure_fp32_runtime, make_optimizer
from scripts.rt_a5_train import WordOrder, cpu_tree, rng_state


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 6)
        self.readout = nn.Linear(6, 4, bias=False)
        self.predictor = nn.Linear(6, 6, bias=False)
        self.initialization = {"fixture": "tiny-fuzzy-trainer"}
        self.stochastic = False


def tiny_objective(model, x, y, *, task, latent_weight):
    assert task == "fuzzy"
    h = model.embedding(x)
    if model.stochastic:
        h = h + torch.randn_like(h) * 0.001
    logits = model.readout(h)
    ce = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100)
    predicted = h[:, :-1] + model.predictor(model.embedding(x[:, 1:]))
    latent = F.smooth_l1_loss(predicted, h[:, 1:].detach())
    return {"loss": ce + latent_weight * latent, "ce": ce,
            "latent": latent, "logits": logits}


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    torch.set_num_threads(1)
    configure_fp32_runtime()
    torch.manual_seed(81)
    monkeypatch.setattr(trainer, "task_loss", tiny_objective)


def batch():
    x = torch.arange(20).reshape(5, 4) % 8
    y = x.cumsum(-1) % 4
    # Unequal supervised-token counts across microbatches exercise both means.
    y[0, :3] = -100
    y[2, :2] = -100
    return x, y


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


def test_accumulation_matches_direct_objective_with_uneven_batches_and_masks():
    actual = TinyModel()
    expected = copy.deepcopy(actual)
    opt, ref_opt = make_optimizer(actual), make_optimizer(expected)
    x, y = batch()
    measured = trainer.train_step(actual, opt, x, y, microbatch=2)
    ref_opt.zero_grad(set_to_none=True)
    objective = tiny_objective(expected, x, y, task="fuzzy", latent_weight=1.0)
    objective["loss"].backward()
    norm = nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
    ref_opt.step()
    assert measured["ce"] == pytest.approx(objective["ce"].item(), abs=2e-7)
    assert measured["latent"] == pytest.approx(objective["latent"].item(), abs=2e-7)
    assert measured["grad_norm"] == pytest.approx(norm.item(), rel=1e-6)
    for left, right in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(left, right, rtol=2e-6, atol=1e-7)
    assert measured["scored_tokens"] == 15
    assert measured["examples"] == 5
    assert all(state["step"].item() == 1 for state in opt.state.values())
    # A second accumulated update clears previous gradients and steps once.
    trainer.train_step(actual, opt, x, y, microbatch=2)
    assert all(state["step"].item() == 2 for state in opt.state.values())


def test_exact_resume_preserves_epoch_boundary_rng_adam_and_order(tmp_path):
    initial = TinyModel()
    initial.stochastic = True
    model = copy.deepcopy(initial)
    optimizer = make_optimizer(model)
    contract = {"batch_size": 5, "train_rows": 7, "source_sha256": "fixture",
                "objective": {"latent_weight": 1.0}}
    order = WordOrder(7, 96)
    chain = trainer.ORDER_INITIAL
    x, y = batch()

    def step(completed, model, optimizer, chain):
        indices = order.indices(completed * 5, 5)
        chain = hashlib.sha256(bytes.fromhex(chain) + indices.astype("<i8").tobytes()).hexdigest()
        values = trainer.train_step(model, optimizer, x, y, microbatch=2)
        # These streams also belong to exact continuation even though this model
        # only uses the torch stream during its stochastic objective.
        random.random()
        np.random.random()
        return values, chain

    _, chain = step(0, model, optimizer, chain)
    path = tmp_path / "step-000001.pt"
    trainer.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                            completed=1, order_chain=chain, initialization=model.initialization)
    expected_metrics, expected_chain = step(1, model, optimizer, chain)
    expected_model = cpu_tree(model.state_dict())
    expected_adam = cpu_tree(optimizer.state_dict())
    expected_rng = rng_state()
    resumed = copy.deepcopy(initial)
    resumed_optimizer = make_optimizer(resumed)
    packet = trainer.load_checkpoint(path, model=resumed, optimizer=resumed_optimizer,
                                     contract=contract)
    assert packet["next_cursor"] == {"absolute_example_offset": 5, "epoch": 0, "position": 5}
    actual_metrics, actual_chain = step(1, resumed, resumed_optimizer, packet["order_chain"])
    assert actual_metrics == expected_metrics
    assert actual_chain == expected_chain
    assert_tree_equal(resumed.state_dict(), expected_model)
    assert_tree_equal(resumed_optimizer.state_dict(), expected_adam)
    assert_tree_equal(rng_state(), expected_rng)


def test_checkpoint_rejects_invalid_cursor_and_nonfinite_adam(tmp_path):
    model = TinyModel()
    optimizer = make_optimizer(model)
    trainer.train_step(model, optimizer, *batch(), microbatch=2)
    contract = {"batch_size": 5, "train_rows": 7, "objective": {"latent_weight": 1.0}}
    path = tmp_path / "good.pt"
    trainer.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                            completed=1, order_chain=trainer.ORDER_INITIAL,
                            initialization=model.initialization)
    packet = torch.load(path, weights_only=False)
    packet["next_cursor"]["position"] = 6
    broken = tmp_path / "broken.pt"
    torch.save(packet, broken)
    with pytest.raises(ValueError, match="cursor"):
        trainer.load_checkpoint(broken, model=model, optimizer=optimizer, contract=contract)
    next(iter(optimizer.state.values()))["exp_avg"].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="optimizer state"):
        trainer.save_checkpoint(tmp_path / "bad.pt", model=model, optimizer=optimizer,
                                contract=contract, completed=1, order_chain=trainer.ORDER_INITIAL,
                                initialization=model.initialization)


def test_source_manifest_covers_production_model_window_and_metrics():
    sources = trainer.source_manifest()
    for name in ("cdrm/rt_nextlat_tasks.py", "cdrm/rt_nextlat_fuzzy_metrics.py",
                 "scripts/rt_a5_window.py", "scripts/rt_nextlat_fuzzy_train.py",
                 "vendors/mad-lab/mad/data/instances.py"):
        assert name in sources
        assert len(sources[name]) == 64


def test_native_evaluation_uses_answer_mask_and_preserves_rng(monkeypatch):
    from cdrm.mad_data import FUZZY_TASK, generate_dataset
    from cdrm.rt_nextlat_tasks import build_model, encode_inputs, task_loss

    monkeypatch.setattr(trainer, "task_loss", task_loss)
    data = generate_dataset(FUZZY_TASK, "dev", 514, 3, {"seq_len": 20})
    model = build_model(backend="naive")
    before = rng_state()
    result = trainer.evaluate(model, data, microbatch=2, device="cpu")
    assert model.training
    assert_tree_equal(rng_state(), before)
    x = encode_inputs(torch.from_numpy(data.input_ids.copy()), task="fuzzy")
    y = torch.from_numpy(data.answer_labels.copy())
    with torch.no_grad():
        expected = task_loss(model, x, y)
    assert result["ce"] == pytest.approx(expected["ce"].item(), rel=1e-6)
    assert result["latent"] == pytest.approx(expected["latent"].item(), rel=1e-6)
    assert result["answer_ce"] == pytest.approx(result["ce"], rel=1e-6)
    assert result["answer_tokens"] == int((data.answer_labels != -100).sum())
    assert result["evaluated_rows"] == result["examples"] == 3


def test_cli_defaults_keep_online_tracking_and_no_endpoint_ceiling():
    args = trainer.parser().parse_args(["--config", "model.json", "--data", "data",
                                       "--output", "new-run", "--updates", "12345"])
    assert args.updates == 12345
    assert args.wandb_mode == "online"
    assert args.wandb_project == "rt-nextlat-fuzzy-a5"
    assert args.batch_size == 128
    assert args.checkpoint_steps == [0, 1000, 5000, 10000]
