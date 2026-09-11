"""CPU-only unit tests of A5 order, checkpoint contracts and evaluation arithmetic."""
import copy
import hashlib

import numpy as np
import pytest
import torch

from scripts.rt_a5_train import (
    WordOrder, cpu_tree, evaluate_arrays, json_sha256, load_checkpoint,
    save_checkpoint, validate_model_state,
)


def test_word_order_visits_every_row_and_carries_epoch_remainder():
    order = WordOrder(11, 9)
    batches = [order.indices(i * 4, 4) for i in range(9)]
    stream = np.concatenate(batches)
    for epoch in range(3):
        assert sorted(stream[epoch * 11:(epoch + 1) * 11]) == list(range(11))
    # A new process can reconstruct an arbitrary cross-epoch window directly.
    np.testing.assert_array_equal(WordOrder(11, 9).indices(8, 20), stream[8:28])
    np.testing.assert_array_equal(WordOrder(11, 9).indices(0, 36), stream)
    assert not np.array_equal(WordOrder(11, 10).indices(0, 36), stream)


def test_checkpoint_preserves_adam_rng_and_rejects_changed_contract(tmp_path):
    model = torch.nn.Linear(2, 3, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.weight.grad = torch.full_like(model.weight, 0.125)
    optimizer.step()
    contract = {"batch_size": 4, "data": "original", "source": "original"}
    path = tmp_path / "state.pt"
    before = cpu_tree(model.state_dict())
    record = save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                             completed=1, order_chain=hashlib.sha256(b"order").hexdigest(), initialization={})
    reference_draw = torch.rand(5)
    replacement = torch.nn.Linear(2, 3, bias=False)
    other_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-4)
    packet = load_checkpoint(path, model=replacement, optimizer=other_optimizer, contract=contract)
    assert packet["examples_seen"] == 4 and len(record["sha256"]) == 64
    assert torch.equal(replacement.weight, before["weight"])
    assert torch.equal(torch.rand(5), reference_draw)
    assert torch.equal(other_optimizer.state[replacement.weight]["exp_avg"], optimizer.state[model.weight]["exp_avg"])
    changed = copy.deepcopy(contract)
    changed["data"] = "different"
    with pytest.raises(ValueError, match="contract differs"):
        load_checkpoint(path, model=replacement, optimizer=other_optimizer, contract=changed)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model=model, optimizer=optimizer, contract=contract,
                        completed=1, order_chain=packet["order_chain"], initialization={})


def test_evaluation_keeps_remainder_and_restores_training_mode():
    class TokenClassifier(torch.nn.Module):
        def forward(self, x):
            from types import SimpleNamespace
            return SimpleNamespace(logits=torch.nn.functional.one_hot(x, 60).float() * 10)
    model = TokenClassifier()
    x = np.arange(21, dtype=np.uint8).reshape(7, 3)
    y = x.copy()
    y[1, 1] = 59
    result = evaluate_arrays(model, x, y, batch_size=4, device="cpu")
    assert result["rows"] == 7 and result["tokens"] == 21
    assert result["token_accuracy"] == pytest.approx(20 / 21)
    assert result["cumulative_prefix_exactness"] == pytest.approx([1., 6 / 7, 6 / 7])
    assert model.training
    assert json_sha256({"b": 1, "a": 2}) == json_sha256({"a": 2, "b": 1})


def test_checkpoint_validation_rejects_dtype_before_load_can_silently_cast():
    expected = {"weight": torch.ones(2, 3)}
    with pytest.raises(ValueError, match="Invalid FP32"):
        validate_model_state({"weight": expected["weight"].bfloat16()}, expected)
    with pytest.raises(ValueError, match="Invalid FP32"):
        validate_model_state({"weight": expected["weight"] * float("nan")}, expected)
