"""Focused checks of the new precision boundary and checkpoint identity."""
import copy
from dataclasses import asdict
import hashlib

import pytest
import torch

from scripts import rt_a5_bf16 as mixed
from scripts import rt_a5_bf16_train as trainer
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context, make_optimizer
from scripts.rt_a5_depth_order import build_model as original_build_model
from scripts.rt_a5_nextlat import nextlat_objective
from scripts.rt_a5_train import json_value


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    configure_fp32_runtime()


def test_actual_width_preserves_initial_tensors_rng_and_all_block_policies():
    original = original_build_model("rt_window2_first", width=512)
    rng = torch.get_rng_state().clone()
    model = mixed.build_model(width=512)
    assert torch.equal(rng, torch.get_rng_state())
    assert list(model.state_dict()) == list(original.state_dict())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.state_dict().items())
    assert model.nextlat_initialization == original.nextlat_initialization
    assert model.experiment_config == original.experiment_config
    for before, after in zip([original.backbone.config, *(b.config for b in original.backbone.transformer.blocks)],
                             [model.backbone.config, *(b.config for b in model.backbone.transformer.blocks)]):
        a, b = asdict(before), asdict(after)
        assert {name for name in a if a[name] != b[name]} == {"recurrent_precision_policy"}
        assert after.recurrent_precision_policy == "bf16_fp32_state"
    assert all(p.dtype == torch.float32 for p in model.parameters())


@pytest.mark.parametrize("latent_weight", [0.0, 1.0])
def test_inactive_adapter_exact_objective_gradients_and_diagnostics(latent_weight):
    original = original_build_model("rt_window2_first", width=128, backend="naive")
    model = mixed.build_model(width=128, backend="naive")
    x = torch.tensor([[0, 3, 7, 4], [2, 11, 14, 6]])
    y = torch.tensor([[0, 6, 12, 8], [2, 5, 17, 32]])
    results = []
    for candidate, objective in ((original, nextlat_objective), (model, mixed.objective)):
        with fp32_context("cpu"):
            result = objective(candidate, x, y, latent_weight=latent_weight, diagnostics=True)
        result["loss"].backward()
        results.append(result)
    for key in ("loss", "state_loss", "latent_loss", "logits"):
        assert torch.equal(results[0][key], results[1][key])
    assert results[0]["diagnostics"].keys() == results[1]["diagnostics"].keys()
    assert all(torch.equal(value, results[1]["diagnostics"][key]) for key, value in results[0]["diagnostics"].items())
    for a, b in zip(original.parameters(), model.parameters()):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)


def test_no_cpu_mixed_precision_fallback():
    with pytest.raises(ValueError, match="requires CUDA"):
        with mixed.mixed_context("cpu"):
            pass


def test_checkpoint_precision_schema_and_contract_are_strict(tmp_path):
    model = mixed.build_model(width=128, backend="naive")
    optimizer = make_optimizer(model)
    contract = {"batch_size": 2, "precision": "bf16_mixed", "objective": {"latent_weight": 1.0}}
    path = tmp_path / "initial.pt"
    trainer.save_checkpoint(path, model=model, optimizer=optimizer, contract=contract, completed=0,
                            order_chain=hashlib.sha256(b"rt-a5-word-order-v1").hexdigest(),
                            initialization=json_value(model.nextlat_initialization))
    packet = trainer.load_checkpoint(path, model=model, optimizer=optimizer, contract=contract)
    assert packet["schema"] == "rt-a5-bf16-training-v1"
    with pytest.raises(ValueError, match="contract differs"):
        trainer.load_checkpoint(path, model=model, optimizer=optimizer, contract={**contract, "precision": "fp32"})
    changed = copy.deepcopy(packet)
    changed["schema"] = "rt-a5-depth-order-training-v1"
    torch.save(changed, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="contract differs"):
        trainer.load_checkpoint(tmp_path / "old.pt", model=model, optimizer=optimizer, contract=contract)


def test_same_optimizer_parameter_groups_and_no_half_storage():
    original = original_build_model("rt_window2_first", width=128)
    model = mixed.build_model(width=128)
    a, b = make_optimizer(original), make_optimizer(model)
    assert a.state_dict() == b.state_dict()
    assert trainer.optimizer_names(original, a) == trainer.optimizer_names(model, b)


def test_other_architectures_are_not_silently_admitted():
    with pytest.raises(ValueError, match="original L1R"):
        mixed.build_model("seq4_alibi", width=128)
