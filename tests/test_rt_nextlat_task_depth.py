"""Bounded CPU checks for fresh three-layer construction and unchanged objectives."""
import copy

import pytest
import torch

from cdrm import rt_nextlat_task_depth as depth
from cdrm import rt_nextlat_tasks as original
from olmo.model import OLMoRecurrentBlockTiled, OLMoRecurrentAutogradBlock
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context
from scripts.rt_a5_window import WindowTwoRecurrentAutogradBlock, WindowTwoRecurrentBlockTiled


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    configure_fp32_runtime()


@pytest.mark.parametrize("backend,window,full", [
    ("tiled", WindowTwoRecurrentBlockTiled, OLMoRecurrentBlockTiled),
    ("naive", WindowTwoRecurrentAutogradBlock, OLMoRecurrentAutogradBlock),
])
def test_three_layer_initialization_counts_types_and_predictor_identity(backend, window, full):
    config = depth.read_configuration()
    untouched = copy.deepcopy(config)
    rng = torch.get_rng_state().clone()
    model = depth.build_model(config, backend=backend)
    repeated = depth.build_model(config, backend=backend)
    baseline = original.build_model(backend=backend)
    assert torch.equal(rng, torch.get_rng_state())
    assert config == untouched
    assert model.initialization == repeated.initialization
    assert model.initialization["depth"] == 3
    assert model.initialization["backbone_parameter_count"] == 610944
    assert model.initialization["predictor_parameter_count"] == 65792
    assert model.initialization["parameter_count"] == 676736
    assert model.initialization["predictor_sha256"] == baseline.initialization["predictor_sha256"]
    assert model.nextlat_config == baseline.nextlat_config
    assert model.initialization["canonical_sha256"] != baseline.initialization["canonical_sha256"]
    assert len(model.backbone.transformer.blocks) == 3
    for index, block in enumerate(model.backbone.transformer.blocks):
        assert type(block) is (window if index == 0 else full)
        assert block.layer_id == index
        assert block.pre_attention_block.q_proj is block.q_proj
        assert block.pre_attention_block.kv_proj is block.kv_proj
        if index == 0:
            assert block.attention_window == 2
        else:
            assert not hasattr(block, "attention_window")
    assert "wpe" not in model.backbone.transformer
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert type(model) is original.TaskNextLat


@pytest.mark.parametrize("task", ["a5", "fuzzy"])
def test_existing_task_objective_reaches_every_layer_and_predictor(task):
    config = depth.read_configuration()
    config["backbone"].update(d_model=32, n_heads=4, n_kv_heads=4, mlp_hidden_size=128)
    config["predictor_hidden_width"] = 32
    model = depth.build_model(config, backend="naive")
    local = torch.tensor([[1, 4, 2, 9], [3, 2, 8, 1]])
    inputs = original.encode_inputs(local, task)
    labels = local.flip(1)
    with fp32_context("cpu"):
        losses = original.task_loss(model, inputs, labels, task, latent_weight=1.0)
        assert torch.isfinite(losses["loss"])
        assert losses["logits"].shape == (2, 4, 60 if task == "a5" else 16)
        losses["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for block in model.backbone.transformer.blocks:
        assert block.ff_proj.weight.grad.abs().max() > 0
    assert any(p.grad.abs().max() > 0 for p in model.predictor.parameters())


@pytest.mark.parametrize("mutation", [
    lambda c: c["backbone"].update(n_layers=2),
    lambda c: c["backbone"].update(n_layers=True),
    lambda c: c["backbone"].update(block_type="sequential"),
    lambda c: c["backbone"].update(recurrent_layers=[0, 1]),
    lambda c: c["backbone"].update(alibi=False),
    lambda c: c["backbone"].update(precision="amp_bf16"),
    lambda c: c.update(embedding_injection={"variant": "input"}),
    lambda c: c.update(window_layer=1),
    lambda c: c.update(window_size=3),
    lambda c: c.update(schema="rt-nextlat-tasks-model-v1"),
])
def test_explicit_depth_route_rejects_unapproved_configuration(mutation):
    config = depth.read_configuration()
    mutation(config)
    with pytest.raises(ValueError):
        depth.read_configuration(config)


def test_configuration_difference_is_only_depth_and_explicit_schema():
    candidate = depth.read_configuration()
    candidate["backbone"]["n_layers"] = 2
    candidate["schema"] = "rt-nextlat-tasks-model-v1"
    assert candidate == original.read_configuration()
