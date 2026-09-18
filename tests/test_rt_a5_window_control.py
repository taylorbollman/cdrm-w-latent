"""Bounded CPU checks for exact backbone pairing and removal of NextLat."""
from dataclasses import asdict

import pytest
import torch

from scripts.rt_a5_common import canonical_parameter_sha256, configure_fp32_runtime, fp32_context, task_loss
from scripts.rt_a5_depth_order import build_model as build_reference
from scripts.rt_a5_window_control import build_model


@pytest.fixture(autouse=True)
def cpu_runtime():
    torch.set_num_threads(1)
    configure_fp32_runtime()


@pytest.mark.parametrize("width", [128, 512])
def test_control_is_exact_reference_backbone_without_registered_predictor(width):
    reference = build_reference("rt_window2_first", width=width)
    initial_rng = torch.get_rng_state().clone()
    control = build_model(width=width)
    assert torch.equal(initial_rng, torch.get_rng_state())
    expected = {key.removeprefix("backbone."): value for key, value in reference.state_dict().items()
                if key.startswith("backbone.")}
    assert list(control.state_dict()) == list(expected)
    assert all(torch.equal(value, expected[name]) for name, value in control.state_dict().items())
    assert asdict(control.config) == asdict(reference.backbone.config)
    assert canonical_parameter_sha256(control) == reference.nextlat_initialization["canonical_sha256"]
    assert not hasattr(control, "predictor") and not hasattr(control, "backbone")
    assert not any(name.startswith(("predictor.", "backbone.")) for name, _ in control.named_parameters())
    assert len(list(control.parameters())) == 21
    assert control.a5_initialization["reference_nextlat_initialization"] == reference.nextlat_initialization
    assert control.a5_initialization["reference_backbone_initialization"] == reference.backbone.a5_initialization
    assert control.a5_initialization["exact_reference_backbone_initialization"] is True
    assert control.a5_initialization["changed_backbone_parameter_slices"] == []
    assert control.a5_initialization["predictor_parameter_count"] == 0
    assert control.experiment_config["nextlat_enabled"] is False
    assert control.experiment_config["attention"] == reference.experiment_config["attention"]
    assert type(control.transformer.blocks[0]) is type(reference.backbone.transformer.blocks[0])
    assert type(control.transformer.blocks[1]) is type(reference.backbone.transformer.blocks[1])
    assert control.transformer.blocks[0].pre_attention_block.q_proj is control.transformer.blocks[0].q_proj
    assert control.transformer.blocks[0].post_attention_block.ff_proj is control.transformer.blocks[0].ff_proj
    if width == 512:
        assert sum(p.numel() for p in control.parameters()) == 6357504
        assert control.a5_initialization["canonical_sha256"] == "0380e2fd4cdd4db63ce6d732a0834ad054e185c2276da55d733839276d8c55ad"


def test_same_position_ce_gradients_match_frozen_reference_backbone_exactly():
    reference = build_reference("rt_window2_first", width=128, backend="naive")
    control = build_model(width=128, backend="naive")
    inputs = torch.tensor([[0, 7, 3, 9, 1, 11], [2, 1, 5, 0, 4, 16]])
    targets = torch.tensor([[2, 4, 6, 8, 10, 12], [3, 5, 7, 9, 11, 13]])
    with fp32_context("cpu"):
        actual_logits = control(inputs).logits
        expected_logits = reference(inputs).logits
        actual_loss = task_loss(actual_logits, targets)
        expected_loss = task_loss(expected_logits, targets)
        actual_loss.backward()
        expected_loss.backward()
    assert torch.equal(actual_logits, expected_logits) and torch.equal(actual_loss, expected_loss)
    expected_parameters = dict(reference.backbone.named_parameters())
    for name, parameter in control.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.equal(parameter.grad, expected_parameters[name].grad)
    assert all(parameter.grad is None for parameter in reference.predictor.parameters())


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"width": 65}, {"backend": "unknown"}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)
