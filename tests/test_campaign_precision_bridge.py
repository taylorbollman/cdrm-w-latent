"""Contracts that prevent a precision/backend bridge from changing its question."""
from types import SimpleNamespace

import pytest
import torch

from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, component_backward
from scripts.olmo_campaign_precision_bridge import (
    capture_passes, configure_path, forward_geometry, parse_args, tensor_geometry,
)


def production_flags():
    return dict(attention_backend="sdpa", ordinary_attention_backend="sdpa",
        attention_precision="mixed", tile_backend="triton", backward_tile_backend="triton",
        backward_memory="recompute", rt_implementation="native", ordinary_pointwise_backend="eager",
        ordinary_rope_backend="native", ordinary_activation_checkpointing=True,
        cast_weights_once=True, reuse_rope=True, kv_only_writes=True)


def test_bridge_restores_mixed_attention_after_fp32_and_changes_only_tile_backends():
    base = SimpleNamespace(**production_flags())
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2)))
    model.backbone = SimpleNamespace(backbone=base)
    flags = production_flags()
    assert set(flags) == set(RUNTIME_FLAGS)
    assert configure_path(model, flags, "fp32_math_eager")["runtime_flags"]["attention_precision"] == "fp32"
    bridge = configure_path(model, flags, "bf16_math_eager")
    assert bridge["precision"] == "bf16_mixed"
    assert bridge["ordinary_sdpa"] == "MATH"
    changed = {k for k, value in bridge["runtime_flags"].items() if value != flags[k]}
    assert changed == {"tile_backend", "backward_tile_backend"}
    assert bridge["runtime_flags"]["attention_precision"] == "mixed"
    assert configure_path(model, flags, "bf16_flash_triton")["runtime_flags"] == flags
    assert flags == production_flags()
    with pytest.raises(ValueError, match="incomplete"):
        configure_path(model, {k: v for k, v in flags.items() if k != "reuse_rope"}, "bf16_math_eager")
    with pytest.raises(ValueError, match="FP32 master"):
        configure_path(model.bfloat16(), flags, "bf16_math_eager")


def test_geometry_respects_valid_positions_and_does_not_hide_zero_reference():
    record = {"batch": {"valid_mask": torch.tensor([[True, False]])},
              "pass_hidden_states": (torch.tensor([[[3., 4.], [float('nan'), float('nan')]]]),),
              "total_incoming_cotangents": [torch.tensor([[[0., 0.], [float('nan'), 1.]]])]}
    result = forward_geometry([record], [record])[0]
    assert result["hidden"]["finite"] and result["hidden"]["norm"] == 5
    assert result["hidden"]["elements"] == 2
    assert result["hidden"]["relative_l2"] == 0
    assert result["total_incoming_cotangent"]["reference_is_zero"]
    assert result["total_incoming_cotangent"]["cosine"] is None
    rotated = tensor_geometry(torch.tensor([0., 1.]), torch.tensor([1., 0.]))
    assert rotated["cosine"] == 0
    assert rotated["relative_l2"] == pytest.approx(2**.5)
    other = {**record, "batch": {"valid_mask": torch.tensor([[True, True]])}}
    with pytest.raises(ValueError, match="masks"):
        forward_geometry([record], [other])


def test_pass_observation_keeps_real_backward_exact_and_records_each_pass_once():
    torch.set_num_threads(1)
    model, recipe, _, ids, eos = construct(SimpleNamespace(scale="tiny", length=8), "NFR", torch.device("cpu"))
    fixture = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
        length=8, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
    baseline = component_backward(model, recipe, fixture, precision="fp32", layout="sparse", objective="combined")
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    with capture_passes(model, fixture) as records:
        actual = component_backward(model, recipe, fixture, precision="fp32", layout="sparse", objective="combined")
    assert actual == baseline
    assert all(torch.equal(p.grad, gradients[n]) for n, p in model.named_parameters())
    assert len(records) == 2
    assert all(len(r["pass_hidden_states"]) == 4 for r in records)
    assert all(g is not None for r in records for g in r["total_incoming_cotangents"])
    assert all(not h.requires_grad for r in records for h in r["pass_hidden_states"])
    assert len(forward_geometry(records)) == 8


@pytest.mark.parametrize("extra", [["--length", "1024"], ["--steps", "2"], ["--batch-size", "8"]])
def test_bridge_cli_does_not_broaden_into_training_or_sweeps(extra):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/tmp/bridge", *extra])
