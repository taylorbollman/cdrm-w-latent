"""Actual CPU module/observer oracles; no CUDA Flash numerical claim."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.olmo_campaign_aux_cotangents import _decode_tensor
from scripts.olmo_campaign_boundary_precision import (
    capture_boundaries, compare_vjps, local_vjp, match_sites, parse_args, write_fixture,
)
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
from scripts.olmo_campaign_precision_components import component_backward
from scripts.olmo_campaign_recurrence_precision import fixture_pins, state_pins
from scripts.olmo_lm_common import tree_digests
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def model_fixture():
    model, recipe, _, ids, eos = construct(SimpleNamespace(scale="tiny", length=16), "NF", torch.device("cpu"))
    model.backbone.backbone.ordinary_activation_checkpointing = True
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
                length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
    return model, recipe, fixtures


def collect(model, recipe, fixtures, precision):
    with capture_boundaries(model, fixtures) as observed, sdpa_kernel(SDPBackend.MATH):
        metrics = component_backward(model, recipe, fixtures, precision=precision, layout="sparse", objective="ce")
    model.zero_grad(set_to_none=True)
    return observed, metrics


def hook_counts(model):
    return {name: (len(module._forward_hooks), len(module._forward_pre_hooks)) for name, module in model.named_modules()}


def test_boundary_observer_preserves_actual_checkpointed_model_and_excludes_recompute():
    model, recipe, fixtures = model_fixture()
    hooks, rng, pins = hook_counts(model), torch.get_rng_state().clone(), state_pins(model)
    expected = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    with capture_boundaries(model, fixtures) as observed:
        actual = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    assert actual == expected
    assert all(torch.equal(p.grad, gradients[name]) for name, p in model.named_parameters())
    assert hook_counts(model) == hooks and torch.equal(torch.get_rng_state(), rng) and state_pins(model) == pins
    assert observed["outer_forwards"] == 2
    assert observed["original_stack_calls"] == 8 and observed["original_fusion_calls"] == 6
    assert observed["original_layer_entries"] == 8*model.backbone.config.num_layers
    assert observed["excluded_recompute_layer_entries"] == observed["original_layer_entries"]
    assert {(site["record"], site["pass"], site["kind"]) for site in observed["sites"].values()} == {
        (0, 1, "fusion"), (0, 1, "stack"), (0, 3, "fusion"), (0, 3, "stack")}
    for site in observed["sites"].values():
        assert site["cotangent_calls"] == 1
        assert all(not value.requires_grad and value.device.type == "cpu" for value in site["tensors"].values())
        assert site["tensors"]["cotangent"].dtype == torch.float32
        invalid = ~site["tensors"]["valid_output_mask"]
        assert torch.count_nonzero(site["tensors"]["cotangent"][invalid]) == 0
        if site["kind"] == "stack":
            assert site["kwargs"] == {"return_logits": False, "right_padded_causal": True,
                                      "mode": {"selected_layers": (), "alpha": 1.0}}
            assert torch.equal(site["tensors"]["attention_mask"], fixtures[0][0][0].valid_mask)


def test_observer_restores_all_module_hooks_when_inner_checkpointed_forward_raises(monkeypatch):
    model, recipe, fixtures = model_fixture()
    hooks = hook_counts(model)
    def fail(*args, **kwargs):
        raise RuntimeError("injected block error")
    monkeypatch.setattr(model.backbone.backbone.layers[1], "forward", fail)
    with pytest.raises(RuntimeError, match="injected block error"):
        with capture_boundaries(model, fixtures):
            component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective="ce")
    assert hook_counts(model) == hooks


@pytest.mark.parametrize("kind", ["fusion", "stack"])
def test_fixed_boundary_vjp_matches_direct_original_module_gradients(kind):
    model, recipe, fixtures = model_fixture()
    captured, _ = collect(model, recipe, fixtures, "fp32")
    site = captured["sites"][f"record-0/pass-1/{kind}"]
    before = tree_digests(site)
    with sdpa_kernel(SDPBackend.MATH):
        actual = local_vjp(model, site, site, precision="fp32")
    module = model.backbone.fusion if kind == "fusion" else model.backbone.backbone
    leaves = {name: site["tensors"][name].detach().clone().requires_grad_() for name in site["input_names"]}
    parameters = dict(module.named_parameters())
    with sdpa_kernel(SDPBackend.MATH):
        if kind == "fusion":
            output = module(**leaves)
        else:
            kwargs = {**site["kwargs"], **leaves, **{name: site["tensors"][name] for name in site["tensor_kwarg_names"]}}
            kwargs["mode"] = RTMode(**kwargs["mode"])
            output = module(**kwargs).last_hidden_state
        expected = torch.autograd.grad(output, (*leaves.values(), *parameters.values()),
                                       site["tensors"]["cotangent"], allow_unused=True)
    assert torch.equal(actual["values"]["output"], output)
    assert torch.equal(output, site["tensors"]["output"])
    for name, gradient in zip(leaves, expected[:len(leaves)]):
        torch.testing.assert_close(actual["values"]["inputs"][name], gradient, rtol=0, atol=0)
        assert actual["input_layouts"][name]["stride"] == site["layouts"][name]["stride"]
        assert actual["input_layouts"][name]["storage_offset"] == 0
    prefix = "backbone.fusion." if kind == "fusion" else "backbone.backbone."
    for (name, parameter), gradient in zip(parameters.items(), expected[len(leaves):]):
        if gradient is None:
            assert prefix+name in actual["unused_parameters"]
        else:
            torch.testing.assert_close(actual["values"]["parameters"][prefix+name], gradient, rtol=0, atol=0)
    assert len(actual["unused_parameters"]) == (1 if kind == "stack" else 0)
    assert all(actual["health"].values()) and tree_digests(site) == before
    assert all(p.grad is None for p in model.parameters())


def test_bf16_origin_common_cotangent_scope_and_json_fixture_roundtrip(tmp_path):
    model, recipe, fixtures = model_fixture()
    state, inputs, rng = state_pins(model), fixture_pins(fixtures), torch.get_rng_state().clone()
    fp32, _ = collect(model, recipe, fixtures, "fp32")
    bf16, _ = collect(model, recipe, fixtures, "bf16_mixed")
    match_sites(fp32["sites"], bf16["sites"])
    origins = {"fp32": fp32["sites"], "bf16": bf16["sites"]}
    receipt = write_fixture(tmp_path/"fixture.json", origins, provenance={"reference_sha256": "a"*64})
    payload = json.loads((tmp_path/"fixture.json").read_text())
    assert receipt["tensor_payloads"] == 44 and receipt["sites_per_origin"] == 4
    for origin, sites in origins.items():
        for key, site in sites.items():
            encoded = payload["origins"][origin][key]
            assert tree_digests({name: _decode_tensor(value) for name, value in encoded["tensors"].items()}) == tree_digests(site["tensors"])
    for kind in ("fusion", "stack"):
        key = f"record-0/pass-3/{kind}"
        a, common = fp32["sites"][key], bf16["sites"][key]
        with sdpa_kernel(SDPBackend.MATH):
            A = local_vjp(model, a, common, precision="fp32")
            B = local_vjp(model, common, common, precision="fp32")
            C = local_vjp(model, common, common, precision="bf16_mixed")
        assert A["input_cotangent_pins"]["cotangent"] == B["input_cotangent_pins"]["cotangent"] == C["input_cotangent_pins"]["cotangent"]
        assert B["input_cotangent_pins"] == C["input_cotangent_pins"]
        assert torch.equal(A["values"]["output"], a["tensors"]["output"])
        assert torch.equal(C["values"]["output"], common["tensors"]["output"])
        comparison = compare_vjps(C["values"], B["values"], common["tensors"]["valid_output_mask"])
        assert comparison["output_all"]["finite"] and comparison["parameter_vjps"]["all"]["parameter_tensors"] > 0
        assert comparison["output_valid"]["elements"] < comparison["output_all"]["elements"]
        assert all(row["health"]["parameter_grads_remain_none"] for row in (A, B, C))
    assert state_pins(model) == state and fixture_pins(fixtures) == inputs
    assert torch.equal(torch.get_rng_state(), rng)


def test_origin_metadata_mutation_and_non_fp32_boundaries_are_rejected():
    model, recipe, fixtures = model_fixture()
    observed, _ = collect(model, recipe, fixtures, "fp32")
    other = copy.deepcopy(observed["sites"])
    other["record-0/pass-1/stack"]["tensors"]["attention_mask"][1, 3] = False
    with pytest.raises(ValueError, match="masks/positions"):
        match_sites(observed["sites"], other)
    other = copy.deepcopy(observed["sites"])
    other["record-0/pass-1/fusion"]["layouts"]["token_input"]["stride"][0] += 64
    with pytest.raises(ValueError, match="dtype/shape/strides"):
        match_sites(observed["sites"], other)
    site = copy.deepcopy(observed["sites"]["record-0/pass-1/fusion"])
    site["tensors"]["previous_hidden"] = site["tensors"]["previous_hidden"].bfloat16()
    with pytest.raises(ValueError, match="actually be FP32"):
        local_vjp(model, site, site, precision="fp32")


@pytest.mark.parametrize("extra", [["--steps", "5"], ["--length", "1024"], ["--passes", "1,2,3"], ["--record", "1"]])
def test_cli_does_not_expand_selected_boundary_scope(extra):
    args = ["--reference-report", "matrix.json", "--reference-sha256", "a"*64, "--output-dir", "/tmp/boundary"]
    with pytest.raises(SystemExit):
        parse_args(args+extra)
