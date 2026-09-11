"""Bounded CPU checks of the NextLat port and its backbone-only interface."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from scripts.rt_a5_common import (
    build_model, canonical_parameter_sha256, fp32_context, make_optimizer, task_loss,
)
from scripts.rt_a5_nextlat import (
    A5NextLat, BASE_CONFIG_PATH, NextLatDynamicsModel, build_nextlat_model,
    nextlat_configuration, nextlat_objective,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def literal_prediction(predictor, h, embedding):
    """Independent arithmetic: RMS normalization, three affine-free GELU layers."""
    joined = torch.cat((embedding, h), dim=-1)
    normalized = joined * (joined.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt()
    normalized = normalized * predictor.norm_x.weight
    hidden = F.gelu(normalized @ predictor.mlp[0].weight.T)
    hidden = F.gelu(hidden @ predictor.mlp[2].weight.T)
    return h + hidden @ predictor.mlp[4].weight.T


class IndependentStateBackbone(nn.Module):
    """Independent h leaves distinguish stopped targets from attached sources."""

    def __init__(self, batch=2, length=4, width=4):
        super().__init__()
        self.h = nn.Parameter(torch.randn(batch, length, width))
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(60, width), "ff_out": nn.Linear(width, 60, bias=False),
        })

    def forward(self, inputs, return_pre_logits=False):
        assert inputs.shape == self.h.shape[:2]
        return SimpleNamespace(logits=self.transformer.ff_out(self.h),
                               pre_logits=self.h if return_pre_logits else None)


def independent_model(length=4):
    return A5NextLat(IndependentStateBackbone(length=length), NextLatDynamicsModel(4, 8))


def test_predictor_matches_literal_upstream_math_and_gradients():
    torch.manual_seed(27)
    predictor = NextLatDynamicsModel(4, 8)
    reference = copy.deepcopy(predictor)
    h = (torch.randn(2, 3, 4) + 2).requires_grad_()
    embedding = (torch.randn(2, 3, 4) - 1).requires_grad_()
    target = torch.randn_like(h) * 2
    ref_h = h.detach().clone().requires_grad_()
    ref_e = embedding.detach().clone().requires_grad_()
    actual = predictor(h, embedding)
    expected = literal_prediction(reference, ref_h, ref_e)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    F.smooth_l1_loss(actual, target, beta=1.0).backward()
    error = (expected - target).abs()
    # beta=1 SmoothL1, including linear tails, averaged over B*T*D.
    literal_loss = torch.where(error < 1, 0.5 * error.square(), error - 0.5).mean()
    literal_loss.backward()
    torch.testing.assert_close(h.grad, ref_h.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(embedding.grad, ref_e.grad, rtol=1e-6, atol=1e-7)
    for (_, parameter), (_, expected_parameter) in zip(predictor.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=2e-6, atol=1e-7)
    assert predictor.norm_x.eps == 1e-5
    assert all(module.bias is None for module in predictor.mlp if isinstance(module, nn.Linear))


def test_latent_loss_keeps_source_and_embedding_routes_but_stops_targets_and_head():
    torch.manual_seed(37)
    model = independent_model()
    inputs = torch.tensor([[13, 0, 7, 9], [14, 0, 8, 9]])
    labels = torch.tensor([[13, 13, 22, 31], [14, 14, 25, 35]])
    result = nextlat_objective(model, inputs, labels)
    result["latent_loss"].backward()
    assert model.backbone.h.grad[:, :-1].abs().sum() > 0
    assert torch.count_nonzero(model.backbone.h.grad[:, -1]) == 0
    embedding_gradient = model.backbone.transformer.wte.weight.grad
    for operation in (0, 7, 8, 9):
        assert embedding_gradient[operation].abs().sum() > 0
    assert torch.count_nonzero(embedding_gradient[[13, 14]]) == 0
    assert model.backbone.transformer.ff_out.weight.grad is None
    assert all(parameter.grad is not None for parameter in model.predictor.parameters())


def test_all_eleven_transitions_are_within_words_and_independent_of_labels():
    model = independent_model(length=12)
    inputs = torch.zeros(2, 12, dtype=torch.long)
    inputs[1] = torch.arange(12)
    labels = torch.zeros_like(inputs)
    captured = []
    handle = model.predictor.register_forward_pre_hook(
        lambda _module, args: captured.append(tuple(value.detach().clone() for value in args)))
    try:
        result = nextlat_objective(model, inputs, labels)
        other = nextlat_objective(model, inputs, labels + 1)
    finally:
        handle.remove()
    source, embedding = captured[0]
    assert source.shape == embedding.shape == (2, 11, 4)
    assert torch.equal(source, model.backbone.h[:, :-1])
    assert torch.equal(embedding, model.backbone.transformer.wte(inputs[:, 1:]))
    assert torch.equal(result["logits"], other["logits"])
    assert torch.equal(result["latent_loss"], other["latent_loss"])
    assert not torch.equal(result["state_loss"], other["state_loss"])
    predicted = model.predictor(source, embedding)
    error = (predicted - model.backbone.h[:, 1:].detach()).abs()
    expected = torch.where(error < 1, 0.5 * error.square(), error - 0.5).sum() / (2 * 11 * 4)
    torch.testing.assert_close(result["latent_loss"], expected)
    assert torch.equal(result["loss"], result["state_loss"] + result["latent_loss"])


def test_post_final_normalization_latents_are_used_without_normalizing_predictions():
    model = build_nextlat_model("seq", width=128)
    tokens = torch.tensor([[0, 1, 2, 3]])
    normalized = []
    source = []
    norm_hook = model.backbone.transformer.ln_f.register_forward_hook(
        lambda _module, _args, output: normalized.append(output.detach().clone()))
    predictor_hook = model.predictor.register_forward_pre_hook(
        lambda _module, args: source.append(args[0].detach().clone()))
    try:
        with fp32_context("cpu"):
            nextlat_objective(model, tokens, tokens)
    finally:
        norm_hook.remove()
        predictor_hook.remove()
    assert len(normalized) == len(source) == 1
    assert torch.equal(normalized[0][:, :-1], source[0])
    # With a zero delta, an unnormalized source is returned unchanged.
    with torch.no_grad():
        model.predictor.mlp[-1].weight.zero_()
    source_h = torch.full((1, 2, 128), 7.0)
    assert torch.equal(model.predictor(source_h, torch.zeros_like(source_h)), source_h)


def test_direct_autograd_matches_upstream_detached_leaf_gradient_reinjection():
    direct = build_nextlat_model("seq", width=128)
    reinjected = copy.deepcopy(direct)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 0, 6, 7]])
    labels = torch.tensor([[1, 3, 6, 9], [4, 4, 9, 2]])
    with fp32_context("cpu"):
        nextlat_objective(direct, tokens, labels)["loss"].backward()
        output = reinjected.backbone(tokens, return_pre_logits=True)
        hidden = output.pre_logits
        embedding = reinjected.backbone.transformer.wte(tokens[:, 1:])
        hidden_leaf = hidden.detach().requires_grad_()
        embedding_leaf = embedding.detach().requires_grad_()
        predicted = reinjected.predictor(hidden_leaf[:, :-1], embedding_leaf)
        logits = reinjected.backbone.transformer.ff_out(hidden_leaf)
        loss = task_loss(logits, labels) + F.smooth_l1_loss(predicted, hidden_leaf[:, 1:].detach())
        loss.backward()
        torch.autograd.backward((hidden, embedding), (hidden_leaf.grad, embedding_leaf.grad))
    for name, parameter in direct.named_parameters():
        counterpart = dict(reinjected.named_parameters())[name]
        assert parameter.grad is not None and counterpart.grad is not None, name
        torch.testing.assert_close(parameter.grad, counterpart.grad, rtol=2e-5, atol=2e-7, msg=name)


def test_builder_pairs_initialization_preserves_rng_and_has_exact_primary_counts():
    torch.random.default_generator.manual_seed(71)
    before = torch.random.get_rng_state().clone()
    seq = build_nextlat_model("seq")
    rt = build_nextlat_model("rt")
    assert torch.equal(before, torch.random.get_rng_state())
    assert canonical_parameter_sha256(seq.backbone) == canonical_parameter_sha256(rt.backbone)
    assert seq.nextlat_initialization["canonical_sha256"] == canonical_parameter_sha256(build_model("seq", seed=1234))
    assert seq.nextlat_initialization["predictor_sha256"] == rt.nextlat_initialization["predictor_sha256"]
    assert seq.nextlat_initialization["backbone_parameter_count"] == 6357504
    assert seq.nextlat_initialization["predictor_parameter_count"] == 1049600
    assert seq.nextlat_initialization["parameter_count"] == 7407104
    assert all(p.dtype == torch.float32 for p in rt.parameters())
    assert seq.nextlat_config["predictor"]["hidden_width"] == 512
    json.dumps(seq.nextlat_config)
    json.dumps(seq.nextlat_initialization)
    other = build_nextlat_model("seq", predictor_seed=1236)
    assert other.nextlat_initialization["canonical_sha256"] == seq.nextlat_initialization["canonical_sha256"]
    assert other.nextlat_initialization["predictor_sha256"] != seq.nextlat_initialization["predictor_sha256"]


def test_shared_embedding_head_and_optimizer_parameters_are_not_duplicated():
    model = build_nextlat_model("seq", width=128)
    optimizer = make_optimizer(model)
    names = list(dict(model.named_parameters()))
    assert sum("wte.weight" in name for name in names) == 1
    assert "backbone.transformer.ff_out.weight" in names
    assert all("embedding" not in name and "head" not in name for name, _ in model.predictor.named_parameters())
    assert model.backbone.transformer.wte.weight is not model.backbone.transformer.ff_out.weight
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len(set(id(p) for p in parameters)) == len(list(model.parameters()))
    assert set(map(id, parameters)) == set(map(id, model.parameters()))


def test_forward_and_zero_weight_bypass_predictor_and_match_baseline_update():
    model = build_nextlat_model("seq", width=128)
    baseline = build_model("seq", width=128, seed=1234)
    inputs = torch.tensor([[0, 1, 2], [3, 4, 5]])
    labels = torch.tensor([[0, 1, 4], [3, 6, 7]])
    def forbidden(_module, _args):
        raise AssertionError("Backbone-only evaluation and zero auxiliary weight must bypass predictor")
    handle = model.predictor.register_forward_pre_hook(forbidden)
    try:
        with fp32_context("cpu"):
            baseline_logits = baseline(inputs).logits
            assert torch.equal(model(inputs).logits, baseline_logits)
            result = nextlat_objective(model, inputs, labels, latent_weight=0, diagnostics=True)
            assert result["diagnostics"] == {} and float(result["latent_loss"]) == 0
            expected_loss = task_loss(baseline_logits, labels)
            assert torch.equal(result["loss"], expected_loss)
            result["loss"].backward()
            expected_loss.backward()
    finally:
        handle.remove()
    for name, parameter in model.backbone.named_parameters():
        assert torch.equal(parameter.grad, dict(baseline.named_parameters())[name].grad)
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
    before = copy.deepcopy(model.predictor.state_dict())
    make_optimizer(model).step()
    make_optimizer(baseline).step()
    for name, parameter in model.backbone.named_parameters():
        assert torch.equal(parameter, dict(baseline.named_parameters())[name])
    for name, value in before.items():
        assert torch.equal(value, model.predictor.state_dict()[name])


def test_diagnostics_are_detached_teacher_conditioned_scalars_without_changing_gradients():
    model = independent_model()
    comparison = copy.deepcopy(model)
    inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    enabled = nextlat_objective(model, inputs, inputs, diagnostics=True)
    disabled = nextlat_objective(comparison, inputs, inputs, diagnostics=False)
    assert disabled["diagnostics"] == {}
    assert set(enabled["diagnostics"]) == {
        "latent_rms", "predicted_latent_rms", "latent_variation_rms", "latent_relative_l2",
        "latent_cosine", "predicted_state_ce", "teacher_predicted_kl",
    }
    for value in enabled["diagnostics"].values():
        assert value.shape == () and not value.requires_grad and torch.isfinite(value)
    assert torch.equal(enabled["loss"], disabled["loss"])
    enabled["loss"].backward()
    disabled["loss"].backward()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.grad, dict(comparison.named_parameters())[name].grad)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_auxiliary_weights_rejected(weight):
    model = independent_model()
    inputs = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        nextlat_objective(model, inputs, inputs, latent_weight=weight)


def test_input_contract_rejects_shifted_labels_and_missing_transition():
    model = independent_model(length=1)
    inputs = torch.zeros(2, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="two operations"):
        nextlat_objective(model, inputs, inputs)
    with pytest.raises(ValueError, match="same-position"):
        nextlat_objective(model, inputs, inputs[:, :0])
    with pytest.raises(TypeError, match="int64"):
        nextlat_objective(model, inputs.int(), inputs)
    assert torch.isfinite(nextlat_objective(model, inputs, inputs, latent_weight=0)["loss"])


def test_config_preserves_released_width_rule_and_explicit_paper_alternative():
    assert json.loads(BASE_CONFIG_PATH.read_text())["evaluation"] == "backbone_only"
    assert nextlat_configuration(512)["predictor"]["hidden_width"] == 512
    assert nextlat_configuration(512, 1024)["predictor"]["hidden_width"] == 1024
    assert nextlat_configuration(128)["prediction_horizon"] == 1
    with pytest.raises(ValueError, match="explicit width"):
        nextlat_configuration(64)
    assert nextlat_configuration(64, 64)["predictor"]["hidden_width"] == 64
