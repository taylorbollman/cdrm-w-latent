"""Permanent KV-only projection preserves packed ownership and recurrent credit."""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM, recurrent_layer_reference
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM, _project_kv, tiled_recurrent_layer
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260924)


def _close(actual, expected, *, label=None):
    if expected is None:
        assert actual is None or torch.count_nonzero(actual) == 0, label
    elif actual is None:
        assert torch.count_nonzero(expected) == 0, label
    else:
        torch.testing.assert_close(actual, expected, atol=8e-6, rtol=1e-4, msg=label)


def _parameter_grads(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    for name, parameter in expected.named_parameters():
        _close(actual_parameters[name].grad, parameter.grad, label=name)


def _models():
    config = OLMoConfig.tiny()
    scan = OLMoRTForCausalLM(config, attention_backend="math")
    arms = [scan]
    for enabled in (False, True):
        model = OLMoTiledRTForCausalLM(config, attention_backend="math",
            attention_precision="fp32", kv_only_writes=enabled)
        model.load_state_dict(scan.state_dict(), strict=True)
        arms.append(model)
    return arms


@pytest.mark.parametrize("length", [1, 7])
def test_kv_helper_preserves_outputs_source_and_full_packed_weight_gradients(length):
    config = OLMoConfig.tiny()
    d = config.model_dim
    source = torch.randn(2, length, d, requires_grad=True)
    weight = torch.randn(3*d, d, requires_grad=True)
    expected_source = source.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    # The independent reference deliberately computes the original complete QKV
    # projection before discarding Q. Both paths own the same packed parameter.
    projected = F.linear(F.layer_norm(expected_source, (d,), eps=config.layer_norm_eps), expected_weight)
    _, key, value = projected.split(d, dim=-1)
    expected = tuple(part.view(2, length, config.num_heads, config.head_dim).transpose(1, 2)
        for part in (key, value))
    actual = _project_kv(source, weight, config)
    cotangents = tuple(torch.randn_like(part) for part in actual)
    torch.autograd.backward(actual, cotangents)
    torch.autograd.backward(expected, cotangents)
    for observed, target in zip(actual, expected):
        _close(observed, target)
    _close(source.grad, expected_source.grad)
    _close(weight.grad, expected_weight.grad)
    assert weight.grad.shape == (3*d, d)
    assert torch.count_nonzero(weight.grad[:d]) == 0
    assert weight.grad[d:2*d].norm() > 0
    assert weight.grad[2*d:].norm() > 0


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("backward_memory", ["materialized", "recompute"])
def test_old_and_kv_only_tiling_match_scan_with_prefix_padding_and_cache_cotangents(alpha, backward_memory):
    models = _models()
    config = models[0].config
    inputs = [torch.randn(2, 5, config.model_dim, requires_grad=True)]
    inputs += [inputs[0].detach().clone().requires_grad_() for _ in range(2)]
    original_prefix = tuple(torch.randn(2, config.num_heads, 3, config.head_dim, requires_grad=True)
        for _ in range(2))
    prefixes = [original_prefix]
    prefixes += [tuple(value.detach().clone().requires_grad_() for value in original_prefix) for _ in range(2)]
    positions = torch.tensor([[19, 22, 26, 31, 37], [13, 15, 17, 21, 22]])
    key_positions = torch.cat((torch.tensor([[7, 10, 14], [2, 4, 8]]), positions), dim=1)
    valid = torch.tensor([[True, False, True, True, True, False, True, True],
                          [False, False, False, False, False, True, True, True]])
    outputs = []
    for index, (model, source, prefix) in enumerate(zip(models, inputs, prefixes)):
        options = dict(alpha=alpha, past=prefix, query_positions=positions,
            key_positions=key_positions, key_valid=valid)
        if index == 0:
            outputs.append(recurrent_layer_reference(model.layers[0], source, attention_backend="math", **options))
        else:
            outputs.append(tiled_recurrent_layer(model.layers[0], source, attention_precision="fp32",
                backward_memory=backward_memory, kv_only_writes=(index == 2), **options))
    probes = (torch.randn_like(outputs[0][0]), *(torch.randn_like(value) for value in outputs[0][1]))
    for hidden, cache in outputs:
        torch.autograd.backward((hidden, *cache), probes)
    for index in (1, 2):
        _close(outputs[index][0], outputs[0][0])
        for actual, expected in zip(outputs[index][1], outputs[0][1]):
            _close(actual, expected)
        _close(inputs[index].grad, inputs[0].grad)
        for actual, expected in zip(prefixes[index], prefixes[0]):
            _close(actual.grad, expected.grad)
        _parameter_grads(models[index], models[0])


@pytest.mark.parametrize("length", [1, 2, 3, 8, 9])
def test_kv_only_terminal_cache_loss_and_dyadic_boundaries_match_scan(length):
    scan, _, candidate = _models()
    source = torch.randn(1, length, scan.config.model_dim, requires_grad=True)
    expected_source = source.detach().clone().requires_grad_()
    positions = torch.arange(length).unsqueeze(0)
    options = dict(alpha=0.37, past=None, query_positions=positions,
        key_positions=positions, key_valid=torch.ones((1, length), dtype=torch.bool))
    hidden, cache = tiled_recurrent_layer(candidate.layers[0], source,
        attention_precision="fp32", kv_only_writes=True, **options)
    expected_hidden, expected_cache = recurrent_layer_reference(scan.layers[0], expected_source,
        attention_backend="math", **options)
    # The last write never enters a later position's attention in this call;
    # its credit must be carried by the explicit cache-output cotangent.
    probes = tuple(torch.randn_like(value[:, :, -1]) for value in cache)
    sum((value[:, :, -1]*probe).sum() for value, probe in zip(cache, probes)).backward()
    sum((value[:, :, -1]*probe).sum() for value, probe in zip(expected_cache, probes)).backward()
    _close(hidden, expected_hidden)
    _close(source.grad, expected_source.grad)
    _parameter_grads(candidate, scan)
    assert source.grad[:, -1].norm() > 0


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("freeze_all", [False, True])
def test_kv_only_shared_calls_preserve_input_and_unfrozen_parameter_credit(alpha, freeze_all):
    models = _models()
    for model in models:
        for name, parameter in model.named_parameters():
            if freeze_all or ".att_proj." in name:
                parameter.requires_grad_(False)
    inputs = [torch.randn(1, 5, models[0].config.model_dim, requires_grad=True)]
    inputs += [inputs[0].detach().clone().requires_grad_() for _ in range(2)]
    losses = []
    for model, value in zip(models, inputs):
        terms = []
        for index, layers in enumerate(((0,), (1,), (0, 1))):
            output = model(inputs_embeds=value, mode=RTMode(layers, alpha), return_logits=False)
            hidden = output.last_hidden_state
            terms.append((index+1)*hidden.sin().mean())
            shifted = torch.cat((torch.zeros_like(hidden[:, :1]), hidden[:, :-1]), dim=1)
            value = value + .15*shifted
        loss = sum(terms)
        loss.backward()
        losses.append(loss)
    for index in (1, 2):
        _close(losses[index], losses[0])
        _close(inputs[index].grad, inputs[0].grad)
        _parameter_grads(models[index], models[0])
        for parameter in models[index].parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None
    assert inputs[2].grad.norm() > 0


def test_kv_only_autograd_grad_returns_credit_without_mutating_parameter_grads():
    scan, _, candidate = _models()
    x = torch.randn(1, 5, scan.config.model_dim)
    expected = scan(inputs_embeds=x, mode=RTMode((0, 1), 0.37), return_logits=False)
    actual = candidate(inputs_embeds=x, mode=RTMode((0, 1), 0.37), return_logits=False)
    candidate_parameters = tuple(candidate.layers.parameters())
    reference_parameters = tuple(scan.layers.parameters())
    for parameter in candidate_parameters:
        parameter.grad = torch.full_like(parameter, .125)
    snapshots = deepcopy([parameter.grad for parameter in candidate_parameters])
    probe = torch.randn_like(actual.last_hidden_state)
    gradients = torch.autograd.grad((actual.last_hidden_state*probe).sum(), candidate_parameters)
    expected_gradients = torch.autograd.grad((expected.last_hidden_state*probe).sum(), reference_parameters)
    for observed, target in zip(gradients, expected_gradients):
        _close(observed, target)
    for parameter, before in zip(candidate_parameters, snapshots):
        torch.testing.assert_close(parameter.grad, before, atol=0, rtol=0)


def test_kv_only_keeps_native_checkpoint_layout_and_parameter_ownership():
    scan, baseline, candidate = _models()
    assert baseline.state_dict().keys() == candidate.state_dict().keys() == scan.state_dict().keys()
    assert candidate.readout_weight is candidate.token_embeddings.weight
    for key, value in scan.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[key], value, atol=0, rtol=0)
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in scan.parameters())
