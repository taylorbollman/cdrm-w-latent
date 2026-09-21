"""Independent derivative and cache-credit checks for native OLMo tiling."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM, recurrent_layer_reference
from cdrm.pretrained.olmo_recurrent_oracle import olmo_recurrent_model_oracle
from cdrm.pretrained.olmo_reference import build_olmo_reference
from cdrm.pretrained import olmo_tiled as tiled_module
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM, tiled_recurrent_layer
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(2501)


def _models():
    config = OLMoConfig.tiny()
    reference = OLMoRTForCausalLM(config, attention_backend="math")
    tiled = OLMoTiledRTForCausalLM(config, attention_backend="math", attention_precision="fp32")
    tiled.load_state_dict(reference.state_dict(), strict=True)
    return reference, tiled


def _close(actual, expected, *, atol=8e-6, rtol=1e-4, label=None):
    if expected is None:
        assert actual is None or torch.count_nonzero(actual) == 0, label
    elif actual is None:
        assert torch.count_nonzero(expected) == 0, label
    else:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=label)


def _parameter_grads(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    for name, expected_parameter in expected.named_parameters():
        _close(actual_parameters[name].grad, expected_parameter.grad, label=name)


def _block(function, layer, x, alpha, *, past=None, positions=None, key_positions=None, valid=None):
    prefix = 0 if past is None else past[0].shape[-2]
    if positions is None:
        positions = torch.arange(prefix, prefix + x.shape[1], device=x.device).expand(x.shape[0], -1)
    if key_positions is None:
        key_positions = torch.arange(prefix + x.shape[1], device=x.device).expand(x.shape[0], -1)
    if valid is None:
        valid = torch.ones((x.shape[0], prefix + x.shape[1]), dtype=torch.bool, device=x.device)
    options = dict(alpha=alpha, past=past, query_positions=positions, key_positions=key_positions, key_valid=valid)
    if function is tiled_recurrent_layer:
        options["attention_precision"] = "fp32"
    else:
        options["attention_backend"] = "math"
    return function(layer, x, **options)


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("selected", [(0,), (1,), (0, 1)])
def test_tiled_full_model_matches_scan_and_independent_native_history_oracle(alpha, selected):
    scan, tiled = _models()
    native = build_olmo_reference(scan.config)
    native.load_state_dict(scan.state_dict(), strict=True)
    inputs = [torch.randn(2, 7, scan.config.model_dim, requires_grad=True)]
    inputs += [inputs[0].detach().clone().requires_grad_() for _ in range(2)]
    labels = torch.randint(0, scan.config.vocab_size, (2, 7))
    with sdpa_kernel(SDPBackend.MATH):
        outputs = [
            tiled(inputs_embeds=inputs[0], mode=RTMode(selected, alpha)),
            scan(inputs_embeds=inputs[1], mode=RTMode(selected, alpha)),
            olmo_recurrent_model_oracle(native, inputs_embeds=inputs[2], selected_layers=selected, alpha=alpha),
        ]
        for result in outputs:
            F.cross_entropy(result.logits.flatten(0, 1), labels.flatten()).backward()
    for i, expected in enumerate(outputs[1:], start=1):
        _close(outputs[0].logits, expected.logits, atol=4e-6, rtol=4e-6)
        _close(outputs[0].last_hidden_state, expected.last_hidden_state, atol=4e-6, rtol=4e-6)
        _close(inputs[0].grad, inputs[i].grad)
    _parameter_grads(tiled, scan)
    _parameter_grads(tiled, native)


@pytest.mark.parametrize("length", [1, 2, 3, 8, 9])
def test_dyadic_boundary_lengths_preserve_block_outputs_and_all_derivatives(length):
    reference, tiled = _models()
    x = torch.randn(1, length, reference.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    actual, _ = _block(tiled_recurrent_layer, tiled.layers[0], x, 0.37)
    expected, _ = _block(recurrent_layer_reference, reference.layers[0], expected_x, 0.37)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    _close(actual, expected, atol=4e-6, rtol=4e-6)
    _close(x.grad, expected_x.grad)
    _parameter_grads(tiled.layers[0], reference.layers[0])


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
def test_external_prefix_values_positions_masks_and_cache_output_adjoints(alpha):
    reference, tiled = _models()
    config = reference.config
    x = torch.randn(2, 5, config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    past = tuple(torch.randn(2, config.num_heads, 3, config.head_dim, requires_grad=True) for _ in range(2))
    expected_past = tuple(value.detach().clone().requires_grad_() for value in past)
    positions = torch.tensor([[19, 22, 26, 31, 37], [13, 15, 17, 21, 22]])
    key_positions = torch.cat((torch.tensor([[7, 10, 14], [2, 4, 8]]), positions), dim=1)
    valid = torch.tensor([[True, False, True, True, True, False, True, True], [False, False, False, False, False, True, True, True]])
    actual, actual_cache = _block(tiled_recurrent_layer, tiled.layers[0], x, alpha, past=past, positions=positions, key_positions=key_positions, valid=valid)
    expected, expected_cache = _block(recurrent_layer_reference, reference.layers[0], expected_x, alpha, past=expected_past, positions=positions, key_positions=key_positions, valid=valid)
    cotangents = [torch.randn_like(actual), *(torch.randn_like(value) for value in actual_cache)]
    torch.autograd.backward((actual, *actual_cache), cotangents)
    torch.autograd.backward((expected, *expected_cache), cotangents)
    _close(actual, expected, atol=4e-6, rtol=4e-6)
    for value, target in zip(actual_cache, expected_cache):
        _close(value, target, atol=4e-6, rtol=4e-6)
    _close(x.grad, expected_x.grad)
    for value, target in zip(past, expected_past):
        _close(value.grad, target.grad)
    _parameter_grads(tiled.layers[0], reference.layers[0])


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
def test_cache_only_terminal_write_loss_reaches_source_and_block_weights(alpha):
    reference, tiled = _models()
    x = torch.randn(1, 4, reference.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    _, actual_cache = _block(tiled_recurrent_layer, tiled.layers[0], x, alpha)
    _, expected_cache = _block(recurrent_layer_reference, reference.layers[0], expected_x, alpha)
    probes = tuple(torch.randn_like(value[:, :, -1]) for value in actual_cache)
    # The terminal permanent write is absent from this chunk's attention
    # history. Its credit must nevertheless enter custom backward explicitly.
    actual_loss = sum((value[:, :, -1] * probe).sum() for value, probe in zip(actual_cache, probes))
    expected_loss = sum((value[:, :, -1] * probe).sum() for value, probe in zip(expected_cache, probes))
    actual_loss.backward()
    expected_loss.backward()
    _close(x.grad, expected_x.grad)
    _parameter_grads(tiled.layers[0], reference.layers[0])
    assert x.grad[:, -1].norm() > 0
    if alpha == 0:
        assert torch.count_nonzero(x.grad[:, :-1]) == 0
    else:
        assert x.grad[:, :-1].norm() > 0


def test_parameter_only_autograd_grad_returns_values_without_mutating_dot_grad():
    reference, tiled = _models()
    x = torch.randn(2, 5, reference.config.model_dim)  # Deliberately no input gradient.
    expected, _ = _block(recurrent_layer_reference, reference.layers[0], x, 0.37)
    actual, _ = _block(tiled_recurrent_layer, tiled.layers[0], x, 0.37)
    parameters = tuple(tiled.layers[0].parameters())
    expected_parameters = tuple(reference.layers[0].parameters())
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.125)
    snapshots = tuple(parameter.grad.clone() for parameter in parameters)
    probe = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), parameters)
    expected_gradients = torch.autograd.grad((expected * probe).sum(), expected_parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        _close(actual_gradient, expected_gradient)
    for parameter, snapshot in zip(parameters, snapshots):
        torch.testing.assert_close(parameter.grad, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("freeze_all", [False, True])
def test_frozen_parameters_do_not_remove_input_or_unfrozen_parameter_credit(freeze_all):
    reference, tiled = _models()
    for model in (reference, tiled):
        for name, parameter in model.layers[0].named_parameters():
            if freeze_all or name.startswith(("att_proj.", "ff_out.")):
                parameter.requires_grad_(False)
    x = torch.randn(1, 5, reference.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    actual, _ = _block(tiled_recurrent_layer, tiled.layers[0], x, 0.37)
    expected, _ = _block(recurrent_layer_reference, reference.layers[0], expected_x, 0.37)
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    _close(x.grad, expected_x.grad)
    _parameter_grads(tiled.layers[0], reference.layers[0])
    for parameter in tiled.layers[0].parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None


def test_shared_three_forward_modes_return_attached_parameter_and_input_credit():
    reference, tiled = _models()
    inputs = torch.randn(1, 5, reference.config.model_dim, requires_grad=True)
    expected_inputs = inputs.detach().clone().requires_grad_()
    modes = (RTMode((0,), 0.37), RTMode((1,), 1.0), RTMode((0, 1), 0.0))

    def compose(model, value):
        losses = []
        for index, mode in enumerate(modes):
            output = model(inputs_embeds=value, mode=mode)
            losses.append((index + 1) * output.logits.sin().mean())
            shifted = torch.cat((torch.zeros_like(output.last_hidden_state[:, :1]), output.last_hidden_state[:, :-1]), dim=1)
            value = value + 0.15 * shifted
        return sum(losses)

    actual_loss = compose(tiled, inputs)
    expected_loss = compose(reference, expected_inputs)
    actual_loss.backward()
    expected_loss.backward()
    _close(actual_loss, expected_loss, atol=4e-6, rtol=4e-6)
    _close(inputs.grad, expected_inputs.grad)
    _parameter_grads(tiled, reference)


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
def test_attached_cached_suffix_loss_matches_full_scan_with_nonuniform_offsets(alpha):
    reference, tiled = _models()
    tokens = torch.tensor([[1, 2, 3, 5, 8, 13, 21, 34, 55]])
    positions = torch.tensor([[3, 5, 8, 12, 17, 23, 30, 38, 47]])
    mode = RTMode((0, 1), alpha)
    expected = reference(tokens, position_ids=positions, mode=mode, use_cache=True)
    cache, offset, chunks = None, 0, []
    for length in (2, 3, 4):
        output = tiled(tokens[:, offset:offset + length], position_ids=positions[:, offset:offset + length], mode=mode, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        chunks.append(output.logits)
        offset += length
    _close(torch.cat(chunks, dim=1), expected.logits, atol=4e-6, rtol=4e-6)
    # Direct cache losses also exercise last-token writes from every block.
    cache_loss = sum(value[:, :, -1].square().mean() for pair in cache.key_values for value in pair)
    expected_cache_loss = sum(value[:, :, -1].square().mean() for pair in expected.past_key_values.key_values for value in pair)
    (chunks[-1].square().mean() + 0.1 * cache_loss).backward()
    (expected.logits[:, -4:].square().mean() + 0.1 * expected_cache_loss).backward()
    _parameter_grads(tiled, reference)


def test_future_inputs_cannot_change_prefix_outputs_or_derivatives():
    _, tiled = _models()
    x = torch.randn(1, 9, tiled.config.model_dim, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 5:] += 30 * torch.randn_like(changed[:, 5:])
    mode = RTMode((0, 1), 0.37)
    actual = tiled(inputs_embeds=x, mode=mode)
    other = tiled(inputs_embeds=changed, mode=mode)
    torch.testing.assert_close(actual.logits[:, :5], other.logits[:, :5], atol=0, rtol=0)
    derivative = torch.autograd.grad(actual.logits[:, :5].square().sum(), x)[0]
    assert torch.count_nonzero(derivative[:, 5:]) == 0


def test_forward_keeps_native_parameter_layout_and_input_storage_unchanged():
    reference, tiled = _models()
    assert list(tiled.state_dict()) == list(reference.state_dict())
    assert sum(p.numel() for p in tiled.parameters()) == sum(p.numel() for p in reference.parameters())
    assert tiled.readout_weight is tiled.token_embeddings.weight
    x = torch.randn(1, 1, tiled.config.model_dim, requires_grad=True)
    saved = x.detach().clone()
    result = tiled(inputs_embeds=x, mode=RTMode((0, 1), 1.0), return_logits=False)
    assert result.logits is None
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    result.last_hidden_state.square().sum().backward()
    torch.testing.assert_close(x, saved, atol=0, rtol=0)


def test_custom_backward_does_not_replay_sequential_or_dyadic_forward(monkeypatch):
    reference, tiled = _models()
    x = torch.randn(1, 7, tiled.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    actual, _ = _block(tiled_recurrent_layer, tiled.layers[0], x, 0.37)
    expected, _ = _block(recurrent_layer_reference, reference.layers[0], expected_x, 0.37)

    def forbidden(*args, **kwargs):
        raise AssertionError("Backward replayed a recurrent forward")

    monkeypatch.setattr(tiled_module._TiledRecurrence, "forward", staticmethod(forbidden))
    monkeypatch.setattr(tiled_module, "_add_tile", forbidden)
    monkeypatch.setattr(tiled.layers[0], "forward", forbidden)
    # Reconstructing attention from already completed states is allowed and
    # necessary; regenerating those states through the recurrence is not.
    calls = []
    reconstruct = tiled_module._attention_from_completed

    def observe(*args, **kwargs):
        calls.append(True)
        return reconstruct(*args, **kwargs)

    monkeypatch.setattr(tiled_module, "_attention_from_completed", observe)
    cotangent = torch.randn_like(actual)
    actual.backward(cotangent)
    expected.backward(cotangent)
    assert len(calls) == 1
    _close(x.grad, expected_x.grad)
    _parameter_grads(tiled.layers[0], reference.layers[0])


def test_forward_saved_activation_state_scales_linearly_not_as_attention_history():
    _, tiled = _models()
    layer = tiled.layers[0]
    parameter_storage = {parameter.data_ptr() for parameter in layer.parameters()}

    def retained_elements(length):
        retained = []

        def pack(tensor):
            if tensor.data_ptr() not in parameter_storage:
                retained.append((tuple(tensor.shape), tensor.numel()))
            return tensor

        x = torch.randn(2, length, tiled.config.model_dim, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            _block(tiled_recurrent_layer, layer, x, 0.37)
        assert not any(shape == (2, tiled.config.num_heads, length, length) for shape, _ in retained)
        return sum(elements for _, elements in retained)

    short, longer = retained_elements(9), retained_elements(17)
    assert short > 0
    assert longer <= short * 17 / 9 + 1


def test_backward_owns_positions_and_validity_when_caller_reuses_metadata():
    reference, tiled = _models()
    x = torch.randn(1, 5, tiled.config.model_dim, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    positions = torch.tensor([[3, 5, 8, 12, 17]])
    valid = torch.tensor([[True, False, True, True, True]])
    expected, _ = _block(recurrent_layer_reference, reference.layers[0], expected_x, 0.37,
                         positions=positions.clone(), key_positions=positions.clone(), valid=valid.clone())
    actual, _ = _block(tiled_recurrent_layer, tiled.layers[0], x, 0.37,
                       positions=positions, key_positions=positions, valid=valid)
    positions[:, 1::2] += 7
    valid.logical_not_()
    probe = torch.randn_like(actual)
    actual.backward(probe)
    expected.backward(probe)
    _close(x.grad, expected_x.grad)
    _parameter_grads(tiled.layers[0], reference.layers[0])


def test_tiled_cache_rejects_other_execution_histories_and_changed_precision():
    reference, tiled = _models()
    ids = torch.tensor([[1, 2, 3]])
    mode = RTMode((0,), 0.37)
    prefix = tiled(ids[:, :2], mode=mode, use_cache=True)
    with pytest.raises(TypeError):
        reference(ids[:, 2:], mode=mode, past_key_values=prefix.past_key_values)
    scan_prefix = reference(ids[:, :2], mode=mode, use_cache=True)
    with pytest.raises(TypeError):
        tiled(ids[:, 2:], mode=mode, past_key_values=scan_prefix.past_key_values)
    with pytest.raises(ValueError, match="mode"):
        tiled(ids[:, 2:], mode=RTMode((0,), 1.0), past_key_values=prefix.past_key_values)
    tiled.attention_precision = "mixed"
    with pytest.raises(ValueError, match="precision|context"):
        tiled(ids[:, 2:], mode=mode, past_key_values=prefix.past_key_values)
