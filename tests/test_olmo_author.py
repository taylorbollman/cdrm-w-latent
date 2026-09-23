"""Author-derived recurrence against the unchanged native FP32 scan.

CPU tests explicitly disable compiled helpers. GPU precision, compilation and
capture are separately measured; these checks do not claim that qualification.
"""
from copy import deepcopy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_author import author_recurrent_reference, author_tiled_recurrent_layer
from cdrm.pretrained.olmo_recurrent import recurrent_layer_reference
from cdrm.pretrained.olmo_rope import RopeTables, build_rope_tables


class MatrixCounter(TorchDispatchMode):
    """Independent executed mm/bmm arithmetic, excluding pointwise T=1 tiles."""
    def __init__(self):
        super().__init__()
        self.dense = self.attention = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.mm.default:
            a, b = args[:2]
            self.dense += 2*a.shape[0]*a.shape[1]*b.shape[1]
        elif func is torch.ops.aten.bmm.default:
            a, b = args[:2]
            self.attention += 2*a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]
        return func(*args, **(kwargs or {}))


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260925)


def _close(actual, expected, *, label=None):
    if expected is None:
        assert actual is None or torch.count_nonzero(actual) == 0, label
    elif actual is None:
        assert torch.count_nonzero(expected) == 0, label
    else:
        torch.testing.assert_close(actual, expected, atol=8e-6, rtol=1e-4, msg=label)


def _parameter_grads(actual, expected):
    params = dict(actual.named_parameters())
    for name, parameter in expected.named_parameters():
        _close(params[name].grad, parameter.grad, label=name)


def _positions(batch, length, *, irregular=True):
    steps = torch.arange(length)
    row = 3 + steps * (steps + 1) // 2 if irregular else steps
    return row[None].expand(batch, -1).clone() + 7 * torch.arange(batch)[:, None]


def _tables(layer, positions):
    return build_rope_tables(positions, layer.config.head_dim, layer.config.rope_freq_constant)


def _native(layer, x, positions):
    return recurrent_layer_reference(layer, x, alpha=1.0, past=None,
        query_positions=positions, key_positions=positions,
        key_valid=torch.ones(positions.shape, dtype=torch.bool), attention_backend="math")[0]


def _author(layer, x, positions, *, implementation="tiled", chunks=4, cache=True):
    tables = _tables(layer, positions)
    if implementation == "scan":
        return author_recurrent_reference(layer, x, tables, autocast_cache=cache)
    return author_tiled_recurrent_layer(layer, x, tables, compiled_helpers=False,
        bwd_mlp_chunks=chunks, autocast_cache=cache)


@pytest.mark.parametrize("length", [1, 2, 3, 5, 8, 17])
@pytest.mark.parametrize("chunks", [1, 4])
def test_author_scan_and_tiled_match_native_outputs_inputs_and_every_parameter(length, chunks):
    native = OLMoBlock(OLMoConfig.tiny())
    scan, tiled = deepcopy(native), deepcopy(native)
    xs = [torch.randn(2, length, native.config.model_dim, requires_grad=True)]
    xs += [xs[0].detach().clone().requires_grad_() for _ in range(2)]
    positions = _positions(2, length)
    outputs = (_native(native, xs[0], positions),
        _author(scan, xs[1], positions, implementation="scan"),
        _author(tiled, xs[2], positions, chunks=chunks))
    # An arbitrary, unnormalized output cotangent probes the block VJP directly.
    cotangent = torch.randn_like(outputs[0])
    for output in outputs:
        output.backward(cotangent)
    for index, layer in enumerate((scan, tiled), start=1):
        _close(outputs[index], outputs[0])
        _close(xs[index].grad, xs[0].grad)
        _parameter_grads(layer, native)
    # Q and KV share one packed optimizer-owned tensor, not split Parameters.
    assert tiled.att_proj.weight.grad.shape == native.att_proj.weight.shape
    if length > 1:
        assert tiled.att_proj.weight.grad[:native.config.model_dim].norm() > 0
    assert tiled.att_proj.weight.grad[native.config.model_dim:].norm() > 0


@pytest.mark.parametrize("implementation", ["scan", "tiled"])
@pytest.mark.parametrize("irregular", [False, True])
def test_two_recurrent_layers_preserve_interlayer_gradient_credit(implementation, irregular):
    config = OLMoConfig.tiny()
    native = torch.nn.ModuleList([OLMoBlock(config), OLMoBlock(config)])
    candidate = deepcopy(native)
    x = torch.randn(2, 7, config.model_dim, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    positions = _positions(2, 7, irregular=irregular)
    expected, actual = x, y
    for old, new in zip(native, candidate):
        expected = _native(old, expected, positions)
        actual = _author(new, actual, positions, implementation=implementation)
    probe = torch.randn_like(expected)
    expected.backward(probe)
    actual.backward(probe)
    _close(actual, expected)
    _close(y.grad, x.grad)
    _parameter_grads(candidate, native)


@pytest.mark.parametrize("frozen", ["input", "qkv", "all_weights"])
@pytest.mark.parametrize("implementation", ["scan", "tiled"])
def test_freezing_preserves_unfrozen_parameter_or_input_credit(frozen, implementation):
    native = OLMoBlock(OLMoConfig.tiny())
    candidate = deepcopy(native)
    for layer in (native, candidate):
        for name, parameter in layer.named_parameters():
            if frozen == "all_weights" or (frozen == "qkv" and name == "att_proj.weight"):
                parameter.requires_grad_(False)
    x = torch.randn(2, 5, native.config.model_dim, requires_grad=frozen != "input")
    y = x.detach().clone().requires_grad_(x.requires_grad)
    positions = _positions(2, 5)
    expected = _native(native, x, positions)
    actual = _author(candidate, y, positions, implementation=implementation)
    probe = torch.randn_like(expected)
    expected.backward(probe)
    actual.backward(probe)
    _close(actual, expected)
    _close(y.grad, x.grad)
    _parameter_grads(candidate, native)
    for parameter in candidate.parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None


@pytest.mark.parametrize("cache", [False, True])
def test_shared_tiled_calls_accumulate_only_through_outer_autograd(cache):
    native = OLMoBlock(OLMoConfig.tiny())
    candidate = deepcopy(native)
    xs = [torch.randn(1, 6, native.config.model_dim, requires_grad=True)]
    xs.append(xs[0].detach().clone().requires_grad_())
    positions = _positions(1, 6)
    losses = []
    for layer, source, is_candidate in zip((native, candidate), xs, (False, True)):
        terms = []
        value = source
        for index in range(3):
            current_positions = positions + 11 * index
            hidden = (_author(layer, value, current_positions, cache=cache)
                if is_candidate else _native(layer, value, current_positions))
            terms.append((index + 1) * hidden.sin().sum())
            shifted = torch.cat((torch.zeros_like(hidden[:, :1]), hidden[:, :-1]), dim=1)
            value = value + .15 * shifted
        loss = sum(terms)
        loss.backward()
        losses.append(loss)
    _close(losses[1], losses[0])
    _close(xs[1].grad, xs[0].grad)
    _parameter_grads(candidate, native)


def test_autograd_grad_keeps_real_parameter_and_input_dot_grad_untouched():
    native = OLMoBlock(OLMoConfig.tiny())
    candidate = deepcopy(native)
    x = torch.randn(2, 5, native.config.model_dim, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    positions = _positions(2, 5)
    expected = _native(native, x, positions)
    actual = _author(candidate, y, positions)
    targets = (y, *candidate.parameters())
    for target in targets:
        target.grad = torch.full_like(target, .125)
    before = [target.grad.clone() for target in targets]
    probe = torch.randn_like(expected)
    gradients = torch.autograd.grad(actual, targets, grad_outputs=probe)
    expected_gradients = torch.autograd.grad(expected, (x, *native.parameters()), grad_outputs=probe)
    for gradient, wanted in zip(gradients, expected_gradients):
        _close(gradient, wanted)
    for target, saved in zip(targets, before):
        torch.testing.assert_close(target.grad, saved, atol=0, rtol=0)


@pytest.mark.parametrize("length", [1, 5])
def test_zero_cotangent_produces_exact_zero_credit(length):
    layer = OLMoBlock(OLMoConfig.tiny())
    source = torch.randn(2, length, layer.config.model_dim, requires_grad=True)
    output = _author(layer, source, _positions(2, length))
    output.backward(torch.zeros_like(output))
    assert torch.count_nonzero(source.grad) == 0
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_future_inputs_cannot_change_prefix_outputs_or_derivatives():
    layer = OLMoBlock(OLMoConfig.tiny())
    source = torch.randn(1, 9, layer.config.model_dim, requires_grad=True)
    changed = source.detach().clone()
    changed[:, 4:] += 100 * torch.randn_like(changed[:, 4:])
    positions = _positions(1, 9)
    output = _author(layer, source, positions)
    other = _author(layer, changed, positions)
    torch.testing.assert_close(output[:, :4], other[:, :4], atol=0, rtol=0)
    derivative = torch.autograd.grad(output[:, :4].sin().sum(), source)[0]
    assert torch.count_nonzero(derivative[:, 4:]) == 0


def test_author_forward_preserves_input_weights_and_checkpoint_ownership():
    layer = OLMoBlock(OLMoConfig.tiny())
    source = torch.randn(1, 5, layer.config.model_dim)
    before_x = source.clone()
    before_state = {name: value.detach().clone() for name, value in layer.state_dict().items()}
    ownership = [(name, id(parameter), parameter.data_ptr()) for name, parameter in layer.named_parameters()]
    with torch.no_grad():
        expected = _native(layer, source, _positions(1, 5))
        actual = _author(layer, source, _positions(1, 5))
    _close(actual, expected)
    assert not actual.requires_grad
    torch.testing.assert_close(source, before_x, atol=0, rtol=0)
    assert ownership == [(name, id(parameter), parameter.data_ptr()) for name, parameter in layer.named_parameters()]
    assert layer.state_dict().keys() == before_state.keys()
    for name, tensor in layer.state_dict().items():
        torch.testing.assert_close(tensor, before_state[name], atol=0, rtol=0)


@pytest.mark.parametrize("cache", [False, True])
def test_cpu_bf16_private_leaf_replay_has_finite_owned_gradients(cache):
    # A small ownership/replay smoke, not a GPU numerical qualification.
    layer = OLMoBlock(OLMoConfig.tiny())
    source = torch.randn(2, 5, layer.config.model_dim, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = _author(layer, source, _positions(2, 5), cache=cache)
    assert hidden.dtype == torch.float32
    assert torch.isfinite(hidden).all()
    # Match the intended lifecycle: forward autocast exits before backward.
    gradients = torch.autograd.grad(hidden, (source, *layer.parameters()),
        grad_outputs=torch.randn_like(hidden))
    for gradient in gradients:
        assert gradient.dtype == torch.float32
        assert torch.isfinite(gradient).all()
    assert source.grad is None
    assert all(parameter.grad is None for parameter in layer.parameters())


@pytest.mark.parametrize("mutation", ["rank", "empty", "width", "dtype", "table_batch", "table_length", "table_head"])
@pytest.mark.parametrize("implementation", ["scan", "tiled"])
def test_invalid_tensor_metadata_rejected(mutation, implementation):
    layer = OLMoBlock(OLMoConfig.tiny())
    x = torch.randn(2, 5, layer.config.model_dim, requires_grad=True)
    tables = _tables(layer, _positions(2, 5))
    if mutation == "rank": x = x[0]
    if mutation == "empty": x = x[:, :0]
    if mutation == "width": x = x[:, :, :-1]
    if mutation == "dtype": x = x.double()
    if mutation == "table_batch": tables = RopeTables(tables.cos[:1], tables.sin[:1])
    if mutation == "table_length": tables = tables.slice(0, 4)
    if mutation == "table_head": tables = RopeTables(tables.cos[..., :-2], tables.sin[..., :-2])
    with pytest.raises((TypeError, ValueError)):
        if implementation == "scan":
            author_recurrent_reference(layer, x, tables)
        else:
            author_tiled_recurrent_layer(layer, x, tables, compiled_helpers=False)


@pytest.mark.parametrize("arguments", [{"bwd_mlp_chunks": 0}, {"bwd_mlp_chunks": True},
    {"bwd_mlp_chunks": 1.5}, {"precision_policy": "unknown"}, {"autocast_cache": 1},
    {"compiled_helpers": "false"}])
def test_invalid_tiled_execution_options_rejected(arguments):
    layer = OLMoBlock(OLMoConfig.tiny())
    x = torch.randn(1, 5, layer.config.model_dim, requires_grad=True)
    options = {"compiled_helpers": False, **arguments}
    with pytest.raises((TypeError, ValueError)):
        author_tiled_recurrent_layer(layer, x, _tables(layer, _positions(1, 5)), **options)


@pytest.mark.parametrize("length", [3, 5, 8])
@pytest.mark.parametrize("chunks", [1, 4])
def test_author_matrix_accounting_matches_executed_forward_and_backward(length, chunks):
    config = OLMoConfig.tiny()
    layer = OLMoBlock(config)
    batch, d, m = 2, config.model_dim, config.mlp_intermediate_size
    x = torch.randn(batch, length, d, requires_grad=True)
    tables = _tables(layer, _positions(batch, length))
    with MatrixCounter() as observed:
        hidden = author_tiled_recurrent_layer(layer, x, tables, compiled_helpers=False,
            bwd_mlp_chunks=chunks)
        hidden.backward(torch.randn_like(hidden))
    n, history = batch*length, length*(length-1)//2
    expected_dense = 50*n*d*d + 36*n*d*m
    # Historical forward has two matmuls per pair; reverse has three, except
    # singleton tiles where the author uses pointwise multiply/reduce. Full
    # attention reconstruction has two square matmuls and final dQ has one.
    expected_attention = 10*batch*d*history + 6*batch*d*length*length - 6*batch*d*(length//2)
    assert observed.dense == expected_dense
    assert observed.attention == expected_attention
