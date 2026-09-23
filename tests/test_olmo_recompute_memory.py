"""Independent arithmetic and allocation checks for bounded-memory RT backward.

These are CPU tests of the explicit fallback. CUDA kernel parity and allocated
GPU peaks are measured by the milestone probes, not inferred from these tests.
The observer records tensor metadata without retaining tensors or inspecting
their values. The old implementation is a positive control for its sensitivity.
"""

from dataclasses import replace
import copy
import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM, recurrent_layer_reference
from cdrm.pretrained.olmo_tiled import _Invocation, OLMoTiledRTForCausalLM, tiled_recurrent_layer
from cdrm.pretrained import olmo_rt_memory as memory
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(3901)


def _config():
    return replace(OLMoConfig.tiny(), model_dim=64, num_heads=4,
                   mlp_intermediate_size=128, max_context_length=512)


def _spec(precision="fp32"):
    return _Invocation(_config(), 0.37, precision, False, torch.bfloat16,
                       backward_memory="recompute")


def _product(a, b, precision, dtype):
    """Independent full-product oracle with explicit operand/result rounding."""
    arithmetic = torch.float32 if precision == "fp32" else dtype
    a, b = a.to(arithmetic).double(), b.to(arithmetic).double()
    return (a @ b).to(arithmetic).float()


def _frozen(length=37, prefix=7, *, precision="fp32", masked=True):
    dtype = torch.bfloat16 if precision == "mixed" else torch.float32
    shape = (2, 4, length, 16)
    query, temporary_key, temporary_value = [torch.randn(shape).to(dtype) for _ in range(3)]
    all_key, all_value = [torch.randn(2, 4, prefix + length, 16).to(dtype) for _ in range(2)]
    valid = torch.ones(2, prefix + length, dtype=torch.bool)
    if masked:
        valid[0, ::3] = False
        valid[1] = False  # Every row must safely produce zero attention.
    return query, temporary_key, temporary_value, all_key, all_value, valid, prefix, _spec(precision), dtype


def _oracle_attention(args):
    q, kt, vt, key, value, valid, prefix, spec, dtype = args
    length = q.shape[-2]
    score = _product(q, key.transpose(-1, -2), spec.attention_precision, dtype) / math.sqrt(q.shape[-1])
    # Construct row semantics separately from the implementation under test.
    for row in range(length):
        score[:, :, row, prefix + row] = (q[:, :, row].float() * kt[:, :, row].float()).sum(-1) / math.sqrt(q.shape[-1])
        score[:, :, row, prefix + row + 1:] = -torch.inf
    score.masked_fill_(~valid[:, None, None, :], -torch.inf)
    maximum = score.max(-1).values
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    weight = torch.exp(score - safe_maximum.unsqueeze(-1))
    denominator = weight.sum(-1)
    probability = weight / denominator.clamp_min(1e-30).unsqueeze(-1)
    diagonal = torch.stack([probability[:, :, row, prefix + row] for row in range(length)], dim=-1)
    historical = probability.clone()
    for row in range(length):
        historical[:, :, row, prefix + row] = 0
    attention = _product(historical, value, spec.attention_precision, dtype)
    attention += diagonal.unsqueeze(-1) * vt.float()
    return maximum, denominator, diagonal, attention, probability


@pytest.mark.parametrize("precision", ["fp32", "mixed"])
@pytest.mark.parametrize("length,prefix", [(1, 0), (33, 7), (65, 41)])
def test_reconstruction_matches_independent_full_product_oracle(precision, length, prefix):
    args = _frozen(length, prefix, precision=precision)
    actual = memory.attention_from_completed(*args)
    expected = _oracle_attention(args)
    for result, target in zip(actual, expected[:4]):
        torch.testing.assert_close(result, target, atol=3e-6, rtol=2e-5)
    assert torch.isfinite(actual[1]).all()
    assert torch.isfinite(actual[2]).all()
    assert torch.isfinite(actual[3]).all()
    assert torch.count_nonzero(actual[1][1]) == 0
    assert torch.count_nonzero(actual[2][1]) == 0
    assert torch.count_nonzero(actual[3][1]) == 0


@pytest.mark.parametrize("precision", ["fp32", "mixed"])
def test_prefix_and_query_adjoint_products_keep_complete_reduction_rounding(precision):
    args = _frozen(67, 39, precision=precision, masked=False)
    q, kt, vt, keys, values, valid, prefix, spec, dtype = args
    maximum, denominator, diagonal, attention, p = _oracle_attention(args)
    ga = torch.randn_like(attention) * 3.7
    dot = (ga * attention).sum(-1)
    error = p * (_product(ga, values.transpose(-1, -2), precision, dtype) - dot.unsqueeze(-1))
    for row in range(q.shape[-2]):
        error[:, :, row, prefix + row] = 0
    self_error = diagonal * ((ga * vt.float()).sum(-1) - dot)
    expected_dq = _product(error, keys, precision, dtype) / 4.0
    expected_dq += self_error.unsqueeze(-1) * kt.float() / 4.0
    expected = (expected_dq, self_error.unsqueeze(-1) * q.float() / 4.0,
                diagonal.unsqueeze(-1) * ga,
                _product(error[:, :, :, :prefix].transpose(-1, -2), q, precision, dtype) / 4.0,
                _product(p[:, :, :, :prefix].transpose(-1, -2), ga, precision, dtype))
    actual = memory.query_and_prefix_backward(
        q, kt, vt, keys, values, ga, dot, maximum, denominator, diagonal,
        valid, prefix, spec, dtype)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=4e-6, rtol=3e-5)
    if precision == "mixed":
        # This fixture distinguishes one final BF16 rounding from the tempting
        # but wrong sum of independently rounded query-partial dV products.
        partial = sum(_product(p[:, :, start:start + 32, :prefix].transpose(-1, -2),
                               ga[:, :, start:start + 32], precision, dtype)
                      for start in range(0, q.shape[-2], 32))
        assert not torch.equal(partial, expected[-1])


@pytest.mark.parametrize("precision", ["fp32", "mixed"])
def test_historical_adjoint_recomputation_preserves_full_query_reduction(precision):
    args = _frozen(97, 41, precision=precision, masked=True)
    q, _, _, keys, values, valid, prefix, spec, dtype = args
    maximum, denominator, _, attention, probability = _oracle_attention(args)
    ga = torch.randn_like(attention)
    dot = (ga * attention).sum(-1)
    # An irregular genuine history rectangle straddles the 32-column bound.
    qslice, kslice = slice(17, 94), slice(0, prefix + 3)
    p = probability[:, :, qslice, kslice]
    expected_dv = _product(p.transpose(-1, -2), ga[:, :, qslice], precision, dtype)
    error = p * (_product(ga[:, :, qslice], values[:, :, kslice].transpose(-1, -2), precision, dtype)
                 - dot[:, :, qslice].unsqueeze(-1))
    expected_dk = _product(error.transpose(-1, -2), q[:, :, qslice], precision, dtype) / 4.0
    actual = memory.historical_backward(
        q[:, :, qslice], keys[:, :, kslice], values[:, :, kslice],
        ga[:, :, qslice], dot[:, :, qslice], maximum[:, :, qslice],
        denominator[:, :, qslice], valid[:, kslice], spec, dtype)
    for result, target in zip(actual, (expected_dk, expected_dv)):
        torch.testing.assert_close(result, target, atol=4e-6, rtol=3e-5)


class _AttentionAllocationObserver(TorchDispatchMode):
    """Record attention-shaped operation outputs, including aliasing views.

    This is deliberately stronger than checking only explicit ``empty`` calls:
    full-size softmax, clone, masking or pointwise error outputs are all seen.
    It does not measure CUDA allocator peaks or retained live-storage totals.
    """

    def __init__(self, batch, heads, length, prefix):
        super().__init__()
        self.batch, self.heads = batch, heads
        self.length, self.total = length, length + prefix
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(output)[0]:
            if not isinstance(tensor, torch.Tensor):
                continue
            shape = tuple(tensor.shape)
            headed = (len(shape) == 4 and shape[0] in (1, self.batch)
                      and shape[1] in (1, self.heads))
            flattened = len(shape) == 3 and shape[0] == self.batch * self.heads
            global_mask = (len(shape) in (2, 3) and shape[-2:] in
                           ((self.length, self.total), (self.total, self.length)))
            if headed or flattened or global_mask:
                self.records.append((str(func), shape, tensor.numel()))
        return output

    @property
    def oversized(self):
        return [record for record in self.records if min(record[1][-2:]) > 32]


def _observed_block(length, prefix, backend):
    config = _config()
    layer = OLMoBlock(config)
    x = torch.randn(1, length, config.model_dim, requires_grad=True)
    past = tuple(torch.randn(1, config.num_heads, prefix, config.head_dim,
                             requires_grad=True) for _ in range(2))
    positions = torch.arange(prefix, prefix + length)[None]
    key_positions = torch.arange(prefix + length)[None]
    valid = torch.ones(1, prefix + length, dtype=torch.bool)
    valid[:, ::7] = False
    saved = []
    parameter_storage = {p.untyped_storage().data_ptr() for p in layer.parameters()}

    def pack(tensor):
        if tensor.untyped_storage().data_ptr() not in parameter_storage:
            saved.append((tuple(tensor.shape), tensor.numel() * tensor.element_size()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output, cache = tiled_recurrent_layer(
            layer, x, alpha=0.37, past=past, query_positions=positions,
            key_positions=key_positions, key_valid=valid,
            attention_precision="fp32", backward_memory=backend)
    cotangents = tuple(torch.randn_like(value) for value in (output, *cache))
    observer = _AttentionAllocationObserver(1, config.num_heads, length, prefix)
    with observer:
        torch.autograd.backward((output, *cache), cotangents)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in past)
    return observer, saved


def test_allocation_observer_detects_materialized_reference_quadratic_arrays():
    observer, _ = _observed_block(65, 17, "materialized")
    assert observer.oversized
    assert any(record[1][-2:] == (65, 82) for record in observer.oversized)
    assert any("softmax" in record[0] for record in observer.oversized)


def test_recompute_backward_scratch_and_saved_state_scale_linearly():
    observations = [_observed_block(length, prefix, "recompute")
                    for length, prefix in ((65, 17), (129, 33))]
    for (observer, saved), (length, prefix) in zip(observations, ((65, 17), (129, 33))):
        assert observer.records, "The allocation observer must have seen the actual backward"
        assert not observer.oversized, observer.oversized[:5]
        assert max(record[2] for record in observer.records) <= 4 * 32 * (length + prefix)
        assert not any(len(shape) == 4 and min(shape[-2:]) > 32 for shape, _ in saved)
    smaller_scratch, larger_scratch = [max(record[2] for record in observer.records)
                                      for observer, _ in observations]
    smaller_saved, larger_saved = [sum(size for _, size in saved) for _, saved in observations]
    assert 1.8 < larger_scratch / smaller_scratch < 2.1
    assert 1.8 < larger_saved / smaller_saved < 2.1


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("input_requires_grad", [False, True])
def test_recomputed_full_block_matches_independent_sequential_derivatives(alpha, input_requires_grad):
    layer = OLMoBlock(_config())
    reference = copy.deepcopy(layer)
    if not input_requires_grad:
        # A frozen input and a frozen parameter must not remove the remaining
        # parameter/cache derivatives or acquire fabricated .grad values.
        layer.ff_out.weight.requires_grad_(False)
        reference.ff_out.weight.requires_grad_(False)
    length, prefix = 35, 5
    x = torch.randn(2, length, 64)
    past = tuple(torch.randn(2, 4, prefix, 16) for _ in range(2))
    positions = (torch.arange(length) * 3 + 19)[None].expand(2, -1)
    key_positions = torch.cat((torch.arange(prefix)[None].expand(2, -1) * 2, positions), dim=-1)
    valid = torch.ones(2, prefix + length, dtype=torch.bool)
    valid[0, ::5] = False
    valid[1, :prefix + 3] = False
    cotangents = (torch.randn_like(x), *(torch.randn(2, 4, prefix + length, 16) for _ in range(2)))
    outputs, derivatives = [], []
    for block, recompute in ((layer, True), (reference, False)):
        local_x = x.clone().requires_grad_(input_requires_grad)
        local_past = tuple(value.clone().requires_grad_() for value in past)
        options = dict(alpha=alpha, past=local_past, query_positions=positions,
                       key_positions=key_positions, key_valid=valid)
        if recompute:
            z, cache = tiled_recurrent_layer(block, local_x, **options,
                attention_precision="fp32", backward_memory="recompute")
        else:
            z, cache = recurrent_layer_reference(block, local_x, **options, attention_backend="math")
        targets = ((local_x,) if input_requires_grad else ()) + local_past
        targets += tuple(parameter for parameter in block.parameters() if parameter.requires_grad)
        for parameter in block.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, 0.125)
        derivatives.append(torch.autograd.grad((z, *cache), targets, cotangents))
        outputs.append((z, *cache))
        for parameter in block.parameters():
            if parameter.requires_grad:
                assert torch.all(parameter.grad == 0.125)
            else:
                assert parameter.grad is None
    for actual, expected in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(actual, expected, atol=6e-6, rtol=1e-5)
    for actual, expected in zip(derivatives[0], derivatives[1]):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


def test_recompute_shared_model_calls_preserve_tied_parameter_gradients():
    config = _config()
    reference = OLMoRTForCausalLM(config, attention_backend="math")
    candidate = OLMoTiledRTForCausalLM(config, attention_backend="math",
                                     attention_precision="fp32", backward_memory="recompute")
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert candidate.readout_weight is candidate.token_embeddings.weight
    assert tuple(candidate.state_dict()) == tuple(reference.state_dict())
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19, 23]])
    modes = (RTMode((0, 1), 0.37), RTMode((0,), 1.0), RTMode((1,), 0.0))
    probes = [torch.randn(1, ids.shape[1], config.vocab_size) for _ in modes]
    gradients = []
    for model in (candidate, reference):
        loss = sum((model(ids, mode=mode).logits * probe).sum()
                   for mode, probe in zip(modes, probes))
        gradients.append(torch.autograd.grad(loss, tuple(model.parameters())))
        assert all(parameter.grad is None for parameter in model.parameters())
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected, atol=6e-5, rtol=5e-4)


@pytest.mark.parametrize("initial,changed", [("materialized", "recompute"), ("recompute", "materialized")])
def test_cache_records_and_rejects_changed_backward_memory(initial, changed):
    model = OLMoTiledRTForCausalLM(_config(), attention_backend="math", backward_memory=initial)
    mode = RTMode((0,))
    with torch.no_grad():
        cached = model(torch.tensor([[2, 3, 5]]), mode=mode, use_cache=True).past_key_values
        assert cached.backward_memory == initial
        model.backward_memory = changed
        with pytest.raises(ValueError, match="Cached RT kernel execution differs"):
            model(torch.tensor([[7]]), mode=mode, past_key_values=cached)


@pytest.mark.parametrize("initial,changed", [("materialized", "recompute"), ("recompute", "materialized")])
def test_prepared_execution_signature_rejects_changed_backward_memory(initial, changed):
    from cdrm.pretrained.nextlat import NextLatBatch
    from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
    from cdrm.pretrained.olmo_static import PreparedFBTLayout

    core = OLMoFBT(OLMoTiledRTForCausalLM(_config(), attention_backend="math", backward_memory=initial))
    ids = torch.tensor([[2, 3, 5]])
    batch = NextLatBatch(ids, torch.ones_like(ids, dtype=torch.bool), torch.zeros_like(ids))
    layout = PreparedFBTLayout(core, batch)
    mode = FBTMode(rt_mode=RTMode((0,)))
    assert layout.validate_execution(mode)["backward_memory"] == initial
    core.backbone.backward_memory = changed
    with pytest.raises(ValueError, match="execution flags changed"):
        layout.validate_execution(mode)


@pytest.mark.parametrize("invalid", [None, True, "streaming"])
def test_backward_memory_option_is_validated_in_model_and_raw_block(invalid):
    with pytest.raises(ValueError, match="backward_memory"):
        OLMoTiledRTForCausalLM(_config(), backward_memory=invalid)
    with pytest.raises(ValueError, match="backward_memory"):
        tiled_recurrent_layer(OLMoBlock(_config()), torch.randn(1, 2, 64), alpha=1.0,
            past=None, query_positions=torch.arange(2)[None], key_positions=torch.arange(2)[None],
            key_valid=torch.ones(1, 2, dtype=torch.bool), backward_memory=invalid)
