"""Execution flags preserve RT math and invalidate incompatible cached plans."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, FBTOnlineMode, OLMoFBT
from cdrm.pretrained.olmo_static import PreparedFBTLayout
from cdrm.pretrained import olmo_tiled as tiled_module
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM, tiled_recurrent_layer
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(602729)


def model(**options):
    return OLMoTiledRTForCausalLM(replace(OLMoConfig.tiny(), num_heads=2),
                                 attention_backend="math", **options)


def block_call(layer, x, past, *, alpha, precision, cast):
    positions = torch.tensor([[4, 6, 8, 10, 12], [3, 5, 7, 11, 13]])
    key_positions = torch.cat((torch.tensor([[0, 2], [0, 1]]), positions), dim=1)
    valid = torch.tensor([[True, False, True, True, False, True, True],
                          [False, False, False, True, True, False, True]])
    return tiled_recurrent_layer(layer, x, alpha=alpha, past=past,
        query_positions=positions, key_positions=key_positions, key_valid=valid,
        attention_precision=precision, cast_weights_once=cast)


def exact_tensor_tree(left, right):
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["mixed", "fp32"])
@pytest.mark.parametrize("alpha", [0.0, .37, 1.0])
def test_cpu_bf16_raw_block_cast_once_preserves_outputs_cache_and_gradients(precision, alpha):
    reference = model().layers[0]
    candidate = copy.deepcopy(reference)
    x = torch.randn(2, 5, 32)
    prefix = tuple(torch.randn(2, 2, 2, 16) for _ in range(2))
    probes = (torch.randn_like(x), torch.randn(2, 2, 7, 16), torch.randn(2, 2, 7, 16))
    observed = []
    gradients = []
    for layer, cast in ((reference, False), (candidate, True)):
        local_x = x.clone().requires_grad_()
        local_past = tuple(value.clone().requires_grad_() for value in prefix)
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            z, cache = block_call(layer, local_x, local_past, alpha=alpha, precision=precision, cast=cast)
            output = (z, *cache)
            # Raw, unnormalized cotangents cover recurrent credit and both
            # exported cache tensors, rather than relying on a reduced CE loss.
            grads = torch.autograd.grad(output, (local_x, *local_past, *layer.parameters()),
                                        grad_outputs=probes)
        observed.append(output); gradients.append(grads)
        assert all(value.dtype == torch.float32 for value in grads)
        assert all(torch.isfinite(value).all() for value in grads)
        assert all(parameter.grad is None for parameter in layer.parameters())
    exact_tensor_tree(*observed)
    exact_tensor_tree(*gradients)


@pytest.mark.parametrize("precision", ["mixed", "fp32"])
def test_no_autocast_cast_once_flag_is_exact_fp32_noop(precision):
    baseline = model(attention_precision=precision)
    candidate = copy.deepcopy(baseline); candidate.cast_weights_once = True
    ids = torch.tensor([[2, 3, 7, 11, 17]])
    mode = RTMode((0, 1), .37)
    outputs = [item(ids, mode=mode).logits for item in (baseline, candidate)]
    exact_tensor_tree(outputs[:1], outputs[1:])
    probe = torch.randn_like(outputs[0])
    gradients = [torch.autograd.grad(out, tuple(item.parameters()), probe)
                 for item, out in zip((baseline, candidate), outputs)]
    exact_tensor_tree(*gradients)


def test_cast_once_rereads_in_place_updated_weights_on_each_invocation():
    baseline = model()
    candidate = copy.deepcopy(baseline); candidate.cast_weights_once = True
    ids = torch.tensor([[2, 3, 5, 7, 11]])
    mode = RTMode((0,), 1.0)
    before = None
    for update in range(2):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            want = baseline(ids, mode=mode).last_hidden_state
            got = candidate(ids, mode=mode).last_hidden_state
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        if update == 0:
            before = got.clone()
            with torch.no_grad():
                delta = torch.randn_like(baseline.layers[0].ff_out.weight) * .02
                baseline.layers[0].ff_out.weight.add_(delta)
                candidate.layers[0].ff_out.weight.add_(delta)
    assert not torch.equal(got, before)


def test_execution_flags_do_not_change_native_parameter_or_state_layout():
    baseline = model()
    candidate = model(cast_weights_once=True, tile_backend="triton", backward_tile_backend="triton")
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    assert candidate.readout_weight is candidate.token_embeddings.weight
    assert tuple(baseline.state_dict()) == tuple(candidate.state_dict())
    assert tuple(dict(baseline.named_parameters())) == tuple(dict(candidate.named_parameters()))
    assert not any("cast_weights" in name or "tile_backend" in name for name in candidate.state_dict())


@pytest.mark.parametrize("value", [1, None, "true", ()])
def test_constructor_rejects_nonboolean_cast_flag(value):
    with pytest.raises(TypeError, match="must be boolean"):
        model(cast_weights_once=value)


@pytest.mark.parametrize("value", [None, "flash", False, "TRITON"])
def test_constructor_rejects_unknown_tile_backend(value):
    with pytest.raises(ValueError, match="tile_backend"):
        model(tile_backend=value)


@pytest.mark.parametrize("name,value", [("cast_weights_once", 1), ("tile_backend", "unknown"), ("backward_tile_backend", "unknown")])
def test_raw_layer_revalidates_execution_options(name, value):
    layer = model().layers[0]
    options = dict(alpha=1.0, past=None, query_positions=torch.arange(2)[None],
        key_positions=torch.arange(2)[None], key_valid=torch.ones(1, 2, dtype=torch.bool))
    options[name] = value
    with pytest.raises((TypeError, ValueError)):
        tiled_recurrent_layer(layer, torch.randn(1, 2, 32), **options)


def test_requested_triton_rt_rejects_cpu_before_any_kernel_call(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CPU RT reached the CUDA helper")
    monkeypatch.setattr("cdrm.pretrained.olmo_rt_kernels.add_tile", unexpected)
    base = model(tile_backend="triton")
    with pytest.raises(ValueError, match="require CUDA"):
        base(torch.tensor([[2, 3]]), mode=RTMode((0,)))


def test_inactive_rt_does_not_invoke_triton_backend(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("ordinary mode invoked an RT kernel")
    monkeypatch.setattr("cdrm.pretrained.olmo_rt_kernels.add_tile", unexpected)
    baseline = model(); candidate = copy.deepcopy(baseline)
    candidate.tile_backend = "triton"; candidate.cast_weights_once = True
    candidate.backward_tile_backend = "triton"
    ids = torch.tensor([[2, 3, 5]])
    torch.testing.assert_close(candidate(ids, mode=RTMode(())).logits,
                               baseline(ids, mode=RTMode(())).logits, rtol=0, atol=0)


@pytest.mark.parametrize("name,value", [("cast_weights_once", True), ("tile_backend", "triton"), ("backward_tile_backend", "triton")])
def test_cached_rt_execution_flags_are_pinned(name, value):
    base = model().eval()
    with torch.no_grad():
        cached = base(torch.tensor([[2, 3]]), mode=RTMode((0,)), use_cache=True).past_key_values
        assert cached.cast_weights_once is False and cached.tile_backend == "eager"
        setattr(base, name, value)
        with pytest.raises(ValueError, match="Cached RT kernel execution"):
            base(torch.tensor([[5]]), mode=RTMode((0,)), past_key_values=cached, use_cache=True)


@pytest.mark.parametrize("name,value", [("cast_weights_once", True), ("tile_backend", "triton"), ("backward_tile_backend", "triton")])
def test_online_fbt_inner_cache_also_rejects_rt_kernel_option_changes(name, value):
    core = OLMoFBT(model()).eval()
    mode = FBTOnlineMode(rt_mode=RTMode((0,)))
    with torch.no_grad():
        cached = core.forward_online(torch.tensor([[2, 3]]), mode=mode).past_key_values
        setattr(core.backbone, name, value)
        with pytest.raises(ValueError, match="Cached RT kernel execution"):
            core.forward_online(torch.tensor([[5]]), mode=mode, past_key_values=cached)


@pytest.mark.parametrize("name,value", [("cast_weights_once", True), ("tile_backend", "triton"), ("backward_tile_backend", "triton")])
def test_static_layout_guards_execution_flags_before_capture_or_replay(name, value):
    core = OLMoFBT(model())
    ids = torch.tensor([[2, 3, 5]])
    batch = NextLatBatch(ids, torch.ones_like(ids, dtype=torch.bool), torch.zeros_like(ids))
    layout = PreparedFBTLayout(core, batch)
    mode = FBTMode(rt_mode=RTMode((0,)))
    signature = layout.validate_execution(mode)
    assert signature["cast_weights_once"] is False
    assert signature["tile_backend"] == "eager"
    assert signature["backward_tile_backend"] == "eager"
    setattr(core.backbone, name, value)
    with pytest.raises(ValueError, match="execution flags changed"):
        layout.validate_execution(mode, expected_signature=signature)


def test_static_layout_allows_weight_value_updates_with_cast_once_enabled():
    core = OLMoFBT(model(cast_weights_once=True))
    ids = torch.tensor([[2, 3, 5]])
    batch = NextLatBatch(ids, torch.ones_like(ids, dtype=torch.bool), torch.zeros_like(ids))
    layout = PreparedFBTLayout(core, batch)
    mode = FBTMode(rt_mode=RTMode((0,)))
    signature = layout.validate_execution(mode)
    with torch.no_grad():
        core.backbone.layers[0].ff_out.weight.add_(.01)
    layout.validate_execution(mode, expected_signature=signature)


@pytest.mark.parametrize("precision,projection_dtype,key_dtype,dim,target,source,expected", [
    ("mixed", torch.bfloat16, torch.bfloat16, 16, 3, 5, "triton"),
    ("fp32", torch.bfloat16, torch.bfloat16, 16, 3, 5, "eager"),
    ("mixed", torch.float32, torch.float32, 16, 3, 5, "eager"),
    ("mixed", torch.bfloat16, torch.bfloat16, 8, 3, 5, "eager"),
    ("mixed", torch.bfloat16, torch.bfloat16, 16, 257, 5, "eager"),
    ("mixed", torch.bfloat16, torch.bfloat16, 16, 3, 257, "eager"),
    ("mixed", torch.bfloat16, torch.float32, 16, 3, 5, "eager"),
])
def test_tile_dispatch_uses_supported_metadata_without_allocating_cuda(
        monkeypatch, precision, projection_dtype, key_dtype, dim, target, source, expected):
    # These are metadata-only stubs, not fake allocations or CUDA execution.
    class MetadataTensor:
        device = SimpleNamespace(type="cuda")
        dtype = projection_dtype
        def __init__(self, shape):
            self.shape = shape
        def transpose(self, *args):
            return self
    class ReachedEager(Exception):
        pass
    def eager(*args, **kwargs):
        raise ReachedEager
    def triton(*args, **kwargs):
        return "triton"
    monkeypatch.setattr(tiled_module, "_mm", eager)
    monkeypatch.setattr("cdrm.pretrained.olmo_rt_kernels.add_tile", triton)
    spec = SimpleNamespace(tile_backend="triton", attention_precision=precision,
                           config=SimpleNamespace(head_dim=dim))
    args = (MetadataTensor((1, 2, target, dim)), MetadataTensor((1, 2, source, dim)),
            MetadataTensor((1, 2, source, dim)), None, None, None, None, spec, projection_dtype)
    args[1].dtype = key_dtype
    if expected == "triton":
        assert tiled_module._add_tile(*args) == "triton"
    else:
        with pytest.raises(ReachedEager):
            tiled_module._add_tile(*args)
