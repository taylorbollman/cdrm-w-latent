"""Ordinary-only execution switches preserve native ownership and RT scope."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained import olmo_ordinary as ordinary
from cdrm.pretrained import olmo_tiled as tiled
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_static import PreparedFBTLayout
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(73618)


def model(**options):
    backend = options.pop("attention_backend", "math")
    return OLMoTiledRTForCausalLM(replace(OLMoConfig.tiny(), num_layers=3),
                                 attention_backend=backend, **options)


def batch(padded=False):
    ids = torch.tensor([[2, 3, 5, 7], [11, 13, 17, 19]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    if padded:
        valid[0, 0] = False
    return NextLatBatch(ids, valid, torch.zeros_like(ids).masked_fill(~valid, -1))


def test_defaults_preserve_ordinary_reference_outputs_gradients_and_state_names():
    actual = model()
    expected = OLMoForCausalLM(actual.config, attention_backend="math")
    expected.load_state_dict(actual.state_dict(), strict=True)
    ids = batch().input_ids
    want, got = expected(ids).logits, actual(ids, mode=RTMode(())).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    probe = torch.randn_like(want)
    want_grad = torch.autograd.grad(want, tuple(expected.parameters()), probe)
    got_grad = torch.autograd.grad(got, tuple(actual.parameters()), probe)
    for a, b in zip(want_grad, got_grad):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert tuple(actual.state_dict()) == tuple(expected.state_dict())
    assert actual.readout_weight is actual.token_embeddings.weight


@pytest.mark.parametrize("options, error", [
    ({"ordinary_attention_backend": "fa3"}, ValueError),
    ({"ordinary_pointwise_backend": "jit"}, ValueError),
    ({"ordinary_attention_backend": "fa4"}, ValueError),  # math oracle conflict
    ({"ordinary_checkpoint_layers": (0,)}, ValueError),
    ({"ordinary_checkpoint_layers": []}, TypeError),
    ({"ordinary_checkpoint_layers": (False,), "ordinary_activation_checkpointing": True}, TypeError),
    ({"ordinary_checkpoint_layers": (2, 0), "ordinary_activation_checkpointing": True}, ValueError),
    ({"ordinary_checkpoint_layers": (0, 0), "ordinary_activation_checkpointing": True}, ValueError),
    ({"ordinary_checkpoint_layers": (-1,), "ordinary_activation_checkpointing": True}, ValueError),
    ({"ordinary_checkpoint_layers": (3,), "ordinary_activation_checkpointing": True}, ValueError),
])
def test_constructor_rejects_unsupported_options(options, error):
    with pytest.raises(error):
        model(**options)


@pytest.mark.parametrize("selection", [None, (), (0,), (0, 2)])
@pytest.mark.parametrize("rt_layers", [(), (1,)])
def test_selective_checkpoint_calls_and_all_gradients_match(selection, rt_layers, monkeypatch):
    actual = model(ordinary_activation_checkpointing=True, ordinary_checkpoint_layers=selection)
    expected = copy.deepcopy(actual)
    expected.ordinary_activation_checkpointing = False
    expected.ordinary_checkpoint_layers = None
    calls = []
    real_checkpoint = tiled.checkpoint

    def observe(function, *args, **kwargs):
        calls.append(function.keywords["layer"])
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(tiled, "checkpoint", observe)
    mode = RTMode(rt_layers)
    want = expected(batch().input_ids, mode=mode).last_hidden_state
    got = actual(batch().input_ids, mode=mode).last_hidden_state
    effective = set(range(3) if selection is None else selection) - set(rt_layers)
    assert calls == [actual.layers[index] for index in sorted(effective)]
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    probe = torch.randn_like(want)
    want_grad = torch.autograd.grad(want, tuple(expected.parameters()), probe)
    got_grad = torch.autograd.grad(got, tuple(actual.parameters()), probe)
    for a, b in zip(want_grad, got_grad):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fbt_bootstrap_and_later_pass_checkpoint_correct_layers(monkeypatch):
    core = OLMoFBT(model(ordinary_activation_checkpointing=True, ordinary_checkpoint_layers=(0, 1)))
    layout = PreparedFBTLayout(core, batch())
    calls = []
    real_checkpoint = tiled.checkpoint

    def observe(function, *args, **kwargs):
        calls.append(function.keywords["layer"])
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(tiled, "checkpoint", observe)
    mode = FBTMode(num_passes=2, rt_mode=RTMode((1,)))
    layout.validate_execution(mode)
    layout.forward(batch().input_ids, mode).last_hidden_state.square().mean().backward()
    layers = core.backbone.layers
    assert calls == [layers[0], layers[1], layers[0]]


def test_compile_is_lazy_fullgraph_and_preserves_value_gate_order_and_gradients(monkeypatch):
    ordinary._compiled_swiglu.cache_clear()
    real_compile = torch.compile
    calls = []

    def compile_on_cpu(function, **options):
        calls.append(options)
        # Compile through actual Dynamo/AOTAutograd on CPU, without relying on
        # CUDA or claiming CPU fusion measures the production Inductor kernel.
        return real_compile(function, backend="aot_eager", **options)

    monkeypatch.setattr(torch, "compile", compile_on_cpu)
    projected = torch.randn(2, 5, 18)
    ordinary.ordinary_swiglu(projected, backend="eager")
    assert not calls
    want_input = projected.clone().requires_grad_()
    got_input = projected.clone().requires_grad_()
    up, gate = want_input.chunk(2, -1)
    want = up * F.silu(gate)
    got = ordinary.ordinary_swiglu(got_input, backend="compiled")
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    probe = torch.randn_like(want)
    want_grad = torch.autograd.grad(want, want_input, probe)[0]
    got_grad = torch.autograd.grad(got, got_input, probe)[0]
    torch.testing.assert_close(got_grad, want_grad, rtol=0, atol=0)
    ordinary.ordinary_swiglu(projected + 1, backend="compiled")
    assert calls == [{"fullgraph": True, "dynamic": False}]
    ordinary._compiled_swiglu.cache_clear()


def test_fa4_mock_receives_native_views_causal_deterministic_and_returns_attached_result(monkeypatch):
    # CUDA availability is deliberately not mocked. Only the already covered
    # device guard is bypassed for a CPU test of the adapter/interface contract.
    monkeypatch.setattr(ordinary, "_validate_fa4_inputs", lambda *args: None)
    qkv = torch.randn(2, 7, 3 * 4 * 8, requires_grad=True)
    inputs = tuple(piece.view(2, 7, 4, 8).transpose(1, 2) for piece in qkv.chunk(3, -1))
    calls = []

    def fake(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        assert q.shape == k.shape == v.shape == (2, 7, 4, 8)
        assert kwargs == {"causal": True, "deterministic": True}
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
            v.transpose(1, 2), is_causal=True).transpose(1, 2), None

    monkeypatch.setattr(ordinary, "_load_fa4", lambda: fake)
    got = ordinary.flash_attention(*inputs)
    want = F.scaled_dot_product_attention(*inputs, is_causal=True)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    for received, source in zip(calls[0][:3], inputs):
        assert received.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        assert received.stride() == source.transpose(1, 2).stride()
    probe = torch.randn_like(got)
    actual = torch.autograd.grad(got, qkv, probe, retain_graph=True)[0]
    expected = torch.autograd.grad(want, qkv, probe)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fa4_rejects_cpu_before_import_or_fallback(monkeypatch):
    monkeypatch.setattr(ordinary, "_load_fa4", lambda: pytest.fail("FA4 imported on CPU"))
    with pytest.raises(ValueError, match="requires CUDA"):
        ordinary.flash_attention(*(torch.randn(1, 2, 3, 8) for _ in range(3)))


@pytest.mark.parametrize("checkpoint", [False, True])
def test_ordinary_switches_reach_all_blocks_with_attached_parameter_gradients(monkeypatch, checkpoint):
    calls = {"attention": 0, "pointwise": 0}
    monkeypatch.setattr(ordinary, "_validate_fa4_inputs", lambda *args: None)

    def fake_fa4(q, k, v, **kwargs):
        calls["attention"] += 1
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                v.transpose(1, 2), is_causal=kwargs["causal"])
        return output.transpose(1, 2), None

    def fake_compiled(projected):
        calls["pointwise"] += 1
        return ordinary._swiglu(projected)

    monkeypatch.setattr(ordinary, "_load_fa4", lambda: fake_fa4)
    monkeypatch.setattr(ordinary, "_compiled_swiglu", lambda: fake_compiled)
    expected = model()
    actual = model(attention_backend="sdpa", ordinary_attention_backend="fa4",
        ordinary_pointwise_backend="compiled", ordinary_activation_checkpointing=checkpoint,
        ordinary_checkpoint_layers=(1,) if checkpoint else None)
    actual.load_state_dict(expected.state_dict(), strict=True)
    ids = batch().input_ids
    want = expected(ids, mode=RTMode(())).last_hidden_state
    got = actual(ids, mode=RTMode(())).last_hidden_state
    assert calls == {"attention": 3, "pointwise": 3}
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    probe = torch.randn_like(want)
    want_grad = torch.autograd.grad(want, tuple(expected.parameters()), probe)
    got_grad = torch.autograd.grad(got, tuple(actual.parameters()), probe)
    for a, b in zip(want_grad, got_grad):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert calls == {"attention": 4 if checkpoint else 3, "pointwise": 4 if checkpoint else 3}


def test_fa4_rejects_float32_projected_inputs_before_import():
    fake = SimpleNamespace(device=torch.device("cuda"), dtype=torch.float32)
    with pytest.raises(ValueError, match="BF16 or FP16"):
        ordinary._validate_fa4_inputs(fake, fake, fake)


@pytest.mark.parametrize("failure", ["import", "kernel", "signature"])
def test_fa4_errors_propagate_without_fallback(monkeypatch, failure):
    monkeypatch.setattr(ordinary, "_validate_fa4_inputs", lambda *args: None)

    def broken(*args, **kwargs):
        if failure == "import":
            raise ImportError("unavailable pinned wheel")
        if failure == "kernel":
            raise RuntimeError("unsupported launch")
        return torch.empty(1)

    monkeypatch.setattr(ordinary, "_load_fa4", broken if failure == "import" else lambda: broken)
    with pytest.raises((ImportError, RuntimeError)):
        ordinary.flash_attention(*(torch.randn(1, 2, 3, 8) for _ in range(3)))


@pytest.mark.parametrize("kwargs", [
    {"attention_mask": torch.ones(2, 4, dtype=torch.bool)},
    {"use_cache": True},
])
def test_public_fa4_rejects_masks_and_exported_cache_before_kernel(kwargs):
    base = model(attention_backend="sdpa", ordinary_attention_backend="fa4")
    with pytest.raises(ValueError, match="dense causal full sequences"):
        base(batch().input_ids, mode=RTMode(()), **kwargs)


@pytest.mark.parametrize("mode", [FBTMode(enabled=False), FBTMode(num_passes=2, rt_mode=RTMode((0, 1, 2)))])
def test_prepared_fa4_rejects_padding_including_all_rt_fbt_bootstrap(mode):
    core = OLMoFBT(model(attention_backend="sdpa", ordinary_attention_backend="fa4"))
    layout = PreparedFBTLayout(core, batch(True))
    with pytest.raises(ValueError, match="dense causal full sequences"):
        layout.validate_execution(mode)


@pytest.mark.parametrize("name,value", [
    ("ordinary_attention_backend", "fa4"),
    ("ordinary_pointwise_backend", "compiled"),
    ("ordinary_checkpoint_layers", (0, 2)),
])
def test_prepared_and_cache_metadata_pin_ordinary_execution(name, value):
    core = OLMoFBT(model(attention_backend="sdpa", ordinary_activation_checkpointing=True))
    layout = PreparedFBTLayout(core, batch())
    signature = layout.validate_execution(FBTMode(enabled=False))
    assert name in layout.metadata and name in signature
    core.eval()
    with torch.no_grad():
        cached = core.backbone(batch().input_ids, mode=RTMode(()), use_cache=True).past_key_values
    assert getattr(cached, name) == getattr(core.backbone, name)
    setattr(core.backbone, name, value)
    with pytest.raises(ValueError, match="execution flags changed"):
        layout.validate_execution(FBTMode(enabled=False), expected_signature=signature)
    with torch.no_grad(), pytest.raises(ValueError, match="Cached ordinary execution differs"):
        core.backbone(torch.tensor([[23], [29]]), mode=RTMode(()), past_key_values=cached)


def test_ordinary_options_do_not_invoke_helpers_in_all_rt_model(monkeypatch):
    from cdrm.pretrained import olmo
    def unexpected(*args, **kwargs):
        pytest.fail("ordinary helper reached from tiled RT")
    monkeypatch.setattr(olmo, "flash_attention", unexpected)
    monkeypatch.setattr(olmo, "ordinary_swiglu", unexpected)
    baseline = model(attention_backend="sdpa")
    candidate = copy.deepcopy(baseline)
    candidate.ordinary_attention_backend = "fa4"
    candidate.ordinary_pointwise_backend = "compiled"
    candidate.ordinary_activation_checkpointing = True
    candidate.ordinary_checkpoint_layers = (0, 1, 2)
    mode = RTMode((0, 1, 2))
    ids = batch().input_ids
    want, got = baseline(ids, mode=mode).logits, candidate(ids, mode=mode).logits
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert tuple(candidate.state_dict()) == tuple(baseline.state_dict())
    assert not tuple(candidate.buffers())
