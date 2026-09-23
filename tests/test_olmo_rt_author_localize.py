"""Bounded CPU checks for the fixed-input/fixed-cotangent diagnostic.

These establish harness semantics, not GPU mixed-precision qualification.
"""
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained import olmo_author
from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_recurrent import recurrent_layer_reference
from cdrm.pretrained.olmo_rope import build_rope_tables
from scripts import olmo_rt_author_localize as local


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260926)


def fixture():
    layer = OLMoBlock(OLMoConfig.tiny())
    x = torch.randn(2, 5, layer.config.model_dim)
    g = torch.randn_like(x) * 7.0  # Deliberately unnormalized external cotangent.
    positions = torch.tensor([[3, 4, 6, 9, 13], [8, 9, 11, 14, 18]])
    tables = build_rope_tables(positions, layer.config.head_dim, layer.config.rope_freq_constant)
    return layer, x, g, positions, tables


@pytest.mark.parametrize("arm", ["native_fp32", "author_fp32"])
@pytest.mark.parametrize("freeze_projection", [False, True])
def test_local_vjp_matches_independent_scan_with_actual_irregular_positions(arm, freeze_projection):
    layer, x, g, positions, tables = fixture()
    if freeze_projection:
        layer.att_proj.weight.requires_grad_(False)
    expected_layer = deepcopy(layer)
    expected_x = x.clone().requires_grad_()
    expected_output, _ = recurrent_layer_reference(expected_layer, expected_x, alpha=1.0,
        past=None, query_positions=positions, key_positions=positions,
        key_valid=torch.ones(positions.shape, dtype=torch.bool), attention_backend="math")
    expected_named = [(n, p) for n, p in expected_layer.named_parameters() if p.requires_grad]
    expected = torch.autograd.grad(expected_output, (expected_x, *(p for _, p in expected_named)), g)

    sentinels = {}
    for index, (name, p) in enumerate(layer.named_parameters()):
        p.grad = torch.full_like(p, 0.125 * (index + 1))
        sentinels[name] = (p.grad, p.grad.clone(), p.clone(), p.requires_grad)
    before_x, before_g = x.clone(), g.clone()
    before_cos, before_sin = tables.cos.clone(), tables.sin.clone()
    original_helper = olmo_author._helper
    actual = local.local_vjp(layer, x, g, tables, arm, compiled_helpers=False)

    assert actual["arm"] == arm and actual["finite"] and actual["ownership_preserved"]
    assert actual["parameter_gradients"].keys() == dict(expected_named).keys()
    for got, want in [(actual["output"], expected_output), (actual["input_gradient"], expected[0])]:
        torch.testing.assert_close(got, want, atol=8e-6, rtol=1e-4)
        assert not got.requires_grad and got.device.type == "cpu"
    for (name, _), gradient in zip(expected_named, expected[1:]):
        torch.testing.assert_close(actual["parameter_gradients"][name], gradient, atol=8e-6, rtol=1e-4)
        assert not actual["parameter_gradients"][name].requires_grad
    for name, p in layer.named_parameters():
        grad_object, grad_value, value, requires_grad = sentinels[name]
        assert p.grad is grad_object and p.requires_grad == requires_grad
        assert torch.equal(p.grad, grad_value) and torch.equal(p, value)
    assert torch.equal(x, before_x) and torch.equal(g, before_g)
    assert torch.equal(tables.cos, before_cos) and torch.equal(tables.sin, before_sin)
    assert olmo_author._helper is original_helper


@pytest.mark.parametrize("arm", [a for a in local.ARMS if a.endswith("mixed")])
def test_explicit_cpu_mixed_smoke_is_finite_owned_and_restores_helper(arm):
    layer, x, g, _, tables = fixture()
    original = olmo_author._helper
    actual = local.local_vjp(layer, x, g, tables, arm, compiled_helpers=False)
    assert actual["finite"] and actual["ownership_preserved"]
    assert set(actual["parameter_gradients"]) == set(dict(layer.named_parameters()))
    assert all(p.grad is None for p in layer.parameters())
    assert olmo_author._helper is original
    if arm == "author_legacy_separate_self_mixed":
        assert actual["reconstruction"]["token0_vs_temporary_value"]["bitwise_equal"]


def test_gaussian_preserves_reference_scale_without_consuming_global_rng():
    reference = torch.arange(1, 31, dtype=torch.float32).reshape(2, 3, 5) * 0.017
    before = reference.clone()
    rng = torch.random.get_rng_state().clone()
    result = local.normalize_gaussian(reference, seed=123)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(reference, before)
    assert result.shape == reference.shape and result.dtype == reference.dtype and result.device == reference.device
    torch.testing.assert_close(result.double().norm(), reference.double().norm(), atol=0, rtol=1e-7)
    assert torch.equal(result, local.normalize_gaussian(reference, seed=123))
    assert not torch.equal(result, local.normalize_gaussian(reference, seed=124))


@pytest.mark.parametrize("reference", [torch.zeros(2), torch.tensor([float("nan")]),
    torch.tensor([float("inf")]), torch.ones(2, dtype=torch.bfloat16)])
def test_gaussian_rejects_undefined_reference_scale(reference):
    with pytest.raises(ValueError):
        local.normalize_gaussian(reference)


@pytest.mark.parametrize("bad", ["arm", "shape", "dtype", "cpu_compile"])
def test_local_vjp_rejects_ambiguous_fixture_scope(bad):
    layer, x, g, _, tables = fixture()
    arm = "bad" if bad == "arm" else "native_fp32"
    if bad == "shape":
        g = g[:, :-1]
    if bad == "dtype":
        g = g.double()
    with pytest.raises(ValueError):
        local.local_vjp(layer, x, g, tables, arm, compiled_helpers=(bad == "cpu_compile"))


def result(value, *, arm):
    tensor = torch.tensor([value], dtype=torch.float32)
    return {"arm": arm, "output": tensor, "input_gradient": tensor,
        "parameter_gradients": {"weight": tensor}}


def test_comparison_diagnostic_finite_pass_is_not_numerical_qualification():
    candidate, reference = result(2, arm="author"), result(1, arm="native")
    diagnostic = local.compare_local(candidate, reference, name="direction", policy="diagnostic")
    assert diagnostic["passed"] and not diagnostic["gate"]
    assert diagnostic["global_parameter_relative_l2"] == 1.0
    assert diagnostic["output"]["relative_l2"] == 1.0
    for policy in ("fp32", "bf16"):
        compared = local.compare_local(candidate, reference, name=policy, policy=policy)
        assert not compared["passed"] and not compared["gate"]
        identical = local.compare_local(reference, reference, name=policy, policy=policy)
        assert identical["passed"] and identical["global_parameter_relative_l2"] == 0


def test_comparison_rejects_nonfinite_or_missing_parameter_ownership():
    reference = result(1, arm="native")
    nonfinite = local.compare_local(result(float("nan"), arm="author"), reference, name="bad")
    assert not nonfinite["finite"] and not nonfinite["passed"]
    missing = result(1, arm="author")
    missing["parameter_gradients"] = {"other_weight": torch.ones(1)}
    compared = local.compare_local(missing, reference, name="ownership")
    assert not compared["ownership_matches"] and not compared["passed"]


def test_separate_self_override_is_restored_on_failure():
    original = olmo_author._helper
    with pytest.raises(RuntimeError, match="injected"):
        with local._observe_reconstruction(separate_self=True):
            assert olmo_author._helper is not original
            raise RuntimeError("injected")
    assert olmo_author._helper is original


@pytest.mark.parametrize("backend", ["native", "author"])
@pytest.mark.parametrize("fail", [False, True])
def test_full_model_interception_records_fixed_cotangent_and_restores_dispatch(monkeypatch, backend, fail):
    # Exercise interception only: replace CUDA autocast and recurrence with a
    # transparent CPU linear fixture, not a silent CPU fallback of main().
    layer = torch.nn.Linear(3, 3, bias=False)
    x = torch.randn(2, 4, 3, requires_grad=True)
    expected_output = layer(x).detach()
    native_calls, author_calls, validations = [], [], []

    def native(target, values, **kwargs):
        native_calls.append(target)
        return target(values), None

    def author(target, values, *args, **kwargs):
        author_calls.append(target)
        return target(values)

    monkeypatch.setattr(local.olmo_tiled, "tiled_recurrent_layer", native)
    monkeypatch.setattr(local.olmo_author, "author_tiled_recurrent_layer", author)
    monkeypatch.setattr(local.torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(local, "normalized_objective", lambda result: result.sums["ce"])

    def loss_sums():
        if backend == "native":
            output, _ = local.olmo_tiled.tiled_recurrent_layer(layer, x)
        else:
            output = local.olmo_author.author_tiled_recurrent_layer(layer, x, None)
        if fail:
            raise RuntimeError("injected after intercepted forward")
        return SimpleNamespace(sums={"ce": output.square().sum()})

    plan = SimpleNamespace(model=SimpleNamespace(backbone=SimpleNamespace(
        backbone=SimpleNamespace(layers=[layer]))), counts={"ce": 8}, input_tokens=8,
        validate_execution=lambda: validations.append(True), loss_sums=loss_sums)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            local.capture_native_fixture(plan, backend=backend)
    else:
        captured, metadata = local.capture_native_fixture(plan, backend=backend)
        assert torch.equal(captured["x"], x)
        assert torch.equal(captured["output"], expected_output)
        torch.testing.assert_close(captured["cotangent"], 2 * expected_output, atol=0, rtol=0)
        assert not any(captured[key].requires_grad for key in ("x", "output", "cotangent"))
        assert metadata["physical_optimizer_updates"] == 0 and metadata["backward_calls"] == 1
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert validations == [True]
    assert native_calls == ([layer] if backend == "native" else [])
    assert author_calls == ([layer] if backend == "author" else [])
    assert local.olmo_tiled.tiled_recurrent_layer is native
    assert local.olmo_author.author_tiled_recurrent_layer is author


def test_own_parameter_gradient_comparison_distinguishes_exactness_error_and_ownership():
    reference = {"a": torch.tensor([3., 4.]), "b": torch.tensor([0.])}
    same = local.compare_parameter_gradients({n: v.clone() for n, v in reference.items()}, reference)
    assert same["finite"] and same["ownership_matches"] and same["all_bitwise_equal"]
    assert same["global_parameter_relative_l2"] == 0
    altered = local.compare_parameter_gradients({"a": torch.tensor([6., 8.]), "b": torch.tensor([0.])}, reference)
    assert altered["finite"] and altered["ownership_matches"] and not altered["all_bitwise_equal"]
    assert altered["global_parameter_relative_l2"] == 1
    missing = local.compare_parameter_gradients({"a": reference["a"]}, reference)
    assert not missing["ownership_matches"] and not missing["all_bitwise_equal"]
    nonfinite = local.compare_parameter_gradients({"a": torch.tensor([float("nan"), 4.]), "b": reference["b"]}, reference)
    assert not nonfinite["finite"] and not nonfinite["all_bitwise_equal"]
