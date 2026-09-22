"""Observer neutrality and causal, mask-aware F2 reconstruction contracts."""
import copy
from dataclasses import replace
import json

import pytest
import torch

from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTConfig, FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from scripts.olmo_f2_observe import (
    ActivationObserver, activation_summary, distribution,
    reconstructed_attention_summary, ordinary_impl, tiled_impl,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(41872)


def model(*, nextlat=True):
    config = replace(OLMoConfig.tiny(), model_dim=64, mlp_intermediate_size=128)
    backbone = OLMoTiledRTForCausalLM(config, attention_backend="math", attention_precision="fp32")
    return FBTNextLatLM(OLMoFBT(backbone, FBTConfig(seed=812)),
        NextLatConfig(64, lambda_latent=.3, lambda_kl=.7, vocab_chunk_size=3, seed=81), enabled=nextlat)


def batch():
    ids = torch.tensor([[2, 3, 5, 8, 13, 60], [1, 7, 11, 17, 60, 1]])
    valid = torch.tensor([[True] * 6, [False, True, True, True, True, False]])
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, docs)


def identities(module):
    return [(name, id(value), value.data_ptr(), value._version) for name, value in
            (*module.named_parameters(), *module.named_buffers())]


def originals():
    return (ordinary_impl._apply_rope, tiled_impl._apply_rope,
            tiled_impl._project, tiled_impl.tiled_recurrent_layer)


def hook_counts(module):
    return [(name, len(child._forward_hooks), len(child._forward_pre_hooks))
            for name, child in module.named_modules()]


def assert_detached_cpu(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
        assert not value.requires_grad
        assert value.grad_fn is None
    elif isinstance(value, dict):
        for child in value.values():
            assert_detached_cpu(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_detached_cpu(child)


@pytest.mark.parametrize("fbt,layers,passes,nextlat", [
    (False, (), 1, False),
    (False, (0,), 1, False),
    (True, (0,), 2, True),
    (True, (0, 1), 3, True),
])
def test_observer_is_exactly_neutral_for_outputs_losses_gradients_and_runtime_state(fbt, layers, passes, nextlat):
    expected = model(nextlat=nextlat)
    observed = copy.deepcopy(expected)
    mode = FBTMode(enabled=fbt, num_passes=passes, beta=.35, rt_mode=RTMode(layers, .37))
    value = batch()
    reference_states = []
    handle = expected.backbone.register_forward_hook(
        lambda module, args, output: reference_states.extend(h.detach().clone() for h in output.pass_hidden_states))
    expected_result = expected.loss_sums(value, backbone_kwargs={"mode": mode})
    handle.remove()
    expected_result.total.backward()
    state = {name: tensor.clone() for name, tensor in observed.state_dict().items()}
    contracts, functions = identities(observed), originals()
    hooks = hook_counts(observed)
    rng = torch.get_rng_state().clone()
    flags = [child.training for child in observed.modules()]
    with ActivationObserver(observed, layers=(0, 1), max_length=8, max_passes=3) as observer:
        result = observed.loss_sums(value, backbone_kwargs={"mode": mode})
        assert all(p.grad is None for p in observed.parameters())
    assert originals() == functions
    assert hook_counts(observed) == hooks
    assert identities(observed) == contracts
    assert [child.training for child in observed.modules()] == flags
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for name, tensor in observed.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    for name in result.sums:
        torch.testing.assert_close(result.sums[name], expected_result.sums[name], rtol=0, atol=0)
    assert result.counts == expected_result.counts
    for captured, reference_state in zip(observer.passes, reference_states):
        torch.testing.assert_close(captured["output"], reference_state, rtol=0, atol=0)
    result.total.backward()
    for (name, actual), (other_name, reference) in zip(observed.named_parameters(), expected.named_parameters()):
        assert name == other_name
        if actual.grad is None or reference.grad is None:
            assert actual.grad is reference.grad is None
        else:
            torch.testing.assert_close(actual.grad, reference.grad, rtol=0, atol=0, msg=name)
    assert_detached_cpu(observer.records)
    assert_detached_cpu(observer.passes)
    assert_detached_cpu(observer.fusions)
    report = observer.report()
    assert report["pass_count"] == passes
    assert len(report["records"]) == 2 * passes
    assert len(report["fusion"]) == (passes - 1 if fbt else 0)
    for record in report["records"]:
        rt_expected = record["layer"] in layers and (not fbt or record["pass"] > 0)
        assert record["kind"] == ("rt" if rt_expected else "ordinary")
        assert record["attention"]["finite"]
        assert record["attention"]["self_only_head_queries_excluded"] == 2 * observed.backbone.config.num_heads
        assert record["input"]["valid_token_count"] == 10
    # No tensors or nonstandard floating values leak into the durable summary.
    json.dumps(report, allow_nan=False)


def test_observer_captures_actual_temporary_and_persistent_projection_sources():
    module = model()
    mode = FBTMode(enabled=True, num_passes=2, beta=.35, rt_mode=RTMode((0,), .37))
    with ActivationObserver(module, layers=(0, 1), max_length=8) as observer:
        module.loss_sums(batch(), backbone_kwargs={"mode": mode})
    rt = next(record for record in observer.records if record["kind"] == "rt")
    layer = module.backbone.backbone.layers[0]
    q, k, v = tiled_impl._project(rt["input"], layer.att_proj.weight, layer.config)
    torch.testing.assert_close(rt["query_pre_rope"], q, rtol=0, atol=0)
    torch.testing.assert_close(rt["temporary_key_pre_rope"], k, rtol=0, atol=0)
    torch.testing.assert_close(rt["temporary_value"], v, rtol=0, atol=0)
    source = .63 * rt["input"] + .37 * rt["output"]
    torch.testing.assert_close(rt["memory_source"], source, rtol=0, atol=0)
    _, permanent_key, permanent_value = tiled_impl._project(source, layer.att_proj.weight, layer.config)
    # Original permanent projections ran one token at a time; this independent
    # batched recomputation can differ in floating-point reduction order.
    torch.testing.assert_close(rt["permanent_key_pre_rope"], permanent_key, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rt["permanent_value"], permanent_value, rtol=1e-5, atol=1e-6)
    assert not torch.equal(rt["temporary_key_pre_rope"], rt["permanent_key_pre_rope"])
    rotated = ordinary_impl._apply_rope(rt["query_pre_rope"], rt["query_positions"], layer.config.rope_freq_constant)
    torch.testing.assert_close(rt["query_post_rope"], rotated, rtol=0, atol=0)
    for record in observer.records:
        if record["kind"] == "ordinary":
            torch.testing.assert_close(record["temporary_key_pre_rope"], record["permanent_key_pre_rope"], rtol=0, atol=0)


def test_reconstruction_does_not_use_current_permanent_key_or_future_keys():
    q = torch.tensor([[[[1., 0.], [1., 0.], [1., 0.]]]])
    stored = torch.tensor([[[[4., 0.], [8., 0.], [1e6, 0.]]]])
    temporary = torch.zeros_like(stored)
    valid = torch.ones((1, 3), dtype=torch.bool)
    original = reconstructed_attention_summary(q, stored, temporary, valid)
    altered = stored.clone()
    altered[:, :, -1] = -1e6
    assert reconstructed_attention_summary(q, altered, temporary, valid) == original
    assert original["eligible_head_queries"] == 2
    assert original["self_only_head_queries_excluded"] == 1
    assert original["scaled_logits"]["max"] == pytest.approx(8 / 2**.5)
    assert original["maximum_probability"]["max"] < 1
    changed_temporary = temporary.clone()
    changed_temporary[:, :, -1, 0] = 1000
    changed = reconstructed_attention_summary(q, stored, changed_temporary, valid)
    assert changed["self_probability"]["mean"] > original["self_probability"]["mean"] + .4


def test_reconstruction_excludes_invalid_rows_and_self_only_attention_from_saturation():
    q = torch.ones((1, 2, 3, 4))
    keys = torch.ones_like(q)
    valid = torch.tensor([[False, True, True]])
    q[:, :, 0] = float("nan")
    keys[:, :, 0] = float("nan")
    row = reconstructed_attention_summary(q, keys, keys, valid)
    assert row["finite"]
    assert row["valid_head_queries"] == 4
    assert row["self_only_head_queries_excluded"] == 2
    assert row["eligible_head_queries"] == 2
    assert row["maximum_probability"]["mean"] == pytest.approx(.5)
    assert row["entropy_nats"]["mean"] == pytest.approx(.6931471805599453)
    assert row["entropy_fraction_of_uniform"]["mean"] == pytest.approx(1.)
    mask = torch.ones((1, 1, 3, 3), dtype=torch.bool)
    mask[..., -1, :] = False
    empty = reconstructed_attention_summary(q, keys, keys, valid, mask=mask)
    assert empty["eligible_head_queries"] == 0
    assert empty["maximum_probability"]["count"] == 0
    assert empty["maximum_probability"]["max"] is None


def test_summary_excludes_padding_but_reports_nonfinite_valid_values():
    values = torch.tensor([[[3., 4.], [1e20, 1e20], [float("inf"), 0.]]])
    valid = torch.tensor([[True, False, False]])
    good = activation_summary(values, valid)
    assert good["rms"] == pytest.approx((25 / 2)**.5)
    assert good["absolute_components"]["max"] == 4
    assert good["rms_by_position"][1]["valid_rows"] == 0
    valid[0, 2] = True
    bad = activation_summary(values, valid)
    assert not bad["finite"]
    assert bad["rms"] is None
    assert bad["component_values"]["nonfinite_count"] == 1
    assert distribution(torch.tensor([]))["mean"] is None
    json.dumps(bad, allow_nan=False)


def test_summary_does_not_manufacture_rms_overflow_for_finite_fp32_inputs():
    values = torch.full((1, 2, 3), 1e30)
    row = activation_summary(values, torch.ones((1, 2), dtype=torch.bool))
    assert row["finite"]
    assert row["rms"] == pytest.approx(float(values[0, 0, 0]))
    assert row["token_or_head_rms"]["nonfinite_count"] == 0
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("failure", ["raised", "too_long", "too_many_passes", "second_forward", "nested"])
def test_observer_restores_all_hooks_and_wrappers_after_failure(failure):
    module = model()
    before, hooks = originals(), hook_counts(module)
    observer = ActivationObserver(module, layers=(0, 1), max_length=4 if failure == "too_long" else 8,
                                  max_passes=1 if failure == "too_many_passes" else 3)
    with pytest.raises((RuntimeError, ValueError)):
        with observer:
            if failure == "raised":
                raise RuntimeError("injected diagnostic interruption")
            if failure == "nested":
                with ActivationObserver(module, layers=(0,)):
                    pass
            else:
                module.loss_sums(batch(), backbone_kwargs={"mode": FBTMode(num_passes=2)})
                if failure == "second_forward":
                    module.loss_sums(batch(), backbone_kwargs={"mode": FBTMode(num_passes=2)})
    assert originals() == before
    assert hook_counts(module) == hooks
    with pytest.raises(RuntimeError, match="successfully"):
        observer.report()
    # The exclusive observation lock was also released by the failure path.
    with ActivationObserver(module, layers=(0, 1), max_length=8) as retry:
        module.loss_sums(batch(), backbone_kwargs={"mode": FBTMode(enabled=False)})
    assert retry.report()["pass_count"] == 1


def test_zero_feedback_skips_fusion_but_keeps_declared_pass_and_rt_scope():
    module = model()
    mode = FBTMode(enabled=True, num_passes=2, beta=0, rt_mode=RTMode((0,), 0))
    with ActivationObserver(module, layers=(0, 1), max_length=8) as observer:
        module.loss_sums(batch(), backbone_kwargs={"mode": mode})
    report = observer.report()
    assert report["pass_count"] == 2
    assert not report["fusion"]
    assert [(r["pass"], r["layer"]) for r in report["records"] if r["kind"] == "rt"] == [(1, 0)]


def test_report_requires_exit_and_observer_is_single_use():
    module = model()
    observer = ActivationObserver(module, layers=(0, 1), max_length=8)
    with observer:
        module.loss_sums(batch(), backbone_kwargs={"mode": FBTMode(enabled=False)})
        with pytest.raises(RuntimeError, match="successfully"):
            observer.report()
    assert observer.report()["pass_count"] == 1
    with pytest.raises(RuntimeError, match="single-use"):
        with observer:
            pass
