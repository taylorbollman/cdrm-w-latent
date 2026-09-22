"""Ordinary-only recomputation preserves native RT and shared FBT gradients."""
import copy
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained import olmo_tiled as tiled_module
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(910235)


def _base(enabled=False):
    return OLMoTiledRTForCausalLM(
        replace(OLMoConfig.tiny(), num_layers=3), attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=enabled)


def _models():
    baseline = _base()
    candidate = copy.deepcopy(baseline)
    candidate.ordinary_activation_checkpointing = True
    return baseline, candidate


def _batch():
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 23, 60],
                        [4, 9, 14, 18, 27, 60, 1, 1, 1]])
    valid = torch.tensor([[True]*9, [True]*6+[False]*3])
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    ce, latent, kl = [valid.clone() for _ in range(3)]
    ce[0, :3] = False; kl[1, :2] = False; latent[0, 2] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def _assert_grads(left, right):
    actual, expected = dict(left.named_parameters()), dict(right.named_parameters())
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        reference = expected[name]
        assert (value.grad is None) == (reference.grad is None), name
        if value.grad is not None:
            torch.testing.assert_close(value.grad, reference.grad, rtol=1e-6, atol=1e-7, msg=name)
            assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("mode", [RTMode(()), RTMode((0,), .37), RTMode((1,), 1.), RTMode((0, 2), 1.)])
def test_ordinary_and_tiled_paths_preserve_outputs_inputs_and_parameter_gradients(mode):
    baseline, candidate = _models()
    batch = _batch()
    inputs = [torch.randn(2, 9, baseline.config.model_dim, requires_grad=True)]
    inputs.append(inputs[0].detach().clone().requires_grad_())
    positions = torch.tensor([[2, 4, 5, 7, 11, 14, 15, 18, 20], [3, 5, 8, 13, 14, 19, 19, 19, 19]])
    outputs = [model(inputs_embeds=x, mode=mode, attention_mask=batch.valid_mask, position_ids=positions)
               for model, x in zip((baseline, candidate), inputs)]
    torch.testing.assert_close(outputs[0].logits, outputs[1].logits, rtol=0, atol=0)
    cotangent = torch.randn_like(outputs[0].logits)
    for output in outputs:
        output.logits.backward(cotangent)
    torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=1e-6, atol=1e-7)
    _assert_grads(candidate, baseline)


@pytest.mark.parametrize("mode", [
    FBTMode(enabled=True, num_passes=2, beta=1., rt_mode=RTMode((0,), 1.)),
    FBTMode(enabled=True, num_passes=3, beta=.35, rt_mode=RTMode((1,), .37)),
])
def test_shared_fbt_rt_nextlat_losses_and_all_gradients_match(mode):
    baseline = FBTNextLatLM(OLMoFBT(_base()), NextLatConfig(32, vocab_chunk_size=3))
    candidate = copy.deepcopy(baseline)
    candidate.backbone.backbone.ordinary_activation_checkpointing = True
    batch = _batch()
    left, right = [model.loss_sums(batch, backbone_kwargs={"mode": mode}) for model in (baseline, candidate)]
    assert left.counts == right.counts and left.pass_coefficients == right.pass_coefficients
    for name in left.sums:
        torch.testing.assert_close(left.sums[name], right.sums[name], rtol=0, atol=0)
    left.total.backward(); right.total.backward()
    _assert_grads(candidate, baseline)
    assert all(parameter.grad is not None for parameter in candidate.parameters())


def test_recomputation_calls_only_ordinary_layers_and_binds_shared_invocations(monkeypatch):
    model = OLMoFBT(_base(True))
    counts = [0] * model.config.num_layers
    handles = []
    for index, layer in enumerate(model.backbone.layers):
        def before(_layer, _inputs, index=index):
            counts[index] += 1
        handles.append(layer.register_forward_pre_hook(before))
    recurrent_calls = []
    original = tiled_module.tiled_recurrent_layer
    def observed(layer, *args, **kwargs):
        recurrent_calls.append(layer)
        return original(layer, *args, **kwargs)
    monkeypatch.setattr(tiled_module, "tiled_recurrent_layer", observed)
    batch = _batch()
    try:
        mode = FBTMode(num_passes=3, beta=.35, rt_mode=RTMode((1,), .37))
        output = model(batch.input_ids, attention_mask=batch.valid_mask, document_ids=batch.document_ids,
                       mode=mode, return_logits=False)
        assert counts == [3, 1, 3]  # Ordinary bootstrap plus two feedback passes.
        assert recurrent_calls == [model.backbone.layers[1]] * 2
        output.last_hidden_state.backward(torch.randn_like(output.last_hidden_state))
        assert counts == [6, 2, 6]
        assert recurrent_calls == [model.backbone.layers[1]] * 2
    finally:
        for handle in handles: handle.remove()


def test_nonreentrant_path_trains_parameters_when_input_has_no_gradient():
    baseline, candidate = _models()
    x = torch.randn(1, 5, baseline.config.model_dim)
    outputs = [model(inputs_embeds=x, mode=RTMode(())) for model in (baseline, candidate)]
    cotangent = torch.randn_like(outputs[0].logits)
    for output in outputs: output.logits.backward(cotangent)
    _assert_grads(candidate, baseline)
    assert all(parameter.grad is not None for parameter in candidate.parameters())


def test_frozen_backbone_still_transmits_gradients_into_trainable_fusion():
    baseline = FBTNextLatLM(OLMoFBT(_base()), NextLatConfig(32, vocab_chunk_size=3), enabled=False)
    baseline.backbone.backbone.requires_grad_(False)
    candidate = copy.deepcopy(baseline)
    candidate.backbone.backbone.ordinary_activation_checkpointing = True
    mode = FBTMode(num_passes=2, beta=.35, rt_mode=RTMode((0,), .37))
    for model in (baseline, candidate):
        model.loss_sums(_batch(), backbone_kwargs={"mode": mode}).total.backward()
    _assert_grads(candidate, baseline)
    assert all(parameter.grad is None for parameter in candidate.backbone.backbone.parameters())
    assert all(parameter.grad is not None for parameter in candidate.backbone.fusion.parameters())


def test_state_and_tied_parameter_layout_are_unchanged():
    baseline, candidate = _models()
    assert baseline.ordinary_activation_checkpointing is False
    assert candidate.ordinary_activation_checkpointing is True
    assert baseline.config == candidate.config
    assert candidate.readout_weight is candidate.token_embeddings.weight
    assert list(dict(candidate.named_parameters())) == list(dict(baseline.named_parameters()))
    assert candidate.state_dict().keys() == baseline.state_dict().keys()
    assert not any("checkpoint" in name for name in candidate.state_dict())
    for name, tensor in candidate.state_dict().items():
        assert torch.equal(tensor, baseline.state_dict()[name])


@pytest.mark.parametrize("value", [None, 1, "ordinary", ()])
def test_nonboolean_checkpoint_option_rejected(value):
    with pytest.raises(TypeError, match="must be boolean"):
        OLMoTiledRTForCausalLM(OLMoConfig.tiny(), ordinary_activation_checkpointing=value)


@pytest.mark.parametrize("context", ["no_grad_training", "eval"])
def test_inference_and_cache_path_remain_unchanged_without_checkpoint_calls(monkeypatch, context):
    baseline, candidate = _models()
    if context == "eval":
        baseline.eval(); candidate.eval()
    def unexpected(*args, **kwargs):
        raise AssertionError("Inference must not invoke activation checkpointing")
    monkeypatch.setattr(tiled_module, "checkpoint", unexpected)
    ids = _batch().input_ids[:1, :5]
    mode = RTMode((0,), .37)
    # Evaluation with autograd enabled is also supported; no_grad training is
    # the ordinary inference idiom without changing the module's training flag.
    with torch.set_grad_enabled(context == "eval"):
        for model in (baseline, candidate):
            prefix = model(ids[:, :3], mode=mode, use_cache=True)
            suffix = model(ids[:, 3:], mode=mode, past_key_values=prefix.past_key_values, use_cache=True)
            if model is baseline:
                reference_prefix, reference_suffix = prefix, suffix
            else:
                torch.testing.assert_close(prefix.logits, reference_prefix.logits, rtol=0, atol=0)
                torch.testing.assert_close(suffix.logits, reference_suffix.logits, rtol=0, atol=0)
                for actual, expected in zip(suffix.past_key_values.key_values, reference_suffix.past_key_values.key_values):
                    for left, right in zip(actual, expected):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_checkpointed_training_rejects_cache_output_and_cache_input():
    model = _base(True)
    ids = _batch().input_ids[:1, :5]
    with pytest.raises(ValueError, match="cache-free"):
        model(ids, mode=RTMode((0,)), use_cache=True)
    with torch.no_grad():
        cached = model(ids[:, :3], mode=RTMode((0,)), use_cache=True).past_key_values
    with pytest.raises(ValueError, match="cache-free"):
        model(ids[:, 3:], mode=RTMode((0,)), past_key_values=cached)


def test_ordinary_checkpointing_reduces_forward_saved_tensor_volume():
    baseline, candidate = _models()
    totals = []
    for model in (baseline, candidate):
        saved = []
        def pack(tensor):
            saved.append(tensor.numel() * tensor.element_size())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            model(_batch().input_ids, mode=RTMode(()), return_logits=False)
        totals.append(sum(saved))
    # This is saved-tensor volume on the tiny CPU fixture, not a GPU peak or
    # physical-storage count. Its purpose is to verify that the option recomputes.
    assert totals[1] < totals[0]
