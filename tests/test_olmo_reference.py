"""Ordinary adapter parity with independently executed original OLMo source."""

import copy
from pathlib import Path
import shutil

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_reference import (
    build_olmo_reference, native_olmo_reference_config,
    olmo_reference_forward, verify_olmo_reference_sources,
)


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(2401)


def _models():
    config = OLMoConfig.tiny()
    native = build_olmo_reference(config)
    model = OLMoForCausalLM(config, attention_backend="math")
    model.load_state_dict(native.state_dict(), strict=True)
    return native, model


def _compare_gradients(actual, expected, *, atol=4e-6, rtol=5e-5):
    actual_parameters = dict(actual.named_parameters())
    for name, parameter in expected.named_parameters():
        assert parameter.grad is not None, name
        assert actual_parameters[name].grad is not None, name
        torch.testing.assert_close(actual_parameters[name].grad, parameter.grad, atol=atol, rtol=rtol, msg=name)


def test_pristine_source_manifest_and_isolated_namespace(tmp_path):
    manifest = verify_olmo_reference_sources()
    assert manifest["revision"] == "b3741bc21f1dd504838b7dbd9878ee077ded63bd"
    source = Path(__file__).parents[1] / "cdrm/pretrained/_olmo_reference"
    copied = tmp_path / "source"
    shutil.copytree(source, copied)
    with (copied / "model.py").open("a") as handle:
        handle.write("\n# changed\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_olmo_reference_sources(copied)
    model = build_olmo_reference(OLMoConfig.tiny())
    assert type(model).__module__ == "cdrm.pretrained._loaded_olmo_reference"
    assert type(model.transformer.blocks[0]).__module__ == type(model).__module__


def test_native_config_and_meta_geometry_recover_checkpoint_contract():
    config = native_olmo_reference_config()
    assert (config.d_model, config.n_layers, config.n_heads) == (2048, 16, 16)
    assert (config.embedding_size, config.vocab_size) == (50304, 50280)
    assert config.mlp_ratio == 8 and config.mlp_hidden_size is None
    assert not config.layer_norm_with_affine and not config.attention_layer_norm
    assert not config.include_bias and config.rope and config.rope_full_precision
    assert config.weight_tying and not config.scale_logits
    model = build_olmo_reference(device="meta")
    assert len(model.state_dict()) == 65
    assert sum(p.numel() for p in model.parameters()) == 1176764416
    assert model.transformer.wte.padding_idx is None
    assert model.transformer.blocks[0].ff_proj.weight.shape == (16384, 2048)
    assert model.transformer.blocks[0].ff_out.weight.shape == (2048, 8192)


def test_native_constructor_does_not_leave_backend_flags_changed():
    previous = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        build_olmo_reference(OLMoConfig.tiny())
        assert torch.backends.cuda.flash_sdp_enabled()
        assert torch.backends.cuda.mem_efficient_sdp_enabled()
    finally:
        torch.backends.cuda.enable_flash_sdp(previous[0])
        torch.backends.cuda.enable_mem_efficient_sdp(previous[1])


@pytest.mark.parametrize("embedded", [False, True])
def test_ordinary_outputs_loss_all_parameters_and_input_gradients_match_native(embedded):
    native, model = _models()
    ids = torch.tensor([[1, 2, 3, 5, 8, 13], [4, 7, 10, 16, 1, 2]])
    targets = ids.flip(-1)
    actual_x = torch.randn(2, 6, model.config.model_dim, requires_grad=True)
    expected_x = actual_x.detach().clone().requires_grad_()
    with sdpa_kernel(SDPBackend.MATH):
        if embedded:
            actual = model(inputs_embeds=actual_x)
            expected = olmo_reference_forward(native, inputs_embeds=expected_x)
        else:
            actual = model(ids)
            expected = olmo_reference_forward(native, ids)
        actual_loss = F.cross_entropy(actual.logits.flatten(0, 1), targets.flatten())
        expected_loss = F.cross_entropy(expected.logits.flatten(0, 1), targets.flatten())
        actual_loss.backward()
        expected_loss.backward()
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=3e-6)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=2e-6, rtol=3e-6)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-6, rtol=3e-6)
    _compare_gradients(model, native)
    if embedded:
        torch.testing.assert_close(actual_x.grad, expected_x.grad, atol=3e-6, rtol=4e-5)


def test_native_bf16_autocast_outputs_and_all_gradients_match_adapter():
    native, model = _models()
    ids = torch.tensor([[1, 2, 3, 5, 8, 13]])
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(ids)
        expected = olmo_reference_forward(native, ids)
        actual_loss = F.cross_entropy(actual.logits.flatten(0, 1), ids.flip(-1).flatten())
        expected_loss = F.cross_entropy(expected.logits.flatten(0, 1), ids.flip(-1).flatten())
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    _compare_gradients(model, native, atol=0, rtol=0)


def test_native_cached_chunks_and_attached_prefix_gradients_match_full_execution():
    native, model = _models()
    full_native = copy.deepcopy(native)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with sdpa_kernel(SDPBackend.MATH):
        full = olmo_reference_forward(full_native, ids)
        native_prefix = olmo_reference_forward(native, ids[:, :2], use_cache=True)
        native_suffix = olmo_reference_forward(native, ids[:, 2:], past_key_values=native_prefix.past_key_values, use_cache=True)
        prefix = model(ids[:, :2], use_cache=True)
        suffix = model(ids[:, 2:], past_key_values=prefix.past_key_values, use_cache=True)
        full.logits[:, 2:].square().mean().backward()
        native_suffix.logits.square().mean().backward()
        suffix.logits.square().mean().backward()
    torch.testing.assert_close(native_suffix.logits, full.logits[:, 2:], atol=2e-6, rtol=3e-6)
    torch.testing.assert_close(suffix.logits, native_suffix.logits, atol=2e-6, rtol=3e-6)
    _compare_gradients(native, full_native)
    _compare_gradients(model, native)
    for native_pair, actual_pair in zip(native_suffix.past_key_values, suffix.past_key_values.key_values):
        for expected, actual in zip(native_pair, actual_pair):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-6)


def test_native_configured_pad_token_keeps_its_embedding_lookup_gradient():
    native, _ = _models()
    native.transformer.wte(torch.tensor([[native.config.pad_token_id]])).sum().backward()
    assert torch.equal(native.transformer.wte.weight.grad[native.config.pad_token_id], torch.ones(native.config.d_model))


def test_native_swiglu_uses_value_then_gate():
    native, _ = _models()
    x = torch.tensor([[[-3.0, 2.0, 0.4, 1.3]]])
    actual = native.transformer.blocks[0].act(x)
    up, gate = x.chunk(2, dim=-1)
    torch.testing.assert_close(actual, F.silu(gate) * up, atol=0, rtol=0)
    assert not torch.equal(actual, F.silu(up) * gate)
