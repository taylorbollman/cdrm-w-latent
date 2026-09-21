"""Independent upstream fidelity checks for the ordinary OpenELM adapter."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.reference import (
    build_corenet_reference,
    native_reference_config,
    reference_forward,
    verify_reference_sources,
)


def _small_config():
    return SimpleNamespace(
        vocab_size=67,
        padding_idx=64,
        model_dim=32,
        head_dim=8,
        num_query_heads=(2, 4),
        num_kv_heads=(1, 2),
        ffn_intermediate_sizes=(24, 48),
        rms_norm_eps=1e-6,
        rope_freq_constant=10000,
        max_context_length=16,
    )


def test_reference_snapshot_has_verified_source_provenance(tmp_path):
    manifest = verify_reference_sources()
    assert manifest["revision"] == "f9f83e616a34d02c422733a06a3fe5bde63ae575"
    source = Path(__file__).parents[1] / "cdrm/pretrained/_corenet_reference"
    copied = tmp_path / "source"
    shutil.copytree(source, copied)
    with (copied / "general_gpt.py").open("a") as handle:
        handle.write("\n# changed source\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_reference_sources(copied)


def test_native_config_independently_recovers_variable_1b_geometry():
    config = native_reference_config()
    assert (config.vocab_size, config.model_dim, config.head_dim) == (32128, 2048, 64)
    assert config.num_query_heads == [16] * 3 + [20] * 7 + [24] * 8 + [28] * 6 + [32] * 4
    assert config.num_kv_heads == [q // 4 for q in config.num_query_heads]
    assert config.ffn_multipliers[0] == 0.5
    assert config.ffn_multipliers[-1] == 4.0
    assert config.share_input_output_layers


def test_reference_cached_chunks_match_full_native_forward():
    torch.manual_seed(91)
    model = build_corenet_reference(_small_config())
    tokens = torch.randint(0, 64, (2, 7))
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        full = reference_forward(model, tokens)
        prefix = reference_forward(model, tokens[:, :3], use_cache=True)
        suffix = reference_forward(
            model, tokens[:, 3:], past_key_values=prefix.past_key_values, use_cache=True
        )
    torch.testing.assert_close(full.logits[:, 3:], suffix.logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        full.last_hidden_state[:, 3:], suffix.last_hidden_state, atol=2e-6, rtol=2e-6
    )
    assert [tuple(k.shape) for k, _ in suffix.past_key_values] == [(2, 1, 7, 8), (2, 2, 7, 8)]
    assert model.classifier is None
    assert "token_embeddings.weight" in model.state_dict()
    assert "classifier.weight" not in model.state_dict()


def test_adapter_logits_loss_gradients_and_cache_match_native_reference():
    from cdrm.pretrained.openelm import OpenELMConfig, OpenELMModel

    torch.manual_seed(92)
    config = OpenELMConfig(**vars(_small_config()))
    reference = build_corenet_reference(config)
    model = OpenELMModel(config, attention_backend="sdpa")
    model.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, config.vocab_size, (2, 7))
    with sdpa_kernel(SDPBackend.MATH):
        expected = reference_forward(reference, tokens)
        actual = model(tokens)
        expected_loss = F.cross_entropy(expected.logits.flatten(0, 1), targets.flatten())
        actual_loss = F.cross_entropy(actual.logits.flatten(0, 1), targets.flatten())
        expected_loss.backward()
        actual_loss.backward()
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-6, rtol=2e-6)
    actual_parameters = dict(model.named_parameters())
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            actual_parameters[name].grad, parameter.grad, atol=3e-6, rtol=3e-5, msg=name
        )
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        expected_prefix = reference_forward(reference, tokens[:, :3], use_cache=True)
        actual_prefix = model(tokens[:, :3], use_cache=True)
        expected_suffix = reference_forward(
            reference, tokens[:, 3:], past_key_values=expected_prefix.past_key_values, use_cache=True
        )
        actual_suffix = model(
            tokens[:, 3:], past_key_values=actual_prefix.past_key_values, use_cache=True
        )
    torch.testing.assert_close(actual_suffix.logits, expected_suffix.logits, atol=2e-6, rtol=2e-6)
    for actual_kv, expected_kv in zip(actual_suffix.past_key_values.key_values, expected_suffix.past_key_values):
        for actual_tensor, expected_tensor in zip(actual_kv, expected_kv):
            torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-6, rtol=2e-6)
