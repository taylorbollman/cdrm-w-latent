"""Intentional CPU tests of the new ordinary OpenELM mathematical contract."""

import copy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.openelm import OpenELMCache, OpenELMConfig, OpenELMModel, OpenELMRMSNorm


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260921)


def fixture_model(backend="math"):
    return OpenELMModel(OpenELMConfig.tiny(), attention_backend=backend)


def test_bf16_rmsnorm_input_gradients_match_native_near_radial_cancellation():
    """Forward equivalence alone misses separately rounded normalization adjoints.

    A nearly radial incoming cotangent makes the direct and variance-derived
    input adjoints largely cancel. Native CoreNet combines them at its shared
    FP32 cast node before rounding the final input gradient back to BF16.
    """
    from cdrm.pretrained.reference import build_corenet_reference

    torch.manual_seed(20260922)
    native = build_corenet_reference(OpenELMConfig.tiny()).layers[0].attn_norm
    adapter = OpenELMRMSNorm(32, 1e-6)
    with torch.no_grad():
        native.weight.copy_(0.5 + torch.rand(32))
        adapter.weight.copy_(native.weight)
    source = torch.randn(3, 5, 32).to(torch.bfloat16)
    expected_input = source.clone().requires_grad_(True)
    actual_input = source.clone().requires_grad_(True)
    # Account for the learned gain so the input-side cotangent is close to x.
    cotangent = source.float() / native.weight.detach() + 0.01 * torch.randn(3, 5, 32)
    expected = native(expected_input)
    actual = adapter(actual_input)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.backward(cotangent)
    actual.backward(cotangent)
    assert expected_input.grad.abs().sum() > 0
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
    torch.testing.assert_close(adapter.weight.grad, native.weight.grad, rtol=0, atol=0)


def test_native_geometry_and_full_vocabulary_have_one_owned_readout():
    config = OpenELMConfig.native_1_1b()
    model = OpenELMModel(config, device="meta")
    assert config.num_transformer_layers == 28
    assert sum(p.numel() for p in model.parameters()) == 1_080_153_600
    assert model.token_embeddings.weight.shape == (32128, 2048)
    assert model.token_embeddings.padding_idx == 32000
    assert model.readout_weight is model.token_embeddings.weight
    assert model.layers[0].attn.qkv_proj.weight.shape == (1536, 2048)
    assert model.layers[0].attn.out_proj.weight.shape == (2048, 1024)
    assert model.layers[-1].attn.qkv_proj.weight.shape == (3072, 2048)
    assert model.layers[-1].attn.out_proj.weight.shape == (2048, 2048)
    assert model.layers[13].ffn.proj_1.weight.shape == (9216, 2048)
    assert all(layer.attn.q_norm.weight.shape == (64,) for layer in model.layers)
    assert len(model.state_dict()) == 2 + 8 * 28
    assert not any("classifier" in key or "rope" in key for key in model.state_dict())


def test_causal_prefix_outputs_and_gradients_ignore_future_inputs():
    model = fixture_model()
    embeddings = torch.randn(2, 7, model.config.model_dim, requires_grad=True)
    changed = embeddings.detach().clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 3
    first = model(inputs_embeds=embeddings)
    second = model(inputs_embeds=changed)
    torch.testing.assert_close(first.logits[:, :4], second.logits[:, :4], rtol=0, atol=0)
    first.logits[:, :4].square().mean().backward()
    assert torch.count_nonzero(embeddings.grad[:, 4:]) == 0
    assert embeddings.grad[:, :4].abs().sum() > 0


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1, 1), (3, 2, 2), (1, 5, 1)])
def test_cached_chunks_match_full_forward_and_keep_unrotated_native_heads(chunks):
    model = fixture_model()
    tokens = torch.randint(0, 60, (2, 7))
    full = model(tokens)
    cache, offset, outputs = None, 0, []
    for length in chunks:
        result = model(tokens[:, offset:offset + length], past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        outputs.append(result.logits)
        offset += length
        assert cache.sequence_length == offset
        for i, (key, value) in enumerate(cache.key_values):
            assert key.shape == value.shape == (2, model.config.num_kv_heads[i], offset, 8)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full.logits, rtol=3e-6, atol=1e-6)
    first_attention = model.layers[0].attn
    x = model.layers[0].attn_norm(model.token_embeddings(tokens))
    projected = first_attention.qkv_proj(x).reshape(2, 7, 4, 8).transpose(1, 2)
    _, raw_key, raw_value = projected.split((2, 1, 1), dim=1)
    torch.testing.assert_close(cache.key_values[0][0], first_attention.k_norm(raw_key), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cache.key_values[0][1], raw_value, rtol=1e-6, atol=1e-6)


def test_cached_training_preserves_gradients_through_earlier_tokens():
    full_model = fixture_model()
    cached_model = copy.deepcopy(full_model)
    tokens = torch.randint(0, 60, (2, 6))
    full_model(tokens).logits[:, 3:].square().mean().backward()
    prefix = cached_model(tokens[:, :3], use_cache=True)
    suffix = cached_model(tokens[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
    suffix.logits.square().mean().backward()
    for (name, parameter), (other_name, other_parameter) in zip(full_model.named_parameters(), cached_model.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other_parameter.grad, rtol=8e-5, atol=3e-7, msg=name)


@pytest.mark.parametrize("left_padding", [False, True])
def test_padding_matches_unpadded_tokens_and_survives_cached_continuation(left_padding):
    model = fixture_model()
    tokens = torch.tensor([[3, 6, 9, 2], [5, 8, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    if left_padding:
        tokens[1] = torch.tensor([0, 0, 5, 8])
        valid[1] = torch.tensor([0, 0, 1, 1])
    prefixed = model(tokens, attention_mask=valid, use_cache=True)
    short = model(torch.tensor([[5, 8]]), use_cache=True)
    torch.testing.assert_close(prefixed.logits[1, valid[1]], short.logits[0], rtol=3e-6, atol=1e-6)
    continuation = torch.tensor([[11, 12], [11, 12]])
    extended = model(continuation, past_key_values=prefixed.past_key_values, use_cache=True)
    short_extended = model(continuation[1:], past_key_values=short.past_key_values, use_cache=True)
    torch.testing.assert_close(extended.logits[1], short_extended.logits[0], rtol=3e-6, atol=1e-6)
    assert extended.past_key_values.position_ids[1, -2:].tolist() == [2, 3]
    assert torch.equal(extended.past_key_values.attention_mask[:, :4], valid)


def test_entirely_masked_rows_remain_finite_with_finite_backward():
    model = fixture_model()
    tokens = torch.tensor([[3, 4, 5], [0, 0, 0]])
    result = model(tokens, attention_mask=torch.tensor([[1, 1, 1], [0, 0, 0]]))
    assert result.logits.isfinite().all()
    result.logits.square().mean().backward()
    assert all(parameter.grad.isfinite().all() for parameter in model.parameters())


def test_explicit_absolute_positions_continue_without_double_rotation():
    model = fixture_model()
    tokens = torch.tensor([[2, 4, 6, 8, 9]])
    positions = torch.tensor([[100, 101, 102, 103, 104]])
    full = model(tokens, position_ids=positions)
    prefix = model(tokens[:, :3], position_ids=positions[:, :3], use_cache=True)
    suffix = model(tokens[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(suffix.logits, full.logits[:, 3:], rtol=3e-6, atol=1e-6)
    assert suffix.past_key_values.position_ids.tolist() == positions.tolist()


def test_sdpa_and_math_match_outputs_and_parameter_gradients():
    math_model = fixture_model()
    sdpa_model = copy.deepcopy(math_model)
    sdpa_model.attention_backend = "sdpa"
    tokens = torch.randint(0, 60, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    math_output, sdpa_output = math_model(tokens, attention_mask=mask), sdpa_model(tokens, attention_mask=mask)
    torch.testing.assert_close(math_output.logits, sdpa_output.logits, rtol=4e-6, atol=1e-6)
    math_output.logits.square().mean().backward()
    sdpa_output.logits.square().mean().backward()
    for (name, parameter), (_, other) in zip(math_model.named_parameters(), sdpa_model.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=3e-7, msg=name)


def test_tied_padding_row_has_no_lookup_gradient_but_does_have_readout_gradient():
    model = fixture_model()
    tokens = torch.tensor([[model.config.padding_idx, 3, 4]])
    # Native padding is a lookup-gradient rule, not an implicit attention mask.
    model(tokens, return_logits=False).last_hidden_state.square().sum().backward()
    assert torch.count_nonzero(model.readout_weight.grad[model.config.padding_idx]) == 0
    model.zero_grad(set_to_none=True)
    logits = model(tokens).logits
    F.cross_entropy(logits[:, -1], torch.tensor([5])).backward()
    assert model.readout_weight.grad[model.config.padding_idx].abs().sum() > 0
    assert logits.shape[-1] == model.config.vocab_size


def test_inputs_embeds_and_strict_reload_keep_tying_and_native_keys():
    model = fixture_model()
    tokens = torch.randint(0, 60, (1, 4))
    expected = model(tokens)
    actual = model(inputs_embeds=model.token_embeddings(tokens))
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    restored = OpenELMModel(OpenELMConfig.from_dict(model.config.to_dict()), device="meta", attention_backend="math")
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True, assign=True)
    assert restored.readout_weight is restored.token_embeddings.weight
    torch.testing.assert_close(restored(tokens).logits, expected.logits, rtol=0, atol=0)
    assert len({id(p) for p in restored.parameters()}) == len(list(restored.parameters()))
    malformed = dict(model.state_dict())
    malformed["classifier.weight"] = model.readout_weight.detach().clone()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(malformed, strict=True)


def test_serialized_adam_resume_preserves_tying_and_matches_uninterrupted_update(tmp_path):
    model = fixture_model()

    def optimizer_for(backbone):
        optimizer = torch.optim.AdamW(
            backbone.parameters(), lr=3e-4, betas=(0.9, 0.95),
            weight_decay=0.1, foreach=False, fused=False,
        )
        owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        assert sum(parameter is backbone.readout_weight for parameter in owned) == 1
        assert len(owned) == len({id(parameter) for parameter in owned})
        return optimizer

    def update(backbone, optimizer, tokens):
        optimizer.zero_grad(set_to_none=True)
        logits = backbone(tokens).logits
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
        loss.backward()
        assert backbone.readout_weight.grad is not None
        assert all(parameter.grad.isfinite().all() for parameter in backbone.parameters())
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        optimizer.step()
        return loss.detach()

    optimizer = optimizer_for(model)
    first_tokens = torch.tensor([[2, 7, 4, 9], [3, 2, 11, 5]])
    update(model, optimizer, first_tokens)
    checkpoint = tmp_path / "ordinary-openelm-step1.pt"
    torch.save({
        "config": model.config.to_dict(), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = OpenELMModel(OpenELMConfig.from_dict(saved["config"]), device="meta", attention_backend="math")
    restored.load_state_dict(saved["model"], strict=True, assign=True)
    assert restored.readout_weight is restored.token_embeddings.weight
    # Construct the optimizer after assign=True, so it owns the loaded Parameters.
    restored_optimizer = optimizer_for(restored)
    restored_optimizer.load_state_dict(saved["optimizer"])
    before = restored.readout_weight.detach().clone()
    second_tokens = torch.tensor([[6, 13, 5, 1], [9, 17, 2, 8]])
    expected_loss = update(model, optimizer, second_tokens)
    actual_loss = update(restored, restored_optimizer, second_tokens)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert torch.count_nonzero(restored.readout_weight.detach() - before) > 0
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), restored.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[parameter][key], restored_optimizer.state[other][key], rtol=0, atol=0)


@pytest.mark.parametrize("changes", [
    {"head_dim": 7}, {"padding_idx": 67}, {"model_dim": 0},
    {"num_kv_heads": (3, 2)}, {"num_query_heads": (2,)},
    {"rms_norm_eps": float("nan")}, {"rope_freq_constant": 0},
])
def test_invalid_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(OpenELMConfig.tiny(), **changes)


def test_invalid_forward_and_cache_inputs_are_rejected():
    model = fixture_model()
    tokens = torch.tensor([[2, 3, 4]])
    with pytest.raises(ValueError, match="exactly one"):
        model(tokens, inputs_embeds=model.token_embeddings(tokens))
    with pytest.raises(ValueError, match="0/1"):
        model(tokens, attention_mask=torch.tensor([[1, 2, 1]]))
    with pytest.raises(ValueError, match="nonnegative"):
        model(tokens, position_ids=torch.tensor([[0, -1, 2]]))
    cache = model(tokens, use_cache=True).past_key_values
    with pytest.raises(ValueError, match="full cached"):
        model(tokens, past_key_values=cache, attention_mask=torch.ones_like(tokens))
    with pytest.raises(ValueError, match="cannot be changed"):
        model(tokens, past_key_values=cache, attention_mask=torch.tensor([[1, 0, 1, 1, 1, 1]]))
    broken = OpenELMCache(cache.key_values[:-1], cache.attention_mask, cache.position_ids)
    with pytest.raises(ValueError, match="layer count"):
        model(tokens, past_key_values=broken)
    with pytest.raises(TypeError, match="OpenELMCache"):
        model(tokens, past_key_values=cache.key_values)
