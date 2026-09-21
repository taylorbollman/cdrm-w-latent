"""Intentional CPU coverage of native OpenELM's sequential RT contract."""

import copy
from dataclasses import FrozenInstanceError, replace
import io

import pytest
import torch

from cdrm.pretrained.openelm import OpenELMConfig, OpenELMModel
from cdrm.pretrained.recurrent import (
    OpenELMRecurrentCache,
    OpenELMRecurrentModel,
    RTMode,
    recurrent_layer_reference,
)


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260922)


def fixture_model():
    return OpenELMRecurrentModel(OpenELMConfig.tiny(), attention_backend="math")


def assert_parameter_gradients(first, second, *, rtol=1e-4, atol=1e-6):
    for (name, parameter), (other_name, other_parameter) in zip(first.named_parameters(), second.named_parameters()):
        assert name == other_name
        assert parameter.grad is not None, name
        assert other_parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, other_parameter.grad, rtol=rtol, atol=atol, msg=name)


@pytest.mark.parametrize("selected_layers", [(), (0,), (1,), (0, 1)])
def test_alpha_zero_matches_ordinary_outputs_inputs_and_all_parameter_gradients(selected_layers):
    model = fixture_model()
    ordinary = OpenELMModel(model.config, attention_backend="math")
    ordinary.load_state_dict(model.state_dict(), strict=True)
    source = torch.randn(2, 6, model.config.model_dim)
    recurrent_input = source.clone().requires_grad_()
    ordinary_input = source.clone().requires_grad_()
    result = model(inputs_embeds=recurrent_input, mode=RTMode(selected_layers, 0.0))
    expected = ordinary(inputs_embeds=ordinary_input)
    torch.testing.assert_close(result.logits, expected.logits, rtol=4e-6, atol=2e-6)
    cotangent = torch.randn_like(result.logits)
    result.logits.backward(cotangent)
    expected.logits.backward(cotangent)
    torch.testing.assert_close(recurrent_input.grad, ordinary_input.grad, rtol=1e-4, atol=6e-6)
    assert_parameter_gradients(model, ordinary, rtol=2e-4, atol=2e-5)


def test_alpha_zero_really_scans_selected_block():
    model = fixture_model()
    seen_shapes = []
    handle = model.layers[0].register_forward_pre_hook(lambda module, args: seen_shapes.append(tuple(args[0].shape)))
    try:
        model(torch.tensor([[2, 3, 4, 5]]), mode=RTMode((0,), 0.0))
    finally:
        handle.remove()
    assert seen_shapes == [(1, 1, 32)] * 4


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("selected_layers", [(0,), (1,), (0, 1)])
def test_cached_chunks_match_full_outputs_and_training_gradients(alpha, selected_layers):
    full_model = fixture_model()
    cached_model = copy.deepcopy(full_model)
    mode = RTMode(selected_layers, alpha)
    tokens = torch.randint(0, 60, (2, 7))
    full = full_model(tokens, mode=mode, use_cache=True)
    cache, offset, chunks = None, 0, []
    for length in (2, 1, 4):
        output = cached_model(tokens[:, offset:offset + length], mode=mode, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        chunks.append(output.logits)
        offset += length
        assert isinstance(cache, OpenELMRecurrentCache)
        assert cache.mode == mode
        assert cache.sequence_length == offset
        for index, pair in enumerate(cache.key_values):
            expected_shape = (2, cached_model.config.num_kv_heads[index], offset, 8)
            assert pair[0].shape == pair[1].shape == expected_shape
    torch.testing.assert_close(torch.cat(chunks, dim=1), full.logits, rtol=4e-6, atol=2e-6)
    for expected_pair, actual_pair in zip(full.past_key_values.key_values, cache.key_values):
        for expected, actual in zip(expected_pair, actual_pair):
            torch.testing.assert_close(actual, expected, rtol=4e-6, atol=2e-6)
    # Only suffix supervision: equality requires attached prefix memory writes.
    full.logits[:, 3:].square().mean().backward()
    chunks[-1].square().mean().backward()
    assert_parameter_gradients(full_model, cached_model)


@pytest.mark.parametrize("selected_layers", [(0,), (0, 1)])
def test_future_changes_cannot_affect_prefix_outputs_or_gradients(selected_layers):
    model = fixture_model()
    mode = RTMode(selected_layers, 0.61)
    embeddings = torch.randn(2, 7, model.config.model_dim, requires_grad=True)
    changed = embeddings.detach().clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 4
    first = model(inputs_embeds=embeddings, mode=mode)
    second = model(inputs_embeds=changed, mode=mode)
    torch.testing.assert_close(first.logits[:, :4], second.logits[:, :4], rtol=0, atol=0)
    first.logits[:, :4].square().mean().backward()
    assert torch.count_nonzero(embeddings.grad[:, 4:]) == 0
    assert embeddings.grad[:, :4].abs().sum() > 0


@pytest.mark.parametrize("left_padding", [False, True])
def test_padding_and_explicit_positions_survive_cache_continuation(left_padding):
    model = fixture_model()
    mode = RTMode((0, 1), 0.7)
    tokens = torch.tensor([[3, 6, 9, 2], [5, 8, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    positions = torch.tensor([[11, 12, 13, 14], [71, 72, 0, 0]])
    if left_padding:
        tokens[1] = torch.tensor([0, 0, 5, 8])
        valid[1] = torch.tensor([0, 0, 1, 1])
        positions[1] = torch.tensor([0, 0, 71, 72])
    prefixed = model(tokens, mode=mode, attention_mask=valid, position_ids=positions, use_cache=True)
    short = model(torch.tensor([[5, 8]]), mode=mode, position_ids=torch.tensor([[71, 72]]), use_cache=True)
    torch.testing.assert_close(prefixed.logits[1, valid[1]], short.logits[0], rtol=4e-6, atol=2e-6)
    continuation = torch.tensor([[11, 12], [11, 12]])
    extended = model(continuation, mode=mode, past_key_values=prefixed.past_key_values, use_cache=True)
    short_extended = model(continuation[1:], mode=mode, past_key_values=short.past_key_values, use_cache=True)
    torch.testing.assert_close(extended.logits[1], short_extended.logits[0], rtol=4e-6, atol=2e-6)
    assert extended.past_key_values.position_ids[1, -2:].tolist() == [73, 74]
    assert torch.equal(extended.past_key_values.attention_mask[:, :4], valid)


def test_fully_masked_rows_have_finite_outputs_and_backward():
    model = fixture_model()
    result = model(
        torch.tensor([[3, 4, 5], [0, 0, 0]]), mode=RTMode((0, 1), 1.0),
        attention_mask=torch.tensor([[1, 1, 1], [0, 0, 0]]),
    )
    assert result.logits.isfinite().all()
    result.logits.square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in model.parameters())


@pytest.mark.parametrize("alpha", [0.0, 0.35, 1.0])
def test_persistent_cache_is_native_unrotated_memory_from_pre_finalnorm_output(alpha):
    model = fixture_model()
    tokens = torch.tensor([[2, 4, 7]])
    block_outputs = []
    hook = model.layers[0].register_forward_hook(lambda module, args, output: block_outputs.append(output[0]))
    try:
        result = model(tokens, mode=RTMode((0,), alpha), position_ids=torch.tensor([[91, 92, 93]]), use_cache=True)
    finally:
        hook.remove()
    x = model.token_embeddings(tokens)
    z = torch.cat(block_outputs, dim=1)
    attention = model.layers[0].attn
    projected = attention.qkv_proj(model.layers[0].attn_norm((1 - alpha) * x + alpha * z))
    projected = projected.reshape(1, 3, 4, 8).transpose(1, 2)
    _, key, value = projected.split((2, 1, 1), dim=1)
    torch.testing.assert_close(result.past_key_values.key_values[0][0], attention.k_norm(key), rtol=3e-6, atol=1e-6)
    torch.testing.assert_close(result.past_key_values.key_values[0][1], value, rtol=3e-6, atol=1e-6)
    assert result.last_hidden_state.shape == z.shape


def test_later_output_gradient_reaches_earlier_completed_block_output_only_through_memory():
    for alpha in (0.0, 0.35, 1.0):
        model = fixture_model()
        layer = model.layers[0]
        source = torch.randn(1, 4, 32, requires_grad=True)
        produced = []

        def keep_output(module, args, result):
            result[0].retain_grad()
            produced.append(result[0])

        hook = layer.register_forward_hook(keep_output)
        try:
            output, _ = recurrent_layer_reference(
                layer, source, alpha=alpha, past=None,
                query_positions=torch.arange(4).view(1, -1),
                key_positions=torch.arange(4).view(1, -1),
                key_valid=torch.ones(1, 4, dtype=torch.bool), attention_backend="math",
            )
            output[:, -1].square().sum().backward()
        finally:
            hook.remove()
        assert produced[0].grad is not None
        if alpha == 0:
            assert torch.count_nonzero(produced[0].grad) == 0
        else:
            assert produced[0].grad.abs().sum() > 0
        assert source.grad[:, 0].abs().sum() > 0


def test_first_token_is_ordinary_and_alpha_only_changes_its_persistent_write():
    model = fixture_model()
    token = torch.tensor([[12]])
    ordinary = model(token, mode=RTMode((), 1.0), use_cache=True)
    zero = model(token, mode=RTMode((0, 1), 0.0), use_cache=True)
    recurrent = model(token, mode=RTMode((0, 1), 1.0), use_cache=True)
    torch.testing.assert_close(recurrent.logits, ordinary.logits, rtol=0, atol=0)
    torch.testing.assert_close(zero.logits, ordinary.logits, rtol=0, atol=0)
    assert not torch.equal(recurrent.past_key_values.key_values[0][1], ordinary.past_key_values.key_values[0][1])


def test_two_forward_modes_accumulate_shared_parameter_gradients_without_mutation():
    combined = fixture_model()
    first = copy.deepcopy(combined)
    second = copy.deepcopy(combined)
    tokens = torch.randint(0, 60, (2, 5))
    low, high = RTMode((0,), 0.25), RTMode((0, 1), 0.9)
    low_output = combined(tokens, mode=low)
    high_output = combined(tokens, mode=high)
    (low_output.logits.square().mean() + high_output.logits.sin().mean()).backward()
    first(tokens, mode=low).logits.square().mean().backward()
    second(tokens, mode=high).logits.sin().mean().backward()
    for (name, shared), (_, part_a), (_, part_b) in zip(combined.named_parameters(), first.named_parameters(), second.named_parameters()):
        torch.testing.assert_close(shared.grad, part_a.grad + part_b.grad, rtol=8e-5, atol=5e-7, msg=name)


def test_checkpoint_layout_and_tied_ownership_are_unchanged():
    model = fixture_model()
    ordinary = OpenELMModel(model.config)
    assert list(model.state_dict()) == list(ordinary.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in ordinary.parameters())
    ordinary.load_state_dict(model.state_dict(), strict=True)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = fixture_model()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert restored.readout_weight is restored.token_embeddings.weight
    parameters = list(restored.parameters())
    assert sum(parameter is restored.readout_weight for parameter in parameters) == 1
    optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    assert sum(parameter is restored.readout_weight for group in optimizer.param_groups for parameter in group["params"]) == 1
    tokens = torch.tensor([[1, 2, 3, 4]])
    expected = model(tokens, mode=RTMode((0,), 0.6))
    actual = restored(tokens, mode=RTMode((0,), 0.6))
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)


def test_hidden_only_mode_skips_tied_projection():
    model = fixture_model()
    output = model(torch.tensor([[1, 2]]), return_logits=False)
    assert output.logits is None
    assert output.last_hidden_state.shape == (1, 2, 32)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        RTMode(alpha=alpha)


@pytest.mark.parametrize("layers", [(0, 0), (-1,), (True,), (0.0,)])
def test_invalid_layer_selection_is_rejected(layers):
    with pytest.raises(ValueError, match="selected_layers"):
        RTMode(layers)


def test_mode_is_frozen_and_canonical():
    assert RTMode((1, 0), 0.5) == RTMode((0, 1), 0.5)
    with pytest.raises(FrozenInstanceError):
        RTMode().alpha = 0.0
    with pytest.raises(ValueError, match="outside"):
        fixture_model()(torch.tensor([[2]]), mode=RTMode((2,), 0.0))


def test_cache_rejects_changed_mode_model_or_weights():
    model = fixture_model()
    mode = RTMode((0,), 0.4)
    token = torch.tensor([[2]])
    cache = model(token, mode=mode, use_cache=True).past_key_values
    for other_mode in (RTMode((0,), 0.5), RTMode((1,), 0.4), RTMode((), 0.4)):
        with pytest.raises(ValueError, match="mode"):
            model(token, mode=other_mode, past_key_values=cache)
    other_model = copy.deepcopy(model)
    with pytest.raises(ValueError, match="weights"):
        other_model(token, mode=mode, past_key_values=cache)
    with torch.no_grad():
        model.norm.weight.add_(0.1)
    with pytest.raises(ValueError, match="weights"):
        model(token, mode=mode, past_key_values=cache)


def test_plain_cache_and_changed_cached_padding_are_rejected():
    model = fixture_model()
    tokens = torch.tensor([[2, 3]])
    ordinary = OpenELMModel(model.config)
    ordinary_cache = ordinary(tokens, use_cache=True).past_key_values
    with pytest.raises(TypeError, match="RecurrentCache"):
        model(tokens[:, :1], past_key_values=ordinary_cache)
    cache = model(tokens, attention_mask=torch.tensor([[0, 1]]), use_cache=True).past_key_values
    with pytest.raises(ValueError, match="cannot be changed"):
        model(tokens[:, :1], past_key_values=cache, attention_mask=torch.tensor([[1, 1, 1]]))
    with pytest.raises(ValueError, match="native KV shape"):
        model(tokens[:, :1], past_key_values=replace(cache, key_values=((cache.key_values[0][0][:, :, :1], cache.key_values[0][1]),) + cache.key_values[1:]))


def test_ordinary_model_rejects_recurrent_cache_even_with_matching_geometry_and_weights():
    model = fixture_model()
    ordinary = OpenELMModel(model.config, attention_backend="math")
    ordinary.load_state_dict(model.state_dict(), strict=True)
    tokens = torch.tensor([[2, 3]])
    cache = model(tokens, use_cache=True).past_key_values
    with pytest.raises(TypeError, match="OpenELMCache"):
        ordinary(tokens[:, :1], past_key_values=cache)


def test_dtype_roundtrip_invalidates_cache_even_if_parameter_identity_and_versions_survive():
    model = fixture_model()
    tokens = torch.tensor([[2, 3]])
    cache = model(tokens, use_cache=True).past_key_values
    original_weight = model.token_embeddings.weight.detach().clone()
    model.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    assert not torch.equal(model.token_embeddings.weight, original_weight)
    assert model._rt_cache_generation == cache.model_generation + 2
    with pytest.raises(ValueError, match="generation|weights"):
        model(tokens[:, :1], past_key_values=cache)
    # The invalidation metadata is not a checkpoint tensor or extra parameter.
    assert not any("generation" in key for key in model.state_dict())
    assert model.readout_weight is model.token_embeddings.weight
    fresh = model(tokens, use_cache=True)
    model(tokens[:, :1], past_key_values=fresh.past_key_values)


def test_cache_rejects_autocast_enable_and_dtype_changes():
    model = fixture_model()
    tokens = torch.tensor([[2, 3]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cache = model(tokens, use_cache=True).past_key_values
        model(tokens[:, :1], past_key_values=cache)
    with pytest.raises(ValueError, match="execution context"):
        model(tokens[:, :1], past_key_values=cache)
    with torch.autocast("cpu", dtype=torch.float16):
        with pytest.raises(ValueError, match="execution context"):
            model(tokens[:, :1], past_key_values=cache)
    fp32_cache = model(tokens, use_cache=True).past_key_values
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="execution context"):
            model(tokens[:, :1], past_key_values=fp32_cache)


def test_cache_rejects_detached_history_training_and_inference_mode_changes():
    model = fixture_model()
    tokens = torch.tensor([[2, 3]])
    with torch.no_grad():
        cache = model(tokens, use_cache=True).past_key_values
        model(tokens[:, :1], past_key_values=cache)
    with pytest.raises(ValueError, match="execution context"):
        model(tokens[:, :1], past_key_values=cache)
    # Both contexts disable gradients, so the separate inference-mode field
    # prevents treating their distinct tensor/autograd semantics as equivalent.
    with torch.inference_mode():
        with pytest.raises(ValueError, match="execution context"):
            model(tokens[:, :1], past_key_values=cache)
        inference_cache = model(tokens, use_cache=True).past_key_values
        model(tokens[:, :1], past_key_values=inference_cache)
    with torch.no_grad():
        with pytest.raises(ValueError, match="execution context"):
            model(tokens[:, :1], past_key_values=inference_cache)
    attached_cache = model(tokens, use_cache=True).past_key_values
    with torch.no_grad():
        with pytest.raises(ValueError, match="execution context"):
            model(tokens[:, :1], past_key_values=attached_cache)


def test_cache_rejects_configured_attention_backend_change():
    model = fixture_model()
    tokens = torch.tensor([[2, 3]])
    cache = model(tokens, use_cache=True).past_key_values
    model.attention_backend = "sdpa"
    with pytest.raises(ValueError, match="execution context"):
        model(tokens[:, :1], past_key_values=cache)


def test_cache_owns_validity_and_position_metadata_without_aliasing_caller_inputs():
    model = fixture_model()
    tokens = torch.tensor([[2, 3]])
    validity = torch.tensor([[True, True]])
    positions = torch.tensor([[71, 72]])
    with torch.no_grad():
        cache = model(tokens, attention_mask=validity, position_ids=positions, use_cache=True).past_key_values
        validity.zero_()
        positions.zero_()
        assert cache.attention_mask.tolist() == [[True, True]]
        assert cache.position_ids.tolist() == [[71, 72]]
        continued = model(tokens[:, :1], past_key_values=cache, use_cache=True)
        assert continued.past_key_values.position_ids.tolist() == [[71, 72, 73]]
