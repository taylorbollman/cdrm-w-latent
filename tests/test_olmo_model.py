"""Intentional CPU checks of original OLMo's checkpoint and causal contract."""

import copy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained.olmo import OLMoCache, OLMoConfig, OLMoForCausalLM, OLMoLayerNorm


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260921)


def fixture_model(backend="math"):
    return OLMoForCausalLM(OLMoConfig.tiny(), attention_backend=backend)


def test_native_geometry_has_65_tensors_full_vocabulary_and_one_readout():
    config = OLMoConfig.native_1b()
    model = OLMoForCausalLM(config, device="meta")
    assert len(model.layers) == 16
    assert sum(p.numel() for p in model.parameters()) == 1_176_764_416
    assert model.readout_weight.shape == (50304, 2048)
    assert model.token_embeddings.padding_idx is None
    assert model.readout_weight is model.transformer.wte.weight
    assert len(model.state_dict()) == 65
    for block in model.layers:
        assert block.att_proj.weight.shape == (6144, 2048)
        assert block.attn_out.weight.shape == (2048, 2048)
        assert block.ff_proj.weight.shape == (16384, 2048)
        assert block.ff_out.weight.shape == (2048, 8192)
        assert block.attn_norm.weight is block.ff_norm.weight is None
    assert not any("norm" in key or "rope" in key or "lm_head" in key for key in model.state_dict())
    assert all(key.startswith("transformer.") for key in model.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nonaffine_layernorm_preserves_native_function_and_input_gradient(dtype):
    x = (2 + torch.randn(2, 5, 32)).to(dtype).requires_grad_()
    y = x.detach().clone().requires_grad_()
    module = OLMoLayerNorm(32, 1e-5)
    actual = module(x)
    expected = F.layer_norm(y, (32,), weight=None, bias=None, eps=1e-5)
    tangent = torch.randn_like(actual)
    actual.backward(tangent)
    expected.backward(tangent)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    assert not list(module.parameters())


def test_native_padding_row_lookup_gradient_and_full_softmax_are_preserved():
    model = fixture_model()
    tokens = torch.tensor([[model.config.pad_token_id, 3, 4]])
    hidden = model(tokens, return_logits=False).last_hidden_state
    (hidden * torch.randn_like(hidden)).sum().backward()
    assert model.readout_weight.grad[model.config.pad_token_id].abs().sum() > 0
    model.zero_grad(set_to_none=True)
    logits = model(tokens).logits
    F.cross_entropy(logits[:, -1], torch.tensor([5])).backward()
    assert logits.shape[-1] == 67
    # Extra output rows remain in the denominator even though input tokenizer
    # cannot emit them; their readout gradients must not be cropped away.
    assert model.readout_weight.grad[model.config.tokenizer_vocab_size:].abs().sum() > 0


def test_causal_prefix_outputs_and_input_gradients_ignore_future():
    model = fixture_model()
    x = torch.randn(2, 7, 32, requires_grad=True)
    changed = x.detach().clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 3
    first, second = model(inputs_embeds=x), model(inputs_embeds=changed)
    torch.testing.assert_close(first.logits[:, :4], second.logits[:, :4], rtol=0, atol=0)
    first.logits[:, :4].square().mean().backward()
    assert torch.count_nonzero(x.grad[:, 4:]) == 0
    assert x.grad[:, :4].abs().sum() > 0


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1, 1, 1), (3, 2, 2), (1, 5, 1)])
def test_chunks_match_full_and_store_unrotated_native_kv(chunks):
    model = fixture_model()
    tokens = torch.randint(0, 60, (2, 7))
    full = model(tokens)
    cache, offset, outputs = None, 0, []
    for length in chunks:
        output = model(tokens[:, offset:offset + length], past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        outputs.append(output.logits)
        offset += length
        assert cache.sequence_length == offset
        assert all(k.shape == v.shape == (2, 4, offset, 8) for k, v in cache.key_values)
    torch.testing.assert_close(torch.cat(outputs, 1), full.logits, rtol=5e-6, atol=2e-6)
    layer = model.layers[0]
    _, keys, values = layer.project_qkv(layer.attn_norm(model.token_embeddings(tokens)))
    torch.testing.assert_close(cache.key_values[0][0], keys, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cache.key_values[0][1], values, rtol=1e-6, atol=1e-6)


def test_cached_suffix_loss_preserves_prefix_parameter_gradients():
    full, cached = fixture_model(), None
    cached = copy.deepcopy(full)
    tokens = torch.randint(0, 60, (2, 6))
    full(tokens).logits[:, 3:].square().mean().backward()
    prefix = cached(tokens[:, :3], use_cache=True)
    cached(tokens[:, 3:], past_key_values=prefix.past_key_values).logits.square().mean().backward()
    for (name, parameter), (_, other) in zip(full.named_parameters(), cached.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=5e-7, msg=name)


@pytest.mark.parametrize("left_padding", [False, True])
def test_padding_and_absolute_positions_match_unpadded_cache(left_padding):
    model = fixture_model()
    tokens = torch.tensor([[3, 6, 9, 2], [5, 8, 1, 1]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    positions = torch.tensor([[10, 11, 12, 13], [70, 71, 0, 0]])
    if left_padding:
        tokens[1], valid[1], positions[1] = torch.tensor([1, 1, 5, 8]), torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 70, 71])
    padded = model(tokens, attention_mask=valid, position_ids=positions, use_cache=True)
    short = model(torch.tensor([[5, 8]]), position_ids=torch.tensor([[70, 71]]), use_cache=True)
    torch.testing.assert_close(padded.logits[1, valid[1]], short.logits[0], rtol=5e-6, atol=2e-6)
    more = torch.tensor([[11, 12], [11, 12]])
    extended = model(more, past_key_values=padded.past_key_values, use_cache=True)
    short_extended = model(more[1:], past_key_values=short.past_key_values)
    torch.testing.assert_close(extended.logits[1], short_extended.logits[0], rtol=5e-6, atol=2e-6)
    assert extended.past_key_values.position_ids[1, -2:].tolist() == [72, 73]


def test_masked_rows_stay_finite_in_forward_and_backward():
    model = fixture_model()
    output = model(torch.tensor([[3, 4, 5], [1, 1, 1]]), attention_mask=torch.tensor([[1, 1, 1], [0, 0, 0]]))
    assert output.logits.isfinite().all()
    output.logits.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_explicit_offset_cache_and_full_forward_agree():
    model = fixture_model()
    tokens = torch.tensor([[2, 4, 6, 8, 9]])
    positions = torch.tensor([[100, 101, 102, 103, 104]])
    full = model(tokens, position_ids=positions)
    prefix = model(tokens[:, :3], position_ids=positions[:, :3], use_cache=True)
    suffix = model(tokens[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(suffix.logits, full.logits[:, 3:], rtol=5e-6, atol=2e-6)
    assert suffix.past_key_values.position_ids.tolist() == positions.tolist()


def test_sdpa_and_math_backends_agree_on_outputs_and_gradients():
    math_model = fixture_model()
    sdpa_model = copy.deepcopy(math_model)
    sdpa_model.attention_backend = "sdpa"
    tokens = torch.randint(0, 60, (2, 6))
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    a, b = math_model(tokens, attention_mask=mask), sdpa_model(tokens, attention_mask=mask)
    torch.testing.assert_close(a.logits, b.logits, rtol=5e-6, atol=2e-6)
    a.logits.square().mean().backward()
    b.logits.square().mean().backward()
    for (name, parameter), (_, other) in zip(math_model.named_parameters(), sdpa_model.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=5e-7, msg=name)


def test_inputs_embeds_and_strict_meta_materialization_keep_native_tying():
    model = fixture_model()
    tokens = torch.randint(0, 60, (1, 4))
    expected = model(tokens)
    torch.testing.assert_close(model(inputs_embeds=model.token_embeddings(tokens)).logits, expected.logits, rtol=0, atol=0)
    restored = OLMoForCausalLM(OLMoConfig.from_dict(model.config.to_dict()), device="meta", attention_backend="math")
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True, assign=True)
    assert restored.readout_weight is restored.token_embeddings.weight
    assert len(list(restored.parameters())) == len({id(p) for p in restored.parameters()})
    torch.testing.assert_close(restored(tokens).logits, expected.logits, rtol=0, atol=0)
    malformed = dict(model.state_dict(), **{"lm_head.weight": model.readout_weight.detach().clone()})
    with pytest.raises(RuntimeError, match="Unexpected key"):
        restored.load_state_dict(malformed, strict=True)


def test_tied_optimizer_save_reload_matches_uninterrupted_update(tmp_path):
    model = fixture_model()
    def optimizer_for(backbone):
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=3e-4, betas=(0.9, 0.95), foreach=False, fused=False)
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        assert sum(p is backbone.readout_weight for p in owned) == 1
        return optimizer
    def update(backbone, optimizer, tokens):
        optimizer.zero_grad(set_to_none=True)
        logits = backbone(tokens).logits
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
        loss.backward()
        optimizer.step()
        return loss.detach()
    optimizer = optimizer_for(model)
    update(model, optimizer, torch.tensor([[2, 7, 4, 9], [3, 2, 11, 5]]))
    path = tmp_path / "ordinary-olmo.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = OLMoForCausalLM(model.config, device="meta", attention_backend="math")
    restored.load_state_dict(saved["model"], strict=True, assign=True)
    restored_optimizer = optimizer_for(restored)
    restored_optimizer.load_state_dict(saved["optimizer"])
    tokens = torch.tensor([[6, 13, 5, 1], [9, 17, 2, 8]])
    torch.testing.assert_close(update(model, optimizer, tokens), update(restored, restored_optimizer, tokens), rtol=0, atol=0)
    for (name, p), (_, q) in zip(model.named_parameters(), restored.named_parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0, msg=name)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[p][key], restored_optimizer.state[q][key], rtol=0, atol=0)


@pytest.mark.parametrize("changes", [{"num_heads": 3}, {"pad_token_id": 67}, {"model_dim": 0}, {"tokenizer_vocab_size": 68}, {"layer_norm_eps": float("nan")}, {"rope_freq_constant": 0}])
def test_bad_geometry_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(OLMoConfig.tiny(), **changes)


def test_invalid_inputs_and_cache_are_rejected():
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
    with pytest.raises(ValueError, match="layer count"):
        model(tokens, past_key_values=OLMoCache(cache.key_values[:-1], cache.attention_mask, cache.position_ids))
    with pytest.raises(TypeError, match="OLMoCache"):
        model(tokens, past_key_values=cache.key_values)
