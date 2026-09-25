"""Fusion calibration, online replay, boundary and attached-cache contracts."""

import copy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.olmo_fbt import FBTConfig, FBTMode, FBTOnlineMode, OLMoFBT
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(5520)


def _model(tiled=True):
    cls = OLMoTiledRTForCausalLM if tiled else OLMoRTForCausalLM
    return OLMoFBT(cls(OLMoConfig.tiny(), attention_backend="math"))


def _fusion(model, previous, token):
    eps = model.fusion_config.norm_eps
    token_norm = token.float() / (token.float().square().mean(-1, keepdim=True) + eps).sqrt()
    gate = torch.sigmoid(F.linear(token_norm.to(token.dtype), model.fusion.token_gate.weight))
    product = F.linear(previous, model.fusion.state_proj.weight) * gate
    normalized = product.float() / (product.float().square().mean(-1, keepdim=True) + eps).sqrt()
    return (normalized.to(product.dtype) * model.fusion.output_scale).to(token.dtype)


def _grads_match(actual, expected, atol=2e-5, rtol=3e-4):
    for (name, p), (name2, q) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == name2
        if q.grad is None:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, name
        elif p.grad is None:
            assert torch.count_nonzero(q.grad) == 0, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=atol, rtol=rtol, msg=name)


def test_constructor_preserves_native_tensors_rng_and_calibrates_fixed_scale():
    native = OLMoRTForCausalLM(OLMoConfig.tiny())
    before = {name: tensor.clone() for name, tensor in native.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model = OLMoFBT(native)
    assert torch.equal(rng, torch.get_rng_state())
    for name, tensor in native.state_dict().items():
        assert torch.equal(tensor, before[name])
    expected = before["transformer.wte.weight"].float().square().mean().sqrt()
    torch.testing.assert_close(model.fusion.output_scale, expected, atol=0, rtol=0)
    assert "fusion.output_scale" in dict(model.named_buffers())
    assert "fusion.output_scale" not in dict(model.named_parameters())
    assert sum(p.numel() for p in model.fusion.parameters()) == 2 * native.config.model_dim**2
    assert model.token_embeddings is native.token_embeddings
    assert model.readout_weight is native.readout_weight
    twin = OLMoFBT(copy.deepcopy(native))
    for p, q in zip(model.fusion.parameters(), twin.fusion.parameters()):
        assert torch.equal(p, q)
    with torch.no_grad():
        native.readout_weight.mul_(2)
    torch.testing.assert_close(model.fusion.output_scale, expected, atol=0, rtol=0)


def test_fusion_formula_and_asymmetric_gradients():
    model = _model()
    source = torch.randn(2, 4, 32, requires_grad=True)
    token = torch.randn(2, 4, 32, requires_grad=True)
    actual = model.fusion(source, token)
    expected = _fusion(model, source, token)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    gradients = torch.autograd.grad((actual * torch.randn_like(actual)).sum(),
                                    (source, token, *model.fusion.parameters()))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)


def test_plain_backbone_supported_and_rt_request_rejected():
    model = OLMoFBT(OLMoForCausalLM(OLMoConfig.tiny(), attention_backend="math"))
    ids = torch.tensor([[2, 3, 4]])
    assert len(model(ids).pass_hidden_states) == 2
    with pytest.raises(ValueError, match="cache provenance"):
        model.forward_online(ids)
    with pytest.raises(ValueError, match="RT-capable"):
        model(ids, mode=FBTMode(rt_mode=RTMode((0,))))


def test_fbt_off_executes_one_standalone_rt_pass(monkeypatch):
    model = _model()
    ids = torch.tensor([[2, 3, 4]])
    rt = RTMode((0,), .37)
    calls = []
    original = model.backbone.forward
    def tracked(*args, **kwargs):
        calls.append(kwargs.get("mode"))
        return original(*args, **kwargs)
    monkeypatch.setattr(model.backbone, "forward", tracked)
    result = model(ids, mode=FBTMode(enabled=False, num_passes=7, rt_mode=rt))
    assert calls == [rt]
    assert len(result.pass_hidden_states) == 1
    expected = original(ids, mode=rt)
    torch.testing.assert_close(result.logits, expected.logits, atol=1e-6, rtol=1e-6)


def test_beta_zero_never_calls_new_branch(monkeypatch):
    model = _model()
    def forbidden(*args, **kwargs):
        raise AssertionError("Feedback branch ran at beta zero")
    monkeypatch.setattr(model.fusion, "forward", forbidden)
    ids = torch.tensor([[2, 3, 4]])
    model(ids, mode=FBTMode(num_passes=3, beta=0))
    model.forward_online(ids, mode=FBTOnlineMode(beta=0))


def test_shift_preserves_raw_first_valid_and_gap_boundaries():
    model = _model()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    valid = torch.tensor([[False, True, True, False, True, True]])
    docs = torch.tensor([[-1, 8, 8, -1, 8, 8]])
    seen = []
    handle = model.backbone.register_forward_pre_hook(lambda m, args, kwargs: seen.append(kwargs["inputs_embeds"]), with_kwargs=True)
    try:
        output = model(ids, attention_mask=valid, document_ids=docs)
    finally:
        handle.remove()
    embeddings = model.token_embeddings(ids)
    expected = embeddings.clone()
    for target in (2, 5):
        expected[:, target:target + 1] = _fusion(model, output.pass_hidden_states[0][:, target - 1:target], embeddings[:, target:target + 1])
    torch.testing.assert_close(seen[1], expected, atol=2e-7, rtol=2e-6)
    assert torch.equal(seen[1][:, [0, 1, 3, 4]], embeddings[:, [0, 1, 3, 4]])


@pytest.mark.parametrize("alpha,selected", [(0.0, (0,)), (.37, (0, 1)), (1.0, (1,)), (1.0, ())])
def test_online_matches_independent_full_prefix_replay_all_gradients(alpha, selected):
    actual = _model()
    expected = _model(tiled=False)
    expected.load_state_dict(actual.state_dict(), strict=True)
    mode = FBTOnlineMode(beta=.37, rt_mode=RTMode(selected, alpha))
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    valid = torch.tensor([[True, True, True, True, True], [False, True, True, True, False]])
    positions = torch.tensor([[3, 5, 8, 10, 17], [0, 2, 4, 9, 11]])
    got = actual.forward_online(inputs_embeds=x, attention_mask=valid, position_ids=positions, mode=mode)
    inputs, states = [], []
    for t in range(y.shape[1]):
        current = y[:, t:t + 1]
        if t:
            feedback = _fusion(expected, states[-1], current)
            eligible = (valid[:, t:t + 1] & valid[:, t - 1:t]).unsqueeze(-1)
            current = torch.where(eligible, .63 * current + .37 * feedback, current)
        inputs.append(current)
        result = expected.backbone(inputs_embeds=torch.cat(inputs, 1), attention_mask=valid[:, :t + 1],
                                   position_ids=positions[:, :t + 1], mode=mode.rt_mode, return_logits=False)
        states.append(result.last_hidden_state[:, -1:])
    want = torch.cat(states, 1)
    torch.testing.assert_close(got.last_hidden_state, want, atol=4e-6, rtol=2e-5)
    cotangent = torch.randn_like(want)
    got.last_hidden_state.backward(cotangent)
    want.backward(cotangent)
    torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=3e-4)
    _grads_match(actual, expected)


def test_online_chunked_attached_feedback_and_cache_gradients_match_whole():
    whole = _model()
    split = copy.deepcopy(whole)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    mode = FBTOnlineMode(beta=.7, rt_mode=RTMode((0,), .37))
    valid = torch.tensor([[True, True, True, True, True], [False, True, True, True, True]])
    docs = torch.tensor([[5, 5, 5, 5, 5], [-1, 7, 7, 7, 7]])
    full = whole.forward_online(inputs_embeds=x, attention_mask=valid, document_ids=docs, mode=mode)
    prefix = split.forward_online(inputs_embeds=y[:, :2], attention_mask=valid[:, :2], document_ids=docs[:, :2], mode=mode)
    suffix = split.forward_online(inputs_embeds=y[:, 2:], attention_mask=valid, document_ids=docs, past_key_values=prefix.past_key_values, mode=mode)
    got = torch.cat((prefix.last_hidden_state, suffix.last_hidden_state), 1)
    torch.testing.assert_close(got, full.last_hidden_state, atol=0, rtol=0)
    # Loss includes exported terminal KV credit and later-chunk feedback credit.
    probe = torch.randn_like(got)
    for result in (full, suffix):
        result.last_hidden_state.retain_grad()
    full_loss = (full.last_hidden_state * probe).sum()
    split_loss = (got * probe).sum()
    for pair, other in zip(full.past_key_values.backbone_cache.key_values, suffix.past_key_values.backbone_cache.key_values):
        for tensor, target in zip(pair, other):
            cotangent = torch.randn_like(tensor)
            full_loss = full_loss + (tensor * cotangent).sum()
            split_loss = split_loss + (target * cotangent).sum()
    full_loss.backward()
    split_loss.backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=2e-4)
    _grads_match(whole, split)
    assert y.grad[:, :2].norm() > 0


@pytest.mark.parametrize("mutation", ["fusion", "scale", "native", "convert", "child_convert", "mode", "grad", "precision", "owner"])
def test_online_cache_rejects_incompatible_reuse(mutation):
    model = _model()
    ids = torch.tensor([[2, 3]])
    mode = FBTOnlineMode(rt_mode=RTMode((0,)))
    cache = model.forward_online(ids, mode=mode).past_key_values
    next_mode = mode
    if mutation in ("fusion", "scale", "native"):
        tensor = {"fusion": model.fusion.state_proj.weight, "scale": model.fusion.output_scale,
                  "native": model.readout_weight}[mutation]
        with torch.no_grad():
            tensor.add_(.01)
    elif mutation == "convert":
        model.float()
    elif mutation == "child_convert":
        model.fusion.half().float()
    elif mutation == "mode":
        next_mode = replace(mode, beta=.4)
    elif mutation == "precision":
        model.backbone.attention_precision = "fp32"
    elif mutation == "owner":
        model = copy.deepcopy(model)
    if mutation == "grad":
        with torch.no_grad(), pytest.raises(ValueError, match="context"):
            model.forward_online(ids[:, :1], mode=next_mode, past_key_values=cache)
    else:
        with pytest.raises(ValueError, match="Cached"):
            model.forward_online(ids[:, :1], mode=next_mode, past_key_values=cache)


def test_online_cache_metadata_owned_and_document_prefix_checked():
    model = _model()
    ids = torch.tensor([[2, 3]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    docs = torch.tensor([[9, 9]])
    positions = torch.tensor([[10, 13]])
    cache = model.forward_online(ids, attention_mask=valid, document_ids=docs, position_ids=positions).past_key_values
    valid.zero_(); docs.zero_(); positions.zero_()
    assert cache.attention_mask.all()
    assert torch.equal(cache.document_ids, torch.tensor([[9, 9]]))
    assert torch.equal(cache.position_ids, torch.tensor([[10, 13]]))
    result = model.forward_online(ids[:, :1], past_key_values=cache)
    assert result.past_key_values.position_ids[0, -1] == 14
    assert result.past_key_values.document_ids[0, -1] == 9
    with pytest.raises(ValueError, match="document IDs"):
        model.forward_online(ids[:, :1], past_key_values=cache, document_ids=torch.tensor([[3, 3, 3]]))


def test_cache_types_cannot_mix_and_packed_rows_rejected():
    model = _model()
    ids = torch.tensor([[2, 3, 4]])
    native = model.backbone(ids, mode=RTMode(()), use_cache=True).past_key_values
    online = model.forward_online(ids).past_key_values
    with pytest.raises(TypeError, match="FBTOnlineCache"):
        model.forward_online(ids, past_key_values=native)
    with pytest.raises(ValueError, match="Finite"):
        model(ids, past_key_values=online)
    with pytest.raises(ValueError, match="Finite"):
        model(ids, use_cache=True)
    with pytest.raises(TypeError):
        model.backbone(ids, past_key_values=online)
    for method in (model.forward, model.forward_online):
        with pytest.raises(ValueError, match="multi-document"):
            method(ids, document_ids=torch.tensor([[1, 1, 2]]))


def test_save_reload_preserves_fixed_scale_and_outputs():
    model = _model()
    with torch.no_grad():
        model.readout_weight.mul_(1.2)
    copy_model = _model()
    copy_model.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([[2, 3, 4]])
    mode = FBTMode(num_passes=3, rt_mode=RTMode((0,), .37))
    assert FBTConfig.from_dict(model.fusion_config.to_dict()) == model.fusion_config
    torch.testing.assert_close(copy_model(ids, mode=mode).logits, model(ids, mode=mode).logits, atol=0, rtol=0)


@pytest.mark.parametrize("kwargs", [{"num_passes": 0}, {"num_passes": True}, {"beta": float("nan")}, {"beta": -1}, {"rt_mode": None}])
def test_invalid_finite_modes_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        FBTMode(**kwargs)


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("fbt,rt", [(f, r) for f in (False, True) for r in (False, True)])
@pytest.mark.parametrize("offset_positions", [False, True])
def test_full_valid_causal_preserves_explicit_math_outputs_and_all_gradients(tiled, fbt, rt, offset_positions):
    explicit = _model(tiled=tiled)
    implicit = copy.deepcopy(explicit)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    valid = torch.ones((2, 5), dtype=torch.bool)
    docs = torch.tensor([[3]*5, [9]*5])
    positions = torch.tensor([[3, 5, 8, 9, 12], [0, 2, 4, 6, 8]]) if offset_positions else None
    mode = FBTMode(enabled=fbt, num_passes=3, rt_mode=RTMode((0,) if rt else ()))
    kwargs = dict(attention_mask=valid, document_ids=docs, position_ids=positions, mode=mode)
    a = explicit(inputs_embeds=x, **kwargs)
    b = implicit(inputs_embeds=y, full_valid_causal=True, **kwargs)
    torch.testing.assert_close(a.logits, b.logits, atol=2e-6, rtol=2e-5)
    for first, second in zip(a.pass_hidden_states, b.pass_hidden_states):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
    cotangent = torch.randn_like(a.logits)
    (a.logits*cotangent).sum().backward()
    (b.logits*cotangent).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=3e-4)
    _grads_match(implicit, explicit)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("opt_in", [None, False, True])
def test_full_valid_causal_controls_every_stack_call_and_preserves_default(enabled, opt_in):
    model = _model()
    ids = torch.tensor([[2, 3, 4]])
    positions = torch.tensor([[4, 7, 12]])
    seen = []
    handle = model.backbone.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs.copy()), with_kwargs=True)
    try:
        extra = {} if opt_in is None else {"full_valid_causal": opt_in}
        model(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool), position_ids=positions,
              mode=FBTMode(enabled=enabled, num_passes=3, rt_mode=RTMode((0,))), **extra)
    finally:
        handle.remove()
    assert len(seen) == (3 if enabled else 1)
    for call in seen:
        assert torch.equal(call["position_ids"], positions)
        if opt_in:
            assert call["attention_mask"] is None
        else:
            assert torch.equal(call["attention_mask"], torch.ones_like(ids, dtype=torch.bool))


@pytest.mark.parametrize("kwargs,error,match", [
    ({"attention_mask": torch.tensor([[True, False, True]])}, ValueError, "all tokens valid"),
    ({"attention_mask": torch.tensor([[1, 2, 1]])}, ValueError, "0/1"),
    ({"document_ids": torch.tensor([[1, 1, 2]])}, ValueError, "multi-document"),
    ({"document_ids": torch.tensor([[1, -1, 1]])}, ValueError, "nonnegative document"),
    ({"document_ids": torch.tensor([[1., 1., 1.]])}, ValueError, "int64"),
    ({"position_ids": torch.tensor([[0, -1, 2]])}, ValueError, "nonnegative"),
    ({"position_ids": torch.tensor([[0., 1., 2.]])}, ValueError, "int64"),
    ({"position_ids": torch.tensor([[0, 1]])}, ValueError, "int64"),
    ({"use_cache": True}, ValueError, "do not accept caches"),
    ({"past_key_values": object()}, ValueError, "do not accept caches"),
    ({"full_valid_causal": 1}, TypeError, "must be boolean"),
    ({"full_valid_causal": torch.tensor(True)}, TypeError, "must be boolean"),
])
def test_full_valid_causal_rejects_invalid_contract_before_any_stack(monkeypatch, kwargs, error, match):
    model = _model()
    def forbidden(*args, **kwargs):
        raise AssertionError("stack executed before validation")
    monkeypatch.setattr(model, "_stack", forbidden)
    with pytest.raises(error, match=match):
        model(torch.tensor([[2, 3, 4]]), **{"full_valid_causal": True, **kwargs})
