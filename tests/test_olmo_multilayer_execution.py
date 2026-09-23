"""Bounded four-layer RT/FBT/NextLat integration against independent math.

These are CPU FP32 semantic checks, not claims about CUDA graphs or BF16
rounding. Native-weight GPU coverage lives in the F3e protocol and reports.
"""

import copy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from cdrm.pretrained import olmo_tiled
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTConfig, FBTMode, OLMoFBT
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM
from cdrm.pretrained.olmo_recurrent_oracle import olmo_recurrent_model_oracle
from cdrm.pretrained.olmo_reference import build_olmo_reference
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.static_training import StaticFBTTraining, normalized_objective


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(93041)


def _model(memory, *, checkpoint=True):
    config = replace(OLMoConfig.tiny(), num_layers=4)
    base = OLMoTiledRTForCausalLM(config, attention_backend="math",
        attention_precision="fp32", ordinary_activation_checkpointing=checkpoint,
        cast_weights_once=True, backward_memory=memory)
    return FBTNextLatLM(OLMoFBT(base, FBTConfig(seed=61)),
        NextLatConfig(config.model_dim, proj_factor=2.0, lambda_latent=.3,
            lambda_kl=.7, seed=71, vocab_chunk_size=3), gamma=.4)


def _batch():
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17], [19, 23, 29, 31, 37, 1, 1]])
    valid = torch.tensor([[True] * 7, [False, True, True, True, True, False, False]])
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    ce, latent, kl = (valid.clone() for _ in range(3))
    ce[:, 3] = False
    latent[0, 2] = False
    kl[0, 4] = False
    return NextLatBatch(ids, valid, docs, ce, latent, kl)


def _oracle_masks(batch):
    # Explicit indexing deliberately does not reuse production mask lowering.
    masks = {name: torch.zeros(batch.input_ids.shape[0], batch.input_ids.shape[1] - distance,
                              dtype=torch.bool)
             for name, distance in (("ce", 1), ("latent", 1), ("kl", 2))}
    for row in range(batch.input_ids.shape[0]):
        for name, distance in (("ce", 1), ("latent", 1), ("kl", 2)):
            for time in range(batch.input_ids.shape[1] - distance):
                window = range(time, time + distance + 1)
                masks[name][row, time] = (
                    all(bool(batch.valid_mask[row, t]) for t in window)
                    and len({int(batch.document_ids[row, t]) for t in window}) == 1
                    and bool(getattr(batch, name + "_mask")[row, time + distance]))
    return masks


def _oracle_passes(module, batch, mode):
    """Pristine OLMo blocks, explicit RT history and separately built feedback."""
    native = module.backbone.backbone
    embeddings = native.transformer.wte(batch.input_ids)
    positions = (batch.valid_mask.long().cumsum(-1) - 1).clamp_min(0)
    hidden = olmo_recurrent_model_oracle(native, inputs_embeds=embeddings,
        selected_layers=(), position_ids=positions, attention_mask=batch.valid_mask,
        return_logits=False).last_hidden_state
    states = [hidden]
    for _ in range(1, mode.num_passes):
        fusion = module.backbone.fusion
        token = embeddings[:, 1:]
        token_normalized = token / (token.square().mean(-1, keepdim=True) + fusion.norm_eps).sqrt()
        gate = torch.sigmoid(F.linear(token_normalized, fusion.token_gate.weight))
        product = F.linear(hidden[:, :-1], fusion.state_proj.weight) * gate
        feedback = product / (product.square().mean(-1, keepdim=True) + fusion.norm_eps).sqrt()
        feedback = feedback * fusion.output_scale
        eligible = batch.valid_mask[:, :-1] & batch.valid_mask[:, 1:]
        eligible &= batch.document_ids[:, :-1] == batch.document_ids[:, 1:]
        suffix = torch.where(eligible.unsqueeze(-1),
            (1 - mode.beta) * token + mode.beta * feedback, token)
        current = torch.cat((embeddings[:, :1], suffix), dim=1)
        hidden = olmo_recurrent_model_oracle(native, inputs_embeds=current,
            selected_layers=mode.rt_mode.selected_layers, alpha=mode.rt_mode.alpha,
            position_ids=positions, attention_mask=batch.valid_mask,
            return_logits=False).last_hidden_state
        states.append(hidden)
    return embeddings, states


def _oracle_terms(module, embeddings, hidden, batch, masks):
    weight = module.backbone.backbone.transformer.wte.weight
    predicted = module.predictor(hidden[:, :-1], embeddings[:, 1:])
    ce = F.cross_entropy(F.linear(hidden[:, :-1][masks["ce"]], weight),
        batch.input_ids[:, 1:][masks["ce"]], reduction="sum")
    latent = F.smooth_l1_loss(predicted[masks["latent"]],
        hidden[:, 1:][masks["latent"]].detach(), reduction="none").mean(-1).sum()
    student = F.log_softmax(F.linear(predicted[:, :-1][masks["kl"]], weight.detach()), -1)
    teacher = F.log_softmax(F.linear(hidden[:, 1:-1][masks["kl"]].detach(), weight.detach()), -1)
    kl = (teacher.exp() * (teacher - student)).sum()
    return {"ce": ce, "latent": latent, "kl": kl}


def _gradient_close(actual, expected, *, label):
    if actual is None or expected is None:
        other = expected if actual is None else actual
        assert other is None or torch.count_nonzero(other) == 0, label
    else:
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-4, msg=label)


@pytest.mark.parametrize("memory", ["materialized", "recompute"])
@pytest.mark.parametrize("selected,passes,alpha", [
    ((0, 1), 2, 1.0), ((0, 3), 3, .37), ((0, 1, 2, 3), 2, 1.0)],
    ids=["adjacent-k2", "spread-k3", "all-k2"])
def test_static_multilayer_objective_and_all_gradients_match_native_history_oracle(memory, selected, passes, alpha):
    actual = _model(memory)
    expected = copy.deepcopy(actual)
    native = build_olmo_reference(actual.backbone.config)
    native.load_state_dict(actual.backbone.backbone.state_dict(), strict=True)
    expected.backbone.backbone = native
    value = _batch()
    mode = FBTMode(num_passes=passes, beta=.6, rt_mode=RTMode(selected, alpha))
    prepared = StaticFBTTraining(actual, value, mode=mode)
    got = prepared.loss_sums()
    embeddings, states = _oracle_passes(expected, value, mode)
    masks = _oracle_masks(value)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    assert got.counts == counts
    weights = {"ce": 1.0, "latent": .3, "kl": .7}
    objective = 0
    for index, hidden in enumerate(states):
        sums = _oracle_terms(expected, embeddings, hidden, value, masks)
        coefficient = 1 if index == 0 else .4 / (passes - 1)
        for name in weights:
            torch.testing.assert_close(got.pass_losses[index].sums[name], sums[name],
                atol=2e-5, rtol=2e-5, msg=f"pass {index} {name}")
            objective = objective + coefficient * weights[name] * sums[name] / counts[name]
    torch.testing.assert_close(normalized_objective(got), objective, atol=2e-5, rtol=2e-5)
    normalized_objective(got).backward()
    objective.backward()
    expected_parameters = dict(expected.named_parameters())
    assert dict(actual.named_parameters()).keys() == expected_parameters.keys()
    for name, parameter in actual.named_parameters():
        _gradient_close(parameter.grad, expected_parameters[name].grad, label=name)
        # Every selected and ordinary layer, shared predictor and both fusion
        # weights must receive credit from this actual combined objective.
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name


def test_placement_changes_calls_and_invalidates_static_mode_without_duplicating_parameters(monkeypatch):
    module = _model("recompute", checkpoint=False)
    base = module.backbone.backbone
    original = olmo_tiled.tiled_recurrent_layer
    indices = {id(layer): index for index, layer in enumerate(base.layers)}
    calls = []

    def observed(layer, *args, **kwargs):
        calls.append(indices[id(layer)])
        return original(layer, *args, **kwargs)

    monkeypatch.setattr(olmo_tiled, "tiled_recurrent_layer", observed)
    parameters = tuple((name, id(p), p.data_ptr()) for name, p in module.named_parameters())
    assert len(parameters) == len({identifier for _, identifier, _ in parameters})
    assert module.backbone.readout_weight is module.backbone.token_embeddings.weight
    value = _batch()
    plan = None
    results = []
    for selected, passes in (((0, 1), 2), ((0, 3), 3), ((0, 1, 2, 3), 2)):
        mode = FBTMode(num_passes=passes, beta=.6, rt_mode=RTMode(selected))
        if plan is not None:
            plan.mode = mode
            with pytest.raises(ValueError, match="context"):
                plan.validate_execution()
        plan = StaticFBTTraining(module, value, mode=mode)
        calls.clear()
        results.append(plan.loss_sums().total.detach())
        assert calls == list(selected) * (passes - 1)
        assert parameters == tuple((name, id(p), p.data_ptr()) for name, p in module.named_parameters())
    assert not torch.equal(results[0], results[1])


@pytest.mark.parametrize("memory", ["materialized", "recompute"])
def test_spread_multilayer_attached_cache_and_frozen_block_credit_match_sequential_scan(memory):
    tiled = _model(memory, checkpoint=False).backbone.backbone
    scan = OLMoRTForCausalLM(tiled.config, attention_backend="math")
    scan.load_state_dict(tiled.state_dict(), strict=True)
    for model in (tiled, scan):
        # A frozen recurrent block must still pass gradients into the earlier
        # recurrent and ordinary blocks and attached prefix inputs.
        model.layers[3].requires_grad_(False)
    x = torch.randn(1, 7, tiled.config.model_dim, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    positions = torch.tensor([[2, 5, 8, 13, 17, 20, 25]])
    mode = RTMode((0, 3), .37)
    expected = scan(inputs_embeds=y, position_ids=positions, mode=mode, use_cache=True)
    parts, offset, cache = [], 0, None
    for length in (2, 3, 2):
        result = tiled(inputs_embeds=x[:, offset:offset + length],
            position_ids=positions[:, offset:offset + length], mode=mode,
            past_key_values=cache, use_cache=True)
        parts.append(result.last_hidden_state)
        cache = result.past_key_values
        offset += length
    hidden = torch.cat(parts, dim=1)
    torch.testing.assert_close(hidden, expected.last_hidden_state, atol=4e-6, rtol=2e-5)
    cotangent = torch.randn_like(hidden[:, -2:])
    losses = [(hidden[:, -2:] * cotangent).sum(),
              (expected.last_hidden_state[:, -2:] * cotangent).sum()]
    for got_pair, want_pair in zip(cache.key_values, expected.past_key_values.key_values):
        for got, want in zip(got_pair, want_pair):
            torch.testing.assert_close(got, want, atol=4e-6, rtol=2e-5)
            probe = torch.randn_like(got[:, :, -1])
            losses[0] = losses[0] + (got[:, :, -1] * probe).sum()
            losses[1] = losses[1] + (want[:, :, -1] * probe).sum()
    losses[0].backward()
    losses[1].backward()
    _gradient_close(x.grad, y.grad, label="attached prefix input")
    assert x.grad[:, :2].norm() > 0
    for (name, p), (other, q) in zip(tiled.named_parameters(), scan.named_parameters()):
        assert name == other
        _gradient_close(p.grad, q.grad, label=name)
        if not p.requires_grad:
            assert p.grad is None and q.grad is None
