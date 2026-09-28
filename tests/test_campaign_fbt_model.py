"""Portable K4/first-pass RT and externally assigned feedback jitter contracts."""

import copy
from dataclasses import asdict, replace

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_static import PreparedFBTLayout
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode


@pytest.fixture(autouse=True)
def seeded_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(9381)


def model(checkpoint=False):
    return OLMoFBT(OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        ordinary_activation_checkpointing=checkpoint))


def batch(padded=False):
    ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    if padded:
        valid[0, 0] = False
        valid[1, 2:] = False
    docs = torch.tensor([[2] * 5, [3] * 5]).masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, docs)


def noise_for(core, tokens, mode):
    generator = torch.Generator().manual_seed(4753)
    return tuple(torch.empty(tokens.input_ids.shape[0], tokens.input_ids.shape[1] - 1,
        core.config.model_dim).uniform_(-1, 1, generator=generator)
        for _ in range(mode.num_passes - 1))


def assert_grads(actual, expected):
    for (name, parameter), (other, reference) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=4e-5, rtol=4e-4, msg=name)


@pytest.mark.parametrize("policy,expected", [
    ("ordinary-v1", [(), (0, 1), (0, 1), (0, 1)]),
    ("configured-rt-v1", [(0, 1)] * 4),
])
def test_k4_executes_explicit_first_pass_policy_and_attaches_gradients(monkeypatch, policy, expected):
    core = model()
    tokens = batch()
    mode = FBTMode(num_passes=4, rt_mode=RTMode((0, 1)), first_pass_policy=policy)
    calls = []
    original = core._stack

    def observed(embeddings, rt, **kwargs):
        calls.append(rt.selected_layers)
        return original(embeddings, rt, **kwargs)

    monkeypatch.setattr(core, "_stack", observed)
    result = core(tokens.input_ids, mode=mode, return_logits=False)
    assert calls == expected
    for state in result.pass_hidden_states:
        state.retain_grad()
    (result.last_hidden_state * torch.randn_like(result.last_hidden_state)).sum().backward()
    for state in result.pass_hidden_states:
        assert torch.isfinite(state.grad).all() and state.grad.norm() > 0
    assert core.fusion.state_proj.weight.grad.norm() > 0


def test_legacy_mode_dict_defaults_preserve_bootstrap_and_k1():
    rt = RTMode((0, 1))
    legacy = FBTMode(**{"enabled": True, "num_passes": 1, "beta": 1.0, "rt_mode": rt})
    assert legacy.first_pass_policy == "ordinary-v1"
    assert legacy.feedback_jitter == 0
    core = model()
    ids = batch().input_ids
    expected = core.backbone(ids, mode=RTMode(()))
    actual = core(ids, mode=legacy)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    configured = replace(legacy, first_pass_policy="configured-rt-v1")
    torch.testing.assert_close(core(ids, mode=configured).logits,
        core.backbone(ids, mode=rt).logits, rtol=0, atol=0)
    assert asdict(configured)["first_pass_policy"] == "configured-rt-v1"


@pytest.mark.parametrize("policy", ["ordinary-v1", "configured-rt-v1"])
def test_disabled_fbt_always_executes_one_rt_pass(policy):
    core = model()
    mode = FBTMode(enabled=False, num_passes=4, rt_mode=RTMode((0, 1)),
        first_pass_policy=policy, feedback_jitter=.02)
    result = core(batch().input_ids, mode=mode)
    assert len(result.pass_hidden_states) == 1
    expected = core.backbone(batch().input_ids, mode=mode.rt_mode)
    torch.testing.assert_close(result.logits, expected.logits, rtol=0, atol=0)


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("policy,jitter,checkpoint", [
    ("ordinary-v1", 0, False),
    ("configured-rt-v1", 0, False),
    ("configured-rt-v1", .02, False),
    ("configured-rt-v1", .02, True),
])
def test_prepared_k4_matches_eager_states_all_gradients_and_rng(padded, policy, jitter, checkpoint):
    core = model(checkpoint)
    reference = copy.deepcopy(core)
    tokens = batch(padded)
    mode = FBTMode(num_passes=4, rt_mode=RTMode((0,), .7),
        first_pass_policy=policy, feedback_jitter=jitter)
    noise = noise_for(core, tokens, mode) if jitter else None
    layout = PreparedFBTLayout(core, tokens)
    layout.validate_execution(mode, feedback_noise=noise)
    rng_before = torch.get_rng_state().clone()
    actual = layout.forward(tokens.input_ids, mode, feedback_noise=noise)
    expected = reference(tokens.input_ids, attention_mask=tokens.valid_mask,
        document_ids=tokens.document_ids, mode=mode, return_logits=False, feedback_noise=noise)
    assert torch.equal(rng_before, torch.get_rng_state())
    loss, reference_loss = 0, 0
    for state, want in zip(actual.pass_hidden_states, expected.pass_hidden_states):
        torch.testing.assert_close(state, want, atol=3e-6, rtol=2e-5)
        probe = torch.randn_like(state)
        loss = loss + (state * probe).sum()
        reference_loss = reference_loss + (want * probe).sum()
    rng_before_backward = torch.get_rng_state().clone()
    loss.backward()
    reference_loss.backward()
    assert torch.equal(rng_before_backward, torch.get_rng_state())
    assert_grads(core, reference)


def test_training_noise_is_added_before_fusion_only_at_eligible_positions(monkeypatch):
    core = model()
    tokens = batch(padded=True)
    mode = FBTMode(num_passes=4, first_pass_policy="configured-rt-v1", feedback_jitter=.02)
    noise = noise_for(core, tokens, mode)
    seen = []
    original = core.fusion.forward

    def observed(previous, embeddings):
        seen.append(previous.detach().clone())
        return original(previous, embeddings)

    monkeypatch.setattr(core.fusion, "forward", observed)
    result = core(tokens.input_ids, attention_mask=tokens.valid_mask,
        document_ids=tokens.document_ids, mode=mode, feedback_noise=noise)
    eligible = tokens.valid_mask[:, 1:] & tokens.valid_mask[:, :-1]
    for index, previous in enumerate(seen):
        expected = result.pass_hidden_states[index][:, :-1] + torch.where(
            eligible.unsqueeze(-1), noise[index] * .02, torch.zeros_like(noise[index]))
        torch.testing.assert_close(previous, expected, atol=0, rtol=0)


def test_eval_disables_jitter_without_rng_and_rejects_supplied_noise():
    core = model().eval()
    tokens = batch()
    mode = FBTMode(num_passes=4, rt_mode=RTMode((0, 1)),
        first_pass_policy="configured-rt-v1", feedback_jitter=.02)
    noise = noise_for(core, tokens, mode)
    layout = PreparedFBTLayout(core, tokens)
    rng_before = torch.get_rng_state().clone()
    expected = core(tokens.input_ids, mode=replace(mode, feedback_jitter=0)).last_hidden_state
    torch.testing.assert_close(core(tokens.input_ids, mode=mode).last_hidden_state, expected, atol=0, rtol=0)
    layout.validate_execution(mode)
    torch.testing.assert_close(layout.forward(tokens.input_ids, mode).last_hidden_state, expected,
        atol=3e-6, rtol=2e-5)
    assert torch.equal(rng_before, torch.get_rng_state())
    with pytest.raises(ValueError, match="only accepted"):
        core(tokens.input_ids, mode=mode, feedback_noise=noise)
    with pytest.raises(ValueError, match="only accepted"):
        layout.validate_execution(mode, feedback_noise=noise)


def test_k4_jitter_preserves_causality_and_attached_prefix_gradients():
    core = model()
    mode = FBTMode(num_passes=4, rt_mode=RTMode((0, 1)),
        first_pass_policy="configured-rt-v1", feedback_jitter=.02)
    tokens = batch()
    noise = noise_for(core, tokens, mode)
    x = core.token_embeddings(tokens.input_ids).detach().requires_grad_()
    y = x.detach().clone()
    y[:, 3:] += torch.randn_like(y[:, 3:])
    output = core(inputs_embeds=x, mode=mode, feedback_noise=noise)
    changed = core(inputs_embeds=y, mode=mode, feedback_noise=noise)
    for actual, other in zip(output.pass_hidden_states, changed.pass_hidden_states):
        torch.testing.assert_close(actual[:, :3], other[:, :3], atol=0, rtol=0)
    (output.last_hidden_state[:, 2] * torch.randn_like(output.last_hidden_state[:, 2])).sum().backward()
    assert torch.count_nonzero(x.grad[:, 3:]) == 0
    assert x.grad[:, :3].norm() > 0


def test_static_noise_buffers_allow_refill_but_reject_replacement_and_invalid_values():
    core = model()
    tokens = batch()
    mode = FBTMode(num_passes=4, feedback_jitter=.02)
    noise = noise_for(core, tokens, mode)
    layout = PreparedFBTLayout(core, tokens)
    signature = layout.validate_execution(mode, feedback_noise=noise)
    for item in noise:
        item.mul_(.5)
    layout.validate_execution(mode, feedback_noise=noise, expected_signature=signature)
    with pytest.raises(ValueError, match="context differs"):
        layout.validate_execution(mode, feedback_noise=tuple(item.clone() for item in noise),
            expected_signature=signature)
    noise[0][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite unit noise"):
        layout.validate_execution(mode, feedback_noise=noise)


def test_prepared_jitter_body_has_no_host_value_checks(monkeypatch):
    core = model()
    tokens = batch(True)
    mode = FBTMode(num_passes=4, rt_mode=RTMode((0,)),
        first_pass_policy="configured-rt-v1", feedback_jitter=.02)
    noise = noise_for(core, tokens, mode)
    layout = PreparedFBTLayout(core, tokens)
    layout.validate_execution(mode, feedback_noise=noise)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tensor-value synchronization entered capture body")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    layout.forward(tokens.input_ids, mode, feedback_noise=noise).last_hidden_state.sum().backward()


@pytest.mark.parametrize("problem", ["missing", "length", "shape", "dtype", "grad", "range", "nan"])
def test_training_jitter_rejects_bad_external_noise(problem):
    core = model()
    tokens = batch()
    mode = FBTMode(num_passes=4, feedback_jitter=.02)
    noise = list(noise_for(core, tokens, mode))
    if problem == "missing":
        noise = None
    elif problem == "length":
        noise.pop()
    elif problem == "shape":
        noise[0] = noise[0][:, :-1]
    elif problem == "dtype":
        noise[0] = noise[0].double()
    elif problem == "grad":
        noise[0].requires_grad_()
    elif problem == "range":
        noise[0][0, 0, 0] = 1.001
    elif problem == "nan":
        noise[0][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="noise"):
        core(tokens.input_ids, mode=mode, feedback_noise=noise)
    with pytest.raises(ValueError, match="noise"):
        PreparedFBTLayout(core, tokens).validate_execution(mode, feedback_noise=noise)


@pytest.mark.parametrize("kwargs", [
    {"first_pass_policy": "all"}, {"feedback_jitter": True},
    {"feedback_jitter": -.01}, {"feedback_jitter": float("inf")},
])
def test_reject_invalid_new_mode_fields(kwargs):
    with pytest.raises(ValueError):
        FBTMode(**kwargs)
