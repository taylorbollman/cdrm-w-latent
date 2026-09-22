"""Independent causal-cone and shared-gradient checks for the O5 FBT core."""

import copy

import pytest
import torch

from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, FBTOnlineMode, OLMoFBT
from cdrm.pretrained.olmo_recurrent import RTMode
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM


@pytest.fixture(autouse=True)
def cpu_fixture():
    torch.set_num_threads(1)
    torch.manual_seed(20260922)


def model():
    return OLMoFBT(OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math"))


def assert_gradient_close(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    else:
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-4)


@pytest.mark.parametrize("rt_mode", [RTMode(()), RTMode((0,), .37)])
def test_future_and_other_document_interventions_cannot_change_past_outputs_or_jacobian(rt_mode):
    core = model()
    x = torch.randn(2, 5, 32, requires_grad=True)
    mode = FBTMode(num_passes=3, beta=.6, rt_mode=rt_mode)
    first = core(inputs_embeds=x, mode=mode)
    altered = x.detach().clone()
    altered[0, 3:] = torch.randn_like(altered[0, 3:]) * 7
    altered[1] = torch.randn_like(altered[1]) * 5
    second = core(inputs_embeds=altered, mode=mode)
    for left, right in zip(first.pass_hidden_states, second.pass_hidden_states):
        torch.testing.assert_close(left[0, :3], right[0, :3], atol=0, rtol=0)
    loss = (first.last_hidden_state[0, 2] * torch.randn(32)).sum()
    derivative = torch.autograd.grad(loss, x)[0]
    assert derivative[0, :3].abs().sum() > 0
    assert derivative[0, 3:].count_nonzero() == 0
    assert derivative[1].count_nonzero() == 0


@pytest.mark.parametrize("rt_mode", [RTMode(()), RTMode((0,), 1.0)])
def test_feedback_adjoint_has_strict_one_token_shift_per_pass(rt_mode):
    core = model()
    x = torch.randn(1, 5, 32, requires_grad=True)
    output = core(inputs_embeds=x, mode=FBTMode(num_passes=3, beta=.8, rt_mode=rt_mode))
    first, second, third = output.pass_hidden_states
    one_step = (second[:, 2] * torch.randn(1, 32)).sum()
    credit = torch.autograd.grad(one_step, first, retain_graph=True)[0]
    assert credit[:, :2].abs().sum() > 0
    assert credit[:, 2:].count_nonzero() == 0  # Same-position state is forbidden.
    two_step = (third[:, 4] * torch.randn(1, 32)).sum()
    first_credit, second_credit = torch.autograd.grad(two_step, (first, second), retain_graph=True)
    assert first_credit[:, :3].abs().sum() > 0
    assert first_credit[:, 3:].count_nonzero() == 0
    assert second_credit[:, :4].abs().sum() > 0
    assert second_credit[:, 4:].count_nonzero() == 0
    # Position zero cannot use any earlier-pass hidden state.
    start_loss = (third[:, 0] * torch.randn(1, 32)).sum()
    start_credit = torch.autograd.grad(start_loss, (first, second), allow_unused=True)
    assert all(value is None or value.count_nonzero() == 0 for value in start_credit)


@pytest.mark.parametrize("endpoint", ["k1", "beta0"])
def test_ordinary_endpoints_preserve_logits_input_and_parameter_gradients(endpoint):
    core = model()
    reference = copy.deepcopy(core.backbone)
    x = torch.randn(2, 4, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    # K1 is ordinary even if the extra-pass RT mode selects a recurrent layer.
    mode = (FBTMode(num_passes=1, beta=1, rt_mode=RTMode((0,), 1)) if endpoint == "k1"
            else FBTMode(num_passes=3, beta=0, rt_mode=RTMode(())))
    actual = core(inputs_embeds=x, mode=mode)
    expected = reference(inputs_embeds=y, mode=RTMode(()))
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    for states in actual.pass_hidden_states:
        torch.testing.assert_close(states, expected.last_hidden_state, rtol=0, atol=0)
    cotangent = torch.randn_like(actual.logits)
    actual_grads = torch.autograd.grad((actual.logits * cotangent).sum(), (x, *core.parameters()), allow_unused=True)
    expected_grads = torch.autograd.grad((expected.logits * cotangent).sum(), (y, *reference.parameters()), allow_unused=True)
    assert_gradient_close(actual_grads[0], expected_grads[0])
    expected_by_name = dict(zip(dict(reference.named_parameters()), expected_grads[1:]))
    for (name, _), gradient in zip(core.named_parameters(), actual_grads[1:]):
        if name.startswith("backbone."):
            assert_gradient_close(gradient, expected_by_name[name.removeprefix("backbone.")])
        else:
            assert gradient is None or gradient.count_nonzero() == 0


def test_three_outstanding_modes_accumulate_as_independent_calls_without_leaf_side_effects():
    core = model()
    reference = copy.deepcopy(core)
    modes = [FBTMode(num_passes=3, beta=.45, rt_mode=RTMode((0,), .37)),
             FBTMode(num_passes=1, beta=1, rt_mode=RTMode((0,), 1)),
             FBTMode(num_passes=2, beta=0, rt_mode=RTMode(()))]
    inputs = [torch.randn(1, length, 32, requires_grad=True) for length in (5, 3, 4)]
    cotangents = [torch.randn(1, x.shape[1], 67) for x in inputs]
    outputs = [core(inputs_embeds=x, mode=mode).logits for x, mode in zip(inputs, modes)]
    loss = sum((output * cotangent).sum() for output, cotangent in zip(outputs, cotangents))
    actual = torch.autograd.grad(loss, (*core.parameters(), *inputs), allow_unused=True)
    expected_parameters = [None] * len(list(reference.parameters()))
    expected_inputs = []
    for x, mode, cotangent in zip(inputs, modes, cotangents):
        isolated = x.detach().clone().requires_grad_()
        out = reference(inputs_embeds=isolated, mode=mode).logits
        derivatives = torch.autograd.grad((out * cotangent).sum(), (*reference.parameters(), isolated), allow_unused=True)
        for i, derivative in enumerate(derivatives[:-1]):
            if derivative is not None:
                expected_parameters[i] = derivative if expected_parameters[i] is None else expected_parameters[i] + derivative
        expected_inputs.append(derivatives[-1])
    for got, wanted in zip(actual, (*expected_parameters, *expected_inputs)):
        assert_gradient_close(got, wanted)
    assert all(parameter.grad is None for parameter in core.parameters())
    assert all(x.grad is None for x in inputs)


@pytest.mark.parametrize("rt_mode", [RTMode(()), RTMode((0,), .37)])
def test_finite_pass_causal_cone_converges_to_online_without_assuming_k2_equivalence(rt_mode):
    core = model()
    tokens = torch.tensor([[2, 7, 4, 11]])
    beta = .7
    short = core(tokens, mode=FBTMode(num_passes=2, beta=beta, rt_mode=rt_mode))
    # T complete passes suffice for a T-token prefix because feedback shifts by
    # one token. Practical finite K generally differs beyond the first K tokens.
    converged = core(tokens, mode=FBTMode(num_passes=4, beta=beta, rt_mode=rt_mode))
    online = core.forward_online(tokens, mode=FBTOnlineMode(beta=beta, rt_mode=rt_mode))
    torch.testing.assert_close(short.logits[:, :2], online.logits[:, :2], rtol=1e-5, atol=3e-6)
    assert (short.logits[:, 2:] - online.logits[:, 2:]).abs().max() > 1e-4
    torch.testing.assert_close(converged.logits, online.logits, rtol=1e-5, atol=3e-6)
    cotangent = torch.randn_like(converged.logits)
    finite_grads = torch.autograd.grad((converged.logits * cotangent).sum(), tuple(core.parameters()), allow_unused=True)
    online_grads = torch.autograd.grad((online.logits * cotangent).sum(), tuple(core.parameters()), allow_unused=True)
    for actual, expected in zip(finite_grads, online_grads):
        if actual is None or expected is None:
            assert actual is expected
            continue
        # Parallel finite passes and token-wise online execution have different
        # FP32 reduction orders. Use O2/O3's existing joint tensor budget for
        # these raw cotangents, rather than a relative budget at near-zero
        # coordinates. The initial coordinate screen flagged one of 1024
        # entries in one tensor; all tensor relative L2 errors were <1.81e-6.
        difference = (actual - expected).double()
        norm, error = expected.double().norm(), difference.norm()
        assert error <= 2e-6 or error / norm.clamp_min(1e-30) <= 1e-4
        assert difference.abs().max() <= 2e-6 + 1e-4 * expected.abs().max()
