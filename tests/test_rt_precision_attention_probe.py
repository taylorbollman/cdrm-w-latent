"""CPU mathematical checks for the independent conditional attention oracle."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_precision_attention_probe import ADJOINTS, OPERANDS, causal_attention, conditional_vjp, error_metrics


def fixture(length=4):
    generator = torch.Generator().manual_seed(812)
    operands = {name: torch.randn(length, 2, 2, 3, generator=generator, dtype=torch.float64)
                for name in OPERANDS}
    # The fixture stores already-scaled queries, exactly as the production probe.
    operands['q'] *= .25
    position = torch.arange(length, dtype=torch.float64)
    operands['attention_bias'] = (position[None, :] - position[:, None])[None, None] * .1
    return {'operands': operands, 'g': torch.randn(length, 2, 2, 3, generator=generator, dtype=torch.float64)}


def direct_token_attention(values, bias):
    q, k_init, v_init, permanent_k, permanent_v = values
    result = []
    for t in range(len(q)):
        keys = torch.cat((permanent_k[:t], k_init[t:t + 1]), dim=0)
        vals = torch.cat((permanent_v[:t], v_init[t:t + 1]), dim=0)
        logits = (keys * q[t:t + 1]).sum(-1).permute(1, 2, 0)
        weights = (logits + bias[:, :, t, :t + 1]).softmax(-1)
        result.append((weights.permute(2, 0, 1).unsqueeze(-1) * vals).sum(0))
    return torch.stack(result)


def test_matrix_oracle_matches_direct_provisional_self_attention_and_all_vjps():
    packet = fixture()
    values = [packet['operands'][name].clone().requires_grad_(True) for name in OPERANDS]
    expected = direct_token_attention(values, packet['operands']['attention_bias'])
    expected_gradients = torch.autograd.grad(expected, values, grad_outputs=packet['g'])
    actual = conditional_vjp(packet, torch.float64, 'cpu')
    torch.testing.assert_close(actual['attention'], expected, rtol=1e-12, atol=1e-12)
    for name, expected_gradient in zip(ADJOINTS, expected_gradients):
        torch.testing.assert_close(actual['adjoints'][name], expected_gradient, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(actual['adjoints']['permanent_k_grad'][-1]) == 0
    assert torch.count_nonzero(actual['adjoints']['permanent_v_grad'][-1]) == 0


def test_first_query_is_exactly_provisional_value_with_zero_qk_and_permanent_credit():
    packet = fixture(length=1)
    actual = conditional_vjp(packet, torch.float64, 'cpu')
    assert torch.equal(actual['attention'], packet['operands']['v_init'])
    assert torch.equal(actual['adjoints']['v_init_grad'], packet['g'])
    for name in ('q_grad', 'k_init_grad', 'permanent_k_grad', 'permanent_v_grad'):
        assert torch.count_nonzero(actual['adjoints'][name]) == 0


def test_independent_attention_derivatives_pass_finite_difference_gradcheck():
    generator = torch.Generator().manual_seed(42)
    values = tuple(torch.randn(3, 1, 1, 2, generator=generator, dtype=torch.float64,
                               requires_grad=True) for _ in OPERANDS)
    assert torch.autograd.gradcheck(lambda *args: causal_attention(*args)[0], values,
                                    eps=1e-6, atol=1e-5, rtol=1e-4)


def test_same_bf16_rounded_operands_preserve_fp32_reference_and_zero_error_reporting():
    packet = fixture()
    packet['operands'] = {name: tensor.bfloat16() for name, tensor in packet['operands'].items()}
    packet['g'] = packet['g'].bfloat16()
    reference = conditional_vjp(packet, torch.float64, 'cpu')
    candidate = conditional_vjp(packet, torch.float32, 'cpu')
    for name in ADJOINTS:
        row = error_metrics(reference['adjoints'][name], candidate['adjoints'][name])
        assert row['relative_l2'] < 1e-5
    undefined = error_metrics(torch.zeros(3), torch.ones(3))
    assert undefined['relative_l2'] is None and undefined['nonzero_actual_at_zero_reference'] == 3
    assert error_metrics(torch.zeros(3), torch.zeros(3))['relative_l2'] == 0
    with pytest.raises(FloatingPointError):
        error_metrics(torch.ones(1), torch.tensor([float('nan')]))
