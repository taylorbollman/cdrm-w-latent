"""Independent derivative and numerical-interval checks without CUDA."""
import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cdrm_precision_attention_reference import analytic_attention, quantized_interval, validate_capture


def test_analytic_vjp_matches_double_autograd_with_causal_alibi():
    generator = torch.Generator().manual_seed(731)
    q, k, v = [torch.randn(2, 3, 7, 4, dtype=torch.float64, generator=generator).requires_grad_() for _ in range(3)]
    dy = torch.randn(2, 3, 7, 4, dtype=torch.float64, generator=generator)
    position = torch.arange(7)
    bias = -.17 * (position[:, None] - position[None, :]).abs().double()
    bias = bias.masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), float('-inf'))
    out = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(4) + bias, dim=-1) @ v
    gradients = torch.autograd.grad(out, (q, k, v), dy)
    actual = analytic_attention(q, k, v, bias, dy)
    for name, expected in zip(("output", "q_gradient", "k_gradient", "v_gradient"), (out, *gradients)):
        torch.testing.assert_close(actual[name], expected, rtol=1e-12, atol=1e-12)
    # A first-token objective cannot credit future keys or values.
    only_first = torch.zeros_like(dy)
    only_first[..., :1, :] = dy[..., :1, :]
    first = analytic_attention(q, k, v, bias, only_first)
    assert torch.count_nonzero(first['k_gradient'][..., 1:, :]) == 0
    assert torch.count_nonzero(first['v_gradient'][..., 1:, :]) == 0


def test_quantization_interval_allows_boundary_crossing_but_rejects_excess():
    midpoint = 1.0 + 2**-8
    reference = torch.tensor([midpoint - 1e-7], dtype=torch.float64)
    adjacent = torch.tensor([1.0 + 2**-7], dtype=torch.bfloat16)
    assert quantized_interval(reference, adjacent)['interval_pass']
    too_far = torch.tensor([1.0 + 2**-6], dtype=torch.bfloat16)
    result = quantized_interval(reference, too_far)
    assert not result['interval_pass'] and result['interval_failures'] == 1


def test_same_operand_bf16_projection_of_reference_has_zero_interval_failures():
    reference = torch.tensor([-2., -1e-7, 0., .12733, 11.472], dtype=torch.float64)
    assert quantized_interval(reference, reference.bfloat16())['interval_pass']
    assert not quantized_interval(reference, torch.full_like(reference, float('nan')).bfloat16())['interval_pass']
    assert not quantized_interval(torch.full_like(reference, float('nan')), reference.bfloat16())['interval_pass']


def test_capture_rejects_unsupported_dropout_or_missing_causality():
    shape = (2, 3, 7, 4)
    c = {key: torch.ones(shape) for key in ('q','k','v','output','output_gradient','q_gradient','k_gradient','v_gradient')}
    mask = torch.zeros(1,3,7,7).masked_fill(torch.ones(7,7,dtype=torch.bool).triu(1), -1e30)
    c.update(mask=mask, bias_before_cast=mask.clone(), dropout_p=0., is_causal=False)
    validate_capture(c)
    with pytest.raises(ValueError, match='dropout'):
        validate_capture({**c,'dropout_p':.1})
    with pytest.raises(ValueError, match='future'):
        validate_capture({**c,'mask':torch.zeros_like(mask)})
