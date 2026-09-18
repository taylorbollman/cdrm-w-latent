"""Independent two-record scan and temporal-credit checks for the RT window."""
import copy

import pytest
import torch

from olmo.model import alibi_attention_bias, block_attention_add, recurrent_helper
from scripts.rt_a5_common import configure_fp32_runtime, fp32_context
from scripts.rt_a5_window import build_window_model
from scripts.rt_a5_window_reference import DirectWindow2RecurrentBlock, direct_window2


@pytest.fixture(autouse=True)
def cpu_only_fp32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU oracle tests attempted CUDA"))
    torch.set_num_threads(1)
    configure_fp32_runtime()


@pytest.mark.parametrize("length", [1, 2, 3, 12])
@pytest.mark.parametrize("positions", ["alibi", "sinusoidal"])
def test_masked_naive_matches_physical_two_record_scan_and_all_gradients(length, positions):
    model = build_window_model(width=128, backend="naive", position_encoding=positions,
                               second_layer_window=2)
    native = model.backbone.transformer.blocks[1]
    oracle = copy.deepcopy(native)
    oracle.__class__ = DirectWindow2RecurrentBlock
    assert list(native.state_dict()) == list(oracle.state_dict())
    generator = torch.Generator().manual_seed(317 + length)
    x = torch.randn(2, length, 128, generator=generator).requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    cotangent = torch.randn(x.shape, generator=generator)
    bias = (alibi_attention_bias(length, native.config, x.device)
            if positions == "alibi" else None)
    with fp32_context("cpu"):
        actual = native(x, attention_bias=bias)[0]
        expected = oracle(reference_x, attention_bias=bias)[0]
        (actual * cotangent).sum().backward()
        (expected * cotangent).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=2e-5, atol=5e-6)
    for name, parameter in native.named_parameters():
        other = dict(oracle.named_parameters())[name]
        assert parameter.grad is not None and other.grad is not None, name
        assert torch.isfinite(parameter.grad).all() and torch.isfinite(other.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-5, atol=2e-5, msg=name)


@pytest.mark.parametrize("length", [1, 2])
def test_one_and_two_tokens_match_original_full_rt(length):
    full = build_window_model(width=128, backend="naive", position_encoding="alibi")
    window = build_window_model(width=128, backend="naive", position_encoding="alibi", second_layer_window=2)
    tokens = torch.arange(2 * length).reshape(2, length)
    with fp32_context("cpu"):
        torch.testing.assert_close(full(tokens).logits, window(tokens).logits, rtol=0, atol=0)


def test_previous_permanent_write_retains_multistep_credit_but_terminal_write_is_unused():
    model = build_window_model(width=128, backend="naive", position_encoding="sinusoidal",
                               second_layer_window=2)
    block = model.backbone.transformer.blocks[1]
    generator = torch.Generator().manual_seed(831)
    x = torch.randn(2, 12, 128, generator=generator).requires_grad_()
    trace = {}
    with fp32_context("cpu"):
        output = direct_window2(block, x, trace=trace)
        last_cotangent = torch.randn(2, 128, generator=generator)
        (output[:, -1] * last_cotangent).sum().backward()
    assert trace["key_indices"] == [[0]] + [[position - 1, position] for position in range(1, 12)]
    assert [weights.shape[-1] for weights in trace["probabilities"]] == [1] + [2] * 11
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[:, 0]).item() > 0
    for name in ("permanent_keys", "permanent_values"):
        assert trace[name][-1].grad is None
        for record in trace[name][:-1]:
            assert record.grad is not None and torch.isfinite(record.grad).all()
            assert torch.count_nonzero(record.grad).item() > 0


def test_all_forbidden_cross_tile_rows_preserve_finite_self_state_exactly():
    block = build_window_model(width=128, backend="naive").backbone.transformer.blocks[1]
    generator = torch.Generator().manual_seed(93)
    initial_attention = torch.randn(4, 2, 2, 64, generator=generator)
    initial_maximum = torch.randn(4, 2, 2, generator=generator)
    initial_denominator = torch.ones(4, 2, 2)
    q, k, v = [torch.randn(2, 2, 4, 64, generator=generator) for _ in range(3)]
    with fp32_context("cpu"):
        actual = recurrent_helper(block, block_attention_add, initial_attention, initial_maximum,
                                  initial_denominator, k, v, q, torch.full((1, 1, 4, 4), -torch.inf))
    for computed, original in zip(actual, (initial_attention, initial_maximum, initial_denominator)):
        assert torch.isfinite(computed).all()
        assert torch.equal(computed, original)
