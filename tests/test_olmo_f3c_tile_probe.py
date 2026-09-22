"""CPU-only checks of the F3c probe's reference math, masks and declared gates."""
import json

import pytest
import torch

from scripts.olmo_f3c_tile_probe import (
    WIDTHS, backward_fixture, eager_backward_tile, fp64_boundary_oracle,
    gradient_comparison, json_safe,
)


@pytest.mark.parametrize("index,width", list(enumerate(WIDTHS)))
def test_eager_backward_tiles_agree_with_independent_boundary_oracle(index, width):
    torch.set_num_threads(1)
    for variant in range(6):
        operands, metadata = backward_fixture(width, index, variant)
        actual = eager_backward_tile(operands)
        expected = fp64_boundary_oracle(operands)
        result = gradient_comparison(actual, expected)
        assert result["passed"], (metadata, result)
        assert all(x.dtype == torch.float32 for x in actual)
        assert all(x.shape == operands[2].shape for x in actual)


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_masks_and_empty_credit_give_exact_zero_historical_gradients(variant):
    operands, _ = backward_fixture(17, 3, variant)
    dkey, dvalue = eager_backward_tile(operands)
    if variant == 1:
        # Historical keys with zero probability receive no gradient; query
        # rows elsewhere may still contribute to the unmasked keys.
        assert torch.count_nonzero(dkey[..., ::3, :]) == 0
        assert torch.count_nonzero(dvalue[..., ::3, :]) == 0
    else:
        assert torch.count_nonzero(dkey) == torch.count_nonzero(dvalue) == 0


def test_strided_fixture_preserves_noncontiguous_native_dtypes():
    operands, metadata = backward_fixture(32, 4, 1)
    assert metadata["target_width"] != metadata["width"]
    assert all(x.stride()[-1] == 2 for x in operands)
    assert [x.dtype for x in operands] == [torch.float32, torch.float32, torch.bfloat16, torch.bfloat16, torch.float32]
    contiguous = tuple(x.contiguous() for x in operands)
    for a, b in zip(eager_backward_tile(operands), eager_backward_tile(contiguous)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_global_gradient_screen_and_stricter_diagnostics_are_distinct():
    reference = (torch.ones(100), torch.ones(100))
    candidate = tuple(x+.001 for x in reference)
    result = gradient_comparison(candidate, reference)
    assert result["passed"]
    assert result["global_gradient_relative_l2"] == pytest.approx(.001, rel=.001)
    assert all(not row["passed"] for row in result["strict_diagnostics"].values())
    # Each tensor clears 1/32, yet the global 1/64 screen correctly fails.
    result = gradient_comparison(tuple(x+.02 for x in reference), reference)
    assert all(row["passed"] for row in result["gradients"].values())
    assert not result["passed"]


def test_zero_reference_and_nonfinite_results_cannot_pass():
    zero = (torch.zeros(3), torch.zeros(3))
    assert gradient_comparison(zero, zero)["passed"]
    assert not gradient_comparison((torch.ones(3), torch.zeros(3)), zero)["passed"]
    assert not gradient_comparison((torch.full((3,), 1e-40), torch.zeros(3)), zero)["passed"]
    bad = gradient_comparison((torch.tensor([float("nan")]), torch.ones(1)), (torch.ones(1), torch.ones(1)))
    assert not bad["passed"]
    encoded = json.dumps(json_safe(bad), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["global_gradient_relative_l2"] is None
    assert decoded["gradients"]["dkey"]["finite"] is False


def test_deterministic_fixtures_do_not_change_global_rng_or_initialize_cuda():
    before = torch.random.get_rng_state().clone()
    first, _ = backward_fixture(64, 5, 4)
    second, _ = backward_fixture(64, 5, 4)
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not torch.cuda.is_initialized()
