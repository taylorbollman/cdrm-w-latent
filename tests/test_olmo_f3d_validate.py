"""CPU-only reference-zero and fused-dispatch accounting for native F3c checks."""

import math

import pytest
import torch

from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo_fbt import FBTMode
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained import olmo_rt_recompute_kernels
from scripts import olmo_f3d_validate as validation


@pytest.mark.parametrize("rows", [[], [{"delta_sq": 0.0, "reference_sq": 0.0}],
    [{"delta_sq": 0.0, "reference_sq": 0.0}, {"delta_sq": 0.0, "reference_sq": 0.0}]])
def test_global_zero_reference_and_zero_error_are_exactly_zero(rows):
    assert validation.global_gradient_l2(iter(rows)) == 0.0


@pytest.mark.parametrize("error", [1.0, 1e-80])
def test_nonzero_error_against_zero_global_reference_cannot_pass(error):
    value = validation.global_gradient_l2([{"delta_sq": error, "reference_sq": 0.0}])
    assert math.isinf(value) and value > 0


def test_global_gradient_norm_uses_weighted_squared_totals_not_average_tensor_ratios():
    # sqrt((.09+.16)/(9+16)) = .1; tensor ratios would coincide here only
    # accidentally, so a second asymmetric example rules out simple averaging.
    rows = [{"delta_sq": .09, "reference_sq": 9.0},
            {"delta_sq": .16, "reference_sq": 16.0}]
    assert validation.global_gradient_l2(iter(rows)) == pytest.approx(.1)
    rows = [{"delta_sq": 1.0, "reference_sq": 1.0},
            {"delta_sq": 0.0, "reference_sq": 99.0},
            {"delta_sq": 0.0, "reference_sq": 0.0}]
    assert validation.global_gradient_l2(iter(rows)) == pytest.approx(.1)


def batch(length, padded=False):
    ids = torch.full((2, length), 2, dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    if padded:
        valid[1, length // 2:] = False
    documents = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    return NextLatBatch(ids, valid, documents)


@pytest.mark.parametrize("length,enabled,passes,layers,expected", [
    (512, False, 2, (0,), 511),
    (512, True, 2, (0,), 511),
    (512, True, 3, (0,), 1022),
    (32, True, 3, (0, 15), 124),
    (32, False, 4, (0, 15), 62),
    (32, True, 1, (0, 15), 0),
    (32, True, 3, (), 0),
    (32, False, 2, (), 0),
    (1, True, 3, (0, 15), 0),
])
def test_expected_fused_count_matches_actual_rt_pass_selection(length, enabled, passes, layers, expected):
    mode = FBTMode(enabled=enabled, num_passes=passes, rt_mode=RTMode(layers))
    assert validation.expected_fused_backward_calls(batch(length), mode) == expected


def test_expected_count_uses_physical_length_including_padded_loop_positions():
    mode = FBTMode(rt_mode=RTMode((0,)))
    assert validation.expected_fused_backward_calls(batch(17, padded=True), mode) == 16


def test_counter_observes_kernel_entry_without_any_cuda_execution(monkeypatch):
    seen = []
    sentinel = (object(), object())
    def fake(*args, **kwargs):
        seen.append((args, kwargs))
        return sentinel
    monkeypatch.setattr(olmo_rt_recompute_kernels, "backward_recomputed_tile", fake)
    with validation.count_backward_tiles() as count:
        assert count == {"count": 0}
        assert olmo_rt_recompute_kernels.backward_recomputed_tile("metadata only") is sentinel
        assert olmo_rt_recompute_kernels.backward_recomputed_tile("second", flag=True) is sentinel
        assert count == {"count": 2}
    assert seen == [(("metadata only",), {}), (("second",), {"flag": True})]
    assert olmo_rt_recompute_kernels.backward_recomputed_tile is fake


def test_counter_restores_original_callable_even_when_kernel_raises(monkeypatch):
    class DiagnosticFailure(Exception):
        pass
    def fake(*args, **kwargs):
        raise DiagnosticFailure("synthetic CPU failure")
    monkeypatch.setattr(olmo_rt_recompute_kernels, "backward_recomputed_tile", fake)
    with pytest.raises(DiagnosticFailure):
        with validation.count_backward_tiles() as count:
            olmo_rt_recompute_kernels.backward_recomputed_tile()
    assert count == {"count": 1}
    assert olmo_rt_recompute_kernels.backward_recomputed_tile is fake
