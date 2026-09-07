"""CPU-only checks of diagnostic criteria and numerical reporting semantics."""
import json
import math

import pytest
import torch

from scripts.r3_validation_metrics import compare_tensors


def assert_json_safe(report):
    json.dumps(report, allow_nan=False)


def test_matches_original_asymmetric_elementwise_tolerance():
    reference = torch.tensor([0., 1., -1.], dtype=torch.float64)
    tolerance = 2e-6 + 2e-5 * reference.abs()
    inside = compare_tensors(reference, reference + tolerance * 0.999)
    outside = compare_tensors(reference, reference + tolerance * 1.001)
    assert inside["elementwise_pass"]
    assert not outside["elementwise_pass"]
    assert outside["elementwise_failure_count"] == 3
    assert outside["ref_l2"] == pytest.approx(math.sqrt(2))
    assert outside["ref_rms"] == pytest.approx(math.sqrt(2 / 3))
    assert outside["ref_max_abs"] == 1
    assert_json_safe(outside)


def test_scale_diagnostic_invariance_and_original_absolute_floor_are_separate():
    reference = torch.tensor([0., 1., 2., 3.], dtype=torch.float64)
    actual = reference + torch.tensor([1e-5, 0., 0., 0.], dtype=torch.float64)
    reports = [compare_tensors(reference * scale, actual * scale) for scale in (1 / 32, 1, 32)]
    assert [report["elementwise_pass"] for report in reports] == [True, False, False]
    assert all(report["scale_aware_pass"] for report in reports)
    for report in reports[1:]:
        assert report["rel_l2"] == pytest.approx(reports[0]["rel_l2"], rel=1e-12)
        assert report["maxerr_over_ref_rms"] == pytest.approx(reports[0]["maxerr_over_ref_rms"], rel=1e-12)
    # Custom elementwise thresholds cannot silently change the fixed scale test.
    bad = compare_tensors(reference, reference + 1e-3, atol=10., rtol=10.)
    assert bad["elementwise_pass"] and not bad["scale_aware_pass"]
    assert bad["scale_aware_rel_l2_threshold"] == 2e-5


def test_reports_near_zero_and_exact_zero_separately():
    reference = torch.tensor([[0., 1e-5, 1.], [2., -1e-5, -3.]], dtype=torch.float64)
    actual = reference.clone()
    actual[0, 0] = 8e-6
    actual[0, 1] += 1e-5
    actual[1, 1] += 5e-6
    report = compare_tensors(reference, actual)
    assert report["near_zero_ref"]["count"] == 3
    assert report["exact_zero_ref"]["count"] == 1
    assert report["near_zero_nonzero_ref"]["count"] == 2
    assert report["near_zero_ref"]["max_abs_error"] == pytest.approx(1e-5)
    assert report["exact_zero_ref"]["max_abs_error"] == pytest.approx(8e-6)
    assert report["near_zero_ref"]["elementwise_failure_count"] == 3
    assert report["worst_coordinate"]["index"] == [0, 1]
    assert report["top5_offending_coordinates"][0]["index"] == [0, 1]
    assert_json_safe(report)


def test_worst_absolute_error_differs_from_worst_relative_tolerance_violation():
    reference = torch.tensor([0., 1000.], dtype=torch.float64)
    actual = reference + torch.tensor([1e-5, 0.03], dtype=torch.float64)
    report = compare_tensors(reference, actual)
    assert report["worst_coordinate"]["index"] == [1]
    assert report["worst_elementwise_coordinate"]["index"] == [0]
    assert report["top5_offending_coordinates"][0]["index"] == [0]


def test_top_five_violations_are_ranked_and_ties_are_stable():
    reference = torch.zeros(2, 4, dtype=torch.float64)
    actual = torch.tensor([[8., 7., 6., 5.], [4., 3., 2., 8.]], dtype=torch.float64)
    report = compare_tensors(reference, actual)
    assert report["elementwise_failure_count"] == 8
    assert [item["flat_index"] for item in report["top5_offending_coordinates"]] == [0, 7, 1, 2, 3]


@pytest.mark.parametrize("reference,actual,status", [
    (None, torch.zeros(3), "presence_mismatch"),
    (torch.zeros(3), None, "presence_mismatch"),
    (None, None, "missing_both"),
])
def test_missing_gradient_is_never_silently_replaced_by_zero(reference, actual, status):
    report = compare_tensors(reference, actual)
    assert report["status"] == status
    assert not report["elementwise_pass"] and not report["scale_aware_pass"]
    assert report["error_l2"] is None and report["rel_l2"] is None
    assert report["presence_match"] == (reference is None and actual is None)
    assert_json_safe(report)


def test_exact_shape_required_without_broadcasting():
    report = compare_tensors(torch.ones(2, 1), torch.ones(2))
    assert report["status"] == "shape_mismatch"
    assert report["ref_shape"] == [2, 1] and report["actual_shape"] == [2]
    assert report["shape_match"] is False
    assert not report["elementwise_pass"] and not report["scale_aware_pass"]
    assert report["max_abs_error"] is None
    assert_json_safe(report)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_fails_even_when_both_inputs_match(value):
    reference = torch.tensor([1., value], dtype=torch.float64)
    report = compare_tensors(reference, reference.clone())
    assert report["status"] == "nonfinite_input"
    assert report["ref_nonfinite_count"] == report["actual_nonfinite_count"] == 1
    assert not report["finite"] and not report["elementwise_pass"] and not report["scale_aware_pass"]
    assert report["ref_l2"] is None and report["worst_coordinate"] is None
    assert_json_safe(report)


def test_zero_reference_uses_documented_floor_without_hiding_small_errors():
    reference = torch.zeros(4, dtype=torch.float64)
    report = compare_tensors(reference, reference + 1e-8)
    assert report["elementwise_pass"] and not report["scale_aware_pass"]
    assert report["rel_l2"] == pytest.approx(2e4)
    assert report["maxerr_over_ref_rms"] == pytest.approx(1e4)
    assert report["exact_zero_ref"]["count"] == 4
    perfect = compare_tensors(reference, reference)
    assert perfect["elementwise_pass"] and perfect["scale_aware_pass"]
    assert perfect["rel_l2"] == perfect["maxerr_over_ref_rms"] == 0


def test_constant_update_error_can_be_hidden_by_large_post_step_weights():
    weights = torch.full((8,), 1000., dtype=torch.float64)
    reference_update = torch.full((8,), 1e-3, dtype=torch.float64)
    actual_update = reference_update + 1e-4
    post_step = compare_tensors(weights + reference_update, weights + actual_update)
    updates = compare_tensors(reference_update, actual_update)
    assert post_step["elementwise_pass"] and post_step["scale_aware_pass"]
    assert not updates["elementwise_pass"] and not updates["scale_aware_pass"]
    assert updates["rel_l2"] == pytest.approx(0.1)
    assert updates["maxerr_over_ref_rms"] == pytest.approx(0.1)


def test_scalar_empty_and_noncontiguous_inputs_are_json_safe_and_unchanged():
    reference = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    before = reference.clone()
    report = compare_tensors(reference, reference.clone())
    assert torch.equal(reference, before)
    assert report["elementwise_pass"] and report["scale_aware_pass"]
    scalar = compare_tensors(torch.tensor(1.), torch.tensor(1.1))
    assert scalar["ref_shape"] == [] and scalar["worst_coordinate"]["index"] == []
    empty = compare_tensors(torch.empty(0, 2), torch.empty(0, 2))
    assert empty["elementwise_pass"] and empty["scale_aware_pass"]
    assert empty["ref_rms"] == 0 and empty["worst_coordinate"] is None
    assert_json_safe(report)
    assert_json_safe(scalar)
    assert_json_safe(empty)


def test_zero_tolerance_failure_has_json_null_ratio_without_changing_failure():
    report = compare_tensors(torch.zeros(2), torch.tensor([0., 1.]), atol=0., rtol=0.)
    assert not report["elementwise_pass"]
    assert report["elementwise_failure_count"] == 1
    assert report["top5_offending_coordinates"][0]["error_over_tolerance"] is None
    assert_json_safe(report)


@pytest.mark.parametrize("kwargs", [{"norm_floor": 0.}, {"norm_floor": float("nan")},
                                    {"atol": -1.}, {"rtol": float("inf")}, {"rtol": True}])
def test_invalid_tolerances_raise(kwargs):
    with pytest.raises(ValueError):
        compare_tensors(torch.ones(1), torch.ones(1), **kwargs)


def test_subtraction_and_norms_promote_before_fp32_overflow():
    reference = torch.tensor([3e38, -3e38], dtype=torch.float32)
    actual = -reference
    report = compare_tensors(reference, actual)
    assert report["finite"] and report["status"] == "compared"
    assert report["ref_l2"] == pytest.approx(math.sqrt(2) * float(reference[0]), rel=1e-15)
    assert report["max_abs_error"] == 2 * float(reference[0])
    assert report["rel_l2"] == pytest.approx(2.)
    assert not report["elementwise_pass"] and not report["scale_aware_pass"]
    assert_json_safe(report)
