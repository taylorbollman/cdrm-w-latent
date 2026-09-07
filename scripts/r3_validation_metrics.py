"""Strict elementwise and tensor-scale diagnostics for recurrent backward audits.

This module does not change tensors or choose an optimizer acceptance policy.
All reductions occur on detached CPU FP64 values. The original elementwise
criterion and the fixed tensor-scale diagnostic are reported independently.
"""
from __future__ import annotations

import math
from typing import Any

import torch

DEFAULT_ATOL = 2e-6
DEFAULT_RTOL = 2e-5
NORM_FLOOR = 1e-12
SCALE_AWARE_TOLERANCE = 2e-5
NEAR_ZERO_RMS_FRACTION = 1e-3


def _number(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _validate_tensor(value, name):
    if value is not None and not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if value is not None and value.is_complex():
        raise TypeError(f"{name} must be real-valued")
    if value is not None and value.layout != torch.strided:
        raise TypeError(f"{name} must be a dense strided tensor")


def _tensor_stats(tensor: torch.Tensor | None, prefix: str) -> tuple[dict, torch.Tensor | None]:
    if tensor is None:
        return {f"{prefix}_shape": None, f"{prefix}_dtype": None, f"{prefix}_device": None,
                f"{prefix}_numel": None, f"{prefix}_finite": None,
                f"{prefix}_nonfinite_count": None, f"{prefix}_l2": None,
                f"{prefix}_rms": None, f"{prefix}_max_abs": None}, None
    values = tensor.detach().to(device="cpu", dtype=torch.float64)
    finite = torch.isfinite(values)
    is_finite = bool(finite.all().item())
    count = values.numel()
    # For nonfinite tensors, do not report a norm of just the finite subset.
    l2 = _number(torch.linalg.vector_norm(values)) if is_finite else None
    rms = _number(l2 / math.sqrt(count)) if l2 is not None and count else (0.0 if count == 0 else None)
    maximum = _number(values.abs().max()) if count and is_finite else (0.0 if count == 0 else None)
    return {f"{prefix}_shape": list(tensor.shape), f"{prefix}_dtype": str(tensor.dtype),
            f"{prefix}_device": str(tensor.device), f"{prefix}_numel": count,
            f"{prefix}_finite": is_finite,
            f"{prefix}_nonfinite_count": count - int(finite.sum().item()),
            f"{prefix}_l2": l2, f"{prefix}_rms": rms, f"{prefix}_max_abs": maximum}, values


def _index(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    result = []
    for width in reversed(shape):
        result.append(flat_index % width)
        flat_index //= width
    return list(reversed(result))


def _coordinate(flat_index, reference, actual, errors, tolerances, violation_ratios):
    return {"flat_index": int(flat_index), "index": _index(int(flat_index), tuple(reference.shape)),
            "reference": _number(reference.reshape(-1)[flat_index]),
            "actual": _number(actual.reshape(-1)[flat_index]),
            "absolute_error": _number(errors.reshape(-1)[flat_index]),
            "elementwise_tolerance": _number(tolerances.reshape(-1)[flat_index]),
            "error_over_tolerance": _number(violation_ratios.reshape(-1)[flat_index])}


def _subset(mask, errors, elementwise_failures):
    count = int(mask.sum().item())
    return {"count": count, "max_abs_error": _number(errors[mask].max()) if count else None,
            "elementwise_failure_count": int((mask & elementwise_failures).sum().item())}


def compare_tensors(reference: torch.Tensor | None, actual: torch.Tensor | None,
                    atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL,
                    norm_floor: float = NORM_FLOOR) -> dict[str, Any]:
    """Return JSON-safe diagnostics without broadcasting, rescaling, or mutation.

    ``elementwise_pass`` retains ``|actual-reference| <= atol + rtol*|reference|``.
    ``scale_aware_pass`` separately requires both relative L2 error and maximum
    error/reference RMS <= 2e-5. Its thresholds stay fixed even if a caller
    overrides the elementwise tolerances. Ratio denominators are
    ``max(reference L2, norm_floor)`` and ``max(reference RMS, norm_floor)``.

    None is missing, not a zero gradient. Both missing also fail both numerical
    flags: agreement about absence is not a validated intended parameter gradient.
    Nonfinite values, shape mismatches and missing values fail both flags and
    produce JSON null for unavailable/nonfinite diagnostics. Near-zero reference
    entries include exact zeros and satisfy abs(ref) <= 1e-3 * reference RMS;
    exact-zero and nonzero-near-zero subsets are also reported separately.
    """
    for name, value in (("atol", atol), ("rtol", rtol), ("norm_floor", norm_floor)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
        if value < 0 or (name == "norm_floor" and value == 0):
            raise ValueError(f"{name} must be {'positive' if name == 'norm_floor' else 'nonnegative'}")
    _validate_tensor(reference, "reference")
    _validate_tensor(actual, "actual")
    reference_stats, ref = _tensor_stats(reference, "ref")
    actual_stats, act = _tensor_stats(actual, "actual")
    report = {**reference_stats, **actual_stats,
        "reference_present": reference is not None, "actual_present": actual is not None,
        "presence_match": (reference is None) == (actual is None),
        "shape_match": reference.shape == actual.shape if reference is not None and actual is not None else None,
        "atol": float(atol), "rtol": float(rtol), "norm_floor": float(norm_floor),
        "reduction_dtype": "torch.float64", "reduction_device": "cpu",
        "scale_aware_rel_l2_threshold": SCALE_AWARE_TOLERANCE,
        "scale_aware_maxerr_over_ref_rms_threshold": SCALE_AWARE_TOLERANCE,
        "near_zero_ref_rms_fraction": NEAR_ZERO_RMS_FRACTION,
        "status": "uncompared", "finite": False,
        "elementwise_pass": False, "scale_aware_pass": False,
        "elementwise_failure_count": None, "elementwise_failure_fraction": None,
        "error_l2": None, "error_rms": None, "max_abs_error": None,
        "rel_l2": None, "maxerr_over_ref_rms": None,
        "max_error_over_elementwise_tolerance": None,
        "worst_coordinate": None, "worst_elementwise_coordinate": None,
        "top5_offending_coordinates": [], "near_zero_ref_threshold": None,
        "near_zero_ref": None, "exact_zero_ref": None, "near_zero_nonzero_ref": None}
    if ref is None or act is None:
        report["status"] = "missing_both" if ref is None and act is None else "presence_mismatch"
        return report
    if ref.shape != act.shape:
        report["status"] = "shape_mismatch"
        return report
    if not report["ref_finite"] or not report["actual_finite"]:
        report["status"] = "nonfinite_input"
        return report
    errors = (act - ref).abs()
    count = ref.numel()
    report["error_l2"] = _number(torch.linalg.vector_norm(errors))
    report["error_rms"] = _number(report["error_l2"] / math.sqrt(count)) if count and report["error_l2"] is not None else (0.0 if count == 0 else None)
    report["max_abs_error"] = _number(errors.max()) if count else 0.0
    necessary = (report["ref_l2"], report["ref_rms"], report["actual_l2"],
                 report["actual_rms"], report["error_l2"], report["max_abs_error"])
    if any(value is None for value in necessary):
        report["status"] = "nonfinite_reduction"
        return report
    report["rel_l2"] = _number(report["error_l2"] / max(report["ref_l2"], norm_floor))
    report["maxerr_over_ref_rms"] = _number(report["max_abs_error"] / max(report["ref_rms"], norm_floor))
    if report["rel_l2"] is None or report["maxerr_over_ref_rms"] is None:
        report["status"] = "nonfinite_ratio"
        return report
    tolerances = atol + rtol * ref.abs()
    if not bool(torch.isfinite(tolerances).all().item()):
        report["status"] = "nonfinite_tolerance"
        return report
    failures = errors > tolerances
    # Zero tolerance has ratio zero for equality and +inf for a nonzero error.
    # Infinities are used only to rank failures; JSON fields contain null instead.
    ratios = torch.where(tolerances > 0, errors / tolerances,
                         torch.where(errors == 0, 0.0, float("inf")))
    report.update(status="compared", finite=True,
                  elementwise_failure_count=int(failures.sum().item()),
                  elementwise_failure_fraction=float(failures.sum().item()) / count if count else 0.0,
                  elementwise_pass=not bool(failures.any().item()),
                  scale_aware_pass=(report["rel_l2"] <= SCALE_AWARE_TOLERANCE and
                                    report["maxerr_over_ref_rms"] <= SCALE_AWARE_TOLERANCE),
                  max_error_over_elementwise_tolerance=_number(ratios.max()) if count else 0.0)
    if count:
        report["worst_coordinate"] = _coordinate(int(errors.reshape(-1).argmax().item()), ref, act, errors, tolerances, ratios)
        report["worst_elementwise_coordinate"] = _coordinate(int(ratios.reshape(-1).argmax().item()), ref, act, errors, tolerances, ratios)
        offender_indices = failures.reshape(-1).nonzero(as_tuple=False).reshape(-1)
        if len(offender_indices):
            # Stable ties preserve flat-index order for reproducible JSON output.
            ranking = torch.argsort(ratios.reshape(-1)[offender_indices], descending=True, stable=True)[:5]
            report["top5_offending_coordinates"] = [
                _coordinate(int(offender_indices[rank].item()), ref, act, errors, tolerances, ratios)
                for rank in ranking]
    near_threshold = NEAR_ZERO_RMS_FRACTION * report["ref_rms"]
    near_mask = ref.abs() <= near_threshold
    zero_mask = ref == 0
    report.update(near_zero_ref_threshold=near_threshold,
                  near_zero_ref=_subset(near_mask, errors, failures),
                  exact_zero_ref=_subset(zero_mask, errors, failures),
                  near_zero_nonzero_ref=_subset(near_mask & ~zero_mask, errors, failures))
    return report
