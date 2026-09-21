"""The FP32 semantic budget must bound both distributed and isolated errors."""

from pathlib import Path
import sys

import torch
from torch import nn

# The inherited Stage A CLI helpers resolve experiment_tracking from scripts/,
# as Python does when launching the validator directly. Match that CLI context.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.openelm_rt_validate import semantic_gradients


def _compare(reference_gradient, actual_gradient):
    reference = nn.Linear(reference_gradient.numel(), 1, bias=False)
    actual = nn.Linear(reference_gradient.numel(), 1, bias=False)
    reference.weight.grad = reference_gradient.reshape_as(reference.weight)
    actual.weight.grad = actual_gradient.reshape_as(actual.weight)
    return semantic_gradients(actual, reference)


def test_cancellation_floor_is_diagnostic_when_tensor_relative_error_is_tiny():
    reference = torch.tensor([100.0, 0.0])
    actual = torch.tensor([100.0, 3e-6])
    result = _compare(reference, actual)
    assert result["passed"]
    assert result["elementwise_failed"] == ["weight"]
    assert result["tensors"]["weight"]["elementwise_violation_count"] == 1


def test_isolated_error_cannot_hide_in_small_global_l2():
    reference = torch.ones(10000)
    actual = reference.clone()
    actual[0] += 0.001
    result = _compare(reference, actual)
    assert result["relative_l2"] < 1e-4
    assert not result["passed"]


def test_distributed_error_cannot_hide_below_maximum_component_budget():
    reference = torch.zeros(10000)
    reference[0] = 1000
    actual = reference + 0.005
    result = _compare(reference, actual)
    row = result["tensors"]["weight"]
    assert row["max_abs"] < row["maximum_error_limit"]
    assert not result["passed"]


def test_nearly_zero_tensor_requires_small_total_absolute_error():
    reference = torch.zeros(100)
    assert not _compare(reference, reference + 1e-6)["passed"]
    assert _compare(reference, reference + 1e-8)["passed"]
