"""CPU checks of validation decisions; no accelerator initialization or model runs."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rt_a5_validate import compare_tensors, finite_state, timing_summary


def test_coordinate_screen_catches_local_error_hidden_by_global_norm():
    reference = torch.ones(10000)
    actual = reference.clone()
    actual[-1] += .001
    result = compare_tensors(reference, actual)
    assert result["relative_l2"] < 2e-5
    assert not result["passed"]
    assert result["mismatched_coordinates"] == 1


def test_zero_reference_and_nonfinite_are_not_silently_accepted():
    assert compare_tensors(torch.zeros(3), torch.zeros(3))["passed"]
    assert not compare_tensors(torch.zeros(3), torch.tensor([0., 0., 1e-3]))["passed"]
    assert not compare_tensors(torch.ones(3), torch.tensor([1., float("nan"), 1.]))["passed"]


def test_timing_reports_assumed_overhead_and_unstable_window():
    result = timing_summary([.1, .1, .05, .05], words=1024, length=12, overhead_fraction=.2)
    assert result["needs_more_warmup_or_timing"]
    assert result["half_window_ratio"] == 2
    assert result["tokens_per_second"] == pytest.approx(12288 / .075)
    estimate = result["estimated_hours"]["10000"]
    assert estimate["with_assumed_eval_checkpoint_overhead"] == pytest.approx(1.2 * estimate["training_only"])


@pytest.mark.parametrize("values", [[0., .1], [float("nan"), .1], [.1]])
def test_invalid_timings_rejected(values):
    with pytest.raises(ValueError):
        timing_summary(values, words=1024, length=12)


def test_state_audit_requires_gradients_and_finite_fp32_moments():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    assert not finite_state(model, optimizer, require_gradients=True)["passed"]
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    healthy = finite_state(model, optimizer, require_gradients=True)
    assert healthy["passed"] and healthy["optimizer_steps"] == [1]
    parameter = next(model.parameters())
    optimizer.state[parameter]["exp_avg"][0, 0] = float("inf")
    assert not finite_state(model, optimizer, require_gradients=True)["passed"]
    optimizer.state[parameter]["exp_avg"].zero_()
    optimizer.state[parameter]["exp_avg_sq"] = optimizer.state[parameter]["exp_avg_sq"].half()
    assert not finite_state(model, optimizer, require_gradients=True)["passed"]
