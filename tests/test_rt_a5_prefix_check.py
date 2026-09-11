"""Decision checks for a bounded prefix evaluation, with CPU fixtures only."""
import pytest
import torch

from scripts.rt_a5_prefix_check import compare_prefix


def test_same_counts_cannot_hide_changed_predictions():
    ref = torch.zeros(2, 2, 60)
    ref[..., 1] = 1
    got = ref.clone()
    got[..., 1], got[..., 2] = 0, 1
    labels = torch.zeros(2, 2, dtype=torch.long)
    result = compare_prefix(ref, got, labels)
    assert result["accuracy_counts_equal"]
    assert result["prediction_disagreements"] == 4
    assert not result["passed"]


def test_harmless_roundoff_passes_and_counts_remain_integer():
    ref = torch.zeros(2, 2, 60)
    ref[..., 0] = 1
    labels = torch.tensor([[0, 0], [0, 1]])
    result = compare_prefix(ref, ref + 1e-7, labels)
    assert result["passed"]
    assert result["truncated_metrics"]["state_correct_counts"] == [2, 1]
    assert result["truncated_metrics"]["prefix_correct_counts"] == [2, 1]


def test_zero_rounded_percentage_is_not_substituted_for_counts():
    ref = torch.zeros(1001, 1, 60)
    ref[..., 0] = 1
    labels = torch.ones(1001, 1, dtype=torch.long)
    labels[0] = 0
    result = compare_prefix(ref, ref, labels)
    assert result["truncated_metrics"]["prefix_correct_counts"] == [1]


def test_invalid_shapes_and_nonfinite_logits_rejected():
    ref = torch.zeros(1, 2, 60)
    labels = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError):
        compare_prefix(ref, ref[:, :1], labels)
    bad = ref.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        compare_prefix(ref, bad, labels)
    with pytest.raises(TypeError):
        compare_prefix(ref.half(), ref.half(), labels)
