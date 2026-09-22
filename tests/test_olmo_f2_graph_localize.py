"""CPU checks for retained graph-localization comparisons, not CUDA readiness."""
import copy

import pytest
import torch

from scripts.olmo_f2_graph_localize import backend_context, compare_snapshots, snapshot


def reference():
    return {"hidden": torch.ones((1, 3, 4)), "gradients": {
        "transformer.wte.weight": torch.ones((7, 4)),
        "transformer.blocks.0.att_proj.weight": torch.ones((4, 4)),
        "transformer.blocks.11.att_proj.weight": torch.ones((4, 4)),
        "transformer.blocks.15.ff_out.weight": torch.ones((4, 4)),
    }}


def test_identical_snapshots_preserve_exactness_and_original_budgets():
    expected = reference()
    row = compare_snapshots(copy.deepcopy(expected), expected, name="repeat")
    assert row["all_bitwise_equal"] and row["passed"]
    assert row["max_gradient_relative_l2"] == 0
    assert row["highest_layer_with_any_gradient_difference"] is None
    assert all(value["relative_l2_limit"] == 1e-5 for value in row["gradients"].values())


def test_localization_finds_first_backward_layer_not_largest_error_layer():
    expected, actual = reference(), reference()
    actual["gradients"]["transformer.blocks.0.att_proj.weight"][0, 0] += 1
    actual["gradients"]["transformer.blocks.11.att_proj.weight"][0, 0] += .01
    row = compare_snapshots(actual, expected, name="capture")
    assert row["hidden"]["bitwise_equal"]
    assert not row["all_bitwise_equal"] and not row["passed"]
    assert row["highest_layer_with_any_gradient_difference"] == 11
    assert row["highest_layer_outside_original_budget"] == 11
    assert row["layer_max_gradient_relative_l2"]["0"] > row["layer_max_gradient_relative_l2"]["11"]


def test_embedding_only_difference_does_not_fabricate_layer_localization():
    expected, actual = reference(), reference()
    actual["gradients"]["transformer.wte.weight"][0, 0] += .1
    row = compare_snapshots(actual, expected, name="embedding")
    assert not row["passed"]
    assert row["highest_layer_with_any_gradient_difference"] is None


def test_snapshot_copies_gradient_buffers_and_rejects_missing_ownership():
    model = torch.nn.Linear(3, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    hidden = torch.ones((1, 2))
    saved = snapshot(model, hidden)
    hidden.zero_()
    for parameter in model.parameters():
        parameter.grad.zero_()
    assert bool((saved["hidden"] == 1).all())
    assert all(bool((value == 1).all()) and value.device.type == "cpu" and not value.requires_grad
               for value in saved["gradients"].values())
    model.weight.grad = None
    with pytest.raises(AssertionError, match="disappeared"):
        snapshot(model, hidden)


def test_ownership_mismatch_and_unknown_backend_are_rejected():
    expected, actual = reference(), reference()
    actual["gradients"].pop("transformer.wte.weight")
    with pytest.raises(ValueError, match="ownership"):
        compare_snapshots(actual, expected, name="bad")
    with pytest.raises(ValueError, match="backend"):
        backend_context("unsupported")
