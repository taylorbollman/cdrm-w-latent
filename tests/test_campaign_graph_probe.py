"""CPU tests for bounded graph probe fixtures, snapshots and comparison gates."""
import copy

import pytest
import torch

from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts.olmo_campaign_graph_probe import (compare_metrics, compare_optimizer,
    compare_tensor_mappings, fixture_batches, fixture_record, gradients_are_zero,
    optimizer_snapshot, parameter_snapshot, parse_args, restore_parameters_in_place)


@pytest.mark.parametrize("length", [16, 32])
def test_fixtures_have_changing_masks_counts_short_rows_and_complete_empty_slots(length):
    updates = [fixture_batches(list(range(2, 100)), 101, length=length, update=u) for u in (0,1)]
    expected_sizes = [((length,7),(2,0),(0,0)), ((length-3,5),(6,0),(0,0))]
    counts = []
    for u, (batches, keys) in enumerate(updates):
        assert len(batches) == len(keys) == 3
        assert tuple(tuple(int(x) for x in b.valid_mask.sum(-1)) for b in batches) == expected_sizes[u]
        assert all(b.input_ids.shape == (2,length) for b in batches)
        assert all(not bool((b.valid_mask[:,1:] & ~b.valid_mask[:,:-1]).any()) for b in batches)
        assert all(len(k) == int(b.valid_mask.any(-1).sum()) for k,b in zip(keys,batches))
        assert all(not mask.any() for mask in build_nextlat_masks(batches[-1]).values())
        counts.append({term: sum(int(build_nextlat_masks(b)[term].sum()) for b in batches)
                       for term in ("ce","latent","kl")})
        record = fixture_record(batches)
        assert record[0]["valid_mask"] == batches[0].valid_mask.tolist()
    assert counts[0] != counts[1]
    assert all(value > 0 for update in counts for value in update.values())
    assert not torch.equal(updates[0][0][0].input_ids, updates[1][0][0].input_ids)
    assert not torch.equal(updates[0][0][0].latent_mask, updates[1][0][0].latent_mask)


def test_parameter_snapshot_restore_preserves_storages_and_rejects_malformed_snapshots():
    model = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2))
    original = parameter_snapshot(model)
    pointers = {name: p.data_ptr() for name,p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(.3)
    assert not compare_tensor_mappings(dict(model.named_parameters()), original, atol=1e-7,rtol=1e-7)["passed"]
    restore_parameters_in_place(model, original)
    assert compare_tensor_mappings(dict(model.named_parameters()), original, atol=0,rtol=0)["passed"]
    assert pointers == {name:p.data_ptr() for name,p in model.named_parameters()}
    broken = dict(original)
    broken["0.weight"] = broken["0.weight"][:1]
    with pytest.raises(ValueError,match="snapshot differs"):
        restore_parameters_in_place(model,broken)
    assert compare_tensor_mappings(dict(model.named_parameters()), original, atol=0,rtol=0)["passed"]


def test_optimizer_comparison_checks_moments_and_exact_step_counts():
    model = torch.nn.Linear(3,2)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
    model(torch.ones(2,3)).square().sum().backward()
    optimizer.step()
    reference = optimizer_snapshot(model,optimizer)
    assert compare_optimizer(model,optimizer,reference)["passed"]
    first = next(iter(optimizer.state.values()))
    first["step"].add_(1)
    assert not compare_optimizer(model,optimizer,reference)["passed"]
    first["step"].sub_(1)
    first["exp_avg"].add_(.01)
    assert not compare_optimizer(model,optimizer,reference)["passed"]


def test_gradient_zero_gate_rejects_missing_nonzero_and_nonfinite_gradients():
    model = torch.nn.Linear(3,2,bias=False)
    assert not gradients_are_zero(model)
    model.weight.grad = torch.zeros_like(model.weight)
    assert gradients_are_zero(model)
    model.weight.grad[0,0] = 1
    assert not gradients_are_zero(model)
    model.weight.grad[0,0] = float("nan")
    assert not gradients_are_zero(model)


def test_small_update_discrepancy_is_not_hidden_by_large_pretrained_weights():
    initial = {"weight": torch.full((4,), 100., dtype=torch.float64)}
    expected = {"weight": initial["weight"]+.0001}
    actual = {"weight": expected["weight"]+.00001}
    assert compare_tensor_mappings(actual,expected,atol=3e-6,rtol=3e-5)["passed"]
    result = compare_tensor_mappings(actual,expected,atol=3e-6,rtol=3e-5,initial=initial)
    assert not result["passed"]
    assert result["update_relative_l2"] == pytest.approx(.1,rel=1e-6)
    assert result["update_relative_l2_budget"] == 1e-3


def test_metric_comparison_checks_target_counts_and_losses_separately():
    reference = {"counts":{"ce":7,"latent":6,"kl":5},"microbatches":3,
                 "documents":3,"input_tokens":10,"objective":2.5,
                 "loss_sums":{"ce":10.,"latent":4.,"kl":2.}}
    assert compare_metrics(reference,reference)["passed"]
    bad = copy.deepcopy(reference)
    bad["counts"]["kl"] += 1
    assert not compare_metrics(bad,reference)["passed"]
    bad = copy.deepcopy(reference)
    bad["loss_sums"]["kl"] += .1
    assert not compare_metrics(bad,reference)["passed"]
    bad = copy.deepcopy(reference)
    bad["objective"] += .1
    assert not compare_metrics(bad,reference)["passed"]


@pytest.mark.parametrize("extra", [["--length","1024"],["--updates","100"],["--tiny"],["--batch","64"]])
def test_parser_rejects_unbounded_training_or_tiny_model_substitution(extra):
    with pytest.raises(SystemExit):
        parse_args(["--output-dir","/tmp/campaign-graph-probe",*extra])


@pytest.mark.parametrize("arguments", [dict(length=16,update=2),dict(length=10,update=0)])
def test_fixture_scope_rejects_unknown_dimensions_or_updates(arguments):
    with pytest.raises(ValueError):
        fixture_batches(list(range(100)),101,**arguments)
