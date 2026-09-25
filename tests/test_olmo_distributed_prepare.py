"""Pure CPU tests for the bounded GPU harness's references and failure gates."""
import copy

import pytest
import torch

from cdrm.pretrained.distributed_training import ObjectiveForwardAdapter, sum_objective_counts
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from scripts import olmo_distributed_prepare as harness


def fixture():
    torch.manual_seed(212)
    torch.set_num_threads(1)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32")
    model = FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=32, proj_factor=2, vocab_chunk_size=4)).train()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    docs = torch.arange(2)[:, None].expand_as(ids).clone()
    full = NextLatBatch(ids, valid, docs, valid.clone(), valid.clone(), valid.clone())
    mode = FBTMode(num_passes=2, rt_mode=RTMode((0,)))
    return model, full, mode


def test_harness_reference_and_adapter_same_order_are_exact(monkeypatch):
    model, full, mode = fixture()
    monkeypatch.setattr(harness, "batch_for", lambda tokenizer, case, update: copy.deepcopy(full))
    batches = harness.unequal_batches(None, None)
    counts = sum_objective_counts([model.counts(batch) for batch in batches])
    assert model.counts(batches[1])["latent"] == model.counts(batches[1])["kl"] == 0
    expected = harness.backward_sequence(model, batches, mode, counts)
    gradients = harness.gradients_cpu(model)
    actual = harness.backward_sequence(model, batches, mode, counts, adapter=ObjectiveForwardAdapter(model))
    check = harness.check_snapshot(model, gradients, expected, actual, name="unit")
    assert check["passed"] and check["all_bitwise_equal"] and check["losses_and_counts_exact"]


def test_independent_fp32_microbatch_sum_passes_frozen_grouping_budget(monkeypatch):
    model, full, mode = fixture()
    monkeypatch.setattr(harness, "batch_for", lambda tokenizer, case, update: copy.deepcopy(full))
    batches = harness.unequal_batches(None, None)
    counts = sum_objective_counts([model.counts(batch) for batch in batches])
    total, losses = {}, []
    for batch in batches:
        losses.extend(harness.backward_sequence(model, [batch], mode, counts))
        harness.add_gradients_cpu(total, harness.gradients_cpu(model))
    actual = harness.backward_sequence(model, batches, mode, counts, adapter=ObjectiveForwardAdapter(model))
    check = harness.check_snapshot(model, total, losses, actual, name="independent", exact=False)
    assert check["passed"] and check["losses_and_counts_exact"]
    assert check["budgets"] == {"global_relative_l2": 2e-6,
                                "tensor_relative_l2": 2e-6, "tensor_max_relative": 1e-5}


def test_gradient_gate_rejects_ownership_nonfinite_or_material_error():
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    reference = harness.gradients_cpu(model)
    assert harness.gradient_check(model, reference, exact=True)["passed"]
    model.weight.grad[0, 0] += 1e-4
    assert not harness.gradient_check(model, reference, exact=True)["passed"]
    assert not harness.gradient_check(model, reference, exact=False)["passed"]
    model.weight.grad[0, 0] = float("nan")
    assert not harness.gradient_check(model, reference, exact=False)["passed"]
    model.weight.grad = None
    assert not harness.gradient_check(model, reference, exact=True)["passed"]


def test_independent_expected_participation_catches_common_missing_branch():
    model = torch.nn.Linear(2, 2, bias=True)
    model.weight.grad = torch.ones_like(model.weight)
    reference = harness.gradients_cpu(model)
    assert harness.gradient_check(model, reference, exact=True)["passed"]
    participation = harness.expected_ownership(model, {"weight", "bias"})
    assert not participation["expected_participation_matches"]
    assert participation["missing_expected_gradients"] == ["bias"]
    assert not harness.check_snapshot(model, reference, [], [], name="missing",
                                     expected_names={"weight", "bias"})["passed"]


def test_empty_reference_tensor_has_no_absolute_error_escape():
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.zeros_like(model.weight)
    reference = harness.gradients_cpu(model)
    assert harness.gradient_check(model, reference, exact=False)["passed"]
    model.weight.grad[0, 0] = 1e-12
    assert not harness.gradient_check(model, reference, exact=False)["passed"]


def test_cpu_snapshots_and_added_gradients_do_not_alias_live_storage():
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    reference = harness.gradients_cpu(model)
    total = {}
    harness.add_gradients_cpu(total, reference)
    model.weight.grad.add_(2)
    reference["weight"].add_(1)
    assert torch.equal(total["weight"], torch.ones_like(model.weight))
    harness.add_gradients_cpu(total, reference)
    assert torch.equal(total["weight"], torch.full_like(model.weight, 3))


@pytest.mark.parametrize("argument", [["--batch-size", "4"], ["--length", "1024"],
                                     ["--updates", "5"], ["--case", "ordinary"]])
def test_cli_refuses_unbounded_or_out_of_scope_cells(argument):
    with pytest.raises(SystemExit):
        harness.parse_args(["--case", "rt", "--output-dir", "unused", *argument])


def test_primary_defaults_and_frozen_sources_include_all_new_code():
    args = harness.parse_args(["--case", "combined", "--output-dir", "unused"])
    assert (args.batch_size, args.length, args.updates) == (2, 512, 2)
    assert "scripts/olmo_distributed_prepare.py" in harness.SOURCES
    assert "cdrm/pretrained/distributed_training.py" in harness.SOURCES
    assert harness.PROTOCOL.is_file()
