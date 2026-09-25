"""Pure CPU tests for the bounded GPU harness's references and failure gates."""
import copy

import pytest
import torch

from cdrm.pretrained.distributed_training import ObjectiveForwardAdapter, sum_objective_counts
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.lm_training import (
    LMTrainingConfig, TrainingCounters, build_adamw, build_warmup_scheduler,
    optimizer_step, _rng_state, _restore_rng,
)
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTMode, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from scripts import olmo_distributed_prepare as harness


def fixture(*, nextlat=True):
    torch.manual_seed(212)
    torch.set_num_threads(1)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math", attention_precision="fp32")
    model = FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=32, proj_factor=2, vocab_chunk_size=4), enabled=nextlat).train()
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


@pytest.mark.parametrize("fbt,rt,nextlat", [(f, r, n) for f in (False, True) for r in (False, True) for n in (False, True)])
def test_two_complete_accumulated_updates_match_canonical_optimizer_exactly(monkeypatch, fbt, rt, nextlat):
    model, full, _ = fixture(nextlat=nextlat)
    mode = FBTMode(enabled=fbt, num_passes=2, rt_mode=RTMode((0,) if rt else ()))
    monkeypatch.setattr(harness, "batch_for", lambda tokenizer, case, update: copy.deepcopy(full))
    batches = harness.unequal_batches(None, None)
    initial_state = {name: value.clone() for name, value in model.state_dict().items()}
    initial_rng = _rng_state(None)
    addresses = [p.data_ptr() for p in model.parameters()]
    config = LMTrainingConfig(precision="fp32", max_grad_norm=1.)
    references = []
    actual_steps = 0
    for branch in ("reference", "adapter"):
        if branch == "adapter":
            model.load_state_dict(initial_state)
            _restore_rng(initial_rng, None)
            assert [p.data_ptr() for p in model.parameters()] == addresses
        optimizer = build_adamw(model, lr=1e-4, eps=1e-4, fused=False)
        scheduler = build_warmup_scheduler(optimizer, warmup_updates=2)
        counters = TrainingCounters()
        adapter = ObjectiveForwardAdapter(model)
        for update in range(2):
            if branch == "reference":
                metrics = optimizer_step(model, optimizer, batches, config=config,
                    backbone_kwargs={"mode": mode}, scheduler=scheduler, counters=counters)
            else:
                metrics = harness.adapter_optimizer_update(model, adapter, optimizer, scheduler,
                                                           counters, batches, mode, config=config)
            actual_steps += 1
            record = {"metrics": metrics, "batches": [harness.tree_digests(vars(batch)) for batch in batches],
                      "boundary": {"state": harness.boundary_digests(model, optimizer, scheduler, counters),
                                   "rng": harness.tree_digests(_rng_state(None))}}
            if branch == "reference": references.append(record)
            else:
                check = harness.update_equality(references[update], record, "update")
                assert check["passed"], check
            assert counters.optimizer_updates == update+1
            assert counters.microbatches == 2*(update+1)
            assert all(p.grad is None for p in model.parameters())
        del optimizer, scheduler
    assert actual_steps == 4 and counters.optimizer_updates == 2


@pytest.mark.parametrize("field", ["metrics", "batches", "boundary"])
def test_complete_update_gate_detects_each_comparison_component(field):
    expected = {"metrics": {"loss": 1.}, "batches": ["hash"], "boundary": {"rng": "hash", "state": "hash"}}
    actual = copy.deepcopy(expected)
    actual[field] = "changed"
    check = harness.update_equality(expected, actual, "changed")
    assert not check["passed"] and not check["bitwise_manifest_equal"][field]


def test_tracking_rng_scope_restores_all_cpu_random_state():
    import random
    import numpy as np
    before = harness.tree_digests(_rng_state(None))
    with harness.preserve_rng():
        torch.rand(3)
        random.random()
        np.random.rand(3)
    assert harness.tree_digests(_rng_state(None)) == before


def test_reference_cache_policy_does_not_enable_outer_autocast_and_restores_on_error():
    previous = torch.is_autocast_cache_enabled()
    assert not torch.is_autocast_enabled("cpu")
    with pytest.raises(RuntimeError, match="fixture"):
        with harness.disable_autocast_weight_cache():
            assert not torch.is_autocast_enabled("cpu")
            assert not torch.is_autocast_cache_enabled()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                assert torch.is_autocast_enabled("cpu")
                assert not torch.is_autocast_cache_enabled()
            assert not torch.is_autocast_enabled("cpu")
            raise RuntimeError("fixture")
    assert torch.is_autocast_cache_enabled() is previous
