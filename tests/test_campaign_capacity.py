"""CPU timing/accounting tests, not hardware performance acceptance."""
import math

import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignRecipe, CampaignTokenSchedule
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts.olmo_campaign_capacity import (ROOT, adam_residency, capacity_fixture, clock_snapshot,
                                          logical_counts, parse_args, timing_card)


@pytest.mark.parametrize("microbatches", [1, 2])
@pytest.mark.parametrize("batch_size", [8, 12, 16, 32])
def test_capacity_full_valid_fixture_matches_global_input_and_per_term_counts(batch_size, microbatches):
    recipe = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0, 1))
    before = torch.get_rng_state().clone()
    actual = {"ce": 0, "latent": 0, "kl": 0}
    valid_tokens = documents = 0
    fixtures = []
    for rank in (0, 1):
        batches, noises = capacity_fixture(recipe, 32, rank=rank, update=2,
            batch_size=batch_size, microbatches=microbatches, token_ids=[2, 3, 4, 5], eos_id=60, length=8)
        fixtures.append((batches, noises))
        assert len(batches) == len(noises) == microbatches
        for batch, noise in zip(batches, noises):
            assert batch.valid_mask.all() and batch.input_ids.shape == (batch_size, 8)
            assert (batch.input_ids[:, -1] == 60).all()
            assert torch.unique(batch.document_ids[:, 0]).numel() == batch_size
            assert (batch.document_ids == batch.document_ids[:, :1]).all()
            for name, mask in build_nextlat_masks(batch).items():
                actual[name] += int(mask.sum())
            valid_tokens += int(batch.valid_mask.sum())
            documents += int(batch.valid_mask.any(-1).sum())
            assert len(noise) == 3 and all(n.shape == (batch_size, 7, 32) for n in noise)
    expected = logical_counts(batch_size, microbatches, length=8)
    assert actual == expected["counts"]
    assert valid_tokens == expected["input_tokens"] and documents == expected["documents"]
    assert not torch.equal(fixtures[0][1][0][0], fixtures[1][1][0][0])
    assert torch.equal(before, torch.get_rng_state())


def test_capacity_new_updates_change_tokens_noise_and_have_deterministic_repeats():
    recipe = CampaignRecipe("NFR", sequence_length=8, rt_layers=(0, 1))
    kwargs = dict(rank=0, batch_size=2, microbatches=1, token_ids=[2, 3, 4], eos_id=60, length=8)
    first = capacity_fixture(recipe, 32, update=0, **kwargs)
    again = capacity_fixture(recipe, 32, update=0, **kwargs)
    second = capacity_fixture(recipe, 32, update=1, **kwargs)
    assert torch.equal(first[0][0].input_ids, again[0][0].input_ids)
    assert torch.equal(first[1][0][0], again[1][0][0])
    assert not torch.equal(first[0][0].input_ids, second[0][0].input_ids)
    assert not torch.equal(first[1][0][0], second[1][0][0])


def test_timing_aggregates_total_tokens_over_sum_of_slowest_rank_times():
    rows = [{"max_rank_seconds": 2.0}, {"max_rank_seconds": 4.0}]
    card = timing_card(rows, 1024)
    assert card["valid_input_tokens_per_second"] == pytest.approx(2048/6)
    assert card["total_valid_input_tokens"] == 2048
    assert card["median_update_seconds"] == 3.0
    # It is neither the arithmetic mean of per-update speeds nor multiplied by K4.
    assert card["valid_input_tokens_per_second"] != pytest.approx((1024/2+1024/4)/2)
    assert card["measured_updates"] == 2


@pytest.mark.parametrize("seconds", [0, -1, math.inf, math.nan])
def test_timing_rejects_invalid_measurements(seconds):
    with pytest.raises(ValueError):
        timing_card([{"max_rank_seconds": seconds}], 1024)


def test_capacity_cli_fixes_sequence_length_and_bounded_single_candidate():
    common = ["--batch-size", "8", "--output-dir", str(ROOT/".runtime/capacity-test")]
    args = parse_args(common)
    assert args.length == 1024 and args.scale == "pretrained" and args.microbatches == 1
    assert args.warmup_updates == 3 and args.measured_updates == 5
    for extra in (["--batch-size", "512"], ["--microbatches", "3"], ["--warmup-updates", "0"], ["--measured-updates", "500"]):
        with pytest.raises(SystemExit):
            parse_args(common+extra)


@pytest.mark.parametrize("dimensions", [(0, 1, 1024), (1, 0, 1024), (1, 1, 2), (True, 1, 1024)])
def test_counts_reject_invalid_dimensions(dimensions):
    batch_size, microbatches, length = dimensions
    with pytest.raises(ValueError):
        logical_counts(batch_size, microbatches, length=length)


def test_real_adam_residency_requires_complete_initialized_moments_and_exact_step():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
    assert not adam_residency(model, optimizer, expected_steps=3)["passed"]
    for _ in range(3):
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
    actual = adam_residency(model, optimizer, expected_steps=3)
    assert actual["passed"]
    assert actual["actual_moment_bytes"] == 2*4*sum(p.numel() for p in model.parameters())
    assert actual["actual_state_bytes_by_device"]["cpu"] >= actual["actual_moment_bytes"]
    first = next(iter(optimizer.state.values()))
    first["step"].add_(1)
    assert not adam_residency(model, optimizer, expected_steps=3)["passed"]
    first["step"].sub_(1)
    del first["exp_avg_sq"]
    assert not adam_residency(model, optimizer, expected_steps=3)["passed"]


def test_clock_snapshot_does_not_alias_scheduler_and_detects_optimizer_or_token_advance():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, foreach=False)
    schedule = CampaignTokenSchedule(optimizer, [8]*8, warmup_tokens=80)
    counters = TrainingCounters()
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)
    schedule.step()
    counters.optimizer_updates = 1
    counters.input_tokens = 8
    before = clock_snapshot(optimizer, schedule, counters)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.zero_grad(set_to_none=False)
    assert before == clock_snapshot(optimizer, schedule, counters)
    optimizer.step()
    assert before != clock_snapshot(optimizer, schedule, counters)
    after_optimizer = clock_snapshot(optimizer, schedule, counters)
    schedule.step()
    assert after_optimizer != clock_snapshot(optimizer, schedule, counters)
    assert after_optimizer["scheduler"]["last_epoch"] == 1
    after_schedule = clock_snapshot(optimizer, schedule, counters)
    counters.input_tokens += 8
    assert after_schedule != clock_snapshot(optimizer, schedule, counters)
