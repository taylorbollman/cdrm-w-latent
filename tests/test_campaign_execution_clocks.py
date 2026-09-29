"""Boundary clock guards with actual CPU AdamW and the finite token schedule."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignTokenSchedule
from cdrm.pretrained.lm_training import TrainingCounters
from scripts import olmo_campaign_execution as engine
from scripts.olmo_lm_common import tree_digests


TOKENS = (80, 80, 80)


def actual_boundary(completed):
    # Two actual Adam groups exercise the per-owner/per-group checks. A
    # nonuniform gradient gives nontrivial first and second moments.
    parameters = [torch.nn.Parameter(torch.tensor([1., -2., 3.])),
                  torch.nn.Parameter(torch.tensor([[.5, -.75]]))]
    optimizer = torch.optim.AdamW([
        {'params': [parameters[0]], 'lr': 1e-4},
        {'params': [parameters[1]], 'lr': 2e-4}], betas=(.9, .95), fused=False)
    scheduler = CampaignTokenSchedule(optimizer, TOKENS, warmup_tokens=240,
                                      start_fraction=.1)
    counters = TrainingCounters()
    for update in range(completed):
        sum(((index + 1) * p.square()).sum() for index, p in enumerate(parameters)).backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        counters.optimizer_updates += 1
        counters.input_tokens += TOKENS[update]
        counters.microbatches += 4
        counters.documents += 5
        counters.ce_positions += 75
    return parameters, optimizer, scheduler, counters


def snapshot(parameters, optimizer, scheduler, counters):
    return tree_digests({'parameters': parameters, 'adam': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'counters': asdict(counters),
        'rng': torch.get_rng_state()})


@pytest.mark.parametrize('completed', [0, 1, 3])
def test_actual_empty_ordinary_and_terminal_boundaries_are_read_only(completed):
    parameters, optimizer, scheduler, counters = actual_boundary(completed)
    before = snapshot(parameters, optimizer, scheduler, counters)
    engine.validate_clocks(optimizer, scheduler, counters)
    assert snapshot(parameters, optimizer, scheduler, counters) == before
    assert scheduler.last_epoch == completed
    assert bool(optimizer.state) == bool(completed)
    if completed == len(TOKENS):
        # A terminal resume must validate without asking for nonexistent data.
        with pytest.raises(ValueError, match='exhausted'):
            scheduler.validate_next_update(TOKENS[-1])


@pytest.mark.parametrize('mutation', [
    'scheduler_update', 'scheduler_step_count', 'scheduler_last_lr',
    'optimizer_lr', 'cumulative_tokens', 'adam_step', 'missing_owner',
    'missing_first_moment', 'missing_second_moment', 'moment_shape',
    'moment_dtype', 'foreign_owner'])
def test_completed_boundary_rejects_independent_clock_or_adam_drift(mutation):
    parameters, optimizer, scheduler, counters = actual_boundary(2)
    state = optimizer.state[parameters[0]]
    if mutation == 'scheduler_update': scheduler.last_epoch -= 1
    elif mutation == 'scheduler_step_count': scheduler._step_count -= 1
    elif mutation == 'scheduler_last_lr': scheduler._last_lr[1] *= .5
    elif mutation == 'optimizer_lr': optimizer.param_groups[1]['lr'] *= .5
    elif mutation == 'cumulative_tokens': counters.input_tokens -= 1
    elif mutation == 'adam_step': state['step'].sub_(1)
    elif mutation == 'missing_owner': del optimizer.state[parameters[1]]
    elif mutation == 'missing_first_moment': del state['exp_avg']
    elif mutation == 'missing_second_moment': del state['exp_avg_sq']
    elif mutation == 'moment_shape': state['exp_avg'] = state['exp_avg'].reshape(1, -1)
    elif mutation == 'moment_dtype': state['exp_avg_sq'] = state['exp_avg_sq'].to(torch.bfloat16)
    elif mutation == 'foreign_owner':
        optimizer.state[torch.nn.Parameter(torch.ones(1))] = deepcopy(state)
    with pytest.raises(ValueError):
        engine.validate_clocks(optimizer, scheduler, counters)


def test_terminal_resume_rejects_internally_consistent_stale_scheduler():
    parameters, optimizer, scheduler, counters = actual_boundary(3)
    # The generic loader accepts a schedule whose own metadata is consistent
    # but whose epoch disagrees with the independent committed data cursor.
    _, earlier_optimizer, earlier_scheduler, _ = actual_boundary(2)
    scheduler.load_state_dict(deepcopy(earlier_scheduler.state_dict()))
    for current, earlier in zip(optimizer.param_groups, earlier_optimizer.param_groups):
        current['lr'] = earlier['lr']
    assert scheduler._last_lr == scheduler.get_lr()
    assert [group['lr'] for group in optimizer.param_groups] == scheduler.get_lr()
    assert counters.optimizer_updates == 3 and scheduler.last_epoch == 2
    before = snapshot(parameters, optimizer, scheduler, counters)
    with pytest.raises(ValueError):
        engine.validate_clocks(optimizer, scheduler, counters)
    assert snapshot(parameters, optimizer, scheduler, counters) == before


def test_fresh_zero_boundary_rejects_resident_adam_even_with_zero_step():
    parameters, optimizer, scheduler, counters = actual_boundary(0)
    optimizer.state[parameters[0]] = {'step': torch.tensor(0.),
        'exp_avg': torch.zeros_like(parameters[0]),
        'exp_avg_sq': torch.zeros_like(parameters[0])}
    with pytest.raises(ValueError):
        engine.validate_clocks(optimizer, scheduler, counters)


def test_fresh_objects_restore_full_actual_terminal_boundary():
    parameters, optimizer, scheduler, counters = actual_boundary(3)
    new_parameters, new_optimizer, new_scheduler, _ = actual_boundary(0)
    with torch.no_grad():
        for target, source in zip(new_parameters, parameters):
            target.copy_(source)
    new_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    new_scheduler.load_state_dict(deepcopy(scheduler.state_dict()))
    new_counters = TrainingCounters(**asdict(counters))
    engine.validate_clocks(new_optimizer, new_scheduler, new_counters)
    assert tree_digests(new_optimizer.state_dict()) == tree_digests(optimizer.state_dict())
    assert new_scheduler.state_dict() == scheduler.state_dict()
    assert asdict(new_counters) == asdict(counters)
