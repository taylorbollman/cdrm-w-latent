"""Real ordered-data membership and composited live-controller preservation."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from scripts import olmo_fbt_stability_probe as probe
from scripts.olmo_lm_common import tree_digests
from test_pilot_eval_control import spec, live_setup
from test_pilot_ordered_data import fixture, cpu_tokenizer


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_plan_has_pinned_fixed_prefix_shallow_and_deep_schedule(tmp_path):
    data = fixture(tmp_path)
    plan = probe.resolve_probe_plan(spec(data), length=4, updates=192, world_size=2, panel_rows=3)
    assert plan['deep_updates'] == [0, 32, 64, 96, 100, 128, 192]
    assert plan['scheduled_updates'] == sorted(set([0, *range(8, 193, 8), 100]))
    assert plan['panel']['fixed_plan']['totals']['valid_tokens'] == 12
    assert plan['panel']['fixed_plan']['updates'][0]['allocation_by_arm']['F'][1]['dummy_rows'] == 1


def test_composite_preserves_live_state_runs_origin_once_and_keeps_normal_dev(tmp_path):
    data = fixture(tmp_path)
    ordinary, model, optimizer, report, tracker = live_setup(data, tmp_path, 'F')
    plan = probe.resolve_probe_plan(spec(data), length=4, updates=8, world_size=1, panel_rows=3)
    # This test still exercises the exact live controller, bounded more tightly
    # than the production K32 origin to keep unit-test runtime economical.
    plan['deep_passes'] = 4
    controller = probe.StabilityEvaluationController(ordinary, plan)
    rng = torch.Generator().manual_seed(111)
    def boundary():
        return tree_digests({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'rng': torch.get_rng_state(), 'custom_rng': rng.get_state(),
            'gradients': {n: p.grad for n, p in model.named_parameters()}})
    before = boundary(); persisted = []
    with probe.ordered.OrderedCampaignData(data.corpus, data.output/'panels/train') as train:
        cursor = asdict(train.cursor())
        kwargs = dict(model=model, runner=None, generators={'custom': rng}, training_data=train,
                      boundary=boundary, persist=lambda: persisted.append(1))
        controller.run_if_due(0, **kwargs)
        controller.run_if_due(0, **kwargs)
        assert len(report['stability_probes']) == 1 and not report['evaluations']
        controller.run_if_due(1, **kwargs)
        assert asdict(train.cursor()) == cursor
    assert before == boundary()
    assert len(report['evaluations']) == 1 and len(report['stability_probes']) == 1
    assert report['stability_probes'][0]['training_boundary_exact_by_rank'] == [True]
    assert (tmp_path/'stability-update-000000.json').is_file()
    assert 'dev/stability/all/pass_4/ce' in controller.metrics_for(0)
    assert 'dev/main/pass_4/ce' in controller.metrics_for(1)
    assert len(tracker.logs) == 2


def test_probe_failure_never_publishes_and_preserves_boundary(tmp_path, monkeypatch):
    data = fixture(tmp_path)
    ordinary, model, optimizer, report, tracker = live_setup(data, tmp_path, 'F')
    plan = probe.resolve_probe_plan(spec(data), length=4, updates=8, world_size=1, panel_rows=3)
    controller = probe.StabilityEvaluationController(ordinary, plan)
    before = tree_digests(model.state_dict()); rng = torch.get_rng_state().clone()
    def fail(*args, **kwargs):
        torch.rand(7)
        raise RuntimeError('probe injected failure')
    monkeypatch.setattr(probe, 'probe_batch', fail)
    with probe.ordered.OrderedCampaignData(data.corpus, data.output/'panels/train') as train:
        with pytest.raises(RuntimeError, match='probe injected'):
            controller.run_if_due(0, model=model, runner=None, generators={}, training_data=train,
                boundary=lambda: tree_digests(model.state_dict()), persist=lambda: None)
    assert tree_digests(model.state_dict()) == before and torch.equal(torch.get_rng_state(), rng)
    assert report['stability_probes'][0]['status'] == 'failed'
    assert not tracker.logs and not controller.probe_published
    assert not (tmp_path/'stability-update-000000.json').exists()


def test_probe_rejects_changed_membership_before_forward(tmp_path):
    data = fixture(tmp_path)
    ordinary, model, optimizer, report, tracker = live_setup(data, tmp_path, 'F')
    plan = probe.resolve_probe_plan(spec(data), length=4, updates=8, world_size=1, panel_rows=3)
    plan['panel']['fixed_plan']['updates'][0]['membership_sha256'] = '0'*64
    controller = probe.StabilityEvaluationController(ordinary, plan)
    with probe.ordered.OrderedCampaignData(data.corpus, data.output/'panels/train') as train:
        with pytest.raises(ValueError, match='membership'):
            controller.run_if_due(0, model=model, runner=None, generators={}, training_data=train,
                boundary=lambda: {}, persist=lambda: None)
    assert not tracker.logs and not controller.probe_published
