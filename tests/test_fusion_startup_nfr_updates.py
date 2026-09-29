"""Actual tiny paired updates, no-alias CPU restoration and generic recovery."""
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import build_campaign_adamw, CampaignTokenSchedule
from cdrm.pretrained.lm_training import TrainingCounters
from scripts import olmo_fusion_startup_nfr_updates as probe
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, canonical_backward, advance_counters
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, state_pins
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    torch.set_num_threads(1)
    @contextmanager
    def preserve_cpu_rng():
        before = probe._rng_state(None)
        try:
            yield
        finally:
            probe._restore_rng(before, None)
    monkeypatch.setattr(probe, 'preserve_local_rng', preserve_cpu_rng)


def tiny(updates=4):
    model, recipe, _, ids, eos = construct(SimpleNamespace(scale='tiny', length=8), 'NF', torch.device('cpu'))
    recipe, _ = probe.enable_native_rt(model, recipe)
    model.backbone.backbone.attention_precision = 'mixed'
    flags = {name: getattr(model.backbone.backbone, name) for name in RUNTIME_FLAGS}
    fixtures = [[fixture_for_update(recipe, model.config.model_dim, rank, step % 3,
        length=8, token_ids=ids, eos_id=eos, batch_size=2) for rank in (0, 1)] for step in range(updates)]
    metadata = [probe.global_fixture_metadata(model, row) for row in fixtures]
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    scheduler = CampaignTokenSchedule(optimizer, [m['input_tokens'] for m in metadata],
        warmup_tokens=recipe.warmup_tokens, start_fraction=recipe.warmup_start_fraction)
    return model, recipe, fixtures, metadata, flags, optimizer, scheduler


def test_full_update_matches_literal_canonical_backward_clip_adam_and_token_clock():
    model, recipe, fixtures, metadata, flags, optimizer, scheduler = tiny()
    initial = probe.capture_boundary(model, optimizer, scheduler, TrainingCounters(), metadata)
    counters = TrainingCounters()
    row, raw, clipped = probe.update(model, recipe, fixtures[0], optimizer, scheduler, counters,
        path=FP32, original_flags=flags, metadata=metadata)
    actual = probe.capture_boundary(model, optimizer, scheduler, counters, metadata)
    assert row['lr_used'] == [2e-5]*len(optimizer.param_groups)
    counters = probe.restore_boundary(model, optimizer, scheduler, initial, metadata)
    probe.configure_path(model, flags, FP32)
    with probe.sdpa_kernel(probe.SDPBackend.MATH):
        metrics = canonical_backward(model, recipe, fixtures[0], precision='fp32')
    assert metrics == row['metrics']
    assert all(torch.equal(p.grad, raw[name]) for name, p in model.named_parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True, foreach=False)
    assert all(torch.equal(p.grad, clipped[name]) for name, p in model.named_parameters())
    optimizer.step(); scheduler.step(); advance_counters(counters, metrics); model.zero_grad(set_to_none=True)
    expected = probe.capture_boundary(model, optimizer, scheduler, counters, metadata)
    assert tree_digests(expected) == tree_digests(actual)
    assert scheduler.completed_tokens == metadata[0]['input_tokens']
    assert counters.ce_positions == metadata[0]['counts']['ce']


def test_cpu_restore_keeps_parameter_identity_and_saved_adam_does_not_alias_live_updates():
    model, recipe, fixtures, metadata, flags, optimizer, scheduler = tiny()
    counters = TrainingCounters()
    probe.update(model, recipe, fixtures[0], optimizer, scheduler, counters,
        path=FP32, original_flags=flags, metadata=metadata)
    boundary = probe.capture_boundary(model, optimizer, scheduler, counters, metadata)
    before = tree_digests(boundary)
    identities = {name: id(p) for name, p in model.named_parameters()}
    probe.update(model, recipe, fixtures[1], optimizer, scheduler, counters,
        path=FP32, original_flags=flags, metadata=metadata)
    golden = probe.capture_boundary(model, optimizer, scheduler, counters, metadata)
    counters = probe.restore_boundary(model, optimizer, scheduler, boundary, metadata)
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    probe.update(model, recipe, fixtures[1], optimizer, scheduler, counters,
        path=FP32, original_flags=flags, metadata=metadata)
    actual = probe.capture_boundary(model, optimizer, scheduler, counters, metadata)
    assert tree_digests(boundary) == before
    assert tree_digests(actual) == tree_digests(golden)


@pytest.mark.parametrize('mutation', ['counter', 'scheduler', 'scheduler_count', 'scheduler_lr', 'moment', 'variance', 'step', 'ownership', 'lr', 'mode'])
def test_bad_boundary_rejected_before_live_parameter_or_optimizer_mutation(mutation):
    model, recipe, fixtures, metadata, flags, optimizer, scheduler = tiny()
    counters = TrainingCounters()
    probe.update(model, recipe, fixtures[0], optimizer, scheduler, counters,
        path=FP32, original_flags=flags, metadata=metadata)
    snapshot = probe.capture_boundary(model, optimizer, scheduler, counters, metadata)
    before = tree_digests(snapshot)
    bad = probe.cpu_copy(snapshot)
    identifier = next(iter(bad['optimizer']['state']))
    if mutation == 'counter': bad['counters']['ce_positions'] += 1
    elif mutation == 'scheduler': bad['scheduler']['last_epoch'] = 0
    elif mutation == 'scheduler_count': bad['scheduler']['_step_count'] += 1
    elif mutation == 'scheduler_lr': bad['scheduler']['_last_lr'][0] *= 2
    elif mutation == 'moment': bad['optimizer']['state'][identifier]['exp_avg'].view(-1)[0] = float('nan')
    elif mutation == 'variance': bad['optimizer']['state'][identifier]['exp_avg_sq'].view(-1)[0] = -1
    elif mutation == 'step': bad['optimizer']['state'][identifier]['step'] += 1
    elif mutation == 'ownership': bad['ownership'][0][0] = 'foreign'
    elif mutation == 'lr': bad['optimizer']['param_groups'][0]['lr'] *= 2
    elif mutation == 'mode': bad['modes'][''] = False
    with pytest.raises(ValueError):
        probe.restore_boundary(model, optimizer, scheduler, bad, metadata)
    assert tree_digests(probe.capture_boundary(model, optimizer, scheduler, counters, metadata)) == before


def test_four_paired_updates_share_inputs_rng_schedule_and_checkpoint_both_endpoints():
    model, recipe, fixtures, metadata, flags, _, _ = tiny()
    checkpoints, rows = [], []
    def save(path, model, optimizer, scheduler, counters, state):
        assert tree_digests(probe.capture_boundary(model, optimizer, scheduler, counters, metadata)) == tree_digests(state)
        checkpoints.append((path, counters.optimizer_updates))
    result = probe.run_pair(model, recipe, metadata, lambda step: fixtures[step], fixtures[0],
        original_flags=flags, publish=rows.append, checkpoint=save)
    assert result['optimizer_calls'] == 8 and len(rows) == 8
    assert checkpoints == [(FP32, 4), (BF16, 4)]
    assert result['evaluations']['0'][FP32] == result['evaluations']['0'][BF16]
    assert all(all(row['checks'].values()) for row in result['evaluations']['4'].values())
    assert all(row['counts']['ce'] > 0 for row in result['evaluations']['4'].values())
    assert rows[0]['start_boundary_pins'] == rows[1]['start_boundary_pins']
    for pair in result['rows']:
        assert pair[0]['input_pins'] == pair[1]['input_pins']
        assert pair[0]['lr_used'] == pair[1]['lr_used']
        assert set(pair[1]['bf16_vs_fp32']) == {'raw_gradient', 'clipped_gradient', 'actual_master_delta', 'exp_avg', 'exp_avg_sq'}
        assert all(row['reference_gradient_norm'] >= 0 for row in pair[1]['bf16_vs_fp32']['actual_master_delta'].values())
    assert all(p.grad is None for p in model.parameters())
    assert all(getattr(model.backbone.backbone, n) == value for n, value in flags.items())


def test_fp32_evaluation_preserves_complete_boundary_and_reports_all_losses():
    model, recipe, fixtures, metadata, flags, optimizer, scheduler = tiny()
    counters = TrainingCounters()
    before = tree_digests(probe.capture_boundary(model, optimizer, scheduler, counters, metadata))
    result = probe.evaluate_fp32(model, recipe, fixtures[0], original_flags=flags)
    assert all(result['checks'].values()) and set(result['loss_means']) == {'ce', 'latent', 'kl'}
    assert result['combined_objective'] == sum(result['loss_means'].values())
    assert before == tree_digests(probe.capture_boundary(model, optimizer, scheduler, counters, metadata))


def test_lazy_delta_preserves_small_master_changes_without_full_fp64_clone():
    a = {'backbone.backbone.weight': torch.tensor([1e-20, 0.001], dtype=torch.float32)}
    b = {'backbone.backbone.weight': torch.tensor([-1e-12, 0.00100001], dtype=torch.float32)}
    delta = probe.DeltaView(a, b)
    assert delta.actual is a and delta.reference is b
    assert torch.equal(delta['backbone.backbone.weight'], a['backbone.backbone.weight'].double()-b['backbone.backbone.weight'].double())
    geometry = probe.summary(delta)
    assert geometry['all']['relative_l2'] == 0 and geometry['all']['reference_gradient_norm'] > 0


def test_nonfinite_objective_with_finite_gradients_cannot_reach_adam(monkeypatch):
    model, recipe, fixtures, metadata, flags, optimizer, scheduler = tiny()
    original = probe.component_backward
    def poisoned(*args, **kwargs):
        metrics = original(*args, **kwargs)
        metrics['objective'] = float('inf')
        return metrics
    monkeypatch.setattr(probe, 'component_backward', poisoned)
    before = state_pins(model)
    with pytest.raises(FloatingPointError, match='before Adam'):
        probe.update(model, recipe, fixtures[0], optimizer, scheduler, TrainingCounters(),
            path=FP32, original_flags=flags, metadata=metadata)
    assert not optimizer.state and state_pins(model) == before and scheduler.last_epoch == 0
    model.zero_grad(set_to_none=True)


def test_generic_full_checkpoint_fresh_object_restore_and_invalid_cursor_rejection(tmp_path):
    model, recipe, fixtures, metadata, flags, optimizer, scheduler = tiny()
    counters = TrainingCounters()
    probe.update(model, recipe, fixtures[0], optimizer, scheduler, counters,
        path=FP32, original_flags=flags, metadata=metadata)
    original = probe.capture_boundary(model, optimizer, scheduler, counters, metadata)
    config = {'schema': probe.SCHEMA, 'trajectory': FP32, 'schedule': scheduler.checkpoint_contract()}
    source = {'checkpoint_sha256': 'a'*64}
    cursor = lambda step: {'next_update': 144+step, 'data_manifest_sha256': 'b'*64}
    saved = probe.save_endpoint(tmp_path/'endpoint.pt', model, optimizer, scheduler, counters,
        configuration=config, source_fingerprint=source, data_cursor=cursor(1))
    fresh, _, _, fresh_metadata, _, fresh_optimizer, fresh_scheduler = tiny()
    identities = {n: id(p) for n, p in fresh.named_parameters()}
    restored = probe.load_endpoint(tmp_path/'endpoint.pt', fresh, fresh_optimizer, fresh_scheduler,
        configuration=config, source_fingerprint=source, expected_sha256=saved['sha256'], metadata=fresh_metadata, expected_cursor=cursor)
    assert restored['counters'].optimizer_updates == 1
    assert identities == {n: id(p) for n, p in fresh.named_parameters()}
    assert tree_digests(probe.capture_boundary(fresh, fresh_optimizer, fresh_scheduler, restored['counters'], metadata)) == tree_digests(original)
    payload = torch.load(tmp_path/'endpoint.pt', map_location='cpu', weights_only=True)
    payload['data_cursor']['next_update'] += 1; torch.save(payload, tmp_path/'bad.pt')
    before = state_pins(fresh)
    with pytest.raises(ValueError, match='data cursor'):
        probe.load_endpoint(tmp_path/'bad.pt', fresh, fresh_optimizer, fresh_scheduler,
            configuration=config, source_fingerprint=source, expected_sha256=sha256_file(tmp_path/'bad.pt'), metadata=metadata, expected_cursor=cursor)
    assert state_pins(fresh) == before
