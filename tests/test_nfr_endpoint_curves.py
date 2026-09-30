"""NFR64 scope authentication and preserved single-forward finite references."""
from copy import deepcopy

import pytest
import torch

from scripts import olmo_nfr_endpoint_curves as endpoint
from scripts import olmo_campaign_evaluation as evaluation
from scripts.olmo_lm_common import tree_digests
from test_fbt_component_curves import authority, setup


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(71)


def scope():
    ref = {'path': '.runtime/example.json', 'sha256': 'a'*64}
    return {'schema': endpoint.SCOPE_SCHEMA, **deepcopy(endpoint.POLICY),
        'references': {k: deepcopy(ref) for k in endpoint.REFS},
        'checkpoints': {k: deepcopy(ref) for k in ('control', 'reduced')},
        'implementation_sources': {'scripts/olmo_nfr_endpoint_curves.py': 'b'*64}}


@pytest.mark.parametrize('key,value', [('after_update', 32), ('num_passes', 8),
    ('sequence_length', 128), ('rt_layers', []), ('feedback_jitter', .02),
    ('panel_membership_sha256', 'c'*64), ('comparison_passes', [4]), ('extra', 1)])
def test_scope_cannot_silently_expand_historical_or_runtime_conditions(key, value):
    declared = scope()
    assert endpoint.validate_scope(declared) is declared
    declared[key] = value
    with pytest.raises(ValueError):
        endpoint.validate_scope(declared)


@pytest.mark.parametrize('case,weight', [('control', 1.), ('reduced', .1)])
def test_unique_audited_endpoint_authenticates(case, weight):
    report, manifest, spec, sources = authority('NFR', 64, weight)
    assert endpoint.authenticate_endpoint(report, manifest, spec, case=case,
        manifest_sha256='f'*64, sources=sources) == weight


@pytest.mark.parametrize('bad', ['case', 'update', 'cursor', 'publication', 'identity', 'report'])
def test_swapped_or_mutated_endpoint_rejected(bad):
    report, manifest, spec, sources = authority('NFR', 64, .1)
    report, manifest = deepcopy(report), deepcopy(manifest)
    case = 'reduced'
    if bad == 'case': case = 'control'
    elif bad == 'update': manifest['counters']['optimizer_updates'] = 32
    elif bad == 'cursor': manifest['rank_cursors'][1]['cursor']['next_chunk'] += 1
    elif bad == 'publication': report['published_checkpoints'] *= 2
    elif bad == 'identity':
        report['configuration']['execution_identity']['sha256'] = '0'*64
        manifest['metadata']['configuration'] = deepcopy(report['configuration'])
    elif bad == 'report': report['configuration']['different'] = True
    with pytest.raises(ValueError):
        endpoint.authenticate_endpoint(report, manifest, spec, case=case,
            manifest_sha256='f'*64, sources=sources)


def test_one_canonical_forward_preserves_old_metrics_and_matches_direct_residuals(monkeypatch):
    model, recipe, batch = setup('NFR')
    before = tree_digests(model.state_dict()); rng = torch.get_rng_state().clone()
    original = model.backbone.forward; calls = []; snapshots = []
    def tracked(*args, **kwargs):
        calls.append(1)
        output = original(*args, **kwargs)
        snapshots.append(output.pass_hidden_states)
        return output
    def forbidden(*args, **kwargs):
        raise AssertionError('Predictor must not execute in inference observation')
    monkeypatch.setattr(model.predictor, 'forward', forbidden)
    with evaluation.evaluation_runtime(model) as preservation:
        old = endpoint.previous.component_probe_batch(model, batch, recipe, passes=32)
        monkeypatch.setattr(model.backbone, 'forward', tracked)
        actual, direct = endpoint.endpoint_probe_batch(model, batch, recipe)
    assert len(calls) == 1 and actual == old
    assert direct == endpoint.hidden_residual_sums(snapshots[0], batch.valid_mask)
    assert preservation['integrity_passed'] and before == tree_digests(model.state_dict())
    assert torch.equal(rng, torch.get_rng_state()) and all(p.grad is None for p in model.parameters())
    assert not model.backbone._forward_hooks and not model.predictor._forward_pre_hooks


def test_failed_observation_removes_its_hook(monkeypatch):
    model, recipe, batch = setup('NFR')
    def fail(*args, **kwargs): raise RuntimeError('intentional failure')
    monkeypatch.setattr(endpoint.previous, 'component_probe_batch', fail)
    with pytest.raises(RuntimeError, match='intentional failure'):
        endpoint.endpoint_probe_batch(model, batch, recipe)
    assert not model.backbone._forward_hooks


def test_finite_prefix_excludes_first_k_and_aggregation_pools_norms():
    valid = torch.ones((1, 10), dtype=torch.bool)
    states = [torch.ones((1, 10, 2)) for _ in range(32)]
    states[3] = states[3]*2
    first = endpoint.hidden_residual_sums(states, valid)
    assert first['comparisons'][0]['regions']['beyond_guaranteed_prefix']['positions'] == 6
    second_states = [s*3 for s in states]
    second_states[3] = second_states[-1].clone()
    second = endpoint.hidden_residual_sums(second_states, valid)
    result = endpoint.summarize_residuals([first, second])['comparisons'][0]['regions']['all']
    assert result['positions'] == 20
    assert result['relative_l2'] == pytest.approx((10/100)**.5)
    assert result['delta_rms'] == pytest.approx(.5**.5)
    assert result['reference_rms'] == pytest.approx(5**.5)
    # K8 has no unsettled positions when the only valid positions are 0..5.
    valid[:, 6:] = False
    result = endpoint.summarize_residuals([endpoint.hidden_residual_sums(states, valid)])
    assert result['comparisons'][1]['regions']['beyond_guaranteed_prefix']['relative_l2'] is None
