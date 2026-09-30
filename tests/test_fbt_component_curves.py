"""CPU state/metric equivalence and strict saved-condition lineage checks."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_fbt_component_curves as curves
from scripts import olmo_fbt_stability_probe as fprobe
from scripts import olmo_campaign_evaluation as evaluation
from scripts.olmo_kl_branch import declared_recipe
from scripts.olmo_lm_common import tree_digests
from test_fbt_stability_probe import setup as f_setup


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1); torch.manual_seed(71)


def setup(arm):
    recipe = CampaignRecipe(arm, sequence_length=8, rt_layers=(0, 1),
                            document_policy='continuous-stream-v1')
    model = build_campaign_model(OLMoTiledRTForCausalLM(OLMoConfig.tiny(),
        attention_backend='sdpa', attention_precision='mixed', tile_backend='eager',
        backward_tile_backend='eager', reuse_rope=True, kv_only_writes=True), recipe).train()
    batch = NextLatBatch(torch.tensor([[3, 4, 60, 6, 7, 8, 1, 1]]),
        torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
        torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]]))
    return model, recipe, batch


def test_adapter_metrics_exactly_match_f_streaming_measurements():
    model, recipe, batch = f_setup()
    batch = NextLatBatch(**{name: None if value is None else value[:1] for name, value in vars(batch).items()})
    with evaluation.evaluation_runtime(model):
        existing = fprobe.probe_batch(model, batch, recipe, passes=8)
        actual = curves.component_probe_batch(model, batch, recipe, passes=8)
    assert actual == existing


@pytest.mark.parametrize('arm', ['NF', 'NFR'])
def test_nf_nfr_ce_matches_canonical_losses_without_executing_predictor(arm, monkeypatch):
    model, recipe, batch = setup(arm)
    before = tree_digests(model.state_dict()); rng = torch.get_rng_state().clone()
    with evaluation.evaluation_runtime(model) as evidence:
        canonical = evaluation.per_pass_sums(model, batch, recipe)
        def forbidden(*args, **kwargs):
            raise AssertionError('No predictor or auxiliary objective is allowed in inference curves')
        monkeypatch.setattr(model.predictor, 'forward', forbidden)
        monkeypatch.setattr(model, 'loss_sums', forbidden)
        actual = curves.component_probe_batch(model, batch, recipe, passes=8)
    assert evidence['integrity_passed'] and before == tree_digests(model.state_dict())
    assert torch.equal(rng, torch.get_rng_state()) and all(p.grad is None for p in model.parameters())
    assert not model.predictor._forward_pre_hooks
    for index in range(4):
        observed = actual['passes'][index]['regions']['all']
        assert observed['ce_sum'] == pytest.approx(canonical['passes'][index]['sums']['ce'], rel=1e-6)
        assert observed['ce_targets'] == canonical['counts']['ce']
    assert actual['input_tokens'] == 6 and actual['ce_targets'] == 5


def test_forward_failure_restores_hooks_parameters_rng_and_runtime(monkeypatch):
    model, recipe, batch = setup('NFR')
    before = tree_digests(model.state_dict()); rng = torch.get_rng_state().clone()
    flags = {k: getattr(model.backbone.backbone, k) for k in evaluation.RUNTIME_FLAGS}
    original = model.backbone._stack; calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            torch.rand(2)
            raise RuntimeError('injected pass failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(model.backbone, '_stack', fail)
    with pytest.raises(RuntimeError, match='injected'):
        with evaluation.evaluation_runtime(model) as evidence:
            curves.component_probe_batch(model, batch, recipe, passes=8)
    assert before == tree_digests(model.state_dict()) and torch.equal(rng, torch.get_rng_state())
    assert evidence['integrity_passed']
    assert flags == {k: getattr(model.backbone.backbone, k) for k in evaluation.RUNTIME_FLAGS}
    assert not model.backbone.backbone._forward_pre_hooks
    assert not model.backbone.backbone.norm._forward_pre_hooks
    assert not model.predictor._forward_pre_hooks


@pytest.mark.parametrize('bad', ['outside', 'pass_limit', 'batch', 'precision'])
def test_unsupported_runtime_scope_fails_closed(bad):
    model, recipe, batch = setup('NF')
    if bad == 'outside':
        with pytest.raises(ValueError, match='evaluation runtime'):
            curves.component_probe_batch(model, batch, recipe)
        return
    if bad == 'batch':
        batch = NextLatBatch(**{k: v.repeat(2, 1) for k, v in vars(batch).items() if v is not None})
    with evaluation.evaluation_runtime(model):
        if bad == 'precision': model.backbone.backbone.attention_precision = 'mixed'
        with pytest.raises(ValueError):
            curves.component_probe_batch(model, batch, recipe, passes=33 if bad == 'pass_limit' else 4)


def authority(arm='NF', update=32, kl=None):
    recipe = CampaignRecipe(arm, document_policy='continuous-stream-v1')
    cursor = lambda k: {'manifest_sha256': 'a'*64, 'next_chunk': k*512, 'next_update': k, 'split': 'train'}
    plan = {'first_cursor': cursor(0), 'updates': [{'next_cursor': cursor(k)} for k in range(1, 129)]}
    spec = {'recipe': recipe, 'resolved': {'contract_sha256': 'b'*64}, 'plan': plan}
    sources = {'scripts/olmo_fbt_component_curves.py': curves.sha256_file(curves.ROOT/'scripts/olmo_fbt_component_curves.py')}
    operative = recipe.to_dict(); branch = None
    if kl is not None:
        operative = declared_recipe(recipe, kl_weight=kl, parent_manifest_sha256='c'*64)
        branch = {'parent_update': 32, 'review_stop': 64, 'kl_weight': kl,
                  'parent_manifest_sha256': 'c'*64, 'recipe_as_declared': operative}
    identity_payload = {'resolved_contract_sha256': 'b'*64, 'arm': arm, 'sources': sources,
                        'plan': plan, 'recipe': operative}
    identity = {'payload': identity_payload, 'sha256': curves.historical.contract.legacy.digest(identity_payload)}
    config = {'execution_identity': identity, 'recipe': operative}
    if branch is not None: config['objective_branch'] = branch
    state = {'filename': 'state.pt', 'size_bytes': 1, 'sha256': 'd'*64}
    manifest = {'counters': asdict(TrainingCounters(optimizer_updates=update, input_tokens=update*524288)),
        'world_size': 2, 'metadata': {'configuration': config, 'source_fingerprint': {'sha256': 'e'*64}},
        'rank_cursors': [{'rank': r, 'world_size': 2, 'physical_batch_per_rank': 12, 'cursor': cursor(update)} for r in range(2)],
        'state': state}
    report = {'schema': curves.kl_run.SCHEMA if kl is not None else 'olmo-pilot-async-execute-report-v1',
        'status': 'stopped_at_boundary', 'arm': arm,
        'declaration_sha256': curves.historical.DECL, 'resolved_sha256': curves.historical.RESOLVED,
        'configuration': config, 'source_fingerprint': {'sha256': 'e'*64}, 'sources': sources,
        'published_checkpoints': [{'manifest_sha256': 'f'*64, 'state': state}]}
    if branch is not None: report.update(branch=branch, parent_manifest_sha256='c'*64)
    return report, manifest, spec, sources


@pytest.mark.parametrize('arm,update,kl', [('NF', 0, None), ('NF', 32, None), ('NFR', 32, None), ('NF', 64, 1.), ('NF', 64, .1)])
def test_saved_conditions_and_both_nf64_objectives_authenticate(arm, update, kl, monkeypatch):
    report, manifest, spec, sources = authority(arm, update, kl)
    monkeypatch.setattr(curves, '_expected_report_sources', lambda _: sources)
    actual = curves.authenticate_report(report, manifest, spec, expected_manifest_sha256='f'*64)
    assert actual['after_update'] == update and actual['arm'] == arm
    assert actual['kl_weight'] == (1. if kl is None else kl)
    assert actual['optimizer_updates_performed'] == 0


@pytest.mark.parametrize('bad', ['update', 'exposure', 'reference_status', 'reference_config', 'source',
    'publication', 'duplicate_publication', 'identity', 'cursor', 'recipe', 'branch_parent', 'branch_weight'])
def test_other_or_corrupted_saved_lineage_is_rejected(bad, monkeypatch):
    report, manifest, spec, sources = authority(update=64, kl=1.)
    report = deepcopy(report); manifest = deepcopy(manifest)
    monkeypatch.setattr(curves, '_expected_report_sources', lambda _: sources)
    if bad == 'update': manifest['counters']['optimizer_updates'] = 16
    elif bad == 'exposure': manifest['counters']['input_tokens'] += 1
    elif bad == 'reference_status': report['status'] = 'running'
    elif bad == 'reference_config': report['configuration']['different'] = True
    elif bad == 'source': report['sources'] = {}
    elif bad == 'publication': report['published_checkpoints'] = []
    elif bad == 'duplicate_publication': report['published_checkpoints'] *= 2
    elif bad == 'identity':
        report['configuration']['execution_identity']['sha256'] = '0'*64
        manifest['metadata']['configuration'] = deepcopy(report['configuration'])
    elif bad == 'cursor': manifest['rank_cursors'][0]['cursor']['next_chunk'] += 1
    elif bad == 'recipe':
        report['configuration']['recipe']['auxiliary']['latent'] = .2
        manifest['metadata']['configuration'] = deepcopy(report['configuration'])
    elif bad == 'branch_parent': report['branch']['parent_update'] = 4
    elif bad == 'branch_weight': report['branch']['kl_weight'] = .2
    with pytest.raises(ValueError):
        curves.authenticate_report(report, manifest, spec, expected_manifest_sha256='f'*64)
