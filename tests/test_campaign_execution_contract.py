"""CPU metadata/authority tests; no native model or serialized tensor fixture."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from scripts import olmo_campaign_execution_contract as contract
from scripts import olmo_campaign_manifest as legacy
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from test_campaign_manifest import dictionary, corpus, resolved_fixture


def declaration(manifest):
    return {'schema': contract.SCHEMA, 'planning_manifest': copy.deepcopy(manifest),
        'startup': copy.deepcopy(legacy.STARTUP), 'implementation_sources': contract.source_hashes()}


def authority_fixture(tmp_path, monkeypatch, sources):
    """Explicit miniature byte authority, with the same saved metadata topology.

    Only the immutable production constants are replaced for this test. The
    reader, hash comparisons, receipt guards and exact exposure logic are real.
    No tiny authority option is exposed through the production API.
    """
    checkpoint = tmp_path/'fusion.pt'; checkpoint.write_bytes(b'not a tensor file; hashing only')
    monkeypatch.setattr(contract, 'FUSION_CHECKPOINT_SHA', legacy.sha256_file(checkpoint))
    monkeypatch.setattr(contract, 'FUSION_CHECKPOINT_BYTES', checkpoint.stat().st_size)
    recipe = CampaignRecipe('NF')
    state = {'state_proj.weight': {'sha256': 'a'*64}, 'token_gate.weight': {'sha256': 'b'*64},
             'output_scale': {'sha256': 'c'*64}}
    cursor = {'schema': 'olmo-fusion-startup-data-v1', 'manifest_sha256': 'd'*64, 'next_update': 128}
    config = {'kind': 'olmo-fusion-startup-v1', 'recipe': recipe.to_dict(),
        'initial_full_trainability_contract': {'mode': contract.plain(asdict(recipe.mode()))},
        'source_checkpoint': {'sha256': legacy.CHECKPOINT_SHA256}, 'sources': sources,
        'model_config': legacy.OLMoConfig.native_1b().to_dict(), 'fusion_config': {'seed': recipe.fusion_seed},
        'nextlat_config': legacy.nextlat_config(recipe).to_dict(), 'frozen_state_pins': {'native': 'e'*64}}
    checkpoint_row = {'optimizer_updates': 128, 'sha256': contract.FUSION_CHECKPOINT_SHA,
        'size_bytes': checkpoint.stat().st_size, 'data_cursor': cursor,
        'boundary_digests': {'counters': contract.EXPOSURE, 'fusion': state},
        'storage': {'generation': '123', 'sha256': contract.FUSION_CHECKPOINT_SHA,
            'size_bytes': checkpoint.stat().st_size, 'verification': {'download_sha256': True}}}
    report = {'schema': 'olmo-fusion-startup-run-v1', 'status': 'completed_segment', 'passed': True,
        'configuration': config, 'sources': sources, 'integrity': {'frozen_state_exact': True},
        'counters': contract.EXPOSURE, 'data_cursor': cursor, 'source_fingerprint': {'code': sources},
        'checkpoints': [checkpoint_row]}
    report_path = tmp_path/'report.json'; report_path.write_text(json.dumps(report))
    monkeypatch.setattr(contract, 'FUSION_REPORT_SHA', legacy.sha256_file(report_path))
    startup = {**copy.deepcopy(contract.ADAPTED_POLICY),
        'checkpoint': {'path': str(checkpoint), 'sha256': contract.FUSION_CHECKPOINT_SHA},
        'report': {'path': str(report_path), 'sha256': contract.FUSION_REPORT_SHA}}
    return startup, report


def make_identity(result, arm='NFR', **changes):
    return contract.execution_identity(result, arm, runtime=changes.get('runtime', {'torch': 'test-only'}),
        determinism=changes.get('determinism', {'enabled': True}),
        model_contract=changes.get('model_contract', {'tied_readout': True, 'scope': 'metadata fixture'}),
        extra_sources=changes.get('extra_sources'))


def committed(identity, completed):
    p = identity['payload']; plan = p['plan']; world = p['partition']['world_size']
    cursor = plan['first_cursor'] if not completed else plan['updates'][completed-1]['next_cursor']
    return {'schema': 'olmo-replicated-ddp-checkpoint-v1', 'world_size': world,
        'metadata': {'world_size': world, 'configuration': {'execution_identity': identity},
            'source_fingerprint': {'execution_identity_sha256': identity['sha256']}},
        'state': {'filename': 'state.pt', 'size_bytes': 999, 'sha256': 'f'*64},
        'counters': contract.expected_counters(identity, completed),
        'rank_cursors': [{'schema': contract.CURSOR_SCHEMA, 'rank': rank, 'world_size': world,
            'physical_batch_per_rank': p['partition']['physical_batch_per_rank'], 'cursor': cursor}
            for rank in range(world)]}


def test_all_eight_resolution_reuses_exact_old_plan_without_model_or_tensor_load(resolved_fixture, monkeypatch):
    manifest, _ = resolved_fixture
    proposed = declaration(manifest); before = copy.deepcopy(proposed)
    old_result = legacy.resolve(manifest)
    def forbidden(*a, **kw): raise AssertionError('Unexpected model/tensor/GPU construction')
    with monkeypatch.context() as guard:
        guard.setattr(torch.nn.Module, '__init__', forbidden)
        guard.setattr(torch.optim.AdamW, '__init__', forbidden)
        guard.setattr(torch, 'load', forbidden)
        guard.setattr(torch.cuda, 'init', forbidden)
        result = contract.resolve(proposed)
    assert proposed == before and result['planning'] == old_result
    assert result['actual_startup'] == legacy.STARTUP and result['fusion_authority'] is None
    assert result['launch_authorized'] is result['numerical_clearance'] is False
    assert set(result['planning']['recipes']) == set(legacy.ARMS)
    for arm in legacy.ARMS:
        plan = contract.startup_plan(result, arm)
        assert plan['target_recipe'] == old_result['recipes'][arm]
        assert plan['restore_optimizer'] is plan['restore_rng'] is False
        assert 'import_recipe' not in plan


@pytest.mark.parametrize('change', ['unknown', 'outer_source', 'legacy_source', 'unsupported_kind',
    'inherited_optimizer', 'base_startup_rewrite', 'bad_version'])
def test_declarations_reject_unsupported_or_ambiguous_claims(change):
    manifest = dictionary(); manifest['implementation_sources'] = legacy.source_hashes()
    proposed = declaration(manifest)
    if change == 'unknown': proposed['resume_report'] = 'not required'
    elif change == 'outer_source': proposed['implementation_sources'] = {}
    elif change == 'legacy_source': proposed['planning_manifest']['implementation_sources'] = {}
    elif change == 'unsupported_kind': proposed['startup'] = {'kind': 'nfr-update20-inherited-adam'}
    elif change == 'inherited_optimizer': proposed['startup']['inherit_adam'] = True
    elif change == 'base_startup_rewrite': proposed['planning_manifest']['startup']['kind'] = contract.ADAPTED
    else: proposed['schema'] += '-unknown'
    with pytest.raises(ValueError): contract.validate_declaration(proposed)


def test_selected_adapted_origin_resolves_separately_and_weights_only_receipt_matches(resolved_fixture, tmp_path, monkeypatch):
    manifest, _ = resolved_fixture
    manifest['arms'] = ['NF', 'NFR']; manifest['partition']['physical_batch_per_rank'] = {'NF': 3, 'NFR': 3}
    proposed = declaration(manifest)
    proposed['startup'], old = authority_fixture(tmp_path, monkeypatch, proposed['implementation_sources'])
    result = contract.resolve(proposed)
    assert result['planning']['declarations']['startup'] == legacy.STARTUP
    assert result['actual_startup']['kind'] == contract.ADAPTED
    assert result['fusion_authority']['prior_exposure']['input_tokens'] == 1073565
    for arm in ('NF', 'NFR'):
        plan = contract.startup_plan(result, arm)
        assert contract.recipe_from_dict(plan['import_recipe']).document_policy == 'isolated-v1'
        assert plan['target_recipe']['document_policy'] == 'continuous-stream-v1'
        assert plan['transition']['rt_layers'] == ([0, 15] if arm == 'NFR' else [])
        assert plan['optimizer_state'] == 'fresh_all_active_adamw'
        receipt = {'checkpoint_sha256': contract.FUSION_CHECKPOINT_SHA,
            'configuration': old['configuration'], 'source_fingerprint': old['source_fingerprint'],
            'counters': old['counters'], 'data_cursor': old['data_cursor'],
            'fusion_state_pins': old['checkpoints'][0]['boundary_digests']['fusion'],
            'restore_scope': 'fusion weights only; diagnostic flags and RNG preserved',
            'checks': {k: True for k in ('complete_fusion_exact', 'parameter_identities_preserved',
                'trainability_preserved', 'module_modes_preserved', 'frozen_state_exact', 'tied_readout_preserved')}}
        contract.validate_import_receipt(receipt, plan)
        for key in ('configuration', 'source_fingerprint', 'counters', 'data_cursor', 'fusion_state_pins', 'restore_scope', 'checks'):
            corrupt = copy.deepcopy(receipt); corrupt[key] = {} if key != 'restore_scope' else 'fusion+Adam'
            with pytest.raises(ValueError): contract.validate_import_receipt(corrupt, plan)


@pytest.mark.parametrize('change', ['arm', 'checkpoint_pin', 'report_pin', 'prior_exposure', 'optimizer', 'clock',
    'fusion_seed', 'predictor_seed', 'jitter', 'checkpoint_bytes', 'report_bytes'])
def test_adapted_authority_and_branch_changes_rejected(resolved_fixture, tmp_path, monkeypatch, change):
    manifest, _ = resolved_fixture
    manifest['arms'] = ['NFR']; manifest['partition']['physical_batch_per_rank'] = {'NFR': 3}
    proposed = declaration(manifest)
    proposed['startup'], _ = authority_fixture(tmp_path, monkeypatch, proposed['implementation_sources'])
    if change == 'arm': manifest['arms'] = ['FR']; manifest['partition']['physical_batch_per_rank'] = {'FR': 3}; proposed['planning_manifest'] = manifest
    elif change == 'checkpoint_pin': proposed['startup']['checkpoint']['sha256'] = '0'*64
    elif change == 'report_pin': proposed['startup']['report']['sha256'] = '0'*64
    elif change == 'prior_exposure': proposed['startup']['prior_exposure']['input_tokens'] += 1
    elif change == 'optimizer': proposed['startup']['optimizer'] = 'inherit_fusion_adam'
    elif change == 'clock': proposed['startup']['clocks'] = 'continue_128'
    elif change in ('fusion_seed', 'predictor_seed'): proposed['planning_manifest']['recipe'][change] += 1
    elif change == 'jitter': proposed['planning_manifest']['recipe']['feedback_jitter'] = 0
    elif change == 'checkpoint_bytes': Path(proposed['startup']['checkpoint']['path']).write_bytes(b'corruption')
    else: Path(proposed['startup']['report']['path']).write_text('{}')
    with pytest.raises(ValueError): contract.resolve(proposed)


def test_pinned_re_resolution_and_no_silent_source_or_resolved_changes(resolved_fixture, tmp_path):
    manifest, _ = resolved_fixture; proposed = declaration(manifest)
    path = tmp_path/'declaration.json'; path.write_text(json.dumps(proposed))
    _, result = contract.load_declaration(path, legacy.sha256_file(path))
    resolved = tmp_path/'resolved.json'; resolved.write_text(json.dumps(result))
    assert contract.load_declaration(path, legacy.sha256_file(path), resolved, legacy.sha256_file(resolved))[1] == result
    corrupt = copy.deepcopy(result); corrupt['actual_startup']['kind'] = 'different'
    resolved.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError): contract.load_declaration(path, legacy.sha256_file(path), resolved, legacy.sha256_file(resolved))
    with pytest.raises(ValueError): contract.load_declaration(path, legacy.sha256_file(path), resolved, None)
    with pytest.raises(ValueError): contract.startup_plan(corrupt, 'NFR')


def test_identity_is_json_stable_and_binds_actual_runtime_and_owned_sources(resolved_fixture):
    proposed = declaration(resolved_fixture[0]); result = contract.resolve(proposed)
    identity = make_identity(result)
    assert identity == make_identity(json.loads(json.dumps(result)))
    assert identity['sha256'] != make_identity(result, runtime={'torch': 'another-runtime'})['sha256']
    assert identity['sha256'] != make_identity(result, model_contract={'tied_readout': False})['sha256']
    assert identity['sha256'] != make_identity(result, arm='NF')['sha256']
    assert 'observer_mode' not in identity['payload'] and 'stop_update' not in identity['payload']
    for sources in ({'../escape': 'a'*64}, {'missing.py': 'a'*64}, {'scripts/olmo_campaign_manifest.py': 'a'*64}):
        with pytest.raises((ValueError, FileNotFoundError)): make_identity(result, extra_sources=sources)


@pytest.mark.parametrize('arm', legacy.ARMS)
def test_generic_resume_accepts_initial_stopped_and_completed_boundaries_without_report(resolved_fixture, arm):
    result = contract.resolve(declaration(resolved_fixture[0])); identity = make_identity(result, arm)
    for completed in (0, 1, 2):
        manifest = committed(identity, completed)
        answer = contract.validate_resume_metadata(manifest, identity)
        assert answer['completed_updates'] == completed and answer['remaining_updates'] == 2-completed
        assert answer['plan_complete'] == (completed == 2)
        assert answer['completed_reference_report_required'] is False
        assert answer['requires_generic_tensor_optimizer_rng_validation'] is True
        counts = manifest['counters']
        assert counts['input_tokens'] == 2048*completed
        assert counts['documents'] == 2*completed
        assert counts['microbatches'] == 2*completed
        assert bool(counts['latent_pairs']) == bool(completed and 'N' in arm)
        assert counts['ce_positions'] == 2046*completed


@pytest.mark.parametrize('change', ['identity', 'fingerprint', 'world', 'counter', 'counter_type', 'float_count', 'cursor', 'rank',
    'cursor_schema', 'batch', 'state', 'outside_plan', 'missing_rank'])
def test_resume_metadata_rejects_changed_lineage_or_incomplete_boundary(resolved_fixture, change):
    identity = make_identity(contract.resolve(declaration(resolved_fixture[0])))
    manifest = committed(identity, 1)
    if change == 'identity': manifest['metadata']['configuration']['execution_identity'] = make_identity(contract.resolve(declaration(resolved_fixture[0])), arm='B')
    elif change == 'fingerprint': manifest['metadata']['source_fingerprint']['execution_identity_sha256'] = '0'*64
    elif change == 'world': manifest['world_size'] = 3
    elif change == 'counter': manifest['counters']['input_tokens'] += 1
    elif change == 'counter_type': manifest['counters']['optimizer_updates'] = True
    elif change == 'float_count': manifest['counters']['input_tokens'] = float(manifest['counters']['input_tokens'])
    elif change == 'cursor': manifest['rank_cursors'][0]['cursor'] = {**manifest['rank_cursors'][0]['cursor'], 'next_chunk': 0}
    elif change == 'rank': manifest['rank_cursors'][0]['rank'] = 1
    elif change == 'cursor_schema': manifest['rank_cursors'][0]['schema'] = 'old-acceptance-cursor'
    elif change == 'batch': manifest['rank_cursors'][0]['physical_batch_per_rank'] += 1
    elif change == 'state': manifest['state']['filename'] = '../state.pt'
    elif change == 'outside_plan': manifest['counters']['optimizer_updates'] = 3
    else: manifest['rank_cursors'].pop()
    with pytest.raises(ValueError): contract.validate_resume_metadata(manifest, identity)
