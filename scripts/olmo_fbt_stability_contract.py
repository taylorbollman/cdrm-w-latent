"""Explicit F-only lineage from the accepted fusion128/ordered NF startup.

The parent is authenticated through its unchanged resolver. This is a new
weights-only architecture selection, never an NF checkpoint migration.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.fbt_training import FBTNextLatLM
from scripts import olmo_campaign_manifest as legacy
from scripts import olmo_pilot_execution_contract as accepted
from scripts import olmo_pilot_async_execute as accepted_execution
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from scripts.olmo_pilot_eval_control import resolve_evaluation

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'olmo-fbt-stability-declaration-v1'
RESOLVED_SCHEMA = 'olmo-fbt-stability-resolved-v1'
PROTOCOL = 'docs/reports/olmo-fbt-stability/protocol.md'


def source_hashes():
    frozen = json.loads((ROOT/'.runtime/olmo-pilot-async/runtime-sources.json').read_text())
    if len(frozen) != 200 or accepted_execution.source_hashes() != frozen:
        raise ValueError('Historical accepted 200-source runtime changed')
    names = ('scripts/olmo_fbt_stability_contract.py', 'scripts/olmo_fbt_stability_execute.py',
        'scripts/olmo_fbt_stability_engine.py', 'scripts/olmo_fbt_stability_probe.py',
        'tests/test_fbt_stability_execution.py', 'tests/test_fbt_stability_probe.py', 'tests/test_fbt_stability_probe_control.py', PROTOCOL)
    return dict(sorted((frozen | {n: sha256_file(ROOT/n) for n in names}).items()))


def validate_declaration(declaration):
    legacy.exact_fields(declaration, ('schema', 'parent_declaration', 'parent_resolved',
        'parent_report', 'updates', 'initial_stop_after', 'storage_prefix', 'probe'), 'F stability declaration')
    if declaration['schema'] != SCHEMA or declaration['updates'] != 192 or declaration['initial_stop_after'] != 128:
        raise ValueError('F-only study declares 192 ceiling and initial 128 review boundary')
    for name in ('parent_declaration', 'parent_resolved', 'parent_report'):
        legacy.exact_fields(declaration[name], ('path', 'sha256'), name)
        legacy.local_path(declaration[name]['path']); legacy.pin(declaration[name]['sha256'])
    prefix = declaration['storage_prefix']
    from scripts.olmo_campaign_loop_run import storage_location
    storage_location(prefix)
    if declaration['probe'] != {'panel_rows': 8, 'physical_batch': 1}:
        raise ValueError('Use the bounded shared eight-row probe panel')


def check_shared_prefix(new_plan, old_plan, old_arm='NF', new_arm='F'):
    if len(old_plan['updates']) != 128 or len(new_plan['updates']) != 192:
        raise ValueError('Require explicit 128 to 192 ordered plan declaration')
    for new, old in zip(new_plan['updates'][:128], old_plan['updates']):
        old = deepcopy(old); new = deepcopy(new)
        old_alloc = old.pop('allocation_by_arm'); new_alloc = new.pop('allocation_by_arm')
        if new != old or new_alloc != {new_arm: old_alloc[old_arm]}:
            raise ValueError('F-only plan changed original data membership/physical prefix')
    if new_plan['valid_token_prefix'][:129] != old_plan['valid_token_prefix']:
        raise ValueError('F-only plan changed original token exposure')
    return {'shared_128_data_membership_exact': True, 'shared_128_allocation_exact': True,
            'shared_128_token_prefix_exact': True}


def resolve(declaration):
    validate_declaration(declaration)
    ref, resolved_ref = declaration['parent_declaration'], declaration['parent_resolved']
    parent, resolved = accepted.load_declaration(ref['path'], ref['sha256'],
        resolved_ref['path'], resolved_ref['sha256'])
    baseline = legacy.read_json(declaration['parent_report']['path'], declaration['parent_report']['sha256'], limit=128*1024**2)
    if (baseline.get('schema') != accepted_execution.SCHEMA or baseline.get('arm') != 'NF'
            or baseline.get('status') not in ('completed_plan', 'stopped_at_boundary')
            or baseline.get('declaration_sha256') != ref['sha256']
            or baseline.get('resolved_sha256') != resolved_ref['sha256']
            or baseline.get('sources') != accepted_execution.source_hashes()):
        raise ValueError('Require authentic successful NF cohort report')
    old_manifest = parent['planning_manifest']
    if (old_manifest['arms'] != ['NF', 'NFR'] or old_manifest['budget']['updates'] != 128
            or old_manifest['partition'] != {'world_size': 2, 'physical_batch_per_rank': {'NF': 12, 'NFR': 12}}
            or parent['startup']['kind'] != accepted.ADAPTED):
        raise ValueError('Require original two-rank B12 NF128 cohort declaration')
    old_recipe = accepted.recipe_from_dict(resolved['planning']['recipes']['NF'])
    recipe = replace(old_recipe, arm='F')
    expected = CampaignRecipe('F', document_policy='continuous-stream-v1')
    if recipe != expected:
        raise ValueError('F-only study preserves the standard current K4 recipe')
    manifest = deepcopy(old_manifest)
    manifest.update(label='fbt-only-stability-192', arms=['F'])
    manifest['budget']['updates'] = 192
    manifest['partition']['physical_batch_per_rank'] = {'F': 12}
    manifest['retention'].update(storage_prefix=declaration['storage_prefix'], checkpoint_every_updates=16,
        checkpoint_seconds=600, keep_local_completed=2)
    manifest['tracking']['group'] = 'fbt-only-stability'
    manifest['evaluation']['physical_batch_by_arm'] = {'F': old_manifest['evaluation']['physical_batch_by_arm']['NF']}
    manifest['evaluation']['every_updates'] = 16
    data = manifest['data']
    with OrderedCampaignData(legacy.local_path(data['corpus']), legacy.local_path(data['index'])) as reader:
        plan = accepted.plan_updates(reader, manifest['budget'], {'F': (2, 12)})
    checks = check_shared_prefix(plan, resolved['planning']['plan'])
    schedule = {'plan_sha256': legacy.digest({'tokens': [r['counts']['valid_tokens'] for r in plan['updates']],
        'warmup_tokens': recipe.warmup_tokens, 'start_fraction': recipe.warmup_start_fraction}),
        'valid_token_prefix': plan['valid_token_prefix'],
        'lr_at_completed_boundaries': [recipe.plateau_lr*(recipe.warmup_start_fraction +
            (1-recipe.warmup_start_fraction)*min(1, n/recipe.warmup_tokens)) for n in plan['valid_token_prefix']]}
    if schedule['lr_at_completed_boundaries'][:129] != resolved['planning']['schedule']['lr_at_completed_boundaries']:
        raise ValueError('F-only plan changed original LR prefix')
    checks['shared_128_lr_prefix_exact'] = True
    evaluation = resolve_evaluation(data, manifest['evaluation'], length=1024, updates=192, partitions={'F': (2, 12)})
    evaluation['scheduled_updates'] = sorted(set(evaluation['scheduled_updates']) | {0, 100})
    from scripts.olmo_fbt_stability_probe import resolve_probe_plan
    probe = resolve_probe_plan(data, length=1024, updates=192, world_size=2, **declaration['probe'])
    ledger = old_manifest['resource_ledger']
    cards = legacy.resource_cards({'F': recipe}, plan,
        legacy.read_json(legacy.local_path(ledger['path']), ledger['sha256']))
    historical_import_plan = accepted.startup_plan(resolved, 'NF')
    startup = deepcopy(historical_import_plan)
    startup['historical_import_plan'] = historical_import_plan
    startup.update(kind='fusion128-fresh-F-all-adam-v1', target_recipe=recipe.to_dict(),
        target_mode=accepted.plain(asdict(recipe.mode())),
        target_nextlat_config=legacy.nextlat_config(recipe).to_dict(),
        transition={'kind':'strict-NF-import-then-F-wrapper-v1', 'predictor':'removed-before-optimizer',
            'backbone_and_fusion':'same_module_and_parameter_objects', 'optimizer':'fresh_all_active_adamw'})
    payload = {'schema':RESOLVED_SCHEMA, 'declaration':declaration, 'manifest':manifest,
        'planning': {'plan':plan, 'schedule':schedule, 'resource_cards':cards,
            'model_checkpoint_authority':resolved['planning']['model_checkpoint_authority'],
            'native_backbone_config':resolved['planning']['native_backbone_config']},
        'recipe':recipe.to_dict(), 'startup':startup, 'evaluation_plan':evaluation, 'probe_plan':probe,
        'checkpoint_updates':sorted(set(range(16,193,16))|{100}), 'prefix_checks':checks,
        'parent_contract_sha256':resolved['contract_sha256'], 'sources':source_hashes()}
    return {**accepted.plain(payload), 'contract_sha256':legacy.digest(payload)}


def import_f_wrapper(model, historical_recipe, recipe):
    """Keep exactly the imported core; omit predictor before creating fresh Adam."""
    if (historical_recipe.arm != 'NF' or historical_recipe.document_policy != 'isolated-v1'
            or recipe.arm != 'F' or recipe.document_policy != 'continuous-stream-v1'
            or not model.enabled or model.predictor is None or model.config.document_policy != 'isolated-v1'
            or model.pass_loss_policy != 'campaign_v1' or recipe.mode().rt_mode.selected_layers
            or historical_recipe.fusion_seed != recipe.fusion_seed or historical_recipe.predictor_seed != recipe.predictor_seed
            or historical_recipe.feedback_jitter != recipe.feedback_jitter):
        raise ValueError('Require strict historical NF import followed by explicit F-only construction')
    core = model.backbone
    before = tree_digests(core.state_dict())
    identities = {n:id(p) for n,p in core.named_parameters()}
    modes = {n:m.training for n,m in core.named_modules()}
    result = FBTNextLatLM(core, replace(model.config, document_policy=recipe.document_policy),
        enabled=False, pass_loss_policy='campaign_v1')
    result.train(model.training)
    checks = {'core_state_exact':tree_digests(result.backbone.state_dict()) == before,
        'core_parameter_identities_exact':identities == {n:id(p) for n,p in result.backbone.named_parameters()},
        'core_modes_exact':modes == {n:m.training for n,m in result.backbone.named_modules()},
        'predictor_absent':result.predictor is None and not result.enabled,
        'no_predictor_parameters':not any(n.startswith('predictor.') for n,_ in result.named_parameters()),
        'all_active_trainable':all(p.requires_grad for p in result.parameters()),
        'tied_readout_preserved':result.backbone.readout_weight is result.backbone.token_embeddings.weight,
        'only_document_policy_changed':replace(result.config, document_policy='isolated-v1') == model.config}
    if not all(checks.values()): raise ValueError('F-only wrapper changed retained core or ownership')
    return result, {'checks':checks, 'historical_recipe':historical_recipe.to_dict(),
        'target_recipe':recipe.to_dict(), 'loaded_optimizer':False,
        'transition':'Strict NF fusion128 import; retained identical core and removed predictor; new all-active Adam'}
