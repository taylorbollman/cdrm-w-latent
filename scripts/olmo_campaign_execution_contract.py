#!/usr/bin/env python3
"""Versioned CPU planning/startup identities, separate from an execution launcher.

The legacy resolver remains unchanged. Its original-weight planning manifest is
an explicit construction/data subdocument, not the authority for adapted startup.
This module never constructs a model, loads tensors, initializes CUDA or transfers
objects. Actual import and distributed tensor validation remain mandatory.
"""
from __future__ import annotations

from dataclasses import asdict, fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from scripts import olmo_campaign_manifest as legacy
from scripts.olmo_fusion_startup_train import source_hashes as startup_sources

SCHEMA = 'olmo-campaign-execution-declaration-v1'
RESOLVED_SCHEMA = 'olmo-campaign-execution-resolved-v1'
IDENTITY_SCHEMA = 'olmo-campaign-execution-identity-v1'
CURSOR_SCHEMA = 'olmo-campaign-execution-cursor-v1'
ADAPTED = 'fusion128-fresh-all-adam-v1'
PROTOCOL = 'docs/reports/olmo-campaign-execution/contract.md'
FUSION_CHECKPOINT_SHA = '892ff2fdcdeec89e3008a16a12e91158250ebe05adfe0e9efce8f153409b8cfc'
FUSION_CHECKPOINT_BYTES = 103240258
FUSION_REPORT_SHA = '79a148ff18af694fca542b7f02eaca5d3005ab8fd4c140a369071099eb8b3edc'
EXPOSURE = {'optimizer_updates': 128, 'input_tokens': 1073565, 'ce_positions': 1048576,
            'latent_pairs': 0, 'kl_triples': 0, 'documents': 9235, 'microbatches': 1219}
ADAPTED_POLICY = {'kind': ADAPTED, 'weights': 'complete_fusion_only',
    'optimizer': 'fresh_all_active_adamw', 'clocks': 'reset_new_lineage',
    'data_origin': 'packed_prefix_zero', 'all_active_parameters_trainable': True,
    'prior_exposure': EXPOSURE}


def plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def source_hashes():
    result = dict(legacy.source_hashes())
    for name, digest in startup_sources().items():
        if name in result and result[name] != digest:
            raise ValueError('Historical importer/resolver source inventories disagree')
        result[name] = digest
    for name in ('scripts/olmo_campaign_execution_contract.py',
                 'tests/test_campaign_execution_contract.py', PROTOCOL):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def recipe_from_dict(value):
    recipe = CampaignRecipe(**{f.name: value[f.name] for f in fields(CampaignRecipe) if f.name in value})
    if recipe.to_dict() != value:
        raise ValueError('Serialized recipe differs from its exact supported interpretation')
    return recipe


def validate_declaration(declaration, sources=None):
    legacy.exact_fields(declaration, ('schema', 'planning_manifest', 'startup', 'implementation_sources'), 'execution declaration')
    if declaration['schema'] != SCHEMA:
        raise ValueError('Unsupported execution declaration version')
    sources = source_hashes() if sources is None else sources
    if declaration['implementation_sources'] != sources:
        raise ValueError('Execution implementation source inventory differs')
    manifest = declaration['planning_manifest']
    recipes = legacy.validate_manifest(manifest, legacy.source_hashes())
    startup = declaration['startup']
    if startup == legacy.STARTUP:
        return recipes
    legacy.exact_fields(startup, set(ADAPTED_POLICY)|{'checkpoint', 'report'}, 'adapted startup')
    if any(startup[k] != v or type(startup[k]) is not type(v) for k, v in ADAPTED_POLICY.items()):
        raise ValueError('Only selected fusion128 weights with fresh all-active Adam and reset clocks are supported')
    if any(type(v) is not int for v in startup['prior_exposure'].values()):
        raise ValueError('Historical exposure counters must remain integer counts')
    if any(arm not in ('NF', 'NFR') for arm in recipes):
        raise ValueError('Fusion128 startup supports NF/NFR only')
    for name, digest in (('checkpoint', FUSION_CHECKPOINT_SHA), ('report', FUSION_REPORT_SHA)):
        legacy.exact_fields(startup[name], ('path', 'sha256'), 'adapted '+name)
        legacy.local_path(startup[name]['path'])
        if startup[name]['sha256'] != digest:
            raise ValueError('Adapted startup must pin the selected historical '+name)
    return recipes


def load_fusion_authority(startup, recipes, sources):
    """Read the pinned historical JSON and hash compact bytes, without loading tensors."""
    report = legacy.read_json(legacy.local_path(startup['report']['path']), FUSION_REPORT_SHA)
    checkpoint = legacy.local_path(startup['checkpoint']['path'])
    if (checkpoint.is_symlink() or not checkpoint.is_file()
            or checkpoint.stat().st_size != FUSION_CHECKPOINT_BYTES
            or sha256_file(checkpoint) != FUSION_CHECKPOINT_SHA):
        raise ValueError('Selected fusion checkpoint bytes differ')
    config = report.get('configuration', {})
    historical = recipe_from_dict(config.get('recipe', {}))
    records = [r for r in report.get('checkpoints', []) if r.get('optimizer_updates') == 128]
    if (report.get('schema') != 'olmo-fusion-startup-run-v1'
            or report.get('status') != 'completed_segment' or report.get('passed') is not True
            or report.get('counters') != EXPOSURE or not report.get('integrity')
            or not all(v is True for v in report['integrity'].values())
            or config.get('kind') != 'olmo-fusion-startup-v1'
            or config.get('source_checkpoint', {}).get('sha256') != legacy.CHECKPOINT_SHA256
            or config.get('sources') != report.get('sources')
            or not report.get('sources') or any(sources.get(k) != v for k, v in report['sources'].items())
            or historical.arm != 'NF' or historical.document_policy != 'isolated-v1'
            or plain(asdict(historical.mode())) != config.get('initial_full_trainability_contract', {}).get('mode')
            or len(records) != 1):
        raise ValueError('Historical fusion report authority/configuration/exposure differs')
    record = records[0]
    retained = record.get('storage', {})
    if (record.get('sha256') != FUSION_CHECKPOINT_SHA or record.get('size_bytes') != FUSION_CHECKPOINT_BYTES
            or record.get('boundary_digests', {}).get('counters') != EXPOSURE
            or record.get('data_cursor') != report.get('data_cursor')
            or retained.get('sha256') != FUSION_CHECKPOINT_SHA or retained.get('size_bytes') != FUSION_CHECKPOINT_BYTES
            or not retained.get('generation') or not retained.get('verification')
            or not all(v is True for v in retained['verification'].values())):
        raise ValueError('Historical saved boundary/retention authority differs')
    for recipe in recipes.values():
        if (recipe.fusion_seed != historical.fusion_seed or recipe.predictor_seed != historical.predictor_seed
                or recipe.feedback_jitter != historical.feedback_jitter):
            raise ValueError('Adapted fusion requires the saved branch seeds and feedback jitter')
    return plain({'historical_recipe': historical.to_dict(), 'historical_mode': plain(asdict(historical.mode())),
        'model_config': config['model_config'], 'fusion_config': config['fusion_config'],
        'nextlat_config': config['nextlat_config'], 'source_checkpoint': config['source_checkpoint'],
        'historical_configuration_sha256': legacy.digest(config),
        'historical_source_fingerprint_sha256': legacy.digest(report['source_fingerprint']),
        'historical_sources': report['sources'], 'frozen_state_pins': config['frozen_state_pins'],
        'fusion_state_pins': record['boundary_digests']['fusion'], 'prior_exposure': EXPOSURE,
        'data_cursor': record['data_cursor'], 'retained_checkpoint': retained,
        'prior_exposure_scope': 'Historical CE-only fusion adaptation; documents are presentations, not unique documents. New clocks reset; comparison with untouched weights is not exposure matched.'})


def resolve(declaration):
    sources = source_hashes()
    recipes = validate_declaration(declaration, sources)
    planning = legacy.resolve(declaration['planning_manifest'])
    authority = (load_fusion_authority(declaration['startup'], recipes, sources)
                 if declaration['startup']['kind'] == ADAPTED else None)
    if source_hashes() != sources:
        raise ValueError('Execution sources changed during CPU resolution')
    payload = {'schema': RESOLVED_SCHEMA, 'status': 'cpu_execution_contract_validated_not_authorized',
        'launch_authorized': False, 'numerical_clearance': False,
        'declaration': plain(declaration), 'sources': sources, 'planning': planning,
        'actual_startup': plain(declaration['startup']), 'fusion_authority': authority,
        'planning_scope': 'Legacy original-weight construction/data plan only; actual_startup is authoritative for execution origin.',
        'resume_scope': 'Same immutable execution identity and committed distributed boundary; no completed reference report required. Generic tensor/Adam/RNG validation is still mandatory.'}
    return {**payload, 'contract_sha256': legacy.digest(payload)}


def validate_resolved(resolved):
    if not isinstance(resolved, dict) or resolved.get('schema') != RESOLVED_SCHEMA:
        raise ValueError('Unsupported resolved execution contract')
    payload = {k: v for k, v in resolved.items() if k != 'contract_sha256'}
    if (resolved.get('contract_sha256') != legacy.digest(payload)
            or resolved.get('launch_authorized') is not False or resolved.get('numerical_clearance') is not False
            or resolved.get('actual_startup') != resolved.get('declaration', {}).get('startup')):
        raise ValueError('Resolved execution identity was changed')


def load_declaration(path, sha256, resolved_path=None, resolved_sha256=None):
    if (resolved_path is None) != (resolved_sha256 is None):
        raise ValueError('Resolved path and independent SHA must be supplied together')
    declaration = legacy.read_json(path, sha256)
    result = resolve(declaration)
    if resolved_path is not None and legacy.read_json(resolved_path, resolved_sha256, limit=128*1024**2) != result:
        raise ValueError('Pinned resolved execution contract differs from independent CPU resolution')
    if sha256_file(Path(path)) != sha256:
        raise ValueError('Execution declaration changed during resolution')
    return declaration, result


def startup_plan(resolved, arm):
    validate_resolved(resolved)
    planning = resolved['planning']
    if arm not in planning['recipes']:
        raise ValueError('Arm is not declared in the resolved plan')
    recipe = recipe_from_dict(planning['recipes'][arm])
    plan = {'kind': resolved['actual_startup']['kind'], 'target_recipe': recipe.to_dict(),
        'target_mode': plain(asdict(recipe.mode())), 'target_nextlat_config': planning['nextlat_configs'][arm],
        'optimizer_state': 'fresh_all_active_adamw', 'scheduler_and_counters': 'reset_zero',
        'data_origin': planning['plan']['first_cursor'], 'restore_optimizer': False,
        'restore_rng': False, 'require_new_origin_checkpoint': True,
        'original_checkpoint': planning['model_checkpoint_authority']}
    if plan['kind'] == ADAPTED:
        plan.update(import_recipe=resolved['fusion_authority']['historical_recipe'],
            checkpoint=resolved['actual_startup']['checkpoint'],
            report=resolved['actual_startup']['report'], expected_import=resolved['fusion_authority'],
            transition={'kind': 'weights-preserving-isolated-NF-to-declared-packed-arm-v1',
                'from_document_policy': 'isolated-v1', 'to_document_policy': recipe.document_policy,
                'from_arm': 'NF', 'to_arm': arm, 'rt_layers': list(recipe.mode().rt_mode.selected_layers),
                'allowed_state_changes': [], 'active_parameters': 'all',
                'historical_precision': 'fp32_fusion_training',
                'new_precision': resolved['declaration']['planning_manifest']['execution']['precision']})
    return plain(plan)


def validate_import_receipt(receipt, plan):
    """Validate the unchanged historical loader's weights-only receipt."""
    if plan.get('kind') != ADAPTED:
        raise ValueError('Original startup has no adapted import receipt')
    authority = plan['expected_import']
    if (receipt.get('checkpoint_sha256') != plan['checkpoint']['sha256']
            or legacy.digest(receipt.get('configuration')) != authority['historical_configuration_sha256']
            or legacy.digest(receipt.get('source_fingerprint')) != authority['historical_source_fingerprint_sha256']
            or receipt.get('counters') != authority['prior_exposure']
            or receipt.get('data_cursor') != authority['data_cursor']
            or receipt.get('fusion_state_pins') != authority['fusion_state_pins']
            or receipt.get('restore_scope') != 'fusion weights only; diagnostic flags and RNG preserved'
            or set(receipt.get('checks', {})) != {'complete_fusion_exact', 'parameter_identities_preserved',
                'trainability_preserved', 'module_modes_preserved', 'frozen_state_exact', 'tied_readout_preserved'}
            or not all(v is True for v in receipt['checks'].values())):
        raise ValueError('Adapted import receipt differs or restored more than fusion weights')


def execution_identity(resolved, arm, *, runtime, determinism, model_contract, extra_sources=None):
    validate_resolved(resolved)
    if any(not isinstance(v, dict) or not v for v in (runtime, determinism, model_contract)):
        raise ValueError('Actual runtime, determinism and model ownership contracts are required')
    sources = dict(resolved['sources'])
    for name, digest in (extra_sources or {}).items():
        legacy.pin(digest)
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not name or (name in sources and sources[name] != digest):
            raise ValueError('Execution source inventory contains unsafe or conflicting entries')
        if sha256_file(ROOT/path) != digest:
            raise ValueError('Extra execution source differs from its pin')
        sources[name] = digest
    manifest = resolved['declaration']['planning_manifest']
    plan = startup_plan(resolved, arm)
    payload = {'arm': arm, 'resolved_contract_sha256': resolved['contract_sha256'],
        'cursor_schema': CURSOR_SCHEMA,
        'startup': plan, 'recipe': plan['target_recipe'], 'model_contract': plain(model_contract),
        'execution': manifest['execution'], 'data': manifest['data'], 'plan': resolved['planning']['plan'],
        'schedule': resolved['planning']['schedule'], 'partition': {'world_size': manifest['partition']['world_size'],
            'physical_batch_per_rank': manifest['partition']['physical_batch_per_rank'][arm]},
        'evaluation': resolved['planning']['evaluation'], 'runtime': plain(runtime), 'determinism': plain(determinism),
        'sources': dict(sorted(sources.items()))}
    return {'schema': IDENTITY_SCHEMA, 'payload': plain(payload), 'sha256': legacy.digest(payload)}


def validate_identity(identity):
    legacy.exact_fields(identity, ('schema', 'payload', 'sha256'), 'execution identity')
    if identity['schema'] != IDENTITY_SCHEMA or identity['sha256'] != legacy.digest(identity['payload']):
        raise ValueError('Execution identity digest/version differs')


def expected_counters(identity, completed):
    validate_identity(identity)
    payload = identity['payload']; rows = payload['plan']['updates']; arm = payload['arm']
    if type(completed) is not int or not 0 <= completed <= len(rows):
        raise ValueError('Committed update is outside the immutable finite plan')
    result = {key: 0 for key in ('optimizer_updates', 'input_tokens', 'documents', 'microbatches',
                               'ce_positions', 'latent_pairs', 'kl_triples')}
    for row in rows[:completed]:
        result['optimizer_updates'] += 1
        result['input_tokens'] += row['counts']['valid_tokens']
        result['documents'] += row['counts']['packed_rows']
        result['microbatches'] += sum(rank['microbatches'] for rank in row['allocation_by_arm'][arm])
        result['ce_positions'] += row['counts']['ce_targets']
        if 'N' in arm:
            result['latent_pairs'] += row['counts']['latent_pairs']
            result['kl_triples'] += row['counts']['kl_triples']
    return result


def validate_resume_metadata(checkpoint_manifest, identity):
    """Metadata preflight, never a substitute for inspect/load_distributed_checkpoint.

    A stopped, failed or missing run report is intentionally irrelevant. The
    caller must independently pin the committed manifest and verify state bytes.
    """
    validate_identity(identity)
    payload = identity['payload']; world = payload['partition']['world_size']
    meta = checkpoint_manifest.get('metadata', {})
    if (checkpoint_manifest.get('schema') != 'olmo-replicated-ddp-checkpoint-v1'
            or checkpoint_manifest.get('world_size') != world or meta.get('world_size') != world
            or meta.get('configuration', {}).get('execution_identity') != identity
            or meta.get('source_fingerprint', {}).get('execution_identity_sha256') != identity['sha256']):
        raise ValueError('Distributed checkpoint is not the same execution lineage')
    state = checkpoint_manifest.get('state', {})
    if state.get('filename') != 'state.pt':
        raise ValueError('Require the committed distributed state payload')
    legacy.integer(state.get('size_bytes'), 'checkpoint bytes'); legacy.pin(state.get('sha256'))
    counters = checkpoint_manifest.get('counters', {})
    completed = counters.get('optimizer_updates')
    if counters != expected_counters(identity, completed) or any(type(v) is not int for v in counters.values()):
        raise ValueError('Committed counters differ from finite plan exposure')
    plan = payload['plan']
    expected = plan['first_cursor'] if completed == 0 else plan['updates'][completed-1]['next_cursor']
    ranks = checkpoint_manifest.get('rank_cursors', [])
    if len(ranks) != world:
        raise ValueError('Checkpoint rank cursor count differs')
    for rank, record in enumerate(ranks):
        if (record.get('schema') != payload['cursor_schema']
                or record.get('rank') != rank or record.get('world_size') != world
                or record.get('physical_batch_per_rank') != payload['partition']['physical_batch_per_rank']
                or record.get('cursor') != expected):
            raise ValueError('Checkpoint cursor/physical partition differs')
    return {'completed_updates': completed, 'plan_complete': completed == len(plan['updates']),
        'next_cursor': plain(expected), 'remaining_updates': len(plan['updates'])-completed,
        'requires_generic_tensor_optimizer_rng_validation': True,
        'completed_reference_report_required': False}
