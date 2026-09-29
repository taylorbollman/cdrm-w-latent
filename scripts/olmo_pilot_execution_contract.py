#!/usr/bin/env python3
"""CPU-only ordered-pilot planning and fresh-lineage execution authorities.

The accepted runtime and model remain unchanged. This new contract authenticates
round-zero ordered data directly; it never relabels a historical packed manifest
or imports an old cursor. Resolving does not construct tensors or authorize a run.
"""
from __future__ import annotations
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import MANIFEST_FILENAME, validate_prepared_manifest
from scripts import olmo_campaign_manifest as legacy
from scripts import olmo_campaign_execution_contract as accepted
from scripts import olmo_pilot_ordered_data as ordered
from scripts import olmo_pilot_data_plan as data_plan
from scripts import olmo_pilot_eval_control as evaluation_control

SCHEMA = 'olmo-pilot-execution-declaration-v1'
PLANNING_SCHEMA = 'olmo-pilot-execution-planning-v1'
RESOLVED_SCHEMA = 'olmo-pilot-execution-resolved-v1'
PLAN_SCHEMA = 'olmo-pilot-execution-plan-v1'
IDENTITY_SCHEMA = 'olmo-pilot-execution-identity-v1'
# The unchanged engine owns this outer rank wrapper. The new execution identity
# and inner ordered manifest hash forbid migration from a historical data stream.
CURSOR_SCHEMA = accepted.CURSOR_SCHEMA
PROTOCOL = 'docs/reports/olmo-pilot-execution/protocol.md'
ADAPTED = accepted.ADAPTED
ADAPTED_POLICY = {**accepted.ADAPTED_POLICY, 'data_origin': 'ordered_round0_prefix_zero'}
EXPOSURE = accepted.EXPOSURE
plain = accepted.plain
recipe_from_dict = accepted.recipe_from_dict
validate_import_receipt = accepted.validate_import_receipt


def source_hashes():
    result = dict(accepted.source_hashes())
    for name in ('scripts/olmo_pilot_execution_contract.py', 'tests/test_pilot_execution_contract.py',
                 'scripts/olmo_pilot_ordered_data.py', 'scripts/olmo_pilot_data_plan.py',
                 'tests/test_pilot_ordered_data.py', 'tests/test_pilot_data_plan.py',
                 'configs/data/dolma-v1_5-pilot-v1.json',
                 'scripts/olmo_pilot_eval_control.py', 'tests/test_pilot_eval_control.py',
                 'scripts/olmo_pilot_execute.py', 'tests/test_pilot_execute.py', PROTOCOL):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


DATA_PATHS = ('corpus', 'suite', 'index', 'source_plan', 'source_authorities', 'inventory', 'exclusions')
DATA_PINS = ('corpus_manifest_sha256', 'suite_manifest_sha256', 'index_manifest_sha256',
             'source_plan_sha256', 'source_authorities_sha256', 'inventory_sha256', 'exclusions_sha256')


def validate_data_spec(data):
    legacy.exact_fields(data, (*DATA_PATHS, *DATA_PINS, 'split', 'policy'), 'ordered data')
    if data['split'] != 'train' or data['policy'] != ordered.POLICY or data['policy'].get('cycling') is not False:
        raise ValueError('Require finite round-zero ordered training data with its explicit policy')
    for key in DATA_PATHS: legacy.local_path(data[key])
    for key in DATA_PINS: legacy.pin(data[key])


def _read_bytes(path, expected, *, limit=32*1024**2):
    path = legacy.local_path(str(path)); legacy.pin(expected)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Unsupported pinned data authority file')
    before = path.stat()
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected or path.stat() != before:
        raise ValueError('Pinned data authority bytes differ or changed')
    return value


def authenticate_data(data, length):
    """Authenticate a complete ordered suite and its selected training authority.

    This is also usable by the separately named tiny acceptance schema. It opens
    only JSON, SQLite and token bytes for hashing; no token tensors or CUDA.
    Returns authenticated manifest dictionaries, never live reader ownership.
    """
    validate_data_spec(data)
    legacy.integer(length, 'sequence length', maximum=2048)
    corpus, suite_root, index = (legacy.local_path(data[k]) for k in ('corpus', 'suite', 'index'))
    if any(p.is_symlink() or not p.is_dir() for p in (corpus, suite_root, index)):
        raise ValueError('Ordered authority directories must be real directories')
    suite = legacy.read_json(suite_root/'manifest.json', data['suite_manifest_sha256'], limit=32*1024**2)
    if (suite.get('schema') != ordered.SUITE_SCHEMA or type(suite.get('round_id')) is not int
            or suite['round_id'] != 0 or suite.get('identity_sha256') != ordered._digest(
                {k:v for k,v in suite.items() if k != 'identity_sha256'})
            or set(suite.get('panels', {})) != set(ordered.panel_names())):
        raise ValueError('Ordered suite schema, round, identity or panel inventory differs')
    recipe = suite['recipe']; data_plan.validate_recipe(recipe)
    if (suite['recipe_sha256'] != data_plan.recipe_sha256(recipe)
            or recipe['panels']['length'] != length or suite['exclusion_authority'] != recipe['exclusions']
            or suite['split_intersections'] != dict.fromkeys(('content_hashes', 'document_indices', 'readiness_exclusions'), 0)):
        raise ValueError('Ordered recipe/context/exclusion authority differs')
    inventory = _read_bytes(data['inventory'], data['inventory_sha256'])
    if data['inventory_sha256'] != recipe['inventory']['sha256']:
        raise ValueError('Inventory pin differs from ordered recipe')
    selected = data_plan.source_selection(recipe, inventory)
    exclusions = _read_bytes(data['exclusions'], data['exclusions_sha256'])
    excluded = data_plan.verify_exclusions(recipe, exclusions.decode('ascii').splitlines())
    if exclusions != data_plan.exclusion_bytes(excluded) or data['exclusions_sha256'] != recipe['exclusions']['content_ids_sha256']:
        raise ValueError('Exclusion bytes must be the exact canonical declared identities')
    acquisition = legacy.read_json(legacy.local_path(data['source_plan']), data['source_plan_sha256'], limit=32*1024**2)
    authorities = legacy.read_json(legacy.local_path(data['source_authorities']), data['source_authorities_sha256'], limit=32*1024**2)
    if suite['acquisition_authority'] != {'plan_sha256':data['source_plan_sha256'],
            'source_authorities_sha256':data['source_authorities_sha256'], 'source_authorities':authorities}:
        raise ValueError('Suite acquisition mapping/pins differ')
    summary = legacy.read_json(corpus/'manifest.json', data['corpus_manifest_sha256'])
    config = legacy.read_json(corpus/'config.json', suite['corpus_config_sha256'])
    if (not summary.get('completed') or summary.get('config_sha256') != suite['corpus_config_sha256']
            or suite['corpus_manifest_sha256'] != data['corpus_manifest_sha256']
            or suite['source_selection'] != selected
            or config['split_policy'] != {'seed':recipe['split']['seed'], 'weights':recipe['split']['weights']}):
        raise ValueError('Corpus completion/config/split or selected sources differ')
    ordered._source_contract(config, selected, recipe, acquisition, data['source_plan_sha256'], authorities)
    catalog = suite['selection_authority']
    legacy.exact_fields(catalog, ('path', 'sha256', 'size_bytes'), 'selection catalog')
    if catalog['path'] != 'catalog.sqlite': raise ValueError('Unsafe selection catalog path')
    legacy.integer(catalog['size_bytes'], 'catalog bytes'); legacy.pin(catalog['sha256'])
    catalog_path = suite_root/'catalog.sqlite'
    if (catalog_path.is_symlink() or not catalog_path.is_file() or catalog_path.stat().st_size != catalog['size_bytes']
            or sha256_file(catalog_path) != catalog['sha256']):
        raise ValueError('Selection catalog bytes differ')
    for name, record in suite['panels'].items():
        if record['path'] != 'panels/'+name:
            raise ValueError('Ordered panel path differs')
        legacy.pin(record['manifest_sha256']); legacy.pin(record['identity_sha256'])
    train = suite['panels']['train']
    if (index.resolve() != (suite_root/train['path']).resolve()
            or train['manifest_sha256'] != data['index_manifest_sha256']):
        raise ValueError('Training index is not the suite selected train panel')
    with ordered.OrderedCampaignData(corpus, index) as reader:
        manifest = reader.manifest
        common = ('recipe', 'recipe_sha256', 'round_id', 'selection_authority', 'source_selection',
            'acquisition_authority', 'exclusion_authority', 'corpus_manifest_sha256', 'corpus_config_sha256',
            'corpus_files', 'tokenizer', 'pad_id', 'eos_id', 'vocab_size', 'token_dtype')
        if (reader.manifest_sha256 != data['index_manifest_sha256'] or reader.split != 'train'
                or reader.length != length or manifest['panel'] != 'train'
                or any(manifest[k] != suite[k] for k in common)
                or manifest['identity_sha256'] != train['identity_sha256']
                or manifest['total_tokens'] != train['tokens'] or manifest['total_chunks'] != train['chunks']
                or manifest['counts'] != train['counts'] or manifest['stratum_quotas'] != train['stratum_quotas']
                or manifest['stratum_quotas'] != data_plan.panel_quotas(recipe, 'train')):
            raise ValueError('Ordered train manifest differs from suite authority')
        identity = {k:manifest[k] for k in ('identity_sha256', 'order_sha256', 'total_tokens',
            'total_chunks', 'selected_documents', 'policy', 'recipe_sha256', 'round_id', 'selection_authority',
            'exclusion_authority', 'acquisition_authority')}
        identity.update(suite_manifest_sha256=data['suite_manifest_sha256'], suite_identity_sha256=suite['identity_sha256'])
        if reader._token_fds: raise AssertionError('Data authentication opened token materialization')
    return {'suite':suite, 'train_manifest':manifest, 'data_identity':identity}


def validate_manifest(manifest, sources):
    legacy.exact_fields(manifest, ('schema', 'purpose', 'label', 'model', 'data', 'arms', 'recipe',
        'budget', 'partition', 'startup', 'execution', 'retention', 'evaluation', 'tracking',
        'resource_ledger', 'implementation_sources'), 'manifest')
    if manifest['schema'] != PLANNING_SCHEMA or manifest['purpose'] not in ('readiness-example', 'review-draft'):
        raise ValueError('Only explicitly unlaunched draft/readiness manifests are supported')
    if not isinstance(manifest['label'], str) or re.fullmatch('[a-z0-9][a-z0-9-]{0,79}', manifest['label']) is None:
        raise ValueError('Use a short stable draft label')
    if manifest['implementation_sources'] != sources:
        raise ValueError('Implementation source inventory differs; re-review before repinning')
    arms = manifest['arms']
    if not isinstance(arms, list) or not arms or len(arms) != len(set(arms)) or any(a not in legacy.ARMS for a in arms):
        raise ValueError('Select distinct supported arms explicitly')
    common = manifest['recipe']
    legacy.exact_fields(common, {f.name for f in fields(CampaignRecipe)}-{'arm'}, 'recipe')
    recipes = {arm: CampaignRecipe(arm=arm, **common) for arm in arms}
    if common['sequence_length'] != 1024 or common['rt_layers'] != [0, 15] or common['document_policy'] != ordered.POLICY['document_policy']:
        raise ValueError('This draft resolver supports native T1024, RT0/15 and continuous-stream policy only')
    if manifest['startup'] != legacy.STARTUP or manifest['startup'].get('all_active_parameters_trainable') is not True:
        raise ValueError('Unsupported startup: adapted weights, inherited Adam and implicit warmup are not implemented')
    model = manifest['model']
    legacy.exact_fields(model, ('artifacts', 'manifest_sha256', 'checkpoint_sha256', 'repo', 'revision'), 'model')
    if (model['repo'], model['revision'], model['checkpoint_sha256']) != (legacy.REPO_ID, legacy.REVISION, legacy.CHECKPOINT_SHA256):
        raise ValueError('Only the original pinned OLMo-1B authority is supported')
    legacy.pin(model['manifest_sha256']); legacy.local_path(model['artifacts'])
    validate_data_spec(manifest['data'])
    budget = manifest['budget']
    legacy.exact_fields(budget, ('updates', 'target_valid_tokens_per_update', 'whole_chunk_overshoot', 'insufficient_corpus'), 'budget')
    legacy.integer(budget['updates'], 'updates', maximum=legacy.MAX_UPDATES)
    legacy.integer(budget['target_valid_tokens_per_update'], 'valid tokens per update')
    if (budget['whole_chunk_overshoot'] != 'record' or budget['insufficient_corpus'] != 'error'
            or budget['target_valid_tokens_per_update'] != common['effective_valid_tokens']):
        raise ValueError('Require explicit whole-chunk accounting and matching recipe budget; never shorten/cycle')
    if budget['updates']*((budget['target_valid_tokens_per_update']+1023)//1024) > legacy.MAX_CHUNKS:
        raise ValueError('Draft metadata plan exceeds the bounded resolver chunk limit')
    partition = manifest['partition']
    legacy.exact_fields(partition, ('world_size', 'physical_batch_per_rank'), 'partition')
    if not 2 <= legacy.integer(partition['world_size'], 'world size', maximum=64):
        raise ValueError('Declare a multi-GPU partition; no hardware qualification is implied')
    legacy.exact_fields(partition['physical_batch_per_rank'], arms, 'physical batches')
    for value in partition['physical_batch_per_rank'].values(): legacy.integer(value, 'physical batch', maximum=65536)
    execution = manifest['execution']
    legacy.exact_fields(execution, set(legacy.EXECUTION_COMMON)|{'precision', 'graph_mode'}|set(legacy.PATHS['fp32']), 'execution')
    precision = execution['precision']
    if (precision not in legacy.PATHS or any(execution[k] != v or type(execution[k]) is not type(v) for k, v in legacy.EXECUTION_COMMON.items())
            or any(execution[k] != v for k, v in legacy.PATHS[precision].items())
            or execution['graph_mode'] not in ('prepared_eager', 'prepared_cuda_graph')
            or (precision == 'fp32' and execution['graph_mode'] != 'prepared_eager')):
        raise ValueError('Unsupported precision/backend combination; no automatic fallback')
    retention = manifest['retention']
    legacy.exact_fields(retention, ('storage_prefix', 'checkpoint_seconds', 'checkpoint_every_updates', 'keep_local_completed', 'publication', 'resume'), 'retention')
    prefix = retention['storage_prefix']
    root = 'gs://fast-chunks/cdrm-w-latent/'
    if (not isinstance(prefix, str) or not prefix.startswith(root) or not prefix[len(root):]
            or any(p in ('', '.', '..') for p in prefix[len(root):].split('/'))
            or retention['publication'] != 'verified_generation_before_prune'
            or retention['resume'] != 'same_topology_exact_configuration'):
        raise ValueError('Require an explicit immutable fast-chunks retention/resume declaration')
    legacy.integer(retention['checkpoint_seconds'], 'checkpoint cadence', maximum=600)
    legacy.integer(retention['checkpoint_every_updates'], 'checkpoint update cadence')
    legacy.integer(retention['keep_local_completed'], 'local completed checkpoint count')
    evaluation_control.validate_policy(manifest['evaluation'])
    legacy.exact_fields(manifest['tracking'], ('entity', 'project', 'group'), 'tracking')
    if manifest['tracking']['entity'] != 'taylorbollman' or any(not isinstance(manifest['tracking'][k], str) or not manifest['tracking'][k] for k in ('project', 'group')):
        raise ValueError('Declare the authorized tracking account and nonempty project/group')
    legacy.exact_fields(manifest['resource_ledger'], ('path', 'sha256'), 'resource ledger')
    legacy.local_path(manifest['resource_ledger']['path']); legacy.pin(manifest['resource_ledger']['sha256'])
    if manifest['resource_ledger']['sha256'] != legacy.LEDGER_SHA:
        raise ValueError('Use the retained native-size ownership ledger')
    return recipes


def validate_declaration(declaration, sources=None):
    legacy.exact_fields(declaration, ('schema', 'planning_manifest', 'startup', 'implementation_sources'), 'ordered execution declaration')
    if declaration['schema'] != SCHEMA: raise ValueError('Unsupported ordered execution declaration')
    sources = source_hashes() if sources is None else sources
    if declaration['implementation_sources'] != sources:
        raise ValueError('Ordered execution source inventory differs')
    recipes = validate_manifest(declaration['planning_manifest'], sources)
    startup = declaration['startup']
    if startup == legacy.STARTUP and startup.get('all_active_parameters_trainable') is True: return recipes
    legacy.exact_fields(startup, set(ADAPTED_POLICY)|{'checkpoint','report'}, 'adapted ordered startup')
    if any(startup[k] != v or type(startup[k]) is not type(v) for k,v in ADAPTED_POLICY.items()):
        raise ValueError('Only selected fusion128 weights with fresh all-active Adam and ordered origin are supported')
    if any(type(v) is not int for v in startup['prior_exposure'].values()):
        raise ValueError('Historical exposure must remain exact integer counts')
    if any(arm not in ('NF','NFR') for arm in recipes): raise ValueError('Fusion128 startup supports NF/NFR only')
    for name,digest in (('checkpoint',accepted.FUSION_CHECKPOINT_SHA),('report',accepted.FUSION_REPORT_SHA)):
        legacy.exact_fields(startup[name],('path','sha256'),'adapted '+name)
        legacy.local_path(startup[name]['path'])
        if startup[name]['sha256'] != digest: raise ValueError('Adapted startup authority differs')
    return recipes



def plan_updates(data, budget, partitions):
    if not isinstance(data, ordered.OrderedCampaignData):
        raise ValueError('Require an ordered pilot reader; old packed cursors are not interchangeable')
    result = legacy.plan_updates(data, budget, partitions)
    result['scope'] = 'Metadata-only finite round-zero ordered prefix; unique document counts per update are not additive.'
    return result


def resolve(declaration):
    sources = source_hashes()
    recipes = validate_declaration(declaration, sources)
    manifest = declaration['planning_manifest']
    before_rng = torch.random.get_rng_state().clone()
    if torch.cuda.is_initialized(): raise RuntimeError('Ordered CPU resolver cannot run after CUDA initialization')
    model = manifest['model']; artifacts = legacy.local_path(model['artifacts'])
    legacy.read_json(artifacts/MANIFEST_FILENAME, model['manifest_sha256'])
    authority = validate_prepared_manifest(artifacts)
    if authority['checkpoint']['sha256'] != model['checkpoint_sha256']:
        raise ValueError('Actual model authority differs')
    item = manifest['resource_ledger']
    ledger = legacy.read_json(legacy.local_path(item['path']), item['sha256'])
    if (ledger.get('status') != 'complete' or not ledger.get('integrity') or not all(ledger['integrity'].values())
            or set(c['arm'] for c in ledger.get('cards', [])) != set(legacy.ARMS)
            or any(sources.get(k) != v for k,v in ledger['sources'].items())):
        raise ValueError('Require unchanged completed all-eight ownership ledger')
    data_spec = manifest['data']; bound = authenticate_data(data_spec, manifest['recipe']['sequence_length'])
    partitions = {arm:(manifest['partition']['world_size'],manifest['partition']['physical_batch_per_rank'][arm]) for arm in recipes}
    with ordered.OrderedCampaignData(legacy.local_path(data_spec['corpus']),legacy.local_path(data_spec['index'])) as reader:
        legacy.validate_data_binding(reader, data_spec, authority, split='train')
        plan = plan_updates(reader, manifest['budget'], partitions)
    evaluation = evaluation_control.resolve_evaluation(data_spec, manifest['evaluation'],
        length=manifest['recipe']['sequence_length'], updates=manifest['budget']['updates'], partitions=partitions)
    cards = legacy.resource_cards(recipes, plan, ledger)
    first = next(iter(recipes.values()))
    schedule_hash = legacy.digest({'tokens':[r['counts']['valid_tokens'] for r in plan['updates']],
        'warmup_tokens':first.warmup_tokens,'start_fraction':first.warmup_start_fraction})
    lrs = [first.plateau_lr*(first.warmup_start_fraction+(1-first.warmup_start_fraction)*
        (min(1,count/first.warmup_tokens) if first.warmup_tokens else 1)) for count in plan['valid_token_prefix']]
    fusion_authority = (accepted.load_fusion_authority(declaration['startup'], recipes, sources)
                        if declaration['startup']['kind'] == ADAPTED else None)
    checks = {'implementation_sources_unchanged':source_hashes() == sources,
        'model_manifest_unchanged':sha256_file(artifacts/MANIFEST_FILENAME) == model['manifest_sha256'],
        'cpu_rng_unchanged':torch.equal(before_rng,torch.random.get_rng_state()),
        'no_cuda_initialized':not torch.cuda.is_initialized()}
    if not all(checks.values()): raise AssertionError('Ordered CPU resolution integrity failed')
    planning = {'schema':PLAN_SCHEMA,'status':'cpu_ordered_plan_validated_not_authorized',
        'launch_authorized':False,'numerical_clearance':False,'sources':sources,
        'declarations':plain(manifest),'manifest_semantic_sha256':legacy.digest(manifest),
        'label':manifest['label'],'purpose':manifest['purpose'],'model_checkpoint_authority':authority['checkpoint'],
        'data_identity':bound['data_identity'],'recipes':{arm:r.to_dict() for arm,r in recipes.items()},
        'native_backbone_config':OLMoConfig.native_1b().to_dict(),
        'nextlat_configs':{arm:legacy.nextlat_config(r).to_dict() for arm,r in recipes.items()},
        'plan':plan,'plan_sha256':legacy.digest(plan),'schedule':{'plan_sha256':schedule_hash,
            'valid_token_prefix':plan['valid_token_prefix'],'lr_at_completed_boundaries':lrs,
            'scope':'Accepted warmup/plateau formula; no optimizer or scheduler constructed.'},
        'resource_cards':cards,'evaluation':evaluation,'checks':checks}
    payload = {'schema':RESOLVED_SCHEMA,'status':'cpu_ordered_execution_validated_not_authorized',
        'launch_authorized':False,'numerical_clearance':False,'declaration':plain(declaration),
        'sources':sources,'planning':planning,'actual_startup':plain(declaration['startup']),
        'fusion_authority':fusion_authority,
        'planning_scope':'Native construction plus authenticated finite ordered data; actual_startup owns the new lineage.',
        'resume_scope':'Same immutable ordered identity, topology and completed distributed boundary; never migrate old packed cursors.'}
    return {**payload,'contract_sha256':legacy.digest(payload)}


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
            transition={'kind': 'weights-preserving-isolated-NF-to-declared-ordered-arm-v1',
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
