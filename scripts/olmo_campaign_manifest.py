#!/usr/bin/env python3
"""Resolve a pinned campaign draft on CPU; never construct or launch a model."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_recipe import ARMS, CampaignRecipe
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import (CHECKPOINT_SHA256, MANIFEST_FILENAME,
    REPO_ID, REVISION, validate_prepared_manifest)
from cdrm.pretrained.packed_campaign_data import PackedCampaignData, POLICY
from cdrm.pretrained.resource_estimates import architecture_parameter_counts
from scripts.olmo_campaign_resource_ledger import matrix_card, source_hashes as ledger_sources

SCHEMA = 'olmo-campaign-draft-manifest-v1'
RESOLVED_SCHEMA = 'olmo-campaign-resolved-draft-v1'
PROTOCOL = 'docs/reports/olmo-campaign-manifest/protocol.md'
LEDGER_SHA = 'f7d111fdbbac0cc9147766a6b74cd2b09d529e9de856b0608fe5c9bca5f5fea9'
MAX_UPDATES, MAX_CHUNKS = 4096, 1_048_576
STARTUP = {'kind': 'original-pretrained-fresh-optimizer-v1', 'all_active_parameters_trainable': True}
EXECUTION_COMMON = {'backward_memory': 'recompute', 'ordinary_activation_checkpointing': True,
    'reuse_rope': True, 'kv_only_writes': True, 'cast_weights_once': True,
    'ordinary_pointwise_backend': 'eager', 'ordinary_rope_backend': 'native',
    'optimizer': 'adamw_fused', 'master_dtype': 'float32', 'optimizer_state_dtype': 'float32',
    'autocast_cache': False, 'tf32': False,
    'deterministic': True, 'distributed': 'replicated_static_ddp'}
PATHS = {
    'fp32': {'ordinary_attention': 'math_sdpa', 'rt_attention_precision': 'fp32',
             'rt_forward_tiles': 'eager', 'rt_backward_tiles': 'eager'},
    'bf16_mixed': {'ordinary_attention': 'flash_sdpa', 'rt_attention_precision': 'mixed',
                   'rt_forward_tiles': 'triton', 'rt_backward_tiles': 'triton'},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def exact_fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{label}: missing or unsupported fields')


def integer(value, label, *, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f'{label}: require a bounded positive integer')
    return value


def pin(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Require an explicit lowercase SHA256 pin')
    return value


def local_path(value):
    if not isinstance(value, str) or not value or '://' in value or '..' in Path(value).parts:
        raise ValueError('Use a local path without traversal or a URI')
    path = Path(value)
    return path if path.is_absolute() else ROOT/path


def read_json(path, expected, *, limit=16*1024*1024):
    path = Path(path)
    pin(expected)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit or sha256_file(path) != expected:
        raise ValueError('Pinned JSON bytes differ or file is unsupported')
    result = json.loads(path.read_text())
    if sha256_file(path) != expected:
        raise ValueError('Pinned JSON changed while reading')
    return result


def source_hashes():
    result = ledger_sources()
    for name in ('scripts/olmo_campaign_manifest.py', 'tests/test_campaign_manifest.py', PROTOCOL):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def validate_manifest(manifest, sources):
    exact_fields(manifest, ('schema', 'purpose', 'label', 'model', 'data', 'arms', 'recipe',
        'budget', 'partition', 'startup', 'execution', 'retention', 'evaluation', 'tracking',
        'resource_ledger', 'implementation_sources'), 'manifest')
    if manifest['schema'] != SCHEMA or manifest['purpose'] not in ('readiness-example', 'review-draft'):
        raise ValueError('Only explicitly unlaunched draft/readiness manifests are supported')
    if not isinstance(manifest['label'], str) or re.fullmatch('[a-z0-9][a-z0-9-]{0,79}', manifest['label']) is None:
        raise ValueError('Use a short stable draft label')
    if manifest['implementation_sources'] != sources:
        raise ValueError('Implementation source inventory differs; re-review before repinning')
    arms = manifest['arms']
    if not isinstance(arms, list) or not arms or len(arms) != len(set(arms)) or any(a not in ARMS for a in arms):
        raise ValueError('Select distinct supported arms explicitly')
    common = manifest['recipe']
    exact_fields(common, {f.name for f in fields(CampaignRecipe)}-{'arm'}, 'recipe')
    recipes = {arm: CampaignRecipe(arm=arm, **common) for arm in arms}
    if common['sequence_length'] != 1024 or common['rt_layers'] != [0, 15] or common['document_policy'] != POLICY['document_policy']:
        raise ValueError('This draft resolver supports native T1024, RT0/15 and continuous-stream policy only')
    if manifest['startup'] != STARTUP or manifest['startup'].get('all_active_parameters_trainable') is not True:
        raise ValueError('Unsupported startup: adapted weights, inherited Adam and implicit warmup are not implemented')
    model = manifest['model']
    exact_fields(model, ('artifacts', 'manifest_sha256', 'checkpoint_sha256', 'repo', 'revision'), 'model')
    if (model['repo'], model['revision'], model['checkpoint_sha256']) != (REPO_ID, REVISION, CHECKPOINT_SHA256):
        raise ValueError('Only the original pinned OLMo-1B authority is supported')
    pin(model['manifest_sha256']); local_path(model['artifacts'])
    data = manifest['data']
    exact_fields(data, ('corpus', 'corpus_manifest_sha256', 'index', 'index_manifest_sha256', 'split', 'policy'), 'data')
    if data['split'] != 'train' or data['policy'] != POLICY or data['policy'].get('cycling') is not False:
        raise ValueError('Unsupported data policy; no shuffle, cycling, overlapping chunks or tail drop')
    for key in ('corpus_manifest_sha256', 'index_manifest_sha256'): pin(data[key])
    for key in ('corpus', 'index'): local_path(data[key])
    budget = manifest['budget']
    exact_fields(budget, ('updates', 'target_valid_tokens_per_update', 'whole_chunk_overshoot', 'insufficient_corpus'), 'budget')
    integer(budget['updates'], 'updates', maximum=MAX_UPDATES)
    integer(budget['target_valid_tokens_per_update'], 'valid tokens per update')
    if (budget['whole_chunk_overshoot'] != 'record' or budget['insufficient_corpus'] != 'error'
            or budget['target_valid_tokens_per_update'] != common['effective_valid_tokens']):
        raise ValueError('Require explicit whole-chunk accounting and matching recipe budget; never shorten/cycle')
    if budget['updates']*((budget['target_valid_tokens_per_update']+1023)//1024) > MAX_CHUNKS:
        raise ValueError('Draft metadata plan exceeds the bounded resolver chunk limit')
    partition = manifest['partition']
    exact_fields(partition, ('world_size', 'physical_batch_per_rank'), 'partition')
    if not 2 <= integer(partition['world_size'], 'world size', maximum=64):
        raise ValueError('Declare a multi-GPU partition; no hardware qualification is implied')
    exact_fields(partition['physical_batch_per_rank'], arms, 'physical batches')
    for value in partition['physical_batch_per_rank'].values(): integer(value, 'physical batch', maximum=65536)
    execution = manifest['execution']
    exact_fields(execution, set(EXECUTION_COMMON)|{'precision', 'graph_mode'}|set(PATHS['fp32']), 'execution')
    precision = execution['precision']
    if (precision not in PATHS or any(execution[k] != v or type(execution[k]) is not type(v) for k, v in EXECUTION_COMMON.items())
            or any(execution[k] != v for k, v in PATHS[precision].items())
            or execution['graph_mode'] not in ('prepared_eager', 'prepared_cuda_graph')
            or (precision == 'fp32' and execution['graph_mode'] != 'prepared_eager')):
        raise ValueError('Unsupported precision/backend combination; no automatic fallback')
    retention = manifest['retention']
    exact_fields(retention, ('storage_prefix', 'checkpoint_seconds', 'checkpoint_every_updates', 'keep_local_completed', 'publication', 'resume'), 'retention')
    prefix = retention['storage_prefix']
    root = 'gs://fast-chunks/cdrm-w-latent/'
    if (not isinstance(prefix, str) or not prefix.startswith(root) or not prefix[len(root):]
            or any(p in ('', '.', '..') for p in prefix[len(root):].split('/'))
            or retention['publication'] != 'verified_generation_before_prune'
            or retention['resume'] != 'same_topology_exact_configuration'):
        raise ValueError('Require an explicit immutable fast-chunks retention/resume declaration')
    integer(retention['checkpoint_seconds'], 'checkpoint cadence', maximum=600)
    integer(retention['checkpoint_every_updates'], 'checkpoint update cadence')
    integer(retention['keep_local_completed'], 'local completed checkpoint count')
    evaluation = manifest['evaluation']
    if not isinstance(evaluation, dict):
        raise ValueError('Require an explicit evaluation declaration')
    if evaluation.get('kind') == 'deferred':
        exact_fields(evaluation, ('kind', 'reason'), 'deferred evaluation')
        if not isinstance(evaluation['reason'], str) or not evaluation['reason'].strip():
            raise ValueError('Explicitly explain the unresolved evaluation choice')
    elif evaluation.get('kind') == 'finite_pass_teacher_forced':
        exact_fields(evaluation, ('kind', 'index', 'index_manifest_sha256', 'split', 'target_valid_tokens',
            'every_updates', 'precision', 'feedback_jitter', 'report_passes', 'generation'), 'evaluation')
        local_path(evaluation['index']); pin(evaluation['index_manifest_sha256'])
        integer(evaluation['target_valid_tokens'], 'evaluation tokens'); integer(evaluation['every_updates'], 'evaluation interval')
        if (evaluation['split'] != 'dev' or evaluation['precision'] != 'fp32' or evaluation['feedback_jitter'] != 0
                or evaluation['report_passes'] != 'all_trained_passes' or evaluation['generation'] != 'not_implemented'):
            raise ValueError('Only declared no-jitter FP32 finite-pass dev evaluation is supported')
    else:
        raise ValueError('Evaluation must be explicitly deferred or use the supported finite-pass declaration')
    exact_fields(manifest['tracking'], ('entity', 'project', 'group'), 'tracking')
    if manifest['tracking']['entity'] != 'taylorbollman' or any(not isinstance(manifest['tracking'][k], str) or not manifest['tracking'][k] for k in ('project', 'group')):
        raise ValueError('Declare the authorized tracking account and nonempty project/group')
    exact_fields(manifest['resource_ledger'], ('path', 'sha256'), 'resource ledger')
    local_path(manifest['resource_ledger']['path']); pin(manifest['resource_ledger']['sha256'])
    if manifest['resource_ledger']['sha256'] != LEDGER_SHA:
        raise ValueError('Use the retained native-size ownership ledger')
    return recipes


def plan_updates(data, budget, partitions):
    """Metadata only: peek/partition, never read tokens, construct tensors or commit."""
    origin = cursor = data.cursor()
    target, number = budget['target_valid_tokens_per_update'], budget['updates']
    if origin.next_chunk or origin.next_update:
        raise ValueError('Only a fresh source-prefix draft is supported')
    minimum = (number-1)*((target+data.length-1)//data.length)*data.length+target
    if minimum > data.total_tokens:
        raise ValueError('Insufficient corpus for the finite target budget; cycling/shortening forbidden')
    plans, prefix, total_counts = [], [0], None
    for update_index in range(number):
        update = data.peek_update(cursor, target)
        if update is None or not update.reaches_target or update.next_cursor.next_chunk <= cursor.next_chunk:
            raise ValueError('Insufficient corpus or invalid finite update progression')
        allocations = {}
        for arm, (world, batch) in partitions.items():
            slots = data.partition(update, world_size=world, physical_batch_size=batch)
            ranks = []
            for rank in range(world):
                rows = [row for slot in slots for row in slot[rank]]
                physical_rows = len(slots)*batch
                valid = sum(row.length for row in rows)
                ranks.append({'rank': rank, 'microbatches': len(slots), 'packed_rows': len(rows),
                    'physical_rows': physical_rows, 'dummy_rows': physical_rows-len(rows),
                    'valid_tokens': valid, 'padding_tokens': physical_rows*data.length-valid})
            if sum(r['valid_tokens'] for r in ranks) != update.counts.valid_tokens:
                raise AssertionError('Rank metadata failed to reproduce logical input exposure')
            allocations[arm] = ranks
        counts = asdict(update.counts)
        total_counts = counts if total_counts is None else {k: total_counts[k]+v for k, v in counts.items()}
        prefix.append(prefix[-1]+update.counts.valid_tokens)
        plans.append({'update': update_index+1, 'start_cursor': asdict(cursor), 'next_cursor': asdict(update.next_cursor),
            'target_valid_tokens': target, 'overshoot_tokens': update.overshoot_tokens, 'counts': counts,
            'unique_documents_in_this_update': update.unique_document_count,
            'membership_sha256': digest([asdict(row) for row in update.rows]), 'allocation_by_arm': allocations})
        cursor = update.next_cursor
    data.validate_integrity()
    if data.cursor() != origin or data._token_fds:
        raise AssertionError('Planning advanced the reader or opened token materialization handles')
    return {'updates': plans, 'valid_token_prefix': prefix, 'totals': total_counts,
        'first_cursor': asdict(origin), 'final_cursor': asdict(cursor),
        'scope': 'Metadata-only noncycling source prefix; unique document counts are per update and are not additive.'}


def nextlat_config(recipe):
    return NextLatConfig(OLMoConfig.native_1b().model_dim, seed=recipe.predictor_seed,
        vocab_chunk_size=128, ce_chunk_size=2048, document_policy=recipe.document_policy)


def parameter_card(arm, recipe, ledger):
    config = OLMoConfig.native_1b()
    nextlat = nextlat_config(recipe)
    architecture = architecture_parameter_counts(config, fbt=recipe.feedback, nextlat=nextlat if recipe.nextlat else None)
    matches = [card for card in ledger['cards'] if card['arm'] == arm]
    if len(matches) != 1 or matches[0]['parameters']['architecture'] != architecture or not all(matches[0]['parameters']['checks'].values()):
        raise ValueError('Retained actual ownership ledger disagrees with declared architecture')
    return {'architecture': architecture, 'observed_inventory_from_retained_ledger': matches[0]['parameters']['observed_inventory'],
        'groups_from_retained_ledger': matches[0]['parameters']['groups'],
        'scope': 'Reuses the retained actual CPU ownership inventory; this resolver constructs no model or optimizer.'}


def resource_cards(recipes, plan, ledger):
    cards = {}
    config = OLMoConfig.native_1b()
    for arm, recipe in recipes.items():
        ranks = [rank for row in plan['updates'] for rank in row['allocation_by_arm'][arm]]
        physical_rows = sum(rank['physical_rows'] for rank in ranks)
        counts = plan['totals']
        selected = {'ce': counts['ce_targets'], 'latent': counts['latent_pairs'] if recipe.nextlat else 0,
                    'kl': counts['kl_triples'] if recipe.nextlat else 0, 'predictor': counts['latent_pairs'] if recipe.nextlat else 0}
        ncfg = nextlat_config(recipe)
        sparse = matrix_card(config, recipe, ncfg, physical_rows, selected, layout='sparse')
        dense = matrix_card(config, recipe, ncfg, physical_rows, selected, layout='dynamic_dense')
        slots = sum(rank['microbatches'] for rank in ranks)
        cards[arm] = {'parameters': parameter_card(arm, recipe, ledger), 'supervised_positions': selected,
            'physical_microbatches_all_ranks': slots, 'physical_rows': physical_rows,
            'allocated_input_tokens': physical_rows*recipe.sequence_length,
            'valid_input_tokens': counts['valid_tokens'], 'padding_tokens': sum(r['padding_tokens'] for r in ranks),
            'dummy_rows': sum(r['dummy_rows'] for r in ranks),
            'valid_pass_tokens': counts['valid_tokens']*recipe.mode().num_passes,
            'allocated_pass_tokens': physical_rows*recipe.sequence_length*recipe.mode().num_passes,
            'rt_block_invocations': slots*dense['rt_block_calls_per_physical_slot'],
            'ordinary_block_invocations': slots*dense['ordinary_block_calls_per_physical_slot'],
            'sparse_counterfactual': sparse, 'prepared_dense': dense,
            'predictor_union_basis': 'All within-document latent pairs supervised; KL-source positions are a subset.',
            'qualification': 'Analytic matrix work at declared physical slots, not measured FLOPs, GPU capacity or speed.'}
    return cards


def validate_data_binding(data, data_spec, authority, *, split):
    expected_tokenizer = {'repo': REPO_ID, 'revision': REVISION,
        'sha256': authority['artifacts']['native/tokenizer.json']['sha256'],
        'add_special_tokens': False, 'padding': False, 'truncation': False}
    tokenizer = authority['tokenizer']
    if (data.split != split or data.length != 1024
            or data.manifest['corpus_manifest_sha256'] != data_spec['corpus_manifest_sha256']
            or data.manifest['tokenizer'] != expected_tokenizer
            or data.manifest['vocab_size'] != tokenizer['vocab_size']
            or data.manifest['eos_id'] != tokenizer['eos_token_id']
            or data.pad_id != tokenizer['pad_token_id']):
        raise ValueError('Packed corpus/index model-tokenizer authority differs')


def resolve(manifest):
    sources = source_hashes()
    recipes = validate_manifest(manifest, sources)
    before_rng = torch.random.get_rng_state().clone()
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU resolver cannot run after CUDA initialization')
    model = manifest['model']; artifacts = local_path(model['artifacts'])
    read_json(artifacts/MANIFEST_FILENAME, model['manifest_sha256'])
    authority = validate_prepared_manifest(artifacts)  # Offline full-byte checks only.
    if authority['checkpoint']['sha256'] != model['checkpoint_sha256']:
        raise ValueError('Actual model authority differs')
    item = manifest['resource_ledger']
    ledger = read_json(local_path(item['path']), item['sha256'])
    if ledger.get('status') != 'complete' or not all(ledger.get('integrity', {}).values()) or set(c['arm'] for c in ledger.get('cards', [])) != set(ARMS):
        raise ValueError('Require the completed all-eight ownership ledger')
    if any(sources.get(k) != v for k, v in ledger['sources'].items()):
        raise ValueError('Resource-ledger sources changed')
    data_spec = manifest['data']
    corpus = local_path(data_spec['corpus'])
    read_json(corpus/'manifest.json', data_spec['corpus_manifest_sha256'])
    read_json(local_path(data_spec['index'])/'manifest.json', data_spec['index_manifest_sha256'])
    partitions = {arm: (manifest['partition']['world_size'], manifest['partition']['physical_batch_per_rank'][arm]) for arm in recipes}
    with PackedCampaignData(corpus, local_path(data_spec['index'])) as data:
        if data.manifest_sha256 != data_spec['index_manifest_sha256']:
            raise ValueError('Packed corpus/index authority differs')
        validate_data_binding(data, data_spec, authority, split='train')
        plan = plan_updates(data, manifest['budget'], partitions)
        data_identity = {k: data.manifest[k] for k in ('identity_sha256', 'order_sha256', 'total_tokens', 'total_chunks', 'selected_documents', 'policy')}
    evaluation = {'declaration': manifest['evaluation'], 'execution': 'not_run'}
    if manifest['evaluation']['kind'] != 'deferred':
        e = manifest['evaluation']
        read_json(local_path(e['index'])/'manifest.json', e['index_manifest_sha256'])
        with PackedCampaignData(corpus, local_path(e['index'])) as dev:
            if dev.manifest_sha256 != e['index_manifest_sha256']:
                raise ValueError('Evaluation packed-index authority differs')
            validate_data_binding(dev, data_spec, authority, split='dev')
            eval_plan = plan_updates(dev, {'target_valid_tokens_per_update': e['target_valid_tokens'], 'updates': 1}, partitions)
            evaluation['fixed_plan'] = eval_plan
            evaluation['scheduled_updates'] = list(range(e['every_updates'], manifest['budget']['updates']+1, e['every_updates']))
    cards = resource_cards(recipes, plan, ledger)
    first_recipe = next(iter(recipes.values()))
    schedule_tokens = [row['counts']['valid_tokens'] for row in plan['updates']]
    schedule_hash = digest({'tokens': schedule_tokens, 'warmup_tokens': first_recipe.warmup_tokens, 'start_fraction': first_recipe.warmup_start_fraction})
    lrs = [first_recipe.plateau_lr*(first_recipe.warmup_start_fraction+(1-first_recipe.warmup_start_fraction)*
           (min(1, count/first_recipe.warmup_tokens) if first_recipe.warmup_tokens else 1)) for count in plan['valid_token_prefix']]
    checks = {'implementation_sources_unchanged': source_hashes() == sources,
        'model_manifest_unchanged': sha256_file(artifacts/MANIFEST_FILENAME) == model['manifest_sha256'],
        'cpu_rng_unchanged': torch.equal(before_rng, torch.random.get_rng_state()), 'no_cuda_initialized': not torch.cuda.is_initialized()}
    if not all(checks.values()):
        raise AssertionError('CPU resolution integrity failed')
    return {'schema': RESOLVED_SCHEMA, 'status': 'cpu_plan_validated_not_authorized', 'launch_authorized': False,
        'numerical_clearance': False, 'manifest_semantic_sha256': digest(manifest), 'label': manifest['label'], 'purpose': manifest['purpose'],
        'sources': sources, 'declarations': manifest, 'model_checkpoint_authority': authority['checkpoint'],
        'data_identity': data_identity, 'recipes': {arm: recipe.to_dict() for arm, recipe in recipes.items()},
        'native_backbone_config': OLMoConfig.native_1b().to_dict(),
        'nextlat_configs': {arm: nextlat_config(recipe).to_dict() for arm, recipe in recipes.items()},
        'recipe_precision_note': 'Historical recipe serialization retains its nominal BF16 label; the explicit outer execution declaration is authoritative for this draft.',
        'plan': plan, 'plan_sha256': digest(plan), 'schedule': {'plan_sha256': schedule_hash,
            'valid_token_prefix': plan['valid_token_prefix'], 'lr_at_completed_boundaries': lrs,
            'scope': 'Existing warmup/plateau scalar formula; no scheduler or optimizer constructed.'},
        'resource_cards': cards, 'evaluation': evaluation, 'checks': checks,
        'remaining_review': ['No all-arm training entrypoint is implemented or invoked by this resolver.',
            'Production source mixture/order/budgets, startup and evaluation choices require review.',
            'Declared topology/batches/backends need actual-hardware qualification; no VRAM or throughput guarantee.',
            'BF16 sparse/prepared and trajectory qualifications remain open even when this structural plan validates.',
            'Adapted startup, cooldown, SFT and generation are unsupported by this draft.']}


def readiness_example(*, artifacts, corpus, index, ledger, sources):
    recipe = asdict(CampaignRecipe('B', effective_valid_tokens=16384, document_policy=POLICY['document_policy']))
    recipe.pop('arm'); recipe = json.loads(json.dumps(recipe))
    return {'schema': SCHEMA, 'purpose': 'readiness-example', 'label': 'readiness-only-all-eight-no-launch',
        'model': {'artifacts': str(artifacts), 'manifest_sha256': sha256_file(Path(artifacts)/MANIFEST_FILENAME),
                  'checkpoint_sha256': CHECKPOINT_SHA256, 'repo': REPO_ID, 'revision': REVISION},
        'data': {'corpus': str(corpus), 'corpus_manifest_sha256': sha256_file(Path(corpus)/'manifest.json'),
                 'index': str(index), 'index_manifest_sha256': sha256_file(Path(index)/'manifest.json'), 'split': 'train', 'policy': POLICY},
        'arms': list(ARMS), 'recipe': recipe,
        'budget': {'updates': 3, 'target_valid_tokens_per_update': 16384, 'whole_chunk_overshoot': 'record', 'insufficient_corpus': 'error'},
        'partition': {'world_size': 2, 'physical_batch_per_rank': {arm: 8 for arm in ARMS}},
        'startup': STARTUP, 'execution': {**EXECUTION_COMMON, **PATHS['bf16_mixed'], 'precision': 'bf16_mixed', 'graph_mode': 'prepared_cuda_graph'},
        'retention': {'storage_prefix': 'gs://fast-chunks/cdrm-w-latent/campaign-drafts/readiness-example-only',
            'checkpoint_seconds': 600, 'checkpoint_every_updates': 1, 'keep_local_completed': 2,
            'publication': 'verified_generation_before_prune', 'resume': 'same_topology_exact_configuration'},
        'evaluation': {'kind': 'deferred', 'reason': 'Readiness example only; this does not choose a production dev set or evaluation budget.'},
        'tracking': {'entity': 'taylorbollman', 'project': 'pretrained-fbt-rt-nextlat', 'group': 'unlaunched-campaign-draft'},
        'resource_ledger': {'path': str(ledger), 'sha256': LEDGER_SHA}, 'implementation_sources': sources}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd() != Path('/workspace/cdrm-w-latent') or torch.cuda.is_available():
        raise RuntimeError('Use the project CPU container with GPU passthrough disabled')
    manifest = read_json(args.manifest, args.manifest_sha256)
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT): parser.error('Keep resolved evidence under the persistent project')
    output.mkdir(parents=True, exist_ok=False)
    result = resolve(manifest)
    result['manifest_file_sha256'] = args.manifest_sha256
    if sha256_file(args.manifest) != args.manifest_sha256: raise ValueError('Input manifest changed during resolution')
    write_json(output/'resolved.json', result)
    for name in result['sources']:
        target = output/'source-snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    if any(sha256_file(output/'source-snapshot'/name) != value for name, value in result['sources'].items()):
        raise ValueError('Source snapshot differs')
    lines = ['# Unlaunched CPU campaign draft', '', 'Status: **CPU plan validated; no launch authorization or numerical clearance.**', '',
        f"Purpose: `{result['purpose']}`. Updates: {manifest['budget']['updates']}; valid inputs: {result['plan']['totals']['valid_tokens']:,}.", '',
        '| Arm | Trainable parameters | Physical rows | Dummy rows | Estimated dense PFLOPs for entire draft |', '| --- | ---: | ---: | ---: | ---: |']
    for arm, card in result['resource_cards'].items():
        flops = card['prepared_dense']
        lines.append(f"| {arm} | {card['parameters']['architecture']['training_architecture']:,} | {card['physical_rows']} | {card['dummy_rows']} | {flops['matrix_flops_minimum']/1e15:.5f}–{flops['matrix_flops_maximum']/1e15:.5f} |")
    lines += ['', 'The shared example partition is not an all-arm capacity recommendation. Matrix estimates exclude communication, pointwise work, optimizer and graph setup; they do not predict speed or VRAM.', '', *['- '+row for row in result['remaining_review']], '']
    (output/'plan-card.md').write_text('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'arms': list(result['resource_cards']), 'resolved_sha256': sha256_file(output/'resolved.json')}))


if __name__ == '__main__': main()
