#!/usr/bin/env python3
"""JSON-only audit of explicitly forked KL continuations and same-branch restart.

Training/evaluation checks are literal accepted validators with the new report
and configuration schemas, and declared actual objective weights. No input
report is relabeled. Pair checks do not require different objectives to produce
the same learned state; tiny same-branch restart checks require exact parity.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_pilot_execution_audit import (
    Audit, legacy, publication_metadata, expected_counts, expected_cursor,
    file_sha, bounded_json, digest, finite_json, boundary_check,
    OBSERVATION_SCHEMA, TERMS, HEAVY, IDENTITY_SCHEMA, TINY_SCHEMA)
from scripts.olmo_pilot_async_audit import transport_check
from scripts.olmo_pilot_execution_audit_v2 import training_parity

SCHEMA = 'olmo-kl-continuation-audit-v1'
REPORT_SCHEMA = 'olmo-kl-continuation-report-v1'
CONFIG_SCHEMA = 'olmo-kl-continuation-execution-v1'
BRANCH_SCHEMA = 'olmo-kl-objective-branch-v1'

def training_check(audit, report, label, source_root=None):
    check = lambda name, condition: audit.require(label+'/'+name, condition)
    equal = lambda name, actual, expected: audit.equal(label+'/'+name, actual, expected)
    check('finite_json', finite_json(report))
    equal('schema', report['schema'], REPORT_SCHEMA)
    check('closed_success', report['status'] in ('completed_plan', 'stopped_at_boundary'))
    configuration = report['configuration']; identity = configuration['execution_identity']; payload = identity['payload']
    equal('identity_schema', identity['schema'], IDENTITY_SCHEMA)
    equal('identity_digest', identity['sha256'], digest(payload))
    equal('arm', report['arm'], payload['arm'])
    check('known_arm', payload['arm'] in ('B','N','F','R','NF','NR','FR','NFR'))
    equal('two_ranks', payload['partition']['world_size'], 2)
    equal('configuration_recipe', configuration['recipe'], payload['recipe'])
    equal('configuration_ownership', configuration['parameters'], payload['model_contract'])
    equal('configuration_precision', configuration['training']['precision'], payload['execution']['precision'])
    equal('sources', report['sources'], payload['sources'])
    check('nonempty_sources', isinstance(report['sources'], dict) and bool(report['sources']))
    fingerprint = report['source_fingerprint']
    equal('fingerprint_identity', fingerprint['execution_identity_sha256'], identity['sha256'])
    equal('fingerprint_digest', fingerprint['sha256'], identity['sha256'])
    equal('fingerprint_scope', fingerprint['scope'], 'execution_identity')
    equal('fingerprint_sources', fingerprint['sources'], report['sources'])
    equal('fingerprint_source', fingerprint['source_checkpoint'], configuration['source_checkpoint'])
    for name, pin in report['sources'].items():
        path = Path(name)
        check('source_safe/'+name, not path.is_absolute() and '..' not in path.parts and bool(name)
              and isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin) is not None)
        if source_root is not None:
            candidate = Path(source_root)/path
            check('source_snapshot/'+name, candidate.is_file() and not candidate.is_symlink() and file_sha(candidate) == pin)
    final = report['final_counters']['optimizer_updates']; start = report.get('resume', {}).get('completed_update', 0)
    check('integer_final_counters', all(type(v) is int for v in report['final_counters'].values()))
    expected = expected_counts(payload, final)
    equal('final_counters', report['final_counters'], expected)
    check('valid_segment', type(start) is int and 0 <= start <= final <= len(payload['plan']['updates']))
    equal('loop_start', report['loop']['start_update'], start)
    equal('loop_final', report['loop']['completed_update'], final)
    equal('plan_completed', report['plan_completed'], final == len(payload['plan']['updates']))
    equal('status_matches_plan', report['status'] == 'completed_plan', report['plan_completed'])
    updates = report.get('updates', {})
    equal('complete_update_membership', set(updates), {str(i) for i in range(start+1, final+1)})
    equal('graph_prepared', report['graph_prepared'], final > start)
    if report['graph_prepared']:
        equal('preparation_exact', report['preparation_boundary_exact'], [True]*2)
        check('runner_metadata', len(report['runner_by_rank']) == 2 and all(isinstance(x, dict) and x for x in report['runner_by_rank']))
    else:
        check('no_preparation_evidence', 'preparation_boundary_exact' not in report)
        equal('no_captured_runners', report['runner_by_rank'], [None, None])
    if 'resume' in report:
        equal('resume_reference_dependency', report['resume']['reference_report_required'],True)
        equal('resume_parent_files_dependency',report['resume']['parent_files_required'],True)
        equal('resume_parent_boundary_compared',report['resume']['parent_boundary_compared'],
              not report.get('branch_resume',False))
        check('resume_pin', re.fullmatch('[0-9a-f]{64}', report['resume']['manifest_sha256']) is not None)
    boundary_check(audit, label+'/origin', report['origin_boundary_by_rank'], payload, start)
    boundary_check(audit, label+'/final', report['final_boundary_by_rank'], payload, final)
    mode = report['observation_mode']; check('observer_mode', mode in ('lean', 'acceptance'))
    active = {p['name'] for p in payload['model_contract']['parameter_layout'] if p['requires_grad']}
    check('active_parameters', bool(active))
    for label_clock, completed in (('origin_clocks',start),('final_clocks',final)):
        equal(label_clock, report[label_clock], {'adam_parameters':len(active) if completed else 0,
              'input_tokens':expected_counts(payload,completed)['input_tokens'],
              'optimizer_updates':completed,'scheduler_epoch':completed})
    equal('complete_accounting_membership', set(report.get('observations',{})), set(updates))
    for step in range(start+1, final+1):
        rank_data=report['observations'][str(step)]['rank_data']
        check(f'update{step}/accounting_ranks', isinstance(rank_data,list) and len(rank_data)==2)
        plan_row=payload['plan']['updates'][step-1]
        for key, value in plan_row['counts'].items():
            equal(f'update{step}/data_total/{key}',sum(row[key] for row in rank_data),value)
        for rank, allocated in enumerate(plan_row['allocation_by_arm'][payload['arm']]):
            for actual_key, planned_key in (('valid_tokens','valid_tokens'),('packed_rows','packed_rows'),
                    ('physical_rows','physical_rows'),('padding_tokens','padding_tokens'),('empty_rows','dummy_rows')):
                equal(f'update{step}/allocation/rank{rank}/{actual_key}',rank_data[rank][actual_key],allocated[planned_key])
        rows = updates[str(step)]; check(f'update{step}/rank_count', isinstance(rows, list) and len(rows) == 2)
        if mode == 'acceptance':
            boundary_check(audit, label+f'/update{step}/boundary', [row['boundary'] for row in rows], payload, step)
        before, after = expected_counts(payload, step-1), expected_counts(payload, step)
        for rank, row in enumerate(rows):
            prefix = f'update{step}/rank{rank}/'
            equal(prefix+'observation_fields',set(row), {'schema','mode','metrics','cursor','loss_means','clipping','observation_seconds'}
                  | (set(HEAVY) if mode=='acceptance' else set()))
            equal(prefix+'schema', row['schema'], OBSERVATION_SCHEMA)
            equal(prefix+'mode', row['mode'], mode)
            metrics = row['metrics']
            check(prefix+'complete_metrics', isinstance(metrics,dict) and
                  {'loss_sums','counts','objective','microbatches','documents','input_tokens',
                   'gradient_norm_before_clip','lr_used','lr_next','counters'} <= metrics.keys())
            check(prefix+'lr_groups', isinstance(metrics['lr_used'],list) and bool(metrics['lr_used'])
                  and isinstance(metrics['lr_next'],list) and len(metrics['lr_used']) == len(metrics['lr_next'])
                  and all(type(v) in (int,float) and v >= 0 for key in ('lr_used','lr_next') for v in metrics[key]))
            equal(prefix+'replica_metrics', metrics, rows[0]['metrics'])
            equal(prefix+'counters', metrics['counters'], after)
            equal(prefix+'cursor', row['cursor'], expected_cursor(payload, step, rank))
            for key in ('input_tokens','documents','microbatches'):
                equal(prefix+key, metrics[key], after[key]-before[key])
            counts = {term: after[key]-before[key] for term,key in zip(TERMS,('ce_positions','latent_pairs','kl_triples'))}
            equal(prefix+'loss_counts', metrics['counts'], counts)
            equal(prefix+'loss_terms', set(metrics['loss_sums']), set(TERMS))
            equal(prefix+'loss_means', row['loss_means'], {term: metrics['loss_sums'][term]/counts[term] if counts[term] else None for term in TERMS})
            limit = configuration['training']['max_grad_norm']; norm = metrics['gradient_norm_before_clip']
            check(prefix+'gradient_norm', type(norm) in (int,float) and norm >= 0)
            clipping = row['clipping']
            equal(prefix+'clip_limit', clipping['configured_limit'], limit)
            equal(prefix+'clip_exceeded', clipping['norm_exceeds_limit'], limit is not None and norm > limit)
            equal(prefix+'clip_coefficient', clipping['coefficient_estimate'], 1.0 if limit is None else min(1.0,limit/(norm+1e-6)))
            if mode == 'lean':
                check(prefix+'no_heavy_evidence', not set(HEAVY)&row.keys())
            else:
                check(prefix+'input_evidence', isinstance(row['input'], dict) and bool(row['input']))
                equal(prefix+'all_active_gradients', set(row['raw_gradients']), active)
                check(prefix+'gradient_digests', all(isinstance(v, dict) and {'shape','dtype','sha256'} == set(v)
                      and re.fullmatch('[0-9a-f]{64}', v['sha256']) is not None for v in row['raw_gradients'].values()))
                equal(prefix+'replica_gradients', row['raw_gradients'], rows[0]['raw_gradients'])
    if mode == 'acceptance' and final > start:
        equal('last_update_final_boundary', [row['boundary'] for row in updates[str(final)]], report['final_boundary_by_rank'])
    if final == start:
        equal('no_update_state_change', report['final_boundary_by_rank'], report['origin_boundary_by_rank'])
    return payload


def panel_check(audit, entry, panel, payload, scope):
    fixed = panel['fixed_plan']['updates'][0]; raw = fixed['counts']
    arm = payload['arm']; recipe = payload['recipe']
    expected = {'ce':raw['ce_targets'], 'latent':raw['latent_pairs'] if 'N' in arm else 0,
                'kl':raw['kl_triples'] if 'N' in arm else 0}
    count = recipe['fbt_passes'] if 'F' in arm else 1
    coefficients = {'ce':[1.] if count == 1 else [.5]+[.5/(count-1)]*(count-1),
        'latent':[1./count]*count, 'kl':[1./count]*count}
    weights = payload['model_contract']['weights']
    mode = {'enabled':'F' in arm, 'num_passes':count, 'beta':1., 'feedback_jitter':0.,
        'first_pass_policy':recipe['first_pass_policy'], 'document_policy':recipe.get('document_policy','isolated-v1'),
        'rt_mode':{'selected_layers':recipe['rt_layers'] if 'R' in arm else [], 'alpha':1.}}
    audit.equal(scope+'/rank_count',len(entry['by_rank']),2)
    local=[]
    for rank,record in enumerate(entry['by_rank']):
        evidence=record['preservation'];checks=evidence['checks']
        required={'tensor_metadata_unchanged','module_ownership_unchanged','cache_generations_unchanged',
            'gradient_identity_unchanged','gradient_values_remained_zero','runtime_restored','modes_restored',
            'rng_restored','autocast_cache_restored'}
        audit.equal(scope+f'/rank{rank}/check_membership',set(checks),required)
        audit.require(scope+f'/rank{rank}/preserved',evidence['restored'] is True and evidence['integrity_passed'] is True
            and all(value is True for value in checks.values()))
        audit.equal(scope+f'/rank{rank}/precision',evidence['precision'],'fp32_math_eager')
        audit.equal(scope+f'/rank{rank}/noise',evidence['feedback_jitter'],0.)
        audit.equal(scope+f'/rank{rank}/dev_cursor',record['cursor'],panel['fixed_plan']['first_cursor'])
        allocation=fixed['allocation_by_arm'][arm][rank]
        audit.equal(scope+f'/rank{rank}/physical_calls',len(record['rows']),allocation['microbatches'])
        for actual_key,planned_key in (('valid_tokens','valid_tokens'),('packed_rows','packed_rows'),
                ('physical_rows','physical_rows'),('padding_tokens','padding_tokens'),('empty_rows','dummy_rows')):
            audit.equal(scope+f'/rank{rank}/allocation/'+actual_key,record['accounting'][actual_key],allocation[planned_key])
        audit.equal(scope+f'/rank{rank}/local_input_sum',sum(row['input_tokens'] for row in record['rows']),record['accounting']['valid_tokens'])
        for term,key in zip(TERMS,('ce_targets','latent_pairs','kl_triples')):
            audit.equal(scope+f'/rank{rank}/local_target_sum/'+term,sum(row['counts'][term] for row in record['rows']),
                record['accounting'][key] if weights[term] else 0)
        local.extend(record['rows'])
    for key,value in raw.items():
        audit.equal(scope+'/global_accounting/'+key,sum(r['accounting'][key] for r in entry['by_rank']),value)
    for i,row in enumerate(local):
        prefix=scope+'/physical'+str(i)
        audit.equal(prefix+'/schema',row['schema'],'olmo-campaign-local-evaluation-v1')
        audit.equal(prefix+'/policy',row['policy'],'common_fp32_no_jitter_v1')
        audit.equal(prefix+'/mode',row['mode'],mode)
        audit.equal(prefix+'/weights',row['weights'],weights)
        audit.equal(prefix+'/enabled',row['enabled'],{k:bool(v) for k,v in weights.items()})
        audit.equal(prefix+'/coefficients',row['term_pass_coefficients'],coefficients)
        audit.equal(prefix+'/pass_membership',[p['index'] for p in row['passes']],list(range(count)))
        audit.equal(prefix+'/count_terms',set(row['counts']),set(TERMS))
        audit.require(prefix+'/nonnegative_counts',all(type(v) is int and v>=0 for v in row['counts'].values()))
        audit.require(prefix+'/nonnegative_tokens',type(row['input_tokens']) is int and row['input_tokens']>=0)
        audit.equal(prefix+'/aggregate_terms',set(row['aggregate_sums']),set(TERMS))
        for p in row['passes']:
            audit.equal(prefix+f'/pass{p["index"]}/counts',p['counts'],row['counts'])
            audit.equal(prefix+f'/pass{p["index"]}/sum_terms',set(p['sums']),set(TERMS))
        for sums in [row['aggregate_sums'],*[p['sums'] for p in row['passes']]]:
            audit.require(prefix+'/numeric_sums',all(type(v) in (int,float) and math.isfinite(v) for v in sums.values()))
            for term in TERMS:
                if not weights[term]:audit.equal(prefix+'/disabled_'+term,(sums[term],row['counts'][term]),(0.,0))
    counts={t:sum(r['counts'][t] for r in local) for t in TERMS}
    audit.equal(scope+'/global_targets',counts,expected)
    audit.equal(scope+'/global_inputs',sum(r['input_tokens'] for r in local),raw['valid_tokens'])
    def reduce_sums(values):
        sums={t:math.fsum(v[t] for v in values) for t in TERMS}
        for term in TERMS:
            if not weights[term]:audit.equal(scope+'/disabled_'+term,(sums[term],counts[term]),(0.,0))
            else:audit.require(scope+'/positive_'+term,counts[term]>0)
        return {'sums':sums,'counts':counts,'means':{t:sums[t]/counts[t] if counts[t] else None for t in TERMS}}
    passes=[{'index':i,**reduce_sums([r['passes'][i]['sums'] for r in local])} for i in range(count)]
    aggregate=reduce_sums([r['aggregate_sums'] for r in local])
    reconstructed={t:math.fsum(coefficients[t][i]*passes[i]['sums'][t] for i in range(count)) for t in TERMS}
    for term in TERMS:
        audit.require(scope+'/weighted_rounding/'+term,math.isclose(reconstructed[term],aggregate['sums'][term],rel_tol=2e-6,abs_tol=1e-6))
    wanted={'schema':'olmo-kl-continuation-evaluation-v1','passes':passes,'aggregate':aggregate,
        'objective':math.fsum(weights[t]*(aggregate['means'][t] or 0.) for t in TERMS),
        'input_tokens':raw['valid_tokens'],'enabled':{t:bool(weights[t]) for t in TERMS},'weights':weights,
        'term_pass_coefficients':coefficients,'reconstructed_aggregate_sums':reconstructed,'policy':'common_fp32_no_jitter_v1'}
    audit.equal(scope+'/independent_global_reduction',entry['result'],wanted)


def evaluation_check(audit, report, label):
    policy = report['evaluation_policy']; plan = policy['plan']; declaration = plan['declaration']
    payload = report['configuration']['execution_identity']['payload']
    audit.equal(label+'/resume_policy', policy['resume'], 'repeat_scheduled_restored_boundary_once_per_segment')
    audit.equal(label+'/failure_policy', policy['failure'],
                'earlier committed checkpoint remains authoritative; no save of unverified state')
    audit.equal(label+'/training_batch', policy['training_physical_batch_per_rank'], payload['partition']['physical_batch_per_rank'])
    if report['scale'] == 'native':
        audit.equal(label+'/bound_evaluation', payload['evaluation'], plan)
    else:
        audit.equal(label+'/tiny_schema', payload['declaration']['schema'], TINY_SCHEMA)
        audit.equal(label+'/tiny_evaluation', payload['declaration']['evaluation'], declaration)
        audit.equal(label+'/tiny_declaration_digest', payload['resolved_contract_sha256'], digest(payload['declaration']))
    if declaration['kind'] == 'deferred':
        audit.equal(label+'/deferred_fields', set(declaration), {'kind','reason'})
        audit.require(label+'/deferred_reason', isinstance(declaration['reason'],str) and bool(declaration['reason'].strip()))
        audit.equal(label+'/deferred_plan', set(plan), {'declaration','execution'})
        audit.equal(label+'/deferred_evaluations', report['evaluations'], [])
        audit.equal(label+'/deferred_batch', policy['evaluation_physical_batch_per_rank'], None)
        return
    audit.equal(label+'/policy_fields', set(declaration), {'kind','panels','physical_batch_by_arm','every_updates',
        'precision','feedback_jitter','report_passes','generation'})
    for key,value in {'kind':'ordered_named_dev_panels_v1','precision':'fp32','feedback_jitter':0,
                      'report_passes':'all_trained_passes','generation':'not_implemented'}.items():
        audit.equal(label+'/'+key, declaration[key], value)
    audit.equal(label+'/plan_schema', plan['schema'], 'olmo-pilot-evaluation-control-v1')
    names = [row['name'] for row in declaration['panels']]
    allowed = {'dev-main'} | {'dev-source/'+s for s in ('books','c4','cc_en_head','cc_en_middle','cc_en_tail','pes2o','reddit','stack','wiki')}
    audit.require(label+'/unique_dev_only', bool(names) and len(names)==len(set(names)) and set(names)<=allowed)
    audit.equal(label+'/declared_panel_inventory', set(plan['panels']), set(names))
    interval = declaration['every_updates']
    audit.require(label+'/positive_interval', type(interval) is int and interval>0)
    scheduled = list(range(interval, len(payload['plan']['updates'])+1, interval))
    audit.equal(label+'/fixed_schedule', plan['scheduled_updates'], scheduled)
    start=report.get('resume',{}).get('completed_update',0); final=report['final_counters']['optimizer_updates']
    audit.equal(label+'/actual_schedule', [r['after_update'] for r in report['evaluations']], [s for s in scheduled if start<=s<=final])
    arm = payload['arm']; batch = declaration['physical_batch_by_arm'][arm]
    audit.require(label+'/positive_eval_batch', type(batch) is int and batch>0)
    audit.equal(label+'/eval_batch', policy['evaluation_physical_batch_per_rank'], batch)
    audit.equal(label+'/eval_partition', plan['partition_by_arm'][arm], {'world_size':2, 'physical_batch_per_rank':batch})
    audit.equal(label+'/overlap_policy', plan['overlap_policy'],
                'Report each panel separately; main/source overlap is not independent replication')
    for declared in declaration['panels']:
        name=declared['name']; panel=plan['panels'][name]; scope=label+'/'+name
        audit.require(scope+'/pin', re.fullmatch('[0-9a-f]{64}',panel['index_manifest_sha256']) is not None)
        audit.equal(scope+'/budget', panel['target_valid_tokens'], declared['target_valid_tokens'])
        audit.require(scope+'/positive_budget', type(panel['target_valid_tokens']) is int and panel['target_valid_tokens']>0)
        fixed=panel['fixed_plan']; audit.equal(scope+'/one_update',len(fixed['updates']),1)
        audit.equal(scope+'/origin',fixed['first_cursor'],{'manifest_sha256':panel['index_manifest_sha256'], 'split':'dev','next_chunk':0,'next_update':0})
        audit.equal(scope+'/planned_budget', fixed['updates'][0]['target_valid_tokens'], panel['target_valid_tokens'])
        audit.equal(scope+'/selection',panel['selection'],'fixed_ordered_prefix_from_chunk_zero')
        audit.require(scope+'/available_budget', panel['available_panel_tokens']>=panel['target_valid_tokens'])
        audit.equal(scope+'/allocation_rank_count', len(fixed['updates'][0]['allocation_by_arm'][arm]), 2)
        for rank, row in enumerate(fixed['updates'][0]['allocation_by_arm'][arm]):
            audit.equal(scope+f'/rank{rank}/batch_product',row['physical_rows'],row['microbatches']*batch)
    for row in report['evaluations']:
        scope=label+'/update'+str(row['after_update'])
        audit.equal(scope+'/schema',row['schema'],'olmo-kl-continuation-evaluation-v1')
        audit.equal(scope+'/completed',row['status'],'completed')
        audit.equal(scope+'/training_boundary_preserved',row['training_boundary_exact_by_rank'],[True,True])
        audit.equal(scope+'/panels',set(row['panels']),set(names))
        for name, entry in row['panels'].items():
            panel=plan['panels'][name]
            audit.equal(scope+'/'+name+'/index_pin',entry['index_manifest_sha256'],panel['index_manifest_sha256'])
            audit.equal(scope+'/'+name+'/membership_pin',entry['membership_sha256'],panel['fixed_plan']['updates'][0]['membership_sha256'])
            panel_check(audit,entry,panel,payload,scope+'/'+name)


def _same_except_kl(audit, actual, original, label):
    """Compare configuration after separately validating transition metadata."""
    candidate, expected = deepcopy(actual), deepcopy(original)
    for value in (candidate, expected):
        value.pop('execution_identity', None)
        value.pop('objective_branch', None)
        value.pop('schema', None)
        value['model']['lambda_kl'] = 1.
        value['parameters']['weights']['kl'] = 1.
        value['recipe']['auxiliary']['kl'] = 1.
        value['recipe']['optimizer_state'] = 'validated_inherited_state'
        value['recipe'].pop('objective_transition', None)
    audit.equal(label, candidate, expected)


def _recipe_check(audit, recipe, original, branch, label):
    expected = deepcopy(original)
    expected['auxiliary']['kl'] = branch['kl_weight']
    expected['optimizer_state'] = 'inherited_exact_parent_checkpoint'
    expected['objective_transition'] = {
        'schema':'olmo-kl-objective-transition-v1',
        'parent_manifest_sha256':branch['parent_manifest_sha256'],
        'parent_kl_weight':1., 'kl_weight':branch['kl_weight'],
        'changed_field':'lambda_kl', 'same_weight_control':branch['kl_weight']==1.,
        'retained_state':['model','optimizer','scheduler','counters','data_cursor','rng'],
        'resume':'exact_branch_identity_without_reapplying_parent_transition'}
    audit.equal(label, recipe, expected)


def _configuration_receipt(audit, receipt, report, label):
    audit.equal(label+'/fields',set(receipt),{'schema','configuration_before','configuration_after',
        'weights_before','weights_after','checks','scope','before_graph_construction_required'})
    audit.equal(label+'/schema',receipt['schema'],'olmo-kl-objective-transition-v1')
    audit.equal(label+'/before_graph',receipt['before_graph_construction_required'],True)
    required={'parameters_unchanged','buffers_unchanged','modules_unchanged',
        'only_lambda_kl_changed','actual_objective_matches'}
    audit.equal(label+'/checks',receipt['checks'],dict.fromkeys(required,True))
    audit.equal(label+'/weights_before',receipt['weights_before'],{'ce':1.,'latent':1.,'kl':1.})
    audit.equal(label+'/weights_after',receipt['weights_after'],report['configuration']['parameters']['weights'])
    before,after=receipt['configuration_before'],receipt['configuration_after']
    audit.equal(label+'/owners_before',set(before),{'model','predictor'})
    audit.equal(label+'/owners_after',set(after),{'model','predictor'})
    audit.equal(label+'/old_model',before['model'],report['original_configuration']['model'])
    audit.equal(label+'/new_model',after['model'],report['configuration']['model'])
    for owner in before:
        wanted=deepcopy(before[owner]);wanted['lambda_kl']=report['branch']['kl_weight']
        audit.equal(label+'/'+owner+'/old_KL',before[owner]['lambda_kl'],1.)
        audit.equal(label+'/'+owner+'/only_KL',after[owner],wanted)


def _branch_check(audit, report, label, *, expected_stop):
    branch = report['branch']; configuration = report['configuration']
    payload = configuration['execution_identity']['payload']
    audit.equal(label+'/branch_fields', set(branch), {'schema','kl_weight','parent_manifest_sha256',
        'parent_identity_sha256','parent_update','review_stop','recipe_as_declared','parent_report_sha256'})
    audit.equal(label+'/branch_schema', branch['schema'], BRANCH_SCHEMA)
    audit.require(label+'/supported_weight', type(branch['kl_weight']) in (int,float)
                  and branch['kl_weight'] in (1., .1))
    audit.require(label+'/valid_review', type(branch['parent_update']) is int
                  and 0 < branch['parent_update'] < branch['review_stop'])
    audit.equal(label+'/review_stop', branch['review_stop'], expected_stop)
    audit.equal(label+'/completed_review', report['final_counters']['optimizer_updates'], expected_stop)
    for key in ('parent_manifest_sha256','parent_identity_sha256','parent_report_sha256'):
        audit.require(label+'/'+key, isinstance(branch[key],str) and re.fullmatch('[0-9a-f]{64}', branch[key]) is not None)
    audit.equal(label+'/configuration_schema', configuration['schema'], CONFIG_SCHEMA)
    audit.equal(label+'/configuration_branch', configuration['objective_branch'], branch)
    audit.equal(label+'/identity_branch', payload['objective_branch'], branch)
    weights = {'ce':1., 'latent':1., 'kl':branch['kl_weight']}
    audit.equal(label+'/actual_model_KL', configuration['model']['lambda_kl'], branch['kl_weight'])
    audit.equal(label+'/actual_weights', configuration['parameters']['weights'], weights)
    audit.equal(label+'/declared_recipe_KL', configuration['recipe']['auxiliary']['kl'], branch['kl_weight'])
    audit.equal(label+'/operative_recipe', branch['recipe_as_declared'], configuration['recipe'])
    _recipe_check(audit, configuration['recipe'], report['original_configuration']['recipe'], branch,
                  label+'/recipe_KL_and_inherited_metadata_only')
    for key in ('parent_manifest_sha256','parent_report_sha256'):
        audit.equal(label+'/reported_'+key, report[key], branch[key])
    audit.equal(label+'/original_identity', report['original_configuration']['execution_identity']['sha256'],
                branch['parent_identity_sha256'])
    _same_except_kl(audit, configuration, report['original_configuration'], label+'/only_KL_math_changed')
    audit.equal(label+'/Adam_resident_before_DDP', report['adam_resident_before_ddp'], True)
    if report.get('branch_resume'):
        audit.equal(label+'/child_resume_flag', report['branch_resume'], True)
        audit.require(label+'/no_second_transition', 'objective_transition' not in report)
        audit.require(label+'/child_resume_boundary', report['resume']['completed_update'] > branch['parent_update'])
    else:
        audit.require(label+'/no_spurious_resume_flag', 'branch_resume' not in report)
        audit.equal(label+'/fork_origin', report['resume']['completed_update'], branch['parent_update'])
        audit.equal(label+'/parent_pin', report['resume']['manifest_sha256'], branch['parent_manifest_sha256'])
        receipt = report['objective_transition']
        audit.equal(label+'/transition_fields', set(receipt), {'parent_manifest_sha256','before_boundary_by_rank',
                    'after_boundary_by_rank','configuration_change_by_rank'})
        audit.equal(label+'/transition_parent', receipt['parent_manifest_sha256'], branch['parent_manifest_sha256'])
        audit.equal(label+'/complete_state_preserved', receipt['before_boundary_by_rank'], receipt['after_boundary_by_rank'])
        audit.equal(label+'/transition_is_origin', receipt['after_boundary_by_rank'], report['origin_boundary_by_rank'])
        audit.require(label+'/two_configuration_receipts', len(receipt['configuration_change_by_rank']) == 2)
        for rank,change in enumerate(receipt['configuration_change_by_rank']):
            _configuration_receipt(audit,change,report,label+f'/configuration_change/rank{rank}')
        audit.equal(label+'/replicated_configuration_change',receipt['configuration_change_by_rank'][0],
                    receipt['configuration_change_by_rank'][1])
    if report['scale'] == 'native':
        audit.equal(label+'/native_scope', (report['arm'],branch['parent_update'],expected_stop), ('NF',32,64))
        audit.equal(label+'/native_length', payload['recipe']['sequence_length'], 1024)
        audit.equal(label+'/native_effective_batch', payload['recipe']['effective_valid_tokens'], 524288)
        audit.equal(label+'/native_physical_batch', payload['partition']['physical_batch_per_rank'], 12)
    else:
        audit.equal(label+'/tiny_scale', report['scale'], 'tiny')


def validate_report(audit, report, label, *, expected_stop, source_root=None, check_transport=True):
    payload = training_check(audit, report, label, source_root)
    _branch_check(audit, report, label, expected_stop=expected_stop)
    evaluation_check(audit, report, label+'/evaluation')
    objective = report['evaluation_policy']['objective']
    audit.equal(label+'/evaluation_weights', objective['weights'], payload['model_contract']['weights'])
    audit.equal(label+'/evaluation_objective_schema', objective['schema'], 'olmo-kl-continuation-evaluation-v1')
    schedule=payload['schedule']
    audit.equal(label+'/schedule_count',len(schedule['lr_at_completed_boundaries']),len(payload['plan']['updates'])+1)
    audit.equal(label+'/token_prefix',schedule['valid_token_prefix'],
                [expected_counts(payload,n)['input_tokens'] for n in range(len(payload['plan']['updates'])+1)])
    for step,rows in report.get('updates',{}).items():
        for key,index in (('lr_used',int(step)-1),('lr_next',int(step))):
            actual=rows[0]['metrics'][key]
            audit.equal(label+f'/update{step}/'+key,actual,[schedule['lr_at_completed_boundaries'][index]]*len(actual))
    if check_transport:
        transport_check(audit, report, label+'/transport')
    return payload


def _parent_check(audit, parent, child, label, parent_report_sha256=None):
    branch = child['branch']; completed = branch['parent_update']
    audit.equal(label+'/parent_configuration', child['original_configuration'], parent['configuration'])
    audit.equal(label+'/parent_identity', branch['parent_identity_sha256'], parent['configuration']['execution_identity']['sha256'])
    audit.equal(label+'/parent_identity_digest',branch['parent_identity_sha256'],
                digest(parent['configuration']['execution_identity']['payload']))
    if parent_report_sha256 is not None:
        audit.equal(label+'/parent_report_pin',branch['parent_report_sha256'],parent_report_sha256)
    source = [row for row in parent.get('local_checkpoints', []) if row['optimizer_update'] == completed]
    audit.require(label+'/parent_publication_exists', len(source) == 1)
    audit.equal(label+'/parent_manifest', source[0]['receipt']['manifest_sha256'], branch['parent_manifest_sha256'])
    audit.equal(label+'/inherited_complete_state', child['origin_boundary_by_rank'], legacy.boundary_at(parent, completed))
    parent_payload = parent['configuration']['execution_identity']['payload']
    payload = child['configuration']['execution_identity']['payload']
    wanted=deepcopy(parent_payload)
    wanted.update(scope='Explicit saved-Adam KL objective fork; not same-configuration resume',
        objective_branch=branch,recipe=child['configuration']['recipe'],
        model_contract=child['configuration']['parameters'],sources=child['sources'])
    audit.equal(label+'/only_declared_identity_changes',payload,wanted)
    for key, pin in parent['sources'].items():
        audit.equal(label+'/preserved_source/'+key, child['sources'].get(key), pin)


def _raw_evaluation(entry):
    return {name: {'membership_sha256': panel['membership_sha256'],
        'index_manifest_sha256': panel['index_manifest_sha256'],
        'passes': panel['result']['passes'], 'aggregate': panel['result']['aggregate'],
        'input_tokens': panel['result']['input_tokens'],
        'rank_raw': [{'cursor': rank['cursor'], 'accounting': rank['accounting'],
            'rows': [{k:v for k,v in row.items() if k != 'weights'} for row in rank['rows']]}
                     for rank in panel['by_rank']]}
        for name, panel in entry['panels'].items()}


def _result(audit, error=None, **extra):
    result = audit.result(error)
    result.update(schema=SCHEMA, **extra,
        scope='Pinned JSON lineage, exact preserved state digests, ordered membership/accounting and named evaluation; '
              'no new tensor reload or cloud readback. Native lean inputs bind deterministic data/mask/jitter '
              'contracts; exact physical input/noise bytes are checked in tiny acceptance.')
    return result


def audit_pair(parent_report, control_report, reduced_report, *, expected_stop,
               source_roots=None, check_transport=True, parent_report_sha256=None):
    """Audit two new fork reports against the exact retained common parent."""
    audit = Audit(); source_roots = source_roots or {}
    try:
        reports = {'control':control_report, 'reduced':reduced_report}
        for label, report in reports.items():
            validate_report(audit, report, label, expected_stop=expected_stop,
                            source_root=source_roots.get(label), check_transport=check_transport)
            audit.require(label+'/is_first_fork_segment', not report.get('branch_resume',False))
            _parent_check(audit, parent_report, report, label, parent_report_sha256)
        a,b = control_report,reduced_report
        audit.equal('pair/weights', (a['branch']['kl_weight'],b['branch']['kl_weight']), (1.,.1))
        for field in a['branch']:
            if field not in ('kl_weight','recipe_as_declared'):
                audit.equal('pair/branch/'+field, a['branch'][field], b['branch'][field])
        _same_except_kl(audit, a['configuration'], b['configuration'], 'pair/only_declared_KL_change')
        audit.equal('pair/origin_complete_state', a['origin_boundary_by_rank'], b['origin_boundary_by_rank'])
        ap,bp = (r['configuration']['execution_identity']['payload'] for r in (a,b))
        for key in ('data','partition','plan','schedule','execution','sources','runtime','determinism','storage_policy'):
            audit.equal('pair/'+key, ap[key], bp[key])
        audit.equal('pair/evaluation_plan', a['evaluation_policy']['plan'], b['evaluation_policy']['plan'])
        first = a['branch']['parent_update']+1
        for key in ('loss_sums','counts'):
            audit.equal('pair/first_forward_'+key, a['updates'][str(first)][0]['metrics'][key],
                        b['updates'][str(first)][0]['metrics'][key])
        for step in range(first,expected_stop+1):
            audit.equal(f'pair/update{step}/accounting', a['observations'][str(step)]['rank_data'],
                        b['observations'][str(step)]['rank_data'])
            if a['observation_mode'] == b['observation_mode'] == 'acceptance':
                audit.equal(f'pair/update{step}/inputs_noise', [r['input'] for r in a['updates'][str(step)]],
                            [r['input'] for r in b['updates'][str(step)]])
        if a['scale']=='tiny':
            audit.equal('control/acceptance',a['observation_mode'],'acceptance')
            training_parity(audit,parent_report,a)
            previous={row['after_update']:row for row in parent_report['evaluations']}
            for row in a['evaluations']:
                audit.equal('control/raw_evaluation'+str(row['after_update']),_raw_evaluation(row),
                            _raw_evaluation(previous[row['after_update']]))
        origin = first-1
        ae = {r['after_update']:r for r in a['evaluations']}
        be = {r['after_update']:r for r in b['evaluations']}
        if origin in ae or origin in be:
            audit.require('pair/origin_eval_both', origin in ae and origin in be)
            audit.equal('pair/origin_raw_evaluation', _raw_evaluation(ae[origin]), _raw_evaluation(be[origin]))
    except (KeyError,TypeError,ValueError,IndexError,OSError) as error:
        return _result(audit, f'{type(error).__name__}: {error}', comparison_kind='paired_fork')
    return _result(audit, comparison_kind='paired_fork')


def audit_restart(reference, resumed, *, source_publication, expected_stop,
                  source_roots=None, check_transport=True):
    """Require exact tiny continuation after a committed child-branch checkpoint."""
    audit = Audit(); source_roots = source_roots or {}
    try:
        for label, report in (('reference',reference),('resumed',resumed)):
            validate_report(audit,report,label,expected_stop=expected_stop,
                            source_root=source_roots.get(label),check_transport=check_transport)
            audit.equal(label+'/tiny_acceptance',(report['scale'],report['observation_mode']),('tiny','acceptance'))
        audit.equal('restart/child_flag',resumed.get('branch_resume'),True)
        for field in ('configuration','branch','original_configuration','sources','source_fingerprint','evaluation_policy'):
            audit.equal('restart/'+field,reference[field],resumed[field])
        audit.require('restart/source_publication',source_publication is not None)
        manifest,_,identity = publication_metadata(source_publication)
        audit.equal('restart/manifest_pin',resumed['resume']['manifest_sha256'],source_publication['manifest_sha256'])
        audit.equal('restart/identity',identity,resumed['configuration']['execution_identity'])
        audit.equal('restart/counters',manifest['counters'],resumed['origin_boundary_by_rank'][0]['state']['counters'])
        audit.equal('restart/cursors',manifest['rank_cursors'],[r['cursor'] for r in resumed['origin_boundary_by_rank']])
        training_parity(audit, reference, resumed)
        previous={r['after_update']:r for r in reference['evaluations']}
        for row in resumed['evaluations']:
            audit.equal('restart/raw_eval'+str(row['after_update']),_raw_evaluation(row),
                        _raw_evaluation(previous[row['after_update']]))
    except (KeyError,TypeError,ValueError,IndexError,OSError) as error:
        return _result(audit,f'{type(error).__name__}: {error}',comparison_kind='same_branch_restart')
    return _result(audit,comparison_kind='same_branch_restart')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('pair','restart'),required=True)
    parser.add_argument('--input',action='append',nargs=3,metavar=('LABEL','PATH','SHA256'),required=True)
    parser.add_argument('--expected-stop',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();reports={};pins={}
    for label,path,pin in args.input:
        if label in reports:parser.error('Duplicate input label')
        reports[label],pins[label]=bounded_json(Path(path),pin)
    required={'parent','control','reduced'} if args.kind=='pair' else {'reference','resumed','publication'}
    if set(reports)!=required:parser.error('Require exact input labels '+str(sorted(required)))
    result=(audit_pair(reports['parent'],reports['control'],reports['reduced'],expected_stop=args.expected_stop,
                      parent_report_sha256=pins['parent']['sha256'])
        if args.kind=='pair' else audit_restart(reports['reference'],reports['resumed'],
            source_publication=reports['publication'],expected_stop=args.expected_stop))
    for pin in pins.values():
        if file_sha(pin['path'])!=pin['sha256']:raise ValueError('Audit input changed')
    result['inputs']=pins
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'failures':result['failures']}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
