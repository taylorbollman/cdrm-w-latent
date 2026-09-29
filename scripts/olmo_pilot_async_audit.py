#!/usr/bin/env python3
"""JSON-only exact blocking/async transport and same-lineage restart audit.

The training validator is copied literally from the accepted ordered auditor,
with this module's explicit new report schema. No input report is relabeled.
The original named-evaluation, storage and meaningful parity rules are reused.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import olmo_pilot_execution_audit as original
from scripts import olmo_pilot_execution_audit_v2 as previous
from scripts.olmo_pilot_execution_audit import (Audit, legacy, storage_audit,
    publication_metadata, expected_counts, expected_cursor, file_sha, bounded_json,
    storage_check, digest, finite_json, boundary_check, OBSERVATION_SCHEMA,
    COUNTERS, TERMS, HEAVY, CONFIG_SCHEMA, IDENTITY_SCHEMA, TINY_SCHEMA)

SCHEMA='olmo-pilot-async-audit-v1'
REPORT_SCHEMA='olmo-pilot-async-execute-report-v1'
TRANSPORT_SCHEMA='olmo-pilot-async-retention-v1'
RESULT_SCHEMA='olmo-pilot-async-retention-result-v1'


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
        equal('resume_no_reference_dependency', report['resume']['reference_report_required'], False)
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


def nonnegative_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def transport_check(audit, report, label):
    payload=report['configuration']['execution_identity']['payload']
    policy=payload['storage_policy']; transport=policy['transport']; mode=transport['mode']
    audit.require(label+'/known_mode',mode in ('async','blocking'))
    keep=policy['keep_local_completed']
    audit.require(label+'/valid_keep',type(keep) is int and 2<=keep<=32)
    expected={'schema':'olmo-campaign-ssd-policy-v1','location':'local_ssd_verified_gcs',
        'keep_local_completed':keep,'transport':{'schema':TRANSPORT_SCHEMA,'mode':mode,
            'maximum_pending':1,'worker_timeout_seconds':480,'terminal_drain':True,
            'full_cloud_readback':True,'training_state_access':'none'}}
    audit.equal(label+'/bound_policy',policy,expected)
    audit.equal(label+'/reported_policy',report['storage']['policy'],policy)
    audit.equal(label+'/reported_transport',report['checkpoint_transport'],
        {'maximum_pending':1,'worker_timeout_seconds':480,'mode':mode})
    local=report.get('local_checkpoints',[]); publications=report.get('published_checkpoints',[])
    storage=report.get('storage_publications',[])
    submissions=report.get('async_submissions',[]); completions=report.get('async_completions',[])
    audit.require(label+'/lists',all(type(rows) is list for rows in (local,publications,storage,submissions,completions)))
    numbers=[row['optimizer_update'] for row in local]
    audit.require(label+'/ordered_local_updates',all(type(n) is int and n>=0 for n in numbers)
        and numbers==sorted(set(numbers)))
    for name, rows in (('submissions',submissions),('completions',completions),
                       ('published',publications),('storage',storage)):
        audit.equal(label+'/'+name+'_count',len(rows),len(local))
    loop=report['loop']
    for key in ('checkpoints_submitted','checkpoints_published'):
        audit.require(label+'/'+key+'_integer',type(loop[key]) is int)
        audit.equal(label+'/'+key,loop[key],len(local))
    audit.equal(label+'/terminal_drained',loop['checkpoint_pending'],False)
    audit.require(label+'/finite_wait',nonnegative_number(loop['checkpoint_wait_seconds']))
    final=report['final_counters']['optimizer_updates']
    audit.equal(label+'/last_local',loop['last_saved_update'],final)
    audit.equal(label+'/last_remote',loop['last_retained_update'],final)
    if numbers:
        audit.equal(label+'/terminal_saved',numbers[-1],final)
        audit.equal(label+'/reported_last_verified',report['last_verified_cloud_update'],final)
    else:
        audit.require(label+'/no_new_work_only_restored_terminal','resume' in report and
            report['resume']['completed_update']==final)
    previous_finish=None
    for index,(number,submitted,done) in enumerate(zip(numbers,submissions,completions)):
        scope=label+'/update'+str(number)
        audit.equal(scope+'/submission_fields',set(submitted),{'schema','update','job_path','job_sha256',
            'state','submitted_at_unix_seconds'})
        audit.equal(scope+'/submission_schema',submitted['schema'],TRANSPORT_SCHEMA)
        audit.equal(scope+'/submission_update',submitted['update'],number)
        audit.equal(scope+'/submission_not_durable',submitted['state'],'pending_not_cloud_durable')
        audit.require(scope+'/job_pin',isinstance(submitted['job_sha256'],str) and
            re.fullmatch('[0-9a-f]{64}',submitted['job_sha256']) is not None)
        audit.equal(scope+'/job_path',submitted['job_path'],str(Path(report['storage']['evidence_dir'])/
            'async-retention'/f'update-{number:06d}'/'job.json'))
        audit.equal(scope+'/completion_fields',set(done),{'schema','status','update','job_sha256','receipt',
            'storage_publication','timing','worker','submitted_at_unix_seconds','completed_at_unix_seconds'})
        for key,value in (('schema',RESULT_SCHEMA),('status','published'),('update',number),
                          ('job_sha256',submitted['job_sha256'])):
            audit.equal(scope+'/'+key,done[key],value)
        audit.equal(scope+'/local_receipt',local[index]['receipt'],
            {k:v for k,v in done['receipt'].items() if k!='retention'})
        audit.equal(scope+'/published_receipt',done['receipt'],publications[index])
        audit.equal(scope+'/storage_publication',done['storage_publication'],storage[index])
        manifest,_,identity=publication_metadata(done['receipt'])
        audit.equal(scope+'/published_identity',identity,report['configuration']['execution_identity'])
        audit.equal(scope+'/published_configuration',manifest['metadata']['configuration'],report['configuration'])
        audit.equal(scope+'/published_source',manifest['metadata']['source_fingerprint'],report['source_fingerprint'])
        audit.equal(scope+'/published_counters',manifest['counters'],expected_counts(payload,number))
        audit.equal(scope+'/published_cursors',manifest['rank_cursors'],[expected_cursor(payload,number,rank) for rank in range(2)])
        audit.equal(scope+'/submitted_clock',done['submitted_at_unix_seconds'],submitted['submitted_at_unix_seconds'])
        begin,end=done['submitted_at_unix_seconds'],done['completed_at_unix_seconds']
        audit.require(scope+'/clock_interval',nonnegative_number(begin) and nonnegative_number(end) and end>=begin)
        audit.require(scope+'/single_worker_interval',previous_finish is None or begin>=previous_finish)
        previous_finish=end
        timings=done['timing']
        audit.equal(scope+'/timing_fields',set(timings),{'local_validation_seconds','cloud_child_seconds',
            'verified_publication_and_pruning_seconds','total_background_seconds'})
        audit.require(scope+'/timing_values',all(nonnegative_number(v) for v in timings.values()))
        audit.require(scope+'/timing_total',timings['total_background_seconds']>=sum(v for k,v in timings.items()
            if k!='total_background_seconds'))
        worker=done['worker']
        audit.equal(scope+'/worker_fields',set(worker),{'pid','cuda_initialized','distributed_initialized',
            'sdk_seconds','cuda_visible_devices'})
        audit.require(scope+'/worker_pid',type(worker['pid']) is int and worker['pid']>0)
        audit.require(scope+'/worker_cuda_not_initialized',worker['cuda_initialized'] is False)
        audit.require(scope+'/worker_distributed_not_initialized',worker['distributed_initialized'] is False)
        audit.equal(scope+'/worker_cuda_hidden',worker['cuda_visible_devices'],'')
        audit.require(scope+'/worker_sdk_clock',nonnegative_number(worker['sdk_seconds']))
    updates=report.get('updates',{})
    intervals=report.get('update_intervals_unix_seconds',{})
    wall=report.get('update_wall_seconds_by_rank',{})
    audit.equal(label+'/update_interval_membership',set(intervals),set(updates))
    audit.equal(label+'/update_wall_membership',set(wall),set(updates))
    overlaps=[]; previous_finish=None
    for number in sorted(intervals,key=int):
        interval=intervals[number]
        audit.equal(label+'/interval'+number+'/fields',set(interval),{'started','finished'})
        begin,end=interval['started'],interval['finished']
        audit.require(label+'/interval'+number+'/clock',nonnegative_number(begin) and nonnegative_number(end) and end>=begin)
        audit.require(label+'/interval'+number+'/ordered',previous_finish is None or begin>=previous_finish)
        previous_finish=end
        audit.require(label+'/wall'+number,len(wall[number])==2 and all(nonnegative_number(v) for v in wall[number]))
        for done in completions:
            overlap=min(end,done['completed_at_unix_seconds'])-max(begin,done['submitted_at_unix_seconds'])
            if overlap>0:
                overlaps.append({'optimizer_update':int(number),'checkpoint_update':done['update'],
                                 'overlap_seconds':overlap})
    if mode=='blocking':
        audit.equal(label+'/blocking_has_no_update_overlap',overlaps,[])
    return {'mode':mode,'worker_update_interval_intersections':overlaps,
        'qualification':'Wall-interval intersections, not device-level concurrent activity or a throughput gain.'}


def validate_report(audit, report, label, source_root=None):
    payload=training_check(audit,report,label,source_root)
    audit.equal(label+'/configuration_schema',report['configuration']['schema'],CONFIG_SCHEMA)
    original.evaluation_check(audit,report,label+'/evaluation')
    for event in report['evaluations']:
        for name,panel in event['panels'].items():
            for rank,part in enumerate(panel['by_rank']):
                audit.require(f'{label}/eval{event["after_update"]}/{name}/rank{rank}/elapsed',
                    nonnegative_number(part['preservation'].get('elapsed_seconds')))
    return payload,transport_check(audit,report,label+'/transport')


def evaluation_parity(audit, reference, actual, label):
    by_step={row['after_update']:row for row in reference['evaluations']}
    for row in actual['evaluations']:
        previous_event=by_step[row['after_update']]
        audit.equal(label+'/eval'+str(row['after_update'])+'/panels',set(row['panels']),set(previous_event['panels']))
        for name,panel in row['panels'].items():
            other=previous_event['panels'][name]
            for key in ('result','index_manifest_sha256','membership_sha256'):
                audit.equal(f'{label}/eval{row["after_update"]}/{name}/'+key,panel[key],other[key])
            for rank,part in enumerate(panel['by_rank']):
                for key in ('rows','accounting','cursor'):
                    audit.equal(f'{label}/eval{row["after_update"]}/{name}/rank{rank}/'+key,
                        part[key],other['by_rank'][rank][key])
                audit.equal(f'{label}/eval{row["after_update"]}/{name}/rank{rank}/preservation_except_elapsed',
                    {k:v for k,v in part['preservation'].items() if k!='elapsed_seconds'},
                    {k:v for k,v in other['by_rank'][rank]['preservation'].items() if k!='elapsed_seconds'})


def load_transport_evidence(root, report):
    """Read only named, pinned JSON jobs and their saved worker results."""
    root=Path(root); evidence={}; pins={}
    for submission in report.get('async_submissions',[]):
        update=submission['update']
        if type(update) is not int or update<0:
            raise ValueError('Transport update must be a nonnegative integer')
        relative=Path('async-retention')/f'update-{update:06d}'
        job,job_pin=bounded_json(root/relative/'job.json',submission['job_sha256'])
        result_path=root/relative/'result.json'
        result,result_pin=bounded_json(result_path,file_sha(result_path))
        if str(update) in evidence:
            raise ValueError('Duplicate transport job update')
        evidence[str(update)]={'job':job,'result':result}
        pins[str(relative/'job.json')]=job_pin
        pins[str(relative/'result.json')]=result_pin
    return evidence,pins


def transport_evidence_check(audit, report, evidence, label):
    submissions=report['async_submissions']
    audit.equal(label+'/membership',set(evidence),{str(row['update']) for row in submissions})
    payload=report['configuration']['execution_identity']['payload']
    # The producer ran within its declared repository. Do not read those absolute
    # paths here: an audit may run after restoration on a different host.
    producer_root=report['storage']['evidence_dir'].split('/.runtime/',1)[0]
    audit.require(label+'/producer_root',producer_root!=report['storage']['evidence_dir'])
    expected_sources={str(Path(producer_root)/name):pin for name,pin in report['sources'].items()}
    prefix=payload['declaration']['storage_prefix']+'/'+report['arm']+'/'+Path(report['storage']['evidence_dir']).name
    for index,row in enumerate(submissions):
        scope=label+'/update'+str(row['update']); item=evidence[str(row['update'])];job=item['job']
        audit.equal(scope+'/job_fields',set(job),{'schema','receipt','source_pins','storage_prefix',
            'worker_timeout_seconds','submitted_at_unix_seconds'})
        audit.equal(scope+'/schema',job['schema'],TRANSPORT_SCHEMA)
        audit.equal(scope+'/local_receipt',job['receipt'],report['local_checkpoints'][index]['receipt'])
        audit.equal(scope+'/sources',job['source_pins'],expected_sources)
        audit.equal(scope+'/storage_prefix',job['storage_prefix'],prefix)
        audit.equal(scope+'/timeout',job['worker_timeout_seconds'],480)
        audit.equal(scope+'/submitted_at',job['submitted_at_unix_seconds'],row['submitted_at_unix_seconds'])
        audit.equal(scope+'/saved_worker_result',item['result'],report['async_completions'][index])


def compare(reference, actual, *, kind, reference_source_root=None, actual_source_root=None,
            storage_evidence=None, resume_publication=None, reference_transport_evidence=None,
            actual_transport_evidence=None):
    audit=Audit(); overlaps={}
    try:
        if kind not in ('transport','resume'):raise ValueError('Unknown async audit kind')
        if reference.get('scale')!='tiny' or actual.get('scale')!='tiny':
            raise ValueError('Async comparison audit is bounded to tiny acceptance')
        rp,overlaps['reference']=validate_report(audit,reference,'reference',reference_source_root)
        ap,overlaps['actual']=validate_report(audit,actual,'actual',actual_source_root)
        for label,report,evidence in (('reference',reference,reference_transport_evidence),
                                     ('actual',actual,actual_transport_evidence)):
            if evidence is not None:transport_evidence_check(audit,report,evidence,label+'/transport_files')
        audit.equal('comparison/acceptance',(reference['observation_mode'],actual['observation_mode']),('acceptance','acceptance'))
        if kind=='transport':
            audit.require('transport/no_resume','resume' not in reference and 'resume' not in actual)
            audit.equal('transport/modes',(overlaps['reference']['mode'],overlaps['actual']['mode']),('blocking','async'))
            expected_payload=deepcopy(rp)
            expected_payload['storage_policy']['transport']['mode']='async'
            audit.equal('transport/payload_except_mode',ap,expected_payload)
            audit.equal('transport/configuration_except_identity',
                {k:v for k,v in actual['configuration'].items() if k!='execution_identity'},
                {k:v for k,v in reference['configuration'].items() if k!='execution_identity'})
            for key in ('sources','arm','scale','declaration_sha256','resolved_sha256','startup_import','evaluation_policy'):
                audit.equal('transport/'+key,actual.get(key),reference.get(key))
            audit.equal('transport/same_terminal_update',actual['final_counters'],reference['final_counters'])
            audit.require('transport/observed_wall_overlap',bool(overlaps['actual']['worker_update_interval_intersections']))
        else:
            audit.require('resume/present','resume' in actual)
            audit.equal('resume/async_mode',overlaps['actual']['mode'],'async')
            for key in ('configuration','sources','source_fingerprint','arm','scale','declaration_sha256',
                        'resolved_sha256','startup_import','evaluation_policy'):
                audit.equal('resume/'+key,actual.get(key),reference.get(key))
            if resume_publication is None:raise ValueError('Resume audit requires exact pinned source publication')
            manifest,_,identity=publication_metadata(resume_publication)
            audit.equal('resume/published_identity',identity,actual['configuration']['execution_identity'])
            audit.equal('resume/published_configuration',manifest['metadata']['configuration'],actual['configuration'])
            completed=actual['resume']['completed_update']
            audit.equal('resume/published_counters',manifest['counters'],expected_counts(ap,completed))
            audit.equal('resume/published_cursors',manifest['rank_cursors'],[expected_cursor(ap,completed,rank) for rank in range(2)])
            audit.equal('resume/exact_manifest',actual['resume']['manifest_sha256'],resume_publication['manifest_sha256'])
        evaluation_parity(audit,reference,actual,kind)
        previous.training_parity(audit,reference,actual)
        if storage_evidence is not None:storage_check(audit,actual,storage_evidence)
    except (KeyError,TypeError,ValueError,IndexError,OSError) as error:
        result=audit.result(f'{type(error).__name__}: {error}')
    else:result=audit.result()
    result.update(schema=SCHEMA,comparison_kind=kind,storage_evidence_checked=storage_evidence is not None,
        transport_files_checked={'reference':reference_transport_evidence is not None,'actual':actual_transport_evidence is not None},
        overlap_evidence=overlaps,
        scope='JSON-only exact tiny transport/evaluation/restart and declared storage evidence; no tensor reload, new cloud verification or precision clearance')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('transport','resume'),required=True)
    for name in ('reference','actual'):
        parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
        parser.add_argument('--'+name+'-source-root',type=Path)
    parser.add_argument('--storage-evidence-root',type=Path,required=True)
    parser.add_argument('--resume-publication',type=Path)
    parser.add_argument('--resume-publication-sha256')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if (args.resume_publication is None)!=(args.resume_publication_sha256 is None):parser.error('Pair publication and SHA')
    if (args.kind=='resume')!=(args.resume_publication is not None):parser.error('Publication required for resume only')
    reports={};inputs={}
    for name in ('reference','actual'):
        reports[name],inputs[name]=bounded_json(getattr(args,name),getattr(args,name+'_sha256'))
    publication=None
    if args.resume_publication is not None:
        publication,inputs['resume_publication']=bounded_json(args.resume_publication,args.resume_publication_sha256)
    evidence,storage_pins=storage_audit.load_storage_evidence(args.storage_evidence_root)
    transport_evidence={};transport_pins={}
    for name in ('reference','actual'):
        transport_evidence[name],transport_pins[name]=load_transport_evidence(getattr(args,name).parent,reports[name])
    result=compare(reports['reference'],reports['actual'],kind=args.kind,reference_source_root=args.reference_source_root,
        actual_source_root=args.actual_source_root,storage_evidence=evidence,resume_publication=publication,
        reference_transport_evidence=transport_evidence['reference'],actual_transport_evidence=transport_evidence['actual'])
    for pin in [*inputs.values(),*storage_pins.values(),*(pin for group in transport_pins.values() for pin in group.values())]:
        if file_sha(pin['path'])!=pin['sha256']:raise ValueError('Audit input changed')
    result['inputs']=inputs;result['storage_inputs']=storage_pins;result['transport_inputs']=transport_pins
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in ('scripts/olmo_pilot_async_audit.py','tests/test_pilot_async_audit.py',
        'scripts/olmo_pilot_execution_audit_v2.py','tests/test_pilot_execution_audit_v2.py',
        'scripts/olmo_pilot_execution_audit.py','tests/test_pilot_execution_audit.py',
        'scripts/olmo_campaign_ssd_audit.py','scripts/olmo_campaign_execution_audit.py',
        'scripts/olmo_campaign_execution_restore.py','scripts/olmo_campaign_recovery_bundle.py',
        'scripts/olmo_pilot_execution_restore.py','tests/test_pilot_execution_restore.py',
        'scripts/olmo_campaign_ssd_storage.py'):
        destination=args.output_dir/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination);sources[name]=file_sha(destination)
    result['sources']=result['audit_sources']=sources
    (args.output_dir/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'failures':result['failures']}))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
