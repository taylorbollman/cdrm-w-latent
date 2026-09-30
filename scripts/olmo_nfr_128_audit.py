"""Independent JSON-only audit of the exact reduced-NFR64-to128 continuation.

Reuse accepted KL training and async storage validators directly. The copied
named-evaluation validator adds only the declared update100 observation. New
scope checks reconstruct metadata deltas independently of execution helpers.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
from scripts.olmo_kl_continuation_audit import (training_check, panel_check, _raw_evaluation,
    expected_counts, Audit, digest, finite_json, file_sha, bounded_json, TINY_SCHEMA)
from scripts.olmo_pilot_async_audit import transport_check, load_transport_evidence

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='olmo-nfr-128-audit-v1'
AUDIT_SOURCES=('scripts/olmo_nfr_128_audit.py','tests/test_nfr_128_audit.py',
    'scripts/olmo_kl_continuation_audit.py','scripts/olmo_pilot_async_audit.py',
    'scripts/olmo_pilot_execution_audit.py','scripts/olmo_pilot_execution_audit_v2.py',
    'scripts/olmo_campaign_execution_audit.py','scripts/olmo_campaign_ssd_audit.py',
    'scripts/olmo_campaign_eval_audit.py','scripts/olmo_pilot_execution_restore.py',
    'scripts/olmo_campaign_execution_restore.py','scripts/olmo_campaign_ssd_storage.py')
INPUTS=('parent64','continuation','scope','activation','resolved','host_launch')

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
    scheduled = sorted(set(range(interval, len(payload['plan']['updates'])+1, interval)) | {100})
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


def metadata_check(audit,report,parent,scope,activation,resolved,inputs):
    eq=lambda key,actual,wanted:audit.equal('continuation/'+key,actual,wanted)
    req=lambda key,value:audit.require('continuation/'+key,value)
    configuration=report['configuration'];payload=configuration['execution_identity']['payload']
    parent_config=parent['configuration'];old_payload=parent_config['execution_identity']['payload']
    eq('scope_schema',scope['schema'],'olmo-nfr-128-scope-v1')
    eq('bounded_scope',(scope['arm'],scope['origin_update'],scope['stop_update'],scope['planned_updates'],scope['kl_weight']),('NFR',64,128,128,.1))
    eq('extra_eval',scope['extra_evaluation_updates'],[100])
    eq('named_checkpoints',scope['named_checkpoints'],[80,96,100,112,128])
    eq('scope_report',report['nfr128_scope'],{'schema':'olmo-nfr-128-execution-scope-v1',
        'declaration':scope,'sha256':inputs['scope']['sha256']})
    eq('activation',report['activation'],{'receipt':activation,'sha256':inputs['activation']['sha256']})
    eq('activation_schema',activation['schema'],'olmo-nfr-128-activation-v1')
    eq('activation_decision',activation['decision'],'continue_reduced_to128')
    eq('activation_scope',activation['continuation_sha256'],inputs['scope']['sha256'])
    eq('activation_parent',activation['parent64_manifest_sha256'],scope['parent64_checkpoint']['sha256'])
    eq('parent_report_pin',scope['parent64_report']['sha256'],inputs['parent64']['sha256'])
    eq('original_config',report['original_configuration'],parent['original_configuration'])
    eq('old64_config',report['parent64_configuration'],parent_config)
    eq('old64_fingerprint',report['parent64_fingerprint'],parent['source_fingerprint'])
    eq('old64_report_pin',report['parent64_report_sha256'],inputs['parent64']['sha256'])
    eq('parent_state64',parent['final_counters']['optimizer_updates'],64)
    eq('parent_cloud64',parent['last_verified_cloud_update'],64)
    eq('parent_status',parent['status'],'stopped_at_boundary')
    eq('parent_synced',parent['wandb']['status'],'synced')
    publications=[p for p in parent['published_checkpoints'] if p['manifest_sha256']==scope['parent64_checkpoint']['sha256']]
    req('unique_original64',len(publications)==1)
    eq('same_parent64_config',publications[0]['metadata']['configuration'],parent_config)
    boundary=next(r['boundary_by_rank'] for r in parent['local_checkpoints']
        if r['receipt']['manifest_sha256']==scope['parent64_checkpoint']['sha256'])
    eq('parent_recorded_final_boundary',boundary,parent['final_boundary_by_rank'])
    continuation={'schema':'olmo-nfr-128-metadata-continuation-v1','scope_sha256':inputs['scope']['sha256'],
        'parent_manifest_sha256':scope['parent64_checkpoint']['sha256'],
        'parent_report_sha256':inputs['parent64']['sha256'],
        'parent_identity_sha256':parent_config['execution_identity']['sha256'],
        'parent_update':64,'stop_update':128,'named_checkpoints':scope['named_checkpoints'],
        'extra_evaluation_updates':[100],'objective_change':False,
        'state_transition':'identity_only_after_strict_original64_load'}
    eq('metadata',report['continuation'],continuation)
    expected=deepcopy(parent_config)
    expected.update(schema='olmo-nfr-128-execution-v1',continuation=continuation)
    expected_payload=expected['execution_identity']['payload']
    expected_payload.update(scope='Exact saved reduced-KL NFR64 continuation; unchanged objective and128 token schedule',
        sources=report['sources'],continuation=continuation)
    expected_payload['evaluation']['scheduled_updates']=sorted(set(old_payload['evaluation']['scheduled_updates'])|{100})
    expected['execution_identity']['sha256']=digest(expected_payload)
    eq('only_metadata_changed',configuration,expected)
    eq('prepared_configuration',configuration,resolved['configuration'])
    eq('prepared_scope',resolved['scope'],scope)
    eq('prepared_sources',resolved['sources'],report['sources'])
    eq('old_branch_unchanged',report['branch'],parent['branch'])
    eq('KL',payload['model_contract']['weights'],{'ce':1.,'latent':1.,'kl':.1})
    eq('native_mode',payload['model_contract']['mode']['rt_mode'],{'selected_layers':[0,15],'alpha':1.})
    eq('K4',payload['model_contract']['mode']['num_passes'],4)
    eq('source_count',len(report['sources']),222)
    eq('parent_source_count',len(parent['sources']),215)
    for name,pin in parent['sources'].items():eq('frozen_source/'+name,report['sources'].get(name),pin)
    eq('resume64',report['resume']['completed_update'],64)
    eq('resume64_manifest',report['resume']['manifest_sha256'],scope['parent64_checkpoint']['sha256'])
    req('not_a_child_restart',not report.get('branch_resume',False))
    req('no_objective_fork','objective_transition' not in report)
    eq('populated_Adam',report['adam_resident_before_ddp'],True)
    eq('origin_clocks',report['origin_clocks'],{'adam_parameters':71,'input_tokens':33554432,'optimizer_updates':64,'scheduler_epoch':64})
    eq('exact_original64_boundary',report['origin_boundary_by_rank'],boundary)
    transition=report['continuation_transition']
    eq('transition_fields',set(transition),{'parent_manifest_sha256','before_boundary_by_rank','after_boundary_by_rank','configuration_change_by_rank'})
    eq('transition_parent',transition['parent_manifest_sha256'],scope['parent64_checkpoint']['sha256'])
    eq('transition_before',transition['before_boundary_by_rank'],boundary)
    eq('transition_after',transition['after_boundary_by_rank'],boundary)
    expected_receipt={'schema':'olmo-nfr-128-transition-check-v1','metadata_only':True,'objective_unchanged':True,
        'parent_identity_sha256':parent_config['execution_identity']['sha256'],
        'child_identity_sha256':configuration['execution_identity']['sha256']}
    eq('two_rank_transition',transition['configuration_change_by_rank'],[expected_receipt,expected_receipt])
    eq('same_128token_schedule',configuration['schedule'],parent_config['schedule'])
    eq('schedule_hash',configuration['schedule']['plan_sha256'],'993d922340322268d7d0092b3303e8650c9e96a6b33e3c14055353e91041ef1c')
    eq('same_LR_schedule',payload['schedule'],old_payload['schedule'])
    lr=payload['schedule']['lr_at_completed_boundaries']
    eq('129boundaries',len(lr),129)
    eq('warmup_end',lr[100],.0002)
    eq('unchanged_plateau',lr[100:],[.0002]*29)
    req('warmup_still_increases',lr[64]<lr[96]<lr[99]<lr[100])
    eq('token_prefix',payload['schedule']['valid_token_prefix'],[n*524288 for n in range(129)])
    for step,rows in report['updates'].items():
        for key,index in (('lr_used',int(step)-1),('lr_next',int(step))):
            eq('step'+step+'/'+key,rows[0]['metrics'][key],[lr[index]]*len(rows[0]['metrics'][key]))
    eq('stop128',report['final_counters']['optimizer_updates'],128)
    eq('terminal_sync',report['wandb']['status'],'synced')
    saved={r['optimizer_update'] for r in report['local_checkpoints']}
    published={r['counters']['optimizer_updates'] for r in report['published_checkpoints']}
    req('named_local_saves',set(scope['named_checkpoints'])<=saved)
    req('named_cloud_saves',set(scope['named_checkpoints'])<=published)
    eq('cloud128',report['last_verified_cloud_update'],128)
    old_dev=next(e for e in parent['evaluations'] if e['after_update']==64)
    new_dev=next(e for e in report['evaluations'] if e['after_update']==64)
    eq('origin_dev64_reproduced',_raw_evaluation(new_dev),_raw_evaluation(old_dev))


def worker_files_check(audit,report,files):
    submissions=report['async_submissions']
    audit.equal('worker_files/membership',set(files),{str(r['update']) for r in submissions})
    root=report['storage']['evidence_dir'].split('/.runtime/',1)[0]
    expected_sources={str(Path(root)/name):pin for name,pin in report['sources'].items()}
    scope=report['nfr128_scope']['declaration']
    prefix=scope['storage_prefix']+'/reduced/'+Path(report['storage']['evidence_dir']).name
    for i,submission in enumerate(submissions):
        label='worker_files/'+str(submission['update']);job=files[str(submission['update'])]['job']
        audit.equal(label+'/receipt',job['receipt'],report['local_checkpoints'][i]['receipt'])
        audit.equal(label+'/sources',job['source_pins'],expected_sources)
        audit.equal(label+'/prefix',job['storage_prefix'],prefix)
        audit.equal(label+'/timeout',job['worker_timeout_seconds'],480)
        audit.equal(label+'/result',files[str(submission['update'])]['result'],report['async_completions'][i])


def audit_report(report,parent,scope,activation,resolved,launch,inputs,*,source_root=None,transport_files=None):
    audit=Audit()
    try:
        training_check(audit,report,'training',source_root)
        metadata_check(audit,report,parent,scope,activation,resolved,inputs)
        evaluation_check(audit,report,'evaluation')
        audit.equal('evaluation/objective',report['evaluation_policy']['objective']['weights'],{'ce':1.,'latent':1.,'kl':.1})
        transport_check(audit,report,'transport')
        if transport_files is not None:worker_files_check(audit,report,transport_files)
        audit.equal('host/exit',launch['exit_code'],0)
        audit.equal('host/finished',launch['status'],'executor_finished')
        audit.equal('host/report_pin',launch['report_sha256'],inputs['continuation']['sha256'])
        audit.equal('host/final_counters',launch['final_counters'],report['final_counters'])
        audit.equal('host/activation',launch['activation_sha256'],inputs['activation']['sha256'])
        audit.equal('host/scope',launch['scope_sha256'],inputs['scope']['sha256'])
        audit.equal('host/sources',launch['sources'],report['sources'])
        result=audit.result()
    except (ValueError,KeyError,TypeError,IndexError,StopIteration) as exc:
        result=audit.result(exc)
    result.update(schema=SCHEMA,inputs=inputs,
        audit_sources={name:file_sha(ROOT/name) for name in AUDIT_SOURCES},
        scope='Exact pinned terminal JSON, preserved model/Adam/cursor/RNG boundary digests, same128data/LR schedule, '
            'raw dev64 reproduction, named96/100/128 verified publications; no tensor reload or new cloud readback')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in INPUTS:
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
        parser.add_argument('--'+name.replace('_','-')+'-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=False)
    loaded={};inputs={};snapshot=args.output_dir/'input-snapshot';snapshot.mkdir()
    for name in INPUTS:
        path=getattr(args,name);pin=getattr(args,name+'_sha256')
        loaded[name],actual=bounded_json(path,pin)
        inputs[name]=actual
        destination=snapshot/(name+'.json');shutil.copyfile(path,destination)
        if file_sha(destination)!=pin:raise ValueError('Input changed during audit snapshot')
    report=loaded['continuation'];root=args.continuation.parent
    files,pins=load_transport_evidence(root,report)
    result=audit_report(report,loaded['parent64'],loaded['scope'],loaded['activation'],loaded['resolved'],loaded['host_launch'],
        inputs,source_root=root/'source-snapshot',transport_files=files)
    result['transport_file_pins']=pins
    for name,pin in result['audit_sources'].items():
        target=args.output_dir/'audit-source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        if file_sha(target)!=pin:raise ValueError('Audit source changed during copy')
    (args.output_dir/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'failures':result['failures']}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
