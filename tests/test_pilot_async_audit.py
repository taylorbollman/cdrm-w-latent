"""Independent JSON-only transport, worker and exact-restart mutation oracles."""
from copy import deepcopy
import inspect
import math
from pathlib import Path
import subprocess
import sys

import pytest
from scripts import olmo_pilot_async_audit as audit
from test_pilot_execution_audit import pair as ordered_pair, stopped_at_two
from test_campaign_eval_audit import reidentify, resumed
from test_campaign_ssd_audit import encoded, pin, storage


def add_transport(report, mode):
    report['schema']=audit.REPORT_SCHEMA
    policy=report['configuration']['execution_identity']['payload']['storage_policy']
    policy['transport']={'schema':audit.TRANSPORT_SCHEMA,'mode':mode,'maximum_pending':1,
        'worker_timeout_seconds':480,'terminal_drain':True,'full_cloud_readback':True,'training_state_access':'none'}
    report['storage']['policy']=deepcopy(policy)
    reidentify(report)
    for event in report['evaluations']:
        for panel in event['panels'].values():
            for part in panel['by_rank']:
                part['preservation'].update(elapsed_seconds=.1,integrity_scope='complete',rope_scope='fixed',timing_scope='wall')
    evidence=storage(report)
    report['checkpoint_transport']={'maximum_pending':1,'worker_timeout_seconds':480,'mode':mode}
    local=report['local_checkpoints'];final=report['final_counters']['optimizer_updates']
    report['loop'].update(checkpoints_submitted=len(local),checkpoints_published=len(local),checkpoint_pending=False,
        checkpoint_wait_seconds=.25,last_saved_update=final,last_retained_update=final)
    report['last_verified_cloud_update']=final
    report['async_submissions']=[];report['async_completions']=[];transport_evidence={}
    for index,row in enumerate(local):
        number=row['optimizer_update'];begin=float(number*100+21)
        end=float(number*100+(115 if mode=='async' and number not in (0,final) else 30))
        job={'schema':audit.TRANSPORT_SCHEMA,'receipt':deepcopy(row['receipt']),
            'source_pins':{str(Path('/workspace/cdrm-w-latent')/name):sha for name,sha in report['sources'].items()},
            'storage_prefix':evidence['journal']['ownership']['storage_prefix'],'worker_timeout_seconds':480,
            'submitted_at_unix_seconds':begin}
        submission={'schema':audit.TRANSPORT_SCHEMA,'update':number,'job_path':str(Path(report['storage']['evidence_dir'])/
            'async-retention'/f'update-{number:06d}'/'job.json'),'job_sha256':pin(encoded(job))['sha256'],
            'state':'pending_not_cloud_durable','submitted_at_unix_seconds':begin}
        completion={'schema':audit.RESULT_SCHEMA,'status':'published','update':number,'job_sha256':submission['job_sha256'],
            'receipt':deepcopy(report['published_checkpoints'][index]),
            'storage_publication':deepcopy(report['storage_publications'][index]),
            'timing':{'local_validation_seconds':.1,'cloud_child_seconds':.2,
                'verified_publication_and_pruning_seconds':.3,'total_background_seconds':end-begin},
            'worker':{'pid':1234+number,'cuda_initialized':False,'distributed_initialized':False,
                'sdk_seconds':.2,'cuda_visible_devices':''},
            'submitted_at_unix_seconds':begin,'completed_at_unix_seconds':end}
        report['async_submissions'].append(submission);report['async_completions'].append(completion)
        transport_evidence[str(number)]={'job':job,'result':deepcopy(completion)}
    report['update_intervals_unix_seconds']={str(n):{'started':float(100*n+10),'finished':float(100*n+20)}
        for n in map(int,report.get('updates',{}))}
    report['update_wall_seconds_by_rank']={str(n):[10.,9.9] for n in map(int,report.get('updates',{}))}
    return evidence,transport_evidence


def pair():
    _,reference=ordered_pair()
    re,rt=add_transport(reference,'blocking')
    actual=deepcopy(reference)
    actual['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/async'
    actual['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/async'
    ae,at=add_transport(actual,'async')
    return reference,actual,re,ae,rt,at


def run(values,**kwargs):
    reference,actual,_,evidence,rt,at=values
    return audit.compare(reference,actual,kind='transport',storage_evidence=evidence,
        reference_transport_evidence=rt,actual_transport_evidence=at,**kwargs)


def test_exact_transport_parity_with_actual_pending_overlap_and_cpu_workers():
    values=pair();before=deepcopy(values);result=run(values)
    assert result['passed'],result['failures']
    assert values==before
    assert result['overlap_evidence']['reference']['worker_update_interval_intersections']==[]
    overlaps=result['overlap_evidence']['actual']['worker_update_interval_intersections']
    assert [(row['optimizer_update'],row['checkpoint_update'],row['overlap_seconds']) for row in overlaps]==[(2,1,5.),(3,2,5.)]
    assert result['transport_files_checked']=={'reference':True,'actual':True}


@pytest.mark.parametrize('mutation',['model','gradient','input','rng','accounting','recipe','source','hidden_math',
    'worker_cuda','worker_distributed','worker_visibility','worker_pid','submission_durable','job_pin','completion_pin',
    'job_receipt','job_source','job_prefix','job_timeout','worker_result','submission_count','completion_count',
    'loop_pending','loop_submitted','loop_published','last_saved','last_remote','remote_report','worker_timeout',
    'unbounded_queue','weaker_verification','state_access','overlapping_workers','no_training_overlap','bad_interval',
    'missing_interval','bad_wall','negative_time','nan_time','boolean_time','cloud_generation','prune_latest',
    'eval_preservation','eval_extra_preservation','eval_mean','eval_missing_elapsed'])
def test_corrupt_training_storage_transport_or_evaluation_is_rejected(mutation):
    values=pair();reference,actual,_,evidence,_,transport=values
    payload=actual['configuration']['execution_identity']['payload']
    done=actual['async_completions'][1];submitted=actual['async_submissions'][1]
    p=actual['evaluations'][0]['panels']['dev-main']['by_rank'][0]['preservation']
    if mutation=='model':actual['updates']['3'][0]['boundary']['state']['model']['weight']['sha256']='0'*64
    elif mutation=='gradient':actual['updates']['3'][0]['raw_gradients']['weight']['sha256']='0'*64
    elif mutation=='input':actual['updates']['3'][0]['input']['batches']['sha256']='0'*64
    elif mutation=='rng':actual['final_boundary_by_rank'][0]['rng']['python']=[999]
    elif mutation=='accounting':actual['observations']['3']['rank_data'][0]['valid_tokens']+=1
    elif mutation=='recipe':payload['recipe']['fbt_passes']=3
    elif mutation=='source':actual['sources']['scripts/example.py']='0'*64
    elif mutation=='hidden_math':payload['new_math']='unapproved'
    elif mutation=='worker_cuda':done['worker']['cuda_initialized']=True
    elif mutation=='worker_distributed':done['worker']['distributed_initialized']=True
    elif mutation=='worker_visibility':done['worker']['cuda_visible_devices']='0'
    elif mutation=='worker_pid':done['worker']['pid']=False
    elif mutation=='submission_durable':submitted['state']='published'
    elif mutation=='job_pin':submitted['job_sha256']='0'*64
    elif mutation=='completion_pin':done['job_sha256']='0'*64
    elif mutation=='job_receipt':transport['1']['job']['receipt']['state']['sha256']='0'*64
    elif mutation=='job_source':transport['1']['job']['source_pins']['/workspace/cdrm-w-latent/scripts/example.py']='0'*64
    elif mutation=='job_prefix':transport['1']['job']['storage_prefix']+='/other'
    elif mutation=='job_timeout':transport['1']['job']['worker_timeout_seconds']=600
    elif mutation=='worker_result':transport['1']['result']['status']='failed'
    elif mutation=='submission_count':actual['async_submissions'].pop()
    elif mutation=='completion_count':actual['async_completions'].pop()
    elif mutation=='loop_pending':actual['loop']['checkpoint_pending']=True
    elif mutation=='loop_submitted':actual['loop']['checkpoints_submitted']-=1
    elif mutation=='loop_published':actual['loop']['checkpoints_published']-=1
    elif mutation=='last_saved':actual['loop']['last_saved_update']-=1
    elif mutation=='last_remote':actual['loop']['last_retained_update']-=1
    elif mutation=='remote_report':actual['last_verified_cloud_update']-=1
    elif mutation=='worker_timeout':payload['storage_policy']['transport']['worker_timeout_seconds']=600
    elif mutation=='unbounded_queue':payload['storage_policy']['transport']['maximum_pending']=5
    elif mutation=='weaker_verification':payload['storage_policy']['transport']['full_cloud_readback']=False
    elif mutation=='state_access':payload['storage_policy']['transport']['training_state_access']='live'
    elif mutation=='overlapping_workers':done['completed_at_unix_seconds']=250.
    elif mutation=='no_training_overlap':
        for row in actual['async_completions']:row['completed_at_unix_seconds']=row['submitted_at_unix_seconds']+1.
    elif mutation=='bad_interval':actual['update_intervals_unix_seconds']['2']['finished']=100.
    elif mutation=='missing_interval':actual['update_intervals_unix_seconds'].pop('3')
    elif mutation=='bad_wall':actual['update_wall_seconds_by_rank']['2']=[10.]
    elif mutation=='negative_time':done['timing']['cloud_child_seconds']=-1.
    elif mutation=='nan_time':done['timing']['cloud_child_seconds']=math.nan
    elif mutation=='boolean_time':done['timing']['cloud_child_seconds']=True
    elif mutation=='cloud_generation':next(iter(evidence['receipts'].values()))['retention']['objects'][0]['generation']='latest'
    elif mutation=='prune_latest':evidence['journal']['prune_operations'][0]['update']=3
    elif mutation=='eval_preservation':p['checks']['rng_restored']=False
    elif mutation=='eval_extra_preservation':p['unapproved']=True
    elif mutation=='eval_mean':actual['evaluations'][0]['panels']['dev-main']['result']['objective']+=1
    elif mutation=='eval_missing_elapsed':p.pop('elapsed_seconds')
    reidentify(actual)
    result=run(values)
    assert not result['passed'],mutation


def test_elapsed_timings_may_differ_without_changing_meaningful_evaluation():
    values=pair();values[1]['evaluations'][0]['panels']['dev-main']['by_rank'][0]['preservation']['elapsed_seconds']=555.
    result=run(values);assert result['passed'],result['failures']


def resume_pair():
    _,reference=ordered_pair();re,rt=add_transport(reference,'async')
    publication=deepcopy(reference['published_checkpoints'][2])
    actual=resumed(reference);actual['resume']['manifest_sha256']=publication['manifest_sha256']
    actual['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/resumed'
    actual['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/resumed'
    ae,at=add_transport(actual,'async')
    return reference,actual,publication,ae,rt,at


def run_resume(values):
    reference,actual,publication,evidence,rt,at=values
    return audit.compare(reference,actual,kind='resume',resume_publication=publication,
        storage_evidence=evidence,reference_transport_evidence=rt,actual_transport_evidence=at)


def test_pinned_same_lineage_resume_matches_next_update_and_repeated_evaluation():
    values=resume_pair();result=run_resume(values)
    assert result['passed'],result['failures']


@pytest.mark.parametrize('mutation',['missing_publication','different_manifest','wrong_published_state','different_mode',
    'different_identity','replayed_update','terminal_boundary'])
def test_resume_rejects_unpinned_or_changed_lineage(mutation):
    values=list(resume_pair());_,actual,publication,_,_,_=values
    if mutation=='missing_publication':values[2]=None
    elif mutation=='different_manifest':actual['resume']['manifest_sha256']='0'*64
    elif mutation=='wrong_published_state':publication['counters']['input_tokens']+=1
    elif mutation=='different_mode':actual['configuration']['execution_identity']['payload']['storage_policy']['transport']['mode']='blocking'
    elif mutation=='different_identity':actual['configuration']['execution_identity']['payload']['new_math']=True
    elif mutation=='replayed_update':actual['updates']['2']=deepcopy(values[0]['updates']['2'])
    elif mutation=='terminal_boundary':actual['final_boundary_by_rank'][0]['rng']['python']=[88]
    reidentify(actual)
    assert not run_resume(values)['passed']


def test_terminal_resume_performs_no_updates_capture_or_new_publications():
    _,reference=ordered_pair();reference=stopped_at_two(reference);_,rt=add_transport(reference,'async')
    publication=deepcopy(reference['published_checkpoints'][-1]);actual=deepcopy(reference)
    actual['resume']={'completed_update':2,'reference_report_required':False,'manifest_sha256':publication['manifest_sha256']}
    actual['loop']['start_update']=2;actual['origin_boundary_by_rank']=deepcopy(actual['final_boundary_by_rank'])
    actual['origin_clocks']=deepcopy(actual['final_clocks']);actual.pop('updates');actual.pop('observations')
    actual['graph_prepared']=False;actual['runner_by_rank']=[None,None];actual.pop('preparation_boundary_exact')
    actual['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/terminal'
    actual['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/terminal'
    ae,at=add_transport(actual,'async')
    result=run_resume((reference,actual,publication,ae,rt,at))
    assert result['passed'],result['failures']
    assert not actual['async_submissions'] and not ae['receipts']


def test_transport_reader_binds_saved_job_bytes_and_rejects_changes(tmp_path):
    _,actual,_,_,_,transport=pair()
    for number,item in transport.items():
        directory=tmp_path/'async-retention'/f'update-{int(number):06d}';directory.mkdir(parents=True)
        for name,value in item.items():(directory/(name+'.json')).write_bytes(encoded(value))
    actual_before=deepcopy(actual)
    loaded,pins=audit.load_transport_evidence(tmp_path,actual)
    assert loaded==transport and actual==actual_before and len(pins)==8
    (tmp_path/'async-retention/update-000001/job.json').write_text('{}')
    with pytest.raises(ValueError):audit.load_transport_evidence(tmp_path,actual)


def test_training_validator_is_literal_previous_validator_under_new_schema():
    assert inspect.getsource(audit.training_check)==inspect.getsource(audit.original.training_check)


def test_no_model_cuda_or_cloud_imports():
    subprocess.run([sys.executable,'-c','import sys;import scripts.olmo_pilot_async_audit;assert "torch" not in sys.modules;assert "google.cloud.storage" not in sys.modules'],
        cwd=audit.ROOT,check=True,capture_output=True,text=True)
