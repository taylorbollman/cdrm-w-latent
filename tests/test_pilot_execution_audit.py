"""Independent mutation oracles for new schemas, named panels and restart."""
from copy import deepcopy
import subprocess
import sys

import pytest
from scripts import olmo_pilot_execution_audit as audit
from test_campaign_ssd_audit import pair as old_pair, storage
from test_campaign_eval_audit import reidentify, resumed


def pair():
    _, actual = old_pair()
    actual['schema'] = audit.REPORT_SCHEMA
    actual['configuration']['schema'] = audit.CONFIG_SCHEMA
    identity=actual['configuration']['execution_identity']; identity['schema']=audit.IDENTITY_SCHEMA
    payload=identity['payload']; payload.pop('evaluation',None)
    old=actual['evaluation_policy']['plan']
    fixed=deepcopy(old['fixed_plan'])
    fixed['updates'][0]['membership_sha256']='2'*64
    declaration={'kind':'ordered_named_dev_panels_v1', 'panels':[{'name':'dev-main','target_valid_tokens':80}],
        'physical_batch_by_arm':{'NFR':2}, 'every_updates':2,'precision':'fp32','feedback_jitter':0.,
        'report_passes':'all_trained_passes','generation':'not_implemented'}
    panel={'index':'/dev','index_manifest_sha256':'f'*64,'panel_identity_sha256':'3'*64,
        'fixed_plan':fixed,'selection':'fixed_ordered_prefix_from_chunk_zero','target_valid_tokens':80,'available_panel_tokens':288}
    plan={'schema':'olmo-pilot-evaluation-control-v1','declaration':declaration,'execution':'not_run',
        'panels':{'dev-main':panel},'scheduled_updates':[2],
        'partition_by_arm':{'NFR':{'world_size':2,'physical_batch_per_rank':2}},
        'overlap_policy':'Report each panel separately; main/source overlap is not independent replication'}
    actual['evaluation_policy'].update(plan=plan,training_physical_batch_per_rank=2,evaluation_physical_batch_per_rank=2)
    entry=actual['evaluations'][0]
    actual['evaluations']=[{'schema':'olmo-pilot-evaluation-control-v1','after_update':2,'status':'completed',
        'training_boundary_exact_by_rank':[True,True],'total_seconds':1.,'panels':{'dev-main':{
            'result':entry['result'],'by_rank':entry['by_rank'],'index_manifest_sha256':'f'*64,
            'membership_sha256':'2'*64,'total_seconds':1.}}}]
    payload['declaration'].update(schema=audit.TINY_SCHEMA,evaluation=declaration)
    payload['resolved_contract_sha256']=audit.digest(payload['declaration']);reidentify(actual)
    reference=deepcopy(actual)
    deferred={'kind':'deferred','reason':'independent no-evaluation reference'}
    reference['evaluation_policy'].update(plan={'declaration':deferred,'execution':'not_run'},evaluation_physical_batch_per_rank=None)
    reference['evaluations']=[]
    rp=reference['configuration']['execution_identity']['payload']
    rp['declaration']['evaluation']=deferred;rp['resolved_contract_sha256']=audit.digest(rp['declaration']);reidentify(reference)
    return reference,actual


def test_insertion_and_storage_exact_without_schema_relabeling():
    reference,actual=pair(); evidence=storage(actual); before=deepcopy((reference,actual,evidence))
    result=audit.compare(reference,actual,kind='insertion',storage_evidence=evidence)
    assert result['passed'],result['failures']
    assert (reference,actual,evidence)==before
    assert actual['configuration']['execution_identity']['schema']==audit.IDENTITY_SCHEMA


@pytest.mark.parametrize('mutation',['model','gradient','rng','cursor','input','accounting','evaluation_count',
    'evaluation_mean','evaluation_preservation','evaluation_boundary','evaluation_index','evaluation_membership',
    'evaluation_named_panel','confirmation_panel','evaluation_batch','evaluation_schedule','evaluation_weight',
    'recipe','source','new_payload_field','source_authority','storage_generation','storage_identity','prune'])
def test_rejects_state_data_named_evaluation_or_storage_drift(mutation):
    reference,actual=pair();evidence=storage(actual)
    payload=actual['configuration']['execution_identity']['payload'];entry=actual['evaluations'][0]
    panel=entry['panels']['dev-main'];plan=actual['evaluation_policy']['plan']
    if mutation=='model':actual['updates']['3'][0]['boundary']['state']['model']['weight']['sha256']='0'*64
    elif mutation=='gradient':actual['updates']['3'][0]['raw_gradients']['weight']['sha256']='0'*64
    elif mutation=='rng':actual['final_boundary_by_rank'][0]['rng']['python']=[9]
    elif mutation=='cursor':actual['updates']['3'][0]['cursor']['cursor']['next_chunk']+=1
    elif mutation=='input':actual['updates']['3'][0]['input']['batches']['sha256']='0'*64
    elif mutation=='accounting':actual['observations']['3']['rank_data'][0]['valid_tokens']+=1
    elif mutation=='evaluation_count':panel['by_rank'][0]['rows'][0]['counts']['ce']+=1
    elif mutation=='evaluation_mean':panel['result']['passes'][0]['means']['ce']+=1
    elif mutation=='evaluation_preservation':panel['by_rank'][0]['preservation']['checks']['rng_restored']=False
    elif mutation=='evaluation_boundary':entry['training_boundary_exact_by_rank']=[True,False]
    elif mutation=='evaluation_index':panel['index_manifest_sha256']='0'*64
    elif mutation=='evaluation_membership':panel['membership_sha256']='0'*64
    elif mutation=='evaluation_named_panel':entry['panels']['other']=entry['panels'].pop('dev-main')
    elif mutation=='confirmation_panel':plan['declaration']['panels'][0]['name']='confirmation-main'
    elif mutation=='evaluation_batch':plan['declaration']['physical_batch_by_arm']['NFR']=4
    elif mutation=='evaluation_schedule':entry['after_update']=1
    elif mutation=='evaluation_weight':panel['result']['term_pass_coefficients']['ce'][0]=1.
    elif mutation=='recipe':payload['recipe']['fbt_passes']=3
    elif mutation=='source':actual['sources']['scripts/example.py']='0'*64
    elif mutation=='new_payload_field':payload['hidden']=True
    elif mutation=='source_authority':payload['data']['index']='other'
    elif mutation=='storage_generation':next(iter(evidence['receipts'].values()))['retention']['objects'][0]['generation']='latest'
    elif mutation=='storage_identity':evidence['journal']['ownership']['execution_identity_sha256']='0'*64
    elif mutation=='prune':evidence['journal']['prune_operations'][0]['update']=3
    payload['resolved_contract_sha256']=audit.digest(payload['declaration']);reidentify(actual)
    assert not audit.compare(reference,actual,kind='insertion',storage_evidence=evidence)['passed'],mutation


def test_resumed_next_update_and_repeated_named_evaluation_match():
    _,reference=pair();storage(reference);actual=resumed(reference)
    publication=deepcopy(reference['published_checkpoints'][2]);actual['resume']['manifest_sha256']=publication['manifest_sha256']
    actual['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/resume'
    actual['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/pilot-resume'
    evidence=storage(actual)
    result=audit.compare(reference,actual,kind='resume',resume_publication=publication,storage_evidence=evidence)
    assert result['passed'],result['failures']
    actual['evaluations'][0]['panels']['dev-main']['result']['objective']+=1
    assert not audit.compare(reference,actual,kind='resume',resume_publication=publication,storage_evidence=evidence)['passed']


def test_old_identity_and_unpinned_resume_are_rejected():
    _,reference=pair();actual=resumed(reference)
    assert not audit.compare(reference,actual,kind='resume')['passed']
    reference['configuration']['execution_identity']['schema']='olmo-campaign-execution-identity-v1'
    assert not audit.compare(reference,actual,kind='resume')['passed']


def test_import_is_json_only_without_torch_or_cloud():
    subprocess.run([sys.executable,'-c','import sys;import scripts.olmo_pilot_execution_audit;assert "torch" not in sys.modules;assert "google.cloud.storage" not in sys.modules'],
        cwd=audit.ROOT,check=True,capture_output=True,text=True)


def stopped_at_two(report):
    value=deepcopy(report)
    value['status']='stopped_at_boundary';value['plan_completed']=False
    value['loop']['completed_update']=2
    value['final_counters']=deepcopy(value['updates']['2'][0]['metrics']['counters'])
    value['final_boundary_by_rank']=[deepcopy(row['boundary']) for row in value['updates']['2']]
    value['final_clocks']={'adam_parameters':1,'input_tokens':160,'optimizer_updates':2,'scheduler_epoch':2}
    value['updates'].pop('3');value['observations'].pop('3')
    return value


def test_stop_boundary_and_terminal_resume_eval_without_new_update_or_graph():
    no_eval,full=pair();stopped=stopped_at_two(full);evidence=storage(stopped)
    result=audit.compare(no_eval,stopped,kind='insertion',storage_evidence=evidence)
    assert result['passed'],result['failures']
    publication=deepcopy(stopped['published_checkpoints'][-1])
    terminal=stopped_at_two(full)
    terminal['resume']={'completed_update':2,'reference_report_required':False,'manifest_sha256':publication['manifest_sha256']}
    terminal['loop']['start_update']=2
    terminal['origin_boundary_by_rank']=deepcopy(terminal['final_boundary_by_rank'])
    terminal['origin_clocks']=deepcopy(terminal['final_clocks'])
    terminal['updates']={};terminal['observations']={};terminal['graph_prepared']=False
    terminal['runner_by_rank']=[None,None];terminal.pop('preparation_boundary_exact')
    terminal['storage']['checkpoint_root']='/mnt/localssd/cdrm-checkpoints/example/terminal'
    terminal['storage']['evidence_dir']='/workspace/cdrm-w-latent/.runtime/pilot-terminal'
    evidence=storage(terminal)
    result=audit.compare(full,terminal,kind='resume',storage_evidence=evidence,resume_publication=publication)
    assert result['passed'],result['failures']
    assert not evidence['journal']['published']
    terminal['graph_prepared']=True
    assert not audit.compare(full,terminal,kind='resume',storage_evidence=evidence,resume_publication=publication)['passed']
