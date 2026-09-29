"""Fabricated JSON mutation oracles; no model imports, tensor loading or GPU."""
from copy import deepcopy
import json
import math

import pytest
from scripts import olmo_campaign_eval_audit as audit
from test_campaign_execution_audit import fixture


def reidentify(report):
    identity=report['configuration']['execution_identity'];identity['sha256']=audit.digest(identity['payload'])
    report['source_fingerprint']['sha256']=identity['sha256'];report['source_fingerprint']['execution_identity_sha256']=identity['sha256']


def make_pair():
    old=fixture();old['scale']='native';old['configuration']['schema']='olmo-campaign-execution-v1'
    old['startup_import']={}
    payload=old['configuration']['execution_identity']['payload']
    payload['recipe'].update(fbt_passes=4,rt_layers=[0,15],first_pass_policy='configured-rt-v1',document_policy='continuous-stream-v1')
    payload.update(resolved_contract_sha256='c'*64,evaluation={'declaration':{'kind':'deferred','reason':'test'},'execution':'not_run'},
        startup={'kind':'fresh'},runtime={'torch':'test'},determinism={'fixed':True},data={'index':'training'})
    reidentify(old)
    actual=deepcopy(old);actual['schema']=audit.NEW_REPORT_SCHEMA;actual['configuration']['schema']=audit.NEW_CONFIG_SCHEMA
    actual['sources'].update({name:'d'*64 for name in audit.NEW_SOURCES})
    ap=actual['configuration']['execution_identity']['payload'];ap['resolved_contract_sha256']='b'*64
    policy={'kind':'finite_pass_teacher_forced','index':'dev','index_manifest_sha256':'f'*64,'split':'dev',
        'target_valid_tokens':80,'every_updates':2,'precision':'fp32','feedback_jitter':0.,'report_passes':'all_trained_passes','generation':'not_implemented'}
    fixed=deepcopy(ap['plan']);fixed['updates']=fixed['updates'][:1]
    for cursor in (fixed['first_cursor'],fixed['final_cursor'],fixed['updates'][0]['next_cursor']):cursor['split']='dev'
    fixed['first_cursor']['manifest_sha256']=policy['index_manifest_sha256']
    fixed['updates'][0]['target_valid_tokens']=80
    plan={'declaration':policy,'execution':'not_run','fixed_plan':fixed,'scheduled_updates':[2]}
    ap['evaluation']=plan;actual['evaluation_policy']={'plan':plan,'resume':'repeat_scheduled_restored_boundary_once_per_segment',
        'failure':'earlier committed checkpoint remains authoritative; no save of unverified state'}
    coefficients={'ce':[.5,1/6,1/6,1/6],'latent':[.25]*4,'kl':[.25]*4}
    mode={'enabled':True,'num_passes':4,'beta':1.,'feedback_jitter':0.,'first_pass_policy':'configured-rt-v1',
        'document_policy':'continuous-stream-v1','rt_mode':{'selected_layers':[0,15],'alpha':1.}}
    by_rank=[]
    for rank in range(2):
        accounting=actual['observations']['1']['rank_data'][rank]
        counts={t:accounting[k] for t,k in zip(audit.TERMS,('ce_targets','latent_pairs','kl_triples'))}
        rows=[]
        for dummy in (False,True):
            c=dict.fromkeys(audit.TERMS,0) if dummy else counts
            passes=[{'index':i,'counts':c,'sums':{t:c[t]*(i+2) for t in audit.TERMS}} for i in range(4)]
            rows.append({'schema':'olmo-campaign-local-evaluation-v1','policy':'common_fp32_no_jitter_v1','mode':mode,
                'weights':dict.fromkeys(audit.TERMS,1.),'enabled':dict.fromkeys(audit.TERMS,True),
                'term_pass_coefficients':coefficients,'counts':c,'passes':passes,'input_tokens':0 if dummy else accounting['valid_tokens'],
                'aggregate_sums':{t:sum(coefficients[t][i]*passes[i]['sums'][t] for i in range(4)) for t in audit.TERMS}})
        checks=dict.fromkeys(('tensor_metadata_unchanged','module_ownership_unchanged','cache_generations_unchanged',
            'gradient_identity_unchanged','gradient_values_remained_zero','runtime_restored','modes_restored','rng_restored','autocast_cache_restored'),True)
        by_rank.append({'rows':rows,'preservation':{'checks':checks,'restored':True,'integrity_passed':True,
            'precision':'fp32_math_eager','feedback_jitter':0.},'cursor':fixed['first_cursor'],'accounting':accounting})
    local=[r for rank in by_rank for r in rank['rows']];counts={t:sum(r['counts'][t] for r in local) for t in audit.TERMS}
    def reduced(sums):return {'sums':sums,'counts':counts,'means':{t:sums[t]/counts[t] for t in audit.TERMS}}
    passes=[{'index':i,**reduced({t:math.fsum(r['passes'][i]['sums'][t] for r in local) for t in audit.TERMS})} for i in range(4)]
    aggregate=reduced({t:math.fsum(r['aggregate_sums'][t] for r in local) for t in audit.TERMS})
    result={'schema':'olmo-campaign-evaluation-control-v1','passes':passes,'aggregate':aggregate,
        'objective':math.fsum(aggregate['means'].values()),'input_tokens':80,'enabled':dict.fromkeys(audit.TERMS,True),
        'weights':dict.fromkeys(audit.TERMS,1.),'term_pass_coefficients':coefficients,
        'reconstructed_aggregate_sums':{t:math.fsum(coefficients[t][i]*passes[i]['sums'][t] for i in range(4)) for t in audit.TERMS},
        'policy':'common_fp32_no_jitter_v1'}
    actual['evaluations']=[{'after_update':2,'status':'completed','training_boundary_exact_by_rank':[True,True],
        'by_rank':by_rank,'result':result}]
    reidentify(actual)
    return old,actual


def test_cross_version_insertion_preserves_actual_schemas_and_all_evidence():
    old,new=make_pair();before=deepcopy((old,new))
    result=audit.compare(old,new,kind='insertion')
    assert result['passed'],result['failures']
    assert (old,new)==before
    assert old['schema']!=new['schema']


@pytest.mark.parametrize('mutation',['backbone','rng','raw_gradient','input','accounting','lr','recipe','startup',
    'old_source','foreign_source','unknown_row','eval_mode','eval_count','eval_mean','eval_pass','eval_preservation','eval_boundary','eval_schedule',
    'dev_origin','dev_budget','opposite_rank_tokens','opposite_rank_counts'])
def test_rejects_training_or_evaluation_drift(mutation):
    old,new=make_pair();ap=new['configuration']['execution_identity']['payload']
    if mutation=='backbone':new['updates']['3'][0]['boundary']['state']['model']['weight']['sha256']='0'*64
    elif mutation=='rng':new['final_boundary_by_rank'][0]['rng']['python']=[999]
    elif mutation=='raw_gradient':new['updates']['3'][0]['raw_gradients']['weight']['sha256']='0'*64
    elif mutation=='input':new['updates']['3'][0]['input']['batches']['sha256']='0'*64
    elif mutation=='accounting':new['observations']['3']['rank_data'][0]['physical_rows']+=1
    elif mutation=='lr':new['updates']['3'][0]['metrics']['lr_used']=[.002]
    elif mutation=='recipe':ap['recipe']['fbt_passes']=3
    elif mutation=='startup':ap['startup']['kind']='other'
    elif mutation=='old_source':new['sources']['scripts/example.py']='0'*64
    elif mutation=='foreign_source':new['sources']['scripts/unapproved.py']='0'*64
    elif mutation=='unknown_row':new['updates']['3'][0]['new_scientific_field']=1
    elif mutation=='eval_mode':new['evaluations'][0]['by_rank'][0]['rows'][0]['mode']['feedback_jitter']=.02
    elif mutation=='eval_count':new['evaluations'][0]['by_rank'][0]['rows'][0]['counts']['ce']+=1
    elif mutation=='eval_mean':new['evaluations'][0]['result']['passes'][0]['means']['ce']+=1
    elif mutation=='eval_pass':new['evaluations'][0]['result']['passes'].pop()
    elif mutation=='eval_preservation':new['evaluations'][0]['by_rank'][0]['preservation']['checks']['rng_restored']=False
    elif mutation=='eval_boundary':new['evaluations'][0]['training_boundary_exact_by_rank']=[True,False]
    elif mutation=='eval_schedule':new['evaluations'][0]['after_update']=1
    elif mutation=='dev_origin':new['evaluation_policy']['plan']['fixed_plan']['first_cursor']['split']='train'
    elif mutation=='dev_budget':new['evaluation_policy']['plan']['fixed_plan']['updates'][0]['target_valid_tokens']=81
    elif mutation=='opposite_rank_tokens':
        new['evaluations'][0]['by_rank'][0]['rows'][0]['input_tokens']+=1
        new['evaluations'][0]['by_rank'][1]['rows'][0]['input_tokens']-=1
    elif mutation=='opposite_rank_counts':
        new['evaluations'][0]['by_rank'][0]['rows'][0]['counts']['ce']+=1
        new['evaluations'][0]['by_rank'][1]['rows'][0]['counts']['ce']-=1
    reidentify(new)
    assert not audit.compare(old,new,kind='insertion')['passed']


def resumed(reference):
    new=deepcopy(reference);new['resume']={'completed_update':2,'reference_report_required':False,'manifest_sha256':'a'*64}
    new['loop']['start_update']=2;new['origin_boundary_by_rank']=deepcopy(reference['updates']['2'][0:2])
    new['origin_boundary_by_rank']=[r['boundary'] for r in new['origin_boundary_by_rank']]
    new['origin_clocks']={'adam_parameters':1,'input_tokens':160,'optimizer_updates':2,'scheduler_epoch':2}
    new['updates']={'3':new['updates']['3']};new['observations']={'3':new['observations']['3']}
    return new


def test_same_identity_resume_repeats_boundary_eval_and_matches_next_update():
    _,reference=make_pair();actual=resumed(reference)
    result=audit.compare(reference,actual,kind='resume')
    assert result['passed'],result['failures']


def test_resume_rejects_even_allowed_insertion_lineage_changes():
    _,reference=make_pair();actual=resumed(reference)
    actual['configuration']['execution_identity']['payload']['resolved_contract_sha256']='0'*64;reidentify(actual)
    assert not audit.compare(reference,actual,kind='resume')['passed']


def test_cli_refuses_mutated_input_before_publication(tmp_path,monkeypatch):
    old,new=make_pair();paths={}
    for name,value in [('reference',old),('actual',new)]:
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));paths[name]=path
    original=audit.compare
    def corrupt(*args,**kwargs):
        result=original(*args,**kwargs);paths['actual'].write_text('{}');return result
    monkeypatch.setattr(audit,'compare',corrupt)
    argv=['--kind','insertion','--output-dir',str(tmp_path/'out')]
    for name,path in paths.items():argv+=['--'+name,str(path),'--'+name+'-sha256',audit.file_sha(path)]
    with pytest.raises(ValueError,match='changed during audit'):audit.main(argv)
    assert not (tmp_path/'out').exists()


def test_auditor_has_no_training_torch_or_cloud_import():
    assert 'torch' not in audit.__dict__
    assert 'cdrm' not in audit.__dict__
