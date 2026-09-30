"""JSON-only oracles for inherited-state objective forks and child restarts."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import olmo_kl_continuation_audit as audit
from test_pilot_async_audit import add_transport
from test_pilot_execution_audit import pair as ordered_pair
from test_campaign_eval_audit import reidentify, resumed


PARENT_PIN='e'*64
EVAL_SCHEMA='olmo-kl-continuation-evaluation-v1'


def pair():
    _,parent=ordered_pair()
    config=parent['configuration'];payload=config['execution_identity']['payload']
    config['model']={'lambda_kl':1.,'lambda_latent':1.,'document_policy':'continuous-stream-v1'}
    config['parameters']['weights']={'ce':1.,'latent':1.,'kl':1.}
    config['recipe'].update(auxiliary={'latent':1.,'kl':1.},optimizer_state='fresh',jitter_seed=123)
    payload['schedule']={'lr_at_completed_boundaries':[.001]*4,'valid_token_prefix':[0,80,160,240]}
    reidentify(parent);add_transport(parent,'async')
    source=parent['local_checkpoints'][1]['receipt']['manifest_sha256']

    def fork(weight):
        result=deepcopy(parent)
        result['original_configuration']=deepcopy(parent['configuration'])
        result['resume']={'completed_update':1,'reference_report_required':True,'manifest_sha256':source,
            'parent_files_required':True,'parent_boundary_compared':True}
        result['loop']['start_update']=1
        result['origin_boundary_by_rank']=audit.legacy.boundary_at(parent,1)
        result['origin_clocks']={'adam_parameters':1,'input_tokens':80,'optimizer_updates':1,'scheduler_epoch':1}
        result['updates'].pop('1');result['observations'].pop('1')
        result['adam_resident_before_ddp']=True
        config=result['configuration'];payload=config['execution_identity']['payload']
        recipe=deepcopy(config['recipe']);recipe['auxiliary']['kl']=weight
        recipe['optimizer_state']='inherited_exact_parent_checkpoint'
        recipe['objective_transition']={
            'schema':'olmo-kl-objective-transition-v1','parent_manifest_sha256':source,
            'parent_kl_weight':1.,'kl_weight':weight,'changed_field':'lambda_kl',
            'same_weight_control':weight==1.,
            'retained_state':['model','optimizer','scheduler','counters','data_cursor','rng'],
            'resume':'exact_branch_identity_without_reapplying_parent_transition'}
        branch={'schema':audit.BRANCH_SCHEMA,'kl_weight':weight,'parent_manifest_sha256':source,
            'parent_identity_sha256':parent['configuration']['execution_identity']['sha256'],
            'parent_report_sha256':PARENT_PIN,'parent_update':1,'review_stop':3,'recipe_as_declared':recipe}
        result.update(branch=branch,parent_manifest_sha256=source,parent_report_sha256=PARENT_PIN)
        config.update(schema=audit.CONFIG_SCHEMA,objective_branch=branch,recipe=recipe)
        config['model']['lambda_kl']=weight;config['parameters']['weights']['kl']=weight
        payload.update(scope='Explicit saved-Adam KL objective fork; not same-configuration resume',
            objective_branch=branch,recipe=recipe,model_contract=config['parameters'])
        old=deepcopy(parent['configuration']['model']);new=deepcopy(config['model'])
        change={'schema':'olmo-kl-objective-transition-v1',
            'configuration_before':{'model':old,'predictor':deepcopy(old)},
            'configuration_after':{'model':new,'predictor':deepcopy(new)},
            'weights_before':{'ce':1.,'latent':1.,'kl':1.},
            'weights_after':{'ce':1.,'latent':1.,'kl':weight},
            'checks':dict.fromkeys(('parameters_unchanged','buffers_unchanged','modules_unchanged',
                'only_lambda_kl_changed','actual_objective_matches'),True),
            'scope':'Config-only ownership/version checks; caller audits complete restored state',
            'before_graph_construction_required':True}
        result['objective_transition']={'parent_manifest_sha256':source,
            'before_boundary_by_rank':deepcopy(result['origin_boundary_by_rank']),
            'after_boundary_by_rank':deepcopy(result['origin_boundary_by_rank']),
            'configuration_change_by_rank':[change,deepcopy(change)]}
        result['evaluation_policy']['objective']={'schema':EVAL_SCHEMA,
            'weights':deepcopy(config['parameters']['weights']),
            'raw_terms':'Existing unweighted per-pass sums divided by separate counts',
            'weighted_total':'Branch-specific objective; not a cross-branch quality metric'}
        for event in result['evaluations']:
            event['schema']=EVAL_SCHEMA
            for panel in event['panels'].values():
                panel['result'].update(schema=EVAL_SCHEMA,weights=deepcopy(config['parameters']['weights']))
                panel['result']['objective']=math.fsum(config['parameters']['weights'][term]*value
                    for term,value in panel['result']['aggregate']['means'].items())
                for rank in panel['by_rank']:
                    for row in rank['rows']:row['weights']=deepcopy(config['parameters']['weights'])
        for rows in result['updates'].values():
            for row in rows:
                row['metrics']['objective']=math.fsum(config['parameters']['weights'][term]*value
                    for term,value in row['loss_means'].items())
        result['storage']['checkpoint_root']+=f'/KL{weight}'
        result['storage']['evidence_dir']+=f'/KL{weight}'
        reidentify(result);add_transport(result,'async');result['schema']=audit.REPORT_SCHEMA
        return result
    return parent,fork(1.),fork(.1)


def run(values,**kwargs):
    return audit.audit_pair(*values,expected_stop=3,parent_report_sha256=PARENT_PIN,**kwargs)


def test_paired_objective_fork_preserves_parent_Adam_cursor_RNG_and_raw_terms():
    values=pair();before=deepcopy(values);result=run(values)
    assert result['passed'],result['failures']
    assert values==before
    assert values[1]['updates']['2'][0]['metrics']['objective'] != values[2]['updates']['2'][0]['metrics']['objective']
    assert values[1]['evaluations'][0]['panels']['dev-main']['result']['objective'] != values[2]['evaluations'][0]['panels']['dev-main']['result']['objective']


@pytest.mark.parametrize('mutation',[
    'fresh_Adam','reset_scheduler','reset_cursor','reset_rng','missing_state','untruthful_transition',
    'transition_extra_check','transition_predictor_change','transition_predictor_KL','missing_transition',
    'recipe_jitter','fresh_recipe','unknown_recipe','unknown_payload','unknown_branch','parent_pin',
    'parent_report_pin','parent_identity','weight','model_weight','predictor_receipt_weight',
    'first_raw_loss','input_noise','mask_counts','opposite_rank_counts','learning_rate',
    'eval_mean','eval_weight','eval_mask','eval_schedule','eval_membership','eval_preservation',
    'control_gradient','control_final','child_flag','unpublished_final','source_changed'])
def test_rejects_corrupt_lineage_data_masks_schedule_evaluation_or_inheritance(mutation):
    values=pair();parent,control,reduced=values
    payload=reduced['configuration']['execution_identity']['payload']
    origin=reduced['origin_boundary_by_rank'];receipt=reduced['objective_transition']
    panel=reduced['evaluations'][0]['panels']['dev-main']
    if mutation=='fresh_Adam':
        for rank in origin:rank['state']['optimizer']['state']={}
    elif mutation=='reset_scheduler':
        for rank in origin:rank['state']['scheduler']['last_epoch']=0
    elif mutation=='reset_cursor':origin[0]['cursor']['cursor']['next_chunk']=0
    elif mutation=='reset_rng':origin[0]['rng']['python']=[999]
    elif mutation=='missing_state':origin[0]['state'].pop('optimizer')
    elif mutation=='untruthful_transition':receipt['before_boundary_by_rank'][0]['rng']['python']=[999]
    elif mutation=='transition_extra_check':receipt['configuration_change_by_rank'][0]['checks']['unchecked']=True
    elif mutation=='transition_predictor_change':receipt['configuration_change_by_rank'][0]['configuration_after']['predictor']['lambda_latent']=.5
    elif mutation=='transition_predictor_KL':receipt['configuration_change_by_rank'][0]['configuration_before']['predictor']['lambda_kl']=.1
    elif mutation=='missing_transition':reduced.pop('objective_transition')
    elif mutation=='recipe_jitter':payload['recipe']['jitter_seed']=124
    elif mutation=='fresh_recipe':payload['recipe']['optimizer_state']='fresh'
    elif mutation=='unknown_recipe':payload['recipe']['hidden_update']=True
    elif mutation=='unknown_payload':payload['hidden_update']=True
    elif mutation=='unknown_branch':reduced['branch']['hidden_update']=True
    elif mutation=='parent_pin':reduced['branch']['parent_manifest_sha256']='0'*64
    elif mutation=='parent_report_pin':reduced['branch']['parent_report_sha256']='0'*64
    elif mutation=='parent_identity':reduced['branch']['parent_identity_sha256']='0'*64
    elif mutation=='weight':reduced['branch']['kl_weight']=.2
    elif mutation=='model_weight':reduced['configuration']['model']['lambda_kl']=1.
    elif mutation=='predictor_receipt_weight':receipt['configuration_change_by_rank'][0]['weights_after']['kl']=1.
    elif mutation=='first_raw_loss':
        for row in reduced['updates']['2']:
            row['metrics']['loss_sums']['ce']+=1.;row['loss_means']['ce']=row['metrics']['loss_sums']['ce']/75
    elif mutation=='input_noise':reduced['updates']['3'][0]['input']['batches']['sha256']='0'*64
    elif mutation=='mask_counts':reduced['observations']['3']['rank_data'][0]['kl_triples']-=1
    elif mutation=='opposite_rank_counts':
        reduced['observations']['3']['rank_data'][0]['kl_triples']+=1
        reduced['observations']['3']['rank_data'][1]['kl_triples']-=1
    elif mutation=='learning_rate':
        for row in reduced['updates']['2']:row['metrics']['lr_used']=[.002]
    elif mutation=='eval_mean':panel['result']['passes'][0]['means']['ce']+=1
    elif mutation=='eval_weight':panel['result']['weights']['kl']=1.
    elif mutation=='eval_mask':panel['by_rank'][0]['rows'][0]['counts']['kl']+=1
    elif mutation=='eval_schedule':reduced['evaluations'][0]['after_update']=3
    elif mutation=='eval_membership':panel['membership_sha256']='0'*64
    elif mutation=='eval_preservation':panel['by_rank'][0]['preservation']['checks']['rng_restored']=False
    elif mutation=='control_gradient':
        for row in control['updates']['3']:row['raw_gradients']['weight']['sha256']='0'*64
    elif mutation=='control_final':
        for row in control['final_boundary_by_rank']:row['state']['model']['weight']['sha256']='0'*64
    elif mutation=='child_flag':reduced['branch_resume']=True
    elif mutation=='unpublished_final':reduced['loop']['checkpoint_pending']=True
    elif mutation=='source_changed':reduced['sources']['scripts/example.py']='0'*64
    reidentify(reduced)
    result=run(values)
    assert not result['passed'],mutation


def restart_pair():
    _,_,reference=pair()
    source=deepcopy(next(r for r in reference['published_checkpoints'] if r['counters']['optimizer_updates']==2))
    actual=resumed(reference);actual['branch_resume']=True;actual.pop('objective_transition')
    actual['resume'].update(reference_report_required=True,parent_files_required=True,parent_boundary_compared=False)
    actual['resume']['manifest_sha256']=source['manifest_sha256']
    actual['storage']['checkpoint_root']+='/restart';actual['storage']['evidence_dir']+='/restart'
    reidentify(actual);add_transport(actual,'async');actual['schema']=audit.REPORT_SCHEMA
    return reference,actual,source


def run_restart(values):
    return audit.audit_restart(values[0],values[1],source_publication=values[2],expected_stop=3)


def test_same_branch_cloud_resume_matches_next_update_and_repeated_raw_evaluation():
    values=restart_pair();before=deepcopy(values);result=run_restart(values)
    assert result['passed'],result['failures']
    assert values==before


@pytest.mark.parametrize('mutation',['missing_publication','manifest','reapply_parent','objective',
    'input_noise','gradient','state','rng','eval','parent_lineage'])
def test_restart_rejects_wrong_branch_or_inexact_next_update(mutation):
    values=list(restart_pair());_,actual,_=values
    if mutation=='missing_publication':values[2]=None
    elif mutation=='manifest':actual['resume']['manifest_sha256']='0'*64
    elif mutation=='reapply_parent':actual['objective_transition']={}
    elif mutation=='objective':actual['branch']['kl_weight']=1.
    elif mutation=='input_noise':actual['updates']['3'][0]['input']['batches']['sha256']='0'*64
    elif mutation=='gradient':
        for row in actual['updates']['3']:row['raw_gradients']['weight']['sha256']='0'*64
    elif mutation=='state':actual['final_boundary_by_rank'][0]['state']['model']['weight']['sha256']='0'*64
    elif mutation=='rng':actual['origin_boundary_by_rank'][0]['rng']['python']=[999]
    elif mutation=='eval':actual['evaluations'][0]['panels']['dev-main']['result']['passes'][0]['means']['ce']+=1
    elif mutation=='parent_lineage':actual['branch']['parent_manifest_sha256']='0'*64
    reidentify(actual)
    result=run_restart(values)
    assert not result['passed'],mutation


def test_no_model_cuda_or_cloud_imports():
    subprocess.run([sys.executable,'-c','import sys;import scripts.olmo_kl_continuation_audit;assert "torch" not in sys.modules;assert "google.cloud.storage" not in sys.modules'],
        cwd=audit.ROOT,check=True,capture_output=True,text=True)


def test_cli_binds_actual_parent_report_bytes(tmp_path):
    values=list(pair());parent_path=tmp_path/'parent.json';parent_path.write_text(json.dumps(values[0]))
    parent_pin=audit.file_sha(parent_path)
    for report in values[1:]:
        report['branch']['parent_report_sha256']=parent_pin;report['parent_report_sha256']=parent_pin
        reidentify(report);add_transport(report,'async');report['schema']=audit.REPORT_SCHEMA
    command=[sys.executable,str(Path(audit.__file__)),'--kind','pair','--expected-stop','3','--output-dir',str(tmp_path/'audit')]
    for label,report in zip(('parent','control','reduced'),values):
        path=tmp_path/(label+'.json');path.write_text(json.dumps(report))
        command.extend(['--input',label,str(path),audit.file_sha(path)])
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    saved=json.loads((tmp_path/'audit/report.json').read_text())
    assert saved['passed'] and saved['inputs']['parent']['sha256']==parent_pin
