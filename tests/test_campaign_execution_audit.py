"""Literal fabricated JSON evidence; no training imports, CUDA or checkpoints."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import olmo_campaign_execution_audit as audit


def tensor(label, shape=None):
    return {'shape': [1] if shape is None else shape, 'dtype':'torch.float32',
            'sha256':hashlib.sha256(label.encode()).hexdigest()}


def fixture(*, mode='acceptance', start=0, final=3, arm='NFR'):
    def cursor(step):
        return {'manifest_sha256':'d'*64,'next_chunk':step*5,'next_update':step,'split':'train'}
    plan = {'first_cursor':cursor(0),'final_cursor':cursor(3),'updates':[]}
    for step in range(1,4):
        plan['updates'].append({'counts':{'valid_tokens':80,'packed_rows':5,'ce_targets':75,'latent_pairs':70,'kl_triples':65},
            'allocation_by_arm':{arm:[{'rank':rank,'microbatches':2,'valid_tokens':48 if rank==0 else 32,
                'packed_rows':3 if rank==0 else 2,'physical_rows':4,'padding_tokens':16 if rank==0 else 32,
                'dummy_rows':1 if rank==0 else 2} for rank in range(2)]},
            'next_cursor':cursor(step)})
    sources = {'scripts/example.py':hashlib.sha256(b'example\n').hexdigest()}
    ownership = {'parameter_layout':[{'name':'weight','aliases':['weight'],'shape':[1],'dtype':'torch.float32','requires_grad':True}]}
    recipe = {'arm':arm}
    payload = {'arm':arm,'partition':{'world_size':2,'physical_batch_per_rank':2},'plan':plan,
        'cursor_schema':'olmo-campaign-execution-cursor-v1','sources':sources,'recipe':recipe,
        'model_contract':ownership,'execution':{'precision':'fp32'}}
    identity = {'schema':'olmo-campaign-execution-identity-v1','payload':payload,'sha256':audit.digest(payload)}
    def counts(step):
        return {'optimizer_updates':step,'input_tokens':80*step,'documents':5*step,'microbatches':4*step,
                'ce_positions':75*step,'latent_pairs':70*step if 'N' in arm else 0,'kl_triples':65*step if 'N' in arm else 0}
    def boundary(step,rank):
        return {'state':{'model':{'weight':tensor(f'w{step}')},'optimizer':{'param_groups':[{'params':[0]}],
                    'state':{} if step==0 else {'0':{'step':tensor(str(step),[]),'exp_avg':tensor('m'+str(step))}}},
                'scheduler':{'last_epoch':step},'counters':counts(step)},
            'cursor':{'schema':payload['cursor_schema'],'cursor':cursor(step),'rank':rank,'world_size':2,'physical_batch_per_rank':2},
            'rng':{'python':[rank,step],'numpy':[rank,step],'torch_cpu':tensor('cpu'+str(rank)),
                   'torch_cuda':tensor('gpu'+str(rank)),'generators':{'data':tensor('data')},
                   'generator_devices':{'data':'cpu'},'device':'cuda:'+str(rank)}}
    report = {'schema':audit.REPORT_SCHEMA,'status':'completed_plan' if final==3 else 'stopped_at_boundary',
        'configuration':{'execution_identity':identity,'recipe':recipe,'parameters':ownership,
            'source_checkpoint':{'sha256':'c'*64},'training':{'precision':'fp32','max_grad_norm':1.0}},
        'sources':sources,'arm':arm,'scale':'tiny','declaration_sha256':'e'*64,'resolved_sha256':None,
        'source_fingerprint':{'sha256':identity['sha256'],'scope':'execution_identity',
            'execution_identity_sha256':identity['sha256'],'sources':sources,'source_checkpoint':{'sha256':'c'*64}},
        'final_counters':counts(final),'loop':{'start_update':start,'completed_update':final},
        'plan_completed':final==3,'graph_prepared':final>start,'runner_by_rank':[{'captured':True}]*2 if final>start else [None]*2,
        'origin_clocks':{'adam_parameters':1 if start else 0,'input_tokens':80*start,'optimizer_updates':start,'scheduler_epoch':start},
        'final_clocks':{'adam_parameters':1 if final else 0,'input_tokens':80*final,'optimizer_updates':final,'scheduler_epoch':final},
        'origin_boundary_by_rank':[boundary(start,rank) for rank in range(2)],
        'final_boundary_by_rank':[boundary(final,rank) for rank in range(2)],'updates':{},'observations':{},'observation_mode':mode}
    if final>start:report['preparation_boundary_exact']=[True,True]
    if start:report['resume']={'completed_update':start,'reference_report_required':False,'manifest_sha256':'f'*64}
    for step in range(start+1,final+1):
        report['observations'][str(step)]={'rank_data':[{'valid_tokens':48,'packed_rows':3,'ce_targets':45,
            'latent_pairs':42,'kl_triples':39,'physical_rows':4,'padding_tokens':16,'empty_rows':1},
            {'valid_tokens':32,'packed_rows':2,'ce_targets':30,'latent_pairs':28,'kl_triples':26,
            'physical_rows':4,'padding_tokens':32,'empty_rows':2}]}
        rows = []
        for rank in range(2):
            metric = {'counters':counts(step),'counts':{'ce':75,'latent':70 if 'N' in arm else 0,'kl':65 if 'N' in arm else 0},
                'loss_sums':{'ce':150.0+step,'latent':35.0 if 'N' in arm else 0.,'kl':65.0 if 'N' in arm else 0.},
                'objective':3.5+step/75,'gradient_norm_before_clip':2.,'input_tokens':80,'documents':5,'microbatches':4,
                'lr_used':[.001],'lr_next':[.001]}
            row = {'schema':audit.OBSERVATION_SCHEMA,'mode':mode,'metrics':metric,'cursor':boundary(step,rank)['cursor'],
                'loss_means':{term:metric['loss_sums'][term]/metric['counts'][term] if metric['counts'][term] else None for term in audit.TERMS},
                'clipping':{'configured_limit':1.,'norm_exceeds_limit':True,'coefficient_estimate':1./(2.+1e-6),'scope':'unchanged runner'},
                'observation_seconds':{'after_backward':.001,'after_update':.002}}
            if mode=='acceptance':
                row.update(input={'batches':tensor('input'+str(step)+str(rank))},raw_gradients={'weight':tensor('grad'+str(step))},
                           boundary=boundary(step,rank))
            rows.append(row)
        report['updates'][str(step)]=rows
    return report


@pytest.mark.parametrize('arm',['B','N','F','R','NF','NR','FR','NFR'])
def test_all_arms_exact_counts_and_lean_full_parity(arm):
    reference=fixture(arm=arm);actual=fixture(mode='lean',arm=arm)
    result=audit.compare(reference,actual)
    assert result['passed'],result['failures']
    assert not any(key in row for row in actual['updates']['1'] for key in audit.HEAVY)


@pytest.mark.parametrize('actual',[fixture(mode='lean',final=1),fixture(start=1),fixture(start=3),fixture(final=0)])
def test_stop_resume_terminal_and_origin_only(actual):
    result=audit.compare(fixture(),actual)
    assert result['passed'],result['failures']
    if actual['final_counters']['optimizer_updates']==actual.get('resume',{}).get('completed_update',0):
        assert not actual['graph_prepared']


@pytest.mark.parametrize('mutation,expected',[
    ('open','closed_success'),('missing_update','complete_update_membership'),('false_prepare','preparation_exact'),
    ('counter','final_counters'),('rank_cursor','cursor'),('rank_metrics','replica_metrics'),
    ('raw_gradient','replica_gradients'),('source','sources'),('identity','identity_digest'),
    ('lean_hash','observation_fields'),('clipping','clip_coefficient'),('loss_mean','loss_means'),
    ('missing_rng','rng_complete'),('missing_state','state_complete'),('rank_state','replica_state'),
    ('nan','finite_json'),('float_counter','integer_final_counters'),('terminal_capture','graph_prepared'),
    ('scheduler_clock','scheduler_epoch'),('missing_metrics','complete_metrics'),('summary_clock','final_clocks'),
    ('extra_evidence','observation_fields'),('data_accounting','data_total/ce_targets'),
])
def test_structured_mutation_rejections(mutation,expected):
    r=fixture(mode='lean' if mutation=='lean_hash' else 'acceptance',start=3 if mutation=='terminal_capture' else 0)
    if mutation=='open':r['status']='running'
    elif mutation=='missing_update':del r['updates']['2']
    elif mutation=='false_prepare':r['preparation_boundary_exact'][1]=False
    elif mutation=='counter':r['final_counters']['input_tokens']+=1
    elif mutation=='rank_cursor':r['final_boundary_by_rank'][1]['cursor']['rank']=0
    elif mutation=='rank_metrics':r['updates']['1'][1]['metrics']['objective']+=1
    elif mutation=='raw_gradient':r['updates']['1'][1]['raw_gradients']['weight']['sha256']='0'*64
    elif mutation=='source':r['sources']={'new.py':'0'*64}
    elif mutation=='identity':r['configuration']['execution_identity']['payload']['arm']='B'
    elif mutation=='lean_hash':r['updates']['1'][0]['raw_gradients']={}
    elif mutation=='clipping':r['updates']['1'][0]['clipping']['coefficient_estimate']=1.
    elif mutation=='loss_mean':r['updates']['1'][0]['loss_means']['ce']=0.
    elif mutation=='missing_rng':del r['final_boundary_by_rank'][0]['rng']['torch_cuda']
    elif mutation=='missing_state':del r['final_boundary_by_rank'][0]['state']['optimizer']
    elif mutation=='rank_state':r['final_boundary_by_rank'][1]['state']['model']['weight']['sha256']='0'*64
    elif mutation=='nan':r['updates']['1'][0]['metrics']['objective']=float('nan')
    elif mutation=='float_counter':r['final_counters']['optimizer_updates']=3.0
    elif mutation=='terminal_capture':r['graph_prepared']=True
    elif mutation=='scheduler_clock':r['final_boundary_by_rank'][0]['state']['scheduler']['last_epoch']=2
    elif mutation=='missing_metrics':del r['updates']['1'][0]['metrics']['lr_next']
    elif mutation=='summary_clock':r['final_clocks']['scheduler_epoch']=0
    elif mutation=='extra_evidence':r['updates']['1'][0]['new_scientific_result']=42
    elif mutation=='data_accounting':r['observations']['1']['rank_data'][0]['ce_targets']-=1
    result=audit.audit_report(r)
    assert not result['passed'] and expected in result['failures'][0],result


@pytest.mark.parametrize('field',['model','optimizer','scheduler','rng'])
def test_same_replica_but_different_continuation_rejected(field):
    reference=fixture();actual=fixture(start=1)
    for row in actual['origin_boundary_by_rank']:
        if field=='rng':row['rng']['python'].append(42)
        else:row['state'][field]['changed']='intentional mutation'
    result=audit.compare(reference,actual)
    assert not result['passed'] and 'pair/origin_boundary' in result['failures'][0]


def test_operational_timing_differences_excluded_and_inputs_not_masked():
    reference=fixture();actual=deepcopy(reference)
    actual['timing']={'save':999};actual['updates']['1'][0]['observation_seconds']['after_update']=123
    assert audit.compare(reference,actual)['passed']
    actual['updates']['1'][0]['input']['batches']['sha256']='0'*64
    result=audit.compare(reference,actual)
    assert not result['passed'] and '/input' in result['failures'][0]


def test_missing_reference_intermediate_boundary_is_not_silently_skipped():
    result=audit.compare(fixture(mode='lean'),fixture(mode='lean',final=1))
    assert not result['passed'] and 'lacks complete boundary evidence' in result['failures'][0]


def test_optional_snapshot_bytes_and_unsafe_paths(tmp_path):
    report=fixture();path=tmp_path/'scripts/example.py';path.parent.mkdir();path.write_bytes(b'example\n')
    assert audit.audit_report(report,source_root=tmp_path)['passed']
    path.write_bytes(b'mutation\n')
    assert 'source_snapshot/scripts/example.py' in audit.audit_report(report,source_root=tmp_path)['failures'][0]
    report['sources']['../bad.py']='0'*64
    report['configuration']['execution_identity']['sha256']=audit.digest(report['configuration']['execution_identity']['payload'])
    report['source_fingerprint']['sha256']=report['source_fingerprint']['execution_identity_sha256']=report['configuration']['execution_identity']['sha256']
    assert 'source_safe/../bad.py' in audit.audit_report(report)['failures'][0]


def test_cli_pins_atomic_output_and_audit_source_snapshot(tmp_path):
    reference=tmp_path/'reference.json';actual=tmp_path/'actual.json'
    reference.write_text(json.dumps(fixture()));actual.write_text(json.dumps(fixture(start=1)))
    output=tmp_path/'audit'
    command=[sys.executable,str(Path(audit.__file__)), '--reference',str(reference),'--reference-sha256',audit.file_sha(reference),
        '--actual',str(actual),'--actual-sha256',audit.file_sha(actual),'--output-dir',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    saved=json.loads((output/'report.json').read_text())
    assert saved['passed'] and len(saved['audit_sources'])==2
    assert saved['inputs']['actual']['sha256']==audit.file_sha(actual)
    assert not list(output.glob('tmp*'))
    bad=command.copy();bad[bad.index('--actual-sha256')+1]='0'*64;bad[-1]=str(tmp_path/'bad')
    assert subprocess.run(bad,capture_output=True).returncode!=0
    assert not (tmp_path/'bad').exists()


def test_cli_rejects_report_mutation_during_audit(tmp_path,monkeypatch):
    reference=tmp_path/'r.json';actual=tmp_path/'a.json'
    reference.write_text(json.dumps(fixture()));actual.write_text(json.dumps(fixture(start=1)))
    monkeypatch.setattr(sys,'argv',['audit','--reference',str(reference),'--reference-sha256',audit.file_sha(reference),
        '--actual',str(actual),'--actual-sha256',audit.file_sha(actual),'--output-dir',str(tmp_path/'output')])
    original=audit.compare
    def mutation(*args,**kwargs):
        result=original(*args,**kwargs);actual.write_text('{}');return result
    monkeypatch.setattr(audit,'compare',mutation)
    with pytest.raises(ValueError,match='changed during JSON audit'):audit.main()
    assert not (tmp_path/'output').exists()
