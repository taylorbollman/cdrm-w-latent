"""Mutation tests for independent continuation identity and named100 checks."""
import ast
from copy import deepcopy
from functools import lru_cache
import inspect
import json
from pathlib import Path
import pytest
from scripts import olmo_nfr_128_audit as audit
from scripts import olmo_kl_continuation_audit as previous


@lru_cache(maxsize=1)
def authorities():
    root=audit.ROOT/'.runtime/olmo-nfr-stability-128'
    prep=root/'continuation-prepared-01'
    old=audit.ROOT/'.runtime/olmo-nfr-kl-continuation/native-nfr-reduced-32to64-01/report.json'
    parent=json.loads(old.read_text());scope=json.loads((prep/'scope.json').read_text());resolved=json.loads((prep/'resolved.json').read_text())
    # Synthetic terminal envelope around real pinned metadata, not purported run
    # evidence. Full training/transport validators run only on actual completion.
    inputs={'parent64':{'path':str(old),'sha256':audit.file_sha(old)},
        'scope':{'path':str(prep/'scope.json'),'sha256':audit.file_sha(prep/'scope.json')},
        'activation':{'path':'synthetic','sha256':'a'*64}}
    activation={'schema':'olmo-nfr-128-activation-v1','decision':'continue_reduced_to128',
        'continuation_sha256':inputs['scope']['sha256'],'parent64_manifest_sha256':scope['parent64_checkpoint']['sha256']}
    report={'configuration':resolved['configuration'],'nfr128_scope':{'schema':'olmo-nfr-128-execution-scope-v1',
        'declaration':scope,'sha256':inputs['scope']['sha256']},'activation':{'receipt':activation,'sha256':'a'*64},
        'original_configuration':parent['original_configuration'],'parent64_configuration':parent['configuration'],
        'parent64_fingerprint':parent['source_fingerprint'],'parent64_report_sha256':inputs['parent64']['sha256'],
        'continuation':resolved['continuation'],'branch':parent['branch'],'sources':resolved['sources'],
        'resume':{'completed_update':64,'manifest_sha256':scope['parent64_checkpoint']['sha256']},
        'adam_resident_before_ddp':True,'origin_clocks':parent['final_clocks'],
        'origin_boundary_by_rank':parent['final_boundary_by_rank'],
        'updates':{},'final_counters':{'optimizer_updates':128},'wandb':{'status':'synced'},
        'local_checkpoints':[{'optimizer_update':n} for n in scope['named_checkpoints']],
        'published_checkpoints':[{'counters':{'optimizer_updates':n}} for n in scope['named_checkpoints']],
        'last_verified_cloud_update':128,'evaluations':[e for e in parent['evaluations'] if e['after_update']==64]}
    change={'schema':'olmo-nfr-128-transition-check-v1','metadata_only':True,'objective_unchanged':True,
        'parent_identity_sha256':parent['configuration']['execution_identity']['sha256'],
        'child_identity_sha256':resolved['configuration']['execution_identity']['sha256']}
    report['continuation_transition']={'parent_manifest_sha256':scope['parent64_checkpoint']['sha256'],
        'before_boundary_by_rank':parent['final_boundary_by_rank'],'after_boundary_by_rank':parent['final_boundary_by_rank'],
        'configuration_change_by_rank':[change,change]}
    return report,parent,scope,activation,resolved,inputs


def fixture():
    # Independently deserialized files cannot share mutable nested objects.
    return tuple(deepcopy(value) for value in authorities())


@pytest.mark.parametrize('change',['none','weights','Adam_origin','LR','data','warmup','extra_evaluation',
    'ancestral_source','schema','KL','original64','parent_pin','transition','scope_stop','state_fork','cloud100','dev64'])
def test_metadata_only_scope_and_origin_mutations(change):
    report,parent,scope,activation,resolved,inputs=fixture()
    if change=='weights':report['configuration']['model']['lambda_latent']=.2
    elif change=='Adam_origin':report['origin_clocks']['adam_parameters']=0
    elif change=='LR':report['configuration']['recipe']['plateau_lr']=.001
    elif change=='data':report['configuration']['execution_identity']['payload']['plan']['updates'].pop()
    elif change=='warmup':report['configuration']['schedule']['warmup_tokens']=1
    elif change=='extra_evaluation':report['configuration']['execution_identity']['payload']['evaluation']['scheduled_updates'].append(101)
    elif change=='ancestral_source':report['sources'][next(iter(parent['sources']))]='f'*64
    elif change=='schema':report['configuration']['schema']='old'
    elif change=='KL':report['branch']['kl_weight']=1.
    elif change=='original64':report['parent64_fingerprint']={}
    elif change=='parent_pin':scope['parent64_report']['sha256']='0'*64
    elif change=='transition':report['continuation_transition']['after_boundary_by_rank']=[]
    elif change=='scope_stop':scope['stop_update']=192
    elif change=='state_fork':report['objective_transition']={}
    elif change=='cloud100':report['published_checkpoints']=[r for r in report['published_checkpoints'] if r['counters']['optimizer_updates']!=100]
    elif change=='dev64':report['evaluations'][0]['panels']['dev-main']['result']['passes'][0]['means']['ce']+=.01
    checker=audit.Audit()
    if change=='none':
        audit.metadata_check(checker,report,parent,scope,activation,resolved,inputs)
        assert checker.result()['passed']
    else:
        with pytest.raises((ValueError,KeyError,IndexError)):
            audit.metadata_check(checker,report,parent,scope,activation,resolved,inputs)


def test_evaluation_validator_diff_is_only_declared100():
    old=inspect.getsource(previous.evaluation_check)
    old=old.replace('    scheduled = list(range(interval, len(payload[\'plan\'][\'updates\'])+1, interval))',
        '    scheduled = sorted(set(range(interval, len(payload[\'plan\'][\'updates\'])+1, interval)) | {100})')
    assert ast.dump(ast.parse(old))==ast.dump(ast.parse(inspect.getsource(audit.evaluation_check)))


def test_reuse_original_training_and_transport_without_relabel():
    assert audit.training_check is previous.training_check
    assert audit.transport_check is previous.transport_check


@pytest.mark.parametrize('missing',[None,100,96,128])
def test_mandatory_evaluation100_coverage(missing):
    report,parent,scope,activation,resolved,inputs=fixture()
    report['scale']='native'
    report['evaluation_policy']=deepcopy(parent['evaluation_policy'])
    report['evaluation_policy']['plan']=deepcopy(resolved['evaluation_plan'])
    base=deepcopy(report['evaluations'][0]);report['evaluations']=[]
    for n in [64,80,96,100,112,128]:
        if n==missing:continue
        row=deepcopy(base);row['after_update']=n;report['evaluations'].append(row)
    checker=audit.Audit()
    if missing is None:audit.evaluation_check(checker,report,'eval')
    else:
        with pytest.raises(ValueError,match='actual_schedule'):audit.evaluation_check(checker,report,'eval')


def test_cli_pins_are_flat_references_and_snapshots_are_exact(tmp_path,monkeypatch):
    arguments=[];expected={}
    for name in audit.INPUTS:
        path=tmp_path/(name+'.json');path.write_text(json.dumps({'input':name})+'\n')
        pin=audit.file_sha(path);expected[name]=pin
        arguments+=['--'+name.replace('_','-'),str(path),'--'+name.replace('_','-')+'-sha256',pin]
    monkeypatch.setattr(audit,'load_transport_evidence',lambda *a:({},{}))
    def examine(*values,**kwargs):
        refs=values[-1]
        assert all(isinstance(refs[name]['sha256'],str) and refs[name]['sha256']==pin for name,pin in expected.items())
        return {'passed':True,'checks':[],'failures':[],'inputs':refs,
                'audit_sources':{n:audit.file_sha(audit.ROOT/n) for n in audit.AUDIT_SOURCES}}
    monkeypatch.setattr(audit,'audit_report',examine)
    output=tmp_path/'output'
    with pytest.raises(SystemExit) as exc:audit.main(arguments+['--output-dir',str(output)])
    assert exc.value.code==0
    record=json.loads((output/'report.json').read_text())
    assert record['inputs']['continuation']['sha256']==expected['continuation']
    for name,pin in expected.items():assert audit.file_sha(output/'input-snapshot'/(name+'.json'))==pin
