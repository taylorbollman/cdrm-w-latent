"""Narrow resume/metadata guards and frozen mathematical callback equivalence."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import olmo_nfr_128_contract as c
from scripts import olmo_nfr_128_engine as engine
from scripts import olmo_nfr_128_execute as execute
from scripts import olmo_kl_continuation_engine as prior_engine


def scope_fixture():
    return {'schema':c.SCHEMA,'activation':'after_healthy_matched_NFR64_curves','arm':'NFR',
        'origin_update':64,'stop_update':128,'planned_updates':128,'kl_weight':.1,
        'named_checkpoints':[80,96,100,112,128],'extra_evaluation_updates':[100],
        'storage_prefix':'gs://fast-chunks/cdrm-w-latent/test-nfr128',
        **{k:{'path':'/tmp/'+k,'sha256':str(i)*64} for i,k in enumerate(c.REFS,1)}}


@pytest.mark.parametrize('field,value', [('origin_update',32),('stop_update',192),('planned_updates',192),
    ('arm','NF'),('kl_weight',1.),('named_checkpoints',[96,128]),('extra_evaluation_updates',[64,100]),
    ('activation','unconditional')])
def test_scope_limits(field,value):
    scope=scope_fixture();scope[field]=value
    with pytest.raises(ValueError):c.validate_scope(scope)


def model_fixture():
    conf={'lambda_kl':.1,'lambda_latent':1.}
    model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda:copy.deepcopy(conf)),
        predictor=SimpleNamespace(config=SimpleNamespace(lambda_kl=.1)),
        objective_weights=lambda:{'ce':1.,'latent':1.,'kl':.1})
    parent={'schema':'old','model':conf,'schedule':{'planned_updates':128,'warmup_tokens':52428800},
        'objective_branch':{'kl_weight':.1,'review_stop':64},'recipe':{'lr':.0002},
        'execution_identity':{'sha256':'x'*64,'payload':{'sources':{'old':'a'*64},
            'evaluation':{'scheduled_updates':list(range(16,129,16)),'panels':{'same':'unchanged'}},
            'plan':{'unchanged':True}}}}
    meta={'parent_update':64,'stop_update':128,'objective_change':False,'parent_identity_sha256':'x'*64}
    evaluation=c.continuation_evaluation(parent['execution_identity']['payload']['evaluation'],scope_fixture())
    child=c.continued_configuration(parent,meta,{'old':'a'*64,'new':'b'*64},evaluation)
    return model,parent,meta,child


def test_metadata_transition_preserves_parent_and_training_fields():
    model,parent,meta,child=model_fixture();before=copy.deepcopy(parent)
    result=c.validate_continuation_transition(parent,child,model,parent['objective_branch'],meta)
    assert result['metadata_only'] is True and parent==before
    for key in ('model','schedule','objective_branch','recipe'):assert child[key]==parent[key]
    assert child['execution_identity']['payload']['plan']==parent['execution_identity']['payload']['plan']
    assert child['execution_identity']['payload']['evaluation']['scheduled_updates']==[16,32,48,64,80,96,100,112,128]


@pytest.mark.parametrize('change',['lr','model','schedule','objective','data','eval_panel','extra_eval','stop'])
def test_metadata_transition_rejects_training_or_panel_change(change):
    model,parent,meta,child=model_fixture()
    if change=='lr':child['recipe']['lr']=.01
    elif change=='model':child['model']['lambda_kl']=1.
    elif change=='schedule':child['schedule']['warmup_tokens']=1
    elif change=='objective':child['objective_branch']['kl_weight']=1.
    elif change=='data':child['execution_identity']['payload']['plan']={}
    elif change=='eval_panel':child['execution_identity']['payload']['evaluation']['panels']={}
    elif change=='extra_eval':child['execution_identity']['payload']['evaluation']['scheduled_updates'].append(90)
    elif change=='stop':meta['stop_update']=192
    with pytest.raises(ValueError):c.validate_continuation_transition(parent,child,model,parent['objective_branch'],meta)


def callbacks(module):
    tree=ast.parse(inspect.getsource(module.run_segment))
    return {n.name:ast.dump(n,include_attributes=False) for n in tree.body[0].body if isinstance(n,ast.FunctionDef)}


def test_update_save_prepare_and_tracking_math_literal_frozen_copies():
    before,after=callbacks(prior_engine),callbacks(engine)
    # Scope-specific restore handling sits outside these functions. Even
    # materialization, graph preparation, global denominators, update timing,
    # clipping and storage callbacks must remain literally unchanged.
    for name in ('persist','current_boundary','save','create_manager','submit','poll','accept','prepare','update','log'):
        assert after[name]==before[name],name


def test_restore_uses_old_configuration_only_at_transition_and_no_objective_fork():
    tree=ast.parse(inspect.getsource(engine.run_segment))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    load=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='load_distributed_checkpoint')
    kwargs={k.arg:ast.unparse(k.value) for k in load.keywords}
    assert kwargs['configuration']=='parent_configuration if transition_parent else configuration'
    assert kwargs['source_fingerprint']=='parent_fingerprint if transition_parent else source_fingerprint'
    assert not any(isinstance(n.func,ast.Name) and n.func.id=='set_kl_weight' for n in calls)
    assert 'before != parent_boundary_by_rank' in inspect.getsource(engine.run_segment)
    assert 'if before != after:' in inspect.getsource(engine.run_segment)


def test_named100_checkpoint_union_and_unchanged_loop():
    text=inspect.getsource(engine.run_segment)
    assert '|set(named_checkpoints)' in text
    tree=ast.parse(text)
    newcall=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_loop')
    oldtree=ast.parse(inspect.getsource(prior_engine.run_segment))
    oldcall=next(n for n in ast.walk(oldtree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_loop')
    assert ast.dump(newcall)==ast.dump(oldcall)


def test_no_monkeypatch_of_frozen_module_globals():
    for module in (engine,execute,c):
        tree=ast.parse(inspect.getsource(module))
        assert not any(isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)
            and isinstance(n.value,ast.Name) and n.value.id in ('accepted','prior','prior_engine') for n in ast.walk(tree))


def publication_fixture():
    parent={'manifest_sha256':'a'*64,'state':{'sha256':'b'*64},'metadata':{},'counters':{'optimizer_updates':64}}
    pub=copy.deepcopy(parent);pub['retention']={'download_sha256_verified':True,'create_only':True,
        'objects':[{'sha256':pin,'verification':{k:True for k in ('download_sha256','server_md5','server_size','sha256_metadata')}}
                   for pin in ('a'*64,'b'*64)]}
    return parent,pub


@pytest.mark.parametrize('change',['none','state','cloud','object','counter'])
def test_require_verified_matching_cloud64(change):
    parent,pub=publication_fixture()
    if change=='state':pub['state']={}
    elif change=='cloud':pub['retention']['objects'][0]['verification']['download_sha256']=False
    elif change=='object':pub['retention']['objects'].pop()
    elif change=='counter':pub['counters']['optimizer_updates']=63
    if change=='none':c.verify_publication(pub,parent)
    else:
        with pytest.raises(ValueError):c.verify_publication(pub,parent)


def endpoint_fixture(scope):
    common={'declaration':{'checkpoints':{'control':{'path':'/tmp/control','sha256':'e'*64},
                                        'reduced':scope['parent64_checkpoint']}},'sha256':'f'*64}
    endpoints={}
    for name,weight in [('control',1.),('reduced',.1)]:
        endpoints[name]={'schema':'olmo-nfr-endpoint-curves-v1','status':'completed','arm':'NFR',
            'after_update':64,'case':name,'kl_weight':weight,'num_passes':32,'optimizer_updates_performed':0,
            'weights_unchanged':True,'rng_unchanged':True,'gradient_buffers_absent':True,
            'preservation':{'integrity_passed':True,'restored':True,'checks':{'state':True}},
            'endpoint_scope':copy.deepcopy(common),'membership_sha256':'m','index_manifest_sha256':'i',
            'batch_tensor_sha256':['b'],'probe_policy':{'identical':True},
            'input_authorities':{'checkpoint':common['declaration']['checkpoints'][name]},
            'result':{'input_tokens':8192,'ce_targets':8184,'policy':'common_fp32_no_jitter_v1','beta':1.,
                      'passes':[{'pass':i} for i in range(1,33)]}}
    return endpoints


@pytest.mark.parametrize('change',['none','schema','state','case','preservation','panel','source','scope','incomplete'])
def test_activation_requires_matching_completed_endpoint_authorities(monkeypatch,change):
    scope=scope_fixture();reports=endpoint_fixture(scope)
    receipt={'schema':'olmo-nfr-128-activation-v1','decision':'continue_reduced_to128','continuation_sha256':'c'*64,
        'parent64_manifest_sha256':scope['parent64_checkpoint']['sha256'],
        'endpoint_reports':{name:{'path':'/tmp/'+name,'sha256':'a'*64} for name in reports},'rationale':'bounded'}
    if change=='schema':reports['control']['schema']='other'
    elif change=='state':reports['reduced']['input_authorities']['checkpoint']={}
    elif change=='case':reports['control']['case']='reduced'
    elif change=='preservation':reports['control']['rng_unchanged']=False
    elif change=='panel':reports['reduced']['batch_tensor_sha256']=['different']
    elif change=='source':receipt['parent64_manifest_sha256']='0'*64
    elif change=='scope':reports['reduced']['endpoint_scope']['sha256']='0'*64
    elif change=='incomplete':reports['reduced']['status']='running'
    monkeypatch.setattr(c.legacy,'read_json',lambda path,*a,**k:receipt if str(path)=='/tmp/activation' else reports[Path(path).name])
    if change=='none':assert c.validate_activation(Path('/tmp/activation'),'b'*64,scope,'c'*64)==receipt
    else:
        with pytest.raises(ValueError):c.validate_activation(Path('/tmp/activation'),'b'*64,scope,'c'*64)


def test_historical_training215_still_byte_exact():
    sources=json.loads((c.ROOT/'.runtime/olmo-nfr-kl-continuation/runtime-sources.json').read_text())
    assert len(sources)==215
    assert all(c.sha256_file(c.ROOT/name)==pin for name,pin in sources.items())
