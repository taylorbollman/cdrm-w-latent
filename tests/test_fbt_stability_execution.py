"""F-only startup invariants and unchanged accepted execution mathematics."""
import ast
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model, build_campaign_adamw
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_fbt_stability_contract as contract
from scripts import olmo_fbt_stability_execute as execute
from scripts import olmo_fbt_stability_engine as engine
from scripts import olmo_pilot_async_engine as accepted_engine
from scripts import olmo_pilot_execution_contract as ordered_contract
from scripts.olmo_lm_common import tree_digests
from test_pilot_execute import fixture_data


def nested(module, name):
    return next(n for n in ast.walk(ast.parse(inspect.getsource(module)))
                if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize('name', ['save', 'prepare', 'update', 'current_boundary', 'log'])
def test_accepted_update_and_save_callbacks_literal(name):
    assert ast.dump(nested(engine, name)) == ast.dump(nested(accepted_engine, name))


@pytest.fixture
def historical():
    torch.set_num_threads(1)
    recipe = CampaignRecipe('NF', sequence_length=16, rt_layers=(0,1), effective_valid_tokens=80,
        warmup_tokens=240)
    model = build_campaign_model(OLMoTiledRTForCausalLM(OLMoConfig.tiny(),
        attention_backend='math', attention_precision='fp32', tile_backend='eager',
        backward_tile_backend='eager', backward_memory='recompute'), recipe)
    return model, recipe


def test_f_wrapper_retains_imported_core_and_drops_predictor_before_adam(historical):
    original, historical_recipe = historical
    recipe = replace(historical_recipe, arm='F', document_policy='continuous-stream-v1')
    core = original.backbone
    before = tree_digests(core.state_dict()); rng = torch.random.get_rng_state().clone()
    model, receipt = contract.import_f_wrapper(original, historical_recipe, recipe)
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    assert model.backbone is core and tree_digests(core.state_dict()) == before
    assert model.predictor is None and not model.enabled
    assert all(receipt['checks'].values()) and not receipt['loaded_optimizer']
    assert not optimizer.state and torch.equal(rng, torch.random.get_rng_state())
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert owned == {id(p) for p in core.parameters()}
    assert not owned & {id(p) for p in original.predictor.parameters()}
    assert {group['component'] for group in optimizer.param_groups} == {'backbone','fusion'}
    assert engine.model_contract(model,recipe,optimizer)['weights']['latent'] == 0


@pytest.mark.parametrize('change',['arm','policy','historical_arm','seed','jitter','disabled'])
def test_wrapper_rejects_other_startups(historical,change):
    model, old = historical
    new = replace(old,arm='F',document_policy='continuous-stream-v1')
    if change=='arm':new=replace(new,arm='NF')
    elif change=='policy':new=replace(new,document_policy='isolated-v1')
    elif change=='historical_arm':old=replace(old,arm='F')
    elif change=='seed':new=replace(new,fusion_seed=new.fusion_seed+1)
    elif change=='jitter':new=replace(new,feedback_jitter=.03)
    elif change=='disabled':model._enabled=False
    before=tree_digests(model.state_dict())
    with pytest.raises(ValueError):contract.import_f_wrapper(model,old,new)
    assert tree_digests(model.state_dict())==before


def plans():
    old={'updates':[{'counts':{'valid_tokens':1024},'membership_sha256':str(n),
        'allocation_by_arm':{'NF':[{'rank':0}], 'NFR':[{'rank':0}]}} for n in range(128)],
        'valid_token_prefix':[n*1024 for n in range(129)]}
    new={'updates':[{'counts':{'valid_tokens':1024},'membership_sha256':str(n),
        'allocation_by_arm':{'F':[{'rank':0}]}} for n in range(192)],
        'valid_token_prefix':[n*1024 for n in range(193)]}
    return new,old


def test_shared_prefix_allocation_and_membership_are_checked():
    new,old=plans()
    assert all(contract.check_shared_prefix(new,old).values())


@pytest.mark.parametrize('change',['membership','allocation','tokens','horizon'])
def test_shared_prefix_rejects_changes(change):
    new,old=plans()
    if change=='membership':new['updates'][90]['membership_sha256']='changed'
    elif change=='allocation':new['updates'][90]['allocation_by_arm']['F'][0]['rank']=1
    elif change=='tokens':new['valid_token_prefix'][90]+=1
    else:new['updates'].pop()
    with pytest.raises(ValueError):contract.check_shared_prefix(new,old)


@pytest.fixture
def tiny_declaration(fixture_data,tmp_path):
    value={'schema':execute.TINY_SCHEMA,'arm':'F','data':deepcopy(fixture_data),
        'storage_prefix':'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/fbt-stability/cpu',
        'seed':20260929,'probe':{'physical_batch':1,'panel_rows':2},
        'evaluation':{'kind':'ordered_named_dev_panels_v1',
            'panels':[{'name':'dev-main','target_valid_tokens':80}],
            'physical_batch_by_arm':{'F':1},'every_updates':2,'precision':'fp32',
            'feedback_jitter':0.,'report_passes':'all_trained_passes','generation':'not_implemented'}}
    path=tmp_path/'declaration.json';path.write_text(json.dumps(value))
    return value,path


def request(path):
    return SimpleNamespace(declaration=path,declaration_sha256=sha256_file(path),resolved=None,
        resolved_sha256=None,arm='F',stop_after=None,checkpoint_mode='async')


def test_tiny_identity_has_explicit_probe_schedule_and_zero_aux(tiny_declaration):
    value,path=tiny_declaration;spec=execute.load_spec(request(path))
    model,source,imported=execute.construct(spec,torch.device('cpu'))
    optimizer=build_campaign_adamw(model,spec['recipe'],fused=False)
    identity,ownership=execute.construct_identity(spec,spec['recipe'],model,optimizer,{'device':'cpu'},
        {'test':True},{'fixture':'0'*64})
    assert spec['evaluation_plan']['scheduled_updates']==[0,2]
    assert spec['checkpoint_updates']==[] and spec['recipe'].arm=='F'
    assert spec['probe_plan']['scheduled_updates']==[0,2,3]
    assert spec['probe_plan']['acceptance_schedule']=='origin-live-graph-next-update-terminal-v1'
    assert ownership['component_parameters']['predictor']==0 and not optimizer.state
    counts=ordered_contract.expected_counters(identity,3)
    assert counts['input_tokens']==240 and counts['latent_pairs']==counts['kl_triples']==0
    changed=deepcopy(spec);changed['probe_plan']['test_identity_change']=True
    other,_=execute.construct_identity(changed,spec['recipe'],model,optimizer,{'device':'cpu'},
        {'test':True},{'fixture':'0'*64})
    assert other['sha256']!=identity['sha256']


def test_no_historical_source_changed():
    found=contract.source_hashes()
    assert len(found)>200
    assert all(sha256_file(contract.ROOT/name)==digest for name,digest in found.items())


@pytest.mark.parametrize('field,value',[('updates',128),('initial_stop_after',192),('probe',{'panel_rows':16,'physical_batch':1})])
def test_native_scope_rejects_undeclared_changes(field,value):
    declaration={'schema':contract.SCHEMA,'parent_declaration':{'path':'/tmp/parent','sha256':'0'*64},
        'parent_resolved':{'path':'/tmp/resolved','sha256':'0'*64},
        'parent_report':{'path':'/tmp/report','sha256':'0'*64},'updates':192,'initial_stop_after':128,
        'storage_prefix':'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/fbt-stability/cpu',
        'probe':{'panel_rows':8,'physical_batch':1}}
    declaration[field]=value
    with pytest.raises(ValueError):contract.validate_declaration(declaration)


@pytest.mark.parametrize('completed',[0,2,128])
def test_origin_tracking_merges_ordinary_and_probe_metrics_once(completed):
    calls=[]
    class Tracker:
        def log(self, values, *, step):calls.append((step,values))
    class Controller:
        def run_if_due(self, update, **kwargs):
            assert update==completed and kwargs['runner'] is None
            assert kwargs['log_metrics'] is False
            self.values={'dev/main/pass_1/ce':3.,'dev/stability/all/pass_4/ce':4.}
        def metrics_for(self, update):
            assert update==completed
            return self.values
    class Coordinator:
        def call(self, name, action, *, rank_zero):
            assert rank_zero is True
            return action()
    engine.evaluate_origin(evaluation=Controller(),completed=completed,coordinator=Coordinator(),
        tracker=Tracker(),model=object(),generators={},training_data=object(),boundary=lambda:None,persist=lambda:None)
    assert calls==[(completed,{'update':completed,'dev/main/pass_1/ce':3.,'dev/stability/all/pass_4/ce':4.})]


def test_origin_tracking_nonzero_rank_still_enters_collective():
    entered=[]
    class Coordinator:
        def call(self,name,action,*,rank_zero):
            entered.append((name,rank_zero))
            # Rank one participates but never executes rank-zero publication.
    class Controller:
        def run_if_due(self,*args,**kwargs):assert kwargs['log_metrics'] is False
        def metrics_for(self,*args):raise AssertionError('Only rank zero derives published metrics')
    engine.evaluate_origin(evaluation=Controller(),completed=0,coordinator=Coordinator(),tracker=None,
        model=object(),generators={},training_data=object(),boundary=lambda:None,persist=lambda:None)
    assert entered==[('combined origin evaluation tracking',True)]
