"""New NFR declaration authorizes a scope, never a changed execution engine."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from scripts import olmo_nfr_kl_contract as contract
from scripts import olmo_nfr_kl_execute as execute


def fixture():
    scope={'schema':contract.SCHEMA,'activation':'conditional_after_fbt_only_assessment',
        'arm':'NFR','parent_update':32,'review_stop':64,'planned_updates':128,'kl_weights':[1.,.1],
        'storage_prefix':'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/nfr-kl/test',
        **{k:{'path':'/tmp/'+k,'sha256':str(i)*64} for i,k in enumerate(contract.REFS,1)}}
    args=contract.arguments_from_scope(scope)
    sources={'frozen.py':'a'*64}
    plan={'updates':[{}]*128}
    config={'execution_identity':{'payload':{'arm':'NFR','plan':plan,
        'model_contract':{'mode':{'rt_mode':{'selected_layers':[0,15],'alpha':1.},'num_passes':4},
                          'weights':{'ce':1.,'latent':1.,'kl':1.}}}},
        'schedule':{'planned_updates':128}}
    parent={'state':{'sha256':'s'},'counters':{'optimizer_updates':32},
        'metadata':{'configuration':config,'source_fingerprint':{'sources':sources}}}
    reference={'schema':contract.accepted.accepted.SCHEMA,'arm':'NFR','status':'stopped_at_boundary',
        'sources':sources,'configuration':config,'source_fingerprint':parent['metadata']['source_fingerprint'],
        'declaration_sha256':args.declaration_sha256,'resolved_sha256':args.resolved_sha256,
        'published_checkpoints':[{'manifest_sha256':args.parent_manifest_sha256,'state':parent['state']}],
        'local_checkpoints':[{'receipt':{'manifest_sha256':args.parent_manifest_sha256},
                             'boundary_by_rank':[{'rank':0},{'rank':1}]}]}
    args.spec={'kind':'native','plan':plan}
    return scope,args,parent,reference,sources


@pytest.mark.parametrize('change',['none','configuration','publication','boundary','arm','sources','RT','update','plan'])
def test_exact_original_NFR_parent_required(monkeypatch,change):
    scope,args,parent,reference,sources=fixture()
    if change=='configuration':reference['configuration']={}
    elif change=='publication':reference['published_checkpoints']=[]
    elif change=='boundary':reference['local_checkpoints']=[]
    elif change=='arm':reference['arm']='NF'
    elif change=='sources':reference['sources']={}
    elif change=='RT':parent['metadata']['configuration']['execution_identity']['payload']['model_contract']['mode']['rt_mode']['selected_layers']=[]
    elif change=='update':parent['counters']['optimizer_updates']=31
    elif change=='plan':args.spec={'kind':'native','plan':{'updates':[{}]*127}}
    monkeypatch.setattr(contract,'inspect_distributed_checkpoint',lambda *a,**k:parent)
    monkeypatch.setattr(contract.legacy,'read_json',lambda *a,**k:reference)
    monkeypatch.setattr(contract.accepted.accepted,'source_hashes',lambda:sources)
    if change=='none':
        assert contract.authenticated_parent(args,scope)==parent
        assert args.parent_boundary_by_rank==[{'rank':0},{'rank':1}]
    else:
        with pytest.raises(ValueError):contract.authenticated_parent(args,scope)


@pytest.mark.parametrize('field,value',[('arm','NF'),('parent_update',0),('review_stop',128),
    ('planned_updates',192),('kl_weights',[1.,.2]),('activation','launch_automatically')])
def test_scope_rejects_unrequested_variants(field,value):
    scope,*_=fixture();scope[field]=value
    with pytest.raises(ValueError):contract.validate_scope(scope)


@pytest.mark.parametrize('field,value',[('arm','NF'),('stop_after',128),('observation','acceptance'),
    ('checkpoint_mode','blocking'),('branch','other'),('kl_weight',.5),('parent_manifest_sha256','0'*64),
    ('resolved',Path('/tmp/other'))])
def test_command_must_match_declared_scope(field,value):
    scope,args,*_=fixture();setattr(args,field,value)
    with pytest.raises(ValueError):contract.validate_arguments(args,scope)


def test_reduced_branch_same_parent_and_no_scope_mutation():
    scope,args,*_=fixture();before=deepcopy(scope)
    args.branch='reduced';args.kl_weight=.1
    contract.validate_arguments(args,scope)
    assert scope==before


def test_launcher_reuses_accepted_stage_without_assigning_its_globals():
    tree=ast.parse(inspect.getsource(execute))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='run_stage_releasing_failure']
    assert len(calls)==1
    value=next(k.value for k in calls[0].keywords if k.arg=='stage')
    assert ast.unparse(value)=='accepted.run_stage'
    assert not any(isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)
                   and isinstance(n.value,ast.Name) and n.value.id=='accepted' for n in ast.walk(tree))


def test_historical_and_F_only_sources_still_frozen():
    from cdrm.pretrained.artifacts import sha256_file
    for filename,expected in (('olmo-kl-continuation',210),('olmo-fbt-stability',208)):
        sources=json.loads((contract.ROOT/'.runtime'/filename/'runtime-sources.json').read_text())
        assert len(sources)==expected
        assert all(sha256_file(contract.ROOT/name)==pin for name,pin in sources.items())
