"""Branch integration preserves accepted numerical callbacks and strict lineage."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model, build_campaign_adamw
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_kl_continuation as run
from scripts.olmo_kl_branch import set_kl_weight, declared_recipe


ROOT=Path(__file__).resolve().parents[1]


def functions(path):
    tree=ast.parse(path.read_text())
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
            if isinstance(n,ast.FunctionDef)}


def test_mathematical_and_storage_callbacks_are_literal_accepted_code():
    old=functions(ROOT/'scripts/olmo_pilot_async_engine.py')
    new=functions(ROOT/'scripts/olmo_kl_continuation_engine.py')
    for name in ('prepare','update','save','submit','poll','accept','log','current_boundary'):
        assert old[name]==new[name],name


def test_branch_identity_is_explicit_and_does_not_mutate_parent():
    torch.set_num_threads(1)
    recipe=CampaignRecipe('NF',sequence_length=6)
    model=build_campaign_model(OLMoTiledRTForCausalLM(OLMoConfig.tiny(),attention_backend='math'),recipe)
    optimizer=build_campaign_adamw(model,recipe,fused=False)
    set_kl_weight(model,.1)
    parent={'schema':run.contract.IDENTITY_SCHEMA,'payload':{'arm':'NF','plan':{'fixed':[3,4]},
            'schedule':{'fixed':'hash'},'sources':{'old.py':'a'*64}},'sha256':'b'*64}
    before=copy.deepcopy(parent)
    branch={'kl_weight':.1,'parent_manifest_sha256':'c'*64,
        'recipe_as_declared':declared_recipe(recipe,kl_weight=.1,parent_manifest_sha256='c'*64)}
    identity,ownership=run.branch_identity(parent,branch,recipe,model,optimizer,{'new.py':'d'*64})
    assert parent==before
    assert identity['payload']['plan']==parent['payload']['plan']
    assert identity['payload']['schedule']==parent['payload']['schedule']
    assert identity['payload']['objective_branch']==branch
    assert identity['sha256']==run.legacy.digest(identity['payload'])
    assert ownership['weights']=={'ce':1.,'latent':1.,'kl':.1}
    assert identity['payload']['recipe']['optimizer_state']=='inherited_exact_parent_checkpoint'


def test_child_resume_rejects_other_objective_branch():
    identity={'schema':run.contract.IDENTITY_SCHEMA,'payload':{'kl':.1},'sha256':'x'}
    saved={'metadata':{'configuration':{'execution_identity':identity}}}
    run.validate_branch_resume(saved,copy.deepcopy(identity))
    with pytest.raises(ValueError,match='another objective branch'):
        run.validate_branch_resume(saved,{**identity,'payload':{'kl':1.}})


def parent_fixture():
    sources={'frozen.py':'d'*64}
    config={'execution_identity':{'sha256':'i'}}
    fingerprint={'sources':sources}
    parent={'manifest_sha256':'a'*64,'state':{'sha256':'s'},'counters':{'optimizer_updates':32},
            'metadata':{'configuration':config,'source_fingerprint':fingerprint}}
    reference={'schema':run.accepted.SCHEMA,'status':'stopped_at_boundary','sources':sources,
        'configuration':config,'source_fingerprint':fingerprint,'declaration_sha256':'b'*64,
        'resolved_sha256':'c'*64,'published_checkpoints':[{'manifest_sha256':'a'*64,'state':parent['state']}],
        'local_checkpoints':[{'receipt':{'manifest_sha256':'a'*64},'boundary_by_rank':[{'rank':0},{'rank':1}]}]}
    args=SimpleNamespace(parent_checkpoint=Path('/tmp/parent'),parent_manifest_sha256='a'*64,
        parent_report=Path('/tmp/report'),parent_report_sha256='e'*64,declaration_sha256='b'*64,
        resolved_sha256='c'*64,spec={'kind':'native'},arm='NF',stop_after=64,observation='lean')
    return sources,parent,reference,args


@pytest.mark.parametrize('corrupt',('none','configuration','publication','boundary','scope','sources'))
def test_parent_preflight_requires_exact_published_lineage(monkeypatch,corrupt):
    sources,parent,reference,args=parent_fixture()
    if corrupt=='configuration':reference['configuration']={}
    if corrupt=='publication':reference['published_checkpoints']=[]
    if corrupt=='boundary':reference['local_checkpoints']=[]
    if corrupt=='scope':args.stop_after=128
    if corrupt=='sources':reference['sources']={}
    monkeypatch.setattr(run,'inspect_distributed_checkpoint',lambda *a,**k:parent)
    monkeypatch.setattr(run.legacy,'read_json',lambda *a,**k:reference)
    monkeypatch.setattr(run.accepted,'source_hashes',lambda:sources)
    if corrupt!='none':
        with pytest.raises(ValueError):run.authenticated_parent(args)
    else:
        assert run.authenticated_parent(args)==parent
        assert args.parent_boundary_by_rank==[{'rank':0},{'rank':1}]
