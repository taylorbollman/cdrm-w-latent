"""New transport cannot alter accepted math, identities or resume policy silently."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import build_campaign_adamw
from scripts import olmo_pilot_async_execute as new
from scripts import olmo_pilot_execute as old
from scripts import olmo_pilot_async_engine as engine
from scripts import olmo_campaign_ssd_engine as accepted_engine
from test_pilot_execute import declaration, fixture_data, request


def nested_function(module,name):
    tree=ast.parse(inspect.getsource(module))
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)


@pytest.mark.parametrize('name',['save','prepare','current_boundary'])
def test_snapshot_capture_and_preparation_math_remain_literal(name):
    assert ast.dump(nested_function(engine,name))==ast.dump(nested_function(accepted_engine,name))


def test_update_math_calls_and_order_unchanged():
    names={'materialize','sum_objective_counts','CampaignObjective','validate_cursor'}
    methods={'backward','step','commit','after_backward','after_update','counts'}
    def calls(module):
        fn=nested_function(module,'update')
        return [ast.dump(n) for n in ast.walk(fn) if isinstance(n,ast.Call) and
                ((isinstance(n.func,ast.Name) and n.func.id in names) or
                 (isinstance(n.func,ast.Attribute) and n.func.attr in methods))]
    assert calls(engine)==calls(accepted_engine)


def test_native_and_tiny_constructor_are_same_math():
    assert ast.dump(ast.parse(inspect.getsource(new.construct)))==ast.dump(ast.parse(inspect.getsource(old.construct)))


@pytest.mark.parametrize('mode',['async','blocking'])
def test_transport_binds_identity_without_changing_model_or_schedule(declaration,mode):
    torch.set_num_threads(1)
    _,path=declaration;args=request(path);args.checkpoint_mode=mode
    spec=new.load_spec(args);baseline=old.load_spec(args)
    assert spec['plan']==baseline['plan'] and spec['recipe']==baseline['recipe']
    model,_,_=new.construct(spec,torch.device('cpu'))
    optimizer=build_campaign_adamw(model,spec['recipe'],fused=False)
    identity,ownership=new.construct_identity(spec,spec['recipe'],model,optimizer,{'test':'cpu'},{},{'test':'0'*64})
    other=deepcopy(spec);other['checkpoint_mode']='blocking' if mode=='async' else 'async'
    other_identity,other_ownership=new.construct_identity(other,spec['recipe'],model,optimizer,{'test':'cpu'},{},{'test':'0'*64})
    assert identity['sha256']!=other_identity['sha256']
    assert ownership==other_ownership
    assert identity['payload']['storage_policy']['transport']['mode']==mode
    for key in identity['payload']:
        if key!='storage_policy':assert identity['payload'][key]==other_identity['payload'][key]
    assert not optimizer.state


def test_source_inventory_preserves_all_accepted_sources():
    # New tests/protocol are pinned separately, never overwrite old authorities.
    found=new.source_hashes();accepted=old.source_hashes()
    assert all(found[k]==v for k,v in accepted.items())
    assert len(found)>len(accepted)
    assert all(sha256_file(new.ROOT/name)==digest for name,digest in found.items())


def test_worker_not_given_model_optimizer_or_rng():
    tree=ast.parse(inspect.getsource(engine))
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
              and n.func.id=='AsyncCheckpointRetention')
    names={n.id for n in ast.walk(call) if isinstance(n,ast.Name)}
    assert not names & {'model','optimizer','scheduler','generators','device','runner'}
    assert 'retain_without_rng' not in inspect.getsource(engine)


@pytest.mark.parametrize('mode',[None,'thread_inplace','disabled'])
def test_invalid_transport_rejected_before_cuda(declaration,mode,monkeypatch):
    _,path=declaration;args=request(path);args.checkpoint_mode=mode
    def forbidden(*a,**kw):raise AssertionError('No CUDA or model operation during invalid preflight')
    monkeypatch.setattr(new,'configure_cuda_runtime',forbidden)
    monkeypatch.setattr(new,'construct',forbidden)
    with pytest.raises(ValueError,match='checkpoint transport'):new.load_spec(args)
