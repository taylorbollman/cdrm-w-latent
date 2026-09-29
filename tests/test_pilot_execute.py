"""CPU checks for ordered-runner authority, ownership and graph policy."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import ARMS, build_campaign_adamw
from scripts import olmo_pilot_execute as cli
from scripts import olmo_pilot_execution_contract as contract
from scripts.olmo_pilot_execution_fixture import build_fixture


@pytest.fixture(scope='module')
def fixture_data(tmp_path_factory):
    return build_fixture(tmp_path_factory.mktemp('ordered-executor')/'fixture')['data_spec']


@pytest.fixture
def declaration(fixture_data,tmp_path):
    value={'schema':cli.TINY_SCHEMA,'arm':'NFR','data':deepcopy(fixture_data),
        'storage_prefix':'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T190649Z/pilot-execution-cpu',
        'seed':20260929,'evaluation':{'kind':'ordered_named_dev_panels_v1',
            'panels':[{'name':'dev-main','target_valid_tokens':80}],
            'physical_batch_by_arm':{'NFR':1},'every_updates':2,'precision':'fp32',
            'feedback_jitter':0.,'report_passes':'all_trained_passes','generation':'not_implemented'}}
    path=tmp_path/'declaration.json';path.write_text(json.dumps(value));return value,path


def request(path,arm='NFR'):
    return SimpleNamespace(declaration=path,declaration_sha256=sha256_file(path),resolved=None,
        resolved_sha256=None,arm=arm,stop_after=None)


@pytest.mark.parametrize('arm',ARMS)
def test_all_arms_keep_fresh_optimizer_and_bind_ordered_origin(declaration,arm):
    torch.set_num_threads(1)
    value,path=declaration;value['arm']=arm
    value['evaluation']['physical_batch_by_arm']={arm:1};path.write_text(json.dumps(value))
    spec=cli.load_spec(request(path,arm))
    assert [u['counts']['valid_tokens'] for u in spec['plan']['updates']]==[80]*3
    model,source,imported=cli.construct(spec,torch.device('cpu'))
    optimizer=build_campaign_adamw(model,spec['recipe'],fused=False)
    identity,ownership=cli.construct_identity(spec,spec['recipe'],model,optimizer,{'device':'cpu'},
        {'test':True},{'fixture':'0'*64})
    assert not optimizer.state and not imported and ownership['arm']==arm
    assert identity['schema']==contract.IDENTITY_SCHEMA
    assert identity['payload']['scope']=='ordered-tiny-acceptance-not-native'
    assert identity['payload']['data']==value['data']
    assert source['kind']=='deterministic_random_tiny' and ownership['tied_readout']
    counts=contract.expected_counters(identity,3)
    assert counts['input_tokens']==240 and counts['optimizer_updates']==3
    assert bool(counts['latent_pairs'])==('N' in arm)
    altered=deepcopy(spec);altered['declaration']['evaluation']['physical_batch_by_arm'][arm]=2
    other=cli.construct_identity(altered,spec['recipe'],model,optimizer,{'device':'cpu'},
        {'test':True},{'fixture':'0'*64})[0]
    assert other['sha256']!=identity['sha256']
    assert other['payload']['model_contract']==identity['payload']['model_contract']


@pytest.mark.parametrize('change',['pin','seed','old_schema','extra','arm','native_resolved',
    'stop_limit','suite_pin','index_pin','confirmation','no_eval_batch'])
def test_preflight_rejects_incompatible_authorities_without_cuda(declaration,change,monkeypatch):
    value,path=declaration
    if change=='seed':value['seed']+=1
    elif change=='old_schema':value['schema']='olmo-campaign-evaluation-tiny-acceptance-v1'
    elif change=='extra':value['extra']=True
    elif change=='arm':value['arm']='B'
    elif change=='suite_pin':value['data']['suite_manifest_sha256']='0'*64
    elif change=='index_pin':value['data']['index_manifest_sha256']='0'*64
    elif change=='confirmation':value['evaluation']['panels'][0]['name']='confirmation-main'
    elif change=='no_eval_batch':value['evaluation']['physical_batch_by_arm']={}
    path.write_text(json.dumps(value));args=request(path)
    if change=='pin':args.declaration_sha256='0'*64
    elif change=='native_resolved':args.resolved=path;args.resolved_sha256=args.declaration_sha256
    elif change=='stop_limit':args.stop_after=4
    def forbidden(*a,**kw):raise AssertionError('CPU preflight must not initialize CUDA or construct model')
    monkeypatch.setattr(cli,'construct',forbidden);monkeypatch.setattr(cli,'configure_cuda_runtime',forbidden)
    with pytest.raises(ValueError):cli.load_spec(args)


def test_deferred_reference_has_identical_training_plan(declaration):
    value,path=declaration;evaluated=cli.load_spec(request(path))
    value['evaluation']={'kind':'deferred','reason':'same training no-evaluation reference'}
    path.write_text(json.dumps(value));args=request(path);args.stop_after=3
    result=cli.load_spec(args)
    assert result['plan']==evaluated['plan']
    assert result['evaluation_plan']=={'declaration':value['evaluation'],'execution':'not_run'}


@pytest.mark.parametrize('change',['fp32','eager','world','evaluation'])
def test_native_execution_policy_cannot_silently_fallback(change):
    from scripts import olmo_campaign_manifest as legacy
    manifest={'execution':{**legacy.EXECUTION_COMMON,**legacy.PATHS['bf16_mixed'],
        'precision':'bf16_mixed','graph_mode':'prepared_cuda_graph'},'partition':{'world_size':2},
        'evaluation':{'kind':'ordered_named_dev_panels_v1'}}
    cli.require_execution_policy(manifest)
    if change=='fp32':manifest['execution']['precision']='fp32'
    elif change=='eager':manifest['execution']['graph_mode']='prepared_eager'
    elif change=='world':manifest['partition']['world_size']=1
    else:manifest['evaluation']['kind']='finite_pass_teacher_forced'
    with pytest.raises(ValueError):cli.require_execution_policy(manifest)


def test_native_constructor_reuses_accepted_constructor(monkeypatch):
    from scripts import olmo_campaign_ssd_execute as accepted
    marker=object();spec={'kind':'native','recipe':object(),'manifest':{'execution':{}}}
    monkeypatch.setattr(accepted,'construct',lambda actual,device:(marker,actual,device))
    assert cli.construct(spec,'cpu')==(marker,spec,'cpu')
