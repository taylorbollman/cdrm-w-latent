"""Versioned evaluation CLI authorities and actual CPU constructor contracts."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import ARMS, build_campaign_adamw
from cdrm.pretrained.distributed_checkpoint import _metadata
from scripts import olmo_campaign_eval_execute as cli
from scripts import olmo_campaign_execution_contract as contract
from scripts import olmo_campaign_manifest as legacy
from test_campaign_eval_control import evaluation_paths, policy


@pytest.fixture
def declaration(evaluation_paths,tmp_path):
    corpus,indices=evaluation_paths
    value={'schema':cli.TINY_SCHEMA,'arm':'NFR','corpus':str(corpus),
        'corpus_manifest_sha256':sha256_file(corpus/'manifest.json'),'index':str(indices['train']),
        'index_manifest_sha256':sha256_file(indices['train']/'manifest.json'),
        'storage_prefix':'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T140000Z/fixture',
        'seed':20260929,'evaluation':policy(indices['dev'])}
    path=tmp_path/'declaration.json';path.write_text(json.dumps(value))
    return value,path


def args(path,arm='NFR'):
    return SimpleNamespace(declaration=path,declaration_sha256=sha256_file(path),resolved=None,
        resolved_sha256=None,arm=arm,stop_after=None)


@pytest.mark.parametrize('arm',ARMS)
def test_actual_all_arm_identity_binds_evaluation_and_retains_training_ownership(declaration,arm):
    torch.set_num_threads(1)
    value,path=declaration;value['arm']=arm;path.write_text(json.dumps(value))
    spec=cli.load_spec(args(path,arm))
    assert [r['counts']['valid_tokens'] for r in spec['plan']['updates']]==[80]*3
    assert spec['evaluation_plan']['scheduled_updates']==[1,2,3]
    assert spec['evaluation_plan']['fixed_plan']['updates'][0]['counts']['valid_tokens']==48
    model,source,imported=cli.construct(spec,torch.device('cpu'))
    optimizer=build_campaign_adamw(model,spec['recipe'],fused=False)
    identity,ownership=cli.construct_identity(spec,spec['recipe'],model,optimizer,{'device':'cpu'},
        {'test':True},{'fixture':'0'*64})
    assert source['kind']=='deterministic_random_tiny' and not imported and not optimizer.state
    assert ownership['arm']==arm and ownership['tied_readout']
    assert identity['payload']['declaration']['evaluation']==value['evaluation']
    assert identity['payload']['scope']=='tiny-acceptance-not-native'
    counts=contract.expected_counters(identity,3)
    assert counts['input_tokens']==240 and counts['optimizer_updates']==3
    assert bool(counts['latent_pairs'])==('N' in arm)
    assert bool(counts['kl_triples'])==('N' in arm)
    fingerprint=cli.source_fingerprint(identity,source,{'fixture':'0'*64})
    metadata=_metadata(model,optimizer,None,{'execution_identity':identity},fingerprint,2)
    assert metadata['source_fingerprint']['sha256']==identity['sha256']
    # Only the held-out schedule changes here; it must still break resume identity.
    altered=deepcopy(spec);altered['declaration']['evaluation']['every_updates']=2
    other=cli.construct_identity(altered,spec['recipe'],model,optimizer,{'device':'cpu'},
        {'test':True},{'fixture':'0'*64})[0]
    assert other['sha256']!=identity['sha256']
    assert other['payload']['model_contract']==identity['payload']['model_contract']


@pytest.mark.parametrize('change',['pin','seed','schema','extra','arm','native_resolved','limit',
    'missing_evaluation','dev_pin','precision','interval','train_as_dev'])
def test_tiny_declaration_rejects_incompatible_authority_before_model_creation(declaration,change,monkeypatch):
    value,path=declaration
    if change=='seed':value['seed']+=1
    elif change=='schema':value['schema']='olmo-campaign-tiny-acceptance-v1'
    elif change=='extra':value['extra']=True
    elif change=='arm':value['arm']='B'
    elif change=='missing_evaluation':del value['evaluation']
    elif change=='dev_pin':value['evaluation']['index_manifest_sha256']='0'*64
    elif change=='precision':value['evaluation']['precision']='bf16_mixed'
    elif change=='interval':value['evaluation']['every_updates']=0
    elif change=='train_as_dev':
        value['evaluation']['index']=value['index']
        value['evaluation']['index_manifest_sha256']=value['index_manifest_sha256']
    path.write_text(json.dumps(value));request=args(path)
    if change=='pin':request.declaration_sha256='0'*64
    elif change=='native_resolved':request.resolved=path;request.resolved_sha256=request.declaration_sha256
    elif change=='limit':request.stop_after=4
    def forbidden(*a,**kw):raise AssertionError('Preflight may not initialize CUDA or construct a model')
    monkeypatch.setattr(cli,'construct',forbidden)
    monkeypatch.setattr(cli,'configure_cuda_runtime',forbidden)
    with pytest.raises(ValueError):cli.load_spec(request)


def test_explicit_deferred_policy_retains_no_eval_reference_and_terminal_stop(declaration):
    value,path=declaration
    value['evaluation']={'kind':'deferred','reason':'matching no-evaluation reference'}
    path.write_text(json.dumps(value));request=args(path);request.stop_after=3
    result=cli.load_spec(request)
    assert result['evaluation_plan']=={'declaration':value['evaluation'],'execution':'not_run'}
    assert len(result['plan']['updates'])==3


@pytest.mark.parametrize('change',['fp32','eager','world','evaluation'])
def test_native_requires_declared_supported_path_with_evaluation(change):
    manifest={'execution':{**legacy.EXECUTION_COMMON,**legacy.PATHS['bf16_mixed'],
        'precision':'bf16_mixed','graph_mode':'prepared_cuda_graph'},
        'partition':{'world_size':2},'evaluation':{'kind':'finite_pass_teacher_forced'}}
    cli.require_execution_policy(manifest)
    if change=='fp32':manifest['execution']['precision']='fp32'
    elif change=='eager':manifest['execution']['graph_mode']='eager'
    elif change=='world':manifest['partition']['world_size']=1
    else:manifest['evaluation']['kind']='unimplemented'
    with pytest.raises(ValueError):cli.require_execution_policy(manifest)


def test_cli_requires_independent_resume_pin_fresh_output_and_absolute_stop(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    base=['--declaration',str(tmp_path/'d.json'),'--declaration-sha256','a'*64,
        '--output-dir',str(tmp_path/'new'),'--checkpoint-root',str(tmp_path/'checkpoints'),'--arm','B']
    parsed=cli.parse_args(base+['--stop-after','0'])
    assert parsed.stop_after==0 and parsed.observation=='lean'
    with pytest.raises(SystemExit):cli.parse_args(base+['--resume',str(tmp_path/'checkpoint')])
    with pytest.raises(SystemExit):cli.parse_args(base+['--resume-manifest-sha256','b'*64])
    with pytest.raises(SystemExit):cli.parse_args(base+['--stop-after','-1'])
    (tmp_path/'new').mkdir()
    with pytest.raises(SystemExit):cli.parse_args(base)
