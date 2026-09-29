"""Entrypoint guard and real tiny-constructor tests; no CUDA or transfers."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import ARMS, build_campaign_adamw
from cdrm.pretrained.distributed_checkpoint import _metadata
from scripts import olmo_campaign_execute as cli
from scripts import olmo_campaign_execution_contract as contract
from scripts import olmo_campaign_manifest as legacy
from test_campaign_execution_engine import packed_paths


@pytest.fixture
def declaration(packed_paths,tmp_path):
    corpus,index=packed_paths
    value={'schema':cli.TINY_SCHEMA,'arm':'NFR','corpus':str(corpus),
        'corpus_manifest_sha256':sha256_file(corpus/'manifest.json'),'index':str(index),
        'index_manifest_sha256':sha256_file(index/'manifest.json'),
        'storage_prefix':'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T140000Z/fixture',
        'seed':20260929}
    path=tmp_path/'declaration.json'
    path.write_text(json.dumps(value))
    return value,path


def args(path,arm='NFR',**kw):
    return SimpleNamespace(declaration=path,declaration_sha256=sha256_file(path),resolved=None,
        resolved_sha256=None,arm=arm,stop_after=None,**kw)


@pytest.mark.parametrize('arm',ARMS)
def test_actual_tiny_constructor_identity_counts_and_terminal_metadata(declaration,arm):
    torch.set_num_threads(1)
    value,path=declaration;value['arm']=arm;path.write_text(json.dumps(value))
    spec=cli.load_spec(args(path,arm))
    assert [r['counts']['valid_tokens'] for r in spec['plan']['updates']]==[80]*3
    model,source,imported=cli.construct(spec,torch.device('cpu'))
    optimizer=build_campaign_adamw(model,spec['recipe'],fused=False)
    identity,ownership=cli.construct_identity(spec,spec['recipe'],model,optimizer,{'device':'cpu'},
        {'test':True},{'fixture':'0'*64})
    assert source['kind']=='deterministic_random_tiny' and not imported
    base=model.backbone.backbone
    assert base.ordinary_activation_checkpointing is spec['manifest']['execution']['ordinary_activation_checkpointing']
    assert base.cast_weights_once is spec['manifest']['execution']['cast_weights_once']
    assert not optimizer.state and ownership['arm']==arm
    assert identity['payload']['scope']=='tiny-acceptance-not-native'
    assert identity['payload']['model_contract']['tied_readout']
    counts=contract.expected_counters(identity,3)
    assert counts['input_tokens']==240 and counts['optimizer_updates']==3
    assert counts['microbatches']==12
    assert bool(counts['latent_pairs'])==('N' in arm)
    assert bool(counts['kl_triples'])==('N' in arm)
    again=cli.construct_identity(spec,spec['recipe'],model,optimizer,{'device':'cpu'},
        {'test':True},{'fixture':'0'*64})[0]
    assert identity==again
    fingerprint=cli.source_fingerprint(identity,source,{'fixture':'0'*64})
    metadata=_metadata(model,optimizer,None,{'execution_identity':identity},fingerprint,2)
    assert metadata['source_fingerprint']['sha256']==identity['sha256']
    assert metadata['source_fingerprint']['execution_identity_sha256']==identity['sha256']


@pytest.mark.parametrize('change',['pin','seed','schema','extra','arm','native_resolved','limit'])
def test_tiny_declaration_rejects_incompatible_authority(declaration,change):
    value,path=declaration
    if change=='seed':value['seed']+=1
    elif change=='schema':value['schema']='guessed'
    elif change=='extra':value['extra']=True
    elif change=='arm':value['arm']='B'
    path.write_text(json.dumps(value));request=args(path)
    if change=='pin':request.declaration_sha256='0'*64
    elif change=='native_resolved':request.resolved=path;request.resolved_sha256=request.declaration_sha256
    elif change=='limit':request.stop_after=4
    with pytest.raises(ValueError):cli.load_spec(request)


@pytest.mark.parametrize('change',['fp32','eager','world','evaluation'])
def test_native_launcher_rejects_unimplemented_paths(change):
    manifest={'execution':{**legacy.EXECUTION_COMMON,**legacy.PATHS['bf16_mixed'],
        'precision':'bf16_mixed','graph_mode':'prepared_cuda_graph'},
        'partition':{'world_size':2},'evaluation':{'kind':'deferred'}}
    cli.require_execution_policy(manifest)
    if change=='fp32':manifest['execution']['precision']='fp32'
    elif change=='eager':manifest['execution']['graph_mode']='eager'
    elif change=='world':manifest['partition']['world_size']=1
    else:manifest['evaluation']['kind']='unimplemented'
    with pytest.raises(ValueError):cli.require_execution_policy(manifest)


def test_cli_requires_independent_resume_pin_and_new_output(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    base=['--declaration',str(tmp_path/'d.json'),'--declaration-sha256','a'*64,
        '--output-dir',str(tmp_path/'new'),'--checkpoint-root',str(tmp_path/'checkpoints'),'--arm','B']
    parsed=cli.parse_args(base+['--stop-after','0'])
    assert parsed.stop_after==0 and parsed.observation=='lean'
    with pytest.raises(SystemExit):cli.parse_args(base+['--resume',str(tmp_path/'checkpoint')])
    (tmp_path/'new').mkdir()
    with pytest.raises(SystemExit):cli.parse_args(base)
