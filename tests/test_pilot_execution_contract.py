"""Ordered authority, finite planning and fresh execution lineage CPU acceptance."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from scripts import olmo_campaign_manifest as legacy
from scripts import olmo_pilot_execution_contract as contract
from scripts import olmo_pilot_ordered_data as ordered
from scripts import olmo_pilot_data_plan as data_plan
from test_pilot_ordered_data import fixture as ordered_fixture, cpu_tokenizer
from test_campaign_manifest import dictionary as old_dictionary


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True)+'\n')
    return legacy.sha256_file(path)


def data_spec(f):
    inventory = f.output.parent/'inventory.txt'; inventory.write_bytes(f.args['inventory_bytes'])
    excluded = f.output.parent/'exclusions.txt'; excluded.write_bytes(data_plan.exclusion_bytes([f.excluded]))
    result = {'corpus':str(f.corpus),'suite':str(f.output),'index':str(f.output/'panels/train'),
        'source_plan':str(f.args['acquisition_plan_path']), 'source_authorities':str(f.args['source_authorities_path']),
        'inventory':str(inventory),'exclusions':str(excluded),'split':'train','policy':copy.deepcopy(ordered.POLICY)}
    for key in contract.DATA_PATHS:
        target=Path(result[key]); suffix='_manifest_sha256' if key in ('corpus','suite','index') else '_sha256'
        result[key+suffix]=legacy.sha256_file(target/'manifest.json' if target.is_dir() else target)
    return result


@pytest.fixture
def bound(tmp_path):
    f=ordered_fixture(tmp_path,length=1024)
    return f,data_spec(f)


def declaration(data):
    manifest=old_dictionary();manifest['schema']=contract.PLANNING_SCHEMA
    manifest['data']=copy.deepcopy(data)
    manifest['budget'].update(updates=2,target_valid_tokens_per_update=2049)
    manifest['recipe']['effective_valid_tokens']=2049
    manifest['partition']['physical_batch_per_rank']={a:1 if a=='B' else 2 for a in legacy.ARMS}
    sources={'scripts/olmo_pilot_execution_contract.py':legacy.sha256_file(contract.ROOT/'scripts/olmo_pilot_execution_contract.py')}
    manifest['implementation_sources']=sources
    return {'schema':contract.SCHEMA,'planning_manifest':manifest,'startup':copy.deepcopy(legacy.STARTUP),
        'implementation_sources':sources}


@pytest.fixture
def prepared(bound,tmp_path,monkeypatch):
    f,data=bound;d=declaration(data);m=d['planning_manifest'];sources=d['implementation_sources']
    monkeypatch.setattr(contract,'source_hashes',lambda:dict(sources))
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    authority={'checkpoint':{'sha256':legacy.CHECKPOINT_SHA256},
        'artifacts':{'native/tokenizer.json':{'sha256':f.manifest['tokenizer']['sha256']}},
        'tokenizer':{'vocab_size':50280,'eos_token_id':50279,'pad_token_id':1}}
    m['model'].update(artifacts=str(artifacts),manifest_sha256=write(artifacts/contract.MANIFEST_FILENAME,authority))
    # Only immutable multi-GB model authority is represented by tiny metadata.
    # All data/schema, hash, planner, resource estimate and identity code is real.
    monkeypatch.setattr(contract,'validate_prepared_manifest',lambda _:authority)
    cards=[]
    for arm in legacy.ARMS:
        recipe=CampaignRecipe(arm,**m['recipe']);cfg=legacy.OLMoConfig.native_1b()
        architecture=legacy.architecture_parameter_counts(cfg,fbt=recipe.feedback,
            nextlat=legacy.nextlat_config(recipe) if recipe.nextlat else None)
        cards.append({'arm':arm,'parameters':{'architecture':architecture,'checks':{'matched':True},
            'observed_inventory':{'trainable':architecture['training_architecture']},'groups':{}}})
    ledger=tmp_path/'ledger.json';digest=write(ledger,{'status':'complete','integrity':{'ok':True},'sources':sources,'cards':cards})
    monkeypatch.setattr(legacy,'LEDGER_SHA',digest);m['resource_ledger']={'path':str(ledger),'sha256':digest}
    return f,d


def identity(result,arm='NFR'):
    return contract.execution_identity(result,arm,runtime={'kind':'CPU test metadata'},
        determinism={'enabled':True},model_contract={'native':True})


def checkpoint(ident,completed):
    # The old fixture helper is literal metadata only; it calls its old identity
    # validator, so build with our independent new schema/counters directly.
    payload=ident['payload'];plan=payload['plan'];world=payload['partition']['world_size']
    cursor=plan['first_cursor'] if completed==0 else plan['updates'][completed-1]['next_cursor']
    return {'schema':'olmo-replicated-ddp-checkpoint-v1','world_size':world,
        'metadata':{'world_size':world,'configuration':{'execution_identity':ident},
            'source_fingerprint':{'execution_identity_sha256':ident['sha256']}},
        'state':{'filename':'state.pt','size_bytes':999,'sha256':'a'*64},
        'counters':contract.expected_counters(ident,completed),
        'rank_cursors':[{'schema':contract.CURSOR_SCHEMA,'rank':r,'world_size':world,
            'physical_batch_per_rank':payload['partition']['physical_batch_per_rank'],'cursor':cursor}
            for r in range(world)]}


def test_authentication_and_complete_resolution_do_not_construct_models_or_read_token_tensors(prepared,monkeypatch):
    f,d=prepared;before=copy.deepcopy(d)
    def forbidden(*a,**kw):raise AssertionError('Unexpected model, tensor, optimizer or CUDA construction')
    with monkeypatch.context() as guard:
        guard.setattr(torch.nn.Module,'__init__',forbidden);guard.setattr(torch.optim.AdamW,'__init__',forbidden)
        guard.setattr(torch,'load',forbidden);guard.setattr(torch.cuda,'init',forbidden)
        guard.setattr(ordered.OrderedCampaignData,'_token_slice',forbidden)
        result=contract.resolve(d)
    assert d==before and result['schema']==contract.RESOLVED_SCHEMA
    assert result['launch_authorized'] is result['numerical_clearance'] is False
    p=result['planning'];assert p['plan']['valid_token_prefix']==[0,3072,6144]
    assert p['plan']['totals']['valid_tokens']==6144 and all(p['checks'].values())
    assert p['data_identity']['suite_identity_sha256']==f.manifest['identity_sha256']
    assert p['data_identity']['policy']==ordered.POLICY
    for arm in legacy.ARMS:
        start=contract.startup_plan(result,arm)
        assert start['data_origin']==p['plan']['first_cursor']
        assert start['restore_optimizer'] is start['restore_rng'] is False
        assert start['target_recipe']['arm']==arm
    assert p['resource_cards']['NFR']['dummy_rows']==2


@pytest.mark.parametrize('key',contract.DATA_PINS)
def test_independent_data_pins_reject_changed_authority(bound,key):
    _,data=bound;data[key]='f'*64
    with pytest.raises(ValueError):contract.authenticate_data(data,1024)


@pytest.mark.parametrize('kind',['catalog','recipe','round','selected_index','source_mapping','exclusions','old_policy','old_schema'])
def test_resealed_metadata_cannot_swap_round_order_source_or_recipe(bound,kind):
    f,data=bound
    if kind=='catalog':
        with (f.output/'catalog.sqlite').open('ab') as handle:handle.write(b'changed')
    elif kind=='selected_index':
        data['index']=str(f.output/'panels/dev-main');data['index_manifest_sha256']=legacy.sha256_file(Path(data['index'])/'manifest.json')
    elif kind=='source_mapping':
        value=json.loads(Path(data['source_authorities']).read_text());value[next(iter(value))]['upstream_source']['etag']='changed'
        data['source_authorities_sha256']=write(Path(data['source_authorities']),value)
    elif kind=='exclusions':
        Path(data['exclusions']).write_text('f'*64+'\n');data['exclusions_sha256']=legacy.sha256_file(Path(data['exclusions']))
    elif kind=='old_policy':data['policy']=legacy.POLICY
    else:
        suite=json.loads((f.output/'manifest.json').read_text())
        if kind=='recipe':suite['recipe']['ordering']['document_namespace']+='changed'
        elif kind=='round':suite['round_id']=1
        else:suite['schema']='olmo-packed-campaign-index-v1'
        suite['identity_sha256']=ordered._digest({k:v for k,v in suite.items() if k!='identity_sha256'})
        data['suite_manifest_sha256']=write(f.output/'manifest.json',suite)
    with pytest.raises(ValueError):contract.authenticate_data(data,1024)


@pytest.mark.parametrize('kind',['old_declaration','old_plan','source','startup','one_rank','precision','cycle','unbounded','token_budget','missing_inventory'])
def test_structural_declarations_fail_before_file_reads(bound,kind):
    _,data=bound;d=declaration(data);m=d['planning_manifest']
    if kind=='old_declaration':d['schema']=contract.accepted.SCHEMA
    elif kind=='old_plan':m['schema']=legacy.SCHEMA
    elif kind=='source':m['implementation_sources']={}
    elif kind=='startup':d['startup']={'kind':'adapted-full-NFR-with-Adam'}
    elif kind=='one_rank':m['partition']['world_size']=1
    elif kind=='precision':m['execution']['rt_attention_precision']='fp32'
    elif kind=='cycle':m['data']['policy']['cycling']=True
    elif kind=='unbounded':m['budget']['updates']=legacy.MAX_UPDATES+1
    elif kind=='token_budget':m['budget']['target_valid_tokens_per_update']+=1
    else:m['data'].pop('inventory')
    with pytest.raises(ValueError):contract.validate_declaration(d,d['implementation_sources'])


def test_finite_plan_and_new_resume_identity_reject_repartition_and_old_cursor(prepared):
    f,d=prepared;result=contract.resolve(d);ident=identity(result)
    saved=checkpoint(ident,1);status=contract.validate_resume_metadata(saved,ident)
    assert status['remaining_updates']==1 and status['completed_reference_report_required'] is False
    assert contract.expected_counters(ident,2)['input_tokens']==6144
    for key,value in (('schema','unknown-cursor-schema'),('physical_batch_per_rank',99)):
        wrong=copy.deepcopy(saved);wrong['rank_cursors'][0][key]=value
        with pytest.raises(ValueError):contract.validate_resume_metadata(wrong,ident)
    wrong=copy.deepcopy(saved);wrong['counters']['ce_positions']+=1
    with pytest.raises(ValueError):contract.validate_resume_metadata(wrong,ident)
    old=copy.deepcopy(ident);old['schema']=contract.accepted.IDENTITY_SCHEMA
    with pytest.raises(ValueError):contract.validate_identity(old)
    with ordered.OrderedCampaignData(f.corpus,f.output/'panels/train') as reader:
        with pytest.raises(ValueError,match='Insufficient corpus'):
            contract.plan_updates(reader,{'updates':2,'target_valid_tokens_per_update':reader.total_tokens},{'NFR':(2,2)})


def test_new_lineage_identity_binds_data_order_and_partition(prepared):
    _,d=prepared;first=identity(contract.resolve(d))
    m=d['planning_manifest'];m['partition']['physical_batch_per_rank']['NFR']=3
    second=identity(contract.resolve(d));assert first['sha256']!=second['sha256']
    with pytest.raises(ValueError):contract.validate_resume_metadata(checkpoint(first,1),second)


def test_pinned_declaration_resolution_roundtrip_and_reject_resolved_edit(prepared,tmp_path):
    _,d=prepared;result=contract.resolve(d)
    path=tmp_path/'declaration.json';pin=write(path,d)
    resolved=tmp_path/'resolved.json';rpin=write(resolved,result)
    assert contract.load_declaration(path,pin,resolved,rpin)==(d,result)
    changed=copy.deepcopy(result);changed['planning']['plan']['totals']['ce_targets']+=1
    rpin=write(resolved,changed)
    with pytest.raises(ValueError):contract.load_declaration(path,pin,resolved,rpin)


def test_fusion_startup_allowlist_keeps_prior_exposure_separate(bound):
    _,data=bound;d=declaration(data);m=d['planning_manifest']
    m['arms']=['NF','NFR'];m['partition']['physical_batch_per_rank']={'NF':2,'NFR':2}
    d['startup']={**copy.deepcopy(contract.ADAPTED_POLICY),
        'checkpoint':{'path':'unused-fusion','sha256':contract.accepted.FUSION_CHECKPOINT_SHA},
        'report':{'path':'unused-report','sha256':contract.accepted.FUSION_REPORT_SHA}}
    assert set(contract.validate_declaration(d,d['implementation_sources']))=={'NF','NFR'}
    assert d['startup']['prior_exposure']['input_tokens']==1073565
    assert d['startup']['data_origin']=='ordered_round0_prefix_zero'
    d['startup']['data_origin']='packed_prefix_zero'
    with pytest.raises(ValueError):contract.validate_declaration(d,d['implementation_sources'])


def test_fusion128_resolves_real_metadata_authority_and_validates_weights_only_receipt(prepared,tmp_path,monkeypatch):
    from test_campaign_execution_contract import authority_fixture
    _,d=prepared;m=d['planning_manifest']
    m['arms']=['NF','NFR'];m['partition']['physical_batch_per_rank']={'NF':2,'NFR':2}
    d['startup'],historical=authority_fixture(tmp_path,monkeypatch,d['implementation_sources'])
    d['startup']['data_origin']='ordered_round0_prefix_zero'
    result=contract.resolve(d)
    assert result['fusion_authority']['prior_exposure']==contract.EXPOSURE
    for arm in ('NF','NFR'):
        plan=contract.startup_plan(result,arm)
        assert plan['target_recipe']['document_policy']=='continuous-stream-v1'
        assert contract.recipe_from_dict(plan['import_recipe']).document_policy=='isolated-v1'
        assert plan['transition']['kind']=='weights-preserving-isolated-NF-to-declared-ordered-arm-v1'
        receipt={'checkpoint_sha256':d['startup']['checkpoint']['sha256'],
            'configuration':historical['configuration'],'source_fingerprint':historical['source_fingerprint'],
            'counters':historical['counters'],'data_cursor':historical['data_cursor'],
            'fusion_state_pins':historical['checkpoints'][0]['boundary_digests']['fusion'],
            'restore_scope':'fusion weights only; diagnostic flags and RNG preserved',
            'checks':dict.fromkeys(('complete_fusion_exact','parameter_identities_preserved',
                'trainability_preserved','module_modes_preserved','frozen_state_exact','tied_readout_preserved'),True)}
        contract.validate_import_receipt(receipt,plan)
        receipt['restore_scope']='complete trained NFR backbone and Adam'
        with pytest.raises(ValueError):contract.validate_import_receipt(receipt,plan)
    ident=identity(result)
    assert contract.expected_counters(ident,0)['input_tokens']==0
    assert ident['payload']['startup']['expected_import']['prior_exposure']['input_tokens']==1073565
