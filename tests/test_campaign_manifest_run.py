"""Bounded B execution: declaration consumption and literal CPU math oracles."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.campaign_recipe import CampaignRecipe, CampaignTokenSchedule, build_campaign_adamw
from cdrm.pretrained.campaign_training import CampaignGraphTraining, CampaignObjective
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.packed_campaign_data import PackedCampaignData
from scripts import olmo_campaign_manifest_run as runner
from scripts import olmo_campaign_manifest as resolver
from scripts import olmo_campaign_base_loop as base_loop
from scripts.olmo_lm_common import tree_digests
from test_campaign_base_loop import actual_data
from test_campaign_manifest import dictionary


@pytest.fixture(autouse=True)
def fixed_math():
    torch.set_num_threads(1)
    with sdpa_kernel(SDPBackend.MATH):
        yield


def declaration():
    m = dictionary()
    m.update(arms=['B'], implementation_sources=resolver.source_hashes())
    m['recipe']['effective_valid_tokens']=16384
    m['budget'].update(updates=3, target_valid_tokens_per_update=16384)
    m['partition']['physical_batch_per_rank']={'B':8}
    m['retention'].update(checkpoint_every_updates=3, keep_local_completed=3,
        storage_prefix='gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/test-manifest')
    return m


def test_explicit_supported_declaration_and_extra_checkpoint1():
    m=declaration(); before=copy.deepcopy(m)
    recipe=runner.validate_execution(m)
    policy=runner.checkpoint_policy(m)
    assert m==before and recipe.arm=='B' and recipe.effective_valid_tokens==16384
    assert policy.max_updates==3 and policy.checkpoint_updates==(1,3)
    assert policy.checkpoint_seconds==m['retention']['checkpoint_seconds'] and not policy.save_initial
    m['retention']['checkpoint_seconds']=59
    assert runner.checkpoint_policy(m).checkpoint_seconds==59


@pytest.mark.parametrize('case', ['all_arms','alternate_batch','alternate_world','alternate_budget','fp32',
    'eager','eval','adapted','every1','local2','generic_prefix','source','cycle'])
def test_unsupported_declarations_rejected_before_model_or_cuda(case, monkeypatch):
    m=declaration()
    if case=='all_arms': m['arms']=['B','N'];m['partition']['physical_batch_per_rank']['N']=8
    elif case=='alternate_batch':m['partition']['physical_batch_per_rank']['B']=12
    elif case=='alternate_world':m['partition']['world_size']=4
    elif case=='alternate_budget':m['budget']['updates']=4
    elif case=='fp32':m['execution'].update(resolver.PATHS['fp32'],precision='fp32',graph_mode='prepared_eager')
    elif case=='eager':m['execution']['graph_mode']='prepared_eager'
    elif case=='eval':m['evaluation']={'kind':'finite_pass_teacher_forced','index':'unused',
        'index_manifest_sha256':'a'*64,'split':'dev','target_valid_tokens':1024,'every_updates':1,
        'precision':'fp32','feedback_jitter':0,'report_passes':'all_trained_passes','generation':'not_implemented'}
    elif case=='adapted':m['startup']['kind']='adapted-fusion'
    elif case=='every1':m['retention']['checkpoint_every_updates']=1
    elif case=='local2':m['retention']['keep_local_completed']=2
    elif case=='generic_prefix':m['retention']['storage_prefix']='gs://fast-chunks/cdrm-w-latent/campaign-drafts/test'
    elif case=='source':m['implementation_sources']['scripts/olmo_campaign_manifest.py']='f'*64
    elif case=='cycle':m['data']['policy']['cycling']=True
    def forbidden(*args,**kwargs):raise AssertionError('Attempted construction or CUDA')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    monkeypatch.setattr(torch.cuda,'init',forbidden)
    with pytest.raises(ValueError):runner.validate_execution(m)


def test_actual_plan_is_three_equal_updates_and_matches_resolution(actual_data):
    m=declaration()
    with PackedCampaignData(*actual_data) as data:
        plan=resolver.plan_updates(data,m['budget'],{'B':(2,8)})
        original=data.cursor()
        actual=runner.assert_declared_plan(data,m,{'plan':plan})
        assert data.cursor()==original and not data._token_fds
        assert [p.counts.valid_tokens for p in actual]==[16384]*3
        assert [len(p.rows) for p in actual]==[16]*3
        assert plan['valid_token_prefix']==[0,16384,32768,49152]
        expected=base_loop.expected_counters(actual,3,batch_size=8)
        assert asdict(expected)==dict(optimizer_updates=3,microbatches=6,documents=48,
            input_tokens=49152,ce_positions=49104,latent_pairs=0,kl_triples=0)
        changed=copy.deepcopy(plan);changed['updates'][1]['membership_sha256']='f'*64
        with pytest.raises(ValueError,match='pure data plan'):runner.assert_declared_plan(data,m,{'plan':changed})


def test_load_exact_pinned_resolution_before_cuda_and_detect_drift(tmp_path, monkeypatch):
    m=declaration();manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(m))
    observed=[]
    expected={'status':'cpu_plan_validated_not_authorized','launch_authorized':False,
              'numerical_clearance':False,'declarations':m,'plan':{'fixed':'fixture'}}
    def resolve(value):
        assert value==m and not torch.cuda.is_initialized()
        observed.append('resolve');return copy.deepcopy(expected)
    monkeypatch.setattr(resolver,'resolve',resolve)
    resolved=tmp_path/'resolved.json'
    payload={**copy.deepcopy(expected),'manifest_file_sha256':resolver.sha256_file(manifest)}
    resolved.write_text(json.dumps(payload))
    args=SimpleNamespace(manifest=manifest,manifest_sha256=resolver.sha256_file(manifest),
        resolved=resolved,resolved_sha256=resolver.sha256_file(resolved))
    value=runner.load_draft(args)
    assert observed==['resolve'] and value[0]==m and value[1]==payload and value[2].arm=='B'
    payload['plan']['fixed']='drift';resolved.write_text(json.dumps(payload));args.resolved_sha256=resolver.sha256_file(resolved)
    with pytest.raises(ValueError,match='independently re-resolved'):runner.load_draft(args)
    args.manifest_sha256='f'*64
    with pytest.raises(ValueError,match='Pinned JSON'):runner.load_draft(args)


def construct_tiny_declared(monkeypatch):
    config=OLMoConfig.tiny()
    torch.manual_seed(44)
    native=OLMoTiledRTForCausalLM(config,device='cpu',dtype=torch.float32)
    state={k:v.clone() for k,v in native.state_dict().items()}
    recipe=CampaignRecipe('B',sequence_length=8,rt_layers=(0,1),effective_valid_tokens=32,
        plateau_lr=7e-4,warmup_tokens=96,warmup_start_fraction=.2,weight_decay=.07,
        epsilon=3e-6,betas=(.8,.9),max_grad_norm=.8,document_policy='continuous-stream-v1')
    m=declaration();m['recipe']=json.loads(json.dumps(asdict(recipe)));m['recipe'].pop('arm')
    source={'sha256':'a'*64}
    resolved={'model_checkpoint_authority':source,'native_backbone_config':config.to_dict(),
        'nextlat_configs':{'B':resolver.nextlat_config(replace(recipe)).to_dict()},'recipes':{'B':recipe.to_dict()}}
    resolved['nextlat_configs']['B']['model_dim']=config.model_dim
    # NextLat hidden_dim is derived from model width; get the actual config here.
    from cdrm.pretrained.nextlat import NextLatConfig
    resolved['nextlat_configs']['B']=NextLatConfig(config.model_dim,seed=recipe.predictor_seed,
        vocab_chunk_size=128,ce_chunk_size=2048,document_policy=recipe.document_policy).to_dict()
    monkeypatch.setattr(runner,'OLMoConfig',SimpleNamespace(native_1b=lambda:config))
    monkeypatch.setattr(runner,'validate_prepared_manifest',lambda _: {'checkpoint':source})
    monkeypatch.setattr(runner,'load_native_state_dict',lambda _: {k:v.clone() for k,v in state.items()})
    model,observed=runner.construct_declared(m,resolved,recipe,torch.device('cpu'))
    assert observed==source
    return model,recipe,m,resolved


def test_constructor_consumes_recipe_runtime_and_leaves_frozen_globals(monkeypatch):
    globals_before=(base_loop.TARGETS,base_loop.construct,base_loop.run_stage)
    model,recipe,m,resolved=construct_tiny_declared(monkeypatch)
    base=model.backbone.backbone
    assert base.ordinary_activation_checkpointing and base.cast_weights_once and base.reuse_rope and base.kv_only_writes
    assert base.attention_backend=='sdpa' and base.attention_precision=='mixed'
    assert base.tile_backend==base.backward_tile_backend=='triton'
    optimizer=build_campaign_adamw(model,recipe,fused=False)
    assert all(group['betas']==(.8,.9) and group['eps']==3e-6 and group['lr']==7e-4 for group in optimizer.param_groups)
    assert {group['weight_decay'] for group in optimizer.param_groups}=={0,.07}
    contract=runner.model_contract(model,recipe,optimizer)
    inventory={'registered_unique':contract['resident_parameters'],'trainable':contract['trainable_parameters'],
               'optimizer_owned':contract['trainable_parameters']}
    card={'resource_cards':{'B':{'parameters':{'observed_inventory_from_retained_ledger':inventory}}}}
    runner.assert_declared_ownership(contract,card)
    card['resource_cards']['B']['parameters']['observed_inventory_from_retained_ledger']['trainable']+=1
    with pytest.raises(ValueError,match='ownership'):runner.assert_declared_ownership(contract,card)
    assert globals_before==(base_loop.TARGETS,base_loop.construct,base_loop.run_stage)


def test_actual_tiny_ce_update_schedule_and_fresh_state_next_update(monkeypatch):
    model,recipe,_,_=construct_tiny_declared(monkeypatch)
    oracle=copy.deepcopy(model)
    optimizer=build_campaign_adamw(model,recipe,fused=False)
    scheduler=CampaignTokenSchedule(optimizer,[32]*3,warmup_tokens=recipe.warmup_tokens,
        start_fraction=recipe.warmup_start_fraction)
    ids=torch.arange(32).reshape(4,8)%55+2;valid=torch.ones_like(ids,dtype=torch.bool)
    docs=torch.zeros_like(ids);docs[:,4:]=1
    batch=NextLatBatch(ids,valid,docs,valid.clone(),valid.clone(),valid.clone())
    adapter=CampaignObjective(model,batch,mode=recipe.mode(),global_counts={'ce':28,'latent':0,'kl':0},
        config=LMTrainingConfig(precision='fp32',max_grad_norm=recipe.max_grad_norm))
    local=CampaignGraphTraining(adapter)
    measured=local.backward((batch,),replay=False)
    output=oracle.backbone.backbone(batch.input_ids,attention_mask=valid,
        mode=recipe.mode().rt_mode,return_logits=True)
    loss=F.cross_entropy(output.logits[:,:-1].reshape(-1,output.logits.shape[-1]),ids[:,1:].reshape(-1))
    loss.backward()
    assert measured['objective']==pytest.approx(float(loss.detach()),rel=3e-7)
    for (name,p),(_,q) in zip(model.named_parameters(),oracle.named_parameters()):
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=1e-7,msg=name)
        else:assert p.grad is None and q.grad is None
    counters=TrainingCounters();dormant=tree_digests(model.backbone.fusion.state_dict())
    local.optimizer_step(optimizer,(batch,),scheduler=scheduler,counters=counters)
    state=copy.deepcopy(model.state_dict());adam=copy.deepcopy(optimizer.state_dict());schedule=copy.deepcopy(scheduler.state_dict())
    observed=local.optimizer_step(optimizer,(batch,),scheduler=scheduler,counters=counters)
    final=tree_digests({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()})
    fresh=copy.deepcopy(oracle);fresh.load_state_dict(state)
    opt2=build_campaign_adamw(fresh,recipe,fused=False);sch2=CampaignTokenSchedule(opt2,[32]*3,
        warmup_tokens=recipe.warmup_tokens,start_fraction=recipe.warmup_start_fraction)
    opt2.load_state_dict(adam);sch2.load_state_dict(schedule)
    fresh.zero_grad(set_to_none=True)
    second=CampaignGraphTraining(CampaignObjective(fresh,batch,mode=recipe.mode(),global_counts={'ce':28,'latent':0,'kl':0},
        config=LMTrainingConfig(precision='fp32',max_grad_norm=recipe.max_grad_norm)))
    ctr2=TrainingCounters(optimizer_updates=1,microbatches=1,documents=4,input_tokens=32,ce_positions=28)
    replay=second.optimizer_step(opt2,(batch,),scheduler=sch2,counters=ctr2)
    assert replay==observed and asdict(ctr2)==asdict(counters)
    assert tree_digests({'model':fresh.state_dict(),'optimizer':opt2.state_dict(),'scheduler':sch2.state_dict()})==final
    assert tree_digests(fresh.backbone.fusion.state_dict())==dormant
    assert scheduler.completed_tokens==64 and optimizer.param_groups[0]['lr']==pytest.approx(7e-4*(.2+.8*64/96))


def test_reference_authority_and_manifest_final_guards(tmp_path):
    source={'test':'a'*64};config={'explicit':'recipe'};checkpoint='b'*64
    report={'schema':runner.SCHEMA,'phase':'reference','status':'passed','sources':source,
        'configuration':config,'completed_segment':True,'updates':{'1':[], '2':[], '3':[]},
        'local_checkpoints':[{'optimizer_update':1,'receipt':{'manifest_sha256':checkpoint},'boundary_by_rank':[{},{}]}],
        'published_checkpoints':[{'manifest_sha256':checkpoint,'retention':{'download_sha256_verified':True}}]}
    path=tmp_path/'report.json'
    def load(value):
        path.write_text(json.dumps(value))
        return runner.load_reference(path,resolver.sha256_file(path),source,config,checkpoint)
    assert load(report)[1]==[{},{}]
    for field,value in [('schema',base_loop.SCHEMA),('phase','resume'),('completed_segment',False),
                        ('configuration',{}),('sources',{}),('status','running'),('published_checkpoints',[])]:
        changed=copy.deepcopy(report);changed[field]=value
        with pytest.raises(ValueError):load(changed)
    path.write_text('{}');resolved=tmp_path/'resolved.json';resolved.write_text('{}')
    args=SimpleNamespace(manifest=path,manifest_sha256=resolver.sha256_file(path),resolved=resolved,
        resolved_sha256=resolver.sha256_file(resolved),phase='reference')
    runner.assert_inputs(args)
    resolved.write_text('{"changed":true}')
    with pytest.raises(ValueError):runner.assert_inputs(args)
