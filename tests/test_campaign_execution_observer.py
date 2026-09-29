import copy
from dataclasses import asdict
from datetime import timedelta
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist

from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
from cdrm.pretrained.campaign_recipe import (ARMS, CampaignRecipe, CampaignTokenSchedule,
    build_campaign_adamw, build_campaign_model, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.distributed_checkpoint import _local_rng
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_campaign_execution_observer as subject
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(scope='module')
def gloo(tmp_path_factory):
    assert not dist.is_initialized()
    path=tmp_path_factory.mktemp('observer-gloo')/'rendezvous'
    dist.init_process_group('gloo',rank=0,world_size=1,init_method=f'file://{path}',timeout=timedelta(seconds=120))
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def fixed_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(37)


def batches(update):
    result=[]
    for index,lengths in enumerate(((6,4),(5,0))):
        ids=(torch.arange(12).reshape(2,6)+3+update+index)%31
        valid=torch.arange(6)[None]<torch.tensor(lengths)[:,None]
        docs=torch.arange(2)[:,None].expand_as(ids).clone()
        docs[0,3:]=2;docs.masked_fill_(~valid,-1)
        ce,latent,kl=valid.clone(),valid.clone(),valid.clone()
        ce[0,2]=False;latent[1,1]=False;kl[0,2]=False
        result.append(NextLatBatch(ids,valid,docs,ce,latent,kl))
    return result


def signature(model,optimizer,scheduler,counters,generators):
    return tree_digests({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'counters':asdict(counters),
        'gradients':{name:p.grad for name,p in model.named_parameters()},
        'rng':_local_rng(torch.device('cpu'),generators),
        'modes':{name:module.training for name,module in model.named_modules()}})


def addresses(model):
    return {name:(id(p),p.data_ptr(),None if p.grad is None else p.grad.data_ptr()) for name,p in model.named_parameters()}


@pytest.mark.parametrize('arm',ARMS)
def test_all_eight_actual_campaign_arms_observation_cannot_change_updates(gloo,arm):
    recipe=CampaignRecipe(arm,sequence_length=6,rt_layers=(0,1),warmup_tokens=30,max_grad_norm=.2,
                          document_policy='continuous-stream-v1')
    native=OLMoTiledRTForCausalLM(OLMoConfig.tiny(),attention_backend='math',attention_precision='fp32')
    initial=build_campaign_model(native,recipe)
    initial.backbone.fusion.training=False  # Preserve a deliberately heterogeneous mode.
    data=[batches(i) for i in range(2)]
    targets=[sum(int(b.valid_mask.sum()) for b in update) for update in data]
    results={}
    for mode in ('unobserved','lean','acceptance'):
        model=copy.deepcopy(initial)
        optimizer=build_campaign_adamw(model,recipe,fused=False)
        schedule=CampaignTokenSchedule(optimizer,targets,warmup_tokens=recipe.warmup_tokens)
        counters=TrainingCounters()
        generators={'data':torch.Generator().manual_seed(29)}
        random.seed(11);np.random.seed(17);torch.manual_seed(23)
        noise=[[feedback_noise_for_rows(recipe,[f'{slot}-a',f'{slot}-b'],logical_update=i,
            sequence_length=6,width=model.config.model_dim) for slot in range(2)] for i in range(2)]
        count=sum_objective_counts([model.counts(b) for b in data[0]])
        if 'N' not in arm:assert count['latent']==count['kl']==0
        else:assert 0<count['kl']<count['latent']<count['ce']
        adapter=CampaignObjective(model,data[0][0],mode=recipe.mode(),global_counts=count,
            feedback_noise=noise[0][0],config=LMTrainingConfig(precision='fp32',max_grad_norm=recipe.max_grad_norm))
        runner=CampaignDDPGraphTraining(adapter,bucket_cap_mb=.02)
        runner.prepare(warmup=11)
        observer=None if mode=='unobserved' else subject.ExecutionObserver(mode,model=model,optimizer=optimizer,
            scheduler=schedule,device='cpu',generators=generators,max_grad_norm=recipe.max_grad_norm)
        rows=[]
        for update in range(2):
            result=runner.backward(data[update],feedback_noises=noise[update],replay=False)
            before=signature(model,optimizer,schedule,counters,generators);pointers=addresses(model)
            raw=None if observer is None else observer.after_backward()
            assert signature(model,optimizer,schedule,counters,generators)==before
            assert addresses(model)==pointers
            metrics=runner.step(result,optimizer,scheduler=schedule,counters=counters)
            if observer:
                before=signature(model,optimizer,schedule,counters,generators);pointers=addresses(model)
                row=observer.after_update(metrics,counters=counters,cursor={'next_update':update+1},
                    input_record=tree_digests({'batches':[vars(b) for b in data[update]]}) if mode=='acceptance' else None,
                    raw_gradients=raw)
                assert signature(model,optimizer,schedule,counters,generators)==before
                assert addresses(model)==pointers
                assert row['metrics']==metrics and row['cursor']=={'next_update':update+1}
                assert row['loss_means']['ce']==metrics['loss_sums']['ce']/metrics['counts']['ce']
                if mode=='acceptance':
                    expected=copy.deepcopy(row);expected['observation_seconds']={'different_timing':123.}
                    observer.assert_reference(row,expected)
                    expected['boundary']['state']['counters']['input_tokens']+=1
                    with pytest.raises(ValueError,match='exact reference'):observer.assert_reference(row,expected)
                else:assert 'boundary' not in row and 'raw_gradients' not in row
            rows.append(copy.deepcopy(metrics))
        results[mode]=(rows,signature(model,optimizer,schedule,counters,generators))
    assert results['lean']==results['acceptance']==results['unobserved']


def telemetry():
    counters=TrainingCounters(optimizer_updates=1,microbatches=2,documents=3,input_tokens=15,
        ce_positions=10,latent_pairs=8,kl_triples=4)
    metrics={'loss_sums':{'ce':20.,'latent':4.,'kl':1.},'counts':{'ce':10,'latent':8,'kl':4},
        'objective':2.75,'microbatches':2,'documents':3,'input_tokens':15,
        'gradient_norm_before_clip':2.,'lr_used':[.001],'lr_next':[.001],'counters':asdict(counters)}
    return metrics,counters


def small_observer(mode='lean'):
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.1))
    model[1].eval()
    optimizer=torch.optim.AdamW(model.parameters())
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10)
    generator=torch.Generator().manual_seed(9)
    observer=subject.ExecutionObserver(mode,model=model,optimizer=optimizer,scheduler=scheduler,
        device='cpu',generators={'data':generator},max_grad_norm=1.)
    return observer,model,optimizer,scheduler,generator


def test_lean_forbids_all_tensor_scans_and_uses_returned_norm(monkeypatch):
    observer,model,optimizer,scheduler,generator=small_observer()
    metrics,counters=telemetry()
    def forbidden(*args,**kwargs):raise AssertionError('Expensive observation path called in lean')
    for name in ('tree_digests','boundary','_local_rng','_restore_local_rng'):
        monkeypatch.setattr(subject,name,forbidden)
    for obj,name in ((model,'named_parameters'),(model,'parameters'),(model,'state_dict'),
                     (model,'modules'),(optimizer,'state_dict'),(scheduler,'state_dict')):
        monkeypatch.setattr(obj,name,forbidden)
    assert observer.after_backward() is None
    row=observer.after_update(metrics,counters=counters,cursor={'next_update':1})
    assert row['clipping']['norm_exceeds_limit']
    assert row['clipping']['coefficient_estimate']==1/(2+1e-6)
    metrics['counts']['ce']=999
    assert row['metrics']['counts']['ce']==10


@pytest.mark.parametrize('mutation',['nan','bad_count','bool_count','counters','lr','tensor','norm'])
def test_invalid_scalar_telemetry_is_rejected_without_scanning(mutation):
    metrics,counters=telemetry()
    if mutation=='nan':metrics['loss_sums']['kl']=float('nan')
    if mutation=='bad_count':metrics['counts']['ce']=16
    if mutation=='bool_count':metrics['counts']['ce']=True
    if mutation=='counters':counters.input_tokens+=1
    if mutation=='lr':metrics['lr_used']=[-.1]
    if mutation=='tensor':metrics['loss_sums']['ce']=torch.tensor(20.)
    if mutation=='norm':metrics['gradient_norm_before_clip']=-1.
    with pytest.raises((ValueError,TypeError)):subject.validate_metrics(metrics,counters)


def test_acceptance_restores_rng_named_generator_and_modes_on_hash_failure(monkeypatch):
    observer,model,optimizer,scheduler,generator=small_observer('acceptance')
    before=tree_digests(_local_rng(torch.device('cpu'),{'data':generator}))
    modes={name:m.training for name,m in model.named_modules()}
    def fail(*args,**kwargs):
        random.random();np.random.rand();torch.rand(3);torch.rand(3,generator=generator)
        model.eval()
        raise RuntimeError('injected read-only observer failure')
    monkeypatch.setattr(subject,'tree_digests',fail)
    with pytest.raises(RuntimeError):observer.after_backward()
    assert tree_digests(_local_rng(torch.device('cpu'),{'data':generator}))==before
    assert {name:m.training for name,m in model.named_modules()}==modes


def test_observation_contract_requires_explicit_scope_and_step_order():
    observer,model,optimizer,scheduler,_=small_observer()
    metrics,counters=telemetry()
    with pytest.raises(ValueError,match='after_backward'):observer.after_update(metrics,counters=counters,cursor={})
    observer.after_backward()
    with pytest.raises(ValueError,match='Lean'):observer.after_update(metrics,counters=counters,cursor={},raw_gradients={'unexpected':'hash'})
    with pytest.raises(ValueError,match='acceptance'):observer.assert_reference({'mode':'lean'},{'mode':'lean'})
    with pytest.raises(ValueError,match='explicitly'):
        subject.ExecutionObserver('automatic',model=model,optimizer=optimizer,scheduler=scheduler,device='cpu')


def test_observer_uses_new_counter_object_after_restore():
    observer,*_=small_observer()
    metrics,counters=telemetry()
    restored=TrainingCounters(**{**asdict(counters),'optimizer_updates':7,'input_tokens':100})
    metrics['counters']=asdict(restored)
    observer.after_backward()
    row=observer.after_update(metrics,counters=restored,cursor={'next_update':7})
    assert row['metrics']['counters']['optimizer_updates']==7
    assert row['metrics']['counters']['input_tokens']==100
    assert counters.optimizer_updates==1  # Earlier object is neither held nor mutated.


def test_incomplete_acceptance_dicts_cannot_qualify_exact_reference():
    observer,*_=small_observer('acceptance')
    incomplete={'schema':subject.SCHEMA,'mode':'acceptance'}
    with pytest.raises(ValueError,match='complete acceptance evidence'):
        observer.assert_reference(incomplete,incomplete)
