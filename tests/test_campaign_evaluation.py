"""Literal CPU loss/mask oracles and live-state/error preservation; no CUDA claim."""
from copy import deepcopy
from dataclasses import asdict, replace
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cdrm.pretrained.campaign_recipe import (ARMS, CampaignRecipe, build_campaign_model,
    build_campaign_adamw, feedback_noise_for_rows)
from cdrm.pretrained.campaign_training import CampaignObjective, CampaignGraphTraining
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_campaign_evaluation as evaluation
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def setup(arm='NFR'):
    recipe = CampaignRecipe(arm, sequence_length=6, rt_layers=(0,1),
                            document_policy='continuous-stream-v1')
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend='sdpa',
        attention_precision='mixed', tile_backend='eager', backward_tile_backend='eager',
        ordinary_activation_checkpointing=True, cast_weights_once=True,
        reuse_rope=True, kv_only_writes=True, backward_memory='recompute')
    model = build_campaign_model(base, recipe).train()
    batch = NextLatBatch(
        torch.tensor([[3,60,5,6,7,8],[9,10,11,12,1,1],[1,1,1,1,1,1]]),
        torch.tensor([[1,1,1,1,1,1],[1,1,1,1,0,0],[0,0,0,0,0,0]],dtype=torch.bool),
        torch.tensor([[0,0,0,1,1,1],[2,2,2,2,-1,-1],[-1,-1,-1,-1,-1,-1]]),
        ce_mask=torch.tensor([[1,1,0,1,1,1],[1,1,1,1,0,0],[0,0,0,0,0,0]],dtype=torch.bool),
        latent_mask=torch.tensor([[1,1,1,1,0,1],[1,1,0,1,0,0],[0,0,0,0,0,0]],dtype=torch.bool),
        kl_mask=torch.tensor([[1,1,1,1,1,1],[1,1,1,0,0,0],[0,0,0,0,0,0]],dtype=torch.bool))
    return model, recipe, batch


def literal(model,batch,recipe):
    """Independent token-coordinate eligibility and dense mathematical losses."""
    mode = replace(recipe.mode(), feedback_jitter=0.)
    embeddings = model.backbone.token_embeddings(batch.input_ids)
    output = model.backbone(inputs_embeds=embeddings, attention_mask=batch.valid_mask,
        document_ids=batch.document_ids, mode=mode, feedback_noise=None,
        right_padded_causal=True, return_logits=False)
    locations = {term: [] for term in evaluation.TERMS}
    for row in range(batch.input_ids.shape[0]):
        for target in range(1, batch.input_ids.shape[1]):
            adjacent = bool(batch.valid_mask[row,target-1] and batch.valid_mask[row,target])
            same = adjacent and bool(batch.document_ids[row,target-1] == batch.document_ids[row,target])
            if adjacent and bool(batch.ce_mask[row,target]): locations['ce'].append((row,target))
            if recipe.nextlat and same and bool(batch.latent_mask[row,target]): locations['latent'].append((row,target))
            triple = target >= 2 and same and bool(batch.valid_mask[row,target-2]) and bool(batch.document_ids[row,target-2] == batch.document_ids[row,target])
            if recipe.nextlat and triple and bool(batch.kl_mask[row,target]): locations['kl'].append((row,target))
    passes = []
    for hidden in output.pass_hidden_states:
        sums = dict.fromkeys(evaluation.TERMS, 0.)
        for row,target in locations['ce']:
            logits = F.linear(hidden[row,target-1],model.backbone.readout_weight)
            sums['ce'] += float(-F.log_softmax(logits,dim=-1)[batch.input_ids[row,target]])
        for row,target in locations['latent']:
            predicted = model.predictor(hidden[row,target-1:target],embeddings[row,target:target+1])[0]
            difference = (predicted-hidden[row,target]).abs()
            sums['latent'] += float(torch.where(difference<1,.5*difference.square(),difference-.5).mean())
        for row,target in locations['kl']:
            predicted = model.predictor(hidden[row,target-2:target-1],embeddings[row,target-1:target])[0]
            log_student = F.log_softmax(F.linear(predicted,model.backbone.readout_weight),dim=-1)
            log_teacher = F.log_softmax(F.linear(hidden[row,target-1],model.backbone.readout_weight),dim=-1)
            sums['kl'] += float((log_teacher.exp()*(log_teacher-log_student)).sum())
        passes.append(sums)
    k=len(passes)
    coefficients={'ce':[1.] if k==1 else [.5]+[.5/(k-1)]*(k-1),
                  'latent':[1./k]*k,'kl':[1./k]*k}
    return passes,{name:len(rows) for name,rows in locations.items()},coefficients,locations


@pytest.mark.parametrize('arm',ARMS)
def test_all_arms_match_literal_per_token_losses_and_cross_document_masks(arm):
    model,recipe,batch=setup(arm)
    calls=[]
    handle=model.backbone.register_forward_hook(lambda *args:calls.append(1))
    with evaluation.evaluation_runtime(model) as evidence:
        observed=evaluation.per_pass_sums(model,batch,recipe)
        assert len(calls)==1
        handle.remove()
        expected,counts,coefficients,locations=literal(model,batch,recipe)
    assert evidence['restored'] and evidence['integrity_passed']
    assert observed['counts']==counts and counts['ce']==7
    assert (0,3) in locations['ce']  # True source-document boundary remains a CE target.
    if recipe.nextlat:
        assert counts=={'ce':7,'latent':5,'kl':3}
        assert (0,3) not in locations['latent'] and (0,3) not in locations['kl']
    else: assert counts['latent']==counts['kl']==0
    assert observed['term_pass_coefficients']==coefficients
    for row,want in zip(observed['passes'],expected):
        assert row['counts']==counts
        assert row['sums']==pytest.approx(want,rel=1e-5,abs=1e-6)
    for term in evaluation.TERMS:
        expected_total=sum(coefficient*p[term] for coefficient,p in zip(coefficients[term],expected))
        assert observed['aggregate_sums'][term]==pytest.approx(expected_total,rel=1e-5,abs=1e-6)
        assert observed['enabled'][term]==(term=='ce' or recipe.nextlat)
    assert observed['input_tokens']==10 and observed['mode']['feedback_jitter']==0


@pytest.mark.parametrize('arm',ARMS)
def test_entirely_empty_rank_batch_contributes_exact_zero_without_local_mean(arm):
    model,recipe,batch=setup(arm)
    empty=NextLatBatch(**{name:value[2:].clone() for name,value in vars(batch).items()})
    with evaluation.evaluation_runtime(model): row=evaluation.per_pass_sums(model,empty,recipe)
    assert row['counts']==dict.fromkeys(evaluation.TERMS,0)
    assert row['aggregate_sums']==dict.fromkeys(evaluation.TERMS,0.)
    assert all(p['sums']==dict.fromkeys(evaluation.TERMS,0.) for p in row['passes'])
    assert row['input_tokens']==0


def snapshot(model,generators):
    return tree_digests({'state':model.state_dict(),'gradients':{n:p.grad for n,p in model.named_parameters()},
        'rng':evaluation._local_rng(torch.device('cpu'),generators),
        'modes':{n:m.training for n,m in model.named_modules()},
        'flags':{k:getattr(model.backbone.backbone,k) for k in evaluation.RUNTIME_FLAGS}})


@pytest.mark.parametrize('fail',[False,True])
def test_scope_preserves_mixed_modes_named_rng_runtime_and_zero_gradient_owners(fail):
    model,recipe,batch=setup()
    model.backbone.fusion.training=False
    base=model.backbone.backbone
    base.tile_backend=base.backward_tile_backend='triton'
    base.ordinary_pointwise_backend='compiled';base.ordinary_rope_backend='dao'
    gradients={}
    for index,(name,p) in enumerate(model.named_parameters()):
        if index%2:p.grad=torch.zeros_like(p)
        gradients[name]=p.grad
    generators={'data':torch.Generator().manual_seed(23)}
    before=snapshot(model,generators)
    def run():
        with torch.autocast('cpu',dtype=torch.bfloat16):
            with evaluation.evaluation_runtime(model,generators=generators) as evidence:
                assert not torch.is_autocast_enabled('cpu') and not torch.is_grad_enabled()
                assert not torch.is_autocast_cache_enabled()
                row=evaluation.per_pass_sums(model,batch,recipe)
                random.random();np.random.rand();torch.rand(1);torch.rand(1,generator=generators['data'])
                if fail:raise OSError('intentional evaluation failure')
            assert torch.is_autocast_enabled('cpu')
            return row,evidence
    if fail:
        with pytest.raises(OSError,match='intentional'):run()
    else:
        _,evidence=run();assert evidence['integrity_passed']
    assert snapshot(model,generators)==before
    assert all(p.grad is gradients[n] for n,p in model.named_parameters())


@pytest.mark.parametrize('damage',['replace_gradient','nonzero_gradient','weight_version'])
def test_integrity_failures_are_visible_even_after_runtime_restoration(damage):
    model,recipe,batch=setup()
    parameter=next(model.parameters());parameter.grad=torch.zeros_like(parameter)
    original_grad=parameter.grad;flags={key:getattr(model.backbone.backbone,key) for key in evaluation.RUNTIME_FLAGS}
    with pytest.raises(RuntimeError,match='live-state integrity'):
        with evaluation.evaluation_runtime(model) as evidence:
            if damage=='replace_gradient':parameter.grad=original_grad.clone()
            elif damage=='nonzero_gradient':parameter.grad.add_(1)
            else:parameter.add_(1)
    assert evidence['restored'] and not evidence['integrity_passed']
    assert {key:getattr(model.backbone.backbone,key) for key in evaluation.RUNTIME_FLAGS}==flags
    assert parameter.grad is original_grad and not bool(original_grad.any())


@pytest.mark.parametrize('invalid',['outside_scope','nonzero_entry','recipe','length','nonfinite'])
def test_invalid_entry_or_result_cannot_become_a_metric(invalid,monkeypatch):
    model,recipe,batch=setup()
    if invalid=='outside_scope':
        with pytest.raises(ValueError,match='evaluation_runtime'):evaluation.per_pass_sums(model,batch,recipe)
        return
    if invalid=='nonzero_entry':
        next(model.parameters()).grad=torch.ones_like(next(model.parameters()))
        with pytest.raises(ValueError,match='zero or absent'):
            with evaluation.evaluation_runtime(model):pass
        return
    if invalid=='recipe':recipe=replace(recipe,arm='FR')
    elif invalid=='length':batch=NextLatBatch(**{name:value[:,:5] for name,value in vars(batch).items()})
    elif invalid=='nonfinite':
        original=model.loss_sums
        def corrupt(*args,**kwargs):
            value=original(*args,**kwargs);value.pass_losses[0].sums['ce']=torch.tensor(float('nan'));return value
        monkeypatch.setattr(model,'loss_sums',corrupt)
    with evaluation.evaluation_runtime(model):
        with pytest.raises(ValueError):evaluation.per_pass_sums(model,batch,recipe)


@pytest.mark.parametrize('arm',['B','NFR'])
def test_evaluation_between_real_prepared_cpu_updates_leaves_next_update_exact(arm):
    initial,recipe,batch=setup(arm)
    results={}
    for insert in (False,True):
        model=deepcopy(initial)
        optimizer=build_campaign_adamw(model,recipe,fused=False)
        counters=TrainingCounters()
        noise=feedback_noise_for_rows(recipe,['a','b','dummy'],logical_update=0,
            sequence_length=6,width=model.config.model_dim)
        with evaluation.sdpa_kernel(evaluation.SDPBackend.MATH):
            objective=CampaignObjective(model,batch,mode=recipe.mode(),global_counts=model.counts(batch),
                feedback_noise=noise,config=LMTrainingConfig(precision='fp32'))
            runner=CampaignGraphTraining(objective)
            first=runner.optimizer_step(optimizer,(batch,),feedback_noises=(noise,),counters=counters)
            runner.zero_grad()
            pointers={name:None if p.grad is None else p.grad.data_ptr() for name,p in model.named_parameters()}
            inputs=tree_digests(objective.owned_inputs())
            if insert:
                with evaluation.evaluation_runtime(model) as evidence:
                    evaluation.per_pass_sums(model,batch,recipe)
                assert evidence['integrity_passed']
                assert inputs==tree_digests(objective.owned_inputs())
                assert pointers=={name:None if p.grad is None else p.grad.data_ptr() for name,p in model.named_parameters()}
            second=runner.optimizer_step(optimizer,(batch,),feedback_noises=(noise,),counters=counters)
        results[insert]=(first,second,tree_digests({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                                                 'counters':asdict(counters)}))
    assert results[False]==results[True]
