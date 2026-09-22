"""Canonical update parity and persistent-gradient ownership for static plans."""
import copy
from dataclasses import replace

import pytest
import torch

from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters,
    build_adamw, build_warmup_scheduler, optimizer_step)
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT, FBTMode
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.static_training import StaticFBTTraining, normalized_objective


def setup(nextlat=True, checkpoint=True):
    torch.manual_seed(91); torch.set_num_threads(1)
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math",
        ordinary_activation_checkpointing=checkpoint, attention_precision="fp32")
    model = FBTNextLatLM(OLMoFBT(base), NextLatConfig(model_dim=base.config.model_dim,
                        proj_factor=2, vocab_chunk_size=4), enabled=nextlat).train()
    ids = torch.tensor([[2, 3, 4, 5, 6, 7], [3, 7, 4, 2, 1, 1]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2])
    docs = torch.tensor([[0]*6, [1]*4+[-1]*2])
    ce, latent, kl = valid.clone(), valid.clone(), valid.clone()
    ce[0, 2] = False; latent[1, 1] = False; kl[0, 4] = False
    return model, NextLatBatch(ids, valid, docs, ce, latent, kl)


@pytest.mark.parametrize("fbt,rt,nextlat", [(f,r,n) for f in (False,True) for r in (False,True) for n in (False,True)])
def test_static_complete_updates_preserve_canonical_losses_weights_and_moments(fbt, rt, nextlat):
    reference, batch = setup(nextlat)
    actual = copy.deepcopy(reference)
    mode = FBTMode(enabled=fbt, num_passes=3, beta=.35, rt_mode=RTMode((0,) if rt else (), .37))
    plan = StaticFBTTraining(actual, batch, mode=mode)
    optimizers = [build_adamw(m, lr=1e-4, eps=1e-4) for m in (reference, actual)]
    schedulers = [build_warmup_scheduler(o, warmup_updates=2) for o in optimizers]
    counts = [TrainingCounters(), TrainingCounters()]
    for update in range(3):
        changed = replace(batch, input_ids=batch.input_ids.roll(update+1, -1))
        expected = optimizer_step(reference, optimizers[0], [changed], scheduler=schedulers[0],
            counters=counts[0], backbone_kwargs={"mode": mode})
        got = plan.optimizer_step(optimizers[1], changed, scheduler=schedulers[1], counters=counts[1])
        assert got["counts"] == expected["counts"]
        assert got["counters"] == expected["counters"]
        assert got["objective"] == pytest.approx(expected["objective"], rel=3e-6)
        assert got["gradient_norm_before_clip"] == pytest.approx(expected["gradient_norm_before_clip"], rel=5e-6)
        for (name, p), (_, q) in zip(actual.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p, q, atol=3e-7, rtol=2e-5, msg=name)
            left, right = optimizers[1].state.get(p, {}), optimizers[0].state.get(q, {})
            assert left.keys() == right.keys()
            for key in left:
                torch.testing.assert_close(left[key], right[key], atol=3e-7, rtol=3e-5, msg=name+key)


@pytest.mark.parametrize("beta,passes,gamma", [(0.,2,1.), (1.,1,1.), (1.,3,0.)])
def test_inactive_or_zero_weight_branches_preserve_canonical_gradient_ownership(beta, passes, gamma):
    model, batch = setup()
    model._gamma = gamma
    mode = FBTMode(num_passes=passes, beta=beta, rt_mode=RTMode((0,)))
    model.loss_sums(batch, backbone_kwargs={"mode":mode}).total.backward()
    active = {n for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    plan = StaticFBTTraining(model, batch, mode=mode)
    plan.initialize_gradients()
    assert set(plan.active_names) == active
    before = {n:p.detach().clone() for n,p in model.named_parameters() if n not in active}
    plan.optimizer_step(build_adamw(model, lr=.01), batch)
    for n,p in model.named_parameters():
        if n not in active:
            assert p.grad is None
            assert torch.equal(p,before[n])


def test_repeated_eager_backward_overwrites_and_replaced_buffers_reject():
    model, batch = setup()
    plan = StaticFBTTraining(model,batch,mode=FBTMode(rt_mode=RTMode((0,))))
    plan.backward()
    expected={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    plan.backward(); plan.backward()
    for n,p in model.named_parameters():
        if p.grad is not None:
            assert torch.equal(p.grad,expected[n])
    model.zero_grad(set_to_none=True)
    with pytest.raises(ValueError,match="gradient buffers"):
        plan.backward()


def test_changed_masks_fail_before_token_buffer_write():
    model,batch=setup()
    plan=StaticFBTTraining(model,batch,mode=FBTMode())
    original=plan.batch.input_ids.clone()
    mask=batch.ce_mask.clone(); mask[0,2]=True
    with pytest.raises(ValueError):
        plan.load_batch(replace(batch,input_ids=batch.input_ids+1,ce_mask=mask))
    assert torch.equal(plan.batch.input_ids,original)


def test_cpu_graph_request_never_silently_runs_eager():
    model,batch=setup()
    plan=StaticFBTTraining(model,batch,mode=FBTMode())
    with pytest.raises(ValueError,match="CUDA"):
        plan.capture()
    with pytest.raises(ValueError,match="Capture"):
        plan.backward(replay=True)


def test_objective_config_or_parameter_replacement_invalidates_plan():
    model,batch=setup()
    plan=StaticFBTTraining(model,batch,mode=FBTMode())
    model._gamma=.2
    with pytest.raises(ValueError,match="objective configuration"):
        plan.validate_execution()
    model._gamma=1.
    model.predictor.mlp[0].weight = torch.nn.Parameter(model.predictor.mlp[0].weight.clone())
    with pytest.raises(ValueError,match="storage/ownership"):
        plan.validate_execution()
