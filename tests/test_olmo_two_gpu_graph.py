"""CPU checks for graph-probe accounting and safe restore; no CUDA evidence."""
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.lm_training import LMTrainingConfig,TrainingCounters,build_warmup_scheduler
from scripts.olmo_two_gpu_graph import (parse_args,fixed_batch,snapshot_state,restore_state,
    raw_snapshot,exact_raw_check,finish_step)
from scripts.olmo_f1_common import IntegrationCase


def test_tiny_batches_change_tokens_but_keep_graph_contract():
    case=IntegrationCase('combined',fbt=True,nextlat=True,rt_layers=(0,1),batch_size=2,length=8)
    batches=[fixed_batch(case,None,u,r,tiny=True) for u in range(2) for r in range(2)]
    assert len({tuple(b.input_ids.flatten().tolist()) for b in batches})==4
    for batch in batches:
        for name in ('valid_mask','document_ids','ce_mask','latent_mask','kl_mask'):
            assert torch.equal(getattr(batch,name),getattr(batches[0],name))


def test_restore_preserves_weights_and_gradient_addresses():
    model=torch.nn.Linear(3,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=build_warmup_scheduler(optimizer,warmup_updates=2)
    counters=TrainingCounters(optimizer_updates=1)
    model(torch.ones(2,3)).sum().backward();optimizer.step();scheduler.step()
    saved=snapshot_state(model,optimizer,scheduler,counters)
    pointers={n:p.data_ptr() for n,p in model.named_parameters()}
    grads={n:p.grad.data_ptr() for n,p in model.named_parameters()}
    optimizer.step();scheduler.step();counters.optimizer_updates+=1
    restore_state(model,optimizer,scheduler,counters,saved)
    assert counters.optimizer_updates==1
    for n,p in model.named_parameters():
        assert torch.equal(p,saved['model'][n])
        assert p.data_ptr()==pointers[n] and p.grad.data_ptr()==grads[n]


def test_exact_raw_check_detects_gradient_and_loss_discrepancy():
    model=torch.nn.Linear(2,2)
    result={'objective':model(torch.ones(1,2)).sum()}
    result['objective'].backward()
    snapshot=raw_snapshot(model,result)
    assert exact_raw_check(model,result,snapshot)['passed']
    model.weight.grad.add_(.01)
    assert not exact_raw_check(model,result,snapshot)['passed']
    model.weight.grad.copy_(snapshot['gradients']['weight'])
    assert not exact_raw_check(model,{'objective':result['objective']+1},snapshot)['passed']


def test_finish_step_counts_global_inputs_once_and_keeps_gradient_storage(monkeypatch):
    import scripts.olmo_two_gpu_graph as module
    model=torch.nn.Linear(2,2)
    model(torch.ones(1,2)).sum().backward()
    addresses=[p.grad.data_ptr() for p in model.parameters()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=build_warmup_scheduler(optimizer,warmup_updates=2)
    counts={'ce':6,'latent':4,'kl':2};weights={'ce':1.,'latent':.3,'kl':.7}
    adapter=SimpleNamespace(world_size=2,global_counts=counts,plan=SimpleNamespace(
        documents=1,input_tokens=4,weights=weights,config=LMTrainingConfig()))
    runtime=SimpleNamespace(model=model,adapter=adapter)
    def reduce(value,op=None):
        if op is None: value.mul_(2)
    monkeypatch.setattr(module.dist,'all_reduce',reduce)
    counters=TrainingCounters()
    metrics=finish_step(runtime,optimizer,scheduler,counters,
        {'loss_sums':{'ce':torch.tensor(3.),'latent':torch.tensor(2.),'kl':torch.tensor(1.)}})
    assert counters==TrainingCounters(1,2,2,8,6,4,2)
    assert metrics['loss_sums']=={'ce':6.,'latent':4.,'kl':2.}
    assert metrics['objective']==pytest.approx(2.)
    assert [p.grad.data_ptr() for p in model.parameters()]==addresses


def test_nonfinite_global_health_prevents_adam(monkeypatch):
    import scripts.olmo_two_gpu_graph as module
    model=torch.nn.Linear(2,2);model(torch.ones(1,2)).sum().backward()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=build_warmup_scheduler(optimizer,warmup_updates=2)
    runtime=SimpleNamespace(model=model,adapter=SimpleNamespace(plan=SimpleNamespace(config=LMTrainingConfig())))
    monkeypatch.setattr(module.dist,'all_reduce',lambda value,op=None:value.zero_())
    stepped=[];optimizer.register_step_pre_hook(lambda *args:stepped.append(True))
    with pytest.raises(FloatingPointError):
        finish_step(runtime,optimizer,scheduler,TrainingCounters(),
            {'loss_sums':dict.fromkeys(('ce','latent','kl'),torch.tensor(1.))})
    assert not stepped


@pytest.mark.parametrize('extra',[['--batch-size','0'],['--batch-size','9'],['--length','7']])
def test_correctness_cli_rejects_unbounded_or_invalid_shapes(extra):
    with pytest.raises(SystemExit):
        parse_args(['--case','rt','--stage','correctness','--output-dir','.runtime/test',*extra])


def tiny_prepared_model():
    from cdrm.pretrained.fbt_training import FBTNextLatLM
    from cdrm.pretrained.nextlat import NextLatConfig
    from cdrm.pretrained.olmo import OLMoConfig
    from cdrm.pretrained.olmo_fbt import OLMoFBT
    from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
    from cdrm.pretrained.ddp_graph_training import PreparedDDPObjective,DDPGraphTraining
    from scripts.olmo_f1_common import active_names
    torch.set_num_threads(1)
    base=OLMoTiledRTForCausalLM(OLMoConfig.tiny(),attention_backend='math',attention_precision='fp32',
                               reuse_rope=True,kv_only_writes=True)
    model=FBTNextLatLM(OLMoFBT(base),NextLatConfig(model_dim=base.config.model_dim,vocab_chunk_size=4)).train()
    case=IntegrationCase('combined',fbt=True,nextlat=True,rt_layers=(0,1),batch_size=1,length=8)
    batch=fixed_batch(case,None,0,0,tiny=True)
    adapter=PreparedDDPObjective(model,batch,mode=case.mode(),global_counts=model.counts(batch),world_size=1)
    runtime=DDPGraphTraining(adapter,expected_active_names=sorted(active_names(model,case.mode())))
    adapter()['objective'].backward();runtime._freeze_gradients()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=build_warmup_scheduler(optimizer,warmup_updates=2)
    counters=TrainingCounters()
    return model,optimizer,scheduler,counters,runtime


def test_actual_tiny_olmo_prepared_contract_survives_parameter_only_restore():
    model,optimizer,scheduler,counters,runtime=tiny_prepared_model()
    optimizer.step();scheduler.step();counters.optimizer_updates=1
    snapshot=snapshot_state(model,optimizer,scheduler,counters)
    buffers={n:(id(b),b.data_ptr(),b._version,b.clone()) for n,b in model.named_buffers()}
    assert 'backbone.fusion.output_scale' in buffers
    optimizer.step();scheduler.step();counters.optimizer_updates=2
    restore_state(model,optimizer,scheduler,counters,snapshot)
    runtime.validate_execution()  # Real PreparedFBTLayout checks fixed buffer versions.
    for name,buffer in model.named_buffers():
        identity,address,version,value=buffers[name]
        assert (id(buffer),buffer.data_ptr(),buffer._version)==(identity,address,version)
        assert torch.equal(buffer,value)
    assert all(torch.equal(value,snapshot['model'][name]) for name,value in model.state_dict().items())


def test_changed_fixed_buffer_is_rejected_before_parameter_restore():
    model,optimizer,scheduler,counters,runtime=tiny_prepared_model()
    snapshot=snapshot_state(model,optimizer,scheduler,counters)
    with torch.no_grad():
        next(model.parameters()).add_(.1)
        model.backbone.fusion.output_scale.add_(.25)
    current={n:p.detach().clone() for n,p in model.named_parameters()}
    version=model.backbone.fusion.output_scale._version
    with pytest.raises(ValueError,match='unchanged fixed model buffer'):
        restore_state(model,optimizer,scheduler,counters,snapshot)
    assert all(torch.equal(p,current[n]) for n,p in model.named_parameters())
    assert model.backbone.fusion.output_scale._version==version
    with pytest.raises(ValueError,match='buffers changed'):
        runtime.validate_execution()  # Restore must not reset the existing contract.


def test_graph_restore_rejects_inconsistent_tied_parameter_snapshot():
    model=torch.nn.Module()
    model.weight=torch.nn.Parameter(torch.ones(2,2));model.alias=model.weight
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=build_warmup_scheduler(optimizer,warmup_updates=2)
    counters=TrainingCounters();snapshot=snapshot_state(model,optimizer,scheduler,counters)
    snapshot['model']['alias'].add_(1)
    with pytest.raises(ValueError,match='tied parameter aliases disagree'):
        restore_state(model,optimizer,scheduler,counters,snapshot)
    assert torch.equal(model.weight,torch.ones_like(model.weight))
