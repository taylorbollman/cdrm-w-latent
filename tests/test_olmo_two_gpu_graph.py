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
