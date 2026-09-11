"""CPU checks of exact counts, pairing, native supervision and epoch recovery."""
import copy
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cdrm_fuzzy_common as common
import cdrm_fuzzy_train as train


def test_exact_full_width_counts_without_initialization_draws():
    from olmo.model import OLMo
    for arm in common.ARMS:
        model=OLMo(common.config(arm,'fp32','naive',overrides={'init_device':'meta'}))
        counts=common.parameter_counts(model)
        assert counts['total']==common.PARAMETERS[arm]
        assert counts['input_and_output_tables']==32768
        assert counts['adapters']==(2097152 if arm=='cdrm' else 0)
        assert model.config.ordinary_attention_precision_policy=='fp32'
        assert model.config.cdrm_enabled==(arm=='cdrm')


@pytest.fixture(scope='module')
def small_pair():
    torch.set_num_threads(1)
    return common.cpu_initial_pair(17,overrides={'d_model':8,'n_heads':2,'n_kv_heads':2,'n_layers':10})


def test_pairing_preserves_same_shapes_and_native_resized_mlps(small_pair):
    models,record=small_pair
    source=models['cdrm'].state_dict();target=models['seq'].state_dict()
    assert len(record['native_resized_mlp_names'])==20
    assert record['adapter_seed']==record['seed']+100003
    assert all(torch.equal(source[name],target[name]) for name in record['copied_names'])
    assert all(source[name].shape!=target[name].shape for name in record['native_resized_mlp_names'])
    assert set(source)-set(target)==common.ADAPTERS
    assert not models['seq'].config.cdrm_enabled and getattr(models['seq'],'cdrm',None) is None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(record['adapter_seed'])
        for name in ('cdrm.deep_adapter.weight','cdrm.bridge_adapter.weight'):
            expected=torch.empty_like(source[name]);torch.nn.init.normal_(expected,std=8**-.5)
            assert torch.equal(source[name],expected)
    for model in models.values():
        opt=common.optimizer_for(model,1e-3)
        flat=[p for group in opt.param_groups for p in group['params']]
        assert len(flat)==len(set(map(id,flat)))==len(list(model.parameters()))
        assert opt.param_groups[0]['lr']==1e-3 and opt.param_groups[0]['betas']==(.9,.98)


def test_native_dense_training_and_masked_answer_metrics_are_distinct():
    logits=torch.zeros(2,4,16,requires_grad=True)
    dense=torch.tensor([[1,7,15,15],[2,8,15,15]])
    answer=torch.tensor([[-100,7,-100,-100],[-100,8,-100,-100]])
    summed,count=common.native_loss_sum(logits,dense)
    assert count==8 and common.metric_counts(logits,answer)['targets']==2
    (summed/count).backward()
    assert torch.count_nonzero(logits.grad[:,:,15]).item()==8
    with pytest.raises(ValueError):common.native_loss_sum(logits,torch.full_like(dense,-100))


class IndexDataset:
    def __len__(self):return 12
    def take(self,indices):return SimpleNamespace(sha256=common.state_digest(np.asarray(indices)))


def toy_run(target,initial_state,restored=None):
    model=torch.nn.Linear(3,2,bias=False);model.load_state_dict(initial_state)
    optimizer=common.optimizer_for(model,5e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-6)
    identity={'physical_batch':4,'shuffle_seed':23,'base_lr':5e-4}
    completed=epoch=position=0;history=[]
    common.seed_cpu(37)
    if restored:
        model.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler']);common.restore_rng(restored['rng'])
        completed,epoch,position=(restored[key] for key in ('completed_updates','completed_epochs','batch_in_epoch'))
        history=copy.deepcopy(restored['history'])
    dataset=IndexDataset()
    while completed<target:
        indices=train.batch_indices(completed,len(dataset),4,23)
        optimizer.zero_grad(set_to_none=True)
        x=torch.as_tensor(indices[:,None],dtype=torch.float32).expand(-1,3)/12+torch.rand(4,3)*.01
        loss=model(x).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        rate=optimizer.param_groups[0]['lr'];optimizer.step()
        history.append({'update':completed+1,'epoch':epoch+1,'batch_in_epoch':position,'learning_rate':rate,
            'indices_sha256':common.state_digest(indices),'batch_sha256':dataset.take(indices).sha256,
            'native_loss':loss.item(),'seconds':1.})
        completed+=1;epoch,position=train.advance_cursor(epoch,position,3,scheduler)
    return common.cpu_tree({'format':common.FORMAT,'identity':identity,'identity_sha256':common.json_digest(identity),
        'arm':'seq','precision':'fp32','model_config':{},'initialization':{},'initial_checkpoint':{},
        'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':common.rng_state(),
        'completed_updates':completed,'completed_epochs':epoch,'batch_in_epoch':position,'history':history,'development':{},
        'next_batch_sha256':dataset.take(train.batch_indices(completed,len(dataset),4,23)).sha256,'run_target_updates':target})


def test_exact_recovery_spans_epoch_boundary_and_preserves_next_batch():
    state=torch.nn.Linear(3,2,bias=False).state_dict()
    uninterrupted=toy_run(7,state);prefix=toy_run(2,state);resumed=toy_run(7,state,prefix)
    train.validate_position(resumed,IndexDataset())
    assert train.compare_recovery(uninterrupted,resumed)['bitwise_state_and_nontiming_metrics_equal']
    assert resumed['scheduler']['last_epoch']==2 and resumed['batch_in_epoch']==1
    assert resumed['history'][2]['learning_rate']==5e-4
    assert resumed['history'][3]['learning_rate']<5e-4
    corrupt=copy.deepcopy(resumed);corrupt['next_batch_sha256']='wrong'
    with pytest.raises(ValueError,match='next batch'):train.validate_position(corrupt,IndexDataset())
    corrupt=copy.deepcopy(resumed);corrupt['history'][3]['learning_rate']=5e-4
    with pytest.raises(ValueError,match='learning rate'):train.validate_position(corrupt,IndexDataset())


def test_resume_rejects_changed_data_runtime_lr_or_source(tmp_path):
    source=tmp_path/'runner.py';source.write_text('frozen')
    identity={'source_sha256':{str(source):common.file_digest(source)},'model_config':{},'arm':'seq',
        'data':{'sha256':'arrays'},'execution_contract':{'cache':'shared'},'base_lr':5e-4}
    payload={'format':common.FORMAT,'identity':identity,'identity_sha256':common.json_digest(identity),
        'model_config':{},'arm':'seq','completed_updates':2,'run_target_updates':7}
    train.validate_checkpoint(payload,identity,7)
    for key,value in [('data',{'sha256':'other'}),('execution_contract',{'cache':'cold'}),('base_lr',1e-3)]:
        altered=copy.deepcopy(identity);altered[key]=value
        with pytest.raises(ValueError,match='identity differs'):train.validate_checkpoint(payload,altered,7)
    with pytest.raises(ValueError,match='shorten'):train.validate_checkpoint(payload,identity,3)
    train.validate_checkpoint(payload,identity,3,recovery=True)
    source.write_text('changed')
    with pytest.raises(RuntimeError,match='source identity'):train.validate_checkpoint(payload,identity,7)


def test_calibration_cli_caps_budget_and_keeps_fp32_fallback_explicit():
    base=['--arm','seq','--initial-checkpoint','init.pt','--data-root','data','--data-manifest','manifest.json',
          '--protocol','protocol.json','--output-dir','new','--batch-size','64','--stop-updates','100']
    args=train.parse_args(base)
    assert args.epochs==10 and args.stop_updates==100 and args.precision=='bf16'
    assert train.parse_args(base+['--precision','fp32']).precision=='fp32'
    for extra in (['--epochs','11'],['--stop-updates','2001'],['--batch-size','16']):
        with pytest.raises(SystemExit):train.parse_args(base+extra)


def test_protocol_guard_rejects_undeclared_draws_before_initialization(tmp_path):
    protocol=tmp_path/'original-protocol.json'
    protocol.write_text(json.dumps({'schema':'cdrm-fuzzy-prospective-protocol-v1',
        'initialization':{'calibration_model_seed':86100,'additional_numerical_model_seed':86101,'adapter_seed_offset':100003},
        'models':{'cdrm':{'width':1024,'heads':16,'layers':12,'mlp':4096,'parameters':153175040,
                          'early_layer':3,'late_layer':8,'rho':1.,'epsilon':.1,'lambda':.01},
                  'seq':{'width':1024,'heads':16,'layers':12,'mlp':4192,'parameters':153437184}},
        'optimizer':{'name':'AdamW','betas':[.9,.98],'eps':1e-8,'weight_decay':0.,'clip_norm':1.,'foreach':False,'fused':False},
        'schedule':{'epochs':50,'eta_min':1e-6,'name':'epoch_cosine','step':'after completed epoch','warmup_epochs':0},
        'calibration':{'learning_rates':[1e-4,5e-4,1e-3],'epochs':10},
        'data':{'task':'fuzzy-in-context-recall'},'precision':{'ordinary_attention_precision_policy':'fp32'}}))
    document=common.validate_protocol(protocol,86100)
    common.validate_protocol(protocol,86101)
    with pytest.raises(ValueError,match='not declared'):common.validate_protocol(protocol,86102)
    path=tmp_path/'protocol.json';document['schedule']['epochs']=10;path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='schedule'):common.validate_protocol(path,86100)
