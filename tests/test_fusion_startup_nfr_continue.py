"""Real tiny updates and strict scheduler-fork/recovery oracles; no CUDA."""
import copy
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from scripts import olmo_fusion_startup_nfr_continue as probe
from scripts.olmo_lm_common import tree_digests
from test_fusion_startup_packed_bridge import endpoint_fixture,tiny


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    torch.set_num_threads(1)
    @contextmanager
    def preserve():
        state=probe.endpoint._rng_state(None)
        try:yield
        finally:probe.endpoint._restore_rng(state,None)
    monkeypatch.setattr(probe.endpoint,'preserve_local_rng',preserve)


def setup(tmp_path):
    _,_,_,receipt,authority=endpoint_fixture(tmp_path)
    model,recipe,source,fixtures,flags=tiny(packed=False)
    optimizer,scheduler,counters,pins=probe.restore_original(model,recipe,source,tmp_path/'endpoint.pt',receipt['sha256'],authority)
    metadata=copy.deepcopy(authority['training_metadata'])*5
    return SimpleNamespace(model=model,recipe=recipe,source=source,fixtures=fixtures,flags=flags,optimizer=optimizer,
        scheduler=scheduler,counters=counters,pins=pins,metadata=metadata,authority=authority,receipt=receipt)


def fork(obj):
    obj.scheduler,info=probe.extend_schedule(obj.optimizer,obj.scheduler,obj.counters,
        obj.authority['training_metadata'],obj.metadata)
    return info


def config(obj):
    return {'kind':probe.SCHEMA,'plan':probe.PLAN,'frozen_buffer_pins':tree_digests(dict(obj.model.named_buffers())),
        'schedule':obj.scheduler.checkpoint_contract(),'precision':'bf16_mixed'}


def test_same_original_bf16_endpoint_import_and_prefix_fork_preserve_full_adam_and_clock(tmp_path):
    obj=setup(tmp_path)
    before=tree_digests(obj.optimizer.state_dict());weights=probe.state_pins(obj.model)
    ids={n:id(p) for n,p in obj.model.named_parameters()};old=copy.deepcopy(obj.scheduler.state_dict())
    counters=asdict(obj.counters);info=fork(obj)
    assert all(info['checks'].values())
    assert before==tree_digests(obj.optimizer.state_dict()) and weights==probe.state_pins(obj.model)
    assert obj.model.backbone.readout_weight is obj.model.backbone.token_embeddings.weight
    assert ids=={n:id(p) for n,p in obj.model.named_parameters()} and counters==asdict(obj.counters)
    assert len(obj.scheduler.token_prefix)==21 and obj.scheduler.token_prefix[:5]==old['token_prefix']
    assert obj.scheduler.last_epoch==4 and obj.scheduler._step_count==5
    assert obj.scheduler._last_lr==old['_last_lr'] and obj.scheduler.completed_tokens==obj.counters.input_tokens
    assert obj.scheduler.plan_sha256!=old['plan_sha256']
    assert obj.scheduler.warmup_tokens==old['warmup_tokens']
    probe.validate_clock(obj.model,obj.optimizer,obj.scheduler,obj.counters,obj.metadata)


@pytest.mark.parametrize('mutation',['prefix','count','clock','lr','negative_tokens'])
def test_invalid_fork_rejected_before_optimizer_or_scheduler_mutation(tmp_path,mutation):
    obj=setup(tmp_path);metadata=copy.deepcopy(obj.metadata);counters=copy.deepcopy(obj.counters)
    if mutation=='prefix':metadata[0]['input_tokens']+=1
    elif mutation=='count':metadata.pop()
    elif mutation=='clock':counters.input_tokens+=1
    elif mutation=='lr':obj.optimizer.param_groups[0]['lr']*=2
    else:metadata[-1]['input_tokens']=0
    before=tree_digests(obj.optimizer.state_dict()),tree_digests(obj.scheduler.state_dict())
    with pytest.raises(ValueError):
        probe.extend_schedule(obj.optimizer,obj.scheduler,counters,obj.authority['training_metadata'],metadata)
    assert before==(tree_digests(obj.optimizer.state_dict()),tree_digests(obj.scheduler.state_dict()))


def test_first_continued_update_matches_literal_original_backward_clip_adam_and_new_token_clock(tmp_path):
    obj=setup(tmp_path);fork(obj)
    start=probe.endpoint.capture_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters,obj.metadata)
    row=probe.update(obj.model,obj.recipe,obj.fixtures,obj.optimizer,obj.scheduler,obj.counters,
        path=probe.FP32,original_flags=obj.flags,metadata=obj.metadata)
    actual=probe.endpoint.capture_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters,obj.metadata)
    obj.counters=probe.endpoint.restore_boundary(obj.model,obj.optimizer,obj.scheduler,start,obj.metadata)
    original,_,_=probe.endpoint.update(obj.model,obj.recipe,obj.fixtures,obj.optimizer,obj.scheduler,obj.counters,
        path=probe.FP32,original_flags=obj.flags,metadata=obj.metadata)
    expected=probe.endpoint.capture_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters,obj.metadata)
    assert tree_digests(actual)==tree_digests(expected)
    assert row['metrics']==original['metrics'] and row['lr_used']==original['lr_used']
    assert row['gradient_norm_before_clip']==original['gradient_norm_before_clip']
    assert obj.counters.optimizer_updates==5 and obj.scheduler._step_count==6
    assert row['lr_used'][0]==start['optimizer']['param_groups'][0]['lr']
    assert all(v>0 for v in row['raw_gradient_norms'].values())
    assert not any(k.endswith('_pins') for k in row if k!='input_pins')


def test_fresh_process_style_resume_matches_next_update_exactly(tmp_path):
    obj=setup(tmp_path);fork(obj)
    probe.update(obj.model,obj.recipe,obj.fixtures,obj.optimizer,obj.scheduler,obj.counters,
        path=probe.FP32,original_flags=obj.flags,metadata=obj.metadata)
    configuration=config(obj);fingerprint={'checkpoint_sha256':'a'*64}
    cursor=lambda update:{'next_update':update,'manifest_sha256':'c'*64}
    receipt=probe.endpoint.save_endpoint(tmp_path/'continued.pt',obj.model,obj.optimizer,obj.scheduler,obj.counters,
        configuration=configuration,source_fingerprint=fingerprint,data_cursor=cursor(149))
    golden=probe.update(obj.model,obj.recipe,obj.fixtures,obj.optimizer,obj.scheduler,obj.counters,
        path=probe.FP32,original_flags=obj.flags,metadata=obj.metadata)
    boundary=probe.endpoint.capture_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters,obj.metadata)
    other_dir=tmp_path/'fresh';other_dir.mkdir();fresh=setup(other_dir);fork(fresh)
    fresh.counters=probe.load_continuation(tmp_path/'continued.pt',fresh.model,fresh.optimizer,fresh.scheduler,
        digest=receipt['sha256'],configuration=configuration,fingerprint=fingerprint,metadata=fresh.metadata,
        data=SimpleNamespace(cursor=cursor))
    row=probe.update(fresh.model,fresh.recipe,fresh.fixtures,fresh.optimizer,fresh.scheduler,fresh.counters,
        path=probe.FP32,original_flags=fresh.flags,metadata=fresh.metadata)
    assert row==golden
    assert tree_digests(probe.endpoint.capture_boundary(fresh.model,fresh.optimizer,fresh.scheduler,fresh.counters,fresh.metadata))==tree_digests(boundary)


@pytest.mark.parametrize('mutation',['cursor','buffer','configuration','moment','scheduler','ownership'])
def test_invalid_resume_rejected_before_model_adam_or_rng_mutation(tmp_path,mutation):
    obj=setup(tmp_path);fork(obj);configuration=config(obj);fingerprint={'checkpoint_sha256':'a'*64}
    data=SimpleNamespace(cursor=lambda step:{'next_update':step})
    receipt=probe.endpoint.save_endpoint(tmp_path/'continued.pt',obj.model,obj.optimizer,obj.scheduler,obj.counters,
        configuration=configuration,source_fingerprint=fingerprint,data_cursor=data.cursor(148))
    payload=torch.load(tmp_path/'continued.pt',map_location='cpu',weights_only=True)
    if mutation=='cursor':payload['data_cursor']['next_update']+=1
    elif mutation=='buffer':payload['model']['backbone.fusion.output_scale']+=.1
    elif mutation=='configuration':payload['configuration']['precision']='fp32'
    elif mutation=='moment':next(iter(payload['optimizer']['state'].values()))['exp_avg'].view(-1)[0]=float('nan')
    elif mutation=='scheduler':payload['scheduler']['token_prefix']=tuple([0]+[x+1 for x in payload['scheduler']['token_prefix'][1:]])
    else:
        payload['parameter_layout'][0]['aliases'].append('foreign_readout_alias')
    torch.save(payload,tmp_path/'bad.pt')
    before=probe.bridge.current_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters)
    with pytest.raises((ValueError,AssertionError)):
        probe.load_continuation(tmp_path/'bad.pt',obj.model,obj.optimizer,obj.scheduler,
            digest=probe.sha256_file(tmp_path/'bad.pt'),configuration=configuration,fingerprint=fingerprint,
            metadata=obj.metadata,data=data)
    assert before==probe.bridge.current_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters)


def test_heldout_common_fp32_keeps_forked_full_state_and_reports_all_losses(tmp_path):
    obj=setup(tmp_path);fork(obj)
    before=probe.bridge.current_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters)
    result=probe.endpoint.evaluate_fp32(obj.model,obj.recipe,obj.fixtures,original_flags=obj.flags)
    assert all(result['checks'].values()) and set(result['loss_means'])=={'ce','latent','kl'}
    assert before==probe.bridge.current_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters)


def test_per_pass_observation_keeps_old_aggregate_exact_and_adds_no_forward(tmp_path):
    obj=setup(tmp_path);fork(obj)
    before=probe.bridge.current_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters)
    calls=[]
    handle=obj.model.backbone.register_forward_hook(lambda *args:calls.append(1))
    try:
        old=probe.endpoint.evaluate_fp32(obj.model,obj.recipe,obj.fixtures,original_flags=obj.flags)
        expected_calls=len(calls);calls.clear()
        observed=probe.evaluate_fp32(obj.model,obj.recipe,obj.fixtures,original_flags=obj.flags)
        assert len(calls)==expected_calls==2
    finally:handle.remove()
    assert {key:observed[key] for key in old}==old
    assert observed['per_pass_observer']['loss_sums_calls']==2
    assert observed['per_pass_observer']['pass_loss_records']==8
    assert 'loss_sums' not in obj.model.__dict__
    for index in range(4):
        row=observed['per_pass']['pass_'+str(index)]
        assert row['counts']==old['counts'] and set(row['loss_means'])=={'ce','latent','kl'}
        assert all(row['loss_means'][term]==row['loss_sums'][term]/row['counts'][term] for term in row['counts'])
    for term in ('ce','latent','kl'):
        weights=(.5,1/6,1/6,1/6) if term=='ce' else (.25,)*4
        reconstructed=sum(observed['per_pass']['pass_'+str(i)]['loss_sums'][term]*weight for i,weight in enumerate(weights))
        assert reconstructed==pytest.approx(old['loss_sums'][term],rel=2e-7,abs=1e-7)
    assert before==probe.bridge.current_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters)


def test_per_pass_observer_restores_existing_instance_method_on_failure(tmp_path,monkeypatch):
    obj=setup(tmp_path)
    original=obj.model.loss_sums
    def custom(*args,**kwargs):return original(*args,**kwargs)
    object.__setattr__(obj.model,'loss_sums',custom)
    def fail(*args,**kwargs):raise OSError('evaluation interruption')
    monkeypatch.setattr(probe.endpoint,'evaluate_fp32',fail)
    with pytest.raises(OSError):probe.evaluate_fp32(obj.model,obj.recipe,obj.fixtures,original_flags=obj.flags)
    assert obj.model.__dict__['loss_sums'] is custom


def test_nonfinite_loss_never_steps_and_missing_gradient_is_visible(tmp_path,monkeypatch):
    obj=setup(tmp_path);fork(obj)
    original=probe.component_backward
    def poison(*args,**kwargs):
        row=original(*args,**kwargs);row['objective']=float('inf');return row
    monkeypatch.setattr(probe,'component_backward',poison)
    before=probe.bridge.current_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters)
    with pytest.raises(FloatingPointError,match='before Adam'):
        probe.update(obj.model,obj.recipe,obj.fixtures,obj.optimizer,obj.scheduler,obj.counters,
            path=probe.FP32,original_flags=obj.flags,metadata=obj.metadata)
    assert before==probe.bridge.current_boundary(obj.model,obj.optimizer,obj.scheduler,obj.counters)
    with pytest.raises(FloatingPointError,match='participate'):probe.norm_groups(obj.model,gradients=True)


def test_cli_fixed_scope_and_checkpoint_path_pins():
    args=['--precision','bf16_mixed','--origin-checkpoint','origin.pt','--origin-checkpoint-sha256','a'*64,
        '--nfr-report','report.json','--nfr-report-sha256','b'*64,'--fixture','fixture.json','--fixture-sha256','c'*64,
        '--output-dir',str(probe.ROOT/'.runtime/test'),'--storage-prefix','gs://fast-chunks/test']
    assert probe.parse_args(args).precision=='bf16_mixed'
    with pytest.raises(SystemExit):probe.parse_args(args+['--updates','32'])
    with pytest.raises(SystemExit):probe.parse_args(args+['--resume','unverified.pt'])
