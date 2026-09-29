"""Saved-state import and real CPU math oracles; capture orchestration is simulated."""
from dataclasses import replace
import json
import time
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import build_campaign_adamw, CampaignTokenSchedule
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch
from scripts import olmo_fusion_startup_packed_bridge as probe
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, prepared_backward
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, BF16
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def tiny(*, packed=True):
    model, recipe, source, ids, eos = construct(SimpleNamespace(scale='tiny', length=8), 'NFR', torch.device('cpu'))
    model.backbone.backbone.attention_precision = 'mixed'
    flags = {name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS}
    if packed:
        recipe, _, _ = probe.packed_transition(model, recipe)
    fixtures = [fixture_for_update(recipe,model.config.model_dim,rank,0,length=8,
        token_ids=ids,eos_id=eos) for rank in (0,1)]
    return model, recipe, source, fixtures, flags


def test_actual_fp32_sparse_prepared_and_old_prepared_helper_agree_with_refills():
    model, recipe, _, fixtures, flags = tiny()
    before = probe.state_pins(model),probe.fixture_pins(fixtures),tree_digests(probe._rng_state(None))
    probe.configure_path(model,flags,FP32)
    with probe.sdpa_kernel(probe.SDPBackend.MATH):
        sparse=probe.component_backward(model,recipe,fixtures,precision='fp32',layout='sparse',objective='combined')
        _,reference=probe.observed_gradients(model,save=True)
        old=prepared_backward(model,recipe,fixtures,precision='fp32')
        old_pins=tree_digests({n:p.grad for n,p in model.named_parameters()})
        model.zero_grad(set_to_none=True)
        runner,batches,noises=probe.make_runner(model,recipe,fixtures,'fp32')
        actual=runner.backward(batches,feedback_noises=noises,replay=False)
        gradients,_=probe.observed_gradients(model,reference)
    assert probe.compare_case(actual,gradients,sparse)['passed']
    assert actual==old and tree_digests({n:p.grad for n,p in model.named_parameters()})==old_pins
    assert runner.warmup_backward_calls==1 and runner.replay_calls==0
    assert torch.equal(runner.adapter.batch.input_ids,batches[-1].input_ids)
    assert torch.equal(runner.adapter.batch.document_ids,batches[-1].document_ids)
    assert all(torch.equal(a,b) for a,b in zip(runner.adapter.feedback_noise,noises[-1]))
    assert runner.adapter.global_counts==actual['counts']
    with probe.sdpa_kernel(probe.SDPBackend.MATH):runner.discard_backward()
    assert probe.gradients_are_zero(model)
    assert before==(probe.state_pins(model),probe.fixture_pins(fixtures),tree_digests(probe._rng_state(None)))


def test_policy_transition_changes_no_parameter_or_runtime_mode_and_keeps_nfr_execution():
    model, recipe, _, _, _ = tiny(packed=False)
    before=probe.state_pins(model)
    packed,data,record=probe.packed_transition(model,recipe)
    assert all(record['checks'].values()) and before==probe.state_pins(model)
    assert packed.mode().rt_mode.selected_layers==(0,1) and data.mode().rt_mode.selected_layers==()
    assert packed.mode().num_passes==data.mode().num_passes==4
    assert record['original_recipe'].get('document_policy','isolated-v1')=='isolated-v1'
    with pytest.raises(ValueError):probe.packed_transition(model,packed)


def test_data_view_noise_equality_checks_actual_execution_recipe_not_only_saved_bytes():
    model,recipe,_,_,_=tiny()
    recipe=replace(recipe,sequence_length=1024);data=replace(recipe,arm='NF')
    fixtures=[];metadata={'records':[]}
    for index in range(2):
        keys=[f'packed-{index}'];ids=torch.full((1,1024),2,dtype=torch.long);valid=torch.ones_like(ids,dtype=torch.bool)
        batch=NextLatBatch(ids,valid,torch.zeros_like(ids))
        noise=probe.feedback_noise_for_rows(data,keys,logical_update=0,sequence_length=1024,
            width=4,physical_batch_size=1,device='cpu')
        fixtures.append(((batch,),(noise,)));metadata['records'].append({'noise_keys':keys})
    assert probe.validate_noise_view(fixtures,metadata,recipe,data,4)['mode_independent_noise_exact']
    fixtures[0][1][0][0][0,0,0]+=1
    with pytest.raises(ValueError,match='jitter'):probe.validate_noise_view(fixtures,metadata,recipe,data,4)
    with pytest.raises(ValueError,match='physical'):probe.validate_noise_view([((),()),fixtures[1]],metadata,recipe,data,4)


def endpoint_fixture(tmp_path):
    model,recipe,source,fixtures,flags=tiny(packed=False)
    metadata=[probe.global_fixture_metadata(model,fixtures)]*4
    optimizer=build_campaign_adamw(model,recipe,fused=False)
    scheduler=CampaignTokenSchedule(optimizer,[m['input_tokens'] for m in metadata],
        warmup_tokens=recipe.warmup_tokens,start_fraction=recipe.warmup_start_fraction)
    counters=TrainingCounters()
    for _ in range(4):
        probe.endpoint.update(model,recipe,fixtures,optimizer,scheduler,counters,
            path=FP32,original_flags=flags,metadata=metadata)
    sources={'test/source.py':'b'*64}
    selections=[{'source_training_index':144+i,'start_cursor':{'next_update':144+i},
        'next_cursor':{'next_update':145+i},**metadata[i]} for i in range(4)]
    common={'schema':probe.endpoint.SCHEMA,'plan':tree_digests(probe.endpoint.PLAN),'recipe':recipe.to_dict(),
        'sources':sources,'origin_checkpoint_sha256':'a'*64,'source_checkpoint':source,
        'data_manifest_sha256':'c'*64,'training_metadata':metadata}
    fingerprint={'checkpoint_sha256':'a'*64,'base':source,'sources':sources}
    receipt=probe.endpoint.save_endpoint(tmp_path/'endpoint.pt',model,optimizer,scheduler,counters,
        configuration=probe.endpoint.checkpoint_configuration(common,BF16),source_fingerprint=fingerprint,
        data_cursor=selections[-1]['next_cursor'])
    report={'schema':probe.endpoint.SCHEMA,'status':'passed_bounded_functionality','passed':True,
        'optimizer_calls':8,'plan':tree_digests(probe.endpoint.PLAN),'integrity':{'exact':True},
        'determinism':{'deterministic_algorithms':True},'sources':sources,'recipe':recipe.to_dict(),
        'checkpoints':[{'trajectory':BF16,**receipt,'gcs':{'verified':True}}],
        'rows':[{'path':p,'update':i} for i in range(1,5) for p in (FP32,BF16)],
        'checkpoint_configuration':common,'training_metadata':metadata,'training_data_selections':selections,
        'origin_checkpoint_sha256':'a'*64,'data_manifest_sha256':'c'*64,
        'final_boundary_pins':{BF16:probe.current_boundary(model,optimizer,scheduler,counters)}}
    assert report['final_boundary_pins'][BF16]==tree_digests(probe.endpoint.capture_boundary(model,optimizer,scheduler,counters,metadata))
    return model,recipe,source,receipt,report


def test_strict_full_endpoint_import_restores_exact_boundary_then_releases_adam(tmp_path):
    model,recipe,source,receipt,authority=endpoint_fixture(tmp_path)
    fresh,fresh_recipe,fresh_source,_,_=tiny(packed=False)
    identities={n:id(p) for n,p in fresh.named_parameters()}
    info=probe.import_endpoint(fresh,fresh_recipe,fresh_source,tmp_path/'endpoint.pt',receipt['sha256'],authority)
    assert all(info['checks'].values()) and info['counters']['optimizer_updates']==4
    assert probe.state_pins(fresh)==probe.state_pins(model)
    assert identities=={n:id(p) for n,p in fresh.named_parameters()}
    assert all(p.grad is None for p in fresh.parameters())


@pytest.mark.parametrize('mutation',['cursor','ownership','step','policy'])
def test_import_rejects_invalid_saved_contract_before_model_mutation(tmp_path,mutation):
    _,_,_,receipt,authority=endpoint_fixture(tmp_path)
    payload=torch.load(tmp_path/'endpoint.pt',map_location='cpu',weights_only=True)
    if mutation=='cursor':payload['data_cursor']['next_update']+=1
    elif mutation=='ownership':payload['optimizer_ownership'][0][0]='foreign'
    elif mutation=='step':payload['counters']['optimizer_updates']=3
    else:payload['configuration']['recipe']['document_policy']='continuous-stream-v1'
    torch.save(payload,tmp_path/'bad.pt')
    fresh,recipe,source,_,_=tiny(packed=False);before=probe.state_pins(fresh)
    with pytest.raises((ValueError,AssertionError)):
        probe.import_endpoint(fresh,recipe,source,tmp_path/'bad.pt',sha256_file(tmp_path/'bad.pt'),authority)
    assert probe.state_pins(fresh)==before


@pytest.mark.parametrize('mutation',[None,'source','receipt','configuration','metadata','missing_row','policy'])
def test_report_authority_is_pinned_completed_and_internally_consistent(tmp_path,mutation):
    _,_,_,receipt,authority=endpoint_fixture(tmp_path)
    if mutation=='source':authority['sources']['new']='d'*64
    elif mutation=='receipt':authority['checkpoints'][0]['sha256']='e'*64
    elif mutation=='configuration':authority['checkpoint_configuration']['origin_checkpoint_sha256']='e'*64
    elif mutation=='metadata':authority['training_data_selections'][0]['input_tokens']+=1
    elif mutation=='missing_row':authority['rows'].pop()
    elif mutation=='policy':authority['recipe']['document_policy']='continuous-stream-v1'
    path=tmp_path/'authority.json';path.write_text(json.dumps(authority))
    kwargs=dict(checkpoint_sha256=receipt['sha256'],sources={'test/source.py':'b'*64})
    if mutation is None:
        assert probe.load_authority(path,sha256_file(path),**kwargs)['passed']
    else:
        with pytest.raises(ValueError):probe.load_authority(path,sha256_file(path),**kwargs)


def test_missing_gradient_is_rejected_before_observer_materializes_zeros():
    model,_,_,_,_=tiny()
    with pytest.raises(AssertionError,match='lost'):probe.observed_gradients(model)
    assert all(p.grad is None for p in model.parameters())


def test_runtime_import_checks_stack_hardware_and_determinism_allowing_device_count_only():
    runtime={'torch':'fixed','cuda':'fixed','gpu':'H100','nvidia_smi':'H100, 80GB, driver1'}
    determinism={'deterministic_algorithms':True}
    authority={'runtime':dict(runtime,nvidia_smi=runtime['nvidia_smi']+'\n'+runtime['nvidia_smi']),
        'determinism':determinism}
    assert all(probe.runtime_contract(runtime,determinism,authority).values())
    with pytest.raises(ValueError):probe.runtime_contract(dict(runtime,gpu='H200'),determinism,authority)
    with pytest.raises(ValueError):probe.runtime_contract(dict(runtime,nvidia_smi='H100, 80GB, driver2'),determinism,authority)
    with pytest.raises(ValueError):probe.runtime_contract(runtime,{'deterministic_algorithms':False},authority)


def test_five_case_accounting_and_cleanup_with_real_math_and_simulated_capture(monkeypatch):
    """Tests orchestration only; deliberately replaces BF16/CUDA with FP32 CPU math."""
    model,recipe,_,fixtures,flags=tiny()
    configure=probe.configure_path;factory=probe.make_runner
    monkeypatch.setattr(probe,'configure_path',lambda m,f,p:configure(m,f,FP32))
    runners=[]
    def make_runner(model,recipe,fixtures,precision):
        runner,batches,noises=factory(model,recipe,fixtures,'fp32');runners.append(runner)
        backward=runner.backward
        def simulated_backward(*args,replay=False,**kwargs):
            result=backward(*args,replay=False,**kwargs)
            if replay:runner.replay_calls+=len(args[0])
            return result
        def simulated_capture(*,warmup,phase_observer):
            for name in ('gradient_initialization','warmup','capture'):
                phase_observer(name,'begin')
                if name=='gradient_initialization':runner.initialize_gradients()
                else:
                    for _ in range(warmup if name=='warmup' else 1):
                        runner.zero_grad();runner._tensor_backward()
                        if name=='warmup':runner.warmup_backward_calls+=1
                        else:runner.capture_backward_calls+=1
                phase_observer(name,'end')
            runner.zero_grad();runner.graph=object()
        runner.backward=simulated_backward;runner.capture=simulated_capture
        return runner,batches,noises
    monkeypatch.setattr(probe,'make_runner',make_runner)
    rows=[];phases=[]
    result=probe.run_bridge(model,recipe,fixtures,original_flags=flags,publish=rows.append,
        phase=lambda name,details:phases.append(name),deadline=time.monotonic()+1000)
    assert [r['case'] for r in rows]==list(probe.CASES)
    assert all(r['passed'] for r in rows) and all(r['primary_comparison']['passed'] for r in rows[1:])
    assert result['setup_counts']=={'initialization_backward_calls':2,'warmup_backward_calls':10,
        'capture_backward_calls':1,'measured_physical_backward_calls':10,'measured_replay_calls':2}
    assert rows[-1]['bitwise_gradients_exact'] and rows[-1]['bitwise_metrics_exact']
    assert all(p.grad is None for p in model.parameters()) and all(r.graph is None for r in runners)
    assert 'capture/warmup' in phases


def test_deadline_stops_before_any_backward_and_restores_flags():
    model,recipe,_,fixtures,flags=tiny();rows=[]
    with pytest.raises(TimeoutError):
        probe.run_bridge(model,recipe,fixtures,original_flags=flags,publish=rows.append,
            phase=lambda *args:None,deadline=time.monotonic()-1)
    assert not rows and all(p.grad is None for p in model.parameters())
    assert all(getattr(model.backbone.backbone,k)==v for k,v in flags.items())


def test_cli_refuses_unpinned_or_alternate_fixture_and_extra_experiments():
    args=['--checkpoint','/tmp/a','--checkpoint-sha256','a'*64,'--nfr-report','/tmp/b',
        '--nfr-report-sha256','b'*64,'--fixture','/tmp/c','--fixture-sha256',probe.FIXTURE_SHA,
        '--output-dir',str(probe.ROOT/'.runtime/test')]
    assert probe.parse_args(args).fixture_sha256==probe.FIXTURE_SHA
    with pytest.raises(SystemExit):probe.parse_args(args+['--steps','10'])
    args[args.index(probe.FIXTURE_SHA)]='c'*64
    with pytest.raises(SystemExit):probe.parse_args(args)
