"""Literal multi-group Adam and same-state gradient controls; CPU only."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from scripts import olmo_optimizer_history_probe as probe
from scripts.olmo_lm_common import tree_digests


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)


def simple():
    model=torch.nn.Module();model.backbone=torch.nn.Module();model.backbone.fusion=torch.nn.Module();model.predictor=torch.nn.Module()
    for owner in (model.backbone,model.backbone.fusion,model.predictor):
        owner.register_parameter('weight',torch.nn.Parameter(torch.tensor([.02,-.04,1.,-2.,.3,.0])))
    groups=[]
    for i,(name,p) in enumerate(model.named_parameters()):
        groups.append({'params':[p],'param_names':[name],'weight_decay':.1 if i else 0.})
    optimizer=torch.optim.AdamW(groups,lr=2.17e-5,betas=(.9,.95),eps=1e-5,foreach=False,fused=False)
    for i,p in enumerate(model.parameters()):
        optimizer.state[p]={'step':torch.tensor(20.),'exp_avg':torch.tensor([1e-7,-2e-5,.1,-.01,0.,.001])*(i+1),
            'exp_avg_sq':torch.tensor([1e-12,1e-9,.01,.001,1e-8,.0001])*(i+1)}
    fp=torch.tensor([1e-7,1e-4,-.03,.04,0.,-1e-7]);bf=torch.tensor([-2e-7,1.01e-4,-.028,.04,1e-8,0.])
    clipped={probe.FP32:{n:fp.clone() for n,_ in model.named_parameters()},probe.BF16:{n:bf.clone() for n,_ in model.named_parameters()}}
    return model,optimizer,{'optimizer':copy.deepcopy(optimizer.state_dict())},clipped


def literal(model,optimizer,clipped,history,precision):
    candidate=copy.deepcopy(model);by_name=dict(candidate.named_parameters())
    groups=[]
    for saved in optimizer.state_dict()['param_groups']:
        group={k:copy.deepcopy(v) for k,v in saved.items() if k!='params'}
        group['params']=[by_name[n] for n in saved['param_names']];groups.append(group)
    opt=torch.optim.AdamW(groups,foreach=False,fused=False)
    if history=='inherited':opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for name,p in candidate.named_parameters():p.grad=torch.zeros_like(p) if precision=='zero' else clipped[precision][name].clone()
    original={n:p.detach().clone() for n,p in candidate.named_parameters()}
    opt.step()
    return {n:p.detach().double()-original[n].double() for n,p in candidate.named_parameters()},opt


@pytest.mark.parametrize('history',['inherited','reset'])
@pytest.mark.parametrize('precision',[probe.FP32,probe.BF16,'zero'])
def test_streamed_candidate_matches_literal_full_multigroup_adam_without_mutation(history,precision):
    model,opt,payload,clipped=simple();before=tree_digests({'model':model.state_dict(),'optimizer':opt.state_dict(),'payload':payload})
    expected,literal_opt=literal(model,opt,clipped,history,precision)
    for group in payload['optimizer']['param_groups']:
        for identifier,name in zip(group['params'],group['param_names']):
            p=dict(model.named_parameters())[name];gradient=torch.zeros_like(p) if precision=='zero' else clipped[precision][name]
            got,denominator=probe.adam_candidate(p,gradient,payload['optimizer']['state'][identifier],group,reset=history=='reset')
            assert torch.equal(got,expected[name])
            assert torch.isfinite(denominator).all()
    assert before==tree_digests({'model':model.state_dict(),'optimizer':opt.state_dict(),'payload':payload})
    assert all(float(s['step'])==(21 if history=='inherited' else 1) for s in literal_opt.state.values())


def test_counterfactual_geometry_and_epsilon_statistics_match_independent_concatenation():
    model,opt,payload,clipped=simple();before=tree_digests({'model':model.state_dict(),'optimizer':opt.state_dict(),'payload':payload,'clipped':clipped})
    result=probe.optimizer_counterfactuals(model,payload,clipped)
    values={h:{p:literal(model,opt,clipped,h,p)[0] for p in (probe.FP32,probe.BF16,'zero')} for h in ('inherited','reset')}
    for history in ('inherited','reset'):
        for label,control in [('delta',None),('delta_minus_zero_adam','zero')]:
            vectors={p:torch.cat([(v-values[history][control][n] if control else v).flatten() for n,v in values[history][p].items()]) for p in (probe.FP32,probe.BF16)}
            difference=torch.linalg.vector_norm(vectors[probe.BF16]-vectors[probe.FP32]).item()
            refnorm=torch.linalg.vector_norm(vectors[probe.FP32]).item()
            row=result['geometry'][history+'/'+label]['all']
            assert row['difference_norm']==pytest.approx(difference,abs=1e-15)
            assert row['relative_l2']==pytest.approx(difference/refnorm)
    allstats=result['epsilon_and_sign']['all'];fp=torch.cat(list(clipped[probe.FP32].values()));bf=torch.cat(list(clipped[probe.BF16].values()))
    assert allstats['sign_flips']==int(((fp>0)&(bf<0)|(fp<0)&(bf>0)).sum())
    assert allstats['sign_mismatches']==int((fp.sign()!=bf.sign()).sum())
    assert allstats['fp32_abs_le_eps']==int((fp.abs()<=1e-5).sum())
    assert result['parameterwise_adam_calls']==18 and result['conceptual_full_model_candidates']==6
    assert before==tree_digests({'model':model.state_dict(),'optimizer':opt.state_dict(),'payload':payload,'clipped':clipped})


def test_zero_gradient_control_is_not_missing_gradient_and_reset_control_equals_decay():
    model,opt,payload,clipped=simple()
    result=probe.optimizer_counterfactuals(model,payload,clipped)
    assert result['geometry']['inherited/zero_adam']['all']['reference_gradient_norm']>result['geometry']['reset/zero_adam']['all']['reference_gradient_norm']
    assert result['geometry']['reset/zero_adam']==result['geometry']['reset/decay']
    assert result['geometry']['inherited/delta']['all']['difference_norm']==pytest.approx(result['geometry']['inherited/delta_minus_zero_adam']['all']['difference_norm'])


@pytest.mark.parametrize('mutation',['epsilon','beta','step','fused','missing_gradient'])
def test_invalid_adam_contract_rejected_without_touching_live_state(mutation):
    model,opt,payload,clipped=simple();before=tree_digests(model.state_dict())
    if mutation=='epsilon':payload['optimizer']['param_groups'][0]['eps']=1e-8
    elif mutation=='beta':payload['optimizer']['param_groups'][0]['betas']=(.8,.95)
    elif mutation=='step':payload['optimizer']['state'][0]['step'].fill_(19)
    elif mutation=='fused':payload['optimizer']['param_groups'][0]['fused']=True
    else:clipped[probe.FP32].pop(next(iter(clipped[probe.FP32])))
    with pytest.raises(ValueError):probe.optimizer_counterfactuals(model,payload,clipped)
    assert tree_digests(model.state_dict())==before


def test_same_tiny_nfr_gradient_measurement_uses_actual_global_clipping_and_preserves_state():
    from test_fusion_startup_packed_bridge import tiny
    model,recipe,source,fixtures,flags=tiny(packed=False)
    # Tiny construction uses eager attention; production bridge requires mixed masters.
    flags['attention_precision']='mixed';model.backbone.backbone.attention_precision='mixed'
    state=tree_digests(model.state_dict());rng=tree_digests(probe._rng_state(None));calls=[]
    rows,clipped,geometry=probe.measure_gradients(model,recipe,fixtures,flags,observe=lambda label,row:calls.append(label))
    assert calls==[probe.FP32,probe.BF16] and len(rows)==2
    assert tree_digests(model.state_dict())==state and tree_digests(probe._rng_state(None))==rng
    assert all(p.grad is None for p in model.parameters())
    assert {n:getattr(model.backbone.backbone,n) for n in probe.RUNTIME_FLAGS}==flags
    execution=probe.configure_path(model,flags,probe.FP32)
    with probe.sdpa_kernel(probe.SDPBackend.MATH),torch.autocast('cpu',enabled=False):
        expected=probe.component_backward(model,recipe,fixtures,precision='fp32',layout='sparse',objective='combined')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe.max_grad_norm,error_if_nonfinite=True,foreach=False)
    assert rows[0]['metrics']==expected and rows[0]['global_norm_before_clip']==float(norm)
    for name,p in model.named_parameters():assert torch.equal(clipped[probe.FP32][name],p.grad)
    assert geometry['raw_gradient']['all']['reference_gradient_norm']>0


def test_measurement_failure_restores_flags_and_clears_gradients(monkeypatch):
    from test_fusion_startup_packed_bridge import tiny
    model,recipe,source,fixtures,flags=tiny(packed=False);flags['attention_precision']='mixed'
    model.backbone.backbone.attention_precision='mixed'
    def fail(*args,**kwargs):
        next(model.parameters()).grad=torch.ones_like(next(model.parameters()))
        raise RuntimeError('injected')
    monkeypatch.setattr(probe,'component_backward',fail)
    with pytest.raises(RuntimeError,match='injected'):probe.measure_gradients(model,recipe,fixtures,flags)
    assert all(p.grad is None for p in model.parameters())
    assert {n:getattr(model.backbone.backbone,n) for n in probe.RUNTIME_FLAGS}==flags


def test_authority_rejects_altered_source_or_checkpoint_before_tensor_load(tmp_path):
    from test_fusion_startup_nfr_compare import reports
    original,fp,bf=reports();sources={**bf['sources'],**{n:'f'*64 for n in probe.comparison.DEPENDENCIES}}
    original['sources'].update({n:'f'*64 for n in probe.comparison.DEPENDENCIES})
    for r in (fp,bf):r['sources'].update(original['sources'])
    # Tests use explicit small report authorities; native CLI independently pins production bytes.
    paths={}
    for name,data in [('origin_report',original),('fp32_report',fp),('bf16_report',bf)]:
        path=tmp_path/(name+'.json');probe.write_json(path,data);paths[name]=path;paths[name+'_sha256']=probe.sha256_file(path)
    for r in (fp,bf):r['source_fingerprint']['nfr_report_sha256']=paths['origin_report_sha256']
    for name,data in [('fp32_report',fp),('bf16_report',bf)]:
        probe.write_json(paths[name],data);paths[name+'_sha256']=probe.sha256_file(paths[name])
    args=SimpleNamespace(**paths,checkpoint_sha256='b'*64,fixture_sha256='d'*64)
    with pytest.raises(ValueError,match='Checkpoint or fixture'):probe.load_authorities(args,sources)
    args.checkpoint_sha256='a'*64;sources['original.py']='x'*64
    with pytest.raises(ValueError,match='sources'):probe.load_authorities(args,sources)


def test_cli_rejects_foreign_checkpoint_pin_before_execution(tmp_path):
    argv=[]
    for name,digest in probe.PINS.items():argv+=['--'+name.replace('_','-'),str(tmp_path/name),'--'+name.replace('_','-')+'-sha256',digest]
    argv+=['--output-dir',str(probe.ROOT/'.runtime/unused-optimizer-probe-test')]
    assert probe.parse_args(argv).checkpoint_sha256==probe.PINS['checkpoint']
    argv[argv.index('--checkpoint-sha256')+1]='a'*64
    with pytest.raises(SystemExit):probe.parse_args(argv)
