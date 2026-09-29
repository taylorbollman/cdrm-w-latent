#!/usr/bin/env python3
"""Two fixed-state NFR backwards and streamed Adam-history counterfactuals."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_recipe import build_campaign_adamw, CampaignTokenSchedule
from cdrm.pretrained.lm_training import TrainingCounters, _rng_state, _restore_rng, parameter_layout, optimizer_ownership
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct, global_fixture_metadata
from scripts.olmo_campaign_precision_bridge import configure_path
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, component_backward, gradient_geometry, _geometry_finish
from scripts.olmo_campaign_probe import component, memory
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, fixture_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_long_probe import load_long_fixture
from scripts import olmo_fusion_startup_nfr_updates as endpoint
from scripts import olmo_fusion_startup_nfr_continue as continuation
from scripts import olmo_fusion_startup_nfr_compare as comparison
from scripts.olmo_fusion_startup_packed_bridge import current_boundary, runtime_contract
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = 'olmo-optimizer-history-probe-v1'
PINS = {
    'origin_report': 'f24c6035f9035189bcb9fefd8ee12ceb42d3f249aca7c7221beb175d3f0da9dd',
    'fp32_report': '6285917a106a6337a6880172167a4dbbe228cfcc6f369d57e0b686593f88ea9f',
    'bf16_report': 'cef1136ca0e75d77cfd0d840eace2fc44ad267f9cc5ef96b0232ae7f6ccc400d',
    'checkpoint': 'bbd00dc1b70e08cae516ab4bb0f491ff03884efcd6e2d8b92558bf45ddbd7dd6',
    'fixture': '830920f60c687f667baee7f7d6f137b521b22a35604f02e2a2e3b038586e55f9',
}
OWN = ('scripts/olmo_optimizer_history_probe.py', 'tests/test_optimizer_history_probe.py',
       'docs/reports/olmo-optimizer-history/protocol.md')
GROUPS = ('all', 'backbone', 'fusion', 'predictor')
GEOMETRY_FIELDS = ('reference_squared', 'actual_squared', 'error_squared', 'dot', 'parameter_tensors')
PLAN = {'aggregate_backwards': 2, 'physical_backwards': 4, 'conceptual_optimizer_candidates': 6,
        'new_training_updates': 0, 'new_checkpoints': 0, 'saved_update': 20,
        'fixture': 'fixed isolated B2/T128; two records; K4 beta1 jitter0.02; combined loss',
        'optimizer': 'CUDA FP32 AdamW foreach=False fused=False; identical saved group LR/settings',
        'history': 'twenty adaptation updates; not original OLMo pretraining moments',
        'soft_seconds': 2700}


def source_hashes():
    result = continuation.source_hashes()
    for name, digest in comparison.source_hashes().items():
        if name in result and result[name] != digest:
            raise ValueError('Frozen inherited source inventories disagree')
        result[name] = digest
    result.update({name: sha256_file(ROOT/name) for name in OWN})
    return dict(sorted(result.items()))


def load_authorities(args, sources):
    reports = {name: comparison.read_report(getattr(args, name), getattr(args, name+'_sha256'))
               for name in ('origin_report', 'fp32_report', 'bf16_report')}
    origin, fp, bf = (reports[name] for name in ('origin_report', 'fp32_report', 'bf16_report'))
    checks = comparison.report_controls(origin, fp, bf, current_sources=sources,
        origin_sha256=args.origin_report_sha256)
    if any(sources.get(name) != digest for name, digest in bf['sources'].items()):
        raise ValueError('Historical continuation sources changed')
    record = comparison.retained_record(bf, 20)
    if record['sha256'] != args.checkpoint_sha256 or bf['configuration']['fixture_sha256'] != args.fixture_sha256:
        raise ValueError('Checkpoint or fixture differs from the pinned continuation')
    return origin, bf, record, checks


def import_boundary(model, recipe, source, payload, report):
    """Validate the exact historical state before copying into current objects."""
    config = report['configuration']
    expected = {'recipe': tree_digests(recipe.to_dict()), 'model_config': tree_digests(model.backbone.config.to_dict()),
        'nextlat_config': tree_digests(model.config.to_dict()), 'fusion_config': tree_digests(model.backbone.fusion_config.to_dict()),
        'ownership': parameter_layout(model), 'module_modes': {n:m.training for n,m in model.named_modules()},
        'production_flags': {n:getattr(model.backbone.backbone,n) for n in RUNTIME_FLAGS},
        'frozen_buffer_pins': tree_digests(dict(model.named_buffers()))}
    if (any(config.get(k) != v for k,v in expected.items())
            or tree_digests(source) != report['source_fingerprint']['base']
            or recipe.arm != 'NFR' or recipe.document_policy != 'isolated-v1'
            or recipe.epsilon != 1e-5 or recipe.betas != (.9,.95) or recipe.max_grad_norm != 1.
            or payload['data_cursor'] != report['final_cursor']):
        raise ValueError('Current model/recipe/source differs from historical NFR before import')
    optimizer = build_campaign_adamw(model, recipe, fused=False)
    metadata = config['training_metadata']
    scheduler = CampaignTokenSchedule(optimizer, [r['input_tokens'] for r in metadata],
        warmup_tokens=recipe.warmup_tokens, start_fraction=recipe.warmup_start_fraction)
    snapshot = {'schema':endpoint.SCHEMA, 'model':payload['model'], 'optimizer':payload['optimizer'],
        'scheduler':payload['scheduler'], 'rng':payload['rng'], 'counters':payload['counters'],
        'ownership':payload['optimizer_ownership'], 'layout':payload['parameter_layout'], 'modes':payload['module_training']}
    endpoint.validate_boundary(model,optimizer,scheduler,snapshot,metadata)
    identities = {n:id(p) for n,p in model.named_parameters()}
    model.load_state_dict(payload['model'],strict=True,assign=False)
    # CUDA loading creates new device moments; CPU tests clone to avoid aliasing.
    saved_optimizer = endpoint.cpu_copy(payload['optimizer']) if next(model.parameters()).device.type == 'cpu' else payload['optimizer']
    optimizer.load_state_dict(saved_optimizer)
    scheduler.load_state_dict(copy.deepcopy(payload['scheduler']))
    _restore_rng(payload['rng'],None)
    counters = TrainingCounters(**payload['counters'])
    if (counters.optimizer_updates != 20 or identities != {n:id(p) for n,p in model.named_parameters()}
            or current_boundary(model,optimizer,scheduler,counters) != report['final_boundary']
            or model.backbone.readout_weight is not model.backbone.token_embeddings.weight):
        raise AssertionError('Imported complete update20 boundary is not exact')
    continuation.validate_clock(model,optimizer,scheduler,counters,metadata)
    return optimizer,scheduler,counters


def measure_gradients(model, recipe, fixtures, flags, *, observe=lambda label,row:None):
    """Retain two clipped CPU gradients, only one temporary raw CPU reference."""
    inputs = fixture_pins(fixtures); rng = tree_digests(_rng_state(None))
    expected = global_fixture_metadata(model,fixtures)
    rows, clipped, raw_reference = [], {}, None
    comparison_raw = None
    try:
        for path in (FP32,BF16):
            execution = configure_path(model,flags,path)
            device = next(model.parameters()).device
            backend = SDPBackend.FLASH_ATTENTION if path == BF16 and device.type == 'cuda' else SDPBackend.MATH
            with sdpa_kernel(backend),torch.autocast(device.type,enabled=False):
                metrics = component_backward(model,recipe,fixtures,precision=execution['precision'],layout='sparse',objective='combined')
            if (any(metrics[k] != v for k,v in expected.items()) or
                    not all(math.isfinite(v) for v in (metrics['objective'],*metrics['loss_sums'].values()))):
                raise FloatingPointError('Objective counts or finiteness differs')
            live = {n:p.grad for n,p in model.named_parameters()}
            if any(v is None or v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in live.values()):
                raise FloatingPointError('Missing or invalid full-model gradient')
            raw_norms = continuation.norm_groups(model,gradients=True)
            if path == FP32:
                raw_reference = {n:v.detach().cpu().clone() for n,v in live.items()}
            else:
                comparison_raw = gradient_geometry(live,raw_reference)
                del raw_reference
                raw_reference = None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),recipe.max_grad_norm,error_if_nonfinite=True,foreach=False)
            clipped[path] = {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
            rows.append({'path':path,'execution':execution,'metrics':metrics,'raw_norms':raw_norms,
                'global_norm_before_clip':float(norm),'clip_scale':float(torch.clamp(recipe.max_grad_norm/(norm+1e-6),max=1.)),
                'physical_backwards':expected['microbatches']})
            model.zero_grad(set_to_none=True)
            if tree_digests(_rng_state(None)) != rng or fixture_pins(fixtures) != inputs:
                raise AssertionError('Gradient measurement changed RNG or fixed input')
            observe(path,rows[-1])
        return rows,clipped,{'raw_gradient':comparison_raw,
            'clipped_gradient':gradient_geometry(clipped[BF16],clipped[FP32])}
    finally:
        model.zero_grad(set_to_none=True)
        for name,value in flags.items():setattr(model.backbone.backbone,name,value)


def adam_candidate(weight, gradient, state, group, *, reset):
    """Actual stored FP32 update for one independent canonical parameter."""
    if (weight.dtype != torch.float32 or gradient.shape != weight.shape or gradient.dtype != weight.dtype
            or group.get('fused') is not False or group.get('foreach') is not False
            or group.get('amsgrad',False) or group.get('maximize',False) or group.get('capturable',False)
            or group.get('differentiable',False)):
        raise ValueError('Unsupported counterfactual Adam arithmetic')
    p = torch.nn.Parameter(weight.detach().clone())
    optimizer = torch.optim.AdamW([p],lr=group['lr'],betas=tuple(group['betas']),eps=group['eps'],
        weight_decay=group['weight_decay'],foreach=False,fused=False)
    if not reset:
        optimizer.state[p] = {'step':state['step'].detach().clone(),
            'exp_avg':state['exp_avg'].detach().to(weight.device,copy=True),
            'exp_avg_sq':state['exp_avg_sq'].detach().to(weight.device,copy=True)}
    p.grad = gradient.detach().clone()
    optimizer.step()
    current = optimizer.state[p]
    wanted = 1 if reset else float(state['step'])+1
    if float(current['step']) != wanted or not bool(torch.isfinite(p).all()):
        raise FloatingPointError('Candidate Adam step or finiteness differs')
    delta = p.detach().double()-weight.detach().double()
    denominator = (current['exp_avg_sq']/(1-group['betas'][1]**wanted)).sqrt()
    return delta,denominator


def geometry_accumulator():
    return {group:dict.fromkeys(GEOMETRY_FIELDS,0.) for group in GROUPS}


def add_geometry(result,name,actual,reference):
    a,b = actual.double(),reference.double()
    values = {'reference_squared':float(b.square().sum()),'actual_squared':float(a.square().sum()),
        'error_squared':float((a-b).square().sum()),'dot':float((a*b).sum()),'parameter_tensors':1}
    for group in ('all',component(name)):
        for key,value in values.items():result[group][key] += value


@torch.no_grad()
def optimizer_counterfactuals(model,payload,clipped,*,observe=lambda label,row:None):
    """Six candidate maps streamed per parameter; never step the live model."""
    names = dict(model.named_parameters())
    if any(set(v) != set(names) for v in clipped.values()) or set(clipped) != {FP32,BF16}:
        raise ValueError('Candidate gradient ownership differs')
    binding = {}
    for group in payload['optimizer']['param_groups']:
        for identifier,name in zip(group['params'],group['param_names']):
            if name in binding:raise ValueError('Duplicate Adam ownership')
            binding[name] = group,payload['optimizer']['state'][identifier]
    if set(binding) != set(names):raise ValueError('Saved optimizer owner inventory differs')
    labels = [f'{history}/{quantity}' for history in ('inherited','reset') for quantity in
        ('delta','delta_minus_decay','delta_minus_zero_adam','zero_adam','decay')]
    labels += ['history_effect/fp32_delta','history_effect/bf16_delta']
    geometry = {label:geometry_accumulator() for label in labels}
    fields = ('coordinates','sign_flips','sign_mismatches','fp32_abs_le_eps','bf16_abs_le_eps',
        'fp32_gradient_squared','sign_flip_gradient_squared','delta_difference_squared','sign_flip_delta_difference_squared',
        'inherited_denominator_le_eps','reset_denominator_le_eps',
        'inherited_update_squared','inherited_eps_dominated_update_squared',
        'reset_update_squared','reset_eps_dominated_update_squared')
    stats = {group:dict.fromkeys(fields,0.) for group in GROUPS}
    calls = 0
    for name,weight in names.items():
        group,state = binding[name]
        if (tuple(group['betas']) != (.9,.95) or group['eps'] != 1e-5 or float(state['step']) != 20
                or not math.isfinite(group['lr']) or group['lr'] <= 0):
            raise ValueError('Counterfactual changed saved epsilon/betas/update20/current LR')
        fp,bf = (clipped[path][name].to(weight.device) for path in (FP32,BF16))
        zero = torch.zeros_like(weight)
        # PyTorch decoupled decay executes this FP32 multiply before Adam.
        decay = (weight.detach()*(1-group['lr']*group['weight_decay'])).double()-weight.detach().double()
        deltas = {}; denominators = {}
        for history in ('inherited','reset'):
            values = {}
            for precision,gradient in (('fp32',fp),('bf16',bf),('zero',zero)):
                delta,denominator = adam_candidate(weight,gradient,state,group,reset=history=='reset')
                values[precision] = delta; calls += 1
                if precision == 'fp32':denominators[history] = denominator
            deltas[history] = values
            for label,actual,reference in (
                ('delta',values['bf16'],values['fp32']),
                ('delta_minus_decay',values['bf16']-decay,values['fp32']-decay),
                ('delta_minus_zero_adam',values['bf16']-values['zero'],values['fp32']-values['zero']),
                ('zero_adam',values['zero'],values['zero']),('decay',decay,decay)):
                add_geometry(geometry[history+'/'+label],name,actual,reference)
        for precision in ('fp32','bf16'):
            add_geometry(geometry['history_effect/'+precision+'_delta'],name,deltas['reset'][precision],deltas['inherited'][precision])
        flips = (fp>0)&(bf<0)|(fp<0)&(bf>0)
        diff = deltas['reset']['bf16']-deltas['reset']['fp32']
        values = {'coordinates':fp.numel(),'sign_flips':int(flips.sum()),
            'sign_mismatches':int((fp.sign()!=bf.sign()).sum()),'fp32_abs_le_eps':int((fp.abs()<=group['eps']).sum()),
            'bf16_abs_le_eps':int((bf.abs()<=group['eps']).sum()),'fp32_gradient_squared':float(fp.double().square().sum()),
            'sign_flip_gradient_squared':float(fp[flips].double().square().sum()),
            'delta_difference_squared':float(diff.square().sum()),'sign_flip_delta_difference_squared':float(diff[flips].square().sum())}
        for history in ('inherited','reset'):
            mask = denominators[history] <= group['eps']; change = deltas[history]['fp32']
            values[history+'_denominator_le_eps'] = int(mask.sum())
            values[history+'_update_squared'] = float(change.square().sum())
            values[history+'_eps_dominated_update_squared'] = float(change[mask].square().sum())
        for key in ('all',component(name)):
            for field,value in values.items():stats[key][field] += value
        del fp,bf,zero,decay,deltas,denominators,values,diff,flips,change,mask,delta,denominator
    finished = {label:{g:_geometry_finish(row) for g,row in groups.items()} for label,groups in geometry.items()}
    for group,row in stats.items():
        count = row['coordinates']
        for name in ('sign_flips','sign_mismatches','fp32_abs_le_eps','bf16_abs_le_eps','inherited_denominator_le_eps','reset_denominator_le_eps'):
            row[name+'_fraction'] = row[name]/count if count else None
        for numerator,denominator in (
            ('sign_flip_gradient_squared','fp32_gradient_squared'),
            ('sign_flip_delta_difference_squared','delta_difference_squared'),
            ('inherited_eps_dominated_update_squared','inherited_update_squared'),
            ('reset_eps_dominated_update_squared','reset_update_squared')):
            row[numerator+'_fraction'] = row[numerator]/row[denominator] if row[denominator] else None
    for label,row in finished.items():observe(label,row)
    return {'geometry':finished,'epsilon_and_sign':stats,'parameterwise_adam_calls':calls,
        'conceptual_full_model_candidates':6,'canonical_parameter_tensors':len(names),
        'epsilon':1e-5,'denominator_scope':'Bias-corrected sqrt(v) after the FP32 candidate, excluding epsilon; count and FP32 delta-energy fractions',
        'sign_scope':'Clipped-gradient strict sign flips; zero/nonzero differences separately included in sign_mismatches; delta_difference_squared and its sign-flip fraction describe RESET history only',
        'control_scope':'Delta minus explicit-zero Adam is diagnostic, not a linear causal decomposition; decay is the actual FP32 decay multiply'}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in PINS:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
        p.add_argument('--'+name.replace('_','-')+'-sha256',required=True)
    p.add_argument('--artifacts',type=Path,default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    for name,digest in PINS.items():
        if getattr(args,name+'_sha256')!=digest:p.error('Only the declared retained update20 authorities are supported: '+name)
    args.output_dir=args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT) or args.output_dir.exists():p.error('Use a new persistent project evidence directory')
    return args


def main(argv=None):
    args=parse_args(argv); sources=source_hashes()
    origin,authority,receipt,checks=load_authorities(args,sources)
    determinism=configure_determinism(True);runtime=require_container_gpu()
    if torch.distributed.is_initialized():raise RuntimeError('One GPU diagnostic process, no DDP')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');started=time.monotonic()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name in sources:
        target=args.output_dir/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    report={'schema':SCHEMA,'status':'running','passed':False,'sources':sources,'plan':PLAN,'authority_checks':checks,
        'authority_pins':PINS,'runtime':runtime,'determinism':determinism,'started_utc':datetime.now(timezone.utc).isoformat(),
        'qualification':'Fixed-state optimizer-memory diagnostic; no changed epsilon, ongoing training, quality claim or BF16 clearance'}
    tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',output_dir=args.output_dir,
        group='olmo-optimizer-history',name=args.output_dir.name,preserve_state=preserve_local_rng)
    failure=None
    def persist(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-started,wandb=tracker.record);write_json(args.output_dir/'report.json',report)
    def observe(label,row):
        report.setdefault('observations',{})[label]=row
        tracker.log(scalar_metrics(row,'diagnostic/'+label));persist(label)
        print({'stage':label,'elapsed_seconds':report['elapsed_seconds']},flush=True)
        if time.monotonic()-started>PLAN['soft_seconds']:raise TimeoutError('Bounded diagnostic elapsed-time budget exceeded')
    try:
        tracker.start({'plan':PLAN,'authority_pins':PINS,'qualification':report['qualification']})
        report['runtime_controls']=runtime_contract(runtime,determinism,authority)
        persist('strict_checkpoint_load')
        payload,views,file_identity=comparison.load_checkpoint(args.checkpoint,receipt,
            configuration=authority['configuration'],fingerprint=authority['source_fingerprint'],expected_boundary=authority['final_boundary'])
        model,recipe,source,_,_=construct(SimpleNamespace(scale='pretrained',length=16,artifacts=args.artifacts),'NFR',torch.device('cuda'))
        flags={name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS}
        optimizer,scheduler,counters=import_boundary(model,recipe,source,payload,authority)
        before=current_boundary(model,optimizer,scheduler,counters)
        identities={n:id(p) for n,p in model.named_parameters()}
        data_recipe=replace(recipe,arm='NF')
        fixtures,provenance=load_long_fixture(args.fixture,expected_sha256=args.fixture_sha256,recipe=data_recipe,width=model.config.model_dim)
        if fixture_pins(fixtures)!=origin['fixture_pins']:raise ValueError('Actual fixed fixture/noise differs from prior diagnostic')
        if sum(p.numel() for p in model.parameters())!=1_267_879_936:raise ValueError('Native NFR parameter inventory changed')
        report.update(fixture_provenance=provenance,fixture_pins=fixture_pins(fixtures),recipe=recipe.to_dict(),
            data_construction_recipe=data_recipe.to_dict(),actual_mode=asdict(recipe.mode()),
            current_lrs=[g['lr'] for g in optimizer.param_groups],imported_boundary_exact=before==authority['final_boundary'])
        persist('matched_gradients')
        rows,clipped,gradients=measure_gradients(model,recipe,fixtures,flags,observe=observe)
        report.update(gradient_rows=rows,gradient_comparison=gradients,aggregate_backwards=2,
            physical_backwards=sum(r['physical_backwards'] for r in rows))
        persist('streamed_counterfactuals')
        report['optimizer_comparison']=optimizer_counterfactuals(model,payload,clipped,observe=observe)
        del clipped,views;gc.collect()
        report['integrity']={'complete_boundary_unchanged':current_boundary(model,optimizer,scheduler,counters)==before,
            'parameter_objects_unchanged':identities=={n:id(p) for n,p in model.named_parameters()},
            'source_inventory_unchanged':source_hashes()==sources,'checkpoint_file_identity_unchanged':comparison.signature(args.checkpoint)==file_identity,
            'small_authorities_unchanged':all(sha256_file(getattr(args,k))==v for k,v in PINS.items() if k!='checkpoint'),
            'saved_payload_boundary_unchanged':comparison.payload_boundary(payload)==authority['final_boundary'],
            'fixture_tensors_unchanged':fixture_pins(fixtures)==report['fixture_pins'],
            'runtime_flags_restored':flags=={n:getattr(model.backbone.backbone,n) for n in RUNTIME_FLAGS},
            'gradients_cleared':all(p.grad is None for p in model.parameters()),
            'backward_budget_exact':report['physical_backwards']==4,
            'candidate_budget_exact':report['optimizer_comparison']['parameterwise_adam_calls']==6*len(identities)}
        if not all(report['integrity'].values()):raise AssertionError('Final diagnostic integrity failed')
        report.update(status='passed_bounded_diagnostic',passed=True,memory=memory());persist('complete')
    except BaseException as error:
        failure=error;report.update(status='failed',passed=False,error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat();persist(report.get('stage','setup'))
        try:tracker.finish(succeeded=report['passed'])
        except BaseException:
            report.update(status='failed',passed=False,tracking_finish_failed=True)
            if failure is None:raise
        finally:persist(report.get('stage','setup'))


if __name__=='__main__':main()
