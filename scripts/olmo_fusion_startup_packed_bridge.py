#!/usr/bin/env python3
"""Saved NFR endpoint: five fixed packed sparse/prepared/captured backwards."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from cdrm.pretrained.artifacts import sha256_file,write_json
from cdrm.pretrained.campaign_recipe import build_campaign_adamw,CampaignTokenSchedule,feedback_noise_for_rows
from cdrm.pretrained.campaign_training import CampaignObjective,CampaignGraphTraining
from cdrm.pretrained.lm_training import LMTrainingConfig,_rng_state,optimizer_ownership,parameter_layout
from scripts.experiment_tracking import OnlineTracker,scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct,global_fixture_metadata
from scripts.olmo_campaign_graph_probe import compare_metrics,pointer_snapshot,gradients_are_zero
from scripts.olmo_campaign_precision_bridge import configure_path
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS,TERMS,component_backward,record_gradients
from scripts.olmo_campaign_recurrence_precision import FP32,BF16,fixture_pins,state_pins
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts import olmo_fusion_startup_nfr_updates as endpoint
from scripts.olmo_fusion_startup_packed_data import load_packed_fixture
from scripts.olmo_fusion_startup_context_probe import source_hashes as packed_sources
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA='olmo-fusion-startup-packed-bridge-v1'
FIXTURE_SHA='4932410f9fd370d9dae20a1075bf2a191c642e1563eb8557a6fe02fd7e83a975'
CASES=('fp32_sparse','fp32_prepared','bf16_sparse','bf16_prepared','bf16_captured')
SOFT_SECONDS=3300


def source_hashes():
    sources=endpoint.source_hashes()
    for name,digest in packed_sources().items():
        if name in sources and sources[name]!=digest:raise ValueError('Inherited sources disagree')
        sources[name]=digest
    for name in ('scripts/olmo_fusion_startup_packed_bridge.py','tests/test_fusion_startup_packed_bridge.py',
                 'docs/reports/olmo-fusion-startup/packed-bridge-protocol.md'):
        sources[name]=sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def load_authority(path,digest,*,checkpoint_sha256,sources):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>128*1024*1024 or sha256_file(path)!=digest:
        raise ValueError('NFR report differs from bounded immutable authority')
    report=json.loads(path.read_text())
    if (sha256_file(path)!=digest or report.get('schema')!=endpoint.SCHEMA
            or report.get('status')!='passed_bounded_functionality' or report.get('passed') is not True
            or report.get('optimizer_calls')!=8 or report.get('plan')!=tree_digests(endpoint.PLAN)
            or not report.get('integrity') or not all(report['integrity'].values())
            or report.get('determinism',{}).get('deterministic_algorithms') is not True
            or not report.get('sources') or any(sources.get(k)!=v for k,v in report['sources'].items())):
        raise ValueError('Require completed unchanged-source four-update NFR authority')
    receipts=[row for row in report.get('checkpoints',[]) if row.get('trajectory')==BF16 and row.get('optimizer_updates')==4]
    if len(receipts)!=1 or receipts[0].get('sha256')!=checkpoint_sha256 or not receipts[0].get('gcs'):
        raise ValueError('Require retained BF16 update4 receipt matching checkpoint pin')
    if (len(report.get('rows',[]))!=8 or [r.get('path') for r in report['rows']]!=[FP32,BF16]*4
            or [r.get('update') for r in report['rows']]!=[1,1,2,2,3,3,4,4]
            or report.get('recipe',{}).get('arm')!='NFR'
            or report['recipe'].get('document_policy','isolated-v1')!='isolated-v1'):
        raise ValueError('NFR endpoint trajectory or original policy differs')
    common=report.get('checkpoint_configuration',{})
    for key in ('recipe','sources','origin_checkpoint_sha256','data_manifest_sha256','training_metadata'):
        if key not in report or common.get(key)!=report[key]:
            raise ValueError('Checkpoint configuration differs from report authority: '+key)
    selections=report.get('training_data_selections',[])
    if (len(selections)!=4 or [r.get('source_training_index') for r in selections]!=[144,145,146,147]
            or len(report['training_metadata'])!=4
            or any({key:r[key] for key in ('counts','microbatches','documents','input_tokens')}!=meta
                   for r,meta in zip(selections,report['training_metadata']))):
        raise ValueError('Saved training selections differ from four-update metadata')
    return report


def current_boundary(model,optimizer,scheduler,counters):
    """Streaming digests match the old CPU snapshot without cloning 15GB."""
    return tree_digests({'schema':endpoint.SCHEMA,'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':_rng_state(None),
        'counters':asdict(counters),'ownership':optimizer_ownership(model,optimizer),
        'layout':parameter_layout(model),'modes':{n:m.training for n,m in model.named_modules()}})


def runtime_contract(runtime,determinism,authority):
    expected=authority['runtime']
    checks={key:runtime.get(key)==expected.get(key) for key in ('torch','cuda','gpu')}
    checks['gpu_driver_memory_types']=set(runtime['nvidia_smi'].splitlines())==set(expected['nvidia_smi'].splitlines())
    checks['determinism']=determinism==authority['determinism']
    if not all(checks.values()):raise ValueError('Saved endpoint runtime family or deterministic controls differ')
    return checks


def import_endpoint(model,recipe,source,path,digest,authority):
    common=authority['checkpoint_configuration']
    if (tree_digests(recipe.to_dict())!=common['recipe'] or tree_digests(source)!=common['source_checkpoint']
            or recipe.arm!='NFR' or recipe.document_policy!='isolated-v1'
            or not all(p.requires_grad for p in model.parameters())):
        raise ValueError('Reconstructed isolated NFR contract differs before import')
    metadata=authority['training_metadata']
    optimizer=build_campaign_adamw(model,recipe,fused=False)
    scheduler=CampaignTokenSchedule(optimizer,[row['input_tokens'] for row in metadata],
        warmup_tokens=recipe.warmup_tokens,start_fraction=recipe.warmup_start_fraction)
    configuration=endpoint.checkpoint_configuration(common,BF16)
    fingerprint={'checkpoint_sha256':authority['origin_checkpoint_sha256'],
        'base':source,'sources':authority['sources']}
    selections=authority['training_data_selections']
    def cursor(step):
        if not 0<=step<=4:raise ValueError('Endpoint cursor out of range')
        return selections[0]['start_cursor'] if step==0 else selections[step-1]['next_cursor']
    restored=endpoint.load_endpoint(path,model,optimizer,scheduler,configuration=configuration,
        source_fingerprint=fingerprint,expected_sha256=digest,metadata=metadata,expected_cursor=cursor)
    checks={'update4':restored['counters'].optimizer_updates==4,
        'full_saved_boundary_exact':current_boundary(model,optimizer,scheduler,restored['counters'])==authority['final_boundary_pins'][BF16],
        'tied_readout':model.backbone.readout_weight is model.backbone.token_embeddings.weight,
        'gradients_absent':all(p.grad is None for p in model.parameters())}
    if not all(checks.values()):raise AssertionError('Strict saved NFR endpoint import differs')
    info={'checks':checks,'counters':asdict(restored['counters']),'data_cursor':restored['data_cursor'],
        'checkpoint_sha256':digest,'optimizer_scope':'Restored for strict boundary validation only; zero steps; released before measurement'}
    del optimizer,scheduler,restored
    gc.collect()
    if next(model.parameters()).device.type=='cuda':torch.cuda.empty_cache()
    return info


def packed_transition(model,recipe):
    if recipe.arm!='NFR' or recipe.document_policy!='isolated-v1' or model.config.document_policy!='isolated-v1':
        raise ValueError('Policy transition requires the strictly imported isolated NFR model')
    before=state_pins(model);ids={n:id(p) for n,p in model.named_parameters()}
    trainable={n:p.requires_grad for n,p in model.named_parameters()};modes={n:m.training for n,m in model.named_modules()}
    prior_recipe,prior_config=recipe.to_dict(),model.config.to_dict()
    recipe=replace(recipe,document_policy='continuous-stream-v1')
    model.config=replace(model.config,document_policy='continuous-stream-v1')
    data_recipe=replace(recipe,arm='NF')
    checks={'state_exact':before==state_pins(model),'identities_exact':ids=={n:id(p) for n,p in model.named_parameters()},
        'trainability_exact':trainable=={n:p.requires_grad for n,p in model.named_parameters()},
        'modes_exact':modes=={n:m.training for n,m in model.named_modules()},
        'only_recipe_policy':{k:v for k,v in prior_recipe.items() if k!='document_policy'}=={k:v for k,v in recipe.to_dict().items() if k!='document_policy'},
        'only_nextlat_policy':{k:v for k,v in prior_config.items() if k!='document_policy'}=={k:v for k,v in model.config.to_dict().items() if k!='document_policy'},
        'data_view_only_arm':{k:v for k,v in recipe.to_dict().items() if k!='arm'}=={k:v for k,v in data_recipe.to_dict().items() if k!='arm'},
        'execution_keeps_rt':recipe.mode().rt_mode.selected_layers==recipe.rt_layers and recipe.mode().rt_mode.alpha==1.}
    if not all(checks.values()):raise AssertionError('Packed transition changed unrelated state')
    return recipe,data_recipe,{'checks':checks,'original_recipe':prior_recipe,'execution_recipe':recipe.to_dict(),
        'fixture_construction_recipe':data_recipe.to_dict(),'actual_mode':asdict(recipe.mode())}


def validate_noise_view(fixtures,metadata,recipe,data_recipe,width):
    if len(fixtures)!=2 or len(metadata['records'])!=2:raise ValueError('Require both packed records')
    for (batches,noises),record in zip(fixtures,metadata['records']):
        if len(batches)!=1 or len(noises)!=1:raise ValueError('Packed physical shapes differ')
        keys=record['noise_keys'];shape=tuple(batches[0].input_ids.shape)
        if shape!=(1,1024):raise ValueError('Packed physical shapes differ')
        actual=feedback_noise_for_rows(recipe,keys,logical_update=0,sequence_length=1024,width=width,physical_batch_size=1,device='cpu')
        ordinary=feedback_noise_for_rows(data_recipe,keys,logical_update=0,sequence_length=1024,width=width,physical_batch_size=1,device='cpu')
        if tree_digests(actual)!=tree_digests(ordinary) or tree_digests(actual)!=tree_digests(noises[0]):
            raise ValueError('NFR execution jitter differs from authenticated NF data view')
    return {'mode_independent_noise_exact':True,'fixture_pins':fixture_pins(fixtures)}


def make_runner(model,recipe,fixtures,precision):
    batches=tuple(b for bs,_ in fixtures for b in bs);noises=tuple(n for _,ns in fixtures for n in ns)
    adapter=CampaignObjective(model,batches[0],mode=recipe.mode(),
        global_counts=global_fixture_metadata(model,fixtures)['counts'],world_size=1,feedback_noise=noises[0],
        config=LMTrainingConfig(precision=precision,max_grad_norm=recipe.max_grad_norm))
    return CampaignGraphTraining(adapter),batches,noises


def observed_gradients(model,reference=None,*,save=False):
    # Missing gradients must be visible before the historical observer could
    # materialize mathematical zeros for descriptive comparisons.
    if any(p.requires_grad and p.grad is None for p in model.parameters()):
        raise AssertionError('Combined bridge lost a trainable parameter gradient')
    result,snapshot=record_gradients(model,reference,save_cpu=save,
        scope='Fixed saved NFR packed state; existing budgets, no new precision tolerance')
    return result,snapshot


def compare_case(metrics,gradients,reference_metrics):
    return {'metrics':compare_metrics(metrics,reference_metrics),
        'gradient_legacy_budget_passed':gradients['comparison']['all_parameters_close'],
        'passed':compare_metrics(metrics,reference_metrics)['passed'] and gradients['comparison']['all_parameters_close']}


def run_bridge(model,recipe,fixtures,*,original_flags,publish,phase,deadline):
    initial=state_pins(model);inputs=fixture_pins(fixtures);rng=tree_digests(_rng_state(None))
    modes={n:m.training for n,m in model.named_modules()};rows=[];refs={};metrics_by_case={};runner=None
    setup={'initialization_backward_calls':0,'warmup_backward_calls':0,'capture_backward_calls':0,
           'measured_physical_backward_calls':0,'measured_replay_calls':0}
    metadata=global_fixture_metadata(model,fixtures)
    if metadata['microbatches']!=2:raise ValueError('Bridge requires two physical records')
    try:
        for case in CASES:
            if time.monotonic()>=deadline:raise TimeoutError('Packed bridge reached its soft budget')
            phase(case+'/start',dict(setup))
            path=FP32 if case.startswith('fp32') else BF16
            execution=configure_path(model,original_flags,path)
            device=next(model.parameters()).device
            backend=SDPBackend.FLASH_ATTENTION if path==BF16 and device.type=='cuda' else SDPBackend.MATH
            start=time.monotonic();before_warm=before_capture=0
            with sdpa_kernel(backend),torch.autocast(device.type,enabled=False):
                if case.endswith('sparse'):
                    result=component_backward(model,recipe,fixtures,precision=execution['precision'],layout='sparse',objective='combined')
                elif case.endswith('prepared'):
                    model.zero_grad(set_to_none=True)
                    runner,batches,noises=make_runner(model,recipe,fixtures,execution['precision'])
                    result=runner.backward(batches,feedback_noises=noises,replay=False)
                    setup['initialization_backward_calls']+=runner.warmup_backward_calls
                else:
                    if runner is None or runner.graph is not None:raise AssertionError('Capture requires the measured BF16 eager runner')
                    # Eager prepared includes one initialization + two records.
                    expected_seconds=11*(rows[-1]['backward_elapsed_seconds']/3)*1.5+120
                    if time.monotonic()+expected_seconds>deadline:
                        raise TimeoutError('Insufficient remaining budget for required capture warmup')
                    addresses=pointer_snapshot(runner);before_warm=runner.warmup_backward_calls;before_capture=runner.capture_backward_calls
                    def capture_phase(name,action):
                        phase('capture/'+name,{'action':action,**setup})
                    runner.capture(warmup=10,phase_observer=capture_phase)
                    setup['warmup_backward_calls']+=runner.warmup_backward_calls-before_warm
                    setup['capture_backward_calls']+=runner.capture_backward_calls-before_capture
                    if pointer_snapshot(runner)!=addresses or not gradients_are_zero(model):
                        raise AssertionError('Capture changed persistent pointers or retained setup gradients')
                    result=runner.backward(batches,feedback_noises=noises,replay=True)
                    setup['measured_replay_calls']=runner.replay_calls
                    if pointer_snapshot(runner)!=addresses:raise AssertionError('Packed refill/replay changed owned pointers')
            elapsed=time.monotonic()-start;setup['measured_physical_backward_calls']+=2
            primary={'fp32_prepared':'fp32_sparse','bf16_sparse':'fp32_sparse',
                'bf16_prepared':'bf16_sparse','bf16_captured':'bf16_prepared'}.get(case)
            gradients,snapshot=observed_gradients(model,None if primary is None else refs[primary],save=case!='bf16_captured')
            health={'finite_losses':all(math.isfinite(v) for v in (result['objective'],*result['loss_sums'].values())),
                'finite_gradients':gradients['finite'],'full_participation':gradients['participation_intact'],
                'nonzero_all_groups':all(row['norm']>0 for row in gradients['groups'].values()),
                'counts_exact':all(result[k]==v for k,v in metadata.items()),
                'state_exact':state_pins(model)==initial,'inputs_exact':fixture_pins(fixtures)==inputs,
                'rng_exact':tree_digests(_rng_state(None))==rng,
                'modes_exact':{n:m.training for n,m in model.named_modules()}==modes}
            row={'case':case,'path':path,'execution':execution,'metrics':result,'gradients':gradients,
                'gradient_pins':tree_digests(snapshot) if snapshot else tree_digests({n:p.grad for n,p in model.named_parameters()}),
                'primary_reference':primary,'health':health,'passed':all(health.values()),
                'backward_elapsed_seconds':elapsed,'setup_counts':dict(setup)}
            if primary is not None:row['primary_comparison']=compare_case(result,gradients,metrics_by_case[primary])
            if case=='bf16_prepared':
                other,_=observed_gradients(model,refs['fp32_prepared'])
                row['cross_precision_same_layout']={'reference':'fp32_prepared','gradients':other,
                    'metrics':compare_metrics(result,metrics_by_case['fp32_prepared'])}
            if case=='bf16_captured':
                row['bitwise_gradients_exact']=row['gradient_pins']==tree_digests(refs['bf16_prepared'])
                row['bitwise_metrics_exact']=result==metrics_by_case['bf16_prepared']
            rows.append(row);publish(row)
            if not row['passed']:raise AssertionError('Packed bridge operational health failed')
            if case in ('fp32_prepared','bf16_captured') and not row['primary_comparison']['passed']:
                raise AssertionError('Existing same-precision semantic comparison budget failed')
            refs[case]=snapshot;metrics_by_case[case]=result
            if case=='fp32_prepared':
                with sdpa_kernel(backend),torch.autocast(device.type,enabled=False):runner.discard_backward()
                runner=None;model.zero_grad(set_to_none=True)
            elif case=='bf16_prepared':
                with sdpa_kernel(backend),torch.autocast(device.type,enabled=False):runner.discard_backward()
                for key in ('fp32_sparse','fp32_prepared','bf16_sparse'):refs.pop(key,None)
            elif case=='bf16_captured':
                with sdpa_kernel(backend),torch.autocast(device.type,enabled=False):runner.discard_backward()
            else:model.zero_grad(set_to_none=True)
            gc.collect()
        if setup!={'initialization_backward_calls':2,'warmup_backward_calls':10,'capture_backward_calls':1,
                  'measured_physical_backward_calls':10,'measured_replay_calls':2}:
            raise AssertionError('Actual backward accounting differs from bounded plan')
        return {'rows':rows,'setup_counts':setup,
            'bf16_sparse_prepared_legacy_budget_passed':rows[3]['primary_comparison']['passed']}
    finally:
        if runner is not None:
            runner.graph_result=None;runner.graph=None;runner.stream=None
        runner=None;refs.clear();gc.collect()
        model.zero_grad(set_to_none=True);configure_path(model,original_flags,BF16)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','nfr-report','fixture'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--artifacts',type=Path,default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args(argv)
    for digest in (args.checkpoint_sha256,args.nfr_report_sha256,args.fixture_sha256):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):p.error('Require lowercase independent SHA256 pins')
    if args.fixture_sha256!=FIXTURE_SHA:p.error('Only the frozen packed T1024 fixture is authorized')
    args.output_dir=args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):p.error('Evidence must remain on persistent project storage')
    return args


def main(argv=None):
    args=parse_args(argv);determinism=configure_determinism(True);runtime=require_container_gpu()
    if torch.distributed.is_initialized():raise RuntimeError('Packed bridge is one GPU/process without DDP')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');started=time.monotonic()
    args.output_dir.mkdir(parents=True,exist_ok=False);sources=source_hashes()
    for name in sources:
        destination=args.output_dir/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    report={'schema':SCHEMA,'status':'running','passed':False,'sources':sources,'runtime':runtime,'determinism':determinism,
        'checkpoint_sha256':args.checkpoint_sha256,'nfr_report_sha256':args.nfr_report_sha256,'fixture_sha256':args.fixture_sha256,
        'cases':CASES,'aggregate_measured_backwards':5,'physical_measured_backwards':10,'optimizer_updates':0,'rows':[],
        'started_utc':datetime.now(timezone.utc).isoformat(),'soft_budget_seconds':SOFT_SECONDS,
        'qualification':'Saved NFR combined packed execution bridge; no new tolerance, optimizer, DDP, training quality or general BF16 clearance'}
    tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',output_dir=args.output_dir,
        group='olmo-fusion-startup',name=args.output_dir.name,preserve_state=preserve_local_rng)
    failure=None
    def persist(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-started,wandb=tracker.record);write_json(args.output_dir/'report.json',report)
    try:
        tracker.start({k:report[k] for k in ('checkpoint_sha256','fixture_sha256','cases','qualification')})
        authority=load_authority(args.nfr_report,args.nfr_report_sha256,checkpoint_sha256=args.checkpoint_sha256,sources=sources)
        report['runtime_import_checks']=runtime_contract(runtime,determinism,authority)
        persist('construct_original_isolated_nfr')
        model,recipe,source,_,_=construct(SimpleNamespace(scale='pretrained',length=16,artifacts=args.artifacts),'NFR',torch.device('cuda'))
        flags={key:getattr(model.backbone.backbone,key) for key in RUNTIME_FLAGS}
        report['import']=import_endpoint(model,recipe,source,args.checkpoint,args.checkpoint_sha256,authority)
        recipe,data_recipe,report['policy_transition']=packed_transition(model,recipe)
        fixtures,metadata=load_packed_fixture(args.fixture,expected_sha256=args.fixture_sha256,recipe=data_recipe,width=model.config.model_dim)
        if metadata['training_manifest_sha256']!=authority['data_manifest_sha256']:raise ValueError('Packed and endpoint data authorities differ')
        report['noise_view']=validate_noise_view(fixtures,metadata,recipe,data_recipe,model.config.model_dim)
        report.update(initial_state=state_pins(model),fixture_provenance=metadata,recipe=recipe.to_dict(),
            fixture_metadata=global_fixture_metadata(model,fixtures),production_flags=flags)
        def phase(name,details):report['phase_details']=details;persist(name)
        def publish(row):
            row['memory']=memory();report['rows'].append(row);persist(row['case'])
            tracker.log(scalar_metrics(row,'packed_bridge/'+row['case']),step=len(report['rows']))
        result=run_bridge(model,recipe,fixtures,original_flags=flags,publish=publish,phase=phase,deadline=started+SOFT_SECONDS)
        report.update(setup_counts=result['setup_counts'],bf16_sparse_prepared_legacy_budget_passed=result['bf16_sparse_prepared_legacy_budget_passed'])
        report['integrity']={'sources_unchanged':sources==source_hashes(),'saved_state_unchanged':state_pins(model)==report['initial_state'],
            'checkpoint_unchanged':sha256_file(args.checkpoint)==args.checkpoint_sha256,
            'authority_unchanged':sha256_file(args.nfr_report)==args.nfr_report_sha256,
            'fixture_file_unchanged':sha256_file(args.fixture)==args.fixture_sha256,
            'five_cases':len(report['rows'])==5,'gradients_absent':all(p.grad is None for p in model.parameters()),
            'flags_restored':all(getattr(model.backbone.backbone,k)==v for k,v in flags.items())}
        if not all(report['integrity'].values()):raise AssertionError('Final packed bridge integrity failed')
        report.update(passed=True,status='passed_execution_bridge',numerical_qualification=(
            'BF16 sparse/prepared passed existing budget on this state/fixture only' if result['bf16_sparse_prepared_legacy_budget_passed']
            else 'BF16 sparse/prepared remains outside existing budget; operational/graph controls do not clear it'))
        persist('complete')
    except BaseException as error:
        failure=error;report.update(passed=False,status='failed',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat();persist(report.get('stage','setup'))
        try:tracker.finish(succeeded=report['passed'])
        except BaseException as error:
            report.update(passed=False,status='failed',tracking_finish_error={'type':type(error).__name__})
            if failure is None:raise
        finally:persist(report.get('stage','setup'))


if __name__=='__main__':main()
