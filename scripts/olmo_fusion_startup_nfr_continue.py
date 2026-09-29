#!/usr/bin/env python3
"""Conditional sixteen-update NFR continuation from one common BF16 endpoint."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from cdrm.pretrained.artifacts import sha256_file,write_json
from cdrm.pretrained.campaign_recipe import build_campaign_adamw,CampaignTokenSchedule
from cdrm.pretrained.lm_training import (TrainingCounters,_rng_state,optimizer_ownership,parameter_layout,
    load_training_checkpoint)
from scripts.experiment_tracking import OnlineTracker,scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct,global_fixture_metadata,advance_counters
from scripts.olmo_campaign_precision_bridge import configure_path
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS,component_backward
from scripts.olmo_campaign_probe import component,memory
from scripts.olmo_campaign_recurrence_precision import FP32,BF16,fixture_pins,state_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_data import StartupData,DEFAULT_ROOT,DEFAULT_MANIFEST_SHA256
from scripts.olmo_fusion_startup_long_probe import load_long_fixture
from scripts.olmo_fusion_startup_train import retain_checkpoint
from scripts import olmo_fusion_startup_nfr_updates as endpoint
from scripts import olmo_fusion_startup_packed_bridge as bridge
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA='olmo-fusion-startup-nfr-continuation-v1'
PLAN={'start_update':4,'end_update':20,'source_index_offset':144,'new_train_indices':list(range(148,164)),
      'checkpoint_updates':[12,20],'evaluation_updates':[4,12,20],'length':128,'physical_batch':8,
      'ce_targets_per_update':8192,'checkpoint_seconds':600,'max_optimizer_calls':16,'soft_seconds':3300}
PATHS={'fp32':FP32,'bf16_mixed':BF16}


def source_hashes():
    sources=bridge.source_hashes()
    for name in ('scripts/olmo_fusion_startup_nfr_continue.py','tests/test_fusion_startup_nfr_continue.py',
                 'docs/reports/olmo-fusion-startup/nfr-continuation-protocol.md'):
        sources[name]=sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def restore_original(model,recipe,source,path,digest,authority):
    common=authority['checkpoint_configuration']
    if (tree_digests(recipe.to_dict())!=common['recipe'] or tree_digests(source)!=common['source_checkpoint']
            or recipe.arm!='NFR' or recipe.document_policy!='isolated-v1'
            or not all(p.requires_grad for p in model.parameters())):
        raise ValueError('Common starting NFR contract differs before import')
    optimizer=build_campaign_adamw(model,recipe,fused=False)
    metadata=authority['training_metadata']
    scheduler=CampaignTokenSchedule(optimizer,[r['input_tokens'] for r in metadata],
        warmup_tokens=recipe.warmup_tokens,start_fraction=recipe.warmup_start_fraction)
    selections=authority['training_data_selections']
    def cursor(step):
        if not 0<=step<=4:raise ValueError('Original cursor outside update0..4')
        return selections[0]['start_cursor'] if step==0 else selections[step-1]['next_cursor']
    result=endpoint.load_endpoint(path,model,optimizer,scheduler,
        configuration=endpoint.checkpoint_configuration(common,BF16),
        source_fingerprint={'checkpoint_sha256':authority['origin_checkpoint_sha256'],'base':source,'sources':authority['sources']},
        expected_sha256=digest,metadata=metadata,expected_cursor=cursor)
    counters=result['counters'];pins=bridge.current_boundary(model,optimizer,scheduler,counters)
    if counters.optimizer_updates!=4 or pins!=authority['final_boundary_pins'][BF16]:
        raise AssertionError('Original complete BF16 update4 boundary differs')
    return optimizer,scheduler,counters,pins


def extend_schedule(optimizer,scheduler,counters,old_metadata,metadata):
    """Explicit prefix-preserving fork; the old loader remains unchanged."""
    if (len(old_metadata)!=4 or len(metadata)!=20 or metadata[:4]!=old_metadata
            or asdict(counters)!=asdict(endpoint.expected_counters(old_metadata,4))
            or scheduler.last_epoch!=4 or scheduler._step_count!=5
            or scheduler._last_lr!=[g['lr'] for g in optimizer.param_groups]):
        raise ValueError('Schedule fork requires the exact completed original prefix')
    expected=[0]
    for row in metadata:
        value=row['input_tokens']
        if type(value) is not int or value<=0:raise ValueError('Schedule fork needs positive integer token increments')
        expected.append(expected[-1]+value)
    if tuple(expected[:5])!=scheduler.token_prefix or scheduler.completed_tokens!=counters.input_tokens:
        raise ValueError('Original token schedule prefix differs')
    old=copy.deepcopy(scheduler.state_dict());rates=[g['lr'] for g in optimizer.param_groups]
    # Creating a new scheduler temporarily sets floor LR, but never touches Adam
    # moments or weights. Restore the existing completed history before returning.
    new=CampaignTokenSchedule(optimizer,[r['input_tokens'] for r in metadata],
        warmup_tokens=scheduler.warmup_tokens,start_fraction=scheduler.start_fraction)
    extended={**old,'token_prefix':new.token_prefix,'plan_sha256':new.plan_sha256}
    new.load_state_dict(extended)
    for group,lr in zip(optimizer.param_groups,rates):group['lr']=lr
    checks={'prefix_exact':new.token_prefix[:5]==scheduler.token_prefix,
        'completed_tokens_exact':new.completed_tokens==scheduler.completed_tokens,
        'epoch_exact':new.last_epoch==scheduler.last_epoch,'step_count_exact':new._step_count==scheduler._step_count,
        'current_lr_exact':new._last_lr==rates==new.get_lr(),'base_lr_exact':new.base_lrs==scheduler.base_lrs,
        'warmup_exact':new.warmup_tokens==scheduler.warmup_tokens and new.start_fraction==scheduler.start_fraction}
    if not all(checks.values()):raise AssertionError('Schedule fork changed the committed optimizer clock')
    return new,{'checks':checks,'old_scheduler':tree_digests(old),'new_scheduler':tree_digests(new.state_dict()),
        'old_contract':scheduler.checkpoint_contract(),'new_contract':new.checkpoint_contract()}


def norm_groups(model,*,gradients):
    """Only scalar reductions leave the device; no full vectors copied per step."""
    grouped={name:[] for name in ('backbone','fusion','predictor')}
    for name,p in model.named_parameters():
        value=p.grad if gradients else p
        if not p.requires_grad or value is None or value.dtype!=torch.float32:
            raise FloatingPointError('Full NFR requires every master and gradient to participate')
        grouped[component(name)].append(torch.linalg.vector_norm(value.detach()))
    result={name:float(torch.linalg.vector_norm(torch.stack(values))) for name,values in grouped.items()}
    if not all(math.isfinite(v) and v>0 for v in result.values()):
        raise FloatingPointError('NFR group has a nonfinite or zero norm')
    result['all']=math.sqrt(sum(v*v for v in result.values()))
    return result


def validate_clock(model,optimizer,scheduler,counters,metadata):
    step=counters.optimizer_updates
    if (not 4<=step<=20 or asdict(counters)!=asdict(endpoint.expected_counters(metadata,step))
            or scheduler.last_epoch!=step or scheduler._step_count!=step+1
            or scheduler.completed_tokens!=counters.input_tokens
            or scheduler._last_lr!=[g['lr'] for g in optimizer.param_groups]
            or scheduler.get_lr()!=scheduler._last_lr):
        raise ValueError('Continuation clock or LR differs from actual data prefix')
    if any(p.grad is not None for p in model.parameters()):raise ValueError('Committed boundary still has gradients')
    optimizer_ownership(model,optimizer)
    if not all(p.requires_grad for p in model.parameters()):raise ValueError('Continuation lost full trainability')


def update(model,recipe,fixtures,optimizer,scheduler,counters,*,path,original_flags,metadata):
    validate_clock(model,optimizer,scheduler,counters,metadata)
    if counters.optimizer_updates>=20:raise ValueError('Fixed continuation schedule exhausted')
    actual=global_fixture_metadata(model,fixtures)
    if actual!=metadata[counters.optimizer_updates]:raise ValueError('Next update inputs/counts differ')
    scheduler.validate_next_update(actual['input_tokens'])
    rates=[g['lr'] for g in optimizer.param_groups];rng=tree_digests(_rng_state(None));pins=fixture_pins(fixtures)
    execution=configure_path(model,original_flags,path);device=next(model.parameters()).device
    backend=SDPBackend.FLASH_ATTENTION if path==BF16 and device.type=='cuda' else SDPBackend.MATH
    try:
        with sdpa_kernel(backend),torch.autocast(device.type,enabled=False):
            metrics=component_backward(model,recipe,fixtures,precision=execution['precision'],layout='sparse',objective='combined')
        if (any(not math.isfinite(v) for v in (metrics['objective'],*metrics['loss_sums'].values()))
                or any(metrics[key]!=value for key,value in actual.items())):
            raise FloatingPointError('Nonfinite or inconsistent objective before Adam')
        raw=norm_groups(model,gradients=True)
        total=torch.nn.utils.clip_grad_norm_(model.parameters(),recipe.max_grad_norm,error_if_nonfinite=True,foreach=False)
        optimizer.step();scheduler.step();advance_counters(counters,metrics)
        masters=norm_groups(model,gradients=False)
    finally:model.zero_grad(set_to_none=True)
    if rng!=tree_digests(_rng_state(None)) or pins!=fixture_pins(fixtures):
        raise AssertionError('Update consumed global RNG or changed its fixed inputs')
    validate_clock(model,optimizer,scheduler,counters,metadata)
    return {'update':counters.optimizer_updates,'path':path,'metrics':metrics,'counters':asdict(counters),
        'lr_used':rates,'lr_next':[g['lr'] for g in optimizer.param_groups],
        'raw_gradient_norms':raw,'master_parameter_norms':masters,'gradient_norm_before_clip':float(total),
        'clip_scale':min(1.,recipe.max_grad_norm/(float(total)+1e-6)),'input_pins':pins}


def configuration_for(model,recipe,authority,*,origin_sha256,sources,metadata,selections,precision,runtime,determinism,extension,original_flags):
    return {'kind':SCHEMA,'plan':PLAN,'precision':precision,'origin_checkpoint_sha256':origin_sha256,
        'origin_trajectory':BF16,'origin_boundary':authority['final_boundary_pins'][BF16],
        'recipe':recipe.to_dict(),'model_config':model.backbone.config.to_dict(),'nextlat_config':model.config.to_dict(),
        'fusion_config':model.backbone.fusion_config.to_dict(),'sources':sources,'runtime':runtime,'determinism':determinism,
        'training_metadata':metadata,'training_data_selections':selections,'schedule_fork':extension,
        'ownership':parameter_layout(model),'module_modes':{n:m.training for n,m in model.named_modules()},
        'frozen_buffer_pins':tree_digests(dict(model.named_buffers())),'production_flags':original_flags,
        'data_manifest_sha256':authority['data_manifest_sha256'],'fixture_sha256':authority['fixture_sha256'],
        'qualification':'Both paths share BF16-derived update4 state and Adam; conditional history, not independent cold starts'}


def load_continuation(path,model,optimizer,scheduler,*,digest,configuration,fingerprint,metadata,data):
    # Reuse the strict generic full-state loader plus existing clocks/moment
    # validation under this NEW configuration. No historical schema is relaxed.
    if configuration.get('kind')!=SCHEMA or configuration.get('plan')!=PLAN:
        raise ValueError('Foreign continuation lineage')
    path=Path(path)
    if path.is_symlink() or not path.is_file() or sha256_file(path)!=digest:raise ValueError('Continuation checkpoint SHA differs')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    step=payload['counters']['optimizer_updates']
    if (payload['configuration']!=tree_digests(configuration) or payload['source_fingerprint']!=tree_digests(fingerprint)
            or not 4<=step<=20 or payload['data_cursor']!=data.cursor(144+step)
            or tree_digests({name:payload['model'][name] for name,_ in model.named_buffers()})!=configuration['frozen_buffer_pins']):
        raise ValueError('Continuation source, cursor or frozen buffers differ before import')
    snapshot={'schema':endpoint.SCHEMA,'model':payload['model'],'optimizer':payload['optimizer'],
        'scheduler':payload['scheduler'],'rng':payload['rng'],'counters':payload['counters'],
        'ownership':payload['optimizer_ownership'],'layout':payload['parameter_layout'],'modes':payload['module_training']}
    endpoint.validate_boundary(model,optimizer,scheduler,snapshot,metadata)
    del payload,snapshot
    restored=load_training_checkpoint(path,model,optimizer,scheduler=scheduler,configuration=configuration,
        source_fingerprint=fingerprint,expected_sha256=digest)
    validate_clock(model,optimizer,scheduler,restored['counters'],metadata)
    if tree_digests(dict(model.named_buffers()))!=configuration['frozen_buffer_pins']:
        raise AssertionError('Continuation changed frozen model buffers')
    return restored['counters']


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precision',choices=tuple(PATHS),required=True)
    for name in ('origin-checkpoint','nfr-report','fixture'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--resume-sha256')
    p.add_argument('--artifacts',type=Path,default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--storage-prefix',required=True);p.add_argument('--stop-file',type=Path)
    args=p.parse_args(argv)
    for name in ('origin_checkpoint','nfr_report','fixture','resume'):
        path,digest=getattr(args,name),getattr(args,name+'_sha256')
        if (path is None)!=(digest is None) or (digest is not None and (len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest))):
            p.error('All artifact paths require independent lowercase SHA256 pins')
    args.output_dir=args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):p.error('Evidence/checkpoints require persistent project storage')
    return args


def main(argv=None):
    args=parse_args(argv);determinism=configure_determinism(True);runtime=require_container_gpu()
    if torch.distributed.is_initialized():raise RuntimeError('One process per independent trajectory; no DDP')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');started=time.monotonic()
    args.output_dir.mkdir(parents=True,exist_ok=False);sources=source_hashes()
    for name in sources:
        target=args.output_dir/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    report={'schema':SCHEMA,'status':'running','passed':False,'precision':args.precision,'sources':sources,
        'plan':PLAN,'runtime':runtime,'determinism':determinism,'updates':[],'evaluations':[],'checkpoints':[],
        'physical_optimizer_updates':0,'started_utc':datetime.now(timezone.utc).isoformat(),
        'qualification':'Conditional short full-NFR stability diagnostic from common BF16 update4; no quality or production BF16 clearance'}
    tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',output_dir=args.output_dir,group='olmo-fusion-startup',
        name=args.output_dir.name,preserve_state=preserve_local_rng)
    failure=None;requested=[];handlers={n:signal.getsignal(n) for n in (signal.SIGTERM,signal.SIGINT)}
    for number in handlers:signal.signal(number,lambda signum,frame:requested.append(signum))
    def persist(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-started,wandb=tracker.record);write_json(args.output_dir/'report.json',report)
    try:
        tracker.start({'plan':PLAN,'precision':args.precision,'qualification':report['qualification']})
        authority=bridge.load_authority(args.nfr_report,args.nfr_report_sha256,checkpoint_sha256=args.origin_checkpoint_sha256,sources=sources)
        report['runtime_import_checks']=bridge.runtime_contract(runtime,determinism,authority)
        model,recipe,source,_,_=construct(SimpleNamespace(scale='pretrained',length=16,artifacts=args.artifacts),'NFR',torch.device('cuda'))
        flags={n:getattr(model.backbone.backbone,n) for n in RUNTIME_FLAGS};persist('strict_original_import')
        optimizer,scheduler,counters,origin=restore_original(model,recipe,source,args.origin_checkpoint,args.origin_checkpoint_sha256,authority)
        report['origin_boundary']=origin
        data=StartupData.from_prepared(args.data_root,expected_manifest_sha256=DEFAULT_MANIFEST_SHA256)
        if data.manifest_sha256!=authority['data_manifest_sha256']:raise ValueError('Training authority differs')
        selections=[{'source_training_index':i,**data.update_metadata(i)} for i in range(144,164)]
        metadata=[{k:r[k] for k in ('counts','microbatches','documents','input_tokens')} for r in selections]
        if selections[:4]!=authority['training_data_selections'] or any(r['counts']['ce']!=8192 for r in metadata):
            raise ValueError('Data schedule differs from original prefix or fixed CE budget')
        scheduler,extension=extend_schedule(optimizer,scheduler,counters,authority['training_metadata'],metadata)
        if args.fixture_sha256!=authority['fixture_sha256']:raise ValueError('Continuation changed held-out authority')
        data_recipe=replace(recipe,arm='NF')
        fixtures,provenance=load_long_fixture(args.fixture,expected_sha256=args.fixture_sha256,recipe=data_recipe,width=model.config.model_dim)
        if fixture_pins(fixtures)!=authority['fixture_pins']:raise ValueError('Fixed held-out tensors/noise differ')
        config=configuration_for(model,recipe,authority,origin_sha256=args.origin_checkpoint_sha256,sources=sources,
            metadata=metadata,selections=selections,precision=args.precision,runtime=runtime,determinism=determinism,
            extension=extension,original_flags=flags)
        fingerprint={'checkpoint_sha256':source['sha256'],'origin_checkpoint_sha256':args.origin_checkpoint_sha256,'nfr_report_sha256':args.nfr_report_sha256,
            'base':source,'sources':sources,'data_manifest_sha256':data.manifest_sha256,'fixture_sha256':args.fixture_sha256}
        report.update(configuration=config,source_fingerprint=fingerprint,fixture_provenance=provenance,
            data_construction_recipe=data_recipe.to_dict(),actual_execution_mode=asdict(recipe.mode()))
        if args.resume:
            counters=load_continuation(args.resume,model,optimizer,scheduler,digest=args.resume_sha256,
                configuration=config,fingerprint=fingerprint,metadata=metadata,data=data)
            report['resume_checkpoint_sha256']=args.resume_sha256
        if counters.optimizer_updates>=20:raise ValueError('No authorized continuation updates remain')
        expected_flags=configure_path(model,flags,PATHS[args.precision])['runtime_flags']
        report['starting_update']=counters.optimizer_updates
        report['starting_boundary']=bridge.current_boundary(model,optimizer,scheduler,counters)
        last_save=time.monotonic()
        def checkpoint(reason):
            nonlocal last_save
            number=counters.optimizer_updates
            if any(r['optimizer_updates']==number for r in report['checkpoints']):return
            validate_clock(model,optimizer,scheduler,counters,metadata)
            pins=bridge.current_boundary(model,optimizer,scheduler,counters)
            record=endpoint.save_endpoint(args.output_dir/f'update-{number:06d}.pt',model,optimizer,scheduler,counters,
                configuration=config,source_fingerprint=fingerprint,data_cursor=data.cursor(144+number))
            record.update(reason=reason,boundary=pins);report['checkpoints'].append(record);persist('checkpoint/local/'+str(number))
            record['gcs']=retain_checkpoint(record['path'],args.storage_prefix,record['sha256'])
            if pins!=bridge.current_boundary(model,optimizer,scheduler,counters):raise AssertionError('Retention changed committed boundary')
            write_json(Path(record['path']).with_suffix('.receipt.json'),record);last_save=time.monotonic();persist('checkpoint/retained/'+str(number))
        def evaluate():
            before=bridge.current_boundary(model,optimizer,scheduler,counters)
            row={'update':counters.optimizer_updates,**endpoint.evaluate_fp32(model,recipe,fixtures,original_flags=flags)}
            if before!=bridge.current_boundary(model,optimizer,scheduler,counters):raise AssertionError('Read-only eval changed complete boundary')
            report['evaluations'].append(row);tracker.log({'update':counters.optimizer_updates,**scalar_metrics(row,'dev_common_fp32')},step=counters.optimizer_updates);persist('evaluation')
        # Original update4 is already retained. The new scheduler fork is
        # reconstructible from the original authority plus this frozen plan.
        report['retained_origin']={'sha256':args.origin_checkpoint_sha256,'trajectory':BF16,'update':4,'new_weights_written':False}
        if counters.optimizer_updates in PLAN['evaluation_updates']:evaluate()
        while counters.optimizer_updates<20:
            if requested or (args.stop_file and args.stop_file.exists()) or time.monotonic()-started>=PLAN['soft_seconds']:
                checkpoint('bounded_stop');report.update(status='stopped_at_boundary',passed=True);break
            step=counters.optimizer_updates;index=144+step
            batches,noises=data.update_batches(index,recipe,model.config.model_dim,device='cpu');fixtures_train=[(batches,noises)]
            persist('update/'+str(step+1));begin=time.monotonic()
            row=update(model,recipe,fixtures_train,optimizer,scheduler,counters,path=PATHS[args.precision],original_flags=flags,metadata=metadata)
            row.update(source_training_index=index,elapsed_seconds=time.monotonic()-begin,memory=memory())
            report['updates'].append(row);report['physical_optimizer_updates']+=1
            tracker.log({'update':counters.optimizer_updates,**scalar_metrics(row,'train')},step=counters.optimizer_updates);persist('update/complete')
            if report['physical_optimizer_updates']>16:raise AssertionError('Continuation exceeded authorized optimizer calls')
            if counters.optimizer_updates in PLAN['evaluation_updates']:evaluate()
            if counters.optimizer_updates in PLAN['checkpoint_updates'] or time.monotonic()-last_save>=PLAN['checkpoint_seconds']:checkpoint('scheduled')
            del batches,noises,fixtures_train
        validate_clock(model,optimizer,scheduler,counters,metadata)
        report['integrity']={'sources_unchanged':sources==source_hashes(),
            'origin_unchanged':sha256_file(args.origin_checkpoint)==args.origin_checkpoint_sha256,
            'report_authority_unchanged':sha256_file(args.nfr_report)==args.nfr_report_sha256,
            'fixture_unchanged':sha256_file(args.fixture)==args.fixture_sha256,
            'buffers_unchanged':tree_digests(dict(model.named_buffers()))==config['frozen_buffer_pins'],
            'modes_unchanged':{n:m.training for n,m in model.named_modules()}==config['module_modes'],
            'ownership_unchanged':parameter_layout(model)==config['ownership'],
            'gradients_absent':all(p.grad is None for p in model.parameters()),
            'step_count_exact':report['physical_optimizer_updates']==counters.optimizer_updates-report['starting_update'],
            'flags_restored':all(getattr(model.backbone.backbone,n)==v for n,v in expected_flags.items())}
        if args.resume:report['integrity']['resume_unchanged']=sha256_file(args.resume)==args.resume_sha256
        if not all(report['integrity'].values()):raise AssertionError('Continuation final integrity failed')
        report.update(final_counters=asdict(counters),final_cursor=data.cursor(144+counters.optimizer_updates),
            final_boundary=bridge.current_boundary(model,optimizer,scheduler,counters))
        if report['status']!='stopped_at_boundary':report.update(status='completed_segment',passed=True)
        persist('complete')
    except BaseException as error:
        failure=error;report.update(status='failed',passed=False,error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
    finally:
        for number,handler in handlers.items():signal.signal(number,handler)
        report['finished_utc']=datetime.now(timezone.utc).isoformat();persist(report.get('stage','setup'))
        try:tracker.finish(succeeded=report['passed'])
        except BaseException as error:
            report.update(passed=False,status='failed',tracking_finish_error={'type':type(error).__name__})
            if failure is None:raise
        finally:persist(report.get('stage','setup'))


if __name__=='__main__':main()
