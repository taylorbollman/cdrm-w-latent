#!/usr/bin/env python3
"""Strict saved NFR64 continuation through128; no further objective change."""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend,sdpa_kernel
from cdrm.pretrained.artifacts import sha256_file,write_json
from scripts import olmo_kl_continuation as accepted
from scripts import olmo_nfr_128_contract as scope_contract
from scripts import olmo_nfr_128_engine as engine
from scripts.olmo_campaign_loop import Coordinator,LifecycleError
from scripts.olmo_campaign_loop_guarded import run_stage_releasing_failure
from scripts.olmo_campaign_loop_run import finalize_report
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_packed_campaign_run import configure_cuda_runtime
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.experiment_tracking import OnlineTracker

from scripts.olmo_kl_continuation import (construct, build_campaign_adamw, CampaignTokenSchedule,
    contract, declared_recipe, set_kl_weight, branch_identity, source_fingerprint,
    SSDCheckpointStorage, storage_policy, EvaluationController, OrderedCampaignData, legacy,
    validate_branch_resume)



def parse_args(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scope-declaration',type=Path,required=True)
    parser.add_argument('--scope-sha256',required=True)
    parser.add_argument('--activation-receipt',type=Path,required=True)
    parser.add_argument('--activation-sha256',required=True)
    extension,remaining=parser.parse_known_args(argv)
    args=accepted.parse_args(remaining)
    args.scope_declaration=extension.scope_declaration;args.scope_sha256=extension.scope_sha256
    args.activation_receipt=extension.activation_receipt;args.activation_sha256=extension.activation_sha256
    accepted.legacy.pin(args.scope_sha256);accepted.legacy.pin(args.activation_sha256)
    return args


def assert_sources_and_inputs(args,sources):
    if scope_contract.source_hashes(args.scope_declaration,args.continuation_scope)!=sources:
        raise ValueError('NFR scoped execution sources changed')
    checks=((args.scope_declaration,args.scope_sha256),(args.declaration,args.declaration_sha256),
        (args.resolved,args.resolved_sha256),(args.parent_report,args.parent_report_sha256),
        (args.parent_checkpoint/'manifest.json',args.parent_manifest_sha256),
        (args.activation_receipt,args.activation_sha256),
        *[(accepted.legacy.local_path(args.continuation_scope[key]['path'])/
             'manifest.json' if key=='parent64_checkpoint' else
             accepted.legacy.local_path(args.continuation_scope[key]['path']),
            args.continuation_scope[key]['sha256']) for key in scope_contract.REFS])
    if any(path is None or sha256_file(path)!=pin for path,pin in checks):
        raise ValueError('NFR scope or original parent authority changed')


def run_stage(args,coordinator,device,runtime,determinism,report,tracker):
    spec=args.spec;manifest=spec['manifest'];recipe=spec['recipe'];batch_size=manifest['partition']['physical_batch_per_rank'][recipe.arm]
    model,source,imported=coordinator.call('construct and authenticate startup',lambda:construct(spec,device))
    optimizer=build_campaign_adamw(model,recipe,fused=True)
    if optimizer.state:raise ValueError('New constructed optimizer unexpectedly inherited moments')
    tokens=[row['counts']['valid_tokens'] for row in spec['plan']['updates']]
    scheduler=CampaignTokenSchedule(optimizer,tokens,warmup_tokens=recipe.warmup_tokens,start_fraction=recipe.warmup_start_fraction)
    if spec['kind']=='native' and scheduler.plan_sha256!=spec['resolved']['planning']['schedule']['plan_sha256']:
        raise ValueError('Actual scheduler differs from resolved finite token prefix')
    identity,ownership=coordinator.call('original execution identity',lambda:accepted.accepted.construct_identity(spec,recipe,model,optimizer,runtime,determinism,accepted.accepted.source_hashes()))
    configuration=contract.plain({'schema':accepted.accepted.engine.SCHEMA,'execution_identity':identity,'recipe':recipe.to_dict(),
        'model':model.config.to_dict(),'backbone':model.backbone.backbone.config.to_dict(),
        'parameters':ownership,'training':{'precision':manifest['execution']['precision'],'max_grad_norm':recipe.max_grad_norm},
        'schedule':scheduler.checkpoint_contract(),'source_checkpoint':source,
        'world_size':2,'warmup':engine.WARMUP,'seed_policy':'jitter_seed+rank process/data; +world_size CUDA generator',
        'ddp':{'static_graph':True,'find_unused_parameters':False,'broadcast_buffers':False,'gradient_as_bucket_view':False,'bucket_cap_mb':25}})
    original_fingerprint=accepted.accepted.source_fingerprint(identity,source,accepted.accepted.source_hashes())
    if (configuration!=args.original_parent['metadata']['configuration']
            or original_fingerprint!=args.original_parent['metadata']['source_fingerprint']):
        raise ValueError('Actual original runtime/configuration differs from parent')
    original_configuration=copy.deepcopy(configuration)
    branch={'schema':'olmo-kl-objective-branch-v1','kl_weight':args.kl_weight,
        'parent_manifest_sha256':args.parent_manifest_sha256,
        'parent_identity_sha256':identity['sha256'],
        'parent_update':args.original_parent['counters']['optimizer_updates'],'review_stop':64,
        'parent_report_sha256':args.parent_report_sha256,
        'recipe_as_declared':declared_recipe(recipe,kl_weight=args.kl_weight,
            parent_manifest_sha256=args.parent_manifest_sha256)}
    set_kl_weight(model,args.kl_weight)
    identity,ownership=branch_identity(identity,branch,recipe,model,optimizer,args.original_sources)
    configuration.update(schema=accepted.engine.SCHEMA,execution_identity=identity,recipe=branch['recipe_as_declared'],
        model=model.config.to_dict(),parameters=ownership,objective_branch=branch)
    configuration=contract.plain(configuration)
    parent_configuration=copy.deepcopy(configuration)
    parent_fingerprint=source_fingerprint(identity,source,args.original_sources)
    if (parent_configuration!=args.parent64['metadata']['configuration']
            or parent_fingerprint!=args.parent64['metadata']['source_fingerprint']):
        raise ValueError('Actual reduced64 configuration differs before strict resume')
    continuation=scope_contract.continuation_metadata(args.continuation_scope,args.scope_sha256,args.parent64)
    evaluation_plan=scope_contract.continuation_evaluation(spec['evaluation_plan'],args.continuation_scope)
    configuration=scope_contract.continued_configuration(parent_configuration,continuation,report['sources'],evaluation_plan)
    identity=configuration['execution_identity']
    fingerprint=source_fingerprint(identity,source,report['sources'])
    report.update(configuration=configuration,startup_import=imported,source_fingerprint=fingerprint,
        original_configuration=original_configuration,branch=branch,
        parent_report_sha256=args.parent_report_sha256,parent_manifest_sha256=args.parent_manifest_sha256)
    report.update(continuation=continuation,parent64_configuration=parent_configuration,
        parent64_fingerprint=parent_fingerprint,parent64_report_sha256=args.continuation_scope['parent64_report']['sha256'])
    effective_resume=legacy.local_path(args.continuation_scope['parent64_checkpoint']['path']) if args.resume is None else args.resume
    effective_resume_sha=args.continuation_scope['parent64_checkpoint']['sha256'] if args.resume is None else args.resume_manifest_sha256
    storage_prefix=args.storage_prefix+'/'+args.branch+'/'+args.output_dir.name
    storage=None
    def setup_storage():
        nonlocal storage
        storage=SSDCheckpointStorage.create(args.checkpoint_root,args.output_dir,
            execution_identity_sha256=identity['sha256'],
            storage_prefix=storage_prefix,
            keep_local_completed=storage_policy(spec)['keep_local_completed'],resume_source=effective_resume)
        return {'checkpoint_root':str(args.checkpoint_root),'evidence_dir':str(args.output_dir),
                'policy':storage_policy(spec)}
    report['storage']=coordinator.call('SSD ownership registration',setup_storage,rank_zero=True)
    evaluation=EvaluationController(evaluation_plan, manifest['data'], recipe,
        coordinator=coordinator,device=device,batch_size=batch_size,tracker=tracker,
        report=report,output_dir=args.output_dir,acceptance=args.observation=='acceptance',kl_weight=args.kl_weight)
    data_spec=manifest['data']
    with OrderedCampaignData(legacy.local_path(data_spec['corpus']),legacy.local_path(data_spec['index'])) as data:
        if data.manifest_sha256!=data_spec['index_manifest_sha256']:raise ValueError('Actual ordered index changed')
        partitions={arm:(2,batch) for arm,batch in manifest['partition']['physical_batch_per_rank'].items()}
        actual=contract.plan_updates(data,manifest['budget'],partitions)
        if actual!=spec['plan']:raise ValueError('Actual pure data plan differs')
        plans=[];cursor=data.cursor()
        for count in tokens:
            plan=data.peek_update(cursor,count)
            if plan is None or plan.counts.valid_tokens!=count:raise ValueError('Actual logical membership differs')
            plans.append(plan);cursor=plan.next_cursor
        maximum=len(plans) if args.stop_after is None else args.stop_after
        retention=manifest['retention']
        options=engine.SegmentOptions(args.output_dir,args.checkpoint_root,
            storage_prefix,maximum,
            retention['checkpoint_seconds'],retention['checkpoint_every_updates'],args.observation,
            effective_resume,effective_resume_sha,args.stop_file)
        engine.run_segment(options=options,coordinator=coordinator,model=model,recipe=recipe,data=data,plans=plans,
            optimizer=optimizer,scheduler=scheduler,configuration=configuration,source_fingerprint=fingerprint,
            device=device,tracker=tracker,report=report,batch_size=batch_size,
            validate_resume_metadata=validate_branch_resume,storage=storage,evaluation=evaluation,checkpoint_mode=spec['checkpoint_mode'],
            branch=branch,parent_configuration=parent_configuration,parent_fingerprint=parent_fingerprint,
            transition_parent=args.resume is None,parent_boundary_by_rank=args.parent64_boundary_by_rank,
            continuation=continuation,named_checkpoints=args.continuation_scope['named_checkpoints'])
        coordinator.call('final SSD authority validation',lambda:storage.validate(),rank_zero=True)


def main(argv=None):
    args=parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent'):
        raise RuntimeError('Use the required project GPU container')
    if os.environ.get('WORLD_SIZE')!='2' or any(os.environ.get(k)!='0' for k in ('NCCL_ASYNC_ERROR_HANDLING','TORCH_NCCL_ASYNC_ERROR_HANDLING')):
        raise RuntimeError('Use two torchrun ranks, both NCCL async flags0 and a bounded external launcher')
    scope=accepted.legacy.read_json(args.scope_declaration,args.scope_sha256)
    scope_contract.validate_arguments(args,scope)
    args.spec=accepted.load_spec(args)
    scope_contract.authenticate_parent64(args,scope)
    args.continuation_scope=scope
    activation=scope_contract.validate_activation(args.activation_receipt,args.activation_sha256,scope,args.scope_sha256)
    sources=scope_contract.source_hashes(args.scope_declaration,scope)
    rank=int(os.environ['LOCAL_RANK']);device,runtime,determinism=configure_cuda_runtime(rank)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl',timeout=timedelta(seconds=600),device_id=device)
    coordinator=Coordinator();tracker=None;started=time.monotonic()
    # Retain the established KL metrics/report structure. The new execution
    # schema and continuation authority are explicit inside configuration.
    report={'schema':accepted.SCHEMA,'status':'running','scale':args.spec['kind'],'arm':args.arm,'sources':sources,
        'observation_mode':args.observation,'checkpoint_mode':args.checkpoint_mode,'segment_stop_after':args.stop_after,
        'declaration_sha256':args.declaration_sha256,'resolved_sha256':args.resolved_sha256,
        'nfr128_scope':{'schema':'olmo-nfr-128-execution-scope-v1','declaration':scope,'sha256':args.scope_sha256},
        'activation':{'receipt':activation,'sha256':args.activation_sha256},
        'scope':'Same reduced-KL NFR64-to128; populated Adam, unchanged model/objective/data/LR; no precision clearance'}
    def setup():
        nonlocal tracker
        args.output_dir.mkdir(parents=True,exist_ok=False)
        for name,digest in sources.items():
            destination=args.output_dir/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,destination)
            if sha256_file(destination)!=digest:raise ValueError('Source snapshot changed')
        for source,name,digest in ((args.declaration,'declaration.json',args.declaration_sha256),
            (args.resolved,'resolved.json',args.resolved_sha256),
            (args.scope_declaration,'nfr128-scope.json',args.scope_sha256),
            (args.activation_receipt,'activation.json',args.activation_sha256)):
            shutil.copyfile(source,args.output_dir/name)
            if sha256_file(args.output_dir/name)!=digest:raise ValueError('Input snapshot changed')
        tracking=args.spec['manifest']['tracking']
        tracker=OnlineTracker(project=tracking['project'],entity=tracking['entity'],group='nfr-reduced-continuation128',
            output_dir=args.output_dir,name=args.output_dir.name,preserve_state=preserve_local_rng)
        tracker.start({'scope':report['scope'],'arm':args.arm,'startup':args.spec['startup'],
            'observation':args.observation,'kl_weight':args.kl_weight,'parent_manifest':args.parent_manifest_sha256,
            'nfr128_scope_sha256':args.scope_sha256,
            'resume_parent64_manifest':scope['parent64_checkpoint']['sha256']})
    healthy,error=True,None
    try:
        coordinator.call('stage setup/tracking',setup,rank_zero=True)
        with disable_autocast_weight_cache(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            run_stage_releasing_failure(args,coordinator,device,runtime,determinism,report,tracker,
                stage=run_stage)
        coordinator.call('final source/input identity',lambda:assert_sources_and_inputs(args,sources))
        report['status']='completed_plan' if report['plan_completed'] else 'stopped_at_boundary'
    except LifecycleError as exc:
        error=exc;report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)},failure_scope='coordinated host failure')
    except BaseException as exc:
        healthy=False;error=exc;report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},failure_scope='unknown update/CUDA/NCCL; external teardown')
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        if healthy:
            try:finalize_report(args.output_dir,coordinator,report,tracker,succeeded=error is None)
            finally:gc.collect();dist.destroy_process_group()
        elif args.output_dir.is_dir():write_json(args.output_dir/f'rank-{rank}-failure.json',report)
        if error is not None:raise error


if __name__=='__main__':main()
