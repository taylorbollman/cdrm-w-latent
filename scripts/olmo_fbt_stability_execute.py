#!/usr/bin/env python3
"""Clean F-only continuation with fixed K4 and sparse stability diagnostics.

Native declarations extend the accepted ordered cohort prefix to 192 updates;
the initial run stops at 128. Strict historical fusion import is followed by a
predictor-free wrapper and fresh all-active Adam. No old checkpoint is migrated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import re
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
from cdrm.pretrained.campaign_recipe import ARMS,CampaignRecipe,CampaignTokenSchedule,build_campaign_model,build_campaign_adamw
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict,validate_prepared_manifest
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from scripts import olmo_pilot_execution_contract as contract
from scripts import olmo_campaign_manifest as legacy
from scripts import olmo_fbt_stability_engine as engine
from scripts import olmo_fbt_stability_contract as stability
from scripts.olmo_pilot_execution_storage import SSDCheckpointStorage, validate_storage_paths
from scripts.olmo_pilot_eval_control import resolve_evaluation, EvaluationController
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_campaign_loop import Coordinator,LifecycleError
from scripts.olmo_campaign_loop_guarded import run_stage_releasing_failure
from scripts.olmo_campaign_loop_run import finalize_report
from scripts.olmo_campaign_manifest_run import source_hashes as accepted_sources
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_fusion_startup_train import load_fusion_checkpoint
from scripts.olmo_packed_campaign_run import configure_cuda_runtime
from scripts.olmo_two_gpu_validate import preserve_local_rng

SCHEMA='olmo-fbt-stability-execute-report-v1'
TINY_SCHEMA='olmo-fbt-stability-tiny-acceptance-v1'
PROTOCOL=ROOT/stability.PROTOCOL


def source_hashes():
    return stability.source_hashes()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('declaration','output-dir','checkpoint-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--declaration-sha256',required=True)
    parser.add_argument('--resolved',type=Path)
    parser.add_argument('--resolved-sha256')
    parser.add_argument('--arm',choices=('F',),required=True)
    parser.add_argument('--checkpoint-mode',choices=('async','blocking'),default='async')
    parser.add_argument('--observation',choices=('lean','acceptance'),default='lean')
    parser.add_argument('--stop-after',type=int,help='Absolute completed update boundary; immutable finite plan is unchanged')
    parser.add_argument('--stop-file',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-manifest-sha256')
    args=parser.parse_args(argv)
    for first,second in ((args.resolved,args.resolved_sha256),(args.resume,args.resume_manifest_sha256)):
        if (first is None)!=(second is None):parser.error('Path and independent SHA256 must be supplied together')
    for value in (args.declaration_sha256,args.resolved_sha256,args.resume_manifest_sha256):
        if value is not None:legacy.pin(value)
    try:
        validate_storage_paths(args.checkpoint_root,args.output_dir)
    except ValueError as error:parser.error(str(error))
    args.output_dir=args.output_dir.absolute()
    args.checkpoint_root=args.checkpoint_root.absolute()
    if args.checkpoint_root.exists():parser.error('Use a fresh SSD segment root; resumed states go in a separate restore directory')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',args.output_dir.name):
        parser.error('Use a simple distinct segment output name')
    if args.output_dir.exists():parser.error('Use a new evidence directory; never overwrite a completed or failed attempt')
    if args.stop_after is not None and args.stop_after<0:parser.error('stop-after must be a nonnegative absolute update')
    return args


def require_execution_policy(manifest):
    expected={**legacy.EXECUTION_COMMON,**legacy.PATHS['bf16_mixed'],
              'precision':'bf16_mixed','graph_mode':'prepared_cuda_graph'}
    if (manifest['execution']!=expected or manifest['partition']['world_size']!=2
            or manifest['evaluation']['kind'] not in ('deferred','ordered_named_dev_panels_v1')):
        raise ValueError('Native launcher currently accepts two ranks/BF16 captured with declared deferred or finite-pass evaluation; no fallback')


def load_spec(args):
    """All native byte authorities resolve offline before CUDA initialization."""
    if args.arm!='F':raise ValueError('This versioned launcher accepts clean F-only')
    if args.checkpoint_mode not in ('async','blocking'):
        raise ValueError('Require explicit async or blocking checkpoint transport')
    declaration=legacy.read_json(args.declaration,args.declaration_sha256)
    if declaration.get('schema')==stability.SCHEMA:
        if args.resolved is None:raise ValueError('Native execution requires a pinned F-only resolution')
        resolved=stability.resolve(declaration)
        if legacy.read_json(args.resolved,args.resolved_sha256,limit=128*1024**2)!=resolved:
            raise ValueError('Pinned F-only resolution changed')
        manifest=resolved['manifest'];require_execution_policy(manifest)
        recipe=contract.recipe_from_dict(resolved['recipe'])
        result={'kind':'native','declaration':declaration,'resolved':resolved,'manifest':manifest,
                'recipe':recipe,'startup':resolved['startup'],'plan':resolved['planning']['plan'],
                'probe_plan':resolved['probe_plan'],'checkpoint_updates':resolved['checkpoint_updates']}
    elif declaration.get('schema')==TINY_SCHEMA:
        if args.resolved is not None:raise ValueError('Tiny fixture has no native resolved-plan authority')
        legacy.exact_fields(declaration,('schema','arm','data','storage_prefix','seed','evaluation','probe'), 'tiny acceptance')
        if declaration['arm']!=args.arm or declaration['seed']!=20260929:
            raise ValueError('Tiny acceptance arm/seed differs')
        data_spec=declaration['data']
        contract.authenticate_data(data_spec,length=16)
        corpus,index=legacy.local_path(data_spec['corpus']),legacy.local_path(data_spec['index'])
        recipe=CampaignRecipe(args.arm,sequence_length=16,rt_layers=(0,1),effective_valid_tokens=80,
            warmup_tokens=240,document_policy='continuous-stream-v1')
        with OrderedCampaignData(corpus,index) as data:
            if (data.length!=16 or data.split!='train' or data.manifest_sha256!=data_spec['index_manifest_sha256']
                    or data.manifest['corpus_manifest_sha256']!=data_spec['corpus_manifest_sha256']):
                raise ValueError('Tiny fixture data differs')
            plan=contract.plan_updates(data,{'updates':3,'target_valid_tokens_per_update':80},{args.arm:(2,2)})
        manifest={'data':data_spec,
            'execution':{**legacy.EXECUTION_COMMON,**legacy.PATHS['fp32'],'precision':'fp32','graph_mode':'prepared_cuda_graph'},
            'budget':{'updates':3,'target_valid_tokens_per_update':80},
            'partition':{'world_size':2,'physical_batch_per_rank':{args.arm:2}},
            'retention':{'storage_prefix':declaration['storage_prefix'],'checkpoint_seconds':600,'checkpoint_every_updates':1,'keep_local_completed':2},
            'tracking':{'entity':'taylorbollman','project':'pretrained-fbt-rt-nextlat','group':'fbt-only-stability'},
            'evaluation':declaration['evaluation']}
        result={'kind':'tiny','declaration':declaration,'resolved':None,'manifest':manifest,
            'recipe':recipe,'startup':{'kind':'deterministic-tiny-fresh-adam-acceptance-v1'},'plan':plan}
    else:raise ValueError('Unsupported explicit declaration schema')
    manifest=result['manifest']
    result['evaluation_plan']=resolve_evaluation(manifest['data'],manifest['evaluation'],
        length=result['recipe'].sequence_length,updates=len(result['plan']['updates']),
        partitions={arm:(2,batch) for arm,batch in manifest['partition']['physical_batch_per_rank'].items()})
    if result['kind']=='native':
        result['evaluation_plan']['scheduled_updates']=sorted(set(result['evaluation_plan']['scheduled_updates'])|{0,100})
        if result['evaluation_plan']!=result['resolved']['evaluation_plan']:
            raise ValueError('Actual evaluation plan differs from resolved authority')
    else:
        result['evaluation_plan']['scheduled_updates']=sorted(set(result['evaluation_plan'].get('scheduled_updates',[]))|{0})
        from scripts.olmo_fbt_stability_probe import resolve_probe_plan
        result['probe_plan']=resolve_probe_plan(manifest['data'],length=16,updates=3,world_size=2,
            **declaration['probe'])
        # Exercise a probe with live graphs and another actual update afterward.
        result['probe_plan']['scheduled_updates']=[0,2,3]
        result['probe_plan']['acceptance_schedule']='origin-live-graph-next-update-terminal-v1'
        result['checkpoint_updates']=[]
    if args.stop_after is not None and args.stop_after>len(result['plan']['updates']):
        raise ValueError('Segment boundary exceeds immutable finite plan')
    result['checkpoint_mode']=args.checkpoint_mode
    storage_policy(result)
    from scripts.olmo_campaign_loop_run import storage_location
    storage_location(result['manifest']['retention']['storage_prefix'])
    return result


def storage_policy(spec):
    keep=spec['manifest']['retention']['keep_local_completed']
    if type(keep) is not int or not 2<=keep<=32:
        raise ValueError('SSD execution retains 2..32 verified local boundaries')
    return {'schema':'olmo-campaign-ssd-policy-v1','location':'local_ssd_verified_gcs',
            'keep_local_completed':keep,
            'transport':{'schema':'olmo-pilot-async-retention-v1',
                'mode':spec['checkpoint_mode'],'maximum_pending':1,
                'worker_timeout_seconds':480,'terminal_drain':True,
                'full_cloud_readback':True,'training_state_access':'none'}}


def construct(spec,device):
    recipe=spec['recipe'];execution=spec['manifest']['execution']
    if spec['kind']=='native':
        artifacts=legacy.local_path(spec['manifest']['model']['artifacts'])
        authority=validate_prepared_manifest(artifacts)
        if authority['checkpoint']!=spec['resolved']['planning']['model_checkpoint_authority']:
            raise ValueError('Native weight authority changed since resolution')
        state=load_native_state_dict(artifacts)
        base=OLMoTiledRTForCausalLM(OLMoConfig.native_1b(),device='meta',dtype=torch.float32,
            attention_backend='sdpa',attention_precision=execution['rt_attention_precision'],
            ordinary_activation_checkpointing=execution['ordinary_activation_checkpointing'],
            cast_weights_once=execution['cast_weights_once'],tile_backend=execution['rt_forward_tiles'],
            backward_tile_backend=execution['rt_backward_tiles'],backward_memory=execution['backward_memory'],
            reuse_rope=execution['reuse_rope'],kv_only_writes=execution['kv_only_writes'],
            ordinary_pointwise_backend=execution['ordinary_pointwise_backend'],ordinary_rope_backend=execution['ordinary_rope_backend'])
        base.load_state_dict(state,strict=True,assign=True);del state
        source=authority['checkpoint'];startup=spec['startup']
        historical=contract.recipe_from_dict(startup['import_recipe'])
        historical_model=build_campaign_model(base,historical).to(device).train()
        receipt=load_fusion_checkpoint(historical_model,legacy.local_path(startup['checkpoint']['path']),source,
            expected_sha256=startup['checkpoint']['sha256'])
        # Validate against the unmodified original NF importer authority.
        contract.validate_import_receipt(receipt,startup['historical_import_plan'])
        model,transition=stability.import_f_wrapper(historical_model,historical,recipe)
        del historical_model
        imported={'receipt':receipt,'transition':transition}
        if (model.backbone.backbone.config.to_dict()!=spec['resolved']['planning']['native_backbone_config']
                or model.config.to_dict()!=startup['target_nextlat_config']):
            raise ValueError('F-only constructed configuration differs')
    else:
        data=spec['manifest']['data']
        with OrderedCampaignData(legacy.local_path(data['corpus']),legacy.local_path(data['index'])) as reader:
            config=replace(OLMoConfig.tiny(),vocab_size=reader.manifest['vocab_size'],
                tokenizer_vocab_size=reader.manifest['vocab_size'],eos_token_id=reader.manifest['eos_id'],pad_token_id=reader.pad_id)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec['declaration']['seed'])
            base=OLMoTiledRTForCausalLM(config,attention_backend='math',attention_precision='fp32',
                tile_backend='eager',backward_tile_backend='eager',backward_memory='recompute',
                ordinary_activation_checkpointing=execution['ordinary_activation_checkpointing'],
                cast_weights_once=execution['cast_weights_once'],reuse_rope=True,kv_only_writes=True)
        model=build_campaign_model(base,recipe).to(device).train()
        source={'kind':'deterministic_random_tiny','seed':spec['declaration']['seed'],'model':config.to_dict()}
        imported={}
    return model,source,imported


def construct_identity(spec,recipe,model,optimizer,runtime,determinism,sources):
    ownership=engine.model_contract(model,recipe,optimizer)
    payload={'scope':'clean-F-only-stability' if spec['kind']=='native' else 'F-only-tiny-acceptance-not-native',
        'arm':recipe.arm,'cursor_schema':engine.CURSOR_SCHEMA,
        'resolved_contract_sha256':spec['resolved']['contract_sha256'] if spec['resolved'] else legacy.digest(spec['declaration']),
        'startup':spec['startup'],'recipe':recipe.to_dict(),'model_contract':ownership,
        'execution':spec['manifest']['execution'],'data':spec['manifest']['data'],'plan':spec['plan'],
        'partition':{'world_size':2,'physical_batch_per_rank':spec['manifest']['partition']['physical_batch_per_rank'][recipe.arm]},
        'evaluation':spec['evaluation_plan'],'probe_plan':spec['probe_plan'],
        'checkpoint_updates':spec['checkpoint_updates'],
        'runtime':runtime,'determinism':determinism,'sources':sources,'declaration':spec['declaration']}
    if spec['kind']=='native':
        payload['schedule']=spec['resolved']['planning']['schedule']
        inventory=spec['resolved']['planning']['resource_cards'][recipe.arm]['parameters']['observed_inventory_from_retained_ledger']
        if (ownership['resident_parameters']!=inventory['registered_unique']
                or ownership['trainable_parameters']!=inventory['trainable']
                or ownership['trainable_parameters']!=inventory['optimizer_owned']):
            raise ValueError('Actual F ownership differs from native ledger')
    identity={'schema':contract.IDENTITY_SCHEMA,'payload':contract.plain(payload),'sha256':legacy.digest(payload)}
    identity['payload']['storage_policy']=storage_policy(spec)
    identity['sha256']=legacy.digest(identity['payload'])
    return contract.plain(identity),ownership


def source_fingerprint(identity,source,sources):
    # Generic checkpoint metadata requires an explicit top-level byte identity.
    # This digest identifies the whole execution origin, not only native weights.
    return {'sha256':identity['sha256'],'scope':'execution_identity',
        'execution_identity_sha256':identity['sha256'],'source_checkpoint':source,'sources':sources}


def run_stage(args,coordinator,device,runtime,determinism,report,tracker):
    spec=args.spec;manifest=spec['manifest'];recipe=spec['recipe'];batch_size=manifest['partition']['physical_batch_per_rank'][recipe.arm]
    model,source,imported=coordinator.call('construct and authenticate startup',lambda:construct(spec,device))
    optimizer=build_campaign_adamw(model,recipe,fused=True)
    if optimizer.state:raise ValueError('New constructed optimizer unexpectedly inherited moments')
    tokens=[row['counts']['valid_tokens'] for row in spec['plan']['updates']]
    scheduler=CampaignTokenSchedule(optimizer,tokens,warmup_tokens=recipe.warmup_tokens,start_fraction=recipe.warmup_start_fraction)
    if spec['kind']=='native' and scheduler.plan_sha256!=spec['resolved']['planning']['schedule']['plan_sha256']:
        raise ValueError('Actual scheduler differs from resolved finite token prefix')
    identity,ownership=coordinator.call('actual execution identity',lambda:construct_identity(spec,recipe,model,optimizer,runtime,determinism,report['sources']))
    configuration=contract.plain({'schema':engine.SCHEMA,'execution_identity':identity,'recipe':recipe.to_dict(),
        'model':model.config.to_dict(),'backbone':model.backbone.backbone.config.to_dict(),
        'parameters':ownership,'training':{'precision':manifest['execution']['precision'],'max_grad_norm':recipe.max_grad_norm},
        'schedule':scheduler.checkpoint_contract(),'source_checkpoint':source,
        'world_size':2,'warmup':engine.WARMUP,'seed_policy':'jitter_seed+rank process/data; +world_size CUDA generator',
        'ddp':{'static_graph':True,'find_unused_parameters':False,'broadcast_buffers':False,'gradient_as_bucket_view':False,'bucket_cap_mb':25}})
    fingerprint=source_fingerprint(identity,source,report['sources'])
    report.update(configuration=configuration,startup_import=imported,source_fingerprint=fingerprint,
        prefix_checks=spec['resolved']['prefix_checks'] if spec['resolved'] else {})
    storage=None
    def setup_storage():
        nonlocal storage
        storage=SSDCheckpointStorage.create(args.checkpoint_root,args.output_dir,
            execution_identity_sha256=identity['sha256'],
            storage_prefix=manifest['retention']['storage_prefix']+'/'+recipe.arm+'/'+args.output_dir.name,
            keep_local_completed=storage_policy(spec)['keep_local_completed'],resume_source=args.resume)
        return {'checkpoint_root':str(args.checkpoint_root),'evidence_dir':str(args.output_dir),
                'policy':storage_policy(spec)}
    report['storage']=coordinator.call('SSD ownership registration',setup_storage,rank_zero=True)
    evaluation=EvaluationController(spec['evaluation_plan'], manifest['data'], recipe,
        coordinator=coordinator,device=device,batch_size=batch_size,tracker=tracker,
        report=report,output_dir=args.output_dir,acceptance=args.observation=='acceptance')
    from scripts.olmo_fbt_stability_probe import StabilityEvaluationController
    evaluation=StabilityEvaluationController(evaluation,spec['probe_plan'])
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
            retention['storage_prefix']+'/'+recipe.arm+'/'+args.output_dir.name,maximum,
            retention['checkpoint_seconds'],retention['checkpoint_every_updates'],args.observation,
            args.resume,args.resume_manifest_sha256,args.stop_file)
        engine.run_segment(options=options,coordinator=coordinator,model=model,recipe=recipe,data=data,plans=plans,
            optimizer=optimizer,scheduler=scheduler,configuration=configuration,source_fingerprint=fingerprint,
            device=device,tracker=tracker,report=report,batch_size=batch_size,
            validate_resume_metadata=contract.validate_resume_metadata,storage=storage,evaluation=evaluation,checkpoint_mode=spec['checkpoint_mode'],checkpoint_updates=spec['checkpoint_updates'])
        coordinator.call('final SSD authority validation',lambda:storage.validate(),rank_zero=True)


def main(argv=None):
    args=parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent'):
        raise RuntimeError('Use the required project GPU container')
    if os.environ.get('WORLD_SIZE')!='2' or any(os.environ.get(k)!='0' for k in ('NCCL_ASYNC_ERROR_HANDLING','TORCH_NCCL_ASYNC_ERROR_HANDLING')):
        raise RuntimeError('Use two torchrun ranks, both NCCL async flags0 and a bounded external launcher')
    args.spec=load_spec(args)
    sources=source_hashes()
    rank=int(os.environ['LOCAL_RANK']);device,runtime,determinism=configure_cuda_runtime(rank)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl',timeout=timedelta(seconds=600),device_id=device)
    coordinator=Coordinator();tracker=None;started=time.monotonic()
    report={'schema':SCHEMA,'status':'running','scale':args.spec['kind'],'arm':args.arm,'sources':sources,
        'observation_mode':args.observation,'checkpoint_mode':args.checkpoint_mode,'segment_stop_after':args.stop_after,'declaration_sha256':args.declaration_sha256,
        'resolved_sha256':args.resolved_sha256,'scope':'Clean F-only with strict fusion128 weights, fresh Adam, fixed K4 and scheduled stability probes; no quality or precision clearance'}
    def setup():
        nonlocal tracker
        args.output_dir.mkdir(parents=True,exist_ok=False)
        for name,digest in sources.items():
            destination=args.output_dir/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,destination)
            if sha256_file(destination)!=digest:raise ValueError('Source snapshot changed')
        for source,name,digest in ((args.declaration,'declaration.json',args.declaration_sha256),(args.resolved,'resolved.json',args.resolved_sha256)):
            if source is not None:
                shutil.copyfile(source,args.output_dir/name)
                if sha256_file(args.output_dir/name)!=digest:raise ValueError('Input snapshot changed')
        if args.spec['kind']=='native':
            for key in ('parent_declaration','parent_resolved','parent_report'):
                reference=args.spec['declaration'][key]
                destination=args.output_dir/(key+'.json')
                shutil.copyfile(reference['path'],destination)
                if sha256_file(destination)!=reference['sha256']:raise ValueError('Parent authority changed')
        tracking=args.spec['manifest']['tracking']
        tracker=OnlineTracker(project=tracking['project'],entity=tracking['entity'],group=tracking['group'],
            output_dir=args.output_dir,name=args.output_dir.name,preserve_state=preserve_local_rng)
        tracker.start({'scope':report['scope'],'arm':args.arm,'startup':args.spec['startup'],'observation':args.observation})
    healthy,error=True,None
    try:
        coordinator.call('stage setup/tracking',setup,rank_zero=True)
        backend=SDPBackend.FLASH_ATTENTION if args.spec['kind']=='native' else SDPBackend.MATH
        with disable_autocast_weight_cache(),sdpa_kernel(backend):
            run_stage_releasing_failure(args,coordinator,device,runtime,determinism,report,tracker,stage=run_stage)
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


def assert_sources_and_inputs(args,sources):
    if source_hashes()!=sources or sha256_file(args.declaration)!=args.declaration_sha256:
        raise ValueError('Source/declaration changed during execution')
    if args.resolved is not None and sha256_file(args.resolved)!=args.resolved_sha256:
        raise ValueError('Resolved contract changed during execution')
    if args.spec['kind']=='native':
        for key in ('parent_declaration','parent_resolved','parent_report'):
            reference=args.spec['declaration'][key]
            if sha256_file(Path(reference['path']))!=reference['sha256']:
                raise ValueError('Parent authority changed during execution')


if __name__=='__main__':main()
