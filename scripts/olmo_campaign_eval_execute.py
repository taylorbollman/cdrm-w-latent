#!/usr/bin/env python3
"""Explicit two-H100 component execution and same-lineage recovery.

Native declarations are resolved and checked before CUDA. The separately named
tiny acceptance schema exercises the same engine, never masquerading as a
pretrained/native declaration. No quality or precision clearance is implied.
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
from cdrm.pretrained.packed_campaign_data import PackedCampaignData
from scripts import olmo_campaign_execution_contract as contract
from scripts import olmo_campaign_manifest as legacy
from scripts import olmo_campaign_eval_engine as engine
from scripts.olmo_campaign_eval_control import resolve_evaluation, EvaluationController
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_campaign_loop import Coordinator,LifecycleError
from scripts.olmo_campaign_loop_guarded import run_stage_releasing_failure
from scripts.olmo_campaign_loop_run import finalize_report
from scripts.olmo_campaign_manifest_run import source_hashes as accepted_sources
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_fusion_startup_train import load_fusion_checkpoint
from scripts.olmo_packed_campaign_run import configure_cuda_runtime
from scripts.olmo_two_gpu_validate import preserve_local_rng

SCHEMA='olmo-campaign-eval-execute-report-v1'
TINY_SCHEMA='olmo-campaign-evaluation-tiny-acceptance-v1'
PROTOCOL=ROOT/'docs/reports/olmo-campaign-evaluation/protocol.md'


def source_hashes():
    from scripts.olmo_campaign_execute import source_hashes as execution_sources
    sources=execution_sources()
    for name in ('scripts/olmo_campaign_eval_engine.py','scripts/olmo_campaign_eval_execute.py',
                 'scripts/olmo_campaign_eval_control.py','scripts/olmo_campaign_evaluation.py',
                 'tests/test_campaign_eval_control.py','tests/test_campaign_eval_execute.py',
                 'tests/test_campaign_evaluation.py',
                 'docs/reports/olmo-campaign-evaluation/evaluator-contract.md',str(PROTOCOL.relative_to(ROOT))):
        sources[name]=sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('declaration','output-dir','checkpoint-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--declaration-sha256',required=True)
    parser.add_argument('--resolved',type=Path)
    parser.add_argument('--resolved-sha256')
    parser.add_argument('--arm',choices=ARMS,required=True)
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
    for name in ('output_dir','checkpoint_root'):
        path=getattr(args,name).resolve()
        if not path.is_relative_to(ROOT):parser.error('Evidence/checkpoints must stay under persistent project checkout')
        setattr(args,name,path)
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',args.output_dir.name):
        parser.error('Use a simple distinct segment output name')
    if args.output_dir.exists():parser.error('Use a new evidence directory; never overwrite a completed or failed attempt')
    if args.stop_after is not None and args.stop_after<0:parser.error('stop-after must be a nonnegative absolute update')
    return args


def require_execution_policy(manifest):
    expected={**legacy.EXECUTION_COMMON,**legacy.PATHS['bf16_mixed'],
              'precision':'bf16_mixed','graph_mode':'prepared_cuda_graph'}
    if (manifest['execution']!=expected or manifest['partition']['world_size']!=2
            or manifest['evaluation']['kind'] not in ('deferred','finite_pass_teacher_forced')):
        raise ValueError('Native launcher currently accepts two ranks/BF16 captured with declared deferred or finite-pass evaluation; no fallback')


def load_spec(args):
    """All native byte authorities resolve offline before CUDA initialization."""
    declaration=legacy.read_json(args.declaration,args.declaration_sha256)
    if declaration.get('schema')==contract.SCHEMA:
        if args.resolved is None:raise ValueError('Native execution requires a pinned prior resolution')
        declaration,resolved=contract.load_declaration(args.declaration,args.declaration_sha256,args.resolved,args.resolved_sha256)
        manifest=declaration['planning_manifest'];require_execution_policy(manifest)
        if args.arm not in manifest['arms']:raise ValueError('Selected arm is not declared')
        plan=contract.startup_plan(resolved,args.arm)
        recipe=contract.recipe_from_dict(plan['target_recipe'])
        result={'kind':'native','declaration':declaration,'resolved':resolved,'manifest':manifest,
                'recipe':recipe,'startup':plan,'plan':resolved['planning']['plan']}
    elif declaration.get('schema')==TINY_SCHEMA:
        if args.resolved is not None:raise ValueError('Tiny fixture has no native resolved-plan authority')
        legacy.exact_fields(declaration,('schema','arm','corpus','corpus_manifest_sha256','index',
            'index_manifest_sha256','storage_prefix','seed','evaluation'), 'tiny acceptance')
        if declaration['arm']!=args.arm or declaration['seed']!=20260929:
            raise ValueError('Tiny acceptance arm/seed differs')
        corpus,index=legacy.local_path(declaration['corpus']),legacy.local_path(declaration['index'])
        legacy.read_json(corpus/'manifest.json',declaration['corpus_manifest_sha256'])
        legacy.read_json(index/'manifest.json',declaration['index_manifest_sha256'])
        recipe=CampaignRecipe(args.arm,sequence_length=16,rt_layers=(0,1),effective_valid_tokens=80,
            warmup_tokens=240,document_policy='continuous-stream-v1')
        with PackedCampaignData(corpus,index) as data:
            if (data.length!=16 or data.split!='train' or data.manifest_sha256!=declaration['index_manifest_sha256']
                    or data.manifest['corpus_manifest_sha256']!=declaration['corpus_manifest_sha256']):
                raise ValueError('Tiny fixture data differs')
            plan=legacy.plan_updates(data,{'updates':3,'target_valid_tokens_per_update':80},{args.arm:(2,2)})
        manifest={'data':{'corpus':str(corpus),'index':str(index),'index_manifest_sha256':declaration['index_manifest_sha256']},
            'execution':{**legacy.EXECUTION_COMMON,**legacy.PATHS['fp32'],'precision':'fp32','graph_mode':'prepared_cuda_graph'},
            'budget':{'updates':3,'target_valid_tokens_per_update':80},
            'partition':{'world_size':2,'physical_batch_per_rank':{args.arm:2}},
            'retention':{'storage_prefix':declaration['storage_prefix'],'checkpoint_seconds':600,'checkpoint_every_updates':1},
            'tracking':{'entity':'taylorbollman','project':'pretrained-fbt-rt-nextlat','group':'campaign-evaluation'},
            'evaluation':declaration['evaluation']}
        result={'kind':'tiny','declaration':declaration,'resolved':None,'manifest':manifest,
            'recipe':recipe,'startup':{'kind':'deterministic-tiny-fresh-adam-acceptance-v1'},'plan':plan}
    else:raise ValueError('Unsupported explicit declaration schema')
    manifest=result['manifest']
    result['evaluation_plan']=resolve_evaluation(manifest['data'],manifest['evaluation'],
        length=result['recipe'].sequence_length,updates=len(result['plan']['updates']),
        partitions={arm:(2,batch) for arm,batch in manifest['partition']['physical_batch_per_rank'].items()})
    if result['kind']=='native' and result['evaluation_plan']!=result['resolved']['planning']['evaluation']:
        raise ValueError('Actual evaluation plan differs from resolved authority')
    if args.stop_after is not None and args.stop_after>len(result['plan']['updates']):
        raise ValueError('Segment boundary exceeds immutable finite plan')
    from scripts.olmo_campaign_loop_run import storage_location
    storage_location(result['manifest']['retention']['storage_prefix'])
    return result


def construct(spec,device):
    recipe=spec['recipe'];execution=spec['manifest']['execution']
    if spec['kind']=='native':
        artifacts=legacy.local_path(spec['manifest']['model']['artifacts'])
        authority=validate_prepared_manifest(artifacts)
        if authority['checkpoint']!=spec['resolved']['planning']['model_checkpoint_authority']:
            raise ValueError('Native weight authority changed since CPU resolution')
        state=load_native_state_dict(artifacts)
        base=OLMoTiledRTForCausalLM(OLMoConfig.native_1b(),device='meta',dtype=torch.float32,
            attention_backend='sdpa',attention_precision=execution['rt_attention_precision'],
            ordinary_activation_checkpointing=execution['ordinary_activation_checkpointing'],
            cast_weights_once=execution['cast_weights_once'],tile_backend=execution['rt_forward_tiles'],
            backward_tile_backend=execution['rt_backward_tiles'],backward_memory=execution['backward_memory'],
            reuse_rope=execution['reuse_rope'],kv_only_writes=execution['kv_only_writes'],
            ordinary_pointwise_backend=execution['ordinary_pointwise_backend'],ordinary_rope_backend=execution['ordinary_rope_backend'])
        base.load_state_dict(state,strict=True,assign=True);del state
        source=authority['checkpoint']
        startup=spec['startup']
        imported={}
        if startup['kind']==contract.ADAPTED:
            historical=contract.recipe_from_dict(startup['import_recipe'])
            model=build_campaign_model(base,historical).to(device).train()
            receipt=load_fusion_checkpoint(model,legacy.local_path(startup['checkpoint']['path']),source,
                expected_sha256=startup['checkpoint']['sha256'])
            contract.validate_import_receipt(receipt,startup)
            transition=engine.transition_imported_model(model,historical,recipe)
            imported={'receipt':receipt,'transition':transition}
        else:model=build_campaign_model(base,recipe).to(device).train()
        if (model.backbone.backbone.config.to_dict()!=spec['resolved']['planning']['native_backbone_config']
                or model.config.to_dict()!=startup['target_nextlat_config']):
            raise ValueError('Actual constructed/transitioned model differs from native declared configuration')
    else:
        data=spec['manifest']['data']
        with PackedCampaignData(legacy.local_path(data['corpus']),legacy.local_path(data['index'])) as reader:
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
    if spec['kind']=='native':
        identity=contract.execution_identity(spec['resolved'],recipe.arm,runtime=runtime,
            determinism=determinism,model_contract=ownership,extra_sources=sources)
        inventory=spec['resolved']['planning']['resource_cards'][recipe.arm]['parameters']['observed_inventory_from_retained_ledger']
        if (ownership['resident_parameters']!=inventory['registered_unique']
                or ownership['trainable_parameters']!=inventory['trainable']
                or ownership['trainable_parameters']!=inventory['optimizer_owned']):
            raise ValueError('Actual ownership differs from independently retained native ledger')
    else:
        payload={'scope':'tiny-acceptance-not-native','arm':recipe.arm,'cursor_schema':engine.CURSOR_SCHEMA,
            'resolved_contract_sha256':legacy.digest(spec['declaration']),'startup':spec['startup'],
            'recipe':recipe.to_dict(),'model_contract':ownership,'execution':spec['manifest']['execution'],
            'data':spec['manifest']['data'],'plan':spec['plan'],
            'partition':{'world_size':2,'physical_batch_per_rank':2},'runtime':runtime,
            'determinism':determinism,'sources':sources,'declaration':spec['declaration']}
        identity={'schema':contract.IDENTITY_SCHEMA,'payload':contract.plain(payload),'sha256':legacy.digest(payload)}
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
    report.update(configuration=configuration,startup_import=imported,source_fingerprint=fingerprint)
    evaluation=EvaluationController(spec['evaluation_plan'], manifest['data'], recipe,
        coordinator=coordinator,device=device,batch_size=batch_size,tracker=tracker,
        report=report,output_dir=args.output_dir,acceptance=args.observation=='acceptance')
    data_spec=manifest['data']
    with PackedCampaignData(legacy.local_path(data_spec['corpus']),legacy.local_path(data_spec['index'])) as data:
        if data.manifest_sha256!=data_spec['index_manifest_sha256']:raise ValueError('Actual packed index changed')
        partitions={arm:(2,batch) for arm,batch in manifest['partition']['physical_batch_per_rank'].items()}
        actual=legacy.plan_updates(data,manifest['budget'],partitions)
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
            validate_resume_metadata=contract.validate_resume_metadata,evaluation=evaluation)


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
        'observation_mode':args.observation,'segment_stop_after':args.stop_after,'declaration_sha256':args.declaration_sha256,
        'resolved_sha256':args.resolved_sha256,'scope':'Declared per-pass held-out evaluation and preserved training state; no quality or precision clearance'}
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


if __name__=='__main__':main()
