#!/usr/bin/env python3
"""Explicit native NFR32-to64 KL-only scope; reuse accepted execution directly."""
from __future__ import annotations
import argparse
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
from scripts import olmo_nfr_kl_contract as scope_contract
from scripts.olmo_campaign_loop import Coordinator,LifecycleError
from scripts.olmo_campaign_loop_guarded import run_stage_releasing_failure
from scripts.olmo_campaign_loop_run import finalize_report
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_packed_campaign_run import configure_cuda_runtime
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.experiment_tracking import OnlineTracker


def parse_args(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scope-declaration',type=Path,required=True)
    parser.add_argument('--scope-sha256',required=True)
    extension,remaining=parser.parse_known_args(argv)
    args=accepted.parse_args(remaining)
    args.scope_declaration=extension.scope_declaration;args.scope_sha256=extension.scope_sha256
    accepted.legacy.pin(args.scope_sha256)
    return args


def assert_sources_and_inputs(args,sources):
    if scope_contract.source_hashes(args.scope_declaration)!=sources:
        raise ValueError('NFR scoped execution sources changed')
    checks=((args.scope_declaration,args.scope_sha256),(args.declaration,args.declaration_sha256),
        (args.resolved,args.resolved_sha256),(args.parent_report,args.parent_report_sha256),
        (args.parent_checkpoint/'manifest.json',args.parent_manifest_sha256))
    if any(path is None or sha256_file(path)!=pin for path,pin in checks):
        raise ValueError('NFR scope or original parent authority changed')


def main(argv=None):
    args=parse_args(argv)
    if not Path('/.dockerenv').exists() or Path.cwd()!=Path('/workspace/cdrm-w-latent'):
        raise RuntimeError('Use the required project GPU container')
    if os.environ.get('WORLD_SIZE')!='2' or any(os.environ.get(k)!='0' for k in ('NCCL_ASYNC_ERROR_HANDLING','TORCH_NCCL_ASYNC_ERROR_HANDLING')):
        raise RuntimeError('Use two torchrun ranks, both NCCL async flags0 and a bounded external launcher')
    scope=accepted.legacy.read_json(args.scope_declaration,args.scope_sha256)
    scope_contract.validate_arguments(args,scope)
    args.spec=accepted.load_spec(args)
    scope_contract.authenticated_parent(args,scope)
    sources=scope_contract.source_hashes(args.scope_declaration)
    rank=int(os.environ['LOCAL_RANK']);device,runtime,determinism=configure_cuda_runtime(rank)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl',timeout=timedelta(seconds=600),device_id=device)
    coordinator=Coordinator();tracker=None;started=time.monotonic()
    # Report structure and engine schema are intentionally the same; the new
    # scoped authority is separately versioned and bound through source identity.
    report={'schema':accepted.SCHEMA,'status':'running','scale':args.spec['kind'],'arm':args.arm,'sources':sources,
        'observation_mode':args.observation,'checkpoint_mode':args.checkpoint_mode,'segment_stop_after':args.stop_after,
        'declaration_sha256':args.declaration_sha256,'resolved_sha256':args.resolved_sha256,
        'nfr_scope':{'schema':'olmo-nfr-kl-execution-scope-v1','declaration':scope,'sha256':args.scope_sha256},
        'scope':'Paired NFR saved-Adam KL-only continuation; two native RT layers, K4, unchanged128plan; no precision clearance'}
    def setup():
        nonlocal tracker
        args.output_dir.mkdir(parents=True,exist_ok=False)
        for name,digest in sources.items():
            destination=args.output_dir/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,destination)
            if sha256_file(destination)!=digest:raise ValueError('Source snapshot changed')
        for source,name,digest in ((args.declaration,'declaration.json',args.declaration_sha256),
            (args.resolved,'resolved.json',args.resolved_sha256),
            (args.scope_declaration,'nfr-scope.json',args.scope_sha256)):
            shutil.copyfile(source,args.output_dir/name)
            if sha256_file(args.output_dir/name)!=digest:raise ValueError('Input snapshot changed')
        tracking=args.spec['manifest']['tracking']
        tracker=OnlineTracker(project=tracking['project'],entity=tracking['entity'],group='nfr-kl-paired-continuation',
            output_dir=args.output_dir,name=args.output_dir.name,preserve_state=preserve_local_rng)
        tracker.start({'scope':report['scope'],'arm':args.arm,'startup':args.spec['startup'],
            'observation':args.observation,'kl_weight':args.kl_weight,'parent_manifest':args.parent_manifest_sha256,
            'nfr_scope_sha256':args.scope_sha256})
    healthy,error=True,None
    try:
        coordinator.call('stage setup/tracking',setup,rank_zero=True)
        with disable_autocast_weight_cache(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            run_stage_releasing_failure(args,coordinator,device,runtime,determinism,report,tracker,
                stage=accepted.run_stage)
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
