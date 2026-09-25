#!/usr/bin/env python3
"""Matched one-GPU graph baseline for a two-rank equal-global-batch comparison.

The physical batch concatenates the exact rank-0/rank-1 fixtures used by
olmo_two_gpu_graph.py. This uses the established single-device static runtime,
without DDP buckets or collectives. Cross-physical-batch BF16 equality is not
asserted; each candidate receives its own exact eager/graph checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from cdrm.pretrained.artifacts import write_json,sha256_file
from cdrm.pretrained.lm_training import LMTrainingConfig,TrainingCounters,optimizer_state_bytes
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.static_training import StaticFBTTraining
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import IntegrationCase
from scripts.olmo_two_gpu_validate import construct,preserve_local_rng
from scripts.olmo_two_gpu_graph import fixed_batch,steady_memory_summary
from scripts.olmo_large_batch_validation import (snapshot_eager_cpu,replay_compare_cpu,
    snapshot_replay_cpu,release_graph_then_compare_eager_cpu)
from scripts.olmo_rt_efficiency import resource_card
from scripts.olmo_rt_large_batch import (optimizer_for,compiler_configuration,
    configure_determinism,require_container_gpu,backend_context,load_native_tokenizer,
    validate_prepared_manifest,OnlineTracker,MemoryPhases,dependency_record,finish_tracking,
    detailed_memory_snapshot)

PROTOCOL=ROOT/'docs/reports/olmo-two-gpu/protocol.md'


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny',action='store_true')
    parser.add_argument('--case',choices=('rt','combined'),required=True)
    parser.add_argument('--batch-size',type=int,default=128,
                        help='One-GPU physical/global batch; paired DDP uses half on each rank')
    parser.add_argument('--length',type=int,default=512)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    args=parser.parse_args(argv)
    args.output_dir=args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error('Evidence must remain under the persistent project checkout')
    if args.batch_size<2 or args.batch_size%2:
        parser.error('The paired fixture requires a positive even physical batch')
    if not 8<=args.length<=2048 or (not args.tiny and args.length!=512):
        parser.error('Initial full-model scaling uses T512; tiny checks allow T8..2048')
    return args


def paired_batch(case,tokenizer,update,*,tiny):
    """Exact rank-shard token/mask concatenation; each row is its own document."""
    if case.batch_size<2 or case.batch_size%2:
        raise ValueError('Paired fixture requires an even physical batch')
    shard_case=replace(case,batch_size=case.batch_size//2)
    shards=[fixed_batch(shard_case,tokenizer,update,rank,tiny=tiny) for rank in (0,1)]
    fields={}
    for name in vars(shards[0]):
        values=[getattr(shard,name) for shard in shards]
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise ValueError(f'Rank fixtures disagree on optional field {name}')
            fields[name]=None
        else:
            fields[name]=torch.cat(values,dim=0)
    valid=fields['valid_mask']
    rows=torch.arange(case.batch_size,device=valid.device)[:,None].expand_as(valid)
    fields['document_ids']=torch.where(valid,rows,-1)
    return NextLatBatch(**fields)


def run(args,report,tracker):
    device=torch.device('cuda',torch.cuda.current_device())
    case=IntegrationCase(args.case,fbt=args.case=='combined',nextlat=args.case=='combined',
        rt_layers=(0,1 if args.tiny else 15),batch_size=args.batch_size,length=args.length)
    report.update(case=asdict(case),mode=asdict(case.mode()),physical_updates=0,
        configuration={**report['configuration'],'physical_batch_per_gpu':case.batch_size,
            'global_batch':case.batch_size,'paired_ddp_physical_batch_per_gpu':case.batch_size//2,
            'world_size':1,'accumulation_steps':1,'ddp_buckets':False,
            'capture_warmup_requested':11,'preparation_updates':3,'timed_updates':5,
            'fixture':'concatenated rank0/rank1 fixed_batch; independent row documents'})
    def persist():
        if report['status']=='running' and report.get('memory_phases'):
            report['stage']=next(reversed(report['memory_phases']))
        write_json(args.output_dir/'report.json',report)
    phases=MemoryPhases(report,persist)
    torch.random.default_generator.manual_seed(20260925)
    torch.cuda.manual_seed(20260925)
    config=LMTrainingConfig(precision='fp32' if args.tiny else 'bf16_mixed')
    with phases.phase('construct'):
        model=construct(case,args,device)
        tokenizer=None if args.tiny else load_native_tokenizer(args.artifacts)
        batch=paired_batch(case,tokenizer,0,tiny=args.tiny)
        plan=StaticFBTTraining(model,batch,mode=case.mode(),config=config)
        optimizer,scheduler=optimizer_for(model,'compiled-native')
        counters=TrainingCounters()
    with phases.phase('eager_adam_preparation'):
        for index in range(3):
            batch=paired_batch(case,tokenizer,index,tiny=args.tiny)
            metrics=plan.optimizer_step(optimizer,batch,replay=False,scheduler=scheduler,counters=counters)
            report['physical_updates']+=1
            report.setdefault('preparation_updates',[]).append(metrics);persist()
        report['optimizer_state_bytes_before_capture']=optimizer_state_bytes(optimizer)
        plan.validate_execution()
    batch=paired_batch(case,tokenizer,3,tiny=args.tiny)
    with phases.phase('pre_capture_eager_reference'):
        reference=snapshot_eager_cpu(plan,batch)
    plan.capture(warmup=11,release_transient_cache=True,phase_observer=phases.capture_observer)
    report['setup_memory']=phases.setup_summary()
    report['resources']=resource_card(plan,case,optimizer)
    with phases.phase('initial_graph_validation'):
        check=replay_compare_cpu(plan,reference,'pre_capture_eager_vs_graph',replays=2)
        report['checks'].append(check);persist()
        if not check['passed']: raise AssertionError(check['name'])
        del reference
    rates=[]
    for index in range(5):
        batch=paired_batch(case,tokenizer,4+index,tiny=args.tiny)
        with phases.phase(f'timed_update_{index}'):
            torch.cuda.synchronize();started=time.perf_counter()
            metrics=plan.optimizer_step(optimizer,batch,replay=True,scheduler=scheduler,counters=counters)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            report['physical_updates']+=1
            row=dict(update=counters.optimizer_updates,seconds=elapsed,
                input_tokens=case.batch_size*case.length,tokens_per_second=case.batch_size*case.length/elapsed,
                metrics=metrics)
            rates.append(row);report['timed_updates']=rates;persist()
        tracker.log({'update':counters.optimizer_updates,'benchmark/global_tokens_per_second':row['tokens_per_second'],
                     'benchmark/seconds_per_update':elapsed,'train/objective':metrics['objective']})
    total_seconds=sum(row['seconds'] for row in rates)
    report['throughput']=dict(global_tokens_per_second=5*case.batch_size*case.length/total_seconds,
        gpu_seconds_per_input_token=total_seconds/(5*case.batch_size*case.length),
        seconds_per_update=total_seconds/5,physical_batch=case.batch_size,global_batch=case.batch_size,
        timing_scope='batch validation/copy + captured forward/loss/backward + finite checks/clipping/Adam/scheduler; excluding fixture construction, reporting, digests; no distributed communication')
    report['steady_memory']=steady_memory_summary(report['memory_phases'],detailed_memory_snapshot())
    persist()
    # Recheck changed weights without the large eager-plus-live-graph memory peak.
    with phases.phase('final_graph_reference'):
        reference=snapshot_replay_cpu(plan)
    with phases.phase('final_graph_released_eager_validation'):
        check=release_graph_then_compare_eager_cpu(plan,reference,'changed_weight_graph_vs_eager')
        report['checks'].append(check);persist()
        if not check['passed']: raise AssertionError(check['name'])
        del reference
    report.update(status='passed',stage='complete',counters=asdict(counters),
        graph_counters=dict(warmup_backward_calls=plan.warmup_backward_calls,
            capture_backward_calls=plan.capture_backward_calls,replay_calls=plan.replay_calls),
        optimizer_state_bytes=optimizer_state_bytes(optimizer))
    persist()


def main(argv=None):
    args=parse_args(argv)
    configure_determinism(True)
    torch.set_num_threads(4)
    runtime=require_container_gpu()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');compiler_configuration()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=dict(schema='olmo-two-gpu-single-reference-v1',status='running',stage='setup',runtime=runtime,checks=[],
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    tracker=None;original_error=None
    try:
        sources=[*sorted((ROOT/'cdrm/pretrained').glob('*.py')),*sorted((ROOT/'scripts').glob('olmo*.py')),
                 ROOT/'scripts/experiment_tracking.py',ROOT/'scripts/docker_shell.sh']
        if PROTOCOL.exists(): sources.append(PROTOCOL)
        report['sources']={str(path.relative_to(ROOT)):sha256_file(path) for path in sources}
        report['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        report['dependencies']=dependency_record(args.output_dir,include_dao=False,include_fa4=False)
        if not args.tiny: report['checkpoint']=validate_prepared_manifest(args.artifacts)['checkpoint']
        for path in sources:
            target=args.output_dir/'source-snapshot'/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write_json(args.output_dir/'report.json',report)
        tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',group='olmo-two-gpu',
            name=args.output_dir.name,output_dir=args.output_dir,preserve_state=preserve_local_rng)
        tracker.start(report['configuration']);report['wandb']=tracker.record
        write_json(args.output_dir/'report.json',report)
        print(tracker.record['run_url'],flush=True)
        with disable_autocast_weight_cache(),backend_context('math' if args.tiny else 'flash'):
            run(args,report,tracker)
    except BaseException as error:
        original_error=error
        report.update(status='failed',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        write_json(args.output_dir/'report.json',report)
        try:
            if tracker is not None: finish_tracking(tracker,report,original_error=original_error)
        finally:
            write_json(args.output_dir/'report.json',report)


if __name__=='__main__':main()
