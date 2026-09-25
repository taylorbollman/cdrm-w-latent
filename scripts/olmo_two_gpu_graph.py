#!/usr/bin/env python3
"""Bounded two-rank actual-DDP CUDA graph correctness and complete-update timing.

Launch one process per GPU with torchrun. Set TORCH_NCCL_ASYNC_ERROR_HANDLING=0
and NCCL_ASYNC_ERROR_HANDLING=0 before process-group initialization and use an
external launcher timeout. Rank phase/report files are persisted throughout.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import gc
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from cdrm.pretrained.artifacts import write_json, sha256_file
from cdrm.pretrained.ddp_graph_training import PreparedDDPObjective, DDPGraphTraining
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, optimizer_state_bytes
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.resource_estimates import parameter_inventory
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import IntegrationCase, active_names, boundary_digests, inference_names
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import (construct, move_batch, cpu_copy, tensor_check,
    assert_all, preserve_local_rng, gather, TERMS)
from scripts.olmo_rt_large_batch import (batch_for, optimizer_for, compiler_configuration,
    configure_determinism, require_container_gpu, backend_context, load_native_tokenizer,
    validate_prepared_manifest, OnlineTracker, MemoryPhases, dependency_record, finish_tracking)

PROTOCOL = ROOT / 'docs/reports/olmo-two-gpu/protocol.md'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--case', choices=('ordinary','rt','combined'), required=True)
    parser.add_argument('--stage', choices=('correctness','capacity'), required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--bucket-view', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error('Evidence must remain under the persistent project checkout')
    if args.batch_size < 1 or not 8 <= args.length <= 2048:
        parser.error('Require positive physical batch and sequence length 8..2048')
    if args.stage == 'correctness' and args.batch_size > 8:
        parser.error('Complete-update correctness is bounded to physical B1..8')
    if args.stage == 'capacity' and not args.tiny and args.length != 512:
        parser.error('Initial full-model capacity uses T512')
    return args


def fixed_batch(case, tokenizer, update, rank, *, tiny):
    serial = 2*update + rank
    if tiny:
        generator = torch.Generator().manual_seed(20260925+serial)
        ids = torch.randint(2,59,(case.batch_size,case.length),generator=generator)
        valid = torch.ones_like(ids,dtype=torch.bool)
        docs = torch.arange(case.batch_size)[:,None].expand_as(ids).clone()
        return NextLatBatch(ids,valid,docs,valid.clone(),valid.clone(),valid.clone())
    # Preserve the accepted single-GPU supervision fixture and vary rank tokens.
    return batch_for(tokenizer,case,serial)


def snapshot_state(model, optimizer, scheduler, counters):
    return dict(model=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
                scheduler=cpu_copy(scheduler.state_dict()), counters=asdict(counters))


def restore_state(model, optimizer, scheduler, counters, snapshot):
    pointers = {n: None if p.grad is None else p.grad.data_ptr() for n,p in model.named_parameters()}
    model.load_state_dict(snapshot['model'], strict=True, assign=False)
    # Optimizer state is outside the graph; model/gradient storage is not replaced.
    optimizer.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
    for name,value in snapshot['counters'].items(): setattr(counters,name,value)
    if pointers != {n: None if p.grad is None else p.grad.data_ptr() for n,p in model.named_parameters()}:
        raise AssertionError('Restoring weights/optimizer replaced captured gradient storage')


def raw_snapshot(model, result):
    return dict(losses=cpu_copy(result), gradients={n: cpu_copy(p.grad)
        for n,p in model.named_parameters() if p.grad is not None})


def exact_raw_check(model, result, reference):
    actual = {n:p.grad for n,p in model.named_parameters() if p.grad is not None}
    rows = {}
    for name, value in actual.items():
        if name not in reference['gradients']:
            continue
        target = reference['gradients'][name]
        cpu_value = value.detach().cpu()
        # Exact equality needs no billion-element FP64 norm calculation. Keep
        # full diagnostics for mismatches and reject identical infinities.
        if torch.equal(cpu_value, target) and bool(torch.isfinite(cpu_value).all()):
            rows[name] = dict(passed=True, finite=True, exact=True,
                              relative_l2=0., max_absolute=0., max_relative=0.)
        else:
            rows[name] = tensor_check(cpu_value,target,atol=0.,rtol=0.)
    losses = tree_digests(result) == tree_digests(reference['losses'])
    ownership = actual.keys() == reference['gradients'].keys()
    return dict(passed=losses and ownership and all(r['passed'] for r in rows.values()),
                losses_exact=losses, ownership_exact=ownership, gradients=rows)


def finish_step(runtime, optimizer, scheduler, counters, result):
    """Global health, clipping and fused Adam remain outside capture."""
    sums = torch.stack([result['loss_sums'][term].to(torch.float64) for term in TERMS])
    # DDP gradients are replicated. Their norm must not be summed over ranks.
    norm = torch.nn.utils.clip_grad_norm_(runtime.model.parameters(),
        runtime.adapter.plan.config.max_grad_norm or float('inf'),
        error_if_nonfinite=False, foreach=False)
    healthy = (torch.isfinite(sums).all() & torch.isfinite(norm)).to(torch.int32)
    dist.all_reduce(healthy,op=dist.ReduceOp.MIN)
    if not bool(healthy.item()):
        raise FloatingPointError('At least one rank has nonfinite loss/gradient; no Adam step executed')
    dist.all_reduce(sums)
    values = dict(zip(TERMS,sums.cpu().tolist()))
    used_lr = [g['lr'] for g in optimizer.param_groups]
    optimizer.step(); scheduler.step()
    counters.optimizer_updates += 1
    counters.microbatches += runtime.adapter.world_size
    counters.documents += runtime.adapter.plan.documents * runtime.adapter.world_size
    counters.input_tokens += runtime.adapter.plan.input_tokens * runtime.adapter.world_size
    counts = runtime.adapter.global_counts
    counters.ce_positions += counts['ce']; counters.latent_pairs += counts['latent']
    counters.kl_triples += counts['kl']
    weights = runtime.adapter.plan.weights
    means = {t:values[t]/counts[t] if counts[t] else 0. for t in TERMS}
    return dict(loss_sums=values,counts=counts,loss_means=means,
                objective=sum(weights[t]*means[t] for t in TERMS),
                gradient_norm_before_clip=float(norm),lr_used=used_lr,
                lr_next=[g['lr'] for g in optimizer.param_groups],counters=asdict(counters))


def replica_check(model, optimizer, scheduler, counters):
    raw = tree_digests({n:p.grad for n,p in model.named_parameters() if p.grad is not None})
    states = boundary_digests(model,optimizer,scheduler,counters)
    records = gather(dict(gradients=raw,state=states))
    return dict(passed=all(record==records[0] for record in records),
                current_gradients_exact=all(r['gradients']==records[0]['gradients'] for r in records),
                state_exact=all(r['state']==records[0]['state'] for r in records))


def run(args, rank_report, tracker):
    rank = dist.get_rank(); device = torch.device('cuda',int(os.environ['LOCAL_RANK']))
    case = IntegrationCase(args.case, fbt=args.case=='combined',nextlat=args.case=='combined',
        rt_layers=() if args.case=='ordinary' else (0,1 if args.tiny else 15),
        batch_size=args.batch_size,length=args.length)
    rank_report['case'] = asdict(case)
    rank_report['mode'] = asdict(case.mode())
    rank_report['physical_updates'] = 0
    path = args.output_dir/f'rank-{rank}-progress.json'
    def persist():
        if rank_report['status'] == 'running' and rank_report.get('memory_phases'):
            rank_report['stage'] = next(reversed(rank_report['memory_phases']))
        write_json(path,rank_report)
    phases = MemoryPhases(rank_report,persist)
    config = LMTrainingConfig(precision='fp32' if args.tiny else 'bf16_mixed')
    rank_report['precision'] = config.precision
    torch.random.default_generator.manual_seed(20260925)
    torch.cuda.manual_seed(20260925)
    with phases.phase('construct'):
        model = construct(case,args,device)
        tokenizer = None if args.tiny else load_native_tokenizer(args.artifacts)
        cpu_batch = fixed_batch(case,tokenizer,0,rank,tiny=args.tiny)
        counts = torch.tensor([model.counts(cpu_batch)[t] for t in TERMS],device=device,dtype=torch.int64)
        dist.all_reduce(counts)
        adapter = PreparedDDPObjective(model,move_batch(cpu_batch,device),mode=case.mode(),
            global_counts=dict(zip(TERMS,counts.cpu().tolist())),world_size=2,config=config)
        runtime = DDPGraphTraining(adapter,expected_active_names=sorted(active_names(model,case.mode())),
                                   gradient_as_bucket_view=args.bucket_view)
        optimizer,scheduler = optimizer_for(model,'compiled-native')
        counters = TrainingCounters()
    runtime.prepare(warmup=11,phase_observer=phases.capture_observer)
    with phases.phase('eager_adam_preparation'):
        for index in range(3):
            runtime.load_batch(fixed_batch(case,tokenizer,index,rank,tiny=args.tiny))
            metrics = finish_step(runtime,optimizer,scheduler,counters,runtime.backward(replay=False))
            rank_report['physical_updates'] += 1
            rank_report.setdefault('preparation_updates',[]).append(metrics);persist()
        runtime.validate_execution()
        rank_report['optimizer_state_bytes_before_capture'] = optimizer_state_bytes(optimizer)
    # Keep an eager prepared-DDP reference before graph allocation. This avoids
    # the known large-batch eager-validation beside live graph memory overlap.
    runtime.load_batch(fixed_batch(case,tokenizer,3,rank,tiny=args.tiny))
    with phases.phase('pre_capture_eager_reference'):
        reference = raw_snapshot(model,runtime.backward(replay=False))
    runtime.capture(warmup=11,release_transient_cache=True,phase_observer=phases.capture_observer)
    rank_report['setup_memory'] = phases.setup_summary()
    rank_report['parameters'] = parameter_inventory(model,optimizer=optimizer,
        executed_names=runtime.expected_active_names,inference_names=inference_names(model,case))
    with phases.phase('initial_graph_validation'):
        result = runtime.backward(replay=True)
        check = exact_raw_check(model,result,reference)
        rank_report['checks'].append(dict(name='pre_capture_eager_vs_graph',**check));persist()
        assert_all(check['passed'],'initial prepared DDP eager/graph raw values')
        del result,reference
        replicas = replica_check(model,optimizer,scheduler,counters)
        rank_report['checks'].append(dict(name='initial_replicas',**replicas));persist()
        assert_all(replicas['passed'],'initial graph replica state/gradients')
    if args.stage == 'correctness':
        for index in range(2):
            with phases.phase(f'complete_update_parity_{index}'):
                batch = fixed_batch(case,tokenizer,4+index,rank,tiny=args.tiny)
                runtime.load_batch(batch)
                before = snapshot_state(model,optimizer,scheduler,counters)
                eager = runtime.backward(replay=False)
                reference = raw_snapshot(model,eager)
                expected_metrics = finish_step(runtime,optimizer,scheduler,counters,eager)
                rank_report['physical_updates'] += 1;persist()
                expected = boundary_digests(model,optimizer,scheduler,counters)
                restore_state(model,optimizer,scheduler,counters,before)
                runtime.validate_execution()
                del before,eager
                replay = runtime.backward(replay=True)
                check = exact_raw_check(model,replay,reference)
                assert_all(check['passed'],'changed-token graph raw gradient/loss parity')
                metrics = finish_step(runtime,optimizer,scheduler,counters,replay)
                rank_report['physical_updates'] += 1;persist()
                state_exact = boundary_digests(model,optimizer,scheduler,counters)==expected
                metric_exact = metrics==expected_metrics
                replicas = replica_check(model,optimizer,scheduler,counters)
                check.update(state_exact=state_exact,metrics_exact=metric_exact,replicas=replicas)
                check['passed'] &= state_exact and metric_exact and replicas['passed']
                rank_report['checks'].append(dict(name=f'changed_token_complete_update_{index}',**check));persist()
                assert_all(check['passed'],'complete eager/graph Adam update parity')
                del reference,replay,expected
                if rank==0: tracker.log({'update':counters.optimizer_updates,
                    'correctness/passed':int(check['passed']),'train/objective':metrics['objective']})
    else:
        rates=[]
        for index in range(5):
            batch = fixed_batch(case,tokenizer,4+index,rank,tiny=args.tiny)
            with phases.phase(f'timed_update_{index}'):
                dist.barrier();torch.cuda.synchronize(); started=time.perf_counter()
                runtime.load_batch(batch)
                metrics=finish_step(runtime,optimizer,scheduler,counters,runtime.backward(replay=True))
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                rank_report['physical_updates'] += 1
                times=gather(elapsed)
                row=dict(update=counters.optimizer_updates,rank_seconds=times,
                    slowest_rank_seconds=max(times),global_input_tokens=2*case.batch_size*case.length,
                    global_tokens_per_second=2*case.batch_size*case.length/max(times),
                    rank_tokens_per_second=[case.batch_size*case.length/t for t in times])
                rates.append(row);rank_report['timed_updates']=rates;persist()
            if rank==0: tracker.log({'update':counters.optimizer_updates,
                'benchmark/global_tokens_per_second':row['global_tokens_per_second'],
                'benchmark/slowest_rank_seconds':row['slowest_rank_seconds'],'train/objective':metrics['objective']})
        total_seconds=sum(row['slowest_rank_seconds'] for row in rates)
        rank_report['throughput']=dict(global_tokens_per_second=5*2*case.batch_size*case.length/total_seconds,
            slowest_rank_seconds_per_update=total_seconds/5,physical_batch_per_rank=case.batch_size,
            global_batch=2*case.batch_size,accumulation_steps=1,
            timing_scope='batch validation/copy + captured DDP forward/loss/backward + global health/loss + clipping/Adam/scheduler; excluding fixture construction, outer timing barrier, reporting and digests')
        with phases.phase('final_replica_check'):
            check=replica_check(model,optimizer,scheduler,counters)
            rank_report['checks'].append(dict(name='final_replicas',**check));persist()
            assert_all(check['passed'],'final capacity replica agreement')
    rank_report.update(runtime_metadata=runtime.metadata,counters=asdict(counters),
        ddp_logging_data=runtime.ddp._get_ddp_logging_data(),status='passed',stage='complete')
    persist()
    return rank_report


def main(argv=None):
    args=parse_args(argv)
    if os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING',os.environ.get('NCCL_ASYNC_ERROR_HANDLING'))!='0':
        raise RuntimeError('Launch graph probes with TORCH_NCCL_ASYNC_ERROR_HANDLING=0 before initializing NCCL')
    configure_determinism(True)
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    runtime=require_container_gpu()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');compiler_configuration()
    dist.init_process_group('nccl',device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])),
                            timeout=timedelta(minutes=30))
    if dist.get_world_size()!=2: raise RuntimeError('Requires exactly two ranks')
    rank=dist.get_rank();tracker=None;original_error=None
    rank_report=dict(schema='olmo-two-gpu-graph-rank-v1',rank=rank,status='running',stage='setup',checks=[])
    report=dict(schema='olmo-two-gpu-graph-v1',status='running',stage='setup',runtime=runtime,ranks=[],
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        if rank==0:
            args.output_dir.mkdir(parents=True,exist_ok=False)
            sources=[*sorted((ROOT/'cdrm/pretrained').glob('*.py')),*sorted((ROOT/'scripts').glob('olmo*.py')),
                     ROOT/'scripts/experiment_tracking.py',ROOT/'scripts/docker_shell.sh']
            if PROTOCOL.exists(): sources.append(PROTOCOL)
            report['sources']={str(p.relative_to(ROOT)):sha256_file(p) for p in sources}
            report['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
            report['dependencies']=dependency_record(args.output_dir,include_dao=False,include_fa4=False)
            if not args.tiny: report['checkpoint']=validate_prepared_manifest(args.artifacts)['checkpoint']
            for p in sources:
                target=args.output_dir/'source-snapshot'/p.relative_to(ROOT)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            write_json(args.output_dir/'report.json',report)
            tracker=OnlineTracker(project='pretrained-fbt-rt-nextlat',group='olmo-two-gpu',
                name=args.output_dir.name,output_dir=args.output_dir,preserve_state=preserve_local_rng)
            tracker.start(report['configuration']);report['wandb']=tracker.record
            write_json(args.output_dir/'report.json',report)
            print(tracker.record['run_url'],flush=True)
        dist.barrier()
        write_json(args.output_dir/f'rank-{rank}-progress.json',rank_report)
        with disable_autocast_weight_cache(),backend_context('math' if args.tiny else 'flash'):
            run(args,rank_report,tracker)
        reports=gather(rank_report)
        if rank==0:
            report.update(ranks=reports,status='passed',stage='complete',
                physical_updates=reports[0]['physical_updates'],
                rank_optimizer_steps=sum(r['physical_updates'] for r in reports),
                distributed_optimizer_updates=reports[0]['physical_updates'])
    except BaseException as error:
        original_error=error
        details=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        rank_report.update(status='failed',error=details)
        report.update(status='failed',error=details)
        if args.output_dir.exists(): write_json(args.output_dir/f'rank-{rank}-error.json',rank_report)
        raise
    finally:
        if rank==0 and args.output_dir.exists():
            write_json(args.output_dir/'report.json',report)
            try:
                if tracker is not None: finish_tracking(tracker,report,original_error=original_error)
            finally:
                write_json(args.output_dir/'report.json',report)
        # Never barrier in failure cleanup: the other rank can be stranded in a
        # failed collective. torchrun/external timeout owns whole-job termination.
        if original_error is None: dist.destroy_process_group()


if __name__=='__main__': main()
