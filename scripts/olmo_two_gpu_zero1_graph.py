#!/usr/bin/env python3
"""Bounded actual-DDP graph integration/capacity with native ZeRO-1 AdamW.

Forward/loss/backward/reducer collectives are captured. Clipping, health checks,
local Adam and updated-parameter broadcasts remain outside capture and inside
the full-update timing. Parameters/gradients are replicated; moments are
sharded. This probe does not save checkpoints or consolidate optimizer state.
Its separate ZeRO-1 correctness/recovery prerequisite must already be passed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import gc
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
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.zero1_training import zero1_state_inventory, zero1_checkpoint_configuration
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import IntegrationCase, active_names
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_rt_efficiency import resource_card
from scripts.olmo_two_gpu_graph import (fixed_batch, raw_snapshot, exact_raw_check,
    finish_step, aggregate_resources, steady_memory_summary)
from scripts.olmo_two_gpu_recovery import execution_configuration, seed_local
from scripts.olmo_two_gpu_zero1 import make_optimizer
from scripts.olmo_two_gpu_validate import construct, move_batch, gather, assert_all, preserve_local_rng, TERMS
from scripts.olmo_rt_large_batch import (compiler_configuration, configure_determinism,
    require_container_gpu, backend_context, load_native_tokenizer, validate_prepared_manifest,
    OnlineTracker, MemoryPhases, dependency_record, finish_tracking, detailed_memory_snapshot)

PROTOCOL = ROOT/'docs/reports/olmo-two-gpu/protocol.md'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--case', choices=('rt', 'combined'), required=True)
    parser.add_argument('--stage', choices=('integration', 'capacity'), default='capacity')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--bucket-view', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error('Evidence must remain under the persistent project checkout')
    if not 1 <= args.batch_size <= 512 or not 8 <= args.length <= 512:
        parser.error('Require physical B1..512 and T8..512')
    if not args.tiny and args.length != 512:
        parser.error('Initial actual-model graph scope uses T512')
    if args.stage == 'integration' and args.batch_size > 8:
        parser.error('Bound initial integration to physical B1..8')
    return args


def summarize_replica_records(records, expected_owned_names, expected_active_names):
    """Check replicated data and disjoint moment shards without conflating them."""
    if not records:
        raise ValueError('Replica inventory is empty')
    owned = [name for record in records for name in record['inventory']['local_owned_names']]
    states = [name for record in records for name in record['local_moment_digests']]
    owned_exact = len(owned) == len(set(owned)) and set(owned) == set(expected_owned_names)
    state_exact = len(states) == len(set(states)) and set(states) == set(expected_active_names)
    local_scope = all(set(record['local_moment_digests']).issubset(record['inventory']['local_owned_names'])
                      for record in records)
    replicas = all(record['replicated'] == records[0]['replicated'] for record in records)
    healthy = all(record['moment_schema_valid'] and record['finite_local_moments'] and record['step_counters_match'] and
                  record['fp32_local_state'] and not record['inventory']['outer_state_bytes_by_device'] and
                  not record['inventory']['consolidation_cache_present'] for record in records)
    return dict(passed=owned_exact and state_exact and local_scope and replicas and healthy,
        replicated_parameters_gradients_scheduler_counters_exact=replicas,
        parameter_partition_exact=owned_exact, active_moment_partition_exact=state_exact,
        local_state_belongs_to_owner=local_scope, state_health_and_steps_valid=healthy,
        rank_records=records,
        global_optimizer_state_bytes=sum(sum(record['inventory']['local_state_bytes_by_device'].values())
                                         for record in records),
        scope='Parameters, reduced gradients, scheduler and counters are replicas. Local moment hashes describe disjoint shards; moments are not claimed to be replicated or independently revalidated against full Adam in this benchmark.')


def local_state_health(states, expected_step):
    """Inspect the actual local Adam tensors, including scalar fused step state."""
    schema = all(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and
                 all(isinstance(value, torch.Tensor) for value in state.values()) and
                 state['step'].ndim == 0 and
                 state['exp_avg'].shape == parameter.shape and
                 state['exp_avg_sq'].shape == parameter.shape
                 for parameter, state in states.items())
    tensors = [value for state in states.values() for value in state.values()
               if isinstance(value, torch.Tensor)]
    finite = all(bool(torch.isfinite(value).all()) for value in tensors)
    fp32 = all(value.dtype == torch.float32 and value.device == parameter.device
               for parameter, state in states.items() for value in state.values()
               if isinstance(value, torch.Tensor))
    steps = schema and all(state['step'].item() == expected_step for state in states.values())
    return dict(moment_schema_valid=schema, finite_local_moments=finite,
                fp32_local_state=fp32, step_counters_match=steps)


def replica_and_shard_check(runtime, optimizer, scheduler, counters):
    model = runtime.model
    inventory = zero1_state_inventory(model, optimizer)
    by_id = {id(parameter): name for name,parameter in model.named_parameters()}
    local_moments = {by_id[id(parameter)]: tree_digests(state)
                     for parameter,state in optimizer.optim.state.items()}
    replicated = tree_digests(dict(model=model.state_dict(), scheduler=scheduler.state_dict(),
        counters=asdict(counters), gradients={name: parameter.grad for name,parameter in model.named_parameters()
                                            if parameter.grad is not None}))
    record = dict(inventory=inventory, local_moment_digests=local_moments,
                  **local_state_health(optimizer.optim.state, counters.optimizer_updates),
                  replicated=replicated)
    records = gather(record)
    return summarize_replica_records(records,
        [name for name,parameter in model.named_parameters() if parameter.requires_grad],
        runtime.expected_active_names)


def release_graph_for_eager(runtime, *, synchronize=None, cleanup=None):
    """Terminal release of graph handles while retaining the same DDP reducer.

    This deliberately does not permit another capture: the runner remains
    single-use for capture. Persistent gradients were allocated before capture
    and remain owned by the prepared reducer. No grad=None or new DDP wrapper.
    Optional callbacks make the lifetime/order contract CPU-testable.
    """
    runtime.validate_execution()
    if runtime.graph is None:
        raise ValueError('Terminal release requires an existing captured graph')
    synchronize = synchronize or (lambda: torch.cuda.synchronize(runtime.device))
    cleanup = cleanup or (lambda: (gc.collect(), torch.cuda.empty_cache()))
    synchronize()
    addresses = runtime._addresses()
    reducer = runtime.ddp
    runtime.graph_result = None
    runtime.graph = None
    cleanup()
    runtime.validate_execution()
    if runtime.ddp is not reducer or runtime._addresses() != addresses:
        raise AssertionError('Graph release replaced reducer or persistent gradient storage')
    return dict(graph_released=True, same_ddp_reducer=True, gradient_addresses_preserved=True)


def run(args, report, tracker):
    rank = dist.get_rank(); device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    case = IntegrationCase(args.case, fbt=args.case=='combined', nextlat=args.case=='combined',
        rt_layers=(0, 1 if args.tiny else 15), batch_size=args.batch_size, length=args.length)
    config = LMTrainingConfig(precision='fp32' if args.tiny else 'bf16_mixed')
    report.update(case=asdict(case), mode=asdict(case.mode()), physical_updates=0)
    def persist():
        if report['status'] == 'running' and report.get('memory_phases'):
            report['stage'] = next(reversed(report['memory_phases']))
        write_json(args.output_dir/f'rank-{rank}-progress.json', report)
    phases = MemoryPhases(report, persist)
    def gate(name, result):
        report['checks'].append(dict(name=name, **result)); persist()
        assert_all(result['passed'], name)
    with phases.phase('construct'):
        seed_local(20260925, device)
        model = construct(case, args, device)
        tokenizer = None if args.tiny else load_native_tokenizer(args.artifacts)
        batch = fixed_batch(case, tokenizer, 0, rank, tiny=args.tiny)
        counts = torch.tensor([model.counts(batch)[term] for term in TERMS], device=device, dtype=torch.int64)
        dist.all_reduce(counts)
        adapter = PreparedDDPObjective(model, move_batch(batch, device), mode=case.mode(),
            global_counts=dict(zip(TERMS, counts.cpu().tolist())), world_size=2, config=config)
        runtime = DDPGraphTraining(adapter, expected_active_names=sorted(active_names(model, case.mode())),
                                   gradient_as_bucket_view=args.bucket_view)
        optimizer, scheduler = make_optimizer(model)
        counters = TrainingCounters()
        configuration = execution_configuration(model, case, config, args)
        configuration.update(schema='olmo-two-gpu-zero1-graph-execution-v1', graphs=True,
            ddp=dict(static_graph=True, find_unused_parameters=False, broadcast_buffers=False,
                     gradient_as_bucket_view=args.bucket_view),
            zero1=zero1_checkpoint_configuration(model, optimizer, {})['_zero1'],
            optimizer_outside_graph=True, parameter_broadcast_outside_graph=True)
        report['configuration'] = configuration
    runtime.prepare(warmup=11, phase_observer=phases.capture_observer)
    with phases.phase('eager_adam_preparation'):
        for index in range(3):
            runtime.load_batch(fixed_batch(case, tokenizer, index, rank, tiny=args.tiny))
            metrics = finish_step(runtime, optimizer, scheduler, counters, runtime.backward(replay=False))
            report['physical_updates'] += 1
            report.setdefault('preparation_updates', []).append(metrics); persist()
        runtime.validate_execution()
        report['state_inventory_before_capture'] = zero1_state_inventory(model, optimizer)
        report['resources'] = resource_card(adapter.plan, case, optimizer)
        report['resident_state'] = dict(
            parameter_bytes=report['resources']['observed_parameters']['resident_parameter_bytes'],
            logical_gradient_bytes=sum(parameter.grad.numel()*parameter.grad.element_size()
                                       for parameter in model.parameters() if parameter.grad is not None),
            optimizer_state_bytes_by_device=report['state_inventory_before_capture']['local_state_bytes_by_device'],
            optimizer_storage='actual local ZeRO-1 Adam shard; outer optimizer.state is empty',
            scope='Replicated parameters and logical gradients plus local moment/step tensors. Excludes separate communication buffers, graph pools, activations and allocator cache.')
    runtime.load_batch(fixed_batch(case, tokenizer, 3, rank, tiny=args.tiny))
    with phases.phase('pre_capture_eager_reference'):
        reference = raw_snapshot(model, runtime.backward(replay=False))
    runtime.capture(warmup=11, release_transient_cache=True, phase_observer=phases.capture_observer)
    report['setup_memory'] = phases.setup_summary()
    with phases.phase('initial_graph_validation'):
        result = runtime.backward(replay=True)
        gate('pre_capture_eager_vs_graph', exact_raw_check(model, result, reference))
        del result, reference
        gate('initial_replicas_and_shards', replica_and_shard_check(runtime, optimizer, scheduler, counters))
    rates = []
    for index in range(5):
        batch = fixed_batch(case, tokenizer, 4+index, rank, tiny=args.tiny)
        with phases.phase(f'timed_update_{index}'):
            dist.barrier(); torch.cuda.synchronize(); started = time.perf_counter()
            runtime.load_batch(batch)
            metrics = finish_step(runtime, optimizer, scheduler, counters, runtime.backward(replay=True))
            torch.cuda.synchronize(); elapsed = time.perf_counter()-started
            report['physical_updates'] += 1
            times = gather(elapsed)
            row = dict(update=counters.optimizer_updates, rank_seconds=times,
                slowest_rank_seconds=max(times), global_input_tokens=2*case.batch_size*case.length,
                global_tokens_per_second=2*case.batch_size*case.length/max(times),
                rank_tokens_per_second=[case.batch_size*case.length/value for value in times])
            rates.append(row); report['timed_updates'] = rates; persist()
        if rank == 0: tracker.log({'update': counters.optimizer_updates,
            'benchmark/global_tokens_per_second': row['global_tokens_per_second'],
            'benchmark/slowest_rank_seconds': row['slowest_rank_seconds'], 'train/objective': metrics['objective']})
    seconds = sum(row['slowest_rank_seconds'] for row in rates)
    report['throughput'] = dict(global_tokens_per_second=5*2*case.batch_size*case.length/seconds,
        gpu_seconds_per_input_token=seconds/(5*case.batch_size*case.length),
        slowest_rank_seconds_per_update=seconds/5, physical_batch_per_rank=case.batch_size,
        global_batch=2*case.batch_size, accumulation_steps=1,
        timing_scope='batch validation/copy + captured actual DDP forward/loss/backward + coordinated health/loss + clipping + local fused Adam + updated-parameter broadcasts + scheduler; excludes fixtures, outer timing barrier, reporting, hashing and validation')
    report['steady_memory'] = steady_memory_summary(report['memory_phases'], detailed_memory_snapshot())
    persist()
    # The terminal raw comparison uses the changed weights after all five steps,
    # and preserves the same prepared layout/reducer while releasing the graph.
    with phases.phase('terminal_graph_reference'):
        result = runtime.backward(replay=True)
        reference = raw_snapshot(model, result)
        del result
        gate('final_replicas_and_shards', replica_and_shard_check(runtime, optimizer, scheduler, counters))
    with phases.phase('terminal_graph_release'):
        report['terminal_release'] = release_graph_for_eager(runtime)
        persist()
    with phases.phase('terminal_eager_comparison'):
        result = runtime.backward(replay=False)
        check = exact_raw_check(model, result, reference)
        check.update(report['terminal_release'])
        gate('changed_weight_graph_vs_released_graph_eager', check)
        del result, reference
    report.update(status='passed', stage='complete', counters=asdict(counters),
        runtime_metadata=runtime.metadata, ddp_logging_data=runtime.ddp._get_ddp_logging_data(),
        final_state_inventory=zero1_state_inventory(model, optimizer), checkpoint_created=False)
    persist()


def main(argv=None):
    args = parse_args(argv)
    if os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING', os.environ.get('NCCL_ASYNC_ERROR_HANDLING')) != '0':
        raise RuntimeError('Graph probes require TORCH_NCCL_ASYNC_ERROR_HANDLING=0 before NCCL initialization')
    configure_determinism(True); torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    runtime_info = require_container_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest'); compiler_configuration()
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])),
                            timeout=timedelta(minutes=30))
    if dist.get_world_size() != 2: raise RuntimeError('Requires two ranks')
    rank = dist.get_rank(); tracker = None; original_error = None
    local = dict(schema='olmo-two-gpu-zero1-graph-rank-v1', rank=rank, status='running', stage='setup', checks=[])
    report = dict(schema='olmo-two-gpu-zero1-graph-v1', status='running', runtime=runtime_info,
                  configuration={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()})
    try:
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            sources = [*sorted((ROOT/'cdrm/pretrained').glob('*.py')), *sorted((ROOT/'scripts').glob('olmo*.py')),
                       ROOT/'scripts/experiment_tracking.py', ROOT/'scripts/docker_shell.sh', PROTOCOL]
            report['sources'] = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
            report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            report['dependencies'] = dependency_record(args.output_dir, include_dao=False, include_fa4=False)
            if not args.tiny: report['checkpoint'] = validate_prepared_manifest(args.artifacts)['checkpoint']
            for path in sources:
                target = args.output_dir/'source-snapshot'/path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
            write_json(args.output_dir/'report.json', report)
            tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', group='olmo-two-gpu',
                name=args.output_dir.name, output_dir=args.output_dir, preserve_state=preserve_local_rng)
            tracker.start(report['configuration']); report['wandb'] = tracker.record
            write_json(args.output_dir/'report.json', report)
        dist.barrier()
        with disable_autocast_weight_cache(), backend_context('math' if args.tiny else 'flash'):
            run(args, local, tracker)
        records = gather(local)
        if rank == 0: report.update(status='passed', stage='complete', ranks=records,
            distributed_optimizer_updates=8, rank_optimizer_steps=16, checkpoint_created=False,
            resources=aggregate_resources(records))
    except BaseException as error:
        original_error = error
        details = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        local.update(status='failed', error=details); report.update(status='failed', error=details)
        if args.output_dir.exists(): write_json(args.output_dir/f'rank-{rank}-error.json', local)
        raise
    finally:
        if rank == 0 and args.output_dir.exists():
            write_json(args.output_dir/'report.json', report)
            try:
                if tracker is not None: finish_tracking(tracker, report, original_error=original_error)
            finally: write_json(args.output_dir/'report.json', report)
        if original_error is None: dist.destroy_process_group()


if __name__ == '__main__': main()
