#!/usr/bin/env python3
"""Bounded eager NCCL ZeRO-1 fixed-gradient Adam parity and exact recovery.

The full replicated Adam reference consumes the very same reduced/clipped
gradient tensors; it does not run an independent BF16 forward trajectory.
This isolates state sharding and parameter synchronization from DDP numerics.
Reference copies and graph-free checkpointing make this a correctness probe,
not a training-memory/capacity measurement. No local checkpoint is deleted.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import timedelta
import gc
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from cdrm.pretrained.artifacts import write_json, sha256_file
from cdrm.pretrained.ddp_training import EagerDDPTrainer
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, build_warmup_scheduler, optimizer_state_bytes
from cdrm.pretrained.zero1_training import (build_zero1_adamw, zero1_state_inventory,
    save_zero1_checkpoint, load_zero1_checkpoint)
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import IntegrationCase
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import construct, move_batch, gather, assert_all, preserve_local_rng
from scripts.olmo_two_gpu_recovery import recovery_batch, execution_configuration, seed_local
from scripts.olmo_rt_large_batch import (optimizer_for, compiler_configuration, configure_determinism,
    require_container_gpu, backend_context, load_native_tokenizer, validate_prepared_manifest,
    OnlineTracker, MemoryPhases, dependency_record, finish_tracking)

PROTOCOL = ROOT/'docs/reports/olmo-two-gpu/protocol.md'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--case', choices=('ordinary', 'rt', 'combined'), required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--length', type=int, default=512)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'.runtime/olmo1b-step60000/artifacts')
    args = parser.parse_args(argv)
    args.output_dir, args.checkpoint_dir = args.output_dir.resolve(), args.checkpoint_dir.resolve()
    if not all(path.is_relative_to(ROOT.resolve()) for path in (args.output_dir, args.checkpoint_dir)):
        parser.error('Evidence and checkpoints must remain under the persistent project checkout')
    if not 1 <= args.batch_size <= 2 or not 8 <= args.length <= 512:
        parser.error('Correctness only: physical B1..2 and T8..512')
    if args.checkpoint_dir == args.output_dir:
        parser.error('Checkpoint must have its own initially absent directory')
    return args


def make_optimizer(model):
    optimizer = build_zero1_adamw(model, lr=1e-5, betas=(.9, .95), eps=1e-8,
                                  weight_decay=.1, fused=True)
    return optimizer, build_warmup_scheduler(optimizer, warmup_updates=2)


def checkpoint_configuration(model, case, config, args):
    configuration = execution_configuration(model, case, config, args)
    configuration['gradient_reference'] = 'same DDP-reduced gradient; one clip; full replicated fused AdamW'
    configuration['optimizer_sharding'] = 'native-zero1-AdamW; no overlap; no parameter views'
    return configuration


def checkpoint_fingerprint(report):
    """Pin the loaded native weights and the frozen runtime/protocol snapshot."""
    return {'checkpoint_sha256': report['source_checkpoint']['sha256'],
            'source_hashes': dict(report['sources']),
            'protocol_sha256': report['sources'][str(PROTOCOL.relative_to(ROOT))]}


def local_adam_steps(model, optimizer):
    return {name: int(optimizer.optim.state[parameter]['step'])
            for name, parameter in model.named_parameters() if parameter in optimizer.optim.state}


def share_raw_gradients(model, reference):
    """Read-only sharing avoids another full GPU gradient copy in this probe.

    The target clips once, applies ZeRO and clears Parameter.grad references.
    The independent full Adam reference retains those same clipped tensors until
    its immediately following step; no next backward occurs in between.
    """
    expected = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected[name].grad = parameter.grad


def fixed_gradient_check(model, optimizer, reference, reference_optimizer):
    target = dict(reference.named_parameters())
    parameter_errors, moment_errors = [], []
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter, target[name]): parameter_errors.append(name)
    local = {id(parameter) for group in optimizer.optim.param_groups for parameter in group['params']}
    for name, parameter in model.named_parameters():
        if id(parameter) not in local: continue
        actual = optimizer.optim.state.get(parameter, {})
        expected = reference_optimizer.state.get(target[name], {})
        if tree_digests(actual) != tree_digests(expected): moment_errors.append(name)
    groups_exact = len(optimizer.param_groups) == len(reference_optimizer.param_groups)
    for actual, expected in zip(optimizer.param_groups, reference_optimizer.param_groups):
        groups_exact &= actual.keys() == expected.keys()
        groups_exact &= {k:v for k,v in actual.items() if k != 'params'} == {k:v for k,v in expected.items() if k != 'params'}
    return dict(passed=not parameter_errors and not moment_errors and groups_exact,
                parameter_errors=parameter_errors, local_moment_errors=moment_errors,
                global_group_metadata_exact=groups_exact, criterion='bitwise identical fixed-gradient update')


def shard_check(model, optimizer, reference_optimizer):
    inventory = zero1_state_inventory(model, optimizer)
    rows = gather(inventory)
    names = [name for row in rows for name in row['local_owned_names']]
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    total = sum(sum(row['local_state_bytes_by_device'].values()) for row in rows)
    full = sum(optimizer_state_bytes(reference_optimizer).values())
    valid = len(names) == len(set(names)) and set(names) == expected and total == full
    valid &= all(not row['outer_state_bytes_by_device'] and not row['consolidation_cache_present'] for row in rows)
    return dict(passed=valid, rank_inventories=rows, global_sharded_state_bytes=total,
                full_replicated_state_bytes_per_rank=full,
                measured_state_saving_bytes_per_rank=[full-sum(row['local_state_bytes_by_device'].values()) for row in rows])


def local_boundary(model, optimizer, scheduler, counters):
    return tree_digests(dict(model=model.state_dict(), local_optimizer=optimizer.optim.state_dict(),
                            scheduler=scheduler.state_dict(), counters=asdict(counters)))


def rng_draws():
    return tree_digests(dict(python=random.random(), numpy=np.random.rand(4).tolist(),
                            cpu=torch.rand(4), cuda=torch.rand(4, device='cuda')))


def replica_check(model, scheduler, counters):
    records = gather(tree_digests(dict(model=model.state_dict(), scheduler=scheduler.state_dict(),
                                      counters=asdict(counters))))
    return all(record == records[0] for record in records)


def run(args, report, tracker):
    rank = dist.get_rank(); device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    case = IntegrationCase(args.case, fbt=args.case=='combined', nextlat=args.case=='combined',
        rt_layers=() if args.case=='ordinary' else (0, 1 if args.tiny else 15),
        batch_size=args.batch_size, length=args.length, updates=3)
    config = LMTrainingConfig(precision='fp32' if args.tiny else 'bf16_mixed')
    source = checkpoint_fingerprint(report)
    report.update(zero1_physical_updates=0, reference_adam_updates=0)
    def persist(): write_json(args.output_dir/f'rank-{rank}-progress.json', report)
    phases = MemoryPhases(report, persist)
    def check(name, data):
        report['checks'].append(dict(name=name, **data)); persist()
        assert_all(data['passed'], name)
    with phases.phase('construct_with_full_adam_reference'):
        seed_local(20260925, device)
        model = construct(case, args, device)
        configuration = checkpoint_configuration(model, case, config, args)
        report['configuration'] = configuration
        reference = copy.deepcopy(model)
        trainer = EagerDDPTrainer(model)
        reference.load_state_dict(model.state_dict(), strict=True, assign=False)
        optimizer, scheduler = make_optimizer(model)
        reference_optimizer, reference_scheduler = optimizer_for(reference, 'compiled-native')
        counters = TrainingCounters()
        tokenizer = None if args.tiny else load_native_tokenizer(args.artifacts)
        addresses = [parameter.data_ptr() for parameter in model.parameters()]
        seed_local(6300+rank, device)
    def batches(update):
        return [move_batch(recovery_batch(case, tokenizer, update, rank, micro, tiny=args.tiny), device)
                for micro in range(2)]
    for update in range(2):
        with phases.phase(f'fixed_gradient_adam_update_{update}'):
            result = trainer.backward(batches(update), config=config,
                                       backbone_kwargs={'mode': case.mode(), 'full_valid_causal': True})
            raw = gather(tree_digests({name: p.grad for name,p in model.named_parameters() if p.grad is not None}))
            check(f'raw_gradient_replicas_{update}', {'passed': raw[0] == raw[1]})
            share_raw_gradients(model, reference)
            metrics = trainer.step(result, optimizer, scheduler=scheduler, counters=counters)
            report['zero1_physical_updates'] += 1; persist()
            reference_optimizer.step(); reference_scheduler.step()
            report['reference_adam_updates'] += 1; persist()
            reference_optimizer.zero_grad(set_to_none=True)
            parity = fixed_gradient_check(model, optimizer, reference, reference_optimizer)
            parity['scheduler_exact'] = scheduler.state_dict() == reference_scheduler.state_dict()
            parity['parameter_storage_stable'] = addresses == [p.data_ptr() for p in model.parameters()]
            parity['passed'] &= parity['scheduler_exact'] and parity['parameter_storage_stable']
            check(f'full_adam_fixed_gradient_update_{update}', parity)
            check(f'local_state_partition_{update}', shard_check(model, optimizer, reference_optimizer))
            check(f'parameter_replicas_{update}', {'passed': replica_check(model, scheduler, counters)})
            if rank == 0: tracker.log({'update': counters.optimizer_updates, 'train/objective': metrics['objective'],
                                      'zero1/fixed_gradient_adam_exact': int(parity['passed'])})
            del result, raw
    with phases.phase('release_reference_before_checkpoint'):
        del reference_optimizer, reference_scheduler, reference
        gc.collect(); torch.cuda.empty_cache()
    with phases.phase('consolidated_checkpoint_save'):
        trainer.assert_update_boundary()
        receipt = save_zero1_checkpoint(args.checkpoint_dir, model, optimizer, scheduler=scheduler,
            counters=counters, data_cursor={'rank': rank, 'next_update': 2}, configuration=configuration,
            source_fingerprint=source, device=device)
        report['checkpoint'] = receipt
        check('consolidation_cache_released', {'passed': not optimizer._all_state_dicts and not optimizer.state})
    with phases.phase('reference_continuation'):
        steps_before = local_adam_steps(model, optimizer)
        expected_metrics = trainer.optimizer_step(optimizer, batches(2), config=config,
            backbone_kwargs={'mode': case.mode(), 'full_valid_causal': True}, scheduler=scheduler, counters=counters)
        report['zero1_physical_updates'] += 1; persist()
        expected = local_boundary(model, optimizer, scheduler, counters)
        expected_draws = rng_draws()
        steps_after = local_adam_steps(model, optimizer)
        check('reference_continuation_updates_every_active_shard', {
            'passed': bool(steps_before) and steps_before.keys() == steps_after.keys() and
                      all(steps_after[name] == value+1 for name,value in steps_before.items()),
            'steps_before': steps_before, 'steps_after': steps_after})
    with phases.phase('release_and_reconstruct'):
        del trainer, optimizer, scheduler, model
        gc.collect(); torch.cuda.empty_cache()
        model = construct(case, args, device)
        trainer = EagerDDPTrainer(model)
        optimizer, scheduler = make_optimizer(model)
        restored = load_zero1_checkpoint(args.checkpoint_dir, model, optimizer, scheduler=scheduler,
            configuration=checkpoint_configuration(model, case, config, args), source_fingerprint=source, device=device,
            expected_manifest_sha256=receipt['manifest_sha256'])
        counters = restored['counters']
        check('resume_cursor_and_local_state', {'passed': restored['data_cursor'] == {'rank': rank, 'next_update': 2}
              and not optimizer.state and not optimizer._all_state_dicts,
              'inventory': zero1_state_inventory(model, optimizer)})
    with phases.phase('restored_continuation'):
        steps_before = local_adam_steps(model, optimizer)
        actual_metrics = trainer.optimizer_step(optimizer, batches(2), config=config,
            backbone_kwargs={'mode': case.mode(), 'full_valid_causal': True}, scheduler=scheduler, counters=counters)
        report['zero1_physical_updates'] += 1; persist()
        boundary_exact = local_boundary(model, optimizer, scheduler, counters) == expected
        draws_exact = rng_draws() == expected_draws
        steps_after = local_adam_steps(model, optimizer)
        check('restored_continuation_updates_every_active_shard', {
            'passed': bool(steps_before) and steps_before.keys() == steps_after.keys() and
                      all(steps_after[name] == value+1 for name,value in steps_before.items()),
            'steps_before': steps_before, 'steps_after': steps_after})
        replicas = replica_check(model, scheduler, counters)
        check('exact_consolidated_recovery', {'passed': boundary_exact and draws_exact and replicas and actual_metrics == expected_metrics,
            'local_state_exact': boundary_exact, 'rng_draws_exact': draws_exact, 'replicas_exact': replicas,
            'metrics_exact': actual_metrics == expected_metrics})
    report.update(status='passed', stage='complete', counters=asdict(counters),
        final_state_inventory=zero1_state_inventory(model, optimizer),
        checkpoint_retained_locally=True, memory_scope='correctness with replicated-Adam reference; not capacity')
    persist()


def main(argv=None):
    args = parse_args(argv)
    configure_determinism(True); torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    runtime = require_container_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest'); compiler_configuration()
    dist.init_process_group('nccl', device_id=torch.device('cuda', int(os.environ['LOCAL_RANK'])),
                            timeout=timedelta(minutes=30))
    if dist.get_world_size() != 2: raise RuntimeError('Requires two ranks')
    rank = dist.get_rank(); tracker = None; original_error = None
    local = dict(schema='olmo-two-gpu-zero1-rank-v1', rank=rank, status='running', stage='setup', checks=[])
    report = dict(schema='olmo-two-gpu-zero1-v1', status='running', runtime=runtime,
                  configuration={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()})
    try:
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            sources = [*sorted((ROOT/'cdrm/pretrained').glob('*.py')), *sorted((ROOT/'scripts').glob('olmo*.py')),
                       ROOT/'scripts/experiment_tracking.py', ROOT/'scripts/docker_shell.sh', PROTOCOL]
            report['sources'] = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
            report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            report['dependencies'] = dependency_record(args.output_dir, include_dao=False, include_fa4=False)
            report['source_checkpoint'] = ({'sha256': '0'*64, 'kind': 'deterministic_tiny_fixture'} if args.tiny
                                           else validate_prepared_manifest(args.artifacts)['checkpoint'])
            for path in sources:
                target = args.output_dir/'source-snapshot'/path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
            write_json(args.output_dir/'report.json', report)
            tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', group='olmo-two-gpu',
                name=args.output_dir.name, output_dir=args.output_dir, preserve_state=preserve_local_rng)
            tracker.start(report['configuration']); report['wandb'] = tracker.record
            write_json(args.output_dir/'report.json', report)
        dist.barrier()
        shared = gather({key: report[key] for key in ('sources', 'source_checkpoint')} if rank == 0 else None)[0]
        local.update(shared)
        with disable_autocast_weight_cache(), backend_context('math' if args.tiny else 'flash'):
            run(args, local, tracker)
        reports = gather(local)
        if rank == 0: report.update(status='passed', ranks=reports,
            zero1_distributed_updates=4, logical_endpoint_update=3, reference_adam_updates_per_rank=2,
            rank_optimizer_step_calls=12)
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
