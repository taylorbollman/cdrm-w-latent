#!/usr/bin/env python3
"""Bounded real-data memory check for the three-layer mixed-task pilot.

The four optimizer updates exercise the actual effective batch and schedule.
All resulting weights are discarded; the training run starts fresh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import torch

from cdrm.mad_data import FUZZY_TASK, load_dataset
from cdrm.rt_nextlat_task_depth import build_model, read_configuration
from cdrm.rt_nextlat_tasks import encode_inputs
from scripts import rt_nextlat_a5_fuzzy_depth_train as depth
from scripts import rt_nextlat_a5_fuzzy_lr_train as trainer
from scripts.experiment_tracking import OnlineTracker
from scripts.rt_a5_common import configure_fp32_runtime, make_optimizer
from scripts.rt_a5_data import load_split, validate_manifest
from scripts.rt_a5_train import WordOrder, atomic_json, batch_tensors, file_sha256, preserve_rng
from scripts.stage_a_common import require_cuda_container


def run(args):
    hardware = require_cuda_container()
    runtime = configure_fp32_runtime()
    config = read_configuration(args.config)
    if args.microbatch not in (1280, 2560):
        raise ValueError("The bounded pilot tests physical microbatch 2560 or 1280")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    a5 = Path('.runtime/rt-a5/20260911T154748Z/data')
    fuzzy_root = Path('.runtime/rt-nextlat-fuzzy-a5/20260916T154000Z-d128-t400/data')
    validate_manifest(a5)
    if file_sha256(a5 / 'manifest.json') != trainer.A5_MANIFEST_SHA256:
        raise ValueError("Expected the original frozen A5 corpus")
    fuzzy = load_dataset(fuzzy_root, FUZZY_TASK, 'train')
    arrays = {'a5': load_split(a5, 'train'), 'fuzzy': (fuzzy.input_ids, fuzzy.labels)}
    if [arrays[t][0].shape[1] for t in ('a5', 'fuzzy')] != [12, 400]:
        raise ValueError("Expected A5 T12 and Fuzzy T400")
    schedule = trainer.schedule_configuration()
    sources = depth.source_manifest()
    sources['scripts/rt_nextlat_a5_fuzzy_depth_profile.py'] = file_sha256(Path(__file__))
    for name, expected in sources.items():
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trainer.ROOT / name, target)
        if file_sha256(target) != expected:
            raise RuntimeError('Source changed: ' + name)
    report = {'schema': 'rt-nextlat-depth-memory-check-v1', 'status': 'running',
              'hardware': hardware, 'runtime': runtime, 'model_config': config,
              'batch_per_task': 2560, 'microbatch': args.microbatch,
              'requested_updates': 4, 'schedule': schedule, 'source_sha256': sources,
              'weights_discarded': True, 'checkpoint_written': False,
              'data_manifest_sha256': {'a5': file_sha256(a5 / 'manifest.json'),
                                       'fuzzy': file_sha256(fuzzy_root / 'manifest.json')},
              'history': [], 'qualification': 'Bounded memory and finite-state check, not learning results or an optimized throughput benchmark.'}
    tracker = OnlineTracker(project='rt-nextlat-fuzzy-a5', entity='taylorbollman',
                            output_dir=output, group=args.group,
                            name=f'l1r-rt3-memory-b2560-m{args.microbatch}', preserve_state=preserve_rng)
    report['wandb'] = tracker.record
    succeeded = False
    try:
        tracker.start({k: v for k, v in report.items() if k not in ('history', 'wandb')})
        torch.manual_seed(1234)
        model = build_model(config, seed=1234, predictor_seed=1235, fuzzy_seed=1236, device='cuda')
        optimizer = make_optimizer(model, lr=1e-4)
        orders = {task: WordOrder(len(arrays[task][0]), seed)
                  for task, seed in [('a5', 5432), ('fuzzy', 2026091604)]}
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        report['initialization'] = model.initialization
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, 5):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            batches = {}
            for task in ('a5', 'fuzzy'):
                indices = orders[task].indices((step - 1) * 2560, 2560)
                x, y = batch_tensors(*arrays[task], indices, 'cuda')
                batches[task] = encode_inputs(x, task), y
            lr = trainer.set_learning_rate(optimizer, step, schedule)
            values = trainer.train_step(model, optimizer, batches, mode='mixed', microbatch=args.microbatch)
            seconds = time.perf_counter() - begin
            row = {'update': step, 'seconds': seconds, 'learning_rate': lr, **values}
            report['history'].append(row)
            tracker.log({'update': step, 'profile/seconds': seconds,
                         'profile/loss': values['loss'], 'profile/grad_norm': values['grad_norm'],
                         'profile/learning_rate': lr})
            atomic_json(output / 'report.json', report)
        report['finite_state'] = trainer.fuzzy_base.finite_state(model, optimizer, 4)
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        report['mean_last_three_seconds'] = sum(r['seconds'] for r in report['history'][1:]) / 3
        report['status'] = 'passed'
        succeeded = True
    except torch.cuda.OutOfMemoryError:
        report.update(status='oom', peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        tracker.summary({k: report[k] for k in ('status', 'peak_allocated_gib', 'peak_reserved_gib', 'mean_last_three_seconds') if k in report})
        tracker.finish(succeeded=succeeded)
        report['wandb'] = tracker.record
        atomic_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k in ('status', 'microbatch', 'parameter_count', 'peak_allocated_gib', 'peak_reserved_gib', 'mean_last_three_seconds', 'wandb')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rt_nextlat_tasks/fuzzy_d128_l3.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--microbatch', type=int, default=2560)
    parser.add_argument('--group', required=True)
    run(parser.parse_args())
