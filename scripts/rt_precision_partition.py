#!/usr/bin/env python3
"""Validate a strict-FP32 global-batch reference against a physical-batch run.

This bounds partition/reduction-order effects at the tested shape; it is not a
CUDA-graph or physical large-batch throughput measurement.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_cuda_graph_validate import ATOL, RTOL, compare_trees, digest_tensors, validation_config
from rt_precision_compare import (OPTIMIZER_OPTIONS, adam_deltas, compare_arm_packets,
                                  compare_tensors, cpu_tree, load_ids, optimizer_named_tensors,
                                  run_arm, save_packet, validate_starting_state)
from rt_precision_credit import SCALE_LIMITS, snapshot_sources, verify_sources
from stage_a_common import configure_compiled_helpers, require_cuda_container, seed_all
from stage_b_train import atomic_json, file_digest


def partition_comparison(starting, physical, partitioned, *, trained):
    result = compare_arm_packets(starting, physical, partitioned, trained=trained)
    result['comparison_role'] = 'Strict FP32 physical global batch versus explicitly partitioned FP32 reference'
    checks = {
        'raw_gradients': compare_trees(physical['raw_gradients'], partitioned['raw_gradients']),
        'clipped_gradients': compare_trees(physical['clipped_gradients'], partitioned['clipped_gradients']),
        'adam_delta': compare_trees(adam_deltas(starting['model'], physical['model']),
                                    adam_deltas(starting['model'], partitioned['model'])),
        'adam_state': compare_trees(optimizer_named_tensors(physical), optimizer_named_tensors(partitioned)),
    }
    gradient_norms = compare_tensors(physical['raw_gradients'], partitioned['raw_gradients'], limits=SCALE_LIMITS)
    scalars = {}
    for key in ('loss', 'clip_coefficient', 'gradient_norm'):
        left, right = float(physical[key]), float(partitioned[key])
        if not math.isfinite(left) or not math.isfinite(right):
            raise FloatingPointError(f'Nonfinite partition scalar: {key}')
        error, tolerance = abs(right - left), ATOL + RTOL * abs(left)
        scalars[key] = {'reference': left, 'actual': right, 'absolute_error': error,
                        'tolerance': tolerance, 'pass': error <= tolerance}
    result['same_precision_coordinate_checks'] = checks
    result['same_precision_norm_review'] = gradient_norms
    result['same_precision_scalar_checks'] = scalars
    result['partition_validation_pass'] = (all(row['pass'] for row in checks.values())
                                            and all(row['pass'] for row in scalars.values())
                                            and not gradient_norms['requires_review'])
    result['requires_review'] = not result['partition_validation_pass']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', required=True, choices=('tiny', 'full'))
    parser.add_argument('--batch', required=True, type=int)
    parser.add_argument('--microbatch', required=True, type=int)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--ids-path', type=Path)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    if not 1 <= args.microbatch < args.batch <= 1024 or args.seed < 0:
        parser.error('Require 1 <= microbatch < batch <=1024 and nonnegative seed')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh partition-validation directory')
    args.output_dir.mkdir(parents=True)
    tracker, started = None, time.monotonic()
    report = {'schema': 'rt-precision-partition-v1', 'status': 'running',
              'tier': args.tier, 'seed': args.seed, 'batch': args.batch, 'microbatch': args.microbatch,
              'scope': 'Uncaptured strict FP32; common global B*T denominator and one clip/Adam update',
              'qualification': 'Validated only at the recorded shape; larger batch partitions can change FP32 reduction error',
              'runs': {}}
    try:
        report['hardware'] = require_cuda_container()
        torch.set_num_threads(1)
        configure_compiled_helpers(True)
        torch._dynamo.config.suppress_errors = False
        seed_all(args.seed, deterministic=True)
        report['runtime'] = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                             'cache': os.environ.get('TORCHINDUCTOR_CACHE_DIR')}
        report['source_sha256'] = snapshot_sources(args.output_dir, extra=(Path(__file__),))
        criteria = {'schema': 'rt-precision-partition-criteria-v1',
                    'same_precision_coordinate_atol': ATOL, 'same_precision_coordinate_rtol': RTOL,
                    'gradient_norm_limits': SCALE_LIMITS, 'gradient_norm_absolute_floor': 0.,
                    'requires_review': 'Any coordinate/scalar or gradient-norm screen exceeded; no automatic fallback clearance',
                    'source_sha256': report['source_sha256']}
        atomic_json(args.output_dir / 'criteria.json', criteria)
        report['criteria_sha256'] = file_digest(args.output_dir / 'criteria.json')
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({key: value for key, value in report.items() if key not in ('source_sha256', 'wandb')})
        seed_all(args.seed, deterministic=True)
        config = validation_config(args.tier, 'legacy')
        report['model_config'] = dataclasses.asdict(config)
        from olmo.model import OLMo
        model = OLMo(config).to(dtype=torch.float32).train()
        if args.checkpoint:
            starting = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            report['checkpoint'] = {'path': str(args.checkpoint), 'sha256': file_digest(args.checkpoint)}
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_OPTIONS)
            starting = {'schema': 'rt-precision-state-v1', 'model_config': dataclasses.asdict(config),
                        'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                        'optimizer_parameter_names': list(dict(model.named_parameters())), 'seed': args.seed}
            del optimizer
        report['starting_optimizer'] = validate_starting_state(starting, config, list(dict(model.named_parameters())))
        model.load_state_dict(starting['model'], strict=True)
        report['starting_model_digest'] = digest_tensors(starting['model'])
        report['starting_state'] = save_packet(args.output_dir / 'starting-state.pt', starting)
        del model
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        ids = load_ids(args.ids_path, batch=args.batch, length=config.max_sequence_length,
                       vocab=config.vocab_size, seed=args.seed)
        report['ids'] = save_packet(args.output_dir / 'ids.pt', torch.from_numpy(ids))
        report['ids_source'] = {'path': str(args.ids_path) if args.ids_path else None,
                                'sha256': file_digest(args.ids_path) if args.ids_path else None,
                                'selection': 'First batch rows of supplied array' if args.ids_path else 'Seeded random IDs'}
        for index, (name, microbatch) in enumerate((('physical', None), ('partitioned', args.microbatch))):
            destination = args.output_dir / name
            destination.mkdir()
            # Keep run_arm's ordinary packet provenance contract unchanged.
            os.link(args.output_dir / 'starting-state.pt', destination / 'starting-state.pt')
            seed_all(args.seed + 2, deterministic=True)
            print(json.dumps({'run': name, 'status': 'starting'}), flush=True)
            report['runs'][name] = run_arm('C', config, starting, ids, destination,
                                            reference_microbatch=microbatch)
            tracker.log({'update': index + 1, f'partition/{name}/loss': report['runs'][name]['loss']})
            atomic_json(args.output_dir / 'progress.json', report, replace=True)
        physical = torch.load(args.output_dir / 'physical/C-packet.pt', map_location='cpu', weights_only=False)
        partitioned = torch.load(args.output_dir / 'partitioned/C-packet.pt', map_location='cpu', weights_only=False)
        report['comparison'] = partition_comparison(starting, physical, partitioned,
                                                   trained=report['starting_optimizer']['trained_moments'])
        verify_sources(report['source_sha256'])
        report['status'] = 'complete'
        report['partition_validation_pass'] = report['comparison']['partition_validation_pass']
        tracker.summary({'partition/validation_pass': report['partition_validation_pass'],
                         'partition/gradient_relative_l2': report['comparison']['raw_gradients']['relative_l2'],
                         'partition/adam_delta_relative_l2': report['comparison']['adam_delta']['relative_l2']})
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        try:
            if tracker:
                tracker.finish(succeeded=report['status'] == 'complete')
        except BaseException as error:
            report.update(status='execution_failed', sync_error_type=type(error).__name__)
            raise
        finally:
            atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'partition_validation_pass': report['partition_validation_pass']}), flush=True)


if __name__ == '__main__':
    main()
