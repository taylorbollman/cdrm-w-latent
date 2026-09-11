#!/usr/bin/env python3
"""CPU-only preparation of fresh numerical fixtures for the bounded CDRM pilot.

Prior training/development data and seed-7500 checkpoints are referenced without
mutation. The new confirmation examples are allocated prospectively; their
reserved remainder is not a pool from which to choose a better result.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import torch

from cdrm.mad_data import baseline_audit, generate_dataset, overlap_audit, save_dataset
from cdrm_tiled_common import (
    FORMAT, INIT_FORMAT, PRESET, TASK, atomic_json, build_model, cpu_initial_model,
    dataset_identity, epoch_indices, file_digest, json_digest, load_dataset,
    optimizer_for, save_checkpoint, snapshot_sources, sources, state_digest,
    validate_data, verify_sources,
)
from cdrm_tiled_operational import OPTIMIZER, SCHEDULE, SEEDS as PRIOR_SEEDS

LINEAGE = Path('.runtime/cdrm-tiled-pilot/20260907T212606Z')
PRIOR = Path('.runtime/cdrm-tiled-bf16/20260907T191403Z')
MODEL_SEEDS = (7500, 7501)
NUMERICAL_UPDATES = (1000, 2500)
TRAJECTORIES = ('fp32', 'bf16')
CONFIRMATION_SEED = 925801
CONFIRMATION_SIZE = 768
BATCH = 64
SCHEMA = 'cdrm-tiled-pilot-preparation-v1'


def confirmation_allocation():
    slots = []
    for seed in MODEL_SEEDS:
        for update in NUMERICAL_UPDATES:
            for trajectory in TRAJECTORIES:
                offset = len(slots) * BATCH
                slots.append({'slot': f'seed{seed}-u{update:04d}-{trajectory}',
                              'model_seed': seed, 'completed_updates': update,
                              'checkpoint_trajectory': trajectory,
                              'example_offset': offset, 'example_stop': offset + BATCH,
                              'physical_batch': BATCH, 'split': 'dev'})
    return {'seed': CONFIRMATION_SEED, 'examples': CONFIRMATION_SIZE,
            'task': TASK, 'task_overrides': {'num_tokens_to_copy': 96},
            'vocab_size': 16, 'sequence_length': 256, 'slots': slots,
            'reserved': {'example_offset': len(slots) * BATCH,
                         'example_stop': CONFIRMATION_SIZE, 'examples': 256,
                         'access': 'Unused; requires a new explicit prospective purpose before model evaluation'},
            'access_policy': 'Use only the allocated B64 for each declared model seed, checkpoint and trajectory. '
                             'No fixture selection, resampling or threshold tuning from these outcomes.'}


def validate_output_paths(output, confirmation_root, *, lineage=LINEAGE, prior=PRIOR):
    base, old = Path(lineage).resolve(), Path(prior).resolve()
    paths = [Path(output).resolve(), Path(confirmation_root).resolve()]
    if base == old or base.is_relative_to(old) or old.is_relative_to(base):
        raise ValueError('Pilot and prior retained lineages must be separate')
    for path in paths:
        if not path.is_relative_to(base) or path == base:
            raise ValueError('New artifacts must be inside the dedicated pilot lineage')
        if path.exists():
            raise FileExistsError(f'Refusing to reuse a preparation/data output path: {path}')
    if paths[0] == paths[1] or paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        raise ValueError('Preparation and confirmation data require separate sibling subtrees')


def relative_source(path):
    resolved = Path(path).resolve()
    if not resolved.is_file() or not resolved.is_relative_to(Path.cwd()):
        raise ValueError(f'Source must be an existing project file: {path}')
    return resolved.relative_to(Path.cwd())


def checkpoint_record(path, *, expected_updates, manifest):
    path = Path(path)
    digest = file_digest(path)
    relative = path.relative_to(PRIOR).as_posix()
    archived = manifest['files'][relative]
    if digest != archived['sha256'] or path.stat().st_size != archived['bytes']:
        raise ValueError(f'Prior immutable checkpoint differs from its archived manifest: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    required_format = INIT_FORMAT if expected_updates == 0 else FORMAT
    if (payload.get('format') != required_format or payload.get('completed_updates') != expected_updates
            or payload['identity_sha256'] != json_digest(payload['identity'])):
        raise ValueError(f'Unexpected prior checkpoint format/update/identity: {path}')
    if payload['initialization']['seed'] != 7500:
        raise ValueError('Prior lineage must preserve initialization seed 7500')
    verify_sources(payload['identity']['source_sha256'])
    if expected_updates == 0 and payload['optimizer']['state']:
        raise ValueError('Shared initialization must have empty Adam state')
    return payload, {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
                     'format': payload['format'], 'completed_updates': expected_updates,
                     'identity_sha256': payload['identity_sha256'],
                     'source_sha256': copy.deepcopy(payload['identity']['source_sha256']),
                     'model_state_sha256': state_digest(payload['model']),
                     'optimizer_state_sha256': state_digest(payload['optimizer']),
                     'scheduler_state_sha256': state_digest(payload['scheduler']),
                     'initialization': copy.deepcopy(payload['initialization'])}


def fresh_initialization(seed, *, prior_initial, tracked_sources, freeze_path, data,
                         data_root, confirmation_root, parent_record, preset=PRESET):
    if seed != 7501:
        raise ValueError('Only prospectively declared new initialization 7501 may be created')
    model, construction = cpu_initial_model(seed, preset=preset)
    parameters = list(model.named_parameters())
    if (len(parameters) != len({id(value) for _, value in parameters})
            or any(value.device.type != 'cpu' or value.dtype != torch.float32
                   for _, value in parameters)):
        raise AssertionError('Fresh initialization needs unique CPU FP32 master parameters')
    adapter_names = ['cdrm.bridge_adapter.weight', 'cdrm.deep_adapter.weight']
    actual_extras = sorted(name for name, _ in parameters if name.startswith('cdrm.'))
    if actual_extras != adapter_names or any(torch.count_nonzero(dict(parameters)[name]) == 0 for name in adapter_names):
        raise AssertionError('Exactly two nonzero CDRM adapters are required')
    cfg = dataclasses.asdict(model.config)
    if cfg != prior_initial['model_config']:
        raise AssertionError('Fresh seed changed architecture/configuration')
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    if (state_digest(optimizer.state_dict()) != state_digest(prior_initial['optimizer'])
            or state_digest(scheduler.state_dict()) != state_digest(prior_initial['scheduler'])):
        raise AssertionError('Fresh initialization changed optimizer/scheduler semantics')
    seeds = {**PRIOR_SEEDS, 'initialization': seed, 'confirmation': CONFIRMATION_SEED}
    identity = {'schema': INIT_FORMAT, 'source_sha256': tracked_sources,
                'fixture_freeze_sha256': file_digest(freeze_path), 'preset_sha256': file_digest(preset),
                'data': data, 'data_root': str(data_root), 'confirmation_root': str(confirmation_root),
                'task': TASK, 'seeds': seeds, 'optimizer': OPTIMIZER, 'schedule': SCHEDULE,
                'model_config': cfg, 'initialization': construction,
                'pilot_preparation_schema': SCHEMA,
                'parent_initialization': {'path': parent_record['path'], 'sha256': parent_record['sha256'],
                                          'identity_sha256': parent_record['identity_sha256']},
                'parent_source_sha256': copy.deepcopy(prior_initial['identity']['source_sha256'])}
    payload = {'format': INIT_FORMAT, 'identity': identity, 'identity_sha256': json_digest(identity),
               'model_config': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'initialization': construction,
               'completed_updates': 0, 'completed_epochs': 0, 'batch_in_epoch': 0,
               'precision': 'fp32', 'weights_only_initialization': True,
               'memory_state': 'Rebuilt for each independent forward'}
    # Exercise the existing NUM checkpoint loader on CPU; no model forward runs.
    loaded, loaded_record = build_model('tiled', 'bf16', checkpoint=payload, device='cpu', preset=preset)
    if state_digest(loaded.state_dict()) != construction['full_initialization_sha256']:
        raise AssertionError('Legacy NUM checkpoint loader changed the fresh weights')
    return payload, {'legacy_num_loader_compatible': True,
                     'exact_two_nonzero_adapters': True, 'unique_cpu_fp32_parameter_owners': True,
                     'optimizer_and_scheduler_equal_to_prior_initialization': True,
                     'construction': construction, 'legacy_num_loaded_model': loaded_record}


def prepare(args, report):
    if not Path('/.dockerenv').exists() or Path.cwd() != Path('/workspace/cdrm-w-latent'):
        raise RuntimeError('Preparation requires the explicit project CPU container')
    if torch.cuda.is_available():
        raise RuntimeError('Use CDRM_DOCKER_GPUS=none; this preparation never uses a GPU')
    torch.set_num_threads(1)
    protocol = relative_source(args.protocol)
    tracked = sources([relative_source(__file__), protocol,
                       *(relative_source(path) for path in args.extra_source)])
    snapshot_sources(args.output_dir, tracked)
    report.update(source_sha256=tracked, protocol={'path': str(protocol), 'sha256': file_digest(protocol)},
                  execution={'device': 'cpu', 'torch': torch.__version__,
                             'cuda_available': False, 'model_forward_executed': False})
    manifest_path = PRIOR / 'artifact-manifest.json'
    prior_manifest = json.loads(manifest_path.read_text())
    prior_receipt = json.loads((PRIOR / 'storage.json').read_text())
    if file_digest(manifest_path) != prior_receipt['manifest_sha256']:
        raise ValueError('Prior manifest no longer matches its verified archival receipt')
    initial, initial_record = checkpoint_record(PRIOR / 'prepare/init.pt', expected_updates=0, manifest=prior_manifest)
    prior_checkpoints = {'init7500': initial_record}
    for precision in TRAJECTORIES:
        payload, record = checkpoint_record(PRIOR / f'train-{precision}/u0100.pt',
                                            expected_updates=100, manifest=prior_manifest)
        if (payload['identity']['initial_checkpoint_sha256'] != initial_record['sha256']
                or payload['identity']['shuffle_seed'] != PRIOR_SEEDS['shuffle']
                or payload['identity']['optimizer'] != OPTIMIZER or payload['identity']['schedule'] != SCHEDULE):
            raise ValueError('Prior continuation does not share the exact initialization/data-order/optimizer policy')
        prior_checkpoints[f'{precision}_u0100'] = record
    old_data_root = PRIOR / 'data'
    old = {name: load_dataset(old_data_root / 'confirmation' if name == 'old_confirmation' else old_data_root,
                              TASK, 'train' if name == 'train' else 'dev')
           for name in ('train', 'dev', 'old_confirmation')}
    for name, dataset in old.items():
        validate_data(dataset)
        original_key = 'confirmation' if name == 'old_confirmation' else name
        if dataset_identity(dataset) != initial['identity']['data'][original_key]:
            raise ValueError(f'Prior {name} data differs from the original initialization identity')
    if [len(old[name]) for name in ('train', 'dev', 'old_confirmation')] != [12800, 256, 128]:
        raise ValueError('Prior data sizes changed')
    allocation = confirmation_allocation()
    freeze = {'schema': 'cdrm-tiled-pilot-fixture-freeze-v1', 'protocol': report['protocol'],
              'source_sha256': tracked, 'prior_manifest_sha256': file_digest(manifest_path),
              'prior_checkpoints': prior_checkpoints, 'initialization_seeds': list(MODEL_SEEDS),
              'confirmation': {**allocation, 'root': str(args.confirmation_root)},
              'reused_data_root': str(old_data_root),
              'reused_data': {name: dataset_identity(data) for name, data in old.items()},
              'shuffle_seed': PRIOR_SEEDS['shuffle'], 'data_order': 'unchanged shared_epoch_permutation',
              'optimizer': OPTIMIZER, 'schedule': SCHEDULE,
              'purpose': 'Prospective numerical monitoring fixtures; not a model-selection or final task-test set'}
    freeze_path = args.output_dir / 'fixture-freeze.json'
    atomic_json(freeze_path, freeze)
    confirmation = generate_dataset(TASK, 'dev', CONFIRMATION_SEED, CONFIRMATION_SIZE,
                                    {'num_tokens_to_copy': 96})
    validate_data(confirmation)
    saved_data = save_dataset(args.confirmation_root, confirmation)
    confirmation = load_dataset(args.confirmation_root, TASK, 'dev')
    overlap = overlap_audit({**old, 'pilot_confirmation': confirmation})
    report.update(prior_checkpoints=prior_checkpoints, prior_manifest_sha256=file_digest(manifest_path),
                  confirmation_dataset=saved_data, overlap_audit=overlap)
    if (not overlap['no_exact_input_cross_split_overlap']
            or overlap['within_split']['pilot_confirmation']['input']['duplicate_rows']):
        raise AssertionError('Exact input collision detected; retain native draws and investigate without resampling')
    slots = [{**slot, 'array_sha256': confirmation.take(slice(slot['example_offset'], slot['example_stop'])).sha256}
             for slot in allocation['slots']]
    data = {'train': dataset_identity(old['train']), 'dev': dataset_identity(old['dev']),
            'confirmation': dataset_identity(confirmation)}
    payload, fresh_checks = fresh_initialization(7501, prior_initial=initial, tracked_sources=tracked,
        freeze_path=freeze_path, data=data, data_root=old_data_root,
        confirmation_root=args.confirmation_root, parent_record=initial_record, preset=args.preset)
    checkpoint = save_checkpoint(args.output_dir / 'init-7501.pt', payload)
    permutation = epoch_indices(len(old['train']), 0, PRIOR_SEEDS['shuffle'])
    baseline = baseline_audit(old['dev'])
    report.update(fixture_freeze_sha256=file_digest(freeze_path),
                  confirmation={**allocation, 'root': str(args.confirmation_root), 'slots': slots,
                                'dataset_identity': dataset_identity(confirmation)},
                  data_roots={'train_dev': str(old_data_root), 'numerical_confirmation': str(args.confirmation_root)},
                  reused_data={name: dataset_identity(old[name]) for name in ('train', 'dev')},
                  initialization_checkpoints={'7500': initial_record, '7501': checkpoint},
                  fresh_initialization_checks=fresh_checks,
                  training_order={'shuffle_seed': PRIOR_SEEDS['shuffle'], 'updates_per_epoch': 200,
                      'epoch0_permutation_sha256': hashlib.sha256(np.asarray(permutation, dtype='<i8').tobytes()).hexdigest(),
                      'initial_batch_sha256': old['train'].take(permutation[:BATCH]).sha256},
                  development_baselines={'dataset_identity': dataset_identity(old['dev']),
                      'fixed_before_pilot_training': True, 'model_independent': True,
                      'order_ignoring_modal': baseline['query_ignoring_modal'], 'full_audit': baseline},
                  old_sources_unchanged=True, status='complete')
    atomic_json(args.output_dir / 'confirmation-slots.json', report['confirmation'])
    atomic_json(args.output_dir / 'development-baselines.json', report['development_baselines'])
    verify_sources(tracked)
    for record in prior_checkpoints.values():
        if file_digest(Path(record['path'])) != record['sha256']:
            raise AssertionError('A prior checkpoint changed during preparation')
        verify_sources(record['source_sha256'])
    for name, dataset in old.items():
        root = old_data_root / 'confirmation' if name == 'old_confirmation' else old_data_root
        again = load_dataset(root, TASK, 'train' if name == 'train' else 'dev')
        if dataset_identity(again) != dataset_identity(dataset):
            raise AssertionError('A prior dataset changed during preparation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=LINEAGE / 'prepare')
    parser.add_argument('--confirmation-root', type=Path, default=LINEAGE / 'data/confirmation')
    parser.add_argument('--preset', type=Path, default=PRESET)
    parser.add_argument('--extra-source', type=Path, action='append', default=[])
    args = parser.parse_args()
    validate_output_paths(args.output_dir, args.confirmation_root)
    args.output_dir.mkdir(parents=True)
    report = {'schema': SCHEMA, 'status': 'preparation_started',
              'command': shlex.join([sys.executable, *sys.argv]),
              'arguments': {key: str(value) if isinstance(value, Path) else
                            [str(item) for item in value] if isinstance(value, list) else value
                            for key, value in vars(args).items()}}
    try:
        prepare(args, report)
    except BaseException as exc:
        report.update(status='execution_failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(args.output_dir / 'report.json'),
                      'confirmation_array_sha256': report['confirmation']['dataset_identity']['array_sha256'],
                      'slots': len(report['confirmation']['slots']),
                      'modal_development_token_accuracy': report['development_baselines']['order_ignoring_modal']['answer_accuracy']}), flush=True)


if __name__ == '__main__':
    main()
