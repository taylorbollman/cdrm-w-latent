#!/usr/bin/env python3
"""Enforce one prospective pilot fixture around the unchanged CDRM NUM validator.

The wrapper validates role, checkpoint ancestry and fixture identities, then
retains a launch record and an independent completion audit. Numerical screen
failures remain numerical results; a completed process is never called a pass.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
import shlex
import subprocess
import sys

import torch

import cdrm_tiled_common as common
from experiment_tracking import add_wandb_arguments
from stage_a_common import require_cuda_container

ROOT = Path('.runtime/cdrm-tiled-pilot/20260907T212606Z')
VALIDATOR = Path('scripts/cdrm_tiled_validate.py')
WRAPPER = Path('scripts/cdrm_tiled_pilot_numerical.py')
CONTRACT_SHA256 = '26b1756dd958e0ab1c916cc51e46b691393eabb5598c645cb0d63b5bfdd0ea20'
PROTOCOL_SHA256 = '20e5b22b113d8feb6e2614469c1b48af84371ca62b9105edacdd3e6b664efaf7'
POLICIES = {'fp32': 'fp32', 'bf16': 'bf16_fp32_state'}


def file_record(path):
    return {'path': str(path), 'sha256': common.file_digest(path), 'bytes': Path(path).stat().st_size}


def select_slot(preparation, seed, updates, trajectory):
    if seed not in (7500, 7501) or updates not in (1000, 2500) or trajectory not in POLICIES:
        raise ValueError('Undeclared numerical model-seed/update/trajectory role')
    confirmation = preparation['confirmation']
    if (preparation.get('status') != 'complete' or confirmation['seed'] != 925801
            or confirmation['examples'] != 768 or len(confirmation['slots']) != 8):
        raise ValueError('Require the completed frozen pilot confirmation preparation')
    selected = [slot for slot in confirmation['slots']
                if (slot['model_seed'], slot['completed_updates'], slot['checkpoint_trajectory'])
                == (seed, updates, trajectory)]
    offset = ((seed - 7500) * 4 + (0 if updates == 1000 else 2) + (trajectory == 'bf16')) * 64
    if len(selected) != 1:
        raise ValueError('Numerical role must map to exactly one frozen slot')
    slot = selected[0]
    if (slot['example_offset'] != offset or slot['example_stop'] != offset + 64
            or slot['physical_batch'] != 64 or slot['split'] != 'dev'
            or slot['slot'] != f'seed{seed}-u{updates:04d}-{trajectory}'):
        raise ValueError('Prepared slot differs from the prospective physical B64 allocation')
    return copy.deepcopy(slot)


def validate_checkpoint_role(payload, initial, initial_record, preparation, slot):
    identity = payload['identity']
    seed, updates, trajectory = (slot[key] for key in ('model_seed', 'completed_updates', 'checkpoint_trajectory'))
    policy = POLICIES[trajectory]
    if (payload.get('format') != common.FORMAT or payload.get('pilot_schema') != 'cdrm-tiled-pilot-v1'
            or payload.get('identity_sha256') != common.json_digest(identity)):
        raise ValueError('Child checkpoint format or verified identity hash differs')
    if (payload.get('completed_updates') != updates or payload['initialization']['seed'] != seed
            or identity['initialization'] != payload['initialization']
            or payload['initialization'] != initial['initialization']):
        raise ValueError('Checkpoint update/model seed does not match its allocated role')
    if (payload['precision'] != policy or identity['precision'] != policy
            or identity['arm'] != f'cdrm-{trajectory}'):
        raise ValueError('Checkpoint training precision/arm differs from the allocated trajectory')
    expected = dataclasses.asdict(common.config('tiled', policy))
    if payload['model_config'] != expected or identity['model_config'] != expected:
        raise ValueError('Require the unchanged active five-block tiled CDRM configuration')
    if (initial.get('format') != common.INIT_FORMAT
            or initial['identity_sha256'] != common.json_digest(initial['identity'])
            or identity['initial_checkpoint_sha256'] != initial_record['sha256']
            or identity['initial_identity_sha256'] != initial['identity_sha256']
            or payload['initial_checkpoint']['sha256'] != initial_record['sha256']
            or payload['ancestry']['initial']['sha256'] != initial_record['sha256']):
        raise ValueError('Child checkpoint is not linked to the allocated original initialization')
    if (identity['data'] != preparation['reused_data'] or identity['protocol_sha256'] != PROTOCOL_SHA256
            or identity['physical_batch'] != 64 or identity['shuffle_seed'] != 925704
            or identity['accumulation'] is not False):
        raise ValueError('Checkpoint training data/protocol/batch identity differs')
    common.verify_sources(identity['source_sha256'])
    common.verify_sources(initial['identity']['source_sha256'])


def validate_checkpoint_tensors(payload):
    # CPU construction validates canonical names/shapes, without a model forward.
    net, construction = common.build_model('tiled', payload['precision'], checkpoint=payload, device='cpu')
    names = list(dict(net.named_parameters()))
    if len(names) != 45 or list(payload['model']) != list(net.state_dict()):
        raise ValueError('NUM requires all 45 canonical CDRM parameter tensors in their original order')
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
           for value in payload['model'].values()):
        raise ValueError('Checkpoint master weights must all be finite FP32')
    optimizer = common.optimizer_for(net)
    optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
    if len(optimizer.state) != 45:
        raise ValueError('Trained checkpoint must retain all 45 Adam parameter states')
    for parameter, state in optimizer.state.items():
        if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                or state['exp_avg'].shape != parameter.shape or state['exp_avg_sq'].shape != parameter.shape
                or float(state['step']) != payload['completed_updates']
                or any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                       for value in state.values())):
            raise ValueError('Adam moments/steps must match the trained checkpoint and remain finite FP32')
    return {'canonical_parameter_tensors': len(names), 'master_weight_dtype': 'torch.float32',
            'adam_state_tensors_per_parameter': 3,
            'weights_sha256': common.state_digest(payload['model']),
            'optimizer_sha256': common.state_digest(payload['optimizer']),
            'construction': construction}


def verify_fixture(preparation, slot):
    root = Path(preparation['confirmation']['root'])
    dataset = common.load_dataset(root, common.TASK, 'dev')
    common.validate_data(dataset)
    expected = preparation['confirmation']['dataset_identity']
    if (common.dataset_identity(dataset) != expected or len(dataset) != 768
            or dataset.manifest['seed'] != 925801):
        raise ValueError('Frozen confirmation dataset identity changed')
    batch = dataset.take(slice(slot['example_offset'], slot['example_stop']))
    if len(batch) != 64 or batch.sha256 != slot['array_sha256']:
        raise ValueError('Allocated physical minibatch identity changed')
    return {'root': str(root), 'dataset_sha256': dataset.sha256, 'batch_sha256': batch.sha256,
            'manifest_sha256': dataset.manifest['manifest_sha256'], 'shape': [64, 256],
            'native_targets': 6144, 'example_offset': slot['example_offset'], 'split': 'dev'}


def verify_completed_report(report, launch):
    if report.get('status') != 'diagnostics_complete':
        raise ValueError('Underlying NUM execution did not complete')
    if (set(report['construction']) != {'naive_fp32', 'tiled_fp32', 'tiled_bf16'}
            or set(report['comparisons']) != {'tiled_fp32_vs_naive_fp32', 'tiled_bf16_vs_tiled_fp32'}):
        raise ValueError('Completed NUM must retain all three arms and both reference comparisons')
    fixture = report['fixture']
    expected = launch['fixture']
    if (report['checkpoint']['sha256'] != launch['checkpoint']['sha256']
            or report['checkpoint']['completed_updates'] != launch['slot']['completed_updates']
            or report['criteria']['sha256'] != CONTRACT_SHA256
            or report['validator_sha256'] != launch['validator']['sha256']
            or fixture['sha256'] != expected['batch_sha256']
            or fixture['dataset_sha256'] != expected['dataset_sha256']
            or fixture['manifest']['manifest_sha256'] != expected['manifest_sha256']
            or any(fixture[key] != expected[key] for key in ('shape', 'native_targets', 'example_offset', 'split'))
            or fixture['accumulation'] is not False):
        raise ValueError('Completed NUM report differs from its prospective checkpoint/fixture/criteria launch')
    for construction in report['construction'].values():
        if (construction['seed'] != launch['slot']['model_seed']
                or construction['starting_weights_sha256'] != launch['checkpoint_state']['weights_sha256']):
            raise ValueError('NUM arms did not load the allocated same-state weights')
    if report.get('numerical_clearance') is not False:
        raise ValueError('Underlying NUM unexpectedly claims numerical clearance')
    return {'identity_and_fixture_verified': True, 'machine_screens_pass': report['machine_screens_pass'],
            'numerical_clearance': False,
            'disposition': 'Valid same-state numerical observation; retain all machine failures for separate engineering review.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-report', type=Path, required=True)
    parser.add_argument('--model-seed', type=int, choices=[7500, 7501], required=True)
    parser.add_argument('--updates', type=int, choices=[1000, 2500], required=True)
    parser.add_argument('--trajectory', choices=list(POLICIES), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scale-check', action='store_true')
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='cdrm-tiled-learning-pilot', wandb_entity='taylorbollman',
                        wandb_group='20260907T212606Z')
    args = parser.parse_args()
    if args.wandb_project != 'cdrm-tiled-learning-pilot' or args.wandb_entity != 'taylorbollman':
        parser.error('Use the declared pilot W&B project/entity')
    if (not args.output_dir.resolve().is_relative_to(ROOT.resolve())
            or args.output_dir.resolve() == ROOT.resolve()
            or not args.checkpoint.resolve().is_relative_to(ROOT.resolve())
            or args.preparation_report.resolve() != (ROOT / 'prepare/report.json').resolve()):
        raise ValueError('Pilot checkpoint, preparation and new output must use the dedicated child lineage')
    launch_path = args.output_dir.with_name(args.output_dir.name + '.slot-launch.json')
    audit_path = args.output_dir.with_name(args.output_dir.name + '.slot-audit.json')
    source_path = args.output_dir.with_name(args.output_dir.name + '.slot-wrapper.py')
    if any(path.exists() for path in (args.output_dir, launch_path, audit_path, source_path)):
        raise FileExistsError('Use a new numerical case and unused sibling audit paths')
    if not Path('/.dockerenv').exists() or Path.cwd() != Path('/workspace/cdrm-w-latent'):
        raise RuntimeError('Launch only inside the verified project GPU container')
    audit = {'schema': 'cdrm-tiled-pilot-slot-audit-v1', 'status': 'preflight_started',
             'numerical_clearance': False, 'command': shlex.join([sys.executable, *sys.argv])}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        preparation = json.loads(args.preparation_report.read_text())
        common.verify_sources(preparation['source_sha256'])
        freeze = json.loads((ROOT / 'protocol-freeze.json').read_text())
        criteria = ROOT / 'validation-contract.md'
        if (freeze['protocol_sha256'] != PROTOCOL_SHA256 or preparation['protocol']['sha256'] != PROTOCOL_SHA256
                or common.file_digest(ROOT / 'protocol.md') != PROTOCOL_SHA256
                or freeze['numerical_contract_sha256'] != CONTRACT_SHA256
                or common.file_digest(criteria) != CONTRACT_SHA256):
            raise ValueError('Frozen protocol or unchanged numerical criteria identity differs')
        slot = select_slot(preparation, args.model_seed, args.updates, args.trajectory)
        checkpoint_ref = file_record(args.checkpoint)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        initial_record = preparation['initialization_checkpoints'][str(args.model_seed)]
        if common.file_digest(Path(initial_record['path'])) != initial_record['sha256']:
            raise ValueError('Prepared initialization checkpoint identity changed')
        initial = torch.load(initial_record['path'], map_location='cpu', weights_only=False)
        validate_checkpoint_role(payload, initial, initial_record, preparation, slot)
        state = validate_checkpoint_tensors(payload)
        fixture = verify_fixture(preparation, slot)
        tracked = {**payload['identity']['source_sha256'], str(WRAPPER): common.file_digest(WRAPPER),
                   str(VALIDATOR): common.file_digest(VALIDATOR),
                   'scripts/r3_backward_validate.py': common.file_digest(Path('scripts/r3_backward_validate.py'))}
        command = [sys.executable, str(VALIDATOR), '--checkpoint', str(args.checkpoint),
                   '--data-root', fixture['root'], '--split', 'dev', '--batch', '64',
                   '--example-offset', str(slot['example_offset']), '--criteria', str(criteria),
                   '--output-dir', str(args.output_dir), '--wandb-project', args.wandb_project,
                   '--wandb-entity', args.wandb_entity, '--wandb-group', args.wandb_group,
                   '--wandb-run-name', args.wandb_run_name or slot['slot']]
        if args.scale_check:
            command.append('--scale-check')
        launch = {'schema': 'cdrm-tiled-pilot-slot-launch-v1', 'preparation': file_record(args.preparation_report),
                  'protocol': file_record(ROOT / 'protocol.md'), 'criteria': file_record(criteria),
                  'slot': slot, 'fixture': fixture, 'checkpoint': checkpoint_ref,
                  'checkpoint_identity_sha256': payload['identity_sha256'], 'checkpoint_state': state,
                  'initialization': initial_record, 'source_sha256': tracked,
                  'validator': file_record(VALIDATOR), 'wrapper': file_record(WRAPPER),
                  'command': command, 'command_shell_display': shlex.join(command),
                  'same_state_policy': 'Each arm reloads this exact checkpoint weights and Adam moments; the trajectory is its training source.',
                  'criteria_scope': 'Original numerical thresholds; the pilot protocol supersedes only fixture allocation and operational duration.'}
        source_path.write_bytes(WRAPPER.read_bytes())
        common.atomic_json(launch_path, launch)
        audit.update(slot=slot, launch=file_record(launch_path), preflight_verified=True)
        audit['hardware'] = require_cuda_container()
        print(json.dumps({'slot': slot['slot'], 'checkpoint_sha256': checkpoint_ref['sha256'],
                          'fixture_sha256': fixture['batch_sha256'], 'launch': str(launch_path)}), flush=True)
        del payload, initial
        result = subprocess.run(command, check=False)
        audit['validator_returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(f'Underlying numerical validator exited with status {result.returncode}')
        num_report_path = args.output_dir / 'report.json'
        num_report = json.loads(num_report_path.read_text())
        audit.update(verify_completed_report(num_report, launch))
        for name, record in (('checkpoint', checkpoint_ref), ('preparation', launch['preparation']),
                             ('criteria', launch['criteria']), ('protocol', launch['protocol'])):
            if common.file_digest(Path(record['path'])) != record['sha256']:
                raise ValueError(f'{name} changed during numerical validation')
        common.verify_sources(tracked)
        verify_fixture(preparation, slot)
        audit.update(status='verified_observation', numerical_report=file_record(num_report_path))
    except BaseException as exc:
        audit.update(status='execution_or_identity_failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        common.atomic_json(audit_path, audit)
    print(json.dumps({'status': audit['status'], 'machine_screens_pass': audit['machine_screens_pass'],
                      'numerical_clearance': False, 'audit': str(audit_path)}), flush=True)


if __name__ == '__main__':
    main()
