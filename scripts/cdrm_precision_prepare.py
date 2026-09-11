#!/usr/bin/env python3
"""Generate predeclared numerical cases only after an externally anchored freeze.

CPU container only; no model forward, optimizer update, GPU, or resampling.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from cdrm.mad_data import generate_dataset, load_dataset, overlap_audit, save_dataset
from cdrm_precision_confirm import (checked_json, confirmation_sources, validate_checkpoint,
                                    validate_decision, validate_fixture_roles, validate_roles)
from cdrm_tiled_common import (INIT_FORMAT, PRESET, TASK, atomic_json, cpu_initial_model,
    dataset_identity, file_digest, json_digest, optimizer_for, save_checkpoint,
    snapshot_sources, state_digest, validate_data, verify_sources)

LINEAGE = Path('.runtime/cdrm-numerical-resolution/20260907T232931Z')
PILOT = Path('.runtime/cdrm-tiled-pilot/20260907T212606Z')
PREPARATION_SHA256 = '2331eae9100b71a7dc5beec87b5f3d119f27bdfd99d941a8bec52c725fd30cfe'
SEED, EXAMPLES, INITIALIZATION_SEED = 925903, 192, 7502


def required_sources():
    result = confirmation_sources()
    for name in ('scripts/cdrm_precision_prepare.py', 'tests/test_cdrm_precision_prepare.py'):
        result[name] = file_digest(Path(name))
    return result


def validate_frozen_sources(decision):
    tracked = decision['source_sha256']
    required = required_sources()
    if any(tracked.get(name) != expected for name, expected in required.items()):
        raise ValueError('Candidate freeze must include current producer, tests and confirmation dependencies')
    verify_sources(tracked)
    return tracked


def validate_outputs(output, data_root, *, lineage=LINEAGE):
    base = Path(lineage).resolve()
    paths = [Path(output).resolve(), Path(data_root).resolve()]
    for path in paths:
        if not path.is_relative_to(base) or path == base:
            raise ValueError('Preparation and data must use the new numerical-resolution lineage')
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite a preparation/data path: {path}')
    if paths[0] == paths[1] or paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        raise ValueError('Preparation and data need separate, nonoverlapping subtrees')


def checked_inherited(record):
    path = LINEAGE / record['retained_path']
    if file_digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
        raise ValueError('Inherited checkpoint changed')
    return path, torch.load(path, map_location='cpu', weights_only=False)


def preflight(args):
    """All external anchors and source coverage are checked before generation."""
    decision = checked_json(args.decision, args.decision_sha256)
    roles = checked_json(decision['roles']['path'], decision['roles']['sha256'])
    validate_decision(decision, roles)
    declared_roles = validate_roles(roles)
    for key in ('original_criteria', 'reference_contract'):
        checked = decision[key]
        if file_digest(Path(checked['path'])) != checked['sha256']:
            raise ValueError('Frozen numerical criteria bytes changed')
    tracked = validate_frozen_sources(decision)
    prepared = checked_json(args.preparation_report, args.preparation_sha256)
    if prepared.get('schema') != 'cdrm-numerical-resolution-preparation-v1' or prepared.get('status') != 'complete':
        raise ValueError('Require completed numerical lineage provenance setup')
    if prepared['future_roles']['sha256'] != decision['roles']['sha256']:
        raise ValueError('Original role declaration differs from candidate freeze')
    records, checkpoints = {}, {}
    for role in ('fp32_trained_u1000', 'bf16_trained_u1000', 'initial_seed7500'):
        record = prepared['checkpoints'][role]
        if role != 'initial_seed7500' and record['sha256'] != declared_roles[role]['checkpoint_sha256']:
            raise ValueError('Prepared trained checkpoint differs from prospective role before generation')
        path, payload = checked_inherited(record)
        if role != 'initial_seed7500':
            validate_checkpoint(payload, role)
        elif (payload.get('format') != INIT_FORMAT or payload.get('completed_updates') != 0
              or payload['initialization']['seed'] != 7500 or payload['optimizer']['state']
              or payload['identity_sha256'] != json_digest(payload['identity'])):
            raise ValueError('Original initialization identity/optimizer differs')
        verify_sources(payload['identity']['source_sha256'])
        records[role] = {'path': str(path), 'sha256': record['sha256'], 'bytes': record['bytes']}
        checkpoints[role] = payload
    validate_outputs(args.output_dir, args.confirmation_root)
    return decision, roles, prepared, tracked, records, checkpoints


def fresh_initialization(prior, tracked, decision_ref, roles_ref, corpus):
    model, construction = cpu_initial_model(INITIALIZATION_SEED, preset=PRESET)
    config = asdict(model.config)
    if config != prior['model_config']:
        raise AssertionError('Distinct seed changed the scientific architecture')
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    if (state_digest(optimizer.state_dict()) != state_digest(prior['optimizer'])
            or state_digest(scheduler.state_dict()) != state_digest(prior['scheduler'])):
        raise AssertionError('Distinct initialization changed original optimizer/scheduler semantics')
    identity = {'schema': INIT_FORMAT, 'source_sha256': copy.deepcopy(tracked),
                'candidate_freeze': decision_ref, 'prospective_roles': roles_ref,
                'model_config': config, 'initialization': construction,
                'corpus': corpus, 'parent_source_sha256': copy.deepcopy(prior['identity']['source_sha256']),
                'scope': 'Independent initialization for fixed-state numerical confirmation only'}
    payload = {'format': INIT_FORMAT, 'identity': identity, 'identity_sha256': json_digest(identity),
               'model_config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'initialization': construction,
               'completed_updates': 0, 'completed_epochs': 0, 'batch_in_epoch': 0,
               'precision': 'fp32', 'weights_only_initialization': True,
               'memory_state': 'Rebuilt for each independent forward'}
    validate_checkpoint(payload, 'distinct_initialization')
    if state_digest(payload['model']) == state_digest(prior['model']):
        raise AssertionError('Distinct initialization unexpectedly duplicated original weights')
    return payload, {'initialization_seed': INITIALIZATION_SEED, 'canonical_parameter_tensors': 45,
                     'optimizer_and_scheduler_exactly_equal_original_initialization': True,
                     'model_forward_executed': False, 'construction': construction}


def fixture_manifest(decision_sha, roles_sha, roles, dataset, data_root, checkpoints):
    allocation = validate_roles(roles)
    if len(dataset) != EXAMPLES or dataset.manifest['seed'] != SEED:
        raise ValueError('Generated corpus differs from prospective allocation')
    rows = []
    for name in ('fp32_trained_u1000', 'bf16_trained_u1000', 'distinct_initialization'):
        role = allocation[name]
        start, stop = role['example_offset'], role['example_stop']
        rows.append({'name': name, 'example_offset': start, 'example_stop': stop,
                     'batch_sha256': dataset.take(slice(start, stop)).sha256,
                     'checkpoint': checkpoints[name]})
    result = {'schema': 'cdrm-precision-fresh-fixtures-v1', 'status': 'generated',
              'candidate_freeze_sha256': decision_sha, 'roles_sha256': roles_sha,
              'corpus': {'root': str(data_root), 'split': 'dev', 'seed': SEED, 'examples': EXAMPLES,
                         'dataset_sha256': dataset.sha256,
                         'manifest_sha256': dataset.manifest['manifest_sha256']}, 'roles': rows}
    validate_fixture_roles(result, decision_sha, roles_sha)
    return result


def prepare(args, report):
    if (not Path('/.dockerenv').exists() or Path.cwd() != Path('/workspace/cdrm-w-latent')
            or torch.cuda.is_available() or torch.cuda.is_initialized()):
        raise RuntimeError('Use the explicitly GPU-disabled project CPU container')
    torch.set_num_threads(1)
    decision, roles, prepared, tracked, records, checkpoints = preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report['output_directory_created_by_this_invocation'] = True
    report.update(source_sha256=tracked,
                  candidate_freeze={'path': str(args.decision), 'sha256': args.decision_sha256},
                  roles=decision['roles'], execution={'device': 'cpu', 'cuda_available': False,
                      'torch': torch.__version__, 'model_forward_executed': False, 'optimizer_updates': 0})
    snapshot_sources(args.output_dir, tracked)
    (args.output_dir/'candidate-freeze.json').write_bytes(args.decision.read_bytes())
    (args.output_dir/'future-confirmation-roles.json').write_bytes(Path(decision['roles']['path']).read_bytes())
    old_root = PILOT/'inherited/data'
    old = {'training': load_dataset(old_root, TASK, 'train'),
           'development': load_dataset(old_root, TASK, 'dev'),
           'early_confirmation': load_dataset(old_root/'confirmation', TASK, 'dev'),
           'pilot_numerical': load_dataset(LINEAGE/'inherited/pilot/data/confirmation', TASK, 'dev')}
    initial = checkpoints['initial_seed7500']
    for name, key in (('training','train'),('development','dev'),('early_confirmation','confirmation')):
        validate_data(old[name])
        if dataset_identity(old[name]) != initial['identity']['data'][key]:
            raise ValueError(f'Original overlap-audit corpus identity changed: {name}')
    validate_data(old['pilot_numerical'])
    if dataset_identity(old['pilot_numerical'])['array_sha256'] != '4e51bb0f9811e7747972147fc8a5af47ecfa7584eb54ca4dddff389829b5b8f7':
        raise ValueError('Inherited pilot numerical corpus changed')
    # Exactly one native draw, after all candidate/source/role anchors pass.
    dataset = generate_dataset(TASK, 'dev', SEED, EXAMPLES, {'num_tokens_to_copy': 96})
    validate_data(dataset)
    save_dataset(args.confirmation_root, dataset)
    dataset = load_dataset(args.confirmation_root, TASK, 'dev', verify=True)
    overlap = overlap_audit({**old, 'fresh_numerical': dataset})
    atomic_json(args.output_dir/'overlap-audit.json', overlap)
    report['overlap_audit'] = overlap
    report['old_corpora'] = {name: dataset_identity(value) for name,value in old.items()}
    if (not overlap['no_exact_input_cross_split_overlap']
            or overlap['within_split']['fresh_numerical']['input']['duplicate_rows']):
        raise AssertionError('Native draws overlap previous inputs; retained without resampling and require review')
    corpus = dataset_identity(dataset)
    payload, initial_checks = fresh_initialization(initial, tracked,
        report['candidate_freeze'], decision['roles'], corpus)
    records['distinct_initialization'] = save_checkpoint(args.output_dir/'init-7502.pt', payload)
    report['initialization_checks'] = initial_checks
    fixtures = fixture_manifest(args.decision_sha256, decision['roles']['sha256'], roles,
                                dataset, args.confirmation_root, records)
    fixtures.update(source_sha256=tracked, overlap_audit={'path': str(args.output_dir/'overlap-audit.json'),
        'sha256': file_digest(args.output_dir/'overlap-audit.json'), 'no_resampling': True,
        'no_exact_input_cross_split_overlap': True})
    verify_sources(tracked)
    if file_digest(args.decision) != args.decision_sha256 or file_digest(Path(decision['roles']['path'])) != decision['roles']['sha256']:
        raise ValueError('Candidate/role freeze changed during generation')
    for record in records.values():
        if file_digest(Path(record['path'])) != record['sha256']:
            raise ValueError('A role checkpoint changed during preparation')
    if torch.cuda.is_initialized():
        raise AssertionError('CPU preparation initialized CUDA')
    atomic_json(args.output_dir/'fixtures.json', fixtures)
    report.update(status='generated', fixtures={'path': str(args.output_dir/'fixtures.json'),
        'sha256': file_digest(args.output_dir/'fixtures.json')}, checkpoints=records,
        native_draws=EXAMPLES, resampling=False, no_model_forward=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision',type=Path,required=True)
    parser.add_argument('--decision-sha256',required=True)
    parser.add_argument('--preparation-report',type=Path,default=LINEAGE/'reports/preparation.json')
    parser.add_argument('--preparation-sha256',default=PREPARATION_SHA256)
    parser.add_argument('--output-dir',type=Path,default=LINEAGE/'prepare')
    parser.add_argument('--confirmation-root',type=Path,default=LINEAGE/'data/confirmation')
    args=parser.parse_args(argv)
    started=time.monotonic()
    report={'schema':'cdrm-precision-fixture-preparation-v1','status':'preparation_started',
            'arguments':{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}}
    try:prepare(args,report)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        if report.get('output_directory_created_by_this_invocation') and not (args.output_dir/'report.json').exists():
            atomic_json(args.output_dir/'report.json',report)
    print(json.dumps({'status':report['status'],'fixtures':report['fixtures'],'resampling':False,'model_forward':False}))


if __name__=='__main__':main()
