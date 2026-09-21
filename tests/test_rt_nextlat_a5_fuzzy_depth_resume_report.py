import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import rt_nextlat_a5_fuzzy_depth_resume_report as reporter
from scripts.rt_nextlat_a5_fuzzy_lr_train import learning_rate_state, schedule_configuration
from test_rt_nextlat_a5_fuzzy_report import a5_metric, fuzzy_metric


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def build_stage(directory, *, start, endpoint, parent=None, legacy_boundary=False):
    directory.mkdir()
    config, identity, sources, initialization = {'depth': 3}, {'a5': 'a', 'fuzzy': 'f'}, {}, {'seed': 1234}
    for name, value in [('model-config.json', config), ('data-identity.json', identity), ('source-manifest.json', sources)]:
        write_json(directory / name, value)
    contract = {'schema': reporter.lr.TRAIN_SCHEMA, 'model_config': config, 'mode': 'mixed',
        'initialization': initialization, 'configuration_file_sha256': reporter.sha(directory / 'model-config.json'),
        'data_sha256': reporter.json_sha(identity), 'source_sha256': reporter.json_sha(sources),
        'batch_per_task': 2560, 'streams': {task: {'train_rows': 12800} for task in ('a5', 'fuzzy')},
        'learning_rate_schedule': schedule_configuration()}
    report = {'schema': reporter.lr.TRAIN_SCHEMA, 'status': 'running' if parent is None else 'stopped',
        'contract': contract, 'completed_updates': endpoint, 'requested_endpoint': 5000, 'start_update': start,
        'parent_checkpoint': None if parent is None else {key: parent[key] for key in ('path', 'sha256')},
        'initialization': initialization, 'checkpoints': [], 'evaluations': [],
        'confirmation_evaluated': False, 'latent_rollout_evaluated': False}
    history = []
    for update in range(start, endpoint + 1):
        seen = {task: update * 2560 for task in ('a5', 'fuzzy')}
        chains = {task: f'{task}-{update}' for task in seen}
        packet = {'schema': reporter.lr.TRAIN_SCHEMA, 'contract': contract, 'completed_updates': update,
            'examples_seen': seen, 'order_chains': chains, 'initialization': initialization,
            'next_cursors': {task: {'absolute_example_offset': n, 'epoch': n // 12800, 'position': n % 12800}
                             for task, n in seen.items()},
            'model': {'weight': torch.tensor([float(update)])},
            'optimizer': {'state': {}, 'param_groups': [{'lr': learning_rate_state(update, schedule_configuration())['optimizer_lr']}]},
            'rng': {'torch': torch.tensor([1, 2], dtype=torch.uint8), 'numpy': np.array([1, 2], dtype='uint32')},
            'learning_rate_state': learning_rate_state(update, schedule_configuration())}
        path = directory / 'checkpoints' / f'step-{update:06d}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save(packet, path, _use_new_zipfile_serialization=not (legacy_boundary and update == start))
        cp = {'completed_updates': update, 'path': str(path), 'sha256': reporter.sha(path), 'bytes': path.stat().st_size}
        report['checkpoints'].append(cp)
        for metric in [a5_metric(update, role='dev'), a5_metric(update), fuzzy_metric(update)]:
            metric['checkpoint'] = {key: cp[key] for key in ('path', 'sha256', 'completed_updates')}
            report['evaluations'].append(metric)
        if update > start:
            history.append({'update': update, 'order_chains': chains, 'seconds': 1.0, 'loss': 1.0})
    write_json(directory / 'report.json', report)
    (directory / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history))
    return report


@pytest.fixture
def lineage(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    prior = build_stage(old, start=0, endpoint=3)
    with (old / 'history.jsonl').open('ab') as stream:
        stream.write(json.dumps({'update': 4, 'loss': 999}).encode() + b'\n' + b'\0' * 128)
    final = build_stage(new, start=3, endpoint=5, parent=prior['checkpoints'][-1], legacy_boundary=True)
    return old, new, prior, final


def test_exact_state_resume_stitches_once_and_preserves_corrupt_tail(lineage):
    old, new, prior, final = lineage
    before = (old / 'history.jsonl').read_bytes()
    run = reporter.load_run(new)
    assert [row['update'] for row in run['history']] == [1, 2, 3, 4, 5]
    assert all(row['loss'] == 1 for row in run['history'])
    assert (old / 'history.jsonl').read_bytes() == before
    assert run['checkpoints'][0]['verified_local_path'].startswith(str(old))
    assert run['checkpoints'][3]['sha256'] == prior['checkpoints'][-1]['sha256']
    assert len([row for row in run['evaluations'] if row['update'] == 3]) == 3
    assert run['lineage'][0]['history']['discarded_valid_update_range'] == [4, 4]
    assert run['lineage'][0]['history']['first_invalid_tail_byte_offset'] is not None
    assert run['lineage'][1]['boundary_state_audit']['entire_packet_exact']
    assert not run['lineage'][1]['boundary_checkpoint_copy']['byte_identical']
    assert run['lineage'][1]['omitted_child_boundary_evaluations'] == 3


@pytest.mark.parametrize('field', ['model', 'optimizer', 'rng', 'learning_rate_state'])
def test_boundary_state_change_rejected_even_when_new_hash_matches(lineage, field):
    _, new, _, final = lineage
    cp = final['checkpoints'][0]
    path = Path(cp['path'])
    packet = torch.load(path, map_location='cpu', weights_only=False)
    if field == 'model': packet[field]['weight'][0] += 1
    if field == 'optimizer': packet[field]['param_groups'][0]['lr'] = .01
    if field == 'rng': packet[field]['numpy'][0] += 1
    if field == 'learning_rate_state': packet[field]['next_update_lr'] = .02
    torch.save(packet, path)
    cp.update(sha256=reporter.sha(path), bytes=path.stat().st_size)
    write_json(new / 'report.json', final)
    with pytest.raises(ValueError, match='full state differs'):
        reporter.load_run(new)


@pytest.mark.parametrize('damage', ['parent_hash', 'contract', 'missing_update', 'committed_corruption', 'terminal_tail', 'running_terminal', 'subset_terminal'])
def test_invalid_lineage_rejected(lineage, damage):
    old, new, prior, final = lineage
    if damage == 'parent_hash': final['parent_checkpoint']['sha256'] = 'x'
    if damage == 'contract': final['contract']['batch_per_task'] = 1280
    if damage == 'missing_update':
        path = new / 'history.jsonl'
        path.write_text(path.read_text().splitlines(keepends=True)[-1])
    if damage == 'committed_corruption': (old / 'history.jsonl').write_bytes(b'\0')
    if damage == 'terminal_tail':
        with (new / 'history.jsonl').open('ab') as stream: stream.write(b'\0')
    if damage == 'running_terminal': final['status'] = 'running'
    if damage == 'subset_terminal':
        for metric in final['evaluations']:
            if metric['update'] == 5 and metric['task'] == 'a5':
                replacement = a5_metric(5, role=metric['role'], rows=4096)
                replacement['checkpoint'] = metric['checkpoint']
                metric.clear(); metric.update(replacement)
    write_json(new / 'report.json', final)
    with pytest.raises(ValueError): reporter.load_run(new)


def test_exact_tree_rejects_tensor_dtype_and_rng_changes():
    a = {'rng': np.array([1, 2], dtype='uint32'), 'model': torch.tensor([1.])}
    assert reporter.exact_tree(a, copy.deepcopy(a))
    b = copy.deepcopy(a); b['model'] = b['model'].double()
    assert not reporter.exact_tree(a, b)
    b = copy.deepcopy(a); b['rng'][0] += 1
    assert not reporter.exact_tree(a, b)


def test_clean_stop_at_restored_boundary_uses_new_full_evaluation(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    prior = build_stage(old, start=0, endpoint=3)
    for metric in prior['evaluations']:
        if metric['update'] == 3 and metric['task'] == 'a5':
            replacement = a5_metric(3, role=metric['role'], rows=4096)
            replacement['checkpoint'] = metric['checkpoint']
            metric.clear(); metric.update(replacement)
    write_json(old / 'report.json', prior)
    build_stage(new, start=3, endpoint=3, parent=prior['checkpoints'][-1], legacy_boundary=True)
    run = reporter.load_run(new)
    assert [row['update'] for row in run['history']] == [1, 2, 3]
    assert run['lineage'][-1]['history']['used_records'] == 0
    assert run['lineage'][-1]['endpoint_boundary_reevaluation_used']
    assert all(row.get('rows', row.get('examples')) == (102400 if row['task'] == 'a5' else 1280)
               for row in run['evaluations'] if row['update'] == 3)
