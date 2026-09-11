"""CPU-only checks of released scheduling and exact resume/data contracts."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from rt_precision_train import (
    BATCH, OPTIMIZER, SCHEMA, data_window, learning_rate, save_checkpoint,
    validate_model_checkpoint, validate_optimizer_checkpoint, validate_resume_metadata,
)


def test_schedule_matches_released_cosine_at_every_step_and_keeps_long_warmup():
    from olmo.optim import CosWithWarmup
    upstream = CosWithWarmup(grad_clip_warmup_steps=None, grad_clip_warmup_factor=None,
                             warmup_min_lr=None, warmup_steps=5000, alpha_0=.1, alpha_f=.1)
    for update in range(1, 12502):
        assert learning_rate(update) == upstream.get_lr(1e-3, update, 12500)
    assert learning_rate(1) == pytest.approx(.00010018)
    assert learning_rate(100) == pytest.approx(.000118)
    assert learning_rate(500) == pytest.approx(.00019)
    assert learning_rate(5000) == .001
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            learning_rate(invalid)


def test_sequential_data_resume_uses_next_unconsumed_full_batch_and_never_wraps():
    rows = 500 * BATCH
    windows = [data_window(index, rows) for index in range(500)]
    assert windows[0] == (0, 512)
    assert windows[-1] == (rows - BATCH, rows)
    assert all(previous[1] == following[0] for previous, following in zip(windows, windows[1:]))
    assert data_window(100, rows) == (windows[99][1], windows[100][1])
    with pytest.raises(ValueError, match='exhausted'):
        data_window(500, rows)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            data_window(invalid, rows)


def retained_packet(completed=100):
    parameters = {'weight': torch.zeros(3, 2), 'bias': torch.zeros(3)}
    names = list(parameters)
    contract = {
        'model_config': {'recurrent_precision_policy': 'legacy', 'd_model': 1024},
        'optimizer_parameter_names': names, 'seed': 20260910,
        'data': {'manifest_sha256': 'a' * 64, 'shape': [256000, 512],
                 'heldout_exclusion_manifest_sha256': 'b' * 64},
        'source_sha256': {'model.py': 'c' * 64},
        'runtime': {'torch': 'pinned', 'cache': '/retained/cache'}, 'protocol_sha256': 'd' * 64,
    }
    group = dict(OPTIMIZER, params=list(range(len(names))),
                 lr=learning_rate(completed) if completed else OPTIMIZER['lr'])
    optimizer_state = {index: {'step': torch.tensor(float(completed)),
                               'exp_avg': torch.zeros_like(parameters[name]),
                               'exp_avg_sq': torch.ones_like(parameters[name])}
                       for index, name in enumerate(names)} if completed else {}
    packet = {
        'schema': SCHEMA, 'resume_contract': copy.deepcopy(contract),
        'completed_updates': completed, 'next_data_row': completed * BATCH,
        'model_config': contract['model_config'], 'optimizer_parameter_names': names,
        'seed': contract['seed'], 'data_manifest_sha256': contract['data']['manifest_sha256'],
        'source_sha256': contract['source_sha256'], 'runtime': contract['runtime'],
        'protocol_sha256': contract['protocol_sha256'],
        'heldout_manifest_sha256': contract['data']['heldout_exclusion_manifest_sha256'],
        'policy': 'legacy', 'initial_state_sha256': 'e' * 64,
        'rng': dict(python=None, numpy=None, torch_cpu=None, torch_cuda=None),
        'model': copy.deepcopy(parameters),
        'optimizer': {'state': optimizer_state, 'param_groups': [group]},
    }
    return packet, contract, parameters


@pytest.mark.parametrize('completed', [0, 100])
def test_resume_roundtrip_preserves_metadata_and_native_adam_state(tmp_path, completed):
    packet, contract, parameters = retained_packet(completed)
    path = tmp_path / 'state.pt'
    record = save_checkpoint(path, packet)
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    assert len(record['sha256']) == 64 and record['completed_updates'] == completed
    assert validate_resume_metadata(loaded, contract, 500) == completed
    validate_model_checkpoint(loaded, parameters)
    validate_optimizer_checkpoint(loaded, parameters)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, packet)


@pytest.mark.parametrize('change', ['source', 'runtime_cache', 'data', 'policy', 'protocol', 'row_offset', 'endpoint'])
def test_resume_rejects_changed_identity_or_progress(change):
    packet, contract, _ = retained_packet()
    endpoint = 500
    if change == 'source':
        contract['source_sha256'] = {'model.py': 'new-source'}
    elif change == 'runtime_cache':
        contract['runtime'] = {'torch': 'pinned', 'cache': '/different/cache'}
    elif change == 'data':
        contract['data'] = dict(contract['data'], manifest_sha256='other-data')
    elif change == 'policy':
        contract['model_config'] = dict(contract['model_config'], recurrent_precision_policy='bf16_fp32_state')
    elif change == 'protocol':
        contract['protocol_sha256'] = 'new-protocol'
    elif change == 'row_offset':
        packet['next_data_row'] += BATCH
    elif change == 'endpoint':
        endpoint = 100
    with pytest.raises(ValueError):
        validate_resume_metadata(packet, contract, endpoint)


@pytest.mark.parametrize('change', ['missing_moment', 'moment_dtype', 'moment_shape', 'negative_second_moment',
                                  'adam_step', 'adam_lr', 'names'])
def test_resume_rejects_incompatible_optimizer_state(change):
    packet, _, parameters = retained_packet()
    values = packet['optimizer']['state'][0]
    if change == 'missing_moment':
        del values['exp_avg_sq']
    elif change == 'moment_dtype':
        values['exp_avg'] = values['exp_avg'].bfloat16()
    elif change == 'moment_shape':
        values['exp_avg'] = torch.zeros(1)
    elif change == 'negative_second_moment':
        values['exp_avg_sq'][0, 0] = -1
    elif change == 'adam_step':
        values['step'] += 1
    elif change == 'adam_lr':
        packet['optimizer']['param_groups'][0]['lr'] = .001
    elif change == 'names':
        packet['optimizer_parameter_names'] = list(reversed(packet['optimizer_parameter_names']))
    with pytest.raises(ValueError):
        validate_optimizer_checkpoint(packet, parameters)


def test_resume_does_not_silently_cast_nonfp32_or_accept_nonfinite_weights():
    packet, _, parameters = retained_packet()
    packet['model']['weight'] = packet['model']['weight'].bfloat16()
    with pytest.raises(ValueError, match='FP32 checkpoint'):
        validate_model_checkpoint(packet, parameters)
    packet['model']['weight'] = parameters['weight'].clone()
    packet['model']['weight'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='FP32 checkpoint'):
        validate_model_checkpoint(packet, parameters)
