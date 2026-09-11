"""CPU checks for wrong-fixture prevention and honest numerical dispositions."""
import copy
import dataclasses
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import cdrm_tiled_pilot_numerical as numerical
import cdrm_tiled_pilot_prepare as prepare
import cdrm_tiled_common as common


def test_role_selection_rejects_reserved_or_reassigned_examples():
    prep = {'status': 'complete', 'confirmation': prepare.confirmation_allocation()}
    chosen = numerical.select_slot(prep, 7501, 2500, 'bf16')
    assert chosen['example_offset'] == 448 and chosen['example_stop'] == 512
    prep['confirmation']['slots'][-1]['example_offset'] = 512
    prep['confirmation']['slots'][-1]['example_stop'] = 576
    with pytest.raises(ValueError, match='allocation'):
        numerical.select_slot(prep, 7501, 2500, 'bf16')
    with pytest.raises(ValueError, match='Undeclared'):
        numerical.select_slot(prep, 7500, 100, 'fp32')


@pytest.fixture
def checkpoint_fixture():
    assert not torch.cuda.is_available(), 'Use the explicitly GPU-disabled CPU container'
    model, construction = common.cpu_initial_model(7500)
    initial_identity = {'source_sha256': {}, 'initialization': construction}
    initial = {'format': common.INIT_FORMAT, 'identity': initial_identity,
               'identity_sha256': common.json_digest(initial_identity), 'initialization': construction}
    initial_record = {'path': 'initial.pt', 'sha256': 'initial-file-sha'}
    prep = {'reused_data': {'train': {'sha': 'train'}, 'dev': {'sha': 'dev'}}}
    config = dataclasses.asdict(common.config('tiled', 'bf16'))
    identity = {'initialization': construction, 'precision': 'bf16_fp32_state', 'arm': 'cdrm-bf16',
                'model_config': config, 'initial_checkpoint_sha256': initial_record['sha256'],
                'initial_identity_sha256': initial['identity_sha256'], 'source_sha256': {},
                'data': prep['reused_data'], 'protocol_sha256': numerical.PROTOCOL_SHA256,
                'physical_batch': 64, 'shuffle_seed': 925704, 'accumulation': False}
    optimizer = common.optimizer_for(model)
    for parameter in model.parameters():
        optimizer.state[parameter] = {'step': torch.tensor(1000., dtype=torch.float32),
                                      'exp_avg': torch.zeros_like(parameter), 'exp_avg_sq': torch.zeros_like(parameter)}
    payload = {'format': common.FORMAT, 'pilot_schema': 'cdrm-tiled-pilot-v1', 'identity': identity,
               'identity_sha256': common.json_digest(identity), 'completed_updates': 1000,
               'initialization': construction, 'precision': 'bf16_fp32_state', 'model_config': config,
               'initial_checkpoint': initial_record, 'ancestry': {'initial': initial_record},
               'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    slot = {'model_seed': 7500, 'completed_updates': 1000, 'checkpoint_trajectory': 'bf16'}
    return payload, initial, initial_record, prep, slot


def test_checkpoint_role_requires_same_seed_precision_and_initialization(checkpoint_fixture):
    payload, initial, initial_record, prep, slot = checkpoint_fixture
    numerical.validate_checkpoint_role(payload, initial, initial_record, prep, slot)
    for key, wrong in [('model_seed', 7501), ('completed_updates', 2500), ('checkpoint_trajectory', 'fp32')]:
        changed = {**slot, key: wrong}
        with pytest.raises(ValueError):
            numerical.validate_checkpoint_role(payload, initial, initial_record, prep, changed)
    bad = copy.deepcopy(payload)
    bad['identity_sha256'] = 'corrupted'
    with pytest.raises(ValueError, match='identity hash'):
        numerical.validate_checkpoint_role(bad, initial, initial_record, prep, slot)
    with pytest.raises(ValueError, match='initialization'):
        numerical.validate_checkpoint_role(payload, initial, {**initial_record, 'sha256': 'another-init'}, prep, slot)


def test_legacy_loader_retains_all_master_weights_and_trained_moments(checkpoint_fixture):
    payload = checkpoint_fixture[0]
    record = numerical.validate_checkpoint_tensors(payload)
    assert record['canonical_parameter_tensors'] == 45
    assert record['optimizer_sha256'] == common.state_digest(payload['optimizer'])
    bad = copy.deepcopy(payload)
    first = next(iter(bad['optimizer']['state']))
    bad['optimizer']['state'][first]['step'] = torch.tensor(999.)
    with pytest.raises(ValueError, match='Adam moments/steps'):
        numerical.validate_checkpoint_tensors(bad)


def test_completed_formal_failure_is_verified_without_becoming_clearance():
    fixture = {'batch_sha256': 'batch', 'dataset_sha256': 'dataset', 'manifest_sha256': 'manifest',
               'shape': [64, 256], 'native_targets': 6144, 'example_offset': 64, 'split': 'dev'}
    launch = {'fixture': fixture, 'checkpoint': {'sha256': 'checkpoint'},
              'slot': {'completed_updates': 1000, 'model_seed': 7500},
              'checkpoint_state': {'weights_sha256': 'weights'}, 'validator': {'sha256': 'validator'}}
    report = {'status': 'diagnostics_complete', 'machine_screens_pass': False, 'numerical_clearance': False,
              'checkpoint': {'sha256': 'checkpoint', 'completed_updates': 1000},
              'criteria': {'sha256': numerical.CONTRACT_SHA256}, 'validator_sha256': 'validator',
              'fixture': {**fixture, 'sha256': 'batch', 'manifest': {'manifest_sha256': 'manifest'}, 'accumulation': False},
              'construction': {arm: {'seed': 7500, 'starting_weights_sha256': 'weights'}
                               for arm in ('naive_fp32', 'tiled_fp32', 'tiled_bf16')},
              'comparisons': {name: {'machine_screens_pass': False} for name in
                              ('tiled_fp32_vs_naive_fp32', 'tiled_bf16_vs_tiled_fp32')}}
    result = numerical.verify_completed_report(report, launch)
    assert result['identity_and_fixture_verified'] is True
    assert result['machine_screens_pass'] is False and result['numerical_clearance'] is False
    report['fixture']['sha256'] = 'other-batch'
    with pytest.raises(ValueError, match='prospective'):
        numerical.verify_completed_report(report, launch)
