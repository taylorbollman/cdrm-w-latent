"""Reject wrong endpoints, altered authorities and silently cast checkpoint weights."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_precision_eval_contract import validate_evaluation_contract


@pytest.fixture
def packet():
    checkpoint, heldout, protocol, train = 'a' * 64, 'b' * 64, 'c' * 64, 'd' * 64
    source = {'recurrent-transformer/olmo/model.py': 'e' * 64,
              'recurrent-transformer/olmo/config.py': 'f' * 64, 'scripts/training.py': '1' * 64}
    parameters = {'embedding.weight': torch.ones(3, 2), 'output.weight': torch.ones(3, 2)}
    config = {'recurrent_precision_policy': 'legacy'}
    contract = {'data': {'manifest_sha256': train, 'heldout_exclusion_manifest_sha256': heldout},
                'model_config': config, 'seed': 20260910, 'source_sha256': source,
                'protocol_sha256': protocol, 'optimizer_parameter_names': list(parameters)}
    state = {'schema': 'rt-precision-state-v1', 'model_config': config, 'model': parameters,
             'optimizer_parameter_names': list(parameters), 'completed_updates': 100,
             'next_data_row': 51200, 'seed': 20260910, 'data_manifest_sha256': train,
             'source_sha256': source, 'policy': 'legacy', 'protocol_sha256': protocol,
             'heldout_manifest_sha256': heldout, 'resume_contract': contract}
    report = {'schema': 'rt-precision-training-v1', 'status': 'complete', 'precision': 'bf16',
              'policy': 'legacy', 'seed': 20260910, 'completed_updates': 100, 'endpoint': 100,
              'model_config': config, 'source_sha256': source, 'protocol_sha256': protocol,
              'heldout_manifest_sha256': heldout, 'resume_contract': contract,
              'checkpoints': [{'path': 'checkpoints/step-000100.pt', 'sha256': checkpoint,
                               'completed_updates': 100, 'next_data_row': 51200}]}
    return {'training_report': copy.deepcopy(report), 'state': copy.deepcopy(state),
            'checkpoint_sha256': checkpoint, 'heldout_manifest_sha256': heldout,
            'protocol_sha256': protocol, 'current_source_sha256': source.copy(),
            'expected_parameters': {name: value.clone() for name, value in parameters.items()}}


def test_valid_contract_reads_actual_parameter_metadata_without_changing_weights(packet):
    expected = copy.deepcopy(packet['state'])
    result = validate_evaluation_contract(**packet)
    assert result['status'] == 'verified' and result['parameter_count'] == 12
    for name, tensor in expected['model'].items():
        assert torch.equal(packet['state']['model'][name], tensor)


def test_torch_checkpoint_tuples_match_json_roundtripped_report(packet):
    packet['state']['resume_contract']['optimizer'] = {'betas': (0.9, 0.95)}
    packet['state']['model_config']['tuple_metadata'] = (1, 2)
    packet['state']['resume_contract']['model_config'] = copy.deepcopy(packet['state']['model_config'])
    packet['training_report']['resume_contract'] = copy.deepcopy(packet['state']['resume_contract'])
    packet['training_report']['model_config'] = copy.deepcopy(packet['state']['model_config'])
    packet['training_report'] = json.loads(json.dumps(packet['training_report']))
    assert isinstance(packet['state']['resume_contract']['optimizer']['betas'], tuple)
    assert isinstance(packet['training_report']['resume_contract']['optimizer']['betas'], list)
    assert validate_evaluation_contract(**packet)['status'] == 'verified'


@pytest.mark.parametrize('field', ['checkpoint_sha256', 'heldout_manifest_sha256', 'protocol_sha256'])
def test_changed_external_authority_rejected(packet, field):
    packet[field] = '0' * 64
    with pytest.raises(ValueError):
        validate_evaluation_contract(**packet)


@pytest.mark.parametrize('field,value', [
    ('schema', 'rt-precision-arm-v1'), ('seed', 20260911), ('policy', 'bf16_fp32_state'),
    ('completed_updates', 0), ('completed_updates', 500), ('next_data_row', 51199)])
def test_wrong_checkpoint_identity_rejected(packet, field, value):
    packet['state'][field] = value
    with pytest.raises(ValueError):
        validate_evaluation_contract(**packet)


def test_incomplete_training_report_rejected(packet):
    packet['training_report']['status'] = 'running'
    with pytest.raises(ValueError, match='completed'):
        validate_evaluation_contract(**packet)


def test_unlisted_or_duplicated_checkpoint_record_rejected(packet):
    packet['training_report']['checkpoints'] *= 2
    with pytest.raises(ValueError, match='exactly one'):
        validate_evaluation_contract(**packet)


@pytest.mark.parametrize('change', ['missing', 'added', 'changed'])
def test_model_source_coverage_and_hash_are_anchored(packet, change):
    source = packet['current_source_sha256']
    if change == 'missing':
        source.pop('recurrent-transformer/olmo/config.py')
    elif change == 'added':
        source['recurrent-transformer/olmo/new.py'] = '2' * 64
    else:
        source['recurrent-transformer/olmo/model.py'] = '2' * 64
    with pytest.raises(ValueError, match='source'):
        validate_evaluation_contract(**packet)


def test_changed_heldout_exclusion_identity_rejected_even_if_top_level_hashes_match(packet):
    for record in [packet['state'], packet['training_report']]:
        record['resume_contract']['data']['heldout_exclusion_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='exclusion authority'):
        validate_evaluation_contract(**packet)


@pytest.mark.parametrize('change', ['bf16', 'nonfinite', 'wrong_shape', 'missing', 'extra'])
def test_canonical_fp32_finite_saved_parameters_rejected_before_load_cast(packet, change):
    model = packet['state']['model']
    if change == 'bf16': model['embedding.weight'] = model['embedding.weight'].bfloat16()
    elif change == 'nonfinite': model['embedding.weight'][0, 0] = float('nan')
    elif change == 'wrong_shape': model['embedding.weight'] = torch.ones(2, 3)
    elif change == 'missing': model.pop('embedding.weight')
    else: model['extra.weight'] = torch.ones(1)
    with pytest.raises(ValueError, match='parameter'):
        validate_evaluation_contract(**packet)


def test_duplicated_optimizer_parameter_credit_rejected(packet):
    for record in [packet['state'], packet['training_report']]:
        record['resume_contract']['optimizer_parameter_names'].append('output.weight')
    packet['state']['optimizer_parameter_names'].append('output.weight')
    with pytest.raises(ValueError, match='parameter names'):
        validate_evaluation_contract(**packet)


def test_missing_training_manifest_anchor_rejected(packet):
    for record in [packet['state'], packet['training_report']]:
        record['resume_contract']['data'].pop('manifest_sha256')
    packet['state'].pop('data_manifest_sha256')
    with pytest.raises(ValueError, match='training manifest'):
        validate_evaluation_contract(**packet)


def test_missing_seed_rejected_instead_of_matching_missing_fields(packet):
    for record in [packet['state'], packet['training_report']]:
        record.pop('seed')
    with pytest.raises(ValueError, match='seed'):
        validate_evaluation_contract(**packet)
