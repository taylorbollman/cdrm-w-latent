"""CPU contracts for fresh-process capture/Adam continuation evidence."""
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_precision_resume_probe import (BATCH, REPORT_SCHEMA, STATE_SCHEMA, exact_update_comparison,
                                       rng_digest, validate_phase_arguments, validate_proof_state,
                                       validate_reference_report)


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('CPU contracts must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', fail)


@pytest.mark.parametrize('phase,proof,report,valid', [
    ('reference', None, None, True), ('resumed', 'proof.pt', 'report.json', True),
    ('reference', 'proof.pt', None, False), ('reference', None, 'report.json', False),
    ('resumed', None, None, False), ('resumed', 'proof.pt', None, False),
    ('resumed', None, 'report.json', False), ('unknown', None, None, False)])
def test_phase_contract_prevents_incomplete_or_reused_reference_inputs(phase, proof, report, valid):
    if valid:
        validate_phase_arguments(phase, proof, report)
    else:
        with pytest.raises(ValueError):
            validate_phase_arguments(phase, proof, report)


def rng_fixture():
    return {'python': random.Random(21).getstate(),
            'numpy': np.random.RandomState(22).get_state(),
            'torch_cpu': torch.Generator().manual_seed(23).get_state(),
            'torch_cuda': [torch.tensor([1, 2, 3], dtype=torch.uint8)]}


def proof_fixture():
    training = {'model_config': {'recurrent_precision_policy': 'legacy'},
                'optimizer_parameter_names': ['weight'], 'seed': 20260910,
                'data': {'shape': [256000, 512], 'ids_sha256': 'a' * 64},
                'runtime': {'cache': '/retained/cache'}, 'optimizer': {'betas': (.9, .95)}}
    contract = {'training_resume_contract': training, 'source_sha256': {'source.py': 'b' * 64},
                'authority': {'protocol_sha256': 'c' * 64}}
    state = {'schema': STATE_SCHEMA, 'proof_contract': copy.deepcopy(contract),
             'completed_updates': 101, 'next_data_row': 101 * BATCH, 'rng': rng_fixture(),
             **{key: copy.deepcopy(training[key]) for key in ('model_config', 'optimizer_parameter_names', 'seed')}}
    state['rng_sha256'] = rng_digest(state['rng'])
    return state, contract


def update_fixture():
    return {'update': 102, 'row_begin': 101 * BATCH, 'row_end': 102 * BATCH,
            'learning_rate': .00011836, 'loss': 7., 'loss_sha256': 'a' * 64,
            'gradient_norm_before_clip': 4., 'gradient_norm_sha256': 'b' * 64,
            'raw_gradients_sha256': 'c' * 64, 'post_model_sha256': 'd' * 64,
            'post_optimizer_sha256': 'e' * 64, 'post_rng_sha256': 'f' * 64}


def test_rng_digest_is_storage_independent_and_covers_every_rng_family():
    state = rng_fixture()
    assert rng_digest(state) == rng_digest(copy.deepcopy(state))
    for key in ('python', 'numpy', 'torch_cpu', 'torch_cuda'):
        altered = copy.deepcopy(state)
        if key == 'python':
            altered[key] = random.Random(99).getstate()
        elif key == 'numpy':
            altered[key][1][0] += 1
        elif key == 'torch_cpu':
            altered[key][0] ^= 1
        else:
            altered[key][0][0] ^= 1
        assert rng_digest(altered) != rng_digest(state), key


def test_proof_accepts_only_step101_with_identical_data_source_protocol_runtime():
    state, contract = proof_fixture()
    validate_proof_state(state, contract)
    for field in ('completed_updates', 'next_data_row', 'seed'):
        altered = copy.deepcopy(state)
        altered[field] += 1
        with pytest.raises(ValueError):
            validate_proof_state(altered, contract)
    for field in ('data', 'runtime'):
        altered = copy.deepcopy(contract)
        altered['training_resume_contract'][field] = {'changed': True}
        with pytest.raises(ValueError):
            validate_proof_state(state, altered)
    for field in ('authority', 'source_sha256'):
        altered = copy.deepcopy(contract)
        altered[field] = {'changed': True}
        with pytest.raises(ValueError):
            validate_proof_state(state, altered)
    state['rng']['torch_cuda'][0][0] ^= 1
    with pytest.raises(ValueError, match='RNG'):
        validate_proof_state(state, contract)


def test_reference_report_anchors_saved_state_and_requires_a_fresh_process():
    _, contract = proof_fixture()
    process = {'pid': 8, 'hostname': 'container-a', 'start_ticks': '1000'}
    report = {'schema': REPORT_SCHEMA, 'status': 'complete', 'phase': 'reference',
              'proof_contract': json.loads(json.dumps(contract)),
              'proof_checkpoint': {'sha256': 'a' * 64}, 'process': process,
              'expected_update102': update_fixture()}
    kwargs = {'proof_sha256': 'a' * 64, 'current_contract': contract,
              'current_process': dict(process, hostname='container-b')}
    assert validate_reference_report(report, **kwargs)['update'] == 102
    with pytest.raises(ValueError, match='fresh process'):
        validate_reference_report(report, **dict(kwargs, current_process=process))
    with pytest.raises(ValueError, match='retained'):
        validate_reference_report(report, **dict(kwargs, proof_sha256='b' * 64))
    altered = copy.deepcopy(report)
    altered['expected_update102']['row_begin'] += BATCH
    with pytest.raises(ValueError, match='expected-update'):
        validate_reference_report(altered, **kwargs)


@pytest.mark.parametrize('field', ['loss', 'gradient_norm_before_clip', 'raw_gradients_sha256',
                                  'post_model_sha256', 'post_optimizer_sha256', 'post_rng_sha256'])
def test_resume_proof_never_relaxes_scalar_or_tensor_identity(field):
    reference = update_fixture()
    actual = copy.deepcopy(reference)
    actual[field] = np.nextafter(actual[field], np.inf) if isinstance(actual[field], float) else '0' * 64
    result = exact_update_comparison(reference, actual)
    assert not result['pass'] and result['mismatched_fields'] == [field]
    assert exact_update_comparison(reference, copy.deepcopy(reference))['pass']


def test_missing_or_nonfinite_expected_evidence_cannot_pass():
    reference = update_fixture()
    actual = copy.deepcopy(reference)
    actual.pop('post_optimizer_sha256')
    with pytest.raises(ValueError):
        exact_update_comparison(reference, actual)
    actual = copy.deepcopy(reference)
    actual['loss'] = float('nan')
    with pytest.raises(FloatingPointError):
        exact_update_comparison(reference, actual)


@pytest.mark.skipif(not os.environ.get('RT_RESUME_INTEGRATION_DIR'),
                    reason='Opt in to the retained real A100 checkpoint metadata integration')
def test_actual_a100_resume_contract_in_cpu_container(monkeypatch):
    """Use real retained tensors/metadata, while only allocating a meta model.

    Saved GPU identity is supplied to runtime_identity; software settings are
    checked against the actual CPU container build. This makes no GPU claim.
    """
    from dataclasses import replace
    from olmo.model import OLMo
    from rt_cuda_graph_validate import validation_config
    from rt_precision_eval_contract import validate_evaluation_contract
    from rt_precision_resume_probe import runtime_identity, training_resume_contract
    from rt_precision_train import (load_training_data, validate_model_checkpoint,
                                    validate_optimizer_checkpoint, validate_resume_metadata)
    from stage_a_common import seed_all
    from stage_b_train import file_digest

    assert Path('/.dockerenv').is_file()
    assert Path.cwd() == Path('/workspace/cdrm-w-latent')
    lineage = Path(os.environ['RT_RESUME_INTEGRATION_DIR'])
    report_path = lineage / 'train/A-seed0-100/report.json'
    report = json.loads(report_path.read_text())
    checkpoint = next(row for row in report['checkpoints'] if row['completed_updates'] == 100)
    path = Path(checkpoint['path'])
    assert file_digest(path) == checkpoint['sha256']
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR', report['runtime']['cache'])
    seed_all(report['seed'], deterministic=True)
    assert not torch.cuda.is_initialized()
    runtime = runtime_identity(report['hardware'])
    assert runtime == report['runtime']
    # Check the concrete disputed field rather than loosening runtime equality.
    assert runtime['driver_version'] == state['runtime']['driver_version']
    config = validation_config('full', report['policy'])
    model = OLMo(replace(config, init_device='meta'), init_params=False)
    names = list(dict(model.named_parameters()))
    ids, data, _ = load_training_data(lineage / 'data/train/manifest.json', 102)
    source_hashes = {name: file_digest(Path(name)) for name in report['source_sha256']}
    assert source_hashes == report['source_sha256']
    protocol_digest = file_digest(lineage / 'reference/protocol.json')
    contract = training_resume_contract(config, names, seed=report['seed'], data=data,
                                         sources=source_hashes, runtime=runtime,
                                         protocol_sha256=protocol_digest)
    assert contract == state['resume_contract']
    assert json.loads(json.dumps(contract)) == report['resume_contract']
    assert validate_resume_metadata(state, contract, 500) == 100
    validate_model_checkpoint(state, model.state_dict())
    validate_optimizer_checkpoint(state, dict(model.named_parameters()))
    identity = validate_evaluation_contract(
        report, state, checkpoint_sha256=checkpoint['sha256'],
        heldout_manifest_sha256=file_digest(lineage / 'data/heldout/manifest.json'),
        current_source_sha256=source_hashes, protocol_sha256=protocol_digest,
        expected_parameters=dict(model.named_parameters()))
    assert identity['parameter_tensors'] == 111 and identity['parameter_count'] == 216843264
    assert not torch.cuda.is_initialized()
    print(json.dumps({'real_a100_cpu_contract': 'passed', 'checkpoint_sha256': checkpoint['sha256'],
                      'parameter_tensors': 111, 'runtime_fields': list(runtime),
                      'driver_version': runtime['driver_version'], 'cuda_initialized': False}))
