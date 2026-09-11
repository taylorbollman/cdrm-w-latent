"""CPU checks for paired weighting, frozen bootstrap and artifact contracts."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_precision_pair_eval import analyze, paired_bootstrap, sha256, validate_pair_identity


def test_bootstrap_matches_frozen_evaluator_exactly_and_uses_token_weighting():
    from rt_precision_eval import paired_document_interval
    a = [{'text_sha256': str(index), 'targets': count, 'ce_sum': float(count)}
         for index, count in enumerate([0, 1, 10, 100])]
    b = copy.deepcopy(a)
    for row, change in zip(b, [0., .01, .1, -.1]):
        row['ce_sum'] += change
    expected = paired_document_interval(a, b)
    actual = paired_bootstrap(a, b)
    assert actual['candidate_minus_reference_nats_per_target'] == expected['candidate_minus_reference_nats_per_target']
    assert actual['one_sided_95_upper'] == expected['one_sided_95_upper']
    assert actual['two_sided_95_interval'] == expected['interval_95_percent']
    assert actual['excluded_zero_target_documents'] == 1
    assert actual['candidate_minus_reference_nats_per_target'] == pytest.approx(.01 / 111)
    assert actual['candidate_minus_reference_nats_per_target'] != pytest.approx(np.mean([.01, .01, -.001]))


def test_bootstrap_rejects_mispaired_documents_and_counts():
    a = [{'text_sha256': 'x', 'targets': 2, 'ce_sum': 4.},
         {'text_sha256': 'y', 'targets': 3, 'ce_sum': 6.}]
    with pytest.raises(ValueError, match='identities/order'):
        paired_bootstrap(a, list(reversed(a)))
    b = copy.deepcopy(a); b[1]['targets'] += 1
    with pytest.raises(ValueError, match='target counts'):
        paired_bootstrap(a, b)


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def artifacts(tmp_path):
    data = tmp_path / 'data'; data.mkdir()
    tokenizer = data / 'tokenizer'; tokenizer.mkdir()
    tokenizer_file = write(tokenizer / 'tokenizer.json', {'fixture': True})
    ids = (np.arange(1024).reshape(2, 512) % 128).astype(np.uint16)
    np.save(data / 'dev.npy', ids, allow_pickle=False)
    boundaries = [{'text_sha256': str(index) * 64, 'token_begin': begin, 'token_end': end}
                  for index, (begin, end) in enumerate([(0, 5), (5, 520), (520, 1024)])]
    boundary_path = data / 'dev-documents.jsonl'
    boundary_path.write_text(''.join(json.dumps(row) + '\n' for row in boundaries))
    role = {'role': 'dev', 'ids_path': 'dev.npy', 'ids_sha256': sha256(data / 'dev.npy'),
            'dtype': 'uint16', 'shape': [2, 512], 'documents': 3,
            'boundaries_path': boundary_path.name, 'boundaries_sha256': sha256(boundary_path)}
    manifest = write(data / 'manifest.json', {'schema': 'rt-precision-c4-data-v1', 'status': 'complete',
                     'mode': 'heldout', 'roles': {'dev': role},
                     'tokenizer': {'repo_id': 'fixture', 'revision': 'fixed',
                                   'files': {'tokenizer.json': {'sha256': sha256(tokenizer_file)}}}})
    protocol = write(tmp_path / 'protocol.json', {'schema': 'rt-precision-alignment-protocol-v1',
                     'evaluation': {'ce_margin_nats_per_supervised_token': .005,
                                    'common_evaluation': 'FP32 native forward, per supervised token CE',
                                    'uncertainty': 'paired bootstrap document aggregates, 95% upper bound, token-weighted, 10000, seed20260912'}})
    source_name = 'recurrent-transformer/olmo/model.py'
    source_content = b'test-only source snapshot\n'
    import hashlib
    source_digest = hashlib.sha256(source_content).hexdigest()
    sources = {source_name: source_digest}
    reports, training, dirs = {}, {}, {}
    for label, policy, shift in [('A', 'bf16_fp32_state', 0.), ('B', 'legacy', .001)]:
        directory = tmp_path / label; directory.mkdir(); dirs[label] = directory
        source = directory / 'source' / source_name; source.parent.mkdir(parents=True); source.write_bytes(source_content)
        losses = np.linspace(1, 3, 1022, dtype=np.float32).reshape(2, 511) + np.float32(shift)
        np.save(directory / 'token-ce.npy', losses, allow_pickle=False)
        expanded = np.zeros((2, 512), dtype=np.float64); expanded[:, 1:] = losses
        mask = np.ones((2, 512), dtype=np.int64); mask[:, 0] = 0
        docs = [{'text_sha256': row['text_sha256'],
                 'targets': int(mask.reshape(-1)[row['token_begin']:row['token_end']].sum()),
                 'ce_sum': float(expanded.reshape(-1)[row['token_begin']:row['token_end']].sum())}
                for row in boundaries]
        write(directory / 'documents.json', docs)
        ckpt = ('a' if label == 'A' else 'b') * 64
        train = {'schema': 'rt-precision-training-v1', 'status': 'complete', 'policy': policy,
                 'precision': 'bf16', 'seed': 20260910, 'completed_updates': 100, 'endpoint': 100,
                 'checkpoints': [{'sha256': ckpt, 'completed_updates': 100}],
                 'protocol_sha256': sha256(protocol), 'heldout_manifest_sha256': sha256(manifest),
                 'data': {'manifest_sha256': 'c' * 64, 'order': 'sequential'},
                 'initial_state_sha256': 'd' * 64, 'schedule': {'warmup': 5000},
                 'optimizer': {'lr': .001}, 'source_sha256': sources}
        train_path = write(directory / 'training-report.json', train); training[label] = train
        report = {'schema': 'rt-precision-evaluation-v1', 'status': 'complete',
                  'role': 'dev', 'precision': 'fp32', 'batch': 32, 'normalization': 'Per supervised token',
                  'cuda_graphs': False, 'data': {'manifest_sha256': sha256(manifest), **role},
                  'protocol_sha256': sha256(protocol), 'source_sha256': sources,
                  'checkpoint': {'sha256': ckpt}, 'checkpoint_update': 100,
                  'training_report': {'path': str(train_path), 'sha256': sha256(train_path)},
                  'model_config': {'n_layers': 12, 'd_model': 1024, 'precision': None, 'recurrent_precision_policy': policy},
                  'checkpoint_identity': {'status': 'verified', 'training_policy': policy,
                      'seed': 20260910, 'completed_updates': 100, 'checkpoint_sha256': ckpt,
                      'saved_parameters_cpu_fp32_finite': True, 'canonical_parameter_coverage': True,
                      'protocol_sha256': sha256(protocol), 'heldout_manifest_sha256': sha256(manifest),
                      'training_manifest_sha256': 'c' * 64, 'parameter_count': 216843264, 'parameter_tensors': 111,
                      'checked_source_sha256': sources},
                  'supervised_tokens': losses.size, 'documents': len(docs),
                  'ce_nats_per_supervised_token': float(losses.mean(dtype=np.float64)),
                  'token_ce_sha256': sha256(directory / 'token-ce.npy'), 'documents_sha256': sha256(directory / 'documents.json')}
        write(directory / 'report.json', report); reports[label] = report
    return dirs, reports, training, manifest, protocol


def test_complete_analysis_reconciles_documents_crossing_packed_row_boundaries(tmp_path):
    dirs, _, _, manifest, protocol = artifacts(tmp_path)
    result = analyze(dirs['A'], dirs['B'], manifest, protocol)
    assert result['status'] == 'complete' and result['bootstrap']['supervised_tokens'] == 1022
    assert result['bootstrap']['resampled_documents'] == 3
    assert result['identity']['common_fp32_primary']
    assert result['primary_margin_assessment'] == 'within_margin_on_this_pilot_slice'
    assert result['bootstrap']['candidate_minus_reference_nats_per_target'] == pytest.approx(.001, abs=1e-7)


@pytest.mark.parametrize('change', ['seed', 'update', 'role', 'precision', 'initialization', 'source'])
def test_pair_identity_rejects_unpaired_conditions(tmp_path, change):
    _, reports, training, _, _ = artifacts(tmp_path)
    if change == 'seed':
        reports['B']['checkpoint_identity']['seed'] += 1; training['B']['seed'] += 1
    elif change == 'update':
        reports['B']['checkpoint_update'] = 500
    elif change == 'role':
        reports['B']['role'] = 'confirmation'
    elif change == 'precision':
        reports['B']['precision'] = 'native'
    elif change == 'initialization':
        training['B']['initial_state_sha256'] = 'e' * 64
    elif change == 'source':
        reports['B']['source_sha256'] = {'recurrent-transformer/olmo/model.py': 'f' * 64}
    with pytest.raises(ValueError):
        validate_pair_identity(reports['A'], reports['B'], training['A'], training['B'])


def test_native_pair_is_explicitly_secondary(tmp_path):
    _, reports, training, _, _ = artifacts(tmp_path)
    for report in reports.values():
        report['precision'] = 'native'
    identity = validate_pair_identity(reports['A'], reports['B'], training['A'], training['B'])
    assert not identity['common_fp32_primary'] and 'Secondary native' in identity['interpretation']


@pytest.mark.parametrize('change', ['token_hash', 'document_pairing', 'document_sum', 'tokenizer_hash'])
def test_artifact_integrity_and_document_token_join_are_required(tmp_path, change):
    dirs, reports, _, manifest, protocol = artifacts(tmp_path)
    if change == 'token_hash':
        path = dirs['B'] / 'token-ce.npy'; values = np.load(path); values[0, 0] += 1; np.save(path, values)
    elif change == 'tokenizer_hash':
        (manifest.parent / 'tokenizer/tokenizer.json').write_text('changed')
    else:
        path = dirs['B'] / 'documents.json'; docs = json.loads(path.read_text())
        if change == 'document_pairing':
            docs = list(reversed(docs))
        else:
            docs[0]['ce_sum'] += .1
        write(path, docs); reports['B']['documents_sha256'] = sha256(path)
        write(dirs['B'] / 'report.json', reports['B'])
    with pytest.raises(ValueError):
        analyze(dirs['A'], dirs['B'], manifest, protocol)
