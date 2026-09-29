"""Endpoint-only CPU geometry and strict authority checks; no model execution."""
import copy
from dataclasses import asdict
import json
import math

import pytest
import torch

from scripts import olmo_fusion_startup_nfr_compare as probe


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def payload():
    values = {'backbone.weight': torch.tensor([10000., -2.]),
              'backbone.fusion.weight': torch.tensor([3., 4.]),
              'predictor.weight': torch.tensor([.5, -.25])}
    layout = [{'name': name, 'aliases': [name], 'shape': list(value.shape),
               'dtype': str(value.dtype), 'requires_grad': True} for name, value in values.items()]
    layout[0]['aliases'].append('backbone.readout_weight')
    values['backbone.readout_weight'] = values['backbone.weight']
    names = [row['name'] for row in layout]
    return {'schema': probe.CHECKPOINT_SCHEMA, 'model_type': 'cdrm.pretrained.fbt_training.FBTNextLatLM',
        'model': values, 'parameter_layout': layout, 'module_training': {'': True},
        'optimizer_ownership': [names], 'optimizer': {'param_groups': [{'params': [0, 1, 2], 'param_names': names}],
            'state': {i: {'step': torch.tensor(4.), 'exp_avg': torch.tensor([.25, .5]),
                          'exp_avg_sq': torch.tensor([.0625, .25])} for i in range(3)}},
        'counters': {'optimizer_updates': 4}, 'scheduler': {'last_epoch': 4},
        'rng': {'cpu': torch.get_rng_state()}, 'configuration': {'plain_list': [1, 2]},
        'source_fingerprint': {'checkpoint_sha256': 'a'*64}}


def receipt(path='endpoint.pt', *, trajectory=None):
    row = {'path': str(path), 'optimizer_updates': 20, 'size_bytes': 100, 'sha256': 'a'*64,
           'gcs': {'generation': 123, 'sha256': 'a'*64, 'size_bytes': 100,
                   'verification': {k: True for k in ('download_sha256', 'server_md5', 'server_size', 'sha256_metadata')}}}
    if trajectory is not None: row['trajectory'] = trajectory
    return row


def evaluation(update):
    counts = {'ce': 2, 'latent': 2, 'kl': 1}
    passes = {}
    for i in range(4):
        means = {'ce': float(2+i), 'latent': float(.5+i/4), 'kl': float(1+i/2)}
        passes['pass_'+str(i)] = {'counts': counts, 'loss_means': means,
                                 'loss_sums': {k: means[k]*counts[k] for k in counts}}
    means = {k: sum(passes['pass_'+str(i)]['loss_means'][k]*([.5, 1/6, 1/6, 1/6] if k == 'ce' else [.25]*4)[i]
                    for i in range(4)) for k in counts}
    return {'update': update, 'precision': 'fp32_math_eager', 'checks': {'unchanged': True},
            'loss_means': means, 'loss_sums': {k: means[k]*counts[k] for k in counts}, 'counts': counts,
            'combined_objective': sum(means.values()), 'per_pass': passes,
            'per_pass_observer': {'counts_exact': True, 'method_restored': True}}


def reports():
    # Small scalar contracts exercise the actual fixed 4 -> 20 clocks without
    # allocating tensors, checkpoints or rerunning the training implementation.
    metadata = [{'counts': {'ce': 2, 'latent': 2, 'kl': 1}, 'microbatches': 1,
                 'documents': 1, 'input_tokens': 3} for _ in range(20)]
    selections = [{'source_training_index': i, 'next_cursor': {'next_update': i+1}, **metadata[i-144]}
                  for i in range(144, 164)]
    origin_boundary = {'model': 'same-model', 'optimizer': 'same-adam', 'scheduler': {'last_epoch': 4, 'prefix': 4}}
    sources = {'original.py': 'b'*64}
    origin = {'schema': probe.ORIGIN_SCHEMA, 'status': 'passed_bounded_functionality', 'passed': True,
        'optimizer_calls': 8, 'integrity': {'checked': True}, 'sources': sources, 'recipe': {'arm': 'NFR'},
        'runtime': {'gpu': 'same'}, 'determinism': {'enabled': True},
        'data_manifest_sha256': 'c'*64, 'fixture_sha256': 'd'*64,
        'training_metadata': metadata[:4], 'training_data_selections': selections[:4],
        'checkpoints': [{**receipt(trajectory=probe.BF16), 'optimizer_updates': 4}],
        'final_boundary_pins': {probe.BF16: origin_boundary}, 'evaluations': {'4': {probe.BF16: evaluation(4)}}}
    result = []
    for precision in ('fp32', 'bf16_mixed'):
        configuration = {'origin_checkpoint_sha256': 'a'*64, 'origin_trajectory': probe.BF16,
            'origin_boundary': origin_boundary, 'sources': sources, 'precision': precision,
            'recipe': origin['recipe'], 'runtime': origin['runtime'], 'determinism': origin['determinism'],
            'data_manifest_sha256': 'c'*64, 'fixture_sha256': 'd'*64,
            'schedule_fork': {'new_scheduler': {'last_epoch': 4, 'prefix': 20}, 'checks': {'prefix_exact': True}},
            'training_metadata': metadata, 'training_data_selections': selections}
        updates = []
        for step in range(5, 21):
            updates.append({'update': step, 'source_training_index': 144+step-1, 'path': precision,
                'input_pins': {'tokens': str(step)}, 'counters': asdict(probe.expected_counters(metadata, step)),
                'lr_used': [2e-5], 'lr_next': [2e-5], 'metrics': {**metadata[step-1],
                    'objective': 3., 'loss_sums': {'ce': 2., 'latent': 2., 'kl': 1.}},
                'gradient_norm_before_clip': 2., 'clip_scale': .5,
                'raw_gradient_norms': {'all': 2.}, 'master_parameter_norms': {'all': 100.}})
        result.append({'schema': probe.CONTINUATION_SCHEMA, 'status': 'completed_segment', 'passed': True,
            'integrity': {'checked': True}, 'plan': probe.PLAN, 'precision': precision, 'starting_update': 4,
            'physical_optimizer_updates': 16, 'final_counters': asdict(probe.expected_counters(metadata, 20)),
            'configuration': configuration, 'origin_boundary': origin_boundary, 'sources': sources,
            'runtime': origin['runtime'], 'determinism': origin['determinism'],
            'source_fingerprint': {'sources': sources, 'origin_checkpoint_sha256': 'a'*64,
                'data_manifest_sha256': 'c'*64, 'fixture_sha256': 'd'*64, 'nfr_report_sha256': 'e'*64},
            'actual_execution_mode': {'beta': 1., 'document_policy': 'isolated-v1', 'enabled': True,
                'feedback_jitter': .02, 'first_pass_policy': 'configured-rt-v1', 'num_passes': 4,
                'rt_mode': {'alpha': 1., 'selected_layers': [0, 15]}},
            'starting_boundary': {**origin_boundary, 'scheduler': configuration['schedule_fork']['new_scheduler']},
            'updates': updates, 'final_cursor': selections[-1]['next_cursor'],
            'evaluations': [evaluation(i) for i in (4, 12, 20)], 'checkpoints': [receipt()]})
    return copy.deepcopy((origin, *result))


def test_endpoint_and_actual_delta_geometry_match_literal_concatenated_oracle():
    origin = probe.parameter_and_moment_views(payload())
    fp, bf = copy.deepcopy(origin), copy.deepcopy(origin)
    for kind in origin:
        for index, name in enumerate(origin[kind]):
            fp[kind][name] += torch.tensor([.03125*(index+1), -.0625])
            bf[kind][name] += torch.tensor([.0625*(index+1), -.03125])
    observed = []
    actual = probe.endpoint_geometry(origin, fp, bf, observe=lambda name, _: observed.append(name))
    assert len(observed) == 6
    for kind in origin:
        for delta in (False, True):
            label = kind+('_change_from_common_origin' if delta else '')
            for group in probe.COMPONENTS:
                names = [n for n in origin[kind] if group == 'all' or probe.component(n) == group]
                a = torch.cat([(bf[kind][n].double()-(origin[kind][n].double() if delta else 0)).flatten() for n in names])
                b = torch.cat([(fp[kind][n].double()-(origin[kind][n].double() if delta else 0)).flatten() for n in names])
                row = actual[label][group]
                assert row['parameter_tensors'] == len(names)
                assert row['difference_norm'] == pytest.approx(float((a-b).norm()), abs=1e-12)
                assert row['relative_l2'] == pytest.approx(float((a-b).norm()/b.norm()), abs=1e-12)
                assert row['cosine'] == pytest.approx(float(a.dot(b)/(a.norm()*b.norm())), abs=1e-12)
    # Whole weights hide large relative update differences; never substitute them.
    assert actual['model_change_from_common_origin']['backbone']['relative_l2'] > actual['model']['backbone']['relative_l2']*100
    norms = probe.origin_norms(origin)
    assert norms['model']['all'] == pytest.approx(float(torch.cat(list(origin['model'].values())).double().norm()))


def test_mmap_loader_checks_actual_saved_bytes_boundary_and_json_contract(tmp_path):
    obj = payload(); path = tmp_path/'checkpoint.pt'; torch.save(obj, path)
    record = {**receipt(path), 'optimizer_updates': 4, 'sha256': probe.sha256_file(path), 'size_bytes': path.stat().st_size}
    boundary = probe.payload_boundary(obj)
    loaded, views, before = probe.load_checkpoint(path, record, configuration=json.loads(json.dumps(obj['configuration'])),
        fingerprint=obj['source_fingerprint'], expected_boundary=boundary)
    assert before == probe.signature(path) and probe.payload_boundary(loaded) == boundary
    assert len(views['model']) == 3 and len(loaded['model']) == 4
    with pytest.raises(ValueError, match='contract'):
        probe.load_checkpoint(path, record, configuration={'plain_list': [2, 1]}, fingerprint=obj['source_fingerprint'], expected_boundary=boundary)
    bad = copy.deepcopy(boundary); bad['scheduler']['last_epoch'] += 1
    with pytest.raises(ValueError, match='boundary'):
        probe.load_checkpoint(path, record, configuration=obj['configuration'], fingerprint=obj['source_fingerprint'], expected_boundary=bad)
    with path.open('ab') as handle: handle.write(b'changed')
    with pytest.raises(ValueError, match='bytes'):
        probe.load_checkpoint(path, record, configuration=obj['configuration'], fingerprint=obj['source_fingerprint'], expected_boundary=boundary)


@pytest.mark.parametrize('bad', ['tied_alias', 'frozen', 'foreign_owner', 'duplicate_owner', 'negative_variance', 'step', 'nan'])
def test_tensor_inventory_rejects_invalid_ownership_and_adam_state(bad):
    obj = payload()
    if bad == 'tied_alias': obj['model']['backbone.readout_weight'] = torch.ones(2)
    elif bad == 'frozen': obj['parameter_layout'][0]['requires_grad'] = False
    elif bad == 'foreign_owner': obj['optimizer_ownership'][0][0] = 'foreign'
    elif bad == 'duplicate_owner': obj['optimizer']['param_groups'][0]['params'][1] = 0
    elif bad == 'negative_variance': obj['optimizer']['state'][0]['exp_avg_sq'][0] = -1
    elif bad == 'step': obj['optimizer']['state'][0]['step'] += 1
    else: obj['model']['predictor.weight'][0] = float('nan')
    with pytest.raises((ValueError, KeyError)): probe.parameter_and_moment_views(obj)


def test_complete_report_controls_and_per_pass_plot_inputs(tmp_path):
    origin, fp, bf = reports()
    assert all(probe.report_controls(origin, fp, bf, origin_sha256='e'*64).values())
    rows = probe.scalar_rows(fp, bf)
    assert len(rows['training']) == 32
    assert rows['training'][0]['means'] == {'ce': 1., 'latent': 1., 'kl': 1.}
    assert rows['common_fp32_dev']['bf16_mixed'][0]['per_pass']['pass_3']['loss_means']['ce'] == 5.
    pins = probe.plot_rows(rows, tmp_path)
    assert len(pins) == 4 and all(probe.sha256_file(tmp_path/name) == digest for name, digest in pins.items())


@pytest.mark.parametrize('bad', ['unfinished', 'wrong_origin', 'data', 'schedule', 'mode', 'dev_count', 'dev_pass_weight', 'report_pin', 'cloud_bytes'])
def test_invalid_completed_pair_rejected(bad):
    origin, fp, bf = reports()
    if bad == 'unfinished': bf['status'] = 'running'
    elif bad == 'wrong_origin': bf['configuration']['origin_checkpoint_sha256'] = 'f'*64
    elif bad == 'data': bf['updates'][0]['input_pins']['tokens'] = 'wrong'
    elif bad == 'schedule': bf['starting_boundary']['scheduler']['last_epoch'] += 1
    elif bad == 'mode': bf['actual_execution_mode']['rt_mode']['alpha'] = .25
    elif bad == 'dev_count': bf['evaluations'][-1]['per_pass']['pass_3']['counts'] = {'ce': 1, 'latent': 2, 'kl': 1}
    elif bad == 'dev_pass_weight': bf['evaluations'][-1]['per_pass']['pass_0']['loss_means']['ce'] += 1
    elif bad == 'report_pin': bf['source_fingerprint']['nfr_report_sha256'] = 'f'*64
    else: bf['checkpoints'][0]['gcs']['verification']['download_sha256'] = False
    with pytest.raises(ValueError): probe.report_controls(origin, fp, bf, origin_sha256='e'*64)


def test_report_hash_and_mutation_during_read_fail_closed(tmp_path, monkeypatch):
    path = tmp_path/'report.json'; path.write_text('{"status":"completed"}')
    digest = probe.sha256_file(path)
    assert probe.read_report(path, digest) == {'status': 'completed'}
    with pytest.raises(ValueError, match='SHA256'): probe.read_report(path, 'f'*64)
    original = probe.sha256_file
    def mutate(target):
        result = original(target)
        path.write_text('{"status":"mutated"}')
        return result
    monkeypatch.setattr(probe, 'sha256_file', mutate)
    with pytest.raises(ValueError, match='changed while reading'): probe.read_report(path, digest)
