"""Posthoc interpretation guards; CPU only, no live reports or model imports."""
import copy
import json

import pytest

from scripts import olmo_kl_continuation_summary as summary


def loss(ce=3., latent=.2, kl=.4):
    means = {'ce': ce, 'latent': latent, 'kl': kl}
    counts = {'ce': 65472, 'latent': 65338, 'kl': 65140}
    return {'means': means, 'counts': counts,
            'sums': {term: means[term]*counts[term] for term in means}}


def evaluation(update, weight=1., *, ce_offset=0.):
    passes = [{'index': i, **loss(3.+i+ce_offset)} for i in range(4)]
    aggregate = loss(4.+ce_offset)
    result = {'input_tokens': 65536, 'policy': 'common_fp32_no_jitter_v1',
        'enabled': dict.fromkeys(summary.TERMS, True),
        'weights': {'ce': 1., 'latent': 1., 'kl': weight},
        'term_pass_coefficients': summary.COEFFICIENTS, 'passes': passes,
        'aggregate': aggregate, 'objective': aggregate['means']['ce']+.2+weight*.4,
        'reconstructed_aggregate_sums': copy.deepcopy(aggregate['sums'])}
    return {'after_update': update, 'status': 'completed', 'total_seconds': 5.,
        'training_boundary_exact_by_rank': [True, True], 'panels': {'dev-main': {
            'result': result, 'membership_sha256': 'm'*64, 'index_manifest_sha256': 'i'*64,
            'by_rank': [{'preservation': {'integrity_passed': True, 'restored': True,
                        'checks': {'rng_restored': True}}} for _ in range(2)]}}}


def fixture():
    schedule = {'planned_updates': 128, 'warmup_tokens': 52428800, 'start_fraction': .1}
    ownership = {'parameter_layout': [{'name': 'b', 'shape': [2, 2]}, {'name': 'p', 'shape': [2]}],
        'optimizer_ownership': [['b'], ['p']], 'resident_parameters': 6, 'trainable_parameters': 6,
        'mode': {'rt_mode': {'selected_layers': []}},
        'component_parameters': {'backbone': 4, 'fusion': 0, 'predictor': 2}}
    parent_configuration = {'execution_identity': {'sha256': 'identity'}, 'schedule': schedule,
                            'recipe': {'rt_layers': [0, 15]}}
    boundary = [{'rank': n, 'model': 'saved', 'adam': 'populated', 'rng': 'saved'} for n in range(2)]
    parent = {'schema': 'olmo-pilot-async-execute-report-v1', 'status': 'stopped_at_boundary',
        'arm': 'NF', 'scale': 'native', 'final_counters': {'optimizer_updates': 32},
        'configuration': parent_configuration, 'declaration_sha256': 'decl', 'resolved_sha256': 'resolved',
        'local_checkpoints': [{'receipt': {'manifest_sha256': 'manifest'}, 'boundary_by_rank': boundary}],
        'evaluations': [evaluation(32)]}
    memory = {'allocated_gib': 20., 'reserved_gib': 50., 'peak_allocated_gib': 30.,
              'peak_reserved_gib': 52., 'sampled_free_gib': 25., 'total_gib': 80.}
    def branch(weight):
        recipe = {'auxiliary': {'kl': weight, 'latent': 1.}, 'optimizer_state': 'inherited_exact_parent_checkpoint',
            'fbt_passes': 4, 'rt_layers': [0, 15], 'sequence_length': 1024, 'effective_valid_tokens': 524288,
            'feedback_jitter': .02, 'plateau_lr': .0002}
        result = {'schema': 'olmo-kl-continuation-report-v1', 'status': 'stopped_at_boundary',
            'segment_completed': True, 'plan_completed': False, 'scale': 'native', 'arm': 'NF',
            'segment_stop_after': 64, 'checkpoint_mode': 'async', 'observation_mode': 'lean',
            'sources': {str(n): str(n) for n in range(210)}, 'declaration_sha256': 'decl', 'resolved_sha256': 'resolved',
            'parent_report_sha256': 'parent-pin', 'original_configuration': copy.deepcopy(parent_configuration),
            'configuration': {'recipe': recipe, 'schedule': copy.deepcopy(schedule),
                'world_size': 2, 'training': {'precision': 'bf16_mixed'}, 'parameters': ownership,
                'execution_identity': {'sha256': str(weight)}},
            'branch': {'kl_weight': weight, 'parent_update': 32, 'review_stop': 64,
                'parent_report_sha256': 'parent-pin', 'parent_identity_sha256': 'identity',
                'parent_manifest_sha256': 'manifest', 'recipe_as_declared': recipe},
            'objective_transition': {'before_boundary_by_rank': copy.deepcopy(boundary),
                                     'after_boundary_by_rank': copy.deepcopy(boundary)},
            'origin_boundary_by_rank': copy.deepcopy(boundary), 'preparation_boundary_exact': [True, True],
            'adam_resident_before_ddp': True, 'last_verified_cloud_update': 64,
            'loop': {'start_update': 32, 'completed_update': 64, 'last_saved_update': 64,
                'last_retained_update': 64, 'checkpoint_pending': False, 'checkpoint_wait_seconds': 8.},
            'updates': {}, 'observations': {}, 'update_wall_seconds_by_rank': {},
            'memory_after_capture': [copy.deepcopy(memory), copy.deepcopy(memory)],
            'evaluations': [evaluation(n, weight, ce_offset= -.1 if weight == .1 and n > 32 else 0.) for n in (32, 48, 64)],
            'elapsed_seconds': 5000., 'timing': {'preparation_seconds_by_rank': [10., 11.],
                'checkpoint_write_seconds_by_rank': {'64': [4., 5.]}}}
        for n in summary.UPDATES:
            raw = loss(4.)
            metrics = {'input_tokens': 524288, 'world_size': 2, 'local_microbatches': 22,
                'microbatches': 44, 'counters': {'optimizer_updates': n, 'input_tokens': n*524288},
                'loss_sums': raw['sums'], 'counts': raw['counts'], 'objective': 4.+.2+weight*.4,
                'gradient_norm_before_clip': 10., 'lr_used': [.0002*(.1+.9*(n-1)/100)],
                'lr_next': [.0002*(.1+.9*n/100)]}
            result['updates'][str(n)] = [{'metrics': copy.deepcopy(metrics), 'loss_means': raw['means'],
                'clipping': {'configured_limit': 1., 'norm_exceeds_limit': True, 'coefficient_estimate': 1/10.000001},
                'cursor': {'physical_batch_per_rank': 12, 'rank': r, 'next_update': n}} for r in range(2)]
            result['observations'][str(n)] = {'rank_data': [
                {'valid_tokens': 262144, 'ce_targets': 32736, 'latent_pairs': 32669, 'kl_triples': 32570} for _ in range(2)],
                'timing_by_rank': [{'backward': 20., 'optimizer_and_cursor': 1., 'materialization_host': 3.},
                                   {'backward': 21., 'optimizer_and_cursor': 1., 'materialization_host': 2.}],
                'memory_by_rank': [copy.deepcopy(memory), copy.deepcopy(memory)]}
            result['update_wall_seconds_by_rank'][str(n)] = [30., 31.]
        result['final_counters'] = metrics['counters']
        return result
    return branch(1.), branch(.1), parent


def reduce(reports):
    return summary.summarize(*reports, parent_sha='parent-pin')


def test_raw_terms_and_same_origin_not_weighted_objective():
    reports = fixture(); result = reduce(reports)
    left, right = (result['arms'][name] for name in ('KL1', 'KL0.1'))
    assert left['development'][0]['passes'] == right['development'][0]['passes']
    assert left['development'][0]['weighted_objective_branch_specific'] != right['development'][0]['weighted_objective_branch_specific']
    assert result['comparison'][0]['raw_mean_deltas_by_pass'][0]['ce'] == 0
    assert result['comparison'][1]['raw_mean_deltas_by_pass'][0]['ce'] == pytest.approx(-.1)
    assert left['parameters']['optimizer_owned'] == 6
    assert left['clipping']['clipped_updates'] == 32
    assert 'objective' not in result['comparison'][1]


def test_rank_critical_path_not_rank_sum_and_memory_not_gpu_sum():
    result = reduce(fixture())['arms']['KL1']
    assert result['rates']['all33_64']['compute_regions']['seconds'] == 32*22
    assert result['rates']['all33_64']['compute_plus_materialization']['seconds'] == 32*24
    assert result['rates']['all33_64']['complete_update_callback']['seconds'] == 32*31
    assert result['memory_sample_summary_gib']['reserved_gib'] == 50.
    assert result['checkpoint_seconds']['checkpoint_write_seconds_by_rank'] == 5.


@pytest.mark.parametrize('mutation,match', [
    (lambda c, r, p: r.update(status='running'), 'completed native'),
    (lambda c, r, p: r['branch'].update(kl_weight=1.), 'KL branch'),
    (lambda c, r, p: r['updates'].pop('33'), 'complete 33'),
    (lambda c, r, p: r['origin_boundary_by_rank'][0].update(adam='fresh'), 'saved parent state'),
    (lambda c, r, p: r['evaluations'][1]['panels']['dev-main']['result']['passes'][0]['counts'].update(ce=0), 'denominator'),
    (lambda c, r, p: r['evaluations'][1]['panels']['dev-main']['result']['passes'][0]['means'].update(ce=100.), 'numerator/denominator'),
    (lambda c, r, p: r['evaluations'][1]['panels']['dev-main'].update(membership_sha256='different'), 'panel differs'),
    (lambda c, r, p: r['evaluations'][0]['panels']['dev-main']['result']['passes'][0]['sums'].update(ce=10.), 'numerator/denominator'),
    (lambda c, r, p: r['loop'].update(checkpoint_pending=True), 'not durable'),
    (lambda c, r, p: r['configuration']['schedule'].update(warmup_tokens=100), 'schedule'),
    (lambda c, r, p: r['configuration']['parameters']['mode']['rt_mode'].update(selected_layers=[0, 15]), 'shape/objective'),
    (lambda c, r, p: r['sources'].update(extra='new'), 'runtime inventories'),
    (lambda c, r, p: r['updates']['33'][1]['metrics'].update(gradient_norm_before_clip=1.), 'Replica metrics'),
])
def test_reject_misleading_or_incomplete_comparisons(mutation, match):
    reports = fixture(); mutation(*reports)
    with pytest.raises(ValueError, match=match): reduce(reports)


def test_cli_pins_snapshots_and_artifacts(tmp_path):
    reports = fixture(); args = []
    parent_data = json.dumps(reports[2]).encode(); parent_sha = summary.sha(parent_data)
    for report in reports[:2]:
        report['parent_report_sha256'] = report['branch']['parent_report_sha256'] = parent_sha
    for label, report in zip(('control', 'reduced', 'parent'), reports):
        path = tmp_path/(label+'.json'); data = json.dumps(report).encode(); path.write_bytes(data)
        args += ['--'+label, str(path), summary.sha(data)]
    output = tmp_path/'summary'
    summary.main(args+['--output', str(output)])
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'complete_pair'
    assert set(report['artifacts']) == {'training.csv', 'development.csv', 'throughput.csv',
        'development-raw-losses.pdf', 'development-raw-losses.png', 'training-dynamics.pdf',
        'training-dynamics.png', 'timing-memory.pdf', 'timing-memory.png'}
    assert (output/'input-snapshot'/'parent.json').read_bytes() == parent_data
    for name, record in report['artifacts'].items(): assert summary.sha((output/name).read_bytes()) == record['sha256']
    args[2] = '0'*64
    with pytest.raises(ValueError, match='Input SHA'): summary.main(args+['--output', str(tmp_path/'bad')])
    assert not (tmp_path/'bad').exists()
