#!/usr/bin/env python3
"""CPU-only posthoc summaries for completed native NF KL1/KL0.1 forks.

Consumes three explicitly SHA-pinned JSON reports. This is a results reducer,
not a replacement for the separate checkpoint/restart acceptance audit. It
imports no model, Torch, runtime, network, or cloud code. Matplotlib is imported
only when writing static figures. All output goes into a fresh directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = 'olmo-kl-continuation-summary-v1'
TERMS = ('ce', 'latent', 'kl')
UPDATES = list(range(33, 65))
EVAL_UPDATES = (32, 48, 64)
COEFFICIENTS = {'ce': [.5, 1/6, 1/6, 1/6], 'latent': [.25]*4, 'kl': [.25]*4}
TIMING_SCOPES = {
    'compute_regions': 'Sum of per-update maximum-rank (backward + optimizer/cursor) seconds; includes all four passes, backward and communication, excludes materialization, dev evaluation, checkpointing and startup.',
    'compute_plus_materialization': 'Sum of per-update maximum-rank (materialization + backward + optimizer/cursor) seconds; excludes dev evaluation, checkpointing and startup.',
    'complete_update_callback': 'Sum of maximum-rank update callback seconds; includes scalar/report overhead and due dev evaluation, excludes graph preparation and checkpoint callbacks.',
    'stage_wall': 'Reported entire segment wall time including startup, restore, graph preparation, evaluations, checkpoint work and final drain; not optimized steady-state throughput.',
    'memory': 'Per-rank allocator samples after capture and updates: maxima across samples/ranks, never a two-GPU sum. Peak allocator values may include preparation/evaluation; sampled free memory is a minimum, not an allocation peak.',
    'checkpoint': 'Local boundary hashing/write/postcheck and foreground worker wait are separately scoped. Background worker elapsed time overlaps training and must not be added to stage wall time.',
}
QUALIFICATIONS = [
    'Only 32 additional optimizer updates (33–64), one seed and one fixed 64-row development prefix; no held-out quality or statistical generalization claim.',
    'Both branches inherit the same update-32 model, Adam moments, schedule, cursor and RNG. The original 128-update schedule continues; this entire interval remains within its 100-update warmup.',
    'Training uses BF16 mixed precision and feedback jitter; development uses common FP32 without jitter. Their loss curves are different measurement contexts.',
    'Raw CE, latent and KL terms retain separate eligible-target denominators. Weighted total objectives differ by definition between KL1 and KL0.1 and are not compared as quality metrics.',
    'Training raw CE already combines passes with coefficients 1/2, 1/6, 1/6, 1/6; training raw latent/KL each average four passes. Raw here excludes the external auxiliary loss weights, not these canonical pass coefficients.',
    'Later-pass CE is interpreted together with absolute first-pass CE and the later-minus-first gap. A smaller gap alone can result from a worse first pass.',
    'Clipping coefficients are host estimates from returned pre-clip norms. Lower KL weight does not imply a tenfold Adam update because clipping and inherited moments also matter.',
    'This posthoc reducer checks report consistency; it does not rerun GPU math, repeat exact-resume acceptance, or provide new BF16 numerical clearance.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite or nonnumeric measurement')
    return value


def duration(value):
    require(finite(value) >= 0, 'Negative duration')
    return value


def close(actual, expected, label):
    require(math.isclose(finite(actual), finite(expected), rel_tol=2e-6, abs_tol=1e-6), label)


def raw_losses(record):
    require(all(set(record[field]) == set(TERMS) for field in ('sums', 'counts', 'means')), 'Loss term names differ')
    for term in TERMS:
        count = record['counts'][term]
        require(type(count) is int and count > 0, 'Enabled loss denominator must be positive')
        close(record['means'][term], finite(record['sums'][term])/count, 'Loss numerator/denominator disagreement')
    return {field: copy.deepcopy(record[field]) for field in ('sums', 'counts', 'means')}


def evaluate(report, *, weight, expected_updates):
    entries = report['evaluations']
    selected = [e for e in entries if e['after_update'] in expected_updates]
    require(sorted(e['after_update'] for e in selected) == list(expected_updates), 'Missing or duplicate evaluation boundary')
    rows = []
    for entry in sorted(selected, key=lambda e: e['after_update']):
        require(entry['status'] == 'completed' and entry['training_boundary_exact_by_rank'] == [True, True], 'Evaluation not complete/preserved')
        require(set(entry['panels']) == {'dev-main'}, 'Expected the single named development prefix')
        panel = entry['panels']['dev-main']; result = panel['result']
        require(result['policy'] == 'common_fp32_no_jitter_v1'
                and result['input_tokens'] == 65536
                and result['enabled'] == dict.fromkeys(TERMS, True)
                and result['weights'] == {'ce': 1., 'latent': 1., 'kl': weight}
                and result['term_pass_coefficients'] == COEFFICIENTS, 'Evaluation policy/weights differ')
        require(len(panel['by_rank']) == 2, 'Missing evaluation rank')
        for rank in panel['by_rank']:
            preservation = rank['preservation']
            require(preservation['integrity_passed'] is True and preservation['restored'] is True
                    and all(v is True for v in preservation['checks'].values()), 'Evaluation preservation failed')
        require([p['index'] for p in result['passes']] == [0, 1, 2, 3], 'Pass order differs')
        aggregate = raw_losses(result['aggregate'])
        passes = [{'pass': p['index']+1, **raw_losses(p)} for p in result['passes']]
        for term in TERMS:
            require(all(p['counts'][term] == aggregate['counts'][term] for p in passes), 'Pass denominators differ')
            reconstructed = math.fsum(COEFFICIENTS[term][i]*p['sums'][term] for i, p in enumerate(passes))
            close(aggregate['sums'][term], reconstructed, 'Pass aggregation differs')
            close(result['reconstructed_aggregate_sums'][term], reconstructed, 'Recorded reconstruction differs')
        objective = math.fsum(result['weights'][term]*aggregate['means'][term] for term in TERMS)
        close(result['objective'], objective, 'Weighted objective bookkeeping differs')
        rows.append({'after_update': entry['after_update'], 'panel': 'dev-main',
            'membership_sha256': panel['membership_sha256'], 'index_manifest_sha256': panel['index_manifest_sha256'],
            'input_tokens': result['input_tokens'], 'passes': passes, 'aggregate': aggregate,
            'weighted_objective_branch_specific': objective, 'weights': result['weights'],
            'ce_gap_vs_first_pass': [p['means']['ce']-passes[0]['means']['ce'] for p in passes],
            'total_seconds': duration(entry['total_seconds']),
            'preservation_passed': True})
    return rows


def parameter_counts(ownership):
    layout = ownership['parameter_layout']
    sizes = {p['name']: math.prod(p['shape']) for p in layout}
    names = [name for group in ownership['optimizer_ownership'] for name in group]
    require(len(sizes) == len(layout) and len(names) == len(set(names)), 'Duplicate parameter ownership')
    owned = sum(sizes[name] for name in names)
    require(sum(sizes.values()) == ownership['resident_parameters']
            and owned == ownership['trainable_parameters'], 'Parameter counts disagree')
    return {'registered_resident': ownership['resident_parameters'], 'trainable': ownership['trainable_parameters'],
            'optimizer_owned': owned, 'components': ownership['component_parameters'],
            'deployable_active_without_predictor': ownership['resident_parameters']-ownership['component_parameters']['predictor']}


def summarize_branch(report, weight, parent, parent_sha):
    require(report['schema'] == 'olmo-kl-continuation-report-v1' and report['status'] == 'stopped_at_boundary'
            and report['segment_completed'] is True and report['plan_completed'] is False
            and report['scale'] == 'native' and report['arm'] == 'NF'
            and report['segment_stop_after'] == 64 and report['checkpoint_mode'] == 'async'
            and report['observation_mode'] == 'lean', 'Require a completed native NF 32–64 branch')
    require(set(report['updates']) == {str(n) for n in UPDATES}, 'Require one complete 33–64 history; resumed segments need an explicit join')
    branch = report['branch']; config = report['configuration']; recipe = config['recipe']
    require(branch['kl_weight'] == weight and branch['parent_update'] == 32 and branch['review_stop'] == 64
            and branch['parent_report_sha256'] == parent_sha and report['parent_report_sha256'] == parent_sha
            and branch['parent_identity_sha256'] == parent['configuration']['execution_identity']['sha256']
            and report['original_configuration'] == parent['configuration'], 'Parent authority or KL branch differs')
    require(recipe == branch['recipe_as_declared'] and recipe['auxiliary']['kl'] == weight
            and recipe['auxiliary']['latent'] == 1 and recipe['optimizer_state'] == 'inherited_exact_parent_checkpoint'
            and recipe['fbt_passes'] == 4
            and recipe['rt_layers'] == parent['configuration']['recipe']['rt_layers']
            and config['parameters']['mode']['rt_mode']['selected_layers'] == []
            and recipe['sequence_length'] == 1024 and recipe['effective_valid_tokens'] == 524288
            and recipe['feedback_jitter'] == .02 and config['world_size'] == 2
            and config['training']['precision'] == 'bf16_mixed', 'Experiment shape/objective changed')
    schedule = config['schedule']
    require(schedule == parent['configuration']['schedule'] and schedule['planned_updates'] == 128
            and schedule['warmup_tokens'] == 52428800, 'Original schedule was reset or changed')
    parent_boundaries = [c['boundary_by_rank'] for c in parent['local_checkpoints']
                         if c['receipt']['manifest_sha256'] == branch['parent_manifest_sha256']]
    require(len(parent_boundaries) == 1, 'Missing unique saved parent boundary')
    boundary = parent_boundaries[0]
    require(report['objective_transition']['before_boundary_by_rank'] == boundary
            == report['objective_transition']['after_boundary_by_rank'] == report['origin_boundary_by_rank'], 'Objective transition changed saved parent state')
    require(report['preparation_boundary_exact'] == [True, True] and report['adam_resident_before_ddp'] is True, 'Preparation or inherited Adam evidence missing')
    loop = report['loop']
    require(loop['start_update'] == 32 and loop['completed_update'] == loop['last_saved_update']
            == loop['last_retained_update'] == report['last_verified_cloud_update'] == 64
            and loop['checkpoint_pending'] is False, 'Terminal checkpoint not durable')
    memory = list(report['memory_after_capture']); rows = []
    for update in UPDATES:
        key = str(update); ranked = report['updates'][key]; observation = report['observations'][key]
        require(len(ranked) == 2 and ranked[0]['metrics'] == ranked[1]['metrics']
                and ranked[0]['clipping'] == ranked[1]['clipping'], 'Replica metrics disagree')
        metrics = ranked[0]['metrics']; clip = ranked[0]['clipping']
        require(metrics['input_tokens'] == 524288 and metrics['world_size'] == 2
                and metrics['local_microbatches'] == 22 and metrics['microbatches'] == 44
                and metrics['counters']['optimizer_updates'] == update
                and metrics['counters']['input_tokens'] == update*524288, 'Training exposure/counters differ')
        loss = raw_losses({'sums': metrics['loss_sums'], 'counts': metrics['counts'], 'means': ranked[0]['loss_means']})
        close(metrics['objective'], loss['means']['ce']+loss['means']['latent']+weight*loss['means']['kl'], 'Training weighted objective differs')
        norm = finite(metrics['gradient_norm_before_clip']); require(norm >= 0, 'Negative gradient norm')
        require(clip['configured_limit'] == 1 and clip['norm_exceeds_limit'] == (norm > 1), 'Clipping flag differs')
        close(clip['coefficient_estimate'], min(1., 1./(norm+1e-6)), 'Clipping coefficient differs')
        require(all(r['cursor']['physical_batch_per_rank'] == 12 for r in ranked), 'Physical batch changed')
        for term, field in [('ce', 'ce_targets'), ('latent', 'latent_pairs'), ('kl', 'kl_triples')]:
            require(sum(rank[field] for rank in observation['rank_data']) == loss['counts'][term], 'Rank data denominator differs')
        require(sum(rank['valid_tokens'] for rank in observation['rank_data']) == 524288, 'Rank input exposure differs')
        for label, completed in [('lr_used', update-1), ('lr_next', update)]:
            expected = recipe['plateau_lr']*(schedule['start_fraction']+(1-schedule['start_fraction'])*completed*524288/schedule['warmup_tokens'])
            require(len(metrics[label]) > 0, 'Missing optimizer LR groups')
            for value in metrics[label]:
                require(math.isclose(finite(value), expected, rel_tol=1e-12, abs_tol=1e-15), 'LR no longer follows inherited warmup')
        timings = observation['timing_by_rank']; require(len(timings) == 2, 'Missing timing rank')
        scopes = {'compute_regions': ('backward', 'optimizer_and_cursor'),
                  'compute_plus_materialization': ('materialization_host', 'backward', 'optimizer_and_cursor')}
        seconds = {name: max(sum(duration(rank[field]) for field in fields) for rank in timings)
                   for name, fields in scopes.items()}
        seconds['complete_update_callback'] = max(duration(x) for x in report['update_wall_seconds_by_rank'][key])
        require(all(x > 0 for x in seconds.values()), 'Zero throughput denominator')
        memory.extend(observation['memory_by_rank'])
        rows.append({'update': update, 'input_tokens': metrics['input_tokens'],
            'cumulative_input_tokens': metrics['counters']['input_tokens'], 'additional_input_tokens': (update-32)*524288,
            'loss_sums': loss['sums'], 'counts': loss['counts'], 'loss_means': loss['means'],
            'weighted_objective_branch_specific': metrics['objective'],
            'gradient_norm_before_clip': norm, 'clip_coefficient_estimate': clip['coefficient_estimate'],
            'clipped': clip['norm_exceeds_limit'], 'lr_used': metrics['lr_used'], 'lr_next': metrics['lr_next'],
            'seconds': seconds, 'memory_by_rank': observation['memory_by_rank'],
            'cursor_by_rank': [r['cursor'] for r in ranked], 'rank_data': observation['rank_data']})
    require(report['final_counters'] == ranked[0]['metrics']['counters'], 'Final counters differ from last update')
    rates = {}
    for label, selected in [('all33_64', rows), ('updates34_64', rows[1:]), ('updates33_48', rows[:16]), ('updates49_64', rows[16:])]:
        tokens = sum(r['input_tokens'] for r in selected)
        rates[label] = {'input_tokens': tokens}
        for scope in TIMING_SCOPES.keys() & rows[0]['seconds'].keys():
            seconds = sum(row['seconds'][scope] for row in selected)
            rates[label][scope] = {'seconds': seconds, 'input_tokens_per_second': tokens/seconds}
    elapsed = duration(report['elapsed_seconds']); require(elapsed > 0, 'Missing segment wall time')
    memory_summary = {name: (min if name == 'sampled_free_gib' else max)(finite(m[name]) for m in memory)
                      for name in ('allocated_gib', 'reserved_gib', 'peak_allocated_gib', 'peak_reserved_gib', 'sampled_free_gib', 'total_gib')}
    checkpoint = {name: sum(max(duration(x) for x in ranks) for ranks in report['timing'].get(name, {}).values())
                  for name in ('checkpoint_observation_seconds_by_rank', 'checkpoint_write_seconds_by_rank', 'checkpoint_postcheck_seconds_by_rank')}
    checkpoint['foreground_worker_wait_seconds'] = duration(loop['checkpoint_wait_seconds'])
    return {'kl_weight': weight, 'updates': rows, 'development': evaluate(report, weight=weight, expected_updates=EVAL_UPDATES),
        'parameters': parameter_counts(config['parameters']), 'schedule': schedule,
        'clipping': {'clipped_updates': sum(row['clipped'] for row in rows), 'total_updates': len(rows),
            'norm_min': min(row['gradient_norm_before_clip'] for row in rows),
            'norm_median': statistics.median(row['gradient_norm_before_clip'] for row in rows),
            'norm_max': max(row['gradient_norm_before_clip'] for row in rows),
            'coefficient_median': statistics.median(row['clip_coefficient_estimate'] for row in rows)},
        'rates': rates, 'stage_wall_seconds': elapsed, 'stage_wall_input_tokens_per_second': 32*524288/elapsed,
        'preparation_seconds_max_rank': max(duration(x) for x in report['timing']['preparation_seconds_by_rank']),
        'memory_sample_summary_gib': memory_summary, 'checkpoint_seconds': checkpoint,
        'terminal_cloud_update': report['last_verified_cloud_update'], 'wandb': report.get('wandb'),
        'parent_manifest_sha256': branch['parent_manifest_sha256'], 'execution_identity_sha256': config['execution_identity']['sha256']}


def summarize(control, reduced, parent, *, parent_sha):
    require(parent['schema'] == 'olmo-pilot-async-execute-report-v1' and parent['status'] == 'stopped_at_boundary'
            and parent['arm'] == 'NF' and parent['scale'] == 'native'
            and parent['final_counters']['optimizer_updates'] == 32, 'Invalid completed parent report')
    require(control['sources'] == reduced['sources'] and len(control['sources']) == 210, 'Frozen branch runtime inventories differ')
    for key in ('declaration_sha256', 'resolved_sha256'):
        require(control[key] == reduced[key] == parent[key], 'Data declaration authority differs')
    arms = {'KL1': summarize_branch(control, 1., parent, parent_sha),
            'KL0.1': summarize_branch(reduced, .1, parent, parent_sha)}
    require(arms['KL1']['parent_manifest_sha256'] == arms['KL0.1']['parent_manifest_sha256'], 'Branches have different parents')
    parent_eval = evaluate(parent, weight=1., expected_updates=(32,))[0]
    origins = [arm['development'][0] for arm in arms.values()]
    for origin in origins:
        require(origin['passes'] == parent_eval['passes'] and origin['aggregate'] == parent_eval['aggregate'], 'Restored origin raw evaluation differs from parent')
    comparisons = []
    for left, right in zip(arms['KL1']['updates'], arms['KL0.1']['updates']):
        for key in ('update', 'input_tokens', 'cumulative_input_tokens', 'counts', 'lr_used', 'lr_next', 'cursor_by_rank', 'rank_data'):
            require(left[key] == right[key], 'Paired input exposure, cursor or LR differs: '+key)
    for left, right in zip(arms['KL1']['development'], arms['KL0.1']['development']):
        for key in ('after_update', 'membership_sha256', 'index_manifest_sha256', 'input_tokens'):
            require(left[key] == right[key], 'Paired development panel differs')
        require(left['membership_sha256'] == parent_eval['membership_sha256']
                and left['index_manifest_sha256'] == parent_eval['index_manifest_sha256'], 'Development prefix changed since parent')
        comparisons.append({'after_update': left['after_update'], 'direction': 'KL0.1 minus KL1; negative raw CE is lower',
            'raw_mean_deltas_by_pass': [{'pass': l['pass'], **{term: r['means'][term]-l['means'][term] for term in TERMS}}
                                      for l, r in zip(left['passes'], right['passes'])],
            'ce_gap_delta_vs_first_pass': [r-l for l, r in zip(left['ce_gap_vs_first_pass'], right['ce_gap_vs_first_pass'])]})
    return {'schema': SCHEMA, 'status': 'complete_pair', 'arms': arms,
        'parent_development_update32': parent_eval, 'comparison': comparisons,
        'timing_scopes': TIMING_SCOPES, 'qualifications': QUALIFICATIONS,
        'runtime_sources': control['sources'],
        'checks': {'same_saved_parent_boundary': True, 'same_restored_raw_evaluation': True,
                   'same_data_exposure_and_lr': True, 'common_development_prefix': True,
                   'completed_retained_endpoints': True}}


def csv_file(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_csvs(summary, output):
    training, evaluation, resources = [], [], []
    for label, arm in summary['arms'].items():
        for row in arm['updates']:
            training.append({'branch': label, **{key: row[key] for key in ('update', 'input_tokens', 'cumulative_input_tokens', 'additional_input_tokens', 'gradient_norm_before_clip', 'clip_coefficient_estimate', 'clipped')},
                'lr_used_all_groups': json.dumps(row['lr_used']),
                **{'raw_'+term: row['loss_means'][term] for term in TERMS},
                **{'seconds_'+key: value for key, value in row['seconds'].items()}})
        for entry in arm['development']:
            for p in entry['passes']:
                evaluation.append({'branch': label, 'after_update': entry['after_update'], 'pass': p['pass'],
                    **{term+'_'+field: p[field][term] for term in TERMS for field in ('sums', 'counts', 'means')},
                    'ce_gap_vs_first_pass': entry['ce_gap_vs_first_pass'][p['pass']-1]})
        for window, rate in arm['rates'].items():
            for scope, values in rate.items():
                if scope != 'input_tokens':
                    resources.append({'branch': label, 'window': window, 'scope': scope,
                        'input_tokens': rate['input_tokens'], **values})
        resources.append({'branch': label, 'window': 'all33_64', 'scope': 'stage_wall',
            'input_tokens': 32*524288, 'seconds': arm['stage_wall_seconds'],
            'input_tokens_per_second': arm['stage_wall_input_tokens_per_second']})
    csv_file(output/'training.csv', training); csv_file(output/'development.csv', evaluation)
    csv_file(output/'throughput.csv', resources)


def write_plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2166ac', '#b2182b', '#1b7837', '#762a83']
    def save(fig, name):
        fig.savefig(output/(name+'.pdf'), bbox_inches='tight')
        fig.savefig(output/(name+'.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    for col, (label, arm) in enumerate(summary['arms'].items()):
        xs = [d['after_update'] for d in arm['development']]
        for row, term in enumerate(TERMS):
            ax = axes[row, col]
            for p in range(4):
                ax.plot(xs, [d['passes'][p]['means'][term] for d in arm['development']], 'o-', color=colors[p], label=f'Pass {p+1}')
            ax.set_title(f'{label}: raw {term.upper()}'); ax.set_ylabel('Mean per eligible target'); ax.grid(alpha=.25)
            if row == 0: ax.legend(fontsize=8)
            if row == 2: ax.set_xlabel('Cumulative optimizer update')
    fig.suptitle('NF continuation: common FP32 development, no jitter\n32 additional updates; one fixed 64-row development prefix')
    fig.tight_layout(); save(fig, 'development-raw-losses')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fields = [('gradient_norm_before_clip', 'Pre-clip gradient norm', True),
              ('clip_coefficient_estimate', 'Clipping coefficient estimate', True),
              ('lr_used', 'Learning rate used', False),
              ('ce', 'Training raw CE', False), ('latent', 'Training raw latent', False), ('kl', 'Training raw KL', False)]
    for label, arm in summary['arms'].items():
        xs = [row['update'] for row in arm['updates']]
        for ax, (field, title, log) in zip(axes.flat, fields):
            ys = [row['loss_means'][field] if field in TERMS else row['lr_used'][0] if field == 'lr_used' else row[field] for row in arm['updates']]
            ax.plot(xs, ys, label=label); ax.set_title(title); ax.set_xlabel('Cumulative optimizer update'); ax.grid(alpha=.25)
            if log: ax.set_yscale('log')
    for ax in axes.flat: ax.legend(fontsize=8)
    fig.suptitle('BF16 training, same data and inherited Adam; branch-specific total objectives omitted')
    fig.tight_layout(); save(fig, 'training-dynamics')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for label, arm in summary['arms'].items():
        xs = [row['update'] for row in arm['updates']]
        axes[0].plot(xs, [row['input_tokens']/row['seconds']['compute_regions'] for row in arm['updates']], label=label)
        axes[1].plot(xs, [row['input_tokens']/row['seconds']['complete_update_callback'] for row in arm['updates']], label=label)
        axes[2].plot(xs, [max(m['reserved_gib'] for m in row['memory_by_rank']) for row in arm['updates']], label=label)
    for ax, title, ylabel in zip(axes, ['Compute regions only', 'Complete update callback', 'Maximum per-rank reserved memory'], ['Input tokens/second', 'Input tokens/second', 'GiB']):
        ax.set_title(title); ax.set_xlabel('Cumulative optimizer update'); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('Scoped measurements, not optimized throughput\nCallback includes due dev evaluation; both rates exclude checkpoint callbacks and graph preparation')
    fig.tight_layout(); save(fig, 'timing-memory')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ('parent', 'control', 'reduced'):
        parser.add_argument('--'+label, nargs=2, metavar=('REPORT_JSON', 'SHA256'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv); inputs = {}; pins = {}
    for label in ('parent', 'control', 'reduced'):
        path, expected = getattr(args, label); path = Path(path).resolve(); data = path.read_bytes()
        require(len(expected) == 64 and sha(data) == expected, 'Input SHA differs: '+label)
        inputs[label] = (data, json.loads(data))
        pins[label] = {'original_path': str(path), 'sha256': expected, 'size_bytes': len(data), 'snapshot': 'input-snapshot/'+label+'.json'}
    summary = summarize(inputs['control'][1], inputs['reduced'][1], inputs['parent'][1], parent_sha=pins['parent']['sha256'])
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'input-snapshot').mkdir()
    for label, (data, _) in inputs.items(): (output/pins[label]['snapshot']).write_bytes(data)
    producer = Path(__file__).resolve(); producer_data = producer.read_bytes()
    (output/'source-snapshot').mkdir(); (output/'source-snapshot'/producer.name).write_bytes(producer_data)
    summary['input_pins'] = pins
    summary['analysis_producer'] = {'original_path': str(producer), 'sha256': sha(producer_data), 'snapshot': 'source-snapshot/'+producer.name}
    write_csvs(summary, output); write_plots(summary, output)
    summary['artifacts'] = {p.name: {'sha256': sha(p.read_bytes()), 'size_bytes': p.stat().st_size}
                            for p in sorted(output.iterdir()) if p.is_file()}
    (output/'report.json').write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': summary['status'], 'report': str(output/'report.json'), 'sha256': sha((output/'report.json').read_bytes())}))


if __name__ == '__main__':
    main()
