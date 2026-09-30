#!/usr/bin/env python3
"""CPU-only, audit-bound NFR32→64 KL1/KL0.1 comparison; optional F64 context.

No report relabeling or execution imports. Generic loss checks, parameter
accounting, CSV output and timing definitions come from the accepted NF reducer.
The NF-only branch validator and plot titles are deliberately not reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from scripts import olmo_kl_continuation_summary as common

SCHEMA = 'olmo-nfr-kl-summary-v1'
ROOT = Path(__file__).resolve().parents[1]
TERMS, UPDATES = common.TERMS, common.UPDATES
require, finite, duration, close = common.require, common.finite, common.duration, common.close
SOURCE_NAMES = ('scripts/olmo_nfr_kl_summary.py', 'scripts/olmo_kl_continuation_summary.py')


def check_authorities(control, reduced, parent, audit, pins):
    require(audit['schema'] == 'olmo-nfr-kl-audit-v1'
            and audit['comparison_kind'] == 'paired_fork' and audit['passed'] is True
            and not audit['failures'] and audit['checks']
            and all(check['passed'] is True for check in audit['checks']),
            'Require a passed independent NFR pair audit')
    for label in ('parent', 'control', 'reduced'):
        require(audit['inputs'][label]['sha256'] == pins[label]['sha256'],
                'Pair audit is not bound to input report: ' + label)
    require(parent['arm'] == 'NFR' and parent['scale'] == 'native'
            and parent['final_counters']['optimizer_updates'] == 32
            and parent['status'] == 'stopped_at_boundary', 'Require completed original NFR32 parent')
    require(control['sources'] == reduced['sources'] and len(control['sources']) == 215,
            'Require identical frozen 215-source NFR runtimes')
    require(control['nfr_scope'] == reduced['nfr_scope'], 'NFR scope differs')
    for weight, report in ((1., control), (.1, reduced)):
        require(report['schema'] == 'olmo-kl-continuation-report-v1' and report['arm'] == 'NFR'
                and report['scale'] == 'native' and report['status'] == 'stopped_at_boundary'
                and report['segment_completed'] is True and report['plan_completed'] is False
                and report['segment_stop_after'] == 64, 'Require completed native NFR32-to64 branch')
        require(set(report['updates']) == {str(n) for n in UPDATES},
                'Require a complete unjoined 33-to64 history')
        config, branch = report['configuration'], report['branch']
        mode = config['parameters']['mode']
        require(mode['rt_mode'] == {'selected_layers': [0, 15], 'alpha': 1.}
                and mode['num_passes'] == 4 and mode['enabled'] is True
                and mode['beta'] == 1. and mode['feedback_jitter'] == .02,
                'NFR model mode differs')
        require(branch['kl_weight'] == weight and branch['parent_update'] == 32
                and branch['review_stop'] == 64
                and branch['parent_report_sha256'] == pins['parent']['sha256']
                and report['original_configuration'] == parent['configuration'],
                'Parent authority or KL coefficient differs')
        require(config['schedule'] == parent['configuration']['schedule']
                and config['schedule']['planned_updates'] == 128, 'Inherited schedule differs')
        require(report['nfr_scope']['sha256'] == audit['inputs']['scope']['sha256']
                and report['nfr_scope']['sha256'] in report['sources'].values(), 'Scope is not audit-bound')
        loop = report['loop']
        require(loop['start_update'] == 32 and loop['completed_update'] == loop['last_saved_update']
                == loop['last_retained_update'] == report['last_verified_cloud_update'] == 64
                and loop['checkpoint_pending'] is False, 'Terminal checkpoint is not durable')
    require(control['origin_boundary_by_rank'] == reduced['origin_boundary_by_rank'],
            'Pair restored different complete states')


def summarize_branch(report, weight):
    """Aggregate already-audited records, preserving denominator and time scopes."""
    rows, memory = [], list(report['memory_after_capture'])
    for update in UPDATES:
        key = str(update)
        ranked, observation = report['updates'][key], report['observations'][key]
        require(len(ranked) == 2 and ranked[0]['metrics'] == ranked[1]['metrics']
                and ranked[0]['clipping'] == ranked[1]['clipping'], 'Rank metrics differ')
        metrics, clip = ranked[0]['metrics'], ranked[0]['clipping']
        raw = common.raw_losses({'sums': metrics['loss_sums'], 'counts': metrics['counts'],
                                 'means': ranked[0]['loss_means']})
        # Bookkeeping only: objective totals never enter comparison outputs.
        close(metrics['objective'], raw['means']['ce'] + raw['means']['latent']
              + weight * raw['means']['kl'], 'Weighted bookkeeping differs')
        tokens = metrics['input_tokens']
        require(tokens == 524288 and metrics['counters']['optimizer_updates'] == update,
                'Update exposure differs')
        norm = finite(metrics['gradient_norm_before_clip'])
        require(norm >= 0 and clip['configured_limit'] == 1
                and clip['norm_exceeds_limit'] == (norm > 1), 'Clipping flag differs')
        close(clip['coefficient_estimate'], min(1., 1. / (norm + 1e-6)), 'Clipping estimate differs')
        timings = observation['timing_by_rank']
        require(len(timings) == 2, 'Missing rank timing')
        seconds = {scope: max(sum(duration(rank[field]) for field in fields) for rank in timings)
            for scope, fields in {
                'compute_regions': ('backward', 'optimizer_and_cursor'),
                'compute_plus_materialization': ('materialization_host', 'backward', 'optimizer_and_cursor')}.items()}
        seconds['complete_update_callback'] = max(duration(x) for x in report['update_wall_seconds_by_rank'][key])
        require(all(value > 0 for value in seconds.values()), 'Zero throughput denominator')
        memory.extend(observation['memory_by_rank'])
        rows.append({'update': update, 'input_tokens': tokens,
            'cumulative_input_tokens': metrics['counters']['input_tokens'],
            'additional_input_tokens': (update - 32) * tokens,
            'loss_sums': raw['sums'], 'counts': raw['counts'], 'loss_means': raw['means'],
            'gradient_norm_before_clip': norm, 'clip_coefficient_estimate': clip['coefficient_estimate'],
            'clipped': clip['norm_exceeds_limit'], 'lr_used': metrics['lr_used'], 'lr_next': metrics['lr_next'],
            'seconds': seconds, 'memory_by_rank': observation['memory_by_rank'],
            'cursor_by_rank': [rank['cursor'] for rank in ranked], 'rank_data': observation['rank_data']})
    rates = {}
    for label, selected in [('all33_64', rows), ('updates34_64', rows[1:]),
                            ('updates33_48', rows[:16]), ('updates49_64', rows[16:])]:
        tokens = sum(row['input_tokens'] for row in selected)
        rates[label] = {'input_tokens': tokens}
        for scope in rows[0]['seconds']:
            seconds = sum(row['seconds'][scope] for row in selected)
            rates[label][scope] = {'seconds': seconds, 'input_tokens_per_second': tokens / seconds}
    development = common.evaluate(report, weight=weight, expected_updates=(32, 48, 64))
    for entry in development:
        entry.pop('weighted_objective_branch_specific')
    elapsed = duration(report['elapsed_seconds'])
    require(elapsed > 0, 'Missing segment wall time')
    checkpoints = {name: sum(max(duration(x) for x in ranks) for ranks in report['timing'].get(name, {}).values())
        for name in ('checkpoint_observation_seconds_by_rank', 'checkpoint_write_seconds_by_rank',
                     'checkpoint_postcheck_seconds_by_rank')}
    checkpoints['foreground_worker_wait_seconds'] = duration(report['loop']['checkpoint_wait_seconds'])
    return {'kl_weight': weight, 'updates': rows, 'development': development,
        'parameters': common.parameter_counts(report['configuration']['parameters']),
        'schedule': report['configuration']['schedule'], 'rates': rates,
        'clipping': {'clipped_updates': sum(row['clipped'] for row in rows), 'total_updates': len(rows),
            'norm_min': min(row['gradient_norm_before_clip'] for row in rows),
            'norm_median': statistics.median(row['gradient_norm_before_clip'] for row in rows),
            'norm_max': max(row['gradient_norm_before_clip'] for row in rows),
            'coefficient_median': statistics.median(row['clip_coefficient_estimate'] for row in rows)},
        'memory_sample_summary_gib': {name: (min if name == 'sampled_free_gib' else max)(finite(m[name]) for m in memory)
            for name in ('allocated_gib', 'reserved_gib', 'peak_allocated_gib', 'peak_reserved_gib', 'sampled_free_gib', 'total_gib')},
        'stage_wall_seconds': elapsed, 'stage_wall_input_tokens_per_second': 32 * 524288 / elapsed,
        'preparation_seconds_max_rank': max(duration(x) for x in report['timing']['preparation_seconds_by_rank']),
        'checkpoint_seconds': checkpoints, 'terminal_cloud_update': 64,
        'wandb': report.get('wandb'), 'parent_manifest_sha256': report['branch']['parent_manifest_sha256']}


def f64_context(evaluation, publication, paired_development):
    """Allow only same-panel F-only update64 CE, not F128 or a matched fork claim."""
    config = publication['metadata']['configuration']
    payload = config['execution_identity']['payload']
    mode = payload['model_contract']['mode']
    require(publication['counters']['optimizer_updates'] == 64 and payload['arm'] == 'F'
            and mode['num_passes'] == 4 and mode['rt_mode']['selected_layers'] == []
            and mode['enabled'] is False and mode['beta'] == 1.
            and payload['recipe']['sequence_length'] == 1024
            and payload['recipe']['effective_valid_tokens'] == 524288,
            'Context must be native F-only at update64')
    require(evaluation['after_update'] == 64 and evaluation['status'] == 'completed'
            and evaluation['training_boundary_exact_by_rank'] == [True, True],
            'F context must be preserved update64 evaluation')
    require(set(evaluation['panels']) == {'dev-main'}, 'F context panel differs')
    panel = evaluation['panels']['dev-main']; result = panel['result']
    require(len(panel['by_rank']) == 2, 'Missing F context rank')
    for rank in panel['by_rank']:
        preservation = rank['preservation']
        require(preservation['integrity_passed'] is True and preservation['restored'] is True
                and all(value is True for value in preservation['checks'].values()),
                'F context preservation failed')
    require(panel['membership_sha256'] == paired_development['membership_sha256']
            and panel['index_manifest_sha256'] == paired_development['index_manifest_sha256'],
            'F context development membership differs')
    require(result['policy'] == 'common_fp32_no_jitter_v1' and result['input_tokens'] == 65536
            and result['enabled'] == {'ce': True, 'latent': False, 'kl': False}
            and result['term_pass_coefficients']['ce'] == common.COEFFICIENTS['ce'],
            'F context measurement policy differs')
    require([p['index'] for p in result['passes']] == [0, 1, 2, 3], 'F context pass order differs')
    passes = []
    for source, reference in zip(result['passes'], paired_development['passes']):
        require(source['counts']['ce'] == reference['counts']['ce'], 'F context CE denominator differs')
        ce = finite(source['means']['ce'])
        close(ce, source['sums']['ce'] / source['counts']['ce'], 'F context CE denominator disagreement')
        passes.append({'pass': source['index'] + 1, 'ce': ce, 'count': source['counts']['ce']})
    return {'after_update': 64, 'passes': passes,
        'ce_gap_vs_first_pass': [p['ce'] - passes[0]['ce'] for p in passes],
        'scope': 'Descriptive F-only same-update/same-panel context; different architecture and optimizer trajectory, not a paired KL intervention'}


def summarize(control, reduced, parent, audit, pins, *, f_evaluation=None, f_publication=None):
    check_authorities(control, reduced, parent, audit, pins)
    arms = {'KL1': summarize_branch(control, 1.), 'KL0.1': summarize_branch(reduced, .1)}
    parent_eval = common.evaluate(parent, weight=1., expected_updates=(32,))[0]
    parent_eval.pop('weighted_objective_branch_specific')
    for arm in arms.values():
        require(arm['development'][0]['passes'] == parent_eval['passes']
                and arm['development'][0]['aggregate'] == parent_eval['aggregate'],
                'Restored raw origin differs from parent')
    for left, right in zip(arms['KL1']['updates'], arms['KL0.1']['updates']):
        for key in ('update', 'input_tokens', 'cumulative_input_tokens', 'counts', 'lr_used',
                    'lr_next', 'cursor_by_rank', 'rank_data'):
            require(left[key] == right[key], 'Paired exposure or LR differs: ' + key)
    comparison = []
    for left, right in zip(arms['KL1']['development'], arms['KL0.1']['development']):
        for key in ('after_update', 'membership_sha256', 'index_manifest_sha256', 'input_tokens'):
            require(left[key] == right[key], 'Paired development differs: ' + key)
        require(left['membership_sha256'] == parent_eval['membership_sha256']
                and left['index_manifest_sha256'] == parent_eval['index_manifest_sha256'],
                'Development prefix changed since parent')
        comparison.append({'after_update': left['after_update'],
            'direction': 'KL0.1 minus KL1; negative raw CE is lower',
            'raw_mean_deltas_by_pass': [{'pass': l['pass'], **{term: r['means'][term] - l['means'][term]
                for term in TERMS}} for l, r in zip(left['passes'], right['passes'])],
            'ce_gap_delta_vs_first_pass': [r - l for l, r in zip(left['ce_gap_vs_first_pass'], right['ce_gap_vs_first_pass'])]})
    require((f_evaluation is None) == (f_publication is None), 'Provide both F64 authorities or neither')
    context = None if f_evaluation is None else f64_context(f_evaluation, f_publication, arms['KL1']['development'][-1])
    return {'schema': SCHEMA, 'status': 'complete_pair', 'arm': 'NFR', 'arms': arms,
        'parent_development_update32': parent_eval, 'comparison': comparison, 'f64_context': context,
        'timing_scopes': common.TIMING_SCOPES, 'qualifications': common.QUALIFICATIONS + [
            'Native RT is active at layers 0 and 15 in both NFR branches; F64 is optional descriptive CE context only.',
            'Weighted objective totals are checked for bookkeeping but excluded from this summary, plots and comparison tables.'],
        'runtime_sources': control['sources'], 'independent_audit': {'passed': True,
            'sha256': pins['audit']['sha256'], 'checks': len(audit['checks'])},
        'checks': {'same_saved_parent_boundary': True, 'same_restored_raw_evaluation': True,
            'same_data_exposure_and_lr': True, 'common_development_prefix': True,
            'completed_retained_endpoints': True}}


def write_plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2166ac', '#b2182b', '#1b7837', '#762a83']
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(output / (name + '.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey='row')
    for col, (label, arm) in enumerate(summary['arms'].items()):
        xs = [row['after_update'] for row in arm['development']]
        for i, term in enumerate(TERMS):
            ax = axes[i, col]
            for p, color in enumerate(colors):
                ax.plot(xs, [row['passes'][p]['means'][term] for row in arm['development']], 'o-', color=color, label=f'Pass {p+1}')
            if i == 0 and summary['f64_context'] is not None:
                ax.scatter([64]*4, [p['ce'] for p in summary['f64_context']['passes']],
                           marker='x', color='black', label='F64 context')
            ax.set_title(f'NFR {label}: raw {term.upper()}'); ax.set_ylabel('Mean per eligible target')
            ax.grid(alpha=.25); ax.legend(fontsize=8)
            if i == 2: ax.set_xlabel('Cumulative optimizer update')
    fig.suptitle('NFR: common FP32 development, no jitter; fixed 64-row prefix')
    save(fig, 'development-raw-losses')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fields = [('gradient_norm_before_clip', 'Pre-clip gradient norm'),
              ('clip_coefficient_estimate', 'Clipping coefficient estimate'), ('lr_used', 'Learning rate'),
              ('ce', 'Training raw CE'), ('latent', 'Training raw latent'), ('kl', 'Training raw KL')]
    for label, arm in summary['arms'].items():
        for ax, (field, title) in zip(axes.flat, fields):
            ys = [row['loss_means'][field] if field in TERMS else row['lr_used'][0] if field == 'lr_used'
                  else row[field] for row in arm['updates']]
            ax.plot([row['update'] for row in arm['updates']], ys, label=label)
            ax.set_title(title); ax.set_xlabel('Cumulative optimizer update'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0, 0].set_yscale('symlog', linthresh=1.)
    fig.suptitle('NFR BF16 training; inherited Adam; objective totals omitted')
    save(fig, 'training-dynamics')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for label, arm in summary['arms'].items():
        xs = [row['update'] for row in arm['updates']]
        for ax, scope in zip(axes[:2], ('compute_regions', 'complete_update_callback')):
            ax.plot(xs, [row['input_tokens'] / row['seconds'][scope] for row in arm['updates']], label=label)
        axes[2].plot(xs, [max(m['reserved_gib'] for m in row['memory_by_rank']) for row in arm['updates']], label=label)
    for ax, title, unit in zip(axes, ('Compute regions', 'Full update callback', 'Max per-rank reserved memory'),
                               ('Input tokens/s', 'Input tokens/s', 'GiB')):
        ax.set_title(title); ax.set_ylabel(unit); ax.set_xlabel('Optimizer update'); ax.grid(alpha=.25); ax.legend()
    fig.suptitle('NFR scoped timing; callback includes due dev, both exclude checkpoint callbacks and startup')
    save(fig, 'timing-memory')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, arm in summary['arms'].items():
        xs = [row['after_update'] for row in arm['development']]
        for p in (0, 3):
            axes[0].plot(xs, [row['passes'][p]['means']['ce'] for row in arm['development']], 'o-', label=f'{label} pass {p+1}')
        axes[1].plot(xs, [row['ce_gap_vs_first_pass'][3] for row in arm['development']], 'o-', label=label)
    if summary['f64_context'] is not None:
        context = summary['f64_context']
        axes[0].scatter([64, 64], [context['passes'][p]['ce'] for p in (0, 3)], marker='x', color='black', label='F64 context')
        axes[1].scatter([64], [context['ce_gap_vs_first_pass'][3]], marker='x', color='black', label='F64 context')
    axes[0].set_title('Absolute first/final pass CE'); axes[1].set_title('Pass 4 minus pass 1 CE')
    for ax in axes:
        ax.set_ylabel('Nats per CE target'); ax.set_xlabel('Optimizer update'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('A smaller pass gap is useful only alongside absolute CE')
    save(fig, 'ce-refinement')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ('parent', 'control', 'reduced', 'audit', 'f64-evaluation', 'f64-publication'):
        parser.add_argument('--' + label, nargs=2, metavar=('JSON', 'SHA256'),
                            required=label in ('parent', 'control', 'reduced', 'audit'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require((args.f64_evaluation is None) == (args.f64_publication is None), 'Provide both F64 authorities or neither')
    inputs, pins = {}, {}
    for label in ('parent', 'control', 'reduced', 'audit', 'f64_evaluation', 'f64_publication'):
        argument = getattr(args, label)
        if argument is None:
            continue
        path, expected = argument; path = Path(path).resolve(); data = path.read_bytes()
        require(len(expected) == 64 and common.sha(data) == expected, 'Input SHA differs: ' + label)
        inputs[label] = (data, json.loads(data))
        pins[label] = {'path': str(path), 'sha256': expected, 'size_bytes': len(data),
                       'snapshot': 'input-snapshot/' + label + '.json'}
    summary = summarize(*(inputs[label][1] for label in ('control', 'reduced', 'parent', 'audit')), pins,
        f_evaluation=inputs.get('f64_evaluation', (None, None))[1],
        f_publication=inputs.get('f64_publication', (None, None))[1])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'input-snapshot').mkdir()
    for label, (data, _) in inputs.items():
        (args.output / pins[label]['snapshot']).write_bytes(data)
    summary['sources'] = {}
    for relative in SOURCE_NAMES:
        data = (ROOT / relative).read_bytes()
        path = args.output / 'source-snapshot' / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data); summary['sources'][relative] = common.sha(data)
    common.write_csvs(summary, args.output); write_plots(summary, args.output)
    summary['input_pins'] = pins
    summary['artifacts'] = {p.name: {'sha256': common.sha(p.read_bytes()), 'size_bytes': p.stat().st_size}
                            for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': summary['status'], 'report': str(args.output / 'report.json')}))


if __name__ == '__main__':
    main()
