#!/usr/bin/env python3
"""CPU figures from explicit closed RT reports; no token or checkpoint reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_tracking import OnlineTracker, add_wandb_arguments
from rt_precision_pair_eval import BOOTSTRAP_SEED, MARGIN, RESAMPLES, validate_protocol
from rt_precision_pilot_report import COLORS, POLICIES, matched_training

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_training_segments(training, protocol_sha, seeds, endpoint):
    """Bind resumed histories without reading checkpoints or incomplete runs."""
    common = None
    for data in training:
        if (data.get('schema') != 'rt-precision-training-v1' or data.get('status') != 'complete'
                or data['protocol_sha256'] != protocol_sha or data['seed'] not in seeds
                or data['endpoint'] not in (100, endpoint)
                or data['completed_updates'] != data['endpoint']):
            raise ValueError('Training report differs from the frozen protocol or completed endpoint')
        config = dict(data['model_config']); config.pop('recurrent_precision_policy')
        authority = {key: data[key] for key in ('data', 'schedule', 'optimizer', 'source_sha256',
                     'runtime', 'heldout_manifest_sha256', 'shape', 'head_chunk_size', 'loss',
                     'cuda_graphs', 'backbone_accumulation_steps', 'optimizer_captured', 'precision')}
        authority['model_config'] = config
        if common is not None and authority != common:
            raise ValueError('Training architecture, execution, data or source identity changes across seeds/segments')
        common = authority
        start, stop = data['starting_updates'], data['completed_updates']
        if start not in (0, 100) or start >= stop:
            raise ValueError('Unexpected retained training segment range')
        if [row['update'] for row in data['updates']] != list(range(start + 1, stop + 1)):
            raise ValueError('Closed training segment has missing, duplicate or unordered updates')
        batch, length = data['shape']
        for row in data['updates']:
            if (row['row_begin'] != (row['update'] - 1) * batch or row['row_end'] != row['update'] * batch
                    or not row['all_finite_fp32_parameters_gradients_moments']
                    or not all(math.isfinite(row[key]) for key in ('train_ce_b_times_t',
                               'train_ce_supervised_token', 'gradient_norm_before_clip', 'learning_rate'))
                    or not math.isclose(row['train_ce_supervised_token'],
                                        row['train_ce_b_times_t'] * length / (length - 1),
                                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('Training segment has invalid coverage, precision or CE normalization')
        if start:
            resume = data.get('resume', {})
            if resume.get('completed_updates') != start or resume.get('next_data_row') != start * batch:
                raise ValueError('Resumed segment lacks matching checkpoint/update identity')
            predecessors = [previous for previous in training if previous['seed'] == data['seed']
                            and previous['policy'] == data['policy'] and previous['endpoint'] == start]
            for previous in predecessors:
                endpoint_checkpoints = [item for item in previous['checkpoints']
                                        if item['completed_updates'] == start]
                if (len(endpoint_checkpoints) != 1 or endpoint_checkpoints[0]['sha256'] != resume.get('sha256')
                        or previous['final_model_sha256'] != data['starting_model_sha256']):
                    raise ValueError('Resumed curve is not continued from its supplied predecessor')


def validate_shared_sources(candidate, training):
    model_sources = {name for name in training if name.startswith('recurrent-transformer/olmo/')}
    if not model_sources or not model_sources <= set(candidate):
        raise ValueError('Summary input lacks complete model source identity')
    if any(candidate[name] != training[name] for name in candidate.keys() & training.keys()):
        raise ValueError('Summary input source identity differs from supplied training')


def bind_evaluation_training(data, training, training_sha):
    """A matching seed alone cannot establish which trained endpoint was tested."""
    identity = data['identity']
    bound = {}
    for policy, arm in POLICIES.items():
        matches = [index for index, item in enumerate(training)
                   if (item['seed'], item['policy'], item['endpoint']) ==
                   (identity['seed'], policy, identity['completed_updates'])]
        if len(matches) != 1:
            raise ValueError('Supply exactly one completed training endpoint per paired evaluation arm')
        index = matches[0]; item = training[index]
        if (training_sha[index] not in data['input_sha256'].values()
                or item['initial_state_sha256'] != identity['initial_state_sha256']
                or item['heldout_manifest_sha256'] not in data['input_sha256'].values()):
            raise ValueError('Paired summary does not reference the supplied training/data identity')
        validate_shared_sources(data['evaluation_source_sha256'], item['source_sha256'])
        bound[arm] = training_sha[index]
    return bound


def bind_numerical_training(data, training):
    matches = [item for item in training if item['seed'] == data['seed']
               and item['completed_updates'] == data['starting_optimizer']['step']
               and any(row['sha256'] == data['checkpoint']['sha256']
                       and row['completed_updates'] == item['completed_updates'] for row in item['checkpoints'])]
    if len(matches) != 1:
        raise ValueError('Numerical anchor must match exactly one supplied training endpoint checkpoint')
    item = matches[0]
    validate_shared_sources(data['source_sha256'], item['source_sha256'])
    if data['starting_model_digest'] != item['final_model_sha256']:
        raise ValueError('Numerical starting weights differ from their supplied training anchor')
    return {'policy': item['policy'], 'seed': item['seed'], 'completed_updates': item['completed_updates'],
            'checkpoint_sha256': data['checkpoint']['sha256']}


def evaluation_row(data, protocol_sha, seeds):
    """Validate the summary contract, trusting the paired driver's packet audit."""
    if data.get('schema') != 'rt-precision-paired-evaluation-v1' or data.get('status') != 'complete':
        raise ValueError('Require completed paired-analysis reports')
    identity, interval = data['identity'], data['bootstrap']
    if (identity['role'] not in ('dev', 'confirmation')
            or identity['precision'] not in ('fp32', 'native')
            or identity['seed'] not in seeds
            or identity['completed_updates'] not in (100, 500)
            or identity['common_fp32_primary'] != (identity['precision'] == 'fp32')):
        raise ValueError('Unexpected paired evaluation identity')
    if (data['margin_nats_per_supervised_token'] != MARGIN
            or interval['resamples'] != RESAMPLES or interval['seed'] != BOOTSTRAP_SEED
            or interval['quantile_method'] != 'linear'
            or protocol_sha not in data['input_sha256'].values()):
        raise ValueError('Paired analysis does not match the supplied frozen protocol')
    low, high = interval['two_sided_95_interval']
    point, upper = interval['candidate_minus_reference_nats_per_target'], interval['one_sided_95_upper']
    if (not all(math.isfinite(x) for x in (low, high, point, upper))
            or low > upper or upper > high
            or not math.isclose(data['ce_nats_per_supervised_token']['B']
                                - data['ce_nats_per_supervised_token']['A'], point,
                                rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError('Paired summary has invalid intervals or inconsistent CE')
    return {'identity': identity, 'bootstrap': interval,
            'ce_nats_per_supervised_token': data['ce_nats_per_supervised_token'],
            'reported_margin_assessment': data['primary_margin_assessment'],
            'upstream_input_sha256': data['input_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--paired-report', type=Path, action='append', required=True,
                        help='Completed paired-analysis JSON; repeat for each explicit role/seed/precision')
    parser.add_argument('--training-report', type=Path, action='append', required=True,
                        help='Completed training JSON; include both 100 and resumed 500 segments')
    parser.add_argument('--numerical-report', type=Path, action='append', default=[],
                        help='Optional completed trained-state A/B/C comparison JSON')
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh outcome-report directory')
    if not args.wandb_project:
        parser.error('Online W&B is required for graphable results')
    args.output_dir.mkdir(parents=True)
    report = {'schema': 'rt-precision-outcome-figures-v1', 'status': 'running',
              'input_sha256': {}, 'source_sha256': {}, 'figures': [],
              'policy_decision': 'No automatic policy promotion or rejection',
              'scope': 'Explicit completed reports only; per-seed document intervals are not pooled',
              'qualification': '100/500 updates are within the 5000-update warmup. No peak-LR or convergence clearance. Only report JSON is read, never token/document packets or checkpoints.'}
    tracker = None

    def snapshot(path, *, completed=True):
        path = path.resolve(); raw = path.read_bytes(); data = json.loads(raw)
        if completed and data.get('status') != 'complete':
            raise ValueError(f'Explicit input is not complete: {path}')
        sha = hashlib.sha256(raw).hexdigest()
        report['input_sha256'][str(path)] = sha
        target = args.output_dir / 'input-reports' / f'{sha}.json'
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        return data

    def save_figure(fig, name):
        for extension in ('png', 'svg'):
            path = args.output_dir / f'{name}.{extension}'
            fig.savefig(path, dpi=170, bbox_inches='tight')
            report['figures'].append({'path': str(path), 'sha256': digest(path)})
        plt.close(fig)

    try:
        for name in ('rt_precision_outcome_report.py', 'rt_precision_pilot_report.py',
                     'rt_precision_pair_eval.py', 'experiment_tracking.py'):
            path = ROOT / 'scripts' / name; raw = path.read_bytes()
            relative = str(path.relative_to(ROOT))
            report['source_sha256'][relative] = hashlib.sha256(raw).hexdigest()
            target = args.output_dir / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        protocol = snapshot(args.protocol, completed=False); validate_protocol(protocol)
        protocol_sha = report['input_sha256'][str(args.protocol.resolve())]
        seeds = protocol['training']['seeds']; endpoint = protocol['training']['continuation_updates']
        report['protocol_sha256'] = protocol_sha
        report['expected_seeds'] = seeds; report['expected_endpoint'] = endpoint
        training = [snapshot(path) for path in args.training_report]
        training_sha = [report['input_sha256'][str(path.resolve())] for path in args.training_report]
        if len(set(training_sha)) != len(training_sha):
            raise ValueError('Duplicate supplied training report')
        validate_training_segments(training, protocol_sha, seeds, endpoint)
        report['matched_training'] = matched_training(training)
        if not report['matched_training']:
            raise ValueError('At least one completed matched training pair is required')
        report['missing_matched_training_seeds'] = sorted(set(seeds) - {
            row['seed'] for row in report['matched_training']})
        histories = {}
        for data in training:
            history = histories.setdefault((data['seed'], POLICIES[data['policy']]), {})
            history.update({row['update']: row for row in data['updates']})
        for series in report['matched_training']:
            present = {row['update'] for row in series['rows']}
            series['missing_updates_through_endpoint'] = sorted(set(range(1, endpoint + 1)) - present)
            series['complete_through_endpoint'] = not series['missing_updates_through_endpoint']

        pairs = []
        evaluation_sources = None
        for path in args.paired_report:
            data = snapshot(path)
            row = evaluation_row(data, protocol_sha, seeds)
            row['training_report_sha256'] = bind_evaluation_training(data, training, training_sha)
            if evaluation_sources is not None and data['evaluation_source_sha256'] != evaluation_sources:
                raise ValueError('Evaluation source identity changes across supplied pairs')
            evaluation_sources = data['evaluation_source_sha256']
            pairs.append(row)
        pairs.sort(key=lambda row: tuple(row['identity'][key] for key in
                                        ('role', 'precision', 'seed', 'completed_updates', 'evaluation_batch')))
        identities = [tuple(row['identity'][key] for key in
                            ('role', 'precision', 'seed', 'completed_updates', 'evaluation_batch')) for row in pairs]
        if len(identities) != len(set(identities)):
            raise ValueError('Duplicate paired analysis identity')
        report['paired_evaluations'] = pairs
        report['coverage'] = [{
            'role': role, 'precision': precision, 'seed': seed, 'endpoint': endpoint,
            'supplied': any(all(row['identity'][key] == value for key, value in
                               [('role', role), ('precision', precision), ('seed', seed),
                                ('completed_updates', endpoint)]) for row in pairs)}
            for role in ('dev', 'confirmation') for precision in ('fp32', 'native') for seed in seeds]
        confirmation = [row for row in pairs if row['identity']['role'] == 'confirmation']
        report['confirmation_scope'] = ('Explicit confirmation summary reports supplied; coverage listed separately'
                                        if confirmation else 'No confirmation report supplied; confirmation unassessed')

        numerical = []
        for path in args.numerical_report:
            data = snapshot(path)
            if (data.get('schema') != 'rt-precision-comparison-v1'
                    or not data['starting_optimizer']['trained_moments']):
                raise ValueError('Numerical panel requires completed trained-state comparisons')
            numerical.append({'case': path.parent.name, 'seed': data['seed'], 'batch': data['batch'],
                              'training_anchor': bind_numerical_training(data, training),
                              'reference_scope': data['reference_scope'],
                              'starting_model_digest': data['starting_model_digest'],
                              'checkpoint': data['checkpoint'],
                              'anchor_step': data['starting_optimizer']['step'],
                              'saved_learning_rate': data['starting_optimizer']['group']['lr'],
                              'requires_review': data['requires_review'],
                              'comparisons': {arm: {
                                  'gradient_relative_l2': data['comparisons'][arm + '_vs_C']['raw_gradients']['relative_l2'],
                                  'adam_delta_relative_l2': data['comparisons'][arm + '_vs_C']['adam_delta']['relative_l2'],
                                  'gradient_tensor_flags': len(data['comparisons'][arm + '_vs_C']['raw_gradients']['tensor_review_flags']),
                                  'requires_review': data['comparisons'][arm + '_vs_C']['requires_review']}
                                  for arm in ('A', 'B')}})
        report['trained_numerics'] = numerical
        report['numerical_endpoint_coverage'] = [{
            'seed': seed, 'anchor_policy': policy, 'endpoint': endpoint, 'batch': 512,
            'supplied': any((row['training_anchor']['policy'], row['seed'], row['anchor_step'])
                == (policy, seed, endpoint)
                and row['batch'] == 512 and row['reference_scope'] == 'Full physical-batch FP32 reference'
                for row in numerical)} for seed in seeds for policy in POLICIES]

        series_list = report['matched_training']
        fig, axes = plt.subplots(2, len(series_list), figsize=(6.5 * len(series_list), 8), squeeze=False)
        for column, series in enumerate(series_list):
            steps = [row['update'] for row in series['rows']]; seed = series['seed']
            for arm in ('A', 'B'):
                rows = histories[seed, arm]
                axes[0, column].plot(steps, [rows[step]['train_ce_supervised_token'] for step in steps],
                                     color=COLORS[arm], linewidth=1., label=f'{arm}: ' +
                                     ('protected' if arm == 'A' else 'legacy'))
            delta = np.array([row['B_minus_A'] for row in series['rows']])
            axes[1, column].plot(steps, delta, color='#477baa', alpha=.65, linewidth=1., label='Matched batch')
            if len(delta) >= 10 and all(b - a == 1 for a, b in zip(steps, steps[1:])):
                axes[1, column].plot(steps[9:], np.convolve(delta, np.ones(10) / 10, mode='valid'),
                                     color='#174e7c', linewidth=1.5, label='Trailing 10 updates')
            axes[1, column].axhline(0, color='#555555', linewidth=1.)
            scope = 'complete to endpoint' if series['complete_through_endpoint'] else 'partial supplied history'
            axes[0, column].set(title=f'Seed {seed} · {scope}', ylabel='Native CE / supervised token')
            axes[1, column].set(xlabel='Optimizer update', ylabel='B − A native CE / supervised token')
            for ax in axes[:, column]:
                ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
        fig.suptitle('Matched RT training · separate initialization seeds')
        fig.tight_layout(rect=[0, 0, 1, .96]); save_figure(fig, 'training-curves')

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
        for i, role in enumerate(('dev', 'confirmation')):
            for j, precision in enumerate(('fp32', 'native')):
                ax = axes[i, j]
                rows = [row for row in pairs if row['identity']['role'] == role and row['identity']['precision'] == precision]
                for x, row in enumerate(rows):
                    identity, interval = row['identity'], row['bootstrap']
                    color = plt.get_cmap('tab10')(seeds.index(identity['seed']))
                    ax.vlines(x, *interval['two_sided_95_interval'], color=color, linewidth=2.)
                    ax.scatter(x, interval['candidate_minus_reference_nats_per_target'], color=color, zorder=3)
                    ax.scatter(x, interval['one_sided_95_upper'], color=color, marker='_', s=180, zorder=3)
                ax.axhline(0, color='#777777', linewidth=1.)
                ax.axhline(MARGIN, color='#a13131', linestyle='--', label='Frozen 0.005 margin')
                ax.set_xticks(range(len(rows)), [f"seed {row['identity']['seed']}\nu{row['identity']['completed_updates']} · B{row['identity']['evaluation_batch']}" for row in rows], fontsize=8)
                ax.set(title=f'{role.capitalize()} · ' + ('common FP32 (primary)' if precision == 'fp32' else 'native (secondary)'),
                       ylabel='B − A nats / supervised token')
                if not rows:
                    ax.text(.5, .55, 'No completed report supplied\nUnassessed', ha='center', transform=ax.transAxes)
                ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
        fig.suptitle('Per-seed paired document intervals · no pooling over seeds')
        fig.text(.5, .01, 'Dot: gap · line: two-sided 95% interval · cap: one-sided 95% upper. Document resampling does not quantify seed uncertainty.', ha='center', fontsize=9)
        fig.tight_layout(rect=[0, .035, 1, .96]); save_figure(fig, 'paired-evaluation-gaps')

        if numerical:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, key, title in zip(axes, ('gradient_relative_l2', 'adam_delta_relative_l2'),
                                      ('Global gradient error vs C', 'Applied Adam-delta error vs C')):
                x = np.arange(len(numerical))
                for arm, offset in [('A', -.18), ('B', .18)]:
                    values = [row['comparisons'][arm][key] for row in numerical]
                    if any(value is None or not math.isfinite(value) for value in values):
                        raise ValueError('Undefined numerical error needs an explanatory report')
                    ax.bar(x + offset, np.array(values) * 100, .36, color=COLORS[arm], label=arm)
                ax.axhline(.015625 * 100, color='#777777', linestyle='--', label='1.5625% review trigger')
                ax.set_xticks(x, [row['case'] for row in numerical], rotation=20, ha='right', fontsize=8)
                ax.set(title=title, ylabel='Relative L2 (%)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
            fig.suptitle('Conditional numerical checks at separate trained anchors')
            fig.tight_layout(rect=[0, 0, 1, .94]); save_figure(fig, 'trained-numerics')

        lines = ['# RT precision evidence from explicit completed reports', '', report['scope'] + '.', '',
                 '**' + report['confirmation_scope'] + '.**', '', report['qualification'], '',
                 'This report displays evidence and does not choose or promote a precision policy.', '',
                 '| Role | Evaluation | Seed | Update | B−A CE | Two-sided 95% interval | One-sided 95% upper |',
                 '| --- | --- | ---: | ---: | ---: | --- | ---: |']
        for row in pairs:
            identity, interval = row['identity'], row['bootstrap']; low, high = interval['two_sided_95_interval']
            lines.append(f"| {identity['role']} | {identity['precision']} | {identity['seed']} | {identity['completed_updates']} | {interval['candidate_minus_reference_nats_per_target']:.8f} | [{low:.8f}, {high:.8f}] | {interval['one_sided_95_upper']:.8f} |")
        lines += ['', 'The frozen margin is 0.005 nats per supervised token. Native evaluation is secondary and includes inference-rounding differences. Each interval is conditional on one trained seed; no pooled interval or seed-variance estimate is constructed.', '',
                  '| Seed | Paired supplied updates | Complete through endpoint | Final paired training gap |',
                  '| ---: | --- | --- | ---: |']
        for series in series_list:
            steps = [row['update'] for row in series['rows']]
            lines.append(f"| {series['seed']} | {min(steps)}–{max(steps)} ({len(steps)} updates) | {series['complete_through_endpoint']} | {series['last_difference']:.8f} |")
        missing = [row for row in report['coverage'] if not row['supplied']]
        missing_numerics = [row for row in report['numerical_endpoint_coverage'] if not row['supplied']]
        lines += ['', 'Missing matched training seeds: ' + (', '.join(map(str, report['missing_matched_training_seeds'])) or 'none') + '.', '',
                  'Missing endpoint evaluation coverage: ' + (', '.join(f"{row['role']}/{row['precision']}/seed{row['seed']}" for row in missing) if missing else 'none among the frozen roles, policies and seeds') + '.', '',
                  'Missing physical-B512 endpoint numerical anchors: ' + (', '.join(f"{POLICIES[row['anchor_policy']]}-trained/seed{row['seed']}" for row in missing_numerics) if missing_numerics else 'none') + '.', '',
                  'Trained-state numerical inputs: ' + (', '.join(row['case'] for row in numerical) if numerical else 'none supplied; unassessed') + '.', '',
                  'Numerical inputs retain their own reference batch and saved optimizer learning rate; they are conditional comparisons, not differences between separately trained checkpoints. Review flags remain in report.json.', '',
                  'Input report hashes and snapshots are retained with this build. Token/document pairing validation and raw-packet checks remain the responsibility of the linked upstream reports; this figure build does not repeat them.', '']
        if numerical:
            lines += ['| Anchor | Objective batch | Saved LR | A/C gradient L2 | B/C gradient L2 | A/C applied delta L2 | B/C applied delta L2 | Review flagged |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
            for row in numerical:
                a, b = row['comparisons']['A'], row['comparisons']['B']
                lines.append(f"| {row['case']} | {row['batch']} | {row['saved_learning_rate']:.8g} | {a['gradient_relative_l2'] * 100:.4f}% | {b['gradient_relative_l2'] * 100:.4f}% | {a['adam_delta_relative_l2'] * 100:.4f}% | {b['adam_delta_relative_l2'] * 100:.4f}% | {row['requires_review']} |")
            lines.append('')
        for figure in report['figures']:
            if figure['path'].endswith('.png'):
                lines += [f"![{Path(figure['path']).stem}]({Path(figure['path']).name})", '']
        (args.output_dir / 'results.md').write_text('\n'.join(lines))
        for path, expected in report['input_sha256'].items():
            if digest(path) != expected:
                raise ValueError('Input changed during report build')
        for path, expected in report['source_sha256'].items():
            if digest(ROOT / path) != expected:
                raise ValueError('Report source changed during build')
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                                name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({'scope': report['scope'], 'input_sha256': report['input_sha256'],
                       'source_sha256': report['source_sha256'], 'confirmation_scope': report['confirmation_scope']})
        import wandb
        tracker.log({f"outcome/{Path(row['path']).stem}": wandb.Image(row['path'])
                     for row in report['figures'] if row['path'].endswith('.png')})
        tracker.summary({'outcome/paired_reports': len(pairs), 'outcome/training_pairs': len(series_list),
                         'outcome/numerical_anchors': len(numerical), 'outcome/confirmation_reports': len(confirmation)})
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='execution_failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        try:
            if tracker is not None:
                tracker.finish(succeeded=report['status'] == 'complete')
        except BaseException as error:
            report.update(status='execution_failed', sync_error_type=type(error).__name__); raise
        finally:
            (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'figures': report['figures'],
                      'confirmation_scope': report['confirmation_scope']}), flush=True)


if __name__ == '__main__':
    main()
