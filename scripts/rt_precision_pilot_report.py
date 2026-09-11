#!/usr/bin/env python3
"""CPU figures exposing closed RT pilot differences; confirmation stays unopened."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_tracking import OnlineTracker, add_wandb_arguments

POLICIES = {'bf16_fp32_state': 'A', 'legacy': 'B'}
COLORS = {'A': '#2475ad', 'B': '#d87517'}
MARGIN = .005
REVIEW = .015625


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched_training(reports):
    series, authorities = {}, {}
    for data in reports:
        if data.get('schema') != 'rt-precision-training-v1' or data.get('status') != 'complete':
            raise ValueError('Training figures require closed training reports')
        key = (data['seed'], POLICIES[data['policy']])
        authority = {name: data[name] for name in ('initial_state_sha256', 'data', 'schedule', 'optimizer', 'source_sha256')}
        if key in authorities and authorities[key] != authority:
            raise ValueError('Training identity changes across retained segments')
        authorities[key] = authority
        rows = series.setdefault(key, {})
        for row in data['updates']:
            update = row['update']
            if update in rows and rows[update] != row:
                raise ValueError('Conflicting histories for one seed/arm/update')
            rows[update] = row
    result = []
    for seed in sorted({key[0] for key in series}):
        if (seed, 'A') not in series or (seed, 'B') not in series:
            continue
        if authorities[seed, 'A'] != authorities[seed, 'B']:
            raise ValueError('Training curves do not share initialization/data/schedule/source')
        a, b = series[seed, 'A'], series[seed, 'B']
        common = sorted(set(a) & set(b))
        if not common:
            continue
        rows = []
        for update in common:
            for field in ('row_begin', 'row_end', 'learning_rate'):
                if a[update][field] != b[update][field]:
                    raise ValueError('Training difference uses unmatched data rows or learning rate')
            rows.append({'update': update, 'B_minus_A': b[update]['train_ce_supervised_token'] - a[update]['train_ce_supervised_token']})
        result.append({'seed': seed, 'rows': rows, 'A_available_updates': len(a), 'B_available_updates': len(b),
                       'paired_updates': len(rows), 'last_difference': rows[-1]['B_minus_A'],
                       'mean_last_20_difference': float(np.mean([row['B_minus_A'] for row in rows[-20:]]))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lineage', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project='rt-precision-alignment')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh pilot figure-build directory')
    if not args.wandb_project:
        parser.error('Online W&B is required for these graphable results')
    args.output_dir.mkdir(parents=True)
    report = {'schema': 'rt-precision-pilot-figures-v1', 'status': 'running',
              'sources': {}, 'source_sha256': {}, 'skipped': [], 'figures': [],
              'scope': 'Completed paired training segments, development pairs and trained-state numerical checks only',
              'qualification': 'Development at update100 is a screen, not final confirmation or convergence. Numerical review triggers are not automatic failures. No confirmation inputs are opened.'}
    tracker = None

    def read_closed(path):
        if not path.exists():
            report['skipped'].append({'path': str(path), 'reason': 'no final report'})
            return None
        raw = path.read_bytes(); data = json.loads(raw)
        if data.get('status') != 'complete':
            report['skipped'].append({'path': str(path), 'reason': data.get('status', 'missing status')})
            return None
        sha = hashlib.sha256(raw).hexdigest()
        report['sources'][str(path.resolve())] = sha
        target = args.output_dir / 'source-reports' / f'{sha}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
        return data

    try:
        root = Path(__file__).resolve().parents[1]
        for source in (Path(__file__), root / 'scripts/experiment_tracking.py'):
            name = str(source.relative_to(root)); raw = source.read_bytes()
            report['source_sha256'][name] = hashlib.sha256(raw).hexdigest()
            destination = args.output_dir / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(raw)
        training = []
        for directory in sorted((args.lineage / 'train').glob('*')):
            if directory.is_dir():
                data = read_closed(directory / 'report.json')
                if data is not None:
                    training.append(data)
        report['matched_training'] = matched_training(training)

        evaluations = []
        # Scan only directories explicitly named as development pairs. Never
        # enumerate/read the confirmation data or confirmation evaluation tree.
        for directory in sorted((args.lineage / 'analysis').glob('pair-*-dev-*')):
            if not directory.is_dir():
                continue
            data = read_closed(directory / 'report.json')
            if data is None:
                continue
            if data.get('schema') != 'rt-precision-paired-evaluation-v1' or data['identity']['role'] != 'dev':
                raise ValueError('Development figure input has an unexpected role/schema')
            if (data['margin_nats_per_supervised_token'] != MARGIN
                    or data['bootstrap']['resamples'] != 10000 or data['bootstrap']['seed'] != 20260912):
                raise ValueError('Paired interval or margin differs from the frozen protocol')
            evaluations.append({'identity': data['identity'], 'bootstrap': data['bootstrap'],
                                'assessment': data['primary_margin_assessment'],
                                'ce_nats_per_supervised_token': data['ce_nats_per_supervised_token']})
        evaluations.sort(key=lambda row: (row['identity']['seed'], row['identity']['completed_updates'],
                                          row['identity']['precision'], row['identity']['evaluation_batch']))
        identities = [(row['identity']['seed'], row['identity']['completed_updates'], row['identity']['precision'],
                       row['identity']['evaluation_batch']) for row in evaluations]
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate paired development result; choose one explicit closed report build')
        report['development_pairs'] = evaluations

        numerical = []
        for directory in sorted((args.lineage / 'numerics').glob('trained-*')):
            if not directory.is_dir():
                continue
            data = read_closed(directory / 'report.json')
            if data is None:
                continue
            if (data.get('schema') != 'rt-precision-comparison-v1'
                    or not data['starting_optimizer']['trained_moments']):
                raise ValueError('Trained-state figures require nonempty trained Adam anchors')
            match = re.fullmatch(r'trained-([AB])-seed(\d+)-(\d+)-b(\d+)', directory.name)
            label = (f"{match[1]}-trained · seed {match[2]}\nu{match[3]} · B{data['batch']}"
                     if match else directory.name)
            numerical.append({'case': directory.name, 'label': label, 'batch': data['batch'],
                              'anchor_step': data['starting_optimizer']['step'], 'requires_review': data['requires_review'],
                              'comparisons': {arm: {
                                  'gradient_relative_l2': data['comparisons'][arm + '_vs_C']['raw_gradients']['relative_l2'],
                                  'adam_delta_relative_l2': data['comparisons'][arm + '_vs_C']['adam_delta']['relative_l2'],
                                  'requires_review': data['comparisons'][arm + '_vs_C']['requires_review'],
                                  'flagged_gradient_tensors': len(data['comparisons'][arm + '_vs_C']['raw_gradients']['tensor_review_flags'])}
                                  for arm in ('A', 'B')}})
        report['trained_numerics'] = numerical
        if not report['matched_training'] and not evaluations and not numerical:
            raise ValueError('No completed paired pilot evidence to plot')

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax = axes[0, 0]
        for index, series in enumerate(report['matched_training']):
            steps = [row['update'] for row in series['rows']]
            delta = np.array([row['B_minus_A'] for row in series['rows']])
            color = plt.get_cmap('tab10')(index)
            ax.plot(steps, delta, color=color, alpha=.55, linewidth=1., label=f"seed {series['seed']}")
            if len(delta) >= 10:
                ax.plot(steps[9:], np.convolve(delta, np.ones(10) / 10, mode='valid'), color=color, linewidth=2.)
        ax.axhline(0, color='#555555', linewidth=1.)
        ax.set(title='Matched training CE difference: B − A', xlabel='Optimizer update', ylabel='Native CE difference / supervised token')
        if report['matched_training']:
            ax.legend(fontsize=8); ax.text(.02, .03, 'Thin: each batch · thick: trailing 10 updates', transform=ax.transAxes, fontsize=8)
        else:
            ax.text(.5, .5, 'No completed matched training pair', ha='center', transform=ax.transAxes)

        ax = axes[0, 1]
        ax.axhline(0, color='#777777', linewidth=1.)
        ax.axhline(MARGIN, color='#a13131', linestyle='--', linewidth=1.3, label='Frozen 0.005 margin')
        labels = []
        for index, row in enumerate(evaluations):
            identity, interval = row['identity'], row['bootstrap']
            color = '#2475ad' if identity['precision'] == 'fp32' else '#d87517'
            lower, upper = interval['two_sided_95_interval']
            ax.vlines(index, lower, upper, color=color, linewidth=2.)
            ax.scatter(index, interval['candidate_minus_reference_nats_per_target'], color=color, s=40, zorder=3)
            ax.scatter(index, interval['one_sided_95_upper'], color=color, marker='_', s=180, zorder=4)
            labels.append(f"seed {identity['seed'] - 20260910} · u{identity['completed_updates']}\n{identity['precision']} · eval B{identity['evaluation_batch']}")
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set(title='Development CE gap: B − A', ylabel='Nats per supervised token')
        ax.legend(fontsize=8, loc='best')
        ax.text(.02, .03, 'Dot: gap · line: two-sided 95% CI · cap: one-sided 95% upper', transform=ax.transAxes, fontsize=8)
        if not evaluations:
            ax.text(.5, .5, 'No completed development pair', ha='center', transform=ax.transAxes)

        for ax, key, title in [(axes[1, 0], 'gradient_relative_l2', 'Trained anchors: global gradient error vs FP32'),
                               (axes[1, 1], 'adam_delta_relative_l2', 'Trained anchors: actual Adam-delta error vs FP32')]:
            x = np.arange(len(numerical)); width = .35
            for arm, offset in [('A', -width / 2), ('B', width / 2)]:
                values = [row['comparisons'][arm][key] for row in numerical]
                if any(value is None or not np.isfinite(value) for value in values):
                    raise ValueError('Undefined trained-state error requires a separate explanatory figure')
                ax.bar(x + offset, np.array(values) * 100, width, color=COLORS[arm],
                       label='Protected arithmetic' if arm == 'A' else 'Legacy arithmetic')
            ax.axhline(REVIEW * 100, color='#777777', linestyle='--', linewidth=1., label='1.5625% review trigger')
            ax.set_xticks(x, [row['label'] for row in numerical], fontsize=8)
            ax.set(title=title, ylabel='Relative L2 (%)')
            ax.legend(fontsize=8)
            if not numerical:
                ax.text(.5, .5, 'Trained-state checks not closed yet', ha='center', transform=ax.transAxes)
        for ax in axes.flat:
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        fig.suptitle('RT precision pilot · closed development evidence only', fontsize=14)
        fig.text(.5, .01, '100/500 updates are within the 5000-update warmup. Document intervals do not measure training-seed uncertainty. Confirmation untouched.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=[0, .035, 1, .96])
        for extension in ('png', 'svg'):
            path = args.output_dir / f'pilot-differences.{extension}'
            fig.savefig(path, dpi=170, bbox_inches='tight')
            report['figures'].append({'path': str(path), 'sha256': digest(path)})
        plt.close(fig)
        for path, expected in report['sources'].items():
            if digest(path) != expected:
                raise ValueError('Closed input report changed during figure build')
        for name, expected in report['source_sha256'].items():
            if digest(root / name) != expected:
                raise ValueError('Plotting source changed during figure build')
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name, output_dir=args.output_dir)
        report['wandb'] = tracker.record
        tracker.start({'scope': report['scope'], 'input_reports': report['sources'],
                       'source_sha256': report['source_sha256'], 'skipped_cases': report['skipped']})
        import wandb
        tracker.log({'pilot/differences': wandb.Image(str(args.output_dir / 'pilot-differences.png'))})
        tracker.summary({'pilot/closed_training_pairs': len(report['matched_training']),
                         'pilot/closed_dev_pairs': len(evaluations), 'pilot/closed_trained_anchors': len(numerical)})
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
                      'closed_trained_anchors': len(numerical), 'skipped': report['skipped']}), flush=True)


if __name__ == '__main__':
    main()
