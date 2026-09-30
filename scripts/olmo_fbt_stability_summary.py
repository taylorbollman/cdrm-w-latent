"""Plot immutable completed pass probes; no model execution or optimizer updates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probes(paths):
    records = []
    for path in paths:
        raw = path.read_bytes()
        record = json.loads(raw)
        if record.get('status') != 'completed':
            raise ValueError('Only completed immutable probes can be plotted: '+str(path))
        result = record['result']
        if not result['passes']:
            raise ValueError('Missing pass observations')
        records.append((path, hashlib.sha256(raw).hexdigest(), record))
    records.sort(key=lambda row: row[2]['after_update'])
    updates = [row[2]['after_update'] for row in records]
    if len(set(updates)) != len(updates):
        raise ValueError('Do not silently combine repeat evaluations of one update')
    first = records[0][2]
    for _path, _pin, record in records:
        if (record.get('training_boundary_exact_by_rank') != [True, True]
                or any(record.get(key) != first.get(key) for key in
                       ('membership_sha256', 'index_manifest_sha256', 'panel'))
                or any(record['result'][key] != first['result'][key] for key in
                       ('input_tokens', 'ce_targets', 'beta', 'policy'))):
            raise ValueError('Only preserved, matched-panel evaluations belong on one curve figure')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probes', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    records = load_probes(args.probes)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = args.output/'input-snapshot'
    snapshots.mkdir()
    rows, inputs = [], []
    for path, pin, record in records:
        copied = snapshots/f"update-{record['after_update']:06d}.json"
        shutil.copyfile(path, copied)
        if digest(copied) != pin:
            raise ValueError('Input changed while snapshotting')
        inputs.append({'path': str(path), 'sha256': pin, 'snapshot': str(copied.relative_to(args.output))})
        for observation in record['result']['passes']:
            for region, values in observation['regions'].items():
                rows.append({'update': record['after_update'], 'pass': observation['pass'],
                             'region': region, **values['metrics']})
    columns = list(rows[0])
    with (args.output/'pass-curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    updates = [r[2]['after_update'] for r in records]
    colors = {update: plt.get_cmap('viridis')(i/max(1, len(updates)-1))
              for i, update in enumerate(updates)}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True,
                                sharey='col', constrained_layout=True)
    for q, (ce_region, delta_region, title) in enumerate([
            ('all', 'unsettled_suffix', 'All targets / unsettled state suffix'),
            ('tail_128', 'tail_128', 'Last 128 token positions')]):
        for update in updates:
            ce = [r for r in rows if r['update'] == update and r['region'] == ce_region]
            delta = [r for r in rows if r['update'] == update and r['region'] == delta_region
                     and r['relative_delta_rms'] is not None]
            axes[q, 0].plot([r['pass'] for r in ce], [r['ce'] for r in ce],
                            color=colors[update], label=f'update {update}', linewidth=1.6)
            axes[q, 1].plot([r['pass'] for r in delta],
                            [max(1e-12, r['relative_delta_rms']) for r in delta],
                            color=colors[update], linewidth=1.6)
        axes[q, 0].set(title=title, ylabel='CE (nats / eligible target)')
        axes[q, 1].set(title=title, ylabel='RMS(state change) / RMS(previous state)', yscale='log')
        for axis in axes[q]:
            axis.axvline(4, color='grey', linestyle=':', linewidth=1)
            axis.grid(alpha=.2)
            axis.set_xlabel('Total pass (1 = ordinary; training uses 4)')
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.suptitle('FBT-only: held-out repeated-pass behavior at saved training boundaries\n'
                   'Common FP32, no jitter; shrinking state changes are not a proof of contraction', fontsize=11)
    for suffix in ('pdf', 'png'):
        figure.savefig(args.output/f'figure3-style.{suffix}', dpi=165)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for index, (metric, label) in enumerate([
            ('pre_norm_rms', 'Pre-final-normalization hidden RMS'),
            ('entropy', 'Predictive entropy (nats)'),
            ('hidden_rms', 'Final hidden RMS'),
            ('input_rms', 'Stack input RMS')]):
        axis = axes.flat[index]
        for update in updates:
            values = [r for r in rows if r['update'] == update and r['region'] == 'all']
            axis.plot([r['pass'] for r in values], [r[metric] for r in values],
                      color=colors[update], label=f'update {update}')
        axis.set(xlabel='Total pass', ylabel=label)
        axis.axvline(4, color='grey', linestyle=':', linewidth=1)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.suptitle('Scale and entropy accompany the stability curves; low state change alone is insufficient', fontsize=11)
    for suffix in ('pdf', 'png'):
        figure.savefig(args.output/f'scale-entropy.{suffix}', dpi=165)
    plt.close(figure)

    tracking = None
    if args.publish:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', entity='taylorbollman',
            output_dir=args.output, group='fbt-stability',
            name=f'fbt-stability-curves-through-{max(updates):06d}')
        tracker.start({'scope': 'Completed held-out curve snapshots; no training or quality clearance',
                       'updates': updates, 'inputs': inputs})
        tracker.log({'figure3-style': wandb.Image(str(args.output/'figure3-style.png')),
                     'scale-entropy': wandb.Image(str(args.output/'scale-entropy.png')),
                     'pass_curves': wandb.Table(columns=columns,
                         data=[[row[key] for key in columns] for row in rows])})
        tracker.finish(succeeded=True)
        tracking = tracker.record
    source = 'scripts/olmo_fbt_stability_summary.py'
    target = args.output/'source-snapshot'/source
    target.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__), target)
    result = {'schema': 'olmo-fbt-stability-summary-v1', 'status': 'completed',
              'updates': updates, 'inputs': inputs, 'sources': {source: digest(target)},
              'tracking': tracking, 'artifacts': {p.name: {'sha256': digest(p), 'size_bytes': p.stat().st_size}
                  for p in args.output.iterdir() if p.suffix in ('.csv', '.pdf', '.png')},
              'scope': 'Fixed held-out panel; one training seed; finite-pass diagnostics; no global contraction claim'}
    for path, pin, _record in records:
        if digest(path) != pin:
            raise ValueError('Input mutated during analysis')
    (args.output/'report.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'updates': updates, 'output': str(args.output), 'tracking': tracking}))


if __name__ == '__main__':
    main()
