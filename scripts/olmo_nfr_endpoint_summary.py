"""Compare the two authenticated NFR64 saved-state pass probes (CPU only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pair(paths):
    records, common = [], None
    for path in paths:
        record = json.loads(path.read_text())
        if (record.get('schema') != 'olmo-nfr-endpoint-curves-v1'
                or record.get('status') != 'completed' or record.get('arm') != 'NFR'
                or record.get('after_update') != 64 or record.get('kl_weight') not in (1., .1)):
            raise ValueError('Require completed explicitly scoped NFR64 endpoint probes')
        result = record['result']
        signature = (record['membership_sha256'], record['index_manifest_sha256'],
                     result['input_tokens'], result['ce_targets'], result['policy'], result['beta'])
        if common is None:
            common = signature
        if (signature != common or result['policy'] != 'common_fp32_no_jitter_v1'
                or result['beta'] != 1. or [p['pass'] for p in result['passes']] != list(range(1, 33))):
            raise ValueError('Mismatched panel, masks, precision, fusion strength or pass coverage')
        records.append({'path': path, 'sha256': digest(path), 'record': record,
                        'label': f'NFR64 KL {record["kl_weight"]:g}'})
    if len(records) != 2 or {r['record']['kl_weight'] for r in records} != {1., .1}:
        raise ValueError('Require exactly one control and one reduced-KL endpoint')
    return sorted(records, key=lambda r: -r['record']['kl_weight'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs=2, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(argv)
    records = load_pair(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = args.output_dir/'input-snapshot'
    snapshots.mkdir()
    rows, authorities, key_metrics = [], [], []
    for i, item in enumerate(records):
        record = item['record']
        target = snapshots/f'{i:02d}-report.json'
        shutil.copyfile(item['path'], target)
        if digest(target) != item['sha256']:
            raise ValueError('Input changed during snapshot')
        authorities.append({'path': str(item['path']), 'sha256': item['sha256'],
                            'label': item['label'], 'snapshot': str(target.relative_to(args.output_dir))})
        passes = record['result']['passes']
        for p in passes:
            for name, region in p['regions'].items():
                rows.append({'label': item['label'], 'kl_weight': record['kl_weight'],
                             'pass': p['pass'], 'region': name, **region['metrics']})
        key_metrics.append({'label': item['label'], 'kl_weight': record['kl_weight'],
            'ce_by_pass': {str(k): passes[k-1]['regions']['all']['metrics']['ce'] for k in (1, 4, 8, 32)},
            'tail_relative_change_by_pass': {str(k): passes[k-1]['regions']['tail_128']['metrics']['relative_delta_rms']
                                             for k in (4, 8, 16, 32)},
            'finite_residuals': record['finite_residuals']})
    columns = list(rows[0])
    with (args.output_dir/'endpoint-curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True, sharex=True, sharey='col')
    for i, item in enumerate(records):
        color = plt.get_cmap('tab10')(i)
        for row, (ce_region, change_region) in enumerate((('all', 'unsettled_suffix'), ('tail_128', 'tail_128'))):
            ce = [r for r in rows if r['label'] == item['label'] and r['region'] == ce_region]
            delta = [r for r in rows if r['label'] == item['label'] and r['region'] == change_region
                     and r['relative_delta_rms'] is not None]
            axes[row, 0].plot([r['pass'] for r in ce], [r['ce'] for r in ce], color=color, label=item['label'])
            axes[row, 1].plot([r['pass'] for r in delta], [max(1e-12, r['relative_delta_rms']) for r in delta], color=color)
    for row in range(2):
        axes[row, 0].set(ylabel='CE (nats / eligible target)', title='All CE targets' if row == 0 else 'Final 128 positions')
        axes[row, 1].set(ylabel='RMS(state change) / RMS(previous)', yscale='log',
                         title='Unsettled state suffix' if row == 0 else 'Final 128 positions')
        for axis in axes[row]:
            axis.axvline(4, color='grey', linestyle=':', linewidth=.8)
            axis.set_xlabel('Total feedback pass (training uses 4)')
            axis.grid(alpha=.2)
    axes[0, 0].legend()
    figure.suptitle('NFR64 saved checkpoints: settling and predictive loss\n'
                   'Same eight packed T1024 rows; FP32; no jitter; no optimizer updates', fontsize=11)
    for suffix in ('pdf', 'png'):
        figure.savefig(args.output_dir/f'endpoint-curves.{suffix}', dpi=165)
    plt.close(figure)
    tracking = None
    if args.publish:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', entity='taylorbollman',
            output_dir=args.output_dir, group='nfr-stability-128', name='nfr64-paired-pass-curves')
        tracker.start({'scope': 'Matched saved-state settling diagnostics, not quality or BF16 clearance',
                       'inputs': authorities})
        tracker.log({'endpoint-curves': wandb.Image(str(args.output_dir/'endpoint-curves.png')),
                     'pass_curves': wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in rows])})
        tracker.finish(succeeded=True)
        tracking = tracker.record
    source_target = args.output_dir/'source-snapshot'/Path(__file__).name
    source_target.parent.mkdir()
    shutil.copyfile(__file__, source_target)
    report = {'schema': 'olmo-nfr-endpoint-summary-v1', 'status': 'completed',
        'inputs': authorities, 'key_metrics': key_metrics, 'tracking': tracking,
        'source_sha256': digest(source_target), 'optimizer_updates_performed': 0,
        'qualification': 'K32 is a finite reference, not exact online; settling does not establish useful refinement.',
        'artifacts': {p.name: {'sha256': digest(p), 'size_bytes': p.stat().st_size}
                      for p in args.output_dir.iterdir() if p.suffix in ('.csv', '.pdf', '.png')}}
    for item in records:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Input changed during analysis')
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'output': str(args.output_dir), 'tracking': tracking, 'key_metrics': key_metrics}))


if __name__ == '__main__':
    main()
