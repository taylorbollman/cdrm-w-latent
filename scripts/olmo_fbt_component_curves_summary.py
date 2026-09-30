"""CPU-only F/NF/NFR curve overlays with strict common-panel checks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


SCHEMAS = ('olmo-fbt-stability-probe-v1', 'olmo-fbt-component-curves-v1')


def load_inputs(paths):
    records = []; common = None; seen = set()
    for path in paths:
        raw = path.read_bytes(); record = json.loads(raw)
        if record.get('schema') not in SCHEMAS or record.get('status') != 'completed':
            raise ValueError('Only completed F or saved-component pass curves are accepted')
        result = record['result']; arm = record.get('arm', 'F'); update = record['after_update']
        weight = record.get('kl_weight') if arm != 'F' else None
        if arm not in ('F', 'NF', 'NFR') or (arm != 'F' and weight not in (1., .1)):
            raise ValueError('Unknown condition or missing NextLat objective identity')
        signature = (record['membership_sha256'], record['index_manifest_sha256'],
                     result['input_tokens'], result['ce_targets'], result['policy'], result['beta'])
        if common is None: common = signature
        if signature != common or result['policy'] != 'common_fp32_no_jitter_v1' or result['beta'] != 1.:
            raise ValueError('Comparison would mix panel membership, masks, precision or feedback strength')
        passes = [p['pass'] for p in result['passes']]
        if not passes or passes != list(range(1, max(passes)+1)) or max(passes) > 32:
            raise ValueError('Missing, unordered or unsupported pass observations')
        key = (arm, update, weight)
        if key in seen:
            raise ValueError('Duplicate condition/update: select one explicit complete input')
        seen.add(key)
        label = f'{arm} update {update}'+('' if weight is None else f' KL {weight:g}')
        records.append({'path': path, 'sha256': hashlib.sha256(raw).hexdigest(),
            'record': record, 'label': label, 'arm': arm, 'update': update, 'kl_weight': weight})
    if not records:
        raise ValueError('At least one completed input is required')
    return sorted(records, key=lambda r: (('F', 'NF', 'NFR').index(r['arm']), r['update'], r['kl_weight'] or 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    records = load_inputs(args.inputs); args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = args.output_dir/'input-snapshot'; snapshots.mkdir()
    rows = []; authorities = []
    for index, item in enumerate(records):
        target = snapshots/f'{index:02d}-{item["arm"]}-update-{item["update"]:06d}.json'
        shutil.copyfile(item['path'], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Input changed during immutable snapshot')
        authorities.append({'path': str(item['path']), 'sha256': item['sha256'], 'label': item['label'],
                            'snapshot': str(target.relative_to(args.output_dir))})
        for p in item['record']['result']['passes']:
            for name, region in p['regions'].items():
                rows.append({'label': item['label'], 'arm': item['arm'], 'update': item['update'],
                    'kl_weight': item['kl_weight'], 'pass': p['pass'], 'region': name, **region['metrics']})
    with (args.output_dir/'component-curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True, sharey='col')
    for index, item in enumerate(records):
        color = plt.get_cmap('tab10')(index % 10)
        style = {'F': '-', 'NF': '--', 'NFR': ':'}[item['arm']]
        for row, (ce_region, delta_region) in enumerate((('all', 'unsettled_suffix'), ('tail_128', 'tail_128'))):
            ce = [r for r in rows if r['label'] == item['label'] and r['region'] == ce_region]
            delta = [r for r in rows if r['label'] == item['label'] and r['region'] == delta_region
                     and r['relative_delta_rms'] is not None]
            axes[row, 0].plot([r['pass'] for r in ce], [r['ce'] for r in ce],
                color=color, linestyle=style, label=item['label'])
            axes[row, 1].plot([r['pass'] for r in delta], [max(1e-12, r['relative_delta_rms']) for r in delta],
                color=color, linestyle=style)
    for row in range(2):
        axes[row, 0].set(ylabel='CE (nats / eligible target)',
                         title='All CE targets' if row == 0 else 'Final 128 positions')
        axes[row, 1].set(ylabel='RMS(state change) / RMS(previous)', yscale='log',
                         title='Unsettled state suffix' if row == 0 else 'Final 128 positions')
        for axis in axes[row]:
            axis.axvline(4, color='grey', linestyle=':', linewidth=.8)
            axis.set_xlabel('Total pass (training uses 4)'); axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle('Saved F / NF / NFR pass dynamics on identical packed development rows\n'
                   'Common FP32; no jitter; NextLat predictor not executed; no quality or BF16 clearance', fontsize=11)
    for suffix in ('pdf', 'png'):
        figure.savefig(args.output_dir/f'component-curves.{suffix}', dpi=165)
    plt.close(figure)
    report = {'schema': 'olmo-fbt-component-curves-summary-v1', 'status': 'completed',
        'inputs': authorities, 'condition_count': len(records), 'optimizer_updates_performed': 0,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'artifacts': {name: hashlib.sha256((args.output_dir/name).read_bytes()).hexdigest()
                      for name in ('component-curves.csv', 'component-curves.pdf', 'component-curves.png')}}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
