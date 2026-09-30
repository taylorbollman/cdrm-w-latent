"""Audit-bound CPU summary of the unchanged reduced-KL NFR64-to128 continuation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if not math.isfinite(value):
        raise ValueError('Nonfinite reported metric')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent64', 'continuation', 'audit', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(argv)
    inputs = {name: {'path': str(getattr(args, name)), 'sha256': digest(getattr(args, name))}
              for name in ('parent64', 'continuation', 'audit')}
    parent, run, audit = [json.loads(getattr(args, name).read_text()) for name in ('parent64', 'continuation', 'audit')]
    if (audit.get('schema') != 'olmo-nfr-128-audit-v1' or audit.get('passed') is not True
            or audit.get('failures') or not audit.get('checks')
            or not all(check['passed'] is True for check in audit['checks'])
            or any(audit['inputs'][key]['sha256'] != inputs[key]['sha256'] for key in ('parent64', 'continuation'))):
        raise ValueError('Require a passed independent audit bound to these exact reports')
    if (run['status'] != 'completed_plan' or run['final_counters']['optimizer_updates'] != 128
            or run['resume']['completed_update'] != 64 or run['arm'] != 'NFR'
            or set(run['updates']) != {str(i) for i in range(65, 129)}
            or run['configuration']['parameters']['weights'] != {'ce': 1., 'latent': 1., 'kl': .1}
            or parent['final_counters']['optimizer_updates'] != 64):
        raise ValueError('Require the completed unchanged KL0.1 NFR64-to128 segment')
    development = {}
    for source in (parent, run):
        for evaluation in source['evaluations']:
            number = evaluation['after_update']
            result = evaluation['panels']['dev-main']['result']
            if number in development and development[number] != result:
                raise ValueError('Repeated development64 result is not identical')
            development[number] = result
    if not {64, 80, 96, 100, 112, 128}.issubset(development):
        raise ValueError('Missing declared development boundary')
    dev_rows = []
    for update, result in sorted(development.items()):
        for p in result['passes']:
            dev_rows.append({'update': update, 'pass': p['index']+1,
                            **{key: finite(p['means'][key]) for key in ('ce', 'latent', 'kl')}})
    training = []
    for update in range(65, 129):
        record = run['updates'][str(update)][0]
        metrics = record['metrics']
        timing = run['observations'][str(update)]['timing_by_rank']
        seconds = max(sum(finite(t[field]) for field in ('materialization_host', 'backward', 'optimizer_and_cursor')) for t in timing)
        if seconds <= 0:
            raise ValueError('Nonpositive timed update region')
        training.append({'update': update, 'input_tokens': metrics['input_tokens'],
            'gradient_norm_before_clip': finite(metrics['gradient_norm_before_clip']),
            'clipped': record['clipping']['norm_exceeds_limit'],
            'lr_used': metrics['lr_used'][0], 'lr_next': metrics['lr_next'][0],
            'compute_plus_materialization_seconds': seconds,
            'complete_update_callback_seconds': max(run['update_wall_seconds_by_rank'][str(update)]),
            **{key: finite(record['loss_means'][key]) for key in ('ce', 'latent', 'kl')}})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = args.output_dir/'input-snapshot'; snapshots.mkdir()
    for name, ref in inputs.items():
        target = snapshots/(name+'.json'); shutil.copyfile(ref['path'], target)
        if digest(target) != ref['sha256']:
            raise ValueError('Analysis input changed during snapshot')
    for name, rows in (('development', dev_rows), ('training', training)):
        with (args.output_dir/(name+'.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for p in range(1, 5):
        rows = [r for r in dev_rows if r['pass'] == p]
        for axis, term in zip(axes, ('ce', 'latent', 'kl')):
            axis.plot([r['update'] for r in rows], [r[term] for r in rows], marker='o', markersize=3, label=f'Pass {p}')
            axis.set(xlabel='Cumulative optimizer update', ylabel={'ce': 'CE (nats / target)',
                'latent': 'Raw latent loss', 'kl': 'Raw KL loss'}[term])
            axis.grid(alpha=.2)
            axis.axvline(64, color='grey', linestyle=':', linewidth=.8)
            axis.axvline(100, color='grey', linestyle='--', linewidth=.8)
    axes[0].legend(fontsize=8)
    figure.suptitle('Reduced-KL NFR: common FP32/no-jitter development panel\n'
                   'Continued from64; unchanged KL0.1, latent1, K4, RT0/15; warmup ends at100')
    for suffix in ('pdf', 'png'):
        figure.savefig(args.output_dir/f'development.{suffix}', dpi=165)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    updates = [r['update'] for r in training]
    axes[0].plot(updates, [r['gradient_norm_before_clip'] for r in training])
    axes[0].axhline(1, linestyle=':', color='grey'); axes[0].set_ylabel('Global norm before clipping')
    axes[1].plot(updates, [r['lr_used'] for r in training]); axes[1].set_ylabel('Learning rate used')
    for term in ('ce', 'latent', 'kl'):
        axes[2].plot(updates, [r[term] for r in training], label=term)
    axes[2].set_ylabel('Unweighted training loss'); axes[2].legend()
    for axis in axes:
        axis.set_xlabel('Cumulative optimizer update'); axis.grid(alpha=.2)
        axis.axvline(100, color='grey', linestyle='--', linewidth=.8)
    figure.suptitle('NFR64-to128 training dynamics; raw losses are reported separately')
    for suffix in ('pdf', 'png'):
        figure.savefig(args.output_dir/f'training-dynamics.{suffix}', dpi=165)
    plt.close(figure)
    norms = [r['gradient_norm_before_clip'] for r in training]
    tokens = sum(r['input_tokens'] for r in training)
    seconds = sum(r['compute_plus_materialization_seconds'] for r in training)
    memories = list(run['memory_after_capture'])
    for observation in run['observations'].values():
        memories.extend(observation['memory_by_rank'])
    resources = {'new_input_tokens': tokens, 'compute_plus_materialization_seconds': seconds,
        'compute_plus_materialization_inputs_per_second': tokens/seconds,
        'executor_seconds': run['elapsed_seconds'], 'executor_inputs_per_second': tokens/run['elapsed_seconds'],
        'peak_reserved_gib': max(m['peak_reserved_gib'] for m in memories),
        'peak_allocated_gib': max(m['peak_allocated_gib'] for m in memories),
        'scope': 'Real input tokens counted once despite K4; compute scope excludes startup, evaluations and checkpointing; executor scope includes these.'}
    tracking = None
    if args.publish:
        import wandb
        from scripts.experiment_tracking import OnlineTracker
        tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', entity='taylorbollman',
            output_dir=args.output_dir, group='nfr-stability-128', name='nfr-reduced64-to128-summary')
        tracker.start({'inputs': inputs, 'scope': 'Audited functionality/adaptation continuation; no matched128 KL control'})
        tracker.log({name: wandb.Image(str(args.output_dir/(name+'.png'))) for name in ('development', 'training-dynamics')})
        tracker.finish(succeeded=True); tracking = tracker.record
    source_target = args.output_dir/'source-snapshot'/Path(__file__).name
    source_target.parent.mkdir(); shutil.copyfile(__file__, source_target)
    result = {'schema': 'olmo-nfr-128-summary-v1', 'status': 'completed', 'inputs': inputs,
        'source_sha256': digest(source_target), 'tracking': tracking, 'development': dev_rows,
        'resources': resources, 'parameters': run['configuration']['parameters'],
        'gradient_norm': {'min': min(norms), 'median': statistics.median(norms), 'max': max(norms), 'final': norms[-1]},
        'clipped_updates': sum(r['clipped'] for r in training), 'updates': 64,
        'artifacts': {p.name: {'sha256': digest(p), 'size_bytes': p.stat().st_size}
                      for p in args.output_dir.iterdir() if p.suffix in ('.csv', '.pdf', '.png')}}
    for ref in inputs.values():
        if digest(Path(ref['path'])) != ref['sha256']:
            raise ValueError('Analysis input changed')
    (args.output_dir/'report.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'output': str(args.output_dir), 'tracking': tracking, 'resources': resources,
                      'gradient_norm': result['gradient_norm'], 'clipped_updates': result['clipped_updates']}))


if __name__ == '__main__':
    main()
