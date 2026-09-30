"""Explicitly publish an already completed, pinned NFR pair summary to W&B."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.olmo_nfr_kl_summary import SCHEMA as SUMMARY_SCHEMA, common, require


def validate_summary(summary):
    require(summary['schema'] == SUMMARY_SCHEMA and summary['status'] == 'complete_pair'
            and summary['arm'] == 'NFR' and set(summary['arms']) == {'KL1', 'KL0.1'}
            and summary['independent_audit']['passed'] is True
            and all(summary['checks'].values()), 'Only a validated completed NFR pair may be published')
    require(all(arm['terminal_cloud_update'] == 64 for arm in summary['arms'].values()),
            'Pair has not finished its retained endpoints')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    data = args.summary.read_bytes()
    require(common.sha(data) == args.summary_sha256, 'Summary SHA256 differs')
    summary = json.loads(data); validate_summary(summary)
    for name, entry in summary['artifacts'].items():
        require(common.sha((args.summary.parent / name).read_bytes()) == entry['sha256'],
                'Summary artifact SHA256 differs: ' + name)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'input-summary.json').write_bytes(data)
    from scripts.experiment_tracking import OnlineTracker
    import wandb
    tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', entity='taylorbollman',
        output_dir=args.output, group='nfr-kl-paired-continuation', name='paired-nfr-kl-continuation-summary')
    tracker.start({'scope': 'NFR32-to64, saved Adam, KL-only intervention; optional F64 descriptive context; no precision clearance',
        'summary_sha256': args.summary_sha256, 'qualifications': summary['qualifications']})
    images = {}
    for name in ('development-raw-losses', 'training-dynamics', 'timing-memory', 'ce-refinement'):
        source = args.summary.parent / (name + '.png')
        shutil.copyfile(source, args.output / source.name)
        images[name] = wandb.Image(str(source))
    rows = []
    for label, arm in summary['arms'].items():
        for entry in arm['development']:
            for p in entry['passes']:
                rows.append([label, entry['after_update'], p['pass'],
                    *[p['means'][term] for term in common.TERMS], entry['ce_gap_vs_first_pass'][p['pass']-1]])
    tracker.log({**images, 'development': wandb.Table(columns=['branch', 'update', 'pass',
        'raw_ce', 'raw_latent', 'raw_kl', 'ce_gap_vs_first_pass'], data=rows)})
    tracker.finish(succeeded=True)
    require(common.sha(args.summary.read_bytes()) == args.summary_sha256, 'Summary changed during publication')
    sources = {}
    for relative in ('scripts/olmo_nfr_kl_tracking.py', 'scripts/olmo_nfr_kl_summary.py',
                     'scripts/olmo_kl_continuation_summary.py', 'scripts/experiment_tracking.py'):
        path = Path(__file__).resolve().parents[1] / relative
        target = args.output / 'source-snapshot' / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); sources[relative] = common.sha(path.read_bytes())
    report = {'schema': 'olmo-nfr-kl-summary-tracking-v1', 'status': 'completed',
        'summary_sha256': args.summary_sha256, 'sources': sources, 'tracking': tracker.record,
        'scope': 'Presentation only; no optimizer updates and no objective-total comparisons'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(tracker.record['run_url'])


if __name__ == '__main__':
    main()
