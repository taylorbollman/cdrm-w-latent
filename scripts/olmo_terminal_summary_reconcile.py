"""Reconcile completed W&B checkpoint summary from pinned terminal evidence.

Uses the Public API summary mutation only. It does not initialize/resume a W&B
run, write history, load model tensors, or change the training evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

SCHEMA = 'olmo-terminal-summary-reconciliation-v1'
ANNOTATION = 'checkpoint/terminal_summary_reconciliation'
FIELDS = ('checkpoint/last_local_update', 'checkpoint/last_verified_cloud_update',
          'checkpoint/worker_pending')
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_summary(value):
    """W&B flattens nested provenance dictionaries into dotted summary keys."""
    result = dict(value)
    annotation = result.pop(ANNOTATION, None)
    if annotation is not None:
        require(isinstance(annotation, dict), 'Unexpected provenance encoding')
        result.update({ANNOTATION + '.' + key: item for key, item in annotation.items()})
    return result


def verify_summary(before, after, patch):
    before, after, patch = map(normalized_summary, (before, after, patch))
    require(all(after.get(key) == value for key, value in patch.items()),
            'Fresh readback does not match the requested correction')
    unrelated_before = {k: v for k, v in before.items() if k not in patch}
    unrelated_after = {k: v for k, v in after.items() if k not in patch}
    require(unrelated_after == unrelated_before, 'Unrelated summary metrics changed')
    return json_digest(unrelated_after)


def readback(api_factory, run_path, before, patch):
    for attempt in range(4):
        run = api_factory().run(run_path)
        require(run.state == 'finished', 'Remote run state changed')
        after = dict(run.summary)
        try:
            digest_after = verify_summary(before, after, patch)
        except ValueError:
            if attempt == 3:
                raise
            time.sleep(2)
        else:
            return after, digest_after, attempt + 1


def validate_evidence(report, publication, publication_sha256):
    require(report.get('schema') in ('olmo-pilot-async-execute-report-v1',
                                    'olmo-fbt-stability-execute-report-v1'),
            'Unsupported terminal report schema')
    require(report.get('status') in ('completed_plan', 'stopped_at_boundary')
            and report.get('segment_completed'), 'Execution segment is not complete')
    loop = report['loop']
    update = report['final_counters']['optimizer_updates']
    require(type(update) is int and update >= 0, 'Invalid completed update')
    require(loop['checkpoint_pending'] is False and all(value == update for value in (
        loop['completed_update'], loop['last_saved_update'], loop['last_retained_update'],
        report['last_verified_cloud_update'])), 'Final checkpoint drain is incomplete')
    require(publication in report['published_checkpoints'], 'Publication is not in terminal report')
    require(publication['counters']['optimizer_updates'] == update,
            'Publication is not the terminal checkpoint')
    require(any(p['update'] == update and p['receipt_sha256'] == publication_sha256
                for p in report['storage_publications']), 'Publication digest is not authoritative')
    retention = publication['retention']
    require(retention['create_only'] is True and retention['download_sha256_verified'] is True,
            'Checkpoint retention is not verified')
    objects = retention['objects']
    require(len(objects) == 2, 'Expected state and manifest objects')
    by_name = {obj['uri'].rsplit('/', 1)[-1]: obj for obj in objects}
    require(set(by_name) == {'state.pt', 'manifest.json'}, 'Unexpected checkpoint objects')
    for obj in objects:
        require(obj['generation'] and all(obj['verification'].get(key) is True for key in
            ('server_size', 'server_md5', 'sha256_metadata', 'download_sha256')),
            'Cloud object verification is incomplete')
    require(by_name['manifest.json']['sha256'] == publication['manifest_sha256'],
            'Cloud manifest pin differs')
    require(by_name['state.pt']['sha256'] == publication['state']['sha256'],
            'Cloud state pin differs')
    tracking = report['wandb']
    require(tracking['enabled'] is True and tracking['status'] == 'synced'
            and tracking['mode'] == 'online', 'Tracking is not finalized')
    require(tracking['entity'] == 'taylorbollman'
            and tracking['project'] == 'pretrained-fbt-rt-nextlat', 'Unexpected tracking scope')
    return update, '/'.join(tracking[key] for key in ('entity', 'project', 'run_id'))


def reconcile(args, *, api_factory, sdk_version):
    require(digest(args.report) == args.report_sha256, 'Report SHA256 differs')
    require(digest(args.publication) == args.publication_sha256, 'Publication SHA256 differs')
    report = json.loads(args.report.read_text())
    publication = json.loads(args.publication.read_text())
    update, run_path = validate_evidence(report, publication, args.publication_sha256)
    run = api_factory().run(run_path)
    require(run.state == 'finished' and '/'.join(run.path) == run_path,
            'Remote run must be finished and match the pinned report')
    before = dict(run.summary)
    require(before.get('update') == update and before.get('_step') == update,
            'Remote summary is not at the same completed training boundary')
    require(not any(key == ANNOTATION or key.startswith(ANNOTATION + '.') for key in before),
            'A terminal summary correction already exists')
    for key in FIELDS:
        require(key in before, 'Expected checkpoint summary field is absent')
    source_name = 'scripts/olmo_terminal_summary_reconcile.py'
    source_sha256 = digest(__file__)
    provenance = {'schema': SCHEMA, 'completed_update': update,
        'report_sha256': args.report_sha256, 'publication_sha256': args.publication_sha256,
        'source_sha256': source_sha256,
        'scope': 'Terminal checkpoint summary only; history and training state unchanged'}
    patch = normalized_summary(dict(zip(FIELDS, (update, update, False))) | {ANNOTATION: provenance})
    excluded = set(patch)
    unrelated = {k: v for k, v in before.items() if k not in excluded}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'summary-before.json', before)
    source = args.output / 'source-snapshot' / source_name
    source.parent.mkdir(parents=True)
    shutil.copyfile(__file__, source)
    for name, path in [('terminal-report.json', args.report), ('publication.json', args.publication)]:
        shutil.copyfile(path, args.output / name)
    receipt = {'schema': SCHEMA, 'status': 'pending', 'run_path': run_path,
        'sdk_version': sdk_version, 'sources': {source_name: source_sha256},
        'inputs': {'report': {'path': str(args.report), 'sha256': args.report_sha256},
                   'publication': {'path': str(args.publication), 'sha256': args.publication_sha256}},
        'before': {key: before[key] for key in FIELDS}, 'patch': patch,
        'unrelated_summary_sha256_before': json_digest(unrelated),
        'history_write_called': False, 'training_state_touched': False,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.output / 'report.json', receipt)
    try:
        # Installed SDK writes only summaryMetrics, not history or run state.
        run.summary.update(patch)
        after, digest_after, attempts = readback(api_factory, run_path, before, patch)
        write_json(args.output / 'summary-after.json', after)
        require(digest(args.report) == args.report_sha256
                and digest(args.publication) == args.publication_sha256,
                'Pinned local evidence changed during correction')
        receipt.update(status='verified', after={key: after[key] for key in FIELDS},
            unrelated_summary_sha256_after=digest_after, readback_attempts=attempts,
            readback_verified=True, pinned_inputs_unchanged=True,
            summary_step_before=before['_step'], summary_step_after=after['_step'])
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__)
        write_json(args.output / 'report.json', receipt)
        raise RuntimeError('Terminal summary reconciliation failed; inspect retained receipt') from None
    write_json(args.output / 'report.json', receipt)
    return receipt


def verify_attempt(args, *, api_factory, sdk_version):
    """Read-only recovery after a mutation whose immediate readback was stale."""
    require(digest(args.report) == args.report_sha256, 'Report SHA256 differs')
    require(digest(args.publication) == args.publication_sha256, 'Publication SHA256 differs')
    report = json.loads(args.report.read_text())
    publication = json.loads(args.publication.read_text())
    update, run_path = validate_evidence(report, publication, args.publication_sha256)
    attempt_path = args.verify_attempt / 'report.json'
    attempt = json.loads(attempt_path.read_text())
    require(attempt['schema'] == SCHEMA and attempt['run_path'] == run_path,
            'Attempt belongs to another run or schema')
    require(attempt['inputs']['report']['sha256'] == args.report_sha256
            and attempt['inputs']['publication']['sha256'] == args.publication_sha256,
            'Attempt input pins differ')
    for relative, expected in attempt['sources'].items():
        require(digest(args.verify_attempt / 'source-snapshot' / relative) == expected,
                'Attempt source snapshot differs')
    before = json.loads((args.verify_attempt / 'summary-before.json').read_text())
    require(before['update'] == update and before['_step'] == update, 'Attempt boundary differs')
    after, digest_after, attempts = readback(api_factory, run_path, before, attempt['patch'])
    require(digest(args.report) == args.report_sha256
            and digest(args.publication) == args.publication_sha256, 'Pinned inputs changed')
    args.output.mkdir(parents=True, exist_ok=False)
    source_name = 'scripts/olmo_terminal_summary_reconcile.py'
    destination = args.output / 'source-snapshot' / source_name
    destination.parent.mkdir(parents=True)
    shutil.copyfile(__file__, destination)
    shutil.copyfile(attempt_path, args.output / 'attempt-report.json')
    write_json(args.output / 'summary-before.json', before)
    write_json(args.output / 'summary-after.json', after)
    receipt = {'schema': SCHEMA, 'status': 'verified', 'run_path': run_path,
        'scope': 'Read-only verification of earlier summary mutation; no repeated write',
        'sdk_version': sdk_version, 'sources': {source_name: digest(__file__)},
        'inputs': attempt['inputs'], 'attempt_report_sha256': digest(attempt_path),
        'before': attempt['before'], 'after': {key: after[key] for key in FIELDS},
        'patch': attempt['patch'], 'readback_verified': True, 'readback_attempts': attempts,
        'summary_step_before': before['_step'], 'summary_step_after': after['_step'],
        'unrelated_summary_sha256_before': attempt['unrelated_summary_sha256_before'],
        'unrelated_summary_sha256_after': digest_after,
        'history_write_called': False, 'summary_write_called': False,
        'training_state_touched': False, 'pinned_inputs_unchanged': True,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.output / 'report.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'publication', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('report-sha256', 'publication-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--verify-attempt', type=Path,
                        help='Read-only verification of an earlier correction directory')
    args = parser.parse_args()
    import wandb
    function = verify_attempt if args.verify_attempt is not None else reconcile
    result = function(args, api_factory=lambda: wandb.Api(timeout=30), sdk_version=wandb.__version__)
    print(json.dumps({key: result[key] for key in ('status', 'run_path', 'before', 'after')}, indent=2))


if __name__ == '__main__':
    main()
