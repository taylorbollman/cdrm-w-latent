"""Bounded metadata checks for summary-only terminal reconciliation."""
import copy
import json
from argparse import Namespace

import pytest

from scripts import olmo_terminal_summary_reconcile as reconcile


def evidence():
    verification = dict.fromkeys(('server_size', 'server_md5', 'sha256_metadata', 'download_sha256'), True)
    publication = {'counters': {'optimizer_updates': 128}, 'manifest_sha256': 'manifest',
        'state': {'sha256': 'state'}, 'retention': {'create_only': True,
        'download_sha256_verified': True, 'objects': [
            {'uri': 'gs://scope/state.pt', 'sha256': 'state', 'generation': '1', 'verification': verification},
            {'uri': 'gs://scope/manifest.json', 'sha256': 'manifest', 'generation': '2', 'verification': verification}]}}
    report = {'schema': 'olmo-pilot-async-execute-report-v1', 'status': 'completed_plan',
        'segment_completed': True, 'loop': {'checkpoint_pending': False, 'completed_update': 128,
        'last_saved_update': 128, 'last_retained_update': 128}, 'last_verified_cloud_update': 128,
        'final_counters': {'optimizer_updates': 128}, 'published_checkpoints': [publication],
        'storage_publications': [{'update': 128, 'receipt_sha256': 'pin'}],
        'wandb': {'enabled': True, 'status': 'synced', 'mode': 'online', 'entity': 'taylorbollman',
                  'project': 'pretrained-fbt-rt-nextlat', 'run_id': 'example'}}
    return report, publication


@pytest.mark.parametrize('mutation', ['active', 'pending', 'wrong_publication', 'unverified', 'tracking_live'])
def test_reject_unfinished_or_mismatched_evidence(mutation):
    report, publication = evidence()
    if mutation == 'active': report['status'] = 'running'
    if mutation == 'pending': report['loop']['checkpoint_pending'] = True
    if mutation == 'wrong_publication': report['storage_publications'][0]['receipt_sha256'] = 'other'
    if mutation == 'unverified': publication['retention']['objects'][0]['verification']['server_md5'] = False
    if mutation == 'tracking_live': report['wandb']['status'] = 'running'
    with pytest.raises(ValueError):
        reconcile.validate_evidence(report, publication, 'pin')


def test_one_summary_mutation_and_fresh_readback_preserve_unrelated_metrics(tmp_path):
    report, publication = evidence()
    publication_path = tmp_path / 'publication.json'
    reconcile.write_json(publication_path, publication)
    pub_sha = reconcile.digest(publication_path)
    report['storage_publications'][0]['receipt_sha256'] = pub_sha
    report_path = tmp_path / 'terminal.json'
    reconcile.write_json(report_path, report)
    summary = {'update': 128, '_step': 128, 'train/loss': 2.5,
               reconcile.FIELDS[0]: 96, reconcile.FIELDS[1]: 64, reconcile.FIELDS[2]: True}
    calls = []
    class Summary(dict):
        def update(self, patch):
            calls.append(copy.deepcopy(patch))
            summary.update(patch)
    class API:
        def run(self, path):
            return Namespace(state='finished', path=path.split('/'), summary=Summary(summary))
    args = Namespace(report=report_path, report_sha256=reconcile.digest(report_path),
                     publication=publication_path, publication_sha256=pub_sha, output=tmp_path/'result')
    receipt = reconcile.reconcile(args, api_factory=API, sdk_version='fake')
    assert receipt['status'] == 'verified' and len(calls) == 1
    assert receipt['after'] == dict(zip(reconcile.FIELDS, (128, 128, False)))
    assert summary['train/loss'] == 2.5 and summary['_step'] == 128
    assert receipt['unrelated_summary_sha256_before'] == receipt['unrelated_summary_sha256_after']
    assert json.loads((args.output/'summary-before.json').read_text())[reconcile.FIELDS[2]] is True


def test_reject_wrong_input_hash_before_any_api_call(tmp_path):
    path = tmp_path/'report.json'
    path.write_text('{}')
    args = Namespace(report=path, report_sha256='wrong')
    with pytest.raises(ValueError, match='Report SHA256 differs'):
        reconcile.reconcile(args, api_factory=lambda: pytest.fail('API called'), sdk_version='fake')


def test_flattened_provenance_and_stale_readback_retry(monkeypatch):
    before = {'update': 128, '_step': 128, 'loss': 2.5}
    patch = {reconcile.ANNOTATION: {'schema': 'proof'}, reconcile.FIELDS[0]: 128}
    after = before | reconcile.normalized_summary(patch)
    reads = []
    class API:
        def run(self, path):
            reads.append(path)
            return Namespace(state='finished', summary=before if len(reads) == 1 else after)
    monkeypatch.setattr(reconcile.time, 'sleep', lambda seconds: None)
    value, digest, attempts = reconcile.readback(API, 'a/b/c', before, patch)
    assert value == after and attempts == 2
    assert digest == reconcile.json_digest(before)


def test_reject_unrelated_summary_change():
    with pytest.raises(ValueError, match='Unrelated'):
        reconcile.verify_summary({'loss': 2.5}, {'loss': 3., 'checkpoint': 128}, {'checkpoint': 128})


def test_kl_continuation_schema_retains_terminal_drain_gate():
    report, publication = evidence()
    report['schema'] = 'olmo-kl-continuation-report-v1'
    report['arm'] = 'NFR'
    report['status'] = 'stopped_at_boundary'
    assert reconcile.validate_evidence(report, publication, 'pin') == (
        128, 'taylorbollman/pretrained-fbt-rt-nextlat/example')
    report['loop']['checkpoint_pending'] = True
    with pytest.raises(ValueError, match='drain is incomplete'):
        reconcile.validate_evidence(report, publication, 'pin')
