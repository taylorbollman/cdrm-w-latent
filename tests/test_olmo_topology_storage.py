"""Campaign identity, authenticated CPU publication and bounded child teardown."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from scripts import olmo_topology_storage as module
from scripts.olmo_campaign_execution_restore import committed_metadata, MANIFEST_FIELDS
from test_campaign_ssd_storage import manager, checkpoint
from test_campaign_execution_restore import publication


def job_fixture(manager):
    full = checkpoint(manager, 1)
    local = deepcopy(full)
    del local['retention']
    path = manager.evidence / 'job.json'
    job = {'schema': module.SCHEMA, 'receipt': local, 'storage_prefix': manager.prefix,
        'source_pins': {str(module.ROOT / name): pin for name, pin in module.source_hashes().items()}}
    path.write_text(json.dumps(job))
    return path, module._sha(path), job, full


def cpu_environment(monkeypatch):
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')


def test_campaign_authority_accepted_and_pilot_authority_rejected(manager):
    path, pin, job, _ = job_fixture(manager)
    assert module.authenticate_job(path, pin) == job
    from scripts.olmo_pilot_execution_restore import committed_metadata as pilot_authority
    with pytest.raises(ValueError, match='identity'):
        pilot_authority({key: job['receipt'][key] for key in MANIFEST_FIELDS})
    assert committed_metadata({key: job['receipt'][key] for key in MANIFEST_FIELDS})['schema'] == 'olmo-campaign-execution-identity-v1'


@pytest.mark.parametrize('fault', ['job_hash', 'source_hash', 'missing_source', 'manifest', 'state_symlink', 'prefix', 'pilot'])
def test_bad_authority_rejected_before_cloud_access(manager, fault):
    path, pin, job, _ = job_fixture(manager)
    if fault == 'job_hash': pin = '0' * 64
    elif fault == 'source_hash': job['source_pins'][str(Path(module.__file__))] = '0' * 64
    elif fault == 'missing_source': job['source_pins'].pop(str(Path(module.__file__)))
    elif fault == 'manifest': (Path(job['receipt']['directory']) / 'manifest.json').write_text('{}')
    elif fault == 'state_symlink':
        state = Path(job['receipt']['directory']) / 'state.pt'
        state.rename(state.with_name('outside')); state.symlink_to(state.with_name('outside'))
    elif fault == 'prefix': job['storage_prefix'] = 'gs://foreign/elsewhere'
    elif fault == 'pilot': job['receipt']['metadata']['configuration']['execution_identity']['schema'] = 'olmo-pilot-execution-identity-v1'
    if fault != 'job_hash':
        path.write_text(json.dumps(job)); pin = module._sha(path)
    with pytest.raises(ValueError): module.authenticate_job(path, pin)


@pytest.mark.parametrize('changed_state', [False, True])
def test_worker_uses_campaign_identity_and_verified_create_only_publication(manager, monkeypatch, changed_state):
    from google.cloud import storage
    from scripts import olmo_two_gpu_retain
    path, pin, job, full = job_fixture(manager)
    cpu_environment(monkeypatch)
    calls = []
    monkeypatch.setattr(storage, 'Client', lambda: SimpleNamespace(bucket=lambda name: SimpleNamespace(name=name)))
    def upload(bucket, key, path, expected, *, download_sha256):
        assert download_sha256 is True and module._sha(path) == expected['sha256']
        calls.append(Path(path).name)
        return deepcopy(next(row for row in full['retention']['objects'] if row['uri'].endswith('/' + Path(path).name)))
    monkeypatch.setattr(olmo_two_gpu_retain, 'upload_verified', upload)
    result_path = manager.evidence / 'result.json'
    if changed_state:
        (Path(job['receipt']['directory']) / 'state.pt').write_bytes(b'corrupt')
        with pytest.raises(ValueError, match='bytes changed'): module.worker(path, pin, result_path)
        assert not calls and not result_path.exists()
    else:
        result = module.worker(path, pin, result_path)
        assert result['retention'] == full['retention']
        assert calls == ['state.pt', 'manifest.json']
        assert result['worker']['cuda_initialized'] is False
        assert result['worker']['distributed_initialized'] is False
        assert result['worker']['identity_validator'] == 'olmo-campaign-execution-identity-v1'
        assert result == json.loads(result_path.read_text())


@pytest.mark.parametrize('cancel', [False, True])
def test_child_timeout_or_cancellation_kills_and_reaps_own_process_group(tmp_path, monkeypatch, cancel):
    events = []
    class Process:
        pid = 999999
        args = ['test-retainer']
        def wait(self, timeout=None):
            events.append(('wait', timeout))
            if timeout is not None: raise subprocess.TimeoutExpired(self.args, timeout)
            return -9
    def launch(args, **kwargs):
        assert args[1] == str(Path(module.__file__).resolve())
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '' and kwargs['start_new_session'] is True
        assert not any(k in kwargs['env'] for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
        return Process()
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    monkeypatch.setattr(module.os, 'killpg', lambda pid, sig: events.append(('killpg', pid, sig)))
    event = threading.Event()
    if cancel: event.set()
    with pytest.raises(module.AsyncRetentionError if cancel else subprocess.TimeoutExpired):
        module.retain_in_child(tmp_path/'job', '0'*64, tmp_path/'result', tmp_path/'log',
            timeout_seconds=.001, cancel_event=event)
    assert any(row[0] == 'killpg' for row in events) and events[-1] == ('wait', None)


@pytest.mark.parametrize('fault', ['nonzero', 'wrong_job', 'symlink'])
def test_child_does_not_accept_unbound_or_failed_result(tmp_path, monkeypatch, fault):
    result_path = tmp_path / 'result.json'
    class Process:
        pid = 999999
        def wait(self, timeout=None):
            payload = {'schema': module.RESULT_SCHEMA, 'job_sha256': '1'*64 if fault == 'wrong_job' else '0'*64}
            if fault == 'symlink':
                outside = tmp_path / 'outside.json'; outside.write_text(json.dumps(payload)); result_path.symlink_to(outside)
            else: result_path.write_text(json.dumps(payload))
            return 7 if fault == 'nonzero' else 0
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: Process())
    with pytest.raises((module.AsyncRetentionError, ValueError)):
        module.retain_in_child(tmp_path/'job', '0'*64, result_path, tmp_path/'log', timeout_seconds=1)


def test_plain_import_does_not_load_torch_or_cloud_sdk():
    subprocess.run([sys.executable, '-c', 'import sys; import scripts.olmo_topology_storage; '
        'assert "torch" not in sys.modules; assert "google.cloud.storage" not in sys.modules'],
        cwd=module.ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize('environment', ['cuda', 'rank'])
def test_worker_rejects_gpu_or_process_group_environment_before_job_read(tmp_path, monkeypatch, environment):
    cpu_environment(monkeypatch)
    if environment == 'cuda': monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    else: monkeypatch.setenv('RANK', '0')
    with pytest.raises(RuntimeError): module.worker(tmp_path/'unused', '0'*64, tmp_path/'result')
