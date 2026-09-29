"""Bounded background retention, unchanged owned-file publication and teardown."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from scripts import olmo_pilot_async_storage as module
from test_pilot_execution_storage import manager, original_publication
from test_campaign_ssd_storage import checkpoint, latest, journal


def setup_job(manager, update=0):
    full = checkpoint(manager, update)
    local = deepcopy(full)
    del local['retention']
    return local, full


def hook_for(full, *, entered=None, release=None, error=None):
    def hook(path, pin, result_path, log_path, **kwargs):
        if entered is not None:
            entered.set()
        if release is not None:
            assert release.wait(4), 'Test did not release worker'
        if error is not None:
            raise error
        return {'schema': module.RESULT_SCHEMA, 'job_sha256': pin,
                'retention': deepcopy(full['retention']), 'worker': {'fake': True}}
    return hook


def retention(manager, full, **kwargs):
    return module.AsyncCheckpointRetention(manager.storage, manager.evidence,
        manager.prefix, retain_hook=hook_for(full), **kwargs)


def test_submit_is_nonblocking_single_outstanding_and_receipt_is_immutable(manager):
    local, full = setup_job(manager)
    entered, release = threading.Event(), threading.Event()
    worker = module.AsyncCheckpointRetention(manager.storage, manager.evidence, manager.prefix,
        retain_hook=hook_for(full, entered=entered, release=release))
    try:
        pending = worker.submit(local)
        assert entered.wait(4)
        assert worker.pending and worker.poll() is None
        assert pending['state'] == 'pending_not_cloud_durable'
        with pytest.raises(RuntimeError, match='Drain'):
            worker.submit(local)
        local['counters']['optimizer_updates'] = 999
        pending['update'] = 999
        release.set()
        result = worker.drain()
        assert result['update'] == 0 and result['receipt'] == full
        assert latest(manager) == full
        assert worker.poll() is None and worker.drain() is None and not worker.pending
        assert result['completed_at_unix_seconds'] >= result['submitted_at_unix_seconds']
        assert all(value >= 0 for value in result['timing'].values())
    finally:
        release.set()
        worker.close()


def test_completed_but_unconsumed_job_also_blocks_second_submit(manager):
    local, full = setup_job(manager)
    worker = retention(manager, full)
    try:
        worker.submit(local)
        worker._future.result(timeout=4)
        with pytest.raises(RuntimeError, match='Drain'):
            worker.submit(local)
        assert worker.poll()['receipt'] == full
    finally:
        worker.close()


def test_all_storage_hash_publish_prune_work_occurs_in_background_thread(manager, monkeypatch):
    caller = threading.get_ident()
    seen = []
    original_validate = manager.storage.validate_local
    original_publish = manager.storage.publish_retained
    def validate(receipt):
        seen.append(('validate', threading.get_ident()))
        return original_validate(receipt)
    def publish(receipt):
        seen.append(('publish', threading.get_ident()))
        return original_publish(receipt)
    monkeypatch.setattr(manager.storage, 'validate_local', validate)
    monkeypatch.setattr(manager.storage, 'publish_retained', publish)
    local, full = setup_job(manager)
    worker = retention(manager, full)
    try:
        worker.submit(local)
        worker.drain()
    finally:
        worker.close()
    assert seen and all(tid != caller for _, tid in seen)
    assert len({tid for _, tid in seen}) == 1


def test_old_retained_bytes_prune_only_after_new_verified_publication(manager):
    worker = None
    try:
        for step in range(4):
            local, full = setup_job(manager, step)
            if worker is None:
                worker = retention(manager, full)
            else:
                worker._retain = hook_for(full)
            worker.submit(local)
            result = worker.drain()
        assert result['storage_publication']['kept_updates'] == [2, 3]
        assert result['storage_publication']['pruned_updates'] == [0, 1]
        assert all(row['status'] == 'completed' for row in journal(manager)['prune_operations'])
    finally:
        if worker is not None:
            worker.close()


@pytest.mark.parametrize('fault', ['cloud_error', 'wrong_job', 'corrupt_bytes', 'unverified'])
def test_failed_retention_preserves_last_authority_and_pending_bytes(manager, fault):
    old = checkpoint(manager, 0)
    manager.storage.publish_retained(old)
    local, full = setup_job(manager, 1)
    def bad(path, pin, result_path, log_path, **kwargs):
        if fault == 'cloud_error':
            raise OSError('simulated upload failure')
        if fault == 'corrupt_bytes':
            (Path(local['directory'])/'state.pt').write_bytes(b'corrupt')
        retained = deepcopy(full['retention'])
        if fault == 'unverified':
            retained['download_sha256_verified'] = False
        return {'schema': module.RESULT_SCHEMA, 'job_sha256': '0'*64 if fault == 'wrong_job' else pin,
                'retention': retained}
    worker = module.AsyncCheckpointRetention(manager.storage, manager.evidence, manager.prefix, retain_hook=bad)
    try:
        worker.submit(local)
        with pytest.raises(module.AsyncRetentionError):
            worker.drain()
        assert latest(manager) == old
        assert (Path(local['directory'])/'state.pt').exists()
        assert (worker.root/'update-000001'/'failure.json').is_file()
        assert not journal(manager)['prune_operations']
        with pytest.raises(RuntimeError, match='failed'):
            worker.submit(local)
    finally:
        worker.close()


def test_abort_before_publication_preserves_local_checkpoint(manager):
    local, full = setup_job(manager)
    entered, release = threading.Event(), threading.Event()
    worker = module.AsyncCheckpointRetention(manager.storage, manager.evidence, manager.prefix,
        retain_hook=hook_for(full, entered=entered, release=release))
    worker.submit(local)
    assert entered.wait(4)
    worker._cancel.set()
    release.set()
    result = worker.abort()
    assert result['status'] == 'cancelled_or_failed'
    assert not (manager.evidence/'latest-checkpoint.json').exists()
    assert (Path(local['directory'])/'state.pt').is_file()
    assert not worker.pending


def test_manager_does_not_change_python_rng(manager):
    local, full = setup_job(manager)
    before = random.getstate()
    worker = retention(manager, full)
    worker.submit(local)
    worker.close()
    assert random.getstate() == before


def test_source_mutation_rejected_before_submission(manager, tmp_path):
    extra = tmp_path/'source.py'
    extra.write_text('# frozen\n')
    local, full = setup_job(manager)
    worker = retention(manager, full, source_pins={extra: module._sha(extra)})
    try:
        extra.write_text('# changed\n')
        with pytest.raises(ValueError, match='source changed'):
            worker.submit(local)
        assert not worker.pending
    finally:
        worker.close()


@pytest.mark.parametrize('timeout', [0, -1, True, float('nan'), float('inf'), 481])
def test_invalid_timeout_rejected(manager, timeout):
    with pytest.raises(ValueError, match='timeout'):
        module.AsyncCheckpointRetention(manager.storage, manager.evidence, manager.prefix,
                                        worker_timeout_seconds=timeout)


def test_only_owner_thread_can_submit_poll_drain_or_close(manager):
    local, full = setup_job(manager)
    worker = retention(manager, full)
    errors = []
    def outside():
        for action in (lambda: worker.submit(local), worker.poll, worker.drain, worker.close):
            try:
                action()
            except RuntimeError as error:
                errors.append(str(error))
    thread = threading.Thread(target=outside)
    thread.start(); thread.join()
    worker.close()
    assert len(errors) == 4 and all('creating training thread' in value for value in errors)


def test_child_environment_removes_distributed_and_hides_cuda():
    initial = {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '2', 'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '4444', 'TORCHELASTIC_RUN_ID': 'abc', 'NCCL_DEBUG': 'WARN',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING': '0', 'CUDA_VISIBLE_DEVICES': '0,1',
        'GOOGLE_APPLICATION_CREDENTIALS': '/private/credentials', 'PATH': '/usr/bin'}
    actual = module.cpu_child_environment(initial)
    assert actual['CUDA_VISIBLE_DEVICES'] == '' and actual['NVIDIA_VISIBLE_DEVICES'] == 'void'
    assert actual['GOOGLE_APPLICATION_CREDENTIALS'] == initial['GOOGLE_APPLICATION_CREDENTIALS']
    assert not any(key in actual for key in initial if key.startswith(('RANK', 'LOCAL_', 'WORLD_',
        'MASTER_', 'TORCHELASTIC_', 'NCCL_', 'TORCH_NCCL_')))
    assert initial['CUDA_VISIBLE_DEVICES'] == '0,1'


@pytest.mark.parametrize('cancel', [False, True])
def test_child_timeout_or_cancellation_kills_and_reaps_process_group(tmp_path, monkeypatch, cancel):
    events = []
    class Process:
        pid = 999999
        args = ['test-retainer']
        def wait(self, timeout=None):
            events.append(('wait', timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return -9
    def launch(args, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['start_new_session'] is True
        return Process()
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    monkeypatch.setattr(module.os, 'killpg', lambda pid, sig: events.append(('killpg', pid, sig)))
    event = threading.Event()
    if cancel:
        event.set()
    expected = module.AsyncRetentionError if cancel else subprocess.TimeoutExpired
    with pytest.raises(expected):
        module.retain_in_child(tmp_path/'job.json', '0'*64, tmp_path/'result.json',
            tmp_path/'log.txt', timeout_seconds=0.001, cancel_event=event)
    assert any(row[0] == 'killpg' for row in events)
    assert events[-1] == ('wait', None)


def test_child_refuses_gpu_visibility_before_cloud_import(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(RuntimeError, match='disabled CUDA'):
        module.worker(tmp_path/'unused', '0'*64, tmp_path/'result')


def test_child_refuses_distributed_environment_before_cloud_import(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('RANK', '0')
    with pytest.raises(RuntimeError, match='process-group'):
        module.worker(tmp_path/'unused', '0'*64, tmp_path/'result')


def test_child_nonzero_exit_does_not_accept_fake_result(tmp_path, monkeypatch):
    class Process:
        pid = 999999
        def wait(self, timeout=None):
            return 7
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: Process())
    result = tmp_path/'result.json'
    result.write_text('{}')
    with pytest.raises(module.AsyncRetentionError, match='exited 7'):
        module.retain_in_child(tmp_path/'job', '0'*64, result, tmp_path/'log', timeout_seconds=1)


def test_plain_module_import_does_not_import_torch_or_cloud_sdk():
    command = [sys.executable, '-c', 'import sys; import scripts.olmo_pilot_async_storage; '
        'assert "torch" not in sys.modules; assert "google.cloud.storage" not in sys.modules']
    subprocess.run(command, cwd=module.ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize('fault', [None, 'state_bytes', 'manifest_bytes'])
def test_cpu_worker_uploads_state_then_manifest_with_exact_readback(manager, monkeypatch, fault):
    # Real local bytes and pinned request; the cloud boundary alone is faked.
    from google.cloud import storage
    from scripts import olmo_two_gpu_retain
    local, full = setup_job(manager)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    directory = manager.evidence/'worker-test'
    directory.mkdir()
    job = {'schema': module.SCHEMA, 'receipt': local, 'storage_prefix': manager.prefix,
           'source_pins': {str(Path(module.__file__)): module._sha(module.__file__)}}
    job_path = directory/'job.json'
    job_path.write_text(json.dumps(job))
    pin = module._sha(job_path)
    calls = []
    monkeypatch.setattr(storage, 'Client', lambda: SimpleNamespace(bucket=lambda name: SimpleNamespace(name=name)))
    def upload(bucket, key, path, expected, *, download_sha256):
        assert download_sha256 is True
        assert expected['sha256'] == module._sha(path)
        calls.append(Path(path).name)
        return deepcopy(next(row for row in full['retention']['objects'] if row['uri'].endswith('/'+Path(path).name)))
    monkeypatch.setattr(olmo_two_gpu_retain, 'upload_verified', upload)
    if fault:
        target = 'state.pt' if fault == 'state_bytes' else 'manifest.json'
        (Path(local['directory'])/target).write_bytes(b'changed')
        with pytest.raises(ValueError, match='bytes changed'):
            module.worker(job_path, pin, directory/'result.json')
        assert not (directory/'result.json').exists()
        assert calls == ([] if fault == 'state_bytes' else ['state.pt'])
    else:
        result = module.worker(job_path, pin, directory/'result.json')
        assert calls == ['state.pt', 'manifest.json']
        assert result['retention'] == full['retention']
        assert result['worker']['cuda_initialized'] is False
        assert result['worker']['distributed_initialized'] is False
        assert result == json.loads((directory/'result.json').read_text())


def test_close_drains_terminal_job_and_forbids_further_use(manager):
    local, full = setup_job(manager)
    worker = retention(manager, full)
    worker.submit(local)
    assert worker.close()['receipt'] == full
    assert latest(manager) == full
    with pytest.raises(RuntimeError, match='closed'):
        worker.submit(local)
    assert worker.close() is None
