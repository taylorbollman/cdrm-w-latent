"""One outstanding CPU retention job after an immutable SSD checkpoint commit.

The trainer must drain before accessing storage again. The background thread
exclusively owns that storage object; it never sees model/optimizer/CUDA state.
Only a fresh CPU child touches the cloud SDK, isolating SDK randomness from the
training process. Publication retains all existing hashes and pruning checks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_campaign_ssd_storage import _write_durable

SCHEMA = 'olmo-pilot-async-retention-v1'
RESULT_SCHEMA = 'olmo-pilot-async-retention-result-v1'


class AsyncRetentionError(RuntimeError):
    """A retention job failed; already published recovery authority is preserved."""


def _clone(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_sources(sources):
    for path, expected in sources.items():
        if _sha(path) != expected:
            raise ValueError('Async retention runtime source changed: ' + path)


def cpu_child_environment(environment=None):
    """SDK credentials survive; launch/rank variables and GPU visibility do not."""
    env = dict(os.environ if environment is None else environment)
    for key in tuple(env):
        if (key in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                    'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}
                or key.startswith(('TORCHELASTIC_', 'NCCL_', 'TORCH_NCCL_'))):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void', OMP_NUM_THREADS='1')
    return env


def retain_in_child(job_path, job_sha256, result_path, log_path, *, timeout_seconds,
                    cancel_event=None):
    """Launch/reap a separate session; timeout kills its entire process group."""
    with Path(log_path).open('xb') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker',
            str(job_path), '--job-sha256', job_sha256, '--result', str(result_path)],
            cwd=ROOT, env=cpu_child_environment(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout_seconds
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise AsyncRetentionError('CPU retention child cancelled by trainer teardown')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(process.args, timeout_seconds)
                try:
                    returncode = process.wait(timeout=min(0.25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    if returncode != 0:
        raise AsyncRetentionError(f'CPU retention child exited {returncode}; inspect {log_path}')
    if not Path(result_path).is_file():
        raise AsyncRetentionError('CPU retention child returned without committed result')
    result = json.loads(Path(result_path).read_text())
    if result.get('schema') != RESULT_SCHEMA or result.get('job_sha256') != job_sha256:
        raise AsyncRetentionError('CPU retention result does not bind the submitted job')
    return result


class AsyncCheckpointRetention:
    """Bounded owner-thread interface; no second queued or unconsumed job.

    `submit` requires a complete immutable distributed saver receipt. While
    `pending` is true, only the worker may touch `storage`, including validation
    and destination registration. `poll`/`drain` consume a result exactly once.
    The caller updates training reports/collectives only after that return.
    """
    def __init__(self, storage, evidence_dir, storage_prefix, *, worker_timeout_seconds=480,
                 retain_hook=None, source_pins=None):
        if (isinstance(worker_timeout_seconds, bool) or not math.isfinite(worker_timeout_seconds)
                or not 0 < worker_timeout_seconds <= 480):
            raise ValueError('CPU retention timeout must be positive and at most 480 seconds')
        self.storage = storage
        self.prefix = storage_prefix
        self.root = Path(evidence_dir) / 'async-retention'
        self.root.mkdir(exist_ok=False)
        self.timeout = worker_timeout_seconds
        self._retain = retain_in_child if retain_hook is None else retain_hook
        self._owner = threading.get_ident()
        self._sources = {str(Path(__file__).resolve()): _sha(__file__)}
        for path, digest in (source_pins or {}).items():
            resolved = Path(path) if Path(path).is_absolute() else ROOT / path
            self._sources[str(resolved)] = digest
        _frozen_sources(self._sources)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-retention')
        self._future = None
        self._pending = None
        self._closed = False
        self._failed = False
        self._last_submitted = -1
        self._cancel = threading.Event()

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Only the creating training thread may manage retention')

    @property
    def pending(self):
        self._check_owner()
        return self._future is not None

    def submit(self, receipt):
        self._check_owner()
        if self._closed or self._failed:
            raise RuntimeError('Retention manager is closed or failed')
        if self._future is not None:
            raise RuntimeError('Drain the outstanding checkpoint before submitting another')
        frozen = _clone(receipt)
        if 'retention' in frozen:
            raise ValueError('Submit a local committed receipt, not an already published receipt')
        update = frozen.get('counters', {}).get('optimizer_updates')
        if type(update) is not int or update < 0 or update <= self._last_submitted:
            raise ValueError('Checkpoint updates must advance monotonically')
        _frozen_sources(self._sources)
        directory = self.root / f'update-{update:06d}'
        directory.mkdir(exist_ok=False)
        job = {'schema': SCHEMA, 'receipt': frozen, 'storage_prefix': self.prefix,
               'source_pins': dict(self._sources), 'worker_timeout_seconds': self.timeout,
               'submitted_at_unix_seconds': time.time()}
        path = directory / 'job.json'
        pin = _write_durable(path, job, once=True)
        self._pending = {'schema': SCHEMA, 'update': update, 'job_path': str(path),
                         'job_sha256': pin, 'state': 'pending_not_cloud_durable',
                         'submitted_at_unix_seconds': job['submitted_at_unix_seconds']}
        self._last_submitted = update
        self._future = self._executor.submit(self._work, job, path, pin)
        return _clone(self._pending)

    def _work(self, job, path, pin):
        started = time.perf_counter()
        timings = {}
        try:
            if self._cancel.is_set():
                raise AsyncRetentionError('Retention cancelled before local validation')
            _frozen_sources(job['source_pins'])
            phase = time.perf_counter()
            self.storage.validate_local(job['receipt'])
            timings['local_validation_seconds'] = time.perf_counter() - phase
            phase = time.perf_counter()
            remote = self._retain(path, pin, path.parent/'remote-result.json',
                                  path.parent/'worker.log', timeout_seconds=self.timeout,
                                  cancel_event=self._cancel)
            timings['cloud_child_seconds'] = time.perf_counter() - phase
            if remote.get('schema') != RESULT_SCHEMA or remote.get('job_sha256') != pin:
                raise AsyncRetentionError('Remote result does not bind the immutable job')
            if self._cancel.is_set():
                raise AsyncRetentionError('Retention cancelled before durable publication')
            # The existing publisher revalidates all bytes, metadata and generations
            # before replacing the durable recovery authority or pruning anything.
            receipt = {**job['receipt'], 'retention': _clone(remote['retention'])}
            phase = time.perf_counter()
            publication = self.storage.publish_retained(receipt)
            timings['verified_publication_and_pruning_seconds'] = time.perf_counter() - phase
            timings['total_background_seconds'] = time.perf_counter() - started
            result = {'schema': RESULT_SCHEMA, 'status': 'published', 'update':
                receipt['counters']['optimizer_updates'], 'job_sha256': pin,
                'receipt': receipt, 'storage_publication': publication, 'timing': timings,
                'worker': remote.get('worker', {}),
                'submitted_at_unix_seconds': job['submitted_at_unix_seconds'],
                'completed_at_unix_seconds': time.time()}
            _write_durable(path.parent/'result.json', result, once=True)
            return result
        except BaseException as error:
            failure = {'schema': RESULT_SCHEMA, 'status': 'failed', 'job_sha256': pin,
                'update': job['receipt']['counters']['optimizer_updates'],
                'error': f'{type(error).__name__}: {error}',
                'elapsed_seconds': time.perf_counter() - started,
                'recovery': 'Use last verified publication; failed local checkpoint is not deleted.'}
            try:
                _write_durable(path.parent/'failure.json', failure, once=True)
            except Exception:
                pass
            raise AsyncRetentionError(f"Checkpoint {failure['update']} retention failed: {failure['error']}") from None

    def _consume(self, wait):
        self._check_owner()
        if self._future is None or (not wait and not self._future.done()):
            return None
        future = self._future
        try:
            result = future.result()
        except BaseException:
            self._failed = True
            raise
        finally:
            self._future = None
            self._pending = None
        return _clone(result)

    def poll(self):
        return self._consume(False)

    def drain(self):
        return self._consume(True)

    def close(self, *, cancel=False):
        self._check_owner()
        if cancel:
            self._cancel.set()
        try:
            return self.drain()
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=False)

    def abort(self):
        """Reap the child on trainer failure; never delete a pending checkpoint.

        In-progress local hash/publication finishes before the thread can join.
        If publication already started it is allowed to finish consistently.
        No collectives or model operations are attempted during teardown.
        """
        try:
            result = self.close(cancel=True)
            return {'status': 'closed', 'result': result}
        except AsyncRetentionError as error:
            return {'status': 'cancelled_or_failed', 'error': str(error)}


def worker(job_path, job_sha256, result_path):
    """CPU process entry. Reuse accepted cloud verification without tensor loads."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Retention child requires explicitly disabled CUDA visibility')
    if any(key in os.environ for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT')):
        raise RuntimeError('Retention child must not inherit a process-group environment')
    if _sha(job_path) != job_sha256:
        raise ValueError('Retention job SHA256 differs')
    job = json.loads(Path(job_path).read_text())
    if job.get('schema') != SCHEMA:
        raise ValueError('Unknown async retention job schema')
    _frozen_sources(job['source_pins'])
    # These imports stay exclusively in this fresh, CPU-hidden process. The
    # historical helper imports torch but performs no tensor/model operation.
    from google.cloud import storage
    from scripts.openelm_retain import file_digest
    from scripts.olmo_two_gpu_retain import upload_verified
    from scripts.olmo_pilot_execution_restore import committed_metadata, MANIFEST_FIELDS
    import torch
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError('Retention child unexpectedly initialized CUDA or distributed state')
    receipt = job['receipt']
    committed_metadata({key: receipt[key] for key in MANIFEST_FIELDS})
    prefix = job['storage_prefix']
    if (not isinstance(prefix, str) or not prefix.startswith('gs://fast-chunks/cdrm-w-latent/')
            or any(part in ('', '.', '..') for part in prefix[5:].split('/'))
            or any(char in prefix for char in ('?', '#', '%'))):
        raise ValueError('Require an explicit project checkpoint prefix')
    bucket_name, key = prefix[5:].split('/', 1)
    bucket = storage.Client().bucket(bucket_name)
    directory = Path(receipt['directory'])
    objects = []
    started = time.perf_counter()
    for name in ('state.pt', 'manifest.json'):
        digest = file_digest(directory/name)
        expected = receipt['state'] if name == 'state.pt' else {'sha256': receipt['manifest_sha256']}
        if any(digest[key] != value for key, value in expected.items() if key in digest):
            raise ValueError('Committed local checkpoint bytes changed before upload')
        objects.append(upload_verified(bucket, f'{key}/{directory.name}/{name}', directory/name,
                                       digest, download_sha256=True))
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError('Retention child unexpectedly initialized CUDA or distributed state')
    result = {'schema': RESULT_SCHEMA, 'job_sha256': job_sha256,
        'retention': {'objects': objects, 'create_only': True, 'download_sha256_verified': True},
        'worker': {'pid': os.getpid(), 'cuda_initialized': False, 'distributed_initialized': False,
                   'sdk_seconds': time.perf_counter()-started, 'cuda_visible_devices': ''}}
    _write_durable(result_path, result, once=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--job-sha256', required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args(argv)
    worker(args.worker, args.job_sha256, args.result)


if __name__ == '__main__':
    main()
