#!/usr/bin/env python3
"""CPU-only fixed-generation readback memory comparison; never loads tensors."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / 'docs/reports/olmo-campaign-lifecycle/readback-protocol.md'
REPORT = ROOT / '.runtime/olmo-campaign-lifecycle/base-reference-01/report.json'
REPORT_SHA = '6e945cf271c1791a3dcb4e71f2f481e8389bb59477dad1f78097db2a23563d25'
STATE = {'uri': 'gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/20260929T075900Z/base-loop-base-reference-01/update-000001/state.pt',
         'generation': '1790680957093316', 'size_bytes': 14154933413,
         'sha256': '5ed5a23f7d9f2753e76bbf95ed45797f8ddafb30f82ca7b276dc8e98ce2203e5',
         'md5_base64': 'oNJ04OABVkvef0iAR49pVA=='}
SCHEMA = 'olmo-checkpoint-readback-probe-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def source_hashes():
    paths = [Path(__file__), ROOT/'tests/test_checkpoint_readback_probe.py', PROTOCOL,
             ROOT/'scripts/experiment_tracking.py', ROOT/'scripts/olmo_two_gpu_retain.py',
             ROOT/'scripts/openelm_retain.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def snapshot_sources(directory, sources):
    for name in sources:
        destination = directory/'source-snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())


def validate_child_environment(observed, parent, previous_pids):
    keys = ('python', 'platform', 'google_cloud_storage', 'sdk_source_sha256', 'rss_units')
    if (any(observed[k] != parent[k] for k in keys) or observed['pid'] in {parent['pid'], *previous_pids}
            or observed['torch_loaded'] or observed['cuda_device_nodes_present']):
        raise ValueError('Fresh CPU child or environment contract differs')


def endpoint_authority(path=REPORT, expected_sha=REPORT_SHA):
    """Authenticate the saved local/publication chain, not freeform CLI metadata."""
    path = Path(path)
    if sha(path) != expected_sha:
        raise ValueError('Completed reference report SHA differs')
    report = json.loads(path.read_text())
    if report.get('status') != 'passed' or report.get('phase') != 'reference':
        raise ValueError('Require completed reference')
    local = [r['receipt'] for r in report.get('local_checkpoints', []) if r.get('optimizer_update') == 1]
    if len(local) != 1:
        raise ValueError('Require unique local update-1 receipt')
    published = [r for r in report.get('published_checkpoints', [])
                 if r.get('manifest_sha256') == local[0]['manifest_sha256']]
    if len(published) != 1 or published[0]['counters']['optimizer_updates'] != 1:
        raise ValueError('Require matching published update-1 receipt')
    receipt = published[0]
    if (receipt['state'] != local[0]['state'] or not receipt['retention'].get('download_sha256_verified')
            or not receipt['retention'].get('create_only')):
        raise ValueError('Retained checkpoint authority differs')
    objects = receipt['retention']['objects']
    states = [r for r in objects if r['uri'] == STATE['uri']]
    manifests = [r for r in objects if r['uri'] == STATE['uri'].rsplit('/', 1)[0] + '/manifest.json']
    if len(objects) != 2 or len(states) != 1 or len(manifests) != 1:
        raise ValueError('Require exact state/manifest object pair')
    state = states[0]
    if (any(state.get(k) != v for k, v in STATE.items())
            or any(receipt['state'].get(k) != STATE[k] for k in ('sha256', 'size_bytes'))
            or manifests[0]['sha256'] != receipt['manifest_sha256']
            or not all(r.get('verification', {}).get('download_sha256') for r in objects)):
        raise ValueError('Pinned retained object metadata differs')
    if sha(path) != expected_sha:
        raise ValueError('Reference changed while reading')
    return {'report': str(path), 'report_sha256': expected_sha,
            'checkpoint_manifest_sha256': receipt['manifest_sha256'], 'state': dict(STATE)}


class HashSink:
    """Append-only sink: no allocation proportional to the downloaded object."""
    def __init__(self, expected):
        self.expected = dict(expected)
        self.count = 0
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()
        self.write_calls = 0
        self.max_write_bytes = 0

    def write(self, data):
        view = memoryview(data)
        length = view.nbytes
        if self.count + length > self.expected['size_bytes']:
            raise ValueError('Readback overflow')
        self.sha256.update(view)
        self.md5.update(view)
        self.count += length
        self.write_calls += 1
        self.max_write_bytes = max(self.max_write_bytes, length)
        return length

    def seek(self, *args):
        raise ValueError('Hash sink cannot seek or accept transcoding rewind')

    def tell(self):
        return self.count

    def finish(self):
        result = {'size_bytes': self.count, 'sha256': self.sha256.hexdigest(),
                  'md5_base64': base64.b64encode(self.md5.digest()).decode('ascii')}
        if any(result[k] != self.expected[k] for k in result):
            raise ValueError('Readback length or digest differs')
        return result


def validate_remote(blob, expected):
    if blob is None or (str(blob.generation) != expected['generation']
            or blob.size != expected['size_bytes'] or blob.md5_hash != expected['md5_base64']
            or (blob.metadata or {}).get('sha256') != expected['sha256']
            or blob.content_encoding is not None):
        raise ValueError('Exact-generation remote metadata or encoding differs')
    return {'generation': str(blob.generation), 'size_bytes': blob.size,
            'md5_base64': blob.md5_hash, 'sha256_metadata': blob.metadata['sha256'],
            'content_encoding': blob.content_encoding, 'chunk_size': blob.chunk_size}


def current_rss_kib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])
    raise RuntimeError('Linux current RSS unavailable')


def measure(method, blob, expected, *, clock=time.perf_counter):
    if method not in ('bytes', 'stream'):
        raise ValueError('Unknown readback method')
    if blob.chunk_size is not None:
        raise ValueError('Keep default streaming single-resource SDK path')
    sink = HashSink(expected)
    before_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    before_current = current_rss_kib()
    started = clock()
    kwargs = {'if_generation_match': int(expected['generation']), 'single_shot_download': False,
              'checksum': 'auto', 'timeout': (15, 60)}
    # SDK DEFAULT_RETRY remains active; children are independently time-bounded.
    if method == 'bytes':
        payload = blob.download_as_bytes(**kwargs)
        returned = clock()
        sink.write(payload)
        result = sink.finish()
        verified = clock()
        current = current_rss_kib()  # Intentionally measure while payload is live.
    else:
        blob.download_to_file(sink, **kwargs)
        returned = clock()
        result = sink.finish()
        verified = clock()
        current = current_rss_kib()
    after_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'method': method, 'verified': result, 'total_seconds': verified-started,
            'download_call_seconds': returned-started, 'post_download_verification_seconds': verified-returned,
            'download_call_scope': 'network plus SDK buffering/checksum; Python SHA256/MD5 included only for stream',
            'rss_kib': {'peak_before': before_peak, 'peak_after': after_peak,
                        'peak_increase': after_peak-before_peak, 'current_before': before_current,
                        'current_after_with_bytes_payload_live': current,
                        'peak_minus_baseline_current': after_peak-before_current},
            'write_calls': sink.write_calls, 'max_write_bytes': sink.max_write_bytes,
            'source_generation': expected['generation']}


def environment():
    from google.cloud.storage import blob as sdk_blob
    from google.cloud.storage._media.requests import download as sdk_download
    return {'python': sys.version, 'platform': platform.platform(), 'pid': os.getpid(),
            'google_cloud_storage': importlib.metadata.version('google-cloud-storage'),
            'sdk_source_sha256': {str(Path(x.__file__)): sha(x.__file__) for x in (sdk_blob, sdk_download)},
            'rss_units': 'Linux KiB', 'torch_loaded': 'torch' in sys.modules,
            'cuda_device_nodes_present': Path('/dev/nvidia0').exists()}


def run_child(method, output_dir):
    from google.cloud import storage
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    snapshot_sources(output_dir, sources)
    report = {'schema': SCHEMA, 'status': 'running', 'method': method, 'sources': sources}
    def timeout_handler(*_):
        raise TimeoutError('Bounded child deadline')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)
    try:
        authority = endpoint_authority()
        report.update(authority=authority, environment=environment())
        started = time.perf_counter()
        bucket, key = STATE['uri'][5:].split('/', 1)
        blob = storage.Client().bucket(bucket).get_blob(key, generation=int(STATE['generation']))
        report['remote'] = validate_remote(blob, STATE)
        report['metadata_seconds'] = time.perf_counter()-started
        report['measurement'] = measure(method, blob, STATE)
        if source_hashes() != sources or endpoint_authority() != authority:
            raise ValueError('Source or authority changed')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        write_json(output_dir/'report.json', report)
    return report


def run_parent(output_dir):
    from scripts.experiment_tracking import OnlineTracker, scalar_metrics
    output_dir.mkdir(parents=True, exist_ok=False)
    sources, authority = source_hashes(), endpoint_authority()
    snapshot_sources(output_dir, sources)
    report = {'schema': SCHEMA, 'status': 'running', 'sources': sources, 'authority': authority,
              'environment': environment(), 'methods': [],
              'scope': 'One ordered CPU readback pair; memory diagnostic, no speed superiority or retainer adoption claim'}
    tracker = OnlineTracker(project='pretrained-fbt-rt-nextlat', output_dir=output_dir,
                            group='checkpoint-readback', name=output_dir.name)
    def persist():
        report['wandb'] = tracker.record
        write_json(output_dir/'report.json', report)
    try:
        tracker.start({'scope': report['scope'], 'authority': authority, 'sources': sources})
        persist()
        for index, method in enumerate(('bytes', 'stream')):
            child = output_dir/method
            with (output_dir/(method+'.log')).open('x') as log:
                completed = subprocess.run([sys.executable, '-m', 'scripts.olmo_checkpoint_readback_probe',
                    '--child', method, '--output-dir', str(child)], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT, timeout=330, check=False)
            path = child/'report.json'
            result = json.loads(path.read_text()) if path.exists() else {'status': 'missing_report'}
            report['methods'].append({'method': method, 'exit_code': completed.returncode,
                                      'report_sha256': sha(path) if path.exists() else None, 'result': result})
            persist()
            if (completed.returncode != 0 or result['status'] != 'passed' or result['authority'] != authority
                    or result['sources'] != sources or result['measurement']['verified'] != {k:STATE[k] for k in ('size_bytes','sha256','md5_base64')}):
                raise ValueError('Child did not verify the exact authority')
            validate_child_environment(result['environment'], report['environment'],
                                       [r['result']['environment']['pid'] for r in report['methods'][:-1]])
            tracker.log({'update': index+1, **scalar_metrics(result['measurement'], 'benchmark/'+method)}, step=index+1)
        if source_hashes() != sources or endpoint_authority() != authority:
            raise ValueError('Source or authority changed')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        persist()
        try:
            tracker.finish(succeeded=report['status'] == 'passed')
        except BaseException as error:
            report.update(status='failed', tracking_error_type=type(error).__name__)
            raise
        finally:
            persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--child', choices=('bytes','stream'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not Path('/.dockerenv').exists() or Path('/dev/nvidia0').exists() or Path.cwd() != ROOT:
        raise RuntimeError('Use the project CPU-only container')
    if args.child:
        run_child(args.child, args.output_dir)
    else:
        run_parent(args.output_dir)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # Do not emit SDK exception text or chained auth/request details.
        print('Readback probe failed:', type(error).__name__, file=sys.stderr)
        sys.exit(1)
