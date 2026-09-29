#!/usr/bin/env python3
"""Restore one published distributed boundary from pinned cloud generations.

Operational CPU asset download only: no torch imports, tensors, model execution,
data restoration or resume launch. A successful/complete run report is irrelevant.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olmo_campaign_recovery_bundle import (SHA_PATTERN, pinned_json,
    regular, relative_name, sha, write_manifest_once)

SCHEMA = 'olmo-campaign-execution-restore-v1'
CHECKPOINT_SCHEMA = 'olmo-replicated-ddp-checkpoint-v1'
IDENTITY_SCHEMA = 'olmo-campaign-execution-identity-v1'
REMOTE_ROOT = 'gs://fast-chunks/cdrm-w-latent/'
MAX_STATE_BYTES = 20*1024**3
MAX_MANIFEST_BYTES = 16*1024**2
MAX_PUBLICATION_BYTES = 32*1024**2
PROTOCOL = 'docs/reports/olmo-campaign-execution/recovery.md'
MANIFEST_FIELDS = {'schema', 'world_size', 'metadata', 'counters', 'rank_cursors', 'state'}
VERIFICATION = {'download_sha256', 'server_md5', 'server_size', 'sha256_metadata'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pin(value):
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError('Require an explicit lowercase SHA256 pin')
    return value


def positive(value, *, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError('Object size or count exceeds its positive bound')
    return value


def source_hashes():
    names = ('scripts/olmo_campaign_execution_restore.py', 'tests/test_campaign_execution_restore.py',
             'scripts/olmo_campaign_recovery_bundle.py', PROTOCOL)
    return {name: sha(ROOT/name) for name in names}


def object_record(record, filename):
    if not isinstance(record, dict) or filename not in ('state.pt', 'manifest.json'):
        raise ValueError('Require a state/manifest object record')
    uri = record.get('uri', '')
    if not isinstance(uri, str):
        raise ValueError('Cloud URI must be a string')
    parsed = urlparse(uri)
    if (not uri.startswith(REMOTE_ROOT) or parsed.scheme != 'gs' or parsed.netloc != 'fast-chunks'
            or parsed.query or parsed.fragment or '%' in parsed.path or not parsed.path.endswith('/'+filename)):
        raise ValueError('Restore only explicit fast-chunks project checkpoint objects')
    relative_name(parsed.path.lstrip('/'))
    generation = record.get('generation')
    if not isinstance(generation, str) or not generation.isascii() or not generation.isdigit() or int(generation) < 1:
        raise ValueError('Each object requires its own exact positive generation')
    positive(record.get('size_bytes'), maximum=MAX_STATE_BYTES if filename == 'state.pt' else MAX_MANIFEST_BYTES)
    pin(record.get('sha256'))
    try:
        md5 = base64.b64decode(record.get('md5_base64', ''), validate=True)
    except (ValueError, TypeError):
        raise ValueError('Require the retained MD5 authority') from None
    if len(md5) != 16:
        raise ValueError('Require the retained MD5 authority')
    checks = record.get('verification', {})
    if set(checks) != VERIFICATION or not all(v is True for v in checks.values()):
        raise ValueError('Object lacks complete producer verification')
    return parsed.netloc, parsed.path.lstrip('/'), int(generation)


def committed_metadata(manifest):
    """Basic JSON self-consistency only; strict execution/tensor load remains separate."""
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS or manifest['schema'] != CHECKPOINT_SCHEMA:
        raise ValueError('Require the complete committed distributed manifest')
    world = positive(manifest['world_size'], maximum=64)
    meta = manifest['metadata']
    if (not isinstance(meta, dict) or meta.get('schema') != CHECKPOINT_SCHEMA or meta.get('world_size') != world
            or not isinstance(meta.get('model_type'), str) or not meta['model_type']):
        raise ValueError('Committed checkpoint topology/model metadata differs')
    identity = meta.get('configuration', {}).get('execution_identity', {})
    if (set(identity) != {'schema', 'payload', 'sha256'} or identity.get('schema') != IDENTITY_SCHEMA
            or not isinstance(identity.get('payload'), dict) or identity['sha256'] != digest(identity['payload'])
            or meta.get('source_fingerprint', {}).get('execution_identity_sha256') != identity['sha256']):
        raise ValueError('Execution identity hash/source metadata differ')
    payload = identity['payload']
    if (payload.get('arm') not in ('B', 'N', 'F', 'R', 'NF', 'NR', 'FR', 'NFR')
            or payload.get('partition', {}).get('world_size') != world
            or payload.get('cursor_schema') != 'olmo-campaign-execution-cursor-v1'):
        raise ValueError('Execution identity arm/topology/cursor differs')
    batch = positive(payload['partition'].get('physical_batch_per_rank'), maximum=65536)
    counters = manifest['counters']
    if (not isinstance(counters, dict) or set(counters) != {'optimizer_updates', 'input_tokens', 'documents',
            'microbatches', 'ce_positions', 'latent_pairs', 'kl_triples'}
            or any(type(v) is not int or v < 0 for v in counters.values())):
        raise ValueError('Committed counters must be complete nonnegative integer counts')
    ranks = manifest['rank_cursors']
    if not isinstance(ranks, list) or len(ranks) != world:
        raise ValueError('Committed rank cursor count differs')
    first = None
    for rank, row in enumerate(ranks):
        if (not isinstance(row, dict) or set(row) != {'schema', 'rank', 'world_size', 'physical_batch_per_rank', 'cursor'}
                or row['schema'] != payload['cursor_schema'] or type(row['rank']) is not int or row['rank'] != rank
                or row['world_size'] != world or row['physical_batch_per_rank'] != batch
                or not isinstance(row['cursor'], dict) or row['cursor'].get('next_update') != counters['optimizer_updates']):
            raise ValueError('Committed cursor and optimizer boundary disagree')
        if first is None:
            first = row['cursor']
        elif row['cursor'] != first:
            raise ValueError('Ranks disagree on committed logical cursor')
    state = manifest['state']
    if not isinstance(state, dict) or set(state) != {'filename', 'size_bytes', 'sha256'} or state['filename'] != 'state.pt':
        raise ValueError('Committed state filename differs')
    positive(state['size_bytes'], maximum=MAX_STATE_BYTES); pin(state['sha256'])
    return identity


def publication_metadata(receipt):
    if not isinstance(receipt, dict) or set(receipt) != MANIFEST_FIELDS|{'directory', 'manifest_sha256', 'retention'}:
        raise ValueError('Require a published latest-checkpoint receipt')
    manifest = {key: receipt[key] for key in MANIFEST_FIELDS}
    identity = committed_metadata(manifest)
    pin(receipt['manifest_sha256'])
    retention = receipt['retention']
    if (not isinstance(retention, dict) or set(retention) != {'objects', 'create_only', 'download_sha256_verified'}
            or retention['create_only'] is not True or retention['download_sha256_verified'] is not True
            or not isinstance(retention['objects'], list) or len(retention['objects']) != 2):
        raise ValueError('Receipt lacks a completely retained object pair')
    objects = {}
    for row in retention['objects']:
        filename = row.get('uri', '').rsplit('/', 1)[-1] if isinstance(row, dict) else None
        object_record(row, filename)
        if filename in objects:
            raise ValueError('Duplicate checkpoint object')
        objects[filename] = row
    if (set(objects) != {'manifest.json', 'state.pt'}
            or objects['manifest.json']['uri'].rsplit('/', 1)[0] != objects['state.pt']['uri'].rsplit('/', 1)[0]
            or objects['manifest.json']['sha256'] != receipt['manifest_sha256']
            or any(objects['state.pt'][k] != manifest['state'][k] for k in ('sha256', 'size_bytes'))):
        raise ValueError('Retained objects do not bind the same checkpoint boundary')
    return manifest, objects, identity


class HashingWriter:
    """Append-only disk sink with constant-size digest state and strict byte cap."""
    def __init__(self, stream, expected):
        self.stream, self.expected = stream, expected
        self.count = 0
        self.sha256, self.md5 = hashlib.sha256(), hashlib.md5()

    def write(self, data):
        view = memoryview(data)
        if self.count+view.nbytes > self.expected['size_bytes']:
            raise ValueError('Download exceeds its declared byte bound')
        written = self.stream.write(view)
        if written != view.nbytes:
            raise OSError('Incomplete local checkpoint write')
        self.sha256.update(view); self.md5.update(view); self.count += written
        return written

    def tell(self):
        return self.count

    def seek(self, *_args, **_kwargs):
        raise ValueError('Streaming restore rejects rewind/transcoding')

    def finish(self):
        observed = {'size_bytes': self.count, 'sha256': self.sha256.hexdigest(),
                    'md5_base64': base64.b64encode(self.md5.digest()).decode('ascii')}
        if any(observed[k] != self.expected[k] for k in observed):
            raise ValueError('Downloaded size/SHA256/MD5 differs from pinned object')
        self.stream.flush(); os.fsync(self.stream.fileno())
        return observed


def google_fetch(record, sink):
    """No list/latest fallback; SDK default retry keeps the generation precondition."""
    from google.cloud import storage
    filename = record['uri'].rsplit('/', 1)[-1]
    bucket, key, generation = object_record(record, filename)
    blob = storage.Client().bucket(bucket).blob(key, generation=generation)
    blob.reload(if_generation_match=generation, timeout=(15, 60))
    if (str(blob.generation) != record['generation'] or blob.size != record['size_bytes']
            or blob.md5_hash != record['md5_base64'] or (blob.metadata or {}).get('sha256') != record['sha256']
            or blob.content_encoding is not None):
        raise ValueError('Exact-generation cloud metadata or content encoding differs')
    blob.download_to_file(sink, if_generation_match=generation, checksum='auto',
                          single_shot_download=False, raw_download=True, timeout=(15, 60))
    return {'generation': str(blob.generation), 'size_bytes': blob.size,
            'md5_base64': blob.md5_hash, 'sha256_metadata': blob.metadata['sha256'], 'content_encoding': None}


def download(record, path, fetch):
    filename = record['uri'].rsplit('/', 1)[-1]
    object_record(record, filename)
    with Path(path).open('xb') as stream:
        sink = HashingWriter(stream, record)
        remote = fetch(record, sink)
        verified = sink.finish()
    return {'object': record, 'verified': verified, 'remote_metadata': remote}


def durable_publish(partial, destination):
    os.link(partial, destination)  # No overwrite, even if another process races us.
    Path(partial).unlink()
    descriptor = os.open(Path(destination).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fresh_destination(path, persistent_root):
    path, root = Path(path).absolute(), Path(persistent_root).resolve()
    if '..' in path.parts or not path.is_relative_to(root) or path == root or path.exists() or path.is_symlink():
        raise ValueError('Use a fresh destination under the persistent project checkout')
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Recovery destination has a symlink parent')
    return path


def restore(publication_path, publication_sha256, output_dir, *, fetch=google_fetch, persistent_root=ROOT):
    publication_path = regular(publication_path)
    if publication_path.stat().st_size > MAX_PUBLICATION_BYTES:
        raise ValueError('Publication receipt exceeds its metadata bound')
    receipt = pinned_json(publication_path, pin(publication_sha256))
    manifest, objects, identity = publication_metadata(receipt)
    output = fresh_destination(output_dir, persistent_root)
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    need = sum(o['size_bytes'] for o in objects.values())+64*1024**2
    if shutil.disk_usage(ancestor).free < need:
        raise ValueError('Insufficient persistent disk for the pinned state and metadata')
    sources = source_hashes(); started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output/'checkpoint'; checkpoint.mkdir()
    authority = output/'authorities'; authority.mkdir()
    (authority/'latest-checkpoint.json').write_bytes(publication_path.read_bytes())
    if sha(authority/'latest-checkpoint.json') != publication_sha256:
        raise ValueError('Publication receipt changed before recovery')
    for name, digest_ in sources.items():
        destination = output/'source-snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
        if sha(destination) != digest_:
            raise ValueError('Recovery source snapshot differs')
    downloads = []
    try:
        partial_manifest, partial_state = checkpoint/'manifest.json.partial', checkpoint/'state.pt.partial'
        downloads.append(download(objects['manifest.json'], partial_manifest, fetch))
        decoded = pinned_json(partial_manifest, receipt['manifest_sha256'])
        committed_metadata(decoded)
        if decoded != manifest:
            raise ValueError('Downloaded manifest differs from the published boundary')
        downloads.append(download(objects['state.pt'], partial_state, fetch))
        if source_hashes() != sources or sha(publication_path) != publication_sha256:
            raise ValueError('Source or publication authority changed during download')
        durable_publish(partial_state, checkpoint/'state.pt')
        durable_publish(partial_manifest, checkpoint/'manifest.json')
        report = {'schema': SCHEMA, 'status': 'checkpoint_assets_verified_launch_pending', 'sources': sources,
            'publication_sha256': publication_sha256, 'checkpoint_manifest_sha256': receipt['manifest_sha256'],
            'execution_identity_sha256': identity['sha256'], 'completed_optimizer_updates': manifest['counters']['optimizer_updates'],
            'checkpoint_directory': str(checkpoint), 'downloads': downloads,
            'total_downloaded_bytes': sum(o['size_bytes'] for o in objects.values()),
            'elapsed_seconds': time.monotonic()-started, 'model_tensors_loaded': False,
            'gpu_used': False, 'cloud_writes': False, 'original_receipt_local_paths_used': False,
            'qualification': 'Only checkpoint asset bytes and basic committed metadata verified. Matching execution contract, sources, corpus/index, runtime/topology and full tensor/Adam/scheduler/RNG validation remain the runner responsibility. No launch or numerical clearance.'}
        write_manifest_once(output/'report.json', report)
        return report
    except Exception as error:
        write_manifest_once(output/'failure.json', {'schema': SCHEMA, 'status': 'failed_assets_not_completed',
            'error': f'{type(error).__name__}: {error}', 'completed_downloads': downloads,
            'checkpoint_manifest_published': (checkpoint/'manifest.json').exists(),
            'partial_bytes_retained': True, 'publication_sha256': publication_sha256})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--publication-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if (not Path('/.dockerenv').exists() or Path.cwd() != Path('/workspace/cdrm-w-latent')
            or list(Path('/dev').glob('nvidia[0-9]*')) or 'torch' in sys.modules):
        raise RuntimeError('Use the project CPU-only container; no torch/model/GPU process')
    result = restore(args.publication, args.publication_sha256, args.output_dir)
    print(json.dumps({'status': result['status'], 'checkpoint_manifest_sha256': result['checkpoint_manifest_sha256'],
                     'completed_optimizer_updates': result['completed_optimizer_updates']}))


if __name__ == '__main__':
    main()
