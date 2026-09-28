#!/usr/bin/env python3
"""Retain bounded complete-document prefixes from explicitly selected Dolma URLs.

This produces a source-coverage fixture, not a random or representative corpus.
Only extracted bytes are SHA256 verified; remote multipart ETags are provenance,
not a claim of verification of the entire upstream gzip object.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.olmo_artifacts import FILE_SPECS, OLMoNativeTokenizer, REVISION, REPO_ID


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


class LimitedReader:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.count = stream, limit, 0

    def read(self, n=-1):
        n = min(1024 * 1024 if n < 0 else n, self.limit - self.count + 1)
        value = self.stream.read(n)
        self.count += len(value)
        if self.count > self.limit:
            raise ValueError('Compressed source-byte limit exceeded')
        return value


def extract_source(source, destination, **kwargs):
    """One writer per source; an interrupted partial is never a committed input."""
    name = source['name']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
        raise ValueError('Unsafe source name')
    root = Path(destination)
    if root.is_symlink():
        raise ValueError('Symlink destination is not supported')
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ('.extract-' + name + '.lock')
    if lock_path.is_symlink():
        raise ValueError('Symlink lock is not supported')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _extract_source(source, destination, **kwargs)


def _extract_source(source, destination, *, tokenizer, token_target, max_documents,
                   max_compressed_bytes, max_line_bytes, upstream_manifest, tokenizer_sha):
    name = source['name']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name):
        raise ValueError('Unsafe source name')
    url = source['url']
    if not url.startswith('https://olmo-data.org/'):
        raise ValueError('Only explicitly pinned official Dolma source URLs are accepted')
    for value in (token_target, max_documents, max_compressed_bytes, max_line_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError('Extraction bounds must be positive integers')
    dest = Path(destination) / name
    config = {'source': source, 'upstream_manifest': upstream_manifest,
              'tokenizer_sha256': tokenizer_sha, 'target_tokens': token_target,
              'max_documents': max_documents, 'max_compressed_bytes': max_compressed_bytes,
              'max_line_bytes': max_line_bytes,
              'policy': 'complete-document source prefix; no truncation; stop after token target',
              'empty_text_policy': 'skip empty/whitespace-only string with audited source line; reject nonstring'}
    if dest.exists():
        if dest.is_symlink():
            raise ValueError('Symlink destination is not supported')
        report = json.loads((dest/'manifest.json').read_text())
        if (report.get('schema') != 'olmo-dolma-source-extract-v2'
                or report.get('status') != 'complete'
                or set(report.get('files', {})) != {'raw.jsonl', 'source-lines.jsonl'}):
            raise ValueError('Existing extraction commit marker is incomplete')
        if report['config'] != config or any((dest/name).is_symlink()
                or (dest/name).stat().st_size != record['size_bytes']
                or digest(dest/name) != record['sha256'] for name, record in report['files'].items()):
            raise ValueError('Existing extraction configuration or raw bytes differ')
        return report
    partial = dest.with_name(dest.name + '.partial')
    if partial.exists():
        if partial.is_symlink() or not partial.is_dir():
            raise ValueError('Unsafe extraction partial path')
        # This directory belongs only to an uncommitted attempt at this source.
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    started = time.monotonic()
    count = tokens = lines = skipped = 0
    special_tokens = 0
    request = urllib.request.Request(url, headers={'User-Agent': 'curl/8.0', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=60) as response:
        remote = {'requested_url': url, 'resolved_url': response.geturl(),
                  'etag': response.headers.get('ETag'),
                  'last_modified': response.headers.get('Last-Modified'),
                  'content_length_header': response.headers.get('Content-Length'),
                  'status': response.status}
        if source.get('etag') is not None and remote['etag'] != source['etag']:
            raise ValueError('Upstream ETag changed from the pinned source selection')
        compressed = LimitedReader(response, max_compressed_bytes)
        with gzip.GzipFile(fileobj=compressed) as gz, (partial/'raw.jsonl').open('xb') as out, \
                (partial/'source-lines.jsonl').open('x') as mapping:
            while lines < max_documents and tokens < token_target:
                line = gz.readline(max_line_bytes + 1)
                if not line:
                    break
                lines += 1
                if len(line) > max_line_bytes:
                    raise ValueError('Source row exceeds byte bound; document was not truncated')
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
                    raise ValueError('Expected original nonempty string document id')
                text = row.get('text')
                if not isinstance(text, str):
                    raise ValueError('Nonstring source text')
                if not text.strip():
                    skipped += 1
                    mapping.write(json.dumps({'source_line': lines, 'id': row['id'],
                        'raw_line_sha256': hashlib.sha256(line).hexdigest(),
                        'status': 'skipped_empty_text'}) + '\n')
                    continue
                ids = tokenizer.encode(text)
                if not ids:
                    raise ValueError('Document tokenized to zero content tokens')
                special_tokens += ids.count(tokenizer.eos_token_id)
                tokens += len(ids) + int(ids[-1] != tokenizer.eos_token_id)
                # Preserve original source record bytes and original row order.
                out.write(line if line.endswith(b'\n') else line + b'\n')
                count += 1
                mapping.write(json.dumps({'source_line': lines, 'output_line': count,
                    'id': row['id'], 'status': 'retained'}) + '\n')
            out.flush(); os.fsync(out.fileno())
            mapping.flush(); os.fsync(mapping.fileno())
    if not count:
        raise ValueError('No source documents extracted')
    file = partial/'raw.jsonl'
    report = {'schema': 'olmo-dolma-source-extract-v2', 'status': 'complete',
              'config': config, 'remote': remote,
              'source_rows': [1, lines], 'documents': count, 'skipped_empty_text': skipped,
              'tokens_including_terminal_eos': tokens,
              'encoded_eos_in_source_text': special_tokens,
              'stop_reason': 'token_target' if tokens >= token_target else (
                  'document_limit' if lines == max_documents else 'source_eof'),
              'compressed_bytes_read': compressed.count,
              'full_upstream_gzip_sha256_verified': False,
              'elapsed_seconds': time.monotonic() - started,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'files': {name: {'size_bytes': (partial/name).stat().st_size, 'sha256': digest(partial/name)}
                        for name in ('raw.jsonl', 'source-lines.jsonl')}}
    write_json(partial/'manifest.json', report)
    partial_fd = os.open(partial, os.O_RDONLY)
    try:
        os.fsync(partial_fd)
    finally:
        os.close(partial_fd)
    partial.rename(dest)
    parent_fd = os.open(dest.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=ROOT/'.runtime/olmo1b-step60000/artifacts/native/tokenizer.json')
    parser.add_argument('--tokens-per-source', type=int, default=1_000_000)
    parser.add_argument('--only-source')
    args = parser.parse_args(argv)
    if digest(args.tokenizer) != FILE_SPECS['tokenizer.json'][1]:
        raise ValueError('Tokenizer differs from pinned OLMo checkpoint')
    tokenizer = OLMoNativeTokenizer(args.tokenizer)
    source_plan = json.loads(args.source_plan.read_text())
    plan_sha = digest(args.source_plan)
    selected = [s for s in source_plan['sources'] if args.only_source is None or s['name'] == args.only_source]
    if not selected:
        raise ValueError('No requested source selected')
    for source in selected:
        report = extract_source(source, args.output_dir, tokenizer=tokenizer,
            token_target=args.tokens_per_source, max_documents=100_000,
            max_compressed_bytes=256 * 1024**2, max_line_bytes=64 * 1024**2,
            upstream_manifest={'source_plan_sha256': plan_sha, 'release': 'v1_5',
                               'tokenizer_repo': REPO_ID, 'tokenizer_revision': REVISION},
            tokenizer_sha=FILE_SPECS['tokenizer.json'][1])
        print(json.dumps({k: report[k] for k in ('documents', 'tokens_including_terminal_eos', 'elapsed_seconds')}
                         | {'source': source['name'], 'status': report['status']}), flush=True)


if __name__ == '__main__':
    main()
