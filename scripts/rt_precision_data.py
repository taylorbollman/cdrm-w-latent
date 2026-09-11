#!/usr/bin/env python3
"""Prepare bounded, pinned C4/T5 token matrices for RT precision experiments."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np

C4_REPO = 'allenai/c4'
C4_REVISION = '1588ec454efa1a09f29cd18ddd04fe05fc8653a2'
TOKENIZER_REPO = 'google-t5/t5-base'
TOKENIZER_REVISION = 'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
TOKENIZER_FILES = ('tokenizer.json', 'spiece.model', 'config.json')
ROLES = ('diagnostic', 'dev', 'confirmation')
LENGTH = 512
VOCAB_SIZE = 32100
EOS_ID = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def heldout_role(text_hash: str, seed: int) -> str:
    digest = hashlib.sha256(f'{seed}:{text_hash}'.encode()).digest()
    return ROLES[int.from_bytes(digest[:8], 'big') % len(ROLES)]


def write_json(path: Path, value) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


class TokenPacker:
    """Append whole-document token streams, retaining final truncation metadata."""

    def __init__(self, directory: Path, role: str, rows: int, length: int = LENGTH):
        if rows < 1 or length < 2:
            raise ValueError('Require positive row count and sequence length >=2')
        self.path = directory / f'{role}.npy'
        self.boundaries_path = directory / f'{role}-documents.jsonl'
        if self.path.exists() or self.boundaries_path.exists():
            raise FileExistsError('Refusing to overwrite packed data')
        self.ids = np.lib.format.open_memmap(self.path, mode='w+', dtype=np.uint16, shape=(rows, length))
        self.flat = self.ids.reshape(-1)
        self.boundaries = self.boundaries_path.open('x')
        self.position = 0
        self.documents = 0
        self.max_token_id = 0
        self.min_token_id = VOCAB_SIZE
        self.role = role

    @property
    def complete(self) -> bool:
        return self.position == self.flat.size

    def append(self, token_ids, *, text_hash: str, source_file: str, source_row: int) -> None:
        if self.complete:
            raise ValueError('Packed corpus is already complete')
        tokens = np.asarray(token_ids)
        if tokens.ndim != 1 or tokens.dtype.kind not in 'iu' or not len(tokens):
            raise ValueError('Require a nonempty integer document token sequence')
        if int(tokens.min()) < 0 or int(tokens.max()) >= VOCAB_SIZE or int(tokens[-1]) != EOS_ID:
            raise ValueError('Require valid T5 IDs and exactly the explicitly appended final EOS')
        count = min(len(tokens), self.flat.size - self.position)
        selected = tokens[:count]
        self.flat[self.position:self.position + count] = selected
        record = {'text_sha256': text_hash, 'source_file': source_file, 'source_row': source_row,
                  'token_begin': self.position, 'token_end': self.position + count,
                  'full_document_tokens_including_eos': len(tokens),
                  'eos_retained': count == len(tokens), 'truncated_at_corpus_end': count < len(tokens)}
        self.boundaries.write(json.dumps(record, sort_keys=True) + '\n')
        self.position += count
        self.documents += 1
        self.max_token_id = max(self.max_token_id, int(selected.max()))
        self.min_token_id = min(self.min_token_id, int(selected.min()))

    def finish(self) -> dict:
        self.ids.flush()
        self.boundaries.close()
        if not self.complete:
            raise RuntimeError(f'{self.role}: only {self.position}/{self.flat.size} tokens prepared')
        return {'role': self.role, 'ids_path': self.path.name, 'ids_sha256': file_sha256(self.path),
                'ids_bytes': self.path.stat().st_size, 'dtype': 'uint16', 'shape': list(self.ids.shape),
                'tokens': self.position, 'documents': self.documents,
                'min_token_id': self.min_token_id, 'max_token_id': self.max_token_id,
                'boundaries_path': self.boundaries_path.name,
                'boundaries_sha256': file_sha256(self.boundaries_path)}


def excluded_documents(manifest_path: Path | None) -> tuple[set[str], dict | None]:
    if manifest_path is None:
        return set(), None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'rt-precision-c4-data-v1' or manifest.get('mode') != 'heldout':
        raise ValueError('Exclusion manifest must describe this preparer’s held-out data')
    hashes = set()
    for role in manifest['roles'].values():
        path = (manifest_path.parent / role['boundaries_path']).resolve()
        if path.parent != manifest_path.parent.resolve() or file_sha256(path) != role['boundaries_sha256']:
            raise ValueError('Invalid or changed document boundary file')
        for line in path.read_text().splitlines():
            hashes.add(json.loads(line)['text_sha256'])
    return hashes, {'manifest_path': str(manifest_path.resolve()),
                    'manifest_sha256': file_sha256(manifest_path), 'excluded_documents': len(hashes)}


def fetch_file(repo: str, revision: str, filename: str, directory: Path, *, repo_type='model') -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, revision=revision, repo_type=repo_type, local_dir=directory))


def prepare(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory; incomplete outputs are retained for diagnosis')
    if args.mode == 'train' and args.exclude_manifest is None:
        raise ValueError('Training preparation requires the held-out exclusion manifest')
    if args.rows < 1 or args.diagnostic_rows < 512 or args.eval_rows < 1 or args.max_shards < 1:
        raise ValueError('Invalid bounded corpus sizes')
    excluded, exclusion = excluded_documents(args.exclude_manifest)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    tokenizer_files = {}
    tokenizer_dir = args.output_dir / 'tokenizer'
    tokenizer_dir.mkdir()
    for name in TOKENIZER_FILES:
        source = fetch_file(TOKENIZER_REPO, TOKENIZER_REVISION, name, args.download_dir / 'tokenizer')
        destination = tokenizer_dir / name
        shutil.copyfile(source, destination)
        tokenizer_files[name] = {'sha256': file_sha256(destination), 'bytes': destination.stat().st_size}
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / 'tokenizer.json'))
    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB_SIZE or tokenizer.token_to_id('</s>') != EOS_ID:
        raise ValueError('Unexpected tokenizer vocabulary or EOS ID')
    tokenizer.no_padding()
    tokenizer.no_truncation()
    counts = {'train': args.rows} if args.mode == 'train' else {
        'diagnostic': args.diagnostic_rows, 'dev': args.eval_rows, 'confirmation': args.eval_rows}
    packers = {role: TokenPacker(args.output_dir, role, rows) for role, rows in counts.items()}
    sources = []
    seen = set(excluded)
    duplicate_or_excluded = 0
    read_documents = 0
    pending = []

    def flush():
        nonlocal pending
        if not pending:
            return
        encodings = tokenizer.encode_batch([item[0] for item in pending], add_special_tokens=False)
        for (_, text_hash, filename, row, role), encoding in zip(pending, encodings):
            packer = packers[role]
            if not packer.complete:
                packer.append(encoding.ids + [EOS_ID], text_hash=text_hash, source_file=filename, source_row=row)
        pending = []

    split, shard_count = ('train', 1024) if args.mode == 'train' else ('validation', 8)
    for shard in range(min(args.max_shards, shard_count)):
        filename = f'en/c4-{split}.{shard:05d}-of-{shard_count:05d}.json.gz'
        source = fetch_file(C4_REPO, C4_REVISION, filename, args.download_dir / 'c4', repo_type='dataset')
        sources.append({'file': filename, 'path': str(source.resolve()), 'sha256': file_sha256(source),
                        'bytes': source.stat().st_size})
        with gzip.open(source, 'rt', encoding='utf-8') as stream:
            for row, line in enumerate(stream):
                if all(packer.complete for packer in packers.values()):
                    break
                document = json.loads(line)
                text = document['text']
                if not isinstance(text, str) or not text:
                    raise ValueError('Unexpected empty or nontext C4 document')
                read_documents += 1
                text_hash = text_sha256(text)
                if text_hash in seen:
                    duplicate_or_excluded += 1
                    continue
                seen.add(text_hash)
                role = 'train' if args.mode == 'train' else heldout_role(text_hash, args.seed)
                if packers[role].complete:
                    continue
                pending.append((text, text_hash, filename, row, role))
                if len(pending) == args.tokenizer_batch:
                    flush()
                    if read_documents % 10240 < args.tokenizer_batch:
                        print(json.dumps({'status': 'tokenizing', 'read_documents': read_documents,
                                          'tokens': {r: p.position for r, p in packers.items()},
                                          'elapsed_seconds': time.monotonic() - started}), flush=True)
        flush()
        if all(packer.complete for packer in packers.values()):
            break
    role_records = {role: packer.finish() for role, packer in packers.items()}
    source_snapshot = args.output_dir / 'rt_precision_data.py'
    shutil.copyfile(Path(__file__), source_snapshot)
    report = {'schema': 'rt-precision-c4-data-v1', 'status': 'complete', 'mode': args.mode,
              'dataset': {'repo_id': C4_REPO, 'revision': C4_REVISION, 'language': 'en', 'split': split},
              'tokenizer': {'repo_id': TOKENIZER_REPO, 'revision': TOKENIZER_REVISION,
                            'files': tokenizer_files, 'vocab_size': VOCAB_SIZE, 'padded_model_vocab': 32128,
                            'eos_id': EOS_ID, 'pad_id': 0, 'add_special_tokens': False},
              'packing': {'sequence_length': LENGTH, 'append_eos_per_document': True,
                          'cross_document_attention': True, 'extra_attention_masks': False,
                          'drop_only_final_unused_document_suffix': True,
                          'document_order': 'official source shard and JSONL row order',
                          'role_assignment': 'SHA256(seed:text_sha256) first 8 bytes big-endian modulo 3',
                          'role_seed': args.seed, 'heldout_role_order': list(ROLES),
                          'exact_text_deduplication': True},
              'sources': sources, 'roles': role_records, 'exclusion': exclusion,
              'read_documents': read_documents, 'duplicates_or_excluded_skipped': duplicate_or_excluded,
              'elapsed_seconds': time.monotonic() - started,
              'preparer_sha256': file_sha256(source_snapshot),
              'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'tokenizers', 'huggingface_hub']},
              'qualification': 'Bounded fresh C4 fixture with T5-base tokenization, not the authors’ unavailable Dolma-preprocessed corpus. No corpus repetition is required for 500 B512 updates when train rows=256000.'}
    write_json(args.output_dir / 'manifest.json', report)
    print(json.dumps({'status': 'complete', 'manifest': str(args.output_dir / 'manifest.json'),
                      'roles': {role: record['tokens'] for role, record in role_records.items()},
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['heldout', 'train'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--download-dir', type=Path, required=True)
    parser.add_argument('--exclude-manifest', type=Path)
    parser.add_argument('--rows', type=int, default=256000)
    parser.add_argument('--diagnostic-rows', type=int, default=512)
    parser.add_argument('--eval-rows', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--max-shards', type=int, default=2)
    parser.add_argument('--tokenizer-batch', type=int, default=256)
    args = parser.parse_args()
    if args.tokenizer_batch < 1:
        parser.error('--tokenizer-batch must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
