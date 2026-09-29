"""Pinned isolated-document data for bounded fusion startup; no ingestion.

Pure lookup maps logical CE-target intervals to full-context document windows.
A window straddling two updates is presented twice with complementary target
masks. No CE target repeats, no corpus cycling occurs, and lookup never commits
the trainer's cursor. Embedded EOS tokens do not imply document boundaries.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import torch

from cdrm.pretrained.campaign_data import TokenizedDocument
from cdrm.pretrained.campaign_recipe import feedback_noise_for_rows
from cdrm.pretrained.document_shards import iter_documents, verify_document_shards
from cdrm.pretrained.nextlat import NextLatBatch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/mnt/localssd/cdrm-data/olmo-dolma-v1_5-readiness-20260928/tokenized')
DEFAULT_MANIFEST_SHA256 = 'f5135df838cb44241284fe807991d6a76d5a8be663256bc53d86d8ac9ab4ab76'
SCHEMA = 'olmo-fusion-startup-data-v1'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _positive(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _file_signatures(root):
    paths = [root/'config.json', root/'manifest.json']
    for shard in sorted(root.glob('shard-*')):
        if shard.is_symlink() or not shard.is_dir():
            raise ValueError('Expected regular shard directories')
        paths.extend(shard/name for name in ('manifest.json', 'tokens.bin', 'documents.jsonl'))
    signatures = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected immutable regular prepared data files')
        stat = path.stat()
        signatures[path.relative_to(root).as_posix()] = tuple(
            getattr(stat, name) for name in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    return signatures


def _tensor_pin(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
            'sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}


def _noise_pins(noise):
    return None if noise is None else [_tensor_pin(tensor) for tensor in noise]


def _recipe_noise_contract(recipe, width):
    _positive(width, 'width')
    if (not recipe.feedback or recipe.mode().num_passes != 4
            or getattr(recipe, 'document_policy', 'isolated-v1') != 'isolated-v1'):
        raise ValueError('Startup data requires isolated K4 feedback')
    return {'jitter_seed': recipe.jitter_seed, 'feedback_jitter': recipe.feedback_jitter,
            'fbt_passes': 4, 'logical_update': 0, 'width': width, 'dtype': 'torch.float32'}


@dataclass(frozen=True)
class StartupWindow:
    key: str
    document_key: str
    document_index: int
    start: int
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class SelectedWindow:
    window: StartupWindow
    target_start: int  # Inclusive target slot in window, never zero.
    target_stop: int   # Exclusive target slot in window.

    @property
    def key(self):
        return _digest([self.window.key, self.target_start, self.target_stop])


class StartupData:
    """Bounded in-memory view of the existing ~7M-token readiness corpus.

    Prepared-file hashes are verified at construction; all later reads use the
    owned token tuples. Data and schedule identity omit the relocatable path.
    This coverage fixture and deterministic document shuffle are not a production
    mixture or a language-model quality benchmark.
    """

    @classmethod
    def from_prepared(cls, root=DEFAULT_ROOT, *, expected_manifest_sha256=DEFAULT_MANIFEST_SHA256,
                      length=128, ce_per_update=8192, physical_batch_size=8, seed=20260929):
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError('Expected regular prepared corpus directory')
        if (not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_manifest_sha256)):
            raise ValueError('An explicit lowercase prepared-manifest SHA256 is required')
        before = _file_signatures(root)
        if _file_sha(root/'manifest.json') != expected_manifest_sha256:
            raise ValueError('Prepared corpus manifest differs from its pin')
        verified = verify_document_shards(root)
        if not verified['completed']:
            raise ValueError('Prepared corpus must be complete')
        config = json.loads((root/'config.json').read_bytes())
        # Existing verifier checks shard hashes, offsets, split assignment and
        # global duplicate identity; iter_documents also checks every shard hash.
        documents = tuple(iter_documents(root, verify=False))
        if before != _file_signatures(root) or _file_sha(root/'manifest.json') != expected_manifest_sha256:
            raise ValueError('Prepared corpus changed during data construction')
        provenance = {'manifest_sha256': expected_manifest_sha256,
                      'config_sha256': verified['config_sha256'],
                      'shards': verified['shards'], 'tokenizer': config['tokenizer'],
                      'split_policy': config['split_policy'], 'sources': config['sources']}
        # Source descriptors must not make dataset identity depend on relocation.
        if any('path' in source for source in provenance['sources']):
            raise ValueError('Prepared source identity unexpectedly contains a local path')
        return cls._from_documents(documents, provenance=provenance, length=length,
            ce_per_update=ce_per_update, physical_batch_size=physical_batch_size, seed=seed,
            eos_id=config['eos_id'], vocab_size=config['vocab_size'])

    @classmethod
    def _from_documents(cls, documents: Iterable[TokenizedDocument], *, provenance,
                        length=128, ce_per_update=8192, physical_batch_size=8,
                        seed=20260929, eos_id=50279, vocab_size=50280):
        """Internal construction seam for tiny independently supplied CPU fixtures."""
        _positive(length, 'length', minimum=3)
        if length > 1024:
            raise ValueError('Startup contexts are bounded at 1024')
        _positive(ce_per_update, 'ce_per_update')
        _positive(physical_batch_size, 'physical_batch_size')
        _positive(seed, 'seed', minimum=0)
        self = cls()
        self.length, self.ce_per_update, self.physical_batch_size = length, ce_per_update, physical_batch_size
        self.seed, self.eos_id, self.vocab_size, self.pad_id = seed, eos_id, vocab_size, 1
        if not 0 <= eos_id < vocab_size or eos_id == self.pad_id:
            raise ValueError('Invalid EOS/vocabulary/PAD contract')
        records, identities, content_splits = [], set(), {}
        for doc in documents:
            if not isinstance(doc, TokenizedDocument):
                raise TypeError('Expected verified TokenizedDocument records')
            identity = (doc.source.name, doc.document_id)
            if identity in identities:
                raise ValueError('Duplicate source/document identity')
            identities.add(identity)
            if doc.tokens[-1] != eos_id or any(t >= vocab_size for t in doc.tokens):
                raise ValueError('Prepared documents must retain actual terminal EOS and valid tokens')
            content = _digest(doc.tokens[:-1])
            if content in content_splits:
                raise ValueError('Duplicate token content within or across data splits')
            content_splits[content] = doc.split
            key = _digest({'source': asdict(doc.source), 'document_id': doc.document_id,
                           'text_sha256': doc.text_sha256, 'content_sha256': content, 'split': doc.split})
            order = _digest(['fusion-startup-order-v1', seed, key])
            records.append((order, key, doc))
        records.sort(key=lambda r: (r[0], r[1]))
        self._windows, heldout = [], []
        skipped, document_records = 0, []
        for index, (_, key, doc) in enumerate(records):
            document_records.append({'key': key, 'source': doc.source.name, 'split': doc.split,
                                     'tokens': len(doc.tokens), 'content_sha256': _digest(doc.tokens[:-1])})
            if doc.split == 'dev' and len(doc.tokens) >= 16:
                heldout.append((key, index, doc))
            if doc.split != 'train':
                continue
            if len(doc.tokens) < 2:
                skipped += 1
                continue
            for start in range(0, len(doc.tokens)-1, length-1):
                tokens = doc.tokens[start:start+length]
                self._windows.append(StartupWindow(_digest([key, start, length]), key, index, start, tokens))
        if len(heldout) < 4 or not self._windows:
            raise ValueError('Need training windows and at least four dev documents of >=16 tokens')
        self._windows = tuple(self._windows)
        self._target_prefix = [0]
        for row in self._windows:
            self._target_prefix.append(self._target_prefix[-1] + len(row.tokens)-1)
        self.total_updates = self._target_prefix[-1] // ce_per_update
        if self.total_updates < 1:
            raise ValueError('Corpus has fewer CE targets than one complete logical update')
        self._fresh_rows = tuple(StartupWindow(_digest(['startup-fresh-v1', key, size]), key, index, 0,
                                              doc.tokens[:size])
                                 for (key, index, doc), size in zip(heldout[:4], (16, 5, 6, 2)))
        self._fresh_documents = tuple({'key': key, 'document_index': index, 'source': asdict(doc.source),
            'document_id': doc.document_id, 'split': doc.split, 'text_sha256': doc.text_sha256,
            'full_tokens_sha256': _digest(doc.tokens), 'full_token_count': len(doc.tokens),
            'slice_start': 0, 'slice_length': size, 'slice_tokens_sha256': _digest(doc.tokens[:size])}
            for (key, index, doc), size in zip(heldout[:4], (16, 5, 6, 2)))
        self._manifest_json = _json({'schema': SCHEMA, 'provenance': provenance,
            'length': length, 'physical_batch_size': physical_batch_size, 'ce_per_update': ce_per_update,
            'seed': seed, 'eos_id': eos_id, 'pad_id': self.pad_id, 'vocab_size': vocab_size,
            'documents': document_records, 'training_windows': len(self._windows),
            'total_train_ce_targets': self._target_prefix[-1], 'total_updates': self.total_updates,
            'unused_final_ce_targets': self._target_prefix[-1] % ce_per_update,
            'skipped_eos_only_train_documents': skipped,
            'policy': {'packing': False, 'document_policy': 'isolated-v1', 'window_stride': length-1,
                       'update_boundary': 'repeat full context window with complementary target masks',
                       'eos': 'preserve prepared tokens; no synthetic window EOS or EOS-derived boundaries',
                       'order': 'deterministic SHA document shuffle; windows ordered inside each document',
                       'cycling': False, 'cursor': 'trainer commits completed logical update only'},
            'fresh_fixture': {'split': 'dev', 'length': 16, 'physical_batch_size': 2,
                              'documents': self._fresh_documents,
                              'rows': [asdict(r) for r in self._fresh_rows]}})
        self.manifest_sha256 = hashlib.sha256(self._manifest_json.encode()).hexdigest()
        return self

    @property
    def manifest(self):
        return json.loads(self._manifest_json)

    def cursor(self, next_update=0):
        _positive(next_update, 'next_update', minimum=0)
        if next_update > self.total_updates:
            raise ValueError('Cursor exceeds the noncycling schedule')
        return {'schema': SCHEMA, 'manifest_sha256': self.manifest_sha256, 'next_update': next_update}

    def restore_cursor(self, state):
        if not isinstance(state, dict) or set(state) != {'schema', 'manifest_sha256', 'next_update'}:
            raise ValueError('Malformed startup data cursor')
        if state != self.cursor(state['next_update']):
            raise ValueError('Cursor data/schedule identity differs')
        return state['next_update']

    def selections(self, update):
        _positive(update, 'update', minimum=0)
        if update >= self.total_updates:
            raise ValueError('Update exceeds the noncycling complete-target schedule')
        begin, end = update*self.ce_per_update, (update+1)*self.ce_per_update
        index = bisect_right(self._target_prefix, begin)-1
        rows = []
        while index < len(self._windows) and self._target_prefix[index] < end:
            start = 1 + max(begin, self._target_prefix[index])-self._target_prefix[index]
            stop = 1 + min(end, self._target_prefix[index+1])-self._target_prefix[index]
            rows.append(SelectedWindow(self._windows[index], start, stop))
            index += 1
        return tuple(rows)

    def update_metadata(self, update):
        rows = self.selections(update)
        inputs = sum(len(r.window.tokens) for r in rows)
        micros = (len(rows)+self.physical_batch_size-1)//self.physical_batch_size
        return {'counts': {'ce': self.ce_per_update, 'latent': self.ce_per_update,
                           'kl': sum(max(0, r.target_stop-max(2, r.target_start)) for r in rows)},
                'input_tokens': inputs, 'microbatches': micros, 'documents': len(rows),
                'unique_documents': len({r.window.document_key for r in rows}),
                'padding_tokens': micros*self.physical_batch_size*self.length-inputs,
                'window_keys': [r.window.key for r in rows], 'occurrence_keys': [r.key for r in rows],
                'selected_ce_targets': self.ce_per_update, 'start_cursor': self.cursor(update),
                'next_cursor': self.cursor(update+1)}

    def _batch(self, rows, *, length, batch_size, device):
        ids = torch.full((batch_size, length), self.pad_id, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        docs = torch.full_like(ids, -1)
        selected = torch.zeros_like(valid)
        for index, row in enumerate(rows):
            size = len(row.window.tokens)
            ids[index, :size] = torch.tensor(row.window.tokens)
            valid[index, :size] = True
            docs[index, :size] = row.window.document_index
            selected[index, row.target_start:row.target_stop] = True
        return NextLatBatch(*(value.to(device) for value in (ids, valid, docs, selected,
                                                            selected.clone(), selected.clone())))

    def update_batches(self, update, recipe, width, device='cpu'):
        _recipe_noise_contract(recipe, width)
        rows = self.selections(update)
        batches, noise = [], []
        for start in range(0, len(rows), self.physical_batch_size):
            selected = rows[start:start+self.physical_batch_size]
            batches.append(self._batch(selected, length=self.length,
                                      batch_size=self.physical_batch_size, device=device))
            noise.append(feedback_noise_for_rows(recipe, [r.key for r in selected], logical_update=update,
                sequence_length=self.length, width=width, physical_batch_size=self.physical_batch_size,
                device=device))
        return tuple(batches), tuple(noise)

    def fresh_fixture(self, recipe, width, device='cpu'):
        """Four disjoint dev documents, two physical B2/T16 records; no fake EOS."""
        _recipe_noise_contract(recipe, width)
        rows = tuple(SelectedWindow(r, 1, len(r.tokens)) for r in self._fresh_rows)
        records = []
        for start in (0, 2):
            pair = rows[start:start+2]
            batch = self._batch(pair, length=16, batch_size=2, device=device)
            noise = feedback_noise_for_rows(recipe, [r.key for r in pair], logical_update=0,
                sequence_length=16, width=width, physical_batch_size=2, device=device)
            records.append(((batch,), (noise,)))
        return records

    def export_fresh_fixture(self, path, *, recipe, width):
        """Small pinned CPU-list artifact, usable without reloading training data."""
        records = []
        fixtures = self.fresh_fixture(recipe, width)
        for index, ((batch,), (noise,)) in enumerate(fixtures):
            pair = self._fresh_rows[index*2:index*2+2]
            keys = [SelectedWindow(row, 1, len(row.tokens)).key for row in pair]
            records.append({'batch': {name: value.tolist() for name, value in vars(batch).items()},
                            'batch_pins': {name: _tensor_pin(value) for name, value in vars(batch).items()},
                            'noise_keys': keys, 'noise_pins': _noise_pins(noise)})
        train_keys = sorted({row.document_key for row in self._windows})
        if set(train_keys) & {row.document_key for row in self._fresh_rows}:
            raise AssertionError('Fresh numerical documents overlap training')
        metadata = {'schema': 'olmo-fusion-startup-fresh-v1',
            'training_manifest_sha256': self.manifest_sha256,
            'prepared_manifest_sha256': self.manifest['provenance']['manifest_sha256'],
            'training_document_keys_sha256': _digest(train_keys),
            'training_document_count': len(train_keys), 'document_policy': 'isolated-v1',
            'length': 16, 'physical_batch_size': 2, 'pad_id': self.pad_id,
            'vocab_size': self.vocab_size, 'counts': {'ce': 25, 'latent': 25, 'kl': 21},
            'input_tokens': 29, 'microbatches': 2, 'documents': self._fresh_documents,
            'document_disjointness': 'verified dev/train split document keys in pinned training manifest',
            'noise_contract': _recipe_noise_contract(recipe, width), 'records': records}
        raw = (_json(metadata)+'\n').encode()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != raw:
                raise ValueError('Existing fresh fixture differs; use a new artifact path')
        else:
            temporary = path.with_name(path.name+'.tmp')
            with temporary.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(path)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                'size_bytes': len(raw), 'metadata': json.loads(raw)}


def fresh_fixture(data, recipe, width, device='cpu'):
    return data.fresh_fixture(recipe, width, device=device)


def load_fresh_fixture(path, *, expected_sha256, recipe, width):
    """Load the immutable fresh CPU fixture and regenerate/hash its keyed jitter.

    The caller's SHA pin authenticates the export-time train/dev disjointness
    proof; this deliberately does not re-open the training corpus on each probe.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024*1024:
        raise ValueError('Expected a bounded regular fresh-fixture artifact')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Fresh fixture bytes differ from their pin')
    data = json.loads(raw)
    contract = _recipe_noise_contract(recipe, width)
    if (data.get('schema') != 'olmo-fusion-startup-fresh-v1'
            or data.get('noise_contract') != contract or data.get('document_policy') != 'isolated-v1'
            or data.get('length') != 16 or data.get('physical_batch_size') != 2
            or data.get('counts') != {'ce': 25, 'latent': 25, 'kl': 21}
            or data.get('input_tokens') != 29 or data.get('microbatches') != 2
            or len(data.get('documents', [])) != 4 or len(data.get('records', [])) != 2):
        raise ValueError('Fresh fixture contract differs')
    docs = data['documents']
    if (len({d['key'] for d in docs}) != 4 or len({d['document_index'] for d in docs}) != 4
            or any(d['split'] != 'dev' or d['full_token_count'] < 16 or d['slice_start'] != 0
                   or d['slice_length'] != size for d, size in zip(docs, (16,5,6,2)))):
        raise ValueError('Fresh fixture requires four disjoint sufficiently long dev documents')
    fields = {'input_ids', 'valid_mask', 'document_ids', 'ce_mask', 'latent_mask', 'kl_mask'}
    fixtures = []
    for record_index, record in enumerate(data['records']):
        if set(record['batch']) != fields or set(record['batch_pins']) != fields:
            raise ValueError('Fresh batch tensor inventory differs')
        tensors = {}
        for name in fields:
            dtype = torch.long if name in ('input_ids', 'document_ids') else torch.bool
            value = record['batch'][name]
            scalar_type = int if dtype == torch.long else bool
            if (not isinstance(value, list) or len(value) != 2
                    or any(not isinstance(row, list) or len(row) != 16
                           or any(type(v) is not scalar_type for v in row) for row in value)):
                raise ValueError('Fresh batch shape or scalar type differs')
            tensors[name] = torch.tensor(value, dtype=dtype)
            if _tensor_pin(tensors[name]) != record['batch_pins'][name]:
                raise ValueError('Fresh batch hash differs')
        batch = NextLatBatch(**tensors)
        pair = docs[record_index*2:record_index*2+2]
        keys = []
        for row, doc in enumerate(pair):
            size = doc['slice_length']
            values = tuple(record['batch']['input_ids'][row][:size])
            if (_digest(values) != doc['slice_tokens_sha256']
                    or any(not 0 <= value < data['vocab_size'] for value in values)
                    or record['batch']['input_ids'][row][size:] != [data['pad_id']]*(16-size)
                    or record['batch']['document_ids'][row] != [doc['document_index']]*size+[-1]*(16-size)
                    or record['batch']['valid_mask'][row] != [True]*size+[False]*(16-size)
                    or any(record['batch'][name][row] != [False]+[True]*(size-1)+[False]*(16-size)
                           for name in ('ce_mask','latent_mask','kl_mask'))):
                raise ValueError('Fresh actual-token slice, validity, document or target mask differs')
            window = StartupWindow(_digest(['startup-fresh-v1',doc['key'],size]),doc['key'],doc['document_index'],0,values)
            keys.append(SelectedWindow(window,1,size).key)
        if record['noise_keys'] != keys:
            raise ValueError('Fresh keyed noise identities differ')
        noise = feedback_noise_for_rows(recipe, keys, logical_update=0,
            sequence_length=16, width=width, physical_batch_size=2)
        if _noise_pins(noise) != record['noise_pins']:
            raise ValueError('Fresh noise pins differ')
        fixtures.append(((batch,), (noise,)))
    if path.read_bytes() != raw:
        raise ValueError('Fresh fixture changed while reading')
    return fixtures, data


def source_hashes():
    names = ('scripts/olmo_fusion_startup_data.py', 'tests/test_fusion_startup_data.py',
             'cdrm/pretrained/document_shards.py', 'cdrm/pretrained/campaign_data.py',
             'cdrm/pretrained/campaign_recipe.py', 'cdrm/pretrained/nextlat.py')
    return {name: _file_sha(ROOT/name) for name in names}
