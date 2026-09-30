#!/usr/bin/env python3
"""CPU materialization of fixed PR52 rows for saved-state feedback diagnostics.

This does not advance a training cursor, tokenize data, or construct a model.
The conditional training batch is retained in advance but is not authorization
to execute it. All row identities remain those of the accepted ordered reader.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from cdrm.pretrained.packed_campaign_data import PackedCounts
from scripts.olmo_pilot_ordered_data import OrderedCampaignData

SCHEMA = 'olmo-feedback-fixture-v1'
DECLARATION_SHA = '259fc4b84f0b5356bd3a038cee9142ca73fe65efed1cdb7a56ed63a325222109'
RESOLVED_SHA = '44d0e8ca4bf8b4d7dbd9a767980faca7c1951375f701e5f6dc0d4ff4ff238a8d'
POLICY = 'continuous-stream-v1'
SELECTIONS = {'dev': tuple(range(8)), 'train_primary': (16384, 16385),
              'train_conditional': (16386, 16387)}
FIELDS = ('input_ids', 'valid_mask', 'document_ids', 'ce_mask', 'latent_mask', 'kl_mask')


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _digest(value):
    # Match the accepted logical membership digest exactly (no trailing newline).
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _pinned(path, expected):
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'Authority bytes differ: {path}')
    return raw


def _tensor_record(value):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
        raise ValueError('Fixture tensor must be on CPU')
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}


def _selection_contract(declaration, resolved):
    if resolved['declaration'] != declaration:
        raise ValueError('Resolved embedded declaration differs')
    manifest = declaration['planning_manifest']
    planning = resolved['planning']
    steps = planning['plan']['updates']
    previous, upcoming = steps[31:33]
    if (previous['update'] != 32 or upcoming['update'] != 33
            or previous['next_cursor'] != upcoming['start_cursor']
            or upcoming['start_cursor']['next_chunk'] != 16384
            or upcoming['start_cursor']['next_update'] != 32
            or upcoming['next_cursor']['next_chunk'] != 16896
            or upcoming['counts']['valid_tokens'] != 524288
            or upcoming['counts']['packed_rows'] != 512):
        raise ValueError('Expected next unused logical update 33 at row 16384')
    dev = planning['evaluation']['panels']['dev-main']
    fixed = dev['fixed_plan']
    if (dev['selection'] != 'fixed_ordered_prefix_from_chunk_zero'
            or dev['target_valid_tokens'] != 65536
            or len(fixed['updates']) != 1
            or fixed['first_cursor']['next_chunk'] != 0
            or fixed['final_cursor']['next_chunk'] != 64
            or manifest['data']['policy']['document_policy'] != POLICY):
        raise ValueError('Expected unchanged fixed dev-main prefix and document policy')
    return manifest['data'], dev, upcoming


def _verify_logical(data, declared):
    original = asdict(data.cursor())
    data.restore_cursor(declared['start_cursor'])
    logical = data.peek_update(data.cursor(), declared['target_valid_tokens'])
    if logical is None:
        raise ValueError('Declared logical batch is exhausted')
    observed = {'start_cursor': asdict(logical.start_cursor),
                'next_cursor': asdict(logical.next_cursor),
                'counts': asdict(logical.counts),
                'membership_sha256': _digest([asdict(row) for row in logical.rows])}
    if any(value != declared[key] for key, value in observed.items()):
        raise ValueError('Declared logical membership, counts or cursor differs')
    if asdict(data.cursor()) != declared['start_cursor']:
        raise ValueError('Metadata read advanced the cursor')
    return {'reader_created_at': original, 'reader_left_at': asdict(data.cursor()),
            'cursor_committed': False, 'declared_logical_update': declared,
            'verified_row_descriptors': [asdict(row) for row in logical.rows]}


def _materialize(data, indices):
    before = asdict(data.cursor())
    rows = tuple(data.descriptor(index) for index in indices)
    if not rows or any(row.length != data.length for row in rows):
        raise ValueError('Only complete nonempty rows are permitted')
    batch = data.batch(rows, physical_batch_size=len(rows))
    tensors = {key: getattr(batch, key) for key in FIELDS}
    masks = build_nextlat_masks(batch, document_policy=POLICY)
    counts = PackedCounts()
    membership = []
    for row in rows:
        counts += data.chunk_counts(row)
        chunk = data.read_chunk(row.index)
        membership.append({'descriptor': asdict(row), 'source_chunk': data.source_chunk(row.index),
                           'segments': [asdict(segment) for segment in chunk.segments]})
    actual = {key: int(mask.sum()) for key, mask in masks.items()}
    if actual != counts.objective_counts or int(batch.valid_mask.sum()) != counts.valid_tokens:
        raise ValueError('Materialized objective counts differ from ordered metadata')
    if asdict(data.cursor()) != before:
        raise ValueError('Fixture materialization advanced the cursor')
    payload = {'batch': tensors, 'effective_loss_masks': masks}
    metadata = {'indices': list(indices), 'panel_manifest_sha256': data.manifest_sha256,
                'panel_identity_sha256': data.manifest['identity_sha256'],
                'counts': asdict(counts), 'objective_counts': actual,
                'membership_sha256': _digest([asdict(row) for row in rows]),
                'rows': membership, 'cursor_before': before, 'cursor_after': before,
                'tensors': {key: _tensor_record(value) for key, value in tensors.items()},
                'effective_loss_masks': {key: _tensor_record(value) for key, value in masks.items()}}
    return payload, metadata


def build_fixture(declaration_path, resolved_path, output_dir, *,
                  declaration_sha256=DECLARATION_SHA, resolved_sha256=RESOLVED_SHA):
    """Create one immutable fixture directory; caller must use the CPU container."""
    if not Path('/.dockerenv').exists() or Path.cwd() != Path('/workspace/cdrm-w-latent'):
        raise RuntimeError('Materialize inside the required project CPU container')
    if torch.cuda.is_available():
        raise RuntimeError('Materialization requires explicit CDRM_DOCKER_GPUS=none')
    declaration_raw = _pinned(declaration_path, declaration_sha256)
    resolved_raw = _pinned(resolved_path, resolved_sha256)
    declaration, resolved = json.loads(declaration_raw), json.loads(resolved_raw)
    data_spec, dev, upcoming = _selection_contract(declaration, resolved)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix='.pending-feedback-fixture-', dir=output.parent))
    try:
        authorities = {'declaration': (Path(declaration_path), declaration_sha256),
                       'resolved': (Path(resolved_path), resolved_sha256)}
        for name in ('corpus', 'suite', 'index'):
            authorities[name] = (Path(data_spec[name])/'manifest.json',
                                 data_spec[name+'_manifest_sha256'])
        for name in ('source_plan', 'source_authorities', 'inventory', 'exclusions'):
            authorities[name] = (Path(data_spec[name]), data_spec[name+'_sha256'])
        authorities['dev_index'] = (Path(dev['index'])/'manifest.json', dev['index_manifest_sha256'])
        retained = {}
        (pending/'authorities').mkdir()
        for name, (path, sha) in authorities.items():
            raw = _pinned(path, sha)
            target = 'authorities/'+name+path.suffix
            (pending/target).write_bytes(raw)
            retained[name] = {'source_path': str(path), 'snapshot': target,
                              'sha256': sha, 'size_bytes': len(raw)}
        selections, payload, logical = {}, {}, {}
        for panel, index, expected, declared, groups in (
            ('dev-main', dev['index'], dev['index_manifest_sha256'],
             dev['fixed_plan']['updates'][0], ('dev',)),
            ('train', data_spec['index'], data_spec['index_manifest_sha256'], upcoming,
             ('train_primary', 'train_conditional'))):
            with OrderedCampaignData(data_spec['corpus'], index) as reader:
                if reader.length != 1024 or reader.manifest_sha256 != expected:
                    raise ValueError('Expected pinned T1024 ordered reader')
                logical[panel] = _verify_logical(reader, declared)
                for name in groups:
                    payload[name], selections[name] = _materialize(reader, SELECTIONS[name])
                    selections[name]['panel'] = panel
        torch.save(payload, pending/'fixture.pt')
        report = {'schema': SCHEMA, 'status': 'materialized_cpu_only',
                  'document_policy': POLICY, 'selections': selections,
                  'logical_authorities': logical, 'authorities': retained,
                  'payload': {'path': 'fixture.pt', 'sha256': _sha(pending/'fixture.pt'),
                              'size_bytes': (pending/'fixture.pt').stat().st_size},
                  'producer': {'path': 'producer.py', 'sha256': _sha(Path(__file__))},
                  'scope': ['No optimizer updates and no committed training cursor.',
                            'dev is a subset of the existing PR52 dev prefix, not a new evaluation split.',
                            'Training ordinals are unused by the first32 continuation; no additional document or pretraining overlap claim.',
                            'train_conditional is materialized but execution remains conditional.']}
        shutil.copyfile(__file__, pending/'producer.py')
        (pending/'report.json').write_bytes(_json(report))
        digest = _sha(pending/'report.json')
        (pending/'report.sha256').write_text(digest+'\n')
        for name in SELECTIONS:
            load_fixture(pending, expected_report_sha256=digest, selection=name)
        pending.rename(output)
        return report
    except BaseException:
        shutil.rmtree(pending)
        raise


def load_fixture(output_dir, *, expected_report_sha256, selection='dev'):
    """Return (CPU NextLatBatch, provenance) after hash and loss-mask checks."""
    directory = Path(output_dir)
    report = json.loads(_pinned(directory/'report.json', expected_report_sha256))
    if report['schema'] != SCHEMA or report['document_policy'] != POLICY or selection not in SELECTIONS:
        raise ValueError('Unknown fixture schema, policy or selection')
    if set(report['selections']) != set(SELECTIONS):
        raise ValueError('Fixture selection set differs')
    for record in report['authorities'].values():
        path = Path(record['snapshot'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe authority snapshot path')
        _pinned(directory/path, record['sha256'])
    payload_path = report['payload']['path']
    if payload_path != 'fixture.pt' or _sha(directory/payload_path) != report['payload']['sha256']:
        raise ValueError('Fixture payload bytes differ')
    payload = torch.load(directory/payload_path, map_location='cpu', weights_only=True)
    if set(payload) != set(SELECTIONS):
        raise ValueError('Payload selection set differs')
    item, meta = payload[selection], report['selections'][selection]
    if meta['indices'] != list(SELECTIONS[selection]) or set(item['batch']) != set(FIELDS):
        raise ValueError('Fixture row selection or batch fields differ')
    batch = NextLatBatch(**item['batch'])
    if batch.input_ids.shape != (len(SELECTIONS[selection]), 1024) or not bool(batch.valid_mask.all()):
        raise ValueError('Fixture must contain complete T1024 rows without padding')
    for name, value in item['batch'].items():
        if _tensor_record(value) != meta['tensors'][name]:
            raise ValueError('Retained batch tensor differs')
    effective = build_nextlat_masks(batch, document_policy=POLICY)
    if set(item['effective_loss_masks']) != set(effective):
        raise ValueError('Retained loss mask terms differ')
    for name, value in effective.items():
        if (not torch.equal(value, item['effective_loss_masks'][name])
                or _tensor_record(value) != meta['effective_loss_masks'][name]
                or int(value.sum()) != meta['objective_counts'][name]):
            raise ValueError('Retained loss masks or counts differ')
    if int(batch.valid_mask.sum()) != meta['counts']['valid_tokens']:
        raise ValueError('Retained valid input count differs')
    return batch, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', required=True)
    parser.add_argument('--resolved', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    report = build_fixture(args.declaration, args.resolved, args.output_dir)
    print(json.dumps({'status': report['status'], 'selections': {
        key: value['counts'] for key, value in report['selections'].items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
