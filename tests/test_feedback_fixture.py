"""Bounded CPU row-identity, mask, cursor and payload integrity checks."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest
import torch

from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts import olmo_feedback_fixture as probe
from scripts.olmo_pilot_ordered_data import OrderedCampaignData
from test_pilot_ordered_data import fixture as ordered_fixture, cpu_tokenizer


def _contract():
    cursor = {'next_chunk': 16384, 'next_update': 32, 'split': 'train', 'manifest_sha256': 'a'*64}
    declaration = {'planning_manifest': {'data': {'policy': {'document_policy': probe.POLICY}}}}
    previous = {'update': 32, 'next_cursor': cursor}
    upcoming = {'update': 33, 'start_cursor': cursor,
                'next_cursor': {**cursor, 'next_chunk': 16896, 'next_update': 33},
                'counts': {'valid_tokens': 524288, 'packed_rows': 512}}
    dev = {'selection': 'fixed_ordered_prefix_from_chunk_zero', 'target_valid_tokens': 65536,
           'fixed_plan': {'updates': [{}], 'first_cursor': {'next_chunk': 0},
                          'final_cursor': {'next_chunk': 64}}}
    resolved = {'declaration': deepcopy(declaration), 'planning': {
        'plan': {'updates': [{}]*31+[previous, upcoming]}, 'evaluation': {'panels': {'dev-main': dev}}}}
    return declaration, resolved


def test_next_unused_row_identity_and_conditional_second_batch_are_fixed():
    declaration, resolved = _contract()
    _, _, upcoming = probe._selection_contract(declaration, resolved)
    assert probe.SELECTIONS == {'dev': tuple(range(8)), 'train_primary': (16384, 16385),
                                'train_conditional': (16386, 16387)}
    assert upcoming['start_cursor']['next_update'] == 32
    assert not set(probe.SELECTIONS['train_primary']) & set(probe.SELECTIONS['train_conditional'])


@pytest.mark.parametrize('mutation', ['next_chunk', 'next_update', 'dev_origin', 'embedded', 'policy'])
def test_rejects_shifted_cursor_prefix_policy_or_authority(mutation):
    declaration, resolved = _contract()
    if mutation in ('next_chunk', 'next_update'):
        resolved['planning']['plan']['updates'][32]['start_cursor'][mutation] += 1
    elif mutation == 'dev_origin':
        resolved['planning']['evaluation']['panels']['dev-main']['fixed_plan']['first_cursor']['next_chunk'] = 1
    elif mutation == 'embedded':
        resolved['declaration']['extra'] = True
    else:
        declaration['planning_manifest']['data']['policy']['document_policy'] = 'isolated-documents-v1'
        resolved['declaration'] = deepcopy(declaration)
    with pytest.raises(ValueError):
        probe._selection_contract(declaration, resolved)


def test_actual_reader_tokens_boundaries_masks_and_cursor(tmp_path):
    fixture = ordered_fixture(tmp_path, length=1024)
    with OrderedCampaignData(fixture.corpus, fixture.output/'panels/dev-main') as reader:
        cursor = reader.cursor()
        payload, meta = probe._materialize(reader, tuple(range(8)))
        batch = probe.NextLatBatch(**payload['batch'])
        assert reader.cursor() == cursor
        assert batch.input_ids.shape == (8, 1024)
        assert meta['counts']['valid_tokens'] == 8192
        assert meta['objective_counts']['ce'] == 8184
        for index in range(8):
            literal = reader.read_chunk(index)
            assert tuple(batch.input_ids[index].tolist()) == literal.tokens
            assert tuple(batch.document_ids[index].tolist()) == literal.document_ids
            ids = batch.document_ids[index].tolist()
            assert meta['rows'][index]['segments'] == [asdict(s) for s in literal.segments]
            expected_latent = [ids[t] == ids[t+1] for t in range(1023)]
            expected_kl = [ids[t] == ids[t+1] == ids[t+2] for t in range(1022)]
            assert payload['effective_loss_masks']['latent'][index].tolist() == expected_latent
            assert payload['effective_loss_masks']['kl'][index].tolist() == expected_kl
        assert meta['objective_counts']['latent'] < 8184
        assert meta['cursor_before'] == meta['cursor_after'] == asdict(cursor)


def test_logical_membership_check_does_not_commit_or_read_tokens(tmp_path, monkeypatch):
    fixture = ordered_fixture(tmp_path, length=4)
    with OrderedCampaignData(fixture.corpus, fixture.output/'panels/train') as reader:
        logical = reader.peek_update(reader.cursor(), 16)
        declared = {'start_cursor': asdict(logical.start_cursor), 'next_cursor': asdict(logical.next_cursor),
                    'counts': asdict(logical.counts), 'target_valid_tokens': 16,
                    'membership_sha256': probe._digest([asdict(row) for row in logical.rows])}
        monkeypatch.setattr(reader, '_token_slice', lambda *_: pytest.fail('metadata read materialized tokens'))
        observed = probe._verify_logical(reader, declared)
        assert observed['cursor_committed'] is False
        assert reader.cursor() == logical.start_cursor
        declared['membership_sha256'] = 'b'*64
        with pytest.raises(ValueError, match='membership'):
            probe._verify_logical(reader, declared)


def _stored_fixture(tmp_path, monkeypatch):
    fixture = ordered_fixture(tmp_path, length=1024)
    # Tiny corpus row coordinates only; production selection is not configurable.
    monkeypatch.setattr(probe, 'SELECTIONS', {'dev': tuple(range(8)),
        'train_primary': (8, 9), 'train_conditional': (10, 11)})
    payload, selections = {}, {}
    with OrderedCampaignData(fixture.corpus, fixture.output/'panels/train') as reader:
        for name, indices in probe.SELECTIONS.items():
            payload[name], selections[name] = probe._materialize(reader, indices)
    directory = tmp_path/'fixture'
    directory.mkdir()
    torch.save(payload, directory/'fixture.pt')
    report = {'schema': probe.SCHEMA, 'document_policy': probe.POLICY,
              'authorities': {}, 'selections': selections,
              'payload': {'path': 'fixture.pt', 'sha256': probe._sha(directory/'fixture.pt')}}
    (directory/'report.json').write_bytes(probe._json(report))
    return directory, report, payload


def test_roundtrip_both_training_batches_retain_exact_masks_and_disjoint_rows(tmp_path, monkeypatch):
    directory, report, _ = _stored_fixture(tmp_path, monkeypatch)
    batches = {}
    for name in probe.SELECTIONS:
        batch, provenance = probe.load_fixture(directory,
            expected_report_sha256=probe._sha(directory/'report.json'), selection=name)
        batches[name] = batch
        assert provenance['indices'] == list(probe.SELECTIONS[name])
        assert {key: int(value.sum()) for key, value in
                build_nextlat_masks(batch, document_policy=probe.POLICY).items()} == provenance['objective_counts']
    assert not torch.equal(batches['train_primary'].input_ids, batches['train_conditional'].input_ids)


@pytest.mark.parametrize('mutation', ['report_hash', 'payload_hash', 'effective_mask', 'objective_count'])
def test_load_rejects_corruption_and_semantically_wrong_masks(tmp_path, monkeypatch, mutation):
    directory, report, payload = _stored_fixture(tmp_path, monkeypatch)
    old_sha = probe._sha(directory/'report.json')
    if mutation == 'report_hash':
        (directory/'report.json').write_bytes((directory/'report.json').read_bytes()+b' ')
    elif mutation == 'payload_hash':
        with (directory/'fixture.pt').open('ab') as out:
            out.write(b'corrupt')
    elif mutation == 'effective_mask':
        payload['dev']['effective_loss_masks']['latent'][0, 0].logical_not_()
        torch.save(payload, directory/'fixture.pt')
        report['payload']['sha256'] = probe._sha(directory/'fixture.pt')
        (directory/'report.json').write_bytes(probe._json(report))
    else:
        report['selections']['dev']['objective_counts']['ce'] -= 1
        (directory/'report.json').write_bytes(probe._json(report))
    report_sha = old_sha if mutation == 'report_hash' else probe._sha(directory/'report.json')
    with pytest.raises(ValueError):
        probe.load_fixture(directory, expected_report_sha256=report_sha)
