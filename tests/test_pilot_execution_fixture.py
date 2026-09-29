"""Native-tokenizer tiny ordered execution fixture checks; CPU only."""
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import pytest
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts import olmo_pilot_execution_fixture as fixture
from scripts import olmo_pilot_ordered_data as ordered


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    if not fixture.DEFAULT_TOKENIZER.is_file():
        pytest.skip('Pinned native tokenizer asset not installed; no synthetic fallback')
    root = tmp_path_factory.mktemp('pilot-execution')/'fixture'
    return root, fixture.build_fixture(root)


def test_three_uneven_updates_have_literal_masks_and_complete_token_budget(built):
    root, report = built
    with ordered.OrderedCampaignData(root/'corpus', root/'ordered/panels/train') as data:
        cursor = data.cursor(); total = 0; crossed = 0
        for _ in range(3):
            update = data.peek_update(cursor, 80)
            assert len(update.rows) == 5
            ranks = [data.rank_batches(update, rank=r, world_size=2, physical_batch_size=2) for r in range(2)]
            assert [len(x.batches) for x in ranks] == [2, 2]
            assert sum(x.empty_rows for x in ranks) == 3
            assert sum(x.counts.valid_tokens for x in ranks) == 80
            sums = dict.fromkeys(('ce','latent','kl'), 0)
            for rank in ranks:
                for batch in rank.batches:
                    masks = build_nextlat_masks(batch, document_policy='continuous-stream-v1')
                    for key in sums: sums[key] += int(masks[key].sum())
            assert sums == update.counts.objective_counts
            crossed += update.counts.cross_document_ce_targets
            total += update.counts.valid_tokens
            cursor = update.next_cursor
        assert total == 240 and crossed > 0
        assert crossed == report['first_three_updates_cross_document_ce_targets']


def test_train_dev_confirmation_identity_disjoint_and_true_internal_eos(built):
    root, report = built
    manifest = json.loads((root/'ordered/manifest.json').read_text())
    assert manifest['split_intersections'] == dict(content_hashes=0, document_indices=0, readiness_exclusions=0)
    assert report['scope'].startswith('Synthetic CPU fixture')
    from cdrm.pretrained.document_shards import iter_documents
    documents = list(iter_documents(root/'corpus'))
    assert all(doc.tokens[-1] == 50279 and 50279 in doc.tokens[:-1] for doc in documents)
    assert len(documents) == 9*(64+32+32)


def test_fixture_pins_are_literal_and_build_rejects_reuse(built):
    root, report = built
    for path, pin in report['pins'].items(): assert fixture.sha(root/path) == pin
    assert report['data_spec']['split'] == 'train'
    assert report['data_spec']['policy'] == ordered.POLICY
    with pytest.raises(ValueError, match='fresh'): fixture.build_fixture(root)


def test_bad_native_tokenizer_fails_before_output_creation(tmp_path):
    tokenizer = tmp_path/'bad.json'; tokenizer.write_text('{}')
    output = tmp_path/'unused'
    with pytest.raises(ValueError, match='pinned OLMo tokenizer'):
        fixture.build_fixture(output, tokenizer_path=tokenizer)
    assert not output.exists()


def test_rebuild_has_identical_content_index_and_membership_pins(built, tmp_path):
    root, first = built
    second = fixture.build_fixture(tmp_path/'second')
    assert first['pins'] == second['pins']
    assert first['suite_identity_sha256'] == second['suite_identity_sha256']
    assert first['updates'] == second['updates']
    assert first['data_spec']['corpus'] != second['data_spec']['corpus']
