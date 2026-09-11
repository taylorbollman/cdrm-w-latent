"""CPU checks for disjoint C4 roles, exact packing and exclusion integrity."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rt_precision_data import TokenPacker, excluded_documents, file_sha256, heldout_role, text_sha256


def test_packing_crosses_document_boundary_once_and_records_truncated_tail(tmp_path):
    packer = TokenPacker(tmp_path, 'train', rows=2, length=4)
    packer.append([4, 5, 1], text_hash='a', source_file='c4', source_row=0)
    packer.append([6, 7, 8, 9, 10, 1], text_hash='b', source_file='c4', source_row=1)
    record = packer.finish()
    np.testing.assert_array_equal(np.load(tmp_path / 'train.npy'), [[4, 5, 1, 6], [7, 8, 9, 10]])
    assert record['tokens'] == 8 and record['dtype'] == 'uint16'
    docs = [json.loads(line) for line in (tmp_path / 'train-documents.jsonl').read_text().splitlines()]
    assert [(d['token_begin'], d['token_end']) for d in docs] == [(0, 3), (3, 8)]
    assert docs[0]['eos_retained'] and not docs[0]['truncated_at_corpus_end']
    assert not docs[1]['eos_retained'] and docs[1]['truncated_at_corpus_end']
    with pytest.raises(ValueError, match='already complete'):
        packer.append([1], text_hash='c', source_file='c4', source_row=2)


@pytest.mark.parametrize('ids', [[-1, 1], [32100, 1], [2, 3], [1.5, 1], [], [[1, 2]]])
def test_invalid_token_values_rejected_before_cast(tmp_path, ids):
    packer = TokenPacker(tmp_path, 'dev', 1, 4)
    try:
        with pytest.raises(ValueError):
            packer.append(ids, text_hash='a', source_file='c4', source_row=0)
        assert packer.position == 0
    finally:
        packer.boundaries.close()


def test_incomplete_corpus_has_no_success_record(tmp_path):
    packer = TokenPacker(tmp_path, 'dev', 1, 4)
    packer.append([2, 1], text_hash='a', source_file='c4', source_row=0)
    with pytest.raises(RuntimeError, match='only 2/4'):
        packer.finish()


def test_document_role_is_stable_and_exclusive():
    texts = ['same document', 'another document', 'more text'] * 100
    allocation = {}
    for text in texts:
        digest = text_sha256(text)
        role = heldout_role(digest, 20260910)
        assert role in {'diagnostic', 'dev', 'confirmation'}
        assert allocation.setdefault(digest, role) == role
    assert len({heldout_role(text_sha256(str(i)), 20260910) for i in range(100)}) == 3


def test_exclusion_manifest_verifies_boundaries_and_collects_document_hashes(tmp_path):
    boundaries = tmp_path / 'dev-documents.jsonl'
    boundaries.write_text(json.dumps({'text_sha256': 'abc'}) + '\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'rt-precision-c4-data-v1', 'mode': 'heldout',
                                   'roles': {'dev': {'boundaries_path': boundaries.name,
                                                     'boundaries_sha256': file_sha256(boundaries)}}}))
    hashes, reference = excluded_documents(manifest)
    assert hashes == {'abc'} and reference['manifest_sha256'] == file_sha256(manifest)
    boundaries.write_text(json.dumps({'text_sha256': 'changed'}) + '\n')
    with pytest.raises(ValueError, match='changed'):
        excluded_documents(manifest)


def test_refuse_overwrite(tmp_path):
    packer = TokenPacker(tmp_path, 'dev', 1, 4)
    packer.boundaries.close()
    with pytest.raises(FileExistsError):
        TokenPacker(tmp_path, 'dev', 1, 4)
