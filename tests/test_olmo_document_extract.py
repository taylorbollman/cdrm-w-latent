import gzip
import io
import json

import pytest

from scripts.olmo_document_extract import LimitedReader, extract_source


class Tokenizer:
    eos_token_id = 9

    def encode(self, text):
        return [1] * len(text)


class Response(io.BytesIO):
    headers = {'ETag': 'pinned'}
    status = 200

    def geturl(self):
        return 'https://olmo-data.org/fixture'


def prepare(tmp_path, monkeypatch, *, text='longer than budget', **kwargs):
    payload = b'\n'.join(json.dumps({'id': str(i), 'text': text}).encode() for i in range(3)) + b'\n'
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: Response(gzip.compress(payload)))
    options = dict(tokenizer=Tokenizer(), token_target=3, max_documents=100,
                   max_compressed_bytes=4096, max_line_bytes=4096,
                   upstream_manifest={'revision': 'pinned'}, tokenizer_sha='a' * 64)
    options.update(kwargs)
    return extract_source({'name': 'test', 'url': 'https://olmo-data.org/fixture', 'etag': 'pinned'},
                          tmp_path, **options)


def test_complete_document_and_retry(tmp_path, monkeypatch):
    r = prepare(tmp_path, monkeypatch)
    assert r['documents'] == 1
    assert r['tokens_including_terminal_eos'] > 3
    assert json.loads((tmp_path/'test/raw.jsonl').read_text())['text'] == 'longer than budget'
    assert r == prepare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='configuration'):
        prepare(tmp_path, monkeypatch, token_target=4)
    (tmp_path/'test/raw.jsonl').write_text('corrupt')
    with pytest.raises(ValueError, match='raw bytes'):
        prepare(tmp_path, monkeypatch)


def test_failure_does_not_publish_and_retry_restarts(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='row exceeds'):
        prepare(tmp_path, monkeypatch, max_line_bytes=5)
    assert not (tmp_path/'test').exists()
    assert (tmp_path/'test.partial').is_dir()
    assert prepare(tmp_path, monkeypatch)['status'] == 'complete'


def test_bounded_compressed_bytes():
    r = LimitedReader(io.BytesIO(b'1234567'), 3)
    assert r.read(2) == b'12'
    with pytest.raises(ValueError, match='byte limit'):
        r.read(2)


def test_reject_changed_etag(tmp_path, monkeypatch):
    monkeypatch.setattr(Response, 'headers', {'ETag': 'changed'})
    with pytest.raises(ValueError, match='ETag changed'):
        prepare(tmp_path, monkeypatch)
    assert not (tmp_path/'test').exists()


def test_empty_source_rows_are_counted_and_audited(tmp_path, monkeypatch):
    payload = b'{"id":"empty","text":" "}\n{"id":"valid","text":"abc"}\n'
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: Response(gzip.compress(payload)))
    r = extract_source({'name':'test', 'url':'https://olmo-data.org/fixture'}, tmp_path,
        tokenizer=Tokenizer(), token_target=3, max_documents=10, max_compressed_bytes=4096,
        max_line_bytes=4096, upstream_manifest={}, tokenizer_sha='a'*64)
    assert r['documents'] == 1 and r['skipped_empty_text'] == 1
    rows = [json.loads(line) for line in (tmp_path/'test/source-lines.jsonl').read_text().splitlines()]
    assert rows[0]['status'] == 'skipped_empty_text'
    assert rows[1]['source_line'] == 2 and rows[1]['output_line'] == 1


def test_retry_requires_complete_inventory(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    path = tmp_path/'test/manifest.json'
    record = json.loads(path.read_text())
    del record['files']['source-lines.jsonl']
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='incomplete'):
        prepare(tmp_path, monkeypatch)
