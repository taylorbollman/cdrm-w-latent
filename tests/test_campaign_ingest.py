"""Actual raw-byte pinning and deterministic bounded local ingestion fixtures."""
from dataclasses import replace
import gzip
import hashlib
import json

import pytest

from cdrm.pretrained.campaign_data import SourcePin, TokenizerPin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy, prepare_local_preflight


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


TOKENIZER = TokenizerPin("fixture", "fixture-v1", sha(b"declared fixture callable"))
SPLITS = SplitPolicy(624, (("train", 8), ("dev", 1), ("test", 1)))


def source(tmp_path, name, rows=None, raw=None, compression="none", **fields):
    if raw is None:
        raw = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    if compression == "gzip":
        raw = gzip.compress(raw, mtime=0)
    path = tmp_path / f"{name}.jsonl"
    path.write_bytes(raw)
    return LocalJSONLSource(SourcePin(name, f"local:{name}", "fixture-v1", sha(raw)), path,
                           compression=compression, **fields)


def characters(text, *, add_special_tokens):
    assert add_special_tokens is False
    return [ord(c) + 2 for c in text]


def prepare(sources, **kw):
    return prepare_local_preflight(sources, tokenizer=TOKENIZER, split_policy=SPLITS,
                                  tokenize=characters, **kw)


def test_pinned_jsonl_and_gzip_ingest_verbatim_text_with_reproducible_splits_and_order(tmp_path):
    first = source(tmp_path, "a", [{"id": "one", "text": "  café\n"}, {"id": "two", "text": "second"}])
    second = source(tmp_path, "b", [{"name": "three", "content": "third"}], compression="gzip",
                    text_field="content", id_field="name")
    left, right = prepare([first, second]), prepare([first, second])
    assert left.manifest_sha256 == right.manifest_sha256
    assert left.data.manifest_sha256 == right.data.manifest_sha256
    docs = left.data.manifest["documents"]
    assert [d["document_id"] for d in docs] == ["one", "two", "three"]
    texts = ["  café\n", "second", "third"]
    assert [d["text_sha256"] for d in docs] == [sha(t.encode()) for t in texts]
    assert [d["split"] for d in docs] == [SPLITS.split_for_text_hash(sha(t.encode())) for t in texts]
    assert left.manifest["source_bytes_verified"] is True
    assert left.manifest["tokenizer_verification"] == "declared_callable_unverified"
    assert left.manifest["counts"]["normalized_tokens"] == sum(len(t) + 1 for t in texts)
    for rows in left.data._windows.values():
        for row in rows:
            assert row.tokens[-1] == 50279
            assert row.tokens.count(50279) == 1
    left.manifest["bounds"]["max_documents"] = 1
    assert left.manifest["bounds"]["max_documents"] != 1


def test_every_source_hash_is_verified_before_any_tokenization(tmp_path):
    good = source(tmp_path, "first", [{"id": "a", "text": "good"}])
    bad = source(tmp_path, "second", [{"id": "b", "text": "bad"}])
    bad.path.write_bytes(bad.path.read_bytes() + b" ")
    calls = []
    def encode(text, *, add_special_tokens):
        calls.append(text)
        return [5]
    with pytest.raises(ValueError, match="pinned SHA256"):
        prepare_local_preflight([good, bad], tokenizer=TOKENIZER, split_policy=SPLITS, tokenize=encode)
    assert calls == []


def test_input_snapshot_survives_file_change_during_callable_tokenization(tmp_path):
    item = source(tmp_path, "input", [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}])
    def encode(text, *, add_special_tokens):
        item.path.write_text("mutated after byte verification")
        return characters(text, add_special_tokens=add_special_tokens)
    result = prepare_local_preflight([item], tokenizer=TOKENIZER, split_policy=SPLITS, tokenize=encode)
    assert [d["text_sha256"] for d in result.data.manifest["documents"]] == [sha(b"first"), sha(b"second")]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_unicode_line_separators_inside_json_text_do_not_split_records(tmp_path, newline):
    text = "alpha\u2028beta\u2029gamma"
    raw = (json.dumps({"id": "a", "text": text}, ensure_ascii=False) + newline).encode()
    item = source(tmp_path, "unicode", raw=raw)
    result = prepare([item])
    record = result.data.manifest["documents"][0]
    assert result.data.windows(record["split"])[0].tokens == tuple(ord(c) + 2 for c in text) + (50279,)
    assert record["text_sha256"] == sha(text.encode())


def test_native_tokenizer_file_verified_and_postprocessor_special_tokens_disabled(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    native = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "<|endoftext|>": 4}, unk_token="[UNK]"))
    native.pre_tokenizer = pre_tokenizers.Whitespace()
    native.post_processor = processors.TemplateProcessing(single="$A <|endoftext|>", special_tokens=[("<|endoftext|>", 4)])
    path = tmp_path / "tokenizer.json"
    native.save(str(path))
    pin = TokenizerPin("fixture/native", "fixture-v1", sha(path.read_bytes()))
    item = source(tmp_path, "input", [{"id": "a", "text": "alpha beta"}])
    result = prepare_local_preflight([item], tokenizer=pin, split_policy=SplitPolicy(0, (("train", 1),)),
                                    tokenizer_path=path, eos_id=4, vocab_size=5)
    assert result.data.windows("train")[0].tokens == (2, 3, 4)
    assert result.data.manifest["documents"][0]["eos_appended"] is True
    assert result.manifest["tokenizer_verification"] == "verified_tokenizer_json_bytes"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Tokenizer bytes differ"):
        prepare_local_preflight([item], tokenizer=pin, split_policy=SPLITS, tokenizer_path=path, eos_id=4, vocab_size=5)


def test_one_existing_terminal_eos_is_not_duplicated(tmp_path):
    item = source(tmp_path, "input", [{"id": "a", "text": "alpha"}])
    def encode(text, *, add_special_tokens):
        assert not add_special_tokens
        return [5, 50279]
    result = prepare_local_preflight([item], tokenizer=TOKENIZER, split_policy=SplitPolicy(0, (("train", 1),)), tokenize=encode)
    assert result.data.windows("train")[0].tokens == (5, 50279)
    assert result.manifest["counts"]["normalized_tokens"] == 2


@pytest.mark.parametrize("row", [[], {}, {"text": "hello"}, {"id": "a", "text": 3},
                                 {"id": "a", "text": " "}, {"id": 3, "text": "hello"}])
def test_missing_or_malformed_fields_fail_explicitly(tmp_path, row):
    item = source(tmp_path, "bad", [row])
    with pytest.raises(ValueError, match="JSONL object|nonempty string"):
        prepare([item])


@pytest.mark.parametrize("raw,match", [(b"\n", "Invalid JSONL"), (b"oops\n", "Invalid JSONL"),
                                      (b"\xff\n", "UTF-8"), (b"", "no JSONL")])
def test_invalid_json_utf8_and_blank_rows_are_not_silently_skipped(tmp_path, raw, match):
    item = source(tmp_path, "bad", raw=raw)
    with pytest.raises(ValueError, match=match):
        prepare([item])


def test_hash_split_keeps_duplicate_text_together_but_ingest_rejects_duplicates(tmp_path):
    first = source(tmp_path, "one", [{"id": "a", "text": "same text"}])
    second = source(tmp_path, "two", [{"id": "b", "text": "same text"}])
    assert SPLITS.split_for_text_hash(sha(b"same text")) == SPLITS.split_for_text_hash(sha(b"same text"))
    with pytest.raises(ValueError, match="Duplicate document text/tokens"):
        prepare([first, second])
    a, b = prepare([first]), prepare([second])
    assert a.data.manifest["documents"][0]["split"] == b.data.manifest["documents"][0]["split"]


@pytest.mark.parametrize("bound,value,match", [("max_documents", 1, "document bound"),
    ("max_source_bytes", 1, "source-byte bound"), ("max_uncompressed_bytes", 1, "uncompressed-byte bound"),
    ("max_tokens", 2, "normalized-token bound")])
def test_bounds_fail_instead_of_publishing_a_selected_prefix(tmp_path, bound, value, match):
    item = source(tmp_path, "input", [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}], compression="gzip")
    with pytest.raises(ValueError, match=match):
        prepare([item], **{bound: value})


def test_preprocessing_seed_and_explicit_field_names_are_pinned(tmp_path):
    item = source(tmp_path, "input", [{"id": "a", "alias": "a", "text": "hello", "copy": "hello"}])
    a = prepare([item])
    b = prepare([replace(item, text_field="copy", id_field="alias")])
    assert a.data.manifest_sha256 == b.data.manifest_sha256
    assert a.manifest_sha256 != b.manifest_sha256
    c = prepare_local_preflight([item], tokenizer=TOKENIZER, split_policy=SplitPolicy(999, SPLITS.weights), tokenize=characters)
    assert a.manifest_sha256 != c.manifest_sha256


def test_source_order_is_frozen_but_split_assignment_ignores_it(tmp_path):
    one = source(tmp_path, "one", [{"id": "a", "text": "first"}])
    two = source(tmp_path, "two", [{"id": "b", "text": "second"}])
    a, b = prepare([one, two]), prepare([two, one])
    assert a.manifest_sha256 != b.manifest_sha256
    assert a.data.manifest_sha256 != b.data.manifest_sha256
    assert {(d["document_id"], d["split"]) for d in a.data.manifest["documents"]} == {(d["document_id"], d["split"]) for d in b.data.manifest["documents"]}


@pytest.mark.parametrize("weights", [(), (("train", 0),), (("train", True),), (("train", 1), ("train", 2))])
def test_split_policy_requires_explicit_valid_allocation(weights):
    with pytest.raises(ValueError):
        SplitPolicy(0, weights)
