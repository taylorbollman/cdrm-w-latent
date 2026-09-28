"""CPU integrity, deterministic split, and interruption tests for document shards."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unicodedata

import pytest

from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained import document_shards as ds


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        parts = unicodedata.normalize("NFC", text).split("<|endoftext|>")
        tokens = []
        for i, part in enumerate(parts):
            if i:
                tokens.append(ds.EOS_ID)
            tokens.extend(100 + b for b in part.encode("utf-8"))
        return SimpleNamespace(ids=tokens)


@pytest.fixture
def fake_tokenizer(monkeypatch):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(ds, "_load_tokenizer", lambda path: tokenizer)
    return tokenizer


def source(tmp_path, rows, *, name="source", compressed=False):
    raw = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
    if compressed:
        raw = gzip.compress(raw, mtime=0)
    path = tmp_path / (name + (".jsonl.gz" if compressed else ".jsonl"))
    path.write_bytes(raw)
    pin = SourcePin(name, "https://example.invalid/" + name, "fixed-revision", hashlib.sha256(raw).hexdigest())
    return LocalJSONLSource(pin, path, compression="gzip" if compressed else "none")


def prepare(sources, output, **kwargs):
    return ds.prepare_document_shards(sources, output, tokenizer_path="unused.json",
                                     split_policy=SplitPolicy(19, (("train", 9), ("dev", 1))),
                                     **kwargs)


def records(output):
    result = []
    for path in sorted(output.glob("shard-*/documents.jsonl")):
        result.extend(json.loads(raw) for raw in path.read_bytes().splitlines())
    return result


def artifact_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file() and not path.name.startswith(".")}


def test_complete_documents_unicode_eos_long_and_dedup(tmp_path, fake_tokenizer):
    texts = ["cafe\u0301", "café", "a<|endoftext|>b", "done<|endoftext|>",
             "done", "x" * 1300, "", "<|endoftext|>", "z<|endoftext|><|endoftext|>"]
    src = source(tmp_path, [{"id": str(i), "text": t} for i, t in enumerate(texts)])
    output = tmp_path / "output"
    result = prepare([src], output, max_documents_per_shard=2, target_tokens_per_shard=1000)
    assert result["completed"]
    assert result["rows"] == 9
    assert result["documents"] == 6
    assert result["duplicate_rows"] == 3
    assert result == ds.verify_document_shards(output)
    docs = list(ds.iter_documents(output))
    assert len(docs) == 6
    assert len(docs[3].tokens) == 1301  # No 1024-token truncation.
    assert all(doc.tokens[-1] == ds.EOS_ID for doc in docs)
    assert docs[1].tokens.count(ds.EOS_ID) == 2  # Literal internal EOS kept.
    assert docs[-1].tokens[-2:] == (ds.EOS_ID, ds.EOS_ID)
    meta = records(output)
    assert meta[0]["split"] == meta[1]["split"]
    assert meta[0]["text_sha256"] != meta[1]["text_sha256"]
    assert meta[0]["content_token_sha256"] == meta[1]["content_token_sha256"]
    assert meta[3]["had_terminal_eos"] is True
    assert meta[4]["kind"] == "duplicate"
    assert meta[8]["embedded_eos_count"] == 1
    # uint16 little endian; first encoded character is c (99)+100.
    assert (output / "shard-000000/tokens.bin").read_bytes()[:2] == b"\xc7\x00"


@pytest.mark.parametrize("compressed", [False, True])
def test_resume_matches_uninterrupted_bytes(tmp_path, fake_tokenizer, compressed):
    a = source(tmp_path, [{"id": f"a{i}", "text": str(i)} for i in range(5)], compressed=compressed)
    b = source(tmp_path, [{"id": f"b{i}", "text": str(i)} for i in range(3, 9)], name="second", compressed=compressed)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    expected = prepare([a, b], full, max_documents_per_shard=3)
    first = prepare([a, b], resumed, max_documents_per_shard=3, max_new_shards=1)
    assert not first["completed"] and first["rows"] == 3
    assert not (resumed / "manifest.json").exists()
    assert ds.verify_document_shards(resumed) == first
    # Simulate power interruption leaving uncommitted bytes. They are not read.
    pending = resumed / ".pending-crash"
    pending.mkdir()
    (pending / "tokens.bin").write_bytes(b"garbage")
    assert len(list(ds.iter_documents(resumed))) == 3
    second = prepare([a, b], resumed, max_documents_per_shard=3, max_new_shards=1)
    assert not second["completed"] and second["rows"] == 6
    assert not pending.exists()
    actual = prepare([a, b], resumed, max_documents_per_shard=3)
    assert actual == expected
    assert artifact_bytes(resumed) == artifact_bytes(full)
    assert prepare([a, b], resumed, max_documents_per_shard=3) == actual


def test_crash_before_commit_does_not_publish_partial_shard(tmp_path, fake_tokenizer, monkeypatch):
    src = source(tmp_path, [{"id": str(i), "text": str(i)} for i in range(5)])
    output = tmp_path / "out"
    original = fake_tokenizer.encode

    def fail(text, **kwargs):
        if text == "3":
            raise RuntimeError("simulated interruption")
        return original(text, **kwargs)

    monkeypatch.setattr(fake_tokenizer, "encode", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        prepare([src], output, max_documents_per_shard=2)
    assert len(list(output.glob("shard-*"))) == 1
    assert ds.verify_document_shards(output)["rows"] == 2
    monkeypatch.setattr(fake_tokenizer, "encode", original)
    assert prepare([src], output, max_documents_per_shard=2)["rows"] == 5


def test_all_sources_verified_before_any_tokenization(tmp_path, fake_tokenizer, monkeypatch):
    a = source(tmp_path, [{"id": "a", "text": "a"}])
    b = source(tmp_path, [{"id": "b", "text": "b"}], name="bad")
    b.path.write_bytes(b.path.read_bytes() + b" ")
    calls = []
    monkeypatch.setattr(fake_tokenizer, "encode", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="pinned SHA256"):
        prepare([a, b], tmp_path / "out")
    assert not calls


def test_source_changes_during_tokenization_not_committed(tmp_path, fake_tokenizer, monkeypatch):
    src = source(tmp_path, [{"id": "a", "text": "a"}])
    original = fake_tokenizer.encode

    def change(text, **kwargs):
        src.path.write_bytes(src.path.read_bytes() + b" ")
        return original(text, **kwargs)

    monkeypatch.setattr(fake_tokenizer, "encode", change)
    with pytest.raises(ValueError, match="Source changed"):
        prepare([src], tmp_path / "out", max_documents_per_shard=1)
    assert not list((tmp_path / "out").glob("shard-*"))


def test_conflicting_document_identity_rejected_across_resume(tmp_path, fake_tokenizer):
    src = source(tmp_path, [{"id": "same", "text": "first"}, {"id": "same", "text": "second"}])
    output = tmp_path / "out"
    prepare([src], output, max_documents_per_shard=1, max_new_shards=1)
    with pytest.raises(ValueError, match="Conflicting content tokens"):
        prepare([src], output, max_documents_per_shard=1)
    assert ds.verify_document_shards(output)["rows"] == 1


def test_same_identity_same_tokens_deduplicates_and_sources_are_independent(tmp_path, fake_tokenizer):
    src = source(tmp_path, [{"id": "same", "text": "café"}, {"id": "same", "text": "cafe\u0301"}])
    other = source(tmp_path, [{"id": "same", "text": "different"}], name="other")
    output = tmp_path / "out"
    result = prepare([src, other], output, max_documents_per_shard=1)
    assert result["documents"] == 2 and result["duplicate_rows"] == 1
    assert ds.verify_document_shards(output) == result


def test_reader_rejects_conflicting_identity_even_with_updated_metadata_hash(tmp_path, fake_tokenizer):
    src = source(tmp_path, [{"id": "a", "text": "abc"}, {"id": "b", "text": "def"}])
    output = tmp_path / "out"
    prepare([src], output)
    path = output / "shard-000000/documents.jsonl"
    rows = [json.loads(row) for row in path.read_bytes().splitlines()]
    rows[1]["document_id"] = "a"
    path.write_bytes(b"".join(ds._json(row) for row in rows))
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][path.name] = {"size_bytes": path.stat().st_size, "sha256": ds._sha(path)}
    manifest_path.write_bytes(ds._json(manifest))
    with pytest.raises(ValueError, match="Conflicting content tokens"):
        ds.verify_document_shards(output)


@pytest.mark.parametrize("change", ["config", "source", "split"])
def test_resume_rejects_configuration_or_source_identity_change(tmp_path, fake_tokenizer, change):
    src = source(tmp_path, [{"id": str(i), "text": str(i)} for i in range(4)])
    output = tmp_path / "out"
    prepare([src], output, max_documents_per_shard=2, max_new_shards=1)
    kwargs = {"max_documents_per_shard": 2}
    if change == "config":
        kwargs["target_tokens_per_shard"] = 500
    if change == "source":
        src = replace(src, pin=replace(src.pin, revision="different"))
    with pytest.raises(ValueError, match="configuration"):
        if change == "split":
            ds.prepare_document_shards([src], output, tokenizer_path="unused",
                                       split_policy=SplitPolicy(20, (("train", 9), ("dev", 1))), **kwargs)
        else:
            prepare([src], output, **kwargs)


@pytest.mark.parametrize("filename", ["tokens.bin", "documents.jsonl", "manifest.json"])
def test_corrupt_shard_bytes_rejected(tmp_path, fake_tokenizer, filename):
    src = source(tmp_path, [{"id": "a", "text": "abc"}])
    output = tmp_path / "out"
    prepare([src], output)
    path = output / "shard-000000" / filename
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        ds.verify_document_shards(output)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        prepare([src], output)


def test_offsets_checked_even_if_metadata_hash_updated(tmp_path, fake_tokenizer):
    src = source(tmp_path, [{"id": "a", "text": "abc"}])
    output = tmp_path / "out"
    prepare([src], output)
    path = output / "shard-000000/documents.jsonl"
    record = json.loads(path.read_bytes())
    record["token_offset"] = 1
    path.write_bytes(ds._json(record))
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][path.name] = {"size_bytes": path.stat().st_size, "sha256": ds._sha(path)}
    manifest_path.write_bytes(ds._json(manifest))
    with pytest.raises(ValueError, match="offset"):
        ds.verify_document_shards(output)


def test_record_bound_rejects_without_truncation(tmp_path, fake_tokenizer):
    src = source(tmp_path, [{"id": "a", "text": "abc" * 100}])
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="max_record_bytes"):
        prepare([src], output, max_record_bytes=100)
    assert not list(output.glob("shard-*"))


def test_pinned_tokenizer_hash_required(tmp_path):
    src = source(tmp_path, [{"id": "a", "text": "a"}])
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    with pytest.raises(ValueError, match="pinned OLMo"):
        ds.prepare_document_shards([src], tmp_path / "out", tokenizer_path=tokenizer,
                                   split_policy=SplitPolicy(0, (("train", 1),)))


@pytest.mark.parametrize("field,value", [("max_documents_per_shard", 0), ("target_tokens_per_shard", -1),
                                         ("max_record_bytes", True), ("max_new_shards", 0)])
def test_invalid_limits_rejected(tmp_path, fake_tokenizer, field, value):
    src = source(tmp_path, [{"id": "a", "text": "a"}])
    with pytest.raises(ValueError, match="positive integer"):
        prepare([src], tmp_path / "out", **{field: value})


def test_symlink_source_rejected(tmp_path, fake_tokenizer):
    src = source(tmp_path, [{"id": "a", "text": "a"}])
    linked = tmp_path / "symlink.jsonl"
    linked.symlink_to(src.path)
    with pytest.raises(ValueError, match="non-symlink"):
        prepare([replace(src, path=linked)], tmp_path / "out")


def test_empty_sources_and_exact_threshold_completion(tmp_path, fake_tokenizer):
    empty = source(tmp_path, [], name="empty")
    assert prepare([empty], tmp_path / "zero")["tokens"] == 0
    assert ds.verify_document_shards(tmp_path / "zero")["completed"]
    src = source(tmp_path, [{"id": "a", "text": "a"}])
    output = tmp_path / "one"
    first = prepare([src], output, max_documents_per_shard=1, max_new_shards=1)
    assert not first["completed"]
    final = prepare([src], output, max_documents_per_shard=1, max_new_shards=1)
    assert final["completed"] and len(final["shards"]) == 1
    assert ds.verify_document_shards(output) == final
