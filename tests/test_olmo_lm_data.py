"""Download-free evidence for document windows, provenance and exact data cursors."""
import json
from pathlib import Path
from types import SimpleNamespace
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from cdrm.pretrained import lm_data as data
from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.nextlat import build_nextlat_masks


class CharacterTokenizer:
    def token_to_id(self, token):
        assert token == "<|endoftext|>"
        return data.EOS_ID

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[ord(c) + 2 for c in unicodedata.normalize("NFC", text)])


@pytest.fixture
def sources(tmp_path, monkeypatch):
    import tokenizers
    raw = tmp_path / "raw"
    values = {
        "train": ["train function one with many tokens", "train function two distinct", "train three", "same across splits", "cafe\u0301", ""],
        "dev": ["development unique function", "same across splits", "café"],
        "test": ["test unique function", "same across splits"],
        "retention_dev": ["", " = Dev Article = \n", "paragraph A\n", "", " = = Subsection = = \n", "paragraph B\n", " = Another Dev = \n", "other\n"],
        "retention_test": [" = Test Article = \n", "test paragraph\n"],
    }
    specs = {}
    for split, original in data.RAW_SPECS.items():
        path = data._raw_path(raw, original)
        path.parent.mkdir(parents=True, exist_ok=True)
        if split in ("train", "dev", "test"):
            rows = [{"func_code_string": text, "repository_name": f"{split}/repo", "func_code_url": f"https://example/{split}/{i}"}
                    for i, text in enumerate(values[split])]
        else:
            rows = [{"text": text} for text in values[split]]
        pq.write_table(pa.Table.from_pylist(rows), path)
        specs[split] = (*original[:3], path.stat().st_size, sha256_file(path))
    monkeypatch.setattr(data, "RAW_SPECS", specs)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("test tokenizer bytes")
    monkeypatch.setattr(data, "FILE_SPECS", {"tokenizer.json": (tokenizer.stat().st_size, sha256_file(tokenizer))})
    monkeypatch.setattr(tokenizers, "Tokenizer", SimpleNamespace(from_file=lambda path: CharacterTokenizer()))
    return SimpleNamespace(raw=raw, tokenizer=tokenizer, output=tmp_path / "prepared", values=values)


def prepare(sources, **kwargs):
    return data.prepare_lm_data(raw_root=sources.raw, tokenizer_path=sources.tokenizer,
                               output_dir=sources.output, train_token_budget=50, length=8, **kwargs)


def metadata(root, split):
    return [json.loads(line) for line in (root / f"{split}.documents.jsonl").read_text().splitlines()]


def rewrite_manifest(root, edit):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text()); edit(manifest)
    path.write_text(json.dumps(manifest))


def test_windows_preserve_all_ce_pairs_with_one_context_overlap(sources):
    manifest = prepare(sources)
    corpus = data.load_lm_data(sources.output)
    for split in data.SPLITS:
        all_tokens = corpus._tokens[split]
        observed_targets = []
        for record in metadata(sources.output, split):
            selected = list(range(record["window_start"], record["window_start"] + record["windows"]))
            batch = corpus.batch(split, selected)
            assert batch.input_ids.shape == (len(selected), 8)
            assert batch.ce_mask is batch.latent_mask is batch.kl_mask is None
            target_indices = []
            for i, row_index in enumerate(selected):
                offset, count, doc_id, start = corpus._windows[split][row_index]
                assert doc_id == record["document_id"]
                target_indices.extend(range(start + 1, start + count))
                assert all(batch.document_ids[i, :count].tolist()[j] == doc_id for j in range(count))
                assert bool((batch.document_ids[i, count:] == -1).all())
                assert bool((batch.input_ids[i, count:] == data.PAD_ID).all())
                if start + count < record["token_count"]:
                    assert batch.input_ids[i, count-1] != data.EOS_ID
                else:
                    assert batch.input_ids[i, count-1] == data.EOS_ID
            assert target_indices == list(range(1, record["token_count"]))
            stored = all_tokens[record["token_offset"]:record["token_offset"]+record["token_count"]]
            assert (stored == data.EOS_ID).sum() == 1
            observed_targets.extend(target_indices)
        assert len(observed_targets) == manifest["splits"][split]["ce_targets"]
        batch = corpus.batch(split, range(corpus.split_sizes[split]))
        masks = build_nextlat_masks(batch)
        assert int(masks["ce"].sum()) == manifest["splits"][split]["ce_targets"]
        assert int(masks["latent"].sum()) == manifest["splits"][split]["latent_pairs"]
        assert int(masks["kl"].sum()) == manifest["splits"][split]["kl_triples"]
        assert int(batch.valid_mask.sum()) == manifest["splits"][split]["window_input_tokens"]
        assert manifest["splits"][split]["kl_boundary_triples_omitted"] == manifest["splits"][split]["windows"] - manifest["splits"][split]["documents"]


def test_native_token_and_byte_dedup_have_heldout_precedence(sources):
    prepare(sources)
    records = {split: metadata(sources.output, split) for split in data.SPLITS}
    for key in ("text_sha256", "token_sha256"):
        sets = {split: {row[key] for row in rows} for split, rows in records.items()}
        for a in data.SPLITS:
            for b in data.SPLITS:
                if a != b:
                    assert not sets[a] & sets[b]
    cross = data._text_hash("same across splits")
    assert any(row["text_sha256"] == cross for row in records["test"])
    assert all(row["text_sha256"] != cross for row in records["dev"] + records["train"])
    assert all(row["text_sha256"] != data._text_hash("cafe\u0301") for row in records["train"])


def test_wikitext_headings_preserve_articles_and_subheadings(sources):
    path = data._raw_path(sources.raw, data.RAW_SPECS["retention_dev"])
    records = data._read_wikitext(path)
    assert len(records) == 2
    assert "Subsection" in records[0]["text"]
    assert "paragraph A\n\n" in records[0]["text"]
    assert "Another Dev" not in records[0]["text"]
    assert records[1]["source_row"] == 6


def test_reproducible_preparation_and_cursor_resume(sources, tmp_path):
    first = prepare(sources)
    other = tmp_path / "other"
    second = data.prepare_lm_data(raw_root=sources.raw, tokenizer_path=sources.tokenizer,
                                 output_dir=other, train_token_budget=50, length=8)
    assert first["files"] == second["files"]
    corpus = data.load_lm_data(sources.output, expected_manifest_sha256=sha256_file(sources.output / "manifest.json"))
    assert np.array_equal(corpus.train_lengths, corpus.lengths("train"))
    initial, cursor = corpus.next_train_batch(0, 2)
    expected, end = corpus.next_train_batch(cursor, 2)
    restored = data.load_lm_data(sources.output)
    actual, actual_end = restored.next_train_batch(cursor, 2)
    assert end == actual_end == 4
    for name in ("input_ids", "valid_mask", "document_ids"):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    with pytest.raises(StopIteration, match="cycling"):
        corpus.next_train_batch(corpus.split_sizes["train"] - 1, 2)
    with pytest.raises(ValueError): corpus.next_train_batch(-1, 2)
    with pytest.raises(ValueError): corpus.next_train_batch(0, 0)
    with pytest.raises(ValueError): corpus.batch("train", [-1])
    with pytest.raises(ValueError): corpus.batch("train", [0.5])
    with pytest.raises(ValueError): corpus.batch("test", [])


@pytest.mark.parametrize("what", ["raw", "tokenizer"])
def test_prepare_rejects_wrong_source_bytes_before_publication(sources, what):
    path = sources.tokenizer if what == "tokenizer" else data._raw_path(sources.raw, data.RAW_SPECS["train"])
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="[Pp]in"):
        prepare(sources)
    assert not sources.output.exists()


def test_repository_overlap_rejected_before_publication(sources, monkeypatch):
    spec = data.RAW_SPECS["dev"]
    path = data._raw_path(sources.raw, spec)
    rows = pq.read_table(path).to_pylist()
    rows[0]["repository_name"] = "train/repo"
    pq.write_table(pa.Table.from_pylist(rows), path)
    specs = dict(data.RAW_SPECS); specs["dev"] = (*spec[:3], path.stat().st_size, sha256_file(path))
    monkeypatch.setattr(data, "RAW_SPECS", specs)
    with pytest.raises(ValueError, match="repository"):
        prepare(sources)
    assert not sources.output.exists()


@pytest.mark.parametrize("edit,match", [
    (lambda m: m["raw_sources"]["train"].update(revision="wrong"), "provenance"),
    (lambda m: m["tokenizer"].update(add_special_tokens=True), "tokenizer"),
    (lambda m: m["files"].update({"../foreign": {}}), "inventory"),
    (lambda m: m.update(source_sha256="0"*64), "source"),
    (lambda m: m.update(length=513), "length"),
    (lambda m: m["preprocessing"].update(window_context_overlap=0), "contract"),
])
def test_manifest_rejects_provenance_changes(sources, edit, match):
    prepare(sources); rewrite_manifest(sources.output, edit)
    with pytest.raises(ValueError, match=match): data.load_lm_data(sources.output)


def test_prepared_bytes_and_manifest_fingerprint_rejected(sources):
    prepare(sources)
    with pytest.raises(ValueError, match="fingerprint"):
        data.load_lm_data(sources.output, expected_manifest_sha256="0" * 64)
    path = sources.output / "train.tokens.bin"
    payload = bytearray(path.read_bytes()); payload[0] ^= 1; path.write_bytes(payload)
    with pytest.raises(ValueError, match="integrity"): data.load_lm_data(sources.output)


def test_existing_output_and_insufficient_budget_do_not_publish(sources, tmp_path):
    prepare(sources)
    with pytest.raises(FileExistsError): prepare(sources)
    insufficient = tmp_path / "too-small"
    with pytest.raises(ValueError, match="Insufficient"):
        data.prepare_lm_data(raw_root=sources.raw, tokenizer_path=sources.tokenizer,
                             output_dir=insufficient, train_token_budget=1_000_000, length=8)
    assert not (insufficient / "manifest.json").exists()


def test_long_window_bounds_stay_inside_single_document(sources):
    prepare(sources)
    path = sources.output / "train.windows.npy"
    rows = np.load(path); rows[0, 0] = 10**9
    np.save(path, rows, allow_pickle=False)
    rewrite_manifest(sources.output, lambda m: m["files"][path.name].update(sha256=sha256_file(path), size_bytes=path.stat().st_size))
    with pytest.raises(ValueError, match="bounds"): data.load_lm_data(sources.output)


def test_in_bounds_cross_document_window_is_rejected(sources):
    prepare(sources)
    path = sources.output / "train.windows.npy"
    rows = np.load(path); rows[0, 0] += 1
    np.save(path, rows, allow_pickle=False)
    rewrite_manifest(sources.output, lambda m: m["files"][path.name].update(sha256=sha256_file(path), size_bytes=path.stat().st_size))
    with pytest.raises(ValueError, match="document boundaries"): data.load_lm_data(sources.output)
