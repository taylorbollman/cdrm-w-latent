"""CPU stream oracles, bounded I/O, provenance and committed-cursor tests."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained import document_shards as ds
from cdrm.pretrained.campaign_data import SourcePin
from cdrm.pretrained.campaign_ingest import LocalJSONLSource, SplitPolicy
from cdrm.pretrained.campaign_recipe import CampaignRecipe, feedback_noise_for_rows
from cdrm.pretrained.nextlat import build_nextlat_masks
from cdrm.pretrained.packed_campaign_data import (DOCUMENT_POLICY, PackedCampaignData,
    PackedCounts, build_packed_index)

EOS = ds.EOS_ID


@pytest.fixture(autouse=True)
def cpu_fixture_tokenizer(monkeypatch):
    class FixtureTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=[int(v) for v in text.split()] if text else [])
    monkeypatch.setattr(ds, "_load_tokenizer", lambda path: FixtureTokenizer())
    torch.set_num_threads(1)


def corpus(tmp_path, documents=None, *, weights=(("train", 1),), max_rows=1):
    documents = documents or [[2, EOS, 3, 4, EOS], [5, 6, EOS], list(range(7, 18))+[EOS], []]
    source_path = tmp_path/"source.jsonl"
    raw = b"".join((json.dumps({"id": f"doc-{i}", "text": " ".join(map(str, values))})+"\n").encode()
                   for i, values in enumerate(documents))
    source_path.write_bytes(raw)
    source = LocalJSONLSource(SourcePin("fixture", "https://example.invalid/fixture", "fixed",
                                       hashlib.sha256(raw).hexdigest()), source_path)
    root = tmp_path/"corpus"
    ds.prepare_document_shards([source], root, tokenizer_path="unused",
        split_policy=SplitPolicy(19, weights), max_documents_per_shard=max_rows)
    return root


def reader(tmp_path, *, length=4, documents=None, weights=(("train", 1),), split="train", max_rows=1):
    root = corpus(tmp_path, documents, weights=weights, max_rows=max_rows)
    index = tmp_path/"index"
    manifest = build_packed_index(root, index, split=split, length=length)
    return root, index, manifest, PackedCampaignData(root, index)


def test_exact_nonoverlapping_chunks_keep_embedded_eos_and_true_shard_document_offsets(tmp_path):
    root, index, manifest, data = reader(tmp_path)
    with data:
        assert data.total_tokens == 21 and data.total_chunks == 6
        chunks = [data.read_chunk(i) for i in range(data.total_chunks)]
        assert chunks[0].tokens == (2, EOS, 3, 4)
        assert chunks[0].document_ids == (0, 0, 0, 0)  # Embedded EOS is not a boundary.
        assert chunks[1].tokens == (EOS, 5, 6, EOS)
        assert chunks[1].document_ids == (0, 1, 1, 1)
        assert [s.shard for s in chunks[1].segments] == ["shard-000000", "shard-000001"]
        assert [(s.document_token_offset, s.chunk_offset, s.length) for s in chunks[1].segments] == [(4, 0, 1), (0, 1, 3)]
        assert chunks[-1].tokens == (EOS,) and chunks[-1].document_ids == (3,)
        reference = [v for doc in ds.iter_documents(root) for v in doc.tokens]
        assert [v for chunk in chunks for v in chunk.tokens] == reference
        assert sum(len(chunk.tokens) for chunk in chunks) == len(reference)
        assert manifest["policy"]["stride"] == "context_length_no_overlap"
        assert manifest["policy"]["document_policy"] == DOCUMENT_POLICY


def test_target_and_boundary_accounting_matches_independent_global_stream_oracle(tmp_path):
    _, _, _, data = reader(tmp_path)
    with data:
        update = data.peek_update(data.cursor(), 100)
        assert update is not None and not update.reaches_target
        counts = update.counts
        assert counts.packed_rows == 6 and counts.valid_tokens == 21
        assert counts.ce_targets == 15 and counts.latent_pairs == 14 and counts.kl_triples == 9
        assert counts.cross_document_ce_targets == counts.excluded_boundary_latent_pairs == 1
        assert counts.excluded_boundary_kl_triples == 1
        assert counts.omitted_cross_chunk_ce_targets == 5
        assert counts.omitted_cross_chunk_latent_pairs == 3
        assert counts.omitted_cross_chunk_kl_triples == 5
        assert counts.tail_padding_tokens == 3
        assert counts.document_completions == 4 and update.unique_document_count == 4
        ids = [v for i in range(data.total_chunks) for v in data.read_chunk(i).document_ids]
        assert counts.ce_targets+counts.omitted_cross_chunk_ce_targets == len(ids)-1
        assert counts.latent_pairs+counts.omitted_cross_chunk_latent_pairs == sum(a == b for a,b in zip(ids, ids[1:]))
        assert counts.kl_triples+counts.omitted_cross_chunk_kl_triples == sum(a == b == c for a,b,c in zip(ids, ids[1:], ids[2:]))


@pytest.mark.parametrize("world_size,batch_size", [(1, 1), (2, 2), (3, 2), (4, 3)])
def test_partition_is_membership_preserving_and_masks_have_global_true_counts(tmp_path, world_size, batch_size):
    _, _, _, data = reader(tmp_path)
    with data:
        update = data.peek_update(data.cursor(), 17)
        assert update.counts.valid_tokens == 20 and update.overshoot_tokens == 3
        slots = data.partition(update, world_size=world_size, physical_batch_size=batch_size)
        assert tuple(row for slot in slots for rank in slot for row in rank) == update.rows
        global_counts = dict.fromkeys(("ce", "latent", "kl"), 0)
        accounted = PackedCounts()
        presentations = []
        for rank in range(world_size):
            rank_data = data.rank_batches(update, rank=rank, world_size=world_size, physical_batch_size=batch_size)
            assert len(rank_data.batches) == len(slots) == len(rank_data.keys)
            accounted += rank_data.counts
            assert rank_data.accounting["padding_tokens"] == rank_data.physical_rows*data.length-rank_data.counts.valid_tokens
            for batch, keys in zip(rank_data.batches, rank_data.keys):
                assert batch.input_ids.device.type == "cpu" and batch.input_ids.shape == (batch_size, 4)
                assert not (batch.valid_mask[:, 1:] & ~batch.valid_mask[:, :-1]).any()
                assert len(keys) == int(batch.valid_mask.any(-1).sum())
                assert (batch.document_ids[~batch.valid_mask] == -1).all()
                for term, mask in build_nextlat_masks(batch, document_policy=DOCUMENT_POLICY).items():
                    global_counts[term] += int(mask.sum())
                presentations.extend(keys)
        assert accounted == update.counts
        assert global_counts == update.counts.objective_counts
        assert sorted(presentations) == sorted(row.key for row in update.rows)
        if (world_size, batch_size) == (2, 2):
            empty = data.rank_batches(update, rank=1, world_size=2, physical_batch_size=2)
            assert not empty.batches[-1].valid_mask.any() and empty.keys[-1] == ()


def test_committed_cursor_ignores_prefetch_and_restores_next_logical_update(tmp_path):
    root, index, _, data = reader(tmp_path)
    with data:
        start = data.cursor()
        first = data.peek_update(start, 9)
        future = data.peek_update(first.next_cursor, 9)
        data.rank_batches(future, rank=0, world_size=2, physical_batch_size=2)
        assert data.cursor() == start  # Reads/prefetch never commit.
        committed = data.commit(start, first)
        assert committed.next_chunk == 3 and committed.next_update == 1
        with pytest.raises(ValueError, match="stale|out-of-order"):
            data.commit(start, first)
        with pytest.raises(ValueError, match="membership"):
            data.commit(committed, replace(future, counts=replace(future.counts, valid_tokens=99)))
        with PackedCampaignData(root, index) as resumed:
            assert resumed.restore_cursor(asdict(committed)) == committed
            assert resumed.peek_update(resumed.cursor(), 9) == future
            with pytest.raises(ValueError, match="fresh reader"):
                resumed.restore_cursor(asdict(start))
            resumed.commit(committed, future)
            assert resumed.peek_update(resumed.cursor(), 9) is None


def test_cursor_rejects_fingerprint_split_and_position_mismatch_before_advancing(tmp_path):
    _, _, _, data = reader(tmp_path)
    with data:
        original = asdict(data.cursor())
        for key, value in (("manifest_sha256", "0"*64), ("split", "dev"), ("next_chunk", 999),
                           ("next_update", -1), ("next_chunk", True)):
            with pytest.raises(ValueError, match="identity|fields"):
                data.restore_cursor({**original, key: value})
        assert asdict(data.cursor()) == original


def test_actual_t1024_boundary_eos_at_zero_and_final_position_and_tail(tmp_path):
    documents = [list(range(2, 1025))+[EOS], [], [7, EOS, 8, EOS]]
    _, _, _, data = reader(tmp_path, length=1024, documents=documents)
    with data:
        first, tail = data.read_chunk(0), data.read_chunk(1)
        assert len(first.tokens) == 1024 and first.tokens[-1] == EOS
        assert tail.tokens == (EOS, 7, EOS, 8, EOS)
        assert tail.document_ids == (1, 2, 2, 2, 2)
        update = data.peek_update(data.cursor(), 2048)
        batch = data.batch(update.rows, physical_batch_size=2)
        masks = build_nextlat_masks(batch, document_policy=DOCUMENT_POLICY)
        assert masks["ce"][1, 0] and not masks["latent"][1, 0]
        assert masks["latent"][1, 1]  # Embedded EOS in content is same document.
        assert update.counts.omitted_cross_chunk_ce_targets == 1
        assert update.counts.tail_padding_tokens == 1019


def test_existing_split_is_filtered_without_resplitting_or_reordering(tmp_path):
    documents = [[i+2, i+200, EOS] for i in range(40)]
    root, _, manifest, data = reader(tmp_path, documents=documents,
        weights=(("train", 1), ("dev", 1)), split="dev", max_rows=3)
    with data:
        all_docs = list(ds.iter_documents(root))
        expected = [v for doc in all_docs if doc.split == "dev" for v in doc.tokens]
        expected_ids = [i for i, doc in enumerate(all_docs) if doc.split == "dev" for _ in doc.tokens]
        actual = [data.read_chunk(i) for i in range(data.total_chunks)]
        assert [v for c in actual for v in c.tokens] == expected
        assert [v for c in actual for v in c.document_ids] == expected_ids
        assert manifest["selected_documents"] == sum(doc.split == "dev" for doc in all_docs)
        assert all(seg.source_name == "fixture" for c in actual for seg in c.segments)


def test_metadata_peek_reads_no_tokens_and_chunk_reads_are_bounded(tmp_path, monkeypatch):
    _, _, _, data = reader(tmp_path, documents=[list(range(2, 2000))+[EOS]], length=16)
    with data:
        original = data._token_slice
        monkeypatch.setattr(data, "_token_slice", lambda *args: (_ for _ in ()).throw(AssertionError("token read during peek")))
        update = data.peek_update(data.cursor(), 512)
        assert update.counts.valid_tokens == 512 and len(update.rows) == 32
        monkeypatch.setattr(data, "_token_slice", original)
        import cdrm.pretrained.packed_campaign_data as module
        pread = module.os.pread
        reads = []
        def bounded(fd, count, offset):
            reads.append((count, offset))
            return pread(fd, count, offset)
        monkeypatch.setattr(module.os, "pread", bounded)
        assert len(data.read_chunk(50).tokens) == 16
        assert reads == [(32, 1600)]


def test_token_fd_cache_is_bounded_and_replaced_live_files_are_rejected(tmp_path):
    root, _, _, data = reader(tmp_path, documents=[[i+2, EOS] for i in range(20)])
    with data:
        for i in range(data.total_chunks):
            data.read_chunk(i)
        assert len(data._token_fds) <= 8
        path = root/"shard-000000/tokens.bin"
        replacement = path.with_name("replacement.bin")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        with pytest.raises(ValueError, match="changed"):
            data.read_chunk(0)


@pytest.mark.parametrize("relative", ["shard-000000/tokens.bin", "shard-000000/documents.jsonl", "manifest.json"])
def test_open_rejects_modified_corpus_bytes(tmp_path, relative):
    root, index, _, data = reader(tmp_path)
    data.close()
    path = root/relative
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="Corpus bytes"):
        PackedCampaignData(root, index)


def test_index_mutation_bad_provenance_and_runtime_changes_are_rejected(tmp_path):
    _, index, _, data = reader(tmp_path)
    with data:
        row = data.descriptor(0)
        with pytest.raises(ValueError, match="provenance"):
            data.batch((replace(row, key="bad"),), physical_batch_size=1)
        data.length = 8
        with pytest.raises(ValueError, match="runtime contract"):
            data.cursor()
        data.length = 4
        path = index/"documents.sqlite"
        path.write_bytes(path.read_bytes()+b"bad")
        with pytest.raises(ValueError, match="changed"):
            data.cursor()


@pytest.mark.parametrize("relative", ["manifest.json", "documents.sqlite"])
def test_open_rejects_index_replacement_after_verification(tmp_path, monkeypatch, relative):
    root, index, _, data = reader(tmp_path)
    data.close()
    import cdrm.pretrained.packed_campaign_data as module
    original_sha = module._sha
    mutated = False

    def replace_after_hash(path):
        nonlocal mutated
        digest = original_sha(path)
        if path == index/"documents.sqlite" and not mutated:
            target = index/relative
            replacement = index/"replacement"
            replacement.write_bytes(target.read_bytes())
            if relative == "manifest.json":
                replacement.write_bytes(replacement.read_bytes()+b"\n")
            else:
                connection = module.sqlite3.connect(str(replacement))
                try:
                    connection.execute("PRAGMA user_version=1")
                    connection.commit()
                finally:
                    connection.close()
            replacement.replace(target)
            mutated = True
        return digest

    monkeypatch.setattr(module, "_sha", replace_after_hash)
    with pytest.raises(ValueError, match="changed"):
        PackedCampaignData(root, index)
    assert mutated


def test_relocated_index_and_corpus_preserve_identity_cursor_chunks_and_jitter(tmp_path):
    root, index, manifest, data = reader(tmp_path)
    with data:
        update = data.peek_update(data.cursor(), 17)
        original_keys = tuple(row.key for row in update.rows)
        copied = tmp_path/"restored"
        shutil.copytree(root, copied/"corpus")
        shutil.copytree(index, copied/"index")
        with PackedCampaignData(copied/"corpus", copied/"index") as restored:
            assert restored.manifest_sha256 == data.manifest_sha256
            assert restored.manifest == manifest
            other = restored.peek_update(restored.cursor(), 17)
            assert other == update
            recipe = CampaignRecipe("NFR", sequence_length=4, rt_layers=(0, 1), document_policy=DOCUMENT_POLICY)
            noise = feedback_noise_for_rows(recipe, original_keys, logical_update=0, sequence_length=4, width=32)
            recovered = feedback_noise_for_rows(recipe, tuple(r.key for r in other.rows), logical_update=0, sequence_length=4, width=32)
            assert all(torch.equal(a, b) for a, b in zip(noise, recovered))


def test_index_publication_is_atomic_and_policy_order_and_length_are_pinned(tmp_path, monkeypatch):
    root = corpus(tmp_path)
    index = tmp_path/"index"
    import cdrm.pretrained.packed_campaign_data as module
    original = module.os.rename
    monkeypatch.setattr(module.os, "rename", lambda *args: (_ for _ in ()).throw(RuntimeError("interrupted publication")))
    with pytest.raises(RuntimeError, match="interrupted"):
        build_packed_index(root, index, split="train", length=4)
    assert not index.exists() and not list(tmp_path.glob(".pending-packed-index-*"))
    monkeypatch.setattr(module.os, "rename", original)
    first = build_packed_index(root, index, split="train", length=4)
    second = build_packed_index(root, tmp_path/"index-again", split="train", length=4)
    assert first == second
    third = build_packed_index(root, tmp_path/"index-longer", split="train", length=8)
    assert first["identity_sha256"] != third["identity_sha256"]
    assert first["order_sha256"] == third["order_sha256"]
    with pytest.raises(ValueError, match="new"):
        build_packed_index(root, index, split="train", length=4)
    with pytest.raises(ValueError, match="continuous-stream"):
        build_packed_index(root, tmp_path/"isolated", split="train", document_policy="isolated-v1")
    with pytest.raises(ValueError, match="Split"):
        build_packed_index(root, tmp_path/"unknown", split="missing")
