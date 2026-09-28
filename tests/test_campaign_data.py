"""Download-free campaign stream conservation and physical-partition evidence."""
from dataclasses import asdict, replace
import hashlib
import json

import pytest
import torch

from cdrm.pretrained.campaign_data import (
    CampaignData, DataCursor, SourcePin, TokenizedDocument, TokenizerPin,
    count_windows, partition_update,
)
from cdrm.pretrained.nextlat import build_nextlat_masks


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


SOURCE = SourcePin("fixture", "local:fixture.jsonl", "fixture-v1", sha("source bytes"))
TOKENIZER = TokenizerPin("fixture/tokenizer", "immutable-fixture-revision", sha("tokenizer bytes"))


def document(index, tokens, split="train", source=SOURCE):
    return TokenizedDocument(source, str(index), split, sha(f"document {index}"), tuple(tokens))


def corpus(length=8):
    return CampaignData([
        document("short", [20]),
        document("long", range(30, 51)),
        document("exact", range(60, 67)),
        document("tail", range(70, 82)),
        document("dev", range(100, 110), "dev"),
    ], tokenizer=TOKENIZER, length=length)


def test_single_document_windows_preserve_short_documents_tails_and_all_pairs():
    data = corpus()
    manifest = data.manifest
    for split in ("train", "dev"):
        rows = data.windows(split)
        batch = data.batch(rows)
        masks = build_nextlat_masks(batch)
        counts = count_windows(rows)
        assert int(batch.valid_mask.sum()) == counts.presented_tokens
        assert int(masks["ce"].sum()) == counts.ce_targets
        assert int(masks["latent"].sum()) == counts.latent_pairs
        assert int(masks["kl"].sum()) == counts.kl_triples
        assert counts.presented_tokens == counts.new_unique_tokens + counts.overlap_tokens
        assert asdict(counts) == manifest["splits"][split]
        assert batch.input_ids.shape == (len(rows), 8)
        for index, row in enumerate(rows):
            assert bool((batch.document_ids[index, :row.length] == row.document_index).all())
            assert bool((batch.document_ids[index, row.length:] == -1).all())
            assert bool((batch.input_ids[index, row.length:] == data.pad_id).all())
        for index, record in enumerate(manifest["documents"]):
            if record["split"] != split:
                continue
            selected = [row for row in rows if row.document_index == index]
            n = record["token_count"]
            # CE/latent targets occur exactly once, including every window seam.
            assert [target for row in selected for target in range(row.start + 1, row.start + row.length)] == list(range(1, n))
            # KL needs three tokens, so the one-token context overlap omits one
            # boundary triple per additional window, exactly as reported.
            selected_kl = [target for row in selected for target in range(row.start + 2, row.start + row.length)]
            assert len(selected_kl) == len(set(selected_kl))
            assert (n - 2) - len(selected_kl) == len(selected) - 1
            assert sum(row.counts.new_unique_tokens for row in selected) == n
            assert selected[-1].tokens[-1] == data.eos_id
            assert all(data.eos_id not in row.tokens for row in selected[:-1])
    assert data.windows("train")[0].length == 2
    assert any(2 < row.length < data.length for row in data.windows("train"))


def test_terminal_eos_is_retained_once_and_only_appended_at_document_end():
    data = CampaignData([document("supplied", [7, 8, 50279]), document("append", [9, 10])], tokenizer=TOKENIZER)
    assert [row.tokens for row in data.windows("train")] == [(7, 8, 50279), (9, 10, 50279)]
    assert [record["eos_appended"] for record in data.manifest["documents"]] == [False, True]


@pytest.mark.parametrize("tokens", [[50279], [7, 50279, 8], [7, 50279, 50279]])
def test_internal_or_duplicate_eos_is_rejected_not_silently_rewritten(tokens):
    with pytest.raises(ValueError, match="terminal EOS"):
        CampaignData([document("bad", tokens)], tokenizer=TOKENIZER)


def test_updates_cover_corpus_once_and_resume_exactly_at_saved_next_cursor():
    data = corpus()
    cursor = data.cursor("train")
    seen, updates = [], []
    while True:
        try:
            update = data.next_update(cursor, target_valid_tokens=17)
        except StopIteration:
            break
        counts = update.counts
        if update.reaches_target:
            assert 17 <= counts.presented_tokens < 17 + data.length
        else:
            assert update.next_cursor.next_window == len(data.windows("train"))
        restored = data.restore_cursor(json.loads(json.dumps(asdict(cursor))))
        assert data.next_update(restored, 17) == update
        updates.append(update)
        seen.extend(update.rows)
        cursor = update.next_cursor
    assert tuple(seen) == data.windows("train")
    assert len({row.key for row in seen}) == len(seen)
    assert not updates[-1].reaches_target
    assert sum(update.counts.presented_tokens for update in updates) == data.manifest["splits"]["train"]["presented_tokens"]
    assert sum(update.counts.new_unique_tokens for update in updates) == data.manifest["splits"]["train"]["new_unique_tokens"]
    with pytest.raises(StopIteration, match="cycling"):
        data.next_update(cursor, 17)


@pytest.mark.parametrize("world_size,physical_batch_size", [(1, 1), (1, 5), (2, 2), (3, 1), (8, 4)])
def test_physical_partitions_preserve_global_order_counts_and_empty_rank_slots(world_size, physical_batch_size):
    data = corpus()
    update = data.next_update(data.cursor("train"), 10**6)
    partitions = partition_update(update, world_size=world_size, physical_batch_size=physical_batch_size)
    assert all(len(step) == world_size for step in partitions)
    flat = tuple(row for step in partitions for rank_rows in step for row in rank_rows)
    assert flat == update.rows
    actual = {"presented_tokens": 0, "ce_targets": 0, "latent_pairs": 0, "kl_triples": 0}
    for step in partitions:
        for rows in step:
            assert len(rows) <= physical_batch_size
            batch = data.batch(rows, physical_batch_size=physical_batch_size)
            masks = build_nextlat_masks(batch)
            actual["presented_tokens"] += int(batch.valid_mask.sum())
            for term, key in (("ce", "ce_targets"), ("latent", "latent_pairs"), ("kl", "kl_triples")):
                actual[key] += int(masks[term].sum())
            if not rows:
                assert not batch.valid_mask.any()
                assert not any(mask.any() for mask in masks.values())
                assert bool((batch.document_ids == -1).all())
    assert actual == {key: getattr(update.counts, key) for key in actual}


def test_short_local_partition_can_have_no_kl_targets_with_nonempty_global_kl():
    data = corpus()
    update = data.next_update(data.cursor("train"), 11)
    partitions = partition_update(update, world_size=2, physical_batch_size=1)
    local = build_nextlat_masks(data.batch(partitions[0][0]))
    assert int(local["ce"].sum()) == int(local["latent"].sum()) == 1
    assert int(local["kl"].sum()) == 0
    assert update.counts.kl_triples > 0


def test_manifest_and_window_keys_are_reproducible_and_manifest_is_defensive_copy():
    left, right = corpus(), corpus()
    assert left.manifest == right.manifest
    assert left.manifest_sha256 == right.manifest_sha256
    assert left.windows("train") == right.windows("train")
    left.manifest["length"] = 7
    assert left.manifest["length"] == 8
    docs = [document("a", [5, 6]), document("b", [8, 9])]
    a = CampaignData(docs, tokenizer=TOKENIZER)
    b = CampaignData(reversed(docs), tokenizer=TOKENIZER)
    assert a.manifest_sha256 != b.manifest_sha256
    assert {row.key for row in a.windows("train")} == {row.key for row in b.windows("train")}
    # Rank/batch reordering leaves per-window keys unchanged; changing the
    # experiment stream order intentionally invalidates its resume cursor.
    with pytest.raises(ValueError, match="fingerprint"):
        b.restore_cursor(asdict(a.cursor("train")))


@pytest.mark.parametrize("change", ["source", "tokenizer", "tokens", "split", "length"])
def test_provenance_or_contract_change_invalidates_saved_cursor(change):
    docs = [document("a", [5, 6, 7, 8])]
    a = CampaignData(docs, tokenizer=TOKENIZER)
    tokenizer, length = TOKENIZER, 1024
    if change == "source":
        docs[0] = replace(docs[0], source=replace(SOURCE, sha256=sha("changed source")))
    elif change == "tokenizer":
        tokenizer = replace(TOKENIZER, sha256=sha("changed tokenizer"))
    elif change == "tokens":
        docs[0] = replace(docs[0], tokens=(5, 6, 7, 9))
    elif change == "split":
        docs[0] = replace(docs[0], split="dev")
    else:
        length = 512
    b = CampaignData(docs, tokenizer=tokenizer, length=length)
    with pytest.raises(ValueError, match="fingerprint"):
        b.restore_cursor(asdict(a.cursor("train")))


@pytest.mark.parametrize("change", ["raw", "tokens", "identity", "source"])
def test_duplicates_and_conflicting_sources_are_rejected(change):
    first = document("a", [5, 6])
    second = document("b", [7, 8], "dev")
    if change == "raw":
        second = replace(second, text_sha256=first.text_sha256)
    elif change == "tokens":
        second = replace(second, tokens=(5, 6, 50279))
    elif change == "identity":
        second = replace(second, document_id=first.document_id)
    else:
        second = replace(second, source=replace(SOURCE, sha256=sha("other source")))
    with pytest.raises(ValueError, match="Duplicate|Conflicting"):
        CampaignData([first, second], tokenizer=TOKENIZER)


def test_batch_padding_never_disguises_real_content_as_invalid_or_crosses_documents():
    data = CampaignData([document("contains-pad", [1, 2]), document("other", [4])], tokenizer=TOKENIZER)
    batch = data.batch(data.windows("train"), physical_batch_size=3, pad_to=3)
    assert batch.input_ids[0, 0] == data.pad_id
    assert batch.valid_mask[0, 0]
    assert int(batch.valid_mask.sum()) == 5
    assert int(batch.input_ids.numel() - batch.valid_mask.sum()) == 4
    assert torch.equal(batch.document_ids, torch.tensor([[0, 0, 0], [1, 1, -1], [-1, -1, -1]]))
    with pytest.raises(ValueError, match="discard"):
        data.batch(data.windows("train"), physical_batch_size=1)
    with pytest.raises(ValueError, match="padding width"):
        data.batch(data.windows("train"), pad_to=2)
    with pytest.raises(ValueError, match="provenance"):
        data.batch([replace(data.windows("train")[0], tokens=(11, 12, 50279))])


@pytest.mark.parametrize("bad", [-1, True, 0.5, 10000])
def test_resume_rejects_invalid_window_positions(bad):
    data = corpus()
    with pytest.raises(ValueError, match="bounds"):
        data.restore_cursor(asdict(DataCursor(data.manifest_sha256, "train", bad)))


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_update_and_partition_sizes_require_positive_integers(bad):
    data = corpus()
    with pytest.raises(ValueError, match="positive integer"):
        data.next_update(data.cursor("train"), bad)
    update = data.next_update(data.cursor("train"), 5)
    with pytest.raises(ValueError, match="positive integer"):
        partition_update(update, world_size=bad, physical_batch_size=1)
    with pytest.raises(ValueError, match="positive integer"):
        partition_update(update, world_size=1, physical_batch_size=bad)


def test_native_1024_windows_keep_every_tail_without_adding_eos_at_crops():
    data = CampaignData([document("long", range(1000, 3100)), document("tiny", [5])], tokenizer=TOKENIZER)
    rows = data.windows("train")
    assert [row.length for row in rows] == [1024, 1024, 55, 2]
    assert [row.start for row in rows] == [0, 1023, 2046, 0]
    counts = count_windows(rows)
    assert counts.new_unique_tokens == 2103
    assert counts.presented_tokens == 2105
    assert counts.ce_targets == counts.latent_pairs == 2101
    assert counts.kl_triples == 2097
    assert counts.omitted_boundary_kl_triples == 2
