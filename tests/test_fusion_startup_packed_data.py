"""CPU-only packed selection, real-boundary masks and immutable fixture checks."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from cdrm.pretrained.campaign_data import SourcePin, TokenizedDocument
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.document_policy import feedback_eligibility
from cdrm.pretrained.nextlat import build_nextlat_masks
from scripts.olmo_fusion_startup_data import StartupData, _digest, _tensor_pin, _noise_pins
from scripts.olmo_fusion_startup_long_probe import build_long_fixture, document_key
from scripts import olmo_fusion_startup_packed_data as packed


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    documents = []
    for source_index, source_name in enumerate(("c4", "common_crawl", "wiki")):
        source = SourcePin(source_name, "https://example.invalid/"+source_name, "fixed", str(source_index+1)*64)
        for index in range(8):
            values = list(range(1000+10000*source_index+700*index, 1620+10000*source_index+700*index))
            values[11] = 50279  # Literal EOS inside a true document is content.
            documents.append(TokenizedDocument(source, str(index), "dev",
                hashlib.sha256((source_name+str(index)).encode()).hexdigest(), tuple(values)+(50279,)))
        documents.append(TokenizedDocument(source, "train", "train", _digest([source_name, "train"]),
            tuple(range(100+40*source_index, 132+40*source_index))+(50279,)))
    startup = StartupData._from_documents(documents, provenance={"manifest_sha256": "b"*64},
        length=16, ce_per_update=16)
    isolated = CampaignRecipe("NF")
    short = startup.export_fresh_fixture(tmp_path/"short.json", recipe=isolated, width=4)
    long = build_long_fixture(documents, startup.manifest, short["metadata"],
        short_sha256=short["sha256"], recipe=isolated, width=4)
    recipe = replace(isolated, document_policy=packed.POLICY)
    kwargs = dict(short_sha256=short["sha256"], long_sha256=_digest(long), recipe=recipe, width=4)
    before = torch.get_rng_state().clone()
    metadata = packed.build_packed_fixture(documents, startup.manifest, short["metadata"], long, **kwargs)
    assert torch.equal(before, torch.get_rng_state())
    path = tmp_path/"packed.json"
    artifact = packed.publish_fixture(path, metadata)
    loader = dict(recipe=recipe, width=4, expected_training_manifest_sha256=startup.manifest_sha256,
        expected_prepared_manifest_sha256="b"*64, expected_short_fixture_sha256=short["sha256"],
        expected_long_fixture_sha256=_digest(long))
    return documents, metadata, path, artifact, loader


def test_selection_and_tokens_match_independent_filtered_source_stream(fixture):
    documents, metadata, _, _, _ = fixture
    excluded = set(metadata["excluded_short_document_keys"]+metadata["excluded_long_document_keys"])
    assert len(excluded) == 8
    eligible = [d for d in documents if d.split == "dev" and document_key(d) not in excluded]
    source_rows = {}
    for source in sorted({d.source.name for d in eligible}):
        source_docs = [d for d in eligible if d.source.name == source]
        tokens = [t for d in source_docs for t in d.tokens]
        doc_keys = [document_key(d) for d in source_docs for _ in d.tokens]
        for offset in range(0, len(tokens)-1023, 1024):
            keys = doc_keys[offset:offset+1024]
            if len(set(keys)) > 1:
                source_rows[source] = (offset//1024, tokens[offset:offset+1024], keys)
                break
    assert metadata["selection"]["selected_sources"] == sorted(source_rows)[:2]
    index_by_key = {d["key"]:d["document_index"] for d in metadata["documents"]}
    for record in metadata["records"]:
        chunk_index, tokens, keys = source_rows[record["source"]]
        assert record["chunk_index"] == chunk_index
        assert record["batch"]["input_ids"] == [tokens]
        assert record["batch"]["document_ids"] == [[index_by_key[k] for k in keys]]
        assert set(keys).isdisjoint(excluded)
        assert record["stream_start"] == chunk_index*1024


def test_masks_cross_document_ce_feedback_but_exclude_auxiliary_pairs(fixture):
    _, metadata, path, artifact, loader = fixture
    fixtures, loaded = packed.load_packed_fixture(path, expected_sha256=artifact["sha256"], **loader)
    assert loaded == metadata
    counts = dict.fromkeys(("ce", "latent", "kl"), 0)
    boundaries = internal_eos = 0
    for (batch,), (noise,) in fixtures:
        masks = build_nextlat_masks(batch, document_policy=packed.POLICY)
        for name, mask in masks.items(): counts[name] += int(mask.sum())
        boundary = batch.document_ids[:, :-1] != batch.document_ids[:, 1:]
        boundaries += int(boundary.sum())
        assert masks["ce"][boundary].all() and not masks["latent"][boundary].any()
        assert feedback_eligibility(batch.valid_mask, batch.document_ids, packed.POLICY)[boundary].all()
        assert not masks["kl"][(boundary[:, :-1] | boundary[:, 1:])].any()
        embedded = (batch.input_ids[:, :-1] == 50279) & ~boundary
        internal_eos += int(embedded.sum())
        assert masks["latent"][embedded].all()
        assert batch.input_ids.shape == (1, 1024) and batch.valid_mask.all()
        assert all(t.shape == (1, 1023, 4) for t in noise)
    assert internal_eos > 0 and boundaries > 0
    assert counts == metadata["counts"] and counts["ce"] == 2046
    assert counts["latent"] == 2046-boundaries
    assert metadata["accounting"]["cross_document_ce_targets"] == boundaries


def test_exact_roundtrip_rng_noise_and_relocation(fixture, tmp_path):
    _, metadata, path, artifact, loader = fixture
    before = torch.get_rng_state().clone()
    first, _ = packed.load_packed_fixture(path, expected_sha256=artifact["sha256"], **loader)
    relocated = tmp_path/"relocated.json"; relocated.write_bytes(path.read_bytes())
    second, _ = packed.load_packed_fixture(relocated, expected_sha256=artifact["sha256"], **loader)
    assert torch.equal(before, torch.get_rng_state())
    for ((a,), (x,)), ((b,), (y,)) in zip(first, second):
        assert all(torch.equal(value, vars(b)[name]) for name, value in vars(a).items())
        assert _noise_pins(x) == _noise_pins(y)
    assert packed.publish_fixture(path, metadata) == artifact


@pytest.mark.parametrize("mutation", ["mask", "documents", "eos", "noise", "counts", "selection", "exclusion", "split"])
def test_rehashed_internal_contract_corruption_rejected(fixture, mutation):
    _, _, path, _, loader = fixture
    data = json.loads(path.read_text()); row = data["records"][0]
    if mutation == "mask":
        row["batch"]["ce_mask"][0][0] = True
        row["batch_pins"]["ce_mask"] = _tensor_pin(torch.tensor(row["batch"]["ce_mask"]))
    elif mutation == "documents":
        row["batch"]["document_ids"][0][0] += 1
        row["batch_pins"]["document_ids"] = _tensor_pin(torch.tensor(row["batch"]["document_ids"]))
    elif mutation == "eos":
        segment = next(s for s in row["segments"] if s["completes_document"])
        start, length = segment["chunk_offset"], segment["length"]
        row["batch"]["input_ids"][0][start+length-1] = 42
        segment["slice_tokens_sha256"] = _digest(row["batch"]["input_ids"][0][start:start+length])
        row["batch_pins"]["input_ids"] = _tensor_pin(torch.tensor(row["batch"]["input_ids"]))
    elif mutation == "noise": row["noise_pins"][0]["sha256"] = "0"*64
    elif mutation == "counts": data["counts"]["ce"] += 1
    elif mutation == "selection": data["selection"]["selected_sources"].reverse()
    elif mutation == "exclusion": data["excluded_long_document_keys"][0] = data["documents"][0]["key"]
    else: data["documents"][0]["split"] = "train"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        packed.load_packed_fixture(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **loader)


def test_wrong_authority_mode_noise_or_file_rejected(fixture, tmp_path):
    _, _, path, artifact, loader = fixture
    for key, value in (("recipe", replace(loader["recipe"], document_policy="isolated-v1")),
                       ("recipe", replace(loader["recipe"], jitter_seed=17)),
                       ("expected_training_manifest_sha256", "c"*64), ("width", 8)):
        with pytest.raises(ValueError):
            packed.load_packed_fixture(path, expected_sha256=artifact["sha256"], **{**loader, key:value})
    alias = tmp_path/"symlink.json"; alias.symlink_to(path)
    with pytest.raises(ValueError): packed.load_packed_fixture(alias, expected_sha256=artifact["sha256"], **loader)
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError): packed.load_packed_fixture(path, expected_sha256=artifact["sha256"], **loader)


def test_incomplete_corpus_or_exclusions_cannot_build(fixture):
    _, metadata, _, _, _ = fixture
    with pytest.raises(ValueError, match="two sources"):
        packed._select([d for d in metadata["documents"] if d["source"]["name"] == "c4"], "b"*64, set())
    with pytest.raises(ValueError, match="identity/exclusion"):
        packed._select(metadata["documents"], "b"*64, {metadata["documents"][0]["key"]})
    with pytest.raises(ValueError, match="metadata contract"):
        packed._select(list(reversed(metadata["documents"])), "b"*64, set())
