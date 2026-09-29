"""Two held-out packed T1024 rows for bounded NF precision observations.

This exports existing token bytes, not a production dataset or training index.
Eight prior dev documents are removed before per-source stream concatenation.
Selection depends only on source order and true document boundaries.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import torch

from cdrm.pretrained.campaign_data import TokenizedDocument
from cdrm.pretrained.campaign_recipe import CampaignRecipe, feedback_noise_for_rows
from cdrm.pretrained.document_shards import iter_documents, verify_document_shards
from cdrm.pretrained.nextlat import NextLatBatch, build_nextlat_masks
from scripts.olmo_fusion_startup_data import (
    DEFAULT_ROOT, DEFAULT_MANIFEST_SHA256, _digest, _json, _file_sha,
    _file_signatures, _tensor_pin, _noise_pins, load_fresh_fixture,
)
from scripts.olmo_fusion_startup_long_probe import (
    TRAINING_MANIFEST_SHA, SHORT_FIXTURE_SHA, document_key, load_long_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "olmo-fusion-startup-packed-fixture-v1"
POLICY = "continuous-stream-v1"
LONG_FIXTURE_SHA = "830920f60c687f667baee7f7d6f137b521b22a35604f02e2a2e3b038586e55f9"
LENGTH = 1024
FIELDS = {"input_ids", "valid_mask", "document_ids", "ce_mask", "latent_mask", "kl_mask"}
SELECTION = ("exclude eight prior dev documents; concatenate remaining complete dev documents "
    "in canonical corpus order separately per source; choose the first full boundary-containing "
    "chunk per source; select the first two eligible source names alphabetically")
MAX_BYTES = 1024*1024


def _sha_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _noise_contract(recipe, width):
    mode = recipe.mode()
    if (type(width) is not int or width < 1 or recipe.arm != "NF" or not recipe.nextlat
            or recipe.document_policy != POLICY or recipe.sequence_length != LENGTH
            or mode.num_passes != 4 or mode.beta != 1.0 or mode.feedback_jitter != .02
            or mode.rt_mode.selected_layers or not mode.enabled):
        raise ValueError("Packed fixture requires NF K4 beta1 jitter.02 T1024 continuous stream")
    return {"jitter_seed": recipe.jitter_seed, "feedback_jitter": .02, "fbt_passes": 4,
            "logical_update": 0, "width": width, "dtype": "torch.float32"}


def _validate_documents(documents, exclusions):
    seen, identities, indices = set(), set(), set()
    last_ordinal = -1
    for doc in documents:
        source = doc.get("source", {})
        identity = (source.get("name"), doc.get("document_id"))
        if (set(source) != {"name", "uri", "revision", "sha256"}
                or any(not isinstance(v, str) or not v for v in source.values())
                or not _sha_string(source["sha256"]) or doc.get("split") != "dev"
                or type(doc.get("document_index")) is not int or doc["document_index"] < 0
                or type(doc.get("canonical_corpus_ordinal")) is not int
                or doc["canonical_corpus_ordinal"] <= last_ordinal
                or type(doc.get("full_token_count")) is not int or doc["full_token_count"] < 1
                or not isinstance(doc.get("document_id"), str)
                or any(not _sha_string(doc.get(k)) for k in
                       ("key", "text_sha256", "content_sha256", "full_tokens_sha256"))):
            raise ValueError("Packed document metadata contract differs")
        expected_key = _digest({"source": source, "document_id": doc["document_id"],
            "text_sha256": doc["text_sha256"], "content_sha256": doc["content_sha256"], "split": "dev"})
        if (doc["key"] != expected_key or doc["key"] in seen or identity in identities
                or doc["document_index"] in indices or doc["key"] in exclusions):
            raise ValueError("Packed document identity/exclusion differs")
        seen.add(doc["key"]); identities.add(identity); indices.add(doc["document_index"])
        last_ordinal = doc["canonical_corpus_ordinal"]


def _select(documents, prepared_sha, exclusions):
    """Replay selection from authenticated lengths/identities without token I/O."""
    _validate_documents(documents, exclusions)
    streams = {}
    for doc in documents:
        streams.setdefault(doc["source"]["name"], []).append(doc)
    candidates = []
    lengths = {}
    for source, docs in sorted(streams.items()):
        spans, offset = [], 0
        for doc in docs:
            spans.append((doc, offset, offset+doc["full_token_count"]))
            offset += doc["full_token_count"]
        lengths[source] = offset
        for chunk_index in range(offset//LENGTH):
            start, end = chunk_index*LENGTH, (chunk_index+1)*LENGTH
            segments = []
            for doc, left, right in spans:
                if left < end and right > start:
                    a, b = max(start, left), min(end, right)
                    segments.append({"document_key": doc["key"], "document_index": doc["document_index"],
                        "document_offset": a-left, "chunk_offset": a-start, "length": b-a,
                        "completes_document": b == right})
            if len(segments) < 2:
                continue
            key = _digest([SCHEMA, prepared_sha, sorted(exclusions), source, chunk_index, segments])
            candidates.append({"source": source, "chunk_index": chunk_index,
                "stream_start": start, "stream_total_tokens": offset, "key": key, "segments": segments})
            break
    if len(candidates) < 2:
        raise ValueError("Need two sources with a full chunk containing an actual document boundary")
    return candidates[:2], {"policy": SELECTION, "eligible_source_names": [c["source"] for c in candidates],
        "selected_sources": [c["source"] for c in candidates[:2]], "source_stream_tokens": lengths,
        "remaining_dev_documents": len(documents)}


def _counts(segments):
    latent = sum(max(segment["length"]-1, 0) for segment in segments)
    kl = sum(max(segment["length"]-2, 0) for segment in segments)
    return {"ce": LENGTH-1, "latent": latent, "kl": kl}


def build_packed_fixture(documents, training_manifest, short_metadata, long_metadata, *,
                         short_sha256, long_sha256, recipe, width):
    """Pure CPU constructor over a completely verified prepared-document stream."""
    noise_contract = _noise_contract(recipe, width)
    training_sha = _digest(training_manifest)
    prepared_sha = training_manifest["provenance"]["manifest_sha256"]
    if (training_manifest.get("schema") != "olmo-fusion-startup-data-v1"
            or any(meta.get("training_manifest_sha256") != training_sha
                   or meta.get("prepared_manifest_sha256") != prepared_sha
                   for meta in (short_metadata, long_metadata))
            or long_metadata.get("short_fixture_sha256") != short_sha256):
        raise ValueError("Packed fixture source authorities disagree")
    short = {d["key"] for d in short_metadata["documents"]}
    long = {d["key"] for d in long_metadata["documents"]}
    if (len(short) != 4 or len(long) != 4 or short & long
            or set(long_metadata.get("excluded_short_document_keys", [])) != short
            or short != {d["key"] for d in training_manifest["fresh_fixture"]["documents"]}):
        raise ValueError("Require exactly eight distinct prior dev document exclusions")
    excluded = short | long
    inventory = {row["key"]: (i, row) for i, row in enumerate(training_manifest["documents"])}
    if len(inventory) != len(training_manifest["documents"]):
        raise ValueError("Repeated training-manifest document identity")
    observed, eligible, tokens = set(), [], {}
    for corpus_ordinal, document in enumerate(documents):
        if not isinstance(document, TokenizedDocument):
            raise TypeError("Require verified TokenizedDocument records")
        key = document_key(document)
        if key not in inventory or key in observed:
            raise ValueError("Prepared identity differs from complete pinned inventory")
        observed.add(key)
        index, expected = inventory[key]
        record = {"key": key, "source": document.source.name, "split": document.split,
            "tokens": len(document.tokens), "content_sha256": _digest(document.tokens[:-1])}
        if (record != expected or document.tokens[-1] != training_manifest["eos_id"]
                or any(type(t) is not int or not 0 <= t < training_manifest["vocab_size"] for t in document.tokens)):
            raise ValueError("Prepared content, split, length or actual EOS differs")
        if document.split == "dev" and key not in excluded:
            eligible.append({"key": key, "document_index": index, "canonical_corpus_ordinal": corpus_ordinal,
                "source": asdict(document.source),
                "document_id": document.document_id, "split": "dev", "text_sha256": document.text_sha256,
                "content_sha256": record["content_sha256"], "full_token_count": len(document.tokens),
                "full_tokens_sha256": _digest(document.tokens)})
            tokens[key] = document.tokens
    if observed != set(inventory) or any(inventory[k][1]["split"] != "dev" for k in excluded):
        raise ValueError("Incomplete source scan or non-dev exclusion")
    selected, selection = _select(eligible, prepared_sha, excluded)
    records, totals = [], dict.fromkeys(("ce", "latent", "kl"), 0)
    for selected_row in selected:
        row = {**selected_row, "segments": [dict(s) for s in selected_row["segments"]]}
        ids, doc_ids = [], []
        for segment in row["segments"]:
            values = tokens[segment["document_key"]][segment["document_offset"]:segment["document_offset"]+segment["length"]]
            segment["slice_tokens_sha256"] = _digest(values)
            ids.extend(values); doc_ids.extend([segment["document_index"]]*len(values))
        values = torch.tensor([ids], dtype=torch.long)
        valid = torch.ones_like(values, dtype=torch.bool)
        targets = valid.clone(); targets[:, 0] = False
        batch = NextLatBatch(values, valid, torch.tensor([doc_ids]), targets, targets.clone(), targets.clone())
        noise = feedback_noise_for_rows(recipe, [row["key"]], logical_update=0,
            sequence_length=LENGTH, width=width, physical_batch_size=1, device="cpu")
        counts = _counts(row["segments"])
        actual = {name: int(mask.sum()) for name, mask in build_nextlat_masks(batch, document_policy=POLICY).items()}
        if actual != counts:
            raise AssertionError("Independent segment counts differ from model masks")
        for name, count in counts.items(): totals[name] += count
        row.update(batch={name: value.tolist() for name, value in vars(batch).items()},
            batch_pins={name: _tensor_pin(value) for name, value in vars(batch).items()},
            noise_keys=[row["key"]], noise_pins=_noise_pins(noise), counts=counts)
        records.append(row)
    train_keys = sorted(k for k, (_, record) in inventory.items() if record["split"] == "train")
    return {"schema": SCHEMA, "training_manifest_sha256": training_sha,
        "prepared_manifest_sha256": prepared_sha, "short_fixture_sha256": short_sha256,
        "long_fixture_sha256": long_sha256, "excluded_short_document_keys": sorted(short),
        "excluded_long_document_keys": sorted(long), "training_document_count": len(train_keys),
        "training_document_keys_sha256": _digest(train_keys), "selection": selection,
        "documents": eligible, "records": records, "document_policy": POLICY,
        "length": LENGTH, "physical_batch_size": 1, "input_tokens": 2048, "microbatches": 2,
        "padding_tokens": 0, "pad_id": training_manifest["pad_id"],
        "vocab_size": training_manifest["vocab_size"], "eos_id": training_manifest["eos_id"],
        "counts": totals, "noise_contract": noise_contract,
        "accounting": {"cross_document_ce_targets": 2046-totals["latent"],
            "excluded_boundary_latent_pairs": 2046-totals["latent"],
            "excluded_boundary_kl_triples": 2044-totals["kl"],
            "document_segments": sum(len(r["segments"]) for r in records),
            "unique_documents": len({s["document_key"] for r in records for s in r["segments"]}),
            "document_completions": sum(s["completes_document"] for r in records for s in r["segments"])},
        "qualification": "Held-out seven-source coverage corpus, not a production mixture; no training/cursor commit; source-selected numerical fixture, not quality evaluation"}


def publish_fixture(path, metadata):
    path = Path(path)
    raw = (_json(metadata)+"\n").encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("Packed fixture exceeds bounded JSON budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("Existing packed fixture differs")
    else:
        temporary = path.with_name(path.name+".tmp")
        with temporary.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "metadata": metadata}


def load_packed_fixture(path, *, expected_sha256, recipe, width,
                        expected_training_manifest_sha256=TRAINING_MANIFEST_SHA,
                        expected_prepared_manifest_sha256=DEFAULT_MANIFEST_SHA256,
                        expected_short_fixture_sha256=SHORT_FIXTURE_SHA,
                        expected_long_fixture_sha256=LONG_FIXTURE_SHA):
    """Strict CPU reconstruction; expected whole-file SHA authenticates provenance."""
    path = Path(path)
    if (not _sha_string(expected_sha256) or path.is_symlink() or not path.is_file()
            or path.stat().st_size > MAX_BYTES):
        raise ValueError("Require bounded regular packed fixture and explicit SHA256")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Packed fixture bytes differ from pin")
    data = json.loads(raw)
    authorities = {"training_manifest_sha256": expected_training_manifest_sha256,
        "prepared_manifest_sha256": expected_prepared_manifest_sha256,
        "short_fixture_sha256": expected_short_fixture_sha256, "long_fixture_sha256": expected_long_fixture_sha256}
    if (data.get("schema") != SCHEMA or any(data.get(k) != v or not _sha_string(v) for k, v in authorities.items())
            or data.get("noise_contract") != _noise_contract(recipe, width)
            or data.get("document_policy") != POLICY or data.get("length") != LENGTH
            or data.get("physical_batch_size") != 1 or data.get("input_tokens") != 2048
            or data.get("microbatches") != 2 or data.get("padding_tokens") != 0
            or data.get("pad_id") != 1 or type(data.get("vocab_size")) is not int
            or type(data.get("eos_id")) is not int or not 1 < data["eos_id"] < data["vocab_size"]
            or len(data.get("records", [])) != 2):
        raise ValueError("Packed fixture contract differs")
    short, long = data.get("excluded_short_document_keys", []), data.get("excluded_long_document_keys", [])
    if (len(short) != 4 or len(long) != 4 or len(set(short+long)) != 8
            or any(not _sha_string(k) for k in short+long)):
        raise ValueError("Packed exclusion inventory differs")
    selected, selection = _select(data["documents"], data["prepared_manifest_sha256"], set(short+long))
    if data.get("selection") != selection:
        raise ValueError("Packed source-aware selection differs")
    docs = {d["key"]: d for d in data["documents"]}
    fixtures, counts = [], dict.fromkeys(("ce", "latent", "kl"), 0)
    for wanted, record in zip(selected, data["records"]):
        clean = {k: record.get(k) for k in wanted if k != "segments"}
        clean["segments"] = [{k: v for k, v in s.items() if k != "slice_tokens_sha256"} for s in record.get("segments", [])]
        if clean != wanted or set(record.get("batch", {})) != FIELDS or set(record.get("batch_pins", {})) != FIELDS:
            raise ValueError("Packed selected chunk, segment or tensor inventory differs")
        tensors = {}
        for name in FIELDS:
            scalar = int if name in ("input_ids", "document_ids") else bool
            values = record["batch"][name]
            if (not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], list)
                    or len(values[0]) != LENGTH or any(type(v) is not scalar for v in values[0])):
                raise ValueError("Packed tensor shape/scalar type differs")
            tensors[name] = torch.tensor(values, dtype=torch.long if scalar is int else torch.bool)
            if _tensor_pin(tensors[name]) != record["batch_pins"][name]:
                raise ValueError("Packed tensor byte pin differs")
        ids, true_ids = record["batch"]["input_ids"][0], []
        for segment in record["segments"]:
            doc = docs[segment["document_key"]]
            start, size = segment["chunk_offset"], segment["length"]
            values = ids[start:start+size]
            if (_digest(values) != segment.get("slice_tokens_sha256")
                    or (segment["completes_document"] and values[-1] != data["eos_id"])):
                raise ValueError("Packed actual slice or terminal EOS differs")
            if segment["document_offset"] == 0 and size == doc["full_token_count"]:
                if _digest(values) != doc["full_tokens_sha256"] or _digest(values[:-1]) != doc["content_sha256"]:
                    raise ValueError("Packed complete document token pins differ")
            true_ids.extend([segment["document_index"]]*size)
        if (any(not 0 <= token < data["vocab_size"] for token in ids)
                or record["batch"]["document_ids"][0] != true_ids
                or record["batch"]["valid_mask"][0] != [True]*LENGTH
                or any(record["batch"][name][0] != [False]+[True]*(LENGTH-1)
                       for name in ("ce_mask", "latent_mask", "kl_mask"))):
            raise ValueError("Packed true document alignment or target masks differ")
        batch = NextLatBatch(**tensors)
        row_counts = {name: int(mask.sum()) for name, mask in build_nextlat_masks(batch, document_policy=POLICY).items()}
        if row_counts != _counts(record["segments"]) or row_counts != record.get("counts"):
            raise ValueError("Packed objective counts differ")
        for name, value in row_counts.items(): counts[name] += value
        if record.get("noise_keys") != [wanted["key"]]:
            raise ValueError("Packed noise occurrence differs")
        noise = feedback_noise_for_rows(recipe, [wanted["key"]], logical_update=0,
            sequence_length=LENGTH, width=width, physical_batch_size=1, device="cpu")
        if _noise_pins(noise) != record.get("noise_pins"):
            raise ValueError("Packed noise bytes differ")
        fixtures.append(((batch,), (noise,)))
    accounting = {"cross_document_ce_targets": 2046-counts["latent"],
        "excluded_boundary_latent_pairs": 2046-counts["latent"],
        "excluded_boundary_kl_triples": 2044-counts["kl"],
        "document_segments": sum(len(r["segments"]) for r in data["records"]),
        "unique_documents": len({s["document_key"] for r in data["records"] for s in r["segments"]}),
        "document_completions": sum(s["completes_document"] for r in data["records"] for s in r["segments"])}
    if counts != data.get("counts") or counts["ce"] != 2046 or data.get("accounting") != accounting:
        raise ValueError("Packed aggregate counts/accounting differ")
    if path.read_bytes() != raw:
        raise ValueError("Packed fixture changed during load")
    return fixtures, data


def prepare_packed_fixture(output_path, *, corpus=DEFAULT_ROOT,
        training_manifest_path=ROOT/".runtime/olmo-fusion-startup/data-01/manifest.json",
        short_fixture_path=ROOT/".runtime/olmo-fusion-startup/data-01/fresh_fixture.json",
        long_fixture_path=ROOT/".runtime/olmo-fusion-startup/long-data-01/long_fixture.json", recipe, width):
    corpus = Path(corpus)
    before = _file_signatures(corpus)
    if _file_sha(corpus/"manifest.json") != DEFAULT_MANIFEST_SHA256 or _file_sha(training_manifest_path) != TRAINING_MANIFEST_SHA:
        raise ValueError("Prepared corpus/startup manifest differs from fixed authority")
    summary = verify_document_shards(corpus)
    if not summary["completed"]:
        raise ValueError("Prepared corpus is incomplete")
    training = json.loads(Path(training_manifest_path).read_text())
    isolated = replace(recipe, document_policy="isolated-v1")
    _, short = load_fresh_fixture(short_fixture_path, expected_sha256=SHORT_FIXTURE_SHA, recipe=isolated, width=width)
    _, long = load_long_fixture(long_fixture_path, expected_sha256=LONG_FIXTURE_SHA, recipe=isolated, width=width)
    data = build_packed_fixture(iter_documents(corpus, verify=False), training, short, long,
        short_sha256=SHORT_FIXTURE_SHA, long_sha256=LONG_FIXTURE_SHA, recipe=recipe, width=width)
    if before != _file_signatures(corpus) or _file_sha(training_manifest_path) != TRAINING_MANIFEST_SHA:
        raise ValueError("Corpus/manifest changed during packed preparation")
    artifact = publish_fixture(output_path, data)
    load_packed_fixture(output_path, expected_sha256=artifact["sha256"], recipe=recipe, width=width)
    return artifact


def source_hashes():
    names = ("scripts/olmo_fusion_startup_packed_data.py", "tests/test_fusion_startup_packed_data.py",
        "docs/reports/olmo-fusion-startup/packed-probe-protocol.md",
        "scripts/olmo_fusion_startup_data.py", "scripts/olmo_fusion_startup_long_probe.py",
        "cdrm/pretrained/document_shards.py", "cdrm/pretrained/campaign_recipe.py",
        "cdrm/pretrained/nextlat.py", "cdrm/pretrained/document_policy.py")
    return {name: _file_sha(ROOT/name) for name in names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = prepare_packed_fixture(args.output, recipe=CampaignRecipe("NF", document_policy=POLICY), width=2048)
    print(json.dumps({k:v for k,v in artifact.items() if k != "metadata"}, sort_keys=True))
