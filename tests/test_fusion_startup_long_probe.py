"""CPU fixture authority, source coverage and actual T128 gradient controls."""
import copy
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_data import SourcePin, TokenizedDocument
from cdrm.pretrained.campaign_recipe import CampaignRecipe, build_campaign_model
from cdrm.pretrained.nextlat import build_nextlat_masks
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_fusion_startup_long_probe as long
from scripts import olmo_fusion_startup_probe as probe
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_recurrence_precision import FP32, fixture_pins, state_pins
from scripts.olmo_fusion_startup_data import StartupData, _digest, _tensor_pin


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def documents():
    sources = [SourcePin(name, "https://example.invalid/"+name, "pinned", "a"*64) for name in ("alpha", "beta", "gamma", "delta")]
    result = []
    for index in range(18):
        tokens = [2+(position*17+index*3)%57 for position in range(140+index)]
        tokens[70] = 60  # Internal EOS is an actual token, not a split boundary.
        tokens.append(60)
        result.append(TokenizedDocument(sources[index%4], f"doc-{index}", "train" if index < 2 else "dev",
            hashlib.sha256(str(index).encode()).hexdigest(), tuple(tokens)))
    return result


def fixture_metadata():
    docs = documents()
    data = StartupData._from_documents(docs, provenance={"manifest_sha256": "a"*64},
        length=128, ce_per_update=127, physical_batch_size=2, eos_id=60, vocab_size=61)
    short = {"training_manifest_sha256": data.manifest_sha256,
        "prepared_manifest_sha256": "a"*64, "documents": data.manifest["fresh_fixture"]["documents"]}
    recipe = CampaignRecipe("NF", sequence_length=128, rt_layers=(0, 1))
    metadata = long.build_long_fixture(docs, data.manifest, short, short_sha256="b"*64, recipe=recipe, width=32)
    return metadata, docs, data, short, recipe


def export(tmp_path, metadata, name="long.json"):
    path = tmp_path/name
    path.write_text(json.dumps(metadata))
    return path


def load(path, metadata, recipe, **kwargs):
    return long.load_long_fixture(path, expected_sha256=sha256_file(path), recipe=recipe, width=32,
        expected_training_manifest_sha256=metadata["training_manifest_sha256"],
        expected_prepared_manifest_sha256="a"*64, expected_short_fixture_sha256="b"*64, **kwargs)


def test_selection_is_deterministic_source_aware_and_disjoint_with_real_tokens():
    metadata, docs, data, short, recipe = fixture_metadata()
    before = torch.get_rng_state().clone()
    repeated = long.build_long_fixture(reversed(docs), data.manifest, short, short_sha256="b"*64, recipe=recipe, width=32)
    assert repeated == metadata and torch.equal(before, torch.get_rng_state())
    assert [row["source"]["name"] for row in metadata["documents"]] == ["alpha", "beta", "delta", "gamma"]
    chosen = {row["key"] for row in metadata["documents"]}
    assert not chosen & {row["key"] for row in short["documents"]}
    assert not chosen & {row["key"] for row in data.manifest["documents"] if row["split"] == "train"}
    actual_docs = {long.document_key(doc): doc for doc in docs}
    for index, row in enumerate(metadata["documents"]):
        tokens = metadata["records"][index//2]["batch"]["input_ids"][index%2]
        assert tokens == list(actual_docs[row["key"]].tokens[:128])
        assert tokens[70] == 60 and tokens[-1] != 60
    with pytest.raises(ValueError, match="complete pinned corpus"):
        long.build_long_fixture(docs[:-1], data.manifest, short, short_sha256="b"*64, recipe=recipe, width=32)


def test_export_roundtrip_counts_noise_rng_and_portable_hashes(tmp_path):
    metadata, _, _, _, recipe = fixture_metadata()
    artifact = long._publish_fixture(tmp_path/"long.json", metadata)
    before = torch.get_rng_state().clone()
    fixtures, loaded = load(artifact["path"], metadata, recipe)
    assert loaded == metadata and artifact["size_bytes"] < 128*1024
    assert torch.equal(before, torch.get_rng_state())
    totals = dict.fromkeys(long.COUNTS, 0)
    for (batch,), (noise,) in fixtures:
        assert tuple(batch.input_ids.shape) == (2, 128) and batch.valid_mask.all()
        assert len(noise) == 3 and all(tuple(t.shape) == (2, 127, 32) for t in noise)
        for name, mask in build_nextlat_masks(batch).items():
            totals[name] += int(mask.sum())
    assert totals == long.COUNTS
    fixtures2, _ = load(artifact["path"], metadata, recipe)
    assert fixture_pins(fixtures) == fixture_pins(fixtures2)
    with pytest.raises(FileExistsError):
        long._publish_fixture(artifact["path"], metadata)
    with pytest.raises(ValueError, match="contract"):
        load(artifact["path"], metadata, replace(recipe, jitter_seed=recipe.jitter_seed+1))


@pytest.mark.parametrize("corruption", ["schema", "manifest", "prepared", "short", "counts", "short_overlap",
    "duplicate_document", "training_split", "wrong_length", "token", "boolean_token", "mask", "docid", "noise", "noise_keys"])
def test_loader_rejects_semantic_and_pin_mutations_before_execution(tmp_path, corruption):
    metadata, _, _, _, recipe = fixture_metadata()
    changed = copy.deepcopy(metadata)
    row = changed["records"][0]
    if corruption == "schema":
        changed["schema"] = "different"
    elif corruption == "manifest":
        changed["training_manifest_sha256"] = "c"*64
    elif corruption == "prepared":
        changed["prepared_manifest_sha256"] = "c"*64
    elif corruption == "short":
        changed["short_fixture_sha256"] = "c"*64
    elif corruption == "counts":
        changed["counts"]["ce"] -= 1
    elif corruption == "short_overlap":
        changed["excluded_short_document_keys"][0] = changed["documents"][0]["key"]
    elif corruption == "duplicate_document":
        changed["documents"][1]["key"] = changed["documents"][0]["key"]
    elif corruption == "training_split":
        changed["documents"][0]["split"] = "train"
    elif corruption == "wrong_length":
        changed["documents"][0]["full_token_count"] = 127
    elif corruption == "token":
        row["batch"]["input_ids"][0][4] += 1
    elif corruption == "boolean_token":
        row["batch"]["input_ids"][0][4] = True
    elif corruption in ("mask", "docid"):
        key = "ce_mask" if corruption == "mask" else "document_ids"
        row["batch"][key][0][4] = False if corruption == "mask" else row["batch"][key][0][4]+1
        # Make the low-level pin internally consistent to test semantic guards.
        row["batch_pins"][key] = _tensor_pin(torch.tensor(row["batch"][key], dtype=torch.bool if corruption == "mask" else torch.long))
    elif corruption == "noise":
        row["noise_pins"][0]["sha256"] = "c"*64
    elif corruption == "noise_keys":
        row["noise_keys"][0] = "different"
    path = export(tmp_path, changed)
    with pytest.raises(ValueError):
        load(path, metadata, recipe)


def test_actual_t128_tiny_full_gradients_repeat_exactly_and_first_pass_ignores_fusion_change(tmp_path, monkeypatch):
    metadata, _, _, _, recipe = fixture_metadata()
    fixtures, _ = load(export(tmp_path, metadata), metadata, recipe)
    monkeypatch.setattr(probe, "rng_snapshot", lambda: torch.get_rng_state().clone())
    monkeypatch.setattr(probe, "rng_unchanged", lambda old: torch.equal(old, torch.get_rng_state()))
    torch.manual_seed(20260929)
    base = OLMoTiledRTForCausalLM(replace(OLMoConfig.tiny(), max_context_length=128),
        attention_backend="math", attention_precision="mixed", ordinary_activation_checkpointing=True,
        tile_backend="eager", backward_tile_backend="eager")
    model = build_campaign_model(base, recipe).train()
    flags = {name: getattr(base, name) for name in RUNTIME_FLAGS}
    initial, inputs = state_pins(model), fixture_pins(fixtures)
    first, gradients, states = probe.measure_case(model, recipe, fixtures, path=FP32, original_flags=flags)
    repeat, _, _ = probe.measure_case(model, recipe, fixtures, path=FP32, original_flags=flags,
        reference_gradients=gradients, reference_forward=states)
    assert first["metrics"]["counts"] == long.COUNTS
    assert repeat["gradients_vs_fp32"]["geometry"]["all"]["difference_norm"] == 0
    assert first["forward_fingerprints"] == repeat["forward_fingerprints"]
    assert state_pins(model) == initial and fixture_pins(fixtures) == inputs
    assert first["gradients"]["groups"]["backbone"]["norm"] > 0
    assert first["gradients"]["groups"]["fusion"]["norm"] > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.backbone.fusion.token_gate.weight.add_(.01)
    changed, _, _ = probe.measure_case(model, recipe, fixtures, path=FP32, original_flags=flags)
    from scripts.olmo_campaign_crossed_precision import first_pass_identity
    assert first_pass_identity(changed["forward_fingerprints"], first)
    assert changed["forward_fingerprints"] != first["forward_fingerprints"]


def test_cold_reference_is_pinned_and_requires_matching_fixture_sources_and_health(tmp_path):
    report = {"schema": "olmo-fusion-startup-long-probe-v1", "state": "cold", "status": "passed_operational_diagnostic",
        "passed": True, "fixture_sha256": "a"*64, "optimizer_updates": 0, "aggregate_backwards": 2, "physical_backwards": 4,
        "determinism": {"deterministic_algorithms": True}, "integrity": {"fixed": True}, "pair_integrity": {"fixed": True},
        "sources": {"source.py": "b"*64}, "rows": [{"path": p, "objective": "ce", "passed": True, "health": {"finite": True}}
                                                for p in (long.FP32, long.BF16)]}
    path = export(tmp_path, report, name="report.json")
    loaded, endpoints = long.load_cold_report(path, sha256_file(path), report["sources"], fixture_sha256="a"*64)
    assert loaded == report and set(endpoints) == {long.FP32, long.BF16}
    with pytest.raises(ValueError):
        long.load_cold_report(path, sha256_file(path), report["sources"], fixture_sha256="c"*64)
    with pytest.raises(ValueError, match="sources"):
        long.load_cold_report(path, sha256_file(path), {"source.py": "c"*64}, fixture_sha256="a"*64)
    report["rows"][1]["health"] = {}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="endpoint"):
        long.load_cold_report(path, sha256_file(path), report["sources"], fixture_sha256="a"*64)
