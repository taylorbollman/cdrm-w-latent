"""Tiny CPU checks for explicit historical→current weights-only import."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import CHECKPOINT_SCHEMA, parameter_layout
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_campaign_adapted_import as importer
from scripts.olmo_campaign_ddp_probe import construct
from scripts.olmo_lm_common import state_digests
from scripts.olmo_o5b_common import ObservedFBTLM
from scripts.olmo_o5c_common import FUSION_NAMES, freeze_native, frozen_state_digests


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def endpoint(tmp_path):
    torch.manual_seed(712)
    old = ObservedFBTLM(OLMoFBT(OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math")),
        NextLatConfig(32, vocab_chunk_size=128), enabled=False, gamma=1)
    freeze_native(old)
    with torch.no_grad():
        old.backbone.token_embeddings.weight.mul_(1.3)
        old.backbone.fusion.state_proj.weight.add_(.02)
        # A distinct saved value proves import does not recalibrate from wte.
        old.backbone.fusion.output_scale.fill_(.123)
    frozen = frozen_state_digests(old)
    counters = {"optimizer_updates": 1, "input_tokens": 8, "ce_positions": 6, "documents": 2,
                "microbatches": 1, "latent_pairs": 0, "kl_triples": 0}
    construction = {"arm": "fbt", "model_config": old.backbone.config.to_dict(),
        "fusion_config": old.backbone.fusion_config.to_dict(), "nextlat_config": old.config.to_dict(),
        "fusion_output_scale": float(old.backbone.fusion.output_scale), "attention_backend": "math",
        "attention_precision": "mixed", "nextlat_enabled": False, "rt_layers": [], "num_passes": 2, "gamma": 1}
    trainable = [{"name": name, "shape": list(dict(old.named_parameters())[name].shape),
                  "numel": dict(old.named_parameters())[name].numel()} for name in FUSION_NAMES]
    plan = {"arm": "mixed", "total_updates": 1, "ce_per_update": 6,
            "batch_ce_prefix": [0, 6], "batch_token_prefix": [0, 8], "batch_row_prefix": [0, 2]}
    config = {"schema": "olmo-o5c-pilot-config-v1", "arm": "mixed", "storage_prefix": "gs://fixture",
        "native_backbone_frozen": True, "nextlat_enabled": False, "rt_layers": [], "beta": 1.,
        "num_passes": 2, "gamma": 1., "prefix_mixin": False, "hidden_jitter": 0.,
        "model_config": old.backbone.config.to_dict(), "attention_backend": "math", "physical_batch_size": 2,
        "frozen_state_initial": frozen, "trainable_layout": trainable, "source_hashes": {"fixture.py": "a"*64},
        "source_checkpoint_sha256": "b"*64, "data_manifest_sha256": "c"*64,
        "schedule": {"total_updates": 1, "total_ce": 6, "ce_per_update": 6, "arms": {"mixed": plan}}}
    source = {"checkpoint_sha256": "b"*64, "source_checkpoint_sha256": "b"*64,
              "code": config["source_hashes"], "data_manifest_sha256": "c"*64}
    cursor = {"next_update": 1, "bad_evals": 0}
    payload = {"schema": CHECKPOINT_SCHEMA, "model_type": "scripts.olmo_o5b_common.ObservedFBTLM",
        "model": old.state_dict(), "parameter_layout": parameter_layout(old),
        "module_training": {name: False for name, _ in old.named_modules()},
        "optimizer_ownership": [list(FUSION_NAMES)], "configuration": config,
        "source_fingerprint": source, "counters": counters, "data_cursor": cursor,
        "optimizer": {"ignored": True}, "scheduler": {"ignored": True}, "rng": torch.zeros(2, dtype=torch.uint8)}
    path = tmp_path/"endpoint.pt"
    torch.save(payload, path)
    record = {"schema": CHECKPOINT_SCHEMA, "path": str(path), "sha256": sha256_file(path),
              "size_bytes": path.stat().st_size, "optimizer_updates": 1, "input_tokens": 8, "ce_positions": 6}
    authority = importer.AdaptedImportAuthority("tiny-cpu-test", record, config, construction,
        source, counters, cursor, frozen, {"fixture.py": {"scope": "synthetic CPU authority"}})
    current, _, _, _, _ = construct(SimpleNamespace(scale="tiny", length=16), "NF", torch.device("cpu"))
    return current, old, path, payload, authority


def rehash(path, payload, authority):
    torch.save(payload, path)
    return replace(authority, checkpoint={**authority.checkpoint, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})


def test_weights_only_import_preserves_saved_scale_fresh_predictor_ties_and_trainable_identity(endpoint, monkeypatch):
    model, old, path, _, authority = endpoint
    predictor = state_digests(model.predictor)
    identities = {name: id(p) for name, p in model.named_parameters()}
    modes = {name: module.training for name, module in model.named_modules()}
    before_rng = torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("Importer must not construct/restore an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    original_load, loads = torch.load, []
    def observed_load(*args, **kwargs):
        loads.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", observed_load)
    provenance = importer.load_into_current(model, path, authority)
    assert loads == [{"map_location": "cpu", "weights_only": True}]
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(provenance["checks"].values())
    for name, value in old.state_dict().items():
        assert torch.equal(model.state_dict()[name], value)
    assert state_digests(model.predictor) == predictor
    assert {name: id(p) for name, p in model.named_parameters()} == identities
    assert {name: module.training for name, module in model.named_modules()} == modes
    assert all(p.requires_grad and p.dtype == torch.float32 and p.grad is None for p in model.parameters())
    assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
    assert float(model.backbone.fusion.output_scale) == authority.model_configuration["fusion_output_scale"]
    assert float(model.backbone.fusion.output_scale) != float(model.backbone.readout_weight.square().mean().sqrt())
    assert len(provenance["complete_state_pins"]) == len(provenance["imported_state_pins"])+4
    json.dumps(provenance)  # Root can persist it directly without tensor dumps.


@pytest.mark.parametrize("failure", ["missing", "unexpected", "predictor_in_old", "shape", "dtype", "nonfinite",
                                     "scale", "frozen_bytes", "class", "alias", "trainability", "module_keys", "cursor"])
def test_corrupt_historical_payload_is_rejected_before_any_model_copy(endpoint, failure):
    model, _, path, payload, authority = endpoint
    payload = copy.deepcopy(payload)
    native = "backbone.backbone.transformer.wte.weight"
    fusion = FUSION_NAMES[0]
    if failure == "missing": payload["model"].pop(fusion)
    elif failure == "unexpected": payload["model"]["unexpected"] = torch.ones(1)
    elif failure == "predictor_in_old": payload["model"]["predictor.unexpected"] = torch.ones(1)
    elif failure == "shape": payload["model"][fusion] = torch.ones(2, 3)
    elif failure == "dtype": payload["model"][native] = payload["model"][native].bfloat16()
    elif failure == "nonfinite": payload["model"][fusion][0, 0] = float("nan")
    elif failure == "scale": payload["model"]["backbone.fusion.output_scale"].mul_(2)
    elif failure == "frozen_bytes": payload["model"][native][0, 0] += .01
    elif failure == "class": payload["model_type"] = "unexpected.Model"
    elif failure == "alias": payload["parameter_layout"][0]["aliases"].append("unexpected.alias")
    elif failure == "trainability": payload["parameter_layout"][0]["requires_grad"] = True
    elif failure == "module_keys": payload["module_training"].pop("backbone")
    elif failure == "cursor": payload["data_cursor"]["next_update"] = 0
    authority = rehash(path, payload, authority)
    before = state_digests(model)
    with pytest.raises(ValueError):
        importer.load_into_current(model, path, authority)
    assert state_digests(model) == before


@pytest.mark.parametrize("failure", ["checkpoint_hash", "architecture", "fusion_config", "nextlat_config", "predictor_seed_bytes", "frozen_target", "tie"])
def test_independent_authority_or_current_model_drift_is_rejected(endpoint, failure, monkeypatch):
    model, _, path, _, authority = endpoint
    if failure == "checkpoint_hash":
        authority = replace(authority, checkpoint={**authority.checkpoint, "sha256": "0"*64})
    elif failure == "architecture":
        authority = copy.deepcopy(authority)
        authority.configuration["model_config"]["model_dim"] *= 2
    elif failure == "fusion_config":
        authority = copy.deepcopy(authority)
        authority.model_configuration["fusion_config"]["norm_eps"] *= 2
    elif failure == "nextlat_config":
        authority = copy.deepcopy(authority)
        authority.model_configuration["nextlat_config"]["norm_eps"] *= 2
    elif failure == "predictor_seed_bytes":
        with torch.no_grad(): next(model.predictor.parameters()).view(-1)[0] += .01
    elif failure == "frozen_target": next(model.parameters()).requires_grad_(False)
    elif failure == "tie":
        untied = torch.nn.Parameter(model.backbone.readout_weight.detach().clone())
        monkeypatch.setattr(OLMoFBT, "readout_weight", property(lambda self: untied))
    before = state_digests(model)
    with pytest.raises(ValueError):
        importer.load_into_current(model, path, authority)
    assert state_digests(model) == before


def test_exact_source_migrations_are_explicit_not_path_allowlists():
    old = {name: row[0] for name, row in importer.SOURCE_MIGRATIONS.items()}
    current = {name: row[1] for name, row in importer.SOURCE_MIGRATIONS.items()}
    old["unchanged.py"] = current["unchanged.py"] = "a"*64
    mapping = importer.historical_source_mapping(old, current)
    assert sum(record["changed"] for record in mapping.values()) == 6
    assert all(record["interpretation"] for record in mapping.values())
    changed = {**current, next(iter(importer.SOURCE_MIGRATIONS)): "f"*64}
    with pytest.raises(ValueError, match="Unreviewed"):
        importer.historical_source_mapping(old, changed)
    with pytest.raises(ValueError, match="six reviewed"):
        importer.historical_source_mapping(old, {**current, "unchanged.py": "b"*64})


def test_immutable_report_authority_maps_sources_and_production_cannot_fallback_to_cpu(endpoint, tmp_path, monkeypatch):
    model, _, _, _, authority = endpoint
    config = copy.deepcopy(authority.configuration)
    current_sources = {name: row[1] for name, row in importer.SOURCE_MIGRATIONS.items()}
    config["source_hashes"] = {name: row[0] for name, row in importer.SOURCE_MIGRATIONS.items()}
    source = {**authority.source_fingerprint, "code": config["source_hashes"]}
    record = {**authority.checkpoint, "storage": {"sha256": authority.checkpoint["sha256"],
        "size_bytes": authority.checkpoint["size_bytes"], "generation": "42", "uri": "gs://fixture/endpoint.pt"}}
    report = {"schema": "olmo-o5c-arm-v1", "status": "completed", "arm": "mixed", "finished_utc": "fixture",
        "configuration": {k: v for k, v in config.items() if k not in ("arm", "storage_prefix")},
        "storage_prefix": config["storage_prefix"], "counters": authority.counters, "data_cursor": 1,
        "checkpoints": [record], "source_fingerprint": source,
        "frozen_state_initial": authority.frozen_state_digests, "frozen_state_final": authority.frozen_state_digests,
        "endpoint": {"endpoint_configuration": authority.model_configuration, "checkpoint_sha256": config["source_checkpoint_sha256"]}}
    path = tmp_path/"report.json"
    path.write_text(json.dumps(report))
    for name, value in {"ENDPOINT_REPORT_SHA256": sha256_file(path), "ENDPOINT_SHA256": record["sha256"],
        "ENDPOINT_SIZE_BYTES": record["size_bytes"], "ENDPOINT_GENERATION": "42", "ENDPOINT_UPDATE": 1,
        "ENDPOINT_URI": "gs://fixture/endpoint.pt"}.items():
        monkeypatch.setattr(importer, name, value)
    monkeypatch.setattr(importer, "historical_current_sources", lambda: current_sources)
    real_kind = importer.endpoint_authority(path)
    assert real_kind.kind == "o5c-mixed-update512" and len(real_kind.source_mapping) == 6
    with pytest.raises(ValueError, match="without CPU fallback"):
        importer.load_into_current(model, authority.checkpoint["path"], real_kind)
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="immutable"):
        importer.endpoint_authority(path)
