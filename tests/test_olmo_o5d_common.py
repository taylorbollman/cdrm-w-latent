"""Model-only O5c endpoint loading, immutable authority and rejected corruption."""
import copy
from dataclasses import asdict
import json

import pytest
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.nextlat import NextLatBatch, NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts import olmo_o5b_common as o5b
from scripts import olmo_o5c_common as o5c
from scripts import olmo_o5d_common as common


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(5417)


@pytest.fixture
def endpoint(tmp_path):
    base = OLMoTiledRTForCausalLM(OLMoConfig.tiny(), attention_backend="math")
    model = o5b.ObservedFBTLM(OLMoFBT(base), NextLatConfig(32, vocab_chunk_size=3), enabled=False, gamma=1)
    o5c.freeze_native(model)
    frozen = o5c.frozen_state_digests(model)
    optimizer = o5c.build_optimizer(model)
    ids = torch.tensor([[2, 3, 5, 8, 13, 21], [7, 10, 17, 22, 1, 1]])
    valid = torch.tensor([[True]*6, [True]*4+[False]*2])
    docs = torch.arange(2)[:, None].expand_as(ids).clone().masked_fill(~valid, -1)
    counters = TrainingCounters()
    o5c.observed_step(model, optimizer, [NextLatBatch(ids, valid, docs)], counters=counters,
                      backbone_kwargs={"mode": o5c.make_mode(1)})
    construction = {"arm": "fbt", "model_config": model.backbone.config.to_dict(),
        "fusion_config": model.backbone.fusion_config.to_dict(), "nextlat_config": model.config.to_dict(),
        "fusion_output_scale": float(model.backbone.fusion.output_scale),
        "attention_backend": "math", "attention_precision": "mixed", "nextlat_enabled": False,
        "rt_layers": [], "num_passes": 2, "gamma": 1}
    plan = {"arm": "mixed", "total_updates": 1, "ce_per_update": counters.ce_positions,
        "batch_ce_prefix": [0, counters.ce_positions], "batch_token_prefix": [0, counters.input_tokens],
        "batch_row_prefix": [0, counters.documents]}
    config = {"schema": "olmo-o5c-pilot-config-v1", "arm": "mixed", "storage_prefix": "gs://fixture",
        "native_backbone_frozen": True, "nextlat_enabled": False, "rt_layers": [], "beta": 1.,
        "num_passes": 2, "gamma": 1., "prefix_mixin": False, "hidden_jitter": 0.,
        "model_config": model.backbone.config.to_dict(), "attention_backend": "math",
        "physical_batch_size": 16, "frozen_state_initial": frozen,
        "trainable_layout": o5c.trainable_layout(model), "source_hashes": {"fixture.py": "a"*64},
        "source_checkpoint_sha256": "b"*64, "data_manifest_sha256": "c"*64,
        "schedule": {"total_updates": 1, "total_ce": counters.ce_positions,
                     "ce_per_update": counters.ce_positions, "arms": {"mixed": plan}}}
    source = {"checkpoint_sha256": "b"*64, "source_checkpoint_sha256": "b"*64,
        "code": config["source_hashes"], "data_manifest_sha256": "c"*64, "runtime": {"torch": str(torch.__version__)}}
    cursor = {"next_update": 1, "bad_evals": 0}
    path = tmp_path/"endpoint.pt"
    receipt = o5c.save_checkpoint(path, model, optimizer, counters=counters, data_cursor=cursor,
                                   configuration=config, source_fingerprint=source)
    return {"model": model, "checkpoint": path, "receipt": receipt,
        "kwargs": {"expected_sha256": receipt["sha256"], "expected_size_bytes": receipt["size_bytes"],
        "expected_configuration": config, "expected_model_configuration": construction,
        "expected_source_fingerprint": source, "expected_counters": asdict(counters),
        "expected_data_cursor": cursor, "expected_frozen_state_digests": frozen, "device": "cpu"}}


def test_model_only_import_is_exact_frozen_eval_tied_and_does_not_restore_rng_or_optimizer(endpoint, monkeypatch):
    before = torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation loader must never construct an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    model, provenance = common.load_endpoint(endpoint["checkpoint"], **endpoint["kwargs"])
    assert torch.equal(before, torch.get_rng_state())
    original = endpoint["model"].state_dict()
    assert model.state_dict().keys() == original.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    assert all(not module.training for module in model.modules())
    assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
    assert provenance["evaluation_only"] and provenance["trainable_parameters"] == []
    assert provenance["state_digests"] == o5c.state_digests(model)
    assert provenance["frozen_state_digests"] == endpoint["kwargs"]["expected_frozen_state_digests"]
    assert "not constructed" in provenance["optimizer_state"]


@pytest.mark.parametrize("failure", ["hash", "size", "config", "construction", "source", "counters", "cursor", "frozen"])
def test_import_rejects_wrong_independent_authority(endpoint, failure):
    kwargs = copy.deepcopy(endpoint["kwargs"])
    if failure == "hash": kwargs["expected_sha256"] = "0"*64
    elif failure == "size": kwargs["expected_size_bytes"] += 1
    elif failure == "config": kwargs["expected_configuration"]["beta"] = .5
    elif failure == "construction": kwargs["expected_model_configuration"]["model_config"]["model_dim"] *= 2
    elif failure == "source": kwargs["expected_source_fingerprint"]["checkpoint_sha256"] = "0"*64
    elif failure == "counters": kwargs["expected_counters"]["microbatches"] += 1
    elif failure == "cursor": kwargs["expected_data_cursor"]["next_update"] = 0
    elif failure == "frozen": kwargs["expected_frozen_state_digests"]["backbone.fusion.output_scale"] = "0"*64
    with pytest.raises(ValueError):
        common.load_endpoint(endpoint["checkpoint"], **kwargs)


@pytest.mark.parametrize("failure", ["schema", "class", "layout", "optimizer_ownership", "native_shape",
    "fusion_shape", "native_dtype", "nonfinite", "tensor_keys", "module_keys", "scale", "native_bytes", "cursor"])
def test_import_rejects_corrupt_payload_after_independent_rehash(endpoint, failure):
    path = endpoint["checkpoint"]
    payload = torch.load(path, weights_only=True)
    native = "backbone.backbone.transformer.wte.weight"
    fusion = o5c.FUSION_NAMES[0]
    if failure == "schema": payload["schema"] = "wrong"
    elif failure == "class": payload["model_type"] = "wrong.Model"
    elif failure == "layout": payload["parameter_layout"][0]["requires_grad"] = True
    elif failure == "optimizer_ownership": payload["optimizer_ownership"] = [[native]]
    elif failure == "native_shape": payload["model"][native] = torch.zeros(2, 3)
    elif failure == "fusion_shape": payload["model"][fusion] = torch.zeros(2, 3)
    elif failure == "native_dtype": payload["model"][native] = payload["model"][native].bfloat16()
    elif failure == "nonfinite": payload["model"][fusion][0, 0] = float("nan")
    elif failure == "tensor_keys": payload["model"]["unexpected"] = torch.ones(1)
    elif failure == "module_keys": payload["module_training"].pop("backbone")
    elif failure == "scale": payload["model"]["backbone.fusion.output_scale"] *= 2
    elif failure == "native_bytes": payload["model"][native][0, 0] += .01
    elif failure == "cursor": payload["data_cursor"]["next_update"] = 0
    torch.save(payload, path)
    kwargs = {**endpoint["kwargs"], "expected_sha256": sha256_file(path), "expected_size_bytes": path.stat().st_size}
    with pytest.raises(ValueError):
        common.load_endpoint(path, **kwargs)


def _report(endpoint, tmp_path, monkeypatch):
    kwargs = endpoint["kwargs"]
    config = kwargs["expected_configuration"]
    record = {**endpoint["receipt"], "input_tokens": kwargs["expected_counters"]["input_tokens"],
              "ce_positions": kwargs["expected_counters"]["ce_positions"],
              "storage": {"sha256": endpoint["receipt"]["sha256"], "size_bytes": endpoint["receipt"]["size_bytes"],
                          "generation": "42", "uri": "gs://fixture/mixed/endpoint.pt"}}
    report = {"schema": "olmo-o5c-arm-v1", "status": "completed", "arm": "mixed", "finished_utc": "fixture",
        "configuration": {k: v for k, v in config.items() if k not in ("arm", "storage_prefix")},
        "storage_prefix": config["storage_prefix"], "checkpoints": [record],
        "counters": kwargs["expected_counters"], "data_cursor": 1,
        "source_fingerprint": kwargs["expected_source_fingerprint"],
        "frozen_state_initial": kwargs["expected_frozen_state_digests"],
        "frozen_state_final": kwargs["expected_frozen_state_digests"],
        "endpoint": {"checkpoint_sha256": config["source_checkpoint_sha256"],
                     "endpoint_configuration": kwargs["expected_model_configuration"]}}
    path = tmp_path/"report.json"
    path.write_text(json.dumps(report))
    for name, value in {"ENDPOINT_SHA256": record["sha256"], "ENDPOINT_SIZE_BYTES": record["size_bytes"],
                        "ENDPOINT_GENERATION": "42", "ENDPOINT_URI": record["storage"]["uri"],
                        "ENDPOINT_UPDATE": 1, "ENDPOINT_REPORT_SHA256": sha256_file(path)}.items():
        monkeypatch.setattr(common, name, value)
    monkeypatch.setattr(common, "source_hashes", lambda: config["source_hashes"])
    return report, path


def test_report_authority_returns_full_cursor_and_inherited_construction(endpoint, tmp_path, monkeypatch):
    _, path = _report(endpoint, tmp_path, monkeypatch)
    authority = common.endpoint_metadata(path)
    assert authority["data_cursor"] == {"next_update": 1, "bad_evals": 0}
    assert authority["model_configuration"] == endpoint["kwargs"]["expected_model_configuration"]
    assert authority["checkpoint_path"] == str(endpoint["checkpoint"])


@pytest.mark.parametrize("failure", ["hash", "status", "generation", "uri", "source", "duplicate", "frozen", "counter"])
def test_report_rejects_wrong_pin_or_inconsistent_completion(endpoint, tmp_path, monkeypatch, failure):
    report, path = _report(endpoint, tmp_path, monkeypatch)
    if failure == "hash": monkeypatch.setattr(common, "ENDPOINT_REPORT_SHA256", "0"*64)
    else:
        if failure == "status": report["status"] = "paused"
        elif failure == "generation": report["checkpoints"][0]["storage"]["generation"] = "wrong"
        elif failure == "uri": report["checkpoints"][0]["storage"]["uri"] = "gs://different"
        elif failure == "source": monkeypatch.setattr(common, "source_hashes", lambda: {})
        elif failure == "duplicate": report["checkpoints"].append(copy.deepcopy(report["checkpoints"][0]))
        elif failure == "frozen": report["frozen_state_final"] = {}
        elif failure == "counter": report["counters"]["microbatches"] += 1
        path.write_text(json.dumps(report)); monkeypatch.setattr(common, "ENDPOINT_REPORT_SHA256", sha256_file(path))
    with pytest.raises(ValueError): common.endpoint_metadata(path)


def test_production_endpoint_cannot_use_cpu_fallback(endpoint, monkeypatch):
    monkeypatch.setattr(common, "ENDPOINT_SHA256", endpoint["receipt"]["sha256"])
    with pytest.raises(ValueError, match="without CPU fallback"):
        common.load_endpoint(endpoint["checkpoint"], **endpoint["kwargs"])
