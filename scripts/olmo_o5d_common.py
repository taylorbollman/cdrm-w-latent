"""Pinned, model-only import of the completed O5c mixed endpoint for O5d.

This module does not create an optimizer, restore training/RNG state, or alter
the frozen O5b/O5c implementation. The production entry point obtains authority
from endpoint_metadata(); explicit independent authority supports tiny CPU tests.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import CHECKPOINT_SCHEMA, parameter_layout
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTConfig, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_lm_common import state_digests
from scripts.olmo_o5b_common import ObservedFBTLM
from scripts.olmo_o5c_common import FUSION_NAMES, freeze_native, plain_metadata, source_hashes

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_SHA256 = "7bba59ac75478fb15cec5fd0187f306b220da9babb138ebccbf5d88a70609d1a"
ENDPOINT_SIZE_BYTES = 4807843871
ENDPOINT_GENERATION = "1790059437165208"
ENDPOINT_UPDATE = 512
ENDPOINT_URI = ("gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/"
                "olmo1b-o5c-fusion-only/20260922T061000Z/mixed/update-000512.pt")
ENDPOINT_REPORT = ROOT/".runtime/olmo1b-step60000/o5c-pilot-01/mixed/report.json"
ENDPOINT_REPORT_SHA256 = "020a204ae02d3dfa1af2753f278cb465d73120ca8133258bc8a84272e15afbc8"


def _completed_schedule(config, counters, cursor):
    """Reject partial boundaries and inconsistent counts before model import."""
    schedule = config["schedule"]
    updates = schedule["total_updates"]
    plan = schedule["arms"]["mixed"]
    if type(updates) is not int or updates < 1:
        raise ValueError("Endpoint schedule requires a positive integer update count")
    if (plan["arm"] != "mixed" or plan["total_updates"] != updates
            or plan["ce_per_update"] != schedule["ce_per_update"]):
        raise ValueError("Endpoint mixed schedule differs from its configuration")
    expected = {"optimizer_updates": updates, "latent_pairs": 0, "kl_triples": 0}
    for counter, prefix in (("input_tokens", "batch_token_prefix"),
                            ("ce_positions", "batch_ce_prefix"),
                            ("documents", "batch_row_prefix")):
        values = plan[prefix]
        if (len(values) != updates+1 or values[0] != 0
                or any(type(value) is not int for value in values)
                or any(b <= a for a, b in zip(values, values[1:]))):
            raise ValueError("Endpoint schedule prefix is malformed")
        expected[counter] = values[-1]
    physical = config["physical_batch_size"]
    if type(physical) is not int or physical < 1:
        raise ValueError("Endpoint physical batch size differs")
    rows = plan["batch_row_prefix"]
    expected["microbatches"] = sum((b-a+physical-1)//physical for a, b in zip(rows, rows[1:]))
    if (counters != expected or cursor != {"next_update": updates, "bad_evals": 0}
            or expected["ce_positions"] != schedule["total_ce"]
            or schedule["total_ce"] != updates*schedule["ce_per_update"]):
        raise ValueError("Endpoint counters/cursor do not describe its completed schedule")


def _validate_model_config(config, model_config):
    if (config.get("schema") != "olmo-o5c-pilot-config-v1" or config.get("arm") != "mixed"
            or config.get("native_backbone_frozen") is not True or config.get("nextlat_enabled") is not False
            or config.get("rt_layers") != [] or config.get("beta") != 1
            or config.get("num_passes") != 2 or config.get("gamma") != 1
            or config.get("prefix_mixin") is not False or config.get("hidden_jitter") != 0):
        raise ValueError("Endpoint must be the unchanged fusion-only mixed beta1 K2 gamma1 arm")
    if (model_config.get("arm") != "fbt" or model_config.get("nextlat_enabled") is not False
            or model_config.get("rt_layers") != [] or model_config.get("num_passes") != 2
            or model_config.get("gamma") != 1
            or model_config.get("model_config") != config.get("model_config")
            or model_config.get("attention_backend") != config.get("attention_backend")):
        raise ValueError("Endpoint inherited native/fusion construction configuration differs")


def endpoint_metadata(report_path=ENDPOINT_REPORT):
    """Validate the immutable completed report and all inherited frozen sources.

    Storage identity is verified against the previously retained receipt, not by
    a new cloud request. The subsequent load independently hashes local bytes.
    """
    path = Path(report_path)
    if not path.is_file() or path.is_symlink() or sha256_file(path) != ENDPOINT_REPORT_SHA256:
        raise ValueError("O5c mixed endpoint report differs from its immutable pin")
    report = json.loads(path.read_text())
    if (report.get("schema") != "olmo-o5c-arm-v1" or report.get("status") != "completed"
            or report.get("arm") != "mixed" or not report.get("finished_utc")):
        raise ValueError("Require the completed O5c mixed report")
    config = {**report["configuration"], "arm": "mixed", "storage_prefix": report["storage_prefix"]}
    model_config = report["endpoint"]["endpoint_configuration"]
    _validate_model_config(config, model_config)
    counters = report["counters"]
    cursor = {"next_update": report["data_cursor"], "bad_evals": 0}
    _completed_schedule(config, counters, cursor)
    if counters["optimizer_updates"] != ENDPOINT_UPDATE:
        raise ValueError("O5c mixed endpoint update differs from its pin")
    records = [row for row in report["checkpoints"] if row["optimizer_updates"] == ENDPOINT_UPDATE]
    if len(records) != 1 or records[0] != report["checkpoints"][-1]:
        raise ValueError("Require exactly one final retained endpoint record")
    record = records[0]
    storage = record.get("storage", {})
    if (record.get("schema") != CHECKPOINT_SCHEMA or record.get("sha256") != ENDPOINT_SHA256
            or record.get("size_bytes") != ENDPOINT_SIZE_BYTES
            or record.get("input_tokens") != counters["input_tokens"]
            or record.get("ce_positions") != counters["ce_positions"]
            or storage.get("sha256") != ENDPOINT_SHA256 or storage.get("size_bytes") != ENDPOINT_SIZE_BYTES
            or storage.get("generation") != ENDPOINT_GENERATION or storage.get("uri") != ENDPOINT_URI):
        raise ValueError("O5c mixed endpoint checkpoint/retention identity differs")
    source = report["source_fingerprint"]
    if (source.get("code") != config["source_hashes"] or config["source_hashes"] != source_hashes()
            or source.get("checkpoint_sha256") != config["source_checkpoint_sha256"]
            or source.get("source_checkpoint_sha256") != config["source_checkpoint_sha256"]
            or source.get("data_manifest_sha256") != config["data_manifest_sha256"]
            or report["endpoint"]["checkpoint_sha256"] != config["source_checkpoint_sha256"]):
        raise ValueError("Frozen O5c endpoint configuration/source inventory differs")
    frozen = config["frozen_state_initial"]
    if report["frozen_state_initial"] != frozen or report["frozen_state_final"] != frozen:
        raise ValueError("O5c endpoint native state preservation differs")
    checkpoint = Path(record["path"])
    return {"configuration": config, "model_configuration": model_config,
            "source_fingerprint": source, "checkpoint": record,
            "checkpoint_path": str(checkpoint if checkpoint.is_absolute() else ROOT/checkpoint),
            "counters": counters, "data_cursor": cursor, "report_data_cursor": report["data_cursor"],
            "frozen_state_digests": frozen, "report_path": str(path),
            "report_sha256": ENDPOINT_REPORT_SHA256}


def load_endpoint(checkpoint, expected_sha256, *, expected_size_bytes,
                  expected_configuration, expected_model_configuration,
                  expected_source_fingerprint, expected_counters, expected_data_cursor,
                  expected_frozen_state_digests, device="cuda"):
    """Load only native and fusion tensors, then freeze the entire eval model.

    Production callers must pass endpoint_metadata() authority and CUDA. Explicit
    device='cpu' exists only for independently hashed tiny checkpoint fixtures.
    No optimizer is constructed; saved moments/scheduler/RNG are never restored.
    O5c metadata is built-in Python data and needs no pickle-global allowlist.
    """
    path = Path(checkpoint)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)):
        raise ValueError("Expected checkpoint SHA256 must be 64 lowercase hex characters")
    if type(expected_size_bytes) is not int or expected_size_bytes < 1:
        raise ValueError("Expected checkpoint size must be a positive integer")
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != expected_size_bytes
            or sha256_file(path) != expected_sha256):
        raise ValueError("Endpoint checkpoint bytes differ from expected size/SHA256")
    target_device = torch.device(device)
    if target_device.type not in ("cuda", "cpu"):
        raise ValueError("Endpoint evaluation supports explicit CUDA or tiny CPU fixtures only")
    if expected_sha256 == ENDPOINT_SHA256 and target_device.type != "cuda":
        raise ValueError("The production endpoint must run in the GPU container, without CPU fallback")
    config = plain_metadata(expected_configuration)
    construction = plain_metadata(expected_model_configuration)
    source = plain_metadata(expected_source_fingerprint)
    counters = plain_metadata(expected_counters)
    cursor = plain_metadata(expected_data_cursor)
    _validate_model_config(config, construction)
    _completed_schedule(config, counters, cursor)
    if config["frozen_state_initial"] != expected_frozen_state_digests:
        raise ValueError("Expected frozen-state hashes differ from endpoint configuration")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("schema") != CHECKPOINT_SCHEMA or plain_metadata(payload.get("configuration")) != config
            or plain_metadata(payload.get("source_fingerprint")) != source
            or plain_metadata(payload.get("counters")) != counters
            or plain_metadata(payload.get("data_cursor")) != cursor):
        raise ValueError("Endpoint schema/configuration/source/counters/cursor metadata differs")
    base = OLMoTiledRTForCausalLM(OLMoConfig.from_dict(config["model_config"]),
        attention_backend=config["attention_backend"], attention_precision=construction["attention_precision"],
        device="meta", dtype=torch.float32)
    state = payload.get("model", {})
    native = {name.removeprefix("backbone.backbone."): value for name, value in state.items()
              if name.startswith("backbone.backbone.")}
    if set(native) != set(base.state_dict()):
        raise ValueError("Endpoint native tensor keys differ")
    for name, template in base.state_dict().items():
        value = native[name]
        if not isinstance(value, torch.Tensor) or value.shape != template.shape or value.dtype != torch.float32:
            raise ValueError(f"Endpoint native tensor geometry/dtype differs: {name}")
    base.load_state_dict(native, strict=True, assign=True)
    model = ObservedFBTLM(OLMoFBT(base, FBTConfig.from_dict(construction["fusion_config"])),
        NextLatConfig.from_dict(construction["nextlat_config"]), enabled=False, gamma=1)
    training_layout = freeze_native(model)
    if (payload.get("model_type") != type(model).__module__+"."+type(model).__qualname__
            or payload.get("parameter_layout") != parameter_layout(model)
            or config["trainable_layout"] != training_layout
            or payload.get("optimizer_ownership") != [list(FUSION_NAMES)]):
        raise ValueError("Endpoint model class/parameter ownership differs")
    target = model.state_dict()
    if set(state) != set(target):
        raise ValueError("Endpoint model tensor keys differ")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != target[name].shape
                or value.dtype != target[name].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Endpoint tensor shape/dtype/finiteness differs: {name}")
    modes = payload.get("module_training", {})
    if set(modes) != set(dict(model.named_modules())) or any(type(mode) is not bool for mode in modes.values()):
        raise ValueError("Endpoint module ownership/training metadata differs")
    model.load_state_dict(state, strict=True, assign=False)
    if float(model.backbone.fusion.output_scale) != construction["fusion_output_scale"]:
        raise ValueError("Endpoint fixed fusion scale differs from its configuration")
    if model.backbone.readout_weight is not model.backbone.token_embeddings.weight:
        raise ValueError("Endpoint native tied embedding/readout ownership differs")
    del payload, state, native, target
    gc.collect()
    model.requires_grad_(False).to(target_device).eval()
    hashes = state_digests(model)
    frozen = {name: digest for name, digest in hashes.items() if name not in FUSION_NAMES}
    if frozen != expected_frozen_state_digests:
        raise ValueError("Loaded native/fixed-buffer bytes differ from the frozen endpoint")
    return model, {"checkpoint_sha256": expected_sha256, "checkpoint_size_bytes": expected_size_bytes,
        "endpoint_configuration": config, "endpoint_model_configuration": construction,
        "endpoint_source_fingerprint": source, "endpoint_counters": counters, "endpoint_data_cursor": cursor,
        "state_digests": hashes, "frozen_state_digests": frozen, "trainable_parameters": [],
        "optimizer_state": "not constructed; checkpoint optimizer/scheduler/counters/RNG not restored",
        "evaluation_only": True}
