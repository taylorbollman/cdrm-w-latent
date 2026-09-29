"""Explicit weights-only O5c import into today's trainable campaign NF model.

This is not historical execution/resume compatibility. The old O5d validator
is unchanged. Exact historical report/checkpoint authority, reviewed source
mapping and tensor contracts govern this separate diagnostic import.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path

import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.fbt_training import FBTNextLatLM
from cdrm.pretrained.lm_training import CHECKPOINT_SCHEMA, parameter_layout
from cdrm.pretrained.nextlat import NextLatConfig, NextLatPredictor
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_lm_common import state_digests, tree_digests
from scripts.olmo_o5c_common import FUSION_NAMES, plain_metadata, source_hashes as historical_current_sources
from scripts.olmo_o5d_common import (
    ENDPOINT_SHA256, ENDPOINT_SIZE_BYTES, ENDPOINT_GENERATION, ENDPOINT_UPDATE,
    ENDPOINT_URI, ENDPOINT_REPORT, ENDPOINT_REPORT_SHA256, _completed_schedule, _validate_model_config,
)

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_CHECKPOINT = ROOT/".runtime/olmo1b-step60000/o5c-pilot-01/mixed/update-000512.pt"

# Exact reviewed historical/current pairs. Future changes require a new explicit
# import review; this is not a generic allowlist of paths that may change freely.
SOURCE_MIGRATIONS = {
    "cdrm/pretrained/fbt_training.py": (
        "622ce7e14c02736543829a54685a3050f9b412e2401bc862e3a0d75c10f08586",
        "460150379f5401ac4b02ace8feb69056940c95b1582198e32c8641f9ed288668",
        "Adds explicit campaign_v1 pass weighting and document-policy validation; current K4 weights are intentional, historical legacy default remains."),
    "cdrm/pretrained/lm_training.py": (
        "09a7b33eb6928486c8c1d9f52bd2ffc4e0f88fe01baa42527ba56742bfce6a69",
        "22a8f37e4900c74296bbe817a7e946926fe262a9f8d2ed4a8002032d61eecf34",
        "Canonical metadata and optional fused AdamW; import restores no training/optimizer state."),
    "cdrm/pretrained/nextlat.py": (
        "04bf4e4cd9558ad5086170fab8577a3164a8b0a675fc38c9cb33da4e85dbfa3f",
        "2e0edb94d025950269646f8a62cd61e231d28624e2b1721626ec3f77d37a4df8",
        "Adds independent CE chunk size and document policy with legacy omitted defaults; historical predictor absent, current seeded predictor is new."),
    "cdrm/pretrained/olmo.py": (
        "53f1f812007d0fa5c90d93a026ff80fc32c138632d08dfcc92689a51c3db9667",
        "3eef88dee4853be6d0c4901dfd1cc1962dd5458defcfba0a5223a7fb1993c8a0",
        "Adds optional RoPE reuse/Dao, FA4 and pointwise backend dispatch without learned-state renaming; current runtime is recorded separately."),
    "cdrm/pretrained/olmo_fbt.py": (
        "237576033af0d2ae5ee9fd0ee6750299351cabc9e20b07d11d8aaa32e3689fe2",
        "797d1b4cc68f3e34f422f331eef67288da3db38cf1770a540c1a01be50a24e98",
        "Adds explicit first-pass policy, keyed jitter, document policy and padded causal dispatch; current K4/jitter is not old K2 execution."),
    "cdrm/pretrained/olmo_tiled.py": (
        "f9774c8b48d35f5ad4e8371ede70148e8c033d4966d2e997974f0129d041e03c",
        "8df55ddb3c362d7c0394b218e3a9c328b8dbea9a8ae326372b976f6241dd316a",
        "Adds checkpointing, RoPE reuse/KV-only writes, native Triton/recompute/author paths and ordinary backend/padding options; NF executes no RT."),
}


@dataclass(frozen=True)
class AdaptedImportAuthority:
    kind: str
    checkpoint: dict
    configuration: dict
    model_configuration: dict
    source_fingerprint: dict
    counters: dict
    data_cursor: dict
    frozen_state_digests: dict
    source_mapping: dict
    report_path: str | None = None
    report_sha256: str | None = None


def historical_source_mapping(historical, current):
    if set(historical) != set(current):
        raise ValueError("Historical/current source inventory differs")
    changed = {name for name in historical if historical[name] != current[name]}
    if changed != set(SOURCE_MIGRATIONS):
        raise ValueError("Historical source changes differ from the six reviewed migrations")
    result = {}
    for name, old in historical.items():
        reason = "Unchanged historical source bytes"
        if name in changed:
            wanted_old, wanted_current, reason = SOURCE_MIGRATIONS[name]
            if (old, current[name]) != (wanted_old, wanted_current):
                raise ValueError(f"Unreviewed historical/current source revision: {name}")
        result[name] = {"historical_sha256": old, "current_sha256": current[name],
                        "changed": name in changed, "interpretation": reason}
    return dict(sorted(result.items()))


def endpoint_authority(report_path=ENDPOINT_REPORT):
    """Read immutable O5c authority while explicitly mapping current sources."""
    path = Path(report_path)
    if not path.is_file() or path.is_symlink() or sha256_file(path) != ENDPOINT_REPORT_SHA256:
        raise ValueError("O5c report differs from its immutable SHA256 pin")
    report = json.loads(path.read_text())
    if sha256_file(path) != ENDPOINT_REPORT_SHA256:
        raise ValueError("O5c report changed while reading")
    if (report.get("schema") != "olmo-o5c-arm-v1" or report.get("status") != "completed"
            or report.get("arm") != "mixed" or not report.get("finished_utc")):
        raise ValueError("Require the completed historical O5c mixed endpoint")
    config = {**report["configuration"], "arm": "mixed", "storage_prefix": report["storage_prefix"]}
    construction = report["endpoint"]["endpoint_configuration"]
    _validate_model_config(config, construction)
    counters, cursor = report["counters"], {"next_update": report["data_cursor"], "bad_evals": 0}
    _completed_schedule(config, counters, cursor)
    if counters["optimizer_updates"] != ENDPOINT_UPDATE:
        raise ValueError("Historical endpoint update differs")
    records = [row for row in report["checkpoints"] if row["optimizer_updates"] == ENDPOINT_UPDATE]
    if len(records) != 1 or records[0] != report["checkpoints"][-1]:
        raise ValueError("Require one final historical checkpoint record")
    record = records[0]
    storage = record.get("storage", {})
    if (record.get("schema") != CHECKPOINT_SCHEMA or record.get("sha256") != ENDPOINT_SHA256
            or record.get("size_bytes") != ENDPOINT_SIZE_BYTES
            or record.get("input_tokens") != counters["input_tokens"] or record.get("ce_positions") != counters["ce_positions"]
            or storage.get("sha256") != ENDPOINT_SHA256 or storage.get("size_bytes") != ENDPOINT_SIZE_BYTES
            or storage.get("generation") != ENDPOINT_GENERATION or storage.get("uri") != ENDPOINT_URI):
        raise ValueError("Historical checkpoint/retention identity differs")
    source = report["source_fingerprint"]
    if (source.get("code") != config["source_hashes"]
            or source.get("checkpoint_sha256") != config["source_checkpoint_sha256"]
            or source.get("source_checkpoint_sha256") != config["source_checkpoint_sha256"]
            or source.get("data_manifest_sha256") != config["data_manifest_sha256"]
            or report["endpoint"]["checkpoint_sha256"] != config["source_checkpoint_sha256"]):
        raise ValueError("Historical source/configuration authority is inconsistent")
    frozen = config["frozen_state_initial"]
    if report["frozen_state_initial"] != frozen or report["frozen_state_final"] != frozen:
        raise ValueError("Historical native/fixed-buffer preservation differs")
    mapping = historical_source_mapping(source["code"], historical_current_sources())
    return AdaptedImportAuthority("o5c-mixed-update512", record, config, construction, source,
        counters, cursor, frozen, mapping, str(path), ENDPOINT_REPORT_SHA256)


def source_hashes():
    result = historical_current_sources()
    for name in ("scripts/olmo_o5d_common.py", "scripts/olmo_campaign_adapted_import.py",
                 "tests/test_campaign_adapted_import.py"):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def _current_model_contract(model, authority):
    if (type(model) is not FBTNextLatLM or not isinstance(model.backbone.backbone, OLMoTiledRTForCausalLM)
            or not model.enabled or model.predictor is None or model.pass_loss_policy != "campaign_v1"
            or model.gamma != 1 or model.config.seed != 20260921):
        raise ValueError("Import target must be the current seeded campaign NF model")
    if (model.backbone.readout_weight is not model.backbone.token_embeddings.weight
            or len({id(p) for p in model.parameters()}) != len(list(model.parameters()))):
        raise ValueError("Current tied embedding/readout ownership differs")
    if any(p.dtype != torch.float32 or not p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Import target requires trainable FP32 masters and cleared gradients")
    config, construction = authority.configuration, authority.model_configuration
    if (model.backbone.config.to_dict() != config["model_config"]
            or model.backbone.fusion_config.to_dict() != construction["fusion_config"]):
        raise ValueError("Current native/fusion architecture differs from historical tensors")
    old_nextlat = NextLatConfig.from_dict(construction["nextlat_config"]).to_dict()
    current_nextlat = model.config.to_dict()
    # Only this documented chunk-layout difference is allowed. It has no learned
    # state and applies to the CURRENT loss; no historical loss parity is claimed.
    if (any(current_nextlat.get(key) != value for key, value in old_nextlat.items())
            or set(current_nextlat)-set(old_nextlat) != {"ce_chunk_size"}
            or current_nextlat["ce_chunk_size"] != 2048 or model.config.document_policy != "isolated-v1"):
        raise ValueError("Unexplained current NextLat configuration migration")
    predictor = NextLatPredictor(model.config)
    expected_predictor = state_digests(predictor)
    del predictor
    actual_predictor = state_digests(model.predictor)
    if actual_predictor != expected_predictor:
        raise ValueError("Current predictor differs from its intended fresh seed")
    return {"predictor_state_digests": actual_predictor,
        "nextlat_configuration_mapping": {"historical": old_nextlat, "current": current_nextlat,
            "new_fields": {"ce_chunk_size": 2048}, "interpretation": "Historical NextLat disabled/no predictor; current predictor seeded, current CE chunk policy retained"}}


def load_into_current(model, checkpoint, authority):
    """Validate everything before copying native/fusion state in place.

Production authority is checked again against its immutable report. A separate
tiny-cpu-test authority exists solely for small independently hashed tests.
No optimizer/scheduler/cursor/RNG/trainability or module-training state is loaded.
"""
    if not isinstance(authority, AdaptedImportAuthority):
        raise TypeError("Explicit adapted-import authority is required")
    device = next(model.parameters()).device
    if authority.kind == "o5c-mixed-update512":
        if (device.type != "cuda" or not Path("/.dockerenv").exists()
                or Path.cwd() != Path("/workspace/cdrm-w-latent")):
            raise ValueError("Production import requires the GPU container without CPU fallback")
        if authority != endpoint_authority(authority.report_path):
            raise ValueError("Production authority differs from the immutable endpoint")
    elif authority.kind == "tiny-cpu-test":
        if (device.type != "cpu" or model.backbone.config != OLMoConfig.tiny()
                or authority.checkpoint.get("sha256") == ENDPOINT_SHA256):
            raise ValueError("Tiny CPU authority cannot import the production architecture/checkpoint")
    else:
        raise ValueError("Unknown adapted-import authority kind")
    if any(p.device != device for p in model.parameters()):
        raise ValueError("Import target must reside on one device")
    config, construction = plain_metadata(authority.configuration), plain_metadata(authority.model_configuration)
    _validate_model_config(config, construction)
    _completed_schedule(config, authority.counters, authority.data_cursor)
    if config["frozen_state_initial"] != authority.frozen_state_digests:
        raise ValueError("Frozen-state authority differs from configuration")
    current = _current_model_contract(model, authority)
    path = Path(checkpoint)
    digest, size = authority.checkpoint.get("sha256"), authority.checkpoint.get("size_bytes")
    if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or type(size) is not int or size <= 0):
        raise ValueError("Checkpoint needs an independent size/SHA256 authority")
    if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
        raise ValueError("Checkpoint path/size differs")
    before = path.stat()
    if sha256_file(path) != digest:
        raise ValueError("Checkpoint SHA256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Checkpoint changed while hashing/loading")
    if (payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("model_type") != "scripts.olmo_o5b_common.ObservedFBTLM"
            or plain_metadata(payload.get("configuration")) != config
            or plain_metadata(payload.get("source_fingerprint")) != plain_metadata(authority.source_fingerprint)
            or plain_metadata(payload.get("counters")) != plain_metadata(authority.counters)
            or plain_metadata(payload.get("data_cursor")) != plain_metadata(authority.data_cursor)):
        raise ValueError("Historical checkpoint schema/class/configuration/source/counters/cursor differs")
    target = {"backbone."+name: value for name, value in model.backbone.state_dict().items()}
    state = payload.get("model", {})
    if set(state) != set(target):
        raise ValueError("Historical native/fusion tensor keys differ; only predictor may be newly initialized")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != target[name].shape
                or value.dtype != target[name].dtype or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Historical tensor shape/dtype/finiteness differs: {name}")
    source_pins = tree_digests(state)
    frozen = {name: pin["sha256"] for name, pin in source_pins.items() if name not in FUSION_NAMES}
    if frozen != authority.frozen_state_digests:
        raise ValueError("Historical adapted native/fixed-buffer hashes differ")
    if float(state["backbone.fusion.output_scale"]) != construction["fusion_output_scale"]:
        raise ValueError("Saved fusion output_scale differs from historical configuration")
    historical_layout = [record for record in parameter_layout(model) if not record["name"].startswith("predictor.")]
    for record in historical_layout:
        record["requires_grad"] = record["name"] in FUSION_NAMES
    trainable = [{"name": name, "shape": list(target[name].shape), "numel": target[name].numel()} for name in FUSION_NAMES]
    module_names = {name for name, _ in model.named_modules() if name != "predictor" and not name.startswith("predictor.")}
    modes = payload.get("module_training", {})
    if (payload.get("parameter_layout") != historical_layout or payload.get("optimizer_ownership") != [list(FUSION_NAMES)]
            or config["trainable_layout"] != trainable or set(modes) != module_names
            or any(type(value) is not bool for value in modes.values())):
        raise ValueError("Historical parameter/alias/module/optimizer ownership differs")
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    training_modes = {name: module.training for name, module in model.named_modules()}
    model.backbone.load_state_dict({name.removeprefix("backbone."): value for name, value in state.items()}, strict=True, assign=False)
    loaded = tree_digests(model.state_dict())
    checks = {"imported_state_exact": {name: loaded[name] for name in state} == source_pins,
        "predictor_unchanged": state_digests(model.predictor) == current["predictor_state_digests"],
        "parameter_identities_preserved": identities == {name: id(p) for name, p in model.named_parameters()},
        "trainable_fp32_masters": all(p.requires_grad and p.dtype == torch.float32 and p.grad is None for p in model.parameters()),
        "training_modes_unchanged": training_modes == {name: module.training for name, module in model.named_modules()},
        "tied_readout_preserved": model.backbone.readout_weight is model.backbone.token_embeddings.weight}
    if not all(checks.values()):
        raise AssertionError("Adapted import failed a post-copy integrity guard")
    del payload, state, target
    gc.collect()
    return {"schema": "olmo-adapted-weights-import-v1", "scope": "Weights-only import into current runtime, not historical execution or training resume",
        "checkpoint_sha256": digest, "checkpoint_size_bytes": size, "checkpoint_path": str(path),
        "authority_kind": authority.kind, "report_path": authority.report_path, "report_sha256": authority.report_sha256,
        "retained_checkpoint": authority.checkpoint, "source_mapping": authority.source_mapping,
        "historical_configuration": config, "historical_model_configuration": construction,
        "historical_source_fingerprint": authority.source_fingerprint,
        "historical_counters_not_restored": authority.counters, "historical_cursor_not_restored": authority.data_cursor,
        "complete_state_pins": loaded, "imported_state_pins": source_pins,
        "frozen_native_and_buffer_digests": frozen, **current, "checks": checks,
        "optimizer_scheduler_rng": "Not restored or constructed; saved training flags and trainability ignored",
        "qualification": "O5c fusion-only adaptation inherited an already-adapted O5b backbone; current K4/jitter/NextLat fixture differs from historical K2/no-jitter/NextLat-off training"}
