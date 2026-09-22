"""O5c endpoint import and fusion-only adaptation over a frozen native stack.

No backbone, FBT forward, CE reduction or recurrence equation is modified.
Fresh AdamW state owns only the two learned fusion matrices. The inherited
pass-0-plus-pass-1 objective remains intact; pass0 naturally has no autograd
graph because its embedding and all native parameters are frozen.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping

import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import (CHECKPOINT_SCHEMA, parameter_layout, optimizer_ownership,
    build_warmup_scheduler, save_training_checkpoint, load_training_checkpoint)
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_fbt import FBTConfig, OLMoFBT
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
from scripts.olmo_lm_common import state_digests
from scripts.olmo_o5b_common import (ObservedFBTLM, SOURCE_FILES as O5B_SOURCES,
    make_mode, observed_step, microbatches, slice_batch, PilotTracker, preserve_rng)

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_SHA256 = "99585f5e9d666e8dea3f533749d155b0695b8143a6a313b60fb99f8d157faf66"
ENDPOINT_SIZE_BYTES = 14221991781
ENDPOINT_UPDATE = 2634
ENDPOINT_REPORT = ROOT/".runtime/olmo1b-step60000/o5b-pilot-01/fbt/report.json"
ENDPOINT_REPORT_SHA256 = "1e9de0339fcc86efa0eb7856a09d3a7c58e1dec65df9c4b53b696a7548569ab1"
FUSION_NAMES = ("backbone.fusion.state_proj.weight", "backbone.fusion.token_gate.weight")
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5c-fusion-only/"
SOURCE_FILES = tuple(sorted(set(O5B_SOURCES) | {
    "scripts/olmo_o5c_common.py", "scripts/olmo_o5c_train.py", "scripts/olmo_o5c_data.py",
    "scripts/olmo_o5c_preflight.py", "scripts/olmo_o5c_queue.py", "docs/reports/olmo1b-o5c/protocol.md",
}))


def source_hashes():
    return {name: sha256_file(ROOT/name) for name in SOURCE_FILES}


def plain_metadata(value):
    """Built-in JSON values, including conversion of legacy TorchVersion strings."""
    if is_dataclass(value):
        return plain_metadata(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Metadata mapping keys must be strings")
        return {str(key): plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_metadata(item) for item in value]
    if isinstance(value, (str, Path, torch.dtype)):
        return str(value)
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def endpoint_metadata(report_path=ENDPOINT_REPORT):
    """Only the completed retained O5b FBT endpoint is this pilot's starting point."""
    path = Path(report_path)
    if not path.is_file() or path.is_symlink() or sha256_file(path) != ENDPOINT_REPORT_SHA256:
        raise ValueError("O5b endpoint report differs from its immutable pin")
    report = json.loads(path.read_text())
    config = report["configuration"]
    if report.get("status") != "completed" or report.get("arm") != "fbt" or not report.get("finished_utc"):
        raise ValueError("Require the completed O5b FBT report")
    if report["counters"]["optimizer_updates"] != ENDPOINT_UPDATE or config["schedule"]["total_updates"] != ENDPOINT_UPDATE:
        raise ValueError("O5b endpoint update differs")
    records = [item for item in report["checkpoints"] if item["optimizer_updates"] == ENDPOINT_UPDATE]
    if len(records) != 1:
        raise ValueError("Require exactly one retained endpoint record")
    record = records[0]
    if (record.get("sha256") != ENDPOINT_SHA256 or record.get("size_bytes") != ENDPOINT_SIZE_BYTES
            or record.get("storage", {}).get("sha256") != ENDPOINT_SHA256
            or not record.get("storage", {}).get("generation")):
        raise ValueError("O5b endpoint checkpoint/retention identity differs")
    source = report["source_fingerprint"]
    if source.get("code") != config["source_hashes"]:
        raise ValueError("O5b endpoint configuration/source inventory differs")
    for name, digest in config["source_hashes"].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError(f"Frozen O5b source changed: {name}")
    return {"configuration": {**config, "arm": "fbt", "storage_prefix": report["storage_prefix"]},
            "source_fingerprint": plain_metadata(source), "checkpoint": record,
            "checkpoint_path": str(Path(record["path"]) if Path(record["path"]).is_absolute() else ROOT/record["path"]),
            "counters": report["counters"], "data_cursor": report["data_cursor"],
            "report_path": str(path), "report_sha256": ENDPOINT_REPORT_SHA256}


def freeze_native(model):
    """Clear stale gradients, freeze native/tied weights, train only the two gates."""
    if not isinstance(model, ObservedFBTLM) or model.enabled or model.predictor is not None or model.gamma != 1:
        raise ValueError("O5c requires the unchanged NextLat-disabled gamma1 observed FBT wrapper")
    parameters = dict(model.named_parameters())
    expected_native = {"backbone.backbone."+name for name, _ in model.backbone.backbone.named_parameters()}
    if set(parameters) != expected_native | set(FUSION_NAMES):
        raise ValueError("Unexpected parameter ownership in the fusion-only model")
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in FUSION_NAMES)
        parameter.grad = None
    return assert_fusion_only(model)


def assert_fusion_only(model):
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if set(names) != set(FUSION_NAMES) or len(names) != 2:
        raise ValueError("Exactly the two fusion matrices must be trainable")
    if model.backbone.readout_weight is not model.backbone.token_embeddings.weight:
        raise ValueError("Native tied embedding/readout ownership changed")
    width = model.backbone.config.model_dim
    parameters = dict(model.named_parameters())
    if any(parameters[name].shape != (width, width) for name in names):
        raise ValueError("Fusion parameters must be native-width square matrices")
    return [{"name": name, "shape": list(parameters[name].shape), "numel": parameters[name].numel()}
            for name in names]


def trainable_layout(model):
    """JSON-compatible list of name/shape/numel records for exactly two matrices."""
    return assert_fusion_only(model)


def frozen_state_digests(model):
    """Native tensors plus the fixed fusion output-scale buffer, excluding two matrices."""
    assert_fusion_only(model)
    return {name: digest for name, digest in state_digests(model).items() if name not in FUSION_NAMES}


def _load_weights_only(path):
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_frozen_endpoint(checkpoint, expected_sha256, *, expected_configuration,
                         expected_source_fingerprint, device="cuda"):
    """Strict model-only endpoint load; all O5b optimizer/scheduler/RNG is discarded.

    The caller gets configuration/source authority from endpoint_metadata().
    The explicit SHA argument also permits small independently hashed CPU test
    fixtures. The production driver must pass ENDPOINT_SHA256. Construction on
    meta followed by tensor assignment avoids reloading unrelated native weights.
    """
    path = Path(checkpoint)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)):
        raise ValueError("Expected checkpoint SHA256 must be 64 lowercase hex characters")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
        raise ValueError("Endpoint checkpoint bytes differ from expected SHA256")
    payload = _load_weights_only(path)
    config = plain_metadata(expected_configuration)
    source = plain_metadata(expected_source_fingerprint)
    if (payload.get("schema") != CHECKPOINT_SCHEMA or plain_metadata(payload.get("configuration")) != config
            or plain_metadata(payload.get("source_fingerprint")) != source):
        raise ValueError("Endpoint schema/configuration/source metadata differs")
    if (config.get("arm") != "fbt" or config.get("nextlat_enabled") is not False or config.get("rt_layers") != []
            or config.get("num_passes") != 2 or config.get("gamma") != 1):
        raise ValueError("Endpoint must be the FBT-only gamma1 K2 arm")
    if (payload.get("counters", {}).get("optimizer_updates") != config["schedule"]["total_updates"]
            or payload.get("counters", {}).get("input_tokens") != config["schedule"]["total_tokens"]
            or payload.get("data_cursor", {}).get("next_window") != config["schedule"]["used_windows"]):
        raise ValueError("Endpoint counters/cursor do not describe its completed schedule")
    backbone_config = OLMoConfig.from_dict(config["model_config"])
    base = OLMoTiledRTForCausalLM(backbone_config, attention_backend=config["attention_backend"],
        attention_precision=config["attention_precision"], device="meta", dtype=torch.float32)
    state = payload.get("model", {})
    native = {name.removeprefix("backbone.backbone."): value for name, value in state.items()
              if name.startswith("backbone.backbone.")}
    for name, template in base.state_dict().items():
        value = native.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != template.shape or value.dtype != torch.float32:
            raise ValueError(f"Endpoint native tensor geometry/dtype differs: {name}")
    base.load_state_dict(native, strict=True, assign=True)
    model = ObservedFBTLM(OLMoFBT(base, FBTConfig.from_dict(config["fusion_config"])),
        NextLatConfig.from_dict(config["nextlat_config"]), enabled=False, gamma=1)
    expected_class = type(model).__module__+"."+type(model).__qualname__
    if payload.get("model_type") != expected_class or payload.get("parameter_layout") != parameter_layout(model):
        raise ValueError("Endpoint model class/parameter layout differs")
    target = model.state_dict()
    if set(state) != set(target):
        raise ValueError("Endpoint model tensor keys differ")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != target[name].shape or value.dtype != target[name].dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Endpoint tensor shape/dtype/finiteness differs: {name}")
    if set(payload.get("module_training", {})) != set(dict(model.named_modules())):
        raise ValueError("Endpoint module ownership differs")
    model.load_state_dict(state, strict=True, assign=False)
    if float(model.backbone.fusion.output_scale) != config["fusion_output_scale"]:
        raise ValueError("Endpoint fixed fusion scale differs from its configuration")
    old_counters = plain_metadata(payload["counters"])
    del payload, state, native, target
    gc.collect()
    model.to(device).eval()
    trainable = freeze_native(model)
    hashes = state_digests(model)
    return model, {"checkpoint_sha256": expected_sha256, "checkpoint_size_bytes": path.stat().st_size,
        "endpoint_configuration": config, "endpoint_source_fingerprint": source,
        "endpoint_counters": old_counters, "trainable_parameters": trainable,
        "state_digests": hashes, "frozen_state_digests": {k: v for k, v in hashes.items() if k not in FUSION_NAMES},
        "optimizer_state": "fresh; endpoint optimizer/scheduler/counters/RNG not resumed"}


def build_optimizer(model, config=None, *, zero_lr=False):
    assert_fusion_only(model)
    config = {} if config is None else config
    lr, decay = config.get("lr", 1e-4), config.get("weight_decay", .1)
    betas, eps = tuple(config.get("betas", (.9, .95))), config.get("eps", 1e-8)
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{"params": [p for _, p in parameters], "param_names": [name for name, _ in parameters],
        "group_name": "fusion", "lr": 0.0 if zero_lr else lr, "weight_decay": decay}],
        lr=0.0 if zero_lr else lr, weight_decay=decay, betas=betas, eps=eps, foreach=False)
    optimizer_ownership(model, optimizer)
    return optimizer


def build_scheduler(optimizer, *, warmup_updates=50):
    if type(warmup_updates) is not int or warmup_updates < 1:
        raise ValueError("Fusion warmup must be a positive integer")
    return build_warmup_scheduler(optimizer, warmup_updates=warmup_updates)


def save_checkpoint(path, model, optimizer, *, configuration, source_fingerprint, data_cursor, **kwargs):
    assert_fusion_only(model)
    return save_training_checkpoint(path, model, optimizer, configuration=plain_metadata(configuration),
        source_fingerprint=plain_metadata(source_fingerprint), data_cursor=plain_metadata(data_cursor), **kwargs)


def load_checkpoint(path, model, optimizer, *, configuration, source_fingerprint, **kwargs):
    assert_fusion_only(model)
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        return load_training_checkpoint(path, model, optimizer, configuration=plain_metadata(configuration),
            source_fingerprint=plain_metadata(source_fingerprint), **kwargs)


def retain_file(path, uri, *, expected_sha256=None):
    """Create-only retention under the O5c lineage, with verified server checksums."""
    import base64
    from google.cloud import storage
    path = Path(path)
    if not uri.startswith(PREFIX_ROOT) or not path.is_file() or path.is_symlink():
        raise ValueError("O5c retention requires a regular local file and its designated GCS prefix")
    bucket, key = uri[5:].split("/", 1)
    if ".." in key.split("/"):
        raise ValueError("Invalid retention key")
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(16*1024**2):
            sha.update(chunk); md5.update(chunk)
    digest, size = sha.hexdigest(), path.stat().st_size
    encoded = base64.b64encode(md5.digest()).decode()
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("Checkpoint changed before retention")
    blob = storage.Client().bucket(bucket).blob(key)
    if not blob.exists():
        blob.metadata = {"sha256": digest, "artifact_schema": "olmo-o5c-fusion-only-v1"}
        blob.upload_from_filename(str(path), if_generation_match=0, timeout=1200, checksum="md5")
    blob.reload()
    if blob.size != size or blob.md5_hash != encoded or (blob.metadata or {}).get("sha256") != digest:
        raise ValueError("Existing remote object differs from selected local bytes")
    return {"uri": uri, "generation": str(blob.generation), "sha256": digest, "size_bytes": size,
            "md5_base64": encoded, "verification": "GCS generation, size, server MD5 and SHA256 metadata verified"}
