"""Ordinary native-backbone adaptation from the shared O5b source endpoint.

FBT/RT/NextLat stay off. The wrapper retains its unused fusion tensors for
provenance and checkpoint compatibility; only native parameters are optimized.
"""
from __future__ import annotations

from pathlib import Path

import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.lm_training import (optimizer_step, optimizer_ownership,
    build_warmup_scheduler, save_training_checkpoint, load_training_checkpoint)
from cdrm.pretrained.olmo_fbt import FBTMode
from cdrm.pretrained.recurrent import RTMode
from scripts.olmo_o5b_common import ObservedFBTLM
from scripts.olmo_o5c_common import (ENDPOINT_SHA256, FUSION_NAMES, SOURCE_FILES as O5C_SOURCES,
    endpoint_metadata, load_frozen_endpoint, plain_metadata)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = tuple(sorted(set(O5C_SOURCES) | {
    "scripts/olmo_o5e_common.py", "scripts/olmo_o5e_preflight.py", "scripts/olmo_o5e_train.py",
    "scripts/olmo_o5d_common.py",
    "docs/reports/olmo1b-o5e/protocol.md",
}))
FROZEN_NAMES = (*FUSION_NAMES, "backbone.fusion.output_scale")
PREFIX_ROOT = "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo1b-o5e-ordinary-control/"


def source_hashes():
    return {name: sha256_file(ROOT/name) for name in SOURCE_FILES}


def make_mode():
    """One ordinary pass, no feedback, no temporal RT layer selection."""
    return FBTMode(enabled=False, num_passes=1, beta=0., rt_mode=RTMode(()))


def _native_names(model):
    if (not isinstance(model, ObservedFBTLM) or model.enabled or model.predictor is not None
            or model.gamma != 1):
        raise ValueError("Ordinary control requires the unchanged NextLat-disabled gamma1 wrapper")
    names = {"backbone.backbone."+name for name, _ in model.backbone.backbone.named_parameters()}
    if (set(dict(model.named_parameters())) != names | set(FUSION_NAMES)
            or len(names) != 4*model.backbone.config.num_layers+1):
        raise ValueError("Unexpected native/fusion parameter ownership")
    return names


def configure_trainable(model):
    """Unfreeze native matrices including tied readout, clear all stale gradients."""
    native = _native_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in native)
        parameter.grad = None
    model.backbone.fusion.output_scale.requires_grad_(False)
    model.backbone.fusion.output_scale.grad = None
    return assert_native_only(model)


def assert_native_only(model):
    native = _native_names(model)
    parameters = dict(model.named_parameters())
    if {name for name, p in parameters.items() if p.requires_grad} != native:
        raise ValueError("Exactly native backbone parameters must be trainable")
    if model.backbone.readout_weight is not model.backbone.token_embeddings.weight:
        raise ValueError("Native embedding/readout tying differs")
    if any(p.dtype != torch.float32 or p.ndim != 2 for p in parameters.values()):
        raise ValueError("Control requires FP32 native/fusion master matrices")
    if model.backbone.fusion.output_scale.requires_grad:
        raise ValueError("The fixed fusion scale must not require gradients")
    return [{"name": name, "shape": list(p.shape), "numel": p.numel()}
            for name, p in parameters.items() if name in native]


def trainable_layout(model):
    return assert_native_only(model)


def frozen_fusion_digests(model):
    assert_native_only(model)
    values = model.backbone.fusion.state_dict()
    # Hash only three small frozen tensors; do not copy the 1.177B trainable stack.
    from scripts.olmo_lm_common import tensor_digest
    hashes = {"backbone.fusion."+name: tensor_digest(value) for name, value in values.items()}
    if set(hashes) != set(FROZEN_NAMES):
        raise ValueError("Unexpected frozen fusion state")
    return hashes


def load_control(*, device="cuda", endpoint=None):
    """Import model state only; defaults to the immutable completed O5b endpoint.

    Explicit independent endpoint metadata and CPU are available for tiny tests.
    Production cannot silently fall back to CPU. Fresh optimizer construction is
    separate and never copies the endpoint's moments, scheduler, counters or RNG.
    """
    endpoint = endpoint_metadata() if endpoint is None else endpoint
    record = endpoint["checkpoint"]
    device = torch.device(device)
    if device.type not in ("cuda", "cpu") or (record["sha256"] == ENDPOINT_SHA256 and device.type != "cuda"):
        raise ValueError("Production control requires the GPU container; CPU is for tiny fixtures only")
    path = Path(endpoint["checkpoint_path"])
    if not path.is_file() or path.is_symlink() or path.stat().st_size != record["size_bytes"]:
        raise ValueError("Control source endpoint size differs")
    model, provenance = load_frozen_endpoint(path, record["sha256"],
        expected_configuration=endpoint["configuration"],
        expected_source_fingerprint=endpoint["source_fingerprint"], device=device)
    native = configure_trainable(model)
    frozen = frozen_fusion_digests(model)
    return model, {**provenance, "trainable_parameters": native,
        "source_native_state_digests": provenance["frozen_state_digests"],
        "frozen_state_digests": frozen, "frozen_fusion_digests": frozen,
        "optimizer_state": "fresh optimizer required; source moments/scheduler/counters/RNG not restored",
        "training_mode": "ordinary single pass; FBT/RT/NextLat off"}


def build_optimizer(model, config=None, *, zero_lr=False):
    assert_native_only(model)
    config = {} if config is None else config
    pairs = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    lr = 0. if zero_lr else config.get("lr", 1e-5)
    optimizer = torch.optim.AdamW([{"params": [p for _, p in pairs], "param_names": [n for n, _ in pairs],
        "group_name": "native", "lr": lr, "weight_decay": config.get("weight_decay", .1)}],
        lr=lr, betas=tuple(config.get("betas", (.9, .95))), eps=config.get("eps", 1e-8), foreach=False)
    optimizer_ownership(model, optimizer)
    return optimizer


def build_scheduler(optimizer, *, warmup_updates=50):
    if type(warmup_updates) is not int or warmup_updates < 1:
        raise ValueError("Ordinary-control warmup requires a positive integer update count")
    return build_warmup_scheduler(optimizer, warmup_updates=warmup_updates)


def observed_step(model, optimizer, batches, **kwargs):
    """Observe the unchanged single-CE optimizer step; reject recurrent modes."""
    assert_native_only(model)
    options = dict(kwargs.pop("backbone_kwargs", {}) or {})
    if options.get("mode", make_mode()) != make_mode():
        raise ValueError("Ordinary control permits only the single-pass FBT/RT-disabled mode")
    options["mode"] = make_mode()
    model._observations = []
    norms = {}
    def observe(opt, args, kw):
        for group in opt.param_groups:
            values = [torch.linalg.vector_norm(p.grad.detach().float())
                      for p in group["params"] if p.grad is not None]
            norms[group["group_name"]] = float(torch.linalg.vector_norm(torch.stack(values))) if values else 0.
    hook = optimizer.register_step_pre_hook(observe)
    try:
        result = optimizer_step(model, optimizer, batches, backbone_kwargs=options, **kwargs)
        sums = model._observations
        if not sums or any(len(row) != 1 for row in sums):
            raise AssertionError("Ordinary control requires exactly one CE pass per microbatch")
        result["pass_ce_means"] = [sum(row[0] for row in sums)/max(result["counts"]["ce"], 1)]
        result["group_gradient_norm_after_clip"] = norms
        result["stack_input_tokens"] = sum(int(batch.valid_mask.sum()) for batch in batches)
        return result
    finally:
        hook.remove()
        model._observations = None


def save_checkpoint(path, model, optimizer, *, configuration, source_fingerprint, data_cursor, **kwargs):
    assert_native_only(model)
    return save_training_checkpoint(path, model, optimizer, configuration=plain_metadata(configuration),
        source_fingerprint=plain_metadata(source_fingerprint), data_cursor=plain_metadata(data_cursor), **kwargs)


def load_checkpoint(path, model, optimizer, *, configuration, source_fingerprint, **kwargs):
    assert_native_only(model)
    restored = load_training_checkpoint(path, model, optimizer, configuration=plain_metadata(configuration),
        source_fingerprint=plain_metadata(source_fingerprint), **kwargs)
    assert_native_only(model)
    return restored


def retain_file(path, uri, *, expected_sha256=None):
    """Use the existing create-only byte verifier within the new O5e lineage."""
    from scripts.olmo_o4_common import retain_file as retain_verified_file
    if not isinstance(uri, str) or not uri.startswith(PREFIX_ROOT):
        raise ValueError("O5e retention requires its designated GCS prefix")
    return retain_verified_file(Path(path), uri, expected_sha256=expected_sha256)
