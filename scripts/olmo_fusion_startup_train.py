#!/usr/bin/env python3
"""Bounded FP32 fusion-only startup from original OLMo, with compact recovery.

The native backbone, tied readout, predictor and fusion output_scale stay fixed.
Later backbone passes retain autograd into the two trainable fusion matrices.
K4/beta1/jitter0.02 uses campaign CE weighting; latent/KL branches are omitted
only from the warmup objective. Separate diagnostics retain their full contract.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.fbt_training import aggregate_pass_losses
from cdrm.pretrained.lm_training import (CHECKPOINT_SCHEMA, TrainingCounters, _rng_state,
    build_adamw, build_warmup_scheduler, load_training_checkpoint, optimizer_ownership,
    parameter_layout, save_training_checkpoint)
from cdrm.pretrained.nextlat import _validate_batch, build_nextlat_masks, compute_nextlat_loss_sums
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_crossed_precision import source_hashes as prior_sources
from scripts.olmo_campaign_ddp_probe import construct
from scripts.olmo_campaign_fp32_localize import configure_full_fp32
from scripts.olmo_campaign_recurrence_precision import arm_contract, state_pins
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_data import StartupData, DEFAULT_ROOT, DEFAULT_MANIFEST_SHA256, source_hashes as data_sources
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu
from scripts.openelm_retain import file_digest
from scripts.olmo_two_gpu_retain import upload_verified

KIND = "olmo-fusion-startup-v1"
FUSION_NAMES = ("backbone.fusion.state_proj.weight", "backbone.fusion.token_gate.weight")
TRAINING = {"total_updates": 128, "ce_targets_per_update": 8192, "lr": 1e-4,
    "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1, "max_grad_norm": 1.,
    "warmup_updates": 16, "precision": "fp32", "foreach": False, "fused": False,
    "checkpoint_interval_seconds": 600}


def source_hashes():
    result = prior_sources()
    for name, digest in data_sources().items():
        if name in result and result[name] != digest:
            raise ValueError("Data/runtime source inventories disagree")
        result[name] = digest
    for name in ("scripts/olmo_fusion_startup_train.py", "tests/test_fusion_startup_train.py",
                 "scripts/openelm_retain.py", "scripts/olmo_two_gpu_retain.py",
                 "docs/reports/olmo-fusion-startup/protocol.md"):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def assert_fusion_only(model):
    parameters = dict(model.named_parameters())
    active = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if active != set(FUSION_NAMES) or any(p.dtype != torch.float32 for p in parameters.values()):
        raise ValueError("Startup requires exactly two trainable FP32 fusion matrices")
    if model.backbone.readout_weight is not model.backbone.token_embeddings.weight:
        raise ValueError("Startup requires the original tied readout/embedding")
    if any(p.grad is not None for n, p in parameters.items() if n not in FUSION_NAMES):
        raise ValueError("Frozen parameters unexpectedly accumulated gradients")
    return [parameters[name] for name in FUSION_NAMES]


def freeze_for_startup(model):
    if (not model.enabled or model.predictor is None or model.pass_loss_policy != "campaign_v1"
            or model.gamma != 1 or model.config.document_policy != "isolated-v1"):
        raise ValueError("Startup retains the current isolated campaign NF model")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in FUSION_NAMES)
        parameter.grad = None
    assert_fusion_only(model)


def frozen_state_pins(model):
    """Includes full native/predictor state and the unchanged fusion scale."""
    return tree_digests({name: value for name, value in model.state_dict().items() if name not in FUSION_NAMES})


def build_optimizer(model):
    assert_fusion_only(model)
    optimizer = build_adamw(model.backbone.fusion, lr=TRAINING["lr"], betas=tuple(TRAINING["betas"]),
        eps=TRAINING["eps"], weight_decay=TRAINING["weight_decay"], foreach=False, fused=False)
    scheduler = build_warmup_scheduler(optimizer, warmup_updates=TRAINING["warmup_updates"])
    return optimizer, scheduler


def ce_loss_sums(model, recipe, batch, noise):
    """Original FBT/CE functions, with no auxiliary projection work or detaches."""
    mode = recipe.mode()
    if (recipe.arm != "NF" or not mode.enabled or mode.num_passes != 4 or mode.beta != 1
            or mode.feedback_jitter != .02 or mode.rt_mode.selected_layers
            or mode.document_policy != "isolated-v1"):
        raise ValueError("Warmup requires original NF K4 beta1 jitter0.02 without RT")
    _validate_batch(batch, one_document_per_row=True)
    embeddings = model.backbone.token_embeddings(batch.input_ids)
    output = model.backbone(inputs_embeds=embeddings, attention_mask=batch.valid_mask,
        document_ids=batch.document_ids, return_logits=False, mode=mode,
        feedback_noise=noise, right_padded_causal=True)
    if len(output.pass_hidden_states) != 4:
        raise AssertionError("Warmup lost a feedback pass")
    losses = [compute_nextlat_loss_sums(hidden, embeddings, model.backbone.readout_weight,
        batch, None, model.config, enabled=False) for hidden in output.pass_hidden_states]
    return aggregate_pass_losses(losses, gamma=model.gamma, pass_loss_policy=model.pass_loss_policy)


def ce_backward(model, recipe, batches, noises, *, ce_targets):
    assert_fusion_only(model)
    if len(batches) != len(noises) or not batches or type(ce_targets) is not int or ce_targets < 1:
        raise ValueError("Need a nonempty matched batch/noise update with positive global CE count")
    actual = sum(int(build_nextlat_masks(batch, document_policy="isolated-v1")["ce"].sum()) for batch in batches)
    if actual != ce_targets:
        raise ValueError("Update CE mask count differs from its fixed global denominator")
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    metrics = {"objective": 0., "ce_sum": 0., "ce_targets": actual, "microbatches": len(batches),
        "documents": sum(int(batch.valid_mask.any(-1).sum()) for batch in batches),
        "input_tokens": sum(int(batch.valid_mask.sum()) for batch in batches), "pass_ce_sums": [0.]*4}
    for batch, noise in zip(batches, noises):
        local = batch.to(device)
        local_noise = tuple(value.to(device) for value in noise)
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device.type, enabled=False):
            result = ce_loss_sums(model, recipe, local, local_noise)
            objective = result.sums["ce"] / ce_targets
        if not bool(torch.isfinite(objective)) or not objective.requires_grad:
            raise FloatingPointError("CE warmup loss is nonfinite or detached from fusion")
        objective.backward()
        metrics["objective"] += float(objective.detach())
        metrics["ce_sum"] += float(result.sums["ce"].detach())
        for index, loss in enumerate(result.pass_losses):
            metrics["pass_ce_sums"][index] += float(loss.sums["ce"].detach())
        del result, objective, local, local_noise
    parameters = assert_fusion_only(model)
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError("Fusion gradients are missing or nonfinite")
    metrics["fusion_gradient_norms"] = {name: float(parameter.grad.double().norm())
        for name, parameter in zip(FUSION_NAMES, parameters)}
    if not all(value > 0 for value in metrics["fusion_gradient_norms"].values()):
        raise FloatingPointError("A trainable fusion matrix received zero gradient")
    return metrics


def train_update(model, recipe, batches, noises, optimizer, scheduler, counters, *, ce_targets):
    parameters = assert_fusion_only(model)
    optimizer_ownership(model.backbone.fusion, optimizer)
    lr = [group["lr"] for group in optimizer.param_groups]
    before = [parameter.detach().clone() for parameter in parameters]
    try:
        metrics = ce_backward(model, recipe, batches, noises, ce_targets=ce_targets)
        norm = torch.nn.utils.clip_grad_norm_(parameters, TRAINING["max_grad_norm"], error_if_nonfinite=True, foreach=False)
        optimizer.step()
        scheduler.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise FloatingPointError("Fusion update became nonfinite")
        if any(not bool(torch.isfinite(value).all()) for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor)):
            raise FloatingPointError("Fusion Adam state became nonfinite")
        changes = {name: float((parameter.detach()-old).double().norm())
                   for name, parameter, old in zip(FUSION_NAMES, parameters, before)}
        if not all(value > 0 for value in changes.values()):
            raise FloatingPointError("A fusion matrix did not change after an optimizer update")
    finally:
        model.zero_grad(set_to_none=True)
    counters.optimizer_updates += 1
    counters.ce_positions += ce_targets
    for name in ("microbatches", "documents", "input_tokens"):
        setattr(counters, name, getattr(counters, name)+metrics[name])
    return {**metrics, "update": counters.optimizer_updates, "gradient_norm_before_clip": float(norm),
        "max_grad_norm": TRAINING["max_grad_norm"], "lr_used": lr, "lr_next": [g["lr"] for g in optimizer.param_groups],
        "fusion_update_norms": changes, "counters": asdict(counters)}


def checkpoint_configuration(model, recipe, source_checkpoint, *, data_manifest, data_manifest_sha256,
                             sources, determinism, runtime, execution, initial_contract):
    return {"kind": KIND, "training": dict(TRAINING), "source_checkpoint": source_checkpoint,
        "model_config": model.backbone.config.to_dict(), "fusion_config": model.backbone.fusion_config.to_dict(),
        "nextlat_config": model.config.to_dict(), "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256,
        "initial_full_trainability_contract": initial_contract,
        "data_manifest": data_manifest, "data_manifest_sha256": data_manifest_sha256,
        "sources": sources, "determinism": determinism, "runtime": runtime, "execution": execution,
        "frozen_state_pins": frozen_state_pins(model),
        "initial_fusion_pins": tree_digests(model.backbone.fusion.state_dict()),
        "full_module_training": {name: module.training for name, module in model.named_modules()},
        "training_parameter_layout": parameter_layout(model),
        "objective": "CE only; original NF wrapper and predictor retained, latent/KL arithmetic omitted",
        "ce_pass_weights": [.5, 1/6, 1/6, 1/6]}


def _checkpoint_payload(model, path, expected_source_checkpoint, *, expected_sha256):
    path = Path(path)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)
            or not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256):
        raise ValueError("Compact checkpoint differs from its independent SHA256 pin")
    before = path.stat()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Compact checkpoint changed while loading")
    config = payload.get("configuration", {})
    expected_class = type(model.backbone.fusion).__module__+"."+type(model.backbone.fusion).__qualname__
    if (payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("model_type") != expected_class
            or config.get("kind") != KIND or config.get("source_checkpoint") != tree_digests(expected_source_checkpoint)
            or config.get("model_config") != model.backbone.config.to_dict()
            or config.get("fusion_config") != model.backbone.fusion_config.to_dict()
            or config.get("nextlat_config") != model.config.to_dict()
            or config.get("training") != TRAINING or config.get("ce_pass_weights") != [.5, 1/6, 1/6, 1/6]
            or model.pass_loss_policy != "campaign_v1" or not model.enabled or model.predictor is None
            or model.backbone.readout_weight is not model.backbone.token_embeddings.weight):
        raise ValueError("Compact checkpoint source/model/training authority differs")
    if config.get("frozen_state_pins") != frozen_state_pins(model):
        raise ValueError("Original frozen native/predictor/scale state differs")
    saved_recipe = config.get("recipe", {})
    recipe = CampaignRecipe(**{field.name: saved_recipe[field.name] for field in fields(CampaignRecipe)
                              if field.name in saved_recipe})
    mode = recipe.mode()
    if (recipe.to_dict() != saved_recipe or recipe.sha256 != config.get("recipe_sha256")
            or recipe.arm != "NF" or not mode.enabled or mode.num_passes != 4 or mode.beta != 1
            or mode.feedback_jitter != .02 or mode.rt_mode.selected_layers or mode.document_policy != "isolated-v1"
            or recipe.predictor_seed != model.config.seed or recipe.fusion_seed != model.backbone.fusion_config.seed):
        raise ValueError("Compact checkpoint NF recipe/hash/mode differs")
    fingerprint = payload.get("source_fingerprint", {})
    if (fingerprint.get("checkpoint_sha256") != expected_source_checkpoint.get("sha256")
            or fingerprint.get("code") != config.get("sources")
            or fingerprint.get("data_manifest_sha256") != config.get("data_manifest_sha256")):
        raise ValueError("Compact checkpoint source/configuration metadata disagree")
    state, target = payload.get("model", {}), model.backbone.fusion.state_dict()
    if (payload.get("parameter_layout") != parameter_layout(model.backbone.fusion)
            or payload.get("optimizer_ownership") != [["state_proj.weight", "token_gate.weight"]]
            or set(payload.get("module_training", {})) != set(dict(model.backbone.fusion.named_modules()))
            or any(type(v) is not bool for v in payload["module_training"].values())):
        raise ValueError("Compact fusion ownership or module inventory differs")
    if set(state) != set(target) or set(state) != {"state_proj.weight", "token_gate.weight", "output_scale"}:
        raise ValueError("Compact checkpoint must contain exactly the complete fusion state")
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != target[name].shape or tensor.dtype != target[name].dtype
                or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())):
            raise ValueError("Compact fusion tensor geometry/dtype/finiteness differs")
    if tree_digests(state["output_scale"]) != tree_digests(target["output_scale"]):
        raise ValueError("Frozen fusion scale changed in the compact state")
    counters = TrainingCounters(**payload["counters"])
    cursor = payload.get("data_cursor", {})
    if (counters.optimizer_updates > TRAINING["total_updates"]
            or counters.ce_positions != counters.optimizer_updates*TRAINING["ce_targets_per_update"]
            or counters.latent_pairs or counters.kl_triples
            or cursor.get("next_update") != counters.optimizer_updates
            or cursor.get("manifest_sha256") != config.get("data_manifest_sha256")):
        raise ValueError("Compact counters/cursor differ from the fixed CE schedule")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Compact checkpoint load requires cleared model gradients")
    return payload


def load_fusion_checkpoint(model, path, expected_source_checkpoint, *, expected_sha256,
                           configuration=None, source_fingerprint=None, optimizer=None, scheduler=None):
    """Load complete fusion only, optionally restoring its exact training state.

Without optimizer: preserve full diagnostic trainability/modes/RNG and omit all
optimizer/scheduler/RNG restoration. Source/checkpoint authority and every
frozen state byte still have to match. No native/predictor bytes are loaded.
"""
    payload = _checkpoint_payload(model, path, expected_source_checkpoint, expected_sha256=expected_sha256)
    config = payload["configuration"]
    identities = {name: id(p) for name, p in model.named_parameters()}
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    modes = {name: m.training for name, m in model.named_modules()}
    if optimizer is None:
        if scheduler is not None:
            raise ValueError("A scheduler restore requires its optimizer")
        rng = tree_digests(_rng_state(None))
        model.backbone.fusion.load_state_dict(payload["model"], strict=True, assign=False)
        if tree_digests(_rng_state(None)) != rng:
            raise AssertionError("Weights-only compact import changed caller RNG")
    else:
        assert_fusion_only(model)
        if configuration is None or source_fingerprint is None:
            raise ValueError("Exact resume needs explicit current configuration and source fingerprint")
        # Old generic validator handles Adam ownership/dtypes, scheduler identity,
        # RNG topology and strict source/config equality before any mutation.
        load_training_checkpoint(path, model.backbone.fusion, optimizer, scheduler=scheduler,
            configuration=configuration, source_fingerprint=source_fingerprint, expected_sha256=expected_sha256)
        if {name: m.training for name, m in model.named_modules()} != config["full_module_training"]:
            raise AssertionError("Resume module training modes differ from the original configuration")
    checks = {"complete_fusion_exact": tree_digests(model.backbone.fusion.state_dict()) == tree_digests(payload["model"]),
        "parameter_identities_preserved": identities == {name: id(p) for name, p in model.named_parameters()},
        "trainability_preserved": flags == {name: p.requires_grad for name, p in model.named_parameters()},
        "module_modes_preserved": modes == {name: m.training for name, m in model.named_modules()},
        "frozen_state_exact": frozen_state_pins(model) == config["frozen_state_pins"],
        "tied_readout_preserved": model.backbone.readout_weight is model.backbone.token_embeddings.weight}
    if not all(checks.values()):
        raise AssertionError("Compact checkpoint import violated state/ownership guarantees")
    return {"checkpoint_sha256": expected_sha256, "configuration": config,
        "source_fingerprint": payload["source_fingerprint"], "counters": payload["counters"],
        "data_cursor": payload["data_cursor"], "fusion_state_pins": tree_digests(payload["model"]),
        "restore_scope": "fusion+Adam+scheduler+RNG+cursor" if optimizer is not None else "fusion weights only; diagnostic flags and RNG preserved",
        "checks": checks}


def save_fusion_checkpoint(path, model, optimizer, scheduler, counters, *, configuration, source_fingerprint, data_cursor):
    assert_fusion_only(model)
    if frozen_state_pins(model) != configuration["frozen_state_pins"]:
        raise AssertionError("Frozen backbone/predictor/scale changed before checkpoint")
    if (data_cursor.get("next_update") != counters.optimizer_updates
            or counters.ce_positions != counters.optimizer_updates*TRAINING["ce_targets_per_update"]):
        raise ValueError("Save cursor/CE counters do not describe a complete update")
    return save_training_checkpoint(path, model.backbone.fusion, optimizer, scheduler=scheduler,
        counters=counters, data_cursor=data_cursor, configuration=configuration, source_fingerprint=source_fingerprint)


def boundary_digests(model, optimizer, scheduler, counters, cursor):
    return tree_digests({"fusion": model.backbone.fusion.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "counters": asdict(counters), "cursor": cursor, "rng": _rng_state(None)})


def retain_checkpoint(path, prefix, expected_sha256):
    if not re.fullmatch(r"gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/olmo-fusion-startup/[0-9]{8}T[0-9]{6}Z/[A-Za-z0-9][A-Za-z0-9_.-]*/?", prefix):
        raise ValueError("Checkpoint storage must use the declared fusion-startup lineage")
    path = Path(path)
    digest = file_digest(path)
    if digest["sha256"] != expected_sha256:
        raise ValueError("Checkpoint changed before GCS retention")
    from google.cloud import storage
    bucket_name, key = prefix.rstrip("/")[5:].split("/", 1)
    with preserve_local_rng():
        return upload_verified(storage.Client().bucket(bucket_name), key+"/"+path.name,
            path, digest, download_sha256=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--length", type=int, choices=(128, 256), default=128)
    parser.add_argument("--physical-batch-size", type=int, default=8)
    parser.add_argument("--max-updates", type=int, default=128, help="Segment stopping point; fixed training plan stays 128")
    parser.add_argument("--preflight", action="store_true", help="Two data updates plus exact update2 replay from compact update1")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--storage-prefix", help="Create-only GCS checkpoint upload, including generation-pinned download verification")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.length, args.physical_batch_size) not in ((128, 8), (256, 4)):
        parser.error("Predeclared physical layouts are T128/B8 and T256/B4")
    if not 1 <= args.max_updates <= 128 or (args.preflight and (args.max_updates != 2 or args.resume)):
        parser.error("Use --preflight --max-updates2 without resume, or a segment endpoint1..128")
    if (args.resume is None) != (args.resume_sha256 is None):
        parser.error("Resume requires path and independent SHA256")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error("Evidence/checkpoints must remain on persistent project storage")
    return args


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Fusion startup is a one-GPU non-DDP warmup")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    for name in sources:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    report = {"schema": "olmo-fusion-startup-run-v1", "status": "running", "passed": False,
        "scope": __doc__, "sources": sources, "runtime": runtime, "determinism": determinism,
        "started_utc": datetime.now(timezone.utc).isoformat(), "max_updates_this_segment": args.max_updates,
        "preflight": args.preflight, "updates": [], "checkpoints": [], "physical_optimizer_updates": 0,
        "qualification": "Bounded fusion startup, not BF16 clearance, production mixture or quality evaluation"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-fusion-startup", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None
    stop_requested = []
    previous_signals = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous_signals:
        signal.signal(sig, lambda number, frame: stop_requested.append(number))

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({"scope": __doc__, "training": TRAINING, "physical_length": args.length,
            "physical_batch_size": args.physical_batch_size, "preflight": args.preflight})
        persist("construct_original_nf")
        model, recipe, source_checkpoint, _, _ = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NF", torch.device("cuda"))
        initial_contract, cold = arm_contract(model, recipe), state_pins(model)
        execution = configure_full_fp32(model)
        freeze_for_startup(model)
        data = StartupData.from_prepared(args.data_root, expected_manifest_sha256=args.data_manifest_sha256,
            length=args.length, ce_per_update=TRAINING["ce_targets_per_update"], physical_batch_size=args.physical_batch_size)
        if data.total_updates < TRAINING["total_updates"]:
            raise ValueError("Prepared training data is insufficient for the fixed 128-update budget")
        configuration = checkpoint_configuration(model, recipe, source_checkpoint,
            data_manifest=data.manifest, data_manifest_sha256=data.manifest_sha256,
            sources=sources, determinism=determinism, runtime=runtime, execution=execution, initial_contract=initial_contract)
        source = {"checkpoint_sha256": source_checkpoint["sha256"], "code": sources,
            "data_manifest_sha256": data.manifest_sha256}
        optimizer, scheduler = build_optimizer(model)
        counters = TrainingCounters()
        report.update(configuration=configuration, source_fingerprint=source, initial_full_state=cold)
        if args.resume:
            restored = load_fusion_checkpoint(model, args.resume, source_checkpoint, expected_sha256=args.resume_sha256,
                configuration=configuration, source_fingerprint=source, optimizer=optimizer, scheduler=scheduler)
            counters = TrainingCounters(**restored["counters"])
            if data.restore_cursor(restored["data_cursor"]) != counters.optimizer_updates:
                raise ValueError("Restored data cursor differs from checkpoint counters")
            report["resume"] = restored
        if counters.optimizer_updates >= args.max_updates:
            raise ValueError("Requested segment has no remaining updates")
        last_save = time.monotonic()

        def checkpoint(reason):
            nonlocal last_save
            update = counters.optimizer_updates
            previous = next((r for r in report["checkpoints"] if r["optimizer_updates"] == update), None)
            if previous is not None:
                return previous
            cursor = data.cursor(update)
            record = save_fusion_checkpoint(args.output_dir/f"update-{update:06d}.pt", model, optimizer, scheduler, counters,
                configuration=configuration, source_fingerprint=source, data_cursor=cursor)
            record.update(reason=reason, data_cursor=cursor, boundary_digests=boundary_digests(model, optimizer, scheduler, counters, cursor))
            report["checkpoints"].append(record)
            report["counters"] = asdict(counters)
            report["data_cursor"] = cursor
            last_save = time.monotonic()
            persist("checkpoint_"+str(update))
            if args.storage_prefix:
                record["storage"] = retain_checkpoint(record["path"], args.storage_prefix, record["sha256"])
                write_json(Path(record["path"]).with_suffix(".receipt.json"), record)
                persist("checkpoint_"+str(update)+"/retained")
            print({"checkpoint": record["path"], "sha256": record["sha256"], "update": update, "reason": reason}, flush=True)
            return record

        if not args.resume:
            checkpoint("initial_cold_state")
        else:
            report["resumed_boundary"] = boundary_digests(model, optimizer, scheduler, counters, data.cursor(counters.optimizer_updates))
        preflight_one = None
        while counters.optimizer_updates < args.max_updates:
            if stop_requested or (args.stop_file and args.stop_file.exists()):
                checkpoint("requested_stop")
                report.update(status="paused_at_boundary", passed=True)
                break
            logical_update = counters.optimizer_updates
            batches, noises = data.update_batches(logical_update, recipe, model.config.model_dim, device="cpu")
            input_pins = tree_digests({"batches": [vars(b) for b in batches], "noise": noises})
            planned = data.update_metadata(logical_update)
            persist("update_"+str(logical_update+1)+"/backward")
            torch.cuda.reset_peak_memory_stats()
            update_started = time.monotonic()
            metrics = train_update(model, recipe, batches, noises, optimizer, scheduler, counters,
                ce_targets=TRAINING["ce_targets_per_update"])
            elapsed = time.monotonic()-update_started
            report["physical_optimizer_updates"] += 1
            if (metrics["ce_targets"] != planned["counts"]["ce"]
                    or any(metrics[k] != planned[k] for k in ("microbatches", "documents", "input_tokens"))
                    or tree_digests({"batches": [vars(b) for b in batches], "noise": noises}) != input_pins):
                raise AssertionError("Training update counts or input/noise bytes changed")
            row = {**metrics, "elapsed_seconds": elapsed, "input_tokens_per_second": metrics["input_tokens"]/elapsed,
                "ce_targets_per_second": metrics["ce_targets"]/elapsed, "memory": memory(),
                "input_pins": input_pins, "data_metadata": planned}
            report["updates"].append(row)
            report.update(counters=asdict(counters), data_cursor=data.cursor(counters.optimizer_updates),
                estimated_128_update_seconds=128*sum(r["elapsed_seconds"] for r in report["updates"])/len(report["updates"]))
            tracker.log({"update": counters.optimizer_updates, **scalar_metrics(row, "train")}, step=counters.optimizer_updates)
            persist("update_"+str(counters.optimizer_updates)+"/complete")
            print({"update": counters.optimizer_updates, "ce": metrics["objective"], "seconds": elapsed,
                "raw_norm": metrics["gradient_norm_before_clip"], "estimated_128_seconds": report["estimated_128_update_seconds"]}, flush=True)
            if args.preflight and counters.optimizer_updates == 1:
                preflight_one = checkpoint("preflight_restart_boundary")
            if args.preflight and counters.optimizer_updates == 2:
                expected = boundary_digests(model, optimizer, scheduler, counters, data.cursor(2))
                restored = load_fusion_checkpoint(model, preflight_one["path"], source_checkpoint,
                    expected_sha256=preflight_one["sha256"], configuration=configuration,
                    source_fingerprint=source, optimizer=optimizer, scheduler=scheduler)
                counters = TrainingCounters(**restored["counters"])
                next_update = data.restore_cursor(restored["data_cursor"])
                replay_batches, replay_noise = data.update_batches(next_update, recipe, model.config.model_dim, device="cpu")
                replay_inputs = tree_digests({"batches": [vars(b) for b in replay_batches], "noise": replay_noise})
                replay = train_update(model, recipe, replay_batches, replay_noise, optimizer, scheduler, counters,
                    ce_targets=TRAINING["ce_targets_per_update"])
                report["physical_optimizer_updates"] += 1
                actual = boundary_digests(model, optimizer, scheduler, counters, data.cursor(2))
                checks = {"boundary_bitwise_exact": actual == expected,
                    "metrics_bitwise_exact": tree_digests(replay) == tree_digests(metrics),
                    "input_noise_bitwise_exact": replay_inputs == input_pins,
                    "frozen_state_exact": frozen_state_pins(model) == configuration["frozen_state_pins"]}
                report["preflight_recovery"] = {"checks": checks, "uninterrupted_boundary": expected,
                    "replayed_boundary": actual, "scope": "Same-process reload; fresh-process comparison remains separate"}
                if not all(checks.values()):
                    raise AssertionError("Preflight checkpoint continuation differs")
            if (counters.optimizer_updates in (32, 128, args.max_updates)
                    or time.monotonic()-last_save >= TRAINING["checkpoint_interval_seconds"]):
                checkpoint("segment_boundary" if counters.optimizer_updates == args.max_updates else "scheduled_recovery")
            del batches, noises
        integrity = {"frozen_state_exact": frozen_state_pins(model) == configuration["frozen_state_pins"],
            "source_files_unchanged": source_hashes() == sources,
            "data_cursor_exact": data.restore_cursor(data.cursor(counters.optimizer_updates)) == counters.optimizer_updates,
            "cleared_gradients": all(p.grad is None for p in model.parameters()),
            "optimizer_owns_only_fusion": optimizer_ownership(model.backbone.fusion, optimizer) == [["state_proj.weight", "token_gate.weight"]]}
        report["integrity"] = integrity
        if not all(integrity.values()):
            raise AssertionError("Fusion startup final integrity failed")
        if report["status"] != "paused_at_boundary":
            report.update(status="completed_segment", passed=True)
        persist("complete")
    except BaseException as error:
        failure = error
        report.update(status="failed", passed=False,
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist(report.get("stage", "setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed", passed=False, tracking_finish_error={"type": type(error).__name__})
            if failure is None:
                raise
        finally:
            persist(report.get("stage", "setup"))


if __name__ == "__main__":
    main()
