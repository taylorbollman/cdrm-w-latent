"""Small O3 LM training platform: weighted accumulation and exact boundary resume.

Each row is one independent document; padding is allowed, document packing is
not. The NextLat wrapper owns objective definitions and coefficients. This
module only normalizes sums over the *whole optimizer update*, applies AdamW,
and saves complete update-boundary state. No distributed correctness is claimed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .artifacts import sha256_file

TERMS = ("ce", "latent", "kl")
CHECKPOINT_SCHEMA = "olmo-lm-training-checkpoint-v1"


@dataclass(frozen=True)
class LMTrainingConfig:
    precision: str = "fp32"
    max_grad_norm: float | None = 1.0

    def __post_init__(self):
        if self.precision not in ("fp32", "bf16_mixed"):
            raise ValueError("precision must be fp32 or bf16_mixed")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be positive finite or None")


@dataclass
class TrainingCounters:
    optimizer_updates: int = 0
    microbatches: int = 0
    documents: int = 0
    input_tokens: int = 0
    ce_positions: int = 0
    latent_pairs: int = 0
    kl_triples: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("Training counters must be nonnegative integers")


def _plain(value):
    """Canonical configuration/cursor data, without code or tensor payloads."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("Configuration/cursor keys must be strings")
        return {key: _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (torch.dtype, Path)):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported configuration/cursor value: {type(value).__name__}")


def parameter_layout(model: nn.Module) -> list[dict[str, Any]]:
    """Canonical ownership includes aliases but owns every Parameter once."""
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) not in groups:
            groups[id(parameter)] = {"name": name, "aliases": [], "shape": list(parameter.shape),
                                     "dtype": str(parameter.dtype), "requires_grad": parameter.requires_grad}
        groups[id(parameter)]["aliases"].append(name)
    return list(groups.values())


def optimizer_ownership(model: nn.Module, optimizer) -> list[list[str]]:
    by_id = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    seen, groups = set(), []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in by_id or identity in seen:
                raise ValueError("Optimizer has a foreign, frozen or duplicate parameter")
            names.append(by_id[identity]); seen.add(identity)
        if group.get("param_names") is not None and group["param_names"] != names:
            raise ValueError("Optimizer parameter names/order differ from model ownership")
        groups.append(names)
    if seen != set(by_id):
        raise ValueError("Optimizer is missing trainable model parameters")
    return groups


def build_adamw(model: nn.Module, *, lr: float, betas=(0.9, 0.95), eps: float = 1e-8,
                weight_decay: float = 0.1, foreach: bool = False,
                fused: bool | None = None) -> torch.optim.AdamW:
    """Matrix weights decay; vector/scalar parameters do not. Tying stays native."""
    if not math.isfinite(lr) or lr < 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("AdamW learning rate/weight decay must be finite and nonnegative")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("AdamW epsilon must be positive finite")
    if fused is not None and type(fused) is not bool:
        raise TypeError("AdamW fused must be boolean or None")
    if fused and foreach:
        raise ValueError("AdamW fused=True and foreach=True cannot be combined")
    grouped = {True: [], False: []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped[parameter.ndim >= 2].append((name, parameter))
    groups = [{"params": [p for _, p in pairs], "param_names": [name for name, _ in pairs],
               "weight_decay": weight_decay if decay else 0.0}
              for decay, pairs in grouped.items() if pairs]
    if not groups:
        raise ValueError("AdamW requires a trainable parameter")
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps,
                                  weight_decay=weight_decay, foreach=foreach, fused=fused)
    optimizer_ownership(model, optimizer)
    return optimizer


def build_warmup_scheduler(optimizer, *, warmup_updates: int):
    """First update uses base_lr/N, Nth uses base_lr; zero means constant LR."""
    if type(warmup_updates) is not int or warmup_updates < 0:
        raise ValueError("warmup_updates must be a nonnegative integer")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda completed: min(1.0, (completed + 1) / warmup_updates) if warmup_updates else 1.0)
    scheduler._cdrm_warmup_updates = warmup_updates
    return scheduler


def optimizer_state_bytes(optimizer) -> dict[str, int]:
    """Actual initialized state storage by device, excluding model and gradients."""
    seen, totals = set(), {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr(), storage.nbytes())
                if key not in seen:
                    totals[key[0]] = totals.get(key[0], 0) + storage.nbytes()
                    seen.add(key)
    return totals


def _counts(values):
    if set(values) != set(TERMS) or any(type(values[k]) is not int or values[k] < 0 for k in TERMS):
        raise ValueError("Objective counts must be nonnegative integer ce/latent/kl counts")
    return dict(values)


def _batch_statistics(batch):
    valid, documents = batch.valid_mask, batch.document_ids
    if valid.dtype != torch.bool or valid.shape != batch.input_ids.shape or documents.shape != valid.shape:
        raise ValueError("LM batches require aligned token, validity and document tensors")
    for ids, mask in zip(documents, valid):
        active = ids[mask]
        if active.numel() and (bool((active < 0).any()) or active.unique().numel() != 1):
            raise ValueError("Each batch row must contain one independent document; packing is unsupported")
    return int(valid.any(-1).sum()), int(valid.sum())


def optimizer_step(model: nn.Module, optimizer, microbatches: Sequence, *,
                   config: LMTrainingConfig = LMTrainingConfig(), backbone_kwargs=None,
                   scheduler=None, counters: TrainingCounters | None = None) -> dict:
    """Accumulate each objective against its own global valid-position count.

    Failure before optimizer.step leaves weights/moments/counters unchanged and
    clears partial gradients. RNG/data consumption is not rolled back. This is
    a single-process step; DDP would require cross-rank denominators/reduction.
    """
    microbatches = list(microbatches)
    if not microbatches:
        raise ValueError("An optimizer update needs at least one microbatch")
    optimizer_ownership(model, optimizer)
    counters = TrainingCounters() if counters is None else counters
    kwargs = {} if backbone_kwargs is None else dict(backbone_kwargs)
    weights = dict(model.objective_weights())
    if set(weights) != set(TERMS) or any(not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Canonical objective weights must be finite nonnegative ce/latent/kl values")
    batch_counts = [_counts(model.counts(batch)) for batch in microbatches]
    denominators = {term: sum(counts[term] for counts in batch_counts) for term in TERMS}
    if not any(denominators[term] and weights[term] for term in TERMS):
        raise ValueError("Update has no valid positively weighted objective")
    stats = [_batch_statistics(batch) for batch in microbatches]
    totals = {term: 0.0 for term in TERMS}
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    device = next(model.parameters()).device
    if config.precision == "bf16_mixed" and device.type != "cuda":
        raise ValueError("O3 BF16 mixed execution requires the validated GPU environment")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for batch, expected_counts in zip(microbatches, batch_counts):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16_mixed"):
                result = model.loss_sums(batch, backbone_kwargs=kwargs)
                if _counts(result.counts) != expected_counts or set(result.sums) != set(TERMS):
                    raise ValueError("Forward objective counts differ from precomputed denominators")
                parts = []
                for term in TERMS:
                    value = result.sums[term]
                    if value.ndim != 0 or not bool(torch.isfinite(value.detach())):
                        raise FloatingPointError(f"Nonfinite or nonscalar {term} loss sum")
                    totals[term] += float(value.detach())
                    if denominators[term] and weights[term]:
                        parts.append(value * (weights[term] / denominators[term]))
                objective = sum(parts)
            if not isinstance(objective, torch.Tensor) or not objective.requires_grad:
                raise ValueError("Microbatch has no attached objective gradient")
            objective.backward()
        parameters = [p for p in model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(parameters,
                float("inf") if config.max_grad_norm is None else config.max_grad_norm,
                error_if_nonfinite=True, foreach=False)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.zero_grad(set_to_none=True)
    counters.optimizer_updates += 1
    counters.microbatches += len(microbatches)
    counters.documents += sum(value[0] for value in stats)
    counters.input_tokens += sum(value[1] for value in stats)
    counters.ce_positions += denominators["ce"]
    counters.latent_pairs += denominators["latent"]
    counters.kl_triples += denominators["kl"]
    means = {term: totals[term] / denominators[term] if denominators[term] else 0.0 for term in TERMS}
    return {"schema": "olmo-lm-optimizer-step-v1", "update_completed": True,
            "loss_sums": totals, "counts": denominators, "loss_means": means, "objective_weights": weights,
            "objective": sum(weights[term] * means[term] for term in TERMS),
            "gradient_norm_before_clip": float(norm), "max_grad_norm": config.max_grad_norm,
            "lr_used": learning_rates, "lr_next": [g["lr"] for g in optimizer.param_groups],
            "optimizer_state_bytes_by_device": optimizer_state_bytes(optimizer), "counters": asdict(counters)}


def _scheduler_descriptor(scheduler):
    if scheduler is None:
        return None
    return {"class": type(scheduler).__module__ + "." + type(scheduler).__qualname__,
            "warmup_updates": getattr(scheduler, "_cdrm_warmup_updates", None),
            "base_lrs": list(scheduler.base_lrs)}


def _optimizer_descriptor(optimizer):
    return {"class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
            "defaults": _plain(optimizer.defaults)}


def _rng_state(generators):
    import numpy as np
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {name: generator.get_state() for name, generator in (generators or {}).items()}}


def _restore_rng(state, generators):
    import numpy as np
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for name, generator in (generators or {}).items():
        generator.set_state(state["generators"][name])


def _fingerprint(value):
    value = _plain(value)
    digest = value.get("checkpoint_sha256", value.get("sha256", "")) if isinstance(value, dict) else ""
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Source fingerprint must include checkpoint_sha256 (or sha256)")
    return value


def save_training_checkpoint(path: str | Path, model: nn.Module, optimizer, *, scheduler=None,
                             counters: TrainingCounters, data_cursor: Mapping, configuration: Mapping,
                             source_fingerprint: Mapping, generators=None) -> dict:
    """Atomically publish one complete boundary checkpoint without overwriting.

    Construct/load the model before its optimizer. Restore later with ordinary
    load_state_dict(assign=False), keeping those existing parameter references.
    Save only between completed updates, with cleared gradients and no live KV
    cache. Callers place durable files under the project/home, not disposable SSD.
    """
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Save only an update boundary with cleared gradients")
    ownership = optimizer_ownership(model, optimizer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Training checkpoint already exists: {path}")
    payload = {"schema": CHECKPOINT_SCHEMA, "model_type": type(model).__module__ + "." + type(model).__qualname__,
               "model": model.state_dict(), "parameter_layout": parameter_layout(model),
               "module_training": {name: module.training for name, module in model.named_modules()},
               "optimizer": optimizer.state_dict(), "optimizer_ownership": ownership,
               "optimizer_descriptor": _optimizer_descriptor(optimizer),
               "scheduler": None if scheduler is None else scheduler.state_dict(),
               "scheduler_descriptor": _scheduler_descriptor(scheduler),
               "counters": asdict(counters), "data_cursor": _plain(data_cursor),
               "configuration": _plain(configuration), "source_fingerprint": _fingerprint(source_fingerprint),
               "rng": _rng_state(generators)}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        # Same-directory hard-link publication is atomic and refuses races that
        # would overwrite another completed checkpoint. Temporary bytes vanish.
        os.link(temporary, path)
        temporary.unlink(); temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"schema": CHECKPOINT_SCHEMA, "path": str(path), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "optimizer_updates": counters.optimizer_updates}


def load_training_checkpoint(path: str | Path, model: nn.Module, optimizer, *, scheduler=None,
                             configuration: Mapping, source_fingerprint: Mapping,
                             generators=None, expected_sha256: str | None = None) -> dict:
    """Validate ownership/config before loading, then restore optimizer and RNG.

    CUDA RNG topology must match. No cross-topology or distributed resume claim
    is made. AdamW's native loader moves moments to parameter devices while
    keeping noncapturable step scalars on their appropriate device.
    """
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Training checkpoint SHA256 differs from retained record")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema": CHECKPOINT_SCHEMA,
                "model_type": type(model).__module__ + "." + type(model).__qualname__,
                "parameter_layout": parameter_layout(model), "optimizer_ownership": optimizer_ownership(model, optimizer),
                "optimizer_descriptor": _optimizer_descriptor(optimizer),
                "scheduler_descriptor": _scheduler_descriptor(scheduler),
                "configuration": _plain(configuration), "source_fingerprint": _fingerprint(source_fingerprint)}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Training checkpoint {key} differs from the requested resume")
    if (payload.get("scheduler") is None) != (scheduler is None):
        raise ValueError("Training checkpoint scheduler presence differs")
    counters = TrainingCounters(**payload["counters"])
    model_state = model.state_dict()
    if set(payload["model"]) != set(model_state):
        raise ValueError("Training checkpoint model tensor keys differ")
    for name, current in model_state.items():
        saved = payload["model"][name]
        if not isinstance(saved, torch.Tensor) or saved.shape != current.shape or saved.dtype != current.dtype:
            raise ValueError(f"Training checkpoint model tensor shape/dtype differs: {name}")
    for record in payload["parameter_layout"]:
        aliases = record["aliases"]
        if any(not torch.equal(payload["model"][aliases[0]], payload["model"][name]) for name in aliases[1:]):
            raise ValueError("Training checkpoint tied parameter aliases disagree")
    current_groups = optimizer.param_groups
    saved_groups = payload["optimizer"]["param_groups"]
    if len(saved_groups) != len(current_groups):
        raise ValueError("Training checkpoint optimizer group count differs")
    seen = set()
    for saved_group, current_group, names in zip(saved_groups, current_groups, expected["optimizer_ownership"]):
        if len(saved_group["params"]) != len(names) or saved_group.get("param_names") != names:
            raise ValueError("Training checkpoint optimizer parameter order differs")
        for identifier, parameter in zip(saved_group["params"], current_group["params"]):
            if identifier in seen:
                raise ValueError("Training checkpoint optimizer has duplicate parameter ownership")
            seen.add(identifier)
            for key, value in payload["optimizer"]["state"].get(identifier, {}).items():
                if isinstance(value, torch.Tensor) and key != "step" and (value.shape != parameter.shape or value.dtype != parameter.dtype):
                    raise ValueError("Training checkpoint optimizer moment shape/dtype differs")
    if not set(payload["optimizer"]["state"]).issubset(seen):
        raise ValueError("Training checkpoint optimizer contains foreign state")
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(payload["rng"]["torch_cuda"]) != cuda_count:
        raise ValueError("CUDA RNG device topology differs; exact resume is unsupported")
    if set(payload["rng"]["generators"]) != set(generators or {}):
        raise ValueError("Explicit data-generator names differ")
    if set(payload["module_training"]) != set(dict(model.named_modules())):
        raise ValueError("Checkpoint module ownership differs")
    cursor = _plain(payload["data_cursor"])
    # All configuration/ownership checks above precede mutation of live objects.
    model.load_state_dict(payload["model"], strict=True, assign=False)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    for name, module in model.named_modules():
        module.training = payload["module_training"][name]
    optimizer.zero_grad(set_to_none=True)
    _restore_rng(payload["rng"], generators)
    return {"counters": counters, "data_cursor": cursor,
            "configuration": payload["configuration"], "source_fingerprint": payload["source_fingerprint"]}
