"""Crash-safe, same-world-size checkpoints for replicated DDP training.

All ranks call save/load with the canonical (unwrapped) model and its optimizer.
Rank zero writes a single shared state file. A separately published manifest is
the commit record: a directory without that manifest is never resumable. Only
small metadata and rank-local RNG/cursors cross the process group. This module
does not establish model/optimizer replica equality; the training harness must
check that invariant. It does not serialize CUDA graphs or support sharding.

Exceptions raised by live participants are coordinated before the next stage.
Process death or a broken collective still relies on the process-group timeout.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping

import torch
import torch.distributed as dist
from torch import nn

from .artifacts import sha256_file
from .lm_training import (
    TrainingCounters, _fingerprint, _optimizer_descriptor, _plain,
    _scheduler_descriptor, optimizer_ownership, parameter_layout,
)

CHECKPOINT_SCHEMA = "olmo-replicated-ddp-checkpoint-v1"
STATE_FILENAME = "state.pt"
MANIFEST_FILENAME = "manifest.json"


class DistributedCheckpointError(RuntimeError):
    """A checkpoint stage failed on at least one live rank."""


def _context(group):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed checkpointing requires an initialized process group")
    return dist.get_rank(group), dist.get_world_size(group)


def _gather(value, group):
    result = [None] * dist.get_world_size(group)
    dist.all_gather_object(result, value, group=group)
    return result


def _coordinated(stage, work, group):
    """Finish local work, exchange only status, then succeed or fail together."""
    value = None
    error = None
    try:
        value = work()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = _gather(error, group)
    if any(item is not None for item in errors):
        failures = "; ".join(f"rank {rank}: {item}" for rank, item in enumerate(errors) if item is not None)
        raise DistributedCheckpointError(f"Distributed checkpoint {stage} failed: {failures}")
    return value


def _local_device(model, device):
    parameters = list(model.parameters())
    if device is None:
        device = parameters[0].device if parameters else torch.device("cpu")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Checkpoint RNG supports CPU or local CUDA only")
    if device.type == "cuda":
        current = torch.cuda.current_device()
        device = torch.device("cuda", current if device.index is None else device.index)
        if device.index != current:
            raise ValueError("Checkpoint device must be this rank's current CUDA device")
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("Canonical model parameters must all belong to this rank's device")
    return device


def _local_rng(device, generators):
    """Never enumerate or initialize RNG state on the peer rank's GPU."""
    import numpy as np
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "device": str(device),
        "generators": {name: generator.get_state() for name, generator in (generators or {}).items()},
        "generator_devices": {name: str(generator.device) for name, generator in (generators or {}).items()},
    }


def _validate_rng(state, device, generators):
    if state.get("device") != str(device):
        raise ValueError("Rank-local RNG device differs from the requested resume")
    if (state.get("torch_cuda") is None) != (device.type == "cpu"):
        raise ValueError("Rank-local CUDA RNG presence differs")
    for key in ("torch_cpu", "torch_cuda"):
        value = state[key]
        if value is not None and (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1):
            raise ValueError(f"Invalid {key} RNG state")
    expected = {name: str(generator.device) for name, generator in (generators or {}).items()}
    if state.get("generator_devices") != expected or set(state.get("generators", {})) != set(expected):
        raise ValueError("Explicit rank-local generator names/devices differ")
    # Validate Python/NumPy/CPU-generator states without changing live RNGs.
    import numpy as np
    random.Random().setstate(state["python"])
    n = state["numpy"]
    np.random.RandomState().set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    for name, value in state["generators"].items():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError(f"Invalid explicit generator RNG state: {name}")


def _restore_local_rng(state, device, generators):
    import numpy as np
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    for name, generator in (generators or {}).items():
        generator.set_state(state["generators"][name])


def _metadata(model, optimizer, scheduler, configuration, source_fingerprint, world_size):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise ValueError("Pass the canonical model, not its DDP wrapper")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "world_size": world_size,
        "model_type": type(model).__module__ + "." + type(model).__qualname__,
        "parameter_layout": parameter_layout(model),
        "optimizer_ownership": optimizer_ownership(model, optimizer),
        "optimizer_descriptor": _optimizer_descriptor(optimizer),
        "scheduler_descriptor": _scheduler_descriptor(scheduler),
        "configuration": _plain(configuration),
        "source_fingerprint": _fingerprint(source_fingerprint),
    }


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(path, writer):
    """No-overwrite publication, retaining partial bytes on failure for audit."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    _sync_directory(path.parent)


def _write_checkpoint(directory, payload):
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    _sync_directory(directory.parent)
    state_path = directory / STATE_FILENAME
    _publish(state_path, lambda stream: torch.save(payload, stream))
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "world_size": payload["metadata"]["world_size"],
        "metadata": payload["metadata"],
        "counters": payload["counters"],
        "rank_cursors": [record["data_cursor"] for record in payload["rank_states"]],
        "state": {"filename": STATE_FILENAME, "size_bytes": state_path.stat().st_size, "sha256": sha256_file(state_path)},
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _publish(directory / MANIFEST_FILENAME, lambda stream: stream.write(encoded))
    return {**manifest, "directory": str(directory), "manifest_sha256": sha256_file(directory / MANIFEST_FILENAME)}


def save_distributed_checkpoint(directory: str | Path, model: nn.Module, optimizer, *,
                                scheduler=None, counters: TrainingCounters,
                                data_cursor: Mapping, configuration: Mapping,
                                source_fingerprint: Mapping, generators=None,
                                device=None, group=None) -> dict:
    """Save once from rank zero after all ranks reach a cleared update boundary.

    The directory must not exist, even if a previous attempt was incomplete.
    Shared counters/configuration/descriptors must agree on every rank. Replicas
    must already agree on their tensor state. Save canonical model/optimizer
    ownership outside the DDP wrapper, and rebuild graphs after restoring.
    """
    rank, world_size = _context(group)
    directory = Path(directory)

    def prepare():
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Save only an update boundary with cleared gradients")
        local_device = _local_device(model, device)
        metadata = _metadata(model, optimizer, scheduler, configuration, source_fingerprint, world_size)
        global_state = {
            "metadata": metadata,
            "counters": asdict(TrainingCounters(**asdict(counters))),
            "module_training": {name: module.training for name, module in model.named_modules()},
            "scheduler_state": _plain(None if scheduler is None else scheduler.state_dict()),
            "optimizer_groups": _plain([{key: value for key, value in group.items() if key != "params"}
                                         for group in optimizer.param_groups]),
            "directory": str(directory),
        }
        rank_state = {"rank": rank, "data_cursor": _plain(data_cursor), "rng": _local_rng(local_device, generators)}
        _validate_rng(rank_state["rng"], local_device, generators)
        return global_state, rank_state

    local = _coordinated("prepare save", prepare, group)
    records = _gather(local, group)
    if any(record[0] != records[0][0] for record in records):
        raise DistributedCheckpointError("Distributed checkpoint global metadata/counters/directory differ across ranks")
    shared = records[0][0]

    def write():
        if rank != 0:
            return None
        payload = {**shared, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": None if scheduler is None else scheduler.state_dict(),
                   "rank_states": [record[1] for record in records]}
        return _write_checkpoint(directory, payload)

    receipt = _coordinated("write and commit", write, group)
    return _gather(receipt, group)[0]


def inspect_distributed_checkpoint(directory: str | Path, *, expected_manifest_sha256=None,
                                   verify_state=True) -> dict:
    """Read a committed local checkpoint; useful before retaining it in GCS."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError("Incomplete distributed checkpoint: committed manifest is missing")
    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Distributed checkpoint manifest SHA256 differs from retained record")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Distributed checkpoint schema differs")
    state = manifest["state"]
    if state.get("filename") != STATE_FILENAME:
        raise ValueError("Distributed checkpoint state filename differs")
    state_path = directory / STATE_FILENAME
    if not state_path.is_file() or state_path.stat().st_size != state["size_bytes"]:
        raise ValueError("Distributed checkpoint state file is missing or has wrong size")
    if verify_state and sha256_file(state_path) != state["sha256"]:
        raise ValueError("Distributed checkpoint state SHA256 differs from committed manifest")
    return {**manifest, "directory": str(directory), "manifest_sha256": manifest_sha256}


def _validate_payload(payload, expected, manifest, model, optimizer, scheduler, rank, device, generators):
    if payload.get("metadata") != expected or manifest.get("metadata") != expected:
        raise ValueError("Distributed checkpoint configuration/source/ownership metadata differs from requested resume")
    counters = TrainingCounters(**payload["counters"])
    if manifest.get("counters") != asdict(counters):
        raise ValueError("Distributed checkpoint counters disagree with committed manifest")
    ranks = payload["rank_states"]
    if len(ranks) != expected["world_size"] or [record["rank"] for record in ranks] != list(range(expected["world_size"])):
        raise ValueError("Distributed checkpoint rank-state mapping differs")
    if manifest.get("rank_cursors") != [record["data_cursor"] for record in ranks]:
        raise ValueError("Distributed checkpoint cursors disagree with committed manifest")
    local = ranks[rank]
    cursor = _plain(local["data_cursor"])
    _validate_rng(local["rng"], device, generators)
    if (payload.get("scheduler") is None) != (scheduler is None):
        raise ValueError("Distributed checkpoint scheduler presence differs")
    if _plain(payload["scheduler"]) != payload["scheduler_state"]:
        raise ValueError("Distributed checkpoint scheduler state disagrees with shared metadata")
    module_training = payload["module_training"]
    if set(module_training) != set(dict(model.named_modules())) or any(type(value) is not bool for value in module_training.values()):
        raise ValueError("Distributed checkpoint module ownership/training modes differ")
    current_state = model.state_dict()
    if set(payload["model"]) != set(current_state):
        raise ValueError("Distributed checkpoint model tensor keys differ")
    for name, current in current_state.items():
        saved = payload["model"][name]
        if not isinstance(saved, torch.Tensor) or saved.shape != current.shape or saved.dtype != current.dtype:
            raise ValueError(f"Distributed checkpoint model tensor shape/dtype differs: {name}")
    for record in expected["parameter_layout"]:
        aliases = record["aliases"]
        if any(not torch.equal(payload["model"][aliases[0]], payload["model"][alias]) for alias in aliases[1:]):
            raise ValueError("Distributed checkpoint tied parameter aliases disagree")
    saved_optimizer = payload["optimizer"]
    saved_groups = saved_optimizer["param_groups"]
    if _plain([{key: value for key, value in group.items() if key != "params"} for group in saved_groups]) != payload["optimizer_groups"]:
        raise ValueError("Distributed checkpoint optimizer group settings disagree with shared metadata")
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("Distributed checkpoint optimizer group count differs")
    seen = set()
    for saved_group, current_group, names in zip(saved_groups, optimizer.param_groups, expected["optimizer_ownership"]):
        if len(saved_group["params"]) != len(names) or saved_group.get("param_names") != names:
            raise ValueError("Distributed checkpoint optimizer parameter order differs")
        for identifier, parameter in zip(saved_group["params"], current_group["params"]):
            if identifier in seen:
                raise ValueError("Distributed checkpoint optimizer has duplicate parameter ownership")
            seen.add(identifier)
            for key, value in saved_optimizer["state"].get(identifier, {}).items():
                if isinstance(value, torch.Tensor) and key != "step" and (value.shape != parameter.shape or value.dtype != parameter.dtype):
                    raise ValueError("Distributed checkpoint optimizer moment shape/dtype differs")
    if not set(saved_optimizer["state"]).issubset(seen):
        raise ValueError("Distributed checkpoint optimizer contains foreign state")
    return counters, cursor


def load_distributed_checkpoint(directory: str | Path, model: nn.Module, optimizer, *,
                                scheduler=None, configuration: Mapping,
                                source_fingerprint: Mapping, generators=None,
                                expected_manifest_sha256: str | None = None,
                                device=None, group=None) -> dict:
    """Validate on every rank before mutating model, optimizer or RNG state.

    Each rank independently reads the same single shared state file. This avoids
    broadcasting a model-sized Python object and lets native optimizer loading
    place moments on the local parameter devices. Runtime flags, optimizer
    implementation, parameter aliases and world size must match exactly.
    """
    rank, world_size = _context(group)
    directory = Path(directory)

    def prepare():
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Load only with cleared gradients and no live graph")
        local_device = _local_device(model, device)
        expected = _metadata(model, optimizer, scheduler, configuration, source_fingerprint, world_size)
        return local_device, expected

    local_device, expected = _coordinated("prepare load", prepare, group)
    requests = _gather({"directory": str(directory), "metadata": expected, "manifest_sha256": expected_manifest_sha256}, group)
    if any(request != requests[0] for request in requests):
        raise DistributedCheckpointError("Distributed checkpoint resume requests differ across ranks")

    def read_and_validate():
        manifest = inspect_distributed_checkpoint(directory, expected_manifest_sha256=expected_manifest_sha256)
        if manifest.get("world_size") != world_size:
            raise ValueError("Distributed checkpoint world size differs; resharding is unsupported")
        if manifest.get("metadata") != expected:
            raise ValueError("Distributed checkpoint configuration/source/ownership metadata differs from requested resume")
        payload = torch.load(directory / STATE_FILENAME, map_location="cpu", weights_only=True)
        counters, cursor = _validate_payload(payload, expected, manifest, model, optimizer, scheduler, rank, local_device, generators)
        return manifest, payload, counters, cursor

    manifest, payload, counters, cursor = _coordinated("read and validate", read_and_validate, group)

    def restore():
        model.load_state_dict(payload["model"], strict=True, assign=False)
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        for name, module in model.named_modules():
            module.training = payload["module_training"][name]
        optimizer.zero_grad(set_to_none=True)
        _restore_local_rng(payload["rank_states"][rank]["rng"], local_device, generators)

    _coordinated("restore", restore, group)
    return {"counters": counters, "data_cursor": cursor, "configuration": expected["configuration"],
            "source_fingerprint": expected["source_fingerprint"], "manifest": manifest}
