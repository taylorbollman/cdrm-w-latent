"""Authenticated replicated-state migration, separate from the strict loader.

No old checkpoint is rewritten and no validation in the historical loader is
disabled. Surviving streams retain their RNG bytes under an explicit device
mapping; added streams are seeded in the destination lineage. Save the result
with save_distributed_checkpoint, then use its ordinary strict loader for later
same-topology recovery.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from cdrm.pretrained.artifacts import sha256_file
from cdrm.pretrained.campaign_recipe import CampaignTokenSchedule
from cdrm.pretrained.distributed_checkpoint import (
    DistributedCheckpointError, STATE_FILENAME, _context, _coordinated, _gather,
    _local_device, _local_rng, _metadata, _restore_local_rng, _validate_payload,
    _validate_rng, inspect_distributed_checkpoint,
)
from cdrm.pretrained.lm_training import TrainingCounters, _plain
from scripts.olmo_campaign_execution import validate_clocks
from scripts.olmo_topology_contract import (
    SCHEMA, canonical_parent_cursor, cursor_record, destination_configuration,
    destination_fingerprint, digest, validate_topology_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def _signature(path):
    info = path.stat()
    return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def validate_source_pins(migration):
    for name, expected in migration["sources"].items():
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or sha256_file(path) != expected:
            raise ValueError("Destination runtime source differs: " + name)


def _generator_descriptors(rng):
    # _validate_rng reads only generator.device. Descriptors let us validate
    # original rank1 CUDA metadata while running on an isolated rank0 device,
    # without allocating a generator or initializing another GPU's context.
    return {name: SimpleNamespace(device=torch.device(device))
            for name, device in rng["generator_devices"].items()}


def validate_schedule_and_adam(payload, model, optimizer, scheduler, manifest):
    """Prevalidate values which ordinary load_state_dict would otherwise apply."""
    if not isinstance(scheduler, CampaignTokenSchedule):
        raise ValueError("Topology migration requires the unchanged finite campaign token scheduler")
    config = manifest["metadata"]["configuration"]
    if scheduler.checkpoint_contract() != config["schedule"]:
        raise ValueError("Finite schedule contract differs before migration")
    state = payload["scheduler"]
    for name in ("token_prefix", "warmup_tokens", "start_fraction", "plan_sha256", "base_lrs"):
        if state.get(name) != getattr(scheduler, name):
            raise ValueError("Saved scheduler identity differs: " + name)
    counters = TrainingCounters(**payload["counters"])
    completed = counters.optimizer_updates
    if (state.get("last_epoch") != completed or state.get("_step_count") != completed + 1
            or not 0 <= completed < len(scheduler.token_prefix)
            or scheduler.token_prefix[completed] != counters.input_tokens):
        raise ValueError("Scheduler and committed token/update clocks differ")
    progress = min(1., counters.input_tokens / scheduler.warmup_tokens) if scheduler.warmup_tokens else 1.
    factor = scheduler.start_fraction + (1. - scheduler.start_fraction) * progress
    rates = [value * factor for value in scheduler.base_lrs]
    groups = payload["optimizer"]["param_groups"]
    if state.get("_last_lr") != rates or [group["lr"] for group in groups] != rates:
        raise ValueError("Saved learning-rate state differs from the unchanged schedule")
    states = payload["optimizer"]["state"]
    owned = {identifier for group in groups for identifier in group["params"]}
    if (completed == 0 and states) or (completed > 0 and set(states) != owned):
        raise ValueError("Adam moments do not cover the committed optimizer history")
    for identifier, value in states.items():
        if set(value) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Require standard complete Adam state")
        step = value["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != completed:
            raise ValueError("Adam step differs from committed optimizer counter")
        if any(not isinstance(value[name], torch.Tensor) or value[name].dtype != torch.float32
               for name in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Adam moments must be actual FP32 tensors")
    # The checkpoint metadata authenticates configuration; also ensure the
    # caller actually constructed its operative model configs (not just equal
    # tensor shapes with a changed auxiliary loss or document policy).
    if hasattr(model, "config") and model.config.to_dict() != config["model"]:
        raise ValueError("Actual model configuration differs from authenticated source")
    backbone = getattr(getattr(model, "backbone", None), "backbone", None)
    if backbone is not None and backbone.config.to_dict() != config["backbone"]:
        raise ValueError("Actual backbone configuration differs from authenticated source")
    if hasattr(model, "objective_weights"):
        enabled = "N" in config["recipe"]["arm"]
        want = {"ce": 1., "latent": config["model"]["lambda_latent"] if enabled else 0.,
                "kl": config["model"]["lambda_kl"] if enabled else 0.}
        if model.objective_weights() != want:
            raise ValueError("Actual enabled objective differs from authenticated source")
        predictor = getattr(model, "predictor", None)
        if predictor is not None and any(getattr(predictor.config, name) != config["model"][name]
                                         for name in ("lambda_latent", "lambda_kl")):
            raise ValueError("Actual predictor loss weights differ from authenticated source")


def mapped_rng_state(payload, migration, *, rank, device, generators):
    """Construct and validate local RNG state without touching global streams."""
    device = torch.device(device)
    source_rank = migration["rng"]["source_rank_for_destination"][rank]
    if torch.device(payload["rank_states"][0]["rng"]["device"]).type != device.type:
        raise ValueError("Changing RNG device type is outside topology migration")
    expected_devices = {name: str(generator.device) for name, generator in (generators or {}).items()}
    source_names = set(payload["rank_states"][0]["rng"]["generator_devices"])
    if set(expected_devices) != source_names:
        raise ValueError("Destination explicit generator names differ from source")
    if source_rank is not None:
        state = copy.deepcopy(payload["rank_states"][source_rank]["rng"])
        if torch.device(state["device"]).type != device.type:
            raise ValueError("Changing RNG device type is outside topology migration")
        if any(torch.device(state["generator_devices"][name]).type != torch.device(target).type
               for name, target in expected_devices.items()):
            raise ValueError("Explicit generator device type changed")
        state["device"], state["generator_devices"] = str(device), expected_devices
        policy = {"kind": "retained-source-streams", "source_rank": source_rank,
                  "source_device": payload["rank_states"][source_rank]["rng"]["device"],
                  "destination_device": str(device)}
    else:
        def seed(name):
            return int(digest([SCHEMA, migration["lineage"], migration["rng"]["seed"], rank, name])[:16], 16) % (2**63)
        numpy_state = np.random.RandomState(seed("numpy") % 2**32).get_state()
        state = {"python": random.Random(seed("python")).getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]),
            "torch_cpu": torch.Generator(device="cpu").manual_seed(seed("torch_cpu")).get_state(),
            "torch_cuda": None if device.type == "cpu" else torch.Generator(device=device).manual_seed(seed("torch_cuda")).get_state(),
            "device": str(device), "generator_devices": expected_devices,
            "generators": {name: torch.Generator(device=generator.device).manual_seed(seed("generator/" + name)).get_state()
                           for name, generator in (generators or {}).items()}}
        policy = {"kind": "seeded-additional-rank", "source_rank": None, "seed_policy": migration["rng"],
                  "destination_device": str(device)}
    _validate_rng(state, device, generators)
    return state, policy


def validate_parent_payload(payload, manifest, model, optimizer, scheduler, migration):
    """Pure source-state checks, including every old rank's RNG/cursor envelope."""
    validate_topology_contract(migration, manifest)
    metadata = manifest["metadata"]
    expected = _metadata(model, optimizer, scheduler, metadata["configuration"], metadata["source_fingerprint"],
                         manifest["world_size"])
    if expected != metadata:
        raise ValueError("Actual model/optimizer/scheduler descriptors differ from source authority")
    # Reuse the strict tensor/alias/group/configuration validator unchanged.
    # Source-device descriptors validate metadata without opening peer GPUs.
    for rank in range(manifest["world_size"]):
        rng = payload["rank_states"][rank]["rng"]
        _validate_payload(payload, expected, manifest, model, optimizer, scheduler, rank,
                          torch.device(rng["device"]), _generator_descriptors(rng))
    canonical_parent_cursor(manifest)
    validate_schedule_and_adam(payload, model, optimizer, scheduler, manifest)
    return TrainingCounters(**payload["counters"])


def load_topology_checkpoint(directory, model, optimizer, *, scheduler, migration,
                             expected_manifest_sha256, generators=None, device=None, group=None):
    """All ranks validate fully before restoring a declared destination state.

    Caller constructs the unchanged source model and finite scheduler first.
    This method does not save a child, build a graph, advance an update, or
    restore the ordered reader; install the returned inner cursor in a fresh
    reader after checking it against the caller's full logical-data plan.
    """
    rank, world = _context(group)
    directory = Path(directory)
    def prepare():
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Migration requires cleared gradients and no captured graph")
        if world != migration["destination_world_size"] or expected_manifest_sha256 != migration["parent_manifest_sha256"]:
            raise ValueError("Process group or source manifest differs from migration authority")
        validate_source_pins(migration)
        return _local_device(model, device)
    local_device = _coordinated("topology preparation", prepare, group)
    requests = _gather({"directory": str(directory), "migration": migration,
                        "manifest_sha256": expected_manifest_sha256}, group)
    if any(value != requests[0] for value in requests):
        raise DistributedCheckpointError("Topology migration requests differ across ranks")

    def read():
        manifest = inspect_distributed_checkpoint(directory,
            expected_manifest_sha256=expected_manifest_sha256, verify_state=True)
        signature = _signature(directory / STATE_FILENAME)
        payload = torch.load(directory / STATE_FILENAME, map_location="cpu", weights_only=True, mmap=True)
        counters = validate_parent_payload(payload, manifest, model, optimizer, scheduler, migration)
        rng, policy = mapped_rng_state(payload, migration, rank=rank, device=local_device, generators=generators)
        configuration = destination_configuration(manifest["metadata"]["configuration"], migration)
        if signature != _signature(directory / STATE_FILENAME):
            raise ValueError("Original checkpoint state changed during validation")
        return manifest, payload, counters, rng, policy, configuration, signature
    manifest, payload, counters, rng, policy, configuration, signature = _coordinated("topology authenticate", read, group)

    def restore():
        model.load_state_dict(payload["model"], strict=True, assign=False)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        for name, module in model.named_modules():
            module.training = payload["module_training"][name]
        optimizer.zero_grad(set_to_none=True)
        _restore_local_rng(rng, local_device, generators)
        if signature != _signature(directory / STATE_FILENAME):
            raise ValueError("Original checkpoint state changed during restore")
        return validate_clocks(optimizer, scheduler, counters)
    clocks = _coordinated("topology restore", restore, group)
    fingerprint = destination_fingerprint(migration)
    record = cursor_record(migration["origin_cursor"], rank=rank, world_size=world,
                           batch_size=migration["physical_batch_per_rank"])
    receipt = {"schema": SCHEMA + "-receipt", "migration_sha256": migration["sha256"],
        "parent_manifest_sha256": manifest["manifest_sha256"], "parent_state_sha256": manifest["state"]["sha256"],
        "root_manifest_sha256": migration["root_manifest_sha256"],
        "destination_configuration_sha256": digest(configuration), "destination_source_fingerprint": fingerprint,
        "original_counters": asdict(counters), "canonical_cursor": copy.deepcopy(migration["origin_cursor"]),
        "schedule_state": _plain(scheduler.state_dict()), "clocks": clocks,
        "rng_mapping_by_rank": _gather(policy, group),
        "scope": "Authenticated exact replicated-state import; caller audits graph preparation and topology-dependent arithmetic"}
    return {"counters": counters, "data_cursor": record, "configuration": configuration,
            "source_fingerprint": fingerprint, "manifest": manifest, "migration_receipt": receipt}
