"""Explicit topology-transition identities and historical counter accounting.

No checkpoint mutation, tensor loading, RNG changes or GPU execution occurs
here. The original checkpoint format/loader remain unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
import re

from cdrm.pretrained.lm_training import TrainingCounters

SCHEMA = "olmo-topology-migration-v1"
CONFIGURATION_SCHEMA = "olmo-topology-execution-v1"
CURSOR_SCHEMA = "olmo-campaign-execution-cursor-v1"
RNG_POLICY = "retain-declared-source-rank-else-seed-destination-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pin(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Require an explicit lowercase SHA256")
    return value


def _positive(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(name + " must be a positive integer" if not zero else name + " must be a nonnegative integer")
    return value


def canonical_parent_cursor(manifest):
    """All source ranks must agree on one committed ordered-data position."""
    world = _positive(manifest["world_size"], "source world size")
    if (manifest.get("schema") != "olmo-replicated-ddp-checkpoint-v1"
            or manifest["metadata"].get("schema") != manifest["schema"]
            or manifest["metadata"].get("world_size") != world
            or manifest["metadata"]["configuration"].get("world_size") != world):
        raise ValueError("Source checkpoint schema or world-size declarations differ")
    counters = asdict(TrainingCounters(**manifest["counters"]))
    records = manifest["rank_cursors"]
    if len(records) != world:
        raise ValueError("Source cursor rank count differs")
    common, batch = None, None
    for rank, record in enumerate(records):
        if set(record) != {"schema", "cursor", "rank", "world_size", "physical_batch_per_rank"}:
            raise ValueError("Source cursor envelope fields differ")
        if record["schema"] != CURSOR_SCHEMA or record["rank"] != rank or record["world_size"] != world:
            raise ValueError("Source cursor rank/world identity differs")
        _positive(record["physical_batch_per_rank"], "source physical batch")
        cursor = record["cursor"]
        if set(cursor) != {"manifest_sha256", "split", "next_chunk", "next_update"}:
            raise ValueError("Source inner ordered cursor fields differ")
        _pin(cursor["manifest_sha256"])
        if cursor["split"] != "train":
            raise ValueError("Topology continuation requires the training cursor")
        for name in ("next_chunk", "next_update"):
            _positive(cursor[name], name, zero=True)
        if cursor["next_update"] != counters["optimizer_updates"] or cursor["next_chunk"] != counters["documents"]:
            raise ValueError("Source ordered cursor differs from logical counters")
        if common is not None and (cursor != common or record["physical_batch_per_rank"] != batch):
            raise ValueError("Source ranks disagree on cursor or physical batch")
        common, batch = cursor, record["physical_batch_per_rank"]
    return copy.deepcopy(common)


def cursor_record(inner, *, rank, world_size, batch_size):
    _positive(world_size, "world size")
    _positive(batch_size, "physical batch size")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("Destination rank is outside the process group")
    return {"schema": CURSOR_SCHEMA, "cursor": copy.deepcopy(inner), "rank": rank,
            "world_size": world_size, "physical_batch_per_rank": batch_size}


def make_topology_contract(parent_manifest, *, world_size, physical_batch_per_rank,
                           lineage, sources, rng_seed=20261001, rng_rank_map=None):
    """Create one explicitly named destination, with no schedule extension."""
    _positive(world_size, "destination world size")
    _positive(physical_batch_per_rank, "destination physical batch")
    if world_size > 64:
        raise ValueError("Topology metadata is bounded to at most 64 ranks")
    _positive(rng_seed, "RNG seed", zero=True)
    if rng_seed >= 2**63 or not isinstance(lineage, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", lineage) is None:
        raise ValueError("Require a bounded seed and explicit safe lineage name")
    source_world = parent_manifest["world_size"]
    cursor = canonical_parent_cursor(parent_manifest)
    if rng_rank_map is None:
        rng_rank_map = [rank if rank < source_world else None for rank in range(world_size)]
    if (len(rng_rank_map) != world_size or any(value is not None and
            (type(value) is not int or not 0 <= value < source_world) for value in rng_rank_map)
            or len([value for value in rng_rank_map if value is not None]) != len({value for value in rng_rank_map if value is not None})):
        raise ValueError("RNG rank mapping must retain each declared source rank at most once")
    if sum(value is not None for value in rng_rank_map) != min(source_world, world_size):
        raise ValueError("RNG mapping must preserve every surviving stream and seed only added ranks")
    if not sources:
        raise ValueError("Destination runtime source pins are required")
    for name, pin in sources.items():
        if not isinstance(name, str) or not name or name.startswith("/") or ".." in name.split("/"):
            raise ValueError("Destination runtime source paths must be project-relative")
        _pin(pin)
    config = parent_manifest["metadata"]["configuration"]
    arm = config["recipe"]["arm"]
    if arm not in ("B", "N", "F", "R", "NF", "NR", "FR", "NFR"):
        raise ValueError("Source arm is outside the campaign contract")
    previous_migration = config.get("topology_migration")
    comparison_origin = parent_manifest["manifest_sha256"]
    if previous_migration is not None and parent_manifest["counters"] == previous_migration["origin_counters"]:
        comparison_origin = previous_migration.get("comparison_origin_manifest_sha256",
                                                   previous_migration["parent_manifest_sha256"])
    value = {"schema": SCHEMA, "lineage": lineage, "arm": arm,
        "parent_manifest_sha256": _pin(parent_manifest["manifest_sha256"]),
        "comparison_origin_manifest_sha256": _pin(comparison_origin),
        "root_manifest_sha256": _pin(config.get("topology_migration", {}).get(
            "root_manifest_sha256", parent_manifest["manifest_sha256"])),
        "parent_state_sha256": _pin(parent_manifest["state"]["sha256"]),
        "parent_configuration_sha256": digest(config),
        "parent_source_fingerprint": copy.deepcopy(parent_manifest["metadata"]["source_fingerprint"]),
        "source_world_size": source_world, "destination_world_size": world_size,
        "source_physical_batch_per_rank": parent_manifest["rank_cursors"][0]["physical_batch_per_rank"],
        "physical_batch_per_rank": physical_batch_per_rank,
        "origin_counters": copy.deepcopy(parent_manifest["counters"]), "origin_cursor": cursor,
        "schedule_contract": copy.deepcopy(config["schedule"]),
        "rng": {"policy": RNG_POLICY, "seed": rng_seed, "source_rank_for_destination": list(rng_rank_map)},
        "sources": dict(sorted(sources.items())),
        "historical_microbatches": "retain-origin-then-add-current-topology-global-slots",
        "schedule_transition": "unchanged-finite-prefix", "model_or_objective_change": False}
    value["sha256"] = digest(value)
    return value


def validate_topology_contract(migration, parent_manifest):
    if not isinstance(migration, dict) or migration.get("schema") != SCHEMA:
        raise ValueError("Unknown topology migration schema")
    expected = make_topology_contract(parent_manifest,
        world_size=migration["destination_world_size"], physical_batch_per_rank=migration["physical_batch_per_rank"],
        lineage=migration["lineage"], sources=migration["sources"], rng_seed=migration["rng"]["seed"],
        rng_rank_map=migration["rng"]["source_rank_for_destination"])
    if migration != expected:
        raise ValueError("Topology migration differs from authenticated parent or declared policy")
    return expected


def destination_configuration(parent_configuration, migration):
    if digest(parent_configuration) != migration["parent_configuration_sha256"]:
        raise ValueError("Original configuration differs from migration authority")
    required = ("model", "backbone", "recipe", "training", "schedule", "ddp")
    result = {key: copy.deepcopy(parent_configuration[key]) for key in required}
    execution = parent_configuration.get("execution")
    if execution is None:
        execution = parent_configuration["execution_identity"]["payload"]["execution"]
    data = parent_configuration.get("data")
    if data is None:
        data = parent_configuration["execution_identity"]["payload"]["data"]
    result.update(schema=CONFIGURATION_SCHEMA, execution=copy.deepcopy(execution),
        data=copy.deepcopy(data),
        world_size=migration["destination_world_size"], physical_batch_per_rank=migration["physical_batch_per_rank"],
        topology_migration=copy.deepcopy(migration), parent_configuration_sha256=migration["parent_configuration_sha256"],
        execution_identity=execution_identity(migration))
    return result


def execution_identity(migration):
    """Fresh identity compatible with unchanged SSD/cloud receipt validation.

    The outer identity format belongs to the established storage contract;
    the payload explicitly identifies this new topology execution and never
    copies the old world's partition declarations.
    """
    payload = {"schema": "olmo-topology-execution-payload-v1", "arm": migration["arm"],
        "partition": {"world_size": migration["destination_world_size"],
                      "physical_batch_per_rank": migration["physical_batch_per_rank"]},
        "cursor_schema": CURSOR_SCHEMA, "parent_configuration_sha256": migration["parent_configuration_sha256"],
        "topology_migration": copy.deepcopy(migration)}
    return {"schema": "olmo-campaign-execution-identity-v1", "payload": payload, "sha256": digest(payload)}


def destination_fingerprint(migration):
    identity = execution_identity(migration)
    return {"sha256": identity["sha256"], "scope": SCHEMA,
            "execution_identity_sha256": identity["sha256"], "migration_sha256": migration["sha256"],
            "parent_manifest_sha256": migration["parent_manifest_sha256"],
            "root_manifest_sha256": migration["root_manifest_sha256"],
            "parent_source_fingerprint": copy.deepcopy(migration["parent_source_fingerprint"]),
            "sources": copy.deepcopy(migration["sources"])}


def expected_counters_since_origin(origin, plans, completed, recipe, *, world_size, batch_size):
    """`plans` is the complete finite plan; `completed` is an absolute update."""
    origin = asdict(origin) if isinstance(origin, TrainingCounters) else dict(origin)
    result = TrainingCounters(**origin)
    _positive(world_size, "world size"); _positive(batch_size, "physical batch")
    if type(completed) is not int or not result.optimizer_updates <= completed <= len(plans):
        raise ValueError("Completed update is outside the unchanged finite plan")
    for plan in plans[result.optimizer_updates:completed]:
        result.optimizer_updates += 1
        result.input_tokens += plan.counts.valid_tokens
        result.documents += plan.counts.packed_rows
        result.microbatches += world_size * math.ceil(len(plan.rows) / (world_size * batch_size))
        result.ce_positions += plan.counts.ce_targets
        if recipe.nextlat:
            result.latent_pairs += plan.counts.latent_pairs
            result.kl_triples += plan.counts.kl_triples
    return result
