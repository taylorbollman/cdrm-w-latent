#!/usr/bin/env python3
"""Bounded single-device checkpoint recovery with CUDA graphs rebuilt on both branches.

This is a recovery/functionality probe, not a training-quality or throughput run.
The reference retains its model/Adam objects but discards its old graph and plan.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters, _rng_state, _restore_rng,
    save_training_checkpoint, load_training_checkpoint)
from scripts.olmo_f1_common import boundary_digests
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_large_batch_validation import (snapshot_eager_cpu, replay_compare_cpu,
    snapshot_replay_cpu, release_graph_then_compare_eager_cpu)
from scripts.olmo_rt_large_batch import (SOURCES as LARGE_BATCH_SOURCES, ARMS, batch_for,
    set_arm, optimizer_for, optimizer_flags, compiler_configuration, compiler_observations,
    dependency_record, check_dependencies, build_model, new_plan, selected_case,
    state_health, require_container_gpu, configure_determinism, backend_context,
    validate_prepared_manifest, load_native_state_dict, load_native_tokenizer, OnlineTracker)

PROTOCOL = ROOT / "docs/reports/olmo-graph-recovery/protocol.md"
SOURCES = tuple(sorted(set(LARGE_BATCH_SOURCES) | {"scripts/olmo_graph_recovery.py"}))
ARM = "compiled-native"
SEED = 20260922


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("rt", "combined"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(2, 8), default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error("Recovery checkpoint must remain under the persistent project checkout")
    return args


@contextmanager
def preserve_rng():
    """Tracking must never perturb the checkpoint/rebuild experiment's RNG."""
    saved = _rng_state(None)
    try:
        yield
    finally:
        _restore_rng(saved, None)


def rng_digests():
    return tree_digests(_rng_state(None))


def recovery_configuration(model, optimizer, case):
    """Actual execution flags, not merely CLI intent, form the strict resume key."""
    base = model.backbone.backbone
    return {"schema": "olmo-graph-recovery-configuration-v1", "case": asdict(case),
        "mode": asdict(case.mode()), "model": asdict(base.config),
        "nextlat": asdict(model.config), "nextlat_enabled": model.enabled,
        "gamma": model.gamma, "precision": "bf16_mixed", "parameter_dtype": "float32",
        "training": asdict(LMTrainingConfig(precision="bf16_mixed")),
        "tf32": False, "autocast_weight_cache": False, "ordinary_sdpa_dispatch": "flash",
        "ordinary_activation_checkpointing": base.ordinary_activation_checkpointing,
        "ordinary_checkpoint_layers": base.ordinary_checkpoint_layers,
        "ordinary_attention_backend": base.ordinary_attention_backend,
        "ordinary_pointwise_backend": base.ordinary_pointwise_backend,
        "ordinary_rope_backend": base.ordinary_rope_backend,
        "cast_weights_once": base.cast_weights_once, "reuse_rope": base.reuse_rope,
        "kv_only_writes": base.kv_only_writes, "tile_backend": base.tile_backend,
        "backward_tile_backend": base.backward_tile_backend, "backward_memory": base.backward_memory,
        "optimizer_class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
        "optimizer_flags": optimizer_flags(optimizer), "world_size": 1, "accumulation": 1,
        "supervision": "all_valid_ce", "seed": SEED, "capture_warmup_backwards": 10,
        "release_transient_cache": True, "preparation_updates": 2, "continuation_updates": 2,
        "reference_graph_rebuilt": True}


def reference_digests(reference):
    """Compare values across reconstructed objects without comparing addresses."""
    return {"losses": tree_digests(reference.losses), "gradients": tree_digests(reference.gradients),
        "declared_names": sorted(reference.declared_names), "expected_names": sorted(reference.expected_names)}


def equality_check(name, reference, actual):
    keys = sorted(set(reference) | set(actual))
    matches = {key: key in reference and key in actual and reference[key] == actual[key] for key in keys}
    return {"name": name, "passed": bool(keys) and all(matches.values()),
        "bitwise_manifest_equal": matches, "scope": "Exact tensor dtype/shape/SHA256 and scalar equality"}


def boundary_record(model, optimizer, scheduler, counters, cursor):
    return {"state": boundary_digests(model, optimizer, scheduler, counters),
        "rng": rng_digests(), "cursor": dict(cursor)}


def cursor_for(case, update):
    if type(update) is not int or update < 0:
        raise ValueError("Next update index must be a nonnegative integer")
    return {"schema": "olmo-graph-recovery-fixture-v1", "next_update": update,
        "batch_size": case.batch_size, "length": case.length, "fixture": "full-ce-real-text-rotation"}


def validate_cursor(cursor, case, counters):
    expected = cursor_for(case, counters.optimizer_updates)
    if cursor != expected:
        raise ValueError("Recovery cursor differs from the case or completed-update boundary")


def verified_discard_checkpoint(path, receipt, checks, *, expected_updates, actual_updates):
    """Delete only this successful, hash-verified disposable diagnostic checkpoint."""
    path = Path(path)
    if not checks or not all(check.get("passed") is True for check in checks):
        raise ValueError("Retain checkpoint after any incomplete or failed recovery check")
    if actual_updates != expected_updates:
        raise ValueError("Retain checkpoint after an unexpected physical update count")
    if str(path) != receipt["path"] or path.stat().st_size != receipt["size_bytes"]:
        raise ValueError("Checkpoint identity differs from the saved receipt")
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("Checkpoint bytes changed after recovery")
    path.unlink()
    return {"deleted_after_success": True, "sha256": receipt["sha256"],
        "size_bytes": receipt["size_bytes"], "reason": "Disposable two-update diagnostic; source checkpoint retained separately"}


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)  # Must precede any CUDA initialization.
    runtime = require_container_gpu()  # No silent CPU fallback.
    torch.set_num_threads(4)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    compiler = compiler_configuration()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        destination = args.output_dir / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    case = selected_case(args.case, batch=args.batch_size, length=512)
    report = {"schema": "olmo-graph-recovery-v1", "status": "running", "stage": "load",
        "runtime": runtime, "determinism": determinism, "checks": [],
        "started_utc": datetime.now(timezone.utc).isoformat(), "physical_optimizer_updates": 0,
        "branch_physical_optimizer_updates": {"preparation": 0, "reference": 0, "restored": 0},
        "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "compiler_configuration": compiler,
        "scope": "Both continuation branches rebuild CUDA graphs. Not an uninterrupted-live-graph save test."}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-graph-recovery",
        name=args.output_dir.name, output_dir=args.output_dir, preserve_state=preserve_rng)
    hook = None
    plan = None
    branch = "preparation"

    def persist(stage=None):
        if stage is not None:
            report["stage"] = stage
            print({"stage": stage, "utc": datetime.now(timezone.utc).isoformat()}, flush=True)
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)

    def publish(check):
        report["checks"].append(check)
        persist()
        tracker.log({"correctness/passed": int(check["passed"]),
            "physical_optimizer_updates": report["physical_optimizer_updates"]})
        if not check["passed"]:
            raise AssertionError(check["name"])

    def install_step_counter(optimizer):
        def completed(_optimizer, _args, _kwargs):
            report["physical_optimizer_updates"] += 1
            report["branch_physical_optimizer_updates"][branch] += 1
            persist()  # Survives a subsequent scheduler/metric failure.
        return optimizer.register_step_post_hook(completed)

    def capture_checked(plan, label, expected=None):
        persist(label + "_capture")
        before_rng = rng_digests()
        refs = snapshot_eager_cpu(plan, plan.batch)
        values = reference_digests(refs)
        if expected is not None:
            publish(equality_check(label + "_boundary_loss_gradients", expected, values))
        plan.capture(warmup=10, release_transient_cache=True)
        publish(replay_compare_cpu(plan, refs, label + "_eager_graph", replays=2))
        del refs
        publish(equality_check(label + "_capture_rng", {"rng": before_rng}, {"rng": rng_digests()}))
        return values

    def updates(plan, optimizer, scheduler, counters, cursor, count, label):
        persist(label + "_updates")
        rows = report.setdefault(label + "_records", [])
        for _ in range(count):
            validate_cursor(cursor, case, counters)
            index = cursor["next_update"]
            batch = batch_for(tokenizer, case, index)
            row = plan.optimizer_step(optimizer, batch, replay=True, scheduler=scheduler, counters=counters)
            cursor = cursor_for(case, index + 1)
            rows.append({"batch": tree_digests(vars(batch)), "metrics": row, "cursor": cursor})
            persist()
            tracker.log({label + "/objective": row["objective"],
                label + "/gradient_norm": row["gradient_norm_before_clip"],
                "physical_optimizer_updates": report["physical_optimizer_updates"]})
        return cursor

    def terminal(plan, label):
        refs = snapshot_replay_cpu(plan)
        values = reference_digests(refs)
        publish(release_graph_then_compare_eager_cpu(plan, refs, label + "_released_graph_eager"))
        del refs
        return values

    try:
        persist()
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        fingerprint = {"checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "source_hashes": report["source_hashes"], "protocol_sha256": report["protocol_sha256"]}
        report["dependencies"] = dependency_record(args.output_dir, include_dao=False, include_fa4=False)
        tracker.start({"case": asdict(case), "arm": ARMS[ARM], "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        tokenizer = load_native_tokenizer(args.artifacts)
        source = load_native_state_dict(args.artifacts)
        model = build_model(source, case)
        del source
        set_arm(model, ARM)
        optimizer, scheduler = optimizer_for(model, ARM)
        counters = TrainingCounters()
        cursor = cursor_for(case, 0)
        hook = install_step_counter(optimizer)
        with backend_context("flash"):
            plan = new_plan(model, batch_for(tokenizer, case, 0), case.mode(), "recompute")
            configuration = recovery_configuration(model, optimizer, case)
            report["configuration"] = configuration
            # A single checkpoint, no duplicate reference checkpoint. Reserve margin
            # for metadata plus failure-safe atomic publication of that one file.
            required = sum(p.numel() * p.element_size() * (3 if p.requires_grad else 1)
                           for p in model.parameters()) + 2**30
            free = shutil.disk_usage(args.output_dir).free
            report["checkpoint_disk_preflight"] = {"estimated_required_bytes": required, "free_bytes": free}
            if free < required:
                raise OSError("Insufficient persistent disk for one complete diagnostic checkpoint")
            capture_checked(plan, "preparation")
            cursor = updates(plan, optimizer, scheduler, counters, cursor, 2, "preparation")
            # Boundary comparison uses the NEXT batch in every rebuilt graph.
            plan.load_batch(batch_for(tokenizer, case, cursor["next_update"]))
            boundary_gradients = terminal(plan, "boundary")
            report["boundary_loss_gradients"] = boundary_gradients
            del plan
            plan = None
            model.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
            boundary = boundary_record(model, optimizer, scheduler, counters, cursor)
            report["boundary"] = boundary
            publish(equality_check("save_execution_contract", configuration, recovery_configuration(model, optimizer, case)))
            persist("save_checkpoint")
            checkpoint_path = args.output_dir / "diagnostic-boundary.pt"
            receipt = save_training_checkpoint(checkpoint_path, model, optimizer, scheduler=scheduler,
                counters=counters, data_cursor=cursor, configuration=configuration, source_fingerprint=fingerprint)
            report["recovery_checkpoint"] = receipt
            publish(equality_check("save_preserves_boundary", boundary,
                boundary_record(model, optimizer, scheduler, counters, cursor)))
            branch = "reference"
            plan = new_plan(model, batch_for(tokenizer, case, cursor["next_update"]), case.mode(), "recompute")
            capture_checked(plan, "reference", boundary_gradients)
            cursor = updates(plan, optimizer, scheduler, counters, cursor, 2, "reference")
            terminal(plan, "reference_final")
            del plan
            plan = None
            model.zero_grad(set_to_none=True)
            reference_final = boundary_record(model, optimizer, scheduler, counters, cursor)
            report["reference_final"] = reference_final
            publish({"name": "reference_state_health", **state_health(model, optimizer)})
            hook.remove(); hook = None
            del model, optimizer, scheduler
            gc.collect(); torch.cuda.empty_cache()
            persist("reconstruct_load_checkpoint")
            source = load_native_state_dict(args.artifacts)
            model = build_model(source, case)
            del source
            set_arm(model, ARM)
            optimizer, scheduler = optimizer_for(model, ARM)
            # Configure the reconstructed runtime before strict checkpoint loading.
            plan = new_plan(model, batch_for(tokenizer, case, 2), case.mode(), "recompute")
            restored_configuration = recovery_configuration(model, optimizer, case)
            del plan
            plan = None
            resumed = load_training_checkpoint(checkpoint_path, model, optimizer, scheduler=scheduler,
                configuration=restored_configuration, source_fingerprint=fingerprint,
                expected_sha256=receipt["sha256"])
            counters, cursor = resumed["counters"], resumed["data_cursor"]
            validate_cursor(cursor, case, counters)
            publish(equality_check("restored_boundary", boundary,
                boundary_record(model, optimizer, scheduler, counters, cursor)))
            branch = "restored"
            hook = install_step_counter(optimizer)
            plan = new_plan(model, batch_for(tokenizer, case, cursor["next_update"]), case.mode(), "recompute")
            capture_checked(plan, "restored", boundary_gradients)
            cursor = updates(plan, optimizer, scheduler, counters, cursor, 2, "restored")
            terminal(plan, "restored_final")
            del plan
            plan = None
            model.zero_grad(set_to_none=True)
            restored_final = boundary_record(model, optimizer, scheduler, counters, cursor)
            report["restored_final"] = restored_final
            publish(equality_check("restored_final_state_rng_cursor", reference_final, restored_final))
            publish(equality_check("restored_update_metrics", {"records": report["reference_records"]},
                {"records": report["restored_records"]}))
            publish({"name": "restored_state_health", **state_health(model, optimizer)})
            publish(equality_check("physical_update_accounting", {"preparation": 2, "reference": 2, "restored": 2},
                report["branch_physical_optimizer_updates"]))
            publish({"name": "frozen_sources_dependencies", "passed": all(
                sha256_file(ROOT / name) == digest and sha256_file(args.output_dir / "source-snapshot" / name) == digest
                for name, digest in report["source_hashes"].items()) and check_dependencies(report["dependencies"])
                and sha256_file(PROTOCOL) == report["protocol_sha256"]
                and sha256_file(args.output_dir / "protocol.md") == report["protocol_sha256"]})
            report["compiler_observations"] = compiler_observations()
            report["checkpoint_disposal"] = verified_discard_checkpoint(checkpoint_path, receipt, report["checks"],
                expected_updates=6, actual_updates=report["physical_optimizer_updates"])
            report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        persist()
        raise
    finally:
        if hook is not None:
            hook.remove()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        except BaseException as error:
            report.update(status="failed", tracking_error_type=type(error).__name__,
                tracking_error=str(error))
            raise
        finally:
            persist()
    return report


if __name__ == "__main__":
    main()
