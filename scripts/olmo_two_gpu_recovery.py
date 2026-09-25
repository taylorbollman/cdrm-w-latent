#!/usr/bin/env python3
"""Bounded eager two-H100 checkpoint/reconstruction continuation checks.

Two preparation updates establish Adam state. A committed shared checkpoint is
followed by one reference update; fresh model/optimizer/DDP construction then
resumes that checkpoint and must reproduce the same update exactly. Graphs are
not captured here. Checkpoints are retained for separately verified GCS upload.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import gc
import json
import os
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
import torch.distributed as dist

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.ddp_training import EagerDDPTrainer
from cdrm.pretrained.distributed_checkpoint import (
    _local_rng, load_distributed_checkpoint, save_distributed_checkpoint,
)
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import IntegrationCase, boundary_digests, state_health
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_rt_large_batch import (
    OnlineTracker, backend_context, compiler_configuration, configure_determinism,
    load_native_tokenizer, optimizer_for, require_container_gpu, validate_prepared_manifest,
    finish_tracking,
)
from scripts.olmo_two_gpu_validate import construct, gather, make_batch, move_batch, preserve_local_rng

PROTOCOL = ROOT / "docs/reports/olmo-two-gpu/protocol.md"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def coordinated(phase, work):
    result = None
    error = None
    try:
        result = work()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = gather(error)
    if any(item is not None for item in errors):
        raise RuntimeError(f"{phase}: " + "; ".join(
            f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None))
    return result


def ensure_persistent_checkpoint_directory(path):
    """This harness deliberately has no disposable-SSD checkpoint mode."""
    path = Path(path).resolve()
    disposable = Path(os.environ.get("CDRM_LOCALSSD_MOUNT", "/mnt/localssd")).resolve()
    if path == disposable or disposable in path.parents:
        raise ValueError("Recovery checkpoint must use persistent project/home storage, not local SSD")
    if path.exists():
        raise FileExistsError(f"Recovery checkpoint directory already exists: {path}")
    return path


def cursor_for(case, rank, update):
    return {"schema": "olmo-two-gpu-recovery-cursor-v1", "rank": rank, "world_size": 2,
            "next_update": update, "microbatches_per_rank": 2,
            "batch_size_per_rank": case.batch_size, "length": case.length,
            "batch_generator": "olmo_two_gpu_recovery.recovery_batch-v1"}


def validate_cursor(cursor, case, rank, counters):
    expected = cursor_for(case, rank, counters.optimizer_updates)
    if cursor != expected:
        raise ValueError("Recovery cursor differs from global counters/case/rank")


def recovery_batch(case, tokenizer, update, rank, micro, *, tiny):
    """Keep auxiliary gradients active on resumed combined updates.

    The eager validation fixture deliberately disables them globally on update3.
    Recovery instead retains rank0/first-microbatch predictor participation on
    every update, preserving independently generated update-specific tokens.
    """
    batch = make_batch(case, tokenizer, update, rank, micro, tiny=tiny)
    if case.nextlat and update >= 2 and rank == 0 and micro == 0:
        latent, kl = batch.valid_mask.clone(), batch.valid_mask.clone()
        latent[:, 1::3] = False
        kl[:, 2::4] = False
        batch = replace(batch, latent_mask=latent, kl_mask=kl)
    return batch


def seed_local(seed, device):
    """Set CPU and current-device seeds without cuda.manual_seed_all."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def draw_rng(device, generators):
    return tree_digests({
        "python": random.random(), "numpy": np.random.random(5).tolist(),
        "torch_cpu": torch.rand(5), "torch_local": torch.rand(5, device=device),
        "explicit_data": torch.rand(5, generator=generators["data"]),
        "explicit_local": torch.rand(5, device=device, generator=generators["local"]),
    })


def local_boundary(model, optimizer, scheduler, counters, cursor, device, generators):
    return {"state": boundary_digests(model, optimizer, scheduler, counters),
            "rng": tree_digests(_local_rng(device, generators)), "cursor": cursor}


def execution_configuration(model, case, config, args):
    base = model.backbone.backbone
    runtime_names = ("attention_backend", "attention_precision", "ordinary_attention_backend",
                     "ordinary_rope_backend", "ordinary_pointwise_backend",
                     "ordinary_activation_checkpointing", "cast_weights_once", "reuse_rope",
                     "kv_only_writes", "tile_backend", "backward_tile_backend", "backward_memory")
    return {"schema": "olmo-two-gpu-recovery-execution-v1", "case": asdict(case),
            "tiny": args.tiny, "world_size": 2, "training": asdict(config),
            "nextlat": model.config.to_dict(), "nextlat_enabled": model.enabled,
            "fbt_enabled": case.fbt, "gamma": model.gamma,
            "runtime": {name: getattr(base, name, None) for name in runtime_names},
            "ordinary_mask": "full_valid_causal", "optimizer_arm": "compiled-native",
            "ddp": {"find_unused_parameters": True, "gradient_as_bucket_view": False,
                    "broadcast_buffers": False, "static_graph": False},
            "autocast_cache_enabled": False, "tf32": False, "graphs": False}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--case", choices=("rt", "combined"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 8 <= args.length <= 2048:
        parser.error("batch size must be positive and length must be between 8 and 2048")
    return args


def run_case(args, report, tracker, persist, publish):
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    case = IntegrationCase(args.case, fbt=args.case == "combined", nextlat=args.case == "combined",
                           rt_layers=(0, 1 if args.tiny else 15), batch_size=args.batch_size,
                           length=args.length, updates=3)
    config = LMTrainingConfig(precision="fp32" if args.tiny else "bf16_mixed")
    seed_local(20260925, device)
    model = construct(case, args, device)
    optimizer, scheduler = optimizer_for(model, "compiled-native")
    trainer = EagerDDPTrainer(model)
    counters = TrainingCounters()
    cursor = cursor_for(case, rank, 0)
    generators = {"data": torch.Generator().manual_seed(6300 + rank),
                  "local": torch.Generator(device=device).manual_seed(6400 + rank)}
    seed_local(6500 + rank, device)
    tokenizer = None if args.tiny else load_native_tokenizer(args.artifacts)
    configuration = execution_configuration(model, case, config, args)
    report["configuration"] = configuration
    report["case"] = asdict(case)
    fingerprint = {"checkpoint_sha256": report["source_checkpoint"]["sha256"],
                   "source_hashes": report["sources"], "protocol_sha256": sha256_file(PROTOCOL)}
    step_calls = 0

    def count_step(*unused):
        nonlocal step_calls
        step_calls += 1
        report["physical_updates_per_rank"] = step_calls

    hook = optimizer.register_step_post_hook(count_step)

    def completed_update(label, model, optimizer, scheduler, trainer, counters, cursor):
        coordinated("cursor/update boundary", lambda: (trainer.assert_update_boundary(),
                                                       validate_cursor(cursor, case, rank, counters)))
        update = cursor["next_update"]
        batches = [move_batch(recovery_batch(case, tokenizer, update, rank, micro, tiny=args.tiny), device)
                   for micro in range(2)]
        batch_digest = tree_digests([vars(batch) for batch in batches])
        result = trainer.backward(batches, config=config,
                                  backbone_kwargs={"mode": case.mode(), "full_valid_causal": True})
        raw_gradients = tree_digests({name: parameter.grad for name, parameter in model.named_parameters()
                                     if parameter.grad is not None})
        replicas = gather(raw_gradients)
        publish(label + "/replica_raw_gradients_exact", replicas[0] == replicas[1])
        metrics = trainer.step(result, optimizer, scheduler=scheduler, counters=counters)
        next_cursor = cursor_for(case, rank, counters.optimizer_updates)
        boundary = local_boundary(model, optimizer, scheduler, counters, next_cursor, device, generators)
        replica_state = gather(boundary["state"])
        publish(label + "/replica_state_exact", replica_state[0] == replica_state[1])
        publish(label + "/finite_state", state_health(model, optimizer)["passed"])
        record = {"metrics": metrics, "raw_gradients": raw_gradients,
                  "batches": batch_digest, "boundary": boundary}
        records = gather(record)
        def record_update():
            if rank == 0:
                report.setdefault("updates", []).append({"label": label, "ranks": records})
                persist(label)
                tracker.log({"physical_updates_per_rank": step_calls,
                             label + "/objective": metrics["objective"],
                             label + "/gradient_norm": metrics["gradient_norm_before_clip"]})
        coordinated("record completed update", record_update)
        return record, next_cursor

    try:
        for index in range(2):
            _, cursor = completed_update(f"preparation_{index + 1}", model, optimizer, scheduler,
                                         trainer, counters, cursor)
        coordinated("save boundary", trainer.assert_update_boundary)
        boundary = local_boundary(model, optimizer, scheduler, counters, cursor, device, generators)
        required = sum(parameter.numel() * parameter.element_size() * (3 if parameter.requires_grad else 1)
                       for parameter in model.parameters()) + 2 ** 30

        def disk_preflight():
            path = ensure_persistent_checkpoint_directory(args.checkpoint_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(path.parent).free
            if free < required:
                raise OSError(f"Checkpoint needs approximately {required} bytes; only {free} are free")
            return {"estimated_required_bytes": required, "free_bytes": free, "path": str(path)}

        preflight = coordinated("checkpoint disk preflight", disk_preflight)
        report["checkpoint_disk_preflight"] = gather(preflight)
        coordinated("record checkpoint preparation", lambda: persist("save_checkpoint"))
        receipt = save_distributed_checkpoint(args.checkpoint_dir, model, optimizer, scheduler=scheduler,
            counters=counters, data_cursor=cursor, configuration=configuration, source_fingerprint=fingerprint,
            generators=generators, device=device)
        report["checkpoint"] = receipt
        publish("save_preserves_boundary", boundary == local_boundary(
            model, optimizer, scheduler, counters, cursor, device, generators))
        def record_checkpoint():
            persist("checkpoint_committed")
            if rank == 0:
                print(json.dumps({"checkpoint_committed": str(args.checkpoint_dir),
                                  "manifest_sha256": receipt["manifest_sha256"],
                                  "state_sha256": receipt["state"]["sha256"]}), flush=True)
        coordinated("record committed checkpoint", record_checkpoint)
        expected_draws = draw_rng(device, generators)
        reference, cursor = completed_update("reference", model, optimizer, scheduler, trainer, counters, cursor)
        hook.remove()
        hook = None
        del trainer, optimizer, scheduler, model
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        coordinated("record reconstruction", lambda: persist("reconstruct_load"))
        model = construct(case, args, device)
        optimizer, scheduler = optimizer_for(model, "compiled-native")
        restored_configuration = execution_configuration(model, case, config, args)
        publish("reconstructed_execution_configuration_exact", restored_configuration == configuration)
        resumed = load_distributed_checkpoint(args.checkpoint_dir, model, optimizer, scheduler=scheduler,
            configuration=restored_configuration, source_fingerprint=fingerprint, generators=generators,
            expected_manifest_sha256=receipt["manifest_sha256"], device=device)
        counters, cursor = resumed["counters"], resumed["data_cursor"]
        coordinated("restored cursor", lambda: validate_cursor(cursor, case, rank, counters))
        publish("restored_boundary_before_ddp_exact", boundary == local_boundary(
            model, optimizer, scheduler, counters, cursor, device, generators))
        trainer = EagerDDPTrainer(model)
        publish("restored_boundary_after_ddp_exact", boundary == local_boundary(
            model, optimizer, scheduler, counters, cursor, device, generators))
        publish("next_rank_local_random_draws_exact", draw_rng(device, generators) == expected_draws)
        hook = optimizer.register_step_post_hook(count_step)
        restored, cursor = completed_update("restored", model, optimizer, scheduler, trainer, counters, cursor)
        publish("continuation_batches_exact", restored["batches"] == reference["batches"])
        publish("continuation_raw_gradients_exact", restored["raw_gradients"] == reference["raw_gradients"])
        publish("continuation_metrics_exact", restored["metrics"] == reference["metrics"])
        publish("continuation_full_state_rng_cursor_exact", restored["boundary"] == reference["boundary"])
        publish("physical_update_accounting", step_calls == 4 and counters.optimizer_updates == 3)
        publish("checkpoint_retained", (args.checkpoint_dir / "manifest.json").is_file()
                and (args.checkpoint_dir / "state.pt").stat().st_size == receipt["state"]["size_bytes"])
        report["physical_updates_per_rank"] = step_calls
        report["total_rank_optimizer_steps"] = sum(gather(step_calls))
        report["logical_endpoint_updates"] = counters.optimizer_updates
        report["checkpoint_disposal"] = {"deleted": False, "policy": "Retain until separately verified GCS upload"}
    finally:
        if hook is not None:
            hook.remove()


def main(argv=None):
    args = parse_args(argv)
    configure_determinism(True)
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    runtime = require_container_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    compiler_configuration()
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
                            timeout=timedelta(minutes=30))
    if dist.get_world_size() != 2:
        raise RuntimeError("Recovery check requires two ranks on two actual GPUs")
    rank = dist.get_rank()
    tracker = None
    original_error = None
    report = {"schema": "olmo-two-gpu-recovery-v1", "started_utc": utc_now(),
              "runtime": runtime, "passed": False, "status": "running", "checks": [],
              "physical_updates_per_rank": 0, "scope": "eager_same_world_size_reconstruction"}

    def persist(stage=None):
        if stage is not None:
            report["stage"] = stage
        if rank == 0:
            write_json(args.output_dir / "progress.json", report)

    def publish(name, passed):
        flags = gather(bool(passed))
        record = {"name": name, "passed": all(flags), "rank_passed": flags}
        report["checks"].append(record)
        coordinated("record check", persist)
        if not record["passed"]:
            raise AssertionError(f"Recovery check failed: {name}; rank flags={flags}")

    try:
        def setup_rank_zero():
            nonlocal tracker
            if rank != 0:
                return None
            args.output_dir.mkdir(parents=True, exist_ok=False)
            sources = [*sorted((ROOT / "cdrm/pretrained").glob("*.py")),
                       *sorted((ROOT / "scripts").glob("olmo*.py")),
                       ROOT / "scripts/experiment_tracking.py", PROTOCOL]
            sources = list(dict.fromkeys(sources))
            hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
            for path in sources:
                target = args.output_dir / "source-snapshot" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            source_checkpoint = ({"sha256": "0" * 64, "kind": "deterministic_tiny_fixture"} if args.tiny else
                                 validate_prepared_manifest(args.artifacts)["checkpoint"])
            tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-two-gpu",
                name=args.output_dir.name, output_dir=args.output_dir, preserve_state=preserve_local_rng)
            tracker.start({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
            print(tracker.record["run_url"], flush=True)
            return {"sources": hashes, "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_checkpoint": source_checkpoint, "wandb": tracker.record}

        setup = coordinated("rank-zero setup", setup_rank_zero)
        report.update(gather(setup)[0])
        if rank == 0:
            report["wandb"] = tracker.record
        coordinated("record setup", lambda: persist("setup"))
        with disable_autocast_weight_cache(), backend_context("math" if args.tiny else "flash"):
            run_case(args, report, tracker, persist, publish)
        publish("frozen_source_snapshot", all(sha256_file(ROOT / name) == digest and
                sha256_file(args.output_dir / "source-snapshot" / name) == digest
                for name, digest in report["sources"].items()))
        report.update(status="passed", passed=True, stage="complete")
    except BaseException as error:
        original_error = error
        report.update(status="failed", passed=False,
                      error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = utc_now()
        if rank == 0:
            if args.output_dir.is_dir():
                write_json(args.output_dir / "report.json", report)
            try:
                if tracker is not None:
                    finish_tracking(tracker, report, original_error=original_error)
            finally:
                report["passed"] = report["status"] == "passed"
                if args.output_dir.is_dir():
                    write_json(args.output_dir / "report.json", report)
                    persist()
        if original_error is None:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
