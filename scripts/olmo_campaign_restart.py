#!/usr/bin/env python3
"""Two separately launched torchrun phases for campaign checkpoint acceptance.

``write`` saves after one accumulated graphed update and records the next update
using the SAME live graph. ``resume`` must be a fresh process: restore before
constructing DDP/graphs, recapture without optimizer/clock/RNG changes, and match
the uninterrupted next update exactly. Both phases require two CUDA devices.
Checkpoint publication to durable storage is a separate launcher operation;
this program never deletes or silently replaces local checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_recipe import build_campaign_adamw, CampaignTokenSchedule
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.distributed_checkpoint import (
    _local_rng, load_distributed_checkpoint, save_distributed_checkpoint,
)
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import boundary_digests, state_health
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_recovery import coordinated, draw_rng, seed_local
from scripts.olmo_two_gpu_validate import gather, preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = "olmo-campaign-fresh-process-restart-v1"
CURSOR_SCHEMA = "olmo-campaign-restart-fixture-cursor-v1"
TERMS = ("ce", "latent", "kl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write", "resume"), required=True)
    parser.add_argument("--scale", choices=("tiny", "pretrained"), required=True)
    parser.add_argument("--arm", choices=("B", "NFR"), default="NFR")
    parser.add_argument("--length", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args(argv)
    args.length = (8 if args.scale == "tiny" else 16) if args.length is None else args.length
    if args.length != (8 if args.scale == "tiny" else 16):
        parser.error("Bounded restart requires T8 tiny or T16 pretrained")
    resume_fields = (args.reference_report, args.reference_sha256, args.expected_manifest_sha256)
    if args.phase == "write" and any(value is not None for value in resume_fields):
        parser.error("Reference and manifest pins belong to the resume phase only")
    if args.phase == "resume" and any(value is None for value in resume_fields):
        parser.error("Resume requires reference report SHA256 and checkpoint manifest SHA256 pins")
    for digest in (args.reference_sha256, args.expected_manifest_sha256):
        if digest is not None and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            parser.error("SHA256 pins must be 64 lowercase hex characters")
    return args


def cursor_for(recipe_sha256, rank, next_update, *, length):
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Restart fixture requires rank zero or one")
    if type(next_update) is not int or not 0 <= next_update <= 2:
        raise ValueError("Restart fixture has exactly two logical updates")
    return {"schema": CURSOR_SCHEMA, "world_size": 2, "rank": rank,
            "recipe_sha256": recipe_sha256, "next_update": next_update,
            "fixture_update": next_update + 1, "physical_batch_per_rank": 2,
            "length": length, "fixture": "olmo_campaign_ddp_probe.fixture_for_update-v1",
            "microbatches_per_rank": (2, 3, None)[next_update]}


def validate_cursor(cursor, recipe_sha256, rank, counters, *, length):
    expected = cursor_for(recipe_sha256, rank, counters.optimizer_updates, length=length)
    if cursor != expected:
        raise ValueError("Restart cursor differs from recipe/rank/counters/fixture")


def compare_continuation(actual, expected):
    """No approximate budget can hide a restart-only difference here."""
    fields = ("input", "raw_gradients", "metrics", "boundary", "rng_draws")
    rows = {name: name in actual and name in expected and actual[name] == expected[name]
            for name in fields}
    return {"passed": all(rows.values()), "bitwise_checks": rows}


def validate_reference(path, expected_sha256, *, scale, arm, length,
                       expected_manifest_sha256, sources):
    if sha256_file(path) != expected_sha256:
        raise ValueError("Reference report SHA256 differs from the supplied pin")
    reference = json.loads(Path(path).read_text())
    if (reference.get("schema") != SCHEMA or reference.get("phase") != "write"
            or reference.get("status") != "passed" or not reference.get("passed")):
        raise ValueError("Reference must be a successfully completed write-phase report")
    if (reference.get("scale"), reference.get("arm"), reference.get("length")) != (scale, arm, length):
        raise ValueError("Reference model/fixture selection differs")
    if reference.get("sources") != sources:
        raise ValueError("Reference source pins differ; resume under the recorded implementation")
    if reference.get("checkpoint", {}).get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Reference checkpoint manifest pin differs")
    if len(reference.get("continuation", [])) != 2 or len(reference.get("saved_boundaries", [])) != 2:
        raise ValueError("Reference is missing a rank's continuation or checkpoint boundary")
    return reference


def source_hashes():
    from scripts.olmo_campaign_ddp_probe import source_hashes as probe_sources
    sources = probe_sources()
    helpers = (Path(__file__), ROOT / "scripts/olmo_two_gpu_recovery.py",
               ROOT / "scripts/olmo_two_gpu_validate.py", ROOT / "scripts/olmo_lm_common.py",
               ROOT / "scripts/olmo_f1_common.py", ROOT / "scripts/olmo_distributed_prepare.py")
    sources.update({str(path.relative_to(ROOT)): sha256_file(path) for path in helpers})
    return dict(sorted(sources.items()))


def boundary(model, optimizer, scheduler, counters, cursor, device, generators):
    return {"state": boundary_digests(model, optimizer, scheduler, counters),
            "cursor": cursor, "rng": tree_digests(_local_rng(device, generators))}


def checkpoint_disk_preflight(path, model):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.numel() * p.element_size() * (3 if p.requires_grad else 1)
                   for p in model.parameters()) + 2 ** 30
    free = shutil.disk_usage(path.parent).free
    if free < required:
        raise OSError(f"Checkpoint needs approximately {required} bytes; only {free} free")
    return {"path": str(path), "estimated_required_bytes": required, "free_bytes": free,
            "durability": "Local publication only; launcher must verify GCS copy before releasing VM"}


def main(argv=None):
    args = parse_args(argv)
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Run inside the required project Docker container")
    if os.environ.get("WORLD_SIZE") != "2":
        raise RuntimeError("Restart acceptance requires torchrun --nproc-per-node=2")
    if any(os.environ.get(key) != "0" for key in ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Distributed graph capture requires both NCCL async-error flags=0 and external timeout")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    runtime_info = require_container_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=device)
    rank = dist.get_rank()
    tracker = None
    start = time.monotonic()
    report = {"schema": SCHEMA, "phase": args.phase, "scale": args.scale, "arm": args.arm,
              "length": args.length, "runtime": runtime_info, "sources": source_hashes(),
              "status": "running", "passed": False, "checks": [],
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Real two-rank NCCL graphs, same-world-size fresh-process restart; uninterrupted live-graph reference; bounded functionality only"}

    def persist(stage=None):
        if stage is not None:
            report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - start
        if rank == 0:
            if tracker is not None:
                report["wandb"] = tracker.record
            write_json(args.output_dir / "report.json", report)

    def publish(name, value):
        flags = gather(bool(value))
        row = {"name": name, "passed": all(flags), "rank_passed": flags}
        report["checks"].append(row)
        coordinated("persist restart gate", lambda: persist(name))
        if not row["passed"]:
            raise AssertionError(f"Restart acceptance failed: {name}; rank flags={flags}")

    error = None
    try:
        def setup():
            nonlocal tracker
            if rank:
                return
            args.output_dir.mkdir(parents=True, exist_ok=False)
            for relative in report["sources"]:
                destination = args.output_dir / "source-snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                group="olmo-campaign-two-gpu", name="campaign-restart-" + args.output_dir.name,
                preserve_state=preserve_local_rng)
            tracker.start({"phase": args.phase, "scale": args.scale, "arm": args.arm,
                           "length": args.length, "scope": report["scope"]})
            print({"wandb": tracker.record["run_url"]}, flush=True)
        coordinated("restart setup", setup)
        coordinated("persist restart setup", lambda: persist("setup"))
        backend = SDPBackend.MATH if args.scale == "tiny" else SDPBackend.FLASH_ATTENTION
        with disable_autocast_weight_cache(), sdpa_kernel(backend):
            run(args, report, device, tracker, persist, publish)
        publish("source_pins_unchanged", source_hashes() == report["sources"])
        report.update(status="passed", passed=True)
    except BaseException as exc:
        error = exc
        report.update(status="failed", passed=False,
                      error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if rank == 0 and args.output_dir.is_dir():
            persist()
            try:
                if tracker is not None:
                    tracker.finish(succeeded=report["passed"])
            except BaseException as finish_error:
                report.update(status="failed", passed=False,
                              tracking_finish_error={"type": type(finish_error).__name__})
                if error is None:
                    raise
            finally:
                persist()
        if error is None:
            dist.destroy_process_group()


def run(args, report, device, tracker, persist, publish):
    from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
    from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update

    rank = dist.get_rank()
    reference = None
    if args.phase == "resume":
        reference = coordinated("validate pinned reference", lambda: validate_reference(
            args.reference_report, args.reference_sha256, scale=args.scale, arm=args.arm,
            length=args.length, expected_manifest_sha256=args.expected_manifest_sha256,
            sources=report["sources"]))
        report["reference_report"] = {"path": str(args.reference_report), "sha256": args.reference_sha256}
    model, recipe, checkpoint, tokens, eos = construct(args, args.arm, device)
    fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, update,
                    length=args.length, token_ids=tokens, eos_id=eos, batch_size=2)
                for update in (1, 2)]
    fixture_inputs = [tree_digests({"batches": [vars(b) for b in batches], "noise": noises})
                      for batches, noises in fixtures]
    plans = gather([{"counts": sum_objective_counts([model.counts(b) for b in batches]),
                     "input_tokens": sum(int(b.valid_mask.sum()) for b in batches),
                     "microbatches": len(batches)} for batches, _ in fixtures])
    global_counts = [sum_objective_counts([p[index]["counts"] for p in plans]) for index in range(2)]
    update_tokens = [sum(p[index]["input_tokens"] for p in plans) for index in range(2)]
    optimizer = build_campaign_adamw(model, recipe, fused=True)
    scheduler = CampaignTokenSchedule(optimizer, update_tokens, warmup_tokens=recipe.warmup_tokens,
                                     start_fraction=recipe.warmup_start_fraction)
    config = LMTrainingConfig(precision="fp32" if args.scale == "tiny" else "bf16_mixed",
                             max_grad_norm=recipe.max_grad_norm)
    configuration = {"schema": SCHEMA, "scale": args.scale, "arm": args.arm, "length": args.length,
                     "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256,
                     "world_size": 2, "precision": config.precision, "training": asdict(config),
                     "schedule": scheduler.checkpoint_contract(), "plans": plans,
                     "fixture_inputs_by_rank": gather(fixture_inputs),
                     "source_checkpoint": checkpoint, "runtime": report["runtime"],
                     "ddp": {"static_graph": True, "find_unused_parameters": False,
                             "gradient_as_bucket_view": False, "broadcast_buffers": False,
                             "bucket_cap_mb": 25}, "warmup": 11}
    fingerprint = {"checkpoint_sha256": checkpoint.get("sha256", "0" * 64),
                   "recipe_sha256": recipe.sha256, "checkpoint": checkpoint,
                   "source_hashes": report["sources"]}
    report.update(configuration=configuration, source_checkpoint=checkpoint)
    generators = {"data": torch.Generator().manual_seed(8300 + rank),
                  "local": torch.Generator(device=device).manual_seed(8400 + rank)}
    seed_local(8500 + rank, device)
    counters = TrainingCounters()
    cursor = cursor_for(recipe.sha256, rank, 0, length=args.length)
    if args.phase == "resume":
        publish("reference_execution_configuration_exact", configuration == reference["configuration"])
        resumed = load_distributed_checkpoint(args.checkpoint_dir, model, optimizer,
            scheduler=scheduler, configuration=configuration, source_fingerprint=fingerprint,
            generators=generators, expected_manifest_sha256=args.expected_manifest_sha256, device=device)
        counters, cursor = resumed["counters"], resumed["data_cursor"]
        report["checkpoint"] = resumed["manifest"]
        coordinated("restored cursor", lambda: validate_cursor(cursor, recipe.sha256, rank, counters, length=args.length))
        publish("restored_saved_boundary_exact", boundary(model, optimizer, scheduler, counters,
            cursor, device, generators) == reference["saved_boundaries"][rank])
    initial_boundary = boundary(model, optimizer, scheduler, counters, cursor, device, generators)
    batches, noises = fixtures[cursor["next_update"]]
    adapter = CampaignObjective(model, batches[0], mode=recipe.mode(),
        global_counts=global_counts[cursor["next_update"]], world_size=2,
        feedback_noise=noises[0], config=config)
    runner = CampaignDDPGraphTraining(adapter)
    coordinated("persist before capture", lambda: persist("prepare_graphs"))
    runner.prepare(warmup=11)
    runner.capture(warmup=11)
    publish("graph_prepare_preserves_model_adam_rng_clocks", boundary(model, optimizer, scheduler,
        counters, cursor, device, generators) == initial_boundary)

    def update(label):
        nonlocal cursor
        coordinated("validate update cursor", lambda: validate_cursor(cursor, recipe.sha256, rank, counters, length=args.length))
        index = cursor["next_update"]
        batches, noises = fixtures[index]
        draws = draw_rng(device, generators)
        before_rng = tree_digests(_local_rng(device, generators))
        result = runner.backward(batches, feedback_noises=noises, replay=True)
        raw = tree_digests({name: p.grad for name, p in model.named_parameters() if p.grad is not None})
        replica_raw = gather(raw)
        publish(label + "/replica_raw_gradients_exact", all(value == replica_raw[0] for value in replica_raw))
        metrics = runner.step(result, optimizer, scheduler=scheduler, counters=counters)
        cursor = cursor_for(recipe.sha256, rank, counters.optimizer_updates, length=args.length)
        current = boundary(model, optimizer, scheduler, counters, cursor, device, generators)
        states = gather(current["state"])
        publish(label + "/replica_state_exact", all(value == states[0] for value in states))
        publish(label + "/finite_state", state_health(model, optimizer)["passed"])
        publish(label + "/execution_preserves_rng", current["rng"] == before_rng)
        row = {"input": fixture_inputs[index], "raw_gradients": raw, "metrics": metrics,
               "boundary": current, "rng_draws": draws}
        records = gather(row)
        report.setdefault("updates", []).append({"label": label, "ranks": records})
        coordinated("persist update", lambda: persist(label))
        coordinated("log completed update", lambda: tracker.log(
            scalar_metrics(metrics, "diagnostic/" + label), step=len(report["updates"])) if rank == 0 else None)
        return records

    if args.phase == "write":
        update("first_update")
        saved = boundary(model, optimizer, scheduler, counters, cursor, device, generators)
        report["saved_boundaries"] = gather(saved)
        report["checkpoint_disk_preflight"] = gather(coordinated("checkpoint disk preflight",
            lambda: checkpoint_disk_preflight(args.checkpoint_dir, model)))
        coordinated("persist before save", lambda: persist("saving_checkpoint"))
        with runner.checkpoint_boundary():
            receipt = save_distributed_checkpoint(args.checkpoint_dir, model, optimizer,
                scheduler=scheduler, counters=counters, data_cursor=cursor,
                configuration=configuration, source_fingerprint=fingerprint,
                generators=generators, device=device)
        report["checkpoint"] = receipt
        publish("checkpoint_save_preserves_live_graph_boundary", saved == boundary(
            model, optimizer, scheduler, counters, cursor, device, generators))
        # Critically, keep the original trained graph. Recapturing the reference
        # here would conceal a rebuild-only difference in the restart comparison.
        report["continuation"] = update("uninterrupted_next_update")
    else:
        report["continuation"] = update("resumed_next_update")
        comparison = compare_continuation(report["continuation"][rank], reference["continuation"][rank])
        report["continuation_comparisons"] = gather(comparison)
        publish("fresh_process_next_update_bitwise_exact", comparison["passed"])
    report["runner"] = gather(runner.metadata)
    torch.cuda.synchronize(device)
    coordinated("persist complete phase", lambda: persist("phase_complete"))
    del runner, adapter, model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
