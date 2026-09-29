#!/usr/bin/env python3
"""Bounded actual-data campaign updates and fresh-process two-GPU recovery.

The write phase saves after one update and records the next update on its
original live graphs. Resume must load a verified cloud copy in new processes
before DDP construction. This is an opt-in readiness runner, not an unattended
quality campaign. The caller supplies an immutable packed index and corpus.
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
from cdrm.pretrained.campaign_ddp_training import CampaignDDPGraphTraining
from cdrm.pretrained.campaign_recipe import (
    CampaignTokenSchedule, build_campaign_adamw, feedback_noise_for_rows,
)
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.distributed_checkpoint import (
    _local_rng, load_distributed_checkpoint, save_distributed_checkpoint,
)
from cdrm.pretrained.distributed_training import sum_objective_counts
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct, source_hashes as probe_sources
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_restart import boundary, checkpoint_disk_preflight, compare_continuation
from scripts.olmo_distributed_prepare import disable_autocast_weight_cache
from scripts.olmo_f1_common import state_health
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_recovery import coordinated, draw_rng, seed_local
from scripts.olmo_two_gpu_validate import gather, preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = "olmo-packed-campaign-readiness-v1"
POLICY = "continuous-stream-v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write", "resume"), required=True)
    parser.add_argument("--arm", choices=("B", "NFR"), default="NFR")
    parser.add_argument("--batch-size", type=int, choices=(8, 12), default=12)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--reference-sha256")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args(argv)
    fields = (args.reference_report, args.reference_sha256, args.expected_manifest_sha256)
    if args.phase == "resume" and any(v is None for v in fields):
        parser.error("Resume requires completed reference report and checkpoint manifest SHA256 pins")
    if args.phase == "write" and any(v is not None for v in fields):
        parser.error("Reference pins belong only to resume")
    for value in (args.index_sha256, args.reference_sha256, args.expected_manifest_sha256):
        if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            parser.error("SHA256 pins must be 64 lowercase hexadecimal characters")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error("Small evidence must remain under the persistent project checkout")
    args.scale, args.length, args.document_policy = "pretrained", 1024, POLICY
    return args


def token_plan(data, target):
    """Derive the finite update schedule by pure peeking, never consume data."""
    cursor, tokens = data.cursor(), []
    while True:
        update = data.peek_update(cursor, target)
        if update is None:
            break
        if update.next_cursor.next_chunk <= cursor.next_chunk:
            raise ValueError("Packed update must advance the chunk cursor")
        value = update.counts.valid_tokens
        if value <= 0:
            raise ValueError("Packed update must contain valid input tokens")
        tokens.append(value)
        cursor = update.next_cursor
    if len(tokens) < 2:
        raise ValueError("Recovery fixture needs at least two logical updates")
    return tokens


def cursor_record(cursor, *, rank, batch_size):
    return {"schema": SCHEMA, "cursor": asdict(cursor), "rank": rank,
            "world_size": 2, "physical_batch_per_rank": batch_size}


def restore_training_cursor(data, record, counters, tokens, *, rank, batch_size, install=False):
    if (record.get("schema") != SCHEMA or record.get("rank") != rank
            or record.get("world_size") != 2 or record.get("physical_batch_per_rank") != batch_size):
        raise ValueError("Cursor execution partition differs")
    cursor_type = type(data.cursor())
    cursor = cursor_type(**record["cursor"])
    if (cursor.next_update != counters.optimizer_updates or cursor.next_update > len(tokens)
            or counters.input_tokens != sum(tokens[:cursor.next_update])):
        raise ValueError("Committed cursor differs from optimizer/token clocks")
    # Independent chunk-boundary check as well as the update counter. The
    # finite metadata plan makes this cheap and excludes consumed prefetch.
    expected = cursor_type(data.manifest_sha256, data.split)
    for _ in range(cursor.next_update):
        update = data.peek_update(expected, tokens[expected.next_update])
        if update is None:
            raise ValueError("Committed cursor is past the corpus")
        expected = update.next_cursor
    if cursor != expected:
        raise ValueError("Committed cursor is not the scheduled chunk boundary")
    if install:
        return data.restore_cursor(record["cursor"])
    if cursor != data.cursor():
        raise ValueError("Reader cursor differs from the committed training cursor")
    return cursor


def validate_reference(path, digest, args, sources):
    if sha256_file(path) != digest:
        raise ValueError("Reference report bytes differ from supplied SHA256")
    result = json.loads(Path(path).read_text())
    if (result.get("schema") != SCHEMA or result.get("phase") != "write"
            or result.get("status") != "passed" or result.get("sources") != sources):
        raise ValueError("Reference must be a completed write under identical source pins")
    if (result.get("arm") != args.arm or result.get("physical_batch_per_rank") != args.batch_size
            or result.get("index_sha256") != args.index_sha256
            or result.get("checkpoint", {}).get("manifest_sha256") != args.expected_manifest_sha256):
        raise ValueError("Reference data/model/checkpoint selection differs")
    if len(result.get("continuation", [])) != 2 or len(result.get("saved_boundaries", [])) != 2:
        raise ValueError("Reference lacks rank-specific recovery evidence")
    return result


def source_hashes():
    result = probe_sources()
    names = (Path(__file__), ROOT / "scripts/olmo_campaign_restart.py",
             ROOT / "scripts/olmo_two_gpu_recovery.py", ROOT / "scripts/olmo_distributed_prepare.py",
             ROOT / "scripts/olmo_f1_common.py", ROOT / "docs/reports/olmo-packed-campaign/protocol.md")
    result.update({str(p.relative_to(ROOT)): sha256_file(p) for p in names})
    return dict(sorted(result.items()))


def main(argv=None):
    args = parse_args(argv)
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Use the project Docker GPU launcher")
    if os.environ.get("WORLD_SIZE") != "2":
        raise RuntimeError("This bounded runner requires exactly two torchrun ranks")
    if any(os.environ.get(k) != "0" for k in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Set both NCCL async-error flags=0 and bound the launcher to 1200 seconds")
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
    torch.cuda.reset_peak_memory_stats()
    tracker = None
    started = time.monotonic()
    report = {"schema": SCHEMA, "phase": args.phase, "arm": args.arm,
        "physical_batch_per_rank": args.batch_size, "length": 1024,
        "document_policy": POLICY, "index_sha256": args.index_sha256,
        "runtime": runtime, "sources": source_hashes(), "status": "running", "checks": [], "phases": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual packed coverage-fixture updates and same-world-size fresh-process recovery. BF16 numerical qualifications remain. Not a production mixture, quality result or changed-hardware recovery.",
        "counter_note": "Legacy counters.documents counts nonempty packed-row presentations, not unique documents"}

    def persist(stage=None):
        if stage is not None:
            report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic()-started
        if rank == 0:
            if tracker is not None:
                report["wandb"] = tracker.record
            write_json(args.output_dir/"report.json", report)

    def publish(stage, passed, details=None):
        flags = gather(bool(passed))
        report["checks"].append({"stage": stage, "passed": all(flags), "ranks": flags, "details": details})
        coordinated("persist gate", lambda: persist(stage))
        if rank == 0:
            print({"stage": stage, "passed": all(flags), "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
        if not all(flags):
            raise AssertionError(f"Packed campaign gate failed: {stage}")

    def phase_observer(phase, event):
        # Called at matched runner boundaries, outside capture/collectives.
        row = {"phase": phase, "event": event, "rank": rank,
               "elapsed_seconds": time.monotonic()-started, "memory": memory()}
        report["phases"].append(row)
        write_json(args.output_dir/f"rank-{rank}-phases.json", report["phases"])
        persist(phase+"/"+event)
        print(row, flush=True)

    error = None
    try:
        def setup():
            nonlocal tracker
            if rank:
                return
            args.output_dir.mkdir(parents=True, exist_ok=False)
            for name in report["sources"]:
                destination = args.output_dir/"source-snapshot"/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/name, destination)
            tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                group="olmo-packed-campaign", name=args.output_dir.name, preserve_state=preserve_local_rng)
            tracker.start({k: report[k] for k in ("phase", "arm", "length", "physical_batch_per_rank", "document_policy", "scope")})
            print({"wandb": tracker.record["run_url"]}, flush=True)
        coordinated("setup", setup)
        coordinated("persist setup", lambda: persist("setup"))
        with disable_autocast_weight_cache(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            run(args, report, rank, device, tracker, persist, publish, phase_observer)
        publish("source_pins_unchanged", report["sources"] == source_hashes())
        report["status"] = "passed"
    except BaseException as exc:
        error = exc
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc),
                                             "traceback": traceback.format_exc()})
        raise
    finally:
        if rank == 0 and args.output_dir.is_dir():
            persist()
            try:
                if tracker is not None:
                    tracker.finish(succeeded=report["status"] == "passed")
            finally:
                persist()
        if error is None:
            dist.destroy_process_group()


def run(args, report, rank, device, tracker, persist, publish, phase_observer):
    from cdrm.pretrained.packed_campaign_data import PackedCampaignData

    data = coordinated("open verified packed index", lambda: PackedCampaignData(args.corpus, args.index))
    try:
        def validate_data_selection():
            if data.manifest_sha256 != args.index_sha256 or data.length != 1024:
                raise ValueError("Actual data/index differs from selected pinned T1024 plan")
        coordinated("validate pinned index selection", validate_data_selection)
        reference = None if args.phase == "write" else coordinated("validate reference", lambda:
            validate_reference(args.reference_report, args.reference_sha256, args, report["sources"]))
        model, recipe, source_checkpoint, _, _ = coordinated("construct pinned model", lambda:
            construct(args, args.arm, device))
        tokens = coordinated("derive finite token schedule", lambda: token_plan(data, recipe.effective_valid_tokens))
        optimizer = build_campaign_adamw(model, recipe, fused=True)
        scheduler = CampaignTokenSchedule(optimizer, tokens, warmup_tokens=recipe.warmup_tokens,
                                          start_fraction=recipe.warmup_start_fraction)
        config = LMTrainingConfig(precision="bf16_mixed", max_grad_norm=recipe.max_grad_norm)
        configuration = {"schema": SCHEMA, "recipe": recipe.to_dict(), "model_config": model.config.to_dict(),
            "data_manifest": data.manifest, "data_manifest_sha256": data.manifest_sha256,
            "world_size": 2, "physical_batch_per_rank": args.batch_size, "training": asdict(config),
            "schedule": scheduler.checkpoint_contract(), "runtime": report["runtime"],
            "ddp": {"static_graph": True, "broadcast_buffers": False, "gradient_as_bucket_view": False,
                    "bucket_cap_mb": 25}, "warmup": 11}
        fingerprint = {"checkpoint_sha256": source_checkpoint["sha256"],
            "checkpoint": source_checkpoint, "recipe_sha256": recipe.sha256,
            "data_manifest_sha256": data.manifest_sha256, "source_hashes": report["sources"]}
        report.update(configuration=configuration, source_checkpoint=source_checkpoint, token_plan=tokens,
            parameters={"resident": sum(p.numel() for p in model.parameters()),
                        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        generators = {"data": torch.Generator().manual_seed(8300+rank),
                      "local": torch.Generator(device=device).manual_seed(8400+rank)}
        seed_local(8500+rank, device)
        counters, cursor = TrainingCounters(), data.cursor()
        if args.phase == "resume":
            publish("execution_configuration_exact", configuration == reference["configuration"])
            resumed = load_distributed_checkpoint(args.checkpoint_dir, model, optimizer,
                scheduler=scheduler, configuration=configuration, source_fingerprint=fingerprint,
                generators=generators, expected_manifest_sha256=args.expected_manifest_sha256, device=device)
            counters = resumed["counters"]
            cursor = coordinated("validate restored real cursor", lambda: restore_training_cursor(
                data, resumed["data_cursor"], counters, tokens, rank=rank, batch_size=args.batch_size, install=True))
            report["checkpoint"] = resumed["manifest"]

        def current_boundary():
            return boundary(model, optimizer, scheduler, counters,
                cursor_record(cursor, rank=rank, batch_size=args.batch_size), device, generators)

        if reference is not None:
            publish("restored_boundary_exact", current_boundary() == reference["saved_boundaries"][rank])
        initial = current_boundary()

        def materialize(update):
            packed = data.rank_batches(update, rank=rank, world_size=2, physical_batch_size=args.batch_size)
            noises = tuple(feedback_noise_for_rows(recipe, keys, logical_update=cursor.next_update,
                sequence_length=1024, width=model.config.model_dim, physical_batch_size=args.batch_size)
                for keys in packed.keys)
            return packed, noises

        first = coordinated("peek preparation update", lambda: data.peek_update(cursor, recipe.effective_valid_tokens))
        packed, noises = coordinated("materialize preparation update", lambda: materialize(first))
        plans = gather(sum_objective_counts([model.counts(b) for b in packed.batches]))
        adapter = CampaignObjective(model, packed.batches[0], mode=recipe.mode(),
            global_counts=sum_objective_counts(plans), world_size=2, feedback_noise=noises[0], config=config)
        runner = CampaignDDPGraphTraining(adapter)
        report["adam_resident_before_ddp"] = bool(optimizer.state)
        coordinated("persist before DDP", lambda: persist("prepare_graphs"))
        runner.prepare(warmup=11, phase_observer=phase_observer)
        runner.capture(warmup=11, release_transient_cache=True, phase_observer=phase_observer)
        publish("preparation_preserves_model_adam_rng_clocks_cursor", current_boundary() == initial)
        report["memory_after_capture_by_rank"] = gather(memory())
        del packed, noises

        def update(label):
            nonlocal cursor
            start_cursor = cursor
            record = cursor_record(cursor, rank=rank, batch_size=args.batch_size)
            coordinated("validate committed cursor", lambda: restore_training_cursor(
                data, record, counters, tokens, rank=rank, batch_size=args.batch_size))
            draws = draw_rng(device, generators)
            rng_before = tree_digests(_local_rng(device, generators))
            dist.barrier()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            logical = coordinated("peek update", lambda: data.peek_update(cursor, recipe.effective_valid_tokens))
            packed, noises = coordinated("materialize update", lambda: materialize(logical))
            loader_seconds = time.perf_counter()-begin
            input_digest = tree_digests({"batches": [vars(b) for b in packed.batches], "noise": noises,
                                        "keys": packed.keys, "start_cursor": asdict(cursor)})
            coordinated("persist before update", lambda: persist(label+"/backward"))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            raw_result = runner.backward(packed.batches, feedback_noises=noises, replay=True)
            torch.cuda.synchronize()
            backward_seconds = time.perf_counter()-begin
            raw = tree_digests({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
            replicas = gather(raw)
            publish(label+"/raw_gradient_replicas_exact", all(v == replicas[0] for v in replicas))
            begin = time.perf_counter()
            metrics = runner.step(raw_result, optimizer, scheduler=scheduler, counters=counters)
            cursor = coordinated("commit completed update cursor", lambda: data.commit(start_cursor, logical))
            torch.cuda.synchronize()
            optimizer_seconds = time.perf_counter()-begin
            current = current_boundary()
            states = gather(current["state"])
            publish(label+"/state_replicas_exact", all(v == states[0] for v in states))
            publish(label+"/finite_state", state_health(model, optimizer)["passed"])
            publish(label+"/rng_and_committed_cursor", current["rng"] == rng_before
                    and cursor.next_update == counters.optimizer_updates
                    and counters.input_tokens == sum(tokens[:cursor.next_update]))
            row = {"input": input_digest, "raw_gradients": raw, "metrics": metrics,
                   "boundary": current, "rng_draws": draws}
            rows = gather(row)
            timings = gather({"rank": rank, "loader_and_jitter_seconds": loader_seconds,
                "backward_seconds": backward_seconds, "optimizer_commit_seconds": optimizer_seconds,
                "training_seconds_excluding_diagnostics": loader_seconds+backward_seconds+optimizer_seconds})
            maximum = max(v["training_seconds_excluding_diagnostics"] for v in timings)
            actual_counts = asdict(logical.counts)
            publish(label+"/actual_data_counts", metrics["input_tokens"] == actual_counts["valid_tokens"]
                    and metrics["documents"] == actual_counts["packed_rows"]
                    and metrics["counts"] == {t: (actual_counts[k] if model.objective_weights()[t] > 0 else 0) for t, k in
                        (("ce", "ce_targets"), ("latent", "latent_pairs"), ("kl", "kl_triples"))})
            telemetry = {"label": label, "rank_timings": timings, "logical_counts": actual_counts,
                "rank_accounting": gather(packed.accounting), "memory_by_rank": gather(memory()),
                "microbatches_per_rank": len(packed.batches),
                "global_valid_input_tokens_per_second": metrics["input_tokens"]/maximum,
                "timing_scope": "Loader/token reads, keyed jitter, preflight/refill, captured backward/NCCL, clip/Adam/scheduler and cursor commit. Sum of timed segments; excludes diagnostic hashing/gates, health scans, W&B and checkpoint I/O."}
            report.setdefault("updates", []).append({"ranks": rows, **telemetry})
            coordinated("persist update", lambda: persist(label))
            if rank == 0:
                tracker.log(scalar_metrics({"metrics": metrics, **telemetry}, "readiness/"+label),
                            step=counters.optimizer_updates)
            return rows

        if args.phase == "write":
            update("first_update")
            saved = current_boundary()
            report["saved_boundaries"] = gather(saved)
            coordinated("checkpoint disk preflight", lambda: checkpoint_disk_preflight(args.checkpoint_dir, model))
            coordinated("persist before save", lambda: persist("saving_checkpoint"))
            with runner.checkpoint_boundary():
                report["checkpoint"] = save_distributed_checkpoint(args.checkpoint_dir, model, optimizer,
                    scheduler=scheduler, counters=counters,
                    data_cursor=cursor_record(cursor, rank=rank, batch_size=args.batch_size),
                    configuration=configuration, source_fingerprint=fingerprint, generators=generators, device=device)
            publish("save_preserves_live_graph_boundary", saved == current_boundary())
            report["continuation"] = update("uninterrupted_next_update")
        else:
            report["continuation"] = update("resumed_next_update")
            comparison = compare_continuation(report["continuation"][rank], reference["continuation"][rank])
            report["continuation_comparisons"] = gather(comparison)
            publish("actual_data_fresh_process_next_update_bitwise_exact", comparison["passed"])
        report["runner"] = gather(runner.metadata)
        del runner, adapter, model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        data.close()


if __name__ == "__main__":
    main()
