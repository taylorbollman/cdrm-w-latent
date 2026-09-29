#!/usr/bin/env python3
"""One bounded two-H100 NFR K4/T1024 capacity/complete-update timing candidate.

Requires prior campaign correctness and fresh-process restart qualification.
Launch with torchrun --nproc_per_node=2 inside the GPU container, both NCCL
async-error flags set to 0, and an external 900-second launcher timeout. This
script never selects another batch size automatically or saves a full model.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gc
import math
import os
from pathlib import Path
import shutil
import statistics
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
from cdrm.pretrained.campaign_recipe import CampaignTokenSchedule, build_campaign_adamw, feedback_noise_for_rows
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, optimizer_state_bytes
from cdrm.pretrained.nextlat import NextLatBatch
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct, gather, source_hashes as probe_sources
from scripts.olmo_campaign_graph_probe import gradients_are_zero, pointer_snapshot, rng_snapshot, rng_unchanged
from scripts.olmo_campaign_probe import memory
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--microbatches", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--measured-updates", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if args.warmup_updates != 3 or args.measured_updates != 5:
        parser.error("Frozen bounded protocol uses exactly 3 warmup and 5 measured Adam updates")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT.resolve()):
        parser.error("Evidence must remain under the persistent project checkout")
    args.scale, args.length = "pretrained", 1024
    return args


def logical_counts(batch_size, microbatches, *, length=1024, world_size=2):
    if any(type(v) is not int or v <= 0 for v in (batch_size, microbatches, length, world_size)) or length < 3:
        raise ValueError("Positive integer dimensions and at least three positions required")
    rows = batch_size * microbatches * world_size
    return {"counts": {"ce": rows*(length-1), "latent": rows*(length-1), "kl": rows*(length-2)},
            "documents": rows, "microbatches": microbatches*world_size, "input_tokens": rows*length}


def capacity_fixture(recipe, width, *, rank, update, batch_size, microbatches, token_ids, eos_id, length=1024):
    """Full-valid one-document rows; repeat operational prose, never corpus data."""
    if rank not in (0, 1) or type(update) is not int or update < 0:
        raise ValueError("Require rank0/1 and nonnegative update")
    logical_counts(batch_size, microbatches, length=length)
    if not token_ids:
        raise ValueError("Fixture source must contain tokens")
    source = torch.tensor(token_ids, dtype=torch.long)
    batches, noises = [], []
    for microbatch in range(microbatches):
        row_ids = torch.arange(batch_size, dtype=torch.long)
        offsets = (update*31 + rank*17 + microbatch*7 + row_ids*3)[:, None]
        positions = torch.arange(length, dtype=torch.long)[None]
        ids = source[(positions+offsets) % source.numel()].clone()
        ids[:, -1] = eos_id
        valid = torch.ones_like(ids, dtype=torch.bool)
        docs = (update*100000 + rank*10000 + microbatch*batch_size + row_ids)[:, None].expand_as(ids).clone()
        batches.append(NextLatBatch(ids, valid, docs, valid.clone(), valid.clone(), valid.clone()))
        keys = [f"capacity-u{update}-r{rank}-m{microbatch}-row{row}" for row in range(batch_size)]
        noises.append(feedback_noise_for_rows(recipe, keys, logical_update=update,
            sequence_length=length, width=width, physical_batch_size=batch_size))
    return tuple(batches), tuple(noises)


def timing_card(rows, input_tokens):
    if not rows or type(input_tokens) is not int or input_tokens <= 0:
        raise ValueError("Timing card requires measured updates and positive valid-input count")
    seconds = [r["max_rank_seconds"] for r in rows]
    if any(not math.isfinite(v) or v <= 0 for v in seconds):
        raise ValueError("Measured update durations must be positive finite")
    total = sum(seconds)
    return {"measured_updates": len(rows), "valid_input_tokens_per_update": input_tokens,
            "total_valid_input_tokens": len(rows)*input_tokens, "total_seconds": total,
            "valid_input_tokens_per_second": len(rows)*input_tokens/total,
            "median_update_seconds": statistics.median(seconds),
            "minimum_update_seconds": min(seconds), "maximum_update_seconds": max(seconds),
            "timing": "Maximum wall time across ranks per complete optimizer update; includes local CPU preflight/refill, graph replay, NCCL, clipping, fused Adam and scheduler; excludes fixture generation, evidence/W&B logging and post-update health scans",
            "denominator": "Global valid input tokens once, not K4 pass tokens or target/padding tokens"}


def adam_residency(model, optimizer, *, expected_steps):
    """Verify real initialized moments, without manufacturing optimizer state."""
    moment_bytes = 0
    rows = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        state = optimizer.state.get(parameter, {})
        step = state.get("step")
        value = float(step) if isinstance(step, torch.Tensor) and step.numel() == 1 else None
        valid = value == expected_steps
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state.get(key)
            valid &= (isinstance(tensor, torch.Tensor) and tensor.shape == parameter.shape
                      and tensor.dtype == torch.float32 and tensor.device == parameter.device)
            if isinstance(tensor, torch.Tensor):
                moment_bytes += tensor.numel()*tensor.element_size()
        rows[name] = {"step": value, "ready": bool(valid)}
    return {"passed": bool(rows) and all(r["ready"] for r in rows.values()),
            "expected_steps": expected_steps, "parameter_states": rows,
            "actual_state_bytes_by_device": optimizer_state_bytes(optimizer),
            "actual_moment_bytes": moment_bytes, "actual_moment_gib": moment_bytes / 2**30}


def clock_snapshot(optimizer, scheduler, counters):
    """Small independent clocks; capture/replay priming must not advance them."""
    return {"counters": asdict(counters), "scheduler": copy.deepcopy(scheduler.state_dict()),
            "adam_steps": [float(state["step"]) for state in optimizer.state.values()],
            "learning_rates": [group["lr"] for group in optimizer.param_groups]}


def source_hashes():
    result = probe_sources()
    result[str(Path(__file__).relative_to(ROOT))] = sha256_file(Path(__file__))
    protocol = ROOT / "docs/reports/olmo-campaign-two-gpu/protocol.md"
    if protocol.is_file():
        result[str(protocol.relative_to(ROOT))] = sha256_file(protocol)
    return result


def main(argv=None):
    args = parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Capacity probe requires exactly two torchrun ranks")
    if any(os.environ.get(k) != "0" for k in ("NCCL_ASYNC_ERROR_HANDLING", "TORCH_NCCL_ASYNC_ERROR_HANDLING")):
        raise RuntimeError("Set both NCCL async-error flags to 0 and use external 900-second timeout")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=torch.device("cuda", rank))
    torch.cuda.reset_peak_memory_stats()
    started, tracker = time.monotonic(), None
    counts = logical_counts(args.batch_size, args.microbatches)
    report = {"schema": "olmo-campaign-two-gpu-capacity-v1", "status": "running", "rank": rank,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime, "sources": source_hashes(),
        "arm": "NFR", "length": 1024, "physical_batch_per_rank": args.batch_size,
        "microbatches_per_rank": args.microbatches, "logical_update": counts,
        "capture_after_warmup_adam": True,
        "setup_order": ["DDP construction and 20 backward warmups without Adam state",
            "3 eager complete Adam updates using the same DDP wrapper",
            "capture local and synchronized graphs with all initialized Adam moments resident",
            "one graph backward and discard, without any optimizer or data-clock advance",
            "5 measured graph complete updates"],
        "execution": {"precision": "BF16 mixed with FP32 master parameters, gradients and Adam state; TF32 off",
            "fbt_passes": 4, "rt_layers": [0, 15], "rt_on_every_pass": True,
            "attention": "ordinary forced Flash SDPA; native Triton RT forward/backward recomputation",
            "nextlat": "SmoothL1 and KL, coefficients1; full CE and all same-document valid auxiliary positions",
            "checkpointing": "ordinary activation checkpointing", "rope": "native reused",
            "pointwise": "eager", "compile": False, "optimizer": "replicated fused AdamW",
            "graphs": "separate local no_sync and synchronized final-backward CUDA graphs",
            "ddp": "static_graph=True, broadcast_buffers=False, gradient_as_bucket_view=False, bucket25MiB"},
        "scope": "Directional capacity/throughput of this complete campaign path on two H100s. Graph capture includes real resident Adam state, but cold DDP construction with resident Adam at T1024 remains unqualified. Fixed operational prose is full-valid single-document rows, not production packing/data loader or quality/numerical equivalence qualification.",
        "checkpoints": "Disposable eight-update diagnostic; no full checkpoint. Reconstruct from retained pinned source after interruption.",
        "rows": [], "phases": []}
    try:
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            for relative in report["sources"]:
                dest = args.output_dir / "source-snapshot" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/relative, dest)
            tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                group="olmo-campaign-two-gpu-capacity", name=args.output_dir.name, preserve_state=preserve_local_rng)
            tracker.start({k: report[k] for k in ("arm", "length", "physical_batch_per_rank", "microbatches_per_rank", "execution", "scope")})
            report["wandb"] = tracker.record
            print({"wandb": tracker.record["run_url"]}, flush=True)
        dist.barrier()

        def save():
            report["elapsed_seconds"] = time.monotonic()-started
            write_json(args.output_dir/f"rank-{rank}.json", report)
            if rank == 0:
                write_json(args.output_dir/"report.json", report)

        def phase_observer(phase, event):
            row = {"phase": phase, "event": event, "elapsed_seconds": time.monotonic()-started}
            try:
                row["memory"] = memory()
            except Exception as error:
                row["memory_error_type"] = type(error).__name__
            report["phases"].append(row)
            save()
            print({"rank": rank, **row}, flush=True)

        def publish(stage, row):
            rows = gather({"rank": rank, **row})
            passed = all(r.get("passed", False) for r in rows)
            report["rows"].append({"stage": stage, "passed": passed, "ranks": rows})
            save()
            if rank == 0:
                tracker.log(scalar_metrics({"ranks": {str(r["rank"]): r for r in rows}}, "benchmark/"+stage), step=len(report["rows"]))
                print({"stage": stage, "passed": passed, "elapsed_seconds": time.monotonic()-started}, flush=True)
            if not passed:
                raise AssertionError(f"Capacity probe failed at {stage}")

        model, recipe, checkpoint, ids, eos = construct(args, "NFR", torch.device("cuda", rank))
        report.update(checkpoint=checkpoint, recipe=recipe.to_dict(),
            parameters={"resident": sum(p.numel() for p in model.parameters()),
                        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
            model_config=model.backbone.config.to_dict())
        phase_observer("model_loaded", "end")
        fixture_kwargs = dict(rank=rank, batch_size=args.batch_size, microbatches=args.microbatches,
                              token_ids=ids, eos_id=eos)
        batches, noises = capacity_fixture(recipe, model.config.model_dim, update=0, **fixture_kwargs)
        # Check the independent full-valid target formula against model masks.
        local = {t: sum(model.counts(b)[t] for b in batches) for t in counts["counts"]}
        if any(local[t]*2 != counts["counts"][t] for t in local):
            raise AssertionError("Full-valid fixture target formula differs from model")
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            adapter = CampaignObjective(model, batches[0], mode=recipe.mode(),
                global_counts=counts["counts"], world_size=2, feedback_noise=noises[0],
                config=LMTrainingConfig(precision="bf16_mixed", max_grad_norm=recipe.max_grad_norm))
            runner = CampaignDDPGraphTraining(adapter)
            before = rng_snapshot()
            runner.prepare(warmup=11, phase_observer=phase_observer)
            pointers = pointer_snapshot(runner)
            publish("prepare-ddp-before-adam", {"metadata": runner.metadata, "memory": memory(),
                "optimizer_updates": 0,
                "gradients_zero": gradients_are_zero(model), "rng_unchanged": rng_unchanged(before),
                "passed": gradients_are_zero(model) and rng_unchanged(before)})
            optimizer = build_campaign_adamw(model, recipe, fused=True)
            total_updates = args.warmup_updates + args.measured_updates
            schedule = CampaignTokenSchedule(optimizer, [counts["input_tokens"]]*total_updates,
                warmup_tokens=recipe.warmup_tokens, start_fraction=recipe.warmup_start_fraction)
            counters, timed_rows = TrainingCounters(), []

            def execute_update(update, *, replay):
                nonlocal batches, noises
                # New keyed noise/data, created outside the complete-update timer.
                del batches, noises
                batches, noises = capacity_fixture(recipe, model.config.model_dim, update=update, **fixture_kwargs)
                before = rng_snapshot()
                dist.barrier()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                metrics = runner.optimizer_step(optimizer, batches, feedback_noises=noises, replay=replay,
                                                scheduler=schedule, counters=counters)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-begin
                max_seconds = torch.tensor(seconds, device="cuda", dtype=torch.float64)
                dist.all_reduce(max_seconds, op=dist.ReduceOp.MAX)
                row = {"measured": replay, "replay": replay, "update": update+1,
                       "local_seconds": seconds, "max_rank_seconds": float(max_seconds),
                       "input_tokens_per_second": counts["input_tokens"]/float(max_seconds),
                       "metrics": metrics, "memory": memory(), "rng_unchanged": rng_unchanged(before),
                       "gradients_zero": gradients_are_zero(model),
                       "pointers_stable": pointers == pointer_snapshot(runner),
                       "parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters())}
                row["passed"] = (row["rng_unchanged"] and row["gradients_zero"] and row["pointers_stable"]
                    and row["parameters_finite"] and counters.optimizer_updates == update+1
                    and counters.input_tokens == (update+1)*counts["input_tokens"]
                    and all(metrics[key] == value for key, value in counts.items()))
                if row["measured"]:
                    timed_rows.append(row)
                publish(("measured-graph" if replay else "warmup-eager")+f"-update-{update+1}", row)

            for update in range(args.warmup_updates):
                execute_update(update, replay=False)
            residency = adam_residency(model, optimizer, expected_steps=args.warmup_updates)
            publish("adam-resident-before-capture", {"adam": residency, "memory": memory(),
                "counters": asdict(counters), "pointers_stable": pointers == pointer_snapshot(runner),
                "passed": residency["passed"] and pointers == pointer_snapshot(runner)})
            boundary = clock_snapshot(optimizer, schedule, counters)
            before = rng_snapshot()
            runner.capture(warmup=11, release_transient_cache=True, phase_observer=phase_observer)
            capture_memory = memory()
            unchanged = boundary == clock_snapshot(optimizer, schedule, counters)
            publish("capture-after-warmup-adam", {"metadata": runner.metadata, "memory": capture_memory,
                "capture_after_warmup_adam": True, "actual_adam_moment_gib": residency["actual_moment_gib"],
                "clocks": boundary, "clocks_unchanged": unchanged,
                "pointers_stable": pointers == pointer_snapshot(runner),
                "gradients_zero": gradients_are_zero(model), "rng_unchanged": rng_unchanged(before),
                "passed": unchanged and gradients_are_zero(model) and rng_unchanged(before)
                          and pointers == pointer_snapshot(runner)})
            del batches, noises
            batches, noises = capacity_fixture(recipe, model.config.model_dim,
                update=args.warmup_updates, **fixture_kwargs)
            before = rng_snapshot()
            prime = runner.backward(batches, feedback_noises=noises, replay=True)
            runner.discard_backward()
            unchanged = boundary == clock_snapshot(optimizer, schedule, counters)
            publish("graph-prime-backward-discard", {"metrics": prime, "clocks": boundary,
                "clocks_unchanged": unchanged, "optimizer_updates_added": 0,
                "pointers_stable": pointers == pointer_snapshot(runner),
                "rng_unchanged": rng_unchanged(before), "gradients_zero": gradients_are_zero(model),
                "passed": unchanged and rng_unchanged(before) and gradients_are_zero(model)
                          and pointers == pointer_snapshot(runner)})
            for update in range(args.warmup_updates, total_updates):
                execute_update(update, replay=True)
            report["timing"] = timing_card(timed_rows, counts["input_tokens"])
            report["final_counters"] = asdict(counters)
            report["memory_by_rank"] = gather({"rank": rank, "capture": capture_memory, "final": memory()})
            report["graph"] = runner.metadata
            if rank == 0:
                tracker.summary(scalar_metrics(report["timing"], "benchmark/timing"))
        if report["sources"] != source_hashes():
            raise AssertionError("Runtime sources changed during capacity probe")
        report["status"] = "passed"
        save()
        del runner, adapter, optimizer, model
        gc.collect()
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        if args.output_dir.exists():
            write_json(args.output_dir/f"rank-{rank}.json", report)
        if tracker is not None:
            try:
                tracker.finish(succeeded=report["status"] == "passed")
            finally:
                report["wandb"] = tracker.record
                write_json(args.output_dir/"report.json", report)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
