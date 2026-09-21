#!/usr/bin/env python3
"""O3 complete optimizer-step profiling with zero LR and unchanged weights."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, load_native_tokenizer, validate_prepared_manifest
from cdrm.pretrained.olmo_recurrent import RTMode
from cdrm.pretrained.olmo_reference import verify_olmo_reference_sources
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters, build_adamw, optimizer_step
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_rt_validate import SOURCE_FILES as O1_SOURCES
from scripts.olmo_validation import require_container_gpu
from scripts.olmo_lm_common import build_model, fixture, fixture_record, state_digests, verify_nextlat_sources

SOURCE_FILES = O1_SOURCES + (
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/nextlat.py", "cdrm/pretrained/lm_training.py",
    "scripts/olmo_lm_common.py", "scripts/olmo_lm_profile.py",
)


def measure(model, batch, mode, *, warmup=3, repeats=3):
    before = state_digests(model)
    optimizer = build_adamw(model, lr=0.0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    counters = TrainingCounters()
    config = LMTrainingConfig(precision="bf16_mixed", max_grad_norm=1.0)
    def step():
        return optimizer_step(model, optimizer, [batch], config=config,
                              counters=counters, backbone_kwargs={"mode": mode})
    for _ in range(warmup):
        last = step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, device = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        start.record()
        last = step()
        end.record()
        end.synchronize()
        wall.append(time.perf_counter() - began)
        device.append(start.elapsed_time(end) / 1000)
    # Capture operational step peaks before external state-integrity checks.
    peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    unchanged = before == state_digests(model)
    moments_finite = all(bool(torch.isfinite(tensor).all()) for state in optimizer.state.values()
                         for tensor in state.values() if isinstance(tensor, torch.Tensor))
    valid_tokens = int(batch.valid_mask.sum())
    seconds = statistics.median(wall)
    row = {"nextlat": model.enabled, "mode": asdict(mode), "precision": config.precision,
           "attention_precision": model.backbone.attention_precision, "ordinary_backend": model.backbone.attention_backend,
           "batch_size": batch.input_ids.shape[0], "length": batch.input_ids.shape[1],
           "fixture": fixture_record(batch), "warmup_iterations": warmup, "timed_iterations": repeats,
           "wall_seconds": wall, "cuda_event_seconds": device, "median_wall_seconds": seconds,
           "median_cuda_seconds": statistics.median(device), "valid_input_tokens_per_second": valid_tokens / seconds,
           "supervised_tokens_per_second": last["counts"]["ce"] / seconds,
           "baseline_allocated_gib": baseline / 2**30, "peak_allocated_gib": peak / 2**30,
           "incremental_peak_allocated_gib": (peak - baseline) / 2**30, "peak_reserved_gib": reserved / 2**30,
           "peak_scope": "complete step including norm clipping and AdamW, excluding subsequent digest/finite-moment checks",
           "last_step": last, "weights_unchanged": unchanged, "optimizer_moments_finite": moments_finite,
           "passed": unchanged and moments_finite}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime = require_container_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"schema": "olmo-lm-profile-v1", "status": "running", "runtime": runtime,
              "started_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
              "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
              "scope": "O3 zero-LR full optimizer steps with NextLat independently off/on; one RT layer; no O4 learning comparison",
              "compile": False, "cuda_graphs": False, "distributed": False, "learning_rate": 0,
              "microbatch_accumulation": 1, "vocabulary_projection_positions_per_chunk": 128}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                            group="olmo1b-step60000-nextlat-platform", name="olmo-1b-o3-profile")
    began = time.monotonic()
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report.update(checkpoint=manifest["checkpoint"], artifacts_manifest=manifest,
                      native_reference_sources=verify_olmo_reference_sources(), nextlat_reference_sources=verify_nextlat_sources())
        tracker.start({"scope": report["scope"], "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                       "protocol": "docs/reports/olmo1b-o3/protocol.md", "learning_rate": 0})
        report["wandb"] = tracker.record
        print({"stage": "load", "wandb": tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        for enabled in (False, True):
            model = build_model(state, enabled=enabled, chunk_size=128, backend="sdpa")
            report["model_config"] = model.backbone.config.to_dict()
            report["nextlat_config"] = model.config.to_dict()
            report.setdefault("parameter_counts", {})["nextlat" if enabled else "baseline"] = sum(p.numel() for p in model.parameters())
            specifications = [(RTMode((), 0), 1, 128), (RTMode((0,), 1), 1, 128)]
            if enabled:
                specifications += [(RTMode((0,), 1), batch, 512) for batch in (1, 4, 8)]
            for mode, batch_size, length in specifications:
                row = measure(model, fixture(tokenizer, batch_size=batch_size, length=length), mode)
                report["cases"].append(row)
                write_json(args.output_dir / "report.json", report)
                tracker.log({"profile/valid_tokens_per_second": row["valid_input_tokens_per_second"],
                             "profile/step_seconds": row["median_wall_seconds"], "profile/peak_allocated_gib": row["peak_allocated_gib"],
                             "profile/batch_size": batch_size, "profile/length": length,
                             "profile/nextlat": enabled}, step=len(report["cases"]))
                print({key: row[key] for key in ("nextlat", "mode", "batch_size", "length", "median_wall_seconds", "valid_input_tokens_per_second", "peak_allocated_gib", "passed")}, flush=True)
                if not row["passed"]:
                    raise AssertionError("Zero-LR weight integrity or finite optimizer-state check failed")
                if row["peak_allocated_gib"] > 60:
                    raise RuntimeError("Stop profiling expansion at 60 GiB allocation")
                gc.collect()
                torch.cuda.empty_cache()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="passed", elapsed_seconds=time.monotonic() - began,
                      executed_zero_lr_steps=sum(c["warmup_iterations"] + c["timed_iterations"] for c in report["cases"]))
        tracker.summary({"profile/status": "passed", "profile/zero_lr_steps": report["executed_zero_lr_steps"]})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir / "report.json", report)
    print({"status": report["status"], "report": str(args.output_dir / "report.json")}, flush=True)


if __name__ == "__main__":
    main()
