#!/usr/bin/env python3
"""Bounded eager native OLMo tiled RT forward/backward profile; no updates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, validate_prepared_manifest
from cdrm.pretrained.olmo_recurrent import RTMode, recurrent_layer_reference
from cdrm.pretrained.olmo_reference import verify_olmo_reference_sources
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM, tiled_recurrent_layer
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_rt_validate import SOURCE_FILES as O1_SOURCE_FILES
from scripts.olmo_validation import autocast, loss_of, require_container_gpu

SOURCE_FILES = O1_SOURCE_FILES + ("cdrm/pretrained/olmo_tiled.py", "scripts/olmo_tiled_profile.py")


def measure(model, *, scope, implementation, batch, length, precision, warmup=3, repeats=5):
    torch.manual_seed(9021 + batch + length)
    ids = torch.randint(0, model.config.tokenizer_vocab_size, (batch, length), device="cuda")
    positions = torch.arange(length, device="cuda")[None].expand(batch, -1)
    valid = torch.ones(batch, length, dtype=torch.bool, device="cuda")
    x = model.token_embeddings(ids).detach().requires_grad_()
    layer = model.layers[0]
    parameters = tuple(model.parameters()) if scope == "full_model" else tuple(layer.parameters())

    def step():
        model.zero_grad(set_to_none=True)
        x.grad = None
        with autocast(precision):
            if scope == "full_model":
                out = model(ids, mode=RTMode((0,), 1))
                loss = loss_of(out.logits, ids)
            else:
                common = dict(alpha=1, past=None, query_positions=positions, key_positions=positions, key_valid=valid)
                if implementation == "tiled":
                    out, _ = tiled_recurrent_layer(layer, x, attention_precision="mixed", **common)
                else:
                    out, _ = recurrent_layer_reference(layer, x, attention_backend="sdpa", **common)
                loss = out.float().square().mean()
        loss.backward()
        return loss.detach()

    for _ in range(warmup):
        loss = step()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    x.grad = None
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, device = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        start.record()
        loss = step()
        end.record()
        end.synchronize()
        wall.append(time.perf_counter() - began)
        device.append(start.elapsed_time(end) / 1000)
    finite = bool(torch.isfinite(loss)) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
    finite &= scope == "full_model" or (x.grad is not None and bool(torch.isfinite(x.grad).all()))
    peak = torch.cuda.max_memory_allocated() / 2**30
    row = {"scope": scope, "implementation": implementation, "precision": precision,
           "attention_precision": "mixed", "ordinary_backend": "sdpa", "batch_size": batch,
           "length": length, "alpha": 1, "selected_layers": [0], "warmup_iterations": warmup,
           "timed_iterations": repeats, "wall_seconds": wall, "cuda_event_seconds": device,
           "median_wall_seconds": statistics.median(wall), "median_cuda_seconds": statistics.median(device),
           "tokens_per_second": batch * length / statistics.median(wall),
           "baseline_allocated_gib": baseline / 2**30, "peak_allocated_gib": peak,
           "incremental_peak_allocated_gib": peak - baseline / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
           "loss": float(loss), "finite_gradients": finite, "passed": finite}
    model.zero_grad(set_to_none=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime = require_container_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260921)
    report = {"schema": "olmo-tiled-rt-profile-v1", "status": "running", "runtime": runtime,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_hashes": {p: sha256_file(ROOT / p) for p in SOURCE_FILES},
              "scope": "eager forward+backward without optimizer updates; RT layer0 only",
              "protocol": "docs/reports/olmo1b-o2/protocol.md", "cases": [],
              "compile": False, "cuda_graphs": False,
              "limits": "Python local VJPs; quadratic backward attention reconstruction; no training/batch capacity claim"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                            group="olmo1b-step60000-tiled-rt", name="olmo-1b-o2-tiled-profile")
    began = time.monotonic()
    try:
        config = OLMoConfig.native_1b()
        manifest = validate_prepared_manifest(args.artifacts)
        report.update(config=config.to_dict(), checkpoint=manifest["checkpoint"], artifacts_manifest=manifest,
                      native_reference_sources=verify_olmo_reference_sources())
        tracker.start({"config": config.to_dict(), "scope": report["scope"], "checkpoint_sha256": manifest["checkpoint"]["sha256"]})
        report["wandb"] = tracker.record
        print({"stage": "load", "wandb": tracker.record["run_url"]}, flush=True)
        model = OLMoTiledRTForCausalLM(config, attention_backend="sdpa", attention_precision="mixed", device="cuda", dtype=torch.float32).eval()
        state = load_native_state_dict(args.artifacts, expected_shapes={k: tuple(v.shape) for k, v in model.state_dict().items()})
        model.load_state_dict(state, strict=True)
        del state
        specifications = [("block", kind, 1, 128, precision)
                          for precision in ("fp32", "bf16_mixed") for kind in ("sequential", "tiled")]
        specifications += [("block", "tiled", batch, length, "bf16_mixed") for batch, length in ((1, 256), (1, 512), (4, 512))]
        specifications += [("full_model", "tiled", 1, length, "bf16_mixed") for length in (128, 512)]
        for scope, implementation, batch, length, precision in specifications:
            row = measure(model, scope=scope, implementation=implementation, batch=batch, length=length, precision=precision)
            report["cases"].append(row)
            write_json(args.output_dir / "report.json", report)
            tracker.log({"profile/tokens_per_second": row["tokens_per_second"],
                         "profile/wall_seconds": row["median_wall_seconds"],
                         "profile/peak_allocated_gib": row["peak_allocated_gib"],
                         "profile/length": length, "profile/batch_size": batch}, step=len(report["cases"]))
            print({k: row[k] for k in ("scope", "implementation", "precision", "batch_size", "length", "tokens_per_second", "peak_allocated_gib", "passed")}, flush=True)
            if not row["passed"]:
                raise AssertionError("Nonfinite or missing gradient in bounded profiling")
            if row["peak_allocated_gib"] > 60:
                raise RuntimeError("Bounded protocol stopped expansion at 60 GiB allocation")
        report.update(status="passed", elapsed_seconds=time.monotonic() - began)
        tracker.summary({"profile/status": "passed", "profile/cases": len(report["cases"])})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir / "report.json", report)
    print({"status": report["status"], "report": str(args.output_dir / "report.json"), "wandb": tracker.record["run_url"]}, flush=True)


if __name__ == "__main__":
    main()
