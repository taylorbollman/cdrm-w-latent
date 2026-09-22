#!/usr/bin/env python3
"""Freeze O4's real-data baseline, comfortable batch and exposure schedule."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.lm_data import load_lm_data
from cdrm.pretrained.lm_evaluation import evaluate_batches
from cdrm.pretrained.lm_schedule import build_schedule
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, validate_prepared_manifest
from cdrm.pretrained.recurrent import RTMode
from scripts.olmo_lm_common import build_model, verify_nextlat_sources
from scripts.olmo_lm_profile import measure
from scripts.olmo_o4_common import PilotTracker, preserve_rng, source_hashes
from scripts.olmo_validation import require_container_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime = require_container_gpu()
    torch.set_num_threads(8)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "olmo-o4-preflight-v1", "status": "running", "runtime": runtime,
              "started_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": source_hashes(),
              "profile": [], "baseline": {}, "precision": "bf16_mixed", "ordinary_backend": "sdpa",
              "rt_attention_precision": "mixed", "compile": False, "cuda_graphs": False}
    tracker = PilotTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                           name="olmo-1b-o4-preflight", group="olmo1b-o4-code-pilot", preserve_state=preserve_rng)
    try:
        native = validate_prepared_manifest(args.artifacts)
        corpus = load_lm_data(args.data)
        report.update(checkpoint=native["checkpoint"], data_manifest_sha256=corpus.manifest_sha256,
                      nextlat_reference=verify_nextlat_sources())
        tracker.start({"scope": "O4 real-data baseline and zero-LR complete steps", "checkpoint": native["checkpoint"],
                       "data_manifest_sha256": corpus.manifest_sha256})
        report["wandb"] = tracker.record
        state = load_native_state_dict(args.artifacts)
        model = build_model(state, enabled=True, chunk_size=128, backend="sdpa", attention_precision="mixed")
        del state
        full_rows = np.flatnonzero(corpus.train_lengths == 512)
        if len(full_rows) < 32:
            raise ValueError("Preflight requires 32 full-length real training windows")
        chosen = None
        for batch_size in (16, 32):
            try:
                row = measure(model, corpus.batch("train", full_rows[:batch_size], device="cuda"),
                              RTMode((0,), 1.0), warmup=2, repeats=3)
                report["profile"].append(row)
                if not row["passed"]:
                    raise AssertionError("Real-data zero-LR profile failed integrity")
                tracker.log({"profile/batch_size": batch_size, "profile/seconds": row["median_wall_seconds"],
                             "profile/peak_allocated_gib": row["peak_allocated_gib"],
                             "profile/input_tokens_per_second": row["valid_input_tokens_per_second"]})
                if row["peak_allocated_gib"] < 55:
                    chosen = batch_size
            except torch.cuda.OutOfMemoryError:
                report["profile"].append({"batch_size": batch_size, "status": "out_of_memory"})
            finally:
                model.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
            write_json(args.output_dir / "report.json", report)
        if chosen is None:
            raise RuntimeError("Neither candidate has the predeclared 55-GiB headroom")
        # A fixed prefix is selected independently of scores. Test splits stay unopened.
        for name, mode in (("ordinary", RTMode((), 0.0)), ("rt_alpha0", RTMode((0,), 0.0)),
                           ("rt_alpha1", RTMode((0,), 1.0))):
            report["baseline"][name] = {}
            for split in ("dev", "retention_dev"):
                count = min(512, corpus.split_sizes[split])
                batches = (corpus.batch(split, range(i, min(i + 8, count)), device="cuda")
                           for i in range(0, count, 8))
                result = evaluate_batches(model, batches, mode=mode, precision="bf16_mixed", include_document_records=True)
                report["baseline"][name][split] = result
                tracker.log({f"baseline/{name}/{split}/nll": result["mean_nll"],
                             f"baseline/{name}/{split}/accuracy": result["next_token_accuracy"]})
                print({"mode": name, "split": split, "nll": result["mean_nll"],
                       "accuracy": result["next_token_accuracy"]}, flush=True)
                write_json(args.output_dir / "report.json", report)
        schedule = build_schedule(corpus.train_lengths, batch_size=chosen, warmup_updates=100,
                                  ramp_min_tokens=10_000_000, ramp_min_updates=200)
        configuration = {"schema": "olmo-o4-pilot-config-v1", "data_manifest_sha256": corpus.manifest_sha256,
                         "source_hashes": report["source_hashes"],
                         "checkpoint_sha256": native["checkpoint"]["sha256"], "seed": 20260922,
                         "predictor_seed": model.config.seed, "nextlat_config": model.config.to_dict(),
                         "model_config": model.backbone.config.to_dict(), "precision": "bf16_mixed",
                         "attention_backend": "sdpa", "attention_precision": "mixed", "rt_layers": [0],
                         "lr": 1e-5, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1,
                         "max_grad_norm": 1.0, "schedule": schedule, "eval_every_tokens": 1_000_000,
                         "eval_rows": 128, "final_eval_rows": 512, "eval_batch_size": 8,
                         "health_gate_update": 50, "catastrophic_nll_increase": 1.5,
                         "catastrophic_consecutive_evals": 2, "checkpoint_interval_seconds": 1800,
                         "compile": False, "cuda_graphs": False, "distributed": False,
                         "masking": "all valid same-document targets; one window per row; no cross-window cache"}
        write_json(args.output_dir / "configuration.json", configuration)
        report.update(status="passed", selected_batch_size=chosen, schedule=schedule)
        tracker.summary({"preflight/status": "passed", "preflight/batch_size": chosen,
                         "preflight/total_updates": schedule["total_updates"]})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        tracker.finish(succeeded=report["status"] == "passed")
        report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "report.json", report)
    print({"status": report["status"], "batch_size": chosen, "wandb": tracker.record["run_url"]}, flush=True)


if __name__ == "__main__":
    main()
