#!/usr/bin/env python3
"""O3 actual-checkpoint objective and disposable optimizer recovery validation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, load_native_tokenizer, validate_prepared_manifest
from cdrm.pretrained.olmo_recurrent import RTMode
from cdrm.pretrained.olmo_reference import verify_olmo_reference_sources
from cdrm.pretrained.lm_training import (
    LMTrainingConfig, TrainingCounters, build_adamw, build_warmup_scheduler,
    load_training_checkpoint, optimizer_step, save_training_checkpoint,
)
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_rt_validate import SOURCE_FILES as O1_SOURCES
from scripts.olmo_validation import autocast, comparison, descriptive, require_container_gpu, semantic_gradients
from scripts.olmo_lm_common import (
    build_model, dense_objective, fixture, fixture_record, state_digests,
    tree_digests, verify_nextlat_sources,
)

SOURCE_FILES = O1_SOURCES + (
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/nextlat.py", "cdrm/pretrained/lm_training.py",
    "scripts/olmo_lm_common.py", "scripts/olmo_lm_validate.py",
)


def objective_pair(model, reference, batch, mode):
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    actual = model.loss_sums(batch, backbone_kwargs={"mode": mode})
    total, expected = dense_objective(reference, batch, mode)
    actual.total.backward()
    total.backward()
    row = {"kind": "objective_parity", "nextlat": model.enabled, "mode": asdict(mode),
           "batch_size": batch.input_ids.shape[0], "length": batch.input_ids.shape[1],
           "counts": actual.counts, "reference_counts": expected["counts"], "weights": actual.weights,
           "loss_means": {name: float(t.detach()) for name, t in actual.means.items()},
           "loss_comparisons": {name: comparison(actual.means[name], expected["means"][name], atol=1e-5, rtol=1e-5)
                                for name in actual.means},
           "total": comparison(actual.total, total, atol=1e-5, rtol=1e-5),
           "gradients": semantic_gradients(model, reference)}
    row["passed"] = actual.counts == expected["counts"] and row["total"]["passed"] and row["gradients"]["passed"] and all(
        item["passed"] for item in row["loss_comparisons"].values())
    saved = {"gradients": {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()},
             "loss_means": row["loss_means"], "total": float(actual.total.detach())}
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    return row, saved


def mixed_observation(model, batch, mode, fp32, policy):
    model.backbone.attention_precision = policy
    model.zero_grad(set_to_none=True)
    with autocast("bf16_mixed"):
        result = model.loss_sums(batch, backbone_kwargs={"mode": mode})
    result.total.backward()
    rows, missing = {}, []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        else:
            rows[name] = descriptive(comparison(parameter.grad.detach().cpu(), fp32["gradients"][name], atol=0, rtol=0))
    finite = not missing and all(row["finite"] for row in rows.values()) and all(bool(torch.isfinite(t)) for t in result.means.values())
    row = {"kind": "mixed_precision_observation", "attention_precision": policy,
           "ordinary_backend": model.backbone.attention_backend, "mode": asdict(mode),
           "counts": result.counts, "loss_means": {k: float(v.detach()) for k, v in result.means.items()},
           "fp32_loss_means": fp32["loss_means"], "gradients": {"tensors": rows, "missing": missing,
               "relative_l2": math.sqrt(sum(x["difference_l2"] ** 2 for x in rows.values()) / max(1e-60, sum(x["reference_l2"] ** 2 for x in rows.values())))},
           "passed": finite, "scope": "finite objective/complete gradients; cross-precision errors descriptive, not training clearance"}
    model.zero_grad(set_to_none=True)
    return row


def recovery_check(state, batch, directory, fingerprint):
    """Two updates, save, compare third update against a freshly built resume."""
    mode = RTMode((0,), 1)
    training = LMTrainingConfig(precision="fp32", max_grad_norm=1)
    model = build_model(state)
    optimizer = build_adamw(model, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scheduler = build_warmup_scheduler(optimizer, warmup_updates=2)
    counters = TrainingCounters()
    config = {"nextlat": model.config.to_dict(), "enabled": True, "mode": asdict(mode),
              "training": asdict(training), "ordinary_backend": "math", "attention_precision": "mixed"}
    history = []
    for _ in range(2):
        history.append(optimizer_step(model, optimizer, [batch], config=training, scheduler=scheduler,
                                      counters=counters, backbone_kwargs={"mode": mode}))
    path = directory / "disposable-update2.pt"
    began = time.monotonic()
    checkpoint = save_training_checkpoint(path, model, optimizer, scheduler=scheduler, counters=counters,
                                          data_cursor={"fixture_next_index": 2}, configuration=config,
                                          source_fingerprint=fingerprint)
    checkpoint["save_and_hash_seconds"] = time.monotonic() - began
    expected_step = optimizer_step(model, optimizer, [batch], config=training, scheduler=scheduler,
                                   counters=counters, backbone_kwargs={"mode": mode})
    expected = {"model": state_digests(model), "optimizer": tree_digests(optimizer.state_dict()),
                "scheduler": tree_digests(scheduler.state_dict()), "counters": asdict(counters),
                "rng_cpu_draw": torch.rand(4).tolist(), "rng_cuda_draw": torch.rand(4, device="cuda").tolist()}
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    model = build_model(state)
    optimizer = build_adamw(model, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scheduler = build_warmup_scheduler(optimizer, warmup_updates=2)
    began = time.monotonic()
    restored = load_training_checkpoint(path, model, optimizer, scheduler=scheduler, configuration=config,
                                        source_fingerprint=fingerprint, expected_sha256=checkpoint["sha256"])
    checkpoint["verify_and_load_seconds"] = time.monotonic() - began
    if restored["data_cursor"] != {"fixture_next_index": 2} or restored["counters"].optimizer_updates != 2:
        raise AssertionError("Recovery cursor/counters mismatch")
    actual_step = optimizer_step(model, optimizer, [batch], config=training, scheduler=scheduler,
                                 counters=restored["counters"], backbone_kwargs={"mode": mode})
    actual = {"model": state_digests(model), "optimizer": tree_digests(optimizer.state_dict()),
              "scheduler": tree_digests(scheduler.state_dict()), "counters": asdict(restored["counters"]),
              "rng_cpu_draw": torch.rand(4).tolist(), "rng_cuda_draw": torch.rand(4, device="cuda").tolist()}
    checks = {name: actual[name] == expected[name] for name in expected}
    checks["third_update_metrics"] = actual_step == expected_step
    row = {"kind": "native_optimizer_recovery", "passed": all(checks.values()), "checks": checks,
           "configuration": config, "checkpoint": checkpoint, "updates_before_save": history,
           "uninterrupted_third_update": expected_step, "resumed_third_update": actual_step,
           "expected_after_update3": expected, "actual_after_update3": actual,
           "executed_optimizer_updates": 4, "scope": "disposable correctness state, not an adaptation candidate"}
    if row["passed"]:
        path.unlink()
        row["checkpoint"]["disposition"] = "removed after exact recovery; hash and evidence retained"
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
    torch.cuda.reset_peak_memory_stats()
    report = {"schema": "olmo-lm-validation-v1", "status": "running", "runtime": runtime,
              "started_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
              "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
              "scope": "O3 loss/platform checks; four disposable AdamW updates including replay; no O4 learning comparison"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                            group="olmo1b-step60000-nextlat-platform", name="olmo-1b-o3-validation")
    began = time.monotonic()
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report.update(checkpoint=manifest["checkpoint"], artifacts_manifest=manifest,
                      native_reference_sources=verify_olmo_reference_sources(), nextlat_reference_sources=verify_nextlat_sources())
        tokenizer = load_native_tokenizer(args.artifacts)
        batch = fixture(tokenizer)
        padded = fixture(tokenizer, batch_size=2, padded=True)
        report["fixtures"] = {"short": fixture_record(batch), "padded": fixture_record(padded)}
        tracker.start({"scope": report["scope"], "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                       "protocol": "docs/reports/olmo1b-o3/protocol.md"})
        report["wandb"] = tracker.record
        print({"stage": "load", "wandb": tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)

        def record(row):
            report["cases"].append(row)
            write_json(args.output_dir / "report.json", report)
            metrics = {"validation/passed": row["passed"]}
            if "loss_means" in row:
                metrics.update({f"validation/{k}": v for k, v in row["loss_means"].items()})
            if "gradients" in row:
                metrics["validation/gradient_relative_l2"] = row["gradients"]["relative_l2"]
            tracker.log(metrics, step=len(report["cases"]))
            print({"kind": row["kind"], "mode": row.get("mode"), "nextlat": row.get("nextlat"),
                   "passed": row["passed"], "gradient_l2": row.get("gradients", {}).get("relative_l2")}, flush=True)
            if not row["passed"]:
                raise AssertionError("O3 validation failed; inspect retained report")

        for enabled in (False, True):
            model = build_model(state, enabled=enabled)
            reference = copy.deepcopy(model)
            report["model_config"] = model.backbone.config.to_dict()
            report["nextlat_config"] = model.config.to_dict()
            report.setdefault("parameter_counts", {})["nextlat" if enabled else "baseline"] = sum(p.numel() for p in model.parameters())
            assert model.backbone.readout_weight is model.backbone.token_embeddings.weight
            for mode in (RTMode((), 0), RTMode((0,), 1)):
                row, saved = objective_pair(model, reference, batch, mode)
                record(row)
            if enabled:
                for policy in ("mixed", "fp32"):
                    record(mixed_observation(model, batch, RTMode((0,), 1), saved, policy))
                model.backbone.attention_precision = "mixed"
                row, _ = objective_pair(model, reference, padded, RTMode((0,), 1))
                record(row)
            del model, reference, saved
            gc.collect()
            torch.cuda.empty_cache()
        fingerprint = {"checkpoint_sha256": manifest["checkpoint"]["sha256"], "source_hashes": report["source_hashes"],
                       "nextlat_revision": report["nextlat_reference_sources"]["revision"]}
        record(recovery_check(state, batch, args.output_dir, fingerprint))
        report.update(status="passed", elapsed_seconds=time.monotonic() - began,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      memory_scope="validation pair/optimizer recovery/digest scratch; not a batch-capacity measure")
        tracker.summary({"validation/status": "passed", "validation/peak_allocated_gib": report["peak_allocated_gib"]})
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
