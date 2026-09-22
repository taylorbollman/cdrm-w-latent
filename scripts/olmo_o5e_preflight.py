#!/usr/bin/env python3
"""Freeze the ordinary mixed-data control after H100 capacity/state checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_data import load_lm_data
from cdrm.pretrained.lm_training import LMTrainingConfig
from scripts.olmo_lm_common import state_digests
from scripts.olmo_o5b_common import PilotTracker, preserve_rng, evaluation, finite_optimizer
from scripts.olmo_o5c_data import load_o5c_data
from scripts.olmo_o5d_common import endpoint_metadata as comparison_metadata
from scripts.olmo_o5e_common import (load_control, build_optimizer, build_scheduler,
    frozen_fusion_digests, trainable_layout, observed_step, make_mode, source_hashes, plain_metadata)
from scripts.olmo_o5e_train import microbatches, validate_configuration
from scripts.olmo_validation import require_container_gpu


def profile(model, batch, config):
    before = state_digests(model)
    optimizer = build_optimizer(model, config, zero_lr=True)
    training = LMTrainingConfig(precision=config["precision"], max_grad_norm=config["max_grad_norm"])
    chunks = microbatches(batch, config["physical_batch_size"])
    def step():
        return observed_step(model, optimizer, chunks, config=training, backbone_kwargs={"mode": make_mode()})
    for _ in range(2): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    seconds = []
    for _ in range(3):
        began = time.monotonic(); last = step(); torch.cuda.synchronize(); seconds.append(time.monotonic()-began)
    result = {"wall_seconds": seconds, "median_wall_seconds": statistics.median(seconds),
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "input_tokens": int(batch.valid_mask.sum()), "segments": batch.input_ids.shape[0],
        "physical_batch_size": config["physical_batch_size"], "warmup_steps": 2, "timed_steps": 3,
        "state_unchanged": before == state_digests(model), "optimizer_finite": finite_optimizer(model, optimizer),
        "last_step": last}
    if not result["state_unchanged"] or not result["optimizer_finite"] or result["peak_allocated_gib"] > 60:
        raise AssertionError("Zero-LR state or bounded capacity preflight failed")
    del optimizer, chunks
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "base-data", "output-dir"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    runtime = plain_metadata(require_container_gpu())
    torch.set_num_threads(8); torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "olmo-o5e-preflight-v1", "status": "running", "runtime": runtime,
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": source_hashes(), "profile": {}}
    tracker = PilotTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        name="olmo-1b-o5e-ordinary-preflight", group="olmo1b-o5e-ordinary-control", preserve_state=preserve_rng)
    try:
        reference = comparison_metadata()
        original = reference["configuration"]
        if runtime != reference["source_fingerprint"]["runtime"]:
            raise ValueError("Runtime differs from the O5c comparison")
        corpus = load_o5c_data(args.data, args.base_data, expected_manifest_sha256=original["data_manifest_sha256"])
        eval_corpus = load_lm_data(args.base_data)
        plan = corpus.plan("mixed")
        if (plan != original["schedule"]["arms"]["mixed"]
                or eval_corpus.manifest_sha256 != original["base_data_manifest_sha256"]):
            raise ValueError("Shared mixed plan or evaluation data differs")
        config = {"schema": "olmo-o5e-config-v1", "source_hashes": report["source_hashes"], "runtime": runtime,
            "data_manifest_sha256": corpus.manifest_sha256, "base_data_manifest_sha256": eval_corpus.manifest_sha256,
            "source_checkpoint_sha256": original["source_checkpoint_sha256"], "seed": 20260922,
            "reference_o5c_mixed_report_sha256": reference["report_sha256"],
            "reference_o5c_mixed_checkpoint_sha256": reference["checkpoint"]["sha256"],
            "lr": 1e-5, "warmup_updates": 50, "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1,
            "max_grad_norm": 1., "precision": "bf16_mixed", "attention_backend": "sdpa",
            "physical_batch_size": 16, "beta": 0., "fbt_enabled": False, "num_passes": 1, "gamma": 1.,
            "rt_layers": [], "nextlat_enabled": False, "native_backbone_frozen": False,
            "objective": "single_ordinary_ce", "data_plan_arm": "mixed", "prefix_mixin": False,
            "hidden_jitter": 0., "compile": False, "cuda_graphs": False, "distributed": False,
            "schedule": {"total_updates": 512, "ce_per_update": 8192, "total_ce": 4194304,
                         "arms": {"mixed": plan}},
            "eval_every_updates": 64, "eval_rows": 128, "final_eval_rows": 512, "eval_batch_size": 8,
            "checkpoint_every_updates": 128, "checkpoint_interval_seconds": 600,
            "checkpoint_estimated_bytes": 15_000_000_000,
            "catastrophic_nll_increase": 1.5, "catastrophic_consecutive_evals": 2}
        validate_configuration(config)
        tracker.start(config)
        model, provenance = load_control()
        report["endpoint"] = provenance
        config["model_config"] = model.backbone.config.to_dict()
        before = config["full_state_initial"] = state_digests(model)
        frozen = config["frozen_state_initial"] = report["frozen_state_initial"] = frozen_fusion_digests(model)
        config["trainable_layout"] = report["trainable_layout"] = trainable_layout(model)
        config["trainable_parameters"] = sum(row["numel"] for row in config["trainable_layout"])
        counts = plan["batch_row_prefix"]
        busiest = max(range(512), key=lambda update: counts[update+1]-counts[update])
        for label, update in (("first", 0), ("largest_row_count", busiest)):
            batch = corpus.batch_for_update("mixed", update, device="cuda")
            row = report["profile"][label] = {"update_index": update, **profile(model, batch, config)}
            tracker.log({f"profile/{label}/seconds": row["median_wall_seconds"],
                         f"profile/{label}/peak_allocated_gib": row["peak_allocated_gib"]})
            print({"profile": label, "update_index": update, "seconds": row["median_wall_seconds"],
                   "peak_gib": row["peak_allocated_gib"]}, flush=True)
            del batch
        saved = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, warmup_updates=50)
        batch = corpus.batch_for_update("mixed", 0, device="cuda")
        result = observed_step(model, optimizer, microbatches(batch, 16),
            config=LMTrainingConfig(precision="bf16_mixed", max_grad_norm=1.),
            backbone_kwargs={"mode": make_mode()}, scheduler=scheduler)
        after = state_digests(model)
        changed = [name for name in before if before[name] != after[name]]
        if set(changed) != set(saved) or frozen_fusion_digests(model) != frozen or not finite_optimizer(model, optimizer):
            raise AssertionError("Disposable step must change native tensors only and stay finite")
        report["nonzero_step"] = {"result": result, "changed_tensors": changed, "fusion_unchanged": True}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in saved: parameter.copy_(saved[name])
        del optimizer, scheduler, batch, saved
        model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        if state_digests(model) != before:
            raise AssertionError("Disposable native step restoration failed")
        report["baseline"] = evaluation(model, eval_corpus, config, make_mode(), 512, documents=True)
        report["baseline_small"] = evaluation(model, eval_corpus, config, make_mode(), 128)
        expected_nll = {"dev": 1.6982163369972814, "retention_dev": 3.1833611041846672}
        for split, value in report["baseline"].items():
            if abs(value["passes"][0]["mean_nll"]-expected_nll[split]) > 1e-6:
                raise AssertionError("Initial ordinary scores do not reproduce the shared source")
            tracker.log({f"baseline/{split}/nll": value["passes"][0]["mean_nll"]})
        if state_digests(model) != before or source_hashes() != config["source_hashes"]:
            raise AssertionError("Preflight model/source mutation")
        write_json(args.output_dir/"configuration.json", config)
        report.update(status="passed", configuration_sha256=sha256_file(args.output_dir/"configuration.json"),
                      full_state_initial=before, weights_unchanged=True)
        tracker.summary({"preflight/status": "passed"})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error)); raise
    finally:
        try: tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json", report)


if __name__ == "__main__": main()
