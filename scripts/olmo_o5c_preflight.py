#!/usr/bin/env python3
"""Freeze O5c after real-data fusion-only capacity and frozen-state checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.lm_data import load_lm_data
from cdrm.pretrained.lm_training import LMTrainingConfig
from scripts.olmo_lm_common import state_digests
from scripts.olmo_o5b_common import (PilotTracker, preserve_rng, make_mode, observed_step,
    evaluation, finite_optimizer)
from scripts.olmo_o5c_common import (endpoint_metadata, load_frozen_endpoint, source_hashes,
    plain_metadata, build_optimizer, build_scheduler, frozen_state_digests, trainable_layout, FUSION_NAMES)
from scripts.olmo_o5c_data import load_o5c_data
from scripts.olmo_o5c_train import microbatches
from scripts.olmo_validation import require_container_gpu


def profile(model, batch, config):
    before = state_digests(model)
    optimizer = build_optimizer(model, config, zero_lr=True)
    training = LMTrainingConfig(precision=config["precision"], max_grad_norm=config["max_grad_norm"])
    chunks = microbatches(batch, config["physical_batch_size"])
    def step():
        return observed_step(model, optimizer, chunks, config=training, backbone_kwargs={"mode": make_mode(1)})
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
        raise AssertionError("Frozen/zero-LR state or bounded capacity preflight failed")
    del optimizer, chunks
    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runtime = plain_metadata(require_container_gpu())
    torch.set_num_threads(8); torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "olmo-o5c-preflight-v1", "status": "running", "runtime": runtime,
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": source_hashes(), "profile": {}}
    tracker = PilotTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        name="olmo-1b-o5c-fusion-only-preflight", group="olmo1b-o5c-fusion-only", preserve_state=preserve_rng)
    try:
        corpus = load_o5c_data(args.data, args.base_data)
        eval_corpus = load_lm_data(args.base_data)
        endpoint = endpoint_metadata()
        config = {"schema": "olmo-o5c-pilot-config-v1", "source_hashes": report["source_hashes"],
            "data_manifest_sha256": corpus.manifest_sha256, "base_data_manifest_sha256": eval_corpus.manifest_sha256,
            "source_checkpoint_sha256": endpoint["checkpoint"]["sha256"], "seed": 20260922,
            "lr": 1e-4, "warmup_updates": 50, "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1,
            "max_grad_norm": 1., "precision": "bf16_mixed", "attention_backend": "sdpa",
            "physical_batch_size": 16, "beta": 1., "num_passes": 2, "gamma": 1., "rt_layers": [],
            "nextlat_enabled": False, "native_backbone_frozen": True, "prefix_mixin": False,
            "hidden_jitter": 0., "compile": False, "cuda_graphs": False, "distributed": False,
            "schedule": {"total_updates": 512, "ce_per_update": 8192, "total_ce": 4194304,
                         "arms": {arm: corpus.plan(arm) for arm in ("code", "mixed")}},
            "eval_every_updates": 64, "eval_rows": 128, "final_eval_rows": 512, "eval_batch_size": 8,
            "checkpoint_every_updates": 128, "checkpoint_interval_seconds": 600,
            "catastrophic_nll_increase": 1.5, "catastrophic_consecutive_evals": 2}
        tracker.start(config); report["wandb"] = tracker.record
        model, provenance = load_frozen_endpoint(endpoint["checkpoint_path"], config["source_checkpoint_sha256"],
            expected_configuration=endpoint["configuration"], expected_source_fingerprint=endpoint["source_fingerprint"])
        report["endpoint"] = provenance
        config["model_config"] = model.backbone.config.to_dict()
        before = state_digests(model)
        config["frozen_state_initial"] = report["frozen_state_initial"] = frozen_state_digests(model)
        config["trainable_layout"] = report["trainable_layout"] = trainable_layout(model)
        for arm in ("code", "mixed"):
            batch = corpus.batch_for_update(arm, 0, device="cuda")
            report["profile"][arm] = profile(model, batch, config)
            row = report["profile"][arm]
            tracker.log({f"profile/{arm}/seconds": row["median_wall_seconds"],
                         f"profile/{arm}/peak_allocated_gib": row["peak_allocated_gib"]})
            print({"profile_arm": arm, "seconds": row["median_wall_seconds"],
                   "peak_gib": row["peak_allocated_gib"]}, flush=True)
            del batch
        # One bounded nonzero step verifies the real frozen-backbone gradient
        # path. Restore both fusion matrices afterwards; discard its AdamW/RNG.
        fusion = {n: p.detach().clone() for n, p in model.named_parameters() if n in FUSION_NAMES}
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, warmup_updates=config["warmup_updates"])
        batch = corpus.batch_for_update("mixed", 0, device="cuda")
        result = observed_step(model, optimizer, microbatches(batch, config["physical_batch_size"]),
            config=LMTrainingConfig(precision=config["precision"], max_grad_norm=1),
            backbone_kwargs={"mode": make_mode(1)}, scheduler=scheduler)
        after = state_digests(model)
        changed = [n for n in before if before[n] != after[n]]
        if set(changed) != set(FUSION_NAMES) or frozen_state_digests(model) != config["frozen_state_initial"]:
            raise AssertionError("Real step must change exactly the two fusion matrices")
        if not finite_optimizer(model, optimizer): raise FloatingPointError("Real fusion step nonfinite")
        report["nonzero_step"] = {"result": result, "changed_tensors": changed, "native_unchanged": True}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in fusion: parameter.copy_(fusion[name])
        del optimizer, scheduler, batch, fusion
        model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        if state_digests(model) != before: raise AssertionError("Disposable step restoration failed")
        report["baseline"] = evaluation(model, eval_corpus, config, make_mode(1), 512, documents=True)
        report["baseline_small"] = evaluation(model, eval_corpus, config, make_mode(1), 128)
        for split, values in report["baseline"].items():
            for p, row in enumerate(values["passes"]): tracker.log({f"baseline/{split}/pass_{p}/nll": row["mean_nll"]})
        if source_hashes() != config["source_hashes"] or state_digests(model) != before:
            raise AssertionError("Sources/state changed during preflight")
        report.update(status="passed", data_manifest_sha256=corpus.manifest_sha256,
            source_checkpoint_sha256=config["source_checkpoint_sha256"], schedule=config["schedule"],
            weights_unchanged=True)
        write_json(args.output_dir/"configuration.json", config)
        tracker.summary({"preflight/status": "passed"})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__); raise
    finally:
        try: tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json", report)
    print({"status": "passed", "wandb": tracker.record["run_url"]}, flush=True)


if __name__ == "__main__": main()
