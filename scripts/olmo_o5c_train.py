#!/usr/bin/env python3
"""Bounded fusion-only adaptation with immutable native weights and recovery."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.lm_data import load_lm_data
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from scripts.olmo_o5b_common import (PilotTracker, preserve_rng, write_event,
    observed_step, make_mode, evaluation, finite_optimizer, slice_batch)
from scripts.olmo_o5c_common import (endpoint_metadata, load_frozen_endpoint,
    build_optimizer, build_scheduler, frozen_state_digests, trainable_layout,
    source_hashes, save_checkpoint, load_checkpoint, retain_file, plain_metadata)
from scripts.olmo_o5c_data import load_o5c_data
from scripts.olmo_validation import require_container_gpu


def microbatches(batch, physical):
    if type(physical) is not int or physical < 1:
        raise ValueError("Positive physical batch size required")
    return [slice_batch(batch, slice(i, i + physical))
            for i in range(0, batch.input_ids.shape[0], physical)]


def exposure(plan, update):
    return {domain: {key: values[update] for key, values in fields.items()}
            for domain, fields in plan["domain_prefixes"].items()}


def check_counters(plan, counters):
    u = counters.optimizer_updates
    if (counters.input_tokens != plan["batch_token_prefix"][u]
            or counters.ce_positions != plan["batch_ce_prefix"][u]
            or counters.documents != plan["batch_row_prefix"][u]):
        raise ValueError("Training counters differ from frozen segmented data plan")


def health_failure(current, initial, margin):
    return any(current[d]["passes"][1]["mean_nll"] > initial[d]["passes"][1]["mean_nll"] + margin
               for d in ("dev", "retention_dev"))


def validate_model_configuration(model, config):
    if (model.backbone.config.to_dict() != config["model_config"]
            or trainable_layout(model) != config["trainable_layout"]):
        raise ValueError("Model configuration or trainable ownership differs from preflight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--arm", choices=("code", "mixed"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-prefix", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    runtime = plain_metadata(require_container_gpu())
    config = json.loads(args.configuration.read_text())
    if config.get("schema") != "olmo-o5c-pilot-config-v1" or config["source_hashes"] != source_hashes():
        raise ValueError("Require frozen O5c sources and configuration")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    corpus = load_o5c_data(args.data, args.base_data, expected_manifest_sha256=config["data_manifest_sha256"])
    eval_corpus = load_lm_data(args.base_data)
    if eval_corpus.manifest_sha256 != config["base_data_manifest_sha256"]:
        raise ValueError("Evaluation/base data changed")
    plan = corpus.plan(args.arm)
    if plan != config["schedule"]["arms"][args.arm]:
        raise ValueError("Frozen data plan differs")
    endpoint = endpoint_metadata()
    model, provenance = load_frozen_endpoint(endpoint["checkpoint_path"], config["source_checkpoint_sha256"],
        expected_configuration=endpoint["configuration"], expected_source_fingerprint=endpoint["source_fingerprint"])
    frozen = frozen_state_digests(model)
    if frozen != config["frozen_state_initial"]:
        raise ValueError("Native endpoint state differs from preflight")
    validate_model_configuration(model, config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, warmup_updates=config["warmup_updates"])
    training = LMTrainingConfig(precision=config["precision"], max_grad_norm=config["max_grad_norm"])
    source = {"checkpoint_sha256": config["source_checkpoint_sha256"],
              "source_checkpoint_sha256": config["source_checkpoint_sha256"],
              "code": source_hashes(), "data_manifest_sha256": corpus.manifest_sha256, "runtime": runtime}
    prefix = args.storage_prefix.rstrip("/")
    checkpoint_config = {**config, "arm": args.arm, "storage_prefix": prefix}
    args.output_dir.mkdir(parents=True, exist_ok=args.resume is not None)
    report_path = args.output_dir / "report.json"
    report = json.loads(report_path.read_text()) if args.resume else {
        "schema": "olmo-o5c-arm-v1", "status": "running", "arm": args.arm,
        "configuration": config, "source_fingerprint": source, "storage_prefix": prefix,
        "started_utc": datetime.now(timezone.utc).isoformat(), "endpoint": provenance,
        "frozen_state_initial": frozen, "trainable_layout": trainable_layout(model),
        "evaluations": [], "checkpoints": [], "resumptions": []}
    if report["configuration"] != config or report["source_fingerprint"] != source or report["storage_prefix"] != prefix:
        raise ValueError("Resume lineage/runtime changed")
    counters, bad_evals = TrainingCounters(), 0
    if args.resume:
        records = report["checkpoints"]
        if not records or Path(records[-1]["path"]).name != args.resume.name:
            raise ValueError("Resume only the latest recorded boundary")
        restored = load_checkpoint(args.resume, model, optimizer, scheduler=scheduler,
            configuration=checkpoint_config, source_fingerprint=source, expected_sha256=records[-1]["sha256"])
        counters = restored["counters"]
        bad_evals = restored["data_cursor"]["bad_evals"]
        if restored["data_cursor"]["next_update"] != counters.optimizer_updates:
            raise ValueError("Restored cursor differs")
        report["resumptions"].append({"from_update": counters.optimizer_updates, "path": str(args.resume),
                                      "utc": datetime.now(timezone.utc).isoformat()})
        abandoned = [row for row in report["evaluations"] if row["update"] > counters.optimizer_updates]
        report.setdefault("abandoned_evaluations", []).extend(abandoned)
        report["evaluations"] = [r for r in report["evaluations"] if r not in abandoned]
        if frozen_state_digests(model) != frozen:
            raise ValueError("Recovered native state differs")
    check_counters(plan, counters)
    tracker = PilotTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        name=f"olmo-1b-o5c-fusion-only-{args.arm}", group="olmo1b-o5c-fusion-only", preserve_state=preserve_rng)
    requested_stop = False
    def stop_handler(signum, frame):
        nonlocal requested_stop
        requested_stop = True
    signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler)
    last_save_time = time.monotonic()
    report["status"] = "running"

    def persist():
        report.update(counters=asdict(counters), data_cursor=counters.optimizer_updates,
                      domain_counts=exposure(plan, counters.optimizer_updates), wandb=tracker.record)
        write_json(report_path, report)

    def checkpoint(reason):
        nonlocal last_save_time
        digest = frozen_state_digests(model)
        if digest != frozen or not finite_optimizer(model, optimizer):
            raise FloatingPointError("Frozen state changed or model/optimizer became nonfinite")
        report["frozen_state_final"] = digest
        name = f"update-{counters.optimizer_updates:06d}.pt"
        path = args.output_dir / name
        previous = next((r for r in report["checkpoints"] if Path(r["path"]).name == name), None)
        if previous is None:
            record = save_checkpoint(path, model, optimizer, scheduler=scheduler, counters=counters,
                data_cursor={"next_update": counters.optimizer_updates, "bad_evals": bad_evals},
                configuration=checkpoint_config, source_fingerprint=source)
            record.update(reason=reason, input_tokens=counters.input_tokens, ce_positions=counters.ce_positions)
            report["checkpoints"].append(record); persist()
        else:
            record = previous
        if not record.get("storage"):
            record["storage"] = retain_file(path, f"{prefix}/{args.arm}/{name}", expected_sha256=record["sha256"])
            write_json(path.with_suffix(".receipt.json"), record)
        persist()
        for older in report["checkpoints"][:-1]:
            old_path = Path(older["path"])
            if older.get("storage") and old_path.parent == args.output_dir and old_path.is_file():
                old_path.unlink(); older["local_file_removed_after_verified_successor"] = True
        last_save_time = time.monotonic(); persist()
        print({"checkpoint": name, "reason": reason, "uri": record["storage"]["uri"]}, flush=True)

    def evaluate(rows, *, full=False):
        values = evaluation(model, eval_corpus, config, make_mode(1), rows, documents=full)
        event = {"kind": "evaluation", "update": counters.optimizer_updates,
            "input_tokens": counters.input_tokens, "ce_positions": counters.ce_positions,
            "domain_counts": exposure(plan, counters.optimizer_updates), "beta": 1.0,
            "full": full, "rows_requested": rows, "metrics": values}
        old = [r for r in report["evaluations"] if r["update"] == event["update"] and r["full"] == full]
        report.setdefault("superseded_evaluations", []).extend(old)
        report["evaluations"] = [r for r in report["evaluations"] if r not in old] + [event]
        write_event(args.output_dir, event)
        prefix = "final" if full else "dev"
        metrics = {"update": counters.optimizer_updates, f"{prefix}/ce_positions": counters.ce_positions}
        for split, result in values.items():
            for p, item in enumerate(result["passes"]):
                for key in ("mean_nll", "perplexity", "next_token_accuracy", "ce_count"):
                    metrics[f"{prefix}/{split}/pass_{p}/{key}"] = item[key]
        tracker.log(metrics); persist()
        print({"eval_update": counters.optimizer_updates, "full": full,
            "code_nll": [p["mean_nll"] for p in values["dev"]["passes"]],
            "retention_nll": [p["mean_nll"] for p in values["retention_dev"]["passes"]]}, flush=True)
        return values

    try:
        tracker.start(config, run_id=report.get("wandb", {}).get("run_id") if args.resume else None)
        persist()
        if args.resume and not report["checkpoints"][-1].get("storage"):
            checkpoint("resume_retention_retry")
        if not args.resume:
            report["initial_small_evaluation"] = evaluate(config["eval_rows"])
            persist()
        initial = report["initial_small_evaluation"]
        while counters.optimizer_updates < plan["total_updates"]:
            if requested_stop or (args.output_dir / "STOP").exists() or (args.output_dir.parent / "STOP").exists():
                checkpoint("requested_stop"); report["status"] = "paused"; break
            batch = corpus.batch_for_update(args.arm, counters.optimizer_updates, device="cuda")
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); began = time.monotonic()
            result = observed_step(model, optimizer, microbatches(batch, config["physical_batch_size"]),
                config=training, backbone_kwargs={"mode": make_mode(1)}, scheduler=scheduler, counters=counters)
            torch.cuda.synchronize(); seconds = time.monotonic() - began
            check_counters(plan, counters)
            event = {"kind": "update", "seconds": seconds, "beta": 1.,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "domain_counts": exposure(plan, counters.optimizer_updates), **result}
            write_event(args.output_dir, event)
            metrics = {"update": counters.optimizer_updates, "train/input_tokens": counters.input_tokens,
                "train/ce_positions": counters.ce_positions, "train/objective": result["objective"],
                "train/gradient_norm": result["gradient_norm_before_clip"], "train/step_seconds": seconds,
                "train/fusion_lr": result["lr_used"][0], "train/stack_input_tokens": 2*counters.input_tokens,
                "train/peak_allocated_gib": event["peak_allocated_gib"]}
            for p, value in enumerate(result["pass_ce_means"]): metrics[f"train/pass_{p}/ce"] = value
            for domain, counts in event["domain_counts"].items():
                for key, value in counts.items(): metrics[f"exposure/{domain}/{key}"] = value
            tracker.log(metrics); report["last_update"] = event
            u = counters.optimizer_updates
            if u % 10 == 0:
                persist(); print({"update": u, "ce_positions": counters.ce_positions,
                                 "pass_ce": result["pass_ce_means"], "seconds": seconds}, flush=True)
            if u == 50 or u % config["eval_every_updates"] == 0 or u == plan["total_updates"]:
                values = evaluate(config["eval_rows"])
                bad_evals = bad_evals + 1 if health_failure(values, initial, config["catastrophic_nll_increase"]) else 0
                if bad_evals >= config["catastrophic_consecutive_evals"]:
                    checkpoint("development_health_gate"); report["status"] = "health_gate_failed"; break
            if u % config["checkpoint_every_updates"] == 0 or time.monotonic()-last_save_time >= config["checkpoint_interval_seconds"]:
                checkpoint("periodic_recovery")
        else:
            checkpoint("final"); evaluate(config["final_eval_rows"], full=True)
            report["status"] = "completed"
        tracker.summary({"pilot/status": report["status"], "pilot/ce_positions": counters.ce_positions,
                         "pilot/updates": counters.optimizer_updates})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__); persist(); raise
    finally:
        try: tracker.finish(succeeded=report["status"] in ("completed", "paused"))
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat(); persist()
    if report["status"] != "completed": raise SystemExit(2)


if __name__ == "__main__": main()
