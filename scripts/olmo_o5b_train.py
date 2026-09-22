#!/usr/bin/env python3
"""One matched O5b learning arm with boundary checkpoints and online metrics."""
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
from cdrm.pretrained.lm_schedule import alpha_for_update, cursor_for_update
from cdrm.pretrained.lm_training import (LMTrainingConfig, TrainingCounters,
    save_training_checkpoint, load_training_checkpoint)
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, validate_prepared_manifest
from scripts.olmo_o5b_common import (ARMS, PilotTracker, preserve_rng, retain_file, source_hashes,
    write_event, build_model, build_optimizer, build_scheduler, observed_step, microbatches,
    make_mode, evaluation, online_evaluation, catastrophic, finite_optimizer, resume_record)
from scripts.olmo_validation import require_container_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-prefix", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    runtime = require_container_gpu()
    configuration = json.loads(args.configuration.read_text())
    if configuration.get("schema") != "olmo-o5b-pilot-config-v1":
        raise ValueError("Expected frozen O5b configuration")
    storage_prefix = args.storage_prefix.rstrip("/")
    checkpoint_configuration = {**configuration, "arm": args.arm, "storage_prefix": storage_prefix}
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(configuration["seed"]); np.random.seed(configuration["seed"])
    torch.manual_seed(configuration["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=args.resume is not None)
    source = {"checkpoint_sha256": configuration["checkpoint_sha256"], "code": source_hashes(),
              "data_manifest_sha256": configuration["data_manifest_sha256"], "runtime": runtime}
    if source["code"] != configuration["source_hashes"]:
        raise ValueError("Pilot source differs from the frozen preflight configuration")
    report_path = args.output_dir / "report.json"
    report = json.loads(report_path.read_text()) if args.resume else {
        "schema": "olmo-o5b-arm-v1", "status": "running", "arm": args.arm,
        "started_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "source_fingerprint": source, "evaluations": [], "online_evaluations": [],
        "storage_prefix": storage_prefix, "checkpoints": [], "resumptions": []}
    if report["configuration"] != configuration or report["source_fingerprint"] != source or report["storage_prefix"] != storage_prefix:
        raise ValueError("Resume configuration/source/runtime differs from recorded arm")
    native = validate_prepared_manifest(args.artifacts)
    if native["checkpoint"]["sha256"] != configuration["checkpoint_sha256"]:
        raise ValueError("Different native starting checkpoint")
    corpus = load_lm_data(args.data)
    if corpus.manifest_sha256 != configuration["data_manifest_sha256"]:
        raise ValueError("Prepared data manifest differs from frozen preflight")
    use_fbt = args.arm == "fbt"
    state = load_native_state_dict(args.artifacts)
    model = build_model(state, backend=configuration["attention_backend"])
    del state
    if model.config.to_dict() != configuration["nextlat_config"] or model.enabled:
        raise ValueError("O5b requires the recorded NextLat-disabled wrapper")
    if model.backbone.config.to_dict() != configuration["model_config"]:
        raise ValueError("Native backbone resolved configuration differs")
    if model.backbone.fusion_config.to_dict() != configuration["fusion_config"] or model.gamma != configuration["gamma"]:
        raise ValueError("Fusion or pass-loss configuration differs")
    optimizer = build_optimizer(model, configuration)
    schedule = configuration["schedule"]
    scheduler = build_scheduler(optimizer, configuration)
    training_config = LMTrainingConfig(precision=configuration["precision"], max_grad_norm=configuration["max_grad_norm"])
    counters, cursor, bad_evals = TrainingCounters(), 0, 0
    if args.resume:
        saved = resume_record(report, args.resume)
        restored = load_training_checkpoint(args.resume, model, optimizer, scheduler=scheduler,
            configuration=checkpoint_configuration, source_fingerprint=source, expected_sha256=saved["sha256"])
        counters = restored["counters"]
        cursor = restored["data_cursor"]["next_window"]
        bad_evals = restored["data_cursor"]["bad_evals"]
        report["resumptions"].append({"from_update": counters.optimizer_updates, "path": str(args.resume),
                                      "utc": datetime.now(timezone.utc).isoformat()})
        for key in ("evaluations","online_evaluations"):
            abandoned=[row for row in report[key] if row["update"]>counters.optimizer_updates]
            if abandoned:
                report.setdefault("abandoned_"+key,[]).extend(abandoned)
                report[key]=[row for row in report[key] if row["update"]<=counters.optimizer_updates]
    if cursor != cursor_for_update(schedule, counters.optimizer_updates):
        raise ValueError("Restored window cursor differs from frozen update schedule")
    tracker = PilotTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        name=f"olmo-1b-o5b-{args.arm}", group="olmo1b-o5b-code-pilot", preserve_state=preserve_rng)
    requested_stop = False
    def stop_handler(signum, frame):
        nonlocal requested_stop
        requested_stop = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    last_save_time = time.monotonic()
    last_eval_bucket = counters.input_tokens // configuration["eval_every_tokens"]
    report["status"] = "running"

    def persist():
        report.update(counters=asdict(counters), data_cursor=cursor, wandb=tracker.record)
        write_json(report_path, report)

    def checkpoint(reason):
        nonlocal last_save_time
        # One canonical checkpoint per update, even when boundaries coincide.
        name = f"update-{counters.optimizer_updates:06d}.pt"
        path = args.output_dir / name
        prior = next((row for row in report["checkpoints"] if Path(row["path"]).name == name), None)
        if prior is not None and prior.get("storage"):
            return prior
        if prior is None:
            if not finite_optimizer(model, optimizer):
                raise FloatingPointError("Nonfinite model/AdamW state at completed boundary")
            record = save_training_checkpoint(path, model, optimizer, scheduler=scheduler, counters=counters,
                data_cursor={"next_window": cursor, "bad_evals": bad_evals}, configuration=checkpoint_configuration,
                source_fingerprint=source)
            record.update(reason=reason, input_tokens=counters.input_tokens)
            # Record local recovery before any network operation. A failed upload
            # leaves this complete file and local hash available for recovery.
            report["checkpoints"].append(record); persist()
        else:
            record = prior
        record["storage"] = retain_file(path, f"{storage_prefix}/{args.arm}/{name}",
                                         expected_sha256=record["sha256"])
        write_json(path.with_suffix(".receipt.json"), record)
        persist()
        # Keep the latest local checkpoint; earlier ones are removed only after
        # their own immutable cloud receipt exists and the successor is retained.
        for older in report["checkpoints"][:-1]:
            older_path = Path(older["path"])
            if older.get("storage") and older_path.parent == args.output_dir and older_path.is_file():
                older_path.unlink()
                older["local_file_removed_after_verified_successor"] = True
        last_save_time = time.monotonic()
        persist()
        print({"checkpoint": name, "reason": reason, "uri": record["storage"]["uri"]}, flush=True)
        return record

    def evaluate(rows, *, full=False):
        beta = alpha_for_update(schedule, counters.optimizer_updates, counters.input_tokens) if use_fbt else 0.0
        mode = make_mode(beta)
        values = evaluation(model, corpus, configuration, mode, rows, documents=full)
        event = {"kind": "evaluation", "update": counters.optimizer_updates, "input_tokens": counters.input_tokens,
                 "beta": beta, "full": full, "rows_requested": rows, "metrics": values}
        old = [row for row in report["evaluations"] if row["update"] == event["update"] and row["full"] == full]
        if old:
            report.setdefault("superseded_evaluations", []).extend(old)
            report["evaluations"] = [row for row in report["evaluations"] if row not in old]
        report["evaluations"].append(event)
        write_event(args.output_dir, event)
        prefix = "final" if full else "dev"
        metrics = {"update": counters.optimizer_updates, f"{prefix}/input_tokens": counters.input_tokens,
                   f"{prefix}/beta": beta}
        for split, result in values.items():
            for p, item in enumerate(result["passes"]):
                for key in ("mean_nll", "perplexity", "next_token_accuracy", "ce_count"):
                    metrics[f"{prefix}/{split}/pass_{p}/{key}"] = item[key]
        tracker.log(metrics)
        print({"eval_update": counters.optimizer_updates, "tokens": counters.input_tokens, "beta": beta,
               "full": full, "code_nll": [r["mean_nll"] for r in values["dev"]["passes"]],
               "retention_nll": [r["mean_nll"] for r in values["retention_dev"]["passes"]]}, flush=True)
        persist()
        return values

    def evaluate_online():
        beta = alpha_for_update(schedule,counters.optimizer_updates,counters.input_tokens) if use_fbt else 0.0
        values = online_evaluation(model,corpus,configuration,beta)
        event = {"kind":"online_evaluation","update":counters.optimizer_updates,
                 "input_tokens":counters.input_tokens,"beta":beta,"rows_requested":configuration["online_eval_rows"],
                 "max_length":configuration["online_eval_length"],"metrics":values}
        old=[r for r in report["online_evaluations"] if r["update"]==event["update"]]
        if old:
            report.setdefault("superseded_online_evaluations",[]).extend(old)
            report["online_evaluations"]=[r for r in report["online_evaluations"] if r not in old]
        report["online_evaluations"].append(event); write_event(args.output_dir,event)
        metrics={"update":counters.optimizer_updates,"online/input_tokens":counters.input_tokens,"online/beta":beta}
        for split,result in values.items():
            for name,item in result.items():
                for key in ("mean_nll","next_token_accuracy"):
                    metrics[f"online/{split}/{name}/{key}"]=item[key]
        tracker.log(metrics);persist()
        print({"online_update":counters.optimizer_updates,"beta":beta,
               "code_nll":{k:v["mean_nll"] for k,v in values["dev"].items()}},flush=True)

    try:
        tracker.start(configuration, run_id=report.get("wandb", {}).get("run_id") if args.resume else None)
        persist()
        if args.resume and not saved.get("storage"):
            checkpoint("resume_retention_retry")
        if not args.resume:
            report["initial_small_evaluation"] = evaluate(configuration["eval_rows"])
            evaluate_online()
            persist()
        initial = report["initial_small_evaluation"]
        while counters.optimizer_updates < schedule["total_updates"]:
            if requested_stop or (args.output_dir / "STOP").exists():
                checkpoint("requested_stop")
                report["status"] = "paused"
                break
            beta = alpha_for_update(schedule, counters.optimizer_updates, counters.input_tokens) if use_fbt else 0.0
            mode = make_mode(beta)
            batch, next_cursor = corpus.next_train_batch(cursor, schedule["batch_size"], device="cuda")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            result = observed_step(model, optimizer, microbatches(batch,configuration["physical_batch_size"]), config=training_config,
                backbone_kwargs={"mode": mode}, scheduler=scheduler, counters=counters)
            torch.cuda.synchronize()
            seconds = time.monotonic() - began
            cursor = next_cursor
            if cursor != cursor_for_update(schedule, counters.optimizer_updates):
                raise AssertionError("Data stream cursor drift")
            # Validate exact token accounting even for ordinary arms.
            alpha_for_update(schedule, counters.optimizer_updates, counters.input_tokens)
            event = {"kind": "update", "beta": beta, "seconds": seconds,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, **result}
            write_event(args.output_dir, event)
            metrics = {"update": counters.optimizer_updates, "train/input_tokens": counters.input_tokens,
                "train/supervised_tokens": counters.ce_positions, "train/beta": beta,
                "train/lr": result["lr_used"][0], "train/objective": result["objective"],
                "train/gradient_norm": result["gradient_norm_before_clip"], "train/step_seconds": seconds,
                "train/peak_allocated_gib": event["peak_allocated_gib"]}
            metrics.update({f"train/{name}": value for name, value in result["loss_means"].items()})
            for p,value in enumerate(result["pass_ce_means"]):
                metrics[f"train/pass_{p}/ce"] = value
            metrics["train/fusion_lr"] = result["lr_used"][1]
            metrics["train/stack_input_tokens"] = 2*counters.input_tokens
            for group,value in result["group_gradient_norm_after_clip"].items():
                metrics[f"train/{group}/gradient_norm_after_clip"] = value
            tracker.log(metrics)
            report["last_update"] = event
            if counters.optimizer_updates % 10 == 0:
                print({"update": counters.optimizer_updates, "tokens": counters.input_tokens, "beta": beta,
                       "ce": result["loss_means"]["ce"], "seconds": seconds}, flush=True)
                persist()
            at_boundary = counters.optimizer_updates in (schedule["warmup_updates"], schedule["ramp_end_update"], schedule["total_updates"])
            bucket = counters.input_tokens // configuration["eval_every_tokens"]
            if bucket > last_eval_bucket or at_boundary or counters.optimizer_updates == configuration["health_gate_update"]:
                values = evaluate(configuration["eval_rows"])
                bad_evals = bad_evals + 1 if catastrophic(values, initial, configuration["catastrophic_nll_increase"]) else 0
                last_eval_bucket = bucket
                if bad_evals >= configuration["catastrophic_consecutive_evals"]:
                    checkpoint("catastrophic_dev_deterioration")
                    report["status"] = "health_gate_failed"
                    break
            if counters.optimizer_updates == schedule["ramp_end_update"]:
                evaluate_online()
            if at_boundary or time.monotonic() - last_save_time >= configuration["checkpoint_interval_seconds"]:
                checkpoint("phase_boundary" if at_boundary else "periodic_recovery")
        else:
            checkpoint("final")
            evaluate(configuration["final_eval_rows"], full=True)
            evaluate_online()
            report["status"] = "completed"
        tracker.summary({"pilot/status": report["status"], "pilot/input_tokens": counters.input_tokens,
                         "pilot/updates": counters.optimizer_updates})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        # Do not serialize a partly applied or suspect update as recoverable.
        persist()
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] in ("completed", "paused"))
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            persist()
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
