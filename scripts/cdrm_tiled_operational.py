#!/usr/bin/env python3
"""Retained five-block CDRM initialization, bounded OPS, recovery and update cost.

Only --mode prepare runs in the explicit CPU container. Model execution requires
the GPU container. The 100-update smoke test keeps the native 200-epoch cosine
schedule, and does not shorten its learning-rate horizon to this diagnostic run.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time

import numpy as np
import torch

from cdrm_tiled_common import (
    PRESET, TASK, INIT_FORMAT, FORMAT, POLICIES, setup, sources, snapshot_sources, verify_sources,
    cpu_initial_model, build_model, dataset_identity, validate_data, load_dataset, epoch_indices,
    optimizer_for, update, evaluate, save_checkpoint, cpu_tree, state_digest, effective_precision,
    atomic_json, append_jsonl, file_digest, json_digest, compiler_audit, rng_state, restore_rng,
    seed_all, preserve_tracking_rng)
from cdrm_train import compare_resumed
from r3_mixed_operational import measured_compiler_audit
from experiment_tracking import OnlineTracker, add_wandb_arguments, scalar_metrics

SEEDS = {"initialization": 7500, "train": 925701, "dev": 925702,
         "confirmation": 925703, "shuffle": 925704}
SIZES = {"train": 12800, "dev": 256, "confirmation": 128}
SCHEDULE = {"type": "CosineAnnealingLR", "T_max_epochs": 200, "eta_min": 1e-6,
            "step": "after each completed epoch", "warmup": False}
OPTIMIZER = {"type": "AdamW", "lr": 5e-4, "betas": [.9, .98], "eps": 1e-8,
             "weight_decay": 0., "foreach": False, "fused": False, "gradient_clip": 1.}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "train", "resume", "benchmark", "repeat"), required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16", "bf16_fp32_state"), default="fp32")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-final", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", type=Path, default=PRESET)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeat-batch", type=int, default=2)
    parser.add_argument("--repeat-steps", type=int, default=30)
    add_wandb_arguments(parser)
    parser.set_defaults(wandb_project="cdrm-tiled-bf16-validation")
    args = parser.parse_args()
    if args.mode != "prepare" and args.initial_checkpoint is None:
        parser.error("Every model run requires --initial-checkpoint from this five-block preparation")
    if (args.mode == "resume") != (args.checkpoint is not None):
        parser.error("Only --mode resume requires --checkpoint")
    if args.reference_final and args.mode != "resume":
        parser.error("--reference-final is only valid for midpoint recovery")
    if not 2 <= args.warmup or args.steps < 1 or args.warmup + args.steps > 100:
        parser.error("Require warmup>=2, steps>=1, and total<=100")
    if not 1 <= args.repeat_batch <= 8 or not 2 <= args.repeat_steps <= 100:
        parser.error("The repeated-batch diagnostic is bounded to B1..8 and 2..100 updates")
    if not args.wandb_project and (args.wandb_group or args.wandb_run_name):
        parser.error("W&B run naming requires --wandb-project")
    return args


def prepare(args, report):
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Preparation requires the project CPU container")
    if torch.cuda.is_available():
        raise RuntimeError("Use CDRM_DOCKER_GPUS=none for the explicitly CPU preparation command")
    if args.data_root.exists() and any(args.data_root.iterdir()):
        raise FileExistsError("Data preparation requires a new or empty data root")
    from cdrm.mad_data import generate_dataset, save_dataset, overlap_audit
    tracked = sources([args.preset])
    snapshot_sources(args.output_dir, tracked)
    freeze = {"schema": "cdrm-tiled-fixture-freeze-v1", "seeds": SEEDS, "sizes": SIZES,
              "task": TASK, "task_overrides": {"num_tokens_to_copy": 96},
              "preset_sha256": file_digest(args.preset), "source_sha256": tracked,
              "confirmation": {"root": str(args.data_root / "confirmation"), "split": "dev",
                               "initialization_example_offset": 0, "trained_example_offset": 64,
                               "physical_batch": 64},
              "training": {"physical_batch": 64, "updates": 100, "midpoint": 50,
                           "optimizer": OPTIMIZER, "schedule": SCHEDULE},
              "scope": "Prospectively chosen new initialization/data; no final-test split is generated"}
    atomic_json(args.output_dir / "fixture-freeze.json", freeze)
    args.data_root.mkdir(parents=True, exist_ok=True)
    data_records, datasets = {}, {}
    for name in ("train", "dev", "confirmation"):
        split = "train" if name == "train" else "dev"
        root = args.data_root / "confirmation" if name == "confirmation" else args.data_root
        data = generate_dataset(TASK, split, SEEDS[name], SIZES[name], {"num_tokens_to_copy": 96})
        validate_data(data)
        data_records[name] = save_dataset(root, data)
        datasets[name] = load_dataset(root, TASK, split)
    overlap = overlap_audit(datasets)
    if not overlap["no_exact_input_cross_split_overlap"]:
        raise AssertionError("Prospective training/development/confirmation draws overlap; retain and investigate")
    model, initialization = cpu_initial_model(SEEDS["initialization"], preset=args.preset)
    cfg = dataclasses.asdict(model.config)
    expected = {"n_layers": 5, "d_model": 128, "n_heads": 16, "n_kv_heads": 16,
                "mlp_hidden_size": 512, "cdrm_early_layer": 1, "cdrm_late_layer": 3,
                "vocab_size": 16, "cdrm_rho": 1.}
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Prepared architecture differs from the frozen five-block D128/H16 profile")
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    identity = {"schema": INIT_FORMAT, "source_sha256": tracked,
                "fixture_freeze_sha256": file_digest(args.output_dir / "fixture-freeze.json"),
                "preset_sha256": file_digest(args.preset),
                "data": {name: dataset_identity(data) for name, data in datasets.items()},
                "data_root": str(args.data_root), "task": TASK, "seeds": SEEDS,
                "optimizer": OPTIMIZER, "schedule": SCHEDULE, "model_config": cfg,
                "initialization": initialization}
    checkpoint = {"format": INIT_FORMAT, "identity": identity, "identity_sha256": json_digest(identity),
                  "model_config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict(), "initialization": initialization,
                  "completed_updates": 0, "completed_epochs": 0, "batch_in_epoch": 0,
                  "precision": "fp32", "weights_only_initialization": True,
                  "memory_state": "Rebuilt for each independent forward"}
    report.update(identity=identity, data_records=data_records, overlap_audit=overlap,
                  initialization=initialization,
                  checkpoint=save_checkpoint(args.output_dir / "init.pt", checkpoint),
                  expected_updates_per_epoch=200)
    verify_sources(tracked)
    report["status"] = "complete"


def operational(args, report, tracker):
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    if (initial.get("format") != INIT_FORMAT or initial["completed_updates"] != 0
            or initial["optimizer"]["state"] or initial["identity_sha256"] != json_digest(initial["identity"])):
        raise ValueError("Require this milestone's shared five-block weights/empty-Adam initialization")
    verify_sources(initial["identity"]["source_sha256"])
    if file_digest(args.preset) != initial["identity"]["preset_sha256"]:
        raise ValueError("Operational preset differs from the prepared initialization")
    report.update(setup(initial["initialization"]["training_seed"]))
    if not report["execution_contract"]["inductor_cache_directory"]:
        raise ValueError("Use the launcher or an explicit retained shared TORCHINDUCTOR_CACHE_DIR")
    tracked = sources([args.preset])
    snapshot_sources(args.output_dir, tracked)
    train, dev = (load_dataset(args.data_root, TASK, split) for split in ("train", "dev"))
    for name, data in (("train", train), ("dev", dev)):
        validate_data(data)
        if dataset_identity(data) != initial["identity"]["data"][name]:
            raise ValueError(f"Prepared {name} data identity changed")
    if len(train) != 12800 or len(dev) != 256:
        raise ValueError("Expected the frozen 12800-train/256-development data scope")
    precision = POLICIES[args.precision]
    model, construction = build_model("tiled", precision, checkpoint=initial, preset=args.preset)
    optimizer = optimizer_for(model)
    optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    scheduler.load_state_dict(initial["scheduler"])
    seed_all(initial["initialization"]["training_seed"], deterministic=True)
    batch_size = args.repeat_batch if args.mode == "repeat" else 64
    updates_per_epoch = math.ceil(len(train) / 64)
    identity = {"schema": FORMAT, "initial_checkpoint_sha256": file_digest(args.initial_checkpoint),
                "initial_identity_sha256": initial["identity_sha256"], "source_sha256": tracked,
                "model_config": construction["model_config"], "precision": precision,
                "execution_contract": report["execution_contract"], "initialization": initial["initialization"],
                "data": {"train": dataset_identity(train), "dev": dataset_identity(dev)},
                "physical_batch": batch_size, "shuffle_seed": SEEDS["shuffle"],
                "data_order": "fixed_first_examples" if args.mode == "repeat" else "shared_epoch_permutation",
                "accumulation": False, "optimizer": OPTIMIZER, "schedule": SCHEDULE,
                "updates_per_epoch": updates_per_epoch,
                "stop_updates": (args.repeat_steps if args.mode == "repeat" else
                                 args.warmup + args.steps if args.mode == "benchmark" else 100),
                "midpoint": 50 if args.mode in ("train", "resume") else None,
                "purpose": args.mode if args.mode in ("benchmark", "repeat") else "paired_training"}
    report.update(identity=identity, identity_sha256=json_digest(identity), construction=construction,
                  model_config=construction["model_config"], initialization=initial["initialization"],
                  precision=precision, supervision="Original MAD native labels; FP32 aligned masked CE; no target shift",
                  data_policy="Fresh retained native train/dev; confirmation examples remain separately held out",
                  effective_precision_at_initialization=effective_precision(model, optimizer, require_gradients=False))
    atomic_json(args.output_dir / "resolved-config.json", identity)
    if tracker:
        tracker.start({"evidence_class": "OPS", "mode": args.mode, "identity": identity})
        atomic_json(args.output_dir / "wandb-run.json", tracker.record)
        print(json.dumps({"wandb_run_url": tracker.record["run_url"]}), flush=True)

    history, development = [], {}
    completed_updates = completed_epochs = batch_in_epoch = 0
    permutations = {}

    def indices():
        if completed_epochs not in permutations:
            permutations.clear()
            permutations[completed_epochs] = epoch_indices(len(train), completed_epochs, SEEDS["shuffle"])
        return permutations[completed_epochs][batch_in_epoch * 64:(batch_in_epoch + 1) * 64]

    def next_hash():
        return train.take(indices()).sha256

    def snapshot():
        return {"format": FORMAT, "identity": identity, "identity_sha256": json_digest(identity),
                "model_config": construction["model_config"], "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(),
                "completed_updates": completed_updates, "completed_epochs": completed_epochs,
                "batch_in_epoch": batch_in_epoch, "next_batch_sha256": next_hash(),
                "history": history, "development": development, "initialization": initial["initialization"],
                "precision": precision, "memory_state": "Rebuilt for each independent forward",
                "initial_checkpoint": {"path": str(args.initial_checkpoint), "sha256": file_digest(args.initial_checkpoint)}}

    if args.mode == "resume":
        restored = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (restored.get("format") != FORMAT or restored["identity"] != identity
                or restored["identity_sha256"] != json_digest(identity) or restored["completed_updates"] != 50):
            raise ValueError("Midpoint config/source/runtime/data/precision identity mismatch")
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        restore_rng(restored["rng"])
        for name, current in (("model", model.state_dict()), ("optimizer", optimizer.state_dict()),
                              ("scheduler", scheduler.state_dict()), ("rng", rng_state())):
            if state_digest(current) != state_digest(restored[name]):
                raise AssertionError(f"{name} was not restored exactly")
        completed_updates, completed_epochs, batch_in_epoch = (restored[name] for name in
            ("completed_updates", "completed_epochs", "batch_in_epoch"))
        history, development = copy.deepcopy(restored["history"]), copy.deepcopy(restored["development"])
        if next_hash() != restored["next_batch_sha256"]:
            raise AssertionError("Resume data position differs")
        report["recovery"] = {"loaded_state_exact": True, "checkpoint": str(args.checkpoint),
                              "checkpoint_sha256": file_digest(args.checkpoint), "cold_process": True,
                              "warmup_updates": 0, "repeated_midpoint_evaluation": False}

    if args.mode == "benchmark":
        batch = train.take(indices())
        ids = torch.as_tensor(batch.input_ids, device="cuda")
        labels = torch.as_tensor(batch.labels, device="cuda")
        started = time.perf_counter()
        warmup = [update(model, optimizer, ids, labels, precision=precision, monitor=False)
                  for _ in range(args.warmup)]
        report.update(warmup=warmup, warmup_seconds_including_compilation=time.perf_counter() - started,
                      benchmark_data_sha256=batch.sha256,
                      compiler_after_warmup=compiler_audit(True),
                      startup_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      startup_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        measured = [update(model, optimizer, ids, labels, precision=precision, monitor=False)
                    for _ in range(args.steps)]
        report.update(measured=measured, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), compiler_after_measurement=compiler_audit(True))
        report["measurement_compilation"] = measured_compiler_audit(report["compiler_after_warmup"], report["compiler_after_measurement"])
        if not report["measurement_compilation"]["steady_compilation_free"]:
            raise RuntimeError("Measured updates compiled additional graphs")
        times = [row["seconds"] for row in measured]
        report["timing"] = {"mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times),
                            "input_tokens_per_second": args.steps * ids.numel() / sum(times),
                            "scored_answers_per_second": sum(row["native_targets"] for row in measured) / sum(times),
                            "scope": "Prepared GPU batch; zero_grad/forward/FP32 CE/backward/clipping/Adam; no monitors",
                            "memory_scope": "Steady updates after Adam initialization; reserved includes warmup pool"}
        if tracker:
            for index, row in enumerate(measured):
                tracker.log({"update": args.warmup + index + 1, **scalar_metrics(row, "benchmark")})
    elif args.mode == "repeat":
        batch = train.take(slice(0, batch_size))
        ids = torch.as_tensor(batch.input_ids, device="cuda")
        labels = torch.as_tensor(batch.labels, device="cuda")
        for index in range(args.repeat_steps):
            row = update(model, optimizer, ids, labels, precision=precision)
            row.update(update=index + 1, batch_sha256=batch.sha256)
            history.append(row)
            append_jsonl(args.output_dir / "learning-curve.jsonl", row)
            if tracker:
                tracker.log({"update": index + 1, **scalar_metrics(row, "train")})
        report.update(history=history, completed_updates=args.repeat_steps,
                      repeat_batch_sha256=batch.sha256,
                      learning_check={"initial_loss": history[0]["native_loss"], "final_loss": history[-1]["native_loss"],
                                      "loss_reduced": history[-1]["native_loss"] < history[0]["native_loss"]},
                      schedule_interpretation="Repeated-batch diagnostic; no completed native-data epoch; LR remains initial")
    else:
        report["checkpoints"] = {}
        if completed_updates == 0:
            development["0"] = evaluate(model, dev, precision=precision)
            report["checkpoints"]["init"] = save_checkpoint(args.output_dir / "init.pt", snapshot())
        if tracker:
            tracker.log({"update": completed_updates, **scalar_metrics(development[str(completed_updates)], "dev")})
        for _ in range(completed_updates, 100):
            batch_indices = indices()
            batch = train.take(batch_indices)
            ids = torch.as_tensor(batch.input_ids, device="cuda")
            labels = torch.as_tensor(batch.labels, device="cuda")
            answers = torch.as_tensor(batch.answer_labels, device="cuda")
            row = update(model, optimizer, ids, labels, answers, precision=precision)
            row.update(update=completed_updates + 1, epoch=completed_epochs + 1,
                       batch_in_epoch=batch_in_epoch, batch_sha256=batch.sha256,
                       indices_sha256=state_digest(batch_indices))
            completed_updates += 1
            batch_in_epoch += 1
            if batch_in_epoch == updates_per_epoch:
                completed_epochs += 1
                batch_in_epoch = 0
                scheduler.step()
            history.append(row)
            append_jsonl(args.output_dir / "learning-curve.jsonl", row)
            logged = {"update": completed_updates, **scalar_metrics(row, "train")}
            if completed_updates in (50, 100):
                development[str(completed_updates)] = evaluate(model, dev, precision=precision)
                verify_sources(tracked)
                report["checkpoints"][str(completed_updates)] = save_checkpoint(
                    args.output_dir / f"u{completed_updates:04d}.pt", snapshot())
                logged.update(scalar_metrics(development[str(completed_updates)], "dev"))
            if tracker:
                tracker.log(logged)
            if completed_updates % 10 == 0:
                print(json.dumps({"update": completed_updates, "native_loss": row["native_loss"], "precision": precision}), flush=True)
        report.update(history=history, development=development, completed_updates=completed_updates,
                      completed_epochs=completed_epochs, batch_in_epoch=batch_in_epoch,
                      training_seconds=sum(row["seconds"] for row in history),
                      clipping_count=sum(row["clipped"] for row in history))
        if args.reference_final:
            reference = torch.load(args.reference_final, map_location="cpu", weights_only=False)
            report["recovery_comparison"] = compare_resumed(reference, cpu_tree(snapshot()))
            atomic_json(args.output_dir / "recovery-comparison.json", report["recovery_comparison"])
            if not report["recovery_comparison"]["bitwise_state_and_nontiming_metrics_equal"]:
                raise AssertionError("Exact recovery failed; differences remain retained")
    report["final_precision"] = effective_precision(model, optimizer, require_gradients=True)
    report["compiler"] = compiler_audit(True)
    verify_sources(tracked)
    if sources([args.preset]) != tracked:
        raise RuntimeError("Source inventory changed during execution")
    report["status"] = "complete"


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    forbidden = [Path(".runtime") / name for name in ("stage-b", "r3-backward", "r3-bf16", "cdrm-naive", "r3-bf16-tiled-resolution")]
    if any(path.resolve() in output.parents for path in forbidden):
        raise ValueError("Prior retained experiment lineages are immutable")
    output.mkdir(parents=True)
    report = {"format": FORMAT, "evidence": "OPS-preparation" if args.mode == "prepare" else "OPS", "status": "running",
              "command": shlex.join([sys.executable, *sys.argv]),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    started = time.monotonic()
    tracker = None
    if args.mode != "prepare" and args.wandb_project:
        tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                                group=args.wandb_group, name=args.wandb_run_name,
                                output_dir=output, preserve_state=preserve_tracking_rng)
        report["wandb"] = tracker.record
    try:
        if args.mode == "prepare":
            prepare(args, report)
        else:
            operational(args, report, tracker)
        if tracker:
            keys = ("timing", "learning_check", "completed_updates", "completed_epochs", "training_seconds",
                    "clipping_count", "peak_allocated_bytes", "peak_reserved_bytes", "recovery_comparison")
            tracker.summary({"experiment_status": report["status"],
                             **scalar_metrics({key: report[key] for key in keys if key in report})})
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if tracker:
                tracker.finish(succeeded=report["status"] == "complete")
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(output / "report.json", report)


if __name__ == "__main__":
    main()
