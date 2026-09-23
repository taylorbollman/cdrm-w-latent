#!/usr/bin/env python3
"""Bounded CE-only chunk integration and native pretrained OLMo throughput."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import TrainingCounters
from scripts.olmo_f3_graph_training import (
    changed_batch, build_model, build_optimizer, state_health, compare_graph,
    full_update_parity, loss_snapshot, timed, configure_determinism,
    require_container_gpu, validate_prepared_manifest, load_native_state_dict,
    load_native_tokenizer, backend_context, OnlineTracker,
)
from scripts.olmo_f3d_validate import new_plan, global_gradient_l2
from scripts.olmo_f3e_validate import comparison_metrics
from scripts.olmo_f4_resources import SOURCES as F4_SOURCES, selected_case, memory_snapshot

PROTOCOL = ROOT / "docs/reports/olmo-ce-integration/protocol.md"
SOURCES = tuple(sorted(set(F4_SOURCES) | {"scripts/olmo_ce_integration.py"}))


def batch_for(tokenizer, case, update, supervision):
    batch = changed_batch(tokenizer, case, update)
    return replace(batch, ce_mask=batch.valid_mask.clone()) if supervision == "full" else batch


def compare_chunks(model, batch, mode):
    """Same weights and RT implementation: change CE chunk128 to2048 only."""
    assert model.config.vocab_chunk_size == 128 and model.config.ce_chunk_size is None
    reference_plan = new_plan(model, batch, mode, "recompute")
    reference = reference_plan.backward(replay=False)
    losses = loss_snapshot(reference)
    gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    counts = dict(reference_plan.counts)
    del reference_plan, reference
    model.zero_grad(set_to_none=True)
    model.config = replace(model.config, ce_chunk_size=2048)
    plan = new_plan(model, batch, mode, "recompute")
    candidate = plan.backward(replay=False)
    actual_losses = loss_snapshot(candidate)
    loss_checks = {n: comparison_metrics(v, losses[n]) for n, v in actual_losses.items()}
    grad_checks = {n: comparison_metrics(p.grad, gradients[n])
                   for n, p in model.named_parameters() if p.grad is not None}
    ownership = set(grad_checks) == set(gradients) == set(plan.active_names)
    auxiliary_exact = all(torch.equal(v, losses[n]) for n, v in actual_losses.items() if not n.endswith("/ce"))
    global_l2 = global_gradient_l2(grad_checks.values())
    finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    passed = (ownership and finite and auxiliary_exact and counts == plan.counts
        and global_l2 <= 1/64
        and all(r["relative_l2"] <= 1/32 and r["max_relative"] <= 1/16 for r in grad_checks.values())
        and all(r["relative_l2"] <= 1e-5 for r in loss_checks.values()))
    return plan, {"name": "ce2048_vs_ce128", "passed": passed,
        "ownership_matches": ownership, "finite_gradients": finite,
        "auxiliary_losses_bitwise_equal": auxiliary_exact, "counts_equal": counts == plan.counts,
        "losses": loss_checks, "gradients": grad_checks, "global_gradient_relative_l2": global_l2,
        "scope": "Same pretrained weights, BF16 mixed; same optimized RT implementation in both arms"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("correctness", "capacity"), required=True)
    parser.add_argument("--case", choices=("ordinary", "nextlat", "combined"), default="ordinary")
    parser.add_argument("--batch-size", type=int, choices=(8, 64), required=True)
    parser.add_argument("--ce-chunk", type=int, choices=(128, 2048), default=2048)
    parser.add_argument("--supervision", choices=("half", "full"), default="half")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if args.stage == "correctness" and (args.batch_size != 8 or args.ce_chunk != 2048 or args.supervision != "half"):
        parser.error("Correctness uses B8/T512, half supervision, CE128 vs2048")
    if args.stage == "capacity" and (args.batch_size != 64 or args.case != "ordinary"):
        parser.error("Capacity is bounded to the original ordinary B64/T512 model")
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        destination = args.output_dir / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    case = selected_case(args.case, batch=args.batch_size, length=512)
    configuration = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "case_specification": asdict(case), "mode": asdict(case.mode()), "precision": "bf16_mixed",
        "ordinary_attention": "deterministic PyTorch Flash SDPA", "ordinary_checkpointing": True,
        "rt_forward_backward": "triton, cast reuse, recompute", "kl_chunk_size": 128,
        "autocast_weight_cache": False, "tf32": False, "world_size": 1, "accumulation": 1,
        "capture_warmup_backwards": 10, "physical_batch": args.batch_size,
        "seed": 20260922, "length": 512}
    report = {"schema": "olmo-ce-integration-v1", "status": "running", "stage": "load",
        "configuration": configuration, "runtime": runtime, "determinism": determinism,
        "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
        "source_hashes": {p: sha256_file(ROOT / p) for p in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "physical_optimizer_updates": 0}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-ce-integration",
        name=args.output_dir.name, output_dir=args.output_dir)

    def save(stage=None):
        if stage is not None:
            report["stage"] = stage
            print({"stage": stage, "utc": datetime.now(timezone.utc).isoformat()}, flush=True)
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)

    def publish(check):
        report["checks"].append(check)
        save()
        tracker.log({"correctness/passed": int(check["passed"])}, step=len(report["checks"]))
        print({"check": check["name"], "passed": check["passed"]}, flush=True)
        if not check["passed"]:
            raise AssertionError(check["name"])

    def exact_graph(plan, name, replays=1):
        row = compare_graph(plan, name, replays=replays)
        row["passed"] = row["passed"] and row["all_bitwise_equal"]
        publish(row)

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        save()
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        model = build_model(state, case)
        report["parameters"] = {"active": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "resident": sum(p.numel() for p in model.parameters())}
        batch = batch_for(tokenizer, case, 0, args.supervision)
        with backend_context("flash"):
            if args.stage == "correctness":
                save("ce_chunk_gradient_comparison")
                plan, check = compare_chunks(model, batch, case.mode())
                publish(check)
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
                report["capture_seconds"] = time.perf_counter() - began
                exact_graph(plan, "candidate_initial_graph")
                plan.load_batch(changed_batch(tokenizer, case, 1))
                exact_graph(plan, "candidate_changed_tokens_overwrite", replays=2)
                save("complete_update_parity")
                check = full_update_parity(plan, tokenizer, case, updates=3)
                report["physical_optimizer_updates"] = check["physical_optimizer_updates"]
                publish(check)
                plan.load_batch(changed_batch(tokenizer, case, 2))
                exact_graph(plan, "candidate_changed_weights")
                report["memory"] = memory_snapshot()
            else:
                model.config = replace(model.config, ce_chunk_size=args.ce_chunk)
                plan = new_plan(model, batch, case.mode(), "recompute")
                optimizer, scheduler = build_optimizer(model)
                counters = TrainingCounters()
                probe = model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096].clone()
                torch.cuda.reset_peak_memory_stats()
                preparation = []
                save("preparation_updates")
                for i in range(3):
                    preparation.append(plan.optimizer_step(optimizer, batch_for(tokenizer, case, i, args.supervision),
                        scheduler=scheduler, counters=counters))
                    report["physical_optimizer_updates"] = counters.optimizer_updates
                    report["preparation_records"] = preparation
                    save()
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
                report["capture_seconds"] = time.perf_counter() - began
                report["setup_memory"] = memory_snapshot()
                torch.cuda.reset_peak_memory_stats()
                records = []
                batches = [batch_for(tokenizer, case, i+3, args.supervision) for i in range(3)]

                def complete():
                    records.append(plan.optimizer_step(optimizer, batches[len(records)], replay=True,
                        scheduler=scheduler, counters=counters))
                    report["physical_optimizer_updates"] = counters.optimizer_updates

                save("timing")
                report["full_update"] = timed(complete, 3)
                report["timed_records"] = records
                report["steady_memory"] = memory_snapshot()
                report["input_tokens_per_second"] = plan.input_tokens / report["full_update"]["median_wall_seconds"]
                report["ce_targets_per_second"] = plan.counts["ce"] / report["full_update"]["median_wall_seconds"]
                save("backward_timing")
                report["forward_loss_backward"] = timed(lambda: plan.backward(replay=True), 3)
                report["health"] = state_health(model, optimizer)
                changed = not torch.equal(probe, model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096])
                finite_grads = all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                   for p in model.parameters() if p.requires_grad)
                publish({"name": "finite_complete_updates", "passed": report["health"]["passed"] and changed and finite_grads,
                         "trainable_weight_changed": changed, "finite_participating_gradients": finite_grads})
                tracker.log({"benchmark/input_tokens_per_second": report["input_tokens_per_second"],
                    "benchmark/ce_targets_per_second": report["ce_targets_per_second"],
                    "benchmark/setup_peak_reserved_gib": report["setup_memory"]["peak_reserved_gib"],
                    "benchmark/steady_peak_allocated_gib": report["steady_memory"]["peak_allocated_gib"]}, step=2)
            report["nextlat_config"] = model.config.to_dict()
            report["counts"] = dict(plan.counts)
            report["input_tokens"] = plan.input_tokens
            report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
                "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
        assert report["source_hashes"] == {p: sha256_file(ROOT / p) for p in SOURCES}
        assert report["protocol_sha256"] == sha256_file(PROTOCOL)
        report["status"] = "passed"
        tracker.summary({"result_status": "passed", "physical_optimizer_updates": report["physical_optimizer_updates"]})
        save("complete")
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save()
        raise
    finally:
        tracker.finish(succeeded=report["status"] == "passed")
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
