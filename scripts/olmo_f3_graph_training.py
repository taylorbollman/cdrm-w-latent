#!/usr/bin/env python3
"""Bounded canonical static-layout training checks and graph capacity on OLMo."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
from pathlib import Path
import statistics
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import LMTrainingConfig, TrainingCounters
from cdrm.pretrained.olmo_artifacts import (load_native_state_dict,
    load_native_tokenizer, validate_prepared_manifest)
from cdrm.pretrained.static_training import StaticFBTTraining
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_validation import require_container_gpu
from scripts.olmo_lm_common import fixture
from scripts.olmo_f1_common import (build_model, build_optimizer, boundary_digests,
    state_health, active_names, inference_names)
from scripts.olmo_f1_observe import parameter_accounting
from scripts.olmo_f2_health_capacity import case_for, SOURCE_FILES as F2_SOURCES
from scripts.olmo_f2_graph_probe import tensor_comparison
from scripts.olmo_f2_graph_backend_probe import configure_determinism, backend_context

SOURCE_FILES = tuple(sorted(set(F2_SOURCES) | {
    "scripts/olmo_f3_graph_training.py", "scripts/olmo_f2_graph_probe.py",
    "scripts/olmo_f2_graph_backend_probe.py", "cdrm/pretrained/static_training.py",
    "cdrm/pretrained/static_nextlat.py", "cdrm/pretrained/olmo_static.py"}))
PROTOCOL = ROOT/"docs/reports/olmo1b-f3/protocol.md"


def changed_batch(tokenizer, case, update):
    # Full independent documents, real-text token rotations, fixed unequal CE/
    # latent/KL supervision. Padding is a separate tiny/explicit-math scope.
    batch = fixture(tokenizer, batch_size=case.batch_size, length=case.length,
                    padded=False, device="cpu")
    for row in range(case.batch_size):
        batch.input_ids[row, :-1] = batch.input_ids[row, :-1].roll(3*update + row)
    return batch


def loss_snapshot(result):
    return {f"pass{p}/{term}": loss.sums[term].detach().clone()
        for p, loss in enumerate(result.pass_losses) for term in ("ce", "latent", "kl")}


def compare_graph(plan, name, *, replays=1):
    expected = plan.backward(replay=False)
    losses = loss_snapshot(expected)
    gradients = {n: p.grad.detach().clone() for n, p in plan.model.named_parameters() if p.grad is not None}
    for _ in range(replays):
        actual = plan.backward(replay=True)
    torch.cuda.synchronize()
    loss_checks = {n: tensor_comparison(v, losses[n]) for n, v in loss_snapshot(actual).items()}
    gradient_checks = {n: tensor_comparison(p.grad, gradients[n]) for n, p in plan.model.named_parameters()
                       if p.grad is not None}
    ownership = set(gradient_checks) == set(plan.active_names) == active_names(plan.model, plan.mode)
    rows = [*loss_checks.values(), *gradient_checks.values()]
    return {"name": name, "losses": loss_checks, "gradients": gradient_checks,
        "ownership_matches": ownership, "all_bitwise_equal": all(x["bitwise_equal"] for x in rows),
        "passed": ownership and all(x["passed"] for x in rows)}


def trace_dispatch(plan):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        plan.backward(replay=False)
    return sorted({event.key for event in profile.key_averages()
                   if "scaled_dot_product" in event.key or "flash_attention" in event.key})


def new_plan(state, tokenizer, case, args):
    model = build_model(state, case)
    model.backbone.backbone.ordinary_activation_checkpointing = args.checkpointing == "on"
    batch = changed_batch(tokenizer, case, 0)
    plan = StaticFBTTraining(model, batch, mode=case.mode(),
                           config=LMTrainingConfig(precision="bf16_mixed"))
    return model, plan


def full_update_parity(plan, tokenizer, case, *, updates=3):
    """Restore weights in place; the graph must reread all changed parameters."""
    model = plan.model
    initial = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    optimizer, scheduler = build_optimizer(model)
    initial_scheduler = copy.deepcopy(scheduler.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    outcomes = []
    for replay in (False, True):
        if replay:
            # Static buffers are immutable, even if a state-dict copy would
            # write identical bytes. Restore only optimizer-owned parameters.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(initial[name])
            if any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_buffers()):
                raise AssertionError("A fixed buffer changed during the eager update arm")
            optimizer.load_state_dict(initial_optimizer)
            scheduler.load_state_dict(initial_scheduler)
        counters = TrainingCounters()
        metrics = [plan.optimizer_step(optimizer, changed_batch(tokenizer, case, step+5),
                        replay=replay, scheduler=scheduler, counters=counters) for step in range(updates)]
        outcomes.append({"replay": replay, "metrics": metrics,
            "boundary": boundary_digests(model, optimizer, scheduler, counters),
            "health": state_health(model, optimizer)})
    weights_changed = any(not torch.equal(v.detach().cpu(), initial[n]) for n, v in model.state_dict().items()
                          if n.endswith("ff_out.weight"))
    exact_metrics = outcomes[0]["metrics"] == outcomes[1]["metrics"]
    exact_boundary = outcomes[0]["boundary"] == outcomes[1]["boundary"]
    return {"name": "complete_adamw_update_parity", "updates_per_arm": updates,
        "physical_optimizer_updates": 2*updates, "metrics_exact": exact_metrics,
        "model_optimizer_scheduler_counters_exact": exact_boundary,
        "weights_changed": weights_changed, "arms": outcomes,
        "passed": exact_metrics and exact_boundary and weights_changed
                  and all(o["health"]["passed"] for o in outcomes)}


def correctness(state, tokenizer, case, args, report, publish):
    model, plan = new_plan(state, tokenizer, case, args)
    report["stage"] = "capture"
    started = time.perf_counter()
    plan.capture(warmup=args.warmup)
    report["capture_seconds"] = time.perf_counter()-started
    report["capture_succeeded"] = True
    report["stage"] = "equivalence"
    for update, name, repeats in ((0, "original_tokens", 1), (1, "changed_tokens", 1),
                                  (2, "repeated_replay_overwrites_gradients", 2)):
        plan.load_batch(changed_batch(tokenizer, case, update))
        publish(compare_graph(plan, name, replays=repeats))
    report["stage"] = "complete_update_parity"
    publish(full_update_parity(plan, tokenizer, case, updates=args.updates))
    plan.load_batch(changed_batch(tokenizer, case, 19))
    publish(compare_graph(plan, "changed_weights_and_tokens"))
    report["stage"] = "dispatch_trace"
    report["observed_attention_operators"] = trace_dispatch(plan)
    report["parameters"] = parameter_accounting(model, active_parameter_names=set(plan.active_names),
                                                inference_parameter_names=inference_names(model, case))
    report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
        "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
    report["memory"] = {"peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}


def timed(function, repeats):
    wall, device = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter(); start.record()
        function(); end.record(); end.synchronize()
        wall.append(time.perf_counter()-began)
        device.append(start.elapsed_time(end)/1000)
    return {"wall_seconds": wall, "cuda_seconds": device,
            "median_wall_seconds": statistics.median(wall), "median_cuda_seconds": statistics.median(device)}


def capacity_arm(state, tokenizer, case, args, *, replay, report):
    report["stage"] = "capacity_graph_prepare" if replay else "capacity_eager_prepare"
    model, plan = new_plan(state, tokenizer, case, args)
    optimizer, scheduler = build_optimizer(model)
    counters = TrainingCounters()
    batches = [changed_batch(tokenizer, case, i) for i in range(3+args.repeats)]
    torch.cuda.reset_peak_memory_stats()
    report["stage"] = "capacity_graph_optimizer_warmup" if replay else "capacity_eager_optimizer_warmup"
    # Three real preparation updates instantiate Adam state before capture;
    # both arms start timing from corresponding changed weights and moments.
    for step in range(3):
        plan.optimizer_step(optimizer, batches[step], scheduler=scheduler, counters=counters)
    capture_seconds = None
    report["stage"] = "capacity_capture" if replay else "capacity_eager_warmup"
    if replay:
        started = time.perf_counter(); plan.capture(warmup=args.warmup)
        capture_seconds = time.perf_counter()-started
        report["capture_succeeded"] = True
    else:
        for _ in range(args.warmup):
            plan.backward(replay=False)
    torch.cuda.synchronize()
    capture_memory = {"allocated_gib": torch.cuda.memory_allocated()/2**30,
        "reserved_gib": torch.cuda.memory_reserved()/2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
    report["stage"] = "capacity_timing"
    records = []
    index = 3
    def complete():
        nonlocal index
        records.append(plan.optimizer_step(optimizer, batches[index], replay=replay,
                                          scheduler=scheduler, counters=counters))
        index += 1
    full = timed(complete, args.repeats)
    region = timed(lambda: plan.backward(replay=replay), args.repeats)
    peak = torch.cuda.max_memory_allocated()/2**30
    row = {"case": case.name, "batch_size": case.batch_size, "length": case.length,
        "replay": replay, "checkpointing": args.checkpointing == "on", "capture_seconds": capture_seconds,
        "warmup_updates": 3, "timed_updates": args.repeats,
        "gradient_initialization_backward": 1,
        "backward_only_warmup": args.warmup, "graph_capture_backward": int(replay),
        "input_tokens_per_update": plan.input_tokens, "counts": plan.counts,
        "full_step": {**full, "valid_input_tokens_per_second": plan.input_tokens/full["median_wall_seconds"]},
        "forward_loss_backward": region, "capture_memory": capture_memory,
        "peak_allocated_gib": peak, "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
        "within_comfortable_budget": peak <= args.comfortable_gib,
        "records": records, "post_timing_health": state_health(model, optimizer),
        "parameters": parameter_accounting(model, optimizer=optimizer, active_parameter_names=set(plan.active_names),
                                            inference_parameter_names=inference_names(model, case)),
        "full_step_scope": "Validated input copy + forward/loss/backward or graph replay + clip + AdamW + scheduler; one physical batch",
        "region_scope": "Prepared forward/loss/backward or graph replay; no input copies, clip or optimizer; host contract checks included"}
    row["passed"] = row["post_timing_health"]["passed"]
    del plan, model, optimizer, scheduler
    gc.collect(); torch.cuda.empty_cache()
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("correctness", "capacity"), default="correctness")
    parser.add_argument("--case", choices=("rt", "combined", "combined-k3"), default="combined")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--checkpointing", choices=("off", "on"), default="on")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--comfortable-gib", type=float, default=65)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 128 or args.length not in (32, 128, 512):
        parser.error("Bounded batches1–128 and lengths32/128/512 only")
    if args.warmup < 10 or args.repeats < 3 or not 1 <= args.updates <= 3 or not 20 <= args.comfortable_gib <= 65:
        parser.error("Require warmup>=10, repeats>=3, updates1–3, memory budget20–65GiB")
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4); torch.manual_seed(20260922)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update(precision="bf16_mixed", ordinary_sdpa_backend="flash", **determinism,
        autocast_weight_cache=False, gradient_accumulation=1, rt_layers=[0])
    report = {"schema": "olmo-f3-graph-training-v1", "status": "running", "stage": "load",
        "configuration": configuration, "runtime": {k: str(v) for k, v in runtime.items()},
        "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "rows": [],
        "source_hashes": {p: sha256_file(ROOT/p) for p in SOURCE_FILES},
        "protocol_sha256": sha256_file(PROTOCOL), "capture_succeeded": False,
        "limitations": ["RT selects layer0 only; no RT Flash/CuTE kernel", "Fixed full-document unpadded layout",
            "Input validation, clipping, AdamW and scheduler outside CUDA graphs", "No accumulation, cache, distributed or quality claim",
            "Static all-valid masks lower to implicit causal attention; canonical semantic parity tested separately"]}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f3-graph-training", name="olmo-1b-f3-"+args.output_dir.name)
    def persist():
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", report)
    def publish(check):
        report["checks"].append(check); persist()
        tracker.log({"correctness/passed": int(check["passed"]),
            "correctness/check": len(report["checks"])}, step=len(report["checks"]))
        print({"check": check["name"], "passed": check["passed"],
               "all_bitwise_equal": check.get("all_bitwise_equal")}, flush=True)
        if not check["passed"]:
            raise AssertionError("F3 check failed: "+check["name"])
    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": manifest["checkpoint"]})
        persist(); print({"wandb": tracker.record["run_url"]}, flush=True)
        state, tokenizer = load_native_state_dict(args.artifacts), load_native_tokenizer(args.artifacts)
        case = case_for(args.case, batch=args.batch_size, length=args.length)
        with backend_context("flash"):
            if args.stage == "correctness":
                correctness(state, tokenizer, case, args, report, publish)
            else:
                for replay in (False, True):
                    row = capacity_arm(state, tokenizer, case, args, replay=replay, report=report)
                    report["rows"].append(row); persist()
                    prefix = "graph" if replay else "eager"
                    tracker.log({f"capacity/{prefix}/input_tokens_per_second": row["full_step"]["valid_input_tokens_per_second"],
                        f"capacity/{prefix}/peak_allocated_gib": row["peak_allocated_gib"]}, step=len(report["checks"])+1)
                    publish({"name": prefix+"_complete_updates_finite", "passed": row["passed"]})
                    print({"capacity": prefix, "batch": args.batch_size,
                        "input_tokens_per_second": row["full_step"]["valid_input_tokens_per_second"],
                        "peak_allocated_gib": row["peak_allocated_gib"]}, flush=True)
                    if not row["within_comfortable_budget"]:
                        report["stopped_at_memory_budget"] = True
                        break
        if report["source_hashes"] != {p: sha256_file(ROOT/p) for p in SOURCE_FILES}:
            raise ValueError("Runtime sources changed during execution")
        report.update(status="passed", stage="complete")
    except Exception as error:
        report.update(status="capture_blocked" if "capture" in report["stage"] else "failed",
                      error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        try:
            tracker.summary({"f3/status": report["status"]})
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir/"report.json", report)
    print({"status": report["status"], "report": str(args.output_dir/"report.json")}, flush=True)


if __name__ == "__main__":
    main()
