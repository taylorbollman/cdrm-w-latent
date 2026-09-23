#!/usr/bin/env python3
"""Bounded native RT efficiency ablations: RoPE reuse and K/V-only writes."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
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
from cdrm.pretrained.resource_estimates import LossWork, estimate_training_resources, parameter_inventory
from scripts.olmo_f1_common import active_names, inference_names
from scripts.olmo_f3_graph_training import (
    changed_batch, build_model, build_optimizer, boundary_digests, state_health,
    compare_graph, loss_snapshot, timed, configure_determinism,
    require_container_gpu, validate_prepared_manifest, load_native_state_dict,
    load_native_tokenizer, backend_context, OnlineTracker,
)
from scripts.olmo_f3d_validate import new_plan, global_gradient_l2
from scripts.olmo_f3e_validate import comparison_metrics
from scripts.olmo_f4_resources import SOURCES as F4_SOURCES, selected_case, memory_snapshot

PROTOCOL = ROOT / "docs/reports/olmo-rt-efficiency/protocol.md"
SOURCES = tuple(sorted(set(F4_SOURCES) | {
    "scripts/olmo_rt_efficiency.py", "cdrm/pretrained/olmo_rope.py"}))
ARMS = {"control": (False, False), "rope": (True, False), "both": (True, True)}
COMPARISONS = {"control-rope": ("control", "rope"), "rope-both": ("rope", "both")}
BUDGETS = {"global_gradient_relative_l2": 1 / 64,
           "tensor_gradient_relative_l2": 1 / 32,
           "tensor_gradient_max_relative": 1 / 16,
           "loss_relative_l2": 1e-5,
           "output_relative_l2": 1 / 64,
           "output_max_relative": 1 / 16}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("correctness", "capacity"), required=True)
    parser.add_argument("--case", choices=("ordinary", "rt", "combined"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--comparison", choices=tuple(COMPARISONS))
    parser.add_argument("--batch-size", type=int, choices=(8, 64, 128), required=True)
    parser.add_argument("--supervision", choices=("half", "full"))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if args.stage == "correctness":
        args.supervision = args.supervision or "half"
        if args.batch_size != 8 or args.supervision != "half" or args.arm is not None or args.comparison is None:
            parser.error("Correctness requires B8/T512, half supervision and --comparison (not --arm)")
        if args.profile:
            parser.error("Profiling is separate from correctness")
        if args.case == "ordinary" and args.comparison != "control-rope":
            parser.error("Ordinary has no permanent RT writes: compare control-rope only")
    else:
        args.supervision = args.supervision or "full"
        if args.batch_size not in (64, 128) or args.arm is None or args.comparison is not None:
            parser.error("Capacity requires B64/128 and --arm (not --comparison)")
        if args.case == "ordinary" and args.arm == "both":
            parser.error("Ordinary has no permanent RT writes: use control or rope")
        if args.batch_size == 128 and args.case != "rt":
            parser.error("Optional B128 is bounded to RT")
        if args.profile and (args.case != "rt" or args.batch_size != 64 or args.arm == "rope"):
            parser.error("Untimed profiling is bounded to RT B64 control/both")
    return args


def batch_for(tokenizer, case, update, supervision):
    if supervision not in ("half", "full"):
        raise ValueError("Unknown supervision")
    batch = changed_batch(tokenizer, case, update)
    return replace(batch, ce_mask=batch.valid_mask.clone()) if supervision == "full" else batch


def set_arm(model, arm):
    """Execution switches are applied before preparing each immutable layout."""
    if arm not in ARMS:
        raise ValueError("Unknown efficiency arm")
    base = model.backbone.backbone
    base.reuse_rope, base.kv_only_writes = ARMS[arm]
    model.config = replace(model.config, ce_chunk_size=2048)
    if model.config.vocab_chunk_size != 128:
        raise ValueError("This protocol retains KL position chunks of 128")


def metric(candidate, reference):
    return {**comparison_metrics(candidate, reference),
            "bitwise_equal": torch.equal(candidate, reference)}


def comparison_passes(*, exact_required, ownership, finite, counts_equal,
                      losses, gradients, outputs):
    global_l2 = global_gradient_l2(gradients.values())
    all_rows = [*losses.values(), *gradients.values(), *outputs.values()]
    exact = bool(all_rows) and all(row["bitwise_equal"] for row in all_rows)
    passed = (ownership and finite and counts_equal and bool(losses) and bool(gradients)
              and bool(outputs) and (exact if exact_required else (
                  global_l2 <= BUDGETS["global_gradient_relative_l2"]
                  and all(row["relative_l2"] <= BUDGETS["tensor_gradient_relative_l2"]
                          and row["max_relative"] <= BUDGETS["tensor_gradient_max_relative"]
                          for row in gradients.values())
                  and all(row["relative_l2"] <= BUDGETS["loss_relative_l2"] for row in losses.values())
                  and all(row["relative_l2"] <= BUDGETS["output_relative_l2"]
                          and row["max_relative"] <= BUDGETS["output_max_relative"]
                          for row in outputs.values()))))
    return {"passed": passed, "all_bitwise_equal": exact,
            "global_gradient_relative_l2": global_l2}


def output_snapshot(plan):
    # This extra, untimed forward checks every pass's final hidden states.
    # The comparison backward below still executes the canonical training path.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        result = plan.forward_layout.forward(plan.batch.input_ids, mode=plan.mode)
    return {f"pass{index}": value.detach().clone()
            for index, value in enumerate(result.pass_hidden_states)}


def compare_arms(model, batch, mode, comparison):
    reference_arm, candidate_arm = COMPARISONS[comparison]
    set_arm(model, reference_arm)
    reference_plan = new_plan(model, batch, mode, "recompute")
    outputs = output_snapshot(reference_plan)
    reference = reference_plan.backward(replay=False)
    losses = loss_snapshot(reference)
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    counts = dict(reference_plan.counts)
    del reference_plan, reference
    model.zero_grad(set_to_none=True)
    set_arm(model, candidate_arm)
    plan = new_plan(model, batch, mode, "recompute")
    candidate_outputs = output_snapshot(plan)
    candidate = plan.backward(replay=False)
    candidate_losses = loss_snapshot(candidate)
    loss_checks = {name: metric(value, losses[name]) for name, value in candidate_losses.items()}
    output_checks = {name: metric(value, outputs[name]) for name, value in candidate_outputs.items()}
    candidate_names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    ownership = candidate_names == set(gradients) == set(plan.active_names)
    gradient_checks = {name: metric(parameter.grad, gradients[name])
                       for name, parameter in model.named_parameters()
                       if parameter.grad is not None and name in gradients}
    finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
                 if parameter.grad is not None)
    finite = finite and all(bool(torch.isfinite(value).all())
                            for value in (*candidate_losses.values(), *candidate_outputs.values()))
    counts_equal = counts == plan.counts
    exact_required = comparison == "control-rope"
    screen = comparison_passes(exact_required=exact_required, ownership=ownership, finite=finite,
        counts_equal=counts_equal, losses=loss_checks, gradients=gradient_checks, outputs=output_checks)
    check = {"name": comparison.replace("-", "_vs_"), **screen,
        "reference_arm": reference_arm, "candidate_arm": candidate_arm,
        "ownership_matches": ownership, "finite": finite, "counts_equal": counts_equal,
        "exact_required": exact_required, "budgets": dict(BUDGETS),
        "losses": loss_checks, "outputs": output_checks, "gradients": gradient_checks,
        "scope": "Same native checkpoint and BF16 mixed runtime; all-pass outputs, loss sums and raw gradients. No optimizer/clipping in this comparison."}
    return plan, check


def track_optimizer_steps(optimizer, report):
    """Count successful optimizer.step calls even if subsequent scheduling fails."""
    def completed(_optimizer, _args, _kwargs):
        report["physical_optimizer_updates"] += 1
    return optimizer.register_step_post_hook(completed)


def complete_update_parity(plan, tokenizer, case, *, report, persist):
    """Three eager and three graph updates, retaining per-update progress."""
    model = plan.model
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer, scheduler = build_optimizer(model)
    initial_scheduler = copy.deepcopy(scheduler.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    hook = track_optimizer_steps(optimizer, report)
    outcomes = []
    report["update_parity_progress"] = []
    try:
        for replay in (False, True):
            if replay:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        parameter.copy_(initial[name])
                if any(not torch.equal(value.detach().cpu(), initial[name])
                       for name, value in model.named_buffers()):
                    raise AssertionError("A fixed buffer changed during eager updates")
                optimizer.load_state_dict(initial_optimizer)
                scheduler.load_state_dict(initial_scheduler)
            counters = TrainingCounters()
            records = []
            for step in range(3):
                record = plan.optimizer_step(optimizer, changed_batch(tokenizer, case, step + 5),
                    replay=replay, scheduler=scheduler, counters=counters)
                records.append(record)
                report["update_parity_progress"].append({"replay": replay, "record": record})
                persist()
            outcomes.append({"replay": replay, "metrics": records,
                "boundary": boundary_digests(model, optimizer, scheduler, counters),
                "health": state_health(model, optimizer)})
    finally:
        hook.remove()
    changed = any(not torch.equal(value.detach().cpu(), initial[name])
                  for name, value in model.state_dict().items() if name.endswith("ff_out.weight"))
    metrics_exact = outcomes[0]["metrics"] == outcomes[1]["metrics"]
    boundary_exact = outcomes[0]["boundary"] == outcomes[1]["boundary"]
    return {"name": "complete_adamw_update_parity", "updates_per_arm": 3,
        "physical_optimizer_updates": 6, "metrics_exact": metrics_exact,
        "model_optimizer_scheduler_counters_exact": boundary_exact, "weights_changed": changed,
        "arms": outcomes, "passed": metrics_exact and boundary_exact and changed
        and all(outcome["health"]["passed"] for outcome in outcomes)}


def operator_profile(plan, directory):
    """Separate captured forward/loss/backward profile; never a timing sample."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
        plan.backward(replay=True)
        torch.cuda.synchronize()
    path = directory / "operator-trace.json"
    profile.export_chrome_trace(str(path))
    compressed = path.with_suffix(".json.gz")
    with path.open("rb") as source, gzip.open(compressed, "wb") as target:
        shutil.copyfileobj(source, target)
    path.unlink()
    events = [event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    totals = {}
    for event in events:
        row = totals.setdefault(event.name, {"calls": 0, "self_device_us": 0.0})
        row["calls"] += 1
        row["self_device_us"] += event.self_device_time_total
    return {"scope": "One untimed graph forward/loss/backward after timing; no optimizer. Summed device event time is not complete-update wall time.",
        "device_event_count": len(events), "device_kernels": totals,
        "trace_file": compressed.name, "trace_bytes": compressed.stat().st_size,
        "trace_sha256": sha256_file(compressed)}


def resource_card(plan, case, optimizer=None):
    base = plan.model.backbone.backbone
    work = LossWork(ce_targets=plan.counts["ce"], latent_pairs=plan.counts["latent"],
        kl_triples=plan.counts["kl"], predictor_positions=plan.loss_layout.needed_source_indices.numel())
    estimate = estimate_training_resources(base.config, batch_size=case.batch_size,
        sequence_length=case.length, mode=plan.mode,
        nextlat=plan.model.config if case.nextlat else None, loss_work=work,
        ordinary_checkpointing=True, backward_memory="recompute", kv_only_writes=base.kv_only_writes)
    return {"analytic_matrix_work": estimate.to_dict(), "loss_work": asdict(work),
        "observed_parameters": parameter_inventory(plan.model, optimizer=optimizer,
            executed_names=active_names(plan.model, plan.mode), inference_names=inference_names(plan.model, case)),
        "scope": "Matrix arithmetic estimate, not measured hardware FLOPs; RoPE table reuse reduces elementwise work outside this estimate."}


def main(argv=None):
    args = parse_args(argv)
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
    configuration = {**{key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "case_specification": asdict(case), "mode": asdict(case.mode()), "precision": "bf16_mixed",
        "ordinary_attention": "deterministic PyTorch Flash SDPA", "ordinary_checkpointing": True,
        "rt_forward_backward": "triton, cast reuse, recompute", "ce_chunk_size": 2048, "kl_chunk_size": 128,
        "autocast_weight_cache": False, "tf32": False, "world_size": 1, "accumulation": 1,
        "capture_warmup_backwards": 10, "physical_batch": args.batch_size, "seed": 20260922, "length": 512,
        "preparation_updates": 3 if args.stage == "capacity" else 0,
        "timed_updates": 5 if args.stage == "capacity" else 0,
        "backward_timing_samples": 3 if args.stage == "capacity" else 0,
        "arm_switches": {name: {"reuse_rope": rope, "kv_only_writes": kv} for name, (rope, kv) in ARMS.items()}}
    report = {"schema": "olmo-rt-efficiency-v1", "status": "running", "stage": "load",
        "configuration": configuration, "runtime": runtime, "determinism": determinism,
        "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
        "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "physical_optimizer_updates": 0}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-rt-efficiency",
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
        print({"check": check["name"], "passed": check["passed"],
               "all_bitwise_equal": check.get("all_bitwise_equal")}, flush=True)
        if not check["passed"]:
            raise AssertionError(check["name"])

    def exact_graph(plan, name, replays=1):
        check = compare_graph(plan, name, replays=replays)
        check["passed"] = check["passed"] and check["all_bitwise_equal"]
        publish(check)

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        save()
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        model = build_model(state, case)
        report["parameters"] = {"active": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "resident": sum(parameter.numel() for parameter in model.parameters())}
        batch = batch_for(tokenizer, case, 0, args.supervision)
        with backend_context("flash"):
            if args.stage == "correctness":
                save("same_state_efficiency_comparison")
                plan, check = compare_arms(model, batch, case.mode(), args.comparison)
                publish(check)
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
                report["capture_seconds"] = time.perf_counter() - began
                exact_graph(plan, "candidate_initial_graph")
                plan.load_batch(changed_batch(tokenizer, case, 1))
                exact_graph(plan, "candidate_changed_tokens_overwrite", replays=2)
                save("complete_update_parity")
                publish(complete_update_parity(plan, tokenizer, case, report=report, persist=save))
                plan.load_batch(changed_batch(tokenizer, case, 2))
                exact_graph(plan, "candidate_changed_weights")
                report["memory"] = memory_snapshot()
                report["resources"] = resource_card(plan, case)
            else:
                set_arm(model, args.arm)
                plan = new_plan(model, batch, case.mode(), "recompute")
                optimizer, scheduler = build_optimizer(model)
                hook = track_optimizer_steps(optimizer, report)
                counters = TrainingCounters()
                probe = model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096].clone()
                torch.cuda.reset_peak_memory_stats()
                preparation = []
                save("preparation_updates")
                for index in range(3):
                    preparation.append(plan.optimizer_step(optimizer,
                        batch_for(tokenizer, case, index, args.supervision), scheduler=scheduler, counters=counters))
                    report["preparation_records"] = preparation
                    save()
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
                report["capture_seconds"] = time.perf_counter() - began
                report["setup_memory"] = memory_snapshot()
                torch.cuda.reset_peak_memory_stats()
                records = []
                report["timed_records"] = records
                batches = [batch_for(tokenizer, case, index + 3, args.supervision) for index in range(5)]

                def complete():
                    records.append(plan.optimizer_step(optimizer, batches[len(records)], replay=True,
                        scheduler=scheduler, counters=counters))

                save("timing")
                report["full_update"] = timed(complete, 5)
                report["steady_memory"] = memory_snapshot()
                report["input_tokens_per_second"] = plan.input_tokens / report["full_update"]["median_wall_seconds"]
                report["ce_targets_per_second"] = plan.counts["ce"] / report["full_update"]["median_wall_seconds"]
                save("backward_timing")
                report["forward_loss_backward"] = timed(lambda: plan.backward(replay=True), 3)
                report["forward_loss_backward_tokens_per_second"] = plan.input_tokens / report["forward_loss_backward"]["median_wall_seconds"]
                report["health"] = state_health(model, optimizer)
                report["resources"] = resource_card(plan, case, optimizer)
                changed = not torch.equal(probe, model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096])
                finite_gradients = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                                       for parameter in model.parameters() if parameter.requires_grad)
                publish({"name": "finite_complete_updates", "passed": report["health"]["passed"] and changed and finite_gradients,
                    "trainable_weight_changed": changed, "finite_participating_gradients": finite_gradients})
                hook.remove()
                tracker.log({"benchmark/input_tokens_per_second": report["input_tokens_per_second"],
                    "benchmark/ce_targets_per_second": report["ce_targets_per_second"],
                    "benchmark/forward_loss_backward_tokens_per_second": report["forward_loss_backward_tokens_per_second"],
                    "benchmark/setup_peak_reserved_gib": report["setup_memory"]["peak_reserved_gib"],
                    "benchmark/steady_peak_allocated_gib": report["steady_memory"]["peak_allocated_gib"]}, step=2)
                if args.profile:
                    save("untimed_operator_profile")
                    report["profile"] = operator_profile(plan, args.output_dir)
            report["nextlat_config"] = model.config.to_dict()
            report["counts"] = dict(plan.counts)
            report["input_tokens"] = plan.input_tokens
            report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
                "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
        if report["source_hashes"] != {path: sha256_file(ROOT / path) for path in SOURCES}:
            raise AssertionError("Runtime source changed during the run")
        if report["protocol_sha256"] != sha256_file(PROTOCOL):
            raise AssertionError("Protocol changed during the run")
        report["status"] = "passed"
        tracker.summary({"result_status": "passed", "physical_optimizer_updates": report["physical_optimizer_updates"]})
        save("complete")
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save()
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()


if __name__ == "__main__":
    main()
