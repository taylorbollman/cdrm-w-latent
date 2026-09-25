#!/usr/bin/env python3
"""Native RT integration and physical-batch capacity with phase memory accounting."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import gzip
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import TrainingCounters, build_adamw, build_warmup_scheduler
from cdrm.pretrained.resource_estimates import LossWork, estimate_training_resources, parameter_inventory
from scripts.olmo_f1_common import active_names, inference_names
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_f3_graph_training import (
    changed_batch, build_model, build_optimizer, state_health, compare_graph,
    loss_snapshot, timed, configure_determinism, require_container_gpu,
    validate_prepared_manifest, load_native_state_dict, load_native_tokenizer,
    backend_context, OnlineTracker,
)
from scripts.olmo_f3d_validate import new_plan
from scripts.olmo_large_batch_validation import (snapshot_eager_cpu, replay_compare_cpu,
    snapshot_replay_cpu, release_graph_then_compare_eager_cpu)
from scripts.olmo_f4_resources import selected_case, memory_snapshot
from scripts.olmo_rt_efficiency import (
    SOURCES as RT_SOURCES, BUDGETS, metric, comparison_passes, output_snapshot,
    complete_update_parity, track_optimizer_steps, operator_profile, resource_card,
)

PROTOCOL = ROOT / "docs/reports/olmo-rt-large-batch/protocol.md"
# Include actual runtime dependencies, including private helper imports. The
# package-wide superset avoids silently dropping indirect prepared-layout code.
SOURCES = tuple(sorted(set(RT_SOURCES) | {
    "scripts/olmo_ordinary_efficiency.py", "scripts/olmo_ordinary_fusions.py",
    "scripts/olmo_rt_large_batch.py",
    "scripts/olmo_large_batch_validation.py",
    "scripts/olmo_ordinary_optimizer_probe.py", "scripts/docker_shell.sh",
    "scripts/experiment_tracking.py", "scripts/olmo_lm_common.py",
} | {str(path.relative_to(ROOT)) for path in (ROOT / "cdrm/pretrained").glob("*.py")}))
ARMS = {
    "control": {"rope": "native", "pointwise": "eager", "attention": "sdpa", "fused_adam": None},
    "optimized": {"rope": "dao", "pointwise": "compiled", "attention": "sdpa", "fused_adam": True},
    "fa4": {"rope": "dao", "pointwise": "compiled", "attention": "fa4", "fused_adam": True},
    "compiled-native": {"rope": "native", "pointwise": "compiled", "attention": "sdpa", "fused_adam": True},
    "fa4-native": {"rope": "native", "pointwise": "compiled", "attention": "fa4", "fused_adam": True},
}


def optimizer_for(model, arm):
    optimizer = build_adamw(model, lr=1e-5, betas=(.9, .95), eps=1e-8,
                            weight_decay=.1, foreach=False, fused=ARMS[arm]["fused_adam"])
    return optimizer, build_warmup_scheduler(optimizer, warmup_updates=2)


def optimizer_flags(optimizer):
    return [{"fused": group.get("fused"), "foreach": group.get("foreach"),
             "capturable": group.get("capturable"),
             "parameter_tensors": len(group["params"])} for group in optimizer.param_groups]



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("correctness", "capacity"), required=True)
    parser.add_argument("--case", choices=("rt", "combined"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--reference-arm", choices=tuple(ARMS), default="control")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--length", type=int, choices=(32, 512), default=512)
    parser.add_argument("--release-transient-cache", action="store_true")
    parser.add_argument("--validation-order", choices=("live-graph", "before-capture"), default="live-graph")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--continue-after-compatibility-miss", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if args.stage == "correctness":
        if args.validation_order != "live-graph":
            parser.error("Validation reordering applies only to capacity")
        if args.batch_size not in (1, 2, 8) or args.profile:
            parser.error("Correctness uses B1/2/8 and no profiler; primary B8/T512")
    else:
        if args.continue_after_compatibility_miss or args.reference_arm != "control":
            parser.error("Comparison flags apply only to correctness")
        if args.length != 512 or args.batch_size not in (32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512):
            parser.error("Capacity is bounded to physical B32..512, T512")
    return args


def can_continue_after_failure(check, *, enabled):
    """Only a finite same-state numeric miss can defer failure until the end.

    This never changes a check's result or budgets. Ownership, supervision,
    dispatch, graph and optimizer failures remain immediately fatal.
    """
    return bool(enabled and check.get("passed") is False
        and check.get("name") == "same_state_candidate_vs_reference"
        and all(check.get(key) is True for key in ("ownership_matches", "finite", "counts_equal"))
        and all(check.get(key) for key in ("losses", "outputs", "gradients")))


def final_check_summary(checks):
    """Report numerical qualification separately, never promote a failed run."""
    numerical = [check for check in checks if check["name"] == "same_state_candidate_vs_reference"]
    operational = [check for check in checks if check["name"] != "same_state_candidate_vs_reference"]
    return {"status": "passed" if checks and all(check["passed"] for check in checks) else "failed",
        "numerical_compatibility_passed": all(check["passed"] for check in numerical) if numerical else None,
        "operational_checks_passed": bool(operational) and all(check["passed"] for check in operational)}


def batch_for(tokenizer, case, update):
    """One fixed all-valid CE contract for comparisons AND every Adam step."""
    batch = changed_batch(tokenizer, case, update)
    return replace(batch, ce_mask=batch.valid_mask.clone())




def set_arm(model, arm):
    options = ARMS[arm]
    base = model.backbone.backbone
    base.ordinary_attention_backend = options["attention"]
    base.ordinary_checkpoint_layers = None
    base.ordinary_pointwise_backend = options["pointwise"]
    base.ordinary_rope_backend = options["rope"]
    base.reuse_rope = True
    base.kv_only_writes = True
    model.config = replace(model.config, ce_chunk_size=2048)
    if model.config.vocab_chunk_size != 128:
        raise ValueError("KL position chunks must remain 128")


def exact_comparison(reference, candidate):
    return all(ARMS[reference][key] == ARMS[candidate][key]
               for key in ("rope", "pointwise", "attention"))


def compare_arms(model, batch, mode, reference_arm, candidate_arm):
    set_arm(model, reference_arm)
    reference_plan = new_plan(model, batch, mode, "recompute")
    outputs = output_snapshot(reference_plan)
    reference = reference_plan.backward(replay=False)
    losses = loss_snapshot(reference)
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    finite_reference = all(bool(torch.isfinite(value).all()) for value in (*outputs.values(), *losses.values(), *gradients.values()))
    counts = dict(reference_plan.counts)
    del reference_plan, reference
    model.zero_grad(set_to_none=True)
    set_arm(model, candidate_arm)
    plan = new_plan(model, batch, mode, "recompute")
    candidate_outputs = output_snapshot(plan)
    candidate_losses = loss_snapshot(plan.backward(replay=False))
    names = {name for name, p in model.named_parameters() if p.grad is not None}
    ownership = names == set(gradients) == set(plan.active_names)
    losses_check = {name: metric(value, losses[name]) for name, value in candidate_losses.items()}
    outputs_check = {name: metric(value, outputs[name]) for name, value in candidate_outputs.items()}
    gradients_check = {name: metric(p.grad, gradients[name]) for name, p in model.named_parameters()
                       if p.grad is not None and name in gradients}
    finite = finite_reference and all(bool(torch.isfinite(value).all())
        for value in (*candidate_losses.values(), *candidate_outputs.values(),
                      *(p.grad for p in model.parameters() if p.grad is not None)))
    exact_required = exact_comparison(reference_arm, candidate_arm)
    counts_equal = counts == plan.counts
    screen = comparison_passes(exact_required=exact_required, ownership=ownership,
        finite=finite, counts_equal=counts_equal, losses=losses_check,
        gradients=gradients_check, outputs=outputs_check)
    return plan, {"name": "same_state_candidate_vs_reference", **screen,
        "reference_arm": reference_arm, "candidate_arm": candidate_arm,
        "ownership_matches": ownership, "finite": finite, "counts_equal": counts_equal,
        "exact_required": exact_required, "budgets": dict(BUDGETS),
        "losses": losses_check, "outputs": outputs_check, "gradients": gradients_check,
        "scope": "Same pretrained state and all-valid CE; unscaled raw gradients before clipping or optimization."}


def compiler_configuration():
    import torch._dynamo
    torch._dynamo.utils.counters.clear()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    # Fullgraph SwiGLU is reused at different batch/sequence and grad-mode
    # specializations. An exhausted guard cache must fail, never fall back.
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
    return {"suppress_errors": False, "cache_size_limit": 64,
        "accumulated_cache_size_limit": 256, "scope": "ordinary SwiGLU only",
        "fullgraph": True, "dynamic": False, "recompile_limit": 64,
        "fail_on_recompile_limit_hit": True}


def compiler_observations():
    from torch._dynamo.utils import counters
    return {str(group): {str(key): int(value) for key, value in rows.items()}
            for group, rows in counters.items()}


def dependency_record(directory, *, include_dao, include_fa4):
    from scripts.olmo_ordinary_fusions import dependency_record as dao_record
    from scripts.olmo_ordinary_efficiency import dependency_record as fa4_record
    result = dao_record(directory, include_dao=include_dao)
    other = fa4_record(directory, include_fa4=include_fa4)
    result["packages"].update(other["packages"])
    result["fa4_sources"] = other["fa4_sources"]
    if "fa4_interface" in other:
        result["fa4_interface"] = other["fa4_interface"]
    return result


def check_dependencies(record):
    return all(sha256_file(Path(row["source"])) == row["sha256"]
               for key in ("dao_sources", "fa4_sources") for row in record[key].values())


def dispatch_probe(plan, arm):
    """CPU-only operator tracing; no GPU profiler competing with capacity."""
    from cdrm.pretrained import olmo_ordinary
    names = {"dao_rope_loader": "_load_dao_rope", "fa4_loader": "_load_fa4",
             "compiled_swiglu": "_compiled_swiglu"}
    calls = dict.fromkeys(names, 0)
    def counted(key, original):
        def call():
            calls[key] += 1
            return original()
        return call
    with ExitStack() as stack:
        for key, name in names.items():
            stack.enter_context(patch.object(olmo_ordinary, name, counted(key, getattr(olmo_ordinary, name))))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            plan.backward(replay=False)
    operators = sorted({event.key for event in profile.key_averages()
                        if "scaled_dot_product" in event.key or "flash_attention" in event.key})
    options = ARMS[arm]
    flash_present = any("flash_attention" in name for name in operators)
    sdpa_present = any("scaled_dot_product" in name for name in operators)
    good = ((calls["fa4_loader"] > 0 and not sdpa_present) if options["attention"] == "fa4"
            else (flash_present and calls["fa4_loader"] == 0))
    good &= (calls["compiled_swiglu"] > 0) == (options["pointwise"] == "compiled")
    good &= (calls["dao_rope_loader"] > 0) == (options["rope"] == "dao")
    observed = compiler_observations()
    if options["pointwise"] == "compiled":
        good &= observed.get("stats", {}).get("unique_graphs", 0) > 0 and not observed.get("graph_break", {})
    return {"name": "ordinary_dispatch_no_fallback", "passed": bool(good),
        "calls": calls, "operators": operators, "compiler_counters": observed,
        "expected_options": options, "scope": "Actual selected RT stack including ordinary bootstrap if FBT; native RT backend unchanged."}


def compare_graph_cpu(plan, name, *, replays=1):
    """Exact all-gradient check, with CPU reference storage outside timing.

    A single tensor at a time is copied from the candidate to CPU. No additional
    model-sized reference-gradient allocation is held beside the live graph.
    """
    if type(replays) is not int or replays < 1:
        raise ValueError("Require at least one graph replay")
    expected = plan.backward(replay=False)
    losses = {n: value.detach().cpu().clone() for n, value in loss_snapshot(expected).items()}
    gradients = {n: p.grad.detach().cpu().clone() for n, p in plan.model.named_parameters() if p.grad is not None}
    del expected
    for _ in range(replays):
        actual = plan.backward(replay=True)
    actual_losses = loss_snapshot(actual)
    loss_names_match = set(actual_losses) == set(losses)
    loss_checks = {n: {"bitwise_equal": n in losses and torch.equal(value.detach().cpu(), losses[n])}
                   for n, value in actual_losses.items()}
    gradient_checks = {n: {"bitwise_equal": n in gradients and torch.equal(p.grad.detach().cpu(), gradients[n])}
                       for n, p in plan.model.named_parameters() if p.grad is not None}
    ownership = set(gradient_checks) == set(gradients) == set(plan.active_names) == active_names(plan.model, plan.mode)
    exact = all(row["bitwise_equal"] for row in (*loss_checks.values(), *gradient_checks.values()))
    return {"name": name, "passed": ownership and loss_names_match and exact,
        "all_bitwise_equal": loss_names_match and exact, "loss_names_match": loss_names_match,
        "ownership_matches": ownership, "losses": loss_checks, "gradients": gradient_checks,
        "reference_storage": "CPU full gradients; one tensor streamed at a time; untimed"}


def detailed_memory_snapshot():
    result = memory_snapshot()
    free, total = torch.cuda.mem_get_info()
    result.update(device_free_gib=free / 2**30, device_total_gib=total / 2**30,
                  device_used_gib=(total-free) / 2**30)
    return result


class MemoryPhases:
    def __init__(self, report, persist):
        self.report, self.persist = report, persist
        self.rows = report.setdefault("memory_phases", {})

    def observe(self, name, event):
        if event == "begin":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.rows[name] = {"start": detailed_memory_snapshot(), "reset_peaks": True,
                "measurement_scope": "Absolute allocator peaks within phase; device usage sampled only at boundaries."}
        else:
            if event == "end":
                torch.cuda.synchronize()
            row = self.rows[name]
            row["end"] = detailed_memory_snapshot()
            row.update({key: row["end"][key] for key in ("peak_allocated_gib", "peak_reserved_gib")})
            if event == "error":
                row["error"] = True
        self.persist()

    @contextmanager
    def phase(self, name):
        self.observe(name, "begin")
        try:
            yield
        except BaseException as error:
            try:
                self.observe(name, "error")
            except Exception as observation_error:
                error.add_note(f"Memory phase observer also failed: {observation_error}")
            raise
        else:
            self.observe(name, "end")

    def capture_observer(self, name, event):
        self.observe("capture_" + name, event)

    def setup_summary(self):
        result = detailed_memory_snapshot()
        for key in ("peak_allocated_gib", "peak_reserved_gib"):
            result[key] = max([result[key], *(row.get("end", row["start"])[key] for row in self.rows.values())])
        return result


def finish_tracking(tracker, report, *, original_error=None):
    """A logging failure invalidates success, without replacing a primary failure."""
    try:
        tracker.finish(succeeded=report["status"] == "passed")
    except BaseException as error:
        report["tracking_finish_error"] = {"type": type(error).__name__, "message": str(error)}
        if original_error is not None:
            original_error.add_note(f"Tracking finalization also failed: {error}")
            return
        if report["status"] == "passed":
            report.update(status="failed", stage="tracking_finish")
            report.setdefault("error", {"type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc()})
        raise


def full_step_profile(plan, optimizer, scheduler, counters, batch, directory):
    """One additional canonical update with annotated CPU phases and device trace."""
    def scoped(function, label):
        def call(*args, **kwargs):
            with torch.profiler.record_function("ordinary_step/" + label):
                return function(*args, **kwargs)
        return call
    torch.cuda.synchronize()
    with ExitStack() as stack:
        for owner, name, label in (
            (plan, "load_batch", "input_and_validation"), (plan, "backward", "forward_loss_backward"),
            (torch.nn.utils, "clip_grad_norm_", "clipping"), (optimizer, "step", "optimizer"),
            (scheduler, "step", "scheduler")):
            stack.enter_context(patch.object(owner, name, scoped(getattr(owner, name), label)))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function("ordinary_step/complete"):
                record = plan.optimizer_step(optimizer, batch, replay=True,
                                              scheduler=scheduler, counters=counters)
            torch.cuda.synchronize()
    path = directory / "full-step-trace.json"
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
    phases = {event.key: {"cpu_us": event.cpu_time_total, "device_us": event.device_time_total,
                         "calls": event.count} for event in profile.key_averages()
              if event.key.startswith("ordinary_step/")}
    return {"scope": "One additional untimed canonical optimizer update after all timings. CPU phase device attribution can omit captured kernels; raw CUDA event names retained. Not a timed sample.",
            "trace_file": compressed.name, "trace_bytes": compressed.stat().st_size,
            "trace_sha256": sha256_file(compressed), "device_kernels": totals,
            "device_event_count": len(events), "cpu_phase_scopes": phases, "update_record": record}




def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    compiler = compiler_configuration()
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
    case = selected_case(args.case, batch=args.batch_size, length=args.length)
    configuration = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "case_specification": asdict(case), "mode": asdict(case.mode()), "precision": "bf16_mixed",
        "parameter_optimizer_dtype": "float32", "active_rt_layer_count": len(case.rt_layers),
        "selected_rt_layers": list(case.rt_layers), "fbt": case.fbt, "nextlat": case.nextlat,
        "reuse_rope": True, "kv_only_writes": True, "physical_batch": args.batch_size,
        "ce_chunk_size": 2048, "kl_chunk_size": 128, "supervision": "all_valid_ce",
        "autocast_weight_cache": False, "tf32": False, "world_size": 1, "accumulation": 1,
        "capture_warmup_backwards": 10, "seed": 20260922,
        "preparation_updates": 3 if args.stage == "capacity" else 0,
        "timed_updates": 5 if args.stage == "capacity" else 0,
        "backward_timing_samples": 3 if args.stage == "capacity" else 0,
        "ordinary_attention": ARMS[args.arm]["attention"], "ordinary_pointwise": ARMS[args.arm]["pointwise"],
        "ordinary_checkpointing": "all", "ordinary_rope_backend": ARMS[args.arm]["rope"],
        "fused_adam": ARMS[args.arm]["fused_adam"], "arm_switches": ARMS}
    report = {"schema": "olmo-rt-large-batch-v1", "status": "running", "stage": "load",
        "configuration": configuration, "runtime": runtime, "determinism": determinism,
        "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "compiler_configuration": compiler, "started_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [], "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "physical_optimizer_updates": 0}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-rt-large-batch",
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
            if can_continue_after_failure(check, enabled=args.continue_after_compatibility_miss):
                report["compatibility_miss_continued"] = True
                save()
                return
            raise AssertionError(check["name"])

    def exact_graph(plan, name, replays=1):
        check = compare_graph_cpu(plan, name, replays=replays)
        check["passed"] = check["passed"] and check["all_bitwise_equal"]
        publish(check)

    phases = MemoryPhases(report, save)
    plan = None
    hook = None
    try:
        save()
        report["dependencies"] = dependency_record(args.output_dir,
            include_dao=any(ARMS[arm]["rope"] == "dao" for arm in (args.arm, args.reference_arm)),
            include_fa4=any(ARMS[arm]["attention"] == "fa4" for arm in (args.arm, args.reference_arm)))
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        save()
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        with phases.phase("load_model"):
            model = build_model(state, case)
        del state
        report["parameters"] = parameter_inventory(model,
            executed_names=active_names(model, case.mode()), inference_names=inference_names(model, case))
        batch = batch_for(tokenizer, case, 0)
        report["initial_batch"] = tree_digests(vars(batch))
        report["comparison_batches"] = {str(index): tree_digests(vars(batch_for(tokenizer, case, index)))
            for index in (1, 2, 5, 6, 7)}
        with backend_context("flash"):
            if args.stage == "correctness":
                save("same_state_comparison")
                plan, check = compare_arms(model, batch, case.mode(), args.reference_arm, args.arm)
                publish(check)
                publish(dispatch_probe(plan, args.arm))
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10, release_transient_cache=args.release_transient_cache,
                    phase_observer=phases.capture_observer)
                report["capture_seconds"] = time.perf_counter() - began
                exact_graph(plan, "candidate_initial_graph")
                plan.load_batch(batch_for(tokenizer, case, 1))
                exact_graph(plan, "candidate_changed_tokens_overwrite", replays=2)
                save("complete_update_parity")
                publish(complete_update_parity(plan, tokenizer, case, report=report,
                    persist=save, batch_factory=batch_for,
                    optimizer_factory=lambda model: optimizer_for(model, args.arm)))
                plan.load_batch(batch_for(tokenizer, case, 2))
                exact_graph(plan, "candidate_changed_weights")
                report["memory"] = memory_snapshot()
                report["resources"] = resource_card(plan, case)
            else:
                set_arm(model, args.arm)
                with phases.phase("prepare_plan"):
                    plan = new_plan(model, batch, case.mode(), "recompute")
                with phases.phase("dispatch"):
                    publish(dispatch_probe(plan, args.arm))
                optimizer, scheduler = optimizer_for(model, args.arm)
                report["optimizer_flags"] = optimizer_flags(optimizer)
                hook = track_optimizer_steps(optimizer, report)
                counters = TrainingCounters()
                probe = model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096].clone()
                preparation = []
                report["preparation_batches"] = []
                save("preparation_updates")
                phases.observe("preparation_updates", "begin")
                for index in range(3):
                    fresh = batch_for(tokenizer, case, index)
                    report["preparation_batches"].append(tree_digests(vars(fresh)))
                    preparation.append(plan.optimizer_step(optimizer, fresh,
                        scheduler=scheduler, counters=counters))
                    report["preparation_records"] = preparation
                    save()
                phases.observe("preparation_updates", "end")
                if args.validation_order == "before-capture":
                    save("validation_references")
                    with phases.phase("validation_references"):
                        initial_refs = snapshot_eager_cpu(plan, batch_for(tokenizer, case, 2))
                        changed_refs = snapshot_eager_cpu(plan, batch_for(tokenizer, case, 1))
                        plan.load_batch(batch_for(tokenizer, case, 2))
                    report["validation_reference_batches"] = {
                        "initial": tree_digests(vars(batch_for(tokenizer, case, 2))),
                        "changed": tree_digests(vars(batch_for(tokenizer, case, 1)))}
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10, release_transient_cache=args.release_transient_cache,
                    phase_observer=phases.capture_observer)
                report["capture_seconds"] = time.perf_counter() - began
                save("validation_initial")
                with phases.phase("validation_initial"):
                    if args.validation_order == "before-capture":
                        publish(replay_compare_cpu(plan, initial_refs, "capacity_initial_graph"))
                    else:
                        exact_graph(plan, "capacity_initial_graph")
                plan.load_batch(batch_for(tokenizer, case, 1))
                save("validation_changed_tokens")
                with phases.phase("validation_changed_tokens"):
                    if args.validation_order == "before-capture":
                        publish(replay_compare_cpu(plan, changed_refs, "capacity_changed_tokens_overwrite", replays=2))
                        del initial_refs, changed_refs
                    else:
                        exact_graph(plan, "capacity_changed_tokens_overwrite", replays=2)
                report["setup_memory"] = phases.setup_summary()
                torch.cuda.reset_peak_memory_stats()
                records = []
                report["timed_records"] = records
                batches = [batch_for(tokenizer, case, index + 3) for index in range(5)]
                report["timed_batches"] = [tree_digests(vars(value)) for value in batches]

                def complete():
                    records.append(plan.optimizer_step(optimizer, batches[len(records)], replay=True,
                        scheduler=scheduler, counters=counters))

                save("timing")
                with phases.phase("timing"):
                    report["full_update"] = timed(complete, 5)
                report["steady_memory"] = detailed_memory_snapshot()
                report["input_tokens_per_second"] = plan.input_tokens / report["full_update"]["median_wall_seconds"]
                report["ce_targets_per_second"] = plan.counts["ce"] / report["full_update"]["median_wall_seconds"]
                save("backward_timing")
                with phases.phase("backward_timing"):
                    report["forward_loss_backward"] = timed(lambda: plan.backward(replay=True), 3)
                report["forward_loss_backward_tokens_per_second"] = plan.input_tokens / report["forward_loss_backward"]["median_wall_seconds"]
                if args.validation_order == "live-graph":
                    save("capacity_changed_weights_validation")
                    with phases.phase("validation_changed_weights"):
                        exact_graph(plan, "capacity_changed_weights")
                with phases.phase("health"):
                    report["health"] = state_health(model, optimizer)
                report["resources"] = resource_card(plan, case, optimizer)
                changed = not torch.equal(probe, model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096])
                finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
                publish({"name": "finite_complete_updates", "passed": report["health"]["passed"] and changed and finite,
                    "trainable_weight_changed": changed, "finite_participating_gradients": finite})
                tracker.log({"benchmark/input_tokens_per_second": report["input_tokens_per_second"],
                    "benchmark/ce_targets_per_second": report["ce_targets_per_second"],
                    "benchmark/forward_loss_backward_tokens_per_second": report["forward_loss_backward_tokens_per_second"],
                    "benchmark/setup_peak_reserved_gib": report["setup_memory"]["peak_reserved_gib"],
                    "benchmark/steady_peak_allocated_gib": report["steady_memory"]["peak_allocated_gib"]}, step=len(report["checks"]) + 1)
                if args.profile:
                    phases.observe("profile", "begin")
                    save("untimed_operator_profile")
                    report["profile"] = operator_profile(plan, args.output_dir)
                    save("untimed_complete_step_profile")
                    report["profile_batch"] = tree_digests(vars(batch_for(tokenizer, case, 9)))
                    report["full_step_profile"] = full_step_profile(plan, optimizer, scheduler, counters,
                        batch_for(tokenizer, case, 9), args.output_dir)
                    report["post_profile_health"] = state_health(model, optimizer)
                    phases.observe("profile", "end")
                    if not report["post_profile_health"]["passed"]:
                        raise AssertionError("Nonfinite state after profiled canonical update")
                if args.validation_order == "before-capture":
                    save("capacity_terminal_validation")
                    with phases.phase("validation_changed_weights"):
                        final_refs = snapshot_replay_cpu(plan)
                        publish(release_graph_then_compare_eager_cpu(plan, final_refs, "capacity_changed_weights"))
                        del final_refs
            report["prepared_layout"] = plan.forward_layout.metadata
            report["nextlat_config"] = model.config.to_dict()
            report["counts"] = dict(plan.counts)
            report["input_tokens"] = plan.input_tokens
            report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
                "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
            report["compiler_observations"] = compiler_observations()
            forbidden = {group: values for group, values in report["compiler_observations"].items()
                         if group in ("graph_break", "unimplemented") and any(values.values())}
            if forbidden:
                raise AssertionError(f"Compiler graph breaks or unsupported paths: {forbidden}")
        if report["source_hashes"] != {path: sha256_file(ROOT / path) for path in SOURCES}:
            raise AssertionError("Runtime source changed during the run")
        if report["protocol_sha256"] != sha256_file(PROTOCOL) or not check_dependencies(report["dependencies"]):
            raise AssertionError("Protocol or external dependency changed during the run")
        report.update(final_check_summary(report["checks"]))
        tracker.summary({"result_status": report["status"],
            "numerical_compatibility_passed": report["numerical_compatibility_passed"],
            "operational_checks_passed": report["operational_checks_passed"],
            "physical_optimizer_updates": report["physical_optimizer_updates"]})
        save("complete")
        if report["status"] != "passed":
            raise AssertionError("Numerical compatibility remains failed; completed operational diagnostics do not clear it")
    except BaseException as error:
        if plan is not None:
            report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
                "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
        for name, row in phases.rows.items():
            if "end" not in row:
                try:
                    phases.observe(name, "error")
                except Exception:
                    row["error"] = "Snapshot unavailable after failure"
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save()
        raise
    finally:
        if hook is not None:
            hook.remove()
        try:
            finish_tracking(tracker, report, original_error=sys.exception())
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()


if __name__ == "__main__":
    main()
