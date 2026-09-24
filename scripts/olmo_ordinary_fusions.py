#!/usr/bin/env python3
"""Bounded native-FP32 Dao RoPE and fused AdamW ordinary OLMo comparisons."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from scripts.olmo_f4_resources import selected_case, memory_snapshot
from scripts.olmo_rt_efficiency import (
    SOURCES as RT_SOURCES, BUDGETS, metric, comparison_passes, output_snapshot,
    complete_update_parity, track_optimizer_steps, operator_profile,
)

PROTOCOL = ROOT / "docs/reports/olmo-ordinary-fusions/protocol.md"
# Include actual runtime dependencies, including private helper imports. The
# package-wide superset avoids silently dropping indirect prepared-layout code.
SOURCES = tuple(sorted(set(RT_SOURCES) | {
    "scripts/olmo_ordinary_efficiency.py", "scripts/olmo_ordinary_fusions.py",
    "scripts/olmo_ordinary_optimizer_probe.py", "scripts/docker_shell.sh",
    "scripts/experiment_tracking.py", "scripts/olmo_lm_common.py",
} | {str(path.relative_to(ROOT)) for path in (ROOT / "cdrm/pretrained").glob("*.py")}))
ARMS = {
    "control": ("native", None),
    "dao-rope": ("dao", None),
    "fused-adam": ("native", True),
    "dao-rope-fused-adam": ("dao", True),
}


def optimizer_for(model, arm):
    optimizer = build_adamw(model, lr=1e-5, betas=(.9, .95), eps=1e-8,
                            weight_decay=.1, foreach=False, fused=ARMS[arm][1])
    return optimizer, build_warmup_scheduler(optimizer, warmup_updates=2)


def optimizer_flags(optimizer):
    return [{"fused": group.get("fused"), "foreach": group.get("foreach"),
             "capturable": group.get("capturable"),
             "parameter_tensors": len(group["params"])} for group in optimizer.param_groups]



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("correctness", "capacity"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--reference-arm", choices=tuple(ARMS), default="control")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--length", type=int, choices=(32, 512, 2048), default=512)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--optimizer-probe", action="store_true")
    parser.add_argument("--continue-after-compatibility-miss", action="store_true",
        help="Correctness only: retain failed numeric compatibility and finish operational checks; still exits unsuccessfully")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if args.stage == "correctness":
        if args.batch_size not in (1, 2, 8) or args.profile:
            parser.error("Correctness uses B1/2/8 and no profiler; primary B8/T512")
    else:
        if args.continue_after_compatibility_miss:
            parser.error("Compatibility-miss continuation applies only to correctness")
        allowed = {512: (16, 32, 64), 2048: (8, 16)}
        if args.batch_size not in allowed.get(args.length, ()):
            parser.error("Capacity uses T512/B16,32,64 or T2048/B8,16")
        if args.reference_arm != "control":
            parser.error("--reference-arm applies only to correctness")
    if args.optimizer_probe and (args.stage != "correctness" or args.arm != "fused-adam" or args.batch_size != 8 or args.length != 512):
        parser.error("Fixed-gradient optimizer probe requires correctness fused-adam B8/T512")
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


def checkpoint_layers(policy, depth):
    if policy == "all":
        return None
    if policy == "none":
        return ()
    if policy == "alternating":
        return tuple(range(0, depth, 2))
    raise ValueError("Unknown ordinary checkpoint policy")


def set_arm(model, arm):
    rope, _ = ARMS[arm]
    base = model.backbone.backbone
    base.ordinary_attention_backend = "sdpa"
    base.ordinary_checkpoint_layers = None
    base.ordinary_pointwise_backend = "compiled"
    base.ordinary_rope_backend = rope
    base.reuse_rope = True
    model.config = replace(model.config, ce_chunk_size=2048)
    if model.config.vocab_chunk_size != 128:
        raise ValueError("KL position chunks must remain 128")


def exact_comparison(reference, candidate):
    return ARMS[reference][0] == ARMS[candidate][0]


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


def dependency_record(directory, *, include_dao):
    packages = {}
    for name in ("torch", "triton", "flash-attn", "flash-attn-4", "transformer-engine"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {"packages": packages, "dao_sources": {}, "environment": {
        key: os.environ.get(key) for key in (
            "CDRM_FLASH_ATTENTION_SOURCE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
            "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")}}
    if include_dao:
        import flash_attn.layers.rotary as layer
        import flash_attn.ops.triton.rotary as kernel
        for relative, module in (("layers/rotary.py", layer), ("ops/triton/rotary.py", kernel)):
            path = Path(module.__file__).resolve()
            target = directory / "dependency-snapshot/flash_attn" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record["dao_sources"][relative] = {"source": str(path), "sha256": sha256_file(path)}
    return record


def check_dependencies(record):
    return all(sha256_file(Path(row["source"])) == row["sha256"]
               for row in record["dao_sources"].values())


def dispatch_probe(plan, arm):
    """Actual ordinary backward, requiring compiled SwiGLU/Flash and chosen RoPE."""
    from cdrm.pretrained import olmo_ordinary
    from scripts.olmo_ordinary_efficiency import dispatch_probe as prior_dispatch
    calls = {"dao_rope_loader": 0}
    original = olmo_ordinary._load_dao_rope
    def loader():
        calls["dao_rope_loader"] += 1
        return original()
    with patch.object(olmo_ordinary, "_load_dao_rope", loader):
        check = prior_dispatch(plan, "compiled")
    expected = ARMS[arm][0]
    good = calls["dao_rope_loader"] >= 32 if expected == "dao" else calls["dao_rope_loader"] == 0
    check.update(passed=check["passed"] and good, rope_calls=calls,
                 expected_rope=expected)
    return check


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


def resource_card(plan, case, optimizer=None):
    base = plan.model.backbone.backbone
    work = LossWork(ce_targets=plan.counts["ce"], latent_pairs=plan.counts["latent"],
        kl_triples=plan.counts["kl"], predictor_positions=plan.loss_layout.needed_source_indices.numel())
    kwargs = dict(batch_size=case.batch_size, sequence_length=case.length, mode=plan.mode,
        nextlat=None, loss_work=work, backward_memory="recompute")
    plain = estimate_training_resources(base.config, ordinary_checkpointing=False, **kwargs).to_dict()
    checkpointed = estimate_training_resources(base.config, ordinary_checkpointing=True, **kwargs).to_dict()
    selected = base.ordinary_checkpoint_layers
    count = base.config.num_layers if selected is None else len(selected)
    by_name = {row["name"]: row for row in plain["components"]}
    # The shared estimator is linear in equal-sized ordinary layer calls. Add
    # only the checkpoint-induced difference, scaled by the selected layer count.
    combined = []
    for row in checkpointed["components"]:
        original = by_name.get(row["name"], {"minimum": 0, "maximum": 0})
        copy = dict(row)
        for boundary in ("minimum", "maximum"):
            extra = row[boundary] - original[boundary]
            numerator = extra * count
            if numerator % base.config.num_layers:
                raise AssertionError("Nonintegral per-layer arithmetic estimate")
            copy[boundary] = original[boundary] + numerator // base.config.num_layers
        if copy["minimum"] or copy["maximum"]:
            combined.append(copy)
    plain.update(components=combined,
        matrix_flops_minimum=sum(row["minimum"] for row in combined),
        matrix_flops_maximum=sum(row["maximum"] for row in combined))
    plain["assumptions"] = [*plain["assumptions"],
        f"Checkpoint-induced arithmetic is scaled by {count}/{base.config.num_layers} ordinary layers."]
    return {"analytic_matrix_work": plain, "loss_work": asdict(work),
        "checkpointed_ordinary_layer_count": count,
        "observed_parameters": parameter_inventory(plan.model, optimizer=optimizer,
            executed_names=active_names(plan.model, plan.mode), inference_names=inference_names(plan.model, case)),
        "scope": "Logical matrix arithmetic only, not hardware FLOPs; excludes pointwise fusion savings and kernel padding."}


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
    case = selected_case("ordinary", batch=args.batch_size, length=args.length)
    configuration = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "case_specification": asdict(case), "mode": asdict(case.mode()), "precision": "bf16_mixed",
        "parameter_optimizer_dtype": "float32", "active_rt_layer_count": 0,
        "fbt": False, "nextlat": False, "reuse_rope": True,
        "ce_chunk_size": 2048, "kl_chunk_size": 128, "supervision": "all_valid_ce",
        "autocast_weight_cache": False, "tf32": False, "world_size": 1, "accumulation": 1,
        "capture_warmup_backwards": 10, "seed": 20260922,
        "preparation_updates": 3 if args.stage == "capacity" else 0,
        "timed_updates": 5 if args.stage == "capacity" else 0,
        "backward_timing_samples": 3 if args.stage == "capacity" else 0,
        "ordinary_attention": "sdpa", "ordinary_pointwise": "compiled", "ordinary_checkpointing": "all",
        "ordinary_rope_backend": ARMS[args.arm][0], "fused_adam": ARMS[args.arm][1],
        "arm_switches": {name: dict(zip(("rope", "fused_adam"), options)) for name, options in ARMS.items()}}
    report = {"schema": "olmo-ordinary-fusions-v1", "status": "running", "stage": "load",
        "configuration": configuration, "runtime": runtime, "determinism": determinism,
        "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "compiler_configuration": compiler, "started_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [], "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "physical_optimizer_updates": 0}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-ordinary-fusions",
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
        check = compare_graph(plan, name, replays=replays)
        check["passed"] = check["passed"] and check["all_bitwise_equal"]
        publish(check)

    hook = None
    try:
        save()
        report["dependencies"] = dependency_record(args.output_dir,
            include_dao=any(ARMS[arm][0] == "dao" for arm in (args.arm, args.reference_arm)))
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": report["checkpoint"]})
        save()
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
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
                if args.optimizer_probe:
                    save("fixed_gradient_optimizer_comparison")
                    from scripts.olmo_ordinary_optimizer_probe import compare_fixed_gradients
                    publish(compare_fixed_gradients(plan, report=report, persist=save))
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
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
                plan = new_plan(model, batch, case.mode(), "recompute")
                publish(dispatch_probe(plan, args.arm))
                optimizer, scheduler = optimizer_for(model, args.arm)
                report["optimizer_flags"] = optimizer_flags(optimizer)
                hook = track_optimizer_steps(optimizer, report)
                counters = TrainingCounters()
                probe = model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096].clone()
                torch.cuda.reset_peak_memory_stats()
                preparation = []
                report["preparation_batches"] = []
                save("preparation_updates")
                for index in range(3):
                    fresh = batch_for(tokenizer, case, index)
                    report["preparation_batches"].append(tree_digests(vars(fresh)))
                    preparation.append(plan.optimizer_step(optimizer, fresh,
                        scheduler=scheduler, counters=counters))
                    report["preparation_records"] = preparation
                    save()
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
                report["capture_seconds"] = time.perf_counter() - began
                exact_graph(plan, "capacity_initial_graph")
                plan.load_batch(batch_for(tokenizer, case, 1))
                exact_graph(plan, "capacity_changed_tokens_overwrite", replays=2)
                report["setup_memory"] = memory_snapshot()
                torch.cuda.reset_peak_memory_stats()
                records = []
                report["timed_records"] = records
                batches = [batch_for(tokenizer, case, index + 3) for index in range(5)]
                report["timed_batches"] = [tree_digests(vars(value)) for value in batches]

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
                save("capacity_changed_weights_validation")
                exact_graph(plan, "capacity_changed_weights")
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
                    save("untimed_operator_profile")
                    report["profile"] = operator_profile(plan, args.output_dir)
                    save("untimed_complete_step_profile")
                    report["profile_batch"] = tree_digests(vars(batch_for(tokenizer, case, 9)))
                    report["full_step_profile"] = full_step_profile(plan, optimizer, scheduler, counters,
                        batch_for(tokenizer, case, 9), args.output_dir)
                    report["post_profile_health"] = state_health(model, optimizer)
                    if not report["post_profile_health"]["passed"]:
                        raise AssertionError("Nonfinite state after profiled canonical update")
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
            raise AssertionError("Protocol or Dao dependency changed during the run")
        report.update(final_check_summary(report["checks"]))
        tracker.summary({"result_status": report["status"],
            "numerical_compatibility_passed": report["numerical_compatibility_passed"],
            "operational_checks_passed": report["operational_checks_passed"],
            "physical_optimizer_updates": report["physical_optimizer_updates"]})
        save("complete")
        if report["status"] != "passed":
            raise AssertionError("Numerical compatibility remains failed; completed operational diagnostics do not clear it")
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save()
        raise
    finally:
        if hook is not None:
            hook.remove()
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()


if __name__ == "__main__":
    main()
