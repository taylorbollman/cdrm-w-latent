#!/usr/bin/env python3
"""Pretrained ordinary OLMo execution ablations; no RT, FBT or NextLat training."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from cdrm.pretrained.lm_training import TrainingCounters
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

PROTOCOL = ROOT / "docs/reports/olmo-ordinary-efficiency/protocol.md"
# Include actual runtime dependencies, including private helper imports. The
# package-wide superset avoids silently dropping indirect prepared-layout code.
SOURCES = tuple(sorted(set(RT_SOURCES) | {
    "scripts/olmo_ordinary_efficiency.py", "scripts/docker_shell.sh",
    "scripts/experiment_tracking.py", "scripts/olmo_lm_common.py",
} | {str(path.relative_to(ROOT)) for path in (ROOT / "cdrm/pretrained").glob("*.py")}))
ARMS = {
    "control": ("sdpa", "all", "eager"),
    "fa4": ("fa4", "all", "eager"),
    "checkpoint-none": ("sdpa", "none", "eager"),
    "checkpoint-alternating": ("sdpa", "alternating", "eager"),
    "compiled": ("sdpa", "all", "compiled"),
    "fa4-compiled": ("fa4", "all", "compiled"),
    "fa4-compiled-checkpoint-alternating": ("fa4", "alternating", "compiled"),
    "fa4-compiled-checkpoint-none": ("fa4", "none", "compiled"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("correctness", "capacity"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--reference-arm", choices=tuple(ARMS), default="control")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--length", type=int, choices=(32, 512, 2048), default=512)
    parser.add_argument("--profile", action="store_true")
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
    if arm not in ARMS:
        raise ValueError("Unknown ordinary efficiency arm")
    attention, checkpoint, pointwise = ARMS[arm]
    base = model.backbone.backbone
    base.ordinary_attention_backend = attention
    base.ordinary_checkpoint_layers = checkpoint_layers(checkpoint, base.config.num_layers)
    base.ordinary_pointwise_backend = pointwise
    base.reuse_rope = True
    model.config = replace(model.config, ce_chunk_size=2048)
    if model.config.vocab_chunk_size != 128:
        raise ValueError("KL position chunks must remain 128")


def exact_comparison(reference, candidate):
    # Recomputing the same operations changes storage, not operation arithmetic.
    return ARMS[reference][::2] == ARMS[candidate][::2]


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


def dependency_record(directory, *, include_fa4):
    packages = {}
    for name in ("torch", "triton", "flash-attn", "flash-attn-4", "nvidia-cutlass-dsl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {"packages": packages, "fa4_sources": {}, "environment": {
        key: os.environ.get(key) for key in (
            "CDRM_FLASH_ATTENTION_SOURCE", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
            "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")}}
    if include_fa4:
        import flash_attn.cute.interface as interface
        package = Path(interface.__file__).resolve().parent
        record["fa4_interface"] = str(Path(interface.__file__).resolve())
        for path in sorted(package.rglob("*.py")):
            relative = str(path.relative_to(package))
            target = directory / "dependency-snapshot/flash_attn/cute" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record["fa4_sources"][relative] = {"source": str(path), "sha256": sha256_file(path)}
    return record


def check_dependencies(record):
    return all(sha256_file(Path(row["source"])) == row["sha256"]
               for row in record["fa4_sources"].values())


def dispatch_probe(plan, arm):
    """Untimed actual prepared-model backward with explicit no-fallback evidence."""
    from cdrm.pretrained import olmo_ordinary
    calls = {"fa4_loader": 0, "compiled_swiglu": 0}
    original_fa4 = olmo_ordinary._load_fa4
    original_compile = olmo_ordinary._compiled_swiglu

    def fa4():
        calls["fa4_loader"] += 1
        return original_fa4()

    def compiled():
        calls["compiled_swiglu"] += 1
        return original_compile()

    with ExitStack() as stack:
        stack.enter_context(patch.object(olmo_ordinary, "_load_fa4", fa4))
        stack.enter_context(patch.object(olmo_ordinary, "_compiled_swiglu", compiled))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            plan.backward(replay=False)
    operators = sorted({event.key for event in profile.key_averages()
                        if "scaled_dot_product" in event.key or "flash_attention" in event.key})
    expected_attention, _, expected_pointwise = ARMS[arm]
    flash_present = any("flash_attention" in name for name in operators)
    sdpa_present = any("scaled_dot_product" in name for name in operators)
    attention_ok = (calls["fa4_loader"] >= 16 and not sdpa_present) if expected_attention == "fa4" else (flash_present and calls["fa4_loader"] == 0)
    pointwise_ok = calls["compiled_swiglu"] >= 16 if expected_pointwise == "compiled" else calls["compiled_swiglu"] == 0
    observed = compiler_observations()
    if expected_pointwise == "compiled":
        pointwise_ok = pointwise_ok and observed.get("stats", {}).get("unique_graphs", 0) > 0 and not observed.get("graph_break", {})
    return {"name": "ordinary_dispatch_no_fallback", "passed": attention_ok and pointwise_ok,
        "calls": calls, "operators": operators, "compiler_counters": observed,
        "expected_attention": expected_attention, "expected_pointwise": expected_pointwise,
        "scope": "Actual 16-layer prepared-model eager backward, outside graph and timing; no GPU profiler."}


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
        "arm_switches": {name: dict(zip(("attention", "checkpointing", "pointwise"), options)) for name, options in ARMS.items()}}
    report = {"schema": "olmo-ordinary-efficiency-v1", "status": "running", "stage": "load",
        "configuration": configuration, "runtime": runtime, "determinism": determinism,
        "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "compiler_configuration": compiler, "started_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [], "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCES},
        "protocol_sha256": sha256_file(PROTOCOL), "physical_optimizer_updates": 0}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-ordinary-efficiency",
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
            include_fa4=any(ARMS[arm][0] == "fa4" for arm in (args.arm, args.reference_arm)))
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
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
                report["capture_seconds"] = time.perf_counter() - began
                exact_graph(plan, "candidate_initial_graph")
                plan.load_batch(batch_for(tokenizer, case, 1))
                exact_graph(plan, "candidate_changed_tokens_overwrite", replays=2)
                save("complete_update_parity")
                publish(complete_update_parity(plan, tokenizer, case, report=report,
                    persist=save, batch_factory=batch_for))
                plan.load_batch(batch_for(tokenizer, case, 2))
                exact_graph(plan, "candidate_changed_weights")
                report["memory"] = memory_snapshot()
                report["resources"] = resource_card(plan, case)
            else:
                set_arm(model, args.arm)
                plan = new_plan(model, batch, case.mode(), "recompute")
                publish(dispatch_probe(plan, args.arm))
                optimizer, scheduler = build_optimizer(model)
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
            raise AssertionError("Protocol or FA4 dependency changed during the run")
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
