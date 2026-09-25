#!/usr/bin/env python3
"""Bounded one-GPU objective/accumulation preparation; no distributed claim."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import math
import gc
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.distributed_training import ObjectiveForwardAdapter, sum_objective_counts
from cdrm.pretrained.lm_training import (
    TERMS, LMTrainingConfig, TrainingCounters, optimizer_step, optimizer_ownership,
    optimizer_state_bytes, _rng_state, _restore_rng,
)
from scripts.olmo_f1_common import boundary_digests
from scripts.olmo_rt_large_batch import (
    SOURCES as LARGE_BATCH_SOURCES, batch_for, compiler_configuration,
    dependency_record, check_dependencies, set_arm, optimizer_for, optimizer_flags,
    build_model, new_plan, selected_case, configure_determinism, require_container_gpu,
    validate_prepared_manifest, load_native_state_dict, load_native_tokenizer,
    backend_context, state_health, OnlineTracker, tree_digests,
    detailed_memory_snapshot, active_names,
)

PROTOCOL = ROOT / "docs/reports/olmo-distributed-prepare/protocol.md"
SOURCES = tuple(sorted(set(LARGE_BATCH_SOURCES) | {
    "scripts/olmo_distributed_prepare.py", "cdrm/pretrained/distributed_training.py",
}))
ACCUMULATION_BUDGETS = {"global_relative_l2": 2e-6,
                        "tensor_relative_l2": 2e-6, "tensor_max_relative": 1e-5}


@contextmanager
def preserve_rng():
    saved = _rng_state(None)
    try:
        yield
    finally:
        _restore_rng(saved, None)


@contextmanager
def disable_autocast_weight_cache():
    """Preserve forward/backward autocast boundaries while changing cache only."""
    previous = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(previous)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("rt", "combined"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--length", type=int, choices=(32, 512), default=512)
    parser.add_argument("--updates", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    return parser.parse_args(argv)


def unequal_batches(tokenizer, case, update=0):
    """Same physical shape; independent masks, with no auxiliary targets in #1."""
    first = batch_for(tokenizer, case, 2 * update)
    second = batch_for(tokenizer, case, 2 * update + 1)
    ce0, latent0, kl0 = [first.valid_mask.clone() for _ in range(3)]
    ce0[:, 2::3] = False
    latent0[:, 1::3] = False
    kl0[:, 2::4] = False
    ce1 = second.valid_mask.clone()
    ce1[:, 1::2] = False
    empty = torch.zeros_like(second.valid_mask)
    return [replace(first, ce_mask=ce0, latent_mask=latent0, kl_mask=kl0),
            replace(second, ce_mask=ce1, latent_mask=empty, kl_mask=empty.clone())]


def canonical_forward(model, batch, mode, global_counts):
    """Independent reference: use existing sums and explicit canonical formula."""
    result = model.loss_sums(batch, backbone_kwargs={"mode": mode})
    objective = sum(result.sums[t] * (result.weights[t] / global_counts[t])
                    for t in TERMS if global_counts[t] and result.weights[t])
    return {"objective": objective, "loss_sums": result.sums,
            "counts": result.counts, "global_counts": dict(global_counts),
            "objective_weights": result.weights,
            "pass_loss_sums": tuple(loss.sums for loss in result.pass_losses),
            "pass_coefficients": result.pass_coefficients}


def loss_record(output):
    return {"objective": output["objective"].detach().cpu().clone(),
            "loss_sums": {t: output["loss_sums"][t].detach().cpu().clone() for t in TERMS},
            "counts": dict(output["counts"]), "global_counts": dict(output["global_counts"]),
            "objective_weights": dict(output["objective_weights"]),
            "pass_loss_sums": tuple({t: loss[t].detach().cpu().clone() for t in TERMS}
                                    for loss in output["pass_loss_sums"]),
            "pass_coefficients": output["pass_coefficients"]}


def backward_sequence(model, batches, mode, global_counts, *, adapter=None):
    """Accumulate in the given order, retaining detached CPU losses only."""
    model.zero_grad(set_to_none=True)
    records = []
    device = next(model.parameters()).device
    for batch in batches:
        context = (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
                   if device.type == "cuda" else nullcontext())
        with context:
            output = (canonical_forward(model, batch, mode, global_counts) if adapter is None
                      else adapter(batch, global_counts=global_counts, world_size=1,
                                   backbone_kwargs={"mode": mode}))
        output["objective"].backward()
        records.append(loss_record(output))
        del output
    return records


def gradients_cpu(model):
    return {name: p.grad.detach().cpu().clone()
            for name, p in model.named_parameters() if p.grad is not None}


def adapter_optimizer_update(model, adapter, optimizer, scheduler, counters, batches, mode, *,
                             config=LMTrainingConfig(precision="bf16_mixed")):
    """Complete accumulated adapter update, compared to canonical optimizer_step.

    The small harness owns this eager loop; it is not a distributed trainer.
    No collectives or graph execution are inserted here.
    """
    optimizer_ownership(model, optimizer)
    expected_precision = "bf16_mixed" if next(model.parameters()).device.type == "cuda" else "fp32"
    if config.precision != expected_precision:
        raise ValueError("This bounded adapter-update harness uses BF16 mixed on CUDA or explicit FP32 CPU tests")
    denominators = sum_objective_counts([model.counts(batch) for batch in batches])
    weights = model.objective_weights()
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    records = backward_sequence(model, batches, mode, denominators, adapter=adapter)
    totals = {term: sum(float(record["loss_sums"][term]) for record in records) for term in TERMS}
    if not all(math.isfinite(value) for value in totals.values()):
        raise FloatingPointError("Nonfinite adapter loss sum before update")
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        float("inf") if config.max_grad_norm is None else config.max_grad_norm,
        error_if_nonfinite=True, foreach=False)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    counters.optimizer_updates += 1
    counters.microbatches += len(batches)
    counters.documents += sum(int(batch.valid_mask.any(-1).sum()) for batch in batches)
    counters.input_tokens += sum(int(batch.valid_mask.sum()) for batch in batches)
    counters.ce_positions += denominators["ce"]
    counters.latent_pairs += denominators["latent"]
    counters.kl_triples += denominators["kl"]
    means = {term: totals[term]/denominators[term] if denominators[term] else 0. for term in TERMS}
    return {"schema": "olmo-lm-optimizer-step-v1", "update_completed": True,
        "loss_sums": totals, "counts": denominators, "loss_means": means,
        "objective_weights": weights, "objective": sum(weights[t]*means[t] for t in TERMS),
        "gradient_norm_before_clip": float(norm), "max_grad_norm": config.max_grad_norm,
        "lr_used": learning_rates, "lr_next": [group["lr"] for group in optimizer.param_groups],
        "optimizer_state_bytes_by_device": optimizer_state_bytes(optimizer), "counters": asdict(counters)}


def update_equality(reference, actual, name):
    keys = ("metrics", "batches", "boundary")
    exact = {key: reference[key] == actual[key] for key in keys}
    return {"name": name, "passed": all(exact.values()), "bitwise_manifest_equal": exact,
            "scope": "Exact complete accumulated update: losses/counts, clipped Adam weights/moments, scheduler, counters, batches and RNG"}


def add_gradients_cpu(total, addition):
    """Explicit FP32 sum of completed same-shaped microbatch VJPs."""
    for name, value in addition.items():
        if name in total:
            total[name].add_(value)
        else:
            total[name] = value.clone()


def gradient_check(model, reference, *, exact):
    names = {name for name, p in model.named_parameters() if p.grad is not None}
    ownership = names == set(reference)
    rows, delta_sq, reference_sq, finite = {}, 0., 0., True
    for name, p in model.named_parameters():
        if p.grad is None or name not in reference:
            continue
        actual, expected = p.grad.detach().cpu(), reference[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(f"Gradient shape/dtype changed: {name}")
        a, b = actual.double(), expected.double()
        difference = a-b
        delta, norm = float(difference.square().sum()), float(b.square().sum())
        peak, max_error = float(b.abs().max()), float(difference.abs().max())
        relative_l2 = math.sqrt(delta/norm) if norm else (0. if not delta else float("inf"))
        max_relative = max_error/peak if peak else (0. if not max_error else float("inf"))
        is_finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        equal = torch.equal(actual, expected)
        rows[name] = {"bitwise_equal": equal, "finite": is_finite,
                      "relative_l2": relative_l2, "max_relative": max_relative,
                      "max_absolute": max_error, "delta_sq": delta, "reference_sq": norm,
                      "passed": is_finite and (equal if exact else (
                          relative_l2 <= ACCUMULATION_BUDGETS["tensor_relative_l2"]
                          and max_relative <= ACCUMULATION_BUDGETS["tensor_max_relative"]))}
        delta_sq += delta
        reference_sq += norm
        finite &= is_finite
    global_l2 = math.sqrt(delta_sq/reference_sq) if reference_sq else (0. if not delta_sq else float("inf"))
    return {"passed": ownership and bool(rows) and finite and all(row["passed"] for row in rows.values())
                      and (exact or global_l2 <= ACCUMULATION_BUDGETS["global_relative_l2"]),
            "ownership_matches": ownership, "finite": finite, "exact_required": exact,
            "all_bitwise_equal": ownership and all(row["bitwise_equal"] for row in rows.values()),
            "global_gradient_relative_l2": global_l2, "gradients": rows,
            "budgets": None if exact else dict(ACCUMULATION_BUDGETS)}


def expected_ownership(model, expected_names):
    actual = {name for name, p in model.named_parameters() if p.grad is not None}
    return {"expected_participation_matches": actual == set(expected_names),
            "missing_expected_gradients": sorted(set(expected_names)-actual),
            "unexpected_gradients": sorted(actual-set(expected_names))}


def check_snapshot(model, reference_gradients, expected_losses, actual_losses, *, name, exact=True,
                   expected_names=None):
    check = gradient_check(model, reference_gradients, exact=exact)
    participation = {} if expected_names is None else expected_ownership(model, expected_names)
    reference_losses, candidate_losses = tree_digests(expected_losses), tree_digests(actual_losses)
    losses_equal = reference_losses == candidate_losses
    finite_losses = all(bool(torch.isfinite(value))
        for records in (expected_losses, actual_losses) for record in records
        for value in (record["objective"], *record["loss_sums"].values(),
                      *(value for loss in record["pass_loss_sums"] for value in loss.values())))
    return {**check, **participation, "name": name, "losses_and_counts_exact": losses_equal,
            "losses_finite": finite_losses,
            "reference_losses": reference_losses, "candidate_losses": candidate_losses,
            "passed": check["passed"] and losses_equal and finite_losses
                      and participation.get("expected_participation_matches", True)}


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
        target = args.output_dir / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    case = selected_case(args.case, batch=args.batch_size, length=args.length)
    config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "case_specification": asdict(case), "mode": asdict(case.mode()),
              "world_size": 1, "real_distributed_execution": False,
              "precision": "bf16_mixed", "parameter_optimizer_dtype": "float32",
              "arm": "compiled-native", "rt_backend": "native_triton_recompute",
              "ordinary_attention": "deterministic_flash_sdpa", "ordinary_rope": "native",
              "ordinary_pointwise": "compiled", "ordinary_checkpointing": "all",
              "reuse_rope": True, "kv_only_writes": True, "cast_weights_once": True,
              "ce_chunk_size": 2048, "kl_chunk_size": 128, "autocast_cache": False,
              "tf32": False, "cuda_graphs": False, "seed": 20260922,
              "update_microbatches": 2, "update_global_batch": 2*args.batch_size,
              "complete_update_branches": ["reference", "adapter"],
              "physical_updates_expected": 2*args.updates, "logical_endpoint_updates": args.updates,
              "accumulation_budgets": ACCUMULATION_BUDGETS,
              "scope": "Single GPU objective/gradient/accumulation preparation, not quality or throughput"}
    report = {"schema": "olmo-distributed-prepare-v1", "status": "running", "stage": "load",
              "configuration": config, "runtime": runtime, "determinism": determinism,
              "compiler_configuration": compiler, "checks": [], "physical_optimizer_updates": 0,
              "branch_physical_optimizer_updates": {"reference": 0, "adapter": 0},
              "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_hashes": {name: sha256_file(ROOT/name) for name in SOURCES},
              "protocol_sha256": sha256_file(PROTOCOL), "started_utc": datetime.now(timezone.utc).isoformat()}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-distributed-prepare",
                            name=args.output_dir.name, output_dir=args.output_dir, preserve_state=preserve_rng)
    hooks = []

    def save(stage=None):
        if stage is not None:
            report["stage"] = stage
            print({"stage": stage, "utc": datetime.now(timezone.utc).isoformat()}, flush=True)
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)

    def publish(check):
        report["checks"].append(check)
        save()
        tracker.log({"correctness/passed": int(check["passed"])})
        print({"check": check["name"], "passed": check["passed"]}, flush=True)
        if not check["passed"]:
            raise AssertionError(check["name"])

    try:
        save()
        report["dependencies"] = dependency_record(args.output_dir, include_dao=False, include_fa4=False)
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": config, "checkpoint": report["checkpoint"]})
        save()
        print({"wandb": tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        model = build_model(state, case)
        del state
        set_arm(model, "compiled-native")
        full = batch_for(tokenizer, case, 0).to("cuda")
        # Reuse the established configuration helper, without preparing or
        # capturing any graph or retaining its fixed batch ownership.
        configuration_plan = new_plan(model, full, case.mode(), "recompute")
        del configuration_plan
        adapter = ObjectiveForwardAdapter(model)
        expected_names = active_names(model, case.mode())
        report["expected_active_parameter_names"] = sorted(expected_names)
        batches = [batch.to("cuda") for batch in unequal_batches(tokenizer, case)]
        counts = sum_objective_counts([model.counts(batch) for batch in batches])
        report["batches"] = {"single": tree_digests(vars(full)),
                             "microbatches": [tree_digests(vars(batch)) for batch in batches],
                             "microbatch_counts": [model.counts(batch) for batch in batches],
                             "global_counts": counts}
        with backend_context("flash"):
            save("single_microbatch_world1")
            local_counts = model.counts(full)
            expected = backward_sequence(model, [full], case.mode(), local_counts)
            reference = gradients_cpu(model)
            actual = backward_sequence(model, [full], case.mode(), local_counts, adapter=adapter)
            publish(check_snapshot(model, reference, expected, actual, name="world1_adapter_exact",
                                   expected_names=expected_names))
            del reference, expected, actual, full

            save("independent_microbatch_references")
            independent, independent_losses = {}, []
            for index, batch in enumerate(batches):
                independent_losses.extend(backward_sequence(model, [batch], case.mode(), counts))
                gradients = gradients_cpu(model)
                add_gradients_cpu(independent, gradients)
                del gradients
                report["independent_reference_microbatches_completed"] = index+1
                save()

            save("canonical_accumulation")
            expected = backward_sequence(model, batches, case.mode(), counts)
            reference = gradients_cpu(model)
            save("adapter_accumulation")
            actual = backward_sequence(model, batches, case.mode(), counts, adapter=adapter)
            publish(check_snapshot(model, reference, expected, actual, name="same_order_accumulation_exact",
                                   expected_names=expected_names))
            del reference
            publish(check_snapshot(model, independent, independent_losses, actual,
                                   name="independent_cpu_vjp_sum", exact=False, expected_names=expected_names))
            del independent, independent_losses, expected, actual, batches

            save("complete_update_reference")
            initial_state = {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}
            initial_rng = _rng_state(None)
            initial_state_digest = tree_digests(initial_state)
            report["initial_update_boundary"] = {"model": initial_state_digest, "rng": tree_digests(initial_rng)}
            training = LMTrainingConfig(precision="bf16_mixed", max_grad_norm=1.)
            for branch in ("reference", "adapter"):
                if branch == "adapter":
                    model.load_state_dict(initial_state, strict=True)
                    model.zero_grad(set_to_none=True)
                    del initial_state
                    gc.collect(); torch.cuda.empty_cache()
                    _restore_rng(initial_rng, None)
                    restored = {"model": tree_digests(model.state_dict()), "rng": tree_digests(_rng_state(None))}
                    publish({"name": "restored_initial_update_boundary", "passed": restored == report["initial_update_boundary"],
                             "bitwise_manifest_equal": {key: restored[key] == report["initial_update_boundary"][key]
                                                        for key in restored}})
                optimizer, scheduler = optimizer_for(model, "compiled-native")
                report["optimizer_flags"] = optimizer_flags(optimizer)
                counters = TrainingCounters()
                report[branch+"_records"] = []

                def validate_step(_optimizer, _args, _kwargs):
                    participation = expected_ownership(model, expected_names)
                    publish({"name": f"{branch}_update_{counters.optimizer_updates+1}_gradient_ownership",
                             **participation, "passed": participation["expected_participation_matches"]})

                def completed_step(_optimizer, _args, _kwargs):
                    report["physical_optimizer_updates"] += 1
                    report["branch_physical_optimizer_updates"][branch] += 1
                    save()

                hooks = [optimizer.register_step_pre_hook(validate_step), optimizer.register_step_post_hook(completed_step)]
                save(branch+"_complete_updates")
                for update in range(args.updates):
                    batches = [batch.to("cuda") for batch in unequal_batches(tokenizer, case, update+1)]
                    probe = model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096].clone()
                    # Match the cache policy in both forward and any nested
                    # RT backward replay, without enabling outer autocast.
                    with disable_autocast_weight_cache():
                        if branch == "reference":
                            metrics = optimizer_step(model, optimizer, batches, config=training,
                                backbone_kwargs={"mode": case.mode()}, scheduler=scheduler, counters=counters)
                        else:
                            metrics = adapter_optimizer_update(model, adapter, optimizer, scheduler, counters,
                                                               batches, case.mode(), config=training)
                    changed = not torch.equal(probe, model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096])
                    record = {"metrics": metrics, "batches": [tree_digests(vars(batch)) for batch in batches],
                        "boundary": {"state": boundary_digests(model, optimizer, scheduler, counters),
                                     "rng": tree_digests(_rng_state(None))},
                        "weights_changed": changed, "state_health": state_health(model, optimizer)}
                    report[branch+"_records"].append(record)
                    save()
                    tracker.log({"update": report["physical_optimizer_updates"],
                                 f"{branch}/gradient_norm_before_clip": metrics["gradient_norm_before_clip"],
                                 **{f"{branch}/{t}": metrics["loss_means"][t] for t in TERMS}})
                    publish({"name": f"{branch}_update_{update+1}_health", **record["state_health"],
                             "weights_changed": changed, "passed": changed and record["state_health"]["passed"]})
                    if branch == "adapter":
                        publish(update_equality(report["reference_records"][update], record,
                                                f"accumulated_update_{update+1}_exact"))
                for hook in hooks: hook.remove()
                hooks = []
                del optimizer, scheduler, counters, batches
                model.zero_grad(set_to_none=True)
                gc.collect(); torch.cuda.empty_cache()
            publish({"name": "physical_update_accounting",
                "physical_updates": report["physical_optimizer_updates"], "logical_endpoint_updates": args.updates,
                "passed": report["physical_optimizer_updates"] == 2*args.updates
                    and report["branch_physical_optimizer_updates"] == {"reference": args.updates, "adapter": args.updates}})
            report["memory"] = detailed_memory_snapshot()
        if any(sha256_file(ROOT/name) != digest for name, digest in report["source_hashes"].items()):
            raise AssertionError("Runtime source changed during the run")
        if sha256_file(PROTOCOL) != report["protocol_sha256"] or not check_dependencies(report["dependencies"]):
            raise AssertionError("Protocol or dependency changed during the run")
        report["status"] = "passed"
        tracker.summary({"result_status": report["status"],
                         "physical_optimizer_updates": report["physical_optimizer_updates"],
                         "real_distributed_execution": False})
        save("complete")
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
                      error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save()
        raise
    finally:
        for hook in hooks:
            hook.remove()
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        except BaseException as error:
            report["tracking_finish_error"] = {"type": type(error).__name__, "message": str(error)}
            if report["status"] == "passed":
                report["status"] = "failed"
            raise
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()


if __name__ == "__main__":
    main()
