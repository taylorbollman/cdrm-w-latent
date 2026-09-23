#!/usr/bin/env python3
"""Bounded native multi-layer RT integration and complete-update resource checks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.resource_estimates import (
    LossWork, estimate_training_resources, parameter_inventory,
)
from scripts.olmo_f1_common import active_names, inference_names
from scripts.olmo_f3d_validate import (
    SOURCES as F3D_SOURCES, global_gradient_l2, new_plan,
)
from scripts.olmo_f3_graph_training import (
    case_for, changed_batch, build_optimizer, build_model, state_health,
    compare_graph, loss_snapshot, full_update_parity, timed, configure_determinism,
    require_container_gpu, validate_prepared_manifest, load_native_state_dict,
    load_native_tokenizer, backend_context, tensor_comparison, OnlineTracker,
)

LAYOUTS = {"single": (0,), "adjacent2": (0, 1), "spread2": (0, 15),
           "spread4": (0, 5, 10, 15), "all16": tuple(range(16))}
SOURCES = tuple(sorted(set(F3D_SOURCES) | {
    "scripts/olmo_f3e_validate.py", "cdrm/pretrained/resource_estimates.py"}))
PROTOCOL = ROOT / "docs/reports/olmo1b-f3e/protocol.md"
EVENTS = ("forward_blocks", "forward_tiles", "forward_fused_tiles", "forward_eager_tiles", "backward_blocks", "recompute_tiles",
          "materialized_tiles", "materialized_fused_tiles")


def selected_case(name, layout, *, batch, length):
    if layout not in LAYOUTS:
        raise ValueError("Unknown F3e RT layout")
    if name not in ("rt", "combined", "combined-k3"):
        raise ValueError("Unknown F3e case")
    return replace(case_for(name, batch=batch, length=length), rt_layers=LAYOUTS[layout])


def layout_role(layout):
    if layout not in LAYOUTS:
        raise ValueError("Unknown F3e RT layout")
    return "optional_all_layer_stress" if layout == "all16" else (
        "single_layer_reference" if layout == "single" else "primary_multi_layer_integration")


def expected_dispatch(length, mode, variant):
    if variant not in ("reference", "recompute"):
        raise ValueError("Unknown backward memory variant")
    passes = mode.num_passes - 1 if mode.enabled else 1
    # The existing materialized kernel deliberately falls back above 256x256.
    eligible = sum((index & -index) <= 256 and
                   min(length - index, index & -index) <= 256
                   for index in range(1, length))
    per_layer = {
        str(index): {"forward_blocks": passes, "backward_blocks": passes,
            "forward_tiles": passes * (length - 1), "forward_fused_tiles": passes * eligible,
            "forward_eager_tiles": passes * (length - 1 - eligible),
            "recompute_tiles": passes * (length - 1) if variant == "recompute" else 0,
            "materialized_tiles": passes * (length - 1) if variant == "reference" else 0,
            "materialized_fused_tiles": passes * eligible if variant == "reference" else 0}
        for index in mode.rt_mode.selected_layers}
    return {"by_layer": per_layer,
            "totals": {key: sum(row[key] for row in per_layer.values()) for key in EVENTS}}


@contextmanager
def count_layer_dispatch(model, selected_layers):
    """Count eager call sites without altering operands or graph-capture code.

    The custom autograd context carries the layer identity from forward to its
    backward. This is necessary for K3, whose backward order is not the forward
    layer order. No tensor values are read and the wrappers restore on failure.
    """
    from cdrm.pretrained import olmo_tiled, olmo_rt_kernels, olmo_rt_backward_kernels, olmo_rt_recompute_kernels
    selected = tuple(selected_layers)
    indices = {id(layer.att_proj.weight): index
               for index, layer in enumerate(model.backbone.backbone.layers)}
    observed = {"by_layer": {str(index): dict.fromkeys(EVENTS, 0) for index in selected},
                "totals": dict.fromkeys(EVENTS, 0)}
    active = []
    original_forward = olmo_tiled._TiledRecurrence.forward
    original_backward = olmo_tiled._TiledRecurrence.backward

    def record(index, event):
        if str(index) not in observed["by_layer"]:
            raise AssertionError(f"Unexpected RT layer dispatch: {index}")
        observed["by_layer"][str(index)][event] += 1
        observed["totals"][event] += 1

    def forward(ctx, x, wq, *args):
        index = indices[id(wq)]
        ctx._f3e_observed_layer = index
        record(index, "forward_blocks")
        active.append(index)
        try:
            return original_forward(ctx, x, wq, *args)
        finally:
            active.pop()

    def backward(ctx, *args):
        index = ctx._f3e_observed_layer
        record(index, "backward_blocks")
        active.append(index)
        try:
            return original_backward(ctx, *args)
        finally:
            active.pop()

    def counter(function, event):
        def counted(*args, **kwargs):
            if not active:
                raise AssertionError("Historical tile dispatch outside an observed RT block")
            record(active[-1], event)
            return function(*args, **kwargs)
        return counted

    original_add_tile = olmo_tiled._add_tile

    def add_tile(*args, **kwargs):
        if not active:
            raise AssertionError("Forward tile dispatch outside an observed RT block")
        index = active[-1]
        record(index, "forward_tiles")
        previous_fused = observed["by_layer"][str(index)]["forward_fused_tiles"]
        result = original_add_tile(*args, **kwargs)
        if observed["by_layer"][str(index)]["forward_fused_tiles"] == previous_fused:
            record(index, "forward_eager_tiles")
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(olmo_tiled._TiledRecurrence, "forward", staticmethod(forward)))
        stack.enter_context(patch.object(olmo_tiled._TiledRecurrence, "backward", staticmethod(backward)))
        stack.enter_context(patch.object(olmo_tiled, "_add_tile", add_tile))
        for module, name, event in (
            (olmo_rt_kernels, "add_tile", "forward_fused_tiles"),
            (olmo_tiled, "_historical_backward_tile", "materialized_tiles"),
            (olmo_rt_backward_kernels, "backward_tile", "materialized_fused_tiles"),
            (olmo_rt_recompute_kernels, "backward_recomputed_tile", "recompute_tiles"),
        ):
            stack.enter_context(patch.object(module, name, counter(getattr(module, name), event)))
        yield observed


def comparison_metrics(candidate, reference):
    candidate, reference = candidate.double(), reference.double()
    delta = candidate - reference
    error = float(delta.square().sum())
    norm = float(reference.square().sum())
    peak = float(reference.abs().max())
    difference = float(delta.abs().max())
    return {"delta_sq": error, "reference_sq": norm,
        "relative_l2": global_gradient_l2([{"delta_sq": error, "reference_sq": norm}]),
        "max_relative": difference / peak if peak else (0. if difference == 0 else float("inf"))}


def compare_variant(model, batch, mode, variant):
    plan = new_plan(model, batch, mode, "reference")
    plan.initialize_gradients()
    with count_layer_dispatch(model, mode.rt_mode.selected_layers) as reference_dispatch:
        reference = plan.backward(replay=False)
    losses = loss_snapshot(reference)
    gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    del plan, reference
    model.zero_grad(set_to_none=True)
    plan = new_plan(model, batch, mode, variant)
    plan.initialize_gradients()
    with count_layer_dispatch(model, mode.rt_mode.selected_layers) as candidate_dispatch:
        candidate = plan.backward(replay=False)
    candidate_losses = loss_snapshot(candidate)
    loss_checks = {n: tensor_comparison(value, losses[n]) for n, value in candidate_losses.items()}
    gradient_checks = {n: tensor_comparison(p.grad, gradients[n])
                       for n, p in model.named_parameters() if p.grad is not None}
    ownership = set(gradient_checks) == set(gradients) == set(plan.active_names)
    rows = [*loss_checks.values(), *gradient_checks.values()]
    expected_reference = expected_dispatch(batch.input_ids.shape[1], mode, "reference")
    expected_candidate = expected_dispatch(batch.input_ids.shape[1], mode, variant)
    dispatch_matches = reference_dispatch == expected_reference and candidate_dispatch == expected_candidate
    forward_equal = all(row["bitwise_equal"] for row in loss_checks.values())
    mixed = {n: comparison_metrics(p.grad, gradients[n]) for n, p in model.named_parameters() if p.grad is not None}
    mixed_losses = {n: comparison_metrics(value, losses[n]) for n, value in candidate_losses.items()}
    global_l2 = global_gradient_l2(mixed.values())
    if variant == "recompute":
        passed = (ownership and dispatch_matches and forward_equal and global_l2 <= 1 / 64
            and all(row["relative_l2"] <= 1 / 32 and row["max_relative"] <= 1 / 16 for row in mixed.values())
            and all(row["relative_l2"] <= 1 / 64 for row in mixed_losses.values()))
    else:
        passed = ownership and dispatch_matches and all(row["passed"] for row in rows)
    report = {"name": "same_state_variant_vs_reference", "passed": passed,
        "all_bitwise_equal": all(row["bitwise_equal"] for row in rows),
        "ownership_matches": ownership, "forward_losses_bitwise_equal": forward_equal,
        "losses": loss_checks, "gradients": gradient_checks,
        "reference_dispatch": reference_dispatch, "candidate_dispatch": candidate_dispatch,
        "expected_reference_dispatch": expected_reference, "expected_candidate_dispatch": expected_candidate,
        "reference_recompute_backward_calls": reference_dispatch["totals"]["recompute_tiles"],
        "candidate_recompute_backward_calls": candidate_dispatch["totals"]["recompute_tiles"],
        "expected_candidate_recompute_backward_calls": expected_candidate["totals"]["recompute_tiles"],
        "dispatch_matches": dispatch_matches, "mixed_loss_screen": mixed_losses,
        "mixed_gradient_screen": mixed, "global_gradient_relative_l2": global_l2}
    return plan, report


def resource_card(plan, case, *, optimizer=None):
    base = plan.model.backbone.backbone
    work = LossWork(ce_targets=plan.counts["ce"], latent_pairs=plan.counts["latent"],
                    kl_triples=plan.counts["kl"],
                    predictor_positions=plan.loss_layout.needed_source_indices.numel())
    estimate = estimate_training_resources(base.config, batch_size=case.batch_size,
        sequence_length=case.length, mode=plan.mode,
        nextlat=plan.model.config if case.nextlat else None, loss_work=work,
        ordinary_checkpointing=True, backward_memory=base.backward_memory)
    executed = active_names(plan.model, plan.mode)
    inference = inference_names(plan.model, case)
    return {"analytic_matrix_work": estimate.to_dict(),
        "observed_parameters": parameter_inventory(plan.model, optimizer=optimizer,
            executed_names=executed, inference_names=inference),
        "named_parameter_shapes": {name: list(parameter.shape) for name, parameter in plan.model.named_parameters()},
        "declared_execution_names": sorted(executed), "declared_inference_names": sorted(inference),
        "loss_work": asdict(work), "loss_weights": dict(plan.weights),
        "optimizer_inventory_scope": "observed_capacity_optimizer" if optimizer is not None else
            "unavailable_in_correctness_card_after_independent_update_parity_helper_returns",
        "scope": "Matrix arithmetic estimate, not measured hardware FLOPs or an optimizer/elementwise-work total."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("rt", "combined", "combined-k3"), default="combined")
    parser.add_argument("--layout", choices=tuple(LAYOUTS), default="adjacent2")
    parser.add_argument("--variant", choices=("reference", "recompute"), default="recompute")
    parser.add_argument("--stage", choices=("correctness", "capacity"), default="correctness")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, choices=(32, 128, 512, 1024, 2048), default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 128:
        parser.error("bounded batch 1..128")
    if args.case == "combined-k3" and args.stage != "correctness":
        parser.error("combined-k3 is correctness-only in F3e")
    return args


def main(argv=None):
    args = parse_args(argv)
    case = selected_case(args.case, args.layout, batch=args.batch_size, length=args.length)
    mode = case.mode()
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
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
    configuration = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "forward_tile_backend": "triton", "cast_weights_once": True,
        "backward_tile_backend": "triton",
        "backward_memory": "recompute" if args.variant == "recompute" else "materialized",
        "case_specification": asdict(case), "mode": asdict(mode),
        "selected_rt_layers": list(case.rt_layers), "layout_role": layout_role(args.layout),
        "rt_block_calls_per_forward_backward": len(case.rt_layers) * (mode.num_passes - 1 if mode.enabled else 1),
        "ordinary_activation_checkpointing": True, "ordinary_attention_backend": "deterministic_flash",
        "precision": "bf16_mixed", "tf32": False, "autocast_weight_cache": False}
    report = {"schema": "olmo-f3e-native-v1", "status": "running", "stage": args.stage,
        "configuration": configuration, "runtime": {k: str(v) for k, v in runtime.items()},
        "determinism": determinism, "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
        "source_hashes": {p: sha256_file(ROOT / p) for p in SOURCES}, "protocol_sha256": sha256_file(PROTOCOL)}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f3e-multi-rt", name="olmo-" + args.output_dir.name)

    def publish(check):
        report["checks"].append(check)
        write_json(args.output_dir / "report.json", report)
        print({"check": check["name"], "passed": check["passed"], "exact": check.get("all_bitwise_equal")}, flush=True)
        tracker.log({"correctness/passed": int(check["passed"])}, step=len(report["checks"]))
        if not check["passed"]:
            raise AssertionError(check["name"])

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        model = build_model(state, case)
        batch = changed_batch(tokenizer, case, 0)
        with backend_context("flash"):
            if args.stage == "correctness":
                plan, comparison = compare_variant(model, batch, mode, args.variant)
                publish(comparison)
                plan.capture(warmup=10)
                publish(compare_graph(plan, "candidate_initial_graph"))
                plan.load_batch(changed_batch(tokenizer, case, 1))
                publish(compare_graph(plan, "candidate_changed_tokens_and_overwrite", replays=2))
                publish(full_update_parity(plan, tokenizer, case, updates=3))
                plan.load_batch(changed_batch(tokenizer, case, 2))
                publish(compare_graph(plan, "candidate_changed_weights"))
                report["resources"] = resource_card(plan, case)
            else:
                plan = new_plan(model, batch, mode, args.variant)
                optimizer, scheduler = build_optimizer(model)
                counters = TrainingCounters()
                for step in range(3):
                    plan.optimizer_step(optimizer, changed_batch(tokenizer, case, step), scheduler=scheduler, counters=counters)
                plan.capture(warmup=10)
                batches = [changed_batch(tokenizer, case, i + 3) for i in range(3)]
                records = []

                def complete():
                    records.append(plan.optimizer_step(optimizer, batches[len(records)], replay=True,
                                                       scheduler=scheduler, counters=counters))

                timing = timed(complete, 3)
                resources = resource_card(plan, case, optimizer=optimizer)
                seconds = timing["median_wall_seconds"]
                analytic = resources["analytic_matrix_work"]
                report["resources"] = resources
                report["capacity"] = {"full_step": timing, "input_tokens_per_second": plan.input_tokens / seconds,
                    "ce_targets_per_second": plan.counts["ce"] / seconds,
                    "estimated_matrix_tflops_per_second_minimum": analytic["matrix_flops_minimum"] / seconds / 1e12,
                    "estimated_matrix_tflops_per_second_maximum": analytic["matrix_flops_maximum"] / seconds / 1e12,
                    "records": records, "warmup_updates": 3, "timed_updates": 3,
                    "physical_optimizer_updates": 6, "backward_only_warmup": 10,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "current_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "health": state_health(model, optimizer),
                    "scope": "Input copy + graph forward/loss/backward + clip + AdamW + scheduler; one physical batch."}
                publish({"name": "finite_complete_updates", "passed": report["capacity"]["health"]["passed"]})
                tracker.log({"capacity/input_tokens_per_second": report["capacity"]["input_tokens_per_second"],
                    "capacity/peak_allocated_gib": report["capacity"]["peak_allocated_gib"]}, step=len(report["checks"]) + 1)
                print(report["capacity"], flush=True)
            report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
                "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
            report["memory"] = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "current_reserved_gib": torch.cuda.memory_reserved() / 2**30}
        assert report["source_hashes"] == {p: sha256_file(ROOT / p) for p in SOURCES}
        assert report["protocol_sha256"] == sha256_file(PROTOCOL)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        tracker.finish(succeeded=report["status"] == "passed")
        report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
