#!/usr/bin/env python3
"""Bounded author RT integration in native 16-layer OLMo and K2 NextLat."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.lm_training import TrainingCounters
from cdrm.pretrained.resource_estimates import LossWork, estimate_training_resources, parameter_inventory
from cdrm.pretrained.rt_block_resources import estimate_rt_block_resources
from cdrm.pretrained.olmo_author import AUTHOR_SOURCE, COMPILED_HELPER_BOUNDARIES
from scripts.olmo_f1_common import active_names, inference_names
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_f3_graph_training import (
    changed_batch, build_model, build_optimizer, state_health, compare_graph, loss_snapshot, timed,
    configure_determinism, require_container_gpu, validate_prepared_manifest, load_native_state_dict,
    load_native_tokenizer, backend_context, OnlineTracker,
)
from scripts.olmo_f3d_validate import new_plan
from scripts.olmo_f4_resources import selected_case, memory_snapshot
from scripts.olmo_rt_efficiency import (batch_for, set_arm, output_snapshot, metric, comparison_passes,
    BUDGETS, complete_update_parity, track_optimizer_steps)
from scripts.olmo_rt_author_compare import SOURCES as BLOCK_SOURCES, configure_compiler, compiler_audit

PROTOCOL = ROOT / "docs/reports/olmo-rt-author-integration/protocol.md"
SOURCES = tuple(sorted(set(BLOCK_SOURCES) | {"scripts/olmo_rt_author_integration.py"}))
SEED = 20260922


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "capacity"), required=True)
    parser.add_argument("--case", choices=("rt", "combined"), required=True)
    parser.add_argument("--backend", choices=("native", "author"), default="author")
    parser.add_argument("--batch-size", type=int, choices=(8, 32, 64), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    args = parser.parse_args(argv)
    if args.stage == "verify" and (args.batch_size != 8 or args.backend != "author"):
        parser.error("Verify compares native/author at B8/T512 and clears author graph execution")
    if args.stage == "capacity" and args.batch_size not in (32, 64):
        parser.error("Capacity is bounded to common B32/64 at T512")
    return args


def set_backend(model, backend, *, compiled_helpers=True):
    if backend not in ("native", "author"):
        raise ValueError("Unknown RT implementation")
    set_arm(model, "both")
    base = model.backbone.backbone
    base.rt_implementation = backend
    base.author_precision = "author_legacy"
    base.author_compiled_helpers = compiled_helpers
    base.author_bwd_mlp_chunks = 4
    base.author_autocast_cache = True


def parameter_signature(model):
    return tuple((name, id(parameter), parameter.data_ptr(), tuple(parameter.shape), parameter.dtype,
                  parameter.device, parameter.requires_grad) for name, parameter in model.named_parameters())


def full_batch(tokenizer, case, update):
    return batch_for(tokenizer, case, update, "full")


def compare_backends(model, batch, mode):
    original_signature = parameter_signature(model)
    set_backend(model, "native")
    reference_plan = new_plan(model, batch, mode, "recompute")
    outputs = output_snapshot(reference_plan)
    reference = reference_plan.backward(replay=False)
    losses = loss_snapshot(reference)
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    counts = dict(reference_plan.counts)
    del reference_plan, reference
    model.zero_grad(set_to_none=True)
    set_backend(model, "author")
    plan = new_plan(model, batch, mode, "recompute")
    candidate_outputs = output_snapshot(plan)
    candidate = plan.backward(replay=False)
    candidate_losses = loss_snapshot(candidate)
    candidate_names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    ownership = candidate_names == set(gradients) == set(plan.active_names)
    gradients_check = {name: metric(parameter.grad, gradients[name])
        for name, parameter in model.named_parameters() if parameter.grad is not None and name in gradients}
    losses_check = {name: metric(value, losses[name]) for name, value in candidate_losses.items()}
    outputs_check = {name: metric(value, outputs[name]) for name, value in candidate_outputs.items()}
    finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
                 if parameter.grad is not None)
    finite = finite and all(bool(torch.isfinite(value).all())
                            for value in (*candidate_losses.values(), *candidate_outputs.values(),
                                          *losses.values(), *outputs.values(), *gradients.values()))
    counts_equal = counts == plan.counts
    identity = parameter_signature(model) == original_signature
    screen = comparison_passes(exact_required=False, ownership=ownership and identity, finite=finite,
        counts_equal=counts_equal, losses=losses_check, gradients=gradients_check, outputs=outputs_check)
    return plan, {"name": "author_vs_native_bf16_mixed", "gate": True, **screen,
        "ownership_matches": ownership, "parameter_identity_preserved": identity,
        "finite": finite, "counts_equal": counts_equal, "budgets": dict(BUDGETS),
        "losses": losses_check, "outputs": outputs_check, "gradients": gradients_check,
        "scope": "Same checkpoint/fusion/predictor weights, losses and examples; pre-clipping raw gradients. Native both/recompute versus author_legacy/materialized."}


def ordinary_flash_probe(model, tokenizer, case):
    """Real mixed-stack dispatch at B1/T8, separately from any timed execution."""
    small_case = replace(case, batch_size=1, length=8)
    model.zero_grad(set_to_none=True)
    plan = new_plan(model, changed_batch(tokenizer, small_case, 0), small_case.mode(), "recompute")
    # Gradient initialization is outside the observer; observe a complete second
    # eager backward through the same canonical stack/loss dispatch.
    plan.initialize_gradients()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        plan.backward(replay=False)
        torch.cuda.synchronize()
    names = sorted({event.key for event in profile.key_averages()
                    if "scaled_dot_product" in event.key or "flash_attention" in event.key})
    forward = "aten::_scaled_dot_product_flash_attention" in names
    backward = "aten::_scaled_dot_product_flash_attention_backward" in names
    other = [name for name in names if any(key in name for key in ("_efficient_attention", "_cudnn_attention", "_attention_math"))]
    del plan
    model.zero_grad(set_to_none=True)
    return {"name": "ordinary_flash_dispatch", "gate": True,
        "passed": forward and backward and not other, "operators": names,
        "flash_forward_observed": forward, "flash_backward_observed": backward,
        "unexpected_other_attention": other, "batch_size": 1, "length": 8,
        "scope": "Separate CPU operator observer around real BF16 mixed native stack backward; author selected layers0/15, unchanged ordinary Flash layers/bootstrap. Never timed."}


def integration_estimate(config, *, batch_size, length, mode, nextlat, work, backend):
    """Replace only RT matrix components; preserve ordinary/objective bounds."""
    if backend not in ("native", "author"):
        raise ValueError("Unknown integrated RT ledger backend")
    original = estimate_training_resources(config, batch_size=batch_size, sequence_length=length,
        mode=mode, nextlat=nextlat, loss_work=work, ordinary_checkpointing=True,
        backward_memory="recompute", kv_only_writes=True).to_dict()
    result = dict(original)
    result["rt_implementation"] = backend
    result["rt_backward_memory"] = "recompute" if backend == "native" else "materialized"
    if backend == "native":
        return result
    per_call = estimate_rt_block_resources(replace(config, num_layers=1), batch_size=batch_size,
        sequence_length=length, backend="author")
    calls = original["rt_block_calls_per_microbatch"]
    removed = [component for component in original["components"] if component["name"].startswith("rt_")]
    components = [component for component in original["components"] if not component["name"].startswith("rt_")]
    for phase in ("forward", "backward"):
        for kind in ("dense", "attention"):
            value = calls * per_call["matrix_breakdown"][phase][kind]
            components.append({"name": f"author_rt_{kind}_{phase}", "minimum": value, "maximum": value,
                "description": "Audited author block schedule multiplied by actual shared RT calls; no duplicated parameter ownership."})
    result.update(components=components, matrix_flops_minimum=sum(row["minimum"] for row in components),
        matrix_flops_maximum=sum(row["maximum"] for row in components), backward_memory="materialized",
        assumptions=[text for text in original["assumptions"] if "dyadic RT/custom VJP" not in text] + [
            "Selected RT calls use the audited author-derived matrix schedule; other layers and objectives retain native accounting.",
            "Repeated FBT calls multiply RT arithmetic, not unique parameter counts."],
        rt_component_replacement={"calls": calls, "removed_native_components": removed,
            "per_call_author_ledger": per_call,
            "parameter_accounting": "Use outer model parameter_counts/observed inventory; inner one-block parameters are descriptive only."})
    return result


def resource_card(plan, case, backend, optimizer=None):
    base = plan.model.backbone.backbone
    work = LossWork(ce_targets=plan.counts["ce"], latent_pairs=plan.counts["latent"],
        kl_triples=plan.counts["kl"], predictor_positions=plan.loss_layout.needed_source_indices.numel())
    estimate = integration_estimate(base.config, batch_size=case.batch_size, length=case.length, mode=plan.mode,
        nextlat=plan.model.config if case.nextlat else None, work=work, backend=backend)
    return {"analytic_matrix_work": estimate, "loss_work": asdict(work),
        "observed_parameters": parameter_inventory(plan.model, optimizer=optimizer,
            executed_names=active_names(plan.model, plan.mode), inference_names=inference_names(plan.model, case)),
        "scope": "Audited matrix arithmetic estimate, not measured hardware FLOPs; unique parameter ownership counted separately from repeated FBT calls."}


def main(argv=None):
    args = parse_args(argv)
    overall_began = time.perf_counter()
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    compilation = configure_compiler()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        destination = args.output_dir / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(PROTOCOL, args.output_dir / "protocol.md")
    case = selected_case(args.case, batch=args.batch_size, length=512)
    supervision = "full"
    configuration = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "case_specification": asdict(case), "mode": asdict(case.mode()), "seed": SEED,
        "precision": "bf16_mixed", "ordinary_attention": "deterministic PyTorch Flash SDPA",
        "ordinary_checkpointing": True, "native_rt": "both improvements, Triton tiles, cast reuse, recompute",
        "author_rt": "author_legacy, compiled helper boundaries, four MLP chunks, fresh cast cache, materialized backward",
        "ce_chunk_size": 2048, "kl_chunk_size": 128, "supervision": supervision,
        "selected_rt_layers": list(case.rt_layers), "physical_batch": args.batch_size, "length": 512,
        "accumulation": 1, "world_size": 1, "tf32": False, "autocast_weight_cache": False,
        "capture_warmup_backwards": 10, "preparation_updates": 3 if args.stage == "capacity" else 0,
        "timed_updates": 5 if args.stage == "capacity" else 0,
        "graph_timing_samples": 3 if args.stage == "capacity" else 0}
    report = {"schema": "olmo-rt-author-integration-v1", "status": "running", "stage": "load",
        "configuration": configuration, "runtime": runtime, "determinism": determinism,
        "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "author_source": dict(AUTHOR_SOURCE), "compiled_helper_boundaries": COMPILED_HELPER_BOUNDARIES,
        "compiler_configuration": compilation, "started_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [], "physical_optimizer_updates": 0,
        "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCES}, "protocol_sha256": sha256_file(PROTOCOL)}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", group="olmo-rt-author-integration",
        name=args.output_dir.name, output_dir=args.output_dir)

    def save(stage=None):
        if stage is not None:
            report["stage"] = stage
            print({"stage": stage, "utc": datetime.now(timezone.utc).isoformat()}, flush=True)
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)

    def publish(check, *, blocking=True):
        report["checks"].append(check)
        save()
        tracker.log({"correctness/passed": int(check["passed"])}, step=len(report["checks"]))
        print({"check": check["name"], "passed": check["passed"],
               "global_gradient_relative_l2": check.get("global_gradient_relative_l2")}, flush=True)
        if blocking and not check["passed"]:
            raise AssertionError(check["name"])

    def exact_graph(plan, name, replays=1):
        check = compare_graph(plan, name, replays=replays)
        check.update(gate=True, passed=check["passed"] and check["all_bitwise_equal"])
        publish(check)

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"configuration": configuration, "checkpoint": report["checkpoint"]})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        save()
        load_began = time.perf_counter()
        state = load_native_state_dict(args.artifacts)
        tokenizer = load_native_tokenizer(args.artifacts)
        model = build_model(state, case)
        del state
        report["load_and_model_setup_seconds"] = time.perf_counter() - load_began
        original_signature = parameter_signature(model)
        set_backend(model, args.backend)
        report["parameters"] = {"active": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "resident": sum(parameter.numel() for parameter in model.parameters()),
            "owned_parameter_tensors": len(list(model.parameters()))}
        report["initial_auxiliary_parameters"] = tree_digests({name: parameter for name, parameter
            in model.named_parameters() if name.startswith(("backbone.fusion.", "predictor."))})
        report["nextlat_config"] = model.config.to_dict()
        batch = batch_for(tokenizer, case, 0, supervision)
        report["initial_batch"] = tree_digests(vars(batch))
        report["fixture_order"] = {"comparison_and_initial_capture": 0,
            "changed_input": 1 if args.stage == "verify" else 3,
            "parity_per_arm": [5, 6, 7] if args.stage == "verify" else [],
            "preparation": list(range(3)) if args.stage == "capacity" else [],
            "timed": list(range(3, 8)) if args.stage == "capacity" else []}
        with backend_context("flash"):
            if args.stage == "verify":
                save("bounded_ordinary_flash_probe")
                publish(ordinary_flash_probe(model, tokenizer, case))
                save("same_state_backend_comparison")
                began = time.perf_counter()
                plan, comparison = compare_backends(model, batch, case.mode())
                publish(comparison, blocking=False)
                report["raw_comparison_seconds"] = time.perf_counter() - began
                if not all(comparison[key] for key in
                           ("finite", "ownership_matches", "parameter_identity_preserved", "counts_equal")):
                    raise AssertionError("Nonfinite or ownership failure blocks graph execution")
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
                report["capture_seconds"] = time.perf_counter() - began
                report["compiler_after_warmup"] = compiler_audit(required=True)
                exact_graph(plan, "candidate_initial_graph")
                plan.load_batch(full_batch(tokenizer, case, 1))
                exact_graph(plan, "candidate_changed_tokens_overwrite", replays=2)
                save("complete_update_parity")
                publish(complete_update_parity(plan, tokenizer, case, report=report, persist=save,
                                              batch_factory=full_batch))
                plan.load_batch(full_batch(tokenizer, case, 2))
                exact_graph(plan, "candidate_changed_weights")
                report["memory"] = memory_snapshot()
                report["resources"] = resource_card(plan, case, "author")
            else:
                plan = new_plan(model, batch, case.mode(), "recompute")
                optimizer, scheduler = build_optimizer(model)
                hook = track_optimizer_steps(optimizer, report)
                counters = TrainingCounters()
                probe = model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096].clone()
                torch.cuda.reset_peak_memory_stats()
                report["preparation_records"] = []
                report["preparation_batches"] = []
                save("preparation_updates")
                began = time.perf_counter()
                for index in range(3):
                    preparation_batch = batch_for(tokenizer, case, index, supervision)
                    report["preparation_batches"].append(tree_digests(vars(preparation_batch)))
                    report["preparation_records"].append(plan.optimizer_step(optimizer,
                        preparation_batch, scheduler=scheduler, counters=counters))
                    save()
                report["preparation_seconds"] = time.perf_counter() - began
                save("capture")
                began = time.perf_counter()
                plan.capture(warmup=10)
                report["capture_seconds"] = time.perf_counter() - began
                report["compiler_after_warmup"] = compiler_audit(required=args.backend == "author")
                save("capacity_graph_validation")
                exact_graph(plan, "capacity_initial_graph")
                plan.load_batch(batch_for(tokenizer, case, 3, supervision))
                exact_graph(plan, "capacity_changed_tokens_overwrite", replays=2)
                report["setup_memory"] = memory_snapshot()
                torch.cuda.reset_peak_memory_stats()
                records = []
                report["timed_records"] = records
                batches = [batch_for(tokenizer, case, index + 3, supervision) for index in range(5)]
                report["timed_batches"] = [tree_digests(vars(value)) for value in batches]

                def complete():
                    records.append(plan.optimizer_step(optimizer, batches[len(records)], replay=True,
                        scheduler=scheduler, counters=counters))

                save("timed_complete_updates")
                report["full_update"] = timed(complete, 5)
                report["steady_memory"] = memory_snapshot()
                report["input_tokens_per_second"] = plan.input_tokens / report["full_update"]["median_wall_seconds"]
                report["ce_targets_per_second"] = plan.counts["ce"] / report["full_update"]["median_wall_seconds"]
                save("timed_graph_replays")
                report["forward_loss_backward"] = timed(lambda: plan.backward(replay=True), 3)
                report["forward_loss_backward_tokens_per_second"] = plan.input_tokens / report["forward_loss_backward"]["median_wall_seconds"]
                save("capacity_changed_weights_validation")
                exact_graph(plan, "capacity_changed_weights")
                report["health"] = state_health(model, optimizer)
                report["resources"] = resource_card(plan, case, args.backend, optimizer)
                changed = not torch.equal(probe, model.backbone.backbone.layers[0].ff_out.weight.detach().flatten()[:4096])
                finite_gradients = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                                       for parameter in model.parameters() if parameter.requires_grad)
                publish({"name": "finite_complete_updates", "gate": True,
                    "passed": report["health"]["passed"] and changed and finite_gradients,
                    "trainable_weight_changed": changed, "finite_participating_gradients": finite_gradients})
                hook.remove()
                tracker.log({"benchmark/input_tokens_per_second": report["input_tokens_per_second"],
                    "benchmark/ce_targets_per_second": report["ce_targets_per_second"],
                    "benchmark/forward_loss_backward_tokens_per_second": report["forward_loss_backward_tokens_per_second"],
                    "benchmark/setup_peak_reserved_gib": report["setup_memory"]["peak_reserved_gib"],
                    "benchmark/steady_peak_allocated_gib": report["steady_memory"]["peak_allocated_gib"]}, step=len(report["checks"]) + 1)
            report["nextlat_config"] = model.config.to_dict()
            report["prepared_layout"] = plan.forward_layout.metadata
            report["counts"] = dict(plan.counts)
            report["input_tokens"] = plan.input_tokens
            report["pass_input_token_work"] = plan.input_tokens * (case.mode().num_passes if case.fbt else 1)
            report["backward_preparation"] = {"warmup": plan.warmup_backward_calls,
                "capture": plan.capture_backward_calls, "replay": plan.replay_calls}
            report["parameter_identity_preserved"] = parameter_signature(model) == original_signature
        report["compiler_final"] = compiler_audit(required=args.backend == "author")
        if not report["parameter_identity_preserved"]:
            raise AssertionError("Integration changed parameter identity/storage")
        if report["source_hashes"] != {name: sha256_file(ROOT / name) for name in SOURCES}:
            raise AssertionError("Runtime sources changed during execution")
        if report["protocol_sha256"] != sha256_file(PROTOCOL):
            raise AssertionError("Frozen protocol changed during execution")
        failed = [check["name"] for check in report["checks"] if check.get("gate", True) and not check["passed"]]
        if failed:
            raise AssertionError("Retained integration screen failures: " + ", ".join(failed))
        report["status"] = "passed"
        tracker.summary({"result_status": "passed", "physical_optimizer_updates": report["physical_optimizer_updates"]})
        save("complete")
    except BaseException as error:
        report.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed",
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        report["compiler_on_exit"] = compiler_audit(required=False)
        save()
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            report["elapsed_seconds"] = time.perf_counter() - overall_began
            save()


if __name__ == "__main__":
    main()
