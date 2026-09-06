#!/usr/bin/env python3
"""Tiny NUM/OPS checks: learnability, weighted accumulation, exact checkpoint resume.

Run inside the project GPU container, from /workspace/cdrm-w-latent:
    python scripts/stage_a_ops.py --output-dir .runtime/stage-a/ops

Checkpoints contain arbitrary smoke-batch training and must never initialize a
research LM baseline. This is deliberately not a configurable long-running trainer.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import os
from pathlib import Path
import random
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from stage_a_common import (ARTIFACT_WARNING, adamw, configure_compiled_helpers, experiment_manifest,
                            load_checkpoint, model_config, provenance, require_cuda_container,
                            rng_state, save_checkpoint, seed_all, set_rho, shifted_ce_sum,
                            update, validate_compiler_execution, write_json)


def compare(left, right, *, atol=0.0, rtol=0.0, path="root") -> dict:
    """Compare every nested tensor; no aggregate metric can hide a failed key."""
    largest = 0.0
    tensors = 0
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(f"{path}: tensor type mismatch")
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=lambda msg: f"{path}: {msg}")
        tensors = 1
        largest = (left.double() - right.double()).abs().max().item() if left.numel() else 0.0
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in left:
            result = compare(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
            largest = max(largest, result["max_absolute_error"])
            tensors += result["compared_tensors"]
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            result = compare(a, b, atol=atol, rtol=rtol, path=f"{path}[{index}]")
            largest = max(largest, result["max_absolute_error"])
            tensors += result["compared_tensors"]
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif left != right:
        raise AssertionError(f"{path}: values differ: {left!r} != {right!r}")
    return {"max_absolute_error": largest, "compared_tensors": tensors, "atol": atol, "rtol": rtol}


def clone_model(model):
    from olmo.model import OLMo
    other = OLMo(copy.deepcopy(model.config)).cuda().to(dtype=next(model.parameters()).dtype).train()
    other.load_state_dict(model.state_dict(), strict=True)
    return other


@torch.no_grad()
def loss_value(model, ids, mask=None):
    logits = model(ids).logits
    total, count = shifted_ce_sum(logits, ids, mask)
    return total.item() / count


def accumulation_check(initial, ids, *, optimizer_equivalence=True):
    # Unequal target counts, including an entirely unsupervised example. All tokens
    # remain valid causal context: this is a loss mask, not padded attention.
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, 4:] = False
    mask[1, :] = False
    mask[2, 2:] = False
    full = clone_model(initial)
    full_opt = adamw(full)
    update(full, full_opt, [(ids, None)])  # nonzero Adam moments before comparing updates
    accumulated = clone_model(full)
    accumulated_opt = adamw(accumulated)
    accumulated_opt.load_state_dict(copy.deepcopy(full_opt.state_dict()))
    full_result = update(full, full_opt, [(ids, mask)])
    accumulated_result = update(accumulated, accumulated_opt,
                                [(ids[index:index + 1], mask[index:index + 1]) for index in range(4)])
    if full_result["valid_target_tokens"] != accumulated_result["valid_target_tokens"]:
        raise AssertionError("Accumulation target counts differ.")
    full_grads = {name: param.grad for name, param in full.named_parameters()}
    accumulated_grads = {name: param.grad for name, param in accumulated.named_parameters()}
    gradient_error = compare(full_grads, accumulated_grads, atol=2e-6, rtol=2e-5)
    if optimizer_equivalence:
        parameter_error = compare(full.state_dict(), accumulated.state_dict(), atol=2e-6, rtol=2e-5)
        optimizer_error = compare(full_opt.state_dict(), accumulated_opt.state_dict(), atol=2e-6, rtol=2e-5)
    else:
        # Key-normalization bias shifts every key logit equally. Its true gradient
        # is zero, so FP32 roundoff divided by AdamW epsilon can produce visible
        # parameter differences. Retain gradient evidence and actual differences;
        # the independent bias-free fixture must still pass optimizer equality.
        differences = {name: (value - accumulated.state_dict()[name]).abs().max().item()
                       for name, value in full.state_dict().items()}
        parameter_error = {"asserted_equivalent": False, "largest_differences":
                           dict(sorted(differences.items(), key=lambda pair: pair[1], reverse=True)[:8]),
                           "reason": "diagnostic for bias-enabled FP32 AdamW sensitivity near mathematically zero gradients"}
        optimizer_error = {"asserted_equivalent": False}
    return {"status": "passed", "valid_target_tokens": full_result["valid_target_tokens"],
            "full_loss": full_result["loss"], "accumulated_loss": accumulated_result["loss"],
            "gradients": gradient_error, "parameters": parameter_error, "optimizer": optimizer_error,
            "full_batch_sequences": 4, "microbatch_sequences": 1, "accumulation": 4,
            "loss_mask": "unequal counts including one example with zero targets",
            "optimizer_equivalence_asserted": optimizer_equivalence,
            "include_bias": initial.config.include_bias,
            "bias_for_layer_norm": initial.config.bias_for_layer_norm}


def next_stream_batch(config, offset):
    # Exercise CPU, CUDA, NumPy and Python RNG restoration. Offset is an independent
    # part of stream state: restoring RNG alone cannot reproduce these token IDs.
    shift = random.randrange(config.vocab_size - 2) + int(np.random.randint(config.vocab_size - 2))
    ids = torch.randint(0, config.vocab_size - 2, (2, config.max_sequence_length))
    ids = (ids + offset + shift) % (config.vocab_size - 2) + 2
    # A CUDA RNG draw genuinely changes the input, although model dropout stays off.
    extra = torch.randint(0, config.vocab_size - 2, (), device="cuda")
    ids = ((ids.cuda() - 2 + extra) % (config.vocab_size - 2)) + 2
    digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    return ids, digest


def trajectory(model, optimizer, *, begin, end, offset, schedule):
    history = []
    for index in range(begin, end):
        rho = min(1.0, (index + 1) / schedule["rho_ramp_updates"]) if schedule["rho_ramp_updates"] else 1.0
        set_rho(model, rho)
        for group in optimizer.param_groups:
            group["lr"] = schedule["base_lr"] * (0.5 + 0.5 * min(1.0, (index + 1) / 3))
        ids, digest = next_stream_batch(model.config, offset)
        result = update(model, optimizer, [(ids, None)])
        offset += ids.shape[0]
        history.append({"update": index + 1, "offset": offset, "rho": rho,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "batch_sha256": digest, "loss": result["loss"]})
    return history, offset


def resume_check(initial, output_dir, manifest, backend):
    uninterrupted = clone_model(initial)
    opt = adamw(uninterrupted)
    schedule = {"kind": "fixed-smoke-linear-rewarm", "base_lr": 1e-3,
                "rho_ramp_updates": 4 if backend == "naive" and initial.config.recurrent_layers else 0,
                "horizon_updates": 4}
    prefix, offset = trajectory(uninterrupted, opt, begin=0, end=2, offset=0, schedule=schedule)
    checkpoint = output_dir / "interrupted-u2.pt"
    save_checkpoint(checkpoint, uninterrupted, opt, update_index=2, data_offset=offset,
                    schedule=schedule, manifest=manifest)
    tail, final_offset = trajectory(uninterrupted, opt, begin=2, end=4, offset=offset, schedule=schedule)
    final_rng = rng_state()
    resumed = clone_model(initial)
    resumed_opt = adamw(resumed)
    state = load_checkpoint(checkpoint, resumed, resumed_opt)
    if state["optimizer_policy"] != "exact_resume":
        raise AssertionError("Resume incorrectly labeled as an optimizer reset.")
    if resumed.config.recurrent_write_rho != state["rho"] or any(
        resumed.transformer.blocks[index].config.recurrent_write_rho != state["rho"]
        for index in resumed.config.recurrent_layers or []
    ):
        raise AssertionError("The saved recurrence gate was not restored before the next update.")
    resumed_tail, resumed_offset = trajectory(resumed, resumed_opt, begin=state["update_index"],
                                             end=state["schedule"]["horizon_updates"],
                                             offset=state["data_offset"], schedule=state["schedule"])
    parameters = compare(uninterrupted.state_dict(), resumed.state_dict())
    optimizer = compare(opt.state_dict(), resumed_opt.state_dict())
    compare(final_rng, rng_state())
    compare(tail, resumed_tail)
    if final_offset != resumed_offset:
        raise AssertionError("Data offsets differ after resume.")
    save_checkpoint(output_dir / "resumed-u4.pt", resumed, resumed_opt, update_index=4,
                    data_offset=resumed_offset, schedule=state["schedule"], manifest=manifest)
    fresh = clone_model(initial)
    fresh_opt = adamw(fresh)
    fresh_state = load_checkpoint(checkpoint, fresh, fresh_opt, fresh_optimizer=True)
    compare(fresh.state_dict(), torch.load(checkpoint, map_location="cuda", weights_only=False)["model"])
    if fresh_opt.state or fresh_state["update_index"] != 0 or fresh_state["parent_update_index"] != 2:
        raise AssertionError("Fresh-optimizer branch did not reset branch-local training state.")
    if fresh_state["data_offset"] != offset or fresh_state["branch_processed_sequences"] != 0:
        raise AssertionError("Fresh optimizer must preserve stream position and reset branch-local counts.")
    _, fresh_next_digest = next_stream_batch(fresh.config, fresh_state["data_offset"])
    if fresh_next_digest != tail[0]["batch_sha256"]:
        raise AssertionError("Fresh optimizer changed the common parent's next data examples.")
    return {"status": "passed", "parameters": parameters, "optimizer": optimizer,
            "rng_states_exact": True, "data_stream_exact": True,
            "uninterrupted_history": prefix + tail, "resumed_history": resumed_tail,
            "resume_checkpoint": str(checkpoint), "final_data_offset": final_offset,
            "fresh_optimizer_branch": {"verified_empty_optimizer": True, "parent_update_index": 2,
                                        "branch_update_index": 0, "preserved_stream_offset": offset,
                                        "branch_processed_sequences": 0, "preserved_next_data_batch": True},
            "rho_schedule": schedule}


def empty_target_check(model, ids):
    optimizer = adamw(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    logits = model(ids[:, :1]).logits
    loss, count = shifted_ce_sum(logits, ids[:, :1])
    if count != 0 or loss.item() != 0 or not torch.isfinite(logits).all():
        raise AssertionError("T=1 must have finite logits and no valid shifted LM target.")
    result = update(model, optimizer, [(ids[:, :1], None)])
    if not result["skipped_empty_targets"] or optimizer.state:
        raise AssertionError("Empty supervision must not create an optimizer update.")
    compare(before, model.state_dict())
    return {"status": "passed", "t1_logits_finite": True, "valid_targets": 0,
            "optimizer_step_skipped": True}


def run(args, report):
    hardware = require_cuda_container()
    seed_all(args.seed, deterministic=True)
    configure_compiled_helpers(args.compiled_helpers)
    report["provenance"] = provenance(hardware)
    report["attention_backend"] = "explicit PyTorch math SDPA for deterministic NUM/OPS"
    report["compiler_fixture_policy"] = {
        "isolate_independent_groups": args.compiled_helpers,
        "reason": "PyTorch compiler config overrides are thread-local; autograd workers may see default cache limits. Reset only code caches between unrelated fixture groups, never within an accumulation or resume comparison.",
        "counter_policy": "cumulative counters retained across cache resets; any unsupported/fallback counter fails the run",
        "groups": [],
    }

    def new_group(name):
        if args.compiled_helpers:
            validate_compiler_execution(True)
            torch._dynamo.reset_code_caches()
            configure_compiled_helpers(True)
            report["compiler_fixture_policy"]["groups"].append(name)

    from olmo.model import OLMo
    from olmo.checkpoint_conversion import convert_model

    source = OLMo(model_config("tiny", "seq", args.backend, 1.0,
                               compiled_helpers=args.compiled_helpers)).cuda().train()
    ids = (torch.arange(32, device="cuda").reshape(4, 8) % 13) + 2
    report["fixture"] = {"shape": list(ids.shape), "sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                          "description": "four fixed length-8 symbolic sequences; no padding or corpus data"}
    report["topologies"] = {}
    for topology in ("seq", "r3"):
        seed_all(args.seed + 1, deterministic=True)
        config = model_config("tiny", topology, args.backend, 1.0,
                              compiled_helpers=args.compiled_helpers)
        model = OLMo(config).cuda().train()
        conversion = convert_model(source, model)
        manifest = experiment_manifest(config, evidence="NUM/OPS", budget={
            "fixed_batch_updates": args.overfit_updates, "accumulation_updates": 6,
            "resume_updates_executed": 6, "precision": "fp32", "seed": args.seed,
        })
        manifest["mask"] += "; accumulation additionally masks target loss positions"
        record = {"manifest": manifest, "conversion": dataclasses.asdict(conversion), "status": "running"}
        report["topologies"][topology] = record
        topology_dir = args.output_dir / topology
        print(f"{topology.upper()}: empty-target and accumulation NUM checks", flush=True)
        new_group(f"{topology}/empty-targets")
        record["empty_targets"] = empty_target_check(model, ids)
        new_group(f"{topology}/bias-enabled-gradient-accumulation")
        record["bias_enabled_accumulation_gradients"] = accumulation_check(model, ids, optimizer_equivalence=False)
        accumulation_config = copy.deepcopy(config)
        accumulation_config.include_bias = False
        accumulation_config.bias_for_layer_norm = False
        accumulation_model = OLMo(accumulation_config).cuda().train()
        new_group(f"{topology}/bias-free-complete-accumulation")
        record["accumulation"] = accumulation_check(accumulation_model, ids)
        del accumulation_model
        print(f"{topology.upper()}: interrupted/resumed trajectory NUM check", flush=True)
        new_group(f"{topology}/complete-resume-comparison")
        record["resume"] = resume_check(model, topology_dir, manifest, args.backend)
        print(f"{topology.upper()}: {args.overfit_updates}-update fixed-batch OPS learnability", flush=True)
        new_group(f"{topology}/fixed-batch-learnability")
        initial_loss = loss_value(model, ids)
        optimizer = adamw(model, learning_rate=3e-3)
        curve = []
        for index in range(args.overfit_updates):
            result = update(model, optimizer, [(ids, None)])
            curve.append({"update": index + 1, "pre_update_loss": result["loss"]})
        final_loss = loss_value(model, ids)
        record["fixed_batch_learnability"] = {"initial_loss": initial_loss, "final_loss": final_loss,
                                               "final_over_initial": final_loss / initial_loss,
                                               "required_final_over_initial_below": 0.8,
                                               "updates": args.overfit_updates, "curve": curve,
                                               "learning_rate": 3e-3}
        if not final_loss < 0.8 * initial_loss:
            raise AssertionError(f"{topology}: fixed-batch loss did not fall by at least 20%")
        save_checkpoint(topology_dir / f"OPS-smoke-{topology.upper()}.pt", model, optimizer,
                        update_index=args.overfit_updates, data_offset=4 * args.overfit_updates,
                        schedule={"kind": "constant", "learning_rate": 3e-3}, manifest=manifest)
        record["status"] = "passed"
        write_json(topology_dir / "manifest.json", record)
        print(f"{topology.upper()} passed: fixed-batch CE {initial_loss:.6f} -> {final_loss:.6f}", flush=True)
        del model, optimizer
    validate_compiler_execution(args.compiled_helpers)
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overfit-updates", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--backend", choices=("naive", "tiled"), default="naive")
    parser.add_argument("--compiled-helpers", action="store_true")
    args = parser.parse_args()
    if not 8 <= args.overfit_updates <= 100:
        parser.error("--overfit-updates must be between 8 and 100; this is a bounded smoke tool")
    report = {"schema": "stage-a-ops-v1", "status": "running", "artifact_warning": ARTIFACT_WARNING}
    result = 0
    try:
        with sdpa_kernel(SDPBackend.MATH):
            run(args, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        result = 1
    finally:
        report["compiler_counters"] = {str(key): dict(value)
                                       for key, value in torch._dynamo.utils.counters.items()}
        write_json(args.output_dir / "report.json", report)
        print(f"Stage A NUM/OPS {report['status']}: {args.output_dir / 'report.json'}", flush=True)
    return result


if __name__ == "__main__":
    sys.exit(main())
