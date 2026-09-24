"""Bounded same-gradient FP32 scalar/fused AdamW diagnostic, outside timing.

One model's parameters and persistent gradient buffers are reused. Initial
weights/raw gradients and reference final state live on CPU, so the diagnostic
never needs two model/Adam copies on GPU. Three updates per implementation use
identical fixed raw gradients, native clipping and the same short LR schedule.
These are disposable optimizer diagnostics, not six data-training updates.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import torch

from cdrm.pretrained.lm_training import (
    TrainingCounters, build_adamw, build_warmup_scheduler, optimizer_ownership,
)
from scripts.olmo_validation import require_container_gpu

BUDGETS = {
    "moment_global_relative_l2": 1e-6,
    "moment_tensor_relative_l2": 1e-6,
    "moment_tensor_max_relative": 1e-6,
    "weight_global_relative_l2": 1e-6,
    "weight_tensor_relative_l2": 1e-6,
    "weight_tensor_max_relative": 1e-6,
    "cumulative_update_global_relative_l2": 1e-3,
}
_FLAG_KEYS = ("fused", "foreach", "capturable", "differentiable", "amsgrad",
              "maximize", "lr", "betas", "eps", "weight_decay", "param_names")


def make_optimizer(model, *, fused: bool):
    if type(fused) is not bool:
        raise TypeError("The optimizer diagnostic requires an explicit boolean fused choice")
    optimizer = build_adamw(model, lr=1e-5, betas=(.9, .95), eps=1e-8,
        weight_decay=.1, foreach=False, fused=True if fused else None)
    return optimizer, build_warmup_scheduler(optimizer, warmup_updates=2)


def _cpu_copy(value):
    return value.detach().to(device="cpu", copy=True)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else (0. if numerator == 0 else float("inf"))


def tensor_metrics(candidate, reference, *, initial=None, chunk_size=1 << 20):
    """FP64 reductions in bounded CPU chunks, including actual update deltas."""
    if candidate.device.type != "cpu" or reference.device.type != "cpu":
        raise ValueError("Optimizer comparison reductions require CPU snapshots")
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise ValueError("Optimizer tensor metadata changed")
    if initial is not None and (initial.device.type != "cpu" or initial.shape != reference.shape):
        raise ValueError("Initial state must match the compared CPU tensor")
    a, b = candidate.reshape(-1), reference.reshape(-1)
    origin = None if initial is None else initial.reshape(-1)
    error_sq = reference_sq = candidate_sq = max_abs = reference_peak = candidate_peak = 0.
    finite = True
    for start in range(0, a.numel(), chunk_size):
        stop = start + chunk_size
        left, right = a[start:stop].double(), b[start:stop].double()
        if origin is not None:
            beginning = origin[start:stop].double()
            left, right = left - beginning, right - beginning
        error = left - right
        finite = finite and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
        error_sq += float(error.square().sum())
        reference_sq += float(right.square().sum())
        candidate_sq += float(left.square().sum())
        max_abs = max(max_abs, float(error.abs().max()))
        reference_peak = max(reference_peak, float(right.abs().max()))
        candidate_peak = max(candidate_peak, float(left.abs().max()))
    return {"delta_sq": error_sq, "reference_sq": reference_sq,
        "candidate_sq": candidate_sq, "relative_l2": math.sqrt(_ratio(error_sq, reference_sq)),
        "max_relative": _ratio(max_abs, reference_peak), "max_abs": max_abs,
        "reference_peak": reference_peak, "candidate_peak": candidate_peak,
        "reference_l2": math.sqrt(reference_sq), "candidate_l2": math.sqrt(candidate_sq),
        "finite": finite, "bitwise_equal": torch.equal(candidate, reference)}


def global_relative_l2(rows):
    rows = list(rows)
    return math.sqrt(_ratio(math.fsum(row["delta_sq"] for row in rows),
                            math.fsum(row["reference_sq"] for row in rows)))


def comparison_screen(*, moments, weights, updates, semantic_matches):
    global_moments = global_relative_l2(moments.values())
    global_weights = global_relative_l2(weights.values())
    global_updates = global_relative_l2(updates.values())
    passed = (semantic_matches and bool(moments) and bool(weights) and bool(updates)
        and all(row["finite"] for row in (*moments.values(), *weights.values(), *updates.values()))
        and global_moments <= BUDGETS["moment_global_relative_l2"]
        and global_weights <= BUDGETS["weight_global_relative_l2"]
        and global_updates <= BUDGETS["cumulative_update_global_relative_l2"]
        and all(row["relative_l2"] <= BUDGETS["moment_tensor_relative_l2"]
                and row["max_relative"] <= BUDGETS["moment_tensor_max_relative"] for row in moments.values())
        and all(row["relative_l2"] <= BUDGETS["weight_tensor_relative_l2"]
                and row["max_relative"] <= BUDGETS["weight_tensor_max_relative"] for row in weights.values()))
    return {"passed": passed, "moment_global_relative_l2": global_moments,
        "weight_global_relative_l2": global_weights,
        "cumulative_update_global_relative_l2": global_updates}


def _snapshot(model, optimizer):
    parameters = dict(model.named_parameters())
    return {"weights": {name: _cpu_copy(p) for name, p in parameters.items() if p.requires_grad},
        "state": {name: {key: _cpu_copy(value) if isinstance(value, torch.Tensor) else value
                         for key, value in optimizer.state[p].items()}
                  for name, p in parameters.items() if p.requires_grad},
        "step_devices": {name: str(optimizer.state[p]["step"].device)
                         for name, p in parameters.items() if p.requires_grad}}


def _compare_from_fixed_gradients(model, *, report, persist, optimizer_factory=make_optimizer):
    """Internal orchestration, separated for explicit tiny CPU fixture tests."""
    parameters = dict(model.named_parameters())
    active = {name: p for name, p in parameters.items() if p.requires_grad}
    if not active or any(p.dtype != torch.float32 or p.grad is None or p.grad.dtype != torch.float32
                         for p in active.values()):
        raise ValueError("Diagnostic requires attached FP32 parameters and raw gradients")
    if any(not bool(torch.isfinite(p.grad).all()) for p in active.values()):
        raise FloatingPointError("Diagnostic raw gradients must be finite")
    initial = {name: _cpu_copy(p) for name, p in parameters.items()}
    raw = {name: _cpu_copy(p.grad) for name, p in active.items()}
    addresses = {name: (id(p), p.data_ptr(), None if p.grad is None else p.grad.data_ptr())
                 for name, p in parameters.items()}
    report.setdefault("physical_optimizer_updates", 0)
    progress = report["optimizer_probe_progress"] = []
    outcomes, reference = [], None
    optimizer = scheduler = hook = None
    comparison = None
    try:
        for fused in (False, True):
            with torch.no_grad():
                for name, p in parameters.items():
                    p.copy_(initial[name])
            optimizer, scheduler = optimizer_factory(model, fused=fused)
            owned = optimizer_ownership(model, optimizer)
            flags = [{key: group.get(key) for key in _FLAG_KEYS} for group in optimizer.param_groups]
            if any(group["fused"] is not (True if fused else None) or group["foreach"] is not False
                   or group["capturable"] or group["differentiable"] for group in flags):
                raise AssertionError("Requested scalar/fused AdamW implementation was not selected")
            counters, records = TrainingCounters(), []

            def count_step(_optimizer, _args, _kwargs):
                report["physical_optimizer_updates"] += 1

            hook = optimizer.register_step_post_hook(count_step)
            for index in range(3):
                with torch.no_grad():
                    for name, p in active.items():
                        p.grad.copy_(raw[name])
                learning_rates = [group["lr"] for group in optimizer.param_groups]
                norm = torch.nn.utils.clip_grad_norm_(list(active.values()), 1., error_if_nonfinite=True, foreach=False)
                optimizer.step()
                scheduler.step()
                counters.optimizer_updates += 1
                row = {"optimizer_update": index + 1, "gradient_norm_before_clip": float(norm),
                    "lr_used": learning_rates, "lr_next": [group["lr"] for group in optimizer.param_groups],
                    "counters": asdict(counters)}
                records.append(row)
                progress.append({"fused": fused, **row})
                persist()
            hook.remove()
            hook = None
            state = _snapshot(model, optimizer)
            moments_fp32 = all(value.dtype == torch.float32 for values in state["state"].values()
                              for key, value in values.items() if key in ("exp_avg", "exp_avg_sq"))
            steps = {name: float(values["step"]) for name, values in state["state"].items()}
            states_complete = all(set(values) == {"step", "exp_avg", "exp_avg_sq"} for values in state["state"].values())
            outcomes.append({"fused": fused, "records": records, "optimizer_groups": flags,
                "ownership": owned, "step_values": steps, "step_devices": state["step_devices"],
                "moments_fp32": moments_fp32, "state_keys_complete": states_complete,
                "scheduler": scheduler.state_dict(), "counters": asdict(counters)})
            if not fused:
                reference = state
            else:
                moments = {f"{name}/{key}": tensor_metrics(values[key], reference["state"][name][key])
                    for name, values in state["state"].items() for key in ("exp_avg", "exp_avg_sq")}
                weights = {name: tensor_metrics(value, reference["weights"][name])
                           for name, value in state["weights"].items()}
                updates = {name: tensor_metrics(value, reference["weights"][name], initial=initial[name])
                           for name, value in state["weights"].items()}
                # Step device legitimately differs; compare values, ownership,
                # LR schedule, clipping and counters instead of storage placement.
                exact_keys = ("records", "ownership", "step_values", "scheduler", "counters")
                semantics = all(outcomes[0][key] == outcomes[1][key] for key in exact_keys)
                matched_groups = [
                    [{key: value for key, value in group.items() if key != "fused"}
                     for group in outcome["optimizer_groups"]]
                    for outcome in outcomes
                ]
                groups_match = matched_groups[0] == matched_groups[1]
                semantics = semantics and groups_match
                semantics = semantics and all(outcome["moments_fp32"] and outcome["state_keys_complete"]
                    and all(value == 3 for value in outcome["step_values"].values()) for outcome in outcomes)
                semantics = semantics and sum(row["reference_sq"] for row in updates.values()) > 0
                comparison = {"name": "fixed_gradient_scalar_vs_fused_adamw",
                    **comparison_screen(moments=moments, weights=weights, updates=updates, semantic_matches=semantics),
                    "budgets": dict(BUDGETS), "semantic_state_matches": semantics,
                    "optimizer_groups_match_except_fused": groups_match,
                    "moments": moments, "weights": weights, "cumulative_updates": updates,
                    "arms": outcomes, "physical_optimizer_updates": 6, "updates_per_arm": 3,
                    "max_grad_norm": 1.0, "raw_gradient_tensor_count": len(raw),
                    "quantization_context": {"parameter_dtype": "float32", "fp32_epsilon": torch.finfo(torch.float32).eps,
                        "update_definition": "FP64 difference between actual final FP32 parameter and identical initial FP32 parameter",
                        "max_update_error_descriptive_only": True,
                        "note": "Small parameter updates are quantized at the weight's FP32 spacing; update L2 remains an independent gate, never waived by whole-weight agreement."},
                    "scope": "One fixed actual-model raw gradient, restored before all three scalar and three fused AdamW updates; no new examples or learning claim. Initial weights and raw gradients restored afterward."}
            del state
            # Release each arm's GPU moment state before constructing the next.
            del optimizer, scheduler
            optimizer = scheduler = None
    finally:
        if hook is not None:
            hook.remove()
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(initial[name])
            for name, p in active.items():
                p.grad.copy_(raw[name])
        unchanged_addresses = addresses == {name: (id(p), p.data_ptr(), None if p.grad is None else p.grad.data_ptr())
                                             for name, p in parameters.items()}
        restored_values = all(torch.equal(_cpu_copy(p), initial[name]) for name, p in parameters.items())
        restored_values = restored_values and all(torch.equal(_cpu_copy(p.grad), raw[name]) for name, p in active.items())
        report["optimizer_probe_state_restored"] = unchanged_addresses and restored_values
        persist()
        if not unchanged_addresses or not restored_values:
            raise AssertionError("Optimizer diagnostic changed a restored value or persistent buffer address")
    if comparison is None:
        raise AssertionError("Optimizer diagnostic did not produce both arm states")
    comparison["initial_state_restored"] = True
    return comparison


def compare_fixed_gradients(plan, *, report, persist):
    """Run only on the required GPU environment; no CPU execution fallback."""
    if any(p.device.type != "cuda" for p in plan.model.parameters()):
        raise ValueError("Fixed-gradient optimizer diagnostic requires CUDA; no CPU fallback")
    require_container_gpu()
    plan.initialize_gradients()
    plan.backward(replay=False)
    return _compare_from_fixed_gradients(plan.model, report=report, persist=persist)
