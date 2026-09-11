#!/usr/bin/env python3
"""CPU-only independent clipping/Adam references for retained CDRM NUM packets.

This script never constructs a model or changes an optimizer hyperparameter.
Local arithmetic checks and original policy-distance failures stay separate.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import sys

import torch

from experiment_tracking import OnlineTracker, add_wandb_arguments

U, SUBNORMAL = 2**-24, 2**-149
OLD_CONTRACT = "26b1756dd958e0ab1c916cc51e46b691393eabb5598c645cb0d63b5bfdd0ea20"
NEW_CONTRACT = "6660b964a2f5a678d4324ac2799c814ad617ce9acceabbed5e0b0e23101714cc"
ARMS = ("naive_fp32", "tiled_fp32", "tiled_bf16")
DEFAULT_ROOT = Path(".runtime/cdrm-numerical-resolution/20260907T232931Z")
OLD_ROOT = Path(".runtime/cdrm-tiled-pilot/20260907T212606Z")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def norm64(values):
    return math.sqrt(sum(float(value.double().square().sum()) for value in values.values()))


def clip64(gradients):
    norm = norm64(gradients)
    coefficient = min(1., 1./(norm+1e-6))
    return {name: value.double()*coefficient for name, value in gradients.items()}, norm, coefficient


def check_hyperparameters(group):
    for key in ("amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused"):
        if group.get(key, False):
            raise ValueError(f"Unsupported optimizer setting: {key}")
    if group.get("weight_decay", 0.) != 0.:
        raise ValueError("Reference scope requires zero weight decay")
    if not (0 < group["lr"] and group["eps"] > 0 and all(0 <= b < 1 for b in group["betas"])):
        raise ValueError("Invalid optimizer hyperparameters")


def adam64(weights, gradients, state, group):
    """Closed-form Adam on exact promoted saved inputs; no torch optimizer."""
    check_hyperparameters(group)
    b1, b2 = group["betas"]
    result = {key: {} for key in ("moments", "increments", "weights", "realized_deltas")}
    for name, weight in weights.items():
        previous = state[name]
        t = float(previous["step"])+1
        if t < 1 or not t.is_integer():
            raise ValueError("Adam step must advance an integer")
        g = gradients[name].double()
        m = b1*previous["exp_avg"].double()+(1-b1)*g
        v = b2*previous["exp_avg_sq"].double()+(1-b2)*g.square()
        if not bool((v >= 0).all()):
            raise ValueError("Negative second moment")
        delta = -(group["lr"]/(1-b1**t))*m/(torch.sqrt(v/(1-b2**t))+group["eps"])
        after = weight.double()+delta
        result["moments"][name] = {"step": t, "exp_avg": m, "exp_avg_sq": v}
        result["increments"][name] = delta
        result["weights"][name] = after
        result["realized_deltas"][name] = after.float().double()-weight.double()
    return result


def increment_from_post_moments(moment, group):
    b1, b2 = group["betas"]
    t = float(moment["step"])
    return -(group["lr"]/(1-b1**t))*moment["exp_avg"].double()/(torch.sqrt(moment["exp_avg_sq"].double()/(1-b2**t))+group["eps"])


def row_metric(reference, actual, mask=None):
    r, a = reference.double(), actual.double()
    e = a-r
    re, ae, ee = (float(x.square().sum()) for x in (r, a, e))
    dot = float((r*a).sum())
    row = {"count": r.numel(), "reference_energy": re, "actual_energy": ae, "error_energy": ee,
           "dot": dot, "max_abs_error": float(e.abs().max()),
           "relative_l2": math.sqrt(ee)/max(math.sqrt(re), 1e-30),
           "strict_sign_flips": int((r*a < 0).sum())}
    if mask is not None:
        row["near_zero"] = {"count": int(mask.sum()), "error_energy": float(e[mask].square().sum())}
        row["outside_near_zero"] = {"count": int((~mask).sum()), "error_energy": float(e[~mask].square().sum())}
    return row


def compare_maps(reference, actual, masks=None):
    if list(reference) != list(actual):
        raise ValueError("Tensor names/order mismatch")
    rows = {name: row_metric(value, actual[name], None if masks is None else masks[name])
            for name, value in reference.items()}
    totals = {key: sum(row[key] for row in rows.values()) for key in
              ("count", "reference_energy", "actual_energy", "error_energy", "dot", "strict_sign_flips")}
    re, ae, ee = (totals[key] for key in ("reference_energy", "actual_energy", "error_energy"))
    cosine = max(-1., min(1., totals["dot"]/math.sqrt(re*ae))) if re*ae else None
    totals.update(relative_l2=math.sqrt(ee)/max(math.sqrt(re), 1e-30), cosine=cosine,
                  angle_degrees=math.degrees(math.acos(cosine)) if cosine is not None else None,
                  norm_ratio=math.sqrt(ae/re) if re else None,
                  maximum_absolute_error=max(row["max_abs_error"] for row in rows.values()))
    if masks is not None:
        for bucket in ("near_zero", "outside_near_zero"):
            totals[bucket] = {key: sum(row[bucket][key] for row in rows.values()) for key in ("count", "error_energy")}
            totals[bucket]["error_energy_fraction"] = totals[bucket]["error_energy"]/ee if ee else 0.
    return {"global": totals, "rows": rows}


def norm_direction(reference, actual):
    ratio = norm64(actual)/max(norm64(reference), 1e-30)
    magnitude = {name: (ratio-1)*value.double() for name, value in reference.items()}
    direction = {name: actual[name].double()-ratio*value.double() for name, value in reference.items()}
    total = {name: actual[name].double()-value.double() for name, value in reference.items()}
    mag, direct, error = (norm64(values)**2 for values in (magnitude, direction, total))
    cross = 2*sum(float((magnitude[name]*direction[name]).sum()) for name in reference)
    return {"norm_ratio": ratio, "magnitude_component_error_energy": mag,
            "direction_component_error_energy": direct, "twice_cross_inner_product": cross,
            "total_error_energy": error, "energy_identity_residual": error-mag-direct-cross,
            "definition": "a-r=(||a||/||r||-1)*r + [a-(||a||/||r||)*r]; components need not be orthogonal"}


def bound_record(error, bound):
    outside = error.abs() > bound
    return {"outside_count": int(outside.sum()), "maximum_error": float(error.abs().max()),
            "maximum_bound_ratio": float((error.abs()/bound.clamp_min(SUBNORMAL)).max()),
            "pass": not bool(outside.any())}


def local_reference(weights, raw, prior, native, group):
    """Fixed-operand optimizer arithmetic, separate from gradient perturbation."""
    independent_clipped, norm, coefficient = clip64(raw)
    native_scalar = torch.clamp(torch.tensor(1., dtype=torch.float32)/(torch.tensor(native["clip_norm"], dtype=torch.float32)+1e-6), max=1.)
    coefficient_exact = float(native_scalar) == native["clip_coefficient"]
    norm_error = abs(native["clip_norm"]-norm)/max(norm, 1e-30)
    oracle = adam64(weights, native["clipped_gradients"], prior, group)
    rows = {}; b1, b2 = group["betas"]
    for name, weight in weights.items():
        g = native["clipped_gradients"][name].double()
        expected_m, expected_v = (oracle["moments"][name][key] for key in ("exp_avg", "exp_avg_sq"))
        m_bound = 8*U*(b1*prior[name]["exp_avg"].double().abs()+(1-b1)*g.abs())+8*SUBNORMAL
        v_bound = 8*U*(b2*prior[name]["exp_avg_sq"].double()+(1-b2)*g.square())+8*SUBNORMAL
        state = native["state"][name]
        conditional_increment = increment_from_post_moments(state, group)
        conditional_after = weight.double()+conditional_increment
        write_bound = 16*U*conditional_increment.abs()+2*U*torch.maximum(weight.double().abs(), conditional_after.abs())+8*SUBNORMAL
        stored = native["weights"][name]
        values = [raw[name], native["clipped_gradients"][name], stored, state["exp_avg"], state["exp_avg_sq"], state["step"]]
        finite_fp32 = all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in values)
        row = {"finite_fp32": finite_fp32,
               "clipped_gradient_single_multiply_exact": torch.equal(raw[name]*native_scalar, native["clipped_gradients"][name]),
               "step_exact": float(state["step"]) == float(prior[name]["step"])+1,
               "second_moment_nonnegative": bool((state["exp_avg_sq"] >= 0).all()),
               "delta_packet_exact": torch.equal(stored.double()-weight.double(), native["deltas"][name]),
               "moment_first": bound_record(state["exp_avg"].double()-expected_m, m_bound),
               "moment_second": bound_record(state["exp_avg_sq"].double()-expected_v, v_bound),
               "conditional_weight_write": bound_record(stored.double()-conditional_after, write_bound),
               "conditional_correct_rounding_exact_count": int((stored == conditional_after.float()).sum())}
        row["pass"] = all(row[key] for key in ("finite_fp32", "clipped_gradient_single_multiply_exact", "step_exact", "second_moment_nonnegative", "delta_packet_exact")) and all(row[key]["pass"] for key in ("moment_first", "moment_second", "conditional_weight_write"))
        rows[name] = row
    result = {"native_norm": native["clip_norm"], "fp64_norm": norm, "norm_relative_error": norm_error,
              "norm_screen_pass": norm_error <= 16*U, "native_coefficient": native["clip_coefficient"],
              "fp64_coefficient": coefficient, "native_coefficient_exact": coefficient_exact,
              "rows": rows, "failed_tensors": [name for name, row in rows.items() if not row["pass"]],
              "native_delta_vs_ideal_increment": compare_maps(oracle["increments"], native["deltas"]),
              "native_delta_vs_fp32_write_realized": compare_maps(oracle["realized_deltas"], native["deltas"]),
              "fp32_write_effect": compare_maps(oracle["increments"], oracle["realized_deltas"])}
    result["pass"] = result["norm_screen_pass"] and coefficient_exact and not result["failed_tensors"]
    return result, oracle


def load_case(case_dir):
    report = json.loads((case_dir/"report.json").read_text())
    if report["status"] != "diagnostics_complete" or report["criteria"]["sha256"] != OLD_CONTRACT:
        raise ValueError("Expected complete old NUM case and unchanged original criteria")
    tensor_path = case_dir/"tensors.pt"
    checkpoint_path = Path(report["checkpoint"]["path"])
    if digest(tensor_path) != report["tensor_artifact"]["sha256"] or digest(checkpoint_path) != report["checkpoint"]["sha256"]:
        raise ValueError("Retained tensor/checkpoint SHA mismatch")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    names = list(payload["packets"]["tiled_fp32"]["parameters"])
    if names != list(checkpoint["model"]) or len(set(names)) != len(names):
        raise ValueError("Canonical optimizer/model ordering is not the validated one-group ordering")
    groups = checkpoint["optimizer"]["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != len(names):
        raise ValueError("Expected one optimizer group with every canonical parameter once")
    group = groups[0]; check_hyperparameters(group)
    if len(set(group["params"])) != len(names):
        raise ValueError("Duplicate optimizer parameter identifier")
    state = {name: checkpoint["optimizer"]["state"][pid] for name, pid in zip(names, group["params"])}
    if checkpoint["completed_updates"] != report["checkpoint"]["completed_updates"]:
        raise ValueError("Checkpoint update metadata differs")
    for name in names:
        weight = checkpoint["model"][name]
        moments = state[name]
        if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unexpected Adam state fields")
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
               for value in (weight, *moments.values())):
            raise ValueError("Retained FP32 weight/moment finiteness failure")
        if any(moments[key].shape != weight.shape for key in ("exp_avg", "exp_avg_sq")) or not bool((moments["exp_avg_sq"] >= 0).all()):
            raise ValueError("Retained moment shape/value failure")
    for arm in ARMS:
        if any(list(payload[section][arm][key]) != names for section, key in
               (("packets", "parameters"), ("steps", "weights"), ("steps", "state"), ("steps", "deltas"), ("steps", "clipped_gradients"))):
            raise ValueError("Native packet canonical ordering mismatch")
    if any(float(value["step"]) != checkpoint["completed_updates"] for value in state.values()):
        raise ValueError("Retained moment step does not match checkpoint")
    return report, payload, checkpoint["model"], state, group


def analyze_case(case_dir, output):
    report, payload, weights, state, group = load_case(case_dir)
    raw = {arm: payload["packets"][arm]["parameters"] for arm in ARMS}
    native = payload["steps"]
    masks = {name: value.double().abs() <= 2*2**-7*float(value.double().square().mean().sqrt())+2e-6 for name, value in raw["tiled_fp32"].items()}
    _, _, common = clip64(raw["tiled_fp32"])
    local, variants, reference_tensors = {}, {}, {}
    for arm in ARMS:
        local[arm], conditioned = local_reference(weights, raw[arm], state, native[arm], group)
        own_clipped, _, _ = clip64(raw[arm])
        common_clipped = {name: value.double()*common for name, value in raw[arm].items()}
        own, shared = (adam64(weights, gradients, state, group) for gradients in (own_clipped, common_clipped))
        variants[arm] = {"native": native[arm]["deltas"], "fp64_own_clip": own["increments"],
                         "fp64_common_clip": shared["increments"], "fp64_native_clipped": conditioned["increments"],
                         "fp32_write_own_clip": own["realized_deltas"], "fp32_write_common_clip": shared["realized_deltas"]}
        reference_tensors[arm] = {"variants": variants[arm], "fp64_own_clipped": own_clipped,
                                 "fp64_common_clipped": common_clipped,
                                 "fp64_post_moments_with_native_clipped": conditioned["moments"]}
    policy = {name: compare_maps(variants["tiled_fp32"][name], variants["tiled_bf16"][name], masks)
              for name in variants["tiled_fp32"]}
    gradients = {"raw": compare_maps(raw["tiled_fp32"], raw["tiled_bf16"], masks),
                 "native_clipped": compare_maps(native["tiled_fp32"]["clipped_gradients"], native["tiled_bf16"]["clipped_gradients"], masks),
                 "fp64_own_clipped": compare_maps(reference_tensors["tiled_fp32"]["fp64_own_clipped"], reference_tensors["tiled_bf16"]["fp64_own_clipped"], masks),
                 "fp64_common_clipped": compare_maps(reference_tensors["tiled_fp32"]["fp64_common_clipped"], reference_tensors["tiled_bf16"]["fp64_common_clipped"], masks)}
    old = report["comparisons"]["tiled_bf16_vs_tiled_fp32"]
    result = {"source_case": str(case_dir), "report_sha256": digest(case_dir/"report.json"),
              "tensor_sha256": report["tensor_artifact"]["sha256"], "checkpoint": report["checkpoint"],
              "fixture": report["fixture"], "original_criteria": report["criteria"],
              "original_machine_screens_pass": report["machine_screens_pass"],
              "original_comparison": old, "optimizer": {key: value for key, value in group.items() if key != "params"},
              "canonical_parameter_tensors": len(weights), "local_optimizer_checks": local,
              "local_optimizer_checks_pass": all(value["pass"] for value in local.values()),
              "gradient_comparisons": gradients,
              "raw_norm_direction_decomposition": norm_direction(raw["tiled_fp32"], raw["tiled_bf16"]),
              "clipped_norm_direction_decomposition": norm_direction(native["tiled_fp32"]["clipped_gradients"], native["tiled_bf16"]["clipped_gradients"]),
              "adam_policy_comparisons": policy, "common_clipping_coefficient": common,
              "numerical_clearance": False,
              "interpretation_scope": "No model execution. Fixed-gradient arithmetic correctness is distinct from unchanged whole-policy gradient/Adam distance guards. Common clipping is a diagnostic counterfactual only."}
    for value in policy.values():
        value["original_trained_adam_screen_pass_if_applied"] = value["global"]["relative_l2"] <= 2**-6
        value["maximum_error_over_lr"] = value["global"]["maximum_absolute_error"]/group["lr"]
    # The original report's schema is retained whole; the independent native
    # reduction is also checked against its named Adam summary when available.
    old_adam = old.get("optimizer", old.get("adam", old.get("step")))
    if isinstance(old_adam, dict) and "global_delta_relative_l2" in old_adam:
        if not math.isclose(old_adam["global_delta_relative_l2"], policy["native"]["global"]["relative_l2"], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("Independent native-delta reduction disagrees with archived report")
        result["original_adam_summary_reproduced"] = True
    else:
        result["original_adam_summary_reproduced"] = None
    if digest(case_dir/"tensors.pt") != result["tensor_sha256"] or digest(Path(report["checkpoint"]["path"])) != result["checkpoint"]["sha256"]:
        raise ValueError("Retained input changed during analysis")
    output.mkdir()
    torch.save({"references": reference_tensors, "gradient_near_zero_masks": masks,
                "source_tensor_sha256": result["tensor_sha256"], "source_checkpoint": result["checkpoint"]}, output/"references.pt")
    result["reference_tensors"] = {"path": str(output/"references.pt"), "sha256": digest(output/"references.pt")}
    save_json(output/"report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path("docs/reports/cdrm-numerical-resolution/reference-contract.md"))
    add_wandb_arguments(parser)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists() or torch.cuda.is_available():
        raise RuntimeError("Use the explicitly GPU-disabled project CPU container")
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(DEFAULT_ROOT.resolve()):
        raise ValueError("Use a new output within the numerical-resolution lineage")
    if digest(args.contract) != NEW_CONTRACT:
        raise ValueError("Prospective reference contract hash changed")
    if not args.wandb_project:
        parser.error("Graphable numerical diagnostics require online --wandb-project")
    if len({str(path.resolve()) for path in args.case_dir}) != len(args.case_dir):
        raise ValueError("Duplicate source case")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True)
    result = {"schema": "cdrm-precision-adam-reference-v1", "evidence": "NUM-CPU", "status": "running",
              "numerical_clearance": False, "runtime": {"torch": torch.__version__, "python": sys.version, "device": "cpu", "cuda_available": False},
              "contract": {"path": str(args.contract), "sha256": digest(args.contract)}}
    tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
                            name=args.wandb_run_name, output_dir=args.output_dir)
    result["wandb"] = tracker.record
    try:
        source_dir = args.output_dir/"source"; source_dir.mkdir()
        files = [Path(__file__), Path(__file__).with_name("experiment_tracking.py"), args.contract,
                 Path(inspect.getfile(torch.optim.AdamW)), Path(inspect.getfile(torch.optim.Adam)),
                 Path(inspect.getfile(torch.nn.utils.clip_grad_norm_))]
        result["source_sha256"] = {}
        for index, source in enumerate(files):
            target = source_dir/f"{index:02d}-{source.name}"
            shutil.copyfile(source, target)
            result["source_sha256"][str(source)] = digest(source)
        tracker.start({"evidence": "NUM-CPU", "sources": result["source_sha256"], "contract": result["contract"]})
        result["wandb"] = tracker.record
        result["cases"] = []
        for index, path in enumerate(args.case_dir):
            case = analyze_case(path, args.output_dir/f"case-{index}")
            result["cases"].append(case)
            metrics = {"case_index": index, "local_optimizer_checks_pass": case["local_optimizer_checks_pass"]}
            metrics.update({f"gradient/{key}/relative_l2": value["global"]["relative_l2"] for key, value in case["gradient_comparisons"].items()})
            metrics.update({f"adam/{key}/relative_l2": value["global"]["relative_l2"] for key, value in case["adam_policy_comparisons"].items()})
            tracker.log(metrics, step=index)
            print(json.dumps({"case": str(path), **metrics}), flush=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(result["cases"]), figsize=(6*len(result["cases"]), 4), squeeze=False)
        keys = ("native", "fp64_own_clip", "fp64_common_clip", "fp32_write_own_clip")
        for axis, case in zip(axes[0], result["cases"]):
            axis.bar(range(len(keys)), [100*case["adam_policy_comparisons"][key]["global"]["relative_l2"] for key in keys])
            axis.axhline(100*2**-6, color="red", linestyle="--", label="Original Adam guard")
            axis.set_xticks(range(len(keys)), ["Native", "FP64 own clip", "FP64 common clip", "FP32 write"], rotation=20)
            axis.set_ylabel("BF16-gradient vs FP32-gradient update error (%)")
            axis.set_title(Path(case["source_case"]).name); axis.legend()
        fig.tight_layout(); fig.savefig(args.output_dir/"adam-reference.png", dpi=160); plt.close(fig)
        import wandb
        tracker.log({"adam_reference_plot": wandb.Image(str(args.output_dir/"adam-reference.png"))})
        result["local_optimizer_checks_pass"] = all(case["local_optimizer_checks_pass"] for case in result["cases"])
        tracker.summary({"local_optimizer_checks_pass": result["local_optimizer_checks_pass"], "numerical_clearance": False,
                         "original_failures_preserved": True})
        if any(digest(Path(path)) != sha for path, sha in result["source_sha256"].items()):
            raise ValueError("Source changed during analysis")
        result["status"] = "diagnostics_complete"
    except BaseException as error:
        result.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            tracker.finish(succeeded=result["status"] == "diagnostics_complete")
        except BaseException as error:
            result.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            save_json(args.output_dir/"report.json", result)


if __name__ == "__main__":
    main()
