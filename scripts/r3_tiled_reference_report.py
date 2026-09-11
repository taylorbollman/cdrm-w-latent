#!/usr/bin/env python3
"""CPU-only reference-centered analysis of saved R3 mixed-precision packets.

This reads trusted local artifacts; it never imports or executes a model.
Numerical screens alone cannot establish local-backward or operational clearance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time

# This utility is intentionally incapable of selecting a CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

EPS = 2.0**-7
ATOL, RTOL, FLOOR = 2e-6, 2e-5, 1e-12
THRESHOLDS = {
    "bf16_epsilon": EPS,
    "fp32_atol": ATOL,
    "fp32_rtol": RTOL,
    "global_gradient_relative_l2": 2 * EPS,
    "per_tensor_gradient_relative_l2": 4 * EPS,
    "per_tensor_gradient_maximum_over_reference_maximum": 8 * EPS,
    "logit_relative_l2": 2 * EPS,
    "absolute_ce_nats": 0.01,
    "initial_adam_delta_cosine_minimum": 0.99,
    "trained_adam_delta_relative_l2": 2 * EPS,
    "historical_backend_relative_l2": 2 * EPS,
    "historical_maximum_over_reference_rms": 8 * EPS,
    "fp32_scale_aware": 2e-5,
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def ratio(a, b):
    return a / max(b, FLOOR)


def norm(value):
    return float(torch.linalg.vector_norm(value))


def maximum(value):
    return float(value.abs().max()) if value.numel() else 0.0


def as_float64(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("Expected a present tensor")
    if value.device.type != "cpu" or value.is_complex() or value.layout != torch.strided:
        raise ValueError("Expected a dense real CPU tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Expected a finite tensor")
    return value.detach().double()


def gradient_map(packet):
    return {**packet["parameters"],
            **{f"input/{name}": value for name, value in packet["inputs"].items()}}


def mask_summary(reference, actual, error, mask):
    count = int(mask.sum())
    return {
        "coordinate_count": count,
        "error_energy": float(error[mask].square().sum()),
        "maximum_absolute_error": maximum(error[mask]),
        "strict_sign_flip_count": int(((reference * actual < 0) & mask).sum()),
        "sign_change_including_zero_count": int(((reference.sign() != actual.sign()) & mask).sum()),
    }


def metrics(reference, actual, *, anchor=None, top_k=5):
    """Linear reductions and small top-k; no full-coordinate sorting."""
    r, a = as_float64(reference), as_float64(actual)
    if r.shape != a.shape:
        raise ValueError(f"Shape mismatch: {r.shape} versus {a.shape}")
    if anchor is None:
        anchor = r
    else:
        anchor = as_float64(anchor)
        if anchor.shape != r.shape:
            raise ValueError("Near-zero anchor shape differs")
    e = a - r
    n = r.numel()
    rn, an, en = norm(r), norm(a), norm(e)
    rms = rn / math.sqrt(n) if n else 0.0
    f = ATOL + RTOL * r.abs()
    active = r != 0
    active_count = int(active.sum())
    active_rms = rn / math.sqrt(active_count) if active_count else 0.0
    near_threshold = 2 * EPS * (norm(anchor) / math.sqrt(n) if n else 0.0) + ATOL
    near = anchor.abs() <= near_threshold
    abs_error = e.abs()
    violations = abs_error > f
    max_error, max_reference = maximum(e), maximum(r)
    dot = float((r * a).sum())
    result = {
        "shape": list(r.shape), "numel": n,
        "reference_dtype": str(reference.dtype), "actual_dtype": str(actual.dtype),
        "reference_l2": rn, "actual_l2": an, "error_l2": en,
        "reference_rms": rms, "error_rms": en / math.sqrt(n) if n else 0.0,
        "reference_maximum": max_reference, "maximum_absolute_error": max_error,
        "relative_l2": ratio(en, rn),
        "maximum_over_reference_maximum": ratio(max_error, max_reference),
        "maximum_over_reference_rms": ratio(max_error, rms),
        "cosine": dot / (rn * an) if rn * an > 1e-24 else None,
        "dot": dot, "fp32_floor_l2": norm(f), "fp32_floor_maximum": maximum(f),
        "mean_signed_error": float(e.mean()) if n else 0.0,
        "historical_fp32_elementwise_failure_count": int(violations.sum()),
        "historical_fp32_elementwise_pass": not bool(violations.any()),
        "historical_fp32_scale_aware_pass": ratio(en, rn) <= 2e-5 and ratio(max_error, rms) <= 2e-5,
        "historical_bf16_relative_l2_pass": ratio(en, rn) <= 2 * EPS,
        "historical_bf16_max_rms_pass": ratio(max_error, rms) <= 8 * EPS,
        "active_reference_count": active_count,
        "exact_zero_reference_count": n - active_count,
        "actual_nonzero_at_reference_zero_count": int(((a != 0) & ~active).sum()),
        "active_reference_rms": active_rms,
        "maximum_over_active_reference_rms": ratio(max_error, active_rms),
        "inactive_error_energy": float(e[~active].square().sum()),
        "near_zero_threshold": near_threshold,
        "near_zero": mask_summary(r, a, e, near),
        "outside_near_zero": mask_summary(r, a, e, ~near),
        "top_absolute_error_coordinates": [],
    }
    if n and top_k:
        indices = torch.topk(abs_error.flatten(), min(top_k, n), sorted=True).indices.tolist()
        for index in indices:
            rv, av = float(r.flatten()[index]), float(a.flatten()[index])
            result["top_absolute_error_coordinates"].append({
                "flat_index": index, "reference": rv, "actual": av,
                "signed_error": av - rv, "absolute_error": abs(av - rv),
                "reference_over_rms": ratio(abs(rv), rms),
                "in_near_zero_set": bool(near.flatten()[index]),
                "strict_sign_flip": rv * av < 0,
                "fp32_elementwise_failure": bool(violations.flatten()[index]),
            })
    result["prospective_l2_pass"] = en <= 4 * EPS * rn + result["fp32_floor_l2"]
    result["prospective_maximum_pass"] = max_error <= 8 * EPS * max_reference + result["fp32_floor_maximum"]
    return result


def aggregate(rows):
    def energy(name):
        return sum(row[name] ** 2 for row in rows.values())
    rn = math.sqrt(energy("reference_l2"))
    an = math.sqrt(energy("actual_l2"))
    en = math.sqrt(energy("error_l2"))
    floor = math.sqrt(energy("fp32_floor_l2"))
    result = {
        "tensor_count": len(rows), "numel": sum(row["numel"] for row in rows.values()),
        "reference_l2": rn, "actual_l2": an, "error_l2": en,
        "relative_l2": ratio(en, rn), "fp32_floor_l2": floor,
        "cosine": sum(row["dot"] for row in rows.values()) / (rn * an) if rn * an > 1e-24 else None,
        "maximum_absolute_error": max((row["maximum_absolute_error"] for row in rows.values()), default=0.0),
        "prospective_global_l2_pass": en <= 2 * EPS * rn + floor,
    }
    for bucket in ("near_zero", "outside_near_zero"):
        result[bucket] = {key: sum(row[bucket][key] for row in rows.values()) for key in (
            "coordinate_count", "error_energy", "strict_sign_flip_count", "sign_change_including_zero_count")}
        result[bucket]["maximum_absolute_error"] = max((row[bucket]["maximum_absolute_error"] for row in rows.values()), default=0.0)
        result[bucket]["fraction_of_error_energy"] = result[bucket]["error_energy"] / (en * en) if en else 0.0
    return result


def audit_state(packets, steps):
    failures = []
    for arm, packet in packets.items():
        checks = list(gradient_map(packet).items())
        for group in ("weights", "clipped_gradients"):
            checks.extend((f"step/{group}/{name}", value) for name, value in steps[arm][group].items())
        for name, state in steps[arm]["state"].items():
            checks.extend((f"state/{name}/{key}", value) for key, value in state.items() if torch.is_tensor(value))
        for name, value in checks:
            if not torch.is_tensor(value) or value.dtype != torch.float32 or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
                failures.append(f"{arm}/{name}")
    return {"present_finite_fp32_pass": not failures, "failures": failures}


def bitwise_tree(reference, actual):
    result = {"checked_tensor_count": 0, "checked_scalar_count": 0,
              "mismatches": [], "missing_from_actual": [], "additional_actual_keys": []}
    def walk(r, a, path):
        if torch.is_tensor(r):
            result["checked_tensor_count"] += 1
            same = (torch.is_tensor(a) and r.dtype == a.dtype and r.shape == a.shape
                    and torch.equal(r.contiguous().reshape(-1).view(torch.uint8),
                                    a.contiguous().reshape(-1).view(torch.uint8)))
            if not same:
                result["mismatches"].append(path)
        elif isinstance(r, dict) and isinstance(a, dict):
            result["missing_from_actual"].extend(f"{path}/{k}" for k in r.keys() - a.keys())
            result["additional_actual_keys"].extend(f"{path}/{k}" for k in a.keys() - r.keys())
            for key in r.keys() & a.keys():
                walk(r[key], a[key], f"{path}/{key}")
        elif isinstance(r, (tuple, list)) and isinstance(a, (tuple, list)):
            if len(r) != len(a):
                result["mismatches"].append(path + "/length")
            for index, (rv, av) in enumerate(zip(r, a)):
                walk(rv, av, f"{path}/{index}")
        else:
            result["checked_scalar_count"] += 1
            if type(r) is not type(a) or r != a:
                result["mismatches"].append(path)
    walk(reference, actual, "root")
    for key in ("mismatches", "missing_from_actual", "additional_actual_keys"):
        result[key].sort()
    result["shared_values_bitwise_equal"] = not result["mismatches"] and not result["missing_from_actual"]
    return result


def reproduction(prior_dir, current, current_report):
    prior_report = json.loads((prior_dir / "report.json").read_text())
    prior = torch.load(prior_dir / "tensors.pt", map_location="cpu", weights_only=False)
    shared = sorted(prior["packets"].keys() & current["packets"].keys())
    out = {"prior_directory": str(prior_dir), "prior_report_sha256": digest(prior_dir / "report.json"),
           "prior_tensor_sha256": digest(prior_dir / "tensors.pt"), "shared_arms": shared,
           "checkpoint_sha_match": prior_report["checkpoint"]["sha256"] == current_report["checkpoint"]["sha256"],
           "fixture_sha_match": prior_report["fixture"]["sha256"] == current_report["fixture"]["sha256"],
           "data": bitwise_tree({k: prior[k] for k in ("tokens", "labels")},
                                 {k: current[k] for k in ("tokens", "labels")}),
           "packets": {}, "steps": {}, "observed_intermediates": {}}
    for arm in shared:
        out["packets"][arm] = bitwise_tree(prior["packets"][arm], current["packets"][arm])
        out["steps"][arm] = bitwise_tree(prior["steps"][arm], current["steps"][arm])
        old = prior.get("observed_intermediates", {}).get(arm, {})
        new = current.get("observed_intermediates", {}).get(arm, {})
        if old.keys() == new.keys() and old:
            out["observed_intermediates"][arm] = {"comparable": True, **bitwise_tree(old, new)}
        else:
            out["observed_intermediates"][arm] = {
                "comparable": False, "reason": "No shared observation set, or observer keys differ",
                "prior_count": len(old), "current_count": len(new),
                "prior_only_keys": sorted(old.keys() - new.keys()),
                "current_only_keys": sorted(new.keys() - old.keys()),
            }
    out["shared_packet_and_step_reproduction_pass"] = (
        bool(shared) and out["checkpoint_sha_match"] and out["fixture_sha_match"]
        and out["data"]["shared_values_bitwise_equal"]
        and all(row["shared_values_bitwise_equal"] for group in ("packets", "steps") for row in out[group].values()))
    out["comparable_observations_bitwise_equal"] = all(
        row["shared_values_bitwise_equal"] for row in out["observed_intermediates"].values() if row["comparable"])
    return out


def analyze(case_dir, criteria, prior_dir=None):
    source = json.loads((case_dir / "report.json").read_text())
    if source["status"] != "diagnostics_complete":
        raise ValueError("Source model run did not complete")
    payload = torch.load(case_dir / "tensors.pt", map_location="cpu", weights_only=False)
    packets, steps = payload["packets"], payload["steps"]
    for required in ("tiled_fp32", "tiled_bf16"):
        if required not in packets or required not in steps:
            raise ValueError(f"Required arm missing: {required}")
    maps = {arm: gradient_map(packet) for arm, packet in packets.items()}
    names = list(maps["tiled_fp32"])
    if any(list(mapping) != names for mapping in maps.values()):
        raise ValueError("Canonical gradient coverage or order differs across arms")
    parameter_names = list(packets["tiled_fp32"]["parameters"])
    coverage_pass = len(parameter_names) == 100 and len(names) == 102 and set(packets["tiled_fp32"]["inputs"]) == {"embedding_output", "block3_input"}
    anchor_arm = "naive_fp32" if "naive_fp32" in maps else "tiled_fp32"
    state = audit_state(packets, steps)
    out = {
        "schema": "r3-tiled-reference-analysis-v1", "evidence": "CPU reductions of saved tensors only; no model execution",
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_directory": str(case_dir), "source_report_sha256": digest(case_dir / "report.json"),
        "source_tensor_sha256": digest(case_dir / "tensors.pt"), "analyzer_sha256": digest(__file__),
        "criteria": {"path": str(criteria), "sha256": digest(criteria), "implemented_thresholds": THRESHOLDS,
                     "note": "Thresholds are implemented above; the supplied document identifies their acceptance context."},
        "checkpoint": source["checkpoint"], "fixture": source["fixture"], "settings": source["settings"],
        "wandb": source.get("wandb"), "near_zero_anchor_arm": anchor_arm,
        "near_zero_formula": "abs(anchor) <= 0.015625 * RMS(anchor_parameter) + 2e-6",
        "expected_stage_b_coverage_pass": coverage_pass, "state_contract": state,
        "gradient_rows": {}, "old_one_sided_budget_rows": {},
        "original_comparison_flags": {}, "fp32_regression": {"available": False},
        "numerical_clearance": False,
        "disposition": "Machine screens only; local backward, tail attribution, training and recovery evidence remain required.",
    }
    if not state["present_finite_fp32_pass"]:
        out["machine_screens_pass"] = False
        out["disposition"] = "Invalid gradient/state contract; no finite-subset numerical reductions performed."
        return out
    for name in names:
        row = metrics(maps["tiled_fp32"][name], maps["tiled_bf16"][name], anchor=maps[anchor_arm][name])
        out["gradient_rows"][name] = row
        if all(arm in maps for arm in ("naive_fp32", "naive_bf16")):
            ref, naive, tiled = (as_float64(maps[arm][name]) for arm in ("naive_fp32", "naive_bf16", "tiled_bf16"))
            floor = ATOL + RTOL * ref.abs()
            ordinary, added = naive - ref, tiled - naive
            ln, an, fn = norm(ordinary), norm(added), norm(floor)
            lm, am, fm = maximum(ordinary), maximum(added), maximum(floor)
            out["old_one_sided_budget_rows"][name] = {
                "ordinary_error_l2": ln, "added_error_l2": an, "fp32_floor_l2": fn,
                "ordinary_error_maximum": lm, "added_error_maximum": am, "fp32_floor_maximum": fm,
                "backend_relative_l2": ratio(an, norm(naive)),
                "backend_relative_l2_pass": ratio(an, norm(naive)) <= 2 * EPS,
                "added_l2_budget_pass": an <= ln + fn,
                "added_maximum_budget_pass": am <= lm + fm,
                "maximum_budget_ratio": ratio(am, lm + fm),
            }
    gradients = out["gradient_rows"]
    out["global_parameters"] = aggregate({name: gradients[name] for name in parameter_names})
    for label, comparison in source.get("comparisons", {}).items():
        group = comparison.get("gradients", {})
        out["original_comparison_flags"][label] = {
            "historical_bf16_failed_tensors": group.get("historical_bf16_failed_tensors"),
            "fp32_elementwise_failed_tensors": group.get("fp32_elementwise_failed_tensors"),
            "logit_elementwise_pass": comparison.get("logits", {}).get("elementwise_pass"),
            "logit_scale_aware_pass": comparison.get("logits", {}).get("scale_aware_pass"),
        }
    if "naive_fp32" in maps:
        rows = {name: metrics(maps["naive_fp32"][name], maps["tiled_fp32"][name], top_k=1) for name in names}
        logits = metrics(packets["naive_fp32"]["logits"], packets["tiled_fp32"]["logits"], top_k=1)
        out["fp32_regression"] = {
            "available": True, "rows": rows, "logits": logits,
            "elementwise_failed_tensors": [name for name, row in rows.items() if not row["historical_fp32_elementwise_pass"]],
            "scale_aware_failed_tensors": [name for name, row in rows.items() if not row["historical_fp32_scale_aware_pass"]],
            "pass": all(row["historical_fp32_elementwise_pass"] and row["historical_fp32_scale_aware_pass"] for row in rows.values())
                    and logits["historical_fp32_elementwise_pass"] and logits["historical_fp32_scale_aware_pass"],
        }
    out["logits"] = metrics(packets["tiled_fp32"]["logits"], packets["tiled_bf16"]["logits"])
    logit = out["logits"]
    logit["prospective_logit_l2_pass"] = logit["error_l2"] <= 2 * EPS * logit["reference_l2"] + logit["fp32_floor_l2"]
    out["ce"] = {"reference": packets["tiled_fp32"]["loss"], "actual": packets["tiled_bf16"]["loss"]}
    out["ce"]["absolute_error"] = abs(out["ce"]["actual"] - out["ce"]["reference"])
    out["ce"]["pass"] = out["ce"]["absolute_error"] <= THRESHOLDS["absolute_ce_nats"]
    delta_rows = {}
    for name in parameter_names:
        row = metrics(steps["tiled_fp32"]["deltas"][name], steps["tiled_bf16"]["deltas"][name], anchor=maps[anchor_arm][name])
        rgrad, agrad, anchor = (as_float64(maps[arm][name]) for arm in ("tiled_fp32", "tiled_bf16", anchor_arm))
        near = anchor.abs() <= 2 * EPS * norm(anchor) / math.sqrt(anchor.numel()) + ATOL
        for bucket, mask in (("near_zero", near), ("outside_near_zero", ~near)):
            row[bucket]["gradient_strict_sign_flip_count"] = int(((rgrad * agrad < 0) & mask).sum())
            row[bucket]["gradient_sign_change_including_zero_count"] = int(((rgrad.sign() != agrad.sign()) & mask).sum())
        delta_rows[name] = row
    delta_global = aggregate(delta_rows)
    for bucket in ("near_zero", "outside_near_zero"):
        for key in ("gradient_strict_sign_flip_count", "gradient_sign_change_including_zero_count"):
            delta_global[bucket][key] = sum(row[bucket][key] for row in delta_rows.values())
    lr = float(source["optimizer"]["lr"])
    delta_global["maximum_error_over_lr"] = delta_global["maximum_absolute_error"] / lr if lr else None
    initialized = source["checkpoint"]["completed_updates"] == 0
    delta_pass = (delta_global["cosine"] is not None and delta_global["cosine"] >= 0.99) if initialized else delta_global["relative_l2"] <= 2 * EPS
    out["optimizer"] = {
        "initialization": initialized, "source_settings": source["optimizer"],
        "clipping": {arm: {key: steps[arm][key] for key in ("clip_norm", "clip_coefficient")} for arm in steps},
        "delta_rows": delta_rows, "global_delta": delta_global, "guardrail_pass": delta_pass,
        "guardrail": "initial global delta cosine >= .99" if initialized else "trained global delta relative L2 <= .015625",
        "note": "Initial sign-sensitive tails remain diagnostics; no universal coordinate delta bound is imposed.",
    }
    failed_l2 = [name for name, row in gradients.items() if not row["prospective_l2_pass"]]
    failed_max = [name for name, row in gradients.items() if not row["prospective_maximum_pass"]]
    old = out["old_one_sided_budget_rows"]
    b64_scope = source["fixture"]["shape"] == [64, 128] and not source["fixture"].get("accumulation", True)
    numeric_pass = (coverage_pass and not failed_l2 and not failed_max
                    and out["global_parameters"]["prospective_global_l2_pass"]
                    and logit["prospective_logit_l2_pass"] and out["ce"]["pass"])
    out["screen_summary"] = {
        "actual_b64_t128_scope": b64_scope,
        "per_tensor_l2_failures": failed_l2, "per_tensor_maximum_failures": failed_max,
        "old_one_sided_available": bool(old),
        "old_added_l2_failures": [name for name, row in old.items() if not row["added_l2_budget_pass"]],
        "old_added_maximum_failures": [name for name, row in old.items() if not row["added_maximum_budget_pass"]],
        "old_backend_relative_l2_failures": [name for name, row in old.items() if not row["backend_relative_l2_pass"]],
        "primary_numerical_screens_pass": numeric_pass,
        "optimizer_guardrail_pass": delta_pass,
        "fp32_regression_pass": out["fp32_regression"].get("pass"),
    }
    # A tiled-only follow-up can pass its measured screens; it must cite a
    # separate matching FP32 regression, never manufacture a missing pass.
    out["machine_screens_pass"] = numeric_pass and delta_pass and out["fp32_regression"].get("pass", True)
    out["machine_b64_screens_pass"] = b64_scope and out["machine_screens_pass"]
    worst = max(gradients, key=lambda name: gradients[name]["relative_l2"])
    out["summary_metrics"] = {
        "gradient/global_relative_l2": out["global_parameters"]["relative_l2"],
        "gradient/worst_tensor_relative_l2": gradients[worst]["relative_l2"],
        "gradient/worst_maximum_over_reference_maximum": max(row["maximum_over_reference_maximum"] for row in gradients.values()),
        "gradient/tensor_l2_failure_count": len(failed_l2),
        "gradient/tensor_maximum_failure_count": len(failed_max),
        "historical/added_maximum_failure_count": len(out["screen_summary"]["old_added_maximum_failures"]) if old else None,
        "forward/logit_relative_l2": logit["relative_l2"], "forward/ce_absolute_error": out["ce"]["absolute_error"],
        "optimizer/delta_relative_l2": delta_global["relative_l2"], "optimizer/delta_cosine": delta_global["cosine"],
        "optimizer/near_zero_error_energy_fraction": delta_global["near_zero"]["fraction_of_error_energy"],
        "optimizer/outside_near_zero_gradient_sign_flips": delta_global["outside_near_zero"]["gradient_strict_sign_flip_count"],
        "screens/machine_b64_pass": out["machine_b64_screens_pass"],
    }
    out["worst_gradient_relative_l2_tensor"] = worst
    if prior_dir:
        out["prior_reproduction"] = reproduction(prior_dir, payload, source)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; existing files are never overwritten")
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--reproduce-prior", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    result = analyze(args.case_dir, args.criteria, args.reproduce_prior)
    result["elapsed_cpu_analysis_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "machine_screens_pass": result["machine_screens_pass"],
                      "numerical_clearance": False, "summary_metrics": result.get("summary_metrics", {})}, allow_nan=False))


if __name__ == "__main__":
    main()
