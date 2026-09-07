#!/usr/bin/env python3
"""CPU-only decomposition of retained first-step Adam numerical differences.

This reads existing numerical artifacts; it neither runs a model nor changes the
predeclared acceptance criteria. Run with CDRM_DOCKER_GPUS=none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RELATIVE_UPDATE_LIMIT = 1e-3
MAX_UPDATE_LR_FRACTION = 0.01


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def location(path):
    return str(Path(path).resolve().relative_to(ROOT))


def metrics(reference, actual, lr=None):
    reference, actual = reference.double(), actual.double()
    error = actual - reference
    assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
    ref_l2 = float(reference.norm())
    ref_rms = ref_l2 / math.sqrt(reference.numel())
    maximum, index = error.abs().reshape(-1).max(dim=0)
    result = {"numel": reference.numel(), "ref_l2": ref_l2, "ref_rms": ref_rms,
            "error_l2": float(error.norm()), "max_abs_error": float(maximum),
            "rel_l2": float(error.norm()) / max(ref_l2, 1e-12),
            "maxerr_over_ref_rms": float(maximum) / max(ref_rms, 1e-12),
            "changed_coordinates": int(torch.count_nonzero(error)),
            "worst_flat_index": int(index)}
    if lr is not None:
        result["coordinates_over_1pct_lr"] = int((error.abs() > MAX_UPDATE_LR_FRACTION * lr).sum())
    return result


def screen(row, lr):
    return (row["rel_l2"] <= RELATIVE_UPDATE_LIMIT
            and row["max_abs_error"] <= MAX_UPDATE_LR_FRACTION * lr)


def index_of(flat, shape):
    result = []
    for width in reversed(shape):
        result.append(flat % width)
        flat //= width
    return list(reversed(result))


def scalar(tensor, flat):
    return float(tensor.reshape(-1)[flat])


def array_summary(rows, lr):
    ref_l2 = math.sqrt(sum(r["ref_l2"] ** 2 for r in rows.values()))
    error_l2 = math.sqrt(sum(r["error_l2"] ** 2 for r in rows.values()))
    return {"tensor_count": len(rows), "coordinate_count": sum(r["numel"] for r in rows.values()),
            "changed_coordinates": sum(r["changed_coordinates"] for r in rows.values()),
            "coordinates_over_1pct_lr": sum(r["coordinates_over_1pct_lr"] for r in rows.values()),
            "global_ref_l2": ref_l2, "global_error_l2": error_l2,
            "global_relative_l2": error_l2 / max(ref_l2, 1e-12),
            "max_abs_error": max(r["max_abs_error"] for r in rows.values()),
            "max_tensor_relative_l2": max(r["rel_l2"] for r in rows.values()),
            "screen_failed_tensors": [n for n, r in rows.items() if not screen(r, lr)]}


def analyze(case_dir, oracle_dir=None):
    report_path = case_dir / "report.json"
    tensor_path = case_dir / "ce-and-update-tensors.pt"
    report = json.loads(report_path.read_text())
    settings = report["optimizer"]
    assert report["status"] == "diagnostics_complete"
    assert settings["initial_state_equal"] and settings["initial_state_steps"] == []
    assert settings["weight_decay"] == 0 and settings["name"] == "AdamW"
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False, mmap=True)
    lr, eps = settings["lr"], settings["eps"]
    beta1, beta2 = settings["betas"]
    sides = ("naive", "tiled")
    source_flags = {key: {k: v for k, v in report[key].items() if k != "rows"}
                    for key in ("gradients", "clipped_gradients", "post_step_weights", "updates", "optimizer_states")}
    result = {"case": case_dir.name, "report": {"path": location(report_path), "sha256": digest(report_path)},
              "tensors": {"path": location(tensor_path), "sha256": digest(tensor_path)},
              "fixture": report["fixture"], "optimizer": settings, "clipping": report["clipping"],
              "original_diagnostics": source_flags, "tensor_updates": {}, "flagged_update_coordinates": {},
              "flagged_moment_coordinates": {}, "assertions": {"initial_weights_paired": True,
                  "reconstructed_initial_weights_exactly_fp32": True, "all_step_counters_one": True}}
    oracle = None
    if oracle_dir is not None:
        oracle_report_path = oracle_dir / "report.json"
        oracle_tensor_path = oracle_dir / "fp64-ce-tensors.pt"
        oracle_report = json.loads(oracle_report_path.read_text())
        assert oracle_report["status"] == "diagnostics_complete"
        assert oracle_report["source_reference"]["report_sha256"] == result["report"]["sha256"]
        assert oracle_report["source_reference"]["tensors_sha256"] == result["tensors"]["sha256"]
        oracle = torch.load(oracle_tensor_path, map_location="cpu", weights_only=False, mmap=True)
        assert torch.equal(oracle["tokens"], payload["tokens"]) and torch.equal(oracle["labels"], payload["labels"])
        oracle_norm = math.sqrt(sum(float(g.square().sum()) for g in oracle["oracle"]["parameters"].values()))
        oracle_coefficient = min(settings["clip"] / (oracle_norm + 1e-6), 1.0)
        result["fp64_oracle"] = {"report": {"path": location(oracle_report_path), "sha256": digest(oracle_report_path)},
            "tensors": {"path": location(oracle_tensor_path), "sha256": digest(oracle_tensor_path)},
            "scope": oracle_report["scope"], "clip_norm_fp64": oracle_norm,
            "clip_coefficient_fp64": oracle_coefficient, "tensor_ideal_updates_vs_oracle": {}}
    update_rows = {key: {} for key in ("recorded", "ideal_from_gradients", "rounded_ideal", "from_recorded_moments")}
    moment_pairs = {key: {} for key in ("exp_avg", "exp_avg_sq")}
    names = list(payload["steps"]["naive"]["weights"])
    assert set(names) == set(payload["steps"]["tiled"]["weights"])
    max_update_name = max(names, key=lambda n: report["updates"]["rows"][n]["max_abs_error"])
    for name in names:
        values = {}
        for side in sides:
            step = payload["steps"][side]
            gradient = step["clipped_gradients"][name]
            assert gradient.dtype == torch.float32 and gradient.device.type == "cpu"
            state = step["state"][name]
            assert float(state["step"]) == 1
            before = step["weights"][name].double() - step["deltas"][name]
            assert torch.equal(before.float().double(), before)
            g = gradient.double()
            ideal = -lr * g / (g.abs() + eps)
            moment_ideal = {"exp_avg": (1 - beta1) * g, "exp_avg_sq": (1 - beta2) * g.square()}
            from_moments = -lr * (state["exp_avg"].double() / (1 - beta1)) / (
                (state["exp_avg_sq"].double() / (1 - beta2)).sqrt() + eps)
            values[side] = {"raw": payload["gradients"][side]["parameters"][name], "g": g,
                "before": before, "after": step["weights"][name].double(),
                "recorded": step["deltas"][name], "ideal_from_gradients": ideal,
                "rounded_ideal": (before + ideal).float().double() - before,
                "from_recorded_moments": from_moments, "state": state, "moment_ideal": moment_ideal}
        assert torch.equal(values["naive"]["before"], values["tiled"]["before"])
        oracle_values = None
        if oracle is not None:
            assert torch.equal(oracle["initial_fp64_weights"][name], values["naive"]["before"])
            oracle_raw = oracle["oracle"]["parameters"][name]
            assert oracle_raw.dtype == torch.float64
            oracle_g = oracle_raw * oracle_coefficient
            oracle_ideal = -lr * oracle_g / (oracle_g.abs() + eps)
            oracle_values = {"raw_gradient": oracle_raw, "clipped_gradient": oracle_g,
                "ideal_update": oracle_ideal,
                "rounded_ideal_update": (values["naive"]["before"] + oracle_ideal).float().double() - values["naive"]["before"]}
            result["fp64_oracle"]["tensor_ideal_updates_vs_oracle"][name] = {
                side: metrics(oracle_ideal, values[side]["ideal_from_gradients"], lr) for side in sides}
        for key, rows in update_rows.items():
            rows[name] = metrics(values["naive"][key], values["tiled"][key], lr)
            rows[name]["screen_pass"] = screen(rows[name], lr)
        original = report["updates"]["rows"][name]
        for field in ("max_abs_error", "rel_l2"):
            assert math.isclose(update_rows["recorded"][name][field], original[field], rel_tol=1e-12, abs_tol=1e-20)
        result["tensor_updates"][name] = {key: rows[name] for key, rows in update_rows.items()}
        result["tensor_updates"][name]["recorded_minus_ideal_per_arm"] = {
            side: metrics(values[side]["ideal_from_gradients"], values[side]["recorded"]) for side in sides}
        if name in report["updates"]["screen_failed_tensors"] or name == max_update_name:
            flat = update_rows["recorded"][name]["worst_flat_index"]
            coordinate = {"index": index_of(flat, values["naive"]["g"].shape), "flat_index": flat,
                          "shape": list(values["naive"]["g"].shape), "arms": {}}
            for side in sides:
                value = values[side]
                before32 = value["before"].reshape(-1)[flat].float()
                g = scalar(value["g"], flat)
                arm = {"raw_gradient": scalar(value["raw"], flat), "clipped_gradient": g,
                       "clipped_gradient_over_epsilon": g / eps,
                       "before_weight": float(before32), "after_weight": scalar(value["after"], flat),
                       "weight_fp32_spacing_up": float(torch.nextafter(before32, torch.full_like(before32, math.inf)).double() - before32.double()),
                       "weight_fp32_spacing_down": float(before32.double() - torch.nextafter(before32, torch.full_like(before32, -math.inf)).double()),
                       "ideal_adam_local_derivative": -lr * eps / (abs(g) + eps) ** 2}
                arm.update({key: scalar(value[key], flat) for key in update_rows})
                arm["recorded_minus_ideal"] = arm["recorded"] - arm["ideal_from_gradients"]
                arm["recorded_minus_rounded_ideal"] = arm["recorded"] - arm["rounded_ideal"]
                arm["recorded_minus_moment_formula"] = arm["recorded"] - arm["from_recorded_moments"]
                arm["moments"] = {key: {"recorded": scalar(value["state"][key], flat),
                                                 "ideal_from_gradient": scalar(value["moment_ideal"][key], flat)}
                                  for key in moment_pairs}
                coordinate["arms"][side] = arm
            a, b = coordinate["arms"]["naive"], coordinate["arms"]["tiled"]
            coordinate["tiled_minus_naive"] = {key: b[key] - a[key] for key in (
                "raw_gradient", "clipped_gradient", *update_rows.keys())}
            coordinate["recorded_error_fraction_of_lr"] = abs(b["recorded"] - a["recorded"]) / lr
            coordinate["recorded_error_relative_to_naive_coordinate_update"] = abs(b["recorded"] - a["recorded"]) / max(abs(a["recorded"]), 1e-30)
            coordinate["pair_difference_residual_after_ideal_gradient_mapping"] = (
                b["recorded"] - a["recorded"] - (b["ideal_from_gradients"] - a["ideal_from_gradients"]))
            if oracle_values is not None:
                coordinate["fp64_oracle"] = {key: scalar(value, flat) for key, value in oracle_values.items()}
                o = coordinate["fp64_oracle"]
                o["arms"] = {}
                for side, arm in coordinate["arms"].items():
                    same_clip_g = o["raw_gradient"] * report["clipping"][side]["clip_coefficient"]
                    same_clip_ideal = -lr * same_clip_g / (abs(same_clip_g) + eps)
                    o["arms"][side] = {"raw_gradient_error": arm["raw_gradient"] - o["raw_gradient"],
                        "clipped_gradient_error": arm["clipped_gradient"] - o["clipped_gradient"],
                        "ideal_update_error": arm["ideal_from_gradients"] - o["ideal_update"],
                        "recorded_update_error": arm["recorded"] - o["ideal_update"],
                        "oracle_ideal_using_this_arm_fp32_clip_coefficient": same_clip_ideal,
                        "ideal_update_error_with_common_clip_coefficient": arm["ideal_from_gradients"] - same_clip_ideal}
                o["closer_raw_gradient_arm"] = min(sides, key=lambda side: abs(o["arms"][side]["raw_gradient_error"]))
                o["closer_ideal_update_arm"] = min(sides, key=lambda side: abs(o["arms"][side]["ideal_update_error"]))
            result["flagged_update_coordinates"][name] = coordinate
        for key in moment_pairs:
            a, b = values["naive"], values["tiled"]
            pair = metrics(a["state"][key], b["state"][key])
            ideal_pair = metrics(a["moment_ideal"][key], b["moment_ideal"][key])
            moment_pairs[key][name] = {"recorded": pair, "ideal_from_gradients": ideal_pair}
            if f"{name}/{key}" not in report["optimizer_states"]["scale_failed_tensors"]:
                continue
            flat = pair["worst_flat_index"]
            entry = {"index": index_of(flat, a["g"].shape), "recorded_pair_metrics": pair,
                     "ideal_pair_metrics": ideal_pair, "arms": {}}
            for side in sides:
                v = values[side]
                entry["arms"][side] = {"raw_gradient": scalar(v["raw"], flat),
                    "clipped_gradient": scalar(v["g"], flat), "moment_recorded": scalar(v["state"][key], flat),
                    "moment_ideal": scalar(v["moment_ideal"][key], flat),
                    "moment_arithmetic_residual": scalar(v["state"][key].double() - v["moment_ideal"][key], flat)}
            av, bv = entry["arms"]["naive"], entry["arms"]["tiled"]
            entry["ideal_pair_difference_at_recorded_worst"] = bv["moment_ideal"] - av["moment_ideal"]
            entry["recorded_pair_difference_at_recorded_worst"] = bv["moment_recorded"] - av["moment_recorded"]
            entry["coordinate_relative_moment_error"] = abs(bv["moment_recorded"] - av["moment_recorded"]) / max(abs(av["moment_recorded"]), 1e-30)
            entry["reference_coordinate_over_tensor_rms"] = abs(av["moment_recorded"]) / max(pair["ref_rms"], 1e-12)
            result["flagged_moment_coordinates"][f"{name}/{key}"] = entry
    result["update_summaries"] = {key: array_summary(rows, lr) for key, rows in update_rows.items()}
    result["moment_pairs"] = moment_pairs
    result["moment_scale_flag_counts"] = {key: {kind: sum(
        row[kind]["rel_l2"] > 2e-5 or row[kind]["maxerr_over_ref_rms"] > 2e-5
        for row in rows.values()) for kind in ("recorded", "ideal_from_gradients")}
        for key, rows in moment_pairs.items()}
    assert result["update_summaries"]["recorded"]["screen_failed_tensors"] == report["updates"]["screen_failed_tensors"]
    result["largest_recorded_update_difference_tensor"] = max_update_name
    if oracle is not None:
        result["fp64_oracle"]["ideal_update_summaries_vs_oracle"] = {
            side: array_summary({name: row[side] for name, row in result["fp64_oracle"]["tensor_ideal_updates_vs_oracle"].items()}, lr)
            for side in sides}
    return result


def render(report):
    lines = ["# First Adam step: retained numerical evidence", "",
        "This CPU-only analysis uses stored FP32 gradients and optimizer results. It runs no model, "
        "changes no acceptance thresholds, and incorporates explicitly supplied retained naive FP64 gradient references. "
        "Those are higher-precision references, not exact real-arithmetic gradients.", "",
        "At the first Adam step with zero initial moments and zero weight decay, ideal real arithmetic gives "
        "`delta = -lr * g / (abs(g) + eps)`, where `g` is the clipped gradient. Both beta bias corrections cancel. "
        "Its derivative is `-lr * eps / (abs(g) + eps)^2`; gradients near epsilon can therefore produce "
        "appreciable update differences from tiny absolute gradient differences. All ideal formulas below run in FP64 on the retained FP32 gradient values.", "",
        "Recorded deltas were computed as the **FP64 difference of the after and before FP32 weights**. "
        "This subtraction is exact for these nearby stored values; the update was already rounded when stored in the FP32 parameter. "
        "An unrounded CUDA Adam increment was not recorded. The FP64 formula using recorded moments separates "
        "moment-storage effects; rounding the gradient-based ideal result into the initial FP32 parameter estimates final storage effects.", "",
        "The unchanged practical update screen requires each tensor's relative update L2 error ≤ 0.001 **and** "
        "maximum absolute update error ≤ 0.01 × LR. The separately reported moment/gradient scale diagnostic "
        "requires relative L2 and maximum error/reference RMS both ≤ 2e-5. Global summaries below are descriptive; they do not replace tensor-level screens.", "",
        "| Case | Shape | Recorded flagged update tensors | Ideal-gradient flagged tensors | Coordinates above 1% LR / all | Recorded global relative update L2 | Max recorded update error/LR | Moment scale flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"]:
        summaries = case["update_summaries"]
        recorded = summaries["recorded"]
        lines.append(f"| {case['case']} | {case['fixture']['shape']} | {len(recorded['screen_failed_tensors'])} | "
            f"{len(summaries['ideal_from_gradients']['screen_failed_tensors'])} | "
            f"{recorded['coordinates_over_1pct_lr']} / {recorded['coordinate_count']} | {recorded['global_relative_l2']:.6g} | "
            f"{recorded['max_abs_error']/case['optimizer']['lr']:.6g} | {len(case['flagged_moment_coordinates'])} |")
    for case in report["cases"]:
        name = case["largest_recorded_update_difference_tensor"]
        coord = case["flagged_update_coordinates"][name]
        a, b = coord["arms"]["naive"], coord["arms"]["tiled"]
        row = case["tensor_updates"][name]["recorded"]
        lines += ["", f"## {case['case']}", "",
            f"The largest recorded update discrepancy is `{name}` at {coord['index']}. Its tensor relative update L2 "
            f"error is {row['rel_l2']:.9g}; the coordinate differs by {row['max_abs_error']:.12g} "
            f"({coord['recorded_error_fraction_of_lr']:.6g} × LR, "
            f"{100*coord['recorded_error_relative_to_naive_coordinate_update']:.6g}% of this coordinate's naive update).", "",
            "| Quantity | Naive | Tiled |", "|---|---:|---:|"]
        for key in ("raw_gradient", "clipped_gradient", "clipped_gradient_over_epsilon", "before_weight", "after_weight",
                    "ideal_from_gradients", "from_recorded_moments", "rounded_ideal", "recorded", "recorded_minus_ideal",
                    "recorded_minus_rounded_ideal", "weight_fp32_spacing_up"):
            lines.append(f"| {key} | {a[key]:.17g} | {b[key]:.17g} |")
        d = coord["tiled_minus_naive"]
        lines += ["", f"The ideal FP64 Adam mapping of the two stored clipped gradients differs by {d['ideal_from_gradients']:.12g}. "
            f"The recorded difference is {d['recorded']:.12g}; its signed residual after subtracting the ideal-gradient "
            f"difference is {coord['pair_difference_residual_after_ideal_gradient_mapping']:.12g}. "
            "Thus the ideal formula tests whether the local update difference persists without FP32 optimizer arithmetic or parameter storage rounding."]
        if "fp64_oracle" in coord:
            o = coord["fp64_oracle"]
            lines += ["", f"At this same coordinate the retained naive FP64 reference has raw gradient {o['raw_gradient']:.17g}, "
                f"clipped gradient {o['clipped_gradient']:.17g}, and ideal first update {o['ideal_update']:.17g}. "
                f"The {o['closer_raw_gradient_arm']} raw gradient and {o['closer_ideal_update_arm']} ideal update are closer here. "
                "Clipping for this reference uses its FP64 global parameter-gradient norm; a common-coefficient comparison is also retained.", "",
                "| Error relative to FP64 reference at this coordinate | Naive | Tiled |", "|---|---:|---:|"]
            for key in ("raw_gradient_error", "clipped_gradient_error", "ideal_update_error", "recorded_update_error",
                        "ideal_update_error_with_common_clip_coefficient"):
                lines.append(f"| {key} | {o['arms']['naive'][key]:.12g} | {o['arms']['tiled'][key]:.12g} |")
            lines += ["", "All previously flagged update tensors' worst coordinates receive the same reference comparison in the JSON. "
                "Coordinate-specific proximity does not identify one implementation as uniformly more accurate."]
            summary = case["fp64_oracle"]["ideal_update_summaries_vs_oracle"]
            lines += ["", "The following uses the same fixed diagnostic limits against the higher-precision reference; "
                "these additional comparisons do not replace the original naive-versus-tiled screen.", "",
                "| FP32 arm ideal update versus FP64-gradient ideal update | Global relative L2 | Max error/LR | Coordinates above 1% LR | Flagged tensors |",
                "|---|---:|---:|---:|---:|"]
            for side in ("naive", "tiled"):
                s = summary[side]
                lines.append(f"| {side} | {s['global_relative_l2']:.6g} | {s['max_abs_error']/case['optimizer']['lr']:.6g} | "
                             f"{s['coordinates_over_1pct_lr']} | {len(s['screen_failed_tensors'])} |")
        lines += ["",
            "Flagged moments retain the original diagnostic result:", "",
            "| Tensor/moment | Relative L2 | Max error/reference RMS | Worst-coordinate relative error | Coordinate magnitude/reference RMS |",
            "|---|---:|---:|---:|---:|"]
        for name, m in case["flagged_moment_coordinates"].items():
            r = m["recorded_pair_metrics"]
            lines.append(f"| `{name}` | {r['rel_l2']:.6g} | {r['maxerr_over_ref_rms']:.6g} | "
                         f"{m['coordinate_relative_moment_error']:.6g} | {m['reference_coordinate_over_tensor_rms']:.6g} |")
        lines += ["", "For the second moment, ideal `v = (1-beta2)*g^2`: a coordinate gradient perturbation `dg` "
            "changes it by `(1-beta2)*(2*g*dg + dg^2)`. The maximum-error/reference-RMS flag can arise at a "
            "large moment coordinate even when its own relative error and the tensor relative L2 error are small. "
            "Exact gradients, moments, ideal differences and arithmetic residuals for every flagged coordinate are retained in the JSON.", "",
            f"The second-moment scale flag count is {case['moment_scale_flag_counts']['exp_avg_sq']['recorded']} in the "
            f"recorded FP32 moments and {case['moment_scale_flag_counts']['exp_avg_sq']['ideal_from_gradients']} when ideal FP64 "
            "moments are computed from those same clipped gradients. Squaring and coordinate concentration explain "
            "the amplification without removing the diagnostic failures.", "",
            f"Source: [`report.json`](../../../{case['report']['path']}) and "
            f"[`ce-and-update-tensors.pt`](../../../{case['tensors']['path']}); SHA-256 values are in the JSON."]
    lines += ["", "The flagged first-step updates are real differences in the stored parameter changes under the "
        "predeclared screen. A small post-step weight error or a passed raw-gradient comparison does not erase them. "
        "Their first-step epsilon sensitivity is an explanation of how the discrepancies propagate, not a waiver "
        "or evidence that either backward implementation is wrong. The retained higher-precision references give "
        "additional coordinate and tensor evidence for assessing floating-point order sensitivity; this one-step analysis "
        "does not establish a material effect on a complete training trajectory.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument("--oracle-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Run in the explicit CPU project container (CDRM_DOCKER_GPUS=none).")
    torch.set_num_threads(4)
    result = {"schema": "r3-first-adam-analysis-v1", "evidence_class": "NUM", "device": "cpu",
              "arithmetic": "FP64 formulas on stored FP32 gradients; no new backward or optimizer run",
              "script": {"path": location(__file__), "sha256": digest(__file__)},
              "thresholds_unchanged": {"update_relative_l2": RELATIVE_UPDATE_LIMIT,
                  "update_max_error_fraction_of_lr": MAX_UPDATE_LR_FRACTION,
                  "moment_relative_l2_and_max_error_over_rms": 2e-5}, "cases": []}
    result["invocation"] = {"case_dirs": [location(path) for path in args.case_dir],
                            "oracle_dirs": [location(path) for path in args.oracle_dir],
                            "output_prefix": str(args.output_prefix)}
    oracles = {}
    for directory in args.oracle_dir:
        reference_hash = json.loads((directory / "report.json").read_text())["source_reference"]["report_sha256"]
        if reference_hash in oracles:
            raise ValueError("Duplicate oracle for the same reference report")
        oracles[reference_hash] = directory
    for directory in args.case_dir:
        oracle_dir = oracles.pop(digest(directory / "report.json"), None)
        result["cases"].append(analyze(directory, oracle_dir))
        print(f"Analyzed {directory}", flush=True)
    if oracles:
        raise ValueError("An oracle did not match any requested case")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.output_prefix.with_suffix(".md").write_text(render(result))
    print(args.output_prefix, flush=True)


if __name__ == "__main__":
    main()
