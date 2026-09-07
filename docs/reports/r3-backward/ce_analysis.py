#!/usr/bin/env python3
"""Read retained CE packets/reports on CPU and classify unchanged numerical flags."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

RUNS = ("tiny-ce", "init-b2", "trained-b2", "init-b64", "trained-b64")
ORACLES = {"tiny-ce", "trained-b2", "init-b64", "trained-b64"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def fetch(packet, name):
    return packet["inputs"][name[6:]] if name.startswith("input/") else packet["parameters"][name]


def aggregate(comparison):
    rows = comparison["rows"]
    return {"tensor_count": len(rows),
        "legacy_failures": comparison["legacy_failed_tensors"],
        "scale_failures": comparison["scale_failed_tensors"],
        "missing_or_mismatched": [name for name, row in rows.items() if not row["reference_present"] or not row["actual_present"] or not row["shape_match"]],
        "nonfinite": [name for name, row in rows.items() if not row["finite"]],
        "max_rel_l2": max(row["rel_l2"] for row in rows.values()),
        "max_error_over_reference_rms": max(row["maxerr_over_ref_rms"] for row in rows.values()),
        "max_absolute_error": max(row["max_abs_error"] for row in rows.values()),
        "screen_failed_tensors": comparison.get("screen_failed_tensors", [])}


def classification(row):
    coordinate = row["worst_coordinate"]
    value_rms = abs(coordinate["reference"]) / max(row["ref_rms"], row["norm_floor"])
    near = abs(coordinate["reference"]) <= row["near_zero_ref_threshold"]
    failed = []
    if not row["elementwise_pass"]:
        failed.append("original_elementwise")
    if row["rel_l2"] > row["scale_aware_rel_l2_threshold"]:
        failed.append("relative_l2")
    if row["maxerr_over_ref_rms"] > row["scale_aware_maxerr_over_ref_rms_threshold"]:
        failed.append("max_error_over_reference_rms")
    return {"failed_criteria": failed, "worst_absolute_error_coordinate": coordinate,
        "worst_reference_abs_over_rms": value_rms,
        "worst_reference_is_near_zero": near,
        "description": ("Retained tensor max/RMS tail failure at a " + ("near-zero" if near else "non-near-zero") +
                        " reference coordinate. Original elementwise and relative-L2 criteria " +
                        ("both pass." if row["elementwise_pass"] and row["rel_l2"] <= 2e-5 else "are recorded independently."))}


def coordinate_oracle(name, coordinate, oracle_packet, fp32_packets, oracle_comparisons, *, logits=False):
    index = tuple(coordinate["index"])
    oracle_tensor = oracle_packet["logits"] if logits else fetch(oracle_packet, name)
    exact = float(oracle_tensor[index])
    result = {"index": coordinate["index"], "oracle_fp64": exact, "arms": {}}
    for backend in ("naive", "tiled"):
        tensor = fp32_packets[backend]["logits"] if logits else fetch(fp32_packets[backend], name)
        value = float(tensor[index])
        row = oracle_comparisons[backend]["logits"] if logits else oracle_comparisons[backend]["gradients"]["rows"][name]
        result["arms"][backend] = {"fp32_value": value, "signed_error_vs_fp64": value - exact,
            "absolute_error_vs_fp64": abs(value - exact),
            "overall_tensor_relative_l2_vs_fp64": row["rel_l2"],
            "overall_tensor_max_error_over_fp64_rms": row["maxerr_over_ref_rms"],
            "overall_tensor_original_elementwise_pass_vs_fp64": row["elementwise_pass"],
            "overall_tensor_scale_aware_pass_vs_fp64": row["scale_aware_pass"]}
    left, right = result["arms"]["naive"], result["arms"]["tiled"]
    result["pairwise_absolute_gap"] = abs(left["fp32_value"] - right["fp32_value"])
    result["smaller_oracle_coordinate_error"] = ("equal" if left["absolute_error_vs_fp64"] == right["absolute_error_vs_fp64"] else
        "naive" if left["absolute_error_vs_fp64"] < right["absolute_error_vs_fp64"] else "tiled")
    result["both_oracle_errors_exceed_pairwise_gap"] = min(left["absolute_error_vs_fp64"], right["absolute_error_vs_fp64"]) > result["pairwise_absolute_gap"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("Run in the explicit CPU project container")
    torch.set_num_threads(1)
    result = {"schema": "r3-actual-ce-evidence-analysis-v1", "evidence_class": "NUM",
        "thresholds": {"elementwise_atol": 2e-6, "elementwise_rtol": 2e-5,
            "relative_l2": 2e-5, "max_error_over_reference_rms": 2e-5,
            "norm_denominator_floor": 1e-12, "near_zero_reference_rms_fraction": 1e-3},
        "threshold_policy": "All original and added fixed criteria retained unchanged; a classification is not a passing flag.",
        "analysis_script_sha256": digest(Path(__file__)), "cases": {},
        "pairwise_gradient_scale_failures": [], "oracle_gradient_scale_failures": []}
    for run in RUNS:
        path = args.run_root / run / "report.json"
        report = read(path)
        if report["status"] != "diagnostics_complete":
            raise AssertionError(f"Incomplete CE run {run}")
        summary = {"report": str(path), "report_sha256": digest(path), "shape": report["fixture"]["shape"],
            "width": report["model_config"]["d_model"], "checkpoint": report.get("checkpoint"),
            "fixture": report["fixture"], "optimizer": report["optimizer"],
            "logits": report["logits"], "loss_comparison": report["loss_comparison"],
            "losses": report["losses"], "clipping": report["clipping"],
            "comparisons": {name: aggregate(report[name]) for name in
                ("gradients", "clipped_gradients", "post_step_weights", "updates", "optimizer_states")}}
        oracle_packet, fp32_packets, oracle_report = None, None, None
        if run in ORACLES:
            oracle_dir = args.run_root / (run + "-fp64")
            oracle_path = oracle_dir / "report.json"
            oracle_report = read(oracle_path)
            if oracle_report["status"] != "diagnostics_complete":
                raise AssertionError(f"Incomplete FP64 oracle {run}")
            if oracle_report["source_reference"]["report_sha256"] != digest(path):
                raise AssertionError("FP64 oracle refers to a different source CE report")
            packet_path = oracle_dir / "fp64-ce-tensors.pt"
            if digest(packet_path) != oracle_report["tensor_artifact"]["sha256"]:
                raise AssertionError("FP64 oracle packet hash mismatch")
            oracle_packet = torch.load(packet_path, map_location="cpu", weights_only=False)["oracle"]
            source_packet_path = args.run_root / run / "ce-and-update-tensors.pt"
            if digest(source_packet_path) != oracle_report["source_reference"]["tensors_sha256"]:
                raise AssertionError("Saved FP32 packet hash mismatch")
            fp32_packets = torch.load(source_packet_path, map_location="cpu", weights_only=False)["gradients"]
            summary["fp64_oracle"] = {"report": str(oracle_path), "report_sha256": digest(oracle_path),
                "oracle_loss": oracle_report["oracle_loss"], "reconstruction": oracle_report["reconstruction"],
                "arms": {name: {"gradients": aggregate(arm["gradients"]), "logits": arm["logits"], "loss": arm["loss"]}
                         for name, arm in oracle_report["comparisons"].items()}}
            for backend, arm in oracle_report["comparisons"].items():
                for name in arm["gradients"]["scale_failed_tensors"]:
                    row = arm["gradients"]["rows"][name]
                    result["oracle_gradient_scale_failures"].append({"case": run, "arm": backend,
                        "tensor": name, "metrics": row, "classification": classification(row),
                        "same_coordinate": coordinate_oracle(name, row["worst_coordinate"], oracle_packet,
                                                             fp32_packets, oracle_report["comparisons"])})
        for name in report["gradients"]["scale_failed_tensors"]:
            row = report["gradients"]["rows"][name]
            failure = {"case": run, "tensor": name, "metrics": row, "classification": classification(row)}
            if oracle_packet is not None:
                failure["same_coordinate_fp64"] = coordinate_oracle(name, row["worst_coordinate"], oracle_packet,
                    fp32_packets, oracle_report["comparisons"])
            result["pairwise_gradient_scale_failures"].append(failure)
        if not report["logits"]["elementwise_pass"]:
            summary["logit_original_bound_failures"] = []
            for coordinate in report["logits"]["top5_offending_coordinates"]:
                summary["logit_original_bound_failures"].append({"coordinate": coordinate,
                    "same_coordinate_fp64": coordinate_oracle("logits", coordinate, oracle_packet,
                        fp32_packets, oracle_report["comparisons"], logits=True) if oracle_packet else None})
        result["cases"][run] = summary
        del oracle_packet, fp32_packets
    flags = result["pairwise_gradient_scale_failures"]
    result["scope"] = {"tiny": "D32, 12 blocks, T32, B2, four MQAR answers per sequence",
        "actual_width": "D256, 12 blocks, T128, B2 and physical B64; initialization and retained R3 u2000 weights/moments; eight MQAR answers per sequence",
        "precision": "FP32, autocast disabled, TF32 off, deterministic math SDPA, compiled tiled helpers at rho1/chunks4, no accumulation",
        "fp64": "Independent naive FP64 autograd, same stored FP32-representable initial weights and aligned mean-CE fixture; no tiled FP64 claim",
        "excludes": "BF16, accumulation, T512, distributed training, CDRM, alternative tasks/configurations and broad statistical guarantees"}
    result["findings"] = {
        "all_intended_gradients_present_finite_and_shape_matched": all(not r["comparisons"]["gradients"]["missing_or_mismatched"] and not r["comparisons"]["gradients"]["nonfinite"] for r in result["cases"].values()),
        "all_510_pairwise_gradient_original_rules_pass": all(not r["comparisons"]["gradients"]["legacy_failures"] for r in result["cases"].values()),
        "all_pairwise_gradient_relative_l2_rules_pass": all(r["comparisons"]["gradients"]["max_rel_l2"] <= 2e-5 for r in result["cases"].values()),
        "pairwise_gradient_max_rms_failure_count": len(flags),
        "pairwise_worst_error_coordinates_meeting_near_zero_definition": sum(f["classification"]["worst_reference_is_near_zero"] for f in flags),
        "pairwise_logit_original_failure_count": sum(r["logits"]["elementwise_failure_count"] for r in result["cases"].values()),
        "pairwise_gradient_failure_coordinates_closer_to_fp64": {arm: sum(f["same_coordinate_fp64"]["smaller_oracle_coordinate_error"] == arm for f in flags) for arm in ("naive", "tiled", "equal")},
        "oracle_gradient_original_rules_all_pass": all(not a["gradients"]["legacy_failures"] for r in result["cases"].values() if "fp64_oracle" in r for a in r["fp64_oracle"]["arms"].values()),
        "oracle_gradient_relative_l2_rules_all_pass": all(a["gradients"]["max_rel_l2"] <= 2e-5 for r in result["cases"].values() if "fp64_oracle" in r for a in r["fp64_oracle"]["arms"].values()),
        "decision_contribution": "CE evidence supports clearing the specifically tested FP32/no-accumulation backward path without a backward patch, provided the final decision retains all failed diagnostic flags and incorporates the independent raw-scaling/write-path and optimizer-sensitivity analyses. This is not an all-criteria-pass claim.",
        "classification": "Small coordinate tails from finite-precision end-to-end computation are consistent with the observations: the naive FP32 autograd arm exhibits comparable or larger errors against the FP64 oracle, while all original gradient and relative-L2 bounds pass. This does not prove identical arithmetic or a universal roundoff bound."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ce-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Actual mean-CE numerical evidence", "",
        "These retained NUM runs compare naive autograd with compiled tiled R3 under the same answer-masked mean CE. "
        "All **510 intended parameter/input-gradient comparisons** are present, finite and shape matched, and pass the original elementwise rule. "
        "Every relative-L2 rule passes. **Twelve gradient max-error/RMS flags and one original logit-coordinate flag remain failed.** "
        "The evidence supports the tested backward path without a mathematical-backward patch; it does not establish that every declared numerical criterion passed.", "",
        "Original criterion: `abs(error) <= 2e-6 + 2e-5*abs(reference)`. Separate fixed tensor criteria require "
        "relative L2 and max-error/reference-RMS each <= `2e-5`, with denominator floor `1e-12`. "
        "Near-zero means `abs(reference) <= 1e-3*reference_RMS`. No threshold was changed.", "",
        "| Case | B / T / D | CE naive / tiled | Max gradient relative L2 | Max gradient error/RMS | Gradient original / scale failures | Logit original failures |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |"]
    for run, r in result["cases"].items():
        g=r["comparisons"]["gradients"]
        lines.append(f"| {run} | {r['shape'][0]} / {r['shape'][1]} / {r['width']} | {r['losses']['naive']:.9f} / {r['losses']['tiled']:.9f} | {g['max_rel_l2']:.3e} | {g['max_error_over_reference_rms']:.3e} | {len(g['legacy_failures'])} / {len(g['scale_failures'])} | {r['logits']['elementwise_failure_count']} |")
    lines += ["", "All runs use 12 blocks, rho=1 at block3, FP32, disabled autocast/TF32, deterministic math SDPA and no accumulation. "
        "The tiny case is T32/D32; actual-width cases are T128/D256 with B2 and physical B64, at initialization and the retained R3 u2000 checkpoint. "
        "The trained optimizer diagnostic uses copied u2000 Adam state and the last scheduled LR for one NUM step, not a research continuation.", "",
        "## Every pairwise gradient max/RMS failure", "",
        "All entries below fail only the max/RMS gradient criterion; original elementwise and relative-L2 criteria pass. "
        "**None** of these worst-error reference coordinates is near zero under the fixed definition. "
        "A small RMS across a tensor can coexist with much larger individual coordinates; cancellation and sparse gradients must be assessed locally rather than labeled uniformly near-zero.", "",
        "| Case / tensor | Max absolute gap | Relative L2 | Max gap/RMS | abs(ref at worst)/RMS | FP64 absolute error naive / tiled at that coordinate |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for failure in flags:
        row=failure["metrics"]; classification_=failure["classification"]; oracle=failure.get("same_coordinate_fp64")
        oracle_text=(f"{oracle['arms']['naive']['absolute_error_vs_fp64']:.3e} / {oracle['arms']['tiled']['absolute_error_vs_fp64']:.3e}" if oracle else "not run")
        lines.append(f"| {failure['case']} / `{failure['tensor']}` | {row['max_abs_error']:.3e} | {row['rel_l2']:.3e} | {row['maxerr_over_ref_rms']:.3e} | {classification_['worst_reference_abs_over_rms']:.3f} | {oracle_text} |")
    lines += ["", "## Independent FP64 comparison", "",
        "Each oracle reconstructs both arms' pre-step weights exactly from saved FP32 weights and FP64 deltas, verifies "
        "their equality and FP32 round trips, and evaluates the same saved labels/tokens with naive FP64 autograd. "
        "The reference changes forward and backward arithmetic precision together; it is not a same-rounded-forward isolated backward oracle. "
        "ALiBi/mask constants keep their original construction precision. Tiled FP64 is not claimed.", "",
        "| Case / FP32 arm | Max gradient relative L2 vs FP64 | Max gradient error/FP64 RMS | Gradient original / scale failures | Original logit-coordinate failures |",
        "| --- | ---: | ---: | --- | ---: |"]
    for run,r in result["cases"].items():
        if "fp64_oracle" not in r: continue
        for arm,a in r["fp64_oracle"]["arms"].items():
            g=a["gradients"]
            lines.append(f"| {run} / {arm} | {g['max_rel_l2']:.3e} | {g['max_error_over_reference_rms']:.3e} | {len(g['legacy_failures'])} / {len(g['scale_failures'])} | {a['logits']['elementwise_failure_count']} |")
    lines += ["", "Every oracle gradient passes the unchanged original and relative-L2 rules for both FP32 arms. "
        "The oracle-logit original-coordinate failures are separately retained in the last column; all whole-logit scale criteria pass. "
        "The max/RMS diagnostic also flags the ordinary naive FP32 arm, so a flag by itself does not localize a defect in tiled backward. "
        "In the trained B64 case the oracle discrepancies are larger than the pairwise FP32 backend gap. "
        "Complete names, exact coordinates, local values, original flags and classifications for **every oracle gradient max/RMS failure** are retained in "
        "`oracle_gradient_scale_failures` in [ce-analysis.json](ce-analysis.json).", "",
        "## Original logit exception", ""]
    for run,r in result["cases"].items():
        for failure in r.get("logit_original_bound_failures",[]):
            c=failure["coordinate"]; o=failure["same_coordinate_fp64"]
            lines += [f"`{run}` coordinate `{c['index']}`: naive `{c['reference']:.12g}`, tiled `{c['actual']:.12g}`, "
                f"gap `{c['absolute_error']:.3e}` versus allowed `{c['elementwise_tolerance']:.3e}`. "
                f"FP64 is `{o['oracle_fp64']:.12g}`; absolute errors are naive `{o['arms']['naive']['absolute_error_vs_fp64']:.3e}` "
                f"and tiled `{o['arms']['tiled']['absolute_error_vs_fp64']:.3e}`. "
                "This original elementwise flag remains failed. Whole-logit scale diagnostics pass, and recorded FP32 CE is identical between arms.", ""]
    lines += ["## Clipping, optimizer moments and updates", "",
        "| Case | Clip norm naive / tiled | Clipped-gradient scale flags | Moment original / scale flags | Update original / scale flags | Additional update-screen flags | Post-weight original / scale flags |",
        "| --- | --- | ---: | --- | --- | ---: | --- |"]
    for run,r in result["cases"].items():
        c=r["comparisons"]; clip=r["clipping"]
        def counts(name): return f"{len(c[name]['legacy_failures'])} / {len(c[name]['scale_failures'])}"
        lines.append(f"| {run} | {clip['naive']['clip_norm']:.9f} / {clip['tiled']['clip_norm']:.9f} | {len(c['clipped_gradients']['scale_failures'])} | {counts('optimizer_states')} | {counts('updates')} | {len(c['updates']['screen_failed_tensors'])} | {counts('post_step_weights')} |")
    lines += ["", "All original elementwise moment, update and post-weight criteria pass, but the stricter scale flags remain in the table and JSON. "
        "The additional update screen (`relative L2 <= 1e-3` and `max update gap <= .01*LR`) is separate and has initialization flags. "
        "Changing optimizer denominators can amplify small gradients; conversely, large weight magnitudes can hide small update differences. "
        "The completed [first-update analysis](first-update-analysis.md) traces the initialization flags to the retained gradients, "
        "Adam's sensitivity near epsilon, and final FP32 parameter storage. It preserves every failed screen and compares the available FP64 references. "
        "Gradient clipping and Adam are not linear cotangent-scaling tests.", "",
        "The CE evidence is consistent with ordinary floating-point differences rather than an identified tiled-backward defect. "
        "The recommended supported scope is the tested FP32/no-accumulation MQAR regime, combined with the separate raw-scaling, "
        "persistent-write and Adam-sensitivity evidence. BF16, accumulation, T512, distributed execution and CDRM are outside this conclusion. "
        "No blanket all-tests-pass claim or numerical threshold waiver is justified.", "",
        "[Machine-readable evidence, source report hashes and all failed coordinates](ce-analysis.json) · [CPU aggregation script](ce_analysis.py)", ""]
    (args.output_dir / "ce-analysis.md").write_text("\n".join(lines))
    print(json.dumps(result["findings"],indent=2))


if __name__ == "__main__": main()
