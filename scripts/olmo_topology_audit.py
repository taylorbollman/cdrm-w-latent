#!/usr/bin/env python3
"""Independent CPU-only audit of pinned rank-migration execution evidence.

Checks exact imported/common state, graph-preparation preservation and canonical
input identity before reading numerical artifacts. A BF16 comparison measures
gradient and actual Adam displacement differences; it does not grant precision
equivalence from finite loss or from a small full-weight-relative difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

SCHEMA = "olmo-topology-audit-v1"
EXECUTION_SCHEMA = "olmo-topology-execution-v1"
MODES = ("same_topology", "changed_fp32", "changed_bf16")
CHUNK_ELEMENTS = 1 << 20
FP32_BUDGETS = {
    "gradient_atol": 8e-6, "gradient_rtol": 8e-4, "gradient_relative_l2": 1e-4,
    "update_atol": 5e-7, "update_rtol": 1e-3, "update_relative_l2": 1e-4,
}


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_report(path, expected_sha256):
    if not _pin(expected_sha256) or file_sha256(path) != expected_sha256:
        raise ValueError("Execution report differs from supplied SHA256 pin")
    def reject(value):
        raise ValueError("Nonfinite JSON value: " + value)
    result = json.loads(Path(path).read_text(), parse_constant=reject)
    if result.get("schema") != EXECUTION_SCHEMA or result.get("status") != "completed":
        raise ValueError("Require a completed topology execution report")
    return result


def select_update(report, update=None):
    """Select an explicitly retained update without changing root completion."""
    if update is None:
        return report
    if type(update) is not int or update < 1:
        raise ValueError("Selected optimizer update must be a positive integer")
    evidence = report.get("update_evidence", {}).get(str(update))
    if not isinstance(evidence, dict):
        raise ValueError("Requested update evidence is absent")
    if any(key in evidence and evidence[key] != report[key] for key in ("schema", "status")):
        raise ValueError("Update evidence cannot change root completion/schema")
    selected = {**report, **evidence}
    selected["selected_update"] = update
    return selected


def _validate_boundary(value, world_size):
    if set(value.get("state", {})) != {"model", "optimizer", "scheduler", "counters"}:
        raise ValueError("Boundary must retain model, Adam, scheduler and counter evidence")
    if not isinstance(value.get("canonical_cursor"), dict) or not value["canonical_cursor"]:
        raise ValueError("Missing canonical logical cursor")
    if set(value.get("rank_rng", {})) != {str(i) for i in range(world_size)}:
        raise ValueError("Missing destination rank RNG evidence")
    if any(not item for item in value["rank_rng"].values()):
        raise ValueError("Empty rank RNG evidence")
    for key in ("model", "optimizer", "scheduler", "counters"):
        if not isinstance(value["state"][key], dict) or not value["state"][key]:
            raise ValueError("Empty boundary state: " + key)


def _validate_inputs(value):
    rows = value.get("rows", [])
    if not rows or len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Canonical input rows must be nonempty and uniquely keyed")
    for row in rows:
        if not isinstance(row["key"], str) or not row["key"]:
            raise ValueError("Empty row key")
        if any(not _pin(row.get(key)) for key in ("input_sha256", "masks_sha256", "jitter_sha256")):
            raise ValueError("Canonical row must authenticate tokens, masks and actual jitter")
    if set(value.get("counts", {})) != {"ce", "latent", "kl"}:
        raise ValueError("Missing global per-objective denominators")
    if any(type(v) is not int or v < 0 for v in value["counts"].values()) or value["counts"]["ce"] <= 0:
        raise ValueError("Invalid global per-objective denominators")
    if type(value.get("logical_update")) is not int or value["logical_update"] < 0:
        raise ValueError("Missing nonnegative logical update")
    rates = value.get("lr_used")
    if not isinstance(rates, list) or not rates or any(type(v) not in (int, float)
            or not math.isfinite(v) or v < 0 for v in rates):
        raise ValueError("Missing finite learning rates")


def structural_checks(reference, actual, mode):
    """Pure JSON checks; no reliance on a producer's summary 'passed' field."""
    if mode not in MODES:
        raise ValueError("Unknown comparison mode")
    for report in (reference, actual):
        if report.get("schema") != EXECUTION_SCHEMA or report.get("status") != "completed":
            raise ValueError("Require completed versioned topology execution evidence")
        world = report.get("world_size")
        if type(world) is not int or world < 1:
            raise ValueError("Invalid world size")
        if type(report.get("physical_batch_per_rank")) is not int or report["physical_batch_per_rank"] < 1:
            raise ValueError("Invalid physical batch")
        if not _pin(report.get("common_origin_manifest_sha256")):
            raise ValueError("Missing common-origin authority")
        for name in ("imported", "prepared", "final"):
            _validate_boundary(report["boundaries"][name], world)
        _validate_inputs(report["next_update_input"])
        names = report.get("active_parameter_names", [])
        if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("Missing unique active-parameter ownership")
    checks = {}
    for label, report in (("reference", reference), ("candidate", actual)):
        initial = report["boundaries"]["imported"]["state"]["counters"]
        final = report["boundaries"]["final"]["state"]["counters"]
        input_record = report["next_update_input"]
        world, batch = report["world_size"], report["physical_batch_per_rank"]
        physical_calls = world * math.ceil(len(input_record["rows"]) / (world * batch))
        checks[label + "_optimizer_clock_advanced_once"] = (
            input_record["logical_update"] == initial["optimizer_updates"]
            and final["optimizer_updates"] == initial["optimizer_updates"] + 1)
        checks[label + "_physical_counter_increment"] = final["microbatches"] == initial["microbatches"] + physical_calls
        if "input_tokens" in input_record:
            checks[label + "_input_counter_increment"] = final["input_tokens"] == initial["input_tokens"] + input_record["input_tokens"]
    for key in ("common_origin_manifest_sha256", "scale", "precision", "active_parameter_names", "next_update_input"):
        checks[key] = reference.get(key) == actual.get(key)
    for key in ("objective_weights", "sources", "runtime_contract"):
        if key in reference or key in actual:
            checks[key] = key in reference and key in actual and reference[key] == actual[key]
    checks["reference_preparation_preserved"] = reference["boundaries"]["imported"] == reference["boundaries"]["prepared"]
    checks["candidate_preparation_preserved"] = actual["boundaries"]["imported"] == actual["boundaries"]["prepared"]
    ref, act = reference["boundaries"]["imported"], actual["boundaries"]["imported"]
    checks["initial_common_state_exact"] = ref["state"] == act["state"]
    checks["initial_canonical_cursor_exact"] = ref["canonical_cursor"] == act["canonical_cursor"]
    rf, af = reference["boundaries"]["final"], actual["boundaries"]["final"]
    checks["final_scheduler_exact"] = rf["state"]["scheduler"] == af["state"]["scheduler"]
    checks["final_canonical_cursor_exact"] = rf["canonical_cursor"] == af["canonical_cursor"]
    logical = lambda boundary: {k: v for k, v in boundary["state"]["counters"].items() if k != "microbatches"}
    checks["final_logical_counters_exact"] = logical(rf) == logical(af)
    if mode == "same_topology":
        checks["same_allocation"] = (reference["world_size"], reference["physical_batch_per_rank"]) == (
            actual["world_size"], actual["physical_batch_per_rank"])
        checks["initial_rank_rng_exact"] = ref["rank_rng"] == act["rank_rng"]
        checks["final_boundary_bitwise_exact"] = rf == af
    else:
        checks["changed_rank_count"] = reference["world_size"] != actual["world_size"]
        checks["same_physical_batch"] = reference["physical_batch_per_rank"] == actual["physical_batch_per_rank"]
        checks["declared_precision"] = actual["precision"] == ("fp32" if mode == "changed_fp32" else "bf16_mixed")
    return {"passed": all(checks.values()), "checks": checks}


def _component(name):
    return "predictor" if name.startswith("predictor.") else "fusion" if name.startswith("backbone.fusion.") else "backbone"


def _stats():
    return {"reference_squared": 0.0, "actual_squared": 0.0, "error_squared": 0.0,
            "dot": 0.0, "maximum_absolute_error": 0.0, "elements": 0,
            "finite": True, "elementwise_close": True, "exact": True}


def _finish(value):
    ref, actual, error = (math.sqrt(value[k]) for k in ("reference_squared", "actual_squared", "error_squared"))
    return {"reference_norm": ref, "actual_norm": actual, "error_norm": error,
            "relative_l2": error / ref if ref else (0.0 if error == 0 else None),
            "cosine": value["dot"] / (ref * actual) if ref and actual else None,
            "maximum_absolute_error": value["maximum_absolute_error"], "elements": value["elements"],
            "finite": value["finite"], "elementwise_close": value["elementwise_close"], "exact": value["exact"],
            "zero_reference_with_nonzero_error": ref == 0 and error != 0}


@torch.no_grad()
def compare_tensors(reference, actual, names, *, initial=None, atol=0.0, rtol=0.0, chunk_elements=CHUNK_ELEMENTS):
    """Chunked CPU norms; optional subtraction measures actual saved Adam updates.

    Only unique active parameter names enter aggregate norms, never repeated
    state_dict aliases for tied embeddings. Displacements subtract saved master
    parameters in FP64, rather than normalize by total pretrained weight size.
    """
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("Chunk size must be positive")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate parameter ownership")
    total, components, rows = _stats(), {}, {}
    for name in names:
        if name not in reference or name not in actual or (initial is not None and name not in initial):
            raise ValueError("Missing numerical tensor: " + name)
        left, right = reference[name], actual[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ValueError("Numerical artifacts must contain tensors")
        if left.device.type != "cpu" or right.device.type != "cpu":
            raise ValueError("Independent topology audit is CPU-only")
        if left.shape != right.shape or left.dtype != right.dtype or not left.is_floating_point():
            raise ValueError("Numerical tensor shape/dtype differs: " + name)
        origin = None if initial is None else initial[name]
        if origin is not None and (origin.device.type != "cpu" or origin.shape != left.shape or origin.dtype != left.dtype):
            raise ValueError("Common initial tensor differs: " + name)
        row = _stats()
        component = components.setdefault(_component(name), _stats())
        for begin in range(0, left.numel(), chunk_elements):
            a = left.reshape(-1)[begin:begin + chunk_elements].to(torch.float64)
            b = right.reshape(-1)[begin:begin + chunk_elements].to(torch.float64)
            if origin is not None:
                start = origin.reshape(-1)[begin:begin + chunk_elements].to(torch.float64)
                a, b = a - start, b - start
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            if not finite:
                raise ValueError("Nonfinite numerical artifact: " + name)
            error = b - a
            part = {"reference_squared": float(a.square().sum()), "actual_squared": float(b.square().sum()),
                "error_squared": float(error.square().sum()), "dot": float((a * b).sum()),
                "maximum_absolute_error": float(error.abs().max()) if error.numel() else 0.0,
                "elements": a.numel(), "finite": finite,
                "elementwise_close": bool(torch.all(error.abs() <= atol + rtol * a.abs())),
                "exact": bool(torch.equal(a, b))}
            for accumulator in (row, total, component):
                for key in ("reference_squared", "actual_squared", "error_squared", "dot", "elements"):
                    accumulator[key] += part[key]
                accumulator["maximum_absolute_error"] = max(accumulator["maximum_absolute_error"], part["maximum_absolute_error"])
                for key in ("finite", "elementwise_close", "exact"):
                    accumulator[key] &= part[key]
        rows[name] = _finish(row)
    return {"aggregate": _finish(total), "components": {n: _finish(v) for n, v in components.items()},
            "parameters": rows, "atol": atol, "rtol": rtol,
            "normalization": "actual Adam displacement from common initial parameters" if initial is not None else "raw gradient"}


def load_tensor_artifact(record, directory, cache):
    path = Path(record["path"])
    path = path if path.is_absolute() else Path(directory) / path
    path = path.resolve()
    pin = record.get("sha256")
    if not _pin(pin):
        raise ValueError("Missing tensor artifact SHA256 pin")
    identity = (str(path), pin)
    if identity not in cache:
        if file_sha256(path) != pin:
            raise ValueError("Tensor artifact SHA256 differs: " + str(path))
        cache[identity] = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    value = cache[identity]
    key = record.get("key")
    if key is not None:
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError("Tensor artifact must select a named mapping")
    return value


def audit_reports(reference, actual, *, reference_dir, actual_dir, mode):
    structural = structural_checks(reference, actual, mode)
    result = {"schema": SCHEMA, "mode": mode, "structural": structural,
              "scope": "Rank migration/restart correctness; no learning-quality conclusion"}
    if not structural["passed"]:
        result.update(status="failed", numerical_acceptance="not_run", acceptance_passed=False)
        return result
    cache = {}
    def tensor(report, directory, name):
        return load_tensor_artifact(report["numeric_artifacts"][name], directory, cache)
    names = reference["active_parameter_names"]
    initial = tensor(reference, reference_dir, "initial")
    candidate_initial = tensor(actual, actual_dir, "initial")
    initial_check = compare_tensors(initial, candidate_initial, names)
    if not initial_check["aggregate"]["exact"]:
        raise ValueError("Actual initial tensor artifacts disagree despite the claimed boundary digests")
    gradients_ref = tensor(reference, reference_dir, "gradients")
    gradients_act = tensor(actual, actual_dir, "gradients")
    if set(gradients_ref) != set(names) or set(gradients_act) != set(names):
        raise ValueError("Raw-gradient ownership differs from unique active parameters")
    gradient = compare_tensors(gradients_ref, gradients_act, names,
        atol=FP32_BUDGETS["gradient_atol"] if mode == "changed_fp32" else 0.,
        rtol=FP32_BUDGETS["gradient_rtol"] if mode == "changed_fp32" else 0.)
    update = compare_tensors(tensor(reference, reference_dir, "final"), tensor(actual, actual_dir, "final"), names,
        initial=initial, atol=FP32_BUDGETS["update_atol"] if mode == "changed_fp32" else 0.,
        rtol=FP32_BUDGETS["update_rtol"] if mode == "changed_fp32" else 0.)
    result.update(initial_tensor_artifacts_exact=True, raw_gradients=gradient, adam_displacement=update,
                  finite=gradient["aggregate"]["finite"] and update["aggregate"]["finite"])
    if mode == "same_topology":
        passed = gradient["aggregate"]["exact"] and update["aggregate"]["exact"]
        result.update(acceptance_passed=passed, numerical_acceptance="bitwise_restart", status="passed" if passed else "failed")
    elif mode == "changed_fp32":
        def close(record, budget):
            aggregate = record["aggregate"]
            return aggregate["elementwise_close"] and aggregate["relative_l2"] is not None and aggregate["relative_l2"] <= budget
        passed = close(gradient, FP32_BUDGETS["gradient_relative_l2"]) and close(update, FP32_BUDGETS["update_relative_l2"])
        result.update(acceptance_passed=passed, numerical_acceptance="bounded_fp32", fp32_budgets=FP32_BUDGETS,
                      status="passed" if passed else "failed")
    else:
        result.update(status="completed", acceptance_passed=None, numerical_acceptance="measured_only",
            qualification="Exact initial state/input/preparation and finite same-precision BF16 comparison; numerical compatibility remains an interpretation, not an automatic tolerance pass")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--update", type=int, help="Compare this retained optimizer update from both reports")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Do not overwrite an existing independent audit")
    torch.set_num_threads(1)
    reference = select_update(load_report(args.reference, args.reference_sha256), args.update)
    actual = select_update(load_report(args.candidate, args.candidate_sha256), args.update)
    result = audit_reports(reference, actual, reference_dir=args.reference.parent, actual_dir=args.candidate.parent, mode=args.mode)
    result["evidence"] = {"reference": {"path": str(args.reference), "sha256": args.reference_sha256},
                          "candidate": {"path": str(args.candidate), "sha256": args.candidate_sha256},
                          "auditor_sha256": file_sha256(__file__)}
    result["selected_update"] = args.update
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    if result["status"] == "failed":
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
