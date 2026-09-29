#!/usr/bin/env python3
"""Bounded fixed-input Flash-SDPA backward repeatability, eager and captured.

One fresh process per length/determinism setting. Non-deterministic differences
are descriptive; deterministic execution must reproduce full Q/K/V gradient
and forward hashes. No model, optimizer, DDP, training or numerical-budget claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tensor_digest
from scripts.olmo_validation import require_container_gpu

NAMES = ("query", "key", "value")
SOURCE_FILES = ("scripts/olmo_flash_backward_repeatability.py", "scripts/experiment_tracking.py",
                "scripts/olmo_f2_graph_backend_probe.py", "scripts/olmo_lm_common.py",
                "scripts/olmo_validation.py", "cdrm/pretrained/artifacts.py")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, choices=(16, 1024), required=True)
    parser.add_argument("--deterministic", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def configure_before_cuda(enabled):
    if type(enabled) is not bool:
        raise TypeError("deterministic must be boolean")
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a fresh process; determinism must be configured before CUDA initialization")
    return configure_determinism(enabled)


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def geometry(numbers):
    if not numbers["finite"]:
        return {**numbers, "relative_l2": None, "cosine": None,
                "reference_norm": None, "actual_norm": None, "difference_norm": None}
    reference = math.sqrt(numbers["reference_squared"])
    actual = math.sqrt(numbers["actual_squared"])
    difference = math.sqrt(numbers["difference_squared"])
    cosine = max(-1., min(1., numbers["dot"] / (reference * actual))) if reference and actual else None
    return {**numbers, "relative_l2": difference / max(reference, 1e-30),
            "cosine": cosine, "reference_norm": reference, "actual_norm": actual,
            "difference_norm": difference}


@torch.no_grad()
def compare_tensor(actual, reference, *, chunk_elements=1 << 20):
    """Compare in bounded FP64 chunks; retain only the baseline full CPU tensor."""
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("Comparison tensors must have matching shape and dtype")
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    # Noncontiguous head views may need one transient device flatten; the
    # retained CPU reference is contiguous and does not grow with repetitions.
    a, b = actual.detach().reshape(-1), reference.detach().reshape(-1)
    numbers = {"reference_squared": 0., "actual_squared": 0., "difference_squared": 0.,
               "dot": 0., "max_absolute_error": 0., "changed_elements": 0, "finite": True}
    for start in range(0, a.numel(), chunk_elements):
        x = a[start:start + chunk_elements].to(torch.float64)
        y = b[start:start + chunk_elements].to(device=x.device, dtype=torch.float64)
        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        numbers["finite"] &= finite
        if not finite:
            continue
        delta = x - y
        numbers["reference_squared"] += float(y.square().sum())
        numbers["actual_squared"] += float(x.square().sum())
        numbers["difference_squared"] += float(delta.square().sum())
        numbers["dot"] += float((x * y).sum())
        numbers["max_absolute_error"] = max(numbers["max_absolute_error"], float(delta.abs().max()))
        numbers["changed_elements"] += int(torch.count_nonzero(delta))
    return geometry(numbers)


def combine_geometry(rows):
    total = {key: sum(row[key] for row in rows) for key in
             ("reference_squared", "actual_squared", "difference_squared", "dot", "changed_elements")}
    total.update(finite=all(row["finite"] for row in rows),
                 max_absolute_error=max((row["max_absolute_error"] for row in rows), default=0.))
    return geometry(total)


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_before_cuda(bool(args.deterministic))
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Repeatability diagnostic must run without DDP")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "olmo-flash-backward-repeatability-v1", "status": "running", "passed": False,
        "length": args.length, "batch_size": 12, "heads": 16, "head_dimension": 128,
        "precision": "BF16 fixed Q/K/V and output cotangent; FP64 diagnostic reductions",
        "layout": "[B,T,H,D] storage transposed into [B,H,T,D] head views", "seed": 20260929, "determinism": determinism,
        "backend": "forced torch FLASH_ATTENTION SDPA, causal, dropout=0", "runtime": runtime,
        "scope": "Three eager backwards and three CUDA-graph replays at fixed inputs. No model, optimizer, DDP or training. Exactness gates only deterministic mode; nondeterministic differences are descriptive.",
        "sources": source_hashes(), "rows": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    for relative in report["sources"]:
        target = args.output_dir / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-flash-backward-repeatability", name=args.output_dir.name)
    started = time.monotonic()

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - started, wandb=tracker.record)
        write_json(args.output_dir / "report.json", report)

    failure = None
    try:
        tracker.start({key: report[key] for key in ("scope", "length", "batch_size", "heads",
                                                   "head_dimension", "precision", "determinism", "backend")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("create_fixed_inputs")
        generator = torch.Generator(device="cpu").manual_seed(report["seed"])
        storage_shape = (12, args.length, 16, 128)
        # One transient host tensor at a time; no dependence on global RNG.
        values = tuple(torch.randn(storage_shape, generator=generator).to(dtype=torch.bfloat16, device="cuda")
                       .transpose(1, 2).requires_grad_(True) for _ in NAMES)
        cotangent = torch.randn(storage_shape, generator=generator).to(dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        for value in values:
            value.grad = torch.zeros_like(value)
        fixed = dict(zip(NAMES, values)) | {"cotangent": cotangent}
        input_hashes = {name: tensor_digest(value) for name, value in fixed.items()}
        generator_hash = tensor_digest(generator.get_state())
        report.update(input_hashes=input_hashes, generator_state_hash=generator_hash,
                      input_metadata={name: {"shape": list(value.shape), "stride": list(value.stride()),
                          "dtype": str(value.dtype)} for name, value in fixed.items()},
                      reference_cpu_bytes=sum(value.numel() * value.element_size() for value in values))
        reference = None
        baseline_forward = None
        mode_hashes = {}

        def backward():
            for value in values:
                value.grad.zero_()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = F.scaled_dot_product_attention(*values, dropout_p=0., is_causal=True)
                output.backward(cotangent)
            return output

        def publish(mode, index, output):
            nonlocal reference, baseline_forward
            torch.cuda.synchronize()
            forward_hash = tensor_digest(output)
            gradient_hashes = {name: tensor_digest(value.grad) for name, value in zip(NAMES, values)}
            if reference is None:
                reference = {name: value.grad.detach().cpu().contiguous() for name, value in zip(NAMES, values)}
                baseline_forward = forward_hash
                report["reference_gradient_hashes"] = gradient_hashes
                report["reference_forward_hash"] = forward_hash
            comparison = {name: compare_tensor(value.grad, reference[name]) for name, value in zip(NAMES, values)}
            current_hashes = {"forward": forward_hash, "gradients": gradient_hashes}
            if mode not in mode_hashes:
                mode_hashes[mode] = current_hashes
            forward_exact = forward_hash == baseline_forward
            gradient_exact = gradient_hashes == report["reference_gradient_hashes"]
            inputs_unchanged = input_hashes == {name: tensor_digest(value) for name, value in fixed.items()}
            finite = bool(torch.isfinite(output).all()) and all(row["finite"] for row in comparison.values())
            row = {"mode": mode, "index": index, "forward_hash": forward_hash,
                "gradient_hashes": gradient_hashes, "forward_exact_vs_first_eager": forward_exact,
                "gradients_exact_vs_first_eager": gradient_exact,
                "exact_vs_first_same_mode": current_hashes == mode_hashes[mode],
                "per_tensor": comparison, "all_gradients": combine_geometry(list(comparison.values())),
                "inputs_unchanged": inputs_unchanged, "finite": finite,
                "passed": finite and inputs_unchanged and
                    (not args.deterministic or (forward_exact and gradient_exact))}
            report["rows"].append(row)
            persist(f"{mode}/{index}")
            tracker.log(scalar_metrics(row, "diagnostic/" + mode), step=len(report["rows"]))
            print({"stage": report["stage"], "passed": row["passed"],
                   "gradient_exact": gradient_exact, "relative_l2": row["all_gradients"]["relative_l2"]}, flush=True)
            if not finite or not inputs_unchanged:
                raise AssertionError("Finite execution or fixed-input integrity failed")
            # Keep all six observations even if deterministic exactness fails.

        for index in range(3):
            publish("eager", index, backward())
        persist("graph_warmup_capture")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                backward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured_output = backward()
        for index in range(3):
            graph.replay()
            publish("graph_replay", index, captured_output)
        report.update(sources_unchanged=source_hashes() == report["sources"],
                      generator_unchanged=tensor_digest(generator.get_state()) == generator_hash,
                      all_repetitions_bitwise_equal=all(row["forward_exact_vs_first_eager"] and
                          row["gradients_exact_vs_first_eager"] for row in report["rows"]))
        report["passed"] = (all(row["passed"] for row in report["rows"]) and
                            report["sources_unchanged"] and report["generator_unchanged"])
        if not report["passed"]:
            raise AssertionError("Deterministic repeatability or source/generator integrity failed")
        report["status"] = "passed_deterministic_repeatability" if args.deterministic else "passed_operational_diagnostic"
        persist("complete")
    except BaseException as error:
        failure = error
        report.update(status="failed", passed=False,
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist(report.get("stage", "setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed", passed=False, tracking_finish_error={"type": type(error).__name__})
            if failure is None:
                raise
        finally:
            persist(report.get("stage", "setup"))


if __name__ == "__main__":
    main()
