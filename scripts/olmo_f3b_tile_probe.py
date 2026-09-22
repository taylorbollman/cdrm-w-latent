#!/usr/bin/env python3
"""Bounded fused-RT tile and raw-cotangent block diagnostics on one GPU.

Root owns execution. Importing this module performs no CUDA or experiment work.
Every candidate call is counted at the fused entry point, so an eager fallback
cannot masquerade as a passing kernel check. No optimizer/training run is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_tiled import _Invocation, _add_tile, tiled_recurrent_layer
from cdrm.pretrained import olmo_rt_kernels
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_validation import require_container_gpu
from scripts.olmo_f2_graph_backend_probe import configure_determinism

PROTOCOL = "docs/reports/olmo1b-f3b/protocol.md"
SOURCE_FILES = (
    "scripts/olmo_f3b_tile_probe.py", "scripts/experiment_tracking.py",
    "scripts/olmo_validation.py", "scripts/olmo_f2_graph_backend_probe.py",
    "scripts/olmo_f2_graph_probe.py", "cdrm/pretrained/artifacts.py",
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_recurrent.py",
    "cdrm/pretrained/olmo_tiled.py", "cdrm/pretrained/olmo_rt_kernels.py",
    "cdrm/pretrained/recurrent.py", "cdrm/pretrained/openelm.py",
    "cdrm/pretrained/olmo_artifacts.py",
)
WIDTHS = (1, 3, 8, 17, 32, 64, 128, 256)
LIMITS = {"tile_output_relative_l2": 1/64, "tensor_relative_l2": 1/32,
          "global_gradient_relative_l2": 1/64, "max_error_reference_max": 1/16,
          "fp32_state_relative_error": 1e-5}


def comparison(actual, expected, *, relative_limit=1/32, maximum_limit=1/16):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    if a.shape != b.shape:
        return {"passed": False, "shape_matches": False}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    error = a-b
    norm, reference_norm = float(torch.linalg.vector_norm(error)), float(torch.linalg.vector_norm(b))
    maximum, reference_maximum = float(error.abs().max()), float(b.abs().max())
    relative, max_relative = norm/max(reference_norm, 1e-30), maximum/max(reference_maximum, 1e-30)
    return {"shape_matches": True, "finite": finite, "bitwise_equal": torch.equal(a, b),
        "relative_l2": relative, "max_abs": maximum, "reference_max_abs": reference_maximum,
        "max_error_reference_max": max_relative, "error_l2": norm, "reference_l2": reference_norm,
        "relative_l2_limit": relative_limit, "max_error_reference_max_limit": maximum_limit,
        "passed": finite and relative <= relative_limit and max_relative <= maximum_limit}


def state_comparison(actual, expected):
    """Maximum permits matching -inf for an empty incoming and added history."""
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    same_negative_inf = torch.isneginf(a) & torch.isneginf(b)
    valid = torch.isfinite(a) & torch.isfinite(b)
    nonfinite_matches = bool((same_negative_inf | valid).all())
    if bool(valid.any()):
        values = comparison(a[valid], b[valid], relative_limit=1e-5, maximum_limit=1e-5)
    else:
        values = {"passed": True, "relative_l2": 0., "max_error_reference_max": 0.,
                  "bitwise_equal": True, "finite": True}
    values.update(nonfinite_pattern_matches=nonfinite_matches,
                  matching_empty_entries=int(same_negative_inf.sum()))
    values["passed"] = values["passed"] and nonfinite_matches
    return values


def tile_oracle(operands):
    """Independent CPU FP64 matmuls with explicit eager conversion boundaries.

    QK and PV complete reductions round to BF16. Score scaling, exponentials,
    state multiply/add and denominator reductions each round to FP32. FP64
    computes the local reference reductions; this is not an unrounded FP64
    attention reference and does not call either implementation's helpers.
    """
    q, k, v, valid, n, m, d = (x.detach().cpu() for x in operands)
    q, k, v = q.double(), k.double(), v.double()
    qk = (q @ k.transpose(-1, -2)).bfloat16().double()
    score = (qk/math.sqrt(q.shape[-1])).float().double()
    score = score.masked_fill(~valid[:, None, None, :], -torch.inf)
    maximum = torch.maximum(m.double(), score.max(-1).values)
    safe = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    old_delta = (m.double()-safe).float().double()
    score_delta = (score-safe.unsqueeze(-1)).float().double()
    factor = old_delta.exp().float().double()
    weights = score_delta.exp().float().double()
    history = (weights.bfloat16().double() @ v).bfloat16().double()
    numerator = ((n.double()*factor.unsqueeze(-1)).float().double()+history).float()
    denominator = ((d.double()*factor).float().double()+weights.sum(-1).float().double()).float()
    return numerator, maximum.float(), denominator


def tile_comparison(actual, expected):
    n, m, d = actual
    rn, rm, rd = expected
    output = n.float()/d.float().clamp_min(1e-30).unsqueeze(-1)
    reference = rn.float()/rd.float().clamp_min(1e-30).unsqueeze(-1)
    result = {
        "numerator": comparison(n, rn, relative_limit=1/64),
        "normalized_output": comparison(output, reference, relative_limit=1/64),
        "maximum": state_comparison(m, rm),
        "denominator": state_comparison(d, rd),
    }
    result["passed"] = all(value["passed"] for value in result.values())
    return result


@contextmanager
def count_fused_calls():
    calls = {"count": 0}
    original = olmo_rt_kernels.add_tile
    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)
    with patch.object(olmo_rt_kernels, "add_tile", counted):
        yield calls


def _strided(tensor):
    # Padding the final axis preserves exact values and gives nonunit feature/
    # mask strides. Slices in ordinary dyadic execution are also noncontiguous.
    shape = (*tensor.shape[:-1], tensor.shape[-1]*2)
    storage = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    storage[..., ::2] = tensor
    return storage[..., ::2]


def tile_fixture(width, index, variant):
    generator = torch.Generator().manual_seed(2026092200+index*31+variant)
    heads, batch = 2, 2
    target = max(1, width-1) if width > 1 else 3
    dim = (16, 32, 64, 128)[index % 4]
    shape = (batch, heads, target, dim)
    q = torch.randn(shape, generator=generator).bfloat16()
    k = torch.randn(batch, heads, width, dim, generator=generator).bfloat16()
    v = torch.randn(batch, heads, width, dim, generator=generator).bfloat16()
    valid = torch.ones(batch, width, dtype=torch.bool)
    n = torch.randn(shape, generator=generator)*.7
    m = torch.randn(shape[:-1], generator=generator)
    d = torch.rand(shape[:-1], generator=generator)+.3
    label = ("nonzero", "strided_mixed_mask", "masked_empty", "masked_nonzero", "large_new_scores", "dominant_old_state")[variant]
    if variant == 1:
        valid[:, ::3] = False
        valid[1, -1] = True
    if variant in (2, 3):
        valid.zero_()
    if variant == 2:
        n.zero_(); d.zero_(); m.fill_(-torch.inf)
    if variant == 4:
        # Scores near +/- 1000 exercise stable max shifting without overflow.
        q = (q.float()*16).bfloat16(); k = (k.float()*16).bfloat16()
        m.fill_(-80)
    if variant == 5:
        m.fill_(80)
    values = (q, k, v, valid, n, m, d)
    if variant == 1:
        values = tuple(_strided(value) for value in values)
    return values, {"width": width, "target_width": target, "head_dim": dim,
                    "batch_size": batch, "heads": heads, "variant": label}


def timing(function, *, warmup=10, repeats=10):
    for _ in range(warmup):
        function()
    wall, device = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter(); start.record()
        function(); end.record(); end.synchronize()
        wall.append(time.perf_counter()-began)
        device.append(start.elapsed_time(end)/1000)
    return {"warmup_calls": warmup, "repeats": repeats,
        "median_wall_seconds": statistics.median(wall),
        "median_cuda_seconds": statistics.median(device),
        "wall_seconds": wall, "cuda_seconds": device}


def run_tiles(report, publish, *, measure_timing=True):
    fixtures = {}
    for index, width in enumerate(WIDTHS):
        for variant in range(6):
            operands, metadata = tile_fixture(width, index, variant)
            name = f"tile_w{width}_{metadata['variant']}"
            fixtures[name] = {"operands": operands, "metadata": metadata}
            # Preserve deliberately noncontiguous views after device transfer.
            gpu = tuple(_strided(x.to("cuda")) if variant == 1 else x.to("cuda") for x in operands)
            config = replace(OLMoConfig.tiny(), model_dim=metadata["head_dim"]*2, num_heads=2)
            spec = _Invocation(config, 1., "mixed", True, torch.bfloat16,
                               cast_weights_once=True, tile_backend="eager")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                eager = _add_tile(*gpu, spec, torch.bfloat16)
                with count_fused_calls() as calls:
                    candidate = _add_tile(*gpu, replace(spec, tile_backend="triton"), torch.bfloat16)
                torch.cuda.synchronize()
            oracle = tile_oracle(operands)
            check = {"name": name, "kind": "frozen_tile", "configuration": metadata,
                "candidate_vs_eager_bf16": tile_comparison(candidate, eager),
                "candidate_vs_fp64_boundary_oracle": tile_comparison(candidate, oracle),
                "eager_bf16_vs_fp64_boundary_oracle": tile_comparison(eager, oracle),
                "candidate_fused_calls": calls["count"],
                "input_strides": [list(x.stride()) for x in gpu]}
            # FP64 and FP32 reduction order can land on opposite BF16 QK
            # rounding ties. The strict state budget gates like-precision
            # candidate/eager execution; independent-oracle state discrepancies
            # remain visible with the SAME budget, including the eager control.
            oracle_check = check["candidate_vs_fp64_boundary_oracle"]
            oracle_outputs_pass = all(oracle_check[key]["passed"] for key in ("numerator", "normalized_output"))
            oracle_nonfinite_match = all(oracle_check[key]["nonfinite_pattern_matches"] for key in ("maximum", "denominator"))
            check["acceptance_policy"] = "Fused call observed; all candidate-vs-eager screens; candidate numerator/output oracle screens and finite patterns. Strict oracle state screens retained diagnostically because QK rounding ties can differ even for eager control."
            check["passed"] = calls["count"] == 1 and check["candidate_vs_eager_bf16"]["passed"] and oracle_outputs_pass and oracle_nonfinite_match
            if measure_timing and variant == 0 and width in (32, 128, 256) and check["passed"]:
                with torch.no_grad():
                    check["timing"] = {
                        "scope": "Historical tile forward only; no graph, backward, optimizer or data copy",
                        "eager": timing(lambda: _add_tile(*gpu, spec, torch.bfloat16)),
                        "triton": timing(lambda: _add_tile(*gpu, replace(spec, tile_backend="triton"), torch.bfloat16)),
                    }
            publish(check)
    return fixtures


def block_fixture(length, prefix, alpha, seed):
    config = replace(OLMoConfig.tiny(), model_dim=64, num_heads=4,
        num_layers=1, mlp_intermediate_size=128, max_context_length=128)
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = OLMoBlock(config)
        with torch.no_grad():
            for p in layer.parameters():
                p.copy_(torch.randn(p.shape, generator=generator)*.02)
    batch = 2
    x = torch.randn(batch, length, config.model_dim, generator=generator)
    cache_shape = (batch, config.num_heads, prefix, config.head_dim)
    # Cache operands originate from BF16 projections; FP32 control receives
    # precisely these rounded values promoted to FP32, not different data.
    pk = torch.randn(cache_shape, generator=generator).bfloat16()
    pv = torch.randn(cache_shape, generator=generator).bfloat16()
    valid = torch.ones(batch, prefix+length, dtype=torch.bool)
    valid[0, ::5] = False
    valid[1, 1::7] = False
    if prefix:
        valid[0, :prefix] = torch.tensor([True, False, True])
        valid[1, :prefix] = torch.tensor([False, True, True])
    key_positions = torch.arange(prefix+length).repeat(batch, 1)*3
    key_positions[1] += 2
    query_positions = key_positions[:, prefix:].clone()
    zgrad = torch.randn(x.shape, generator=generator)
    kv_shape = (batch, config.num_heads, prefix+length, config.head_dim)
    kgrad = torch.randn(kv_shape, generator=generator)
    vgrad = torch.randn(kv_shape, generator=generator)
    return {"state": {n: p.detach().clone() for n, p in layer.named_parameters()},
        "config": config.to_dict(), "x": x, "pk": pk, "pv": pv,
        "valid": valid, "query_positions": query_positions, "key_positions": key_positions,
        "cotangents": (zgrad, kgrad, vgrad), "alpha": alpha,
        "length": length, "prefix": prefix, "seed": seed}


def block_evaluate(fixture, *, candidate=False, fp32=False, cast_once=False):
    config = OLMoConfig.from_dict(fixture["config"])
    layer = OLMoBlock(config, device="cuda", dtype=torch.float32)
    layer.load_state_dict(fixture["state"], strict=True)
    x = fixture["x"].to("cuda").detach().requires_grad_(True)
    dtype = torch.float32 if fp32 else torch.bfloat16
    pk = fixture["pk"].to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    pv = fixture["pv"].to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    valid = fixture["valid"].to("cuda")
    positions = fixture["query_positions"].to("cuda")
    key_positions = fixture["key_positions"].to("cuda")
    context = nullcontext() if fp32 else torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    with context, count_fused_calls() as calls:
        output, memory = tiled_recurrent_layer(layer, x, alpha=fixture["alpha"],
            past=(pk, pv) if fixture["prefix"] else None,
            query_positions=positions, key_positions=key_positions, key_valid=valid,
            attention_precision="fp32" if fp32 else "mixed", cast_weights_once=cast_once,
            tile_backend="triton" if candidate else "eager")
        outputs = (output, *memory)
        gradients = tuple(g.to(device="cuda", dtype=y.dtype) for g, y in zip(fixture["cotangents"], outputs))
        torch.autograd.backward(outputs, gradients)
    torch.cuda.synchronize()
    result = {"outputs": {name: value.detach().cpu() for name, value in zip(("hidden", "key", "value"), outputs)},
        "gradients": {"input": x.grad.detach().cpu(),
            **{name: parameter.grad.detach().cpu() for name, parameter in layer.named_parameters()}},
        "fused_calls": calls["count"]}
    if fixture["prefix"]:
        result["gradients"].update(prefix_key=pk.grad.detach().cpu(), prefix_value=pv.grad.detach().cpu())
    return result


def block_comparison(actual, expected):
    outputs = {name: comparison(value, expected["outputs"][name], relative_limit=1/64)
               for name, value in actual["outputs"].items()}
    ownership = actual["gradients"].keys() == expected["gradients"].keys()
    gradients = {name: comparison(value, expected["gradients"][name])
                 for name, value in actual["gradients"].items()}
    squared_error = sum(row["error_l2"]**2 for row in gradients.values())
    squared_reference = sum(row["reference_l2"]**2 for row in gradients.values())
    global_relative = math.sqrt(squared_error)/max(math.sqrt(squared_reference), 1e-30)
    return {"outputs": outputs, "gradients": gradients, "gradient_ownership_matches": ownership,
        "global_gradient_relative_l2": global_relative,
        "global_gradient_relative_l2_limit": 1/64,
        "passed": ownership and global_relative <= 1/64 and
            all(row["passed"] for row in (*outputs.values(), *gradients.values()))}


def run_blocks(report, publish):
    fixtures = {}
    for length in (9, 17):
        for prefix in (0, 3):
            for index, alpha in enumerate((0., .37, 1.)):
                name = f"block_t{length}_p{prefix}_alpha{alpha:g}"
                fixture = block_fixture(length, prefix, alpha, 2026092250+length*13+prefix*3+index)
                fixtures[name] = fixture
                reference = block_evaluate(fixture, fp32=True)
                eager = block_evaluate(fixture)
                cast_only = block_evaluate(fixture, cast_once=True)
                candidate = block_evaluate(fixture, candidate=True, cast_once=True)
                check = {"name": name, "kind": "native_tiny_block",
                    "configuration": {"length": length, "prefix": prefix, "alpha": alpha,
                        "model_dim": 64, "num_heads": 4, "head_dim": 16,
                        "cotangents": "Raw independent hidden/exported-K/exported-V, no normalization",
                        "candidate_cast_weights_once": True},
                    "candidate_vs_eager_bf16": block_comparison(candidate, eager),
                    "candidate_vs_full_fp32": block_comparison(candidate, reference),
                    "eager_bf16_vs_full_fp32": block_comparison(eager, reference),
                    "cast_only_vs_eager_bf16": block_comparison(cast_only, eager),
                    "candidate_fused_calls": candidate["fused_calls"]}
                expected_fused = length-1 + int(prefix > 0)
                check["expected_fused_calls"] = expected_fused
                # The eager-vs-FP32 control remains a separate diagnostic. A
                # candidate does not inherit a pass when both miss the screen.
                check["passed"] = candidate["fused_calls"] == expected_fused and all(check[key]["passed"] for key in
                    ("candidate_vs_eager_bf16", "candidate_vs_full_fp32", "cast_only_vs_eager_bf16"))
                # block_evaluate includes construction/copies, so do not present
                # its duration as a block-kernel throughput measurement.
                publish(check)
    return fixtures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "tiles", "blocks"), default="all")
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in (*SOURCE_FILES, PROTOCOL):
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    report = {"schema": "olmo-f3b-tile-probe-v1", "status": "running", "stage": "setup",
        "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
        "runtime": {k: str(v) for k, v in runtime.items()}, "limits": LIMITS,
        "configuration": {"stage": args.stage, "timing": not args.no_timing,
            "precision": "bf16_mixed", "parameters_and_gradients": "fp32",
            "autocast_weight_cache": False, "tf32": False, **determinism},
        "source_hashes": {p: sha256_file(ROOT/p) for p in SOURCE_FILES},
        "protocol_sha256": sha256_file(ROOT/PROTOCOL),
        "limitations": ["Synthetic frozen operands and tiny native blocks; not actual-checkpoint clearance",
            "No optimizer, full-model graph, distributed or quality result",
            "Engineering error screens, not author-published numerical tolerances",
            "FP32 reference uses identical starting weights/input and rounded BF16 prefix values",
            "Fixture exported-cache cotangents are raw; prefix gradients have their input dtype",
            "Helper timings exclude block projection/MLP/backward and are not complete-step throughput"]}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f3b-tile-probe", name="olmo-f3b-"+args.output_dir.name)
    fixtures = {}
    def persist():
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", report)
    def publish(check):
        report["checks"].append(check); persist()
        payload = {"probe/passed": int(check["passed"]), "probe/check": len(report["checks"])}
        comparison_name = "candidate_vs_eager_bf16"
        if check["kind"] == "frozen_tile":
            payload["tile/relative_l2"] = check[comparison_name]["normalized_output"]["relative_l2"]
            payload["tile/max_error_reference_max"] = check[comparison_name]["normalized_output"]["max_error_reference_max"]
        else:
            payload["block/global_gradient_relative_l2_vs_bf16"] = check[comparison_name]["global_gradient_relative_l2"]
            payload["block/global_gradient_relative_l2_vs_fp32"] = check["candidate_vs_full_fp32"]["global_gradient_relative_l2"]
        tracker.log(payload, step=len(report["checks"]))
        print({"check": check["name"], "passed": check["passed"],
               "fused_calls": check["candidate_fused_calls"]}, flush=True)
    try:
        tracker.start({"configuration": report["configuration"], "limits": LIMITS,
                       "source_hashes": report["source_hashes"]})
        persist(); print({"wandb": tracker.record["run_url"]}, flush=True)
        if args.stage in ("all", "tiles"):
            report["stage"] = "frozen_tiles"
            fixtures.update(run_tiles(report, publish, measure_timing=not args.no_timing))
        if args.stage in ("all", "blocks"):
            report["stage"] = "tiny_blocks"
            fixtures.update(run_blocks(report, publish))
        if report["source_hashes"] != {p: sha256_file(ROOT/p) for p in SOURCE_FILES}:
            raise RuntimeError("Probe source changed during execution")
        if report["protocol_sha256"] != sha256_file(ROOT/PROTOCOL):
            raise RuntimeError("Frozen protocol changed during execution")
        report.update(status="passed" if report["checks"] and all(x["passed"] for x in report["checks"]) else "failed",
                      stage="complete", checks_passed=sum(x["passed"] for x in report["checks"]))
        if report["status"] != "passed":
            raise AssertionError("One or more F3b engineering screens failed; inspect retained controls")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        if fixtures:
            torch.save(fixtures, args.output_dir/"fixtures.pt")
            report["fixtures"] = {"path": "fixtures.pt", "sha256": sha256_file(args.output_dir/"fixtures.pt"),
                                  "case_count": len(fixtures)}
        try:
            if tracker.record["status"] == "running":
                tracker.summary({"f3b/status": report["status"], "f3b/checks": len(report["checks"]),
                    "f3b/passed": sum(x["passed"] for x in report["checks"])})
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            persist()
    print({"status": report["status"], "report": str(args.output_dir/"report.json")}, flush=True)


if __name__ == "__main__":
    main()
