#!/usr/bin/env python3
"""Frozen backward-tile and raw-cotangent block checks for opt-in F3c RT.

CPU imports/fixture tests are permitted; only root launches this GPU probe.
Primary block arms share F3b fused forward and cast reuse. The sole primary
change is the historical backward-tile backend. Original eager and FP32 arms
remain diagnostics rather than being conflated with that isolated comparison.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_tiled import _Invocation, _mm, tiled_recurrent_layer
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_validation import require_container_gpu
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_f2_graph_probe import tensor_comparison
from scripts.olmo_f3b_tile_probe import (
    SOURCE_FILES as F3B_SOURCES, WIDTHS, _strided, block_fixture,
    block_comparison as f3b_block_comparison, comparison,
    count_fused_calls as count_forward_calls, timing,
)

PROTOCOL = "docs/reports/olmo1b-f3c/protocol.md"
SOURCE_FILES = tuple(sorted(set(F3B_SOURCES) | {
    "scripts/olmo_f3c_tile_probe.py", "cdrm/pretrained/olmo_rt_backward_kernels.py"}))
LIMITS = {"global_gradient_relative_l2": 1/64, "tensor_relative_l2": 1/32,
          "max_error_reference_max": 1/16,
          "strict_diagnostic_relative_l2": 1e-5,
          "strict_diagnostic_max_abs": "1e-6 + 1e-5 * reference_max"}


def backward_module():
    return importlib.import_module("cdrm.pretrained.olmo_rt_backward_kernels")


@contextmanager
def count_backward_calls():
    module = backward_module()
    original = module.backward_tile
    calls = {"count": 0}
    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)
    with patch.object(module, "backward_tile", counted):
        yield calls


def eager_backward_tile(operands):
    """The existing three-_mm historical dK/dV arithmetic, unmodified."""
    probabilities, ga, values, query, dot = operands
    dim = query.shape[-1]
    config = replace(OLMoConfig.tiny(), model_dim=2*dim, num_heads=2)
    spec = _Invocation(config, 1., "mixed", False, torch.bfloat16)
    dvalue = _mm(probabilities.transpose(-1, -2), ga, spec, torch.bfloat16)
    error = probabilities * (_mm(ga, values.transpose(-1, -2), spec, torch.bfloat16) - dot.unsqueeze(-1))
    dkey = _mm(error.transpose(-1, -2), query, spec, torch.bfloat16) / math.sqrt(dim)
    return dkey, dvalue


def fp64_boundary_oracle(operands):
    """CPU FP64 matrix reductions with each native BF16/FP32 boundary explicit.

    This is not unrounded FP64 attention. The three complete matrix outputs
    round to BF16; the probability/error plumbing and final scaling round to
    FP32. Different accumulation orders can cross BF16 rounding ties, so the
    independent oracle is reported with its eager control, not claimed bitwise.
    """
    p, ga, v, q, dot = (x.detach().cpu() for x in operands)
    ga_rounded = ga.bfloat16().double()
    dvalue = (p.bfloat16().double().transpose(-1, -2) @ ga_rounded).bfloat16().float()
    dp = (ga_rounded @ v.double().transpose(-1, -2)).bfloat16().double()
    centered = (dp-dot.double().unsqueeze(-1)).float().double()
    error = (p.double()*centered).float().bfloat16().double()
    dkey = (error.transpose(-1, -2) @ q.double()).bfloat16().double()
    return (dkey/math.sqrt(q.shape[-1])).float(), dvalue


def gradient_comparison(actual, expected):
    rows = {name: comparison(a, b, relative_limit=1/32, maximum_limit=1/16)
            for name, a, b in zip(("dkey", "dvalue"), actual, expected)}
    for row in rows.values():
        row["zero_reference_requires_exact_zero"] = True
        if row["reference_max_abs"] == 0 and row["max_abs"] != 0:
            row["passed"] = False
    error = sum(row["error_l2"]**2 for row in rows.values())
    reference = sum(row["reference_l2"]**2 for row in rows.values())
    global_relative = math.sqrt(error)/max(math.sqrt(reference), 1e-30)
    strict = {name: tensor_comparison(a.detach().cpu(), b.detach().cpu())
              for name, a, b in zip(("dkey", "dvalue"), actual, expected)}
    return {"gradients": rows, "global_gradient_relative_l2": global_relative,
        "global_gradient_relative_l2_limit": 1/64, "strict_diagnostics": strict,
        "all_bitwise_equal": all(row["bitwise_equal"] for row in rows.values()),
        "passed": all(row["passed"] for row in rows.values()) and global_relative <= 1/64}


def block_comparison(actual, expected):
    result = f3b_block_comparison(actual, expected)
    rows = [*result["outputs"].values(), *result["gradients"].values()]
    for row in rows:
        row["zero_reference_requires_exact_zero"] = True
        if row["reference_max_abs"] == 0 and row["max_abs"] != 0:
            row["passed"] = False
    result["passed"] = result["passed"] and all(row["passed"] for row in rows)
    return result


def backward_fixture(width, index, variant):
    generator = torch.Generator().manual_seed(2026092300+index*31+variant)
    batch, heads = 2, 2
    target = max(1, width-1) if width > 1 else 3
    dim = (16, 32, 64, 128)[index % 4]
    p = torch.rand(batch, heads, target, width, generator=generator)
    # A historical rectangle is a subset of the entire softmax history: its
    # row sum need not be one, even when every key in this tile is unmasked.
    p /= p.sum(-1, keepdim=True)+.5
    ga = torch.randn(batch, heads, target, dim, generator=generator)
    v = torch.randn(batch, heads, width, dim, generator=generator).bfloat16()
    q = torch.randn(batch, heads, target, dim, generator=generator).bfloat16()
    labels = ("nonzero", "strided_masked", "zero_probability", "zero_adjoint",
              "large_adjoint", "small_probability_large_dot")
    if variant == 1:
        p[..., ::3] = 0
        p[1, :, ::4] = 0
    elif variant == 2:
        p.zero_()
    elif variant == 3:
        ga.zero_()
    elif variant == 4:
        ga *= 32
    elif variant == 5:
        p *= 1e-5
    attended = p @ v.float() + torch.randn(ga.shape, generator=generator)*.2
    dot = (ga*attended).sum(-1)
    if variant == 5:
        dot = torch.randn(dot.shape, generator=generator)*64
    values = (p, ga, v, q, dot)
    if variant == 1:
        values = tuple(_strided(x) for x in values)
    return values, {"width": width, "target_width": target, "head_dim": dim,
        "batch_size": batch, "heads": heads, "variant": labels[variant],
        "probability_dtype": "float32", "ga_dot_dtype": "float32", "values_query_dtype": "bfloat16"}


def run_tiles(fixtures, publish, *, measure_timing):
    kernel = backward_module()
    for index, width in enumerate(WIDTHS):
        for variant in range(6):
            operands, metadata = backward_fixture(width, index, variant)
            name = f"backward_tile_w{width}_{metadata['variant']}"
            fixtures[name] = {"operands": operands, "configuration": metadata}
            gpu = tuple(_strided(x.to("cuda")) if variant == 1 else x.to("cuda") for x in operands)
            with torch.no_grad():
                eager = eager_backward_tile(gpu)
                with count_backward_calls() as calls:
                    candidate = kernel.backward_tile(*gpu)
                torch.cuda.synchronize()
            oracle = fp64_boundary_oracle(operands)
            check = {"name": name, "kind": "frozen_backward_tile", "configuration": metadata,
                "candidate_vs_eager_bf16": gradient_comparison(candidate, eager),
                "candidate_vs_fp64_boundary_oracle": gradient_comparison(candidate, oracle),
                "eager_bf16_vs_fp64_boundary_oracle": gradient_comparison(eager, oracle),
                "candidate_fused_backward_calls": calls["count"],
                "input_strides": [list(x.stride()) for x in gpu],
                "acceptance_policy": "One fused call; candidate/eager and candidate/independent-boundary-oracle global/per-tensor engineering screens. Same-arithmetic stricter flags remain diagnostic."}
            check["passed"] = calls["count"] == 1 and check["candidate_vs_eager_bf16"]["passed"] and check["candidate_vs_fp64_boundary_oracle"]["passed"]
            if measure_timing and variant == 0 and width in (32, 128, 256) and check["passed"]:
                with torch.no_grad():
                    check["timing"] = {"scope": "Uncaptured historical backward-tile call; CUDA-event intervals include host submission gaps, not isolated kernel latency. No full backward, optimizer or data copy.",
                        "eager": timing(lambda: eager_backward_tile(gpu)),
                        "triton": timing(lambda: kernel.backward_tile(*gpu))}
            publish(check)


def block_evaluate(fixture, *, backward="eager", forward="triton", fp32=False):
    config = OLMoConfig.from_dict(fixture["config"])
    layer = OLMoBlock(config, device="cuda", dtype=torch.float32)
    layer.load_state_dict(fixture["state"], strict=True)
    x = fixture["x"].to("cuda").detach().requires_grad_(True)
    dtype = torch.float32 if fp32 else torch.bfloat16
    pk = fixture["pk"].to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    pv = fixture["pv"].to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    values = {name: fixture[name].to("cuda") for name in ("valid", "query_positions", "key_positions")}
    context = nullcontext() if fp32 else torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    with context, count_forward_calls() as forward_calls, count_backward_calls() as backward_calls:
        output, memory = tiled_recurrent_layer(layer, x, alpha=fixture["alpha"],
            past=(pk, pv) if fixture["prefix"] else None,
            query_positions=values["query_positions"], key_positions=values["key_positions"],
            key_valid=values["valid"], attention_precision="fp32" if fp32 else "mixed",
            cast_weights_once=not fp32 and forward == "triton", tile_backend=forward,
            backward_tile_backend=backward)
        outputs = (output, *memory)
        cotangents = tuple(g.to(device="cuda", dtype=y.dtype) for g, y in zip(fixture["cotangents"], outputs))
        torch.autograd.backward(outputs, cotangents)
    torch.cuda.synchronize()
    result = {"outputs": {name: y.detach().cpu() for name, y in zip(("hidden", "key", "value"), outputs)},
        "gradients": {"input": x.grad.detach().cpu(),
            **{name: p.grad.detach().cpu() for name, p in layer.named_parameters()}},
        "fused_forward_calls": forward_calls["count"], "fused_backward_calls": backward_calls["count"]}
    if fixture["prefix"]:
        result["gradients"].update(prefix_key=pk.grad.detach().cpu(), prefix_value=pv.grad.detach().cpu())
    return result


def strict_block_diagnostics(actual, expected):
    return {"outputs": {n: tensor_comparison(v, expected["outputs"][n]) for n, v in actual["outputs"].items()},
            "gradients": {n: tensor_comparison(v, expected["gradients"][n]) for n, v in actual["gradients"].items()}}


def run_blocks(fixtures, publish):
    for length in (9, 17):
        for prefix in (0, 3):
            for index, alpha in enumerate((0., .37, 1.)):
                name = f"block_t{length}_p{prefix}_alpha{alpha:g}"
                fixture = block_fixture(length, prefix, alpha, 2026092250+length*13+prefix*3+index)
                fixtures[name] = fixture
                reference = block_evaluate(fixture, forward="eager", fp32=True)
                original = block_evaluate(fixture, forward="eager")
                primary = block_evaluate(fixture)
                candidate = block_evaluate(fixture, backward="triton")
                exact_forward = all(torch.equal(candidate["outputs"][n], primary["outputs"][n]) for n in candidate["outputs"])
                expected_forward, expected_backward = length-1+int(prefix > 0), length-1
                calls_ok = (candidate["fused_forward_calls"] == primary["fused_forward_calls"] == expected_forward
                    and candidate["fused_backward_calls"] == expected_backward and primary["fused_backward_calls"] == 0)
                check = {"name": name, "kind": "native_tiny_block",
                    "configuration": {"length": length, "prefix": prefix, "alpha": alpha, "model_dim": 64,
                        "num_heads": 4, "head_dim": 16, "primary_forward_backend": "triton", "cast_weights_once": True,
                        "cotangents": "Raw independent hidden/exported-K/exported-V; no normalization",
                        "candidate_backward_backend": "triton", "control_backward_backend": "eager"},
                    "candidate_vs_f3b_control": block_comparison(candidate, primary),
                    "candidate_vs_original_eager_bf16": block_comparison(candidate, original),
                    "candidate_vs_full_fp32": block_comparison(candidate, reference),
                    "f3b_control_vs_full_fp32": block_comparison(primary, reference),
                    "original_eager_bf16_vs_full_fp32": block_comparison(original, reference),
                    "candidate_vs_f3b_strict_diagnostics": strict_block_diagnostics(candidate, primary),
                    "primary_forward_and_cache_bitwise_equal": exact_forward,
                    "candidate_fused_forward_calls": candidate["fused_forward_calls"],
                    "candidate_fused_backward_calls": candidate["fused_backward_calls"],
                    "control_fused_forward_calls": primary["fused_forward_calls"],
                    "control_fused_backward_calls": primary["fused_backward_calls"],
                    "expected_fused_forward_calls": expected_forward, "expected_fused_backward_calls": expected_backward,
                    "acceptance_policy": "Only backward backend differs in primary arms; exact forward/cache, expected fused-call counts and F3b-control engineering gradient screens gate. Original-eager and full-FP32 comparisons retain diagnostic screen outcomes."}
                check["passed"] = exact_forward and calls_ok and check["candidate_vs_f3b_control"]["passed"]
                publish(check)


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "tiles", "blocks"), default="all")
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in (*SOURCE_FILES, PROTOCOL):
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    report = {"schema": "olmo-f3c-tile-probe-v1", "status": "running", "stage": "setup",
        "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "limits": LIMITS,
        "configuration": {"stage": args.stage, "timing": not args.no_timing,
            "primary_forward_backend": "triton", "primary_cast_weights_once": True,
            "control_backward_backend": "eager", "candidate_backward_backend": "triton",
            "precision": "bf16_mixed", "autocast_weight_cache": False, "tf32": False, **determinism},
        "runtime": {k: str(v) for k, v in runtime.items()},
        "source_hashes": {p: sha256_file(ROOT/p) for p in SOURCE_FILES},
        "protocol_sha256": sha256_file(ROOT/PROTOCOL),
        "limitations": ["Frozen synthetic backward tiles and tiny native blocks; not actual-checkpoint clearance",
            "No optimizer, full-model graph, quality, all-layer RT or distributed result",
            "Full FP32/original eager are separate diagnostic controls; primary isolates the backward backend",
            "Full probability/error arrays remain; this does not establish linear-memory backward",
            "Helper CUDA-event timings include host submission gaps and are not isolated kernel latency",
            "Nonfinite diagnostic numeric fields serialize as null with their failed/finite flags retained"]}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f3c-backward-tile-probe", name="olmo-f3c-"+args.output_dir.name)
    fixtures = {}
    def persist():
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", json_safe(report))
    def publish(check):
        report["checks"].append(check); persist()
        name = "candidate_vs_eager_bf16" if check["kind"] == "frozen_backward_tile" else "candidate_vs_f3b_control"
        metrics = {"probe/passed": int(check["passed"]), "probe/check": len(report["checks"]),
                   "probe/global_gradient_relative_l2": check[name]["global_gradient_relative_l2"]}
        tracker.log({k:v for k,v in metrics.items() if math.isfinite(v)}, step=len(report["checks"]))
        print({"check": check["name"], "passed": check["passed"],
               "fused_backward_calls": check["candidate_fused_backward_calls"]}, flush=True)
    try:
        tracker.start({"configuration": report["configuration"], "limits": LIMITS,
                       "source_hashes": report["source_hashes"]})
        persist(); print({"wandb": tracker.record["run_url"]}, flush=True)
        if args.stage in ("all", "tiles"):
            report["stage"] = "frozen_backward_tiles"
            run_tiles(fixtures, publish, measure_timing=not args.no_timing)
        if args.stage in ("all", "blocks"):
            report["stage"] = "tiny_blocks"
            run_blocks(fixtures, publish)
        if report["source_hashes"] != {p: sha256_file(ROOT/p) for p in SOURCE_FILES}:
            raise RuntimeError("Probe sources changed during execution")
        if report["protocol_sha256"] != sha256_file(ROOT/PROTOCOL):
            raise RuntimeError("Probe protocol changed during execution")
        report.update(status="passed" if report["checks"] and all(c["passed"] for c in report["checks"]) else "failed",
                      stage="complete", checks_passed=sum(c["passed"] for c in report["checks"]))
        if report["status"] != "passed":
            raise AssertionError("F3c primary engineering screen failed; inspect retained controls")
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
                tracker.summary({"f3c/status": report["status"], "f3c/checks": len(report["checks"]),
                                 "f3c/passed": sum(c["passed"] for c in report["checks"])})
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            persist()
    print({"status": report["status"], "report": str(args.output_dir/"report.json")}, flush=True)


if __name__ == "__main__":
    main()
