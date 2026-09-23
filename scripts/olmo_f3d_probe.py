#!/usr/bin/env python3
"""Bounded raw-cotangent and workspace probes for recomputed RT backward.

The primary control is the validated F3c implementation. Forward tile fusion,
backward tile fusion and cast reuse are identical; only backward memory policy
changes. This is a disposable correctness/memory fixture, not a training run.
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
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.olmo import OLMoBlock, OLMoConfig
from cdrm.pretrained.olmo_tiled import (
    _Invocation, _attention_from_completed, _historical_backward_tile, _mm, tiled_recurrent_layer,
)
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_validation import require_container_gpu
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_f3b_tile_probe import block_fixture, comparison
from scripts.olmo_f3c_tile_probe import (
    SOURCE_FILES as F3C_SOURCES, block_comparison, count_backward_calls,
    count_forward_calls, fp64_boundary_oracle, gradient_comparison, json_safe, strict_block_diagnostics,
)

PROTOCOL = "docs/reports/olmo1b-f3d/protocol.md"
SOURCE_FILES = tuple(sorted(set(F3C_SOURCES) | {
    "scripts/olmo_f3d_probe.py", "cdrm/pretrained/olmo_rt_memory.py",
    "cdrm/pretrained/olmo_rt_recompute_kernels.py"}))
LIMITS = {"global_gradient_relative_l2": 1/64, "tensor_relative_l2": 1/32,
          "max_error_reference_max": 1/16, "forward_cache_bitwise_equal": True,
          "zero_reference_requires_exact_zero": True}


def memory_module():
    return importlib.import_module("cdrm.pretrained.olmo_rt_memory")


@contextmanager
def count_recomputed_calls():
    module = importlib.import_module("cdrm.pretrained.olmo_rt_recompute_kernels")
    original = module.backward_recomputed_tile
    counts = {"count": 0}
    def counted(*args, **kwargs):
        counts["count"] += 1
        return original(*args, **kwargs)
    with patch.object(module, "backward_recomputed_tile", counted):
        yield counts


def row_fixture(length, prefix, variant, *, batch=2, heads=2, dim=32):
    seed = 2026092300 + length*37 + prefix*11 + variant
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, heads, length, dim)
    query, temporary_key, temporary_value = (
        torch.randn(shape, generator=generator).bfloat16() for _ in range(3))
    memory_shape = (batch, heads, prefix + length, dim)
    permanent_key, permanent_value = (
        torch.randn(memory_shape, generator=generator).bfloat16() for _ in range(2))
    valid = torch.ones(batch, prefix + length, dtype=torch.bool)
    if variant == 1:
        valid.zero_()
    elif variant == 2:
        valid[:, ::3] = False
        valid[0, :min(prefix+2, prefix+length)] = False
    elif variant == 3:
        query *= 8
        permanent_key *= 8
        temporary_key *= 8
    values = (query, temporary_key, temporary_value, permanent_key, permanent_value, valid)
    if variant == 2:
        # Preserve arithmetic values while forcing a genuine nonunit last stride.
        strided = []
        for value in values:
            storage = torch.empty((*value.shape[:-1], value.shape[-1]*2), dtype=value.dtype)
            storage[..., ::2] = value
            strided.append(storage[..., ::2])
        values = tuple(strided)
    grad_attention = torch.randn(shape, generator=generator)
    return {"operands": values, "grad_attention": grad_attention, "length": length,
        "prefix": prefix, "seed": seed, "batch_size": batch, "heads": heads,
        "head_dim": dim, "variant": ("normal", "all_masked", "strided_masked", "large_scores")[variant]}


def row_spec(fixture, *, backend="eager"):
    config = replace(OLMoConfig.tiny(), model_dim=fixture["heads"]*fixture["head_dim"],
                     num_heads=fixture["heads"])
    return _Invocation(config, 1., "mixed", False, torch.bfloat16,
                       backward_tile_backend=backend)


def fp64_row_oracle(fixture):
    """Independent CPU reductions with native BF16/FP32 boundaries explicit."""
    q, kt, vt, k, v, valid = fixture["operands"]
    length, prefix, dim = fixture["length"], fixture["prefix"], fixture["head_dim"]
    score = (q.double() @ k.double().transpose(-1, -2)).bfloat16().float() / math.sqrt(dim)
    indices = torch.arange(length)
    self_score = (q.float()*kt.float()).sum(-1) / math.sqrt(dim)
    score[:, :, indices, prefix+indices] = self_score
    allowed = (torch.arange(prefix+length)[None, :] <= prefix+indices[:, None])[None, None] & valid[:, None, None, :]
    score.masked_fill_(~allowed, -torch.inf)
    maximum = score.max(-1).values
    safe = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weight = torch.exp(score-safe.unsqueeze(-1))
    denominator = weight.sum(-1)
    probability = weight/denominator.clamp_min(1e-30).unsqueeze(-1)
    diagonal = probability[:, :, indices, prefix+indices]
    history = probability.clone()
    history[:, :, indices, prefix+indices] = 0
    attention = (history.bfloat16().double() @ v.double()).bfloat16().float()
    attention += diagonal.unsqueeze(-1)*vt.float()
    return maximum, denominator, diagonal, attention, probability


def output_comparison(actual, expected):
    row = comparison(actual, expected, relative_limit=1/64, maximum_limit=1/16)
    row["zero_reference_requires_exact_zero"] = True
    if row.get("reference_max_abs") == 0 and row.get("max_abs") != 0:
        row["passed"] = False
    return row


def _to_gpu_preserving_stride(value):
    result = value.to("cuda")
    if value.stride()[-1] != 1:
        storage = torch.empty((*result.shape[:-1], result.shape[-1]*2), device="cuda", dtype=result.dtype)
        storage[..., ::2] = result
        result = storage[..., ::2]
    return result


def run_rows(fixtures, publish):
    module = memory_module()
    for length in (1, 9, 17, 33, 65, 129):
        for prefix in (0, 3):
            for variant in range(4):
                fixture = row_fixture(length, prefix, variant)
                name = f"rows_t{length}_p{prefix}_{fixture['variant']}"
                fixtures[name] = fixture
                gpu = tuple(_to_gpu_preserving_stride(x) for x in fixture["operands"])
                spec = row_spec(fixture, backend="triton")
                q, kt, vt, k, v, valid = gpu
                with torch.no_grad():
                    probability, attention = _attention_from_completed(*gpu, prefix, spec, torch.bfloat16)
                    maximum, denominator, diagonal, candidate = module.attention_from_completed(*gpu, prefix, spec, torch.bfloat16)
                    indices = torch.arange(length, device="cuda")
                    expected_diagonal = probability[:, :, indices, prefix+indices]
                    ga = fixture["grad_attention"].to("cuda")
                    dot = (ga*attention).sum(-1)
                    boundary = max(1, length//2)
                    source_stop = prefix + (boundary if length > 1 else 0)
                    target_start = boundary if length > 1 else 0
                    history_check = None
                    with count_recomputed_calls() as recomputed_calls:
                        if source_stop:
                            historical = module.historical_backward(
                                q[:, :, target_start:], k[:, :, :source_stop], v[:, :, :source_stop],
                                ga[:, :, target_start:], dot[:, :, target_start:], maximum[:, :, target_start:],
                                denominator[:, :, target_start:], valid[:, :source_stop], spec, torch.bfloat16)
                    if source_stop:
                        expected = _historical_backward_tile(
                            probability[:, :, target_start:, :source_stop], ga[:, :, target_start:],
                            v[:, :, :source_stop], q[:, :, target_start:], dot[:, :, target_start:], spec, torch.bfloat16)
                        history_check = gradient_comparison(historical, expected)
                    torch.cuda.synchronize()
                oracle = fp64_row_oracle(fixture)
                check = {"name": name, "kind": "frozen_reconstruction",
                    "configuration": {k: v for k, v in fixture.items() if k not in ("operands", "grad_attention")},
                    "attention_vs_f3c_control": output_comparison(candidate, attention),
                    "diagonal_vs_f3c_control": output_comparison(diagonal, expected_diagonal),
                    "attention_vs_fp64_boundary_oracle": output_comparison(candidate, oracle[3]),
                    "f3c_attention_vs_fp64_boundary_oracle": output_comparison(attention, oracle[3]),
                    "historical_gradients_vs_f3c_control": history_check,
                    "candidate_recomputed_backward_calls": recomputed_calls["count"],
                    "expected_recomputed_backward_calls": int(source_stop > 0),
                    "maximum_matches_empty_pattern": bool(torch.equal(torch.isneginf(maximum).cpu(), torch.isneginf(oracle[0]))),
                    "finite_denominator_diagonal_attention": bool(all(torch.isfinite(x).all() for x in (denominator, diagonal, candidate))),
                    "acceptance_policy": "Attention/self probability and historical-gradient engineering screens against F3c; matching empty-row patterns and finite outputs. Independent FP64 whole-product-boundary oracle is diagnostic."}
                check["passed"] = all(check[k]["passed"] for k in ("attention_vs_f3c_control", "diagonal_vs_f3c_control"))
                check["passed"] &= check["maximum_matches_empty_pattern"] and check["finite_denominator_diagonal_attention"]
                check["passed"] &= history_check is None or history_check["passed"]
                check["passed"] &= recomputed_calls["count"] == int(source_stop > 0)
                publish(check)


def long_history_fixture(rows, columns):
    seed = 2026092300 + rows*31 + columns
    generator = torch.Generator().manual_seed(seed)
    batch, heads, dim = 1, 2, 128
    q = torch.randn(batch, heads, rows, dim, generator=generator).bfloat16()
    k = torch.randn(batch, heads, columns, dim, generator=generator).bfloat16()
    v = torch.randn(batch, heads, columns, dim, generator=generator).bfloat16()
    kt = torch.randn(q.shape, generator=generator).bfloat16()
    vt = torch.randn(q.shape, generator=generator).bfloat16()
    ga = torch.randn(q.shape, generator=generator)
    valid = torch.ones(batch, columns, dtype=torch.bool)
    valid[:, ::7] = False
    return {"operands": (q, k, v, kt, vt, ga, valid), "rows": rows, "columns": columns,
        "seed": seed, "batch_size": batch, "heads": heads, "head_dim": dim}


def materialized_history_fixture(fixture, device="cpu"):
    q, k, v, kt, vt, ga, valid = (x.to(device) for x in fixture["operands"])
    spec = row_spec(fixture, backend="triton" if device == "cuda" else "eager")
    score = _mm(q, k.transpose(-1, -2), spec, torch.bfloat16)/math.sqrt(fixture["head_dim"])
    score.masked_fill_(~valid[:, None, None, :], -torch.inf)
    self_score = (q.float()*kt.float()).sum(-1)/math.sqrt(fixture["head_dim"])
    maximum = torch.maximum(score.max(-1).values, self_score)
    weight = torch.exp(score-maximum.unsqueeze(-1))
    self_weight = torch.exp(self_score-maximum)
    denominator = weight.sum(-1)+self_weight
    p = weight/denominator.unsqueeze(-1)
    attention = _mm(p, v, spec, torch.bfloat16) + (self_weight/denominator).unsqueeze(-1)*vt.float()
    dot = (ga*attention).sum(-1)
    return (q, k, v, ga, dot, maximum, denominator, valid), p, spec


def run_long_histories(fixtures, publish):
    module = memory_module()
    for rows, columns in ((257, 255), (513, 511), (1024, 1024), (2048, 3)):
        fixture = long_history_fixture(rows, columns)
        name = f"long_history_r{rows}_c{columns}"
        fixtures[name] = fixture
        with torch.no_grad():
            operands, probabilities, spec = materialized_history_fixture(fixture, "cuda")
            q, k, v, ga, dot, maximum, denominator, valid = operands
            control = _historical_backward_tile(probabilities, ga, v, q, dot, spec, torch.bfloat16)
            with count_recomputed_calls() as calls:
                candidate = module.historical_backward(*operands, spec, torch.bfloat16)
            torch.cuda.synchronize()
        oracle = fp64_boundary_oracle((probabilities, ga, v, q, dot))
        check = {"name": name, "kind": "frozen_long_history",
            "configuration": {k: v for k, v in fixture.items() if k != "operands"},
            "candidate_vs_f3c_control": gradient_comparison(candidate, control),
            "candidate_vs_fp64_boundary_oracle": gradient_comparison(candidate, oracle),
            "f3c_control_vs_fp64_boundary_oracle": gradient_comparison(control, oracle),
            "candidate_recomputed_backward_calls": calls["count"],
            "expected_recomputed_backward_calls": 1,
            "acceptance_policy": "One actual recompute-kernel call and F3c-control engineering gradient screens. The control materializes P only in this independent synthetic fixture; boundary-oracle results remain diagnostic."}
        check["passed"] = calls["count"] == 1 and check["candidate_vs_f3c_control"]["passed"]
        publish(check)


def measured_reconstruction(function, length, source_length):
    # Warmup/JIT outside the allocator and shape measurement. Caller owns input
    # tensors; outputs are deleted before recording the resident allocation.
    for _ in range(2):
        warm = function()
        del warm
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    observer = AttentionShapeObserver(length, source_length)
    with observer:
        result = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    retained = torch.cuda.memory_allocated()
    finite = all(bool(torch.isfinite(x).all()) for x in result if isinstance(x, torch.Tensor))
    del result
    return {"resident_before_bytes": baseline, "peak_allocated_bytes": peak,
        "peak_above_resident_bytes": peak-baseline, "retained_output_bytes": retained-baseline,
        "all_outputs_finite": finite, "shape_observer": observer.record()}


def run_memory(publish):
    module = memory_module()
    previous = None
    for length in (512, 1024, 2048):
        fixture = row_fixture(length, 0, 0, batch=2, heads=4, dim=64)
        gpu = tuple(x.to("cuda") for x in fixture["operands"])
        spec = row_spec(fixture, backend="triton")
        with torch.no_grad():
            # Numerical comparisons are deliberately outside both peak regions.
            probability, control_attention = _attention_from_completed(*gpu, 0, spec, torch.bfloat16)
            indices = torch.arange(length, device="cuda")
            control_diagonal = probability[:, :, indices, indices]
            _, _, candidate_diagonal, candidate_attention = module.attention_from_completed(*gpu, 0, spec, torch.bfloat16)
            attention_check = output_comparison(candidate_attention, control_attention)
            diagonal_check = output_comparison(candidate_diagonal, control_diagonal)
            del probability, control_attention, control_diagonal, candidate_diagonal, candidate_attention
            control = measured_reconstruction(
                lambda: _attention_from_completed(*gpu, 0, spec, torch.bfloat16), length, length)
            candidate = measured_reconstruction(
                lambda: module.attention_from_completed(*gpu, 0, spec, torch.bfloat16), length, length)
        check = {"name": f"reconstruction_memory_t{length}", "kind": "isolated_reconstruction_memory",
            "configuration": {"length": length, "prefix": 0, "batch_size": 2, "heads": 4, "head_dim": 64,
                "workspace_chunk": module.WORKSPACE_CHUNK},
            "control": control, "candidate": candidate,
            "attention_vs_f3c_control": attention_check,
            "diagonal_vs_f3c_control": diagonal_check,
            "peak_ratio_candidate_to_control": candidate["peak_above_resident_bytes"]/control["peak_above_resident_bytes"],
            "candidate_peak_doubling_ratio": None if previous is None else candidate["peak_above_resident_bytes"]/previous,
            "one_fp32_full_attention_bytes": 2*4*length*length*4,
            "acceptance_policy": "Observed materialized full attention, no candidate full T×S tensor, finite outputs, lower isolated candidate peak and separate attention/self-probability engineering screens. Doubling ratios are descriptive; this is reconstruction only, not the complete layer or model peak."}
        check["passed"] = (not control["shape_observer"]["no_full_attention_shape"]
            and candidate["shape_observer"]["no_full_attention_shape"]
            and candidate["all_outputs_finite"] and control["all_outputs_finite"]
            and attention_check["passed"] and diagonal_check["passed"]
            and candidate["peak_above_resident_bytes"] < control["peak_above_resident_bytes"])
        previous = candidate["peak_above_resident_bytes"]
        publish(check)


class AttentionShapeObserver(TorchDispatchMode):
    """Record dispatch outputs without retaining tensors or synchronizing CUDA.

    This observes tensor allocation/view shapes, including flattened B*H forms.
    It cannot see register/shared-memory Triton intermediates. Such intermediates
    are not persistent global allocations; source review and allocated peaks
    complement this conservative shape guard.
    """
    def __init__(self, length, source_length):
        super().__init__()
        self.length, self.source_length = length, source_length
        self.tensor_outputs = 0
        self.largest_tensor_numel = 0
        self.largest_tensor_shape = None
        self.full_attention_outputs = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if not isinstance(value, torch.Tensor):
                continue
            shape = tuple(value.shape)
            self.tensor_outputs += 1
            if value.numel() > self.largest_tensor_numel:
                self.largest_tensor_numel = value.numel()
                self.largest_tensor_shape = list(shape)
            if value.ndim >= 2 and shape[-2:] in (
                    (self.length, self.source_length),
                    (self.source_length, self.length)):
                self.full_attention_outputs.append({"operation": str(func), "shape": list(shape),
                    "dtype": str(value.dtype), "numel": value.numel()})
        return result

    def record(self):
        return {"tensor_outputs": self.tensor_outputs,
            "largest_tensor_numel": self.largest_tensor_numel,
            "largest_tensor_shape": self.largest_tensor_shape,
            "full_attention_outputs": self.full_attention_outputs,
            "no_full_attention_shape": not self.full_attention_outputs,
            "scope": "Dispatch-output shapes, including views; no tensors retained; device-private kernel scratch is outside this observer."}


def bounded_block_fixture(length, prefix, alpha, seed):
    fixture = block_fixture(length, prefix, alpha, seed)
    fixture["config"]["max_context_length"] = max(128, length + prefix)
    return fixture


def block_cases():
    cases = [(length, prefix, alpha) for length in (9, 17)
             for prefix in (0, 3) for alpha in (0., .37, 1.)]
    return cases + [(65, 3, 1.), (129, 3, 1.)]


def block_evaluate(fixture, *, memory="materialized", fp32=False, observe=False):
    config = OLMoConfig.from_dict(fixture["config"])
    layer = OLMoBlock(config, device="cuda", dtype=torch.float32)
    layer.load_state_dict(fixture["state"], strict=True)
    x = fixture["x"].to("cuda").detach().requires_grad_(True)
    dtype = torch.float32 if fp32 else torch.bfloat16
    pk = fixture["pk"].to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    pv = fixture["pv"].to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    values = {name: fixture[name].to("cuda") for name in ("valid", "query_positions", "key_positions")}
    context = nullcontext() if fp32 else torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
    observer = AttentionShapeObserver(fixture["length"], fixture["length"] + fixture["prefix"])
    with context, count_forward_calls() as forward_calls, count_backward_calls() as backward_calls, count_recomputed_calls() as recomputed_calls:
        output, cache = tiled_recurrent_layer(layer, x, alpha=fixture["alpha"],
            past=(pk, pv) if fixture["prefix"] else None,
            query_positions=values["query_positions"], key_positions=values["key_positions"],
            key_valid=values["valid"], attention_precision="fp32" if fp32 else "mixed",
            cast_weights_once=not fp32, tile_backend="eager" if fp32 else "triton",
            backward_tile_backend="eager" if fp32 else "triton", backward_memory=memory)
        outputs = (output, *cache)
        cotangents = tuple(g.to(device="cuda", dtype=y.dtype) for g, y in zip(fixture["cotangents"], outputs))
        with observer if observe else nullcontext():
            torch.autograd.backward(outputs, cotangents)
    torch.cuda.synchronize()
    result = {"outputs": {name: y.detach().cpu() for name, y in zip(("hidden", "key", "value"), outputs)},
        "gradients": {"input": x.grad.detach().cpu(),
            **{name: p.grad.detach().cpu() for name, p in layer.named_parameters()}},
        "fused_forward_calls": forward_calls["count"], "fused_backward_calls": backward_calls["count"],
        "recomputed_backward_calls": recomputed_calls["count"]}
    if fixture["prefix"]:
        result["gradients"].update(prefix_key=pk.grad.detach().cpu(), prefix_value=pv.grad.detach().cpu())
    if observe:
        result["backward_shape_observer"] = observer.record()
    return result


def run_blocks(fixtures, publish):
    for index, (length, prefix, alpha) in enumerate(block_cases()):
        name = f"block_t{length}_p{prefix}_alpha{alpha:g}"
        fixture = bounded_block_fixture(length, prefix, alpha, 2026092300+length*13+prefix*3+index)
        fixtures[name] = fixture
        primary = block_evaluate(fixture)
        candidate = block_evaluate(fixture, memory="recompute", observe=length >= 65)
        reference = block_evaluate(fixture, fp32=True)
        exact_forward = all(torch.equal(candidate["outputs"][n], primary["outputs"][n]) for n in candidate["outputs"])
        expected_forward = length - 1 + int(prefix > 0)
        calls_ok = candidate["fused_forward_calls"] == primary["fused_forward_calls"] == expected_forward
        expected_recomputed = length - 1 + int(prefix > 0)
        calls_ok &= (candidate["recomputed_backward_calls"] == expected_recomputed
                     and primary["recomputed_backward_calls"] == 0
                     and primary["fused_backward_calls"] == length - 1
                     and candidate["fused_backward_calls"] == 0)
        check = {"name": name, "kind": "native_tiny_block", "configuration": {
                "length": length, "prefix": prefix, "alpha": alpha, "model_dim": 64,
                "num_heads": 4, "head_dim": 16, "primary_forward_backend": "triton",
                "primary_backward_backend": "triton", "cast_weights_once": True,
                "control_backward_memory": "materialized", "candidate_backward_memory": "recompute",
                "cotangents": "Raw independent hidden/exported-K/exported-V; no normalization"},
            "candidate_vs_f3c_control": block_comparison(candidate, primary),
            "candidate_vs_full_fp32": block_comparison(candidate, reference),
            "f3c_control_vs_full_fp32": block_comparison(primary, reference),
            "candidate_vs_f3c_strict_diagnostics": strict_block_diagnostics(candidate, primary),
            "primary_forward_and_cache_bitwise_equal": exact_forward,
            "candidate_fused_forward_calls": candidate["fused_forward_calls"],
            "candidate_fused_backward_calls": candidate["fused_backward_calls"],
            "candidate_recomputed_backward_calls": candidate["recomputed_backward_calls"],
            "control_fused_forward_calls": primary["fused_forward_calls"],
            "control_fused_backward_calls": primary["fused_backward_calls"],
            "control_recomputed_backward_calls": primary["recomputed_backward_calls"],
            "expected_fused_forward_calls": expected_forward,
            "expected_recomputed_backward_calls": expected_recomputed,
            "acceptance_policy": "Exact primary forward/cache and expected forward dispatch; F3c-control global/per-tensor gradient engineering screens. Full FP32 retains diagnostic outcomes."}
        if "backward_shape_observer" in candidate:
            check["backward_shape_observer"] = candidate["backward_shape_observer"]
        check["passed"] = exact_forward and calls_ok and check["candidate_vs_f3c_control"]["passed"]
        if "backward_shape_observer" in check:
            check["passed"] &= check["backward_shape_observer"]["no_full_attention_shape"]
        publish(check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "rows", "blocks", "memory"), default="all")
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
    report = {"schema": "olmo-f3d-probe-v1", "status": "running", "stage": "setup",
        "started_utc": datetime.now(timezone.utc).isoformat(), "checks": [], "limits": LIMITS,
        "configuration": {"stage": args.stage, "primary_forward_backend": "triton",
            "primary_backward_backend": "triton", "primary_cast_weights_once": True,
            "control_backward_memory": "materialized", "candidate_backward_memory": "recompute",
            "precision": "bf16_mixed", "autocast_weight_cache": False, "tf32": False, **determinism},
        "runtime": {k: str(v) for k, v in runtime.items()},
        "source_hashes": {p: sha256_file(ROOT/p) for p in SOURCE_FILES},
        "protocol_sha256": sha256_file(ROOT/PROTOCOL),
        "limitations": ["Frozen synthetic reconstruction and tiny native blocks; not actual-checkpoint clearance",
            "No optimizer, full-model graph, quality, all-layer RT or distributed result",
            "Full FP32 is a separate diagnostic control; primary isolates backward memory policy",
            "Shape observers cannot inspect device-private kernel scratch; allocated peaks and source inspection complement them",
            "Nonfinite diagnostic numeric fields serialize as null with failed/finite flags retained"]}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo1b-f3d-memory-probe", name="olmo-f3d-"+args.output_dir.name)
    fixtures = {}
    def persist():
        report["wandb"] = tracker.record
        write_json(args.output_dir/"report.json", json_safe(report))
    def publish(check):
        report["checks"].append(check)
        persist()
        metrics = {"probe/passed": int(check["passed"]), "probe/check": len(report["checks"])}
        if "candidate_vs_f3c_control" in check:
            metrics["probe/global_gradient_relative_l2"] = check["candidate_vs_f3c_control"]["global_gradient_relative_l2"]
        tracker.log({k: v for k, v in metrics.items() if math.isfinite(v)}, step=len(report["checks"]))
        print({"check": check["name"], "passed": check["passed"]}, flush=True)
    try:
        tracker.start({"configuration": report["configuration"], "limits": LIMITS,
                       "source_hashes": report["source_hashes"]})
        persist()
        print({"wandb": tracker.record["run_url"]}, flush=True)
        if args.stage in ("all", "rows"):
            report["stage"] = "frozen_reconstruction"
            run_rows(fixtures, publish)
            report["stage"] = "frozen_long_histories"
            run_long_histories(fixtures, publish)
        if args.stage in ("all", "blocks"):
            report["stage"] = "tiny_blocks"
            run_blocks(fixtures, publish)
        if args.stage in ("all", "memory"):
            report["stage"] = "isolated_reconstruction_memory"
            run_memory(publish)
        if report["source_hashes"] != {p: sha256_file(ROOT/p) for p in SOURCE_FILES}:
            raise RuntimeError("Probe sources changed during execution")
        if report["protocol_sha256"] != sha256_file(ROOT/PROTOCOL):
            raise RuntimeError("Probe protocol changed during execution")
        report.update(status="passed" if report["checks"] and all(c["passed"] for c in report["checks"]) else "failed",
                      stage="complete", checks_passed=sum(c["passed"] for c in report["checks"]))
        if report["status"] != "passed":
            raise AssertionError("F3d primary engineering screen failed; inspect retained controls")
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
                tracker.summary({"f3d/status": report["status"], "f3d/checks": len(report["checks"]),
                                 "f3d/passed": sum(c["passed"] for c in report["checks"])})
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            persist()
    print({"status": report["status"], "report": str(args.output_dir/"report.json")}, flush=True)


if __name__ == "__main__":
    main()
