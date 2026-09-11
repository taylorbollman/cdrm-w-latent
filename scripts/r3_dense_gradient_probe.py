#!/usr/bin/env python3
"""Fixed-operand BF16 dense-gradient attribution; diagnostic NUM evidence only.

Runs inside the CUDA project container. The synthetic probe holds every linear
operand fixed while changing cast sharing and reduction grouping. The isolated
R3 probe reuses the retained tiny fixture and records actual dense inputs and
output adjoints with eager helpers. It changes no model implementation. Optional
compiled runs remove all diagnostic module/internal hooks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from stage_a_common import (
    configure_compiled_helpers, provenance, require_cuda_container, seed_all,
)
from stage_b_train import compiler_audit


DEFAULT_FIXTURE = Path(".runtime/r3-bf16/20260906T225438Z/candidate-v2-write-credit/fixture-and-weights.pt")
DENSE_NAMES = ("attn_out", "ff_proj", "ff_out")


def cpu(value):
    return None if value is None else value.detach().cpu().clone()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(reference, actual):
    """Compact, unthresholded CPU FP64 comparisons, including missing values."""
    if reference is None or actual is None:
        return {"present": False, "both_missing": reference is None and actual is None}
    if reference.shape != actual.shape:
        return {"present": True, "shape_match": False,
                "reference_shape": list(reference.shape), "actual_shape": list(actual.shape)}
    left, right = reference.detach().cpu().double(), actual.detach().cpu().double()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    result = {"present": True, "shape_match": True, "finite": finite,
              "reference_dtype": str(reference.dtype), "actual_dtype": str(actual.dtype),
              "shape": list(reference.shape)}
    if not finite:
        return result
    error = right - left
    ref_norm, err_norm = float(left.norm()), float(error.norm())
    ref_rms = ref_norm / max(left.numel(), 1) ** .5
    result.update(exact=bool(torch.equal(left, right)),
                  different_elements=int(torch.count_nonzero(error)),
                  reference_l2=ref_norm, error_l2=err_norm,
                  relative_l2=err_norm / max(ref_norm, 1e-12),
                  max_absolute_error=float(error.abs().max()) if error.numel() else 0.,
                  max_error_over_reference_rms=(float(error.abs().max()) if error.numel() else 0.) / max(ref_rms, 1e-12))
    return result


@contextmanager
def cache_policy(enabled):
    previous = torch.is_autocast_cache_enabled()
    torch.clear_autocast_cache()
    torch.set_autocast_cache_enabled(enabled)
    try:
        yield
    finally:
        torch.clear_autocast_cache()
        torch.set_autocast_cache_enabled(previous)


def contraction(x, dy):
    return dy.double().reshape(-1, dy.shape[-1]).T @ x.double().reshape(-1, x.shape[-1])


def fixed_operand_reductions(x, dy):
    """Use the same BF16-representable operands in every contraction."""
    x, dy = x.cuda().bfloat16(), dy.cuda().bfloat16()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        batched_bf16 = dy.reshape(-1, dy.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        batched_fp32 = dy.float().reshape(-1, dy.shape[-1]).T @ x.float().reshape(-1, x.shape[-1])
        pieces = [dy[:, t].T @ x[:, t] for t in range(x.shape[1])]
        sum_fp32 = torch.zeros_like(batched_fp32)
        sum_bf16 = torch.zeros_like(batched_bf16)
        # Explicit order, recorded separately from autograd's scheduling.
        for piece in reversed(pieces):
            sum_fp32.add_(piece.float())
            sum_bf16.add_(piece)
    reference = contraction(cpu(x), cpu(dy))
    values = {"batched_bf16": cpu(batched_bf16), "batched_fp32": cpu(batched_fp32),
              "per_token_bf16_then_fp32_sum_reverse": cpu(sum_fp32),
              "per_token_bf16_then_bf16_sum_reverse": cpu(sum_bf16),
              "fp64_reference": reference}
    return values, {name: metrics(reference, value) for name, value in values.items()}


def synthetic_linear(seed, length):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = {"batch": 2, "length": length, "input_width": 32, "output_width": 64}
    x_saved = torch.randn(2, length, 32, generator=generator).bfloat16()
    dy_saved = torch.randn(2, length, 64, generator=generator).bfloat16()
    weight_saved = torch.randn(64, 32, generator=generator) / 32 ** .5
    reference = contraction(x_saved, dy_saved)
    packets, report = {}, {"shape": shape, "seed": seed, "arms": {}, "comparisons": {}}
    specifications = [
        ("native_per_token_cache_on", "per_token", True),
        ("native_per_token_cache_off", "per_token", False),
        ("native_batched_cache_on", "batched", True),
        ("native_batched_cache_off", "batched", False),
        ("explicit_shared_cast", "shared", False),
        ("explicit_per_token_cast", "separate", False),
    ]
    for name, mode, enabled in specifications:
        print(f"Synthetic fixed-operand linear: {name}", flush=True)
        x = x_saved.float().cuda().requires_grad_(True)
        weight = weight_saved.cuda().requires_grad_(True)
        retained_casts = []
        with cache_policy(enabled):
            if mode in ("per_token", "batched"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = (F.linear(x, weight) if mode == "batched" else
                              torch.cat([F.linear(x[:, t:t + 1], weight) for t in range(length)], dim=1))
            else:
                with torch.autocast("cuda", enabled=False):
                    low_x = x.bfloat16()
                    if mode == "shared":
                        low_weight = weight.bfloat16()
                        low_weight.retain_grad()
                        retained_casts.append(low_weight)
                        output = torch.cat([F.linear(low_x[:, t:t + 1], low_weight)
                                            for t in range(length)], dim=1)
                    else:
                        outputs = []
                        for t in range(length):
                            low_weight = weight.bfloat16()
                            low_weight.retain_grad()
                            retained_casts.append(low_weight)
                            outputs.append(F.linear(low_x[:, t:t + 1], low_weight))
                        output = torch.cat(outputs, dim=1)
            output.backward(dy_saved.cuda())
        packet = {"weight_gradient": cpu(weight.grad), "input_gradient": cpu(x.grad),
                  "output": cpu(output), "cast_gradients": [cpu(value.grad) for value in retained_casts]}
        packets[name] = packet
        report["arms"][name] = {"cache_enabled": enabled, "mode": mode,
                               "weight_gradient_vs_fp64": metrics(reference, packet["weight_gradient"]),
                               "weight_gradient_vs_fp64_rounded_once_to_bf16": metrics(reference.bfloat16().float(), packet["weight_gradient"]),
                               "cast_gradient_dtypes": sorted({str(value.dtype) for value in packet["cast_gradients"]})}
        if mode == "shared":
            report["arms"][name]["master_gradient_vs_shared_cast_gradient"] = metrics(
                packet["cast_gradients"][0].float(), packet["weight_gradient"])
    for left, right in (("native_per_token_cache_on", "explicit_shared_cast"),
                        ("native_per_token_cache_off", "explicit_per_token_cast"),
                        ("native_per_token_cache_on", "native_per_token_cache_off"),
                        ("native_batched_cache_on", "native_batched_cache_off"),
                        ("native_per_token_cache_on", "native_batched_cache_on")):
        report["comparisons"][f"{right}_vs_{left}"] = {
            key: metrics(packets[left][key], packets[right][key])
            for key in ("weight_gradient", "input_gradient", "output")}
    reductions, reduction_report = fixed_operand_reductions(x_saved, dy_saved)
    report["explicit_reductions_vs_fp64"] = reduction_report
    return {"inputs": x_saved, "output_cotangent": dy_saved, "weight": weight_saved,
            "fp64_reference": reference, "arms": packets, "explicit_reductions": reductions}, report


class DenseCapture:
    """Observe existing eager tensors; never replace a module value or gradient."""

    def __init__(self, block, length):
        self.block, self.length = block, length
        self.phase, self.position = "forward", None
        self.records, self.writes, self.hooks = [], [], []
        self.cache_states = {}

    def internal(self, phase, values, token_index=None):
        if phase in ("forward.mlp_input", "backward.mlp_input"):
            self.position = token_index
        self.cache_states.setdefault(phase, set()).add(torch.is_autocast_cache_enabled())

    def linear(self, name):
        def hook(module, args, output):
            if torch.compiler.is_compiling():
                raise AssertionError("Dense capture is restricted to eager helper execution")
            if output.ndim != 3 or output.shape[1] not in (1, self.length):
                raise AssertionError(f"Unexpected {name} output shape: {tuple(output.shape)}")
            enabled = torch.is_autocast_enabled("cuda")
            effective = args[0].to(torch.get_autocast_dtype("cuda")) if enabled else args[0]
            row = {"name": name, "phase": self.phase,
                   "kind": "batched" if output.shape[1] == self.length else "per_token",
                   "position": self.position if output.shape[1] == 1 else None,
                   "input": cpu(args[0]), "effective_input": cpu(effective), "output": cpu(output),
                   "output_gradient": None, "gradient_hook_calls": 0,
                   "autocast_enabled": enabled, "cache_enabled": torch.is_autocast_cache_enabled()}
            self.records.append(row)
            if output.requires_grad:
                def save_gradient(gradient):
                    row["output_gradient"] = cpu(gradient)
                    row["gradient_hook_calls"] += 1
                output.register_hook(save_gradient)
        return hook

    def write(self, module, args, output):
        if torch.is_grad_enabled() and output.requires_grad and output.ndim == 3 and output.shape[1] == 1:
            output.retain_grad()
            self.writes.append((self.phase, output))

    def __enter__(self):
        if hasattr(self.block, "_recurrent_precision_observer"):
            raise AssertionError("Unexpected pre-existing precision observer")
        self.block._recurrent_precision_observer = self.internal
        self.hooks = [getattr(self.block, name).register_forward_hook(self.linear(name)) for name in DENSE_NAMES]
        self.hooks.append(self.block.kv_proj.register_forward_hook(self.write))
        return self

    def __exit__(self, *error):
        for hook in self.hooks:
            hook.remove()
        del self.block._recurrent_precision_observer


def build_block(fixture, backend, bf16, eager):
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    raw = dict(fixture["configurations"]["naive_fp32"])
    raw.update(init_device="cpu", precision=None, recurrent_backend=backend,
               recurrent_precision_policy="bf16_fp32_state" if bf16 else "legacy", reference_eager=eager)
    model = OLMo(ModelConfig(**raw))
    block = model.transformer.blocks[3]
    block.load_state_dict(fixture["block_weights"]["naive_fp32"], strict=True)
    if block.config.reference_eager != eager or block.config.recurrent_precision_policy != raw["recurrent_precision_policy"]:
        raise AssertionError("The executed block did not receive its intended precision configuration")
    return block.cuda().float().train()


def isolated_arm(fixture, backend, bf16, enabled, *, eager=True, capture=True):
    block = build_block(fixture, backend, bf16, eager)
    leaf = fixture["inputs"].cuda().clone().requires_grad_(True)
    cotangent = fixture["cotangent"].cuda()
    bias = fixture["attention_bias"].cuda()
    recorder = DenseCapture(block, leaf.shape[1])
    with cache_policy(enabled), sdpa_kernel(SDPBackend.MATH):
        if capture:
            recorder.__enter__()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                output = block(leaf, attention_bias=bias)[0]
            recorder.phase, recorder.position = "backward", None
            output.backward(cotangent)
        finally:
            if capture:
                recorder.__exit__()
    gradients = {name: cpu(value.grad) for name, value in block.named_parameters()}
    if len(gradients) != 9 or any(value is None or value.dtype != torch.float32 or not torch.isfinite(value).all()
                                  for value in gradients.values()):
        raise AssertionError("Expected all nine intended parameter gradients finite in FP32")
    if leaf.grad is None or leaf.grad.dtype != torch.float32 or not torch.isfinite(leaf.grad).all():
        raise AssertionError("Expected a finite FP32 input gradient")
    if output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise AssertionError("Expected a finite FP32 residual output")
    writes = [{"position": position, "phase": phase, "value": cpu(value), "gradient": cpu(value.grad)}
              for position, (phase, value) in enumerate(recorder.writes)]
    if capture:
        if len(writes) != leaf.shape[1]:
            raise AssertionError("Persistent write coverage differs from the sequence length")
        expected_phase = "forward" if backend == "naive" else "backward"
        if any(row["phase"] != expected_phase for row in writes):
            raise AssertionError("Persistent write capture phase differs from the expected backend")
        if fixture.get("require_all_earlier_writes", True) and any(
                row["gradient"] is None or not torch.isfinite(row["gradient"]).all()
                or float(row["gradient"].double().norm()) <= 1e-12 for row in writes[:-1]):
            raise AssertionError("An earlier permanent write lacks finite nonzero temporal credit")
    return {"backend": backend, "bf16": bf16, "cache_enabled": enabled, "eager": eager,
            "instrumented": capture, "executed_config": asdict(block.config),
            "output": cpu(output), "input_gradient": cpu(leaf.grad), "parameter_gradients": gradients,
            "writes": writes, "dense_records": recorder.records,
            "cache_states": {key: sorted(value) for key, value in recorder.cache_states.items()}}


def dense_operands(packet, name, kind):
    rows = [row for row in packet["dense_records"] if row["name"] == name and row["kind"] == kind
            and row["output_gradient"] is not None]
    if not rows:
        return None
    if any(row["gradient_hook_calls"] != 1 for row in rows):
        raise AssertionError("A captured dense output was differentiated more than once")
    if kind == "per_token":
        rows.sort(key=lambda row: row["position"])
        length = packet["output"].shape[1]
        if [row["position"] for row in rows] != list(range(length)):
            raise AssertionError(f"{name}: dense operands do not cover each token once")
    elif len(rows) != 1:
        raise AssertionError("Expected exactly one final batched MLP parameter-gradient replay")
    return {key: torch.cat([row[key] for row in rows], dim=1)
            for key in ("effective_input", "output_gradient", "output")}


def analyze_isolated(packets):
    report = {"arms": {}, "comparisons": {}, "dense_operand_comparisons": {}}
    for arm, packet in packets.items():
        row = {"backend": packet["backend"], "bf16": packet["bf16"], "cache_enabled": packet["cache_enabled"],
               "eager": packet["eager"], "instrumented": packet["instrumented"],
               "parameter_count": len(packet["parameter_gradients"]), "cache_states": packet["cache_states"],
               "write_gradient_norms": [None if value["gradient"] is None else float(value["gradient"].double().norm())
                                        for value in packet["writes"]], "dense": {}}
        for name in DENSE_NAMES:
            row["dense"][name] = {}
            for kind in ("per_token", "batched"):
                operands = dense_operands(packet, name, kind)
                if operands is None:
                    continue
                reference = contraction(operands["effective_input"], operands["output_gradient"])
                actual = packet["parameter_gradients"][f"{name}.weight"]
                item = {"input_dtype": str(operands["effective_input"].dtype),
                        "output_gradient_dtype": str(operands["output_gradient"].dtype),
                        "actual_weight_gradient_vs_fixed_operand_fp64": metrics(reference, actual)}
                if packet["bf16"]:
                    reductions, reduction_report = fixed_operand_reductions(operands["effective_input"], operands["output_gradient"])
                    item["explicit_reductions_vs_fp64"] = reduction_report
                    item["actual_weight_gradient_vs_batched_bf16_contraction"] = metrics(reductions["batched_bf16"], actual)
                    item["actual_weight_gradient_vs_fp64_rounded_once_to_bf16"] = metrics(reference.bfloat16().float(), actual)
                row["dense"][name][kind] = item
        report["arms"][arm] = row
    comparisons = [("naive_fp32_eager", "tiled_fp32_eager"),
                   ("naive_bf16_cache_on", "naive_bf16_cache_off"),
                   ("tiled_bf16_cache_on", "tiled_bf16_cache_off"),
                   ("naive_bf16_cache_on", "tiled_bf16_cache_on"),
                   ("naive_bf16_cache_off", "tiled_bf16_cache_off"),
                   ("tiled_fp32_eager", "tiled_bf16_cache_on"),
                   ("tiled_fp32_eager", "tiled_bf16_cache_off")]
    comparisons += [(f"tiled_bf16_cache_{state}", f"tiled_bf16_cache_{state}_compiled") for state in ("on", "off")]
    for left, right in comparisons:
        if left not in packets or right not in packets:
            continue
        a, b = packets[left], packets[right]
        comparison = {key: metrics(a[key], b[key]) for key in ("output", "input_gradient")}
        comparison["parameters"] = {name: metrics(value, b["parameter_gradients"][name])
                                    for name, value in a["parameter_gradients"].items()}
        if a["writes"] and b["writes"]:
            comparison["earlier_writes"] = [{key: metrics(a["writes"][t][key], b["writes"][t][key])
                                              for key in ("value", "gradient")}
                                             for t in range(len(a["writes"]) - 1)]
        label = f"{right}_vs_{left}"
        report["comparisons"][label] = comparison
        dense_rows = {}
        for name in DENSE_NAMES:
            a_kind = "per_token" if a["backend"] == "naive" else "batched"
            b_kind = "per_token" if b["backend"] == "naive" else "batched"
            aa, bb = dense_operands(a, name, a_kind), dense_operands(b, name, b_kind)
            if aa is not None and bb is not None:
                dense_rows[name] = {key: metrics(aa[key], bb[key]) for key in aa}
        report["dense_operand_comparisons"][label] = dense_rows
    for arm, packet in packets.items():
        if packet["backend"] != "tiled" or not packet["instrumented"]:
            continue
        comparison = {}
        for name in DENSE_NAMES:
            local, batched = dense_operands(packet, name, "per_token"), dense_operands(packet, name, "batched")
            comparison[name] = {key: metrics(local[key], batched[key]) for key in local}
        report["dense_operand_comparisons"][f"{arm}_batched_vs_local_replay"] = comparison
    return report


def load_fixture(args):
    if args.source_case:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        source = torch.load(args.source_case, map_location="cpu", weights_only=False)
        boundaries = source["packets"][args.source_arm]["boundaries"]
        prefix = "transformer.blocks.3."
        weights = {name[len(prefix):]: value for name, value in checkpoint["model"].items()
                   if name.startswith(prefix)}
        raw = dict(checkpoint["model_config"])
        if raw["recurrent_layers"] != [3] or raw["recurrent_write_rho"] != 1.:
            raise AssertionError("Captured-input fixture requires authoritative R3/rho1 configuration")
        report_path = args.source_case.parent / "report.json"
        if report_path.exists():
            source_report = json.loads(report_path.read_text())
            if source_report["checkpoint"]["sha256"] != digest(args.checkpoint):
                raise AssertionError("Captured inputs and checkpoint identities differ")
        fixture = {"configurations": {"naive_fp32": raw}, "block_weights": {"naive_fp32": weights},
                   "inputs": boundaries["block3_input"], "cotangent": boundaries["block3_output_gradient"],
                   "require_all_earlier_writes": False}
        from olmo.config import ModelConfig
        from olmo.efficient_utils import alibi_attention_bias
        from olmo.model import OLMo
        bias_config = dict(raw)
        bias_config.update(init_device="cpu", precision=None, reference_eager=True)
        bias_model = OLMo(ModelConfig(**bias_config))
        config = SimpleNamespace(model=bias_model.config)
        fixture["attention_bias"] = cpu(alibi_attention_bias(bias_model, config, fixture["inputs"].shape[1]))
        return fixture, {"source_case": str(args.source_case), "source_case_sha256": digest(args.source_case),
                         "source_arm": args.source_arm, "checkpoint": str(args.checkpoint),
                         "checkpoint_sha256": digest(args.checkpoint), "shape": list(fixture["inputs"].shape),
                         "purpose": "Matched captured input and output cotangent; diagnostic isolation of one recurrent block"}
    if args.fixture_seed is not None:
        from r3_write_path_probe import build_models
        from olmo.efficient_utils import alibi_attention_bias
        models, _ = build_models(args.fixture_seed)
        reference = models["naive"]
        generator = torch.Generator(device="cpu").manual_seed(args.fixture_seed + 1)
        inputs = torch.randn(2, 16, 32, generator=generator)
        cotangent = torch.zeros_like(inputs)
        cotangent[:, -1] = torch.randn(2, 32, generator=generator)
        fixture = {"configurations": {"naive_fp32": asdict(reference.config)},
                   "block_weights": {"naive_fp32": {name: cpu(value) for name, value in reference.transformer.blocks[3].state_dict().items()}},
                   "inputs": inputs, "cotangent": cotangent,
                   "attention_bias": cpu(alibi_attention_bias(reference, SimpleNamespace(model=reference.config), 16)),
                   "require_all_earlier_writes": True}
        return fixture, {"source_seed": args.fixture_seed, "shape": list(inputs.shape),
                         "purpose": "New independent initialization and final-position cotangent; caller assigns confirmatory status"}
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    if fixture["inputs"].shape != (2, 16, 32) or torch.count_nonzero(fixture["cotangent"][:, :-1]):
        raise AssertionError("Expected the retained B2/T16/D32 final-position-only cotangent fixture")
    return fixture, {"path": str(args.fixture), "sha256": digest(args.fixture),
                     "shape": list(fixture["inputs"].shape), "source_seed": 3401,
                     "purpose": "Historical diagnostic fixture, not fresh confirmatory evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fixture-seed", type=int, help="Construct a fresh tiny fixture with this initialization/data seed")
    parser.add_argument("--source-case", type=Path, help="Full-model tensors.pt containing captured block3 boundaries")
    parser.add_argument("--checkpoint", type=Path, help="Authoritative checkpoint matching --source-case")
    parser.add_argument("--source-arm", default="tiled_bf16")
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--synthetic-length", type=int, default=128)
    parser.add_argument("--skip-actual", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--compiled-check", action="store_true")
    args = parser.parse_args()
    if args.synthetic_length < 1:
        raise ValueError("Synthetic length must be positive")
    if bool(args.source_case) != bool(args.checkpoint):
        raise ValueError("--source-case and --checkpoint must be specified together")
    if args.fixture_seed is not None and args.source_case:
        raise ValueError("Choose either a fresh tiny seed or captured full-model inputs")
    if args.skip_actual and args.skip_synthetic:
        raise ValueError("At least one probe must execute")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    forbidden = [Path(".runtime") / name for name in ("stage-b", "r3-backward", "r3-bf16", "cdrm-naive")]
    if any(path.resolve() == output or path.resolve() in output.parents for path in forbidden):
        raise ValueError("Retained prior experiment lineages are immutable; choose a new lineage")
    hardware = require_cuda_container()
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": "r3-dense-gradient-attribution-v1", "evidence": "NUM", "status": "running",
              "command": shlex.join([sys.executable, *sys.argv]),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "scope": "Diagnostic attribution only; no new precision acceptance threshold or clearance",
              "instrumentation": "Actual dense operands use eager helper execution; optional compiled arms omit all hooks",
              "settings": {"tf32": False, "deterministic": True, "sdpa": "math", "parameters": "FP32",
                           "mixed_policy": "bf16_fp32_state", "global_cache_policy_spans_forward_and_backward": True}}
    packets = {}
    try:
        seed_all(args.seed, deterministic=True)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        configure_compiled_helpers(True)
        report["provenance"] = provenance(hardware)
        source_path = output / "probe-source.py"
        source_path.write_bytes(Path(__file__).read_bytes())
        report["script_sha256"] = digest(source_path)
        report["script_snapshot"] = source_path.name
        report["settings"]["bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        report["settings"]["math_sdpa_reduced_precision_reduction"] = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
        if not args.skip_synthetic:
            packets["synthetic"], report["synthetic"] = synthetic_linear(args.seed, args.synthetic_length)
        if not args.skip_actual:
            fixture, report["fixture"] = load_fixture(args)
            packets["fixture"] = fixture
            actual = {}
            specifications = [("naive_fp32_eager", "naive", False, True),
                              ("tiled_fp32_eager", "tiled", False, True),
                              ("naive_bf16_cache_on", "naive", True, True),
                              ("naive_bf16_cache_off", "naive", True, False),
                              ("tiled_bf16_cache_on", "tiled", True, True),
                              ("tiled_bf16_cache_off", "tiled", True, False)]
            for name, backend, bf16, enabled in specifications:
                print(f"Isolated fixed-fixture R3: {name}", flush=True)
                actual[name] = isolated_arm(fixture, backend, bf16, enabled)
            if args.compiled_check:
                for enabled in (True, False):
                    name = f"tiled_bf16_cache_{'on' if enabled else 'off'}_compiled"
                    print(f"Uninstrumented isolated R3: {name}", flush=True)
                    torch._dynamo.reset()
                    torch._dynamo.utils.counters.clear()
                    actual[name] = isolated_arm(fixture, "tiled", True, enabled, eager=False, capture=False)
                    report.setdefault("compiled_checks", {})[name] = compiler_audit(True)
            packets["actual"] = actual
            report["actual"] = analyze_isolated(actual)
        report["status"] = "diagnostics_complete"
    except BaseException as exc:
        report.update(status="execution_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        torch.save(packets, output / "tensors.pt")
        report["tensors_sha256"] = digest(output / "tensors.pt")
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
