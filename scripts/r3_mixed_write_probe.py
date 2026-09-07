#!/usr/bin/env python3
"""Isolated BF16 R3 persistent-write triad; new NUM artifacts, no optimizer steps.

Only block3 executes, with B2/T16/D32 FP32 leaf inputs and an FP32 cotangent
supported at the last output position. Projection hooks observe existing tensors
and return None. The old FP32 probe and its retained artifacts remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from r3_validation_metrics import compare_tensors, NORM_FLOOR
from r3_write_path_probe import build_models, cpu_copy, norm, sha256_file
from stage_a_common import configure_compiled_helpers, provenance, require_cuda_container, unique_parameters
from stage_b_train import atomic_json, compiler_audit

ARMS = ("naive_fp32", "naive_bf16", "tiled_bf16")
PAIRS = (("naive_fp32", "naive_bf16"), ("naive_bf16", "tiled_bf16"), ("naive_fp32", "tiled_bf16"))


def candidate_from(reference, backend, *, device="cuda"):
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    raw = asdict(reference.config)
    raw.update(precision=None, recurrent_precision_policy="bf16_fp32_state", init_device="cpu",
               recurrent_backend=backend, reference_eager=backend == "naive")
    model = OLMo(ModelConfig(**raw)).to(device=device, dtype=torch.float32).train()
    model.load_state_dict(reference.state_dict(), strict=True)
    if any(block.config.recurrent_precision_policy != "bf16_fp32_state" for block in model.transformer.blocks):
        raise AssertionError("Policy must be present in every independently copied block config")
    return model


def comparison(reference, actual):
    row = compare_tensors(reference, actual)
    valid = row["reference_present"] and row["actual_present"] and row["shape_match"] and row["finite"]
    row["historical_bf16_gradient_screen"] = bool(valid and row["rel_l2"] <= .015625
                                                and row["maxerr_over_ref_rms"] <= .0625)
    row["cosine"] = None
    if valid:
        a, b = reference.double().reshape(-1), actual.double().reshape(-1)
        denominator = a.norm() * b.norm()
        if denominator > 1e-24:
            row["cosine"] = float(torch.dot(a, b) / denominator)
    return row


def gradient_packet(block, inputs, cotangent, attention_bias, arm):
    leaf = inputs.detach().clone().requires_grad_(True)
    writes, phases, dtypes = [], [], {}
    phase = "forward"
    internal_dtypes = {}
    expected_policy = "legacy" if arm == "naive_fp32" else "bf16_fp32_state"
    if block.config.recurrent_precision_policy != expected_policy or block.config.precision is not None:
        raise AssertionError(f"{arm}: executed block config does not match declared precision policy")

    def internal(phase, tensors, token_index=None):
        internal_dtypes.setdefault(phase, []).append({name: str(value.dtype) for name, value in tensors.items()
                                                     if isinstance(value, torch.Tensor)})

    def permanent_projection(module, args, output):
        if torch.compiler.is_compiling():
            return None
        key = f"{phase}/kv_projection/{output.dtype}"
        dtypes[key] = dtypes.get(key, 0) + 1
        if torch.is_grad_enabled() and output.requires_grad and output.ndim == 3 and output.shape[1] == 1:
            output.retain_grad()
            writes.append(output)
            phases.append(phase)
        return None

    handle = block.kv_proj.register_forward_hook(permanent_projection)
    prior_observer = getattr(block, "_recurrent_precision_observer", None)
    block._recurrent_precision_observer = internal
    block.zero_grad(set_to_none=True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=arm != "naive_fp32"):
            outputs = block(leaf, attention_bias=attention_bias)[0]
        if outputs.dtype != torch.float32:
            raise AssertionError("The selected profile must preserve the FP32 residual/output stream")
        objective = (outputs * cotangent).sum()
        forward_captures = len(writes)
        phase = "backward"
        # Backward is outside outer autocast; the custom function owns its
        # declared recomputation contexts. No later block or output head exists.
        objective.backward()
        expected = inputs.shape[1] if arm.startswith("naive") else 0
        if forward_captures != expected or len(writes) != inputs.shape[1]:
            raise AssertionError(f"{arm}: incorrect permanent-write hook phase/count: {forward_captures}/{len(writes)}")
        parameters = {name: cpu_copy(parameter.grad) for name, parameter in block.named_parameters()}
        if len(parameters) != 9:
            raise AssertionError(f"Expected all9 intended block parameter gradients, observed {len(parameters)}")
        if any(value is None or value.dtype != torch.float32 or not torch.isfinite(value).all()
               for value in parameters.values()):
            raise AssertionError("All intended parameter gradients must be present, finite FP32")
        if leaf.grad is None or leaf.grad.dtype != torch.float32 or not torch.isfinite(leaf.grad).all():
            raise AssertionError("Expected finite FP32 input gradient")
        if any(parameter.dtype != torch.float32 for parameter in block.parameters()):
            raise AssertionError("Parameters must remain FP32")
        if any(not torch.isfinite(write).all() for write in writes):
            raise FloatingPointError("Nonfinite permanent projected K/V")
        if any(write.grad is None or not torch.isfinite(write.grad).all() for write in writes[:-1]):
            raise AssertionError("Every earlier write must receive a finite gradient")
        required_fp32 = ["forward.attention"] if arm != "naive_fp32" else []
        if arm == "tiled_bf16":
            required_fp32 += ["forward.initial_state", "forward.running_state", "backward.recomputed_attention",
                              "backward.buffers", "backward.attention_adjoint", "backward.pre_attention_adjoint"]
        for name in required_fp32:
            observed = internal_dtypes.get(name, [])
            if not observed or any(dtype != "torch.float32" for row in observed for dtype in row.values()):
                raise AssertionError(f"{arm}: candidate FP32 state/adjoints not observed at {name}: {observed}")
        if arm != "naive_fp32" and any(write.dtype != torch.bfloat16 for write in writes):
            raise AssertionError("Candidate permanent projected records must retain BF16 storage")
        return {"arm": arm, "input": cpu_copy(leaf), "input_gradient": cpu_copy(leaf.grad),
                "output": cpu_copy(outputs), "objective": cpu_copy(objective),
                "parameter_gradients": parameters, "parameter_order": list(parameters),
                "writes": [{"position": position, "capture_phase": phases[position],
                            "projected_kv": cpu_copy(write), "projected_kv_gradient": cpu_copy(write.grad),
                            "value_dtype": str(write.dtype), "gradient_dtype": str(write.grad.dtype) if write.grad is not None else None}
                           for position, write in enumerate(writes)],
                "forward_write_captures": forward_captures, "total_write_captures": len(writes),
                "observed_projection_dtypes": dtypes, "input_dtype": str(leaf.dtype),
                "observed_internal_dtypes": internal_dtypes, "executed_block_config": asdict(block.config),
                "active_candidate_state_assertions": required_fp32,
                "output_dtype": str(outputs.dtype), "cotangent_dtype": str(cotangent.dtype),
                "objective_dtype": str(objective.dtype), "parameter_gradient_dtype": "torch.float32"}
    finally:
        handle.remove()
        if prior_observer is None:
            del block._recurrent_precision_observer
        else:
            block._recurrent_precision_observer = prior_observer


def compare_packets(reference, actual):
    if reference["parameter_order"] != actual["parameter_order"]:
        raise AssertionError("Canonical block parameter name/order differs")
    rows = {name: comparison(reference[name], actual[name]) for name in ("output", "objective", "input_gradient")}
    for name, value in reference["parameter_gradients"].items():
        rows[f"parameter:{name}"] = comparison(value, actual["parameter_gradients"][name])
    for position in range(15):
        for name in ("projected_kv", "projected_kv_gradient"):
            rows[f"permanent_write:{position}:{name}"] = comparison(reference["writes"][position][name],
                                                                      actual["writes"][position][name])
    return {"reference": reference["arm"], "actual": actual["arm"], "comparisons": rows,
            "original_fp32_elementwise_failures": [n for n, r in rows.items() if not r["elementwise_pass"]],
            "original_fp32_scale_failures": [n for n, r in rows.items() if not r["scale_aware_pass"]],
            "historical_bf16_gradient_screen_failures": [n for n, r in rows.items() if not r["historical_bf16_gradient_screen"]],
            "historical_bf16_output_elementwise": compare_tensors(reference["output"], actual["output"], atol=.002, rtol=.02),
            "all_present_finite": all(r["reference_present"] and r["actual_present"] and r["shape_match"] and r["finite"]
                                      for r in rows.values())}


def credit(packet):
    rows = []
    for row in packet["writes"][:-1]:
        gradient = row["projected_kv_gradient"]
        rows.append({"position": row["position"], "projected_kv_gradient_l2": norm(gradient),
                     "key_gradient_l2": norm(gradient[..., :32]), "value_gradient_l2": norm(gradient[..., 32:]),
                     "nontrivial": norm(gradient) > NORM_FLOOR})
    terminal = packet["writes"][-1]["projected_kv_gradient"]
    terminal_ok = (terminal is None if packet["arm"].startswith("naive") else
                   terminal is not None and torch.isfinite(terminal).all().item() and torch.count_nonzero(terminal).item() == 0)
    return {"all15_earlier_writes_nontrivial": all(row["nontrivial"] for row in rows),
            "earlier_input_gradient_l2": norm(packet["input_gradient"][:, :-1]),
            "earlier_input_nontrivial": norm(packet["input_gradient"][:, :-1]) > NORM_FLOOR,
            "earlier_writes": rows, "nontrivial_norm_floor": NORM_FLOOR,
            "terminal": {"position": 15, "gradient_missing": terminal is None, "gradient_l2": norm(terminal),
                         "expected_unused_intermediate_rule_passes": terminal_ok,
                         "rule": "Both naive arms: unused None. Tiled: explicitly propagated finite zero.",
                         "scope": "Terminal projected-KV intermediate only; never excuses missing intended parameter/input gradients"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3401)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if any(path in output.parents or path == output for path in
           (Path(".runtime/stage-b").resolve(), Path(".runtime/r3-backward").resolve())):
        raise ValueError("Original research/FP32 validation lineages are immutable")
    if output.exists():
        raise FileExistsError("Use a new output directory")
    hardware = require_cuda_container()
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    configure_compiled_helpers(True)
    report = {"schema": "r3-mixed-isolated-write-v1", "evidence_class": "NUM", "status": "running",
              "command": shlex.join([sys.executable, *sys.argv]), "seed": args.seed,
              "shape": {"batch": 2, "sequence_length": 16, "width": 32, "heads": 4, "mlp_width": 128},
              "executed_block": 3, "initialization_depth": 12, "rho": 1., "backward_mlp_chunks": 4,
              "criteria": {"original_atol": 2e-6, "original_rtol": 2e-5,
                           "historical_bf16_relative_l2": .015625, "historical_bf16_max_error_over_rms": .0625,
                           "historical_bf16_output_atol": .002, "historical_bf16_output_rtol": .02,
                           "policy": "Unchanged historical diagnostics; failed flags remain visible, no blanket clearance"},
              "precision": {"parameters": "FP32", "input": "FP32", "cotangent": "FP32", "residual": "FP32",
                            "bf16_policy": "bf16_fp32_state", "config_precision": None,
                            "outer_autocast": "Forward only for BF16 arms", "grad_scaler": False,
                            "tf32": False, "sdpa": "deterministic math", "whole_model_compile": False},
              "probe_scope": "Only isolated block3 executes; last-position cotangent cannot use a later ordinary block as a history bypass",
              "scaling_scope": "Fixed-forward scaling is covered by r3_mixed_validate.py; not duplicated here", "artifacts": {}}
    packets = {}
    try:
        originals, conversions = build_models(args.seed)
        from olmo.config import ModelConfig
        from olmo.efficient_utils import alibi_attention_bias
        if "recurrent_precision_policy" not in ModelConfig.__dataclass_fields__:
            raise RuntimeError("Selected precision policy is not implemented")
        models = {"naive_fp32": originals["naive"], "naive_bf16": candidate_from(originals["naive"], "naive"),
                  "tiled_bf16": candidate_from(originals["tiled"], "tiled")}
        for arm, model in models.items():
            expected_policy = "legacy" if arm == "naive_fp32" else "bf16_fp32_state"
            if model.config.precision is not None or model.transformer.blocks[3].config.recurrent_precision_policy != expected_policy:
                raise AssertionError("Construct policy-bearing ModelConfig before OLMo copies it into blocks")
            unique_parameters(model.transformer.blocks[3])
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        inputs = torch.randn(2, 16, 32, generator=generator, dtype=torch.float32).cuda()
        cotangent = torch.zeros_like(inputs)
        cotangent[:, -1] = torch.randn(2, 32, generator=generator, dtype=torch.float32).cuda()
        bias = alibi_attention_bias(models["naive_fp32"], SimpleNamespace(model=models["naive_fp32"].config), 16)
        if bias.dtype != torch.float32 or torch.count_nonzero(cotangent[:, :-1]).item():
            raise AssertionError("Expected FP32 attention bias and strictly last-position cotangent")
        configs = {arm: asdict(model.config) for arm, model in models.items()}
        weights = {arm: {name: cpu_copy(value) for name, value in model.transformer.blocks[3].state_dict().items()}
                   for arm, model in models.items()}
        if any(list(weights[arm]) != list(weights["naive_fp32"]) or any(
               not torch.equal(value, weights[arm][name]) for name, value in weights["naive_fp32"].items()) for arm in ARMS):
            raise AssertionError("Every isolated arm must start from exactly identical canonical FP32 weights")
        torch.save({"configurations": configs, "block_weights": weights, "inputs": cpu_copy(inputs),
                    "cotangent": cpu_copy(cotangent), "attention_bias": cpu_copy(bias)}, output / "fixture-and-weights.pt")
        report.update(configurations=configs, conversions=conversions, provenance=provenance(hardware),
                      cotangent_l2=norm(cotangent))
        report["provenance"]["source_sha256"].update({str(path): sha256_file(path) for path in
            (Path("scripts/r3_mixed_write_probe.py"), Path("scripts/r3_write_path_probe.py"),
             Path("scripts/r3_validation_metrics.py"), Path("scripts/stage_b_train.py"))})
        report["precision"].update(
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            math_sdpa_reduced_precision_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
        with sdpa_kernel(SDPBackend.MATH):
            for arm in ARMS:
                print(f"Isolated persistent-write triad: {arm}", flush=True)
                packets[arm] = gradient_packet(models[arm].transformer.blocks[3], inputs, cotangent, bias, arm)
        report["compiler_audit"] = compiler_audit(True)
        report["comparisons"] = {f"{actual}_vs_{reference}": compare_packets(packets[reference], packets[actual])
                                 for reference, actual in PAIRS}
        report["credit"] = {arm: credit(packet) for arm, packet in packets.items()}
        report["all_arms_receive_earlier_write_credit"] = all(row["all15_earlier_writes_nontrivial"] and
            row["earlier_input_nontrivial"] and row["terminal"]["expected_unused_intermediate_rule_passes"]
            for row in report["credit"].values())
        report["all_historical_bf16_gradient_screens_pass"] = all(not row["historical_bf16_gradient_screen_failures"]
                                                                    for row in report["comparisons"].values())
        report["observed_dtypes"] = {arm: {"projection": packet["observed_projection_dtypes"],
            "internal": packet["observed_internal_dtypes"], "executed_block_config": packet["executed_block_config"],
            "output": packet["output_dtype"], "parameter_gradients": packet["parameter_gradient_dtype"],
            "write_values": sorted({row["value_dtype"] for row in packet["writes"]}),
            "write_gradients": sorted({row["gradient_dtype"] for row in packet["writes"] if row["gradient_dtype"]})}
            for arm, packet in packets.items()}
        if not report["all_arms_receive_earlier_write_credit"]:
            raise AssertionError("Missing earlier-write credit or invalid unused-terminal bookkeeping")
        report["status"] = "diagnostics_complete"
    except Exception as error:
        report.update(status="execution_failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if packets:
            torch.save({"format": "r3-mixed-write-packets-v1", "packets": packets}, output / "gradient-packets.pt")
        for path in sorted(output.glob("*.pt")):
            report["artifacts"][path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        atomic_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
