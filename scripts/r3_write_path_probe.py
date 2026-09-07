#!/usr/bin/env python3
"""Isolated FP32 R3 persistent-write gradient proof; NUM only, one GPU.

Only block 3 executes. A cotangent on its final output has no ordinary later
block through which to bypass recurrent history. The projection hooks observe
existing tensors and return None; they never detach or replace model outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from r3_validation_metrics import compare_tensors, NORM_FLOOR
from stage_a_common import (configure_compiled_helpers, provenance, require_cuda_container,
                            seed_all, unique_parameters, write_json)
from stage_b_train import compiler_audit as strict_compiler_audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_copy(value: torch.Tensor | None):
    return value.detach().cpu().clone() if value is not None else None


def norm(value: torch.Tensor | None) -> float | None:
    return float(value.detach().double().norm().cpu()) if value is not None else None


def build_models(seed: int):
    from olmo.checkpoint_conversion import convert_model
    from olmo.config import ModelConfig
    from olmo.model import OLMo

    seed_all(seed, deterministic=True)
    common = dict(n_layers=12, d_model=32, n_heads=4, n_kv_heads=4,
        mlp_hidden_size=128, activation_type="gelu", vocab_size=64, embedding_size=64,
        max_sequence_length=16, eos_token_id=1, pad_token_id=0,
        alibi=True, rope=False, flash_attention=False, norm_after=False,
        attention_dropout=0.0, embedding_dropout=0.0, residual_dropout=0.0,
        attention_layer_norm=True, attention_layer_norm_with_affine=True,
        embedding_layer_norm=False, layer_norm_type="default", layer_norm_with_affine=True,
        include_bias=False, bias_for_layer_norm=False, weight_tying=False,
        block_type="sequential", recurrent_write_rho=1.0, bwd_mlp_chunks=4,
        init_device="cpu", init_fn="mitchell")
    source = OLMo(ModelConfig(**common, recurrent_layers=[], recurrent_backend="naive", reference_eager=True))
    models, conversions = {}, {}
    for backend in ("naive", "tiled"):
        model = OLMo(ModelConfig(**common, recurrent_layers=[3], recurrent_backend=backend,
                                 reference_eager=backend == "naive"))
        conversions[backend] = convert_model(source, model).to_dict()
        models[backend] = model.to(device="cuda", dtype=torch.float32).train()
        unique_parameters(models[backend].transformer.blocks[3])
    return models, conversions


def gradient_packet(block, inputs: torch.Tensor, cotangent: torch.Tensor,
                    attention_bias: torch.Tensor, backend: str) -> dict:
    leaf = inputs.detach().clone().requires_grad_(True)
    writes = []

    def capture_permanent_projection(module, args, output):
        # Do not introduce Python mutation into the compiled temporary-K/V
        # pre-attention helper. Permanent writes are projected outside it.
        if torch.compiler.is_compiling():
            return None
        if not torch.is_grad_enabled() or not output.requires_grad:
            return None
        # T=16 distinguishes the full-sequence temporary projection. In tiled
        # mode, grad-enabled singleton projections occur during backward's
        # persistent-write reconstruction, in increasing token order.
        if output.ndim == 3 and output.shape[1] == 1:
            output.retain_grad()
            writes.append(output)
        return None

    hook = block.kv_proj.register_forward_hook(capture_permanent_projection)
    block.zero_grad(set_to_none=True)
    try:
        with torch.autocast("cuda", enabled=False):
            outputs = block(leaf, attention_bias=attention_bias)[0]
            objective = (outputs * cotangent).sum()
        forward_write_captures = len(writes)
        objective.backward()
        if len(writes) != inputs.shape[1]:
            raise AssertionError(f"{backend}: expected one observed permanent write per token, got {len(writes)}")
        expected_forward = inputs.shape[1] if backend == "naive" else 0
        if forward_write_captures != expected_forward:
            raise AssertionError(f"{backend}: projection-hook phase classification failed")
        parameters = {name: cpu_copy(parameter.grad) for name, parameter in block.named_parameters()}
        return {"backend": backend, "output": cpu_copy(outputs), "objective": cpu_copy(objective),
            "input": cpu_copy(leaf), "input_gradient": cpu_copy(leaf.grad),
            "parameter_gradients": parameters,
            "writes": [{"position": position, "projected_kv": cpu_copy(write),
                        "projected_kv_gradient": cpu_copy(write.grad)}
                       for position, write in enumerate(writes)],
            "forward_write_captures": forward_write_captures,
            "total_write_captures": len(writes),
            "output_dtype": str(outputs.dtype), "input_dtype": str(leaf.dtype)}
    finally:
        hook.remove()


def compare_packets(naive: dict, tiled: dict) -> dict:
    if list(naive["parameter_gradients"]) != list(tiled["parameter_gradients"]):
        raise AssertionError("Canonical block parameter names/order differ")
    comparisons = {"output": compare_tensors(naive["output"], tiled["output"]),
                   "objective": compare_tensors(naive["objective"], tiled["objective"]),
                   "input_gradient": compare_tensors(naive["input_gradient"], tiled["input_gradient"])}
    comparisons.update({f"parameter:{name}": compare_tensors(value, tiled["parameter_gradients"][name])
                        for name, value in naive["parameter_gradients"].items()})
    history = []
    earlier_count = len(naive["writes"]) - 1
    for position in range(earlier_count):
        left, right = naive["writes"][position], tiled["writes"][position]
        prefix = f"permanent_write:{position}"
        comparisons[prefix + ":value"] = compare_tensors(left["projected_kv"], right["projected_kv"])
        comparisons[prefix + ":gradient"] = compare_tensors(left["projected_kv_gradient"], right["projected_kv_gradient"])
        norms = {}
        for backend, row in (("naive", left), ("tiled", right)):
            gradient = row["projected_kv_gradient"]
            norms[backend] = {"kv_gradient_l2": norm(gradient),
                "key_gradient_l2": norm(gradient[..., :32]) if gradient is not None else None,
                "value_gradient_l2": norm(gradient[..., 32:]) if gradient is not None else None}
        history.append({"position": position, "norms": norms,
            "nontrivial_both": all(row["kv_gradient_l2"] is not None and row["kv_gradient_l2"] > NORM_FLOOR
                                   for row in norms.values())})
    terminal_naive = naive["writes"][-1]["projected_kv_gradient"]
    terminal_tiled = tiled["writes"][-1]["projected_kv_gradient"]
    terminal_rule = {"position": earlier_count,
        "expected": "Naive terminal permanent write is unused (None); tiled explicitly backpropagates zero through its reconstructed terminal write",
        "naive_gradient_missing": terminal_naive is None,
        "tiled_gradient_present": terminal_tiled is not None,
        "tiled_gradient_l2": norm(terminal_tiled),
        "passed": terminal_naive is None and terminal_tiled is not None and bool(torch.count_nonzero(terminal_tiled) == 0),
        "scope": "This expected unused-intermediate bookkeeping difference does not waive any intended parameter gradient"}
    earlier_input = {backend: norm(packet["input_gradient"][:, :-1]) if packet["input_gradient"] is not None else None
                     for backend, packet in (("naive", naive), ("tiled", tiled))}
    history_reach = {"earlier_input_gradient_l2": earlier_input,
        "earlier_input_nontrivial_both": all(value is not None and value > NORM_FLOOR for value in earlier_input.values()),
        "every_earlier_permanent_write_nontrivial_both": all(row["nontrivial_both"] for row in history),
        "nontrivial_norm_floor": NORM_FLOOR, "earlier_write_gradients": history,
        "direct_output_cotangent": "Exactly zero at every position except the last",
        "bypass_control": "Only block 3 executes; no ordinary subsequent layers or output head"}
    elementwise_failures = [name for name, row in comparisons.items() if not row["elementwise_pass"]]
    scale_aware_failures = [name for name, row in comparisons.items() if not row["scale_aware_pass"]]
    passed = (not elementwise_failures and not scale_aware_failures and terminal_rule["passed"] and
              history_reach["earlier_input_nontrivial_both"] and history_reach["every_earlier_permanent_write_nontrivial_both"])
    return {"status": "passed" if passed else "failed_bounds_or_history_assertion",
            "parameter_tensor_count": len(naive["parameter_gradients"]),
            "parameter_mapping": "Exact canonical name/order mapping; both recurrent backends share registration",
            "comparisons": comparisons, "elementwise_failures": elementwise_failures,
            "scale_aware_failures": scale_aware_failures,
            "terminal_write_rule": terminal_rule, "history_reach": history_reach}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3401)
    args = parser.parse_args()
    hardware = require_cuda_container()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output directory to retain earlier numerical evidence")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    configure_compiled_helpers(True)
    report = {"schema": "r3-isolated-write-path-v1", "evidence_class": "NUM",
        "seed": args.seed, "precision": "FP32 parameters/input/cotangent; autocast and TF32 disabled",
        "executed_block_index": 3, "model_depth_for_initialization": 12,
        "shape": {"batch": 2, "sequence_length": 16, "width": 32, "heads": 4, "mlp_width": 128},
        "rho": 1.0, "backward_mlp_chunks": 4, "attention_backend": "math SDPA",
        "status": "started", "artifacts": {}}
    packets = {}
    try:
        models, conversions = build_models(args.seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        from olmo.efficient_utils import alibi_attention_bias
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        inputs = torch.randn(2, 16, 32, generator=generator, dtype=torch.float32).cuda()
        cotangent = torch.zeros_like(inputs)
        cotangent[:, -1] = torch.randn(2, 32, generator=generator, dtype=torch.float32).cuda()
        attention_bias = alibi_attention_bias(models["naive"], SimpleNamespace(model=models["naive"].config), 16)
        if attention_bias.dtype != torch.float32:
            raise AssertionError("Expected FP32 combined causal/ALiBi bias")
        configs = {name: asdict(model.config) for name, model in models.items()}
        weights = {name: {key: cpu_copy(value) for key, value in model.transformer.blocks[3].state_dict().items()}
                   for name, model in models.items()}
        if weights["naive"].keys() != weights["tiled"].keys() or any(
            not torch.equal(value, weights["tiled"][name]) for name, value in weights["naive"].items()
        ):
            raise AssertionError("Isolated blocks did not start from identical weights")
        fixture_path = args.output_dir / "fixture-and-weights.pt"
        torch.save({"format": "r3-isolated-write-fixture-v1", "configurations": configs,
            "block_weights": weights, "inputs": cpu_copy(inputs), "cotangent": cpu_copy(cotangent),
            "attention_bias": cpu_copy(attention_bias), "seed": args.seed}, fixture_path)
        report.update(configurations=configs, conversions=conversions, provenance=provenance(hardware),
            cotangent_l2=norm(cotangent), fixture_description="Leaf input independent of embeddings; last-position-only raw random cotangent")
        report["provenance"]["source_sha256"]["scripts/r3_write_path_probe.py"] = sha256_file(Path(__file__))
        report["provenance"]["source_sha256"]["scripts/r3_validation_metrics.py"] = sha256_file(Path(__file__).with_name("r3_validation_metrics.py"))
        with sdpa_kernel(SDPBackend.MATH):
            for backend in ("naive", "tiled"):
                print(f"Persistent-write NUM: {backend}, B2 T16 D32 block3", flush=True)
                packets[backend] = gradient_packet(models[backend].transformer.blocks[3], inputs,
                                                   cotangent, attention_bias, backend)
        report["compiler_audit"] = strict_compiler_audit(True)
        report.update(compare_packets(packets["naive"], packets["tiled"]))
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if packets:
            torch.save({"format": "r3-isolated-write-gradient-packets-v1", "packets": packets}, args.output_dir / "gradient-packets.pt")
        for path in sorted(args.output_dir.glob("*.pt")):
            report["artifacts"][path.name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        write_json(args.output_dir / "summary.json", report)
    print(json.dumps({"status": report["status"], "summary": str(args.output_dir / "summary.json"),
                      "elementwise_failures": report.get("elementwise_failures"),
                      "scale_aware_failures": report.get("scale_aware_failures")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
