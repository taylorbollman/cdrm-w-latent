#!/usr/bin/env python3
"""Stage B NUM/OPS gate on one GPU; synthetic IDs are not research evidence."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import os
from pathlib import Path
import statistics
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from stage_a_common import (
    adamw, configure_compiled_helpers, provenance, require_cuda_container,
    seed_all, tensor_bytes, unique_parameters, validate_compiler_execution, write_json,
)


def config(args, topology="r3", backend="naive"):
    from olmo.config import ModelConfig

    return ModelConfig(
        n_layers=12, d_model=args.width, n_heads=4, n_kv_heads=4,
        mlp_hidden_size=4 * args.width, activation_type="gelu",
        vocab_size=args.vocab, embedding_size=args.vocab,
        max_sequence_length=args.length, eos_token_id=1, pad_token_id=0,
        alibi=True, rope=False, flash_attention=False, norm_after=False,
        attention_dropout=0, embedding_dropout=0, residual_dropout=0,
        attention_layer_norm=True, attention_layer_norm_with_affine=True,
        embedding_layer_norm=False, layer_norm_type="default", layer_norm_with_affine=True,
        include_bias=False, bias_for_layer_norm=False, weight_tying=False,
        block_type="sequential", recurrent_layers=[3] if topology == "r3" else [],
        recurrent_backend=backend, recurrent_write_rho=1.0,
        reference_eager=backend != "tiled", bwd_mlp_chunks=4,
        init_device="cuda", init_fn="mitchell",
    )


def metrics(reference, actual):
    reference, actual = reference.double(), actual.double()
    difference = actual - reference
    rms = reference.square().mean().sqrt().clamp_min(1e-12)
    return {
        "numel": reference.numel(), "finite": bool(torch.isfinite(actual).all()),
        "max_abs": difference.abs().max().item(),
        "reference_rms": rms.item(),
        "relative_l2": (difference.norm() / reference.norm().clamp_min(1e-12)).item(),
        "max_abs_over_reference_rms": (difference.abs().max() / rms).item(),
    }


def gradient_run(model, tokens, cotangent, bf16):
    captures = []
    def capture_input(module, inputs, output):
        output.retain_grad()
        captures.append(output)
    hook = model.transformer.wte.register_forward_hook(capture_input)
    model.zero_grad(set_to_none=True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(tokens).logits
        logits.backward(cotangent.to(logits.dtype))
        if len(captures) != 1 or captures[0].grad is None:
            raise AssertionError("Missing embedding-output input gradient")
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"Missing parameter gradient: {name}")
            gradients[name] = parameter.grad.detach().float().cpu()
        return {
            "logits": logits.detach().float().cpu(),
            "input": captures[0].grad.detach().float().cpu(),
            "parameters": gradients,
        }
    finally:
        hook.remove()


def numerical(args, report):
    from olmo.model import OLMo
    from olmo.checkpoint_conversion import convert_model

    naive = OLMo(config(args)).train()
    tiled = OLMo(config(args, backend="tiled")).train()
    mapping = convert_model(naive, tiled)
    report["conversion"] = mapping.to_dict()
    report["model_config"] = dataclasses.asdict(naive.config)
    tokens = torch.randint(2, args.vocab, (args.batch, args.length), device="cuda")
    # This cotangent is exactly representable in BF16 and is shared by every mode.
    cotangent = torch.randn(args.batch, args.length, args.vocab, device="cuda").bfloat16().float()
    if args.unit_cotangent:
        cotangent = cotangent / cotangent.norm()
    report["fixture"] = {
        "token_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
        "cotangent_sha256": hashlib.sha256(cotangent.cpu().numpy().tobytes()).hexdigest(),
        "cotangent": "unit-L2 FP32 random cotangent" if args.unit_cotangent else "same BF16-representable random output cotangent in all modes",
        "cotangent_l2": cotangent.norm().item(),
        "input_gradient": "gradient at the embedding output; includes embedding parameter gradient separately",
    }
    outputs = {}
    for name, model, bf16 in (
        ("naive_fp32", naive, False), ("tiled_fp32", tiled, False),
        ("naive_bf16", naive, True), ("tiled_bf16", tiled, True),
    ):
        if args.precision == "fp32" and bf16:
            continue
        print(f"NUM {name}: B{args.batch} T{args.length} D{args.width}", flush=True)
        started = time.perf_counter()
        outputs[name] = gradient_run(model, tokens, cotangent, bf16)
        torch.cuda.synchronize()
        report.setdefault("mode_seconds", {})[name] = time.perf_counter() - started
    comparisons = [("tiled_fp32_vs_naive_fp32", "naive_fp32", "tiled_fp32", False)]
    if args.precision != "fp32":
        comparisons += [
            ("naive_bf16_vs_naive_fp32", "naive_fp32", "naive_bf16", True),
            ("tiled_bf16_vs_naive_fp32", "naive_fp32", "tiled_bf16", True),
            ("tiled_bf16_vs_naive_bf16", "naive_bf16", "tiled_bf16", True),
        ]
    failures = []
    report["comparisons"] = {}
    for label, ref_name, actual_name, bf16 in comparisons:
        ref, actual = outputs[ref_name], outputs[actual_name]
        if set(ref["parameters"]) != set(actual["parameters"]):
            raise AssertionError("Parameter gradient key coverage mismatch")
        # Both are recurrent rho=1 with identical projection ownership; the
        # conversion map therefore maps each parameter by its canonical name.
        expected = {key for item in mapping.parameter_mappings for key in item.source_keys}
        if set(ref["parameters"]) != expected:
            raise AssertionError("Conversion-map parameter coverage mismatch")
        rows = {"input": (ref["input"], actual["input"])}
        rows.update({name: (value, actual["parameters"][name]) for name, value in ref["parameters"].items()})
        result = {"parameter_tensor_count": len(expected), "gradients": {}}
        for name, (reference, value) in rows.items():
            row = metrics(reference, value)
            if bf16:
                row["passed"] = row["finite"] and row["relative_l2"] <= 0.015625 and row["max_abs_over_reference_rms"] <= 0.0625
            else:
                row["passed"] = row["finite"] and bool(torch.all((reference - value).abs() <= 2e-6 + 2e-5 * reference.abs()))
            result["gradients"][name] = row
            if not row["passed"]:
                failures.append({"comparison": label, "tensor": name, **row})
        logit_row = metrics(ref["logits"], actual["logits"])
        atol, rtol = (0.002, 0.02) if bf16 else (2e-6, 2e-5)
        logit_row["passed"] = logit_row["finite"] and bool(torch.all((ref["logits"] - actual["logits"]).abs() <= atol + rtol * ref["logits"].abs()))
        result["logits"] = logit_row
        if not logit_row["passed"]:
            failures.append({"comparison": label, "tensor": "logits", **logit_row})
        report["comparisons"][label] = result
    report["bounds"] = {
        "fp32_elementwise_atol": 2e-6, "fp32_elementwise_rtol": 2e-5,
        "bf16_gradient_relative_l2": 0.015625, "bf16_gradient_max_abs_over_reference_rms": 0.0625,
        "bf16_logit_elementwise_atol": 0.002, "bf16_logit_elementwise_rtol": 0.02,
        "selection": "unchanged Stage A bounds; failures are retained",
    }
    report["failures"] = failures
    report["status"] = "failed_bounds" if failures else "passed"
    print(f"NUM {report['status']}: {len(failures)} failed tensor comparisons", flush=True)


def step(model, optimizer, tokens, labels, precision):
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(tokens).logits
        end.record()
        active = labels != -100
        loss = F.cross_entropy(logits[active].float(), labels[active])
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(unique_parameters(model), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"update_seconds": elapsed, "forward_seconds": start.elapsed_time(end) / 1000,
            "answer_ce": loss.item(), "gradient_norm_before_clip": grad_norm.item()}


def operational(args, report):
    from olmo.model import OLMo
    from olmo.checkpoint_conversion import convert_model

    source = OLMo(config(args, topology="seq"))
    model = OLMo(config(args, topology=args.topology, backend="tiled"))
    report["conversion"] = convert_model(source, model).to_dict()
    del source
    gc.collect()
    torch.cuda.empty_cache()
    model.train(args.mode == "benchmark")
    report["model_config"] = dataclasses.asdict(model.config)
    report["parameter_count"] = sum(p.numel() for p in model.parameters())
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    tokens = torch.randint(2, args.vocab, (args.batch, args.length), generator=generator).cuda()
    labels = torch.full_like(tokens, -100)
    labels[:, 15::16] = torch.randint(0, args.vocab, labels[:, 15::16].shape, generator=generator).cuda()
    report["data"] = {
        "kind": "random symbolic IDs; no scientific learning claim",
        "target_alignment": "labels at scored logit positions, no extra shift",
        "answer_count": int((labels != -100).sum()),
        "answer_positions": "every sixteenth position including final position",
        "token_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
    }
    if args.mode == "benchmark":
        optimizer = adamw(model)
        report["optimizer"] = {"name": "AdamW", "lr": 0.001, "betas": [0.9, 0.95], "eps": 1e-8,
                               "weight_decay": 0.0, "clip_grad_norm": 1.0, "fused": False,
                               "precision": "FP32 parameters and moments; BF16 outer autocast" if args.precision == "bf16" else "FP32"}
        run = lambda: step(model, optimizer, tokens, labels, args.precision)
    else:
        def run():
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                logits = model(tokens).logits
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation logits")
            return {"forward_seconds": elapsed}
    report["warmup"] = []
    for index in range(2):
        print(f"{args.mode} {args.topology} warmup {index + 1}/2", flush=True)
        report["warmup"].append(run())
    torch.cuda.reset_peak_memory_stats()
    report["measured"] = []
    for index in range(3):
        value = run()
        report["measured"].append(value)
        print(f"{args.mode} {args.topology} measured {index + 1}/3: {value}", flush=True)
    report["summary"] = {
        "mean_forward_seconds": statistics.mean(row["forward_seconds"] for row in report["measured"]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "input_tokens_per_forward": args.batch * args.length,
        "timing_scope": "prepared-data forward, aligned answer CE, backward, gradient clip, AdamW" if args.mode == "benchmark" else "prepared-data forward only",
    }
    if args.mode == "benchmark":
        duration = statistics.mean(row["update_seconds"] for row in report["measured"])
        report["summary"].update(mean_update_seconds=duration, input_tokens_per_second=args.batch * args.length / duration,
                                 optimizer_state_bytes=tensor_bytes(optimizer.state), projected_2000_update_hours=duration * 2000 / 3600)
    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
        raise FloatingPointError("Nonfinite final model parameters")
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("numerical", "benchmark", "eval"), required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--topology", choices=("seq", "r3"), default="r3")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=937)
    parser.add_argument("--unit-cotangent", action="store_true", help="Unit-L2 directional derivative with unchanged bounds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.unit_cotangent and args.precision != "fp32":
        parser.error("The unit-cotangent diagnostic is FP32-only")
    report = {"schema": "stage-b-backend-v1", "evidence": "NUM" if args.mode == "numerical" else "OPS",
              "status": "running", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    code = 1
    try:
        hardware = require_cuda_container()
        seed_all(args.seed, deterministic=True)
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        configure_compiled_helpers(True)
        report["provenance"] = provenance(hardware)
        report["attention_backend"] = "explicit math SDPA; deterministic algorithms; TF32 off"
        report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with sdpa_kernel(SDPBackend.MATH):
            if args.mode == "numerical":
                numerical(args, report)
            else:
                operational(args, report)
        validate_compiler_execution(True)
        code = 0 if report["status"] == "passed" else 2
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        report["compiler_counters"] = {str(key): dict(value) for key, value in torch._dynamo.utils.counters.items()}
        write_json(args.output, report)
        print(f"Saved {args.output}: {report['status']}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
