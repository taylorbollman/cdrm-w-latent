#!/usr/bin/env python3
"""Bounded whole-model shifted-CE benchmark, one GPU, no external data or uploads.

From the project container:
    python scripts/benchmark_model.py --topology r3 --output .runtime/stage-a/r3.json
    python scripts/benchmark_model.py --preset full --precision bf16 --topology seq \
        --microbatch 1 --output .runtime/stage-a/full-seq-b1.json

The full preset specifies the actual 12x1024, H16, MLP4096, T512 dimensions.
It uses synthetic IDs and cannot measure held-out language-model quality.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch

from stage_a_common import (adamw, configure_compiled_helpers, experiment_manifest, model_config, provenance,
                            require_cuda_container, seed_all, tensor_bytes, unique_parameters,
                            update, validate_compiler_execution, write_json)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--topology", choices=("seq", "r3"), default="seq")
    parser.add_argument("--backend", choices=("naive", "tiled"), default="naive")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--compiled-helpers", action="store_true",
                        help="Opt into upstream helper compilation; warmup includes compile cost.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("microbatch", "accumulation", "steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.warmup < 2:
        parser.error("--warmup must be at least 2 to initialize and exercise optimizer state")
    if args.sequence_length is not None and args.sequence_length < 2:
        parser.error("Training CE needs T>=2; T=1 is covered by NUM output-only tests")
    if not 0 <= args.rho <= 1:
        parser.error("--rho must be in [0,1]")
    if args.topology == "seq" and args.rho != 1:
        parser.error("--rho is unused for SEQ; keep the default 1")
    if args.topology == "r3" and args.backend == "tiled" and args.rho != 1:
        parser.error("Tiled R3 currently supports rho=1 only")
    return args


def run(args, report):
    hardware = require_cuda_container()
    seed_all(args.seed)
    configure_compiled_helpers(args.compiled_helpers)
    report["provenance"] = provenance(hardware)
    config = model_config(args.preset, args.topology, args.backend, args.rho,
                          sequence_length=args.sequence_length, compiled_helpers=args.compiled_helpers)
    report["manifest"] = experiment_manifest(config, evidence="OPS", budget={
        "warmup_updates": args.warmup, "measured_updates": args.steps,
        "microbatch": args.microbatch, "accumulation": args.accumulation,
        "global_sequences_per_update": args.microbatch * args.accumulation,
        "precision": args.precision, "seed": args.seed,
    })
    from olmo.model import OLMo
    from olmo.checkpoint_conversion import convert_model

    report["phase"] = "initialization"
    if args.topology == "r3":
        source = OLMo(model_config(args.preset, "seq", args.backend, 1.0,
                                  sequence_length=args.sequence_length,
                                  compiled_helpers=args.compiled_helpers))
        model = OLMo(config)
        conversion = convert_model(source, model)
        report["conversion"] = dataclasses.asdict(conversion)
        del source
    else:
        model = OLMo(config)
    model = model.cuda().train()
    optimizer = adamw(model)
    params = unique_parameters(model)
    report["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                            "trainable": sum(p.numel() for p in params),
                            "parameter_bytes": tensor_bytes(params),
                            "parameter_dtype": str(params[0].dtype),
                            "forward_autocast_dtype": "torch.bfloat16" if args.precision == "bf16" else None}
    report["optimizer_config"] = {"name": "AdamW", "learning_rate": 1e-3,
                                   "betas": [0.9, 0.95], "eps": 1e-8,
                                   "weight_decay": 0.0, "foreach": False, "fused": False}
    # All microbatches are prepared ahead of timing. No tokenizer/data-loader time is claimed.
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    batches = [(torch.randint(2, config.vocab_size,
                             (args.microbatch, config.max_sequence_length), generator=generator).cuda(), None)
               for _ in range(args.accumulation)]
    report["phase"] = "warmup"
    torch.cuda.synchronize()
    started = time.perf_counter()
    report["warmup"] = []
    for index in range(args.warmup):
        result = update(model, optimizer, batches, precision=args.precision, measure=True)
        report["warmup"].append(result)
        print(f"warmup {index + 1}/{args.warmup}: CE={result['loss']:.6f}, "
              f"update={result['update_seconds']:.4f}s", flush=True)
    torch.cuda.synchronize()
    report["warmup_wall_seconds_including_compilation"] = time.perf_counter() - started
    report["optimizer_state_bytes"] = tensor_bytes(optimizer.state)
    if not optimizer.state:
        raise AssertionError("Warmup did not initialize optimizer state.")
    report["phase"] = "measurement"
    torch.cuda.reset_peak_memory_stats()
    report["updates"] = []
    for index in range(args.steps):
        result = update(model, optimizer, batches, precision=args.precision, measure=True)
        report["updates"].append(result)
        print(f"measured {index + 1}/{args.steps}: CE={result['loss']:.6f}, "
              f"update={result['update_seconds']:.4f}s", flush=True)
    times = [entry["update_seconds"] for entry in report["updates"]]
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if not torch.stack([torch.isfinite(param).all() for param in params]).all().item():
        raise FloatingPointError("The final optimizer update produced nonfinite parameters.")
    report["summary"] = {
        "update_seconds_mean": statistics.mean(times),
        "update_seconds_median": statistics.median(times),
        "forward_seconds_mean": statistics.mean(entry["forward_seconds"] for entry in report["updates"]),
        "input_tokens_per_second": sum(entry["input_tokens"] for entry in report["updates"]) / sum(times),
        "supervised_target_tokens_per_second": sum(entry["valid_target_tokens"] for entry in report["updates"]) / sum(times),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "memory_scope": "steady updates with initialized AdamW; reserved includes warmup allocator pool",
        "update_timing_scope": "zero_grad, forward, shifted CE, backward, AdamW; synchronized wall time; excludes prepared-data transfer",
        "forward_timing_scope": "CUDA event intervals summed across accumulation microbatches; excludes CE",
        "input_tokens_total": sum(entry["input_tokens"] for entry in report["updates"]),
        "supervised_target_tokens_total": sum(entry["valid_target_tokens"] for entry in report["updates"]),
        "final_parameters_finite": True,
    }
    validate_compiler_execution(args.compiled_helpers)
    report.update(status="completed", phase="complete")


def main() -> int:
    args = parse_args()
    report = {"schema": "stage-a-benchmark-v1", "status": "running",
              "arguments": {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()}}
    code = 0
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "failed",
                      error_type=type(exc).__name__, error=str(exc))
        if Path("/.dockerenv").exists() and torch.cuda.is_initialized():
            report["failure_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        traceback.print_exc()
        code = 2 if report["status"] == "oom" else 1
    finally:
        report["compiler_counters"] = {str(key): dict(value)
                                       for key, value in torch._dynamo.utils.counters.items()}
        write_json(args.output, report)
        print(f"Benchmark {report['status']}: {args.output}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
