#!/usr/bin/env python3
"""New-lineage T128 mixed-precision OPS training, cold recovery and masked-CE timing.

Run serially through scripts/docker_shell.sh. Original Stage B artifacts are inputs
only. The 100-update stop preserves the original 2000-update learning-rate horizon.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.synthetic.common import SyntheticBatch
from cdrm.synthetic.experiment import generate_training_batch
from r3_validation_metrics import compare_tensors
from stage_a_common import (configure_compiled_helpers, provenance, require_cuda_container,
                            restore_rng, rng_state, seed_all, tensor_bytes, unique_parameters)
from stage_b_train import (aligned_ce_sum, append_jsonl, atomic_json, compiler_audit,
                           file_digest, json_digest, learning_rate_at_update,
                           load_fixtures, load_training_arrays, source_hashes)

FORMAT = "r3-mixed-operational-v1"
SOURCE_CHANGES = {"recurrent-transformer/olmo/model.py", "recurrent-transformer/olmo/config.py"}
PROFILE_POLICY = {"fp32": "legacy", "bf16": "bf16_fp32_state"}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def state_digest(value):
    """Stable tensor-aware digest, including RNG arrays and nested optimizer state."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"array:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for entry in item:
                visit(entry)
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())
    visit(value)
    return digest.hexdigest()


def retained_save(path, payload):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(cpu_tree(payload), temporary)
    temporary.rename(path)
    return {"path": str(path), "sha256": file_digest(path), "bytes": path.stat().st_size}


def profile_record(name):
    return {"name": name, "recurrent_precision_policy": PROFILE_POLICY[name],
            "parameters": "torch.float32", "parameter_gradients": "torch.float32",
            "optimizer_moments": "torch.float32", "config_precision": None,
            "forward_autocast": name == "bf16", "autocast_dtype": "torch.bfloat16" if name == "bf16" else None,
            "backward_outer_autocast": False, "grad_scaler": False,
            "residual_policy": "FP32", "ce_reduction": "FP32 aligned answer mean",
            "tf32": False, "sdpa": "deterministic_math", "whole_model_compile": False,
            "cuda_graphs": False, "accumulation": False}


def execution_contract():
    """Runtime math settings that must match before an exact recovery attempt."""
    return {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "bf16_gemm_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_gemm_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
            "flash_sdpa_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_sdpa_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdpa_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def measured_compiler_audit(before, after):
    before_graphs = before["counters"].get("stats", {}).get("unique_graphs", 0)
    after_graphs = after["counters"].get("stats", {}).get("unique_graphs", 0)
    return {"unique_graphs_before": before_graphs, "unique_graphs_after": after_graphs,
            "new_graphs_during_measurement": after_graphs - before_graphs,
            "steady_compilation_free": before_graphs == after_graphs}


def effective_precision(model, optimizer, *, require_gradients):
    def check(tensors, label, required_dtype=True):
        entries = list(tensors)
        if required_dtype and any(value.dtype != torch.float32 for _, value in entries):
            raise AssertionError(f"{label} must stay FP32")
        if entries and not torch.stack([torch.isfinite(value).all() for _, value in entries]).all().item():
            raise FloatingPointError(f"Nonfinite {label}")
        return {"tensors": len(entries), "dtypes": sorted({str(value.dtype) for _, value in entries}),
                "finite": True}
    parameters = list(model.named_parameters())
    missing = [name for name, value in parameters if value.requires_grad and value.grad is None]
    if require_gradients and missing:
        raise AssertionError(f"Missing intended gradients: {missing}")
    moments = [(f"{name}/{key}", value) for name, parameter in parameters
               for key, value in optimizer.state.get(parameter, {}).items()
               if key in {"exp_avg", "exp_avg_sq"}]
    steps = [(name, state["step"]) for name, state in enumerate(optimizer.state.values()) if "step" in state]
    return {"parameters": check(parameters, "parameters"),
            "gradients": check(((n, p.grad) for n, p in parameters if p.grad is not None), "gradients"),
            "moments": check(moments, "moments"), "step_counters": check(steps, "optimizer step counters", False),
            "missing_gradients": missing}


def norm(values):
    return float(torch.stack([value.detach().double().square().sum() for value in values]).sum().sqrt().item())


def update(model, optimizer, ids, labels, settings, completed, profile, *, monitor):
    model.train()
    rate = learning_rate_at_update(completed, settings)
    for group in optimizer.param_groups:
        group["lr"] = rate
    before = [p.detach().clone() for p in model.parameters()] if monitor else None
    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=profile == "bf16"):
        logits = model(ids).logits
        summed, count = aligned_ce_sum(logits, labels)
        loss = summed / count
    expected_logits_dtype = torch.bfloat16 if profile == "bf16" else torch.float32
    if logits.dtype != expected_logits_dtype:
        raise AssertionError(f"Expected observed {expected_logits_dtype} logits under {profile}")
    loss.backward()
    if any(p.grad is None or p.grad.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise AssertionError("Every intended trainable parameter must receive an FP32 gradient")
    signals = {}
    if monitor and model.config.recurrent_layers:
        for name in ("transformer.blocks.3.q_proj.weight", "transformer.blocks.3.kv_proj.weight"):
            gradient = dict(model.named_parameters())[name].grad
            signals[name] = {"l2": norm([gradient]), "nonzero": bool(torch.count_nonzero(gradient).item()),
                             "dtype": str(gradient.dtype)}
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"],
                                                       error_if_nonfinite=True).item())
    optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    row = {"update": completed + 1, "loss": float(loss.detach().item()), "learning_rate": rate,
           "gradient_norm_before_clipping": gradient_norm,
           "clip_coefficient": min(1., settings["gradient_clip"] / (gradient_norm + 1e-6)),
           "clipped": gradient_norm > settings["gradient_clip"],
           "update_seconds": seconds, "input_tokens": ids.numel(), "supervised_targets": count,
           "logits_dtype": str(logits.dtype), "ce_dtype": str(loss.dtype)}
    if not math.isfinite(row["loss"]) or loss.dtype != torch.float32:
        raise FloatingPointError("Masked CE must be finite FP32")
    if monitor:
        row.update(update_l2=norm([p.detach().double() - old.double() for p, old in zip(model.parameters(), before)]),
                   block3_projection_gradient_signals=signals,
                   precision=effective_precision(model, optimizer, require_gradients=True))
        mask = labels != -100
        row["answer_accuracy"] = float((logits.detach()[mask].argmax(-1) == labels[mask]).float().mean().item())
    return row


@torch.no_grad()
def evaluate(model, dev, profile):
    model.eval()
    total = count = correct = 0
    for begin in range(0, len(dev.input_ids), 64):
        ids = torch.as_tensor(dev.input_ids[begin:begin + 64], device="cuda")
        labels = torch.as_tensor(dev.labels[begin:begin + 64], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=profile == "bf16"):
            logits = model(ids).logits
            summed, n = aligned_ce_sum(logits, labels)
        if not torch.isfinite(summed).item() or summed.dtype != torch.float32:
            raise FloatingPointError("Development CE must be finite FP32")
        total += summed.item()
        count += n
        mask = labels != -100
        correct += (logits[mask].argmax(-1) == labels[mask]).sum().item()
    return {"answer_ce": total / count, "answer_accuracy": correct / count,
            "supervised_targets": count, "examples": len(dev.input_ids), "fixture_sha256": dev.sha256}


def compare_recovery(reference, actual):
    differences = {}

    def compare(left, right, path):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if state_digest(left) != state_digest(right):
                differences[path] = compare_tensors(left, right)
                differences[path]["bitwise_equal"] = False
        elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if state_digest(left) != state_digest(right):
                differences[path] = {"equal": False, "bitwise_equal": False}
        elif isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
            for key in left:
                compare(left[key], right[key], f"{path}/{key}")
        elif isinstance(left, (list, tuple)) and isinstance(right, type(left)) and len(left) == len(right):
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}/{index}")
        elif isinstance(left, (dict, list, tuple, torch.Tensor, np.ndarray)):
            differences[path] = {"equal": False, "reason": "Container/type/structure mismatch"}
        elif type(left) is not type(right) or state_digest(left) != state_digest(right):
            differences[path] = {"reference": str(left), "actual": str(right), "equal": False}
    for key in ("identity", "model_config", "profile", "model", "optimizer", "rng", "completed_updates",
                "data_offset_examples", "next_data_sha256", "schedule"):
        compare(reference[key], actual[key], key)
    def numerical_history(payload):
        return [{k: v for k, v in row.items() if k != "update_seconds"} for row in payload["history"]]
    compare(numerical_history(reference), numerical_history(actual), "history")
    compare(reference["development"], actual["development"], "development")
    return {"bitwise_state_and_metrics_equal": not differences, "differences": differences,
            "policy": "No numerical tolerance replaces the exact-resume result. Tensor differences retain historical diagnostic metrics.",
            "excluded": ["wall times", "compiler histories/counters", "paths", "checkpoint bytes"]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "resume", "benchmark"), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILE_POLICY), required=True)
    parser.add_argument("--topology", choices=("seq", "r3"), default="r3")
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-final", type=Path)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--midpoint", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.stop != 100 or args.midpoint != 50:
        parser.error("This bounded protocol fixes stop=100 and midpoint=50")
    if args.mode != "benchmark" and args.topology != "r3":
        parser.error("The paired short trajectory is R3 only")
    if (args.mode == "resume") != (args.checkpoint is not None):
        parser.error("Only resume requires --checkpoint")
    if args.reference_final and args.mode != "resume":
        parser.error("--reference-final is only for the independent resume command")
    if args.warmup < 2 or args.steps < 1 or args.warmup + args.steps > 100:
        parser.error("Bounded benchmark needs 2<=warmup and 1<=steps and total<=100")
    return args


def run(args, report):
    hardware = require_cuda_container()
    torch.set_float32_matmul_precision("highest")
    seed_all(0, deterministic=True)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = args.topology == "r3"
    configure_compiled_helpers(compiled)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    report["provenance"] = provenance(hardware)
    report["execution_contract"] = execution_contract()
    parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    expected_layers = [3] if compiled else []
    if (parent.get("format") != "stage-b-training-v1" or parent["completed_updates"] != 0
            or parent["model_config"]["recurrent_layers"] != expected_layers
            or parent["identity"]["task"] != "mqar" or parent["optimizer"]["state"]):
        raise ValueError("Require corresponding retained MQAR initialization with empty Adam state")
    if parent["identity_sha256"] != json_digest(parent["identity"]):
        raise ValueError("Parent identity digest mismatch")
    expected_model = {"d_model": 256, "n_heads": 4, "n_kv_heads": 4, "n_layers": 12,
                      "mlp_hidden_size": 1024, "vocab_size": 1024, "norm_after": False,
                      "recurrent_write_rho": 1.0, "bwd_mlp_chunks": 4}
    if any(parent["model_config"].get(key) != value for key, value in expected_model.items()):
        raise ValueError("Parent model is outside the fixed D256/H4/12-block/rho1 profile")
    plan = copy.deepcopy(parent["identity"]["plan"])
    settings = copy.deepcopy(parent["schedule"])
    if (settings["updates"] != 2000 or settings["warmup_updates"] != 100
            or settings["global_batch"] != 64 or settings["microbatch"] != 64):
        raise ValueError("Expected authoritative 2000-update schedule and physical/global B64")
    changed = {}
    for name, expected in parent["identity"]["source_sha256"].items():
        actual = file_digest(Path(name))
        if actual != expected:
            if name not in SOURCE_CHANGES:
                raise ValueError(f"Unexpected change to original training/data source: {name}")
            changed[name] = {"parent_sha256": expected, "current_sha256": actual,
                             "reason": "Explicit new mixed-precision model/config implementation lineage"}
    fixture_path = args.fixtures_dir / "mqar" / "manifest.json"
    if file_digest(fixture_path) != parent["identity"]["fixture_manifest_sha256"]:
        raise ValueError("Fixture manifest differs from retained parent")
    fixture_manifest, conditions = load_fixtures(args.fixtures_dir, "mqar", plan, "dev")
    arrays = load_training_arrays(fixture_manifest, settings)
    dev = conditions["iid"].take(slice(0, 256))
    if dev.input_ids.shape != (256, 128):
        raise ValueError("Expected fixed first256 IID T128 development examples")
    batches = []
    for index in range(101):
        batch = generate_training_batch(plan, "mqar", index)
        frozen = SyntheticBatch(arrays[0][index * 64:(index + 1) * 64],
                                arrays[1][index * 64:(index + 1) * 64], [{}] * 64)
        if batch.input_ids.shape != (64, 128) or batch.sha256 != frozen.sha256:
            raise ValueError(f"Generated stream differs from frozen B64/T128 stream at update {index}")
        batches.append(batch)
    if parent["next_data_sha256"] != batches[0].sha256:
        raise ValueError("Parent next-data digest mismatch")
    del arrays, conditions
    profile = profile_record(args.profile)
    current_sources = source_hashes()
    current_sources[str(Path(__file__).relative_to(Path.cwd()))] = file_digest(Path(__file__))
    current_sources["scripts/r3_validation_metrics.py"] = file_digest(Path("scripts/r3_validation_metrics.py"))
    identity = {"schema": FORMAT, "parent_checkpoint_sha256": file_digest(args.parent_checkpoint),
                "parent_identity_sha256": parent["identity_sha256"], "source_sha256": current_sources,
                "declared_parent_source_changes": changed, "topology": args.topology,
                "profile": profile, "execution_contract": report["execution_contract"],
                "schedule": settings, "stop": 100, "midpoint": 50,
                "fixture_manifest_sha256": file_digest(fixture_path), "dev_sha256": dev.sha256,
                "ordered_training_sha256": [batch.sha256 for batch in batches], "seed": plan["seed"]}
    from olmo.config import ModelConfig
    from olmo.model import OLMo
    raw = copy.deepcopy(parent["model_config"])
    raw.update(init_device="cpu", precision=None, reference_eager=False, recurrent_backend="tiled")
    if "recurrent_precision_policy" in {field.name for field in dataclasses.fields(ModelConfig)}:
        raw["recurrent_precision_policy"] = profile["recurrent_precision_policy"]
    elif args.profile == "bf16":
        raise RuntimeError("Candidate recurrent_precision_policy field is not available yet")
    model = OLMo(ModelConfig(**raw)).to(device="cuda", dtype=torch.float32)
    model.load_state_dict(parent["model"], strict=True)
    parameters = unique_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], betas=tuple(settings["betas"]),
                                  eps=settings["eps"], weight_decay=settings["weight_decay"], foreach=False, fused=False)
    restore_rng(parent["rng"])
    report.update(identity=identity, identity_sha256=json_digest(identity), profile=profile,
                  model_config=dataclasses.asdict(model.config), parent_checkpoint=str(args.parent_checkpoint),
                  parent_import_policy="Weights-only new diagnostic initialization; fresh Adam; never Stage B exact resume",
                  parameter_count=sum(p.numel() for p in parameters), parameter_bytes=tensor_bytes(parameters),
                  effective_precision_at_initialization=effective_precision(model, optimizer, require_gradients=False),
                  gradient_signal_scope="Block3 Q/KV projection parameter gradients; supporting trajectory signal, not an isolated persistent-write proof",
                  data_policy="Exact counter-generated Stage B train batches checked against frozen arrays; first256 IID dev; no test evaluation")
    initial = {"model_sha256": state_digest(model.state_dict()), "optimizer_sha256": state_digest(optimizer.state_dict()),
               "rng_sha256": state_digest(rng_state())}
    if initial["model_sha256"] != state_digest(parent["model"]):
        raise AssertionError("Diagnostic initialization differs from retained parent weights")
    report["initial_state"] = initial
    atomic_json(args.output_dir / "resolved-config.json", {"identity": identity, "model": report["model_config"],
                                                          "profile": profile, "settings": settings})
    if args.mode == "benchmark":
        batch = batches[0]
        ids = torch.as_tensor(batch.input_ids, device="cuda")
        labels = torch.as_tensor(batch.labels, device="cuda")
        report.update(benchmark_data_sha256=batch.sha256,
                      benchmark_scope="Synchronized zero_grad, forward, aligned mean CE, backward, gradient clipping, AdamW; prepared GPU data; no trajectory-monitor copies/reductions")
        torch.cuda.synchronize()
        started = time.perf_counter()
        warmup = [update(model, optimizer, ids, labels, settings, index, args.profile, monitor=False)
                  for index in range(args.warmup)]
        torch.cuda.synchronize()
        report["warmup_seconds_including_compilation"] = time.perf_counter() - started
        report["warmup"] = warmup
        report["effective_precision_after_warmup"] = effective_precision(model, optimizer, require_gradients=True)
        report["compiler_after_warmup"] = compiler_audit(compiled)
        report["optimizer_state_bytes"] = tensor_bytes(optimizer.state)
        report["startup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["startup_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        measured = [update(model, optimizer, ids, labels, settings, args.warmup + index, args.profile, monitor=False)
                    for index in range(args.steps)]
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        times = [row["update_seconds"] for row in measured]
        report["measured"] = measured
        report["compiler_after_measurement"] = compiler_audit(compiled)
        report["measurement_compilation"] = measured_compiler_audit(report["compiler_after_warmup"],
                                                                    report["compiler_after_measurement"])
        if not report["measurement_compilation"]["steady_compilation_free"]:
            raise RuntimeError("Additional graphs compiled during measured updates; these are not steady-state timings")
        report["summary"] = {"update_seconds_mean": statistics.mean(times), "update_seconds_median": statistics.median(times),
                             "input_tokens_per_second": args.steps * ids.numel() / sum(times),
                             "scored_answers_per_second": sum(row["supervised_targets"] for row in measured) / sum(times),
                             "memory_scope": "Steady updates after initialized Adam; reserved includes warmup allocator pool"}
        report["final_precision"] = effective_precision(model, optimizer, require_gradients=True)
    else:
        history, development, completed = [], {}, 0
        if args.mode == "resume":
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if (payload.get("format") != FORMAT or payload["identity"] != identity
                    or payload["identity_sha256"] != json_digest(identity) or payload["completed_updates"] != 50
                    or payload["model_config"] != report["model_config"] or payload["profile"] != profile
                    or payload["schedule"] != settings or payload["data_offset_examples"] != 50 * 64
                    or payload["next_data_sha256"] != batches[50].sha256):
                raise ValueError("Midpoint recovery identity/config/profile/source/data mismatch; update-zero resume is prohibited")
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            restore_rng(payload["rng"])
            for name, current in (("model", model.state_dict()), ("optimizer", optimizer.state_dict()), ("rng", rng_state())):
                if state_digest(current) != state_digest(payload[name]):
                    raise AssertionError(f"{name} was not restored exactly before the next update")
            completed = payload["completed_updates"]
            history, development = copy.deepcopy(payload["history"]), copy.deepcopy(payload["development"])
            report["recovery"] = {"checkpoint": str(args.checkpoint), "sha256": file_digest(args.checkpoint),
                                  "loaded_state_exact": True, "cold_process": True,
                                  "warmup_updates": 0, "repeated_midpoint_evaluation": False}

        def snapshot():
            return {"format": FORMAT, "identity": identity, "identity_sha256": json_digest(identity),
                    "model_config": report["model_config"], "profile": profile,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
                    "schedule": settings, "completed_updates": completed, "data_offset_examples": completed * 64,
                    "next_data_sha256": batches[completed].sha256, "history": history, "development": development,
                    "optimizer_policy": "exact_new_lineage_resume", "evidence_class": "OPS"}

        report["checkpoints"] = {}
        if completed == 0:
            report["checkpoints"]["init"] = retained_save(args.output_dir / "init.pt", snapshot())
            development["0"] = evaluate(model, dev, args.profile)
        for index in range(completed, 100):
            batch = batches[index]
            ids = torch.as_tensor(batch.input_ids, device="cuda")
            labels = torch.as_tensor(batch.labels, device="cuda")
            row = update(model, optimizer, ids, labels, settings, index, args.profile, monitor=True)
            row["data_sha256"] = batch.sha256
            history.append(row)
            append_jsonl(args.output_dir / "learning-curve.jsonl", row)
            completed = index + 1
            if completed in {50, 100}:
                development[str(completed)] = evaluate(model, dev, args.profile)
                record = retained_save(args.output_dir / f"u{completed:04d}.pt", snapshot())
                report["checkpoints"][str(completed)] = record
            if completed % 10 == 0:
                print(json.dumps({"update": completed, "loss": row["loss"], "profile": args.profile}), flush=True)
        report.update(development=development, completed_updates=completed,
                      clipping_count=sum(row["clipped"] for row in history),
                      final_precision=effective_precision(model, optimizer, require_gradients=True),
                      optimizer_state_bytes=tensor_bytes(optimizer.state),
                      training_seconds=sum(row["update_seconds"] for row in history),
                      history=history)
        if args.reference_final:
            reference = torch.load(args.reference_final, map_location="cpu", weights_only=False)
            report["recovery_comparison"] = compare_recovery(reference, cpu_tree(snapshot()))
            report["recovery_comparison"]["reference_sha256"] = file_digest(args.reference_final)
            atomic_json(args.output_dir / "recovery-comparison.json", report["recovery_comparison"])
    report["compiler_audit"] = compiler_audit(compiled)
    if source_hashes() != {key: value for key, value in current_sources.items()
                          if key not in {str(Path(__file__).relative_to(Path.cwd())), "scripts/r3_validation_metrics.py"}}:
        raise RuntimeError("Runtime sources changed during execution")
    report["status"] = "complete"


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    forbidden = [Path(".runtime/stage-b").resolve(), Path(".runtime/r3-backward").resolve()]
    if any(output == path or path in output.parents for path in forbidden):
        raise ValueError("Original Stage B and FP32 validation lineages are immutable")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory, including for recovery")
    output.mkdir(parents=True, exist_ok=True)
    report = {"format": FORMAT, "evidence_class": "OPS", "status": "running",
              "command": shlex.join([sys.executable, *sys.argv]),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    started = time.perf_counter()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            run(args, report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["invocation_seconds"] = time.perf_counter() - started
        atomic_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
