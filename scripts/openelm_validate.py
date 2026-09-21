#!/usr/bin/env python3
"""Bounded native-checkpoint fidelity checks on a container GPU, with online W&B."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import (
    CHECKPOINT_FILENAME, OpenELMTokenizer, load_native_state_dict,
    sha256_file, state_dict_manifest, validate_prepared_manifest, write_json,
)
from cdrm.pretrained.openelm import OpenELMConfig, OpenELMModel
from cdrm.pretrained.reference import build_corenet_reference, reference_forward, verify_reference_sources
from experiment_tracking import OnlineTracker


PROMPTS = (
    "A program keeps a running total. It starts at zero, adds three, subtracts "
    "one, and then doubles the result. The final total is four. Explain each "
    "operation in order and check the calculation.",
    "def prefix_sums(values):\n    total = 0\n    result = []\n    for value in values:\n"
    "        total += value\n        result.append(total)\n    return result\n\n"
    "For the input [2, 3, -1], the output is [2, 5, 4].",
)


def require_container_gpu() -> dict:
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run with scripts/docker_shell.sh; host GPU execution is forbidden")
    if Path.cwd() != Path("/workspace/cdrm-w-latent"):
        raise RuntimeError("The container working directory must be /workspace/cdrm-w-latent")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; there is no CPU fallback")
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "nvidia_smi": smi}


def comparison(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict:
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    a, b = actual.detach().float(), expected.detach().float()
    difference = a - b
    norm = float(torch.linalg.vector_norm(b))
    error = float(torch.linalg.vector_norm(difference))
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"passed": finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            "finite": finite, "max_abs": float(difference.abs().max()),
            "relative_l2": error / max(norm, 1e-30), "reference_l2": norm,
            "difference_l2": error, "atol": atol, "rtol": rtol}


def compare_gradients(model, reference, *, atol: float, rtol: float) -> dict:
    actual = dict(model.named_parameters())
    expected = dict(reference.named_parameters())
    if actual.keys() != expected.keys():
        raise AssertionError("Native and adapter parameter ownership differ")
    rows, missing = {}, []
    squared_error = squared_reference = 0.0
    for name, parameter in actual.items():
        a, b = parameter.grad, expected[name].grad
        if a is None or b is None:
            missing.append(name)
            continue
        row = comparison(a, b, atol=atol, rtol=rtol)
        rows[name] = row
        squared_error += row["difference_l2"] ** 2
        squared_reference += row["reference_l2"] ** 2
    failed = [name for name, row in rows.items() if not row["passed"]]
    return {"passed": not missing and not failed, "missing": missing, "failed": failed,
            "parameter_tensors": len(actual), "tensors": rows,
            "relative_l2": math.sqrt(squared_error / max(squared_reference, 1e-60)),
            "max_abs": max((row["max_abs"] for row in rows.values()), default=0.0)}


def loss_of(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def autocast(precision: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16_mixed" else nullcontext()


def run_pair(model, reference, ids, *, precision, backward, backend="math"):
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    # Both sides use the same attention backend. This tests import/semantics,
    # rather than conflating a kernel change with the new architecture adapter.
    backend_context = sdpa_kernel(SDPBackend.MATH) if backend == "math" else nullcontext()
    with backend_context, autocast(precision):
        output = model(ids)
        original = reference_forward(reference, ids)
        loss = loss_of(output.logits, ids)
        original_loss = loss_of(original.logits, ids)
    # Source-equivalence tolerances, not a BF16-versus-FP32 training acceptance rule.
    atol, rtol = (3e-5, 3e-5) if precision == "fp32" else (2e-3, 2e-3)
    result = {"precision": precision, "backend": backend, "tokens": ids.numel(),
              "loss": float(loss.detach()), "reference_loss": float(original_loss.detach()),
              "logits": comparison(output.logits, original.logits, atol=atol, rtol=rtol),
              "hidden": comparison(output.last_hidden_state, original.last_hidden_state, atol=atol, rtol=rtol),
              "loss_comparison": comparison(loss, original_loss, atol=atol, rtol=rtol)}
    if backward:
        loss.backward()
        original_loss.backward()
        result["gradients"] = compare_gradients(
            model, reference, atol=2e-6 if precision == "fp32" else 2e-4,
            rtol=3e-5 if precision == "fp32" else 3e-3,
        )
    result["passed"] = all(result[key]["passed"] for key in ("logits", "hidden", "loss_comparison"))
    if backward:
        result["passed"] &= result["gradients"]["passed"]
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    return result


@torch.no_grad()
def cache_and_causality(model, reference, ids):
    ids = ids[:, :13]
    with sdpa_kernel(SDPBackend.MATH):
        full = model(ids).logits
        original_full = reference_forward(reference, ids).logits
        first = model(ids[:, :6], use_cache=True)
        second = model(ids[:, 6:10], past_key_values=first.past_key_values, use_cache=True)
        third = model(ids[:, 10:], past_key_values=second.past_key_values, use_cache=True)
        chunked = torch.cat((first.logits, second.logits, third.logits), dim=1)
        # Native CoreNet's cache supports prefill followed by single-token decode.
        # Do not use its uncorrected multi-token cached mask as a chunk oracle.
        native = reference_forward(reference, ids[:, :6], use_cache=True)
        native_outputs = [native.logits]
        for index in range(6, ids.shape[1]):
            native = reference_forward(reference, ids[:, index:index + 1],
                                       past_key_values=native.past_key_values, use_cache=True)
            native_outputs.append(native.logits)
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 17) % 32000
        changed_logits = model(changed).logits
    rows = {
        "chunked_vs_full": comparison(chunked, full, atol=1e-4, rtol=5e-5),
        "native_cached_vs_full": comparison(torch.cat(native_outputs, dim=1), original_full, atol=1e-4, rtol=5e-5),
        "adapter_cached_vs_native": comparison(chunked, torch.cat(native_outputs, dim=1), atol=1e-4, rtol=5e-5),
        "future_token_isolation": comparison(changed_logits[:, :7], full[:, :7], atol=0, rtol=0),
    }
    cache = third.past_key_values
    rows["native_kv_head_counts"] = all(
        k.shape[1] == count and v.shape[1] == count
        for (k, v), count in zip(cache.key_values, model.config.num_kv_heads)
    )
    rows["passed"] = all(row["passed"] for row in rows.values() if isinstance(row, dict)) and rows["native_kv_head_counts"]
    return rows


@torch.no_grad()
def attention_dispatch(model, ids):
    # Profile the ordinary native-compatible BF16 path separately from the math
    # equivalence checks. This is a bounded dispatch observation, not a benchmark.
    model.attention_backend = "sdpa"
    with autocast("bf16_mixed"):
        model(ids)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        with autocast("bf16_mixed"):
            fused = model(ids).logits
        torch.cuda.synchronize()
    with sdpa_kernel(SDPBackend.MATH), autocast("bf16_mixed"):
        math_logits = model(ids).logits
    events = sorted({event.key for event in profile.key_averages()
                     if any(word in event.key.lower() for word in ("scaled_dot_product", "flash", "efficient_attention"))})
    # Different kernels need not have identical rounding. This is a descriptive
    # comparison with no arbitrary acceptance budget; source parity on the same
    # default backend is checked separately, including its backward.
    difference = comparison(fused, math_logits, atol=0, rtol=0)
    for key in ("passed", "atol", "rtol"):
        difference.pop(key)
    difference.update(loss=float(loss_of(fused, ids)), math_loss=float(loss_of(math_logits, ids)))
    return {"events": events, "flash_observed": any("flash" in name.lower() for name in events),
            "sdpa_backend": "cudnn" if any("cudnn_attention" in event for event in events) else "see_events",
            "fused_vs_math": difference,
            "scope": "one ordinary BF16 forward; no throughput or recurrent claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--wandb-project", default="pretrained-fbt-rt-nextlat")
    parser.add_argument("--wandb-entity", default="taylorbollman")
    args = parser.parse_args()
    if args.max_length < 16:
        parser.error("--max-length must be at least16")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = require_container_gpu()
    torch.set_num_threads(8)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    report = {"schema": "openelm-import-validation-v1", "status": "running", "runtime": runtime,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "native1.1B300k ordinary import; no recurrence or training run",
              "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path)
                                for path in (ROOT / "cdrm/pretrained/openelm.py", ROOT / "cdrm/pretrained/reference.py",
                                             ROOT / "cdrm/pretrained/artifacts.py", Path(__file__))}}
    tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                            output_dir=args.output_dir, group="openelm-import",
                            name="openelm-1.1b-native300k-import")
    started = time.monotonic()
    try:
        config = OpenELMConfig.native_1_1b()
        report["config"] = config.to_dict()
        checkpoint = args.artifacts / "checkpoint" / CHECKPOINT_FILENAME
        artifact_manifest = validate_prepared_manifest(args.artifacts)
        provenance = artifact_manifest["checkpoint"]
        report["checkpoint"] = provenance
        report["artifacts_manifest"] = artifact_manifest
        report["native_reference_sources"] = verify_reference_sources()
        tokenizer = OpenELMTokenizer(args.artifacts / "tokenizer/tokenizer.model")
        sequences = [tokenizer.encode(prompt)[:args.max_length] for prompt in PROMPTS]
        report["fixtures"] = {"texts": PROMPTS, "token_ids": sequences,
                              "tokenizer_sha256": sha256_file(args.artifacts / "tokenizer/tokenizer.model")}
        write_json(args.output_dir / "report.json", report)
        tracker.start({"scope": report["scope"], "config": config.to_dict(),
                       "checkpoint_sha256": provenance["sha256"], "max_length": args.max_length})
        report["wandb"] = tracker.record
        print(json.dumps({"stage": "load", "wandb": tracker.record["run_url"]}), flush=True)
        model = OpenELMModel(config, device="cuda", dtype=torch.float32).eval()
        reference = build_corenet_reference(device="cuda", dtype=torch.float32).eval()
        state = load_native_state_dict(checkpoint,
                                       expected_shapes={name: tuple(tensor.shape) for name, tensor in model.state_dict().items()})
        model.load_state_dict(state, strict=True)
        reference.load_state_dict(state, strict=True)
        report["state"] = state_dict_manifest(state)
        del state
        parameters = list(model.parameters())
        report["tying"] = {"same_parameter": model.readout_weight is model.token_embeddings.weight,
                           "optimizer_ownership_count": sum(p is model.readout_weight for p in parameters),
                           "state_embedding_keys": [name for name in model.state_dict() if "embedding" in name or "classifier" in name]}
        if not report["tying"]["same_parameter"] or report["tying"]["optimizer_ownership_count"] != 1:
            raise AssertionError("Tied readout ownership was lost")
        cases = []
        for precision, backend in (("fp32", "math"), ("bf16_mixed", "math"), ("bf16_mixed", "default")):
            for index, sequence in enumerate(sequences):
                ids = torch.tensor([sequence], dtype=torch.long, device="cuda")
                result = run_pair(model, reference, ids, precision=precision, backward=index == 0, backend=backend)
                result["fixture"] = index
                cases.append(result)
                report["cases"] = cases
                write_json(args.output_dir / "report.json", report)
                metrics = {"validation/loss": result["loss"],
                           "validation/logit_relative_l2": result["logits"]["relative_l2"],
                           "validation/passed": result["passed"]}
                if "gradients" in result:
                    metrics["validation/gradient_relative_l2"] = result["gradients"]["relative_l2"]
                tracker.log(metrics, step=len(cases))
                print(json.dumps({"stage": "equivalence", "precision": precision, "backend": backend, "fixture": index,
                                  "passed": result["passed"], "loss": result["loss"]}), flush=True)
                if not result["passed"]:
                    raise AssertionError(f"Native equivalence failed: {precision}, {backend}, fixture{index}")
        ids = torch.tensor([sequences[0]], dtype=torch.long, device="cuda")
        report["cache_causality"] = cache_and_causality(model, reference, ids)
        if not report["cache_causality"]["passed"]:
            raise AssertionError("Cache or causality check failed")
        report["attention_dispatch"] = attention_dispatch(model, ids)
        if not report["attention_dispatch"]["fused_vs_math"]["finite"]:
            raise AssertionError("The ordinary fused-attention path produced nonfinite outputs")
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["memory_scope"] = "paired reference+adapter validation with gradients; not single-model training"
        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = "passed"
        tracker.summary({"validation/status": "passed", "validation/parameter_count": sum(p.numel() for p in parameters),
                         "validation/flash_observed": report["attention_dispatch"]["flash_observed"],
                         "validation/peak_allocated_gib": report["peak_allocated_gib"]})
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report["wandb"] = tracker.record
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(args.output_dir / "report.json"),
                      "wandb": tracker.record["run_url"]}), flush=True)


if __name__ == "__main__":
    main()
