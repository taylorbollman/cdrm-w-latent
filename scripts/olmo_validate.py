#!/usr/bin/env python3
"""Bounded original OLMo-1B checkpoint fidelity, on a container GPU with W&B."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
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

from cdrm.pretrained.artifacts import sha256_file, state_dict_manifest, write_json
from cdrm.pretrained.olmo_artifacts import (
    load_native_state_dict, validate_prepared_manifest, load_native_tokenizer,
)
from cdrm.pretrained.olmo import OLMoConfig, OLMoForCausalLM
from cdrm.pretrained.olmo_reference import (
    build_olmo_reference, olmo_reference_forward, verify_olmo_reference_sources,
)
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_validation import (
    PROMPTS, require_container_gpu, comparison, compare_gradients, loss_of, autocast,
    snapshot, compare_snapshot,
)

SOURCE_FILES = (
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_reference.py",
    "cdrm/pretrained/olmo_artifacts.py", "cdrm/pretrained/artifacts.py",
    "scripts/olmo_validation.py", "scripts/olmo_validate.py",
    "scripts/experiment_tracking.py",
)


def run_pair(model, reference, ids, *, precision, backward, backend="math", fp32=None):
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    # Both sides use the same attention backend. This tests import/semantics,
    # rather than conflating a kernel change with the new architecture adapter.
    backend_context = sdpa_kernel(SDPBackend.MATH) if backend == "math" else nullcontext()
    with backend_context, autocast(precision):
        x = model.transformer.wte(ids)
        original_x = reference.transformer.wte(ids)
        if backward:
            x.retain_grad()
            original_x.retain_grad()
        output = model(inputs_embeds=x)
        original = olmo_reference_forward(reference, inputs_embeds=original_x)
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
        result["input_gradient"] = comparison(
            x.grad, original_x.grad, atol=2e-6 if precision == "fp32" else 2e-4,
            rtol=3e-5 if precision == "fp32" else 3e-3,
        )
    result["passed"] = all(result[key]["passed"] for key in ("logits", "hidden", "loss_comparison"))
    if backward:
        result["passed"] &= result["gradients"]["passed"] and result["input_gradient"]["passed"]
    saved = snapshot(model, output, x.grad, loss) if backward and precision == "fp32" else None
    if fp32 is not None and backward:
        result["versus_fp32"] = compare_snapshot(model, output, x.grad, loss, fp32)
        result["passed"] &= result["versus_fp32"]["finite"]
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    return result, saved


@torch.no_grad()
def cache_and_causality(model, reference, ids):
    ids = ids[:, :13]
    with sdpa_kernel(SDPBackend.MATH):
        full = model(ids).logits
        original_full = olmo_reference_forward(reference, ids).logits
        first = model(ids[:, :6], use_cache=True)
        second = model(ids[:, 6:10], past_key_values=first.past_key_values, use_cache=True)
        third = model(ids[:, 10:], past_key_values=second.past_key_values, use_cache=True)
        chunked = torch.cat((first.logits, second.logits, third.logits), dim=1)
        # Check native prefill followed by single-token decode, independently of
        # our arbitrary-size cached chunk interface.
        native = olmo_reference_forward(reference, ids[:, :6], use_cache=True)
        native_outputs = [native.logits]
        for index in range(6, ids.shape[1]):
            native = olmo_reference_forward(reference, ids[:, index:index + 1],
                                       past_key_values=native.past_key_values, use_cache=True)
            native_outputs.append(native.logits)
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 17) % 50280
        changed_logits = model(changed).logits
    rows = {
        "chunked_vs_full": comparison(chunked, full, atol=1e-4, rtol=5e-5),
        "native_cached_vs_full": comparison(torch.cat(native_outputs, dim=1), original_full, atol=1e-4, rtol=5e-5),
        "adapter_cached_vs_native": comparison(chunked, torch.cat(native_outputs, dim=1), atol=1e-4, rtol=5e-5),
        "future_token_isolation": comparison(changed_logits[:, :7], full[:, :7], atol=0, rtol=0),
    }
    cache = third.past_key_values
    rows["native_kv_head_counts"] = all(
        k.shape[1] == model.config.n_heads and v.shape[1] == model.config.n_heads
        for (k, v) in cache.key_values
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
    report = {"schema": "olmo-import-validation-v1", "status": "running", "runtime": runtime,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "original OLMo-1B step60000 (~252B), ordinary import; no recurrence or training run",
              "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCE_FILES}}
    tracker = OnlineTracker(project=args.wandb_project, entity=args.wandb_entity,
                            output_dir=args.output_dir, group="olmo1b-step60000-import",
                            name="olmo-1b-step60000-import")
    started = time.monotonic()
    try:
        config = OLMoConfig.native_1b()
        report["config"] = config.to_dict()
        report["protocol"] = {
            "fp32": "TF32 disabled; source outputs atol/rtol 3e-5; gradients atol 2e-6, rtol 3e-5",
            "bf16_source": "same precision/backend source parity; outputs atol/rtol 2e-3; gradients atol 2e-4, rtol 3e-3",
            "bf16_vs_fp32": "descriptive, not training acceptance",
            "not_tested": ["training", "long contexts", "large batches", "RT", "FBT", "NextLat"],
        }
        artifact_manifest = validate_prepared_manifest(args.artifacts)
        provenance = artifact_manifest["checkpoint"]
        report["checkpoint"] = provenance
        report["artifacts_manifest"] = artifact_manifest
        report["native_reference_sources"] = verify_olmo_reference_sources()
        tokenizer = load_native_tokenizer(args.artifacts)
        sequences = [tokenizer.encode(prompt)[:args.max_length] for prompt in PROMPTS]
        report["fixtures"] = {"texts": PROMPTS, "token_ids": sequences,
                              "tokenizer_sha256": sha256_file(args.artifacts / "native/tokenizer.json")}
        write_json(args.output_dir / "report.json", report)
        tracker.start({"scope": report["scope"], "config": config.to_dict(),
                       "checkpoint_sha256": provenance["sha256"], "max_length": args.max_length})
        report["wandb"] = tracker.record
        print(json.dumps({"stage": "load", "wandb": tracker.record["run_url"]}), flush=True)
        model = OLMoForCausalLM(config, device="cuda", dtype=torch.float32).eval()
        reference = build_olmo_reference(device="cuda", dtype=torch.float32).eval()
        report["resolved_native_config"] = asdict(reference.config)
        state = load_native_state_dict(args.artifacts,
                                       expected_shapes={name: tuple(tensor.shape) for name, tensor in model.state_dict().items()})
        model.load_state_dict(state, strict=True)
        reference.load_state_dict(state, strict=True)
        report["state"] = state_dict_manifest(state)
        del state
        parameters = list(model.parameters())
        report["tying"] = {"same_parameter": model.readout_weight is model.transformer.wte.weight,
                           "optimizer_ownership_count": sum(p is model.readout_weight for p in parameters),
                           "state_embedding_keys": [name for name in model.state_dict() if name == "transformer.wte.weight"]}
        if not report["tying"]["same_parameter"] or report["tying"]["optimizer_ownership_count"] != 1:
            raise AssertionError("Tied readout ownership was lost")
        cases, fp32 = [], None
        for precision, backend in (("fp32", "math"), ("bf16_mixed", "math"), ("bf16_mixed", "default")):
            for index, sequence in enumerate(sequences):
                ids = torch.tensor([sequence], dtype=torch.long, device="cuda")
                result, saved = run_pair(model, reference, ids, precision=precision,
                                         backward=index == 0, backend=backend,
                                         fp32=fp32 if precision != "fp32" and index == 0 else None)
                if saved is not None:
                    fp32 = saved
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
        del fp32
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
