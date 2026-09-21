#!/usr/bin/env python3
"""Bounded actual-checkpoint OLMo-1B RT reference checks; no adaptation run."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, state_dict_manifest, write_json
from cdrm.pretrained.olmo_artifacts import (
    load_native_state_dict, validate_prepared_manifest, load_native_tokenizer,
)
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM, RTMode
from cdrm.pretrained.olmo_recurrent_oracle import olmo_recurrent_model_oracle
from cdrm.pretrained.olmo_reference import build_olmo_reference, verify_olmo_reference_sources
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_validation import (
    PROMPTS, autocast, comparison, loss_of, require_container_gpu,
    descriptive, semantic_gradients, snapshot, compare_snapshot,
)

SOURCE_FILES = (
    "cdrm/pretrained/olmo.py", "cdrm/pretrained/olmo_reference.py",
    "cdrm/pretrained/olmo_artifacts.py", "cdrm/pretrained/artifacts.py",
    "cdrm/pretrained/recurrent.py", "cdrm/pretrained/openelm.py",
    "cdrm/pretrained/olmo_recurrent.py", "cdrm/pretrained/olmo_recurrent_oracle.py",
    "scripts/olmo_validation.py", "scripts/olmo_rt_validate.py",
    "scripts/experiment_tracking.py",
)


def run_pair(model, native, ids, *, alpha, ordinary=False, precision="fp32", backend="math", fp32=None):
    model.zero_grad(set_to_none=True)
    native.zero_grad(set_to_none=True)
    context = sdpa_kernel(SDPBackend.MATH) if backend == "math" else nullcontext()
    with context, autocast(precision):
        x = model.transformer.wte(ids)
        expected_x = native.transformer.wte(ids)
        x.retain_grad()
        expected_x.retain_grad()
        output = model(inputs_embeds=x, mode=RTMode((0,), alpha))
        expected = olmo_recurrent_model_oracle(native, inputs_embeds=expected_x,
                                          selected_layers=() if ordinary else (0,), alpha=alpha)
        loss = loss_of(output.logits, ids)
        expected_loss = loss_of(expected.logits, ids)
    loss.backward()
    expected_loss.backward()
    row = {
        "alpha": alpha, "precision": precision, "backend": backend,
        "reference": "native_ordinary" if ordinary else "independent_recomputed_history_oracle",
        "loss": float(loss.detach()), "reference_loss": float(expected_loss.detach()),
        "logits": comparison(output.logits, expected.logits, atol=1e-4, rtol=1e-4),
        "hidden": comparison(output.last_hidden_state, expected.last_hidden_state, atol=1e-4, rtol=1e-4),
        "input_gradient": comparison(x.grad, expected_x.grad, atol=2e-6, rtol=3e-4),
        "loss_comparison": comparison(loss, expected_loss, atol=1e-5, rtol=1e-5),
        "gradients": semantic_gradients(model, native),
    }
    # BF16 scan versus reconstructed-history reference changes GEMM shapes and
    # attention rounding. Report every tensor but do not call these source
    # differences a failed FP32 semantics check or a precision clearance.
    checks = ("logits", "hidden", "input_gradient", "loss_comparison", "gradients")
    finite = all(row[key]["finite"] for key in checks if key != "gradients")
    finite &= not row["gradients"]["missing"] and all(r["finite"] for r in row["gradients"]["tensors"].values())
    passed = finite and (precision != "fp32" or all(row[key]["passed"] for key in checks))
    if precision != "fp32":
        row = descriptive(row)
        row["acceptance_scope"] = "finite output/backward and complete gradient ownership only"
    row.update(passed=passed, finite=finite)
    saved = snapshot(model, output, x.grad, loss) if precision == "fp32" and alpha == 1 else None
    if fp32 is not None:
        row["versus_fp32"] = compare_snapshot(model, output, x.grad, loss, fp32)
        row["passed"] &= row["versus_fp32"]["finite"]
    model.zero_grad(set_to_none=True)
    native.zero_grad(set_to_none=True)
    return row, saved


@torch.no_grad()
def cache_checks(model, ids, alpha):
    ids = ids[:, :13]
    mode = RTMode((0,), alpha)
    with sdpa_kernel(SDPBackend.MATH):
        full = model(ids, mode=mode)
        cache = None
        pieces = []
        for start, stop in ((0, 3), (3, 5), (5, ids.shape[1])):
            out = model(ids[:, start:stop], mode=mode, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            pieces.append(out.logits)
        changed = ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 17) % 50280
        altered = model(changed, mode=mode)
        no_history = model(ids[:, :1], mode=RTMode((), 0))
        wrong_mode_rejected = False
        try:
            model(ids[:, :1], mode=RTMode((0,), 0 if alpha else 1), past_key_values=cache)
        except ValueError:
            wrong_mode_rejected = True
    row = {
        "alpha": alpha,
        "chunked_vs_full": comparison(torch.cat(pieces, dim=1), full.logits, atol=1e-4, rtol=1e-4),
        "causality": comparison(altered.logits[:, :7], full.logits[:, :7], atol=0, rtol=0),
        "temporary_self_first_token": comparison(full.logits[:, :1], no_history.logits, atol=1e-4, rtol=1e-4),
        "wrong_mode_rejected": wrong_mode_rejected,
        "native_kv_heads": all(k.shape[1] == model.config.n_heads and v.shape[1] == model.config.n_heads
                               for (k, v) in cache.key_values),
    }
    row["passed"] = all(v["passed"] for v in row.values() if isinstance(v, dict)) and wrong_mode_rejected and row["native_kv_heads"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=16)
    args = parser.parse_args()
    if not 13 <= args.length <= 32:
        parser.error("This bounded protocol requires 13 <= --length <= 32")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = require_container_gpu()
    torch.set_num_threads(8)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    report = {
        "schema": "olmo-rt-reference-validation-v1", "status": "running", "runtime": runtime,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "original OLMo-1B step60000 (~252B), B1 short sequence, RT layer0 only; no optimizer or learning run",
        "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "protocol": {"alphas": [0, 0.37, 1], "selected_layers": [0], "batch_size": 1,
                     "fp32_semantics_acceptance": "logits/hidden 1e-4 absolute+relative; each parameter gradient tensor L2<=1e-4 (or whole error norm<=2e-6) AND maxerror<=2e-6+1e-4*reference_maxabs; input gradient elementwise 2e-6+3e-4*abs(reference)",
                     "budget_rationale": "initial FP32 semantic screen requires both small tensor-L2 and maximum tensor-scaled error; retains stricter elementwise diagnostics; not a transferred BF16 training budget",
                     "bf16_scope": "finite smoke checks plus descriptive source/runtime/FP32 differences",
                     "not_tested": ["tiled execution", "long contexts", "large batches", "optimizer adaptation", "FBT", "NextLat", "training quality"]},
    }
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", entity="taylorbollman",
                            output_dir=args.output_dir, group="olmo1b-step60000-rt-reference",
                            name="olmo-1b-step60000-rt-reference")
    started = time.monotonic()
    try:
        config = OLMoConfig.native_1b()
        report["config"] = config.to_dict()
        manifest = validate_prepared_manifest(args.artifacts)
        report.update(checkpoint=manifest["checkpoint"], artifacts_manifest=manifest,
                      native_reference_sources=verify_olmo_reference_sources())
        tokenizer = load_native_tokenizer(args.artifacts)
        sequence = tokenizer.encode(PROMPTS[0])[:args.length]
        if len(sequence) < 13:
            raise ValueError("Tokenized fixture must contain at least 13 tokens for cache checks")
        report["fixtures"] = {"text": PROMPTS[0], "token_ids": [sequence], "length": len(sequence)}
        tracker.start({"scope": report["scope"], "config": config.to_dict(), "protocol": report["protocol"],
                       "checkpoint_sha256": manifest["checkpoint"]["sha256"]})
        report["wandb"] = tracker.record
        print({"stage": "load", "wandb": tracker.record["run_url"]}, flush=True)
        model = OLMoRTForCausalLM(config, device="cuda", dtype=torch.float32).eval()
        native = build_olmo_reference(device="cuda", dtype=torch.float32).eval()
        report["resolved_native_config"] = asdict(native.config)
        state = load_native_state_dict(args.artifacts,
                                       expected_shapes={name: tuple(t.shape) for name, t in model.state_dict().items()})
        model.load_state_dict(state, strict=True)
        native.load_state_dict(state, strict=True)
        report["state"] = state_dict_manifest(state)
        del state
        report["tying"] = {"same_parameter": model.readout_weight is model.transformer.wte.weight,
                           "optimizer_ownership_count": sum(p is model.readout_weight for p in model.parameters())}
        if report["tying"] != {"same_parameter": True, "optimizer_ownership_count": 1}:
            raise AssertionError("Native tied parameter ownership changed")
        ids = torch.tensor([sequence], device="cuda", dtype=torch.long)
        cases, fp32 = [], None
        specifications = [(0, True, "fp32", "math"), (0, False, "fp32", "math"),
                          (0.37, False, "fp32", "math"), (1, False, "fp32", "math"),
                          (0, True, "bf16_mixed", "math"), (1, False, "bf16_mixed", "math"),
                          (1, False, "bf16_mixed", "default")]
        for alpha, ordinary, precision, backend in specifications:
            row, saved = run_pair(model, native, ids, alpha=alpha, ordinary=ordinary,
                                  precision=precision, backend=backend,
                                  fp32=fp32 if precision == "bf16_mixed" and alpha == 1 else None)
            if saved is not None:
                fp32 = saved
            cases.append(row)
            report["cases"] = cases
            write_json(args.output_dir / "report.json", report)
            tracker.log({"validation/alpha": alpha, "validation/loss": row["loss"],
                         "validation/logit_relative_l2": row["logits"]["relative_l2"],
                         "validation/gradient_relative_l2": row["gradients"]["relative_l2"],
                         "validation/passed": row["passed"]}, step=len(cases))
            print({"stage": "pair", "alpha": alpha, "precision": precision, "backend": backend,
                   "ordinary": ordinary, "passed": row["passed"], "gradient_l2": row["gradients"]["relative_l2"]}, flush=True)
            if not row["passed"]:
                raise AssertionError("RT reference validation case failed; inspect report")
        del fp32
        report["cache_causality"] = [cache_checks(model, ids, alpha) for alpha in (0, 0.37, 1)]
        if not all(row["passed"] for row in report["cache_causality"]):
            raise AssertionError("Cache/causality check failed")
        report.update(status="passed", elapsed_seconds=time.monotonic() - started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      memory_scope="paired native+RT reference validation with gradients, not training capacity")
        tracker.summary({"validation/status": "passed", "validation/peak_allocated_gib": report["peak_allocated_gib"]})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir / "report.json", report)
    print({"status": report["status"], "report": str(args.output_dir / "report.json"),
           "wandb": tracker.record["run_url"]}, flush=True)


if __name__ == "__main__":
    main()
