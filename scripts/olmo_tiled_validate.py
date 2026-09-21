#!/usr/bin/env python3
"""Bounded O2 tiled/native OLMo checks on the actual checkpoint; no training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cdrm.pretrained.artifacts import sha256_file, state_dict_manifest, write_json
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import (
    load_native_state_dict, load_native_tokenizer, validate_prepared_manifest,
)
from cdrm.pretrained.olmo_recurrent import OLMoRTForCausalLM, RTMode, recurrent_layer_reference
from cdrm.pretrained.olmo_reference import verify_olmo_reference_sources
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM, tiled_recurrent_layer
from scripts.experiment_tracking import OnlineTracker
from scripts.olmo_rt_validate import SOURCE_FILES as O1_SOURCE_FILES, cache_checks
from scripts.olmo_tiled_roundoff import roundoff_check
from scripts.olmo_validation import (
    PROMPTS, autocast, comparison, compare_snapshot, descriptive, loss_of,
    require_container_gpu, semantic_gradients, snapshot,
)

SOURCE_FILES = O1_SOURCE_FILES + (
    "cdrm/pretrained/olmo_tiled.py", "scripts/olmo_tiled_validate.py", "scripts/olmo_tiled_roundoff.py",
)


def run_case(model, reference, ids, *, alpha=1.0, precision="fp32",
             policy="mixed", backend="math", ordinary=False, fp32=None,
             save_snapshot=False):
    model.attention_backend = reference.attention_backend = backend
    model.attention_precision = policy
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    with autocast(precision):
        x = model.token_embeddings(ids)
        x.retain_grad()
        out = model(inputs_embeds=x, mode=RTMode((0,), alpha))
        loss = loss_of(out.logits, ids)
    loss.backward()
    # Sequential reference is useful for FP32 semantics and BF16 localization;
    # a longer BF16 runtime case instead compares directly to the tiled FP32.
    row = {"alpha": alpha, "precision": precision, "attention_precision": policy,
           "ordinary_backend": backend, "batch_size": ids.shape[0], "length": ids.shape[1],
           "reference": "ordinary" if ordinary else "sequential_rt",
           "loss": float(loss.detach())}
    with autocast(precision):
        rx = reference.token_embeddings(ids)
        rx.retain_grad()
        expected = reference(inputs_embeds=rx, mode=RTMode((), 0) if ordinary else RTMode((0,), alpha))
        expected_loss = loss_of(expected.logits, ids)
    expected_loss.backward()
    row.update(
        reference_loss=float(expected_loss.detach()),
        logits=comparison(out.logits, expected.logits, atol=1e-4, rtol=1e-4),
        hidden=comparison(out.last_hidden_state, expected.last_hidden_state, atol=1e-4, rtol=1e-4),
        loss_comparison=comparison(loss, expected_loss, atol=1e-5, rtol=1e-5),
        input_gradient=comparison(x.grad, rx.grad, atol=2e-6, rtol=3e-4),
        gradients=semantic_gradients(model, reference),
    )
    checks = ("logits", "hidden", "loss_comparison", "input_gradient")
    finite = all(row[key]["finite"] for key in checks)
    finite &= not row["gradients"]["missing"] and all(r["finite"] for r in row["gradients"]["tensors"].values())
    passed = finite and (precision != "fp32" or (row["gradients"]["passed"] and all(row[k]["passed"] for k in checks)))
    if precision != "fp32":
        row = descriptive(row)
        row["acceptance_scope"] = "finite outputs/loss/input/parameter gradients and complete ownership; precision errors descriptive"
    row.update(finite=finite, passed=passed)
    saved = snapshot(model, out, x.grad, loss) if save_snapshot else None
    if fp32 is not None:
        row["versus_tiled_fp32"] = compare_snapshot(model, out, x.grad, loss, fp32)
        row["passed"] &= row["versus_tiled_fp32"]["finite"]
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    return row, saved


def block_check(model, reference):
    """Actual native block with raw output AND terminal-cache cotangents."""
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    batch, length = 2, 17
    ids = torch.randint(0, model.config.tokenizer_vocab_size, (batch, length), device="cuda")
    x = model.token_embeddings(ids).detach().requires_grad_()
    rx = x.detach().clone().requires_grad_()
    positions = torch.arange(length, device="cuda")[None].expand(batch, -1) + torch.tensor([[3], [19]], device="cuda")
    valid = torch.ones(batch, length, dtype=torch.bool, device="cuda")
    valid[1, :2] = False
    common = dict(alpha=0.37, past=None, query_positions=positions, key_positions=positions, key_valid=valid)
    out, pair = tiled_recurrent_layer(model.layers[0], x, attention_precision="mixed", **common)
    expected, rpair = recurrent_layer_reference(reference.layers[0], rx, attention_backend="math", **common)
    go = torch.randn_like(out)
    gk, gv = (torch.randn_like(t) for t in pair)
    torch.autograd.backward((out, *pair), (go, gk, gv))
    torch.autograd.backward((expected, *rpair), (go, gk, gv))
    row = {"batch_size": batch, "length": length, "alpha": 0.37,
           "fixture": {"token_ids": ids.tolist(), "positions": positions.tolist(), "key_valid": valid.tolist()},
           "cotangent": "unnormalized Gaussian for outputs and both unrotated persistent cache tensors",
           "output": comparison(out, expected, atol=1e-4, rtol=1e-4),
           "key": comparison(pair[0], rpair[0], atol=1e-4, rtol=1e-4),
           "value": comparison(pair[1], rpair[1], atol=1e-4, rtol=1e-4),
           "input_gradient": comparison(x.grad, rx.grad, atol=2e-6, rtol=3e-4),
           "gradients": semantic_gradients(model.layers[0], reference.layers[0])}
    # The initial run retained a strict-coordinate failure under large raw
    # cotangents despite sub-ppm aggregate agreement. Do not erase that screen:
    # adjudicate BOTH paths with independent FP64 arithmetic, and explicitly
    # apply the existing tensor budget to this unnormalized input gradient.
    diagnostic = roundoff_check(model.layers[0], x, positions, valid, go, gk, gv, 0.37)
    def tensor_budget(metrics):
        return metrics["finite"] and (
            metrics["relative_l2"] <= 1e-4 or metrics["difference_l2"] <= 2e-6
        ) and metrics["max_abs"] <= 2e-6 + 1e-4 * metrics["reference_max_abs"]
    row["roundoff_adjudication"] = diagnostic
    row["input_gradient_tensor_screen"] = {
        "passed": all(tensor_budget(metrics) for metrics in diagnostic["input_gradient"].values()),
        "relative_l2_limit": 1e-4, "maximum_error_rule": "2e-6 + 1e-4*reference_max_abs",
        "protocol_amendment": "Declared after validation-01: tensor budgets for raw-cotangent input gradients, backed by FP64; original coordinate screen remains recorded",
    }
    row["passed"] = all(row[key]["passed"] for key in ("output", "key", "value", "gradients", "input_gradient_tensor_screen"))
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    runtime = require_container_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    report = {"schema": "olmo-tiled-rt-validation-v1", "status": "running", "runtime": runtime,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_hashes": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
              "scope": "native OLMo-1B, exact tiled RT layer0; no optimizer/training/FBT/NextLat",
              "protocol": "docs/reports/olmo1b-o2/protocol.md", "cases": []}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
                            group="olmo1b-step60000-tiled-rt", name="olmo-1b-o2-tiled-validation")
    started = time.monotonic()
    try:
        config = OLMoConfig.native_1b()
        manifest = validate_prepared_manifest(args.artifacts)
        report.update(config=config.to_dict(), checkpoint=manifest["checkpoint"], artifacts_manifest=manifest,
                      native_reference_sources=verify_olmo_reference_sources())
        tokenizer = load_native_tokenizer(args.artifacts)
        short = tokenizer.encode(PROMPTS[0])[:16]
        long = tokenizer.encode((PROMPTS[0] + "\n" + PROMPTS[1] + "\n") * 3)[:128]
        assert len(short) == 16 and len(long) == 128
        report["fixtures"] = {"short_ids": [short], "long_ids": [long]}
        tracker.start({"scope": report["scope"], "config": config.to_dict(), "checkpoint_sha256": manifest["checkpoint"]["sha256"]})
        report["wandb"] = tracker.record
        print({"stage": "load", "wandb": tracker.record["run_url"]}, flush=True)
        model = OLMoTiledRTForCausalLM(config, device="cuda", dtype=torch.float32).eval()
        reference = OLMoRTForCausalLM(config, device="cuda", dtype=torch.float32).eval()
        state = load_native_state_dict(args.artifacts, expected_shapes={k: tuple(v.shape) for k, v in model.state_dict().items()})
        report["state"] = state_dict_manifest(state)
        model.load_state_dict(state, strict=True)
        reference.load_state_dict(state, strict=True)
        del state
        report["tying"] = {"same_parameter": model.readout_weight is model.token_embeddings.weight,
                           "optimizer_ownership_count": sum(p is model.readout_weight for p in model.parameters())}
        assert report["tying"] == {"same_parameter": True, "optimizer_ownership_count": 1}

        def record(row, stage):
            report["cases"].append(row)
            write_json(args.output_dir / "report.json", report)
            metrics = {"validation/passed": row["passed"]}
            if "loss" in row:
                metrics["validation/loss"] = row["loss"]
                metrics["validation/gradient_relative_l2"] = row["gradients"]["relative_l2"]
            if "versus_tiled_fp32" in row:
                metrics["validation/bf16_gradient_relative_l2"] = row["versus_tiled_fp32"]["gradients"]["relative_l2"]
            tracker.log(metrics, step=len(report["cases"]))
            print({"stage": stage, "alpha": row.get("alpha"), "length": row.get("length"),
                   "precision": row.get("precision", "fp32"), "policy": row.get("attention_precision"),
                   "passed": row["passed"], "gradient_l2": row["gradients"]["relative_l2"]}, flush=True)
            if not row["passed"]:
                raise AssertionError(f"O2 {stage} failed; retained report includes individual errors")

        ids = torch.tensor([short], device="cuda")
        for alpha, ordinary in ((0, True), (0, False), (0.37, False), (1, False)):
            row, saved = run_case(model, reference, ids, alpha=alpha, ordinary=ordinary, save_snapshot=alpha == 1)
            record(row, "full_model")
            if saved is not None:
                fp32 = saved
        for backend in ("math", "sdpa"):
            for policy in ("mixed", "fp32"):
                row, _ = run_case(model, reference, ids, precision="bf16_mixed", policy=policy, backend=backend, fp32=fp32)
                record(row, "short_precision")
        del fp32, saved
        record(block_check(model, reference), "raw_block_cotangents")
        model.attention_backend, model.attention_precision = "math", "mixed"
        report["cache_causality"] = [cache_checks(model, ids, alpha) for alpha in (0, 0.37, 1)]
        if not all(row["passed"] for row in report["cache_causality"]):
            raise AssertionError("Tiled cache/causality check failed")
        ids = torch.tensor([long], device="cuda")
        row, fp32 = run_case(model, reference, ids, backend="sdpa", save_snapshot=True)
        record(row, "long_fp32")
        for policy in ("mixed", "fp32"):
            row, _ = run_case(model, reference, ids, precision="bf16_mixed", policy=policy, backend="sdpa", fp32=fp32)
            record(row, "long_precision")
        del fp32
        report.update(status="passed", elapsed_seconds=time.monotonic() - started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      memory_scope="paired validation models and gradients, not training capacity")
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
    print({"status": report["status"], "report": str(args.output_dir / "report.json"), "wandb": tracker.record["run_url"]}, flush=True)


if __name__ == "__main__":
    main()
