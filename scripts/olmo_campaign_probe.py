#!/usr/bin/env python3
"""Bounded actual-checkpoint K4 campaign smoke, never a throughput/quality run.

All stages are independently replayable from the pinned source. Atomically save
progress after each backward/update; no disposable optimizer state is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_recipe import (CampaignRecipe, CampaignTokenSchedule,
    build_campaign_model, build_campaign_adamw, feedback_noise_for_rows)
from cdrm.pretrained.nextlat import NextLatBatch
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.olmo_artifacts import load_native_state_dict, load_native_tokenizer, validate_prepared_manifest
from cdrm.pretrained.olmo_tiled import OLMoTiledRTForCausalLM
import cdrm.pretrained.olmo_tiled as tiled_module
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_validation import require_container_gpu


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(16, 32), default=16)
    parser.add_argument("--updates", type=int, choices=(1, 2, 3), default=2)
    return parser.parse_args(argv)


def explicit_k4_objective(result):
    """Independent literal campaign coefficients; do not use aggregate metadata."""
    if len(result.pass_losses) != 4 or any(value != 1 for value in result.weights.values()):
        raise ValueError("Probe requires K4 and unit objective weights")
    coefficients = {"ce": (.5, 1 / 6, 1 / 6, 1 / 6), "latent": (.25,) * 4, "kl": (.25,) * 4}
    objective = 0
    for term in ("ce", "latent", "kl"):
        count = result.pass_losses[0].counts[term]
        if count <= 0 or any(loss.counts[term] != count for loss in result.pass_losses):
            raise ValueError("Probe requires positive common per-pass target counts")
        objective = objective + sum(coefficient * loss.sums[term]
            for coefficient, loss in zip(coefficients[term], result.pass_losses)) / count
    return objective


def component(name):
    return "predictor" if name.startswith("predictor.") else (
        "fusion" if name.startswith("backbone.fusion.") else "backbone")


def source_hashes():
    files = set((ROOT / "cdrm/pretrained").rglob("*.py")) | {
        Path(__file__), ROOT / "scripts/experiment_tracking.py", ROOT / "scripts/olmo_validation.py"}
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(files)}


def memory():
    free, total = torch.cuda.mem_get_info()
    return {"allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "sampled_free_gib": free / 2**30, "total_gib": total / 2**30}


@torch.no_grad()
def gradient_record(model, reference=None, *, save_cpu=False):
    """Stream one parameter at a time; do not concatenate billion-scale tensors."""
    groups = {name: {"sum_squares": 0.0, "max_abs": 0.0, "tensors": 0, "finite": True}
              for name in ("backbone", "fusion", "predictor")}
    saved, missing, rows = {}, [], {}
    error_sq = reference_sq = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.grad
        if value is None:
            missing.append(name)
            continue
        # Gradients and master parameters remain FP32; aggregate in FP64.
        finite = bool(torch.isfinite(value).all())
        squared = float(value.square().sum(dtype=torch.float64))
        peak = float(value.abs().max())
        group = groups[component(name)]
        group["sum_squares"] += squared
        group["max_abs"] = max(group["max_abs"], peak)
        group["finite"] &= finite
        group["tensors"] += 1
        if save_cpu:
            saved[name] = value.detach().cpu().clone()
        if reference is not None:
            want = reference[name].to(value.device)
            difference = value - want
            err = float(difference.square().sum(dtype=torch.float64))
            ref = float(want.square().sum(dtype=torch.float64))
            maximum = float(difference.abs().max())
            rows[name] = {"max_abs": maximum,
                "relative_l2": math.sqrt(err / max(ref, 1e-60)),
                "allclose": bool(torch.allclose(value, want, atol=3e-5, rtol=3e-4))}
            error_sq += err
            reference_sq += ref
            del want, difference
    for group in groups.values():
        group["norm"] = math.sqrt(group.pop("sum_squares"))
    record = {"groups": groups, "missing_active_gradients": missing,
              "finite": not missing and all(row["finite"] and math.isfinite(row["norm"])
                                            for row in groups.values())}
    if reference is not None:
        record.update(comparison={"relative_l2": math.sqrt(error_sq / max(reference_sq, 1e-60)),
            "parameter_rows": rows, "all_parameters_close": all(row["allclose"] for row in rows.values()),
            "atol": 3e-5, "rtol": 3e-4,
            "scope": "Same BF16 execution and weights; independent pass aggregation only, not FP32 qualification"})
    return record, saved


def backward_probe(model, recipe, tokens, *, update, explicit=False, reference=None, save_cpu=False):
    model.zero_grad(set_to_none=True)
    noise = feedback_noise_for_rows(recipe, ["campaign-smoke-row-0"], logical_update=update,
        sequence_length=tokens.input_ids.shape[1], width=model.backbone.config.model_dim, device="cuda")
    base = model.backbone.backbone
    indices = {id(layer): index for index, layer in enumerate(base.layers)}
    counts = Counter()
    original = tiled_module.tiled_recurrent_layer

    def observed(layer, *args, **kwargs):
        counts[indices[id(layer)]] += 1
        return original(layer, *args, **kwargs)

    started = time.monotonic()
    rng_before = torch.cuda.get_rng_state().clone()
    with patch.object(tiled_module, "tiled_recurrent_layer", observed):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                result = model.loss_sums(tokens, backbone_kwargs={"mode": recipe.mode(),
                    "full_valid_causal": True, "feedback_noise": noise})
                objective = explicit_k4_objective(result) if explicit else result.total
            losses = {"objective": float(objective.detach()), "counts": result.counts,
                "means": {term: float(value.detach()) for term, value in result.means.items()},
                "pass_means": [{term: float(value.detach()) for term, value in loss.means.items()}
                               for loss in result.pass_losses]}
            objective.backward()
    torch.cuda.synchronize()
    gradients, saved = gradient_record(model, reference, save_cpu=save_cpu)
    record = {"explicit_aggregation": explicit, "losses": losses, "gradients": gradients,
              "rt_layer_invocations": {str(key): count for key, count in sorted(counts.items())},
              "cuda_rng_unchanged": torch.equal(rng_before, torch.cuda.get_rng_state()),
              "elapsed_seconds": time.monotonic() - started, "memory": memory()}
    record["passed"] = (gradients["finite"] and math.isfinite(losses["objective"])
        and counts == Counter({0: 4, 15: 4}) and record["cuda_rng_unchanged"]
        and (reference is None or gradients["comparison"]["all_parameters_close"]))
    return record, saved


def main(argv=None):
    args = parse_args(argv)
    runtime = require_container_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260929)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    recipe = CampaignRecipe("NFR", sequence_length=1024)
    report = {"schema": "olmo-campaign-portable-probe-v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256,
        "fixture_length": args.length, "physical_batch": 1, "sources": source_hashes(),
        "scope": "Bounded eager semantic/gradient smoke, not capacity, training quality, FP32 or distributed qualification",
        "execution": {"ordinary_attention": "forced Flash SDPA", "ordinary_pointwise": "eager",
            "rope": "native reused", "native_rt_tiles": "Triton", "backward": "recompute",
            "ordinary_checkpointing": True, "cuda_graphs": False, "torch_compile": False,
            "precision": "BF16 mixed with FP32 parameters/gradients/Adam; TF32 off; autocast cache off"},
        "rows": [], "checkpoints": "No disposable full model checkpoint; each short stage restarts from pinned original"}
    for relative in report["sources"]:
        destination = args.output_dir / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-campaign-training-readiness", name="campaign-portable-" + args.output_dir.name)
    started = time.monotonic()

    def publish(stage, row):
        report["rows"].append({"stage": stage, **row})
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)
        tracker.log(scalar_metrics(row, "diagnostic/" + stage), step=len(report["rows"]))
        print({"stage": stage, "passed": row.get("passed"), "elapsed_seconds": time.monotonic() - started}, flush=True)

    try:
        manifest = validate_prepared_manifest(args.artifacts)
        report["checkpoint"] = manifest["checkpoint"]
        tracker.start({"recipe": report["recipe"], "fixture_length": args.length,
                       "checkpoint": report["checkpoint"], "scope": report["scope"]})
        report["wandb"] = tracker.record
        write_json(args.output_dir / "report.json", report)
        print({"wandb": tracker.record["run_url"]}, flush=True)
        state = load_native_state_dict(args.artifacts)
        base = OLMoTiledRTForCausalLM(OLMoConfig.native_1b(), device="meta", dtype=torch.float32,
            attention_backend="sdpa", attention_precision="mixed", ordinary_activation_checkpointing=True,
            cast_weights_once=True, tile_backend="triton", backward_tile_backend="triton",
            backward_memory="recompute", reuse_rope=True, kv_only_writes=True)
        base.load_state_dict(state, strict=True, assign=True)
        model = build_campaign_model(base, recipe).to("cuda").train()
        del state, base
        gc.collect()
        tokenizer = load_native_tokenizer(args.artifacts)
        ids = tokenizer.encode("The model keeps a record of earlier tokens and predicts the next token. " * 20)[:args.length - 1]
        ids.append(tokenizer.eos_token_id)
        tokens = torch.tensor([ids], device="cuda", dtype=torch.long)
        valid = torch.ones_like(tokens, dtype=torch.bool)
        batch = NextLatBatch(tokens, valid, torch.zeros_like(tokens), valid.clone(), valid.clone(), valid.clone())
        report["fixture"] = {"input_ids": ids, "all_positions_supervised": True,
            "source": "Fixed repeated operational prose, no benchmark or corpus sample"}
        report["parameters"] = {"resident": sum(p.numel() for p in model.parameters()),
                                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        torch.cuda.reset_peak_memory_stats()
        canonical, reference = backward_probe(model, recipe, batch, update=0, save_cpu=True)
        publish("canonical", canonical)
        if not canonical["passed"]:
            raise AssertionError("Canonical campaign backward failed")
        explicit, _ = backward_probe(model, recipe, batch, update=0, explicit=True, reference=reference)
        explicit["objective_abs_difference"] = abs(explicit["losses"]["objective"] - canonical["losses"]["objective"])
        explicit["passed"] &= explicit["objective_abs_difference"] <= 1e-6
        publish("literal-pass-weights", explicit)
        if not explicit["passed"]:
            raise AssertionError("Independent pass-weight aggregation differs")
        del reference
        model.zero_grad(set_to_none=True)
        gc.collect()
        optimizer = build_campaign_adamw(model, recipe, fused=True)
        schedule = CampaignTokenSchedule(optimizer, [args.length] * args.updates,
            warmup_tokens=recipe.warmup_tokens, start_fraction=recipe.warmup_start_fraction)
        for update in range(args.updates):
            schedule.validate_next_update(args.length)
            row, _ = backward_probe(model, recipe, batch, update=update + 1)
            if not row["passed"]:
                publish(f"update-{update + 1}", row)
                raise AssertionError("Nonfinite campaign update backward")
            row["lr_used"] = optimizer.param_groups[0]["lr"]
            row["gradient_norm_before_clip"] = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True))
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            row["completed_valid_tokens"] = schedule.completed_tokens
            row["memory_after_optimizer"] = memory()
            row["parameters_finite_after_update"] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
            row["passed"] &= row["parameters_finite_after_update"]
            publish(f"update-{update + 1}", row)
            if not row["passed"]:
                raise AssertionError("Nonfinite campaign parameters after update")
        if report["sources"] != source_hashes():
            raise AssertionError("Runtime source changed during probe")
        report["status"] = "passed"
        tracker.summary({"diagnostic/status": "passed", "diagnostic/updates": args.updates})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        try:
            tracker.finish(succeeded=report["status"] == "passed")
        finally:
            report.update(wandb=tracker.record, finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.output_dir / "report.json", report)


if __name__ == "__main__":
    main()
