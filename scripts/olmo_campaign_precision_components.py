#!/usr/bin/env python3
"""Bounded initial-state CE/latent/KL precision localization, without training.

For the original, non-packed NFR fixture, compare BF16 sparse and prepared
dense gradients with a common full-FP32 sparse reference. Retain the existing
BF16 layout qualification. Cross-precision errors and old-budget flags are
descriptive: completion gates only finite execution, fixed weights, RNG and
source integrity. No optimizer, DDP, graph capture, or updated-state sweep.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.campaign_losses import compute_dynamic_nextlat_loss_sums
from cdrm.pretrained.campaign_training import CampaignObjective
from cdrm.pretrained.lm_training import LMTrainingConfig
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import (canonical_backward, construct,
    fixture_for_update, global_fixture_metadata, prepared_backward,
    source_hashes as campaign_sources)
from scripts.olmo_campaign_fp32_localize import configure_full_fp32
from scripts.olmo_campaign_graph_probe import compare_metrics, rng_snapshot, rng_unchanged
from scripts.olmo_campaign_probe import component, gradient_record, memory
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

TERMS = ("ce", "latent", "kl")
OBJECTIVES = ("combined", *TERMS)
RUNTIME_FLAGS = ("attention_backend", "ordinary_attention_backend", "attention_precision",
                 "tile_backend", "backward_tile_backend", "backward_memory", "rt_implementation",
                 "ordinary_pointwise_backend", "ordinary_rope_backend", "ordinary_activation_checkpointing",
                 "cast_weights_once", "reuse_rope", "kv_only_writes")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / ".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def pass_sums(passes):
    """Literal campaign-v1 weights: CE .5/1/6/1/6/1/6; auxiliaries uniform."""
    count = len(passes)
    if count not in (1, 4):
        raise ValueError("Diagnostic requires one or four campaign passes")
    ce = (1.,) if count == 1 else (.5, 1/6, 1/6, 1/6)
    return {term: sum(p[term] * (ce[i] if term == "ce" else 1/count)
                      for i, p in enumerate(passes)) for term in TERMS}


def component_backward(model, recipe, fixtures, *, precision, layout, objective):
    """Keep every loss branch enabled; isolate by zeroing its cotangent.

    Stop-gradient rules remain in the original loss functions. Other terms are
    multiplied by zero after their ordinary global normalization, so changing
    the diagnostic objective does not reconfigure predictor participation.
    Combined execution delegates to the original qualification helpers.
    """
    if precision not in ("fp32", "bf16_mixed") or layout not in ("sparse", "prepared") or objective not in OBJECTIVES:
        raise ValueError("Unsupported precision/layout/component diagnostic")
    if objective == "combined":
        function = canonical_backward if layout == "sparse" else prepared_backward
        return function(model, recipe, fixtures, precision=precision)
    model.zero_grad(set_to_none=True)
    metadata = global_fixture_metadata(model, fixtures)
    if any(metadata["counts"][term] <= 0 for term in TERMS):
        raise ValueError("Every diagnostic objective needs positive global targets")
    device = next(model.parameters()).device
    batches = tuple(batch for rank_batches, _ in fixtures for batch in rank_batches)
    noises = tuple(noise for _, rank_noises in fixtures for noise in rank_noises)
    adapter = None
    if layout == "prepared":
        adapter = CampaignObjective(model, batches[0], mode=recipe.mode(),
            global_counts=metadata["counts"], world_size=1, feedback_noise=noises[0],
            config=LMTrainingConfig(precision=precision, max_grad_norm=recipe.max_grad_norm))
    totals, objective_total = dict.fromkeys(TERMS, 0.), 0.
    for batch, noise in zip(batches, noises):
        if adapter is not None:
            adapter.load_batch(batch, feedback_noise=noise, global_counts=metadata["counts"])
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=precision == "bf16_mixed", cache_enabled=False):
            if layout == "sparse":
                noise = None if noise is None else tuple(value.to(device) for value in noise)
                result = model.loss_sums(batch.to(device), backbone_kwargs={"mode": recipe.mode(),
                    "feedback_noise": noise, "right_padded_causal": True})
                sums = pass_sums(tuple(p.sums for p in result.pass_losses))
                normalized = {term: sums[term] * result.weights[term] / metadata["counts"][term]
                              for term in TERMS}
            else:
                output = adapter.forward_layout.forward(adapter.batch.input_ids, adapter.mode,
                                                         feedback_noise=adapter.feedback_noise)
                passes = tuple(compute_dynamic_nextlat_loss_sums(hidden, output.embeddings,
                    model.backbone.readout_weight, adapter.batch.input_ids, model.predictor,
                    model.config, adapter.loss_layout, enabled=model.enabled)
                    for hidden in output.pass_hidden_states)
                sums = pass_sums(passes)
                normalized = {term: sums[term] * adapter.coefficients[index]
                              for index, term in enumerate(TERMS)}
            selected = sum(normalized[term] * float(term == objective) for term in TERMS)
        selected.backward()
        objective_total += float(selected.detach())
        for term in TERMS:
            totals[term] += float(sums[term].detach())
    return {**metadata, "loss_sums": totals, "objective": objective_total,
            "selected_objective": objective, "weights": dict(model.objective_weights())}


def _geometry_finish(record):
    ref, actual, error, dot = (record[key] for key in ("reference_squared", "actual_squared", "error_squared", "dot"))
    ref_norm, actual_norm = math.sqrt(ref), math.sqrt(actual)
    cosine = max(-1., min(1., dot/(ref_norm*actual_norm))) if ref_norm and actual_norm else None
    return {"reference_gradient_norm": ref_norm, "actual_gradient_norm": actual_norm,
            "difference_norm": math.sqrt(error), "relative_l2": math.sqrt(error/max(ref, 1e-60)),
            "norm_ratio": actual_norm/ref_norm if ref_norm else None,
            "cosine": cosine, "angle_degrees": math.degrees(math.acos(cosine)) if cosine is not None else None,
            "reference_is_zero": ref == 0, "actual_is_zero": actual == 0,
            "reference_dot_actual": dot, "parameter_tensors": record["parameter_tensors"]}


@torch.no_grad()
def gradient_geometry(actual, reference):
    """Stream FP32 parameter gradients; aggregate dot products/norms in FP64."""
    if actual.keys() != reference.keys():
        raise ValueError("Gradient names differ")
    groups = {name: dict.fromkeys(("reference_squared", "actual_squared", "error_squared", "dot", "parameter_tensors"), 0.)
              for name in ("all", "backbone", "fusion", "predictor")}
    for name, value in actual.items():
        want = reference[name].to(device=value.device)
        if value.shape != want.shape or value.dtype != want.dtype:
            raise ValueError("Gradient shape/dtype differs")
        # A parameter at a time bounds both host/device peak storage. Float64
        # products avoid artificial angles from subtracting nearly equal norms.
        a, b = value.to(torch.float64), want.to(torch.float64)
        numbers = {"reference_squared": float(b.square().sum()),
                   "actual_squared": float(a.square().sum()),
                   "error_squared": float((a-b).square().sum()),
                   "dot": float((a*b).sum()), "parameter_tensors": 1}
        for key in ("all", component(name)):
            for metric, number in numbers.items():
                groups[key][metric] += number
    return {name: _geometry_finish(row) for name, row in groups.items()}


def record_gradients(model, reference=None, *, save_cpu=False, scope):
    # Components keep zero-cotangent branches active. A missing trainable grad
    # is nevertheless represented as mathematical zero for fair vector alignment
    # and reported explicitly instead of silently dropping parameter dimensions.
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    record, snapshot = gradient_record(model, reference, save_cpu=save_cpu)
    record["originally_missing_zero_materialized"] = missing
    record["participation_intact"] = not missing
    if reference is not None:
        record["comparison"]["scope"] = scope
        record["comparison"]["gating"] = False
        record["geometry"] = gradient_geometry(
            {name: p.grad for name, p in model.named_parameters() if p.requires_grad}, reference)
    return record, snapshot


def source_hashes():
    result = campaign_sources()
    for path in (Path(__file__), ROOT / "scripts/olmo_campaign_fp32_localize.py"):
        result[str(path.relative_to(ROOT))] = sha256_file(path)
    return dict(sorted(result.items()))


def main(argv=None):
    args = parse_args(argv)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Component diagnostic requires one process without DDP")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-campaign-precision-components-v1", "status": "running", "passed": False,
              "runtime": runtime, "sources": source_hashes(), "rows": [],
              "started_utc": datetime.now(timezone.utc).isoformat(), "fixture": "isolated-v1",
              "arm": "NFR", "length": 16, "physical_batch": 2, "virtual_ranks": 2, "fixture_update": 0,
              "objectives": list(OBJECTIVES), "optimizer_updates": 0, "parameter_states": 1,
              "numerical_qualification": "Previous 3.40224% BF16 sparse/prepared failure remains; this report measures component errors and does not clear it",
              "cross_precision_acceptance": "No new BF16-versus-FP32 numerical pass/fail threshold; old-budget flags are descriptive",
              "legacy_budgets": {"raw_gradient_atol": 3e-5, "raw_gradient_rtol": 3e-4,
                                 "loss_sum_atol": 1e-5, "loss_sum_rtol": 3e-6,
                                 "objective_atol": 1e-6, "objective_rtol": 3e-6},
              "scope": "Initial actual pretrained NFR weights only; separate normalized CE, latent and KL gradients; no packing, DDP, graphs, optimizer, throughput or training-quality claim",
              "checkpoints": "Pinned source reused; at most two component gradient references in CPU RAM; no new model checkpoint"}
    for relative in report["sources"]:
        target = args.output_dir / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-campaign-precision-components", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started = time.monotonic()

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir / "report.json", report)

    def publish(stage, row):
        report["rows"].append({"stage": stage, **row})
        persist(stage)
        tracker.log(scalar_metrics(row, "diagnostic/"+stage), step=len(report["rows"]))
        print({"stage": stage, "operational_passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
        if not row["passed"]:
            raise AssertionError("Component diagnostic failed an operational health/state guard")

    failure = None
    try:
        tracker.start({key: report[key] for key in ("scope", "fixture", "objectives", "legacy_budgets", "cross_precision_acceptance")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("loading_source")
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NFR", torch.device("cuda"))
        base = model.backbone.backbone
        bf16_flags = {key: getattr(base, key) for key in RUNTIME_FLAGS}
        report.update(recipe=recipe.to_dict(), recipe_sha256=recipe.sha256, source_checkpoint=checkpoint,
            bf16_execution={"runtime_flags": bf16_flags, "ordinary_sdpa_dispatch": "forced FLASH_ATTENTION",
                            "precision": "BF16 mixed; FP32 master parameters and gradients; autocast cache off",
                            "tf32": False, "float32_matmul_precision": "highest"})
        fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
                       length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
        report["fixture_inputs"] = tree_digests([
            {"batches": [vars(batch) for batch in batches], "noise": noise} for batches, noise in fixtures])
        weights = tree_digests(dict(model.named_parameters()))
        before = rng_snapshot()
        for objective in OBJECTIVES:
            report["fp32_execution"] = configure_full_fp32(model)
            persist(objective+"/fp32_sparse_backward")
            with sdpa_kernel(SDPBackend.MATH):
                fp32_metrics = component_backward(model, recipe, fixtures, precision="fp32", layout="sparse", objective=objective)
            fp32_gradients, fp32_reference = record_gradients(model, save_cpu=True, scope="Full-FP32 sparse reference")
            publish(objective+"/fp32_sparse", {"objective_component": objective, "metrics": fp32_metrics,
                "normalized_loss_means": {term: fp32_metrics["loss_sums"][term]/fp32_metrics["counts"][term] for term in TERMS},
                "gradients": fp32_gradients, "rng_unchanged": rng_unchanged(before), "memory": memory(),
                "passed": fp32_gradients["finite"] and fp32_gradients["participation_intact"] and rng_unchanged(before)})
            for name, value in bf16_flags.items():
                setattr(base, name, value)
            for layout in ("sparse", "prepared"):
                persist(objective+"/bf16_"+layout+"_backward")
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    metrics = component_backward(model, recipe, fixtures, precision="bf16_mixed", layout=layout, objective=objective)
                gradients, snapshot = record_gradients(model, fp32_reference, save_cpu=layout == "sparse",
                    scope="Descriptive BF16 "+layout+" versus common FP32 sparse reference; no cross-precision acceptance budget")
                row = {"objective_component": objective, "metrics": metrics, "versus_fp32": gradients,
                       "normalized_loss_means": {term: metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
                       "loss_comparison_vs_fp32": {**compare_metrics(metrics, fp32_metrics), "gating": False},
                       "rng_unchanged": rng_unchanged(before), "memory": memory(),
                       "passed": gradients["finite"] and gradients["participation_intact"] and rng_unchanged(before)}
                if layout == "sparse":
                    sparse_reference, sparse_metrics = snapshot, metrics
                else:
                    pair, _ = record_gradients(model, sparse_reference,
                        scope="Retained BF16 prepared versus sparse layout comparison; legacy budgets descriptive here")
                    row["versus_bf16_sparse"] = pair
                    row["loss_comparison_vs_bf16_sparse"] = {**compare_metrics(metrics, sparse_metrics), "gating": False}
                    row["passed"] &= pair["finite"]
                publish(objective+"/bf16_"+layout, row)
            del fp32_reference, sparse_reference, snapshot
            model.zero_grad(set_to_none=True)
            gc.collect()
        weights_equal = weights == tree_digests(dict(model.named_parameters()))
        sources_equal = source_hashes() == report["sources"]
        publish("fixed_state_and_source_integrity", {"weights_unchanged": weights_equal,
            "sources_unchanged": sources_equal, "rng_unchanged": rng_unchanged(before),
            "passed": weights_equal and sources_equal and rng_unchanged(before)})
        report.update(status="passed_operational_diagnostic", passed=True)
        persist("complete")
    except BaseException as error:
        failure = error
        report.update(status="failed", passed=False,
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        persist(report.get("stage", "setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed", passed=False, tracking_finish_error={"type": type(error).__name__})
            if failure is None:
                raise
        finally:
            persist(report.get("stage", "setup"))


if __name__ == "__main__":
    main()
