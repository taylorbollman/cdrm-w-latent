#!/usr/bin/env python3
"""Matched NF FP32/BF16 gradients at saved adapted O5c backbone/fusion weights.

New current-runtime weights-only diagnostic, not historical resume. Two aggregate
CE cases/four physical backwards at K4/T16, no RT, updates, DDP or CUDA graphs.
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
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_adapted_import import endpoint_authority, load_into_current, source_hashes as import_sources
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_fusion_precision import load_reference, reference_anchor, state_contract_checks, source_hashes as prior_sources
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_bridge import capture_passes, configure_path, forward_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS, component_backward, record_gradients
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, arm_contract, fixture_pins, gradient_norm_summary, state_pins
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

PATHS = (FP32, BF16)
REFERENCE_SHA = "bfaff91aae8e2625e5f2572cfaf4f33d449b560d5cefbef7ff563c6d820ac412"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path,
        default=ROOT/".runtime/olmo-recurrence-precision/matrix-01/report.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--checkpoint", type=Path,
        default=ROOT/".runtime/olmo1b-step60000/o5c-pilot-01/mixed/update-000512.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error("Evidence must remain within the persistent project")
    return args


def source_hashes():
    result = prior_sources()
    for name, digest in import_sources().items():
        if name in result and result[name] != digest:
            raise ValueError("Current source inventories disagree")
        result[name] = digest
    for name in ("scripts/olmo_campaign_adapted_precision.py",
                 "tests/test_campaign_adapted_precision.py",
                 "docs/reports/olmo-adapted-precision/protocol.md"):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def fixed_contract_checks(current, cold):
    """Only native/fusion tensor state may change during weights-only import."""
    result = {key+"_exact": tree_digests(value) == tree_digests(cold[key])
              for key, value in current.items() if key != "initial_state"}
    result["predictor_state_exact"] = current["initial_state"]["predictor"] == cold["initial_state"]["predictor"]
    return result


def case_health(metrics, gradients, geometry, observed, metadata):
    return {
        "finite_gradients": gradients["finite"],
        "gradient_participation": gradients["participation_intact"],
        "finite_forward_and_cotangents": all(
            row["hidden"]["finite"] and row["total_incoming_cotangent"]["finite"]
            for row in geometry if not row.get("skipped_dummy")),
        "finite_losses": all(math.isfinite(value) for value in (metrics["objective"], *metrics["loss_sums"].values())),
        "four_passes_each_record": len(observed) == 2 and all(len(row["pass_hidden_states"]) == 4 for row in observed),
        "counts_exact": all(metrics[key] == value for key, value in metadata.items()),
        "predictor_zero_cotangent": gradients["groups"]["predictor"]["norm"] == 0.0,
    }


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Adapted diagnostic uses one process without DDP")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-campaign-adapted-precision-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "sources": source_hashes(), "determinism": determinism, "rows": [], "paths": PATHS,
        "reference_report": str(args.reference_report), "reference_sha256": REFERENCE_SHA,
        "objective": "ce", "arm": "NF", "aggregate_backwards": 2, "physical_batch_backwards": 4,
        "optimizer_updates": 0, "scope": __doc__,
        "checkpoint_provenance_scope": "cold_construction/adapted_state source_checkpoint identifies construction artifacts only; adapted_import identifies actual loaded O5c weights",
        "qualification": "Numerical differences descriptive; no new budget, fix or BF16 clearance",
        "state_comparison_scope": "Within adapted state; cold state is context, not a causal fusion-only ablation",
        "cotangent_scope": "TOTAL incoming pass-output gradients include later feedback and differ across precision cases",
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for name in report["sources"]:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-adapted-precision", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({key: report[key] for key in ("scope", "paths", "determinism", "reference_sha256", "qualification")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("verify_reference_and_construct_cold")
        previous = load_reference(args.reference_report, REFERENCE_SHA, report["sources"])
        authority = endpoint_authority()
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NF", torch.device("cuda"))
        original = {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS}
        fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
            length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]

        def info():
            return {"contract": arm_contract(model, recipe), "source_checkpoint": checkpoint,
                "model_config": model.backbone.backbone.config.to_dict(), "nextlat_config": model.config.to_dict(),
                "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256,
                "initial_state": state_pins(model), "production_runtime_flags": {
                    key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS},
                "fixture_inputs": fixture_pins(fixtures), "fixture_metadata": global_fixture_metadata(model, fixtures)}

        cold = info()
        checks = state_contract_checks(cold, previous["arms"]["NF"])
        checks.update(runtime_exact=runtime == previous["runtime"], determinism_exact=determinism == previous["determinism"],
            math_reduction_exact=report["math_sdpa_reduced_precision_reduction"] == previous["math_sdpa_reduced_precision_reduction"],
            bf16_matmul_reduction_exact=report["bf16_matmul_reduced_precision_reduction"] == previous["bf16_matmul_reduced_precision_reduction"])
        report.update(cold_construction=cold, cold_contract_checks=checks)
        if not all(checks.values()):
            raise AssertionError("Pre-import construction/fixture differs from pinned cold NF matrix")
        rng = rng_snapshot()
        persist("import_adapted_weights")
        report["adapted_import"] = load_into_current(model, args.checkpoint, authority)
        adapted = info()
        checks = fixed_contract_checks(adapted, cold)
        checks["rng_unchanged"] = rng_unchanged(rng)
        report.update(adapted_state=adapted, import_contract_checks=checks)
        if not all(checks.values()):
            raise AssertionError("Import altered non-imported state/fixture/runtime or trainability")
        reference_gradients, reference_forward = None, None
        for path in PATHS:
            execution = configure_path(model, original, path)
            persist(path+"/backward")
            torch.cuda.reset_peak_memory_stats()
            case_started = time.monotonic()
            backend = SDPBackend.MATH if path == FP32 else SDPBackend.FLASH_ATTENTION
            with capture_passes(model, fixtures) as observed, sdpa_kernel(backend):
                metrics = component_backward(model, recipe, fixtures, precision=execution["precision"], layout="sparse", objective="ce")
            gradients, snapshot = record_gradients(model, reference_gradients, save_cpu=path == FP32,
                scope="Descriptive versus matched adapted-state full FP32 reference; no numerical gate")
            geometry = forward_geometry(observed, reference_forward)
            fingerprints = tree_digests([{key: row[key] for key in ("batch", "token_embeddings", "pass_hidden_states")} for row in observed])
            health = case_health(metrics, gradients, geometry, observed, adapted["fixture_metadata"])
            health["rng_unchanged"] = rng_unchanged(rng)
            row = {"path": path, "objective": "ce", "execution": execution, "metrics": metrics,
                "ce_pass_weights": adapted["contract"]["ce_pass_weights"],
                "normalized_loss_means": {term: metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
                "gradients" if path == FP32 else "gradients_vs_fp32": gradients,
                "forward_vs_fp32": geometry, "forward_fingerprints": fingerprints,
                "gradient_norms_descriptive_only": gradient_norm_summary(gradients), "health": health,
                "elapsed_seconds_including_observation": time.monotonic()-case_started, "memory": memory(),
                "passed": all(health.values())}
            report["rows"].append(row)
            persist(path)
            tracker.log(scalar_metrics(row, "diagnostic/"+path), step=len(report["rows"]))
            print({"stage": path, "operational_passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
            if not row["passed"]:
                raise AssertionError("Adapted precision case failed health guards")
            if path == FP32:
                reference_gradients, reference_forward = snapshot, observed
        old = reference_anchor(previous, BF16)
        report["cold_context"] = {
            "scope": "Previously saved cold NF precision comparison, not cross-state gradient comparison",
            "fp32_ce": reference_anchor(previous, FP32)["metrics"]["objective"],
            "bf16_ce": old["metrics"]["objective"],
            "gradient_geometry": old["gradients_vs_fp32"]["geometry"],
            "forward_geometry": old["forward_vs_fp32"]}
        report["integrity"] = {
            "adapted_parameters_and_buffers_unchanged": state_pins(model) == adapted["initial_state"],
            "inputs_unchanged": fixture_pins(fixtures) == adapted["fixture_inputs"],
            "rng_unchanged": rng_unchanged(rng),
            "arm_configuration_unchanged": arm_contract(model, recipe) == adapted["contract"],
            "production_flags_restored": all(getattr(model.backbone.backbone, key) == value for key, value in original.items()),
            "sources_unchanged": source_hashes() == report["sources"],
            "reference_unchanged": sha256_file(args.reference_report) == REFERENCE_SHA}
        if not all(report["integrity"].values()):
            raise AssertionError("Fixed adapted diagnostic state changed")
        model.zero_grad(set_to_none=True)
        del reference_gradients, reference_forward, snapshot, observed
        gc.collect()
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
