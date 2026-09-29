#!/usr/bin/env python3
"""Eight CE-only recurrence/precision cases at fixed initial pretrained weights.

N/NR/NF/NFR each compare FP32 math/eager with production BF16 Flash/Triton.
Every NextLat branch remains active with zero auxiliary cotangents. The two
original T16 B2 virtual-rank records are physical backwards on one GPU; there
is no DDP, CUDA graph, optimizer, precision promotion or training experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
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
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_bridge import (
    capture_passes, configure_path, forward_geometry, source_hashes as bridge_sources,
)
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS, component_backward, record_gradients
from scripts.olmo_campaign_probe import component, memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

ARMS = ("N", "NR", "NF", "NFR")
FP32, BF16 = PATHS = ("fp32_math_eager", "bf16_flash_triton")
GRADIENT_SUMMARY_KEYS = ("finite", "groups", "missing_active_gradients",
                         "originally_missing_zero_materialized", "participation_intact")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.reference_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.reference_sha256):
        parser.error("Reference SHA256 must contain 64 lowercase hexadecimal characters")
    return args


def source_hashes():
    result = bridge_sources()
    for name in ("scripts/olmo_campaign_recurrence_precision.py",
                 "tests/test_campaign_recurrence_precision.py",
                 "docs/reports/olmo-recurrence-precision/protocol.md"):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def load_reference(path, expected_sha, sources):
    path = Path(path)
    if path.stat().st_size > 64*1024*1024 or sha256_file(path) != expected_sha:
        raise ValueError("Reference bytes differ from the bounded SHA256 pin")
    report = json.loads(path.read_text())
    if sha256_file(path) != expected_sha:
        raise ValueError("Reference changed while reading")
    if (report.get("schema") != "olmo-campaign-precision-bridge-v1"
            or report.get("status") != "passed_operational_diagnostic" or report.get("passed") is not True):
        raise ValueError("Reference must be a completed operational bridge report")
    if not report.get("sources") or any(sources.get(name) != value for name, value in report["sources"].items()):
        raise ValueError("Reference bridge source pins differ from current sources")
    if (report.get("determinism", {}).get("deterministic_algorithms") is not True
            or not report.get("integrity") or not all(report["integrity"].values())):
        raise ValueError("Reference lacks deterministic fixed-state integrity")
    for path_name in PATHS:
        reference_anchor(report, path_name)
    return report


def reference_anchor(report, path):
    if path not in PATHS:
        raise ValueError("Unsupported recurrence diagnostic path")
    rows = [row for row in report["rows"] if row.get("objective") == "ce" and row.get("path") == path]
    if len(rows) != 1 or rows[0].get("passed") is not True:
        raise ValueError("Reference requires exactly one successful CE endpoint per precision")
    return rows[0]


def anchor_comparison(metrics, fingerprints, gradients, forward, previous):
    """Exact retained summaries and precision geometry, not unsaved vector bytes."""
    key = "gradients" if previous["path"] == FP32 else "gradients_vs_fp32"
    old = previous[key]
    checks = {"metrics_exact": metrics == previous["metrics"],
              "forward_fingerprints_exact": fingerprints == previous["forward_fingerprints"],
              "gradient_group_summaries_exact": all(gradients[k] == old[k] for k in GRADIENT_SUMMARY_KEYS),
              "forward_precision_geometry_exact": forward == previous["forward_vs_fp32"]}
    if previous["path"] == BF16:
        checks["gradient_precision_geometry_exact"] = gradients["geometry"] == old["geometry"]
        checks["per_parameter_precision_summaries_exact"] = all(
            gradients["comparison"][k] == old["comparison"][k]
            for k in ("relative_l2", "parameter_rows", "all_parameters_close", "atol", "rtol"))
    return {"passed": all(checks.values()), "checks": checks,
            "scope": "Prior NFR CE endpoints and precision geometry; original full gradient vectors were not retained"}


def state_pins(model):
    """Include dormant fusion parameters and its measured output_scale buffer."""
    result = {}
    for group in ("backbone", "fusion", "predictor"):
        result[group] = {
            "parameters": tree_digests({n: p for n, p in model.named_parameters() if component(n) == group}),
            "buffers": tree_digests({n: b for n, b in model.named_buffers() if component(n) == group})}
    return result


def arm_contract(model, recipe):
    mode = recipe.mode()
    if (recipe.arm not in ARMS or not recipe.nextlat or not model.enabled or model.predictor is None
            or model.pass_loss_policy != "campaign_v1"
            or model.config.seed != recipe.predictor_seed
            or model.backbone.fusion_config.seed != recipe.fusion_seed
            or dict(model.objective_weights()) != dict.fromkeys(TERMS, 1.0)):
        raise ValueError("Diagnostic requires active campaign-v1 NextLat branches")
    if (mode.num_passes != (4 if recipe.feedback else 1)
            or mode.enabled != recipe.feedback or mode.first_pass_policy != "configured-rt-v1"
            or mode.rt_mode.selected_layers != (recipe.rt_layers if "R" in recipe.arm else ())
            or mode.rt_mode.alpha != 1.0 or mode.beta != 1.0
            or mode.feedback_jitter != (recipe.feedback_jitter if recipe.feedback else 0.0)):
        raise ValueError("Recurrence mode differs from the declared arm")
    parameters = dict(model.named_parameters())
    if any(p.dtype != torch.float32 or p.requires_grad != (component(n) != "fusion" or recipe.feedback)
           for n, p in parameters.items()):
        raise ValueError("Parameter dtype/trainability differs from the arm")
    return {"arm": recipe.arm, "mode": asdict(mode), "nextlat_enabled": True,
        "objective": "ce", "auxiliary_cotangents": {"latent": 0.0, "kl": 0.0},
        "ce_pass_weights": [1.0] if not recipe.feedback else [.5, 1/6, 1/6, 1/6],
        "auxiliary_pass_weights_before_zero_cotangent": [1/mode.num_passes]*mode.num_passes,
        "denominator_scope": "Global selected positions across both physical records, counted once, not once per pass",
        "predictor_seed": recipe.predictor_seed, "fusion_seed": recipe.fusion_seed,
        "jitter_seed": recipe.jitter_seed, "parameter_inventory": {
            group: {"resident_parameters": sum(p.numel() for n, p in parameters.items() if component(n) == group),
                    "trainable_parameters": sum(p.numel() for n, p in parameters.items() if component(n) == group and p.requires_grad),
                    "trainable_tensors": sum(component(n) == group and p.requires_grad for n, p in parameters.items())}
            for group in ("backbone", "fusion", "predictor")}}


def fixture_pins(fixtures):
    return tree_digests([{"batches": [vars(batch) for batch in batches], "noise": noises}
                         for batches, noises in fixtures])


def first_pass_fingerprints(records):
    return tree_digests([{"batch": record["batch"], "token_embeddings": record["token_embeddings"],
                         "first_pass_hidden": record["pass_hidden_states"][0]} for record in records])


def shared_arm_checks(current, baseline, feedback_baseline=None):
    """Architecture/objectives differ; common weights and row occurrences do not."""
    checks = {"checkpoint_exact": current["source_checkpoint"] == baseline["source_checkpoint"],
              "model_config_exact": current["model_config"] == baseline["model_config"],
              "nextlat_config_exact": current["nextlat_config"] == baseline["nextlat_config"],
              "production_flags_exact": current["production_runtime_flags"] == baseline["production_runtime_flags"],
              "metadata_exact": current["fixture_metadata"] == baseline["fixture_metadata"],
              "tokens_masks_exact": [r["batches"] for r in current["fixture_inputs"]]
                                     == [r["batches"] for r in baseline["fixture_inputs"]]}
    for group in ("backbone", "predictor", "fusion"):
        checks[group+"_state_exact"] = current["initial_state"][group] == baseline["initial_state"][group]
    # Recipe differences are explicit: pass count and arm, not unrelated knobs.
    checks["shared_recipe_exact"] = {k: v for k, v in current["recipe"].items() if k not in ("arm", "fbt_passes")} == {
        k: v for k, v in baseline["recipe"].items() if k not in ("arm", "fbt_passes")}
    if current["contract"]["mode"]["enabled"]:
        checks["feedback_noise_present"] = all(len(r["noise"]) == 1 and len(r["noise"][0] or ()) == 3
                                                for r in current["fixture_inputs"])
        if feedback_baseline is not None:
            checks["common_feedback_noise_exact"] = [r["noise"] for r in current["fixture_inputs"]] == [
                r["noise"] for r in feedback_baseline["fixture_inputs"]]
    else:
        checks["feedback_noise_absent"] = all(r["noise"] == [None] for r in current["fixture_inputs"])
    return checks


def gradient_norm_summary(gradients):
    groups = {group: record["norm"] for group, record in gradients["groups"].items()}
    return {"all": math.sqrt(sum(value*value for value in groups.values())), **groups}


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)  # Before CUDA initialization.
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Recurrence precision diagnostic requires one process without DDP")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-campaign-recurrence-precision-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "sources": source_hashes(), "determinism": determinism, "rows": [], "arms": {}, "paths": PATHS,
        "reference_report": str(args.reference_report), "reference_sha256": args.reference_sha256,
        "objective": "ce", "aggregate_backwards": 8, "physical_batch_backwards": 16, "optimizer_updates": 0,
        "fixture": "Original isolated-v1 T16 B2, two virtual-rank records, update0; recipe remains T1024",
        "scope": __doc__, "qualification": "Numerical errors are descriptive; prior qualifications remain open",
        "cotangent_scope": "TOTAL incoming pass-output gradients, including later FBT feedback, not common cotangents across arms",
        "cross_arm_norm_scope": "Descriptive only; different architectures and CE pass weights prevent a causal norm comparison",
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for relative in report["sources"]:
        destination = args.output_dir/"source-snapshot"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-recurrence-precision", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({key: report[key] for key in ("scope", "paths", "determinism", "reference_sha256", "qualification")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("verify_reference")
        previous = load_reference(args.reference_report, args.reference_sha256, report["sources"])
        runtime_checks = {"runtime_exact": runtime == previous["runtime"],
            "determinism_exact": determinism == previous["determinism"],
            "math_reduction_exact": report["math_sdpa_reduced_precision_reduction"] == previous["math_sdpa_reduced_precision_reduction"],
            "bf16_matmul_reduction_exact": report["bf16_matmul_reduced_precision_reduction"] == previous["bf16_matmul_reduced_precision_reduction"]}
        report["reference_runtime_checks"] = runtime_checks
        if not all(runtime_checks.values()):
            raise AssertionError("Runtime differs from pinned bridge")
        first_pass_pins = {}
        for arm in ARMS:
            persist(arm+"/loading_source")
            model, recipe, checkpoint, ids, eos = construct(
                SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), arm, torch.device("cuda"))
            original = {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS}
            fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
                length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
            info = {"contract": arm_contract(model, recipe), "source_checkpoint": checkpoint,
                "model_config": model.backbone.backbone.config.to_dict(), "nextlat_config": model.config.to_dict(),
                "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256,
                "initial_state": state_pins(model), "production_runtime_flags": original,
                "fixture_inputs": fixture_pins(fixtures), "fixture_metadata": global_fixture_metadata(model, fixtures)}
            report["arms"][arm] = info
            info["shared_state_checks"] = shared_arm_checks(info, report["arms"]["N"], report["arms"].get("NF"))
            if not all(info["shared_state_checks"].values()):
                raise AssertionError("Common arm weights/data/configuration differ")
            if arm == "NFR":
                info["previous_state_checks"] = {"checkpoint_exact": checkpoint == previous["source_checkpoint"],
                    "recipe_exact": recipe.to_dict() == previous["recipe"] and recipe.sha256 == previous["recipe_sha256"],
                    "fixture_inputs_exact": info["fixture_inputs"] == previous["fixture_inputs"],
                    "production_flags_exact": original == previous["production_runtime_flags"]}
                if not all(info["previous_state_checks"].values()):
                    raise AssertionError("NFR state/input contract differs from pinned bridge")
            rng = rng_snapshot()
            reference_gradients, reference_forward = None, None
            for path in PATHS:
                execution = configure_path(model, original, path)
                persist(arm+"/"+path+"/backward")
                torch.cuda.reset_peak_memory_stats()
                case_started = time.monotonic()
                backend = SDPBackend.MATH if path == FP32 else SDPBackend.FLASH_ATTENTION
                with capture_passes(model, fixtures) as observed, sdpa_kernel(backend):
                    metrics = component_backward(model, recipe, fixtures,
                        precision=execution["precision"], layout="sparse", objective="ce")
                gradients, snapshot = record_gradients(model, reference_gradients, save_cpu=path == FP32,
                    scope="Descriptive within-arm BF16 Flash/Triton versus FP32 math/eager; no numerical gate")
                geometry = forward_geometry(observed, reference_forward)
                fingerprints = tree_digests([{key: row[key] for key in ("batch", "token_embeddings", "pass_hidden_states")}
                                             for row in observed])
                first = first_pass_fingerprints(observed)
                first_pass_pins[(arm, path)] = first
                health = {"finite_gradients": gradients["finite"], "gradient_participation": gradients["participation_intact"],
                    "finite_forward_and_cotangents": all(row["hidden"]["finite"] and row["total_incoming_cotangent"]["finite"]
                        for row in geometry if not row.get("skipped_dummy")),
                    "finite_losses": all(math.isfinite(value) for value in (metrics["objective"], *metrics["loss_sums"].values())),
                    "correct_pass_count": len(observed) == 2 and all(len(row["pass_hidden_states"]) == recipe.mode().num_passes for row in observed),
                    "counts_exact": metrics["counts"] == info["fixture_metadata"]["counts"],
                    "predictor_zero_cotangent": gradients["groups"]["predictor"]["norm"] == 0.0,
                    "rng_unchanged": rng_unchanged(rng)}
                if recipe.feedback:
                    health["first_pass_matches_standalone_exactly"] = first == first_pass_pins[("NR" if arm == "NFR" else "N", path)]
                row = {"arm": arm, "path": path, "objective": "ce", "execution": execution, "metrics": metrics,
                    "ce_pass_weights": info["contract"]["ce_pass_weights"],
                    "normalized_loss_means": {term: metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
                    "gradients" if path == FP32 else "gradients_vs_fp32": gradients,
                    "forward_vs_fp32": geometry, "forward_fingerprints": fingerprints, "first_pass_fingerprints": first,
                    "gradient_norms_descriptive_only": gradient_norm_summary(gradients),
                    "health": health, "elapsed_seconds_including_observation": time.monotonic()-case_started,
                    "memory": memory(), "passed": all(health.values())}
                if arm == "NFR":
                    row["previous_anchor"] = anchor_comparison(metrics, fingerprints, gradients, geometry, reference_anchor(previous, path))
                    row["passed"] &= row["previous_anchor"]["passed"]
                report["rows"].append(row)
                persist(arm+"/"+path)
                tracker.log(scalar_metrics(row, "diagnostic/"+arm+"/"+path), step=len(report["rows"]))
                print({"stage": report["stage"], "operational_passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
                if not row["passed"]:
                    raise AssertionError("Arm health, common first pass or NFR anchor guard failed")
                if path == FP32:
                    reference_gradients, reference_forward = snapshot, observed
            info["integrity"] = {"parameters_and_buffers_unchanged": state_pins(model) == info["initial_state"],
                "inputs_unchanged": fixture_pins(fixtures) == info["fixture_inputs"], "rng_unchanged": rng_unchanged(rng),
                "arm_configuration_unchanged": arm_contract(model, recipe) == info["contract"]}
            if not all(info["integrity"].values()):
                raise AssertionError("Arm fixed state changed")
            # Only small fingerprints/reports survive between architectures.
            model.zero_grad(set_to_none=True)
            del model, fixtures, reference_gradients, reference_forward, snapshot, observed
            gc.collect()
            torch.cuda.empty_cache()
            persist(arm+"/integrity")
        report["integrity"] = {"sources_unchanged": source_hashes() == report["sources"],
            "reference_unchanged": sha256_file(args.reference_report) == args.reference_sha256,
            "all_arms_fixed": all(all(info["integrity"].values()) for info in report["arms"].values()),
            "eight_cases_complete": len(report["rows"]) == 8 and all(row["passed"] for row in report["rows"])}
        if not all(report["integrity"].values()):
            raise AssertionError("Final integrity failed")
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
