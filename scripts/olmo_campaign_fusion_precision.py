#!/usr/bin/env python3
"""Three NF CE cases isolating the precision of FBTGateProduct.forward only.

Reproduce FP32 math/eager and BF16 Flash anchors, then keep production BF16
everywhere except feedback fusion. This temporary module-local override uses
the original FP32 input/master/output dtypes and disables autocast only within
the original fusion forward. Six physical backwards; no optimizer/DDP/graphs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
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
from cdrm.pretrained.olmo_fbt import FBTGateProduct
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_bridge import capture_passes, configure_path, forward_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS, component_backward, record_gradients
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_recurrence_precision import (
    FP32, BF16, anchor_comparison as recurrence_anchor_comparison, arm_contract,
    first_pass_fingerprints, fixture_pins, gradient_norm_summary, state_pins,
    source_hashes as recurrence_sources,
)
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

CANDIDATE = "bf16_flash_fp32_fusion"
PATHS = (FP32, BF16, CANDIDATE)


@contextmanager
def fp32_fusion_forward(fusion):
    """Override this instance only, preserving the original input/output dtype.

No tensor is promoted or detached. Non-FP32 inputs/masters fail instead of
silently changing this experiment. Restore even an existing instance override
on exceptions; leave the class and global autocast policy untouched.
"""
    if not isinstance(fusion, FBTGateProduct):
        raise TypeError("Fusion precision override requires FBTGateProduct")
    if any(p.dtype != torch.float32 for p in fusion.parameters()):
        raise ValueError("Fusion precision override requires FP32 masters")
    original = fusion.forward
    had_override = "forward" in vars(fusion)
    original_override = vars(fusion).get("forward")
    evidence = {"calls": [], "restored": False, "scope": "Only this FBTGateProduct.forward instance"}

    def forward(previous_hidden, token_input):
        if previous_hidden.dtype != torch.float32 or token_input.dtype != torch.float32:
            raise ValueError("Fusion override requires actual FP32 inputs; no promotion is permitted")
        if previous_hidden.device != token_input.device:
            raise ValueError("Fusion inputs must share a device")
        device = previous_hidden.device.type
        row = {"previous_hidden_dtype": str(previous_hidden.dtype), "token_input_dtype": str(token_input.dtype),
               "outer_autocast_enabled": torch.is_autocast_enabled(device),
               "input_shape": list(previous_hidden.shape)}
        with torch.autocast(device, enabled=False):
            row["inner_autocast_enabled"] = torch.is_autocast_enabled(device)
            result = original(previous_hidden, token_input)
        if result.dtype != token_input.dtype:
            raise AssertionError("Fusion override changed the original return dtype")
        row["output_dtype"] = str(result.dtype)
        row["outer_autocast_restored"] = torch.is_autocast_enabled(device) == row["outer_autocast_enabled"]
        evidence["calls"].append(row)
        return result

    fusion.forward = forward
    try:
        yield evidence
    finally:
        if had_override:
            fusion.forward = original_override
        else:
            del fusion.forward
        evidence["restored"] = fusion.forward == original and ("forward" in vars(fusion)) == had_override


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
    sources = recurrence_sources()
    for name in ("scripts/olmo_campaign_fusion_precision.py", "tests/test_campaign_fusion_precision.py",
                 "docs/reports/olmo-recurrence-precision/fusion-protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def reference_anchor(report, path):
    if path not in (FP32, BF16):
        raise ValueError("Candidate has no previous endpoint")
    rows = [row for row in report["rows"] if row.get("arm") == "NF"
            and row.get("objective") == "ce" and row.get("path") == path]
    if len(rows) != 1 or rows[0].get("passed") is not True:
        raise ValueError("Matrix needs exactly one successful NF CE endpoint per precision")
    return rows[0]


def load_reference(path, expected_sha, sources):
    path = Path(path)
    if path.stat().st_size > 64*1024*1024 or sha256_file(path) != expected_sha:
        raise ValueError("Reference bytes differ from the bounded SHA256 pin")
    report = json.loads(path.read_text())
    if sha256_file(path) != expected_sha:
        raise ValueError("Reference changed while reading")
    if (report.get("schema") != "olmo-campaign-recurrence-precision-v1"
            or report.get("status") != "passed_operational_diagnostic" or report.get("passed") is not True):
        raise ValueError("Reference must be a completed recurrence matrix")
    if not report.get("sources") or any(sources.get(name) != value for name, value in report["sources"].items()):
        raise ValueError("Reference source pins differ from current sources")
    info = report.get("arms", {}).get("NF", {})
    if (report.get("determinism", {}).get("deterministic_algorithms") is not True
            or not report.get("integrity") or not all(report["integrity"].values())
            or not info.get("integrity") or not all(info["integrity"].values())
            or not info.get("shared_state_checks") or not all(info["shared_state_checks"].values())):
        raise ValueError("Reference lacks NF deterministic fixed-state integrity")
    for path_name in (FP32, BF16):
        reference_anchor(report, path_name)
    return report


def anchor_comparison(metrics, fingerprints, gradients, forward, previous):
    result = recurrence_anchor_comparison(metrics, fingerprints, gradients, forward, previous)
    result["scope"] = "Prior NF CE endpoints and precision geometry; original full gradient vectors were not retained"
    return result


def state_contract_checks(current, previous):
    # Dataclass mode tuples serialize as JSON arrays in the retained matrix.
    return {key+"_exact": tree_digests(value) == previous[key] for key, value in current.items()}


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Fusion precision diagnostic uses one process without DDP")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-campaign-fusion-precision-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "sources": source_hashes(), "determinism": determinism, "rows": [], "paths": PATHS,
        "reference_report": str(args.reference_report), "reference_sha256": args.reference_sha256,
        "objective": "ce", "arm": "NF", "aggregate_backwards": 3, "physical_batch_backwards": 6, "optimizer_updates": 0,
        "scope": __doc__, "qualification": "Numerical errors descriptive; no new acceptance budget or production-policy change",
        "cotangent_scope": "TOTAL incoming gradients at valid pass-output positions, including later feedback",
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for name in report["sources"]:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-recurrence-precision", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({key: report[key] for key in ("scope", "paths", "determinism", "reference_sha256", "qualification")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("verify_reference_and_load_source")
        previous = load_reference(args.reference_report, args.reference_sha256, report["sources"])
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NF", torch.device("cuda"))
        original = {key: getattr(model.backbone.backbone, key) for key in RUNTIME_FLAGS}
        fixtures = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
            length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
        info = {"contract": arm_contract(model, recipe), "source_checkpoint": checkpoint,
            "model_config": model.backbone.backbone.config.to_dict(), "nextlat_config": model.config.to_dict(),
            "recipe": recipe.to_dict(), "recipe_sha256": recipe.sha256,
            "initial_state": state_pins(model), "production_runtime_flags": original,
            "fixture_inputs": fixture_pins(fixtures), "fixture_metadata": global_fixture_metadata(model, fixtures)}
        checks = state_contract_checks(info, previous["arms"]["NF"])
        checks.update(runtime_exact=runtime == previous["runtime"], determinism_exact=determinism == previous["determinism"],
            math_reduction_exact=report["math_sdpa_reduced_precision_reduction"] == previous["math_sdpa_reduced_precision_reduction"],
            bf16_matmul_reduction_exact=report["bf16_matmul_reduced_precision_reduction"] == previous["bf16_matmul_reduced_precision_reduction"])
        report.update(arm_state=info, reference_contract_checks=checks)
        if not all(checks.values()):
            raise AssertionError("NF state/input/runtime contract differs from pinned matrix")
        rng = rng_snapshot()
        reference_gradients, reference_forward = {}, {}
        original_forward = model.backbone.fusion.forward
        for path in PATHS:
            execution = configure_path(model, original, BF16 if path == CANDIDATE else path)
            execution["fusion_precision"] = "autocast_disabled_only_inside_FBTGateProduct.forward" if path == CANDIDATE else "unchanged"
            persist(path+"/backward")
            torch.cuda.reset_peak_memory_stats()
            case_started = time.monotonic()
            backend = SDPBackend.MATH if path == FP32 else SDPBackend.FLASH_ATTENTION
            override = fp32_fusion_forward(model.backbone.fusion) if path == CANDIDATE else nullcontext(None)
            with override as fusion_evidence, capture_passes(model, fixtures) as observed, sdpa_kernel(backend):
                metrics = component_backward(model, recipe, fixtures, precision=execution["precision"], layout="sparse", objective="ce")
            gradients, snapshot = record_gradients(model, reference_gradients.get(FP32), save_cpu=path != CANDIDATE,
                scope="Descriptive versus matched NF full FP32 reference; no numerical gate")
            geometry = forward_geometry(observed, reference_forward.get(FP32))
            fingerprints = tree_digests([{key: row[key] for key in ("batch", "token_embeddings", "pass_hidden_states")} for row in observed])
            first = first_pass_fingerprints(observed)
            health = {"finite_gradients": gradients["finite"], "gradient_participation": gradients["participation_intact"],
                "finite_forward_and_cotangents": all(row["hidden"]["finite"] and row["total_incoming_cotangent"]["finite"]
                    for row in geometry if not row.get("skipped_dummy")),
                "finite_losses": all(math.isfinite(value) for value in (metrics["objective"], *metrics["loss_sums"].values())),
                "four_passes_each_record": len(observed) == 2 and all(len(row["pass_hidden_states"]) == 4 for row in observed),
                "counts_exact": metrics["counts"] == info["fixture_metadata"]["counts"],
                "predictor_zero_cotangent": gradients["groups"]["predictor"]["norm"] == 0.0,
                "fusion_forward_restored": model.backbone.fusion.forward == original_forward,
                "rng_unchanged": rng_unchanged(rng)}
            row = {"path": path, "objective": "ce", "execution": execution, "metrics": metrics,
                "ce_pass_weights": info["contract"]["ce_pass_weights"],
                "normalized_loss_means": {term: metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
                "gradients" if path == FP32 else "gradients_vs_fp32": gradients,
                "forward_vs_fp32": geometry, "forward_fingerprints": fingerprints, "first_pass_fingerprints": first,
                "gradient_norms_descriptive_only": gradient_norm_summary(gradients), "health": health,
                "elapsed_seconds_including_observation": time.monotonic()-case_started, "memory": memory()}
            if path == CANDIDATE:
                health["first_pass_matches_production_exactly"] = first == first_pass_fingerprints(reference_forward[BF16])
                health["fusion_override_executed_exact_scope"] = fusion_evidence["restored"] and len(fusion_evidence["calls"]) == 6 and all(
                    call["outer_autocast_enabled"] and not call["inner_autocast_enabled"] and call["outer_autocast_restored"]
                    and call["output_dtype"] == "torch.float32" for call in fusion_evidence["calls"])
                versus, _ = record_gradients(model, reference_gradients[BF16],
                    scope="Descriptive fusion-only candidate versus matched NF production BF16; no numerical gate")
                row.update(fusion_override=fusion_evidence, gradients_vs_production_bf16=versus,
                           forward_vs_production_bf16=forward_geometry(observed, reference_forward[BF16]))
            else:
                row["previous_anchor"] = anchor_comparison(metrics, fingerprints, gradients, geometry, reference_anchor(previous, path))
                health["previous_anchor_exact"] = row["previous_anchor"]["passed"]
                reference_gradients[path], reference_forward[path] = snapshot, observed
            row["passed"] = all(health.values())
            report["rows"].append(row)
            persist(path)
            tracker.log(scalar_metrics(row, "diagnostic/"+path), step=len(report["rows"]))
            print({"stage": report["stage"], "operational_passed": row["passed"], "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
            if not row["passed"]:
                raise AssertionError("NF anchor, override-scope or health guard failed")
        report["integrity"] = {"parameters_and_buffers_unchanged": state_pins(model) == info["initial_state"],
            "inputs_unchanged": fixture_pins(fixtures) == info["fixture_inputs"], "rng_unchanged": rng_unchanged(rng),
            "arm_configuration_unchanged": arm_contract(model, recipe) == info["contract"],
            "fusion_forward_restored": model.backbone.fusion.forward == original_forward,
            "sources_unchanged": source_hashes() == report["sources"],
            "reference_unchanged": sha256_file(args.reference_report) == args.reference_sha256}
        if not all(report["integrity"].values()):
            raise AssertionError("Fixed diagnostic state changed")
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
