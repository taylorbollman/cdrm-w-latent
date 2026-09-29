#!/usr/bin/env python3
"""Two crossed backbone/fusion states, each with matched NF FP32/BF16 CE VJPs.

Four aggregate cases/eight physical backwards, no training. Complete cold and
adapted fusion states include output_scale. Diagonal endpoints are retained
context, not rerun; this is a transfer/coadaptation diagnostic, not an additive
causal decomposition or a numerical clearance for production training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
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
from scripts.olmo_campaign_adapted_import import endpoint_authority, load_into_current
from scripts.olmo_campaign_adapted_precision import (
    REFERENCE_SHA, case_health, fixed_contract_checks, source_hashes as adapted_sources,
)
from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update, global_fixture_metadata
from scripts.olmo_campaign_fusion_precision import load_reference, reference_anchor, state_contract_checks
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_position_geometry import position_geometry
from scripts.olmo_campaign_precision_bridge import capture_passes, configure_path, forward_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS, component_backward, record_gradients
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, arm_contract, fixture_pins, gradient_norm_summary, state_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

ADAPTED_SHA = "6af988581eac19a2d74dcb32558b63c80911f44569ee9dcd2db442dceb0dbfa4"
PATHS = (FP32, BF16)
HYBRIDS = {
    "cold_backbone_adapted_fusion": ("cold", "adapted"),
    "adapted_backbone_cold_fusion": ("adapted", "cold"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path,
        default=ROOT/".runtime/olmo-recurrence-precision/matrix-01/report.json")
    parser.add_argument("--adapted-report", type=Path,
        default=ROOT/".runtime/olmo-adapted-precision/adapted-01/report.json")
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
    sources = adapted_sources()
    for name in ("scripts/olmo_campaign_crossed_precision.py", "tests/test_campaign_crossed_precision.py",
                 "scripts/olmo_campaign_position_geometry.py", "tests/test_campaign_position_geometry.py",
                 "docs/reports/olmo-crossed-precision/protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def adapted_anchor(report, path):
    if path not in PATHS:
        raise ValueError("Unknown adapted endpoint precision")
    rows = [row for row in report["rows"] if row.get("path") == path and row.get("objective") == "ce"]
    if len(rows) != 1 or rows[0].get("passed") is not True:
        raise ValueError("Require one successful adapted CE endpoint per precision")
    return rows[0]


def load_adapted_reference(path, sources):
    path = Path(path)
    if (path.is_symlink() or path.stat().st_size > 64*1024*1024
            or sha256_file(path) != ADAPTED_SHA):
        raise ValueError("Adapted report differs from its bounded immutable pin")
    report = json.loads(path.read_text())
    if sha256_file(path) != ADAPTED_SHA:
        raise ValueError("Adapted report changed while reading")
    if (report.get("schema") != "olmo-campaign-adapted-precision-v1"
            or report.get("status") != "passed_operational_diagnostic" or report.get("passed") is not True
            or report.get("arm") != "NF" or report.get("objective") != "ce"
            or report.get("aggregate_backwards") != 2 or report.get("physical_batch_backwards") != 4
            or report.get("optimizer_updates") != 0 or report.get("reference_sha256") != REFERENCE_SHA
            or report.get("determinism", {}).get("deterministic_algorithms") is not True):
        raise ValueError("Require the completed matched adapted NF diagnostic")
    if not report.get("sources") or any(sources.get(n) != h for n, h in report["sources"].items()):
        raise ValueError("Adapted reference source inventory differs")
    for checks in (report.get("cold_contract_checks"), report.get("import_contract_checks"),
                   report.get("integrity"), report.get("adapted_import", {}).get("checks")):
        if not checks or not all(value is True for value in checks.values()):
            raise ValueError("Adapted reference lacks exact import/fixed-state integrity")
    for path_name in PATHS:
        row = adapted_anchor(report, path_name)
        if not row.get("health") or not all(value is True for value in row["health"].values()):
            raise ValueError("Adapted reference endpoint health is incomplete")
    return report


def snapshot_components(model):
    """Independent CPU copies; never store live module references or predictor bytes."""
    tensors = {
        "backbone": {n: t.detach().to("cpu", copy=True) for n, t in model.backbone.backbone.state_dict().items()},
        "fusion": {n: t.detach().to("cpu", copy=True) for n, t in model.backbone.fusion.state_dict().items()},
    }
    if "output_scale" not in tensors["fusion"]:
        raise ValueError("Complete fusion snapshot must include output_scale")
    return {"tensors": tensors, "tensor_pins": tree_digests(tensors), "original_state_pins": state_pins(model)}


def snapshots_unchanged(snapshots):
    return {origin: tree_digests(value["tensors"]) == value["tensor_pins"] for origin, value in snapshots.items()}


def assemble_hybrid(model, snapshots, *, backbone_origin, fusion_origin):
    """Prevalidate both complete source components, then copy into existing owners.

Only the two designated crossed combinations are supported. Metadata, predictor,
training flags, tied ownership and parameter identities stay current. RNG is
guarded by the caller around all assembly and execution; this helper draws none.
"""
    if (backbone_origin, fusion_origin) not in HYBRIDS.values() or set(snapshots) != {"cold", "adapted"}:
        raise ValueError("Require one of the two declared crossed source selections")
    if any(p.dtype != torch.float32 or not p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Hybrid target requires trainable FP32 masters and cleared gradients")
    if model.backbone.readout_weight is not model.backbone.token_embeddings.weight:
        raise ValueError("Hybrid target lost tied embedding/readout ownership")
    before = state_pins(model)
    if any(before["predictor"] != snapshot["original_state_pins"]["predictor"] for snapshot in snapshots.values()):
        raise ValueError("Hybrid predictor differs from the common fresh source predictor")
    identities = {n: id(p) for n, p in model.named_parameters()}
    modes = {n: m.training for n, m in model.named_modules()}
    modules = {"backbone": model.backbone.backbone, "fusion": model.backbone.fusion}
    selected = {"backbone": backbone_origin, "fusion": fusion_origin}
    for group, origin in selected.items():
        snapshot = snapshots[origin]
        tensors = snapshot["tensors"][group]
        target = modules[group].state_dict()
        if tree_digests(tensors) != snapshot["tensor_pins"][group]:
            raise ValueError("Source snapshot bytes changed before assembly")
        if set(tensors) != set(target) or (group == "fusion" and "output_scale" not in tensors):
            raise ValueError("Hybrid source tensor inventory differs")
        for name, tensor in tensors.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.dtype != target[name].dtype or tensor.shape != target[name].shape
                    or tensor.requires_grad or not bool(torch.isfinite(tensor).all())):
                raise ValueError("Hybrid source tensor shape/dtype/device/finiteness differs")
    # Validation above completes for BOTH components before any target mutation.
    for group, origin in selected.items():
        modules[group].load_state_dict(snapshots[origin]["tensors"][group], strict=True, assign=False)
    after = state_pins(model)
    checks = {group+"_selected_state_exact": after[group] == snapshots[origin]["original_state_pins"][group]
              for group, origin in selected.items()}
    checks.update({
        "predictor_unchanged": after["predictor"] == before["predictor"],
        "predictor_matches_both_origins": all(after["predictor"] == s["original_state_pins"]["predictor"] for s in snapshots.values()),
        "parameter_identities_preserved": identities == {n: id(p) for n, p in model.named_parameters()},
        "training_modes_preserved": modes == {n: m.training for n, m in model.named_modules()},
        "tied_readout_preserved": model.backbone.readout_weight is model.backbone.token_embeddings.weight,
        "trainable_fp32_masters": all(p.requires_grad and p.dtype == torch.float32 and p.grad is None for p in model.parameters()),
    })
    if not all(checks.values()):
        raise AssertionError("Hybrid assembly violated an exact state/ownership guard")
    return {"backbone_origin": backbone_origin, "fusion_origin": fusion_origin,
        "complete_fusion_including_output_scale": True, "initial_state": after, "checks": checks}


def first_pass_identity(fingerprints, diagonal):
    """Fusion cannot affect embeddings/pass zero; incoming cotangents may differ."""
    def selected(rows):
        return [{"batch": r["batch"], "token_embeddings": r["token_embeddings"],
                 "pass_zero": r["pass_hidden_states"][0]} for r in rows]
    return selected(fingerprints) == selected(diagonal["forward_fingerprints"])


def main(argv=None):
    args = parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Crossed diagnostic uses one process without DDP")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-campaign-crossed-precision-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "sources": source_hashes(), "determinism": determinism, "rows": [], "hybrids": {}, "paths": PATHS,
        "reference_report": str(args.reference_report), "reference_sha256": REFERENCE_SHA,
        "adapted_report": str(args.adapted_report), "adapted_sha256": ADAPTED_SHA,
        "objective": "ce", "arm": "NF", "aggregate_backwards": 4, "physical_batch_backwards": 8,
        "optimizer_updates": 0, "scope": __doc__,
        "qualification": "Within-hybrid precision comparisons only; no new budget, training-policy change or BF16 clearance",
        "diagonal_scope": "Prior cold/cold and adapted/adapted reports reused; no new diagonal backwards",
        "cotangent_scope": "Actual TOTAL incoming pass-output gradients, including later feedback; not common cotangents between precisions",
        "position_scope": "Valid positions, direct CE mask and feedback eligibility, plus union of nonzero incoming-cotangent support; no retrospective localization of prior adapted tensors",
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for name in report["sources"]:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-crossed-precision", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({key: report[key] for key in ("scope", "paths", "determinism", "reference_sha256", "adapted_sha256", "qualification")})
        print({"wandb": tracker.record["run_url"]}, flush=True)
        persist("verify_diagonals_and_construct_cold")
        previous = load_reference(args.reference_report, REFERENCE_SHA, report["sources"])
        previous_adapted = load_adapted_reference(args.adapted_report, report["sources"])
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
        checks["cold_construction_matches_adapted_report"] = tree_digests(cold) == previous_adapted["cold_construction"]
        for label, diagonal in (("matrix", previous), ("adapted", previous_adapted)):
            for key in ("runtime", "determinism", "math_sdpa_reduced_precision_reduction", "bf16_matmul_reduced_precision_reduction"):
                checks[label+"_"+key+"_exact"] = report[key] == diagonal[key]
        report.update(cold_construction=cold, cold_contract_checks=checks)
        if not all(checks.values()):
            raise AssertionError("Cold NF construction/fixture/runtime differs from pinned diagonals")
        rng = rng_snapshot()
        snapshots = {"cold": snapshot_components(model)}
        persist("import_and_verify_adapted_weights")
        report["adapted_import"] = load_into_current(model, args.checkpoint, authority)
        adapted = info()
        checks = fixed_contract_checks(adapted, cold)
        checks.update({"adapted_state_matches_prior_report": tree_digests(adapted) == previous_adapted["adapted_state"],
            "adapted_import_matches_prior_report": tree_digests(report["adapted_import"]) == previous_adapted["adapted_import"],
            "rng_unchanged": rng_unchanged(rng)})
        report.update(adapted_state=adapted, import_contract_checks=checks)
        if not all(checks.values()):
            raise AssertionError("Adapted state/import differs from pinned completed endpoint")
        snapshots["adapted"] = snapshot_components(model)
        report["component_snapshots"] = {origin: {key: value for key, value in snapshot.items() if key != "tensors"}
                                         for origin, snapshot in snapshots.items()}
        for hybrid, (backbone_origin, fusion_origin) in HYBRIDS.items():
            model.zero_grad(set_to_none=True)
            configure_path(model, original, BF16)
            persist(hybrid+"/assemble")
            assembly = assemble_hybrid(model, snapshots, backbone_origin=backbone_origin, fusion_origin=fusion_origin)
            hybrid_info = info()
            fixed = fixed_contract_checks(hybrid_info, cold)
            fixed["rng_unchanged"] = rng_unchanged(rng)
            report["hybrids"][hybrid] = {"assembly": assembly, "state": hybrid_info, "fixed_contract_checks": fixed}
            if not all(fixed.values()):
                raise AssertionError("Hybrid changed fixture/configuration/predictor/RNG")
            reference_gradients, reference_forward = None, None
            for path in PATHS:
                execution = configure_path(model, original, path)
                persist(hybrid+"/"+path+"/backward")
                torch.cuda.reset_peak_memory_stats()
                case_started = time.monotonic()
                backend = SDPBackend.MATH if path == FP32 else SDPBackend.FLASH_ATTENTION
                with capture_passes(model, fixtures) as observed, sdpa_kernel(backend):
                    metrics = component_backward(model, recipe, fixtures, precision=execution["precision"], layout="sparse", objective="ce")
                gradients, snapshot = record_gradients(model, reference_gradients, save_cpu=path == FP32,
                    scope="Descriptive versus this hybrid's matched full FP32 reference; no numerical gate")
                geometry = forward_geometry(observed, reference_forward)
                positions = position_geometry(observed, reference_forward, document_policy="isolated-v1")
                fingerprints = tree_digests([{key: row[key] for key in ("batch", "token_embeddings", "pass_hidden_states")} for row in observed])
                health = case_health(metrics, gradients, geometry, observed, cold["fixture_metadata"])
                health["rng_unchanged"] = rng_unchanged(rng)
                diagonal = reference_anchor(previous, path) if backbone_origin == "cold" else adapted_anchor(previous_adapted, path)
                pass_zero = first_pass_identity(fingerprints, diagonal)
                row = {"hybrid": hybrid, "path": path, "objective": "ce", "execution": execution,
                    "metrics": metrics, "ce_pass_weights": cold["contract"]["ce_pass_weights"],
                    "normalized_loss_means": {term: metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
                    "gradients" if path == FP32 else "gradients_vs_fp32": gradients,
                    "forward_vs_fp32": geometry, "position_geometry": positions, "forward_fingerprints": fingerprints,
                    "gradient_norms_descriptive_only": gradient_norm_summary(gradients), "health": health,
                    "first_pass_matches_backbone_diagonal": pass_zero,
                    "elapsed_seconds_including_observation": time.monotonic()-case_started, "memory": memory(),
                    "passed": all(health.values()) and pass_zero}
                report["rows"].append(row)
                persist(hybrid+"/"+path)
                tracker.log(scalar_metrics(row, "diagnostic/"+hybrid+"/"+path), step=len(report["rows"]))
                print({"hybrid": hybrid, "path": path, "operational_passed": row["passed"],
                       "elapsed_seconds": report["elapsed_seconds"]}, flush=True)
                if not row["passed"]:
                    raise AssertionError("Crossed precision case failed operational/first-pass guard")
                if path == FP32:
                    reference_gradients, reference_forward = snapshot, observed
            integrity = {"parameters_and_buffers_unchanged": state_pins(model) == hybrid_info["initial_state"],
                "inputs_unchanged": fixture_pins(fixtures) == cold["fixture_inputs"], "rng_unchanged": rng_unchanged(rng),
                "arm_configuration_unchanged": arm_contract(model, recipe) == cold["contract"],
                "production_flags_restored": all(getattr(model.backbone.backbone, key) == value for key, value in original.items())}
            report["hybrids"][hybrid]["integrity"] = integrity
            if not all(integrity.values()):
                raise AssertionError("Hybrid parameters/buffers/fixture/runtime changed during backwards")
            model.zero_grad(set_to_none=True)
            del reference_gradients, reference_forward, snapshot, observed
            gc.collect()
            persist(hybrid+"/complete")
        report["diagonal_context"] = {}
        for origin in ("cold", "adapted"):
            diagonal = reference_anchor(previous, BF16) if origin == "cold" else adapted_anchor(previous_adapted, BF16)
            report["diagonal_context"][origin] = {"scope": "Retained within-diagonal precision geometry; no new backwards",
                "ce": diagonal["metrics"]["objective"], "gradient_geometry": diagonal["gradients_vs_fp32"]["geometry"],
                "forward_geometry": diagonal["forward_vs_fp32"]}
        snapshot_checks = snapshots_unchanged(snapshots)
        report["integrity"] = {"cold_snapshot_unchanged": snapshot_checks["cold"],
            "adapted_snapshot_unchanged": snapshot_checks["adapted"],
            "sources_unchanged": source_hashes() == report["sources"],
            "matrix_report_unchanged": sha256_file(args.reference_report) == REFERENCE_SHA,
            "adapted_report_unchanged": sha256_file(args.adapted_report) == ADAPTED_SHA,
            "rng_unchanged": rng_unchanged(rng), "four_cases_completed": len(report["rows"]) == 4}
        if not all(report["integrity"].values()):
            raise AssertionError("Crossed diagnostic final integrity failed")
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
