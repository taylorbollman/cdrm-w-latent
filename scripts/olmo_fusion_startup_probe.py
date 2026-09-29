#!/usr/bin/env python3
"""Matched full-gradient precision observations at a saved fusion startup state.

Training freezes the backbone; this separate diagnostic deliberately restores
the full NF trainability contract to measure sensitivity of all parameter groups.
No optimizer is constructed or stepped. Historical diagnostic sources stay frozen.
"""
from __future__ import annotations

import gc
import argparse
from datetime import datetime, timezone
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

from scripts.olmo_campaign_adapted_precision import case_health
from scripts.olmo_campaign_crossed_precision import first_pass_identity
from scripts.olmo_campaign_ddp_probe import global_fixture_metadata
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_position_geometry import position_geometry
from scripts.olmo_campaign_precision_bridge import capture_passes, configure_path, forward_geometry
from scripts.olmo_campaign_precision_components import TERMS, component_backward, record_gradients
from scripts.olmo_campaign_recurrence_precision import (
    FP32, BF16, arm_contract, fixture_pins, gradient_norm_summary, state_pins,
)
from scripts.olmo_lm_common import tree_digests


def measure_case(model, recipe, fixtures, *, path, original_flags,
                 reference_gradients=None, reference_forward=None):
    """Reuse the established sparse CE/zero-aux VJP and observation helpers."""
    if (reference_gradients is None) != (reference_forward is None):
        raise ValueError("Gradient and forward references must be supplied together")
    contract = arm_contract(model, recipe)
    if contract["arm"] != "NF":
        raise ValueError("Startup probe is initially restricted to NF")
    metadata = global_fixture_metadata(model, fixtures)
    if metadata["microbatches"] != 2:
        raise ValueError("Startup fixture must contain exactly two physical backwards")
    rng = rng_snapshot()
    execution = configure_path(model, original_flags, path)
    backend = SDPBackend.MATH if path == FP32 else SDPBackend.FLASH_ATTENTION
    started = time.monotonic()
    with capture_passes(model, fixtures) as observed, sdpa_kernel(backend):
        metrics = component_backward(model, recipe, fixtures,
            precision=execution["precision"], layout="sparse", objective="ce")
    gradients, snapshot = record_gradients(model, reference_gradients,
        save_cpu=reference_gradients is None,
        scope="Within-saved-state FP32 reference; descriptive, not a BF16 acceptance gate")
    geometry = forward_geometry(observed, reference_forward)
    positions = position_geometry(observed, reference_forward,
        document_policy=recipe.document_policy)
    health = case_health(metrics, gradients, geometry, observed, metadata)
    health["rng_unchanged"] = rng_unchanged(rng)
    row = {"path": path, "objective": "ce", "execution": execution,
        "metrics": metrics, "ce_pass_weights": contract["ce_pass_weights"],
        "normalized_loss_means": {term: metrics["loss_sums"][term]/metrics["counts"][term]
                                  for term in TERMS},
        "gradients" if reference_gradients is None else "gradients_vs_fp32": gradients,
        "forward_vs_fp32": geometry, "position_geometry": positions,
        "forward_fingerprints": tree_digests([{key: record[key] for key in
            ("batch", "token_embeddings", "pass_hidden_states")} for record in observed]),
        "gradient_norms_descriptive_only": gradient_norm_summary(gradients),
        "health": health, "passed": all(health.values()),
        "elapsed_seconds_including_observation": time.monotonic()-started}
    if not row["passed"]:
        raise AssertionError("Startup precision case failed operational health")
    return row, snapshot, observed


def measure_pair(model, recipe, fixtures, *, original_flags, publish,
                 diagonal_by_path=None):
    """Two precisions, unchanged state/fixture/RNG, cleared gradients on exit."""
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Probe requires initially cleared gradients")
    before, inputs = state_pins(model), fixture_pins(fixtures)
    modes = {name: module.training for name, module in model.named_modules()}
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    rng = rng_snapshot()
    reference_gradients = reference_forward = None
    rows = []
    try:
        for path in (FP32, BF16):
            row, snapshot, observed = measure_case(model, recipe, fixtures,
                path=path, original_flags=original_flags,
                reference_gradients=reference_gradients, reference_forward=reference_forward)
            if diagonal_by_path is not None:
                row["first_pass_matches_backbone_diagonal"] = first_pass_identity(
                    row["forward_fingerprints"], diagonal_by_path[path])
                row["passed"] &= row["first_pass_matches_backbone_diagonal"]
            rows.append(row)
            publish(row)
            if not row["passed"]:
                raise AssertionError("First pass changed despite frozen original backbone")
            if path == FP32:
                reference_gradients, reference_forward = snapshot, observed
        integrity = {
            "state_unchanged": state_pins(model) == before,
            "fixture_unchanged": fixture_pins(fixtures) == inputs,
            "rng_unchanged": rng_unchanged(rng),
            "modes_unchanged": modes == {n: m.training for n, m in model.named_modules()},
            "trainability_unchanged": flags == {n: p.requires_grad for n, p in model.named_parameters()},
        }
        if not all(integrity.values()):
            raise AssertionError("Precision pair mutated its fixed execution state")
        return {"rows": rows, "integrity": integrity}
    finally:
        model.zero_grad(set_to_none=True)
        configure_path(model, original_flags, BF16)
        del reference_gradients, reference_forward
        gc.collect()


def source_hashes():
    from cdrm.pretrained.artifacts import sha256_file
    from scripts.olmo_campaign_crossed_precision import source_hashes as previous_sources
    sources = previous_sources()
    for name in ("scripts/olmo_fusion_startup_probe.py", "scripts/olmo_fusion_startup_train.py",
                 "scripts/olmo_fusion_startup_data.py", "tests/test_fusion_startup_probe.py",
                 "tests/test_fusion_startup_train.py", "tests/test_fusion_startup_data.py",
                 "docs/reports/olmo-fusion-startup/protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=("cold", "startup", "adapted-o5c"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--fixtures", choices=("original", "both"), default="both")
    parser.add_argument("--fresh-fixture", type=Path)
    parser.add_argument("--fresh-sha256")
    parser.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.state != "cold" and (args.checkpoint is None or args.checkpoint_sha256 is None):
        parser.error("Saved states require checkpoint and SHA256")
    if args.state == "cold" and (args.checkpoint is not None or args.checkpoint_sha256 is not None):
        parser.error("Cold state does not import a checkpoint")
    if args.fixtures == "both" and (args.fresh_fixture is None or args.fresh_sha256 is None):
        parser.error("Fresh fixture path and SHA256 are required")
    for value in (args.checkpoint_sha256, args.fresh_sha256):
        if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            parser.error("SHA256 must contain64 lowercase hexadecimal characters")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        parser.error("Evidence must stay within the persistent project")
    return args


def main(argv=None):
    from cdrm.pretrained.artifacts import sha256_file, write_json
    from scripts.experiment_tracking import OnlineTracker, scalar_metrics
    from scripts.olmo_campaign_adapted_import import endpoint_authority, load_into_current
    from scripts.olmo_campaign_adapted_precision import REFERENCE_SHA
    from scripts.olmo_campaign_crossed_precision import (
        load_adapted_reference, adapted_anchor, ADAPTED_SHA,
    )
    from scripts.olmo_campaign_ddp_probe import construct, fixture_for_update
    from scripts.olmo_campaign_fusion_precision import load_reference, reference_anchor, anchor_comparison
    from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
    from scripts.olmo_campaign_probe import memory
    from scripts.olmo_f2_graph_backend_probe import configure_determinism
    from scripts.olmo_fusion_startup_data import load_fresh_fixture
    from scripts.olmo_fusion_startup_train import load_fusion_checkpoint
    from scripts.olmo_two_gpu_validate import preserve_local_rng
    from scripts.olmo_validation import require_container_gpu

    args = parse_args(argv)
    determinism = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Startup probes require one GPU/process, no DDP")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report = {"schema": "olmo-fusion-startup-probe-v1", "status": "running", "passed": False,
        "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "sources": source_hashes(), "determinism": determinism, "state": args.state,
        "rows": [], "fixture_reports": {}, "optimizer_updates": 0,
        "aggregate_backwards": 4 if args.fixtures == "both" else 2,
        "physical_backwards": 8 if args.fixtures == "both" else 4,
        "objective": "ce", "arm": "NF", "qualification":
            "Full-gradient CE-only NF precision observations; no training, new budget or production clearance",
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "math_sdpa_reduced_precision_reduction": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    for name in report["sources"]:
        dst = args.output_dir/"source-snapshot"/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dst)
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat", output_dir=args.output_dir,
        group="olmo-fusion-startup", name=args.output_dir.name, preserve_state=preserve_local_rng)
    started, failure = time.monotonic(), None

    def persist(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-started, wandb=tracker.record)
        write_json(args.output_dir/"report.json", report)

    try:
        tracker.start({key: report[key] for key in
            ("state", "checkpoint_sha256", "determinism", "qualification", "aggregate_backwards")})
        persist("construct")
        model, recipe, checkpoint, ids, eos = construct(
            SimpleNamespace(scale="pretrained", length=16, artifacts=args.artifacts), "NF", torch.device("cuda"))
        original = {name: getattr(model.backbone.backbone, name) for name in RUNTIME_FLAGS}
        reference_path = ROOT/".runtime/olmo-recurrence-precision/matrix-01/report.json"
        reference = load_reference(reference_path, REFERENCE_SHA, report["sources"])
        cold_pins = state_pins(model)
        if cold_pins != reference["arms"]["NF"]["initial_state"]:
            raise AssertionError("Cold construction differs from the established NF state")
        report.update(source_checkpoint=checkpoint, cold_state=cold_pins,
            contract=arm_contract(model, recipe), recipe=recipe.to_dict())
        diagonals = {path: reference_anchor(reference, path) for path in (FP32, BF16)}
        if args.state == "startup":
            report["import"] = load_fusion_checkpoint(model, args.checkpoint, checkpoint,
                expected_sha256=args.checkpoint_sha256)
            pins = state_pins(model)
            if any(pins[group] != cold_pins[group] for group in ("backbone", "predictor")):
                raise AssertionError("Fusion checkpoint changed original backbone/predictor")
        elif args.state == "adapted-o5c":
            authority = endpoint_authority()
            if authority.checkpoint["sha256"] != args.checkpoint_sha256:
                raise ValueError("Adapted checkpoint differs from declared SHA256")
            # The validated importer hashes the actual large weight file once.
            report["import"] = load_into_current(model, args.checkpoint, authority)
            previous = load_adapted_reference(
                ROOT/".runtime/olmo-adapted-precision/adapted-01/report.json", report["sources"])
            if tree_digests(state_pins(model)) != previous["adapted_state"]["initial_state"]:
                raise AssertionError("Adapted imported state differs from retained diagonal")
            report["adapted_reference_sha256"] = ADAPTED_SHA
            diagonals = {path: adapted_anchor(previous, path) for path in (FP32, BF16)}
        original_fixture = [fixture_for_update(recipe, model.config.model_dim, rank, 0,
            length=16, token_ids=ids, eos_id=eos, batch_size=2) for rank in range(2)]
        fixtures = {"original": original_fixture}
        if args.fixtures == "both":
            fresh, provenance = load_fresh_fixture(args.fresh_fixture,
                expected_sha256=args.fresh_sha256, recipe=recipe, width=model.config.model_dim)
            fixtures["fresh"] = fresh
            report["fresh_fixture_provenance"] = provenance
        report["initial_state"] = state_pins(model)
        report["contract_after_import"] = arm_contract(model, recipe)
        rng = rng_snapshot()
        for name, fixture in fixtures.items():
            model.zero_grad(set_to_none=True)
            report["fixture_reports"][name] = {"pins": fixture_pins(fixture),
                "metadata": global_fixture_metadata(model, fixture)}
            persist(name+"/start")

            def publish(row):
                row["fixture"] = name
                row["memory"] = memory()
                if args.state == "cold" and name == "original":
                    gradient_key = "gradients" if row["path"] == FP32 else "gradients_vs_fp32"
                    row["cold_anchor"] = anchor_comparison(row["metrics"], row["forward_fingerprints"],
                        row[gradient_key], row["forward_vs_fp32"], diagonals[row["path"]])
                    row["passed"] &= row["cold_anchor"]["passed"]
                report["rows"].append(row)
                persist(name+"/"+row["path"])
                tracker.log(scalar_metrics(row, "probe/"+name+"/"+row["path"]), step=len(report["rows"]))
                print({"fixture": name, "path": row["path"], "passed": row["passed"],
                       "elapsed_seconds": report["elapsed_seconds"]}, flush=True)

            pair = measure_pair(model, recipe, fixture, original_flags=original,
                publish=publish, diagonal_by_path=diagonals if name == "original" else None)
            report["fixture_reports"][name]["integrity"] = pair["integrity"]
        report["integrity"] = {
            "state_unchanged": state_pins(model) == report["initial_state"],
            "sources_unchanged": source_hashes() == report["sources"],
            "rng_unchanged": rng_unchanged(rng),
            "all_cases_completed": len(report["rows"]) == report["aggregate_backwards"],
            "gradients_cleared": all(p.grad is None for p in model.parameters()),
            "original_reference_unchanged": sha256_file(reference_path) == REFERENCE_SHA,
            "production_flags_restored": all(getattr(model.backbone.backbone, n) == v for n, v in original.items()),
        }
        if args.state == "startup":
            report["integrity"]["checkpoint_unchanged"] = sha256_file(args.checkpoint) == args.checkpoint_sha256
        if not all(report["integrity"].values()):
            raise AssertionError("Startup probe final integrity failed")
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
