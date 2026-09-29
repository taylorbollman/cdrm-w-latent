#!/usr/bin/env python3
"""Cold/update128 NF CE precision observations on held-out context fixtures.

Import compact checkpoints under their original isolated training contract,
then explicitly change only the document policy. No optimizer or source-model
change is made; the historical warmup/probe contracts remain frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

from cdrm.pretrained.artifacts import sha256_file, write_json
from cdrm.pretrained.document_policy import ISOLATED_DOCUMENTS, CONTINUOUS_STREAM
from scripts.experiment_tracking import OnlineTracker, scalar_metrics
from scripts.olmo_campaign_adapted_precision import REFERENCE_SHA
from scripts.olmo_campaign_ddp_probe import construct, global_fixture_metadata
from scripts.olmo_campaign_fusion_precision import load_reference
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS
from scripts.olmo_campaign_probe import memory
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, arm_contract, fixture_pins, state_pins
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_packed_data import load_packed_fixture, source_hashes as data_sources
from scripts.olmo_fusion_startup_probe import measure_pair, source_hashes as probe_sources
from scripts.olmo_fusion_startup_train import load_fusion_checkpoint
from scripts.olmo_fusion_startup_long_probe import (load_long_fixture, load_cold_report as load_original_long_cold, source_hashes as long_sources)
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = "olmo-fusion-startup-context-probe-v1"


def source_hashes():
    result = long_sources()
    for name, digest in data_sources().items():
        if name in result and result[name] != digest:
            raise ValueError("Packed data and probe source inventories disagree")
        result[name] = digest
    for name in ("scripts/olmo_fusion_startup_context_probe.py", "tests/test_fusion_startup_context_probe.py", "docs/reports/olmo-fusion-startup/context-probe-protocol.md"):
        result[name] = sha256_file(ROOT/name)
    return dict(sorted(result.items()))


def transition_policy(model, recipe):
    """Explicit configuration-only change after checkpoint import, not migration."""
    if recipe.arm != "NF" or recipe.document_policy != ISOLATED_DOCUMENTS or model.config.document_policy != ISOLATED_DOCUMENTS:
        raise ValueError("Packed transition requires the original isolated NF contract")
    before = {"recipe": recipe.to_dict(), "nextlat": asdict(model.config), "state": state_pins(model),
              "trainability": {n: p.requires_grad for n,p in model.named_parameters()}}
    recipe = replace(recipe, document_policy=CONTINUOUS_STREAM)
    model.config = replace(model.config, document_policy=CONTINUOUS_STREAM)
    after = {"recipe": recipe.to_dict(), "nextlat": asdict(model.config), "state": state_pins(model),
             "trainability": {n: p.requires_grad for n,p in model.named_parameters()}}
    checks = {"tensor_state_exact": before["state"] == after["state"],
        "trainability_exact": before["trainability"] == after["trainability"],
        "recipe_only_document_policy_changed": {k:v for k,v in before["recipe"].items() if k != "document_policy"}
            == {k:v for k,v in after["recipe"].items() if k != "document_policy"},
        "nextlat_only_document_policy_changed": {k:v for k,v in before["nextlat"].items() if k != "document_policy"}
            == {k:v for k,v in after["nextlat"].items() if k != "document_policy"}}
    if not all(checks.values()):
        raise AssertionError("Document policy transition changed unrelated state")
    return recipe, {"before": before, "after": after, "checks": checks,
        "scope": "Runtime document-policy selection after strict isolated checkpoint import; no weight or optimizer migration"}


def json_contract(value):
    """Normalize tuple/list representation without dropping fields or values."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_cold(path, digest, sources, fixture_sha):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024*1024 or sha256_file(path) != digest:
        raise ValueError("Packed cold reference differs from bounded independent SHA256")
    report = json.loads(path.read_text())
    if sha256_file(path) != digest:
        raise ValueError("Packed cold reference changed while reading")
    if (report.get("schema") != SCHEMA or report.get("fixture_kind") != "packed"
            or report.get("state") != "cold" or not report.get("passed")
            or report.get("status") != "passed_operational_diagnostic" or report.get("fixture_sha256") != fixture_sha
            or not report.get("integrity") or not all(report["integrity"].values())
            or not report.get("pair_integrity") or not all(report["pair_integrity"].values())
            or report.get("optimizer_updates") != 0 or report.get("aggregate_backwards") != 2
            or report.get("physical_backwards") != 4
            or report.get("determinism",{}).get("deterministic_algorithms") is not True
            or report.get("sources") != sources):
        raise ValueError("Packed cold reference contract or source integrity differs")
    endpoints = {}
    for path_name in (FP32, BF16):
        rows = [r for r in report.get("rows", []) if r.get("path") == path_name and r.get("objective") == "ce"]
        if (len(rows) != 1 or not rows[0].get("passed") or not rows[0].get("health")
                or not all(rows[0]["health"].values())):
            raise ValueError("Packed cold reference lacks a healthy precision endpoint")
        endpoints[path_name] = rows[0]
    return report, endpoints


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", choices=("cold", "startup"), required=True)
    p.add_argument("--fixture-kind", choices=("long", "packed"), default="packed")
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--fixture-sha256", required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-sha256")
    p.add_argument("--cold-report", type=Path)
    p.add_argument("--cold-report-sha256")
    p.add_argument("--artifacts", type=Path, default=ROOT/".runtime/olmo1b-step60000/artifacts")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if args.fixture_kind == "long" and args.state != "startup":
        p.error("Use the retained original long cold state; this driver adds only its corrected endpoint check")
    supplied = (args.checkpoint,args.checkpoint_sha256,args.cold_report,args.cold_report_sha256)
    if (args.state == "startup" and not all(supplied)) or (args.state == "cold" and any(supplied)):
        p.error("Only startup requires complete checkpoint and cold-report path/SHA pairs")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        p.error("Evidence must remain on persistent project storage")
    return args


def main(argv=None):
    args = parse_args(argv)
    deterministic = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Packed precision observation is single-GPU without DDP")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    for name in sources:
        target = args.output_dir/"source-snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name,target)
    report = {"schema": SCHEMA,"status":"running","passed":False,"sources":sources,
        "state":args.state,"fixture_kind":args.fixture_kind,"runtime":runtime,"determinism":deterministic,
        "started_utc":datetime.now(timezone.utc).isoformat(),"fixture":str(args.fixture),"fixture_sha256":args.fixture_sha256,
        "checkpoint":None if args.checkpoint is None else str(args.checkpoint),"checkpoint_sha256":args.checkpoint_sha256,
        "cold_report":None if args.cold_report is None else str(args.cold_report),"cold_report_sha256":args.cold_report_sha256,
        "rows":[],"aggregate_backwards":2,"physical_backwards":4,"optimizer_updates":0,
        "math_sdpa_reduced_precision_reduction":torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "qualification":"NF CE-only held-out context diagnostic; no BF16 acceptance or RT/auxiliary/update clearance"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat",output_dir=args.output_dir,
        group="olmo-fusion-startup",name=args.output_dir.name,preserve_state=preserve_local_rng)
    start,failure = time.monotonic(),None
    def persist(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-start,wandb=tracker.record)
        write_json(args.output_dir/"report.json",report)
    try:
        tracker.start({k:report[k] for k in ("state","fixture_sha256","checkpoint_sha256","qualification","determinism")})
        persist("construct_original_nf")
        model,recipe,checkpoint,_,_ = construct(SimpleNamespace(scale="pretrained",length=16,artifacts=args.artifacts),"NF",torch.device("cuda"))
        original = {n:getattr(model.backbone.backbone,n) for n in RUNTIME_FLAGS}
        cold = state_pins(model)
        matrix_path = ROOT/".runtime/olmo-recurrence-precision/matrix-01/report.json"
        matrix = load_reference(matrix_path,REFERENCE_SHA,sources)
        if cold != matrix["arms"]["NF"]["initial_state"]:
            raise AssertionError("Packed probe cold model differs from original NF authority")
        if args.state == "startup":
            report["import"] = load_fusion_checkpoint(model,args.checkpoint,checkpoint,expected_sha256=args.checkpoint_sha256)
            if report["import"]["counters"]["optimizer_updates"] != 128:
                raise ValueError("Packed endpoint must be the declared update128")
            current = state_pins(model)
            if any(current[g] != cold[g] for g in ("backbone","predictor")):
                raise AssertionError("Compact import changed original frozen groups")
        rng = rng_snapshot()
        if args.fixture_kind == "packed":
            recipe,report["policy_transition"] = transition_policy(model,recipe)
        if not rng_unchanged(rng):
            raise AssertionError("Policy transition consumed RNG")
        loader = load_packed_fixture if args.fixture_kind == "packed" else load_long_fixture
        fixtures,metadata = loader(args.fixture,expected_sha256=args.fixture_sha256,recipe=recipe,width=model.config.model_dim)
        if args.state == "startup" and report["import"]["configuration"]["data_manifest_sha256"] != metadata["training_manifest_sha256"]:
            raise ValueError("Packed fixture and warmup data authorities differ")
        report.update(source_checkpoint=checkpoint,cold_state=cold,initial_state=state_pins(model),
            fixture_provenance=metadata,fixture_pins=fixture_pins(fixtures),
            fixture_metadata=global_fixture_metadata(model,fixtures),contract=json_contract(arm_contract(model,recipe)))
        diagonals = None
        if args.state == "startup":
            if args.fixture_kind == "long":
                reference,diagonals = load_original_long_cold(args.cold_report,args.cold_report_sha256,sources,fixture_sha256=args.fixture_sha256)
            else:
                reference,diagonals = load_cold(args.cold_report,args.cold_report_sha256,sources,args.fixture_sha256)
            checks = {key:json_contract(report[key]) == json_contract(reference[key]) for key in ("source_checkpoint","cold_state","fixture_pins",
                "fixture_metadata","contract","runtime","determinism","math_sdpa_reduced_precision_reduction","bf16_matmul_reduced_precision_reduction")}
            report["cold_reference_checks"] = checks
            if not all(checks.values()):
                raise AssertionError("Packed cold/startup comparison contract differs")
        def publish(row):
            row["memory"] = memory(); report["rows"].append(row); persist(row["path"])
            tracker.log(scalar_metrics(row,"packed_probe/"+row["path"]),step=len(report["rows"]))
            print({"path":row["path"],"passed":row["passed"],"elapsed_seconds":report["elapsed_seconds"]},flush=True)
        pair = measure_pair(model,recipe,fixtures,original_flags=original,publish=publish,diagonal_by_path=diagonals)
        report["pair_integrity"] = pair["integrity"]
        report["integrity"] = {"state_unchanged":state_pins(model)==report["initial_state"],
            "sources_unchanged":source_hashes()==sources,"rng_unchanged":rng_unchanged(rng),
            "fixture_file_unchanged":sha256_file(args.fixture)==args.fixture_sha256,
            "fixture_tensors_unchanged":fixture_pins(fixtures)==report["fixture_pins"],
            "gradients_cleared":all(p.grad is None for p in model.parameters()),"two_cases_completed":len(report["rows"])==2,
            "production_flags_restored":all(getattr(model.backbone.backbone,n)==v for n,v in original.items()),
            "original_matrix_unchanged":sha256_file(matrix_path)==REFERENCE_SHA}
        if args.state == "startup":
            report["integrity"].update(checkpoint_unchanged=sha256_file(args.checkpoint)==args.checkpoint_sha256,
                cold_report_unchanged=sha256_file(args.cold_report)==args.cold_report_sha256)
        if not all(report["integrity"].values()):
            raise AssertionError("Packed probe final integrity failed")
        report.update(status="passed_operational_diagnostic",passed=True);persist("complete")
    except BaseException as exc:
        failure=exc
        report.update(status="failed",passed=False,error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        raise
    finally:
        report["finished_utc"]=datetime.now(timezone.utc).isoformat();persist(report.get("stage","setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as exc:
            report.update(status="failed",passed=False,tracking_finish_error={"type":type(exc).__name__})
            if failure is None: raise
        finally: persist(report.get("stage","setup"))


if __name__ == "__main__":
    main()
