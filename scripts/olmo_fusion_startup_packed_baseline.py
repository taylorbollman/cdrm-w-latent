#!/usr/bin/env python3
"""One ordinary-stack N-only CE precision pair on the fixed packed T1024 rows.

Original OLMo weights and predictor; dormant frozen fusion. Both precision paths
must reproduce the retained cold NF first pass exactly. No optimizer, temporal
RT, feedback or numerical acceptance threshold. Active auxiliary loss branches
receive zero cotangents, keeping the existing CE diagnostic convention.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
from scripts.olmo_campaign_ddp_probe import construct, global_fixture_metadata
from scripts.olmo_campaign_graph_probe import rng_snapshot, rng_unchanged
from scripts.olmo_campaign_position_geometry import position_geometry
from scripts.olmo_campaign_precision_bridge import capture_passes, configure_path, forward_geometry
from scripts.olmo_campaign_precision_components import RUNTIME_FLAGS, TERMS, component_backward, record_gradients
from scripts.olmo_campaign_recurrence_precision import FP32, BF16, arm_contract, fixture_pins, state_pins, gradient_norm_summary
from scripts.olmo_campaign_probe import memory
from scripts.olmo_f2_graph_backend_probe import configure_determinism
from scripts.olmo_fusion_startup_context_probe import (transition_policy, load_cold, json_contract,
    source_hashes as context_sources)
from scripts.olmo_fusion_startup_packed_data import load_packed_fixture
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

SCHEMA = "olmo-fusion-startup-packed-baseline-v1"


def source_hashes():
    sources = context_sources()
    for name in ("scripts/olmo_fusion_startup_packed_baseline.py", "tests/test_fusion_startup_packed_baseline.py",
                 "docs/reports/olmo-fusion-startup/packed-baseline-protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def ordinary_transition(model, recipe, fixtures):
    """Disable feedback without rebuilding parameters or changing packed rows."""
    if recipe.arm != "NF" or recipe.document_policy != "continuous-stream-v1":
        raise ValueError("Start from the validated packed NF fixture and state")
    arm_contract(model, recipe)
    if len(fixtures) != 2 or any(len(batches) != 1 or len(noises) != 1 or len(noises[0]) != 3
                               for batches, noises in fixtures):
        raise ValueError("Require the two validated NF records and three noise arrays each")
    before = state_pins(model)
    flags = {name:parameter.requires_grad for name,parameter in model.named_parameters()}
    ids = {name:id(parameter) for name,parameter in model.named_parameters()}
    changed = replace(recipe, arm="N")
    model.backbone.fusion.requires_grad_(False)
    ordinary = [(batches, (None,)) for batches,_ in fixtures]
    checks = {"state_unchanged":state_pins(model) == before,
        "parameter_identities":ids == {name:id(parameter) for name,parameter in model.named_parameters()},
        "only_fusion_frozen":all(parameter.requires_grad == (False if name.startswith("backbone.fusion.") else flags[name])
                                for name,parameter in model.named_parameters()),
        "tokens_masks_exact":[r["batches"] for r in fixture_pins(ordinary)] == [r["batches"] for r in fixture_pins(fixtures)],
        "noise_absent":all(noises == (None,) for _,noises in ordinary),
        "recipe_only_arm_and_pass_count":{k:v for k,v in changed.to_dict().items() if k not in ("arm","fbt_passes")}
            == {k:v for k,v in recipe.to_dict().items() if k not in ("arm","fbt_passes")}}
    contract = arm_contract(model, changed)
    if not all(checks.values()):
        raise AssertionError("Ordinary baseline transition changed unrelated state")
    return changed, ordinary, {"before_recipe":recipe.to_dict(), "after_recipe":changed.to_dict(),
        "checks":checks, "contract":contract,
        "scope":"Original tensors retained; dormant fusion frozen; feedback/noise and temporal RT absent"}


def first_pass_anchor(fingerprints, anchor):
    expected = anchor["forward_fingerprints"]
    if len(fingerprints) != len(expected) or len(expected) != 2:
        raise ValueError("First-pass anchor requires both physical records")
    checks = {"tokens_masks":all(a["batch"] == b["batch"] for a,b in zip(fingerprints,expected)),
        "token_embeddings":all(a["token_embeddings"] == b["token_embeddings"] for a,b in zip(fingerprints,expected)),
        "one_pass_exact":all(len(a["pass_hidden_states"]) == 1 and len(b["pass_hidden_states"]) == 4
            and a["pass_hidden_states"][0] == b["pass_hidden_states"][0] for a,b in zip(fingerprints,expected))}
    return {"passed":all(checks.values()), "checks":checks,
        "scope":"Same precision's complete cold NF first pass; incoming cotangents intentionally differ"}


def measure_case(model, recipe, fixtures, *, path, original_flags,
                 reference_gradients=None, reference_forward=None):
    if recipe.arm != "N" or (reference_gradients is None) != (reference_forward is None):
        raise ValueError("N-only pair requires matched gradient and forward references")
    contract = arm_contract(model, recipe)
    if any(noises != (None,) for _,noises in fixtures):
        raise ValueError("N-only case must not receive feedback noise")
    metadata = global_fixture_metadata(model, fixtures)
    if metadata["microbatches"] != 2:
        raise ValueError("Exactly two physical backwards required")
    rng = rng_snapshot()
    execution = configure_path(model, original_flags, path)
    device = next(model.parameters()).device
    backend = SDPBackend.FLASH_ATTENTION if path == BF16 and device.type == "cuda" else SDPBackend.MATH
    begin = time.monotonic()
    with capture_passes(model, fixtures) as observed, sdpa_kernel(backend), torch.autocast(device.type,enabled=False):
        metrics = component_backward(model, recipe, fixtures, precision=execution["precision"],layout="sparse",objective="ce")
    gradients,snapshot = record_gradients(model,reference_gradients,save_cpu=reference_gradients is None,
        scope="Same N-only packed state; descriptive BF16/FP32 calibration, no revised numerical threshold")
    forward = forward_geometry(observed,reference_forward)
    health = {"finite_gradients":gradients["finite"],"gradient_participation":gradients["participation_intact"],
        "finite_forward_and_cotangents":all(row["hidden"]["finite"] and row["total_incoming_cotangent"]["finite"]
            for row in forward if not row.get("skipped_dummy")),
        "finite_losses":all(math.isfinite(value) for value in (metrics["objective"],*metrics["loss_sums"].values())),
        "one_pass_per_record":len(observed)==2 and all(len(row["pass_hidden_states"])==1 for row in observed),
        "counts_exact":all(metrics[key]==value for key,value in metadata.items()),
        "backbone_nonzero":gradients["groups"]["backbone"]["norm"]>0,
        "predictor_zero_cotangent":gradients["groups"]["predictor"]["norm"]==0,
        "fusion_frozen_and_unused":all(not p.requires_grad and p.grad is None for p in model.backbone.fusion.parameters()),
        "rng_unchanged":rng_unchanged(rng)}
    row = {"arm":"N","objective":"ce","path":path,"execution":execution,"contract":contract,
        "metrics":metrics,"normalized_loss_means":{term:metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
        "gradients" if reference_gradients is None else "gradients_vs_fp32":gradients,
        "gradient_norms_descriptive_only":gradient_norm_summary(gradients),"forward_vs_fp32":forward,
        "position_geometry":position_geometry(observed,reference_forward,document_policy=recipe.document_policy),
        "forward_fingerprints":tree_digests([{key:record[key] for key in ("batch","token_embeddings","pass_hidden_states")} for record in observed]),
        "health":health,"passed":all(health.values()),"elapsed_seconds_including_observation":time.monotonic()-begin}
    if not row["passed"]:
        raise AssertionError("N-only precision case failed operational health")
    return row,snapshot,observed


def measure_pair(model,recipe,fixtures,*,original_flags,anchors,publish):
    if set(anchors)!={FP32,BF16} or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Start from cleared gradients and both cold NF precision anchors")
    initial,inputs = state_pins(model),fixture_pins(fixtures)
    modes={n:m.training for n,m in model.named_modules()}
    trainability={n:p.requires_grad for n,p in model.named_parameters()}
    rng=rng_snapshot(); rows=[];reference_gradients=reference_forward=None
    try:
        for path in (FP32,BF16):
            if anchors[path]["path"]!=path:
                raise ValueError("Anchor precision differs")
            row,snapshot,observed=measure_case(model,recipe,fixtures,path=path,original_flags=original_flags,
                reference_gradients=reference_gradients,reference_forward=reference_forward)
            row["cold_nf_first_pass_anchor"]=first_pass_anchor(row["forward_fingerprints"],anchors[path])
            row["passed"] &= row["cold_nf_first_pass_anchor"]["passed"]
            rows.append(row);publish(row)
            if not row["passed"]:
                raise AssertionError("N-only forward differs from cold NF first pass")
            if path==FP32:reference_gradients,reference_forward=snapshot,observed
        checks={"state_unchanged":state_pins(model)==initial,"fixture_unchanged":fixture_pins(fixtures)==inputs,
            "modes_unchanged":modes=={n:m.training for n,m in model.named_modules()},
            "trainability_unchanged":trainability=={n:p.requires_grad for n,p in model.named_parameters()},
            "rng_unchanged":rng_unchanged(rng),"two_cases":len(rows)==2}
        if not all(checks.values()):raise AssertionError("Ordinary baseline integrity failed")
        return {"rows":rows,"integrity":checks}
    finally:
        model.zero_grad(set_to_none=True)
        configure_path(model,original_flags,BF16)
        reference_gradients=reference_forward=None
        gc.collect()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture",type=Path,required=True);p.add_argument("--fixture-sha256",required=True)
    p.add_argument("--cold-report",type=Path,required=True);p.add_argument("--cold-report-sha256",required=True)
    p.add_argument("--artifacts",type=Path,default=ROOT/".runtime/olmo1b-step60000/artifacts")
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args(argv)
    for value in (args.fixture_sha256,args.cold_report_sha256):
        if len(value)!=64 or any(c not in "0123456789abcdef" for c in value):p.error("Require lowercase SHA256 pins")
    args.output_dir=args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):p.error("Evidence must stay under persistent project storage")
    return args


def main(argv=None):
    args=parse_args(argv)
    deterministic=configure_determinism(True);runtime=require_container_gpu()
    if torch.distributed.is_initialized():raise RuntimeError("One GPU/process without DDP required")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources=source_hashes()
    for name in sources:
        destination=args.output_dir/"source-snapshot"/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    report={"schema":SCHEMA,"status":"running","passed":False,"sources":sources,"rows":[],
        "started_utc":datetime.now(timezone.utc).isoformat(),"runtime":runtime,"determinism":deterministic,
        "fixture":str(args.fixture),"fixture_sha256":args.fixture_sha256,
        "cold_report":str(args.cold_report),"cold_report_sha256":args.cold_report_sha256,
        "aggregate_backwards":2,"physical_backwards":4,"optimizer_updates":0,
        "math_sdpa_reduced_precision_reduction":torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "qualification":"Ordinary N-only packed CE precision calibration; no RT/FBT/active auxiliary gradient or BF16 production clearance"}
    tracker=OnlineTracker(project="pretrained-fbt-rt-nextlat",output_dir=args.output_dir,
        group="olmo-fusion-startup",name=args.output_dir.name,preserve_state=preserve_local_rng)
    start,failure=time.monotonic(),None
    def persist(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-start,wandb=tracker.record)
        write_json(args.output_dir/"report.json",report)
    try:
        tracker.start({key:report[key] for key in ("fixture_sha256","cold_report_sha256","qualification","determinism")})
        persist("construct_original_nf")
        reference,anchors=load_cold(args.cold_report,args.cold_report_sha256,context_sources(),args.fixture_sha256)
        model,recipe,checkpoint,_,_=construct(SimpleNamespace(scale="pretrained",length=16,artifacts=args.artifacts),"NF",torch.device("cuda"))
        original={n:getattr(model.backbone.backbone,n) for n in RUNTIME_FLAGS}
        cold=state_pins(model)
        if cold!=reference["cold_state"]:raise AssertionError("Original cold state differs")
        recipe,report["policy_transition"]=transition_policy(model,recipe)
        fixtures,metadata=load_packed_fixture(args.fixture,expected_sha256=args.fixture_sha256,recipe=recipe,width=model.config.model_dim)
        report.update(source_checkpoint=checkpoint,initial_state=state_pins(model),fixture_provenance=metadata,
            nf_fixture_pins=fixture_pins(fixtures),fixture_metadata=global_fixture_metadata(model,fixtures),nf_contract=json_contract(arm_contract(model,recipe)))
        checks={key:json_contract(report[key])==json_contract(reference[key]) for key in
            ("source_checkpoint","initial_state","fixture_provenance","fixture_metadata","runtime","determinism",
             "math_sdpa_reduced_precision_reduction","bf16_matmul_reduced_precision_reduction")}
        checks.update(fixture_pins_exact=report["nf_fixture_pins"]==reference["fixture_pins"],nf_contract_exact=report["nf_contract"]==reference["contract"])
        report["cold_reference_checks"]=checks
        if not all(checks.values()):raise AssertionError("Cold packed baseline authority differs")
        rng=rng_snapshot()
        recipe,ordinary,report["ordinary_transition"]=ordinary_transition(model,recipe,fixtures)
        report.update(fixture_pins=fixture_pins(ordinary),contract=json_contract(arm_contract(model,recipe)))
        def publish(row):
            row["memory"]=memory();report["rows"].append(row);persist(row["path"])
            tracker.log(scalar_metrics(row,"ordinary_packed/"+row["path"]),step=len(report["rows"]))
            print({"path":row["path"],"passed":row["passed"],"elapsed_seconds":report["elapsed_seconds"]},flush=True)
        pair=measure_pair(model,recipe,ordinary,original_flags=original,anchors=anchors,publish=publish)
        report["pair_integrity"]=pair["integrity"]
        report["integrity"]={"sources_unchanged":source_hashes()==sources,"state_unchanged":state_pins(model)==report["initial_state"],
            "fixture_file_unchanged":sha256_file(args.fixture)==args.fixture_sha256,
            "cold_report_unchanged":sha256_file(args.cold_report)==args.cold_report_sha256,
            "nf_fixture_tensors_unchanged":fixture_pins(fixtures)==report["nf_fixture_pins"],
            "ordinary_fixture_tensors_unchanged":fixture_pins(ordinary)==report["fixture_pins"],
            "rng_unchanged":rng_unchanged(rng),"gradients_cleared":all(p.grad is None for p in model.parameters()),
            "production_flags_restored":all(getattr(model.backbone.backbone,n)==v for n,v in original.items()),
            "two_cases_completed":len(report["rows"])==2}
        if not all(report["integrity"].values()):raise AssertionError("N-only final integrity failed")
        report.update(status="passed_operational_diagnostic",passed=True);persist("complete")
    except BaseException as exc:
        failure=exc
        report.update(status="failed",passed=False,error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        raise
    finally:
        report["finished_utc"]=datetime.now(timezone.utc).isoformat();persist(report.get("stage","setup"))
        try:tracker.finish(succeeded=report["passed"])
        except BaseException as exc:
            report.update(status="failed",passed=False,tracking_finish_error={"type":type(exc).__name__})
            if failure is None:raise
        finally:persist(report.get("stage","setup"))


if __name__=="__main__":main()
