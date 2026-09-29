#!/usr/bin/env python3
"""Six bounded post-startup NF/NFR objective/precision cases, no updates.

NF combined, NFR CE and NFR combined each compare full FP32 with the current
production BF16 backend. One saved NF fusion is imported before an explicit
recipe-only RT transition. Old diagnostic and model sources remain untouched.
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
from scripts.olmo_fusion_startup_long_probe import load_long_fixture
from scripts.olmo_fusion_startup_context_probe import source_hashes as context_sources
from scripts.olmo_fusion_startup_train import load_fusion_checkpoint
from scripts.olmo_lm_common import tree_digests
from scripts.olmo_two_gpu_validate import preserve_local_rng
from scripts.olmo_validation import require_container_gpu

PLAN = (("NF", "combined"), ("NFR", "ce"), ("NFR", "combined"))
SCHEMA = "olmo-fusion-startup-component-probe-v1"


def source_hashes():
    sources = context_sources()
    for name in ("scripts/olmo_fusion_startup_component_probe.py", "tests/test_fusion_startup_component_probe.py",
                 "docs/reports/olmo-fusion-startup/component-probe-protocol.md"):
        sources[name] = sha256_file(ROOT/name)
    return dict(sorted(sources.items()))


def objective_contract(model, recipe, objective):
    if objective not in ("ce", "combined") or recipe.arm not in ("NF", "NFR"):
        raise ValueError("Component bridge requires NF/NFR and CE/combined")
    contract = arm_contract(model, recipe)
    contract.update(objective=objective,
        auxiliary_cotangents={term:float(objective == "combined") for term in ("latent", "kl")},
        objective_weights={term:float(term == "ce" or objective == "combined") for term in TERMS},
        auxiliary_pass_weights=[.25]*4)
    # Do not retain the old CE-only descriptive label when auxiliaries are active.
    contract.pop("auxiliary_pass_weights_before_zero_cotangent")
    return contract


def enable_native_rt(model, recipe):
    """Select existing native RT layers without rebuilding or changing tensors."""
    if recipe.arm != "NF" or recipe.document_policy != "isolated-v1" or recipe.mode().rt_mode.selected_layers:
        raise ValueError("Native RT transition starts from isolated NF")
    before = state_pins(model)
    original = recipe.to_dict()
    changed = replace(recipe, arm="NFR")
    checks = {"state_unchanged": before == state_pins(model),
        "only_arm_changed": {k:v for k,v in changed.to_dict().items() if k != "arm"}
            == {k:v for k,v in original.items() if k != "arm"},
        "declared_layers_selected": changed.mode().rt_mode.selected_layers == recipe.rt_layers,
        "native_implementation": model.backbone.backbone.rt_implementation == "native"}
    objective_contract(model, changed, "ce")
    if not all(checks.values()):
        raise AssertionError("RT transition changed unrelated configuration or state")
    return changed, {"before_recipe":original, "after_recipe":changed.to_dict(), "checks":checks,
                     "scope":"Existing native RT layers selected by recipe; no state or optimizer migration"}


def forward_anchor(row, anchor):
    """Objective-only changes must preserve all forward bytes and raw losses."""
    if row["path"] != anchor["path"]:
        raise ValueError("Forward anchors must use the same precision/backend")
    checks = {"all_forward_fingerprints_exact": row["forward_fingerprints"] == anchor["forward_fingerprints"],
        "raw_loss_sums_exact": row["metrics"]["loss_sums"] == anchor["metrics"]["loss_sums"],
        "target_counts_exact": row["metrics"]["counts"] == anchor["metrics"]["counts"]}
    return {"passed":all(checks.values()), "checks":checks,
            "scope":"Changing CE/combined objective changes backward only; gradients and cotangents need not match"}


def case_health(metrics, gradients, forward, observed, metadata, objective):
    if objective not in ("ce", "combined"):
        raise ValueError("Unknown component objective")
    norms = {name: gradients["groups"][name]["norm"] for name in ("backbone", "fusion", "predictor")}
    return {"finite_gradients":gradients["finite"],
        "gradient_participation":gradients["participation_intact"],
        "finite_forward_and_cotangents":all(row["hidden"]["finite"] and row["total_incoming_cotangent"]["finite"]
            for row in forward if not row.get("skipped_dummy")),
        "finite_losses":all(math.isfinite(value) for value in (metrics["objective"],*metrics["loss_sums"].values())),
        "four_passes_each_record":len(observed) == 2 and all(len(row["pass_hidden_states"]) == 4 for row in observed),
        "counts_exact":all(metrics[key] == value for key,value in metadata.items()),
        "nonzero_backbone_and_fusion":all(math.isfinite(norms[name]) and norms[name] > 0 for name in ("backbone","fusion")),
        "predictor_objective_participation":math.isfinite(norms["predictor"]) and
            (norms["predictor"] > 0 if objective == "combined" else norms["predictor"] == 0)}


def measure_case(model, recipe, fixtures, *, objective, path, original_flags,
                 reference_gradients=None, reference_forward=None):
    if (reference_gradients is None) != (reference_forward is None):
        raise ValueError("Supply gradient and forward references together")
    contract = objective_contract(model, recipe, objective)
    metadata = global_fixture_metadata(model, fixtures)
    if metadata["microbatches"] != 2:
        raise ValueError("Exactly two physical backwards per aggregate case")
    rng = rng_snapshot()
    execution = configure_path(model, original_flags, path)
    device = next(model.parameters()).device
    backend = SDPBackend.FLASH_ATTENTION if path == BF16 and device.type == "cuda" else SDPBackend.MATH
    started = time.monotonic()
    # The existing backward helpers own forward autocast. Keep only forced SDPA
    # and disabled ambient autocast active through their checkpoint replay.
    with capture_passes(model, fixtures) as observed, sdpa_kernel(backend), torch.autocast(device.type,enabled=False):
        metrics = component_backward(model, recipe, fixtures, precision=execution["precision"],
                                     layout="sparse", objective=objective)
    gradients, snapshot = record_gradients(model, reference_gradients, save_cpu=reference_gradients is None,
        scope="Within the same state/arm/objective; descriptive BF16/FP32 geometry, no revised numerical threshold")
    forward = forward_geometry(observed, reference_forward)
    health = case_health(metrics, gradients, forward, observed, metadata, objective)
    health["rng_unchanged"] = rng_unchanged(rng)
    row = {"arm":recipe.arm, "objective":objective, "path":path, "execution":execution,
        "contract":contract, "metrics":metrics,
        "normalized_loss_means":{term:metrics["loss_sums"][term]/metrics["counts"][term] for term in TERMS},
        "gradients" if reference_gradients is None else "gradients_vs_fp32":gradients,
        "gradient_norms_descriptive_only":gradient_norm_summary(gradients),
        "forward_vs_fp32":forward,
        "position_geometry":position_geometry(observed,reference_forward,document_policy=recipe.document_policy),
        "forward_fingerprints":tree_digests([{key:record[key] for key in
            ("batch","token_embeddings","pass_hidden_states")} for record in observed]),
        "health":health, "passed":all(health.values()),
        "elapsed_seconds_including_observation":time.monotonic()-started}
    if not row["passed"]:
        raise AssertionError("Component precision case failed operational health")
    return row,snapshot,observed


def measure_bridge(model, recipe, fixtures, *, original_flags, nf_ce_anchors, publish):
    if recipe.arm != "NF" or set(nf_ce_anchors) != {FP32,BF16}:
        raise ValueError("Bridge starts from NF with both saved NF CE anchors")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Bridge starts with cleared gradients")
    initial, inputs = state_pins(model), fixture_pins(fixtures)
    flags = {name:p.requires_grad for name,p in model.named_parameters()}
    modes = {name:m.training for name,m in model.named_modules()}
    rng = rng_snapshot()
    rows, transitions, nfr_ce = [], [], {}
    reference_gradients = reference_forward = None
    try:
        for arm,objective in PLAN:
            if arm == "NFR" and recipe.arm == "NF":
                recipe, transition = enable_native_rt(model,recipe)
                transitions.append(transition)
            reference_gradients = reference_forward = None
            for path in (FP32,BF16):
                row,snapshot,observed = measure_case(model,recipe,fixtures,objective=objective,path=path,
                    original_flags=original_flags,reference_gradients=reference_gradients,reference_forward=reference_forward)
                anchor = nf_ce_anchors[path] if arm == "NF" else nfr_ce.get(path) if objective == "combined" else None
                if anchor is not None:
                    row["objective_forward_control"] = forward_anchor(row,anchor)
                    row["passed"] &= row["objective_forward_control"]["passed"]
                rows.append(row); publish(row)
                if not row["passed"]:
                    raise AssertionError("Objective-only change altered forward or raw losses")
                if arm == "NFR" and objective == "ce":
                    nfr_ce[path] = {key:row[key] for key in ("path","metrics","forward_fingerprints")}
                if path == FP32:
                    reference_gradients,reference_forward = snapshot,observed
            model.zero_grad(set_to_none=True)
            reference_gradients = reference_forward = snapshot = observed = None
            gc.collect()
        integrity = {"state_unchanged":state_pins(model) == initial, "fixtures_unchanged":fixture_pins(fixtures) == inputs,
            "rng_unchanged":rng_unchanged(rng), "six_cases":len(rows) == 6,
            "trainability_unchanged":flags == {n:p.requires_grad for n,p in model.named_parameters()},
            "modes_unchanged":modes == {n:m.training for n,m in model.named_modules()}}
        if not all(integrity.values()):
            raise AssertionError("Component bridge changed fixed state/fixture/runtime contract")
        return {"rows":rows, "transitions":transitions, "integrity":integrity}
    finally:
        model.zero_grad(set_to_none=True)
        configure_path(model,original_flags,BF16)
        reference_gradients = reference_forward = None
        gc.collect()


def load_nf_reference(path, expected_sha256, sources, *, fixture_sha256, checkpoint_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128*1024*1024 or sha256_file(path) != expected_sha256:
        raise ValueError("NF endpoint reference differs from bounded immutable pin")
    report = json.loads(path.read_text())
    if sha256_file(path) != expected_sha256:
        raise ValueError("NF endpoint reference changed while reading")
    if (report.get("schema") != "olmo-fusion-startup-context-probe-v1"
            or report.get("fixture_kind") != "long" or report.get("state") != "startup"
            or report.get("status") != "passed_operational_diagnostic" or report.get("passed") is not True
            or report.get("checkpoint_sha256") != checkpoint_sha256 or report.get("fixture_sha256") != fixture_sha256
            or report.get("optimizer_updates") != 0 or report.get("aggregate_backwards") != 2 or report.get("physical_backwards") != 4
            or report.get("import",{}).get("counters",{}).get("optimizer_updates") != 128
            or report.get("determinism",{}).get("deterministic_algorithms") is not True
            or not report.get("integrity") or not all(report["integrity"].values())
            or not report.get("pair_integrity") or not all(report["pair_integrity"].values())
            or not report.get("sources") or any(sources.get(name) != digest for name,digest in report["sources"].items())):
        raise ValueError("Require healthy matched long NF CE endpoint128 authority")
    rows = {}
    for path_name in (FP32,BF16):
        matches = [row for row in report.get("rows",[]) if row.get("path") == path_name and row.get("objective") == "ce"]
        if len(matches) != 1 or not matches[0].get("passed") or not matches[0].get("health") or not all(matches[0]["health"].values()):
            raise ValueError("NF reference lacks both healthy CE endpoints")
        rows[path_name] = matches[0]
    return report,rows


def reference_checks(current, reference):
    keys = ("source_checkpoint","initial_state","contract","fixture_pins","fixture_metadata","runtime",
            "determinism","math_sdpa_reduced_precision_reduction","bf16_matmul_reduced_precision_reduction")
    checks = {key:tree_digests(current[key]) == tree_digests(reference[key]) for key in keys}
    checks["training_data_authority_exact"] = (current["fixture_provenance"]["training_manifest_sha256"]
        == current["import"]["configuration"]["data_manifest_sha256"])
    return checks


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","fixture","nf-ce-report"):
        p.add_argument("--"+name,type=Path,required=True)
        p.add_argument("--"+name+"-sha256",required=True)
    p.add_argument("--artifacts",type=Path,default=ROOT/".runtime/olmo1b-step60000/artifacts")
    p.add_argument("--output-dir",type=Path,required=True)
    args = p.parse_args(argv)
    for value in (args.checkpoint_sha256,args.fixture_sha256,args.nf_ce_report_sha256):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            p.error("Require independent lowercase SHA256 pins")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT):
        p.error("Evidence must remain within the persistent project")
    return args


def main(argv=None):
    args = parse_args(argv)
    deterministic = configure_determinism(True)
    runtime = require_container_gpu()
    if torch.distributed.is_initialized():
        raise RuntimeError("Component probe is one GPU/process, without DDP")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources = source_hashes()
    for name in sources:
        destination = args.output_dir/"source-snapshot"/name
        destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,destination)
    report = {"schema":SCHEMA,"status":"running","passed":False,"sources":sources,
        "runtime":runtime,"determinism":deterministic,"started_utc":datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256":args.checkpoint_sha256,"fixture_sha256":args.fixture_sha256,
        "nf_ce_report_sha256":args.nf_ce_report_sha256,"plan":[list(row) for row in PLAN],"rows":[],
        "aggregate_backwards":6,"physical_backwards":12,"optimizer_updates":0,
        "math_sdpa_reduced_precision_reduction":torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        "bf16_matmul_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "qualification":"Post-NF-startup component functionality/precision; no RT training, acceptance-budget change or production clearance"}
    tracker = OnlineTracker(project="pretrained-fbt-rt-nextlat",output_dir=args.output_dir,
        group="olmo-fusion-startup",name=args.output_dir.name,preserve_state=preserve_local_rng)
    start,failure = time.monotonic(),None
    def persist(stage):
        report.update(stage=stage,elapsed_seconds=time.monotonic()-start,wandb=tracker.record)
        write_json(args.output_dir/"report.json",report)
    try:
        tracker.start({key:report[key] for key in ("checkpoint_sha256","fixture_sha256","nf_ce_report_sha256","qualification")})
        reference,anchors = load_nf_reference(args.nf_ce_report,args.nf_ce_report_sha256,sources,
            fixture_sha256=args.fixture_sha256,checkpoint_sha256=args.checkpoint_sha256)
        persist("construct_nf")
        model,recipe,source,_,_ = construct(SimpleNamespace(scale="pretrained",length=16,artifacts=args.artifacts),"NF",torch.device("cuda"))
        if recipe.rt_layers != (0,15):
            raise ValueError("Production bridge requires existing native RT layers0/15")
        original = {name:getattr(model.backbone.backbone,name) for name in RUNTIME_FLAGS}
        if state_pins(model) != reference["cold_state"]:
            raise AssertionError("Cold construction differs from NF endpoint authority")
        report["import"] = load_fusion_checkpoint(model,args.checkpoint,source,expected_sha256=args.checkpoint_sha256)
        if report["import"]["counters"]["optimizer_updates"] != 128:
            raise ValueError("Component bridge requires saved fusion128")
        fixtures,metadata = load_long_fixture(args.fixture,expected_sha256=args.fixture_sha256,recipe=recipe,width=model.config.model_dim)
        report.update(source_checkpoint=source,initial_state=state_pins(model),contract=arm_contract(model,recipe),
            fixture_pins=fixture_pins(fixtures),fixture_metadata=global_fixture_metadata(model,fixtures),fixture_provenance=metadata)
        checks = reference_checks(report,reference)
        report["nf_reference_checks"] = checks
        if not all(checks.values()):
            raise AssertionError("Component bridge differs from saved-state NF reference")
        def publish(row):
            row["memory"] = memory(); report["rows"].append(row); persist(row["arm"]+"/"+row["objective"]+"/"+row["path"])
            tracker.log(scalar_metrics(row,"components/"+row["arm"]+"/"+row["objective"]+"/"+row["path"]),step=len(report["rows"]))
            print({"arm":row["arm"],"objective":row["objective"],"path":row["path"],"passed":row["passed"]},flush=True)
        bridge = measure_bridge(model,recipe,fixtures,original_flags=original,nf_ce_anchors=anchors,publish=publish)
        report["bridge"] = {key:value for key,value in bridge.items() if key != "rows"}
        report["integrity"] = {"sources_unchanged":sources == source_hashes(),
            "checkpoint_unchanged":sha256_file(args.checkpoint) == args.checkpoint_sha256,
            "fixture_file_unchanged":sha256_file(args.fixture) == args.fixture_sha256,
            "nf_ce_report_unchanged":sha256_file(args.nf_ce_report) == args.nf_ce_report_sha256,
            "state_unchanged":state_pins(model) == report["initial_state"],
            "gradients_cleared":all(p.grad is None for p in model.parameters()),"six_rows":len(report["rows"]) == 6,
            "production_flags_restored":all(getattr(model.backbone.backbone,name) == value for name,value in original.items())}
        if not all(report["integrity"].values()):
            raise AssertionError("Component bridge final integrity failed")
        report.update(status="passed_operational_diagnostic",passed=True); persist("complete")
    except BaseException as error:
        failure=error
        report.update(status="failed",passed=False,error={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
        raise
    finally:
        report["finished_utc"]=datetime.now(timezone.utc).isoformat(); persist(report.get("stage","setup"))
        try:
            tracker.finish(succeeded=report["passed"])
        except BaseException as error:
            report.update(status="failed",passed=False,tracking_finish_error={"type":type(error).__name__})
            if failure is None: raise
        finally: persist(report.get("stage","setup"))


if __name__ == "__main__":
    main()
