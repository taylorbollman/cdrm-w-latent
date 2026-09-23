#!/usr/bin/env python3
"""Validate multiple-layer RT functionality and resource evidence without quality claims."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrm.pretrained.artifacts import write_json
from cdrm.pretrained.olmo import OLMoConfig
from cdrm.pretrained.nextlat import NextLatConfig
from cdrm.pretrained.olmo_fbt import FBTMode
from cdrm.pretrained.recurrent import RTMode
from cdrm.pretrained.resource_estimates import LossWork, estimate_training_resources, architecture_parameter_counts
from scripts import olmo_f3b_report as prior
from scripts.olmo_f3_report import tensor_check, metrics_check, _state_health
from scripts.olmo_f3e_validate import SOURCES as RUNTIME_SOURCES

SCHEMA = "olmo-f3e-native-v1"
PROTOCOL = "docs/reports/olmo1b-f3e/protocol.md"
LAYOUTS = {"single": [0], "adjacent2": [0, 1], "spread2": [0, 15],
           "spread4": [0, 5, 10, 15], "all16": list(range(16))}
require, number, digest = prior.require, prior.number, prior.digest

def source_lineage(report, directory, project_root):
    require(isinstance(report.get("source_hashes"), dict) and set(RUNTIME_SOURCES) <= report["source_hashes"].keys(),
            "Missing complete frozen F3e runtime source inventory")
    # Reuse source validation without replacing its immutable F3b protocol
    # constant or modifying any global state in the earlier reporter.
    without_protocol = {k:v for k,v in report.items() if k != "protocol_sha256"}
    lineage = prior.source_lineage(without_protocol, directory, project_root)
    expected = report.get("protocol_sha256")
    require(isinstance(expected, str) and len(expected) == 64, "Missing frozen F3e protocol hash")
    candidates = ((directory/"protocol.md", "protocol.md"),
        (directory/"source-snapshot"/PROTOCOL, "source-snapshot/"+PROTOCOL),
        (project_root/PROTOCOL, "current:"+PROTOCOL))
    found = []
    for path, label in candidates:
        if not path.is_file():
            continue
        matches = digest(path) == expected
        if not label.startswith("current:"):
            require(matches, "Corrupt F3e protocol snapshot")
        if matches:
            found.append(label)
    require(found, "Unreconstructable F3e protocol")
    lineage["protocol"] = {"sha256": expected, "verified_at": found}
    return lineage


def declared_checks(report):
    return prior.declared_checks({**report, "schema": "olmo-f3b-native-v1"})

def native_checks(report):
    result = prior.native_correctness({**report, "configuration": {**report["configuration"], "variant": "triton" if report["configuration"]["variant"] == "recompute" else "reference"}})
    first = next(c for c in report["checks"] if c["name"] == "same_state_variant_vs_reference")
    require(first.get("forward_losses_bitwise_equal") is True, "Backward-only candidate changed initial forward losses")
    for row in first["losses"].values():
        tensor_check(row)
        require(row.get("bitwise_equal") is True, "Initial loss records contradict forward exactness")
    require(first.get("ownership_matches") is True, "Native gradient participation differs")
    dispatch = dispatch_check(report, first)
    mixed = first.get("mixed_gradient_screen")
    if mixed:
        require(set(mixed) == set(first["gradients"]), "Mixed and strict native gradient inventories differ")
        for name, row in mixed.items():
            diagnostic = first["gradients"][name]
            norm = number(row.get("reference_sq"), "tensor squared reference")
            delta = number(row.get("delta_sq"), "tensor squared error")
            maximum = number(diagnostic.get("reference_max_abs"), "tensor reference maximum")
            difference = number(diagnostic.get("max_abs"), "tensor maximum error")
            relative = math.sqrt(delta/norm) if norm else (0. if delta == 0 else math.inf)
            max_relative = difference/maximum if maximum else (0. if difference == 0 else math.inf)
            require(math.isclose(number(row.get("relative_l2"), "tensor relative error"), relative, rel_tol=1e-10, abs_tol=1e-14)
                    and math.isclose(number(row.get("max_relative"), "tensor maximum ratio"), max_relative, rel_tol=1e-10, abs_tol=1e-14),
                    "Mixed tensor errors contradict recorded norms/maxima")
            require(diagnostic.get("finite") is True, "Native gradient contains nonfinite values")
            if number(diagnostic.get("reference_max_abs"), "native reference maximum") == 0:
                require(number(diagnostic.get("max_abs"), "native absolute error") == 0,
                        "Zero native tensor reference requires exact zero")
            if number(row.get("reference_sq"), "tensor squared reference") == 0:
                require(number(row.get("delta_sq"), "tensor squared error") == 0,
                        "Zero native tensor norm requires exact zero")
        error = sum(number(r.get("delta_sq"), "global squared error") for r in mixed.values())
        reference = sum(number(r.get("reference_sq"), "global squared reference") for r in mixed.values())
        if reference == 0:
            require(error == 0, "Zero global gradient reference requires exact zero")
        expected_global = math.sqrt(error/reference) if reference else 0.
        require(math.isclose(first["global_gradient_relative_l2"], expected_global, rel_tol=1e-10, abs_tol=1e-14),
                "Recorded global gradient error contradicts tensor sums")
    for check in report["checks"]:
        if check["name"] == "complete_adamw_update_parity": continue
        bitwise = all(row.get("bitwise_equal") is True for row in (*check["losses"].values(), *check["gradients"].values()))
        require(check.get("all_bitwise_equal") is bitwise, "Aggregate bitwise flag contradicts tensor observations")
    update = next(check for check in report["checks"] if check["name"] == "complete_adamw_update_parity")
    expected_counts = {key:report["resources"]["analytic_matrix_work"]["objective_positions_per_update"][key] for key in ("ce","latent","kl")}
    for arm in update["arms"]:
        for step,metrics in enumerate(arm["metrics"],1):
            metrics_check(metrics, report["configuration"], step)
            require(metrics["counts"] == expected_counts, "Optimizer objective counts disagree with prepared resource card")
        boundary = arm.get("boundary")
        require(isinstance(boundary, dict) and set(boundary) == {"model","optimizer","scheduler","counters"}
                and boundary["model"] and boundary["optimizer"] and boundary["scheduler"], "Missing complete optimizer-boundary state")
        require(boundary["counters"] == arm["metrics"][-1]["counters"], "Boundary counters differ from final update")
        _state_health(arm.get("health"))
    return {**result, "forward_losses_bitwise_equal": True,
        "global_gradient_relative_l2": first.get("global_gradient_relative_l2"),
        "max_tensor_gradient_relative_l2": max((r["relative_l2"] for r in (first.get("mixed_gradient_screen") or {}).values()), default=0.),
        "max_tensor_gradient_error_reference_max": max((r["max_relative"] for r in (first.get("mixed_gradient_screen") or {}).values()), default=0.),
        "observed_recompute_backward_calls": first["candidate_recompute_backward_calls"], "dispatch": dispatch}


def configuration_check(report):
    c = report.get("configuration", {})
    require(c.get("layout") in LAYOUTS, "Unknown layer layout")
    layers = LAYOUTS[c["layout"]]
    require(c.get("selected_rt_layers") == layers, "Selected RT layers contradict layout")
    role = ("optional_all_layer_stress" if c["layout"] == "all16" else
            "single_layer_reference" if c["layout"] == "single" else "primary_multi_layer_integration")
    require(c.get("layout_role") == role, "Layer role contradicts protocol")
    require(c.get("case") in ("rt", "combined", "combined-k3"), "Unknown architecture case")
    require(c.get("stage") == report.get("stage") and c["stage"] in ("correctness", "capacity"), "Stage/configuration differs")
    require(c.get("case") != "combined-k3" or c["stage"] == "correctness", "K3 capacity is outside protocol")
    require(type(c.get("batch_size")) is int and 1 <= c["batch_size"] <= 128
            and type(c.get("length")) is int and c["length"] in (32,128,512,1024,2048), "Invalid native shape")
    enabled = c["case"] != "rt"
    passes = 3 if c["case"] == "combined-k3" else (2 if enabled else 1)
    mode = FBTMode(enabled=enabled, num_passes=passes, rt_mode=RTMode(tuple(layers)))
    require(c.get("mode") == {"enabled": enabled, "num_passes": passes, "beta": 1.,
                              "rt_mode": {"selected_layers": layers, "alpha": 1.}}, "Mode differs from selected architecture")
    expected_calls = len(layers) * (passes-1 if enabled else 1)
    require(type(c.get("rt_block_calls_per_forward_backward")) is int
            and c["rt_block_calls_per_forward_backward"] == expected_calls, "RT call multiplicity differs")
    case = c.get("case_specification", {})
    require(all(case.get(k) == v for k,v in {"name":c["case"], "fbt":enabled, "nextlat":enabled,
        "rt_layers":layers, "passes":3 if c["case"] == "combined-k3" else 2, "alpha":1., "beta":1.,
        "batch_size":c["batch_size"], "length":c["length"], "updates":3,
        "transition":False, "resume":False, "profile":False}.items()), "Case specification contradicts configuration")
    require(c.get("variant") in ("reference", "recompute") and c.get("backward_memory") ==
            ("recompute" if c["variant"] == "recompute" else "materialized"), "Backward memory contradicts variant")
    require(c.get("cast_weights_once") is True and c.get("forward_tile_backend") == c.get("backward_tile_backend") == "triton"
            and c.get("ordinary_activation_checkpointing") is True and c.get("ordinary_attention_backend") == "deterministic_flash"
            and c.get("precision") == "bf16_mixed" and c.get("tf32") is False and c.get("autocast_weight_cache") is False,
            "Execution configuration differs from frozen protocol")
    return mode


def dispatch_check(report, first):
    c = report["configuration"]
    mode = configuration_check(report)
    passes = mode.num_passes-1 if mode.enabled else 1
    length = c["length"]
    # Count the actual historical rectangles independently of reported totals.
    fused_materialized = sum(max(index & -index, min(index & -index, length-index)) <= 256
                             for index in range(1, length))
    checked = {}
    for arm, variant in (("reference", "reference"), ("candidate", c["variant"])):
        by_layer = {str(index): {"forward_blocks":passes, "backward_blocks":passes,
            "forward_tiles":passes*(length-1), "forward_fused_tiles":passes*fused_materialized,
            "forward_eager_tiles":passes*(length-1-fused_materialized),
            "recompute_tiles":passes*(length-1) if variant == "recompute" else 0,
            "materialized_tiles":passes*(length-1) if variant == "reference" else 0,
            "materialized_fused_tiles":passes*fused_materialized if variant == "reference" else 0}
            for index in c["selected_rt_layers"]}
        totals = {key:sum(row[key] for row in by_layer.values()) for key in next(iter(by_layer.values()))}
        expected = {"by_layer":by_layer, "totals":totals}
        for prefix in ("", "expected_"):
            value = first.get(prefix+arm+"_dispatch")
            require(value == expected, "Per-layer dispatch differs from geometry/pass multiplicity")
            require(all(type(n) is int for row in value["by_layer"].values() for n in row.values())
                    and all(type(n) is int for n in value["totals"].values()), "Dispatch counts must be integer observations")
        checked[arm] = expected
    count = checked["candidate"]["totals"]["recompute_tiles"]
    require(first.get("dispatch_matches") is True and type(first.get("reference_recompute_backward_calls")) is int
            and first["reference_recompute_backward_calls"] == 0
            and type(first.get("candidate_recompute_backward_calls")) is int
            and type(first.get("expected_candidate_recompute_backward_calls")) is int
            and first["candidate_recompute_backward_calls"] == first["expected_candidate_recompute_backward_calls"] == count,
            "Native dispatch aliases disagree with per-layer observations")
    return checked


def resource_check(report):
    c = report["configuration"]
    mode = configuration_check(report)
    card = report.get("resources")
    require(isinstance(card, dict), "Missing actual-layout resource card")
    b, t = c["batch_size"], c["length"]
    auxiliary = c["case"] != "rt"
    expected_work = {"ce_targets":b*(t//2), "latent_pairs":b*(t-1) if auxiliary else 0,
                     "kl_triples":b*(t//2) if auxiliary else 0, "predictor_positions":b*(t-1) if auxiliary else 0}
    require(card.get("loss_work") == expected_work, "Prepared loss counts contradict frozen fixture masks")
    require(card.get("loss_weights") == {"ce":1., "latent":float(auxiliary), "kl":float(auxiliary)}, "Objective weights differ")
    config = OLMoConfig.native_1b()
    nextlat = NextLatConfig(config.model_dim) if auxiliary else None
    expected = estimate_training_resources(config, batch_size=b, sequence_length=t, mode=mode,
        nextlat=nextlat, loss_work=LossWork(**expected_work), ordinary_checkpointing=True,
        backward_memory=c["backward_memory"]).to_dict()
    # Descriptive prose may evolve; arithmetic, count inventories and selected policy may not.
    def arithmetic(value):
        require(isinstance(value, dict), "Missing analytic matrix estimate")
        return {k: [{key:row.get(key) for key in ("name","minimum","maximum")} for row in v]
                if k == "components" else v for k,v in value.items() if k != "assumptions"}
    require(arithmetic(card.get("analytic_matrix_work")) == arithmetic(expected), "Resource matrix/count ledger differs")
    observed = card.get("observed_parameters", {})
    active = expected["parameter_counts"]["training_architecture"]
    resident = architecture_parameter_counts(config, fbt=True, nextlat=nextlat)["training_architecture"]
    expected_observed = {"registered_unique":resident, "resident_parameter_bytes":4*resident,
        "trainable":active, "gradient_participating":active, "executed_declared":active,
        "deployable_inference_declared":expected["parameter_counts"]["deployable_inference"]}
    require(all(type(observed.get(k)) is int and observed[k] == v for k,v in expected_observed.items()),
            "Observed parameter ownership contradicts native architecture/active branches")
    require(observed.get("optimizer_owned") == active if report["stage"] == "capacity" else
            observed.get("optimizer_owned") in (None, active), "Optimizer ownership differs from active weights")
    expected_shapes = parameter_shapes(auxiliary)
    require(card.get("named_parameter_shapes") == expected_shapes, "Parameter shape/name inventory differs")
    executed = set(expected_shapes) if auxiliary else {name for name in expected_shapes if not name.startswith("backbone.fusion.")}
    inference = {name for name in executed if not name.startswith("predictor.")}
    for field,names in (("declared_execution_names",executed),("declared_inference_names",inference)):
        require(card.get(field) == sorted(names), "Declared executed/inference parameter inventory differs")
    if report["stage"] == "correctness":
        passes = mode.num_passes if mode.enabled else 1
        losses = {f"pass{p}/{term}" for p in range(passes) for term in ("ce","latent","kl")}
        for check in report["checks"]:
            if check["name"] != "complete_adamw_update_parity":
                require(set(check.get("gradients",{})) == executed and set(check.get("losses",{})) == losses,
                        "Native correctness omitted active parameter/loss comparisons")
    return card


def parameter_shapes(auxiliary):
    config = OLMoConfig.native_1b()
    d, m = config.model_dim, config.mlp_intermediate_size
    shapes = {"backbone.backbone.transformer.wte.weight":[config.vocab_size,d],
              "backbone.fusion.state_proj.weight":[d,d], "backbone.fusion.token_gate.weight":[d,d]}
    for index in range(config.num_layers):
        for name,shape in (("att_proj",[3*d,d]),("attn_out",[d,d]),("ff_proj",[2*m,d]),("ff_out",[d,m])):
            shapes[f"backbone.backbone.transformer.blocks.{index}.{name}.weight"] = shape
    if auxiliary:
        hidden = NextLatConfig(d).hidden_dim
        shapes.update({"predictor.mlp.0.weight":[hidden,2*d],"predictor.mlp.2.weight":[hidden,hidden],
                       "predictor.mlp.4.weight":[d,hidden],"predictor.norm_x.weight":[2*d]})
    return shapes


def full_step_record(report, run_name):
    record = prior.full_step_record(report, run_name)
    card = resource_check(report)
    c, data = report["configuration"], report["capacity"]
    require({k:data.get(k) for k in ("warmup_updates","timed_updates","physical_optimizer_updates","backward_only_warmup")}
            == {"warmup_updates":3,"timed_updates":3,"physical_optimizer_updates":6,"backward_only_warmup":10},
            "Capacity update/warmup scope differs")
    require(math.isclose(number(data.get("ce_targets_per_second"), "CE targets/s", positive=True),
            record["ce_targets_per_second"], rel_tol=1e-10), "CE throughput contradicts observed counts")
    for step,metrics in enumerate(data["records"],4):
        metrics_check(metrics, c, step)
    _state_health(data.get("health"))
    counts = {key:card["analytic_matrix_work"]["objective_positions_per_update"][key] for key in ("ce","latent","kl")}
    require(all(row.get("counts") == counts for row in data["records"]), "Measured loss counts changed across updates")
    for bound in ("minimum", "maximum"):
        expected = card["analytic_matrix_work"]["matrix_flops_"+bound]/record["full_step"]["median_wall_seconds"]/1e12
        require(math.isclose(number(data.get("estimated_matrix_tflops_per_second_"+bound), "matrix rate", positive=True),
                expected, rel_tol=1e-10), "Analytic matrix rate contradicts timing")
    return {**record, "layout":c["layout"], "selected_rt_layers":c["selected_rt_layers"],
            "layout_role":c["layout_role"], "resources":card}


def validate_success(report):
    require(report.get("schema") == SCHEMA and report.get("status") == "passed", "Require passed native F3e report")
    declared_checks(report)
    configuration_check(report)
    require(report.get("checkpoint",{}).get("sha256") == prior.CHECKPOINT_SHA256, "Wrong native checkpoint")
    resource_check(report)
    require(report.get("backward_preparation") == {"warmup":11,"capture":1,"replay":7 if report["stage"] == "correctness" else 3},
            "Backward preparation/replay counts contradict recorded execution scope")
    if report["stage"] == "correctness":
        return native_checks(report)
    require({c["name"] for c in report["checks"]} == {"finite_complete_updates"}, "Missing finite complete-update gate")
    return full_step_record(report, "validation")


def summarize(runtime_dirs, *, project_root=ROOT, allow_incomplete=False):
    project_root = Path(project_root)
    summary = {"schema":"olmo-f3e-summary-v1", "generated_utc":datetime.now(timezone.utc).isoformat(),
        "status":"passed", "primary_status":"not_assessed", "runs":[], "correctness":[], "full_steps":[],
        "capability_ledger":[], "resource_cards":[], "failed_diagnostics":[], "incomplete":[],
        "successful_f3e_optimizer_updates":{"eager":0,"graph":0,"total":0},
        "scope":"Multiple selected RT layers, not architecture selection. All16 is optional stress. No quality, placement-superiority, FA4 or multi-GPU claim."}
    seen = set()
    for item in runtime_dirs:
        directory = Path(item).resolve()
        require(directory.name not in seen, "Duplicate runtime name"); seen.add(directory.name)
        path = directory/"report.json"
        if not path.is_file():
            require(allow_incomplete and directory.is_dir(), "Missing runtime report")
            summary["incomplete"].append({"run":directory.name, "status":"no_report_yet"}); continue
        report = json.loads(path.read_text())
        require(report.get("schema") == SCHEMA, "Unknown F3e schema")
        if report.get("status") == "running":
            require(allow_incomplete, "Active report requires explicit preview")
            summary["incomplete"].append({"run":directory.name, "status":"running"}); continue
        status = report.get("status")
        require(status in ("passed","failed","capture_blocked") and report.get("finished_utc"), "Report is not completed")
        configuration_check(report)
        checks = declared_checks(report)
        run = {"name":directory.name, "status":status, "configuration":report["configuration"],
            "runtime":report.get("runtime"), "source_lineage":source_lineage(report,directory,project_root),
            "report_path":str(path), "report_sha256":digest(path), "checks":checks,
            "wandb_url":report.get("wandb",{}).get("run_url"), "used_for_performance":False}
        summary["runs"].append(run)
        c = report["configuration"]
        capability = {"run":directory.name, "case":c["case"], "layout":c["layout"], "selected_rt_layers":c["selected_rt_layers"],
            "layout_role":c["layout_role"], "batch_size":c["batch_size"], "length":c["length"],
            "stage":report["stage"], "backward_memory":c["backward_memory"], "status":status}
        summary["capability_ledger"].append(capability)
        if status != "passed":
            require(report.get("error_type") and isinstance(report.get("error_message"), str), "Failed report lacks error")
            summary["failed_diagnostics"].append({**capability, "error_type":report["error_type"], "error_message":report["error_message"],
                "declared_failed_checks":[row["name"] for row in checks if not row["passed"]], "excluded_partial_performance":bool(report.get("capacity"))})
            if c["layout_role"] == "primary_multi_layer_integration": summary["primary_status"] = "completed_with_failed_diagnostics"
            continue
        result = validate_success(report)
        if c["layout_role"] == "primary_multi_layer_integration" and summary["primary_status"] == "not_assessed":
            summary["primary_status"] = "passed"
        if report["stage"] == "correctness":
            summary["correctness"].append({"run":directory.name,"configuration":c,**result})
        else:
            summary["full_steps"].append(full_step_record(report,directory.name)); run["used_for_performance"] = True
        summary["resource_cards"].append({**capability, **report["resources"]})
        for key,count in (("eager",3),("graph",3),("total",6)):
            summary["successful_f3e_optimizer_updates"][key] += count
    require(summary["runs"] or summary["incomplete"], "No F3e runs supplied")
    if summary["incomplete"]: summary["status"] = "partial_preview"
    elif summary["failed_diagnostics"]: summary["status"] = "completed_with_failed_diagnostics"
    summary["declared_check_count"] = sum(len(row["checks"]) for row in summary["runs"])
    summary["declared_checks_passed"] = sum(check["passed"] for row in summary["runs"] for check in row["checks"])
    summary["successful_declared_check_count"] = sum(len(row["checks"]) for row in summary["runs"] if row["status"] == "passed")
    return summary


def markdown(summary):
    lines = ["# F3e multiple selected RT layers", "", f"Status: **{summary['status']}**; primary-layout status: **{summary['primary_status']}**.", "",
        "Statuses describe the supplied completed runs, not automatic clearance of every planned case. "
        "These are functionality and execution measurements. Two/four-layer layouts are the primary integration scope; "
        "all16 is an optional stress case, not an assumed main architecture. No quality, placement-superiority, native FA4 or multi-GPU claim follows.", "",
        "| Run | Case / layout | RT indices | Role | B/T | Stage | Passed / declared |", "| --- | --- | --- | --- | ---: | --- | ---: |"]
    for run in summary["runs"]:
        c=run["configuration"]; label=f"[{run['name']}]({run['wandb_url']})" if run.get("wandb_url") else run["name"]
        lines.append(f"| {label} | {c['case']} / {c['layout']} | {c['selected_rt_layers']} | {c['layout_role']} | {c['batch_size']}/{c['length']} | "
                     f"{c['stage']} ({run['status']}) | {sum(v['passed'] for v in run['checks'])}/{len(run['checks'])} |")
    updates = summary["successful_f3e_optimizer_updates"]
    lines += ["", f"Successful runs contain {updates['total']} physical optimizer updates ({updates['eager']} eager + {updates['graph']} graph). "
        "Warmup/backward-only work and failed attempts are excluded.", "", "## Correctness", "",
        "Initial materialized-versus-recompute gradients retain global relative L2 ≤1/64, per tensor ≤1/32 and maximum error/reference maximum ≤1/16. "
        "Zero references require exact zero. Same-candidate graph checks retain tighter budgets; full Adam/state comparisons require exact parity. "
        "Stricter diagnostic flags remain in summary.json.", "",
        "| Run | Initial loss exact | Global gradient relative L2 | Recompute tiles | Graph exact | Full Adam exact |",
        "| --- | --- | ---: | ---: | --- | --- |"]
    for row in summary["correctness"]:
        lines.append(f"| {row['run']} | {row['forward_losses_bitwise_equal']} | {row['global_gradient_relative_l2']:.6g} | "
                     f"{row['observed_recompute_backward_calls']} | {row['candidate_graph_checks_bitwise']} | {row['complete_updates_exact']} |")
    lines += ["", "## Complete-update resources", "",
        "Wall time includes input copy/validation, graph forward/loss/backward, clipping, AdamW and scheduler. "
        "Three-update medians are directional. Peak allocated includes setup; reserved peak and current reserved are distinct. "
        "Matrix FLOPs are an analytic ledger excluding elementwise/optimizer/communication work and kernel padding, not hardware utilization.", "",
        "| Run | Case / layout | B/T | Input tokens/s | CE targets/s | Seconds/update | Allocated / reserved peak / current GiB | Matrix TFLOPs/update |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["full_steps"]:
        estimate=row["resources"]["analytic_matrix_work"]
        lines.append(f"| {row['run']} | {row['case']} / {row['layout']} | {row['batch_size']}/{row['length']} | {row['input_tokens_per_second']:,.0f} | "
                     f"{row['ce_targets_per_second']:,.0f} | {row['full_step']['median_wall_seconds']:.4f} | "
                     f"{row['peak_allocated_gib']:.3f} / {row['peak_reserved_gib']:.3f} / {row['current_reserved_gib']:.3f} | "
                     f"{estimate['matrix_flops_minimum']/1e12:.2f}–{estimate['matrix_flops_maximum']/1e12:.2f} |")
    lines += ["", "Parameter ownership and actual objective counts are recorded per run in resource-ledger.json. "
        "RT adds no parameters; multiple FBT passes share weights. The NextLat predictor is training-only. "
        "Capacity health at a larger batch is not a full gradient-equivalence check at that batch."]
    for failure in summary["failed_diagnostics"]:
        lines += ["", f"Failed diagnostic `{failure['run']}` ({failure['layout_role']}): {failure['error_type']}: {failure['error_message']}. "
            "Its partial timing is excluded; its original source/protocol/error record is retained."]
    return "\n".join(lines)+"\n"


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=summary["full_steps"]
    if not rows: return []
    figure, axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    labels=[f"{row['case']} {row['layout']}\nB{row['batch_size']} T{row['length']}" for row in rows]
    for axis,field,title in ((axes[0],"input_tokens_per_second","Full-update input tokens/s"),
                             (axes[1],"peak_allocated_gib","Peak allocated GiB, including setup")):
        axis.bar(range(len(rows)),[row[field] for row in rows]); axis.set_title(title)
        axis.set_xticks(range(len(rows)),labels,rotation=30,ha="right",fontsize=8)
    paths=[]
    for suffix in ("pdf","png"):
        path=output/f"multi-rt-resources.{suffix}"; figure.savefig(path,dpi=180); paths.append(str(path))
    plt.close(figure)
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir",type=Path,action="append",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args(argv); summary=summarize(args.runtime_dir,allow_incomplete=args.allow_incomplete)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/"summary.json",summary)
    write_json(args.output_dir/"capability-ledger.json",summary["capability_ledger"])
    write_json(args.output_dir/"resource-ledger.json",summary["resource_cards"])
    (args.output_dir/"results.md").write_text(markdown(summary)); plot(summary,args.output_dir)
    print(json.dumps({"status":summary["status"],"checks":summary["declared_check_count"]}))


if __name__ == "__main__": main()
