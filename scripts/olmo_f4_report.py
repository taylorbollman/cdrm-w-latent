#!/usr/bin/env python3
"""Validate eight independent RT/FBT/NextLat resource cells without quality claims."""
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
from scripts.olmo_f3_report import tensor_check, _state_health, integer, close, healthy, number
from scripts.olmo_f3e_validate import SOURCES as F3E_SOURCES

SCHEMA = "olmo-f4-native-v1"
PROTOCOL = "docs/reports/olmo1b-f4/protocol.md"
RUNTIME_SOURCES = tuple(sorted(set(F3E_SOURCES) | {"scripts/olmo_f4_resources.py"}))
# Independent switches: neither recurrence implies the auxiliary NextLat loss.
FEATURES = {"ordinary": (False, False, False), "rt": (True, False, False),
    "fbt": (False, True, False), "nextlat": (False, False, True),
    "rt-fbt": (True, True, False), "rt-nextlat": (True, False, True),
    "fbt-nextlat": (False, True, True), "combined": (True, True, True)}
EVENTS = ("forward_blocks", "forward_tiles", "forward_fused_tiles", "forward_eager_tiles",
          "backward_blocks", "recompute_tiles", "materialized_tiles", "materialized_fused_tiles")
TERMS = ("ce", "latent", "kl")
require, digest = prior.require, prior.digest


def feature_flags(case):
    require(case in FEATURES, "Unknown independent feature case")
    return dict(zip(("rt", "fbt", "nextlat"), FEATURES[case]))


def metrics_check(metrics, config, update):
    """Validate objective semantics from explicit NextLat, never a case shortcut."""
    require(metrics.get("schema") == "olmo-lm-optimizer-step-v1" and metrics.get("update_completed") is True,
            "Missing completed canonical optimizer update")
    counts = metrics.get("counts", {})
    require(set(counts) == set(TERMS), "Missing objective counts")
    flags = feature_flags(config["case"])
    b, t = config["batch_size"], config["length"]
    expected_counts = {"ce": b*(t//2), "latent": b*(t-1) if flags["nextlat"] else 0,
                       "kl": b*(t//2) if flags["nextlat"] else 0}
    require(counts == expected_counts, "Canonical counts contradict independent NextLat flag")
    for term in TERMS:
        integer(counts[term], term + " count")
        total = number(metrics.get("loss_sums", {}).get(term), term + " loss sum", signed=True)
        close(metrics.get("loss_means", {}).get(term), total/counts[term] if counts[term] else 0, term + " denominator")
        if counts[term] == 0:
            require(total == 0, "Inactive objective has nonzero loss")
    weights = {"ce": 1., "latent": float(flags["nextlat"]), "kl": float(flags["nextlat"])}
    require(metrics.get("objective_weights") == weights, "Canonical objective weights changed")
    close(metrics.get("objective"), sum(weights[t]*metrics["loss_means"][t] for t in TERMS), "objective")
    number(metrics.get("gradient_norm_before_clip"), "gradient norm", positive=True)
    require(metrics.get("max_grad_norm") == 1., "Canonical clipping changed")
    for key in ("lr_used", "lr_next"):
        require(isinstance(metrics.get(key), list) and metrics[key], "Missing learning rates")
        for value in metrics[key]:
            number(value, key, positive=key == "lr_used")
    expected_counters = {"optimizer_updates": update, "microbatches": update,
        "documents": update*b, "input_tokens": update*b*t,
        "ce_positions": update*counts["ce"], "latent_pairs": update*counts["latent"], "kl_triples": update*counts["kl"]}
    require(metrics.get("counters") == expected_counters, "Optimizer counters differ from physical updates")
    healthy(metrics)

def source_lineage(report, directory, project_root, *, sources=RUNTIME_SOURCES, protocol=PROTOCOL):
    require(isinstance(report.get("source_hashes"), dict) and set(sources) <= report["source_hashes"].keys(),
            "Missing complete frozen F4 runtime source inventory")
    # Reuse source validation without replacing its immutable F3b protocol
    # constant or modifying any global state in the earlier reporter.
    without_protocol = {k:v for k,v in report.items() if k != "protocol_sha256"}
    lineage = prior.source_lineage(without_protocol, directory, project_root)
    expected = report.get("protocol_sha256")
    require(isinstance(expected, str) and len(expected) == 64, "Missing frozen F4 protocol hash")
    candidates = ((directory/"protocol.md", "protocol.md"),
        (directory/"source-snapshot"/protocol, "source-snapshot/"+protocol),
        (project_root/protocol, "current:"+protocol))
    found = []
    for path, label in candidates:
        if not path.is_file():
            continue
        matches = digest(path) == expected
        if not label.startswith("current:"):
            require(matches, "Corrupt F4 protocol snapshot")
        if matches:
            found.append(label)
    require(found, "Unreconstructable F4 protocol")
    lineage["protocol"] = {"sha256": expected, "verified_at": found}
    return lineage


def declared_checks(report):
    return prior.declared_checks({**report, "schema": "olmo-f3b-native-v1"})

def native_checks(report):
    result = prior.native_correctness({**report, "configuration": {**report["configuration"], "variant": "triton" if report["configuration"]["variant"] == "recompute" else "reference"}})
    first = next(c for c in report["checks"] if c["name"] == "same_state_variant_vs_reference")
    require(set(first.get("mixed_loss_screen", {})) == set(first.get("losses", {})),
            "Mixed and strict loss inventories differ")
    require(first.get("forward_losses_bitwise_equal") is True, "Backward-only candidate changed initial forward losses")
    for row in first["losses"].values():
        tensor_check(row)
        require(row.get("bitwise_equal") is True, "Initial loss records contradict forward exactness")
    require(first.get("ownership_matches") is True, "Native gradient participation differs")
    if not feature_flags(report["configuration"]["case"])["rt"]:
        require(first.get("all_bitwise_equal") is True, "Unused RT memory switch changed non-RT computation")
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
        if check["name"].startswith("candidate_"):
            require(bitwise, "F4 same-candidate graph checks require bitwise equality")
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
    flags = feature_flags(c.get("case"))
    require(c.get("feature_flags") == flags and all(type(v) is bool for v in c["feature_flags"].values()),
            "Independent feature flags contradict case")
    layers = [0, 15] if flags["rt"] else []
    require(c.get("layout") == ("spread2" if flags["rt"] else "none"), "Layout contradicts RT switch")
    require(c.get("selected_rt_layers") == layers, "Selected RT layers contradict layout")
    require(c.get("layout_role") == "common_feature_comparison", "Layer role contradicts protocol")
    require(c.get("stage") == report.get("stage") and c["stage"] in ("correctness", "capacity"), "Stage/configuration differs")
    require(type(c.get("batch_size")) is int and 1 <= c["batch_size"] <= 96
            and type(c.get("length")) is int and c["length"] in (32,512), "Invalid native shape")
    require(c["stage"] != "capacity" or c["batch_size"] in (64,96) and c["length"] == 512,
            "Capacity is bounded to common B64/T512 or optional B96/T512")
    require(c.get("world_size") == c.get("accumulation_steps") == 1
            and type(c.get("world_size")) is int and type(c.get("accumulation_steps")) is int
            and c.get("physical_batch_per_gpu") == c.get("logical_batch") == c["batch_size"],
            "Physical/logical/world-size accounting differs")
    enabled = flags["fbt"]
    passes = 2 if enabled else 1
    mode = FBTMode(enabled=enabled, num_passes=passes, rt_mode=RTMode(tuple(layers)))
    require(c.get("mode") == {"enabled": enabled, "num_passes": passes, "beta": 1.,
                              "rt_mode": {"selected_layers": layers, "alpha": 1.}}, "Mode differs from selected architecture")
    expected_calls = len(layers) * (passes-1 if enabled else 1)
    require(type(c.get("rt_block_calls_per_forward_backward")) is int
            and c["rt_block_calls_per_forward_backward"] == expected_calls, "RT call multiplicity differs")
    case = c.get("case_specification", {})
    require(all(case.get(k) == v for k,v in {"name":c["case"], "fbt":enabled, "nextlat":flags["nextlat"],
        "rt_layers":layers, "passes":2, "alpha":1., "beta":1.,
        "batch_size":c["batch_size"], "length":c["length"], "updates":3,
        "transition":False, "resume":False, "profile":False}.items()), "Case specification contradicts configuration")
    require(c.get("variant") == "recompute" and c.get("backward_memory") == "recompute", "Backward memory contradicts variant")
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
        totals = {key:sum(row[key] for row in by_layer.values()) for key in EVENTS}
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
    flags = feature_flags(c["case"])
    auxiliary = flags["nextlat"]
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
    executed = {name for name in expected_shapes if flags["fbt"] or not name.startswith("backbone.fusion.")}
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
    snapshots = {}
    for name in ("setup_memory", "steady_memory"):
        value = data.get(name)
        require(isinstance(value, dict) and set(value) == {"allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib"},
                "Missing separately measured setup/steady memory")
        snapshots[name] = {key: number(amount, name + "/" + key, positive=True) for key, amount in value.items()}
        require(value["allocated_gib"] <= value["reserved_gib"] <= value["peak_reserved_gib"]
                and value["allocated_gib"] <= value["peak_allocated_gib"] <= value["peak_reserved_gib"],
                "Memory snapshot peaks contradict allocation/reservation")
    for field in ("peak_allocated_gib", "peak_reserved_gib"):
        require(data[field] == max(value[field] for value in snapshots.values()), "Combined memory peak contradicts setup/steady measurements")
    require(data["current_reserved_gib"] == snapshots["steady_memory"]["reserved_gib"],
            "Current reserved memory contradicts steady snapshot")
    counts = {key:card["analytic_matrix_work"]["objective_positions_per_update"][key] for key in ("ce","latent","kl")}
    require(all(row.get("counts") == counts for row in data["records"]), "Measured loss counts changed across updates")
    for bound in ("minimum", "maximum"):
        expected = card["analytic_matrix_work"]["matrix_flops_"+bound]/record["full_step"]["median_wall_seconds"]/1e12
        require(math.isclose(number(data.get("estimated_matrix_tflops_per_second_"+bound), "matrix rate", positive=True),
                expected, rel_tol=1e-10), "Analytic matrix rate contradicts timing")
    return {**record, "layout":c["layout"], "selected_rt_layers":c["selected_rt_layers"],
            "layout_role":c["layout_role"], "feature_flags":c["feature_flags"],
            **snapshots, "resources":card}


def trace_check(report, directory=None):
    trace = report.get("operator_trace")
    requested = report["configuration"].get("operator_trace", False)
    require(type(requested) is bool, "Operator trace request must be Boolean")
    require((trace is not None) == requested, "Operator trace presence contradicts request")
    if trace is None:
        return None
    c = report["configuration"]
    require(c["stage"] == "correctness" and c["batch_size"] == 1 and c["length"] == 32,
            "Operator trace is outside bounded untimed correctness scope")
    require(isinstance(trace, dict) and trace.get("file") == "operator-trace.json.gz", "Unexpected operator trace artifact")
    sha = trace.get("sha256")
    require(isinstance(sha, str) and len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha), "Invalid trace digest")
    integer(trace.get("bytes"), "trace byte size", minimum=1)
    integer(trace.get("observed_device_event_count"), "device event count", minimum=1)
    names = trace.get("observed_device_event_names")
    require(isinstance(names, list) and names and all(isinstance(name, str) and name for name in names)
            and names == sorted(set(names)), "Missing unique observed device operators")
    rows = trace.get("operator_rows")
    require(isinstance(rows, list) and rows, "Missing profiler operator records")
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"], "Missing operator name")
        integer(row.get("calls"), "operator calls", minimum=1)
        integer(row.get("estimated_flops"), "selected-operator FLOPs")
        number(row.get("self_cpu_time_us"), "operator CPU time")
        number(row.get("self_device_time_us"), "operator device time")
    require(trace.get("pytorch_estimated_flops") == sum(row["estimated_flops"] for row in rows),
            "Profiler FLOP sum contradicts operator rows")
    require(isinstance(trace.get("scope"), str) and trace["scope"], "Missing profiler coverage qualification")
    if directory is not None:
        path = Path(directory) / trace["file"]
        require(path.is_file() and not path.is_symlink(), "Missing or redirected trace artifact")
        require(path.stat().st_size == trace["bytes"] and digest(path) == sha, "Corrupt operator trace artifact")
    return trace


def validate_success(report):
    require(report.get("schema") == SCHEMA and report.get("status") == "passed", "Require passed native F4 report")
    declared_checks(report)
    configuration_check(report)
    require(report.get("checkpoint",{}).get("sha256") == prior.CHECKPOINT_SHA256, "Wrong native checkpoint")
    resource_check(report)
    trace_check(report)
    require(report.get("backward_preparation") == {"warmup":11,"capture":1,"replay":7 if report["stage"] == "correctness" else 3},
            "Backward preparation/replay counts contradict recorded execution scope")
    if report["stage"] == "correctness":
        return native_checks(report)
    require({c["name"] for c in report["checks"]} == {"finite_complete_updates"}, "Missing finite complete-update gate")
    return full_step_record(report, "validation")


ROUNDOFF_ARMS = ("bf16_mixed-materialized", "bf16_mixed-recompute", "bf16_mixed-recompute-eager",
                "fp32-materialized", "fp32-recompute")
ROUNDOFF_COMPARISONS = {
    "bf16_recompute_vs_materialized": (ROUNDOFF_ARMS[1], ROUNDOFF_ARMS[0]),
    "bf16_eager_history_vs_materialized": (ROUNDOFF_ARMS[2], ROUNDOFF_ARMS[0]),
    "bf16_fused_vs_eager_history": (ROUNDOFF_ARMS[1], ROUNDOFF_ARMS[2]),
    **{name+"_vs_fp32": (name, "fp32-materialized") for name in ROUNDOFF_ARMS[:3]},
    "fp32_recompute_vs_materialized": (ROUNDOFF_ARMS[4], ROUNDOFF_ARMS[3]),
}


def validate_roundoff(report, directory=None, project_root=ROOT):
    """Validate a separate diagnostic, without clearing or recounting F4 gates."""
    require(report.get("schema") == "olmo-f4-roundoff-v1"
            and report.get("status") == "completed_diagnostic" and report.get("finished_utc"),
            "Require completed F4 roundoff diagnostic")
    require(report.get("original_screen_cleared") is False, "Diagnostic cannot clear original numerical screen")
    require(report.get("checkpoint", {}).get("sha256") == prior.CHECKPOINT_SHA256, "Wrong diagnostic checkpoint")
    case = report.get("case", {})
    expected_case = {"name":"rt-fbt", "fbt":True, "nextlat":False, "rt_layers":[0,15], "passes":2,
        "alpha":1., "beta":1., "batch_size":8, "length":512, "updates":3,
        "transition":False, "resume":False, "profile":False}
    require(case == expected_case, "Diagnostic case differs from original failed case")
    shapes = parameter_shapes(False)
    gradients = set(shapes)
    losses = {f"pass{p}/{term}" for p in range(2) for term in TERMS}

    def comparison_check(value):
        require(isinstance(value, dict) and value.get("finite") is True, "Nonfinite/missing diagnostic comparison")
        require(set(value.get("gradients", {})) == gradients and set(value.get("losses", {})) == losses,
                "Diagnostic omitted active gradient/loss comparisons")
        for kind in ("gradients", "losses"):
            for row in value[kind].values():
                require(row.get("finite") is True and type(row.get("bitwise_equal")) is bool,
                        "Missing finite diagnostic tensor comparison")
                error = number(row.get("delta_sq"), "diagnostic squared error")
                reference = number(row.get("reference_sq"), "diagnostic squared reference")
                relative = number(row.get("relative_l2"), "diagnostic relative L2")
                maximum = number(row.get("max_relative"), "diagnostic maximum error ratio")
                require(reference > 0 or error == 0, "Nonzero error against zero diagnostic reference")
                close(relative, math.sqrt(error/reference) if reference else 0., "diagnostic relative norm")
                if row["bitwise_equal"]:
                    require(error == relative == maximum == 0, "Diagnostic equality contradicts errors")
        equal = all(row["bitwise_equal"] for kind in ("gradients", "losses") for row in value[kind].values())
        require(value.get("all_bitwise_equal") is equal, "Diagnostic aggregate equality contradicts tensor records")
        error = sum(row["delta_sq"] for row in value["gradients"].values())
        reference = sum(row["reference_sq"] for row in value["gradients"].values())
        close(value.get("global_gradient_relative_l2"), math.sqrt(error/reference) if reference else 0., "diagnostic global gradient norm")
        failures = {name for name, row in value["gradients"].items() if row["max_relative"] > 1/16}
        recorded = value.get("coordinate_screen_failures")
        require(isinstance(recorded, list) and len(recorded) == len(set(recorded)) and set(recorded) == failures,
                "Diagnostic coordinate failures contradict tensor metrics")
        require(set(value.get("worst_coordinates", {})) == failures, "Missing failed-coordinate observations")
        for name, coordinate in value["worst_coordinates"].items():
            require(coordinate.get("shape") == shapes[name], "Failed-coordinate shape differs from native parameter")
            index = integer(coordinate.get("flat_index"), "failed coordinate index")
            require(index < math.prod(shapes[name]), "Failed coordinate exceeds native tensor")
            candidate = number(coordinate.get("candidate"), "failed candidate coordinate", signed=True)
            ref = number(coordinate.get("reference"), "failed reference coordinate", signed=True)
            peak = number(coordinate.get("reference_tensor_max"), "failed reference peak", positive=True)
            require(abs(ref) <= peak, "Reference coordinate exceeds tensor maximum")
            close(value["gradients"][name]["max_relative"], abs(candidate-ref)/peak, "failed-coordinate error ratio")
        return value

    arms = report.get("arms", {})
    require(set(arms) == set(ROUNDOFF_ARMS), "Missing precision/memory diagnostic arm")
    for name, arm in arms.items():
        require(type(arm.get("gradient_tensor_count")) is int and arm["gradient_tensor_count"] == len(gradients)
                and set(arm.get("losses", {})) == losses, "Diagnostic arm inventory differs")
        for key, value in arm["losses"].items():
            number(value, "diagnostic loss", signed=True)
            if key.endswith(("/latent", "/kl")):
                require(value == 0, "NextLat is disabled in diagnostic")
        if name.startswith("bf16"):
            repeated = comparison_check(arm.get("repeat"))
            require(repeated["all_bitwise_equal"], "BF16 diagnostic eager repeats are not exact")
    comparisons = report.get("comparisons", {})
    require(set(comparisons) == set(ROUNDOFF_COMPARISONS), "Missing diagnostic comparison direction")
    for name, value in comparisons.items():
        comparison_check(value)
        candidate, reference = (arms[key]["losses"] for key in ROUNDOFF_COMPARISONS[name])
        for key, row in value["losses"].items():
            close(row["reference_sq"], reference[key]**2, "diagnostic reference loss norm")
            close(row["delta_sq"], (candidate[key]-reference[key])**2, "diagnostic loss error norm")
    first = comparisons["bf16_recompute_vs_materialized"]
    require(first["coordinate_screen_failures"], "Original BF16 coordinate failure was not reproduced")

    checks = report.get("graph_checks")
    expected_checks = prior.NATIVE_CHECKS - {"same_state_variant_vs_reference"}
    require(isinstance(checks, list) and len(checks) == len(expected_checks)
            and {row.get("name") for row in checks} == expected_checks, "Missing diagnostic graph/update checks")
    config = {"case":"rt-fbt", "feature_flags":feature_flags("rt-fbt"), "batch_size":8, "length":512}
    for check in checks:
        require(check.get("passed") is True, "Diagnostic graph/update gate failed")
        if check["name"] == "complete_adamw_update_parity":
            require(check.get("updates_per_arm") == 3 and check.get("physical_optimizer_updates") == 6,
                    "Diagnostic update count differs")
            require(all(check.get(key) is True for key in ("metrics_exact", "model_optimizer_scheduler_counters_exact", "weights_changed")),
                    "Diagnostic update parity/changed weights missing")
            update_arms = check.get("arms")
            require(isinstance(update_arms, list) and len(update_arms) == 2
                    and [arm.get("replay") for arm in update_arms] == [False, True], "Missing diagnostic eager/graph update arms")
            for arm in update_arms:
                require(len(arm.get("metrics", [])) == 3, "Diagnostic optimizer metrics omitted")
                for step, metrics in enumerate(arm["metrics"], 1): metrics_check(metrics, config, step)
                boundary = arm.get("boundary")
                require(isinstance(boundary, dict) and set(boundary) == {"model","optimizer","scheduler","counters"}
                        and boundary["model"] and boundary["optimizer"] and boundary["scheduler"], "Missing diagnostic full-state boundary")
                require(boundary["counters"] == arm["metrics"][-1]["counters"], "Diagnostic boundary counters differ")
                _state_health(arm.get("health"))
            require(update_arms[0]["metrics"] == update_arms[1]["metrics"]
                    and update_arms[0]["boundary"] == update_arms[1]["boundary"], "Recorded diagnostic optimizer parity is not exact")
        else:
            require(check.get("ownership_matches") is True and check.get("all_bitwise_equal") is True
                    and set(check.get("gradients", {})) == gradients and set(check.get("losses", {})) == losses,
                    "Diagnostic graph omitted inventory or exactness")
            for row in (*check["gradients"].values(), *check["losses"].values()):
                tensor_check(row)
                require(row["bitwise_equal"], "Diagnostic graph tensor is not bitwise equal")
    lineage = None if directory is None else source_lineage(report, Path(directory), Path(project_root),
        sources=(*RUNTIME_SOURCES, "scripts/olmo_f4_roundoff.py"),
        protocol="docs/reports/olmo1b-f4/roundoff-protocol.md")
    return {"schema":"olmo-f4-roundoff-summary-v1", "status":"completed_diagnostic", "case":case,
        "original_screen_cleared":False, "physical_optimizer_updates":6,
        "optimizer_updates":{"eager":3,"graph":3,"total":6}, "counted_in_f4_matrix_updates":False,
        "bf16_repeats_exact":True, "candidate_graph_and_full_adam_exact":True,
        "comparisons":comparisons, "source_lineage":lineage,
        "scope":"Separate fixed-state diagnosis. FP32 changes forward states/backends; descriptive comparisons do not clear the original BF16 screen."}


def summarize(runtime_dirs, *, project_root=ROOT, allow_incomplete=False, f3e_references=()):
    project_root = Path(project_root)
    summary = {"schema":"olmo-f4-summary-v1", "generated_utc":datetime.now(timezone.utc).isoformat(),
        "status":"passed", "runs":[], "correctness":[], "full_steps":[],
        "capability_ledger":[], "resource_cards":[], "failed_diagnostics":[], "incomplete":[],
        "operator_traces":[], "reused_f3e_correctness":[], "reused_optimizer_updates_excluded":0,
        "successful_f4_optimizer_updates":{"eager":0,"graph":0,"total":0},
        "scope":"Eight independent RT/FBT/NextLat training resource cells; RT=(0,15), FBT=K2. No quality, placement-superiority, inference-throughput, hardware-FLOP, FA4 or multi-GPU claim."}
    seen = set()
    for item in runtime_dirs:
        directory = Path(item).resolve()
        require(directory.name not in seen, "Duplicate runtime name"); seen.add(directory.name)
        path = directory/"report.json"
        if not path.is_file():
            require(allow_incomplete and directory.is_dir(), "Missing runtime report")
            summary["incomplete"].append({"run":directory.name, "status":"no_report_yet"}); continue
        report = json.loads(path.read_text())
        require(report.get("schema") == SCHEMA, "Unknown F4 schema")
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
            "feature_flags":c["feature_flags"], "evidence_origin":"new_f4",
            "stage":report["stage"], "backward_memory":c["backward_memory"], "status":status}
        summary["capability_ledger"].append(capability)
        if status != "passed":
            require(report.get("error_type") and isinstance(report.get("error_message"), str), "Failed report lacks error")
            summary["failed_diagnostics"].append({**capability, "error_type":report["error_type"], "error_message":report["error_message"],
                "declared_failed_checks":[row["name"] for row in checks if not row["passed"]], "excluded_partial_performance":bool(report.get("capacity"))})
            continue
        result = validate_success(report)
        trace = trace_check(report, directory)
        if trace is not None:
            summary["operator_traces"].append({"run":directory.name, "case":c["case"], **trace})
        if report["stage"] == "correctness":
            summary["correctness"].append({"run":directory.name,"configuration":c,**result})
        else:
            summary["full_steps"].append(full_step_record(report,directory.name)); run["used_for_performance"] = True
        summary["resource_cards"].append({**capability, **report["resources"]})
        for key,count in (("eager",3),("graph",3),("total",6)):
            summary["successful_f4_optimizer_updates"][key] += count
    if f3e_references:
        from scripts import olmo_f3e_report
        historical = olmo_f3e_report.summarize(f3e_references, project_root=project_root)
        require(historical["status"] == "passed" and not historical["full_steps"], "Reused F3e scope must be passed correctness")
        for row in historical["correctness"]:
            c = row["configuration"]
            require(c["case"] in ("rt", "combined") and c["layout"] == "spread2"
                    and c["batch_size"] == 8 and c["length"] == 512 and c["variant"] == "recompute",
                    "Reused F3e correctness must be RT/combined spread2 B8/T512")
            summary["reused_f3e_correctness"].append(row)
            summary["reused_optimizer_updates_excluded"] += row["physical_optimizer_updates"]
        summary["reused_f3e_lineage"] = historical["runs"]
        summary["capability_ledger"].extend({**row, "evidence_origin":"historical_f3e"}
                                            for row in historical["capability_ledger"])
    require(summary["runs"] or summary["incomplete"], "No F4 runs supplied")
    common = [row for row in summary["full_steps"] if row["batch_size"] == 64 and row["length"] == 512]
    require(len(common) == len({row["case"] for row in common}), "Duplicate successful common-matrix cell")
    present = {row["case"] for row in common}
    summary["common_matrix_complete"] = present == set(FEATURES)
    summary["common_matrix_completion_scope"] = "Completed resource measurements only; numerical qualifications remain independent."
    summary["common_matrix_present"] = [case for case in FEATURES if case in present]
    summary["common_matrix_missing"] = [case for case in FEATURES if case not in present]
    summary["numerical_qualifications"] = []
    for failure in summary["failed_diagnostics"]:
        if failure["stage"] != "correctness" or "same_state_variant_vs_reference" not in failure["declared_failed_checks"]:
            continue
        run = next(row for row in summary["runs"] if row["name"] == failure["run"])
        first = next(row["record"] for row in run["checks"] if row["name"] == "same_state_variant_vs_reference")
        mixed = first.get("mixed_gradient_screen") or {}
        summary["numerical_qualifications"].append({"case":failure["case"], "source_run":failure["run"],
            "original_screen_cleared":False, "global_gradient_relative_l2":first.get("global_gradient_relative_l2"),
            "failed_tensor_screens":{name:row for name,row in mixed.items()
                if row.get("relative_l2", math.inf) > 1/32 or row.get("max_relative", math.inf) > 1/16},
            "scope":"Initial materialized-versus-recompute engineering screen failed. Resource measurement or separate repeat/graph/FP32 diagnostics do not clear it."})
    passed_cases = {row["configuration"]["case"] for row in [*summary["correctness"], *summary["reused_f3e_correctness"]]}
    summary["all_feature_cases_have_passing_bounded_checks"] = passed_cases == set(FEATURES) and not summary["numerical_qualifications"]
    for row in summary["full_steps"]:
        qualifications = [item for item in summary["numerical_qualifications"] if item["case"] == row["case"]]
        row["numerical_qualifications"] = qualifications
        row["numerical_status"] = ("qualified_unresolved_initial_gradient_screen" if qualifications else
            "bounded_correctness_evidence_recorded" if row["case"] in passed_cases else "capacity_health_only")
    for row in summary["resource_cards"]:
        row["numerical_qualifications"] = [item for item in summary["numerical_qualifications"] if item["case"] == row["case"]]
    if summary["common_matrix_complete"]:
        require(any(item["case"] == "rt-fbt" for item in summary["numerical_qualifications"]),
                "Complete F4 matrix requires retained RT+FBT failed-screen provenance; resource coverage cannot silently clear it")
    summary["larger_batch_eligibility"] = []
    for row in summary["full_steps"]:
        if row["batch_size"] != 96: continue
        reference = next((item for item in common if item["case"] == row["case"]), None)
        if reference is not None:
            require(reference["peak_reserved_gib"] < 65., "B96 run violates measured B64 reservation prerequisite")
        summary["larger_batch_eligibility"].append({"run":row["run"], "case":row["case"],
            "b64_reference_run":reference["run"] if reference else None,
            "b64_below_65_gib": True if reference else None,
            "tight_setup_headroom": row["peak_reserved_gib"] > 72.})
    if summary["incomplete"]: summary["status"] = "partial_preview"
    elif summary["failed_diagnostics"]: summary["status"] = "completed_with_failed_diagnostics"
    summary["declared_check_count"] = sum(len(row["checks"]) for row in summary["runs"])
    summary["declared_checks_passed"] = sum(check["passed"] for row in summary["runs"] for check in row["checks"])
    summary["successful_declared_check_count"] = sum(len(row["checks"]) for row in summary["runs"] if row["status"] == "passed")
    return summary


def markdown(summary):
    lines = ["# F4 independent RT / FBT / NextLat training resources", "", f"Status: **{summary['status']}**; complete common B64/T512 matrix: **{summary['common_matrix_complete']}**.", "",
        "Statuses describe the supplied completed runs, not automatic clearance of every planned case. "
        "These are functionality and execution measurements with independently switched RT, FBT and NextLat. "
        "RT selects layers (0,15); FBT uses K2. Native Q/K math stays unchanged. "
        "No quality, placement-superiority, inference-throughput, native FA4 or multi-GPU claim follows.", "",
        "| Run | Case / layout | RT indices | Role | B/T | Stage | Passed / declared |", "| --- | --- | --- | --- | ---: | --- | ---: |"]
    for run in summary["runs"]:
        c=run["configuration"]; label=f"[{run['name']}]({run['wandb_url']})" if run.get("wandb_url") else run["name"]
        lines.append(f"| {label} | {c['case']} / {c['layout']} | {c['selected_rt_layers']} | {c['layout_role']} | {c['batch_size']}/{c['length']} | "
                     f"{c['stage']} ({run['status']}) | {sum(v['passed'] for v in run['checks'])}/{len(run['checks'])} |")
    updates = summary["successful_f4_optimizer_updates"]
    lines += ["", f"Successful runs contain {updates['total']} physical optimizer updates ({updates['eager']} eager + {updates['graph']} graph). "
        f"Warmup/backward-only work, failed attempts and {summary['reused_optimizer_updates_excluded']} historical F3e updates are excluded.",
        "", "Common cells still missing: " + (", ".join(summary["common_matrix_missing"]) or "none") + ".",
        "", "## Correctness", "",
        "Initial materialized-versus-recompute gradients retain global relative L2 ≤1/64, per tensor ≤1/32 and maximum error/reference maximum ≤1/16. "
        "Zero references require exact zero. Same-candidate graph and full Adam/state comparisons require exact parity. "
        "Stricter diagnostic flags remain in summary.json.", "",
        "| Run | Initial loss exact | Global gradient relative L2 | Recompute tiles | Graph exact | Full Adam exact |",
        "| --- | --- | ---: | ---: | --- | --- |"]
    for row in summary["correctness"]:
        lines.append(f"| {row['run']} | {row['forward_losses_bitwise_equal']} | {row['global_gradient_relative_l2']:.6g} | "
                     f"{row['observed_recompute_backward_calls']} | {row['candidate_graph_checks_bitwise']} | {row['complete_updates_exact']} |")
    for row in summary["reused_f3e_correctness"]:
        lines.append(f"| {row['run']} (historical F3e, reused) | {row['forward_losses_bitwise_equal']} | {row['global_gradient_relative_l2']:.6g} | "
                     f"{row['observed_recompute_backward_calls']} | {row['candidate_graph_checks_bitwise']} | {row['complete_updates_exact']} |")
    lines += ["", "## Complete-update resources", "",
        "Wall time includes input copy/validation, graph forward/loss/backward, clipping, AdamW and scheduler. "
        "Three-update medians are directional. Setup and timed steady peaks are measured separately; combined allocated/reserved peaks are their maxima. "
        "Matrix FLOPs are an analytic ledger excluding elementwise/optimizer/communication work and kernel padding, not hardware utilization.", "",
        "| Run | Case / layout | B/T | Input tokens/s | CE targets/s | Seconds/update | Allocated / reserved peak / current GiB | Matrix TFLOPs/update | Numerical scope |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary["full_steps"]:
        estimate=row["resources"]["analytic_matrix_work"]
        lines.append(f"| {row['run']} | {row['case']} / {row['layout']} | {row['batch_size']}/{row['length']} | {row['input_tokens_per_second']:,.0f} | "
                     f"{row['ce_targets_per_second']:,.0f} | {row['full_step']['median_wall_seconds']:.4f} | "
                     f"{row['peak_allocated_gib']:.3f} / {row['peak_reserved_gib']:.3f} / {row['current_reserved_gib']:.3f} | "
                     f"{estimate['matrix_flops_minimum']/1e12:.2f}–{estimate['matrix_flops_maximum']/1e12:.2f} | {row['numerical_status']} |")
    for qualification in summary["numerical_qualifications"]:
        lines += ["", f"**{qualification['case']} remains numerically qualified.** Its original screen in `{qualification['source_run']}` is not cleared. "
            "Successful resource measurements or separate repeat/graph/FP32 diagnostics do not change that outcome. "
            "Common-matrix completion means resource coverage, not numerical clearance."]
    lines += ["", "Parameter ownership and actual objective counts are recorded per run in resource-ledger.json. "
        "RT adds no parameters; multiple FBT passes share weights. The NextLat predictor is training-only. "
        "Frozen fusion weights remain registered when FBT is disabled. Capacity health at a larger batch is not a full gradient-equivalence check at that batch.",
        "", "| Case | Registered | Trainable / gradient / optimizer | Deployable | Setup allocated / reserved peak GiB | Steady allocated / reserved peak GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["full_steps"]:
        p = row["resources"]["observed_parameters"]
        setup, steady = row["setup_memory"], row["steady_memory"]
        lines.append(f"| {row['case']} B{row['batch_size']} | {p['registered_unique']:,} | {p['trainable']:,} / {p['gradient_participating']:,} / {p['optimizer_owned']:,} | "
                     f"{p['deployable_inference_declared']:,} | {setup['peak_allocated_gib']:.3f} / {setup['peak_reserved_gib']:.3f} | "
                     f"{steady['peak_allocated_gib']:.3f} / {steady['peak_reserved_gib']:.3f} |")
    if summary["operator_traces"]:
        lines += ["", "## Operator coverage audit", "",
            "Untimed eager B1/T32 traces include shapes and selected PyTorch operation FLOPs. Fused Flash/Triton and custom/recomputed work can be undercounted. "
            "These figures are neither measured hardware FLOPs nor complete-update timing; the analytic resource ledger remains the comparison."]
        for trace in summary["operator_traces"]:
            lines.append(f"- `{trace['run']}`: {trace['observed_device_event_count']:,} device events; "
                         f"{trace['pytorch_estimated_flops']:,} selected-operation FLOPs; retained `{trace['file']}`.")
    for row in summary["larger_batch_eligibility"]:
        if row["tight_setup_headroom"]:
            lines += ["", f"`{row['run']}` exceeds 72 GiB peak reservation: successful but tight setup headroom."]
    for failure in summary["failed_diagnostics"]:
        lines += ["", f"Failed diagnostic `{failure['run']}` ({failure['layout_role']}): {failure['error_type']}: {failure['error_message']}. "
            "Its partial timing is excluded; its original source/protocol/error record is retained."]
    return "\n".join(lines)+"\n"


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=summary["full_steps"]
    if not rows: return []
    cases = [case for case in FEATURES if any(row["case"] == case for row in rows)]
    batches = sorted({row["batch_size"] for row in rows})
    labels = {"ordinary":"Ordinary", "rt":"RT", "fbt":"FBT", "nextlat":"NextLat",
        "rt-fbt":"RT +\nFBT *", "rt-nextlat":"RT +\nNextLat",
        "fbt-nextlat":"FBT +\nNextLat", "combined":"RT + FBT\n+ NextLat"}
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    figure.subplots_adjust(left=.06, right=.985, bottom=.23, top=.80, wspace=.20)
    width = .36 if len(batches) == 2 else .52
    colors = {64:"#356DA9", 96:"#E69138"}
    for axis, field, title, ylabel in (
        (axes[0], "input_tokens_per_second", "Full-update throughput", "Input tokens/s"),
        (axes[1], "peak_allocated_gib", "Peak allocated memory, including setup", "GiB"),
    ):
        for index, batch in enumerate(batches):
            selected = [row for row in rows if row["batch_size"] == batch]
            offset = (index - (len(batches)-1)/2) * .40
            # Plot each recorded median directly; do not average or rewrite rows.
            positions = [cases.index(row["case"]) + offset for row in selected]
            axis.bar(positions, [row[field] for row in selected], width=width,
                     color=colors.get(batch), label=f"B{batch}", zorder=3)
        axis.set_title(title, fontsize=12, pad=12)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(cases)), [labels[case] for case in cases], fontsize=9)
        axis.set_xlim(-.65, len(cases)-.35)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#DDDDDD", linewidth=.7, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}k" if value else "0"))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, title="Physical batch per GPU", loc="upper center",
                  bbox_to_anchor=(.5, .98), ncol=len(batches), frameon=False)
    caption = "T512; three-update medians. RT uses layers (0,15); FBT uses K2."
    if "rt-fbt" in cases:
        caption += "\n* RT + FBT retains its failed initial gradient screen; resource results do not clear it."
    missing96 = [case for case in cases if not any(row["case"] == case and row["batch_size"] == 96 for row in rows)]
    if missing96:
        caption += "\nB96 not measured: " + ", ".join(missing96) + ". Missing bars are not zero measurements."
    figure.text(.06, .035, caption, fontsize=9, ha="left", va="bottom", color="#333333")
    paths=[]
    for suffix in ("pdf","png"):
        path=output/f"feature-training-resources.{suffix}"; figure.savefig(path,dpi=180); paths.append(str(path))
    plt.close(figure)
    return paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir",type=Path,action="append",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--f3e-reference-dir",type=Path,action="append",default=[])
    args=parser.parse_args(argv); summary=summarize(args.runtime_dir,allow_incomplete=args.allow_incomplete,
                                                   f3e_references=args.f3e_reference_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/"summary.json",summary)
    write_json(args.output_dir/"capability-ledger.json",summary["capability_ledger"])
    write_json(args.output_dir/"resource-ledger.json",summary["resource_cards"])
    (args.output_dir/"results.md").write_text(markdown(summary)); plot(summary,args.output_dir)
    print(json.dumps({"status":summary["status"],"checks":summary["declared_check_count"],
                      "common_matrix_complete":summary["common_matrix_complete"]}))


if __name__ == "__main__": main()
