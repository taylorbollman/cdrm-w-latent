"""F4 evidence checks independent feature semantics and complete resource accounting."""
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts import olmo_f4_report as reporter

TRACE_BYTES = gzip.compress(b'{"traceEvents": []}', mtime=0)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


def tensor_check():
    return {"passed": True, "finite": True, "bitwise_equal": True, "relative_l2": 0.,
            "max_abs": 0., "reference_max_abs": 1., "relative_l2_limit": 1e-5, "max_abs_limit": 1.1e-5}


def metric_fixture(config, counts, step):
    auxiliary = config["feature_flags"]["nextlat"]
    weights = {"ce": 1., "latent": float(auxiliary), "kl": float(auxiliary)}
    means = {key: 1. if count else 0. for key, count in counts.items()}
    return {"schema": "olmo-lm-optimizer-step-v1", "update_completed": True, "counts": copy.deepcopy(counts),
        "loss_means": means, "loss_sums": {key: count*means[key] for key, count in counts.items()},
        "objective_weights": weights, "objective": sum(means[key]*weights[key] for key in counts),
        "gradient_norm_before_clip": 1., "max_grad_norm": 1., "lr_used": [1e-5], "lr_next": [1e-5],
        "counters": {"optimizer_updates": step, "microbatches": step, "documents": step*config["batch_size"],
            "input_tokens": step*config["batch_size"]*config["length"], "ce_positions": step*counts["ce"],
            "latent_pairs": step*counts["latent"], "kl_triples": step*counts["kl"]}}


def make_report(*, stage="capacity", case="combined", length=512, batch=64, trace=False):
    flags = reporter.feature_flags(case)
    layers = [0, 15] if flags["rt"] else []
    passes = 2 if flags["fbt"] else 1
    mode = reporter.FBTMode(enabled=flags["fbt"], num_passes=passes,
                           rt_mode=reporter.RTMode(tuple(layers)))
    config = {"case": case, "layout": "spread2" if flags["rt"] else "none", "stage": stage,
        "variant": "recompute", "batch_size": batch, "length": length, "selected_rt_layers": layers,
        "layout_role": "common_feature_comparison", "feature_flags": flags, "operator_trace": trace,
        "world_size": 1, "accumulation_steps": 1, "physical_batch_per_gpu": batch, "logical_batch": batch,
        "mode": json.loads(json.dumps(asdict(mode))), "rt_block_calls_per_forward_backward": len(layers),
        "cast_weights_once": True, "forward_tile_backend": "triton", "backward_tile_backend": "triton",
        "backward_memory": "recompute", "ordinary_activation_checkpointing": True,
        "ordinary_attention_backend": "deterministic_flash", "precision": "bf16_mixed", "tf32": False,
        "autocast_weight_cache": False,
        "case_specification": {"name": case, "fbt": flags["fbt"], "nextlat": flags["nextlat"], "rt_layers": layers,
            "passes": 2, "alpha": 1., "beta": 1., "batch_size": batch, "length": length, "updates": 3,
            "transition": False, "resume": False, "profile": False}}
    auxiliary = flags["nextlat"]
    work = reporter.LossWork(batch*(length//2), batch*(length-1) if auxiliary else 0,
        batch*(length//2) if auxiliary else 0, batch*(length-1) if auxiliary else 0)
    native = reporter.OLMoConfig.native_1b()
    nextlat = reporter.NextLatConfig(native.model_dim) if auxiliary else None
    estimate = reporter.estimate_training_resources(native, batch_size=batch, sequence_length=length, mode=mode,
        nextlat=nextlat, loss_work=work, ordinary_checkpointing=True, backward_memory="recompute").to_dict()
    active = estimate["parameter_counts"]["training_architecture"]
    resident = reporter.architecture_parameter_counts(native, fbt=True, nextlat=nextlat)["training_architecture"]
    shapes = reporter.parameter_shapes(auxiliary)
    executed = sorted(n for n in shapes if flags["fbt"] or not n.startswith("backbone.fusion."))
    resource = {"analytic_matrix_work": estimate, "loss_work": asdict(work),
        "loss_weights": {"ce": 1., "latent": float(auxiliary), "kl": float(auxiliary)},
        "observed_parameters": {"registered_unique": resident, "resident_parameter_bytes": resident*4,
            "trainable": active, "gradient_participating": active, "optimizer_owned": active if stage == "capacity" else None,
            "executed_declared": active, "deployable_inference_declared": estimate["parameter_counts"]["deployable_inference"]},
        "named_parameter_shapes": shapes, "declared_execution_names": executed,
        "declared_inference_names": [n for n in executed if not n.startswith("predictor.")], "scope": "matrix estimate"}
    result = {"schema": reporter.SCHEMA, "status": "passed", "finished_utc": "now", "stage": stage,
        "configuration": config, "checkpoint": {"sha256": reporter.prior.CHECKPOINT_SHA256}, "resources": resource,
        "wandb": {"status": "synced", "run_url": "https://wandb.ai/taylorbollman/test/runs/123"},
        "backward_preparation": {"warmup": 11, "capture": 1, "replay": 7 if stage == "correctness" else 3}}
    counts = {key: estimate["objective_positions_per_update"][key] for key in reporter.TERMS}
    if trace:
        result["operator_trace"] = {"file": "operator-trace.json.gz", "sha256": hashlib.sha256(TRACE_BYTES).hexdigest(),
            "bytes": len(TRACE_BYTES), "operator_rows": [{"name": "aten::mm", "calls": 2, "estimated_flops": 42,
                "self_cpu_time_us": 10., "self_device_time_us": 5.}], "observed_device_event_count": 2,
            "observed_device_event_names": ["flash_forward", "mm_kernel"], "pytorch_estimated_flops": 42,
            "scope": "Untimed selected-operation count, not complete-update hardware FLOPs."}
    if stage == "capacity":
        result["checks"] = [{"name": "finite_complete_updates", "passed": True}]
        result["capacity"] = {"full_step": {"wall_seconds": [2.]*3, "cuda_seconds": [1.8]*3,
                "median_wall_seconds": 2., "median_cuda_seconds": 1.8},
            "input_tokens_per_second": batch*length/2, "ce_targets_per_second": work.ce_targets/2,
            "estimated_matrix_tflops_per_second_minimum": estimate["matrix_flops_minimum"]/2e12,
            "estimated_matrix_tflops_per_second_maximum": estimate["matrix_flops_maximum"]/2e12,
            "peak_allocated_gib": 40., "peak_reserved_gib": 64., "current_reserved_gib": 43.,
            "setup_memory": {"allocated_gib": 35., "reserved_gib": 43., "peak_allocated_gib": 40., "peak_reserved_gib": 64.},
            "steady_memory": {"allocated_gib": 35., "reserved_gib": 43., "peak_allocated_gib": 38., "peak_reserved_gib": 43.},
            "health": {"passed": True, "nonfinite_parameters": [], "nonfinite_optimizer_tensors": []},
            "warmup_updates": 3, "timed_updates": 3, "physical_optimizer_updates": 6, "backward_only_warmup": 10,
            "records": [metric_fixture(config, counts, i) for i in (4, 5, 6)]}
        return result
    losses = {f"pass{p}/{term}": tensor_check() for p in range(passes) for term in reporter.TERMS}
    row = {"passed": True, "all_bitwise_equal": True, "ownership_matches": True,
           "gradients": {n: tensor_check() for n in executed}, "losses": losses}
    result["checks"] = [{**copy.deepcopy(row), "name": name}
                       for name in sorted(reporter.prior.NATIVE_CHECKS - {"complete_adamw_update_parity"})]
    first = next(v for v in result["checks"] if v["name"] == "same_state_variant_vs_reference")
    for arm in ("reference", "candidate"):
        by_layer = {str(i): {"forward_blocks": 1, "forward_tiles": length-1, "forward_fused_tiles": length-1,
            "forward_eager_tiles": 0, "backward_blocks": 1, "recompute_tiles": length-1 if arm == "candidate" else 0,
            "materialized_tiles": length-1 if arm == "reference" else 0,
            "materialized_fused_tiles": length-1 if arm == "reference" else 0} for i in layers}
        dispatch = {"by_layer": by_layer, "totals": {key: sum(v[key] for v in by_layer.values()) for key in reporter.EVENTS}}
        first[arm+"_dispatch"] = copy.deepcopy(dispatch)
        first["expected_"+arm+"_dispatch"] = copy.deepcopy(dispatch)
    first.update(forward_losses_bitwise_equal=True, dispatch_matches=True, reference_recompute_backward_calls=0,
        candidate_recompute_backward_calls=(length-1)*len(layers),
        expected_candidate_recompute_backward_calls=(length-1)*len(layers), global_gradient_relative_l2=0.,
        mixed_gradient_screen={n: {"delta_sq": 0., "reference_sq": 3., "relative_l2": 0., "max_relative": 0.} for n in executed},
        mixed_loss_screen={n: {"relative_l2": 0., "max_relative": 0.} for n in losses})
    result["checks"].append({"name": "complete_adamw_update_parity", "passed": True, "metrics_exact": True,
        "model_optimizer_scheduler_counters_exact": True, "weights_changed": True, "updates_per_arm": 3,
        "physical_optimizer_updates": 6, "arms": [{"replay": replay,
            "metrics": [metric_fixture(config, counts, i) for i in (1, 2, 3)],
            "boundary": {"model": {"tensor": "digest"}, "optimizer": {"moment": "digest"}, "scheduler": {"step": 3},
                "counters": metric_fixture(config, counts, 3)["counters"]},
            "health": {"passed": True, "nonfinite_parameters": [], "nonfinite_optimizer_tensors": []}}
            for replay in (False, True)]})
    return result


def make_failed_screen():
    report = make_report(stage="correctness", case="rt-fbt", batch=8, length=512)
    first = next(row for row in report["checks"] if row["name"] == "same_state_variant_vs_reference")
    first["passed"] = False
    first["mixed_gradient_screen"]["backbone.backbone.transformer.blocks.11.ff_proj.weight"]["max_relative"] = .064
    report.update(status="failed", error_type="AssertionError", error_message="same_state_variant_vs_reference", checks=[first])
    return report


def make_roundoff_report():
    report = make_report(stage="correctness", case="rt-fbt", batch=8, length=512)
    graph_checks = [row for row in report["checks"] if row["name"] != "same_state_variant_vs_reference"]
    shapes = reporter.parameter_shapes(False)
    losses = {f"pass{p}/{term}": 1. if term == "ce" else 0. for p in range(2) for term in reporter.TERMS}

    def comparison(failure=False):
        value = {"losses": {name: {"delta_sq":0., "reference_sq":amount**2, "relative_l2":0.,
                "max_relative":0., "bitwise_equal":True, "finite":True} for name, amount in losses.items()},
            "gradients": {name: {"delta_sq":0., "reference_sq":100., "relative_l2":0., "max_relative":0.,
                "bitwise_equal":True, "finite":True} for name in shapes},
            "all_bitwise_equal":not failure, "finite":True, "global_gradient_relative_l2":0.,
            "coordinate_screen_failures":[], "worst_coordinates":{}}
        if failure:
            name = "backbone.backbone.transformer.blocks.11.ff_proj.weight"
            value["gradients"][name].update(delta_sq=.064**2, relative_l2=.0064, max_relative=.064, bitwise_equal=False)
            value["global_gradient_relative_l2"] = .064 / (len(shapes)*100.)**.5
            value["coordinate_screen_failures"] = [name]
            value["worst_coordinates"] = {name:{"flat_index":0, "shape":shapes[name], "candidate":1.064,
                "reference":1., "reference_tensor_max":1.}}
        return value
    return {"schema":"olmo-f4-roundoff-v1", "status":"completed_diagnostic", "finished_utc":"now",
        "case":report["configuration"]["case_specification"], "checkpoint":report["checkpoint"],
        "original_screen_cleared":False, "graph_checks":graph_checks,
        "arms":{name:{"losses":copy.deepcopy(losses), "gradient_tensor_count":len(shapes),
            **({"repeat":comparison()} if name.startswith("bf16") else {})} for name in reporter.ROUNDOFF_ARMS},
        "comparisons":{name:comparison(failure=name == "bf16_recompute_vs_materialized")
            for name in reporter.ROUNDOFF_COMPARISONS}}


@pytest.fixture
def evidence(tmp_path):
    project, directory = tmp_path/"project", tmp_path/"runtime"/"native"
    report = make_report()
    write(project/reporter.PROTOCOL, "protocol")
    for name in reporter.RUNTIME_SOURCES:
        write(project/name, "source " + name)
    report.update(source_hashes={name: reporter.digest(project/name) for name in reporter.RUNTIME_SOURCES},
                  protocol_sha256=reporter.digest(project/reporter.PROTOCOL))
    write(directory/"report.json", report)
    return project, directory, report


def install(evidence, report, directory=None):
    _, default, original = evidence
    directory = default if directory is None else directory
    report.update({key: original[key] for key in ("source_hashes", "protocol_sha256")})
    write(directory/"report.json", report)
    if report.get("operator_trace"):
        write(directory/"operator-trace.json.gz", TRACE_BYTES)
    return directory


@pytest.mark.parametrize("case", reporter.FEATURES)
@pytest.mark.parametrize("stage", ["correctness", "capacity"])
def test_eight_switches_have_independent_loss_and_parameter_semantics(evidence, case, stage):
    report = make_report(case=case, stage=stage)
    install(evidence, report)
    summary = reporter.summarize([evidence[1]], project_root=evidence[0])
    card = summary["resource_cards"][0]
    flags = reporter.feature_flags(case)
    assert bool(card["loss_work"]["latent_pairs"]) == flags["nextlat"]
    assert card["analytic_matrix_work"]["rt_block_calls_per_microbatch"] == (2 if flags["rt"] else 0)
    inventory = card["observed_parameters"]
    assert inventory["registered_unique"] - inventory["trainable"] == (0 if flags["fbt"] else 8388608)
    assert summary["successful_f4_optimizer_updates"] == {"eager": 3, "graph": 3, "total": 6}
    if stage == "correctness":
        assert summary["correctness"][0]["observed_recompute_backward_calls"] == ((512-1)*2 if flags["rt"] else 0)
    assert not summary["common_matrix_complete"]


def test_complete_common_matrix_and_optional_large_cells_are_explicit(evidence, tmp_path):
    directories = [install(evidence, make_report(case=case), directory=tmp_path/"runs"/case) for case in reporter.FEATURES]
    directories.append(install(evidence, make_report(case="ordinary", batch=96), directory=tmp_path/"runs"/"ordinary-b96"))
    directories.append(install(evidence, make_failed_screen(), directory=tmp_path/"runs"/"rt-fbt-failed-screen"))
    summary = reporter.summarize(directories, project_root=evidence[0])
    assert summary["common_matrix_complete"] and summary["common_matrix_missing"] == []
    assert summary["larger_batch_eligibility"][0]["b64_below_65_gib"] is True
    assert summary["successful_f4_optimizer_updates"]["total"] == 54
    assert summary["status"] == "completed_with_failed_diagnostics"
    assert not summary["all_feature_cases_have_passing_bounded_checks"]
    qualified = next(row for row in summary["full_steps"] if row["case"] == "rt-fbt")
    assert qualified["numerical_status"] == "qualified_unresolved_initial_gradient_screen"
    assert qualified["numerical_qualifications"][0]["original_screen_cleared"] is False
    assert "independently switched" in reporter.markdown(summary)
    assert all(Path(p).stat().st_size > 1000 for p in reporter.plot(summary, tmp_path/"plots"))


def test_complete_matrix_cannot_drop_original_failed_screen(evidence, tmp_path):
    directories = [install(evidence, make_report(case=case), directory=tmp_path/"runs"/case) for case in reporter.FEATURES]
    with pytest.raises(ValueError, match="failed-screen provenance"):
        reporter.summarize(directories, project_root=evidence[0])


def test_duplicate_successful_common_cell_is_ambiguous(evidence, tmp_path):
    duplicate = install(evidence, make_report(), directory=tmp_path/"second-combined")
    with pytest.raises(ValueError, match="Duplicate successful common"):
        reporter.summarize([evidence[1], duplicate], project_root=evidence[0])


def test_larger_batch_cannot_ignore_measured_reservation_prerequisite(evidence, tmp_path):
    report = make_report()
    report["capacity"]["peak_reserved_gib"] = report["capacity"]["setup_memory"]["peak_reserved_gib"] = 65.
    install(evidence, report)
    larger = install(evidence, make_report(batch=96), directory=tmp_path/"combined-b96")
    with pytest.raises(ValueError, match="reservation prerequisite"):
        reporter.summarize([evidence[1], larger], project_root=evidence[0])


def test_historical_correctness_is_revalidated_but_never_counted_as_new_updates(evidence, tmp_path):
    from test_olmo_f3e_report import make_report as make_f3e_report
    from scripts import olmo_f3e_report
    project = evidence[0]
    historical = make_f3e_report(stage="correctness", case="rt", layout="spread2", batch=8, length=512)
    protocol = write(project/olmo_f3e_report.PROTOCOL, "F3e original protocol")
    historical["source_hashes"] = {name: reporter.digest(project/name) for name in olmo_f3e_report.RUNTIME_SOURCES}
    historical["protocol_sha256"] = reporter.digest(protocol)
    directory = tmp_path/"historical-f3e"
    write(directory/"report.json", historical)
    summary = reporter.summarize([evidence[1]], project_root=project, f3e_references=[directory])
    assert len(summary["reused_f3e_correctness"]) == 1
    assert summary["reused_optimizer_updates_excluded"] == 6
    assert summary["successful_f4_optimizer_updates"]["total"] == 6
    assert summary["declared_check_count"] == 1
    assert summary["capability_ledger"][-1]["evidence_origin"] == "historical_f3e"


def test_partial_and_failed_reports_do_not_fabricate_common_matrix_completion(evidence):
    project, directory, report = evidence
    report.update(status="running")
    write(directory/"report.json", report)
    with pytest.raises(ValueError): reporter.summarize([directory], project_root=project)
    summary = reporter.summarize([directory], project_root=project, allow_incomplete=True)
    assert summary["status"] == "partial_preview" and not summary["common_matrix_complete"]
    assert summary["successful_f4_optimizer_updates"]["total"] == 0
    report.update(status="failed", error_type="SyntheticFailure", error_message="test only")
    write(directory/"report.json", report)
    summary = reporter.summarize([directory], project_root=project)
    assert not summary["full_steps"] and not summary["common_matrix_complete"]
    assert summary["failed_diagnostics"][0]["excluded_partial_performance"]


@pytest.mark.parametrize("damage", ["auxiliary_flag", "fbt_flag", "rt_flag", "layer", "batch", "world", "mode",
    "case", "backend", "counter", "loss_weight", "inactive_loss", "flops", "parameter", "shape", "union",
    "setup_peak", "steady_peak", "timing", "rate", "missing_source", "protocol", "checkpoint"])
def test_tampering_cannot_promote_false_resource_evidence(evidence, damage):
    project, directory, original = evidence
    report = make_report(case="fbt")
    install(evidence, report)
    c, d, card = report["configuration"], report["capacity"], report["resources"]
    if damage == "auxiliary_flag": c["feature_flags"]["nextlat"] = True
    if damage == "fbt_flag": c["feature_flags"]["fbt"] = False
    if damage == "rt_flag": c["feature_flags"]["rt"] = True
    if damage == "layer": c["selected_rt_layers"] = [0, 15]
    if damage == "batch": c["physical_batch_per_gpu"] = 32
    if damage == "world": c["world_size"] = 2
    if damage == "mode": c["mode"]["num_passes"] = 1
    if damage == "case": c["case_specification"]["nextlat"] = True
    if damage == "backend": c["ordinary_attention_backend"] = "math"
    if damage == "counter": d["records"][0]["counters"]["input_tokens"] *= 2
    if damage == "loss_weight": d["records"][0]["objective_weights"]["latent"] = 1.
    if damage == "inactive_loss": d["records"][0]["loss_sums"]["latent"] = 1.
    if damage == "flops": card["analytic_matrix_work"]["matrix_flops_minimum"] += 1
    if damage == "parameter": card["observed_parameters"]["optimizer_owned"] += 1
    if damage == "shape": card["named_parameter_shapes"].pop(next(iter(card["named_parameter_shapes"])))
    if damage == "union": card["loss_work"]["predictor_positions"] = 1
    if damage == "setup_peak": d["setup_memory"]["peak_reserved_gib"] = 66.
    if damage == "steady_peak": d["steady_memory"]["peak_allocated_gib"] = 99.
    if damage == "timing": d["full_step"]["median_wall_seconds"] = 3.
    if damage == "rate": d["input_tokens_per_second"] *= 2
    if damage == "missing_source": report["source_hashes"].pop(next(iter(report["source_hashes"])))
    if damage == "protocol": report["protocol_sha256"] = "a"*64
    if damage == "checkpoint": report["checkpoint"]["sha256"] = "a"*64
    write(directory/"report.json", report)
    with pytest.raises(ValueError): reporter.summarize([directory], project_root=project)


@pytest.mark.parametrize("damage", ["zero_rt_dispatch", "missing_gradient", "missing_loss", "missing_mixed_loss",
    "nonexact_graph", "zero_reference", "global_sum", "optimizer_boundary", "update_count"])
def test_correctness_zero_rt_and_full_tensor_state_inventories(evidence, damage):
    report = make_report(case="ordinary", stage="correctness", batch=1, length=32)
    first = next(c for c in report["checks"] if c["name"] == "same_state_variant_vs_reference")
    graph = next(c for c in report["checks"] if c["name"] == "candidate_initial_graph")
    update = next(c for c in report["checks"] if c["name"] == "complete_adamw_update_parity")
    name = next(iter(first["gradients"]))
    if damage == "zero_rt_dispatch": first["candidate_dispatch"]["totals"]["recompute_tiles"] = 31
    if damage == "missing_gradient": graph["gradients"].pop(name)
    if damage == "missing_loss": graph["losses"].pop("pass0/kl")
    if damage == "missing_mixed_loss": first["mixed_loss_screen"].pop("pass0/kl")
    if damage == "nonexact_graph": graph["gradients"][name]["bitwise_equal"] = False; graph["all_bitwise_equal"] = False
    if damage == "zero_reference": first["mixed_gradient_screen"][name].update(reference_sq=0., delta_sq=1e-12)
    if damage == "global_sum": first["global_gradient_relative_l2"] = .001
    if damage == "optimizer_boundary": update["arms"][1]["boundary"]["optimizer"]["moment"] = "different"
    if damage == "update_count": update["physical_optimizer_updates"] = 5
    install(evidence, report)
    with pytest.raises(ValueError): reporter.summarize([evidence[1]], project_root=evidence[0])


@pytest.mark.parametrize("damage", [None, "bytes", "sha", "missing", "flops", "device", "request"])
def test_operator_trace_artifact_and_incomplete_flop_scope(evidence, damage):
    report = make_report(case="ordinary", stage="correctness", batch=1, length=32, trace=True)
    directory = install(evidence, report)
    if damage == "bytes": report["operator_trace"]["bytes"] += 1
    if damage == "sha": report["operator_trace"]["sha256"] = "a"*64
    if damage == "missing": (directory/"operator-trace.json.gz").unlink()
    if damage == "flops": report["operator_trace"]["pytorch_estimated_flops"] += 1
    if damage == "device": report["operator_trace"]["observed_device_event_count"] = 0
    if damage == "request": report["configuration"]["operator_trace"] = False
    write(directory/"report.json", report)
    if damage:
        with pytest.raises(ValueError): reporter.summarize([directory], project_root=evidence[0])
    else:
        summary = reporter.summarize([directory], project_root=evidence[0])
        assert summary["operator_traces"][0]["pytorch_estimated_flops"] == 42
        assert "neither measured hardware FLOPs" in reporter.markdown(summary)


@pytest.mark.parametrize("damage", [None, "cleared", "case", "checkpoint", "arm", "repeat", "comparison",
    "gradient", "loss", "global_norm", "coordinate", "no_failure", "nonexact_graph", "optimizer", "counts"])
def test_roundoff_diagnostic_validates_but_never_clears_original_screen(damage):
    report = make_roundoff_report()
    first = report["comparisons"]["bf16_recompute_vs_materialized"]
    weight = first["coordinate_screen_failures"][0]
    if damage == "cleared": report["original_screen_cleared"] = True
    if damage == "case": report["case"]["nextlat"] = True
    if damage == "checkpoint": report["checkpoint"]["sha256"] = "a"*64
    if damage == "arm": report["arms"].pop("fp32-recompute")
    if damage == "repeat": report["arms"]["bf16_mixed-recompute"]["repeat"]["all_bitwise_equal"] = False
    if damage == "comparison": report["comparisons"].pop("fp32_recompute_vs_materialized")
    if damage == "gradient": first["gradients"].pop(weight)
    if damage == "loss": first["losses"].pop("pass0/kl")
    if damage == "global_norm": first["global_gradient_relative_l2"] = .1
    if damage == "coordinate": first["worst_coordinates"][weight]["candidate"] = 2.
    if damage == "no_failure": report["comparisons"]["bf16_recompute_vs_materialized"] = copy.deepcopy(report["comparisons"]["fp32_recompute_vs_materialized"])
    if damage == "nonexact_graph":
        graph = next(row for row in report["graph_checks"] if row["name"] == "candidate_initial_graph")
        graph["gradients"][weight]["bitwise_equal"] = False
    if damage in ("optimizer", "counts"):
        update = next(row for row in report["graph_checks"] if row["name"] == "complete_adamw_update_parity")
        if damage == "optimizer": update["arms"][1]["boundary"]["optimizer"] = {"changed":"digest"}
        if damage == "counts": update["arms"][0]["metrics"][0]["counts"]["latent"] = 4088
    if damage:
        with pytest.raises(ValueError): reporter.validate_roundoff(report)
    else:
        result = reporter.validate_roundoff(report)
        assert result["status"] == "completed_diagnostic"
        assert result["physical_optimizer_updates"] == 6
        assert not result["original_screen_cleared"] and not result["counted_in_f4_matrix_updates"]
        assert result["candidate_graph_and_full_adam_exact"] and result["bf16_repeats_exact"]


def test_roundoff_lineage_requires_exact_diagnostic_sources_and_protocol(evidence, tmp_path):
    project = evidence[0]
    source = write(project/"scripts/olmo_f4_roundoff.py", "diagnostic source")
    protocol = write(project/"docs/reports/olmo1b-f4/roundoff-protocol.md", "separate bounded protocol")
    report = make_roundoff_report()
    report["source_hashes"] = {**evidence[2]["source_hashes"], "scripts/olmo_f4_roundoff.py": reporter.digest(source)}
    report["protocol_sha256"] = reporter.digest(protocol)
    directory = tmp_path/"roundoff"
    write(directory/"report.json", report)
    result = reporter.validate_roundoff(report, directory, project)
    assert result["source_lineage"]["protocol"]["sha256"] == reporter.digest(protocol)
    source.write_text("later changed source")
    with pytest.raises(ValueError, match="Unreconstructable"):
        reporter.validate_roundoff(report, directory, project)
