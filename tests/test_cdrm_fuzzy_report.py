"""Self-contained CPU report tests; no native data, models or live-run reads."""
import copy
import json
from pathlib import Path

import pytest

from scripts import cdrm_fuzzy_report as report


def protocol():
    return {"calibration": {"epochs": 1, "learning_rates": [1e-4, 5e-4, 1e-3]},
            "models": {"cdrm": {"parameters": 153175040}, "seq": {"parameters": 153437184}}}


def numerical(initial=True, adam_l2=.08, cosine=.997):
    full = {"global_parameter_relative_l2": .004, "global_parameter_l2_pass": True,
            "per_tensor_l2_failures": [], "per_tensor_maximum_failures": [], "fp32_elementwise_failures": ["tiny-cancellation"], "pass": True}
    adam = {"global_delta_relative_l2": adam_l2, "global_delta_cosine": cosine,
            "guardrail_pass": cosine >= .99 if initial else adam_l2 <= .015625,
            "guardrail": "initial delta cosine >=.99" if initial else "trained delta relative L2 <=.015625"}
    return {"schema": report.NUM_SCHEMA, "status": "diagnostics_complete", "arm": "cdrm",
            "completed_updates": 0 if initial else 100, "fixture": {"shape": [128, 300], "split": "train"},
            "compiler": {arm: {"required": True, "validation_status": "passed", "require_graphs": True} for arm in ("tiled_fp32", "mixed")},
            "comparisons": {"mixed_vs_tiled_fp32": {"actual_ce_gradients": full, "independent_side_gradients": copy.deepcopy(full),
                "adam": adam, "machine_screens_pass": adam["guardrail_pass"], "absolute_ce_difference": .0001}},
            "candidate_checks_pass": adam["guardrail_pass"], "raw_machine_screens_pass": adam["guardrail_pass"], "scale_check_performed": False}


def entry(value, name):
    return {"path": f"{name}/report.json", "sha256": "a" * 64, "report": value, "observation": "closed_report"}


def calibration(arm, lr, steps=2, accuracy=.6):
    identity = {"base_lr": lr, "physical_batch": 128, "updates_per_epoch": 2,
                "shared_initialization_sha256": "shared", "data": {"train": "same"}, "source_sha256": {"code": "same"}}
    metric = {"ce": 1., "token_accuracy": accuracy, "sequence_exact_match": .1, "targets": 20, "examples": 4}
    history = [{"update": u, "epoch": 1, "batch_in_epoch": u - 1, "learning_rate": lr,
                "seconds": float(u), "native_loss": 1.5} for u in range(1, steps + 1)]
    return {"schema": report.TRAIN_SCHEMA, "status": "complete", "arm": arm, "precision": "bf16", "identity": identity,
            "history": history, "development": {str(u): {"native": copy.deepcopy(metric), "answer": copy.deepcopy(metric)} for u in (0, steps)},
            "completed_epochs": 1 if steps == 2 else 0}


def test_initial_adam_distance_does_not_inherit_the_trained_l2_gate():
    case = numerical(initial=True, adam_l2=.08, cosine=.997)
    row = report.numerical_rows(case)[0]
    assert row["adam_guardrail_pass"] is True
    assert row["initial_relative_l2_is_descriptive"] is True
    assert row["adam_relative_l2"] == .08
    assert row["fp32_elementwise_failures"] == ["tiny-cancellation"]
    assert report.numerical_consistency_errors(case) == []
    trained = numerical(initial=False, adam_l2=.08, cosine=.997)
    assert report.numerical_rows(trained)[0]["adam_guardrail_pass"] is False
    assert report.numerical_consistency_errors(trained) == []


def test_wrong_adam_summary_and_missing_compiler_cannot_appear_verified():
    case = numerical()
    case["comparisons"]["mixed_vs_tiled_fp32"]["adam"]["guardrail_pass"] = False
    del case["compiler"]["mixed"]
    errors = report.numerical_consistency_errors(case)
    assert any("Adam criterion" in error for error in errors)
    assert any("compiler arms" in error for error in errors)


def test_failed_numerical_attempt_is_retained_without_invented_metrics():
    case = {"schema": report.NUM_SCHEMA, "status": "execution_failed", "arm": "cdrm", "error_type": "OutOfMemoryError"}
    row = report.numerical_rows(case)[0]
    assert row["status"] == "execution_failed"
    assert row["comparison"] is None and row["candidate_checks_pass"] is None


def test_output_scaling_leaf_failure_cannot_be_hidden_by_pass_summary():
    case = numerical()
    case["scale_check_performed"] = True
    check = {"pass": True, "fixed_forward_logits_bitwise_equal": True,
             "scaling": {scale: {"parameter": {"bitwise_normalized_equal": True}} for scale in ("0.03125", "32.0")}}
    pair = case["comparisons"]["mixed_vs_tiled_fp32"]
    pair.update(scaling_pass=True, output_scaling={"mixed": copy.deepcopy(check), "tiled_fp32": copy.deepcopy(check)})
    assert not report.numerical_consistency_errors(case)
    pair["output_scaling"]["mixed"]["scaling"]["32.0"]["parameter"]["bitwise_normalized_equal"] = False
    assert any("Scaling leaves" in error for error in report.numerical_consistency_errors(case))


def test_partial_six_lr_cohort_does_not_select_winners():
    value = calibration("cdrm", 5e-4)
    result = report.calibration_cohort([entry(value, "one")], protocol())
    assert len(result["selected_runs"]) == 6
    assert result["all_six_frozen_endpoints_complete"] is False
    assert result["selection"] == {}


def test_all_six_lr_selections_use_same_endpoint_and_predeclared_ce_tie_break():
    entries = []
    for arm in ("cdrm", "seq"):
        for lr in protocol()["calibration"]["learning_rates"]:
            value = calibration(arm, lr)
            value["development"]["2"]["native"]["ce"] = 0.9 if lr == 5e-4 else 1.
            value["development"]["2"]["answer"] = copy.deepcopy(value["development"]["2"]["native"])
            entries.append(entry(value, f"calibration/{arm}-lr{lr}-e1"))
    result = report.calibration_cohort(entries, protocol())
    assert result["all_six_frozen_endpoints_complete"]
    assert not result["errors"]
    assert {arm: row["lr"] for arm, row in result["selection"].items()} == {"cdrm": 5e-4, "seq": 5e-4}


def test_continuation_uses_longest_prefix_and_detects_conflicts():
    prior, current = calibration("cdrm", 5e-4, steps=1), calibration("cdrm", 5e-4)
    current["development"]["1"] = copy.deepcopy(prior["development"]["1"])
    prior["history"][0]["seconds"] = 17.
    result = report.calibration_cohort([entry(prior, "prior"), entry(current, "current")], protocol())
    assert not result["errors"]
    assert len(result["training_rows"]) == 2
    prior["history"][0]["native_loss"] += .1
    assert any("Conflicting continuation" in error for error in report.calibration_cohort([entry(prior, "prior"), entry(current, "current")], protocol())["errors"])


def test_profile_excludes_warmup_and_rejects_summary_inconsistency():
    value = {"schema": report.PROFILE_SCHEMA, "status": "profile_complete", "arm": "cdrm", "shape": [128, 300],
             "precision": "bf16", "updates": [{"seconds": 100., "warmup": True}, {"seconds": 2., "warmup": False}, {"seconds": 4., "warmup": False}],
             "timing": {"mean_seconds": 3., "median_seconds": 3., "steady_updates": 2}, "compiler": {"required": True}}
    assert report.profile_row(value)["mean_seconds"] == 3.
    assert not report.profile_consistency_errors(value)
    value["timing"]["mean_seconds"] = 100.
    assert report.profile_consistency_errors(value)


def test_cost_projection_requires_common_fit_and_correct24run_accounting():
    base = {"fit_success": True, "precision": "bf16", "length": 300, "batch": 128}
    profiles = [{**base, "arm": "cdrm", "mean_seconds": 2.}, {**base, "arm": "seq", "mean_seconds": 1.}]
    calibration = {"selected_runs": []}
    output = report.cost_projections(profiles, calibration)
    assert output["rows"][0]["updates_total"] == 60000
    assert output["rows"][0]["training_hours"] == 25.
    assert output["rows"][1]["updates_total"] == 120000
    assert "not a proven bound" in output["rows"][0]["estimate"]
    profiles[1]["fit_success"] = False
    assert report.cost_projections(profiles, calibration)["status"] == "unavailable"


def test_report_scan_excludes_source_snapshots_and_includes_orphan_progress(tmp_path):
    for directory, name in (("source/nested", "report.json"), ("staged-retention/authorities/copy", "report.json"),
                            ("failed", "progress.json"), ("complete", "report.json"), ("recovery/cdrm-control", "report.json")):
        path = tmp_path / directory / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"schema": report.NUM_SCHEMA, "status": "running"}))
    entries = report.scan_reports(tmp_path)
    assert len(entries) == 3
    assert any(row["observation"] == "orphan_progress_incomplete" for row in entries)
    assert any("/recovery/" in row["path"] for row in entries)


def test_prefix_or_recovery_cannot_substitute_for_an_authoritative_endpoint():
    entries = []
    for arm in ("cdrm", "seq"):
        for lr in protocol()["calibration"]["learning_rates"]:
            value = calibration(arm, lr)
            entries.append(entry(value, f"calibration/{arm}-lr{lr}-prefix"))
    recovered = copy.deepcopy(entries[0])
    recovered["path"] = "recovery/cdrm-lr0.0001-recovery/report.json"
    recovered["report"]["recovery_comparison"] = {"bitwise_state_and_nontiming_metrics_equal": True}
    result = report.calibration_cohort(entries + [recovered], protocol())
    assert not result["errors"]
    assert not result["all_six_frozen_endpoints_complete"] and result["selection"] == {}
    assert len(result["recovery"]) == 1
    for item in list(entries):
        endpoint = copy.deepcopy(item)
        endpoint["path"] = endpoint["path"].replace("-prefix/", "-e1/")
        entries.append(endpoint)
    result = report.calibration_cohort(entries + [recovered], protocol())
    assert not result["errors"] and result["all_six_frozen_endpoints_complete"]
    assert all(row["role"] == "authoritative_endpoint" for row in result["selected_runs"])
    assert all(row["reference_reports"] for row in result["selected_runs"])


def test_recovery_name_remains_ineligible_even_if_it_ends_like_an_endpoint():
    value = calibration("cdrm", 5e-4)
    candidate = entry(value, "recovery/cdrm-lr5e-4-e1")
    assert report.calibration_role(candidate, 1) == "recovery_reference"
    result = report.calibration_cohort([candidate], protocol())
    assert not any(row.get("frozen_endpoint_complete") for row in result["selected_runs"])


def test_blob_audit_rejects_modified_bytes_and_reference_outside_new_lineage(tmp_path):
    lineage = tmp_path / "lineage"; lineage.mkdir()
    blob = lineage / "packet.pt"; blob.write_bytes(b"frozen packet")
    audit = report.ArtifactAudit(lineage, project=tmp_path)
    reference = {"path": str(blob), "sha256": report.digest(blob), "bytes": blob.stat().st_size}
    audit.reference(reference)
    blob.write_bytes(b"changed packet")
    with pytest.raises(ValueError, match="changed during"):
        audit.reference(reference)
    outside = tmp_path / "old.pt"; outside.write_bytes(b"old")
    with pytest.raises(ValueError, match="outside"):
        audit.reference({"path": str(outside), "sha256": report.digest(outside)})


def test_failed_cases_render_without_completed_six_run_or_final_test_claim(tmp_path):
    failure = entry({"schema": report.NUM_SCHEMA, "status": "execution_failed", "arm": "cdrm"}, "failed")
    summary = report.summarize([failure], protocol())
    assert summary["attempts"][0]["status"] == "execution_failed"
    assert summary["final_test_evaluated"] is False
    assert summary["final_sweep_performed"] is False
    assert "pending" in report.markdown(summary)
    paths = report.make_plots(summary, tmp_path)
    assert len(paths) == 6 and all(path.is_file() for path in paths)
