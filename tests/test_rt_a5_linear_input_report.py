"""Bounded CPU report guards and shared-prefix plotting checks; no live inputs."""
import copy
import hashlib
import json

import pytest

from scripts import rt_a5_linear_input_report as report


def terminal(end=10000):
    return {"schema": report.TRAIN_SCHEMA, "start_update": 0, "parent_checkpoint": None,
        "endpoint": 10000, "status": "complete", "completed_updates": end,
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
        "requested_endpoint_reached": True, "injection_coefficient": report.coefficient(end),
        "wandb": {"status": "synced"}, "checkpoints": [{"completed_updates": 1000}, {"completed_updates": end}]}


def test_completed_means_actual_10k_not_a_shorter_successful_job():
    assert report.validate_endpoint(terminal(), {}, None) == (10000, False)
    with pytest.raises(ValueError, match="requested 10k"):
        report.validate_endpoint(terminal(5000), {}, None)
    bad = terminal(); bad["injection_coefficient"] = .02
    with pytest.raises(ValueError, match="coefficient"):
        report.validate_endpoint(bad, {}, None)


def test_graceful_stop_requires_bound_retained_sentinel(tmp_path):
    stop = tmp_path / "STOP_AFTER_UPDATE"
    stop.write_text("Requested stop\n")
    data = terminal(1234)
    data.update(status="stopped", requested_endpoint_reached=False,
        stop_request={"reason": "user_stop_file", "observed_after_update": 1234,
            "path": str(stop), "sha256": hashlib.sha256(stop.read_bytes()).hexdigest()})
    protocol = {"resolved_args": {"stop_file": str(stop)}}
    assert report.validate_endpoint(data, protocol, None) == (1234, False)
    stop.write_text("changed\n")
    with pytest.raises(ValueError, match="sentinel changed"):
        report.validate_endpoint(data, protocol, None)


def test_partial_reporting_requires_an_explicit_retained_checkpoint_and_no_failure():
    data = terminal(1000); data.update(status="running", wandb={"status": "running"})
    assert report.validate_endpoint(data, {}, 1000) == (1000, True)
    with pytest.raises(ValueError, match="retained checkpoint"):
        report.validate_endpoint(data, {}, 999)
    with pytest.raises(ValueError, match="finished synced"):
        report.validate_endpoint(data, {}, None)
    data["status"] = "failed"
    with pytest.raises(ValueError, match="failed training"):
        report.validate_endpoint(data, {}, 1000)


def history():
    return [{"update": step, "order_chain": str(step), "examples_seen": step * 1024,
        "injection_coefficient": report.coefficient(step), "loss": .3,
        "state_loss": .2, "latent_loss": .1, "weighted_latent_loss": .1,
        "grad_norm": 1., "seconds": .1, "token_accuracy": .9} for step in range(1, 4)]


def test_history_uses_exact_global_schedule_and_order_and_excludes_partial_tail():
    rows = history()
    assert len(report.validate_history(rows, 2, rows, exact=False)) == 2
    with pytest.raises(ValueError, match="excess terminal"):
        report.validate_history(rows, 2, rows, exact=True)
    for field, value in (("injection_coefficient", .02), ("order_chain", "different"), ("latent_loss", float("nan"))):
        bad = copy.deepcopy(rows); bad[1][field] = value
        with pytest.raises(ValueError):
            report.validate_history(bad, 3, rows, exact=True)


def test_full_and_boundary_rows_and_plots_use_same_saved36_prefixes(tmp_path):
    rows = [{"update": 1000, "role": "ood_dev", "length": t, "E": 1 / t, "A": .8, "M": .9}
            for t in range(1, 37)]
    summary = {"curves": {"1000": {"ood_dev": rows}}, "checkpoint_updates": [0, 1000],
        "report_scope": "partial retained checkpoint", "through_update": 1000,
        "training_curve": [{"update": 1000, "state_ce": .2, "latent_loss": .1}]}
    assert report.plot_rows(summary, 1000) is rows
    figures = report.plots(summary, tmp_path)
    assert figures == ["length-full", "length-boundary", "length36-vs-updates", "training-and-schedule"]
    assert all((tmp_path / f"{name}.{suffix}").stat().st_size > 1000
               for name in figures for suffix in ("png", "pdf"))
    rows[0]["length"] = 2
    with pytest.raises(ValueError, match="identical 36-prefix"):
        report.plot_rows(summary, 1000)


def test_control_table_uses_only_common_saved_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "bound", lambda row, expected=None: row)
    monkeypatch.setattr(report, "hash_file", lambda path: {"sha256": "h"})
    directory = tmp_path / ".runtime/rt-a5/20260915T164000Z-l1r-four-layer-control10k/train-control"
    common = {key: "same" for key in ("model_config", "batch_size", "length", "train_rows", "data_manifest_sha256", "data_order_seed",
        "seed", "predictor_seed", "objective", "optimizer", "nextlat_config", "precision")}
    sources = {str(i): "s" for i in range(58)}
    initial = {"parameter_count": 13702656, "parameter_tensors": 43, "model_parameter_sha256": "a" * 64}
    protocol = {"schema": "rt-a5-l1r-four-layer-control-protocol-v2", "training_directory": str(directory),
                "strict_contract": common, "initialization": initial, "source_files": sources}
    metric = {"rows": 102400, "length": 36, "role": "ood_dev", "update": 5000, "route": "backbone_only",
        "tokens": 102400 * 36, "isolated_state_accuracy": [1.] * 36, "cumulative_prefix_exactness": [1.] * 36,
        "per_position_ce": [.1] * 36, "token_accuracy": 1., "whole_word_exact_match": 1.,
        "final_state_accuracy": 1., "ce": sum([.1] * 36) / 36}
    old = {"schema": "rt-a5-l1r-depth-training-v1", "status": "complete", "completed_updates": 10000,
        "endpoint": 10000, "start_update": 0, "parent_checkpoint": None, "wandb": {"status": "synced"},
        "contract": common, "initialization": initial, "source_files": sources,
        "checkpoints": [{"completed_updates": step, "sha256": "h"} for step in (0, 1000, 5000, 10000)],
        "evaluations": [metric]}
    state = {"schema": "rt-a5-l1r-four-layer-control-final-state-validation-v2", "passed": True,
        "completed_updates": 10000, "model_parameter_tensors": 43, "report": {"sha256": "h"}, "protocol": {"sha256": "h"},
        "checkpoint_inputs": {"5000": {"sha256": "h"}}}
    reference = [{"update": step, "order_chain": str(step)} for step in range(1, 10001)]
    payloads = {"control_protocol": json.dumps(protocol).encode(), "control_report": json.dumps(old).encode(),
        "control_state": json.dumps(state).encode(), "control_history": "\n".join(json.dumps(r) for r in reference).encode()}
    current = {"contract": common, "source_files": sources,
               "initialization": {"shared_four_layer_model_sha256": "a" * 64}}
    curves = {str(step): {"ood_dev": [{"E": .3, "A": .4, "M": .5}]} for step in (5000, 9665)}
    result = report.control_comparison(lambda name, path: payloads[name], current, curves, reference)
    assert [row["update"] for row in result] == [5000]
    assert result[0]["control"]["coefficient"] == 0
    assert result[0]["linear"]["coefficient"] == .001
