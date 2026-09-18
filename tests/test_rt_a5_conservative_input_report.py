"""Bounded CPU report guards and shared-prefix plotting checks; no live inputs."""
import copy
import hashlib
import json

import pytest

from scripts import rt_a5_conservative_input_report as report


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


def test_saved_comparison_keeps_all_arms_at10k_and_rejects_initialization_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "bound", lambda row, expected=None: row)
    common = {key: "same" for key in ("model_config", "batch_size", "length", "train_rows", "data_manifest_sha256", "data_order_seed",
        "seed", "predictor_seed", "projection_seed", "objective", "optimizer", "nextlat_config", "precision")}
    initial = {"model_parameter_sha256": "a" * 64}
    metric = {"rows": 102400, "length": 36, "role": "ood_dev", "update": 10000,
        "tokens": 102400 * 36, "evaluated_rows": 102400, "route": "backbone_only",
        "isolated_state_accuracy": [1.] * 36, "cumulative_prefix_exactness": [1.] * 36,
        "per_position_ce": [.1] * 36, "token_accuracy": 1., "whole_word_exact_match": 1.,
        "final_state_accuracy": 1., "ce": sum([.1] * 36) / 36}
    order = [{"update": step, "order_chain": str(step)} for step in range(1, 10001)]
    history = "\n".join(json.dumps(row) for row in order).encode()
    payloads = {}
    for name, relative, directory, schema, value in (
        ("fixed", "20260915T150000Z-embedding-input10k", "train-injection", "rt-a5-embedding-injection-training-v1", .02),
        ("previous_quadratic", "20260915T160000Z-input-quadratic50k", "train-quadratic", "rt-a5-quadratic-input-training-v1", .05 * (.2 ** 2)),
    ):
        protocol = {"training_directory": str(tmp_path / ".runtime/rt-a5" / relative / directory),
            "strict_contract": common, "initialization": initial, "coefficient": .02,
            "maximum_coefficient": .05, "warmup_updates": 50000}
        old = {"schema": schema, "status": "complete", "completed_updates": 10000,
            "wandb": {"status": "synced", "run_url": "fixture"}, "contract": common,
            "initialization": initial, "checkpoints": [{"completed_updates": 10000}],
            "evaluations": [{**metric, "injection_coefficient": value}]}
        payloads[f"comparison_{name}_protocol"] = json.dumps(protocol).encode()
        payloads[f"comparison_{name}_report"] = json.dumps(old).encode()
        payloads[f"comparison_{name}_history"] = history
    current = {"contract": common, "initialization": initial, "wandb": {"run_url": "fixture"}}
    curves = {"10000": {"ood_dev": [{"E": 1., "A": 1., "M": 1.}]}}
    capture = lambda name, _path: payloads[name]
    rows = report.ten_k_comparison(capture, current, curves, order)
    assert [row["update"] for row in rows] == [10000] * 3
    assert [row["injection_coefficient"] for row in rows] == [.02, .05 * (.2 ** 2), .001 * (.2 ** 2)]
    changed = copy.deepcopy(current); changed["initialization"]["model_parameter_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="every initial learned tensor"):
        report.ten_k_comparison(capture, changed, curves, order)
