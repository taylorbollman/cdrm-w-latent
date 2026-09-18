"""Bounded CPU report guards and shared-prefix plotting checks; no live inputs."""
import copy
import hashlib

import pytest

from scripts import rt_a5_quadratic_input_report as report


def terminal(end=50000):
    return {"schema": report.TRAIN_SCHEMA, "start_update": 0, "parent_checkpoint": None,
        "endpoint": 50000, "status": "complete", "completed_updates": end,
        "confirmation_evaluated": False, "latent_rollout_evaluated": False,
        "requested_endpoint_reached": True, "injection_coefficient": report.coefficient(end),
        "wandb": {"status": "synced"}, "checkpoints": [{"completed_updates": 1000}, {"completed_updates": end}]}


def test_completed_means_actual_50k_not_a_shorter_successful_job():
    assert report.validate_endpoint(terminal(), {}, None) == (50000, False)
    with pytest.raises(ValueError, match="requested 50k"):
        report.validate_endpoint(terminal(10000), {}, None)
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
