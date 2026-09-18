"""Bounded CPU report guards and shared-prefix plotting checks; no live inputs."""
import copy
import hashlib
import json

import pytest

from scripts import rt_a5_six_layer_input_report as report


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


def test_history_uses_exact_fixed_coefficient_and_order_and_excludes_partial_tail():
    rows = history()
    assert len(report.validate_history(rows, 2, rows, exact=False)) == 2
    with pytest.raises(ValueError, match="excess terminal"):
        report.validate_history(rows, 2, rows, exact=True)
    for field, value in (("injection_coefficient", .02), ("order_chain", "different"), ("latent_loss", float("nan"))):
        bad = copy.deepcopy(rows); bad[1][field] = value
        with pytest.raises(ValueError):
            report.validate_history(bad, 3, rows, exact=True)



def fixture_summary(end=10000):
    def rows(step):
        return [{"update": step, "role": "ood_dev", "length": t, "E": 1 / t, "A": .8, "M": .9} for t in range(1, 37)]
    curves = {str(step): {"ood_dev": rows(step)} for step in (1000, 5000, 10000)}
    current = {step: value for step, value in curves.items() if int(step) <= end}
    current[str(end)] = {"ood_dev": rows(end)}
    common = sorted(set(map(int, current)) & set(map(int, curves)))
    bins = [{"update": end, "state_ce": .2, "latent_loss": .1}]
    return {"curves": current, "checkpoint_updates": [0, *sorted(map(int, current))],
        "report_scope": "terminal saved endpoint", "through_update": end,
        "matched_update": max(common, default=None), "training_curve": bins,
        "training_status_at_snapshot": "complete" if end == 10000 else "stopped",
        "training_wandb": {"run_url": "https://example.test/current"}, "qualification": "One seed.",
        "baseline": {"curves": curves, "checkpoint_updates": [0,1000,5000,10000], "training_curve": bins,
                     "comparisons": [], "wandb": {"run_url": "https://example.test/baseline"}}}


@pytest.mark.parametrize("end", [10000, 5666])
def test_full_boundary_share_rows_and_unequal_terminal_is_separate(tmp_path, end):
    summary = fixture_summary(end)
    matched = summary["matched_update"]
    assert report.plot_rows(summary, matched) is summary["curves"][str(matched)]["ood_dev"]
    figures = report.plots(summary, tmp_path)
    expected = ["length-full", "length-boundary"]
    if end < 10000:
        expected += ["length-unequal-terminals"]
    assert figures == expected + ["length36-vs-updates", "training-losses"]
    assert all((tmp_path / f"{name}.{suffix}").stat().st_size > 1000 for name in figures for suffix in ("png", "pdf"))
    text = report.markdown(summary)
    assert "λ is fixed at 0.01" in text and "warmup" not in text
    assert ("Actual endpoints differ" in text) == (end < 10000)
    summary["curves"][str(matched)]["ood_dev"][0]["length"] = 2
    with pytest.raises(ValueError, match="identical 36-prefix"):
        report.plot_rows(summary, matched)


def test_actual_closed_six_layer_baseline_and_only_common_budgets():
    """Read saved baseline evidence only; never construct or evaluate a model."""
    lineage = report.ROOT / ".runtime/rt-a5/20260915T141414Z-l1r-six-layer-nextlat10k"
    old = json.loads((lineage / "train-depth/report.json").read_text())
    proto = json.loads((lineage / "protocol.json").read_text())
    history = [json.loads(line) for line in (lineage / "train-depth/history.jsonl").read_text().splitlines()]
    source = json.loads((report.LINEAGE / "implementation-tests.json").read_text())["source_files"]
    current = {"contract": old["contract"], "source_files": source, "initialization": {
        "shared_six_layer_model_sha256": old["initialization"]["model_parameter_sha256"],
        "reference_six_layer_initialization": old["initialization"]}}
    binding = {key: report.hash_file(lineage / "train-depth" / filename) for key, filename in
        (("report", "report.json"), ("config", "config.json"), ("history", "history.jsonl"))}
    binding["protocol"] = report.hash_file(lineage / "protocol.json")
    binding["checkpoints"] = {str(c["completed_updates"]): c for c in old["checkpoints"]}
    curves = {str(step): {"ood_dev": [{"E": .3, "A": .4, "M": .5}]} for step in (1000,5000,5666)}
    captured = {}
    def capture(name, path):
        captured[name] = path.read_bytes()
        return captured[name]
    baseline = report.baseline_evidence(capture, current, curves, history, binding)
    assert baseline["common_checkpoint_updates"] == [1000,5000]
    assert [row["update"] for row in baseline["comparisons"]] == [1000,5000]
    assert baseline["parameter_tensors"] == 61
    assert baseline["curves"]["10000"]["ood_dev"][-1]["E"] == .820771484375
    assert set(baseline["checkpoints"]) == {"0","1000","5000","10000"}
    current["initialization"]["shared_six_layer_model_sha256"] = "different"
    with pytest.raises(ValueError, match="exact 61"):
        report.baseline_evidence(capture, current, curves, history, binding)
