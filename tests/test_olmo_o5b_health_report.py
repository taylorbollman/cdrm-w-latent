"""Recorded health is distinct from task quality and bitwise replay."""
import copy
import hashlib
import json
import math
import sys

import pytest

from scripts import olmo_o5b_health_report as health


def counters(update):
    return {"optimizer_updates": update, "microbatches": 2 * update,
            "documents": 32 * update, "input_tokens": 320 * update,
            "ce_positions": 288 * update, "latent_pairs": 0, "kl_triples": 0}


def events():
    return [{"kind": "update", "schema": "olmo-lm-optimizer-step-v1", "update_completed": True,
             "counters": counters(u), "counts": {"ce": 288, "latent": 0, "kl": 0},
             "beta": 0., "objective": 4., "pass_ce_means": [2., 2.],
             "gradient_norm_before_clip": 10. if u == 50 else 1. + u / 100,
             "group_gradient_norm_after_clip": {"backbone": 1., "fusion": 0.},
             "max_grad_norm": 1., "lr_used": [1e-5, 1e-4], "lr_next": [1e-5, 1e-4],
             "objective_weights": {"ce": 1., "latent": 0., "kl": 0.},
             "optimizer_state_bytes_by_device": {"cpu": 4, "cuda:0": 100},
             "stack_input_tokens": 640, "seconds": .75, "peak_allocated_gib": 49.}
            for u in range(1, 103)]


@pytest.fixture
def evidence(tmp_path):
    for arm in health.ARMS:
        directory = tmp_path / arm
        directory.mkdir()
        value = {"schema": "olmo-o5b-arm-v1", "arm": arm, "status": "completed",
                 "finished_utc": "2026-09-22", "configuration": {"same": True},
                 "source_fingerprint": {"source": "frozen"}, "counters": counters(102),
                 "data_cursor": 3264, "resumptions": []}
        rows = events()
        if arm == "fbt":
            rows[0]["gradient_norm_before_clip"] += .001
            rows[1]["objective"] += .002
            rows[1]["pass_ce_means"] = [2.001, 2.001]
            for row in rows:
                row["seconds"] = .8
            for row in rows[-2:]:
                row["beta"] = 1.
                row["group_gradient_norm_after_clip"] = {"backbone": .8, "fusion": .6}
        (directory / "report.json").write_text(json.dumps(value))
        write_events(directory / "events.jsonl", rows)
    return tmp_path


def write_events(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def mutate_report(root, fn):
    path = root / "fbt/report.json"
    value = json.loads(path.read_text())
    fn(value)
    path.write_text(json.dumps(value))


def mutate_events(root, fn):
    path = root / "fbt/events.jsonl"
    rows = [json.loads(row) for row in path.read_text().splitlines()]
    fn(rows)
    write_events(path, rows)


def test_reports_observed_health_and_nonexact_warmup(evidence):
    summary = health.build_summary(evidence)
    assert summary["schema"] == "olmo-o5b-optimization-summary-v1"
    assert summary["arms"]["ordinary"]["clipping"]["above_limit"] == 102
    assert summary["arms"]["ordinary"]["gradient_norm_before_clip"]["maximum_update"] == 50
    assert summary["arms"]["ordinary"]["gradient_norm_before_clip"]["max"] == 10.
    recent = summary["arms"]["fbt"]["last100"]
    assert recent["first_update"] == 3 and recent["last_update"] == 102
    assert recent["fusion_squared_norm_fraction"]["max"] == pytest.approx(.36)
    assert recent["fusion_squared_norm_fraction"]["mean"] == pytest.approx(.0072)
    warmup = summary["first100_warmup"]
    assert warmup["all_structural_fields_match"] and warmup["both_beta_zero"]
    assert warmup["exact_non_timing_update_matches"] == 98
    assert warmup["objective"]["exact_matches"] == 99
    assert warmup["objective"]["max_absolute_difference"] == pytest.approx(.002)
    assert warmup["objective"]["maximum_difference_update"] == 2
    assert warmup["gradient_norm_before_clip"]["maximum_difference_update"] == 1
    assert summary["input_sha256"]["fbt"]["events.jsonl"] == hashlib.sha256((evidence / "fbt/events.jsonl").read_bytes()).hexdigest()
    assert summary["script_sha256"] == hashlib.sha256(open(health.__file__, "rb").read()).hexdigest()
    assert "not a scan of model tensors" in summary["arms"]["fbt"]["finite_event_check"]["scope"]


def test_quantiles_interpolate_instead_of_nearest_rank():
    assert health.quantile([1., 2., 3., 10.], .95) == pytest.approx(8.95)
    assert health.quantile([1., 2., 3., 10.], .99) == pytest.approx(9.79)


@pytest.mark.parametrize("location", ["norm", "unscored_event", "nested_audit", "report"])
def test_rejects_nonfinite_even_outside_selected_update_records(evidence, location):
    if location == "report":
        mutate_report(evidence, lambda value: value.update(observation=float("nan")))
    else:
        def change(rows):
            if location == "norm": rows[0]["gradient_norm_before_clip"] = float("inf")
            elif location == "unscored_event": rows.append({"kind": "evaluation", "metric": float("nan")})
            else: rows.append({"kind": "audit", "nested": [{"value": -float("inf")}]})
        mutate_events(evidence, change)
    with pytest.raises(ValueError, match="Nonfinite"):
        health.build_summary(evidence)


@pytest.mark.parametrize("case", ["running", "config", "source", "exposure", "missing_update", "future_update", "unfinished_step",
                                  "missing_group", "duplicate", "truncated_json"])
def test_rejects_incomplete_or_incompatible_evidence(evidence, case):
    if case == "running": mutate_report(evidence, lambda value: value.update(status="running"))
    elif case == "config": mutate_report(evidence, lambda value: value.update(configuration={"different": True}))
    elif case == "source": mutate_report(evidence, lambda value: value.update(source_fingerprint={"different": True}))
    elif case == "exposure": mutate_report(evidence, lambda value: value["counters"].update(input_tokens=1))
    elif case == "missing_update": mutate_events(evidence, lambda rows: rows.pop(1))
    elif case == "future_update": mutate_events(evidence, lambda rows: rows[-1]["counters"].update(optimizer_updates=103))
    elif case == "unfinished_step": mutate_events(evidence, lambda rows: rows[1].update(update_completed=False))
    elif case == "missing_group": mutate_events(evidence, lambda rows: rows[1]["group_gradient_norm_after_clip"].pop("fusion"))
    elif case == "duplicate": mutate_events(evidence, lambda rows: rows.append(copy.deepcopy(rows[1])))
    elif case == "truncated_json":
        with (evidence / "fbt/events.jsonl").open("a") as stream: stream.write('{"kind":')
    with pytest.raises(ValueError):
        health.build_summary(evidence)


def test_resumed_duplicate_selects_latest_and_counts_audit_events(evidence):
    mutate_report(evidence, lambda value: value.update(resumptions=[{"from_update": 1}]))
    def change(rows):
        abandoned = copy.deepcopy(rows[1])
        abandoned["gradient_norm_before_clip"] = 1000.
        rows.insert(1, abandoned)
    mutate_events(evidence, change)
    summary = health.build_summary(evidence)
    selected = summary["arms"]["fbt"]["event_selection"]
    assert selected["raw_update_events"] == 103 and selected["selected_updates"] == 102
    assert selected["superseded_update_events"] == 1
    assert summary["arms"]["fbt"]["gradient_norm_before_clip"]["max"] == 10.


def test_warmup_metadata_mismatch_is_exposed_not_called_exact(evidence):
    mutate_events(evidence, lambda rows: rows[3].update(lr_used=[2e-5, 1e-4]))
    summary = health.build_summary(evidence)
    assert not summary["first100_warmup"]["all_structural_fields_match"]
    assert summary["first100_warmup"]["structural_fields"]["lr_used"]["mismatch_updates"] == [4]


def test_cli_writes_only_after_both_arms_complete(evidence, monkeypatch):
    output = evidence / "summary.json"
    monkeypatch.setattr(sys, "argv", ["health", "--runs", str(evidence), "--output", str(output)])
    mutate_report(evidence, lambda value: value.update(status="running"))
    with pytest.raises(ValueError, match="not completed"):
        health.main()
    assert not output.exists()
    mutate_report(evidence, lambda value: value.update(status="completed"))
    health.main()
    assert json.loads(output.read_text())["status"] == "completed"


def test_cli_cannot_overwrite_input_evidence(evidence, monkeypatch):
    path = evidence / "ordinary/report.json"
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["health", "--runs", str(evidence), "--output", str(path)])
    with pytest.raises(ValueError, match="overwrite"):
        health.main()
    assert path.read_bytes() == before
