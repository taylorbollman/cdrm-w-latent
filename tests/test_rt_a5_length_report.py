"""Count semantics and retained/reporting outputs; no models or GPU execution."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import rt_a5_length_report as report


def metric(step=10000):
    n = report.EXPECTED_ROWS
    exact = [n] * 12 + [80000, 20000, 1000, 1] + [0] * 20
    state = [n] * 12 + [80000, 21000, 2000, 1000] + [1700] * 20
    if step == 5000:
        exact[12], state[12] = 82000, 82000
    return {"rows": n, "evaluated_rows": n, "length": 36, "tokens": n * 36,
            "role": "ood_dev", "update": step, "isolated_state_accuracy": [x / n for x in state],
            "cumulative_prefix_exactness": [x / n for x in exact], "token_accuracy": sum(state) / (n * 36),
            "whole_word_exact_match": exact[-1] / n, "final_state_accuracy": state[-1] / n}


def fixture_arm(tmp_path, monkeypatch):
    directory = tmp_path / "train-rt"
    directory.mkdir()
    checkpoints = []
    for step in report.STEPS:
        path = directory / f"step-{step}.pt"
        path.write_bytes(f"opaque saved checkpoint {step}".encode())
        checkpoints.append({**report.hash_file(path), "completed_updates": step})
    packet = {"start_update": 0, "completed_updates": 10000,
              "contract": {"architecture": "rt", "length": 12, "width": 512, "seed": 1234,
                           "source_sha256": "a" * 64, "data_manifest_sha256": "b" * 64},
              "evaluations": [metric(step) for step in report.STEPS], "checkpoints": checkpoints,
              "initialization": {"parameter_count": 6357504, "canonical_sha256": "c" * 64},
              "source_files": {"fixture.py": "d" * 64}, "wandb": {"run_url": "https://example.test/historical"}}
    report.write_json(directory / "report.json", packet)
    arm = {"report": packet, "input_files": {"report": report.hash_file(directory / "report.json")}}
    monkeypatch.setattr(report, "read_arm", lambda directory, architecture: arm)
    return directory, packet


def prefix_report(directory, packet):
    value = {"schema": "rt-a5-trained-prefix-v1", "status": "complete", "architecture": "rt", "passed": True,
             "role": "ood_dev", "rows": 1024, "confirmation_evaluated": False, "completed_updates": 10000,
             "contract": packet["contract"], "checkpoint": packet["checkpoints"][-1],
             "parameters_unchanged": True, "no_gradients_created": True, "training_updates_performed": 0,
             "checks": {str(length): {"passed": True, "prediction_disagreements": 0, "accuracy_counts_equal": True}
                        for length in (12, 13, 14, 16)}}
    path = directory / "prefix.json"
    report.write_json(path, value)
    return path, value


def test_counts_means_and_zero_rule_preserve_nonzero_rounded_percentages():
    rows = report.curve_rows(metric())
    assert len(rows) == 36
    assert rows[15]["prefix_exact_count"] == 1
    assert f"{100 * rows[15]['E']:.2f}" == "0.00"
    assert rows[15]["token_correct_count"] == sum(r["state_correct_count"] for r in rows[:16])
    assert rows[15]["M"] == rows[15]["token_correct_count"] / (102400 * 16)
    assert report.horizons(rows)["first_observed_zero_length"] == 17
    assert rows[-1]["A"] > 0 and rows[-1]["M"] > rows[-1]["A"]
    assert not any("M_low" in k or "M_high" in k for k in rows[0])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1, 1.01, 0.000001])
def test_invalid_or_nonintegral_recorded_fraction_rejected(value):
    with pytest.raises(ValueError):
        report.recover_count(value, 102400)


def test_wilson_handles_zero_full_and_half_counts():
    low, high = report.wilson95(0, 102400)
    assert low == pytest.approx(0, abs=1e-15)
    assert high == pytest.approx(3.75128308e-5, rel=1e-5)
    low_full, high_full = report.wilson95(102400, 102400)
    assert low_full == pytest.approx(1 - high)
    assert high_full == pytest.approx(1)
    assert sum(report.wilson95(51200, 102400)) == pytest.approx(1)


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(rows=4096),
    lambda m: m.update(evaluated_rows=4096),
    lambda m: m.update(tokens=100),
    lambda m: m.update(role="confirmation"),
    lambda m: m.update(token_accuracy=.5),
    lambda m: m["cumulative_prefix_exactness"].__setitem__(0, .5),
    lambda m: m["cumulative_prefix_exactness"].__setitem__(17, 1 / 102400),
])
def test_bad_curve_denominators_or_semantics_are_rejected(mutate):
    value = metric()
    mutate(value)
    with pytest.raises(ValueError):
        report.curve_rows(value)


def test_thresholds_include_equality_and_missing_threshold():
    rows = [{"length": i + 1, "E": value, "prefix_exact_count": round(value * 100), "E_high95": 1}
            for i, value in enumerate((.95, .50, .10, .01, 0.))]
    result = report.horizons(rows)
    assert result["last_length_at_or_above"] == {"95%": 1, "50%": 2, "10%": 3, "1%": 4}
    assert report.horizons(rows[1:])["last_length_at_or_above"]["95%"] is None


def test_summary_verifies_checkpoint_and_prefix_identity(tmp_path, monkeypatch):
    directory, packet = fixture_arm(tmp_path, monkeypatch)
    prefix_path, prefix = prefix_report(directory, packet)
    result = report.make_summary(directory, "rt", prefix_path)
    assert result["delta_10000_minus_5000"][12]["E"] < 0
    assert result["prefix_check"]["input"]["sha256"] == report.hash_file(prefix_path)["sha256"]
    prefix["checkpoint"] = {**prefix["checkpoint"], "sha256": "wrong"}
    report.write_json(prefix_path, prefix)
    with pytest.raises(ValueError, match="checkpoint differs"):
        report.make_summary(directory, "rt", prefix_path)
    Path(packet["checkpoints"][0]["path"]).write_bytes(b"corrupt 5k checkpoint")
    with pytest.raises(ValueError, match="Checkpoint hash/size differs at 5000"):
        report.make_summary(directory, "rt")


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(passed=False),
    lambda p: p.update(architecture="seq"),
    lambda p: p.update(confirmation_evaluated=True),
    lambda p: p.update(rows=32),
    lambda p: p["checks"].pop("13"),
    lambda p: p["checks"]["13"].update(passed=False),
    lambda p: p.update(contract={"source_sha256": "wrong"}),
])
def test_failed_or_unmatched_prefix_check_rejected(tmp_path, monkeypatch, mutate):
    directory, packet = fixture_arm(tmp_path, monkeypatch)
    path, prefix = prefix_report(directory, packet)
    mutate(prefix)
    report.write_json(path, prefix)
    with pytest.raises(ValueError):
        report.make_summary(directory, "rt", path)


def test_outputs_and_tracking_retain_exact_curves_and_hashes(tmp_path, monkeypatch):
    directory, packet = fixture_arm(tmp_path, monkeypatch)
    prefix, _ = prefix_report(directory, packet)
    calls = []

    class FakeTracker:
        def __init__(self, **kwargs):
            self.record = {"run_url": "https://example.test/length-report", "status": "pending"}
        def start(self, config):
            calls.append(("start", config))
        def log(self, values):
            calls.append(("log", values))
        def summary(self, values):
            calls.append(("summary", values))
        def finish(self, succeeded):
            self.record["status"] = "synced" if succeeded else "failed"

    monkeypatch.setattr(report, "OnlineTracker", FakeTracker)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=lambda p: {"image": p},
                                                            Table=lambda **kw: {"table": kw}))
    output = tmp_path / "output"
    args = SimpleNamespace(run_dir=directory, architecture="rt", prefix_report=prefix,
                           output_dir=output, wandb_group="fixture", prefix_assessment=None)
    result = report.run(args)
    assert result["status"] == "complete"
    assert len((output / "length-curves.csv").read_text().splitlines()) == 73
    assert json.loads((output / "plot-data.json").read_text())["curves"] == result["curves"]
    for name in ("length-full", "length-zoom"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert (output / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    markdown = (output / "report.md").read_text()
    assert "1,024 words" in markdown and "102,400-word" in markdown
    assert "Final confirmation remains unevaluated" in markdown
    assert result["artifacts"]["inputs/prefix-check.json"]["sha256"] == report.hash_file(prefix)["sha256"]
    assert any("length/table" in values for kind, values in calls if kind == "log")
    with pytest.raises(FileExistsError):
        report.run(args)


def test_strict_failed_screen_requires_linked_qualified_assessment(tmp_path, monkeypatch):
    directory, packet = fixture_arm(tmp_path, monkeypatch)
    path, prefix = prefix_report(directory, packet)
    prefix.update(status="failed", passed=False, error_type="AssertionError")
    prefix["checks"]["12"]["passed"] = False
    report.write_json(path, prefix)
    with pytest.raises(ValueError, match="qualified assessment"):
        report.make_summary(directory, "rt", path)
    assessment = {"schema": "rt-a5-prefix-assessment-v1", "status": "complete", "architecture": "rt",
                  "completed_updates": 10000, "usable_for_length_metrics": True,
                  "original_prefix_report_sha256": report.hash_file(path)["sha256"],
                  "source_sha256": packet["contract"]["source_sha256"],
                  "data_manifest_sha256": packet["contract"]["data_manifest_sha256"],
                  "qualification": "Strict elementwise screen failed; sampled accuracy counts agree exactly.",
                  "checks": {length: {"correctness_disagreements": 0, "prediction_disagreements": 0,
                                      "accuracy_counts_equal": True} for length in prefix["checks"]}}
    assessment_path = directory / "assessment.json"
    report.write_json(assessment_path, assessment)
    summary = report.make_summary(directory, "rt", path, assessment_path)
    assert summary["prefix_check"]["strict_logit_screen_passed"] is False
    assert summary["prefix_check"]["report"]["status"] == "failed"
    assert "remains recorded as failed" in report.markdown_report(summary)
    # Prior assessment schema can infer identical correctness only from identical predictions.
    for item in assessment["checks"].values():
        item.pop("correctness_disagreements")
    report.write_json(assessment_path, assessment)
    assert report.make_summary(directory, "rt", path, assessment_path)["prefix_check"]["strict_logit_screen_passed"] is False
    assessment["original_prefix_report_sha256"] = "wrong"
    report.write_json(assessment_path, assessment)
    with pytest.raises(ValueError, match="this exact report"):
        report.make_summary(directory, "rt", path, assessment_path)


def test_changed_wrong_class_requires_equal_correctness_masks_not_cancelled_counts(tmp_path, monkeypatch):
    directory, packet = fixture_arm(tmp_path, monkeypatch)
    packet["contract"]["architecture"] = "seq"
    path, prefix = prefix_report(directory, packet)
    prefix.update(architecture="seq", status="failed", passed=False, error_type="AssertionError")
    prefix["checks"] = {length: prefix["checks"][length] for length in ("12", "13")}
    prefix["checks"]["13"].update(passed=False, prediction_disagreements=1)
    report.write_json(path, prefix)
    with pytest.raises(ValueError, match="qualified assessment"):
        report.make_summary(directory, "seq", path)
    assessment = {"schema": "rt-a5-prefix-assessment-v1", "status": "complete", "architecture": "seq",
                  "completed_updates": 10000, "usable_for_length_metrics": True,
                  "original_prefix_report_sha256": report.hash_file(path)["sha256"],
                  "source_sha256": packet["contract"]["source_sha256"],
                  "data_manifest_sha256": packet["contract"]["data_manifest_sha256"],
                  "qualification": "One wrong class changed; per-word/position correctness stayed identical.",
                  "checks": {length: {"correctness_disagreements": 0,
                                      "prediction_disagreements": prefix["checks"][length]["prediction_disagreements"],
                                      "accuracy_counts_equal": True} for length in prefix["checks"]}}
    assessment_path = directory / "assessment.json"
    report.write_json(assessment_path, assessment)
    summary = report.make_summary(directory, "seq", path, assessment_path)
    assert summary["prefix_check"]["prediction_disagreements_by_length"] == {"12": 0, "13": 1}
    assert "changed predictions remained wrong" in report.markdown_report(summary)
    # Equal aggregate accuracy can conceal one correct->wrong and one wrong->correct change.
    prefix["checks"]["13"]["prediction_disagreements"] = 2
    report.write_json(path, prefix)
    assessment["original_prefix_report_sha256"] = report.hash_file(path)["sha256"]
    assessment["checks"]["13"].update(correctness_disagreements=2, prediction_disagreements=2)
    report.write_json(assessment_path, assessment)
    with pytest.raises(ValueError, match="aggregate counts are insufficient"):
        report.make_summary(directory, "seq", path, assessment_path)
