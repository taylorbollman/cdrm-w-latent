"""The automatic mixed-run decision uses checkpoint-bound full-word counts."""
import hashlib
import json

import pytest

from scripts.rt_nextlat_a5_fuzzy_gate import select_gate


def make_report(tmp_path, count=0):
    checkpoint = tmp_path / "step-010000.pt"
    checkpoint.write_bytes(b"fixture-checkpoint-content")
    row = {"task": "a5", "role": "ood_dev", "scope": "full", "update": 10000,
           "length": 36, "evaluated_rows": 102400, "whole_word_correct": count,
           "whole_word_exact_match": count / 102400,
           "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          "completed_updates": 10000}}
    report = {"status": "complete", "completed_updates": 10000,
              "contract": {"mode": "a5-only"}, "evaluations": [row]}
    path = tmp_path / "report.json"
    return path, report


def write(path, report):
    path.write_text(json.dumps(report))
    return select_gate(path)


def test_single_full_word_is_positive(tmp_path):
    path, report = make_report(tmp_path, 1)
    result = write(path, report)
    assert result["eligible"]
    assert result["first_positive"]["whole_word_correct"] == 1
    assert not result["positive_by_3000"]


def test_token_or_final_accuracy_cannot_enable_gate(tmp_path):
    path, report = make_report(tmp_path, 0)
    report["evaluations"][0].update(token_accuracy=.99, final_state_accuracy=1.0)
    assert not write(path, report)["eligible"]


def test_positive_subset_is_not_confirmation(tmp_path):
    path, report = make_report(tmp_path)
    subset = dict(report["evaluations"][0], scope="subset", evaluated_rows=4096,
                  whole_word_correct=1, whole_word_exact_match=1/4096)
    report["evaluations"].append(subset)
    assert not write(path, report)["eligible"]


@pytest.mark.parametrize("change", ["checkpoint", "count", "rows", "nonfinite", "unfinished"])
def test_invalid_evidence_rejected(tmp_path, change):
    path, report = make_report(tmp_path, 1)
    row = report["evaluations"][0]
    if change == "checkpoint":
        row["checkpoint"]["sha256"] = "0" * 64
    elif change == "count":
        row["whole_word_correct"] = 2
    elif change == "rows":
        row["evaluated_rows"] = 4096
    elif change == "nonfinite":
        row["whole_word_exact_match"] = float("nan")
    else:
        report["status"] = "running"
    with pytest.raises(ValueError):
        write(path, report)


def test_prior_positive_survives_resume_but_late_parent_rows_do_not(tmp_path):
    path, report = make_report(tmp_path, 0)
    prior_cp = tmp_path / "step-001000.pt"
    prior_cp.write_bytes(b"positive-prior-state")
    prior_row = dict(report["evaluations"][0], update=1000, whole_word_correct=7,
                     whole_word_exact_match=7/102400,
                     checkpoint={"path": str(prior_cp), "sha256": hashlib.sha256(prior_cp.read_bytes()).hexdigest(),
                                 "completed_updates": 1000})
    parent = {"contract": report["contract"], "evaluations": [prior_row, dict(prior_row, update=9000)]}
    parent_path = tmp_path / "parent-report.json"
    parent_path.write_text(json.dumps(parent))
    report["a5_positive_gate"] = {"inherited_from": {"report": str(parent_path),
        "report_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(), "restored_update": 3000}}
    result = write(path, report)
    assert result["eligible"] and result["positive_by_3000"]
    assert result["first_positive"]["update"] == 1000
    assert [row["update"] for row in result["evidence"]] == [1000, 10000]
