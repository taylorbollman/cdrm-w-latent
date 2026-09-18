"""Scope of historical gates versus later observed development outcomes."""
import json
from types import SimpleNamespace

import pytest

from scripts import rt_nextlat_a5_fuzzy_extension_report as extension


def metric(update, correct=0, rows=102400):
    accuracy = correct / rows
    return {"task": "a5", "role": "ood_dev", "update": update,
            "rows": rows, "evaluated_rows": rows, "length": 36, "tokens": rows * 36,
            "ce": 1.0, "token_accuracy": accuracy, "final_state_accuracy": accuracy,
            "whole_word_exact_match": accuracy, "whole_word_correct": correct,
            "isolated_state_accuracy": [accuracy] * 36,
            "cumulative_prefix_exactness": [accuracy] * 36,
            "per_position_ce": [1.0] * 36, "checkpoint": {"sha256": "a" * 64}}


@pytest.mark.parametrize("endpoint, first_positive", [(20000, 15000), (14000, None), (14000, 14000)])
def test_extension_and_stopped_endpoint_do_not_rewrite_historical_gate(endpoint, first_positive):
    observations = [metric(10000), metric(11000, 1, rows=4096)]
    if first_positive is not None and first_positive < endpoint:
        observations.append(metric(first_positive, 10))
    observations.append(metric(endpoint, 5 if first_positive is not None else 0))
    historical = extension.legacy.a5_gate(observations)
    result = extension.extension_observations(observations, endpoint)
    assert historical["passed"] is False
    assert result["full_budget"]["first_positive_update"] == first_positive
    assert result["post_10k"]["first_positive_update"] == first_positive
    assert 11000 not in result["full_budget"]["full_evaluation_updates"]
    assert result["actual_endpoint"] == endpoint


def test_new_report_supersedes_only_interpretation_and_preserves_legacy_bytes(tmp_path):
    original, output = tmp_path / "legacy", tmp_path / "extension"
    original.mkdir()
    observations = [metric(10000), metric(15000, 100), metric(20000, 0)]
    evidence = {"schema": extension.legacy.SCHEMA, "status": "complete", "mode": "mixed",
                "source_sha256": extension.legacy.sha(extension.legacy.__file__),
                "endpoint": 20000, "evaluations": observations, "figures": [],
                "final": {"a5/ood_dev": observations[-1]}, "checkpoint": {"sha256": "a" * 64},
                "a5_positive_gate": extension.legacy.a5_gate(observations),
                "qualification": "Historical qualification.", "lineage": [{"directory": "retained-parent"}]}
    for name in ("report.json", "evidence.json"):
        (original / name).write_text(json.dumps(evidence))
    (original / "report.md").write_text(
        "# Original\n\n## A5 continuation gate\n\n"
        "This pilot does not establish length-36 state tracking.\n\n"
        "## Controls and qualifications\n\nHistorical qualification.\n")
    previous = {path.name: path.read_bytes() for path in original.iterdir()}
    result = extension.build_report(SimpleNamespace(legacy_report=original, output=output))
    assert {path.name: path.read_bytes() for path in original.iterdir()} == previous
    assert result["a5_positive_gate"] == result["historical_a5_10k_gate"] == evidence["a5_positive_gate"]
    assert result["a5_extension_observations"]["full_budget"]["first_positive_update"] == 15000
    assert result["a5_extension_observations"]["endpoint"]["whole_word_correct"] == 0
    markdown = (output / "report.md").read_text()
    assert "Authoritative extended-budget interpretation" in markdown
    assert "update 15,000" in markdown
    assert "**0.000000%**" in markdown
    assert "This pilot does not establish length-36 state tracking." not in markdown
    assert "does not establish reliable or general state tracking" in markdown
    assert (output / "legacy-evidence.json").read_bytes() == previous["evidence.json"]


def test_reject_inconsistent_integer_count_and_future_observation():
    broken = metric(20000, 1)
    broken["whole_word_correct"] = 2
    with pytest.raises(ValueError, match="whole_word_correct"):
        extension.extension_observations([broken], 20000)
    with pytest.raises(ValueError, match="exceeds endpoint"):
        extension.extension_observations([metric(20001)], 20000)
