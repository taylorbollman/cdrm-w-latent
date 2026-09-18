"""Guard cached baseline identity and the scope of the two-model report."""
import copy
import json

import pytest

from scripts import rt_nextlat_fuzzy_length_value_compare as compare


def test_cache_requires_matching_data_complete_scope_and_unchanged_evidence(tmp_path, monkeypatch):
    evidence = {"schema": compare.baseline_probe.SCHEMA, "status": "complete",
                "optimizer_updates": 0, "confirmation_evaluated": False, "latent_rollout_evaluated": False,
                "data_manifest_sha256": "shared-data", "source_manifest": {},
                "arms": {"baseline": {"model_unchanged": True, "model_sha256_before": "same",
                          "model_sha256_after": "same", "metrics": {str(n): {} for n in (400, 512, 1024)}}}}
    monkeypatch.setattr(compare.common, "check_metric", lambda metric: metric)
    for name in ("evidence.json", "report.json"):
        (tmp_path / name).write_text(json.dumps(evidence))
    arm, receipt = compare.load_cached_baseline(tmp_path, "shared-data")
    assert arm["model_unchanged"] and len(receipt["evidence_sha256"]) == 64
    with pytest.raises(ValueError, match="different length-evaluation data"):
        compare.load_cached_baseline(tmp_path, "wrong-data")
    changed = copy.deepcopy(evidence)
    changed["optimizer_updates"] = 1
    (tmp_path / "report.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="unchanged completed baseline"):
        compare.load_cached_baseline(tmp_path, "shared-data")


def test_two_arm_report_explains_cache_and_does_not_infer_internal_mechanisms():
    metric = {key: .5 for key in ("answer_accuracy", "first_value_token_accuracy", "terminal_probe_accuracy",
                                "terminal_first_value_token_accuracy", "answer_motif_exact_match", "sequence_exact_match")}
    summary = {"arms": {label: {"checkpoint": {"sha256": label}, "parameters": count,
                      "metrics": {str(n): copy.deepcopy(metric) for n in (400, 512, 1024)}}
                       for label, count in (("baseline", 479616), ("value", 496000))},
               "eval_microbatch": 64, "data_manifest_sha256": "shared-data", "figures": []}
    summary["arms"]["value"]["metrics"]["1024"]["answer_accuracy"] = .6
    text = compare.markdown(summary)
    assert "+10.0000" in text
    assert "baseline measurements are reused" in text
    assert "cannot identify the baseline's internal mechanism" in text
    assert "all four" not in text
