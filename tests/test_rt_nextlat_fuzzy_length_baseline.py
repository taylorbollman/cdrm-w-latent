"""Bounded CPU checks for the baseline-only wrapper; common evaluator is frozen."""
import copy

import pytest

from scripts import rt_nextlat_fuzzy_length_baseline as baseline


def test_baseline_guard_and_saved_t400_checkpoint_binding(monkeypatch):
    current = {"endpoint": 15000, "checkpoints": {15000: {"sha256": "selected"}},
               "report": {"status": "complete", "contract": {"mode": "mixed", "batch_per_task": 2560,
                   "streams": {"fuzzy": {"length": 400}},
                   "model_config": {"backbone": {"d_model": 128, "n_layers": 2, "max_sequence_length": 1024}}}}}
    metric = {"checkpoint": {"sha256": "selected"}}
    monkeypatch.setattr(baseline.saved, "load_run", lambda path: current)
    monkeypatch.setattr(baseline.saved, "endpoint_metrics", lambda run: {"fuzzy/dev": metric})
    monkeypatch.setattr(baseline.length_eval, "check_metric", lambda value: value)
    assert baseline.load_baseline("unused") is current
    current["report"]["contract"]["model_config"]["embedding_injection"] = {"variant": "head"}
    with pytest.raises(ValueError, match="cannot evaluate an embedding variant"):
        baseline.load_baseline("unused")
    del current["report"]["contract"]["model_config"]["embedding_injection"]
    metric["checkpoint"]["sha256"] = "other"
    with pytest.raises(ValueError, match="another checkpoint"):
        baseline.load_baseline("unused")
    current["endpoint"] = 14999
    with pytest.raises(ValueError, match="completed baseline 15k"):
        baseline.load_baseline("unused")


def test_single_baseline_narrative_does_not_claim_four_arm_evaluation():
    metric = {"answer_accuracy": .95, "answer_motif_exact_match": .9, "sequence_exact_match": .5,
              "first_value_token_accuracy": .93, "terminal_probe_accuracy": .92, "known_history_accuracy": .96,
              "answer_tokens": 100, "first_value_token_tokens": 50, "known_history_tokens": 99,
              "unavailable_history_tokens": 1, "oracle_coverage": .99}
    summary = {"arms": {"baseline": {"parameters": 479616, "checkpoint": {"sha256": "selected"},
                "metrics": {str(length): copy.deepcopy(metric) for length in (400, 512, 1024)}}},
               "eval_microbatch": 64, "data_manifest_sha256": "data", "figures": []}
    text = baseline.markdown(summary)
    assert "Baseline-only" in text and "zero training updates" in text
    assert "no embedding variant is evaluated" in text
    assert "shared by all four arms" not in text
    assert "not necessarily harder" in text


def test_source_manifest_includes_transitive_adapter_and_new_wrapper(monkeypatch):
    monkeypatch.setattr(baseline.length_eval, "verify_live_sources", lambda runs: {"original": "hash"})
    monkeypatch.setattr(baseline.adapter, "source_manifest", lambda: {"adapter": "extra"})
    result = baseline.source_manifest({})
    assert result["original"] == "hash" and result["adapter"] == "extra"
    assert len(result["scripts/rt_nextlat_fuzzy_length_baseline.py"]) == 64
