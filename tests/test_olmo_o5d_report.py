"""Matched finite/online selections and descriptive transfer contrasts."""
import copy
import json
import math

import pytest

from scripts import olmo_o5d_report as report


def container(case, endpoint, split):
    online = case["passes"] is None
    mode = {"beta": 1., "rt_mode": {"selected_layers": [], "alpha": 1.}}
    if not online:
        mode.update(enabled=True, num_passes=case["passes"])
    metrics = []
    for index in range(1 if online else case["passes"]):
        depth = 4 if online else index
        # Mixed repairs .4 NLL at K2, but loses .1 of that repair online.
        delta = 0. if depth == 0 else 1. + .02 * (depth - 1)
        if endpoint == "mixed" and depth:
            delta -= .4
            if online:
                delta += .1
        records = []
        for i in range(case["rows"]):
            count = min(100 + i % 7, case["max_length"] - 1)
            nll = 2. + .01 * (i % 11) + delta + (1. if split == "retention_dev" else 0.)
            records.append({"batch_index": i // 8, "row_index": i % 8, "document_id": i // 3,
                "ce_count": count, "ce_sum": nll * count, "mean_nll": nll, "next_token_correct": count // 2})
        count = sum(r["ce_count"] for r in records)
        total = math.fsum(r["ce_sum"] for r in records)
        hits = sum(r["next_token_correct"] for r in records)
        metric = {"schema": "olmo-fbt-pass-evaluation-v1", "execution": "online" if online else "finite",
            "pass_index": None if online else index, "precision": "bf16_mixed", "mode": mode,
            "positions_per_chunk": 128, "batches": math.ceil(case["rows"] / 8), "documents": case["rows"],
            "input_tokens": count + case["rows"], "ce_count": count, "ce_sum": total,
            "mean_nll": total / count, "perplexity": math.exp(total / count), "next_token_correct": hits,
            "next_token_accuracy": hits / count, "document_records": records}
        metrics.append(metric)
    keys = ("precision", "mode", "execution", "positions_per_chunk", "batches", "documents", "input_tokens")
    return {"schema": "olmo-fbt-evaluation-v1", **{k: metrics[0][k] for k in keys}, "passes": metrics}


@pytest.fixture
def diagnostic():
    native = {"backbone.backbone.transformer.wte.weight": "a" * 64,
              "backbone.fusion.output_scale": "b" * 64}
    endpoints = {}
    cases = []
    for endpoint in report.ENDPOINTS:
        state = {**native, **{k: ("c" if endpoint == "source" else "d") * 64 for k in report.FUSION_NAMES}}
        endpoints[endpoint] = {"state_before": state, "state_after": copy.deepcopy(state),
            "weights_unchanged": True, "checkpoint": {"sha256": ("e" if endpoint == "source" else "f") * 64,
                                                       "size_bytes": 10000}}
        for section, (rows, length) in report.SECTIONS.items():
            for passes in (2, 3, 4, None):
                case = {"section": section, "beta": 1., "passes": passes, "rows": rows, "max_length": length}
                cases.append({"endpoint": endpoint, "case": case, "label": endpoint + "-" + section + "-" + report.execution(case),
                    "metrics": {split: container(case, endpoint, split) for split in report.SPLITS},
                    "elapsed_seconds": .2, "split_elapsed_seconds": {split: .1 for split in report.SPLITS}})
    return {"schema": report.SCHEMA, "status": "passed", "finished_utc": "now",
        "configuration": {"precision": "bf16_mixed", "batch_size": 8,
            "bootstrap_repetitions": 1000, "bootstrap_seed": 20260922}, "cases": cases, "endpoints": endpoints,
        "source_hashes": {"model.py": "a" * 64}, "diagnostic_source_hashes": {"driver.py": "b" * 64},
        "wandb": {"run_url": "https://wandb.ai/test/project/runs/diagnostic"}}


def test_fixed_grid_matched_contrasts_and_original_document_interaction(diagnostic):
    before = copy.deepcopy(diagnostic)
    result = report.summarize_cases(diagnostic)
    assert diagnostic == before
    for section, value in result.items():
        for split, item in value["splits"].items():
            assert item["mixed_minus_source"]["K2"]["estimate"] == pytest.approx(-.4)
            assert item["mixed_minus_source"]["online"]["estimate"] == pytest.approx(-.3)
            assert item["execution_gap_interaction"]["online"]["estimate"] == pytest.approx(.1)
            assert item["execution_gap_interaction"]["online"]["ci95"] == pytest.approx([.1, .1])
            assert item["document_clusters"] == math.ceil(value["rows"] / 3)
            assert item["endpoints"]["source"]["versus_K2"]["online"]["estimate"] == pytest.approx(.06)
            assert item["endpoints"]["mixed"]["versus_K2"]["online"]["estimate"] == pytest.approx(.16)
    assert result["short_prefix"]["splits"]["dev"]["ce_count"] == 32 * 63
    assert result["full_context"]["splits"]["dev"]["ce_count"] > 512 * 63


@pytest.mark.parametrize("fault", ["schema", "failed", "unfinished", "precision", "batch", "bootstrap_count", "bootstrap_seed", "missing_endpoint",
    "state_change", "backbone_difference", "missing_scale", "checkpoint", "source_hash", "missing_case", "duplicate_case",
    "unexpected_execution", "beta", "length", "rows", "split", "time", "split_time", "pass_count", "execution",
    "rt", "mode_k", "chunk", "target_sum", "doc_alignment", "cross_selection", "ordinary"])
def test_rejects_incompatible_or_misleading_records(diagnostic, fault):
    row = diagnostic["cases"][-1]
    case = row["case"]
    metric = row["metrics"]["dev"]["passes"][-1]
    if fault == "schema": diagnostic["schema"] = "wrong"
    elif fault == "failed": diagnostic["status"] = "failed"
    elif fault == "unfinished": diagnostic["status"] = "running"
    elif fault == "precision": diagnostic["configuration"]["precision"] = "fp32"
    elif fault == "batch": diagnostic["configuration"]["batch_size"] = 16
    elif fault == "bootstrap_count": diagnostic["configuration"]["bootstrap_repetitions"] = 10
    elif fault == "bootstrap_seed": diagnostic["configuration"]["bootstrap_seed"] = 42
    elif fault == "missing_endpoint": del diagnostic["endpoints"]["source"]
    elif fault == "state_change": diagnostic["endpoints"]["mixed"]["state_after"]["backbone.fusion.state_proj.weight"] = "a" * 64
    elif fault == "backbone_difference":
        for when in ("state_before", "state_after"):
            diagnostic["endpoints"]["mixed"][when]["backbone.backbone.transformer.wte.weight"] = "b" * 64
    elif fault == "missing_scale":
        for endpoint in diagnostic["endpoints"].values():
            for when in ("state_before", "state_after"): del endpoint[when]["backbone.fusion.output_scale"]
    elif fault == "checkpoint": diagnostic["endpoints"]["mixed"]["checkpoint"]["sha256"] = "wrong"
    elif fault == "source_hash": diagnostic["diagnostic_source_hashes"] = {}
    elif fault == "missing_case": diagnostic["cases"].pop()
    elif fault == "duplicate_case": diagnostic["cases"].append(copy.deepcopy(row))
    elif fault == "unexpected_execution": case["passes"] = 5
    elif fault == "beta": case["beta"] = .5
    elif fault == "length": case["max_length"] = 256
    elif fault == "rows": case["rows"] = 32
    elif fault == "split": row["metrics"]["test"] = row["metrics"].pop("retention_dev")
    elif fault == "time": row["elapsed_seconds"] = float("nan")
    elif fault == "split_time": del row["split_elapsed_seconds"]["dev"]
    elif fault == "pass_count": row["metrics"]["dev"]["passes"].append(copy.deepcopy(metric))
    elif fault == "execution": metric["pass_index"] = 0
    elif fault == "rt": metric["mode"]["rt_mode"]["selected_layers"] = [0]
    elif fault == "mode_k": diagnostic["cases"][0]["metrics"]["dev"]["passes"][0]["mode"]["num_passes"] = 3
    elif fault == "chunk": metric["positions_per_chunk"] = 64
    elif fault == "target_sum": metric["ce_count"] -= 1
    elif fault == "doc_alignment": metric["document_records"][0]["document_id"] = 999
    elif fault == "cross_selection":
        for r in diagnostic["cases"]:
            if r["case"]["section"] == "short_prefix":
                for m in r["metrics"]["dev"]["passes"]: m["document_records"][0]["document_id"] = 999
    elif fault == "ordinary":
        target = diagnostic["cases"][8]["metrics"]["dev"]["passes"][0]
        target["next_token_correct"] += 1
        target["next_token_accuracy"] = target["next_token_correct"] / target["ce_count"]
        target["document_records"][0]["next_token_correct"] += 1
    with pytest.raises(ValueError): report.validate_report(diagnostic)


def test_running_can_be_summarized_but_not_presented_as_completed(diagnostic):
    diagnostic["status"] = "running"
    del diagnostic["finished_utc"]
    assert report.summarize_cases(diagnostic)["full_context"]["rows"] == 512
    with pytest.raises(ValueError, match="completed"): report.validate_report(diagnostic)


def test_report_serialization_preserves_pass_records_and_clear_scope(diagnostic, tmp_path):
    diagnostic["summary"] = report.summarize_cases(diagnostic)
    text = report.markdown(diagnostic)
    assert "not free-running" in text and "one training seed" in text
    assert "First 32 development windows" in text and "First 512 development windows" in text
    assert "(Online − K2) mixed minus source" in text
    assert "Shared frozen ordinary" in text
    assert "O5c mixed / online" in text
    report.write_figures(diagnostic, tmp_path)
    for name in report.FIGURES:
        assert (tmp_path / name).stat().st_size > 1000
    source = tmp_path / "input.json"
    source.write_text(json.dumps(diagnostic))
    destination = tmp_path / "published"
    report.main(["--report", str(source), "--output-dir", str(destination)])
    saved = json.loads((destination / "report.json").read_text())
    assert saved["cases"] == diagnostic["cases"]
    assert saved["summary"] == diagnostic["summary"]
