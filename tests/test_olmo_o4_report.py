"""Completed-lineage checks and original-document paired inference for O4."""
import copy
import json
import math
from types import SimpleNamespace

import pytest

from cdrm.pretrained.lm_schedule import alpha_for_update, build_schedule
from scripts import olmo_o4_report as report


def metric(means=(2.0, 2.5, 3.0), counts=(2, 3, 5), docs=(0, 0, 1), *, mode=None, records=True):
    rows = [{"batch_index": 0, "row_index": i, "document_id": doc,
             "ce_sum": mean * count, "ce_count": count, "mean_nll": mean,
             "next_token_correct": count // 2}
            for i, (mean, count, doc) in enumerate(zip(means, counts, docs))]
    count = sum(counts)
    total = math.fsum(row["ce_sum"] for row in rows)
    correct = sum(row["next_token_correct"] for row in rows)
    result = {"schema": "olmo-lm-evaluation-v1", "precision": "bf16_mixed",
              "mode": mode or {"selected_layers": [], "alpha": 0.0},
              "ce_sum": total, "ce_count": count, "mean_nll": total / count,
              "perplexity": math.exp(total / count), "next_token_correct": correct,
              "next_token_accuracy": correct / count, "documents": len(rows)}
    if records:
        result["document_records"] = rows
    return result


@pytest.fixture
def inputs():
    schedule = build_schedule([8] * 12, batch_size=2, warmup_updates=1,
                              ramp_min_tokens=16, ramp_min_updates=1)
    sources = {"scripts/olmo_o4_train.py": "b" * 64, "cdrm/pretrained/nextlat.py": "c" * 64}
    configuration = {"schema": "olmo-o4-pilot-config-v1", "source_hashes": sources,
                     "data_manifest_sha256": "d" * 64, "checkpoint_sha256": "e" * 64,
                     "schedule": schedule, "precision": "bf16_mixed", "eval_rows": 128, "final_eval_rows": 512}
    runtime = {"torch": "test", "gpu": "H100", "cuda": "test"}
    preflight = {"schema": "olmo-o4-preflight-v1", "status": "passed", "finished_utc": "2026-09-22",
                 "source_hashes": sources, "runtime": runtime, "schedule": schedule,
                 "selected_batch_size": 2, "data_manifest_sha256": "d" * 64,
                 "checkpoint": {"sha256": "e" * 64},
                 "baseline": {"ordinary": {split: metric() for split in report.SPLITS}}}
    arms = {}
    shifts = {"ordinary": -.1, "ordinary-nextlat": -.2, "rt": .1, "rt-nextlat": -.3}
    for arm in report.ARMS:
        use_rt, nextlat = arm.startswith("rt"), "nextlat" in arm
        counters = {"optimizer_updates": 3, "microbatches": 3, "documents": 6, "input_tokens": 48,
                    "ce_positions": 42, "latent_pairs": 42 if nextlat else 0,
                    "kl_triples": 36 if nextlat else 0}
        prefix = "gs://fast-chunks/test"
        checkpoint = {"optimizer_updates": 3, "input_tokens": 48, "size_bytes": 1000, "sha256": "f" * 64,
                      "storage": {"uri": f"{prefix}/{arm}/update-000003.pt", "generation": "123",
                                  "size_bytes": 1000, "sha256": "f" * 64, "md5_base64": "test", "verification": "verified"}}
        evaluations = []
        for update, tokens in enumerate(schedule["batch_token_prefix"]):
            alpha = alpha_for_update(schedule, update, tokens) if use_rt else 0.0
            mode = {"selected_layers": [0] if use_rt else [], "alpha": alpha}
            values = tuple(mean + shifts[arm] * update / 3 for mean in (2., 2.5, 3.))
            evaluations.append({"kind": "evaluation", "update": update, "input_tokens": tokens,
                                "alpha": alpha, "full": False, "rows_requested": 128,
                                "metrics": {split: metric(values, mode=mode, records=False) for split in report.SPLITS}})
        final = {**evaluations[-1], "full": True, "rows_requested": 512,
                 "metrics": {split: metric(tuple(mean + shifts[arm] for mean in (2., 2.5, 3.)),
                                            mode=mode) for split in report.SPLITS}}
        arms[arm] = {"schema": "olmo-o4-arm-v1", "status": "completed", "arm": arm,
                     "configuration": {**configuration, "arm": arm, "storage_prefix": prefix},
                     "source_fingerprint": {"checkpoint_sha256": "e" * 64, "code": sources,
                                            "data_manifest_sha256": "d" * 64, "runtime": runtime},
                     "started_utc": "2026-09-21", "finished_utc": "2026-09-22",
                     "counters": counters, "data_cursor": 6, "evaluations": [*evaluations, final],
                     "checkpoints": [checkpoint], "wandb": {"run_url": f"https://wandb.ai/test/project/runs/{arm}"}}
    # Remove aliasing so mutation tests alter exactly the requested evidence.
    return json.loads(json.dumps([preflight, configuration, arms]))


def test_completed_matched_arms_compare_against_original_and_each_other(inputs):
    result = report.build_comparison(*inputs)
    assert result["status"] == "completed"
    assert result["exposure"]["input_tokens"] == 48
    comparisons = result["comparisons"]["dev"]
    assert len(comparisons["between_arms"]) == 6
    assert comparisons["between_arms"]["rt-nextlat_minus_ordinary"]["estimate"] == pytest.approx(-.2)
    assert comparisons["versus_preflight_ordinary"]["ordinary"]["estimate"] == pytest.approx(-.1)
    assert comparisons["interaction"]["estimate"] == pytest.approx(-.3)
    assert all(row["full"] is False and row["rows_requested"] == 128 for rows in result["curves"].values() for row in rows)


def test_bootstrap_combines_repeated_windows_and_weights_tokens_not_document_means():
    left = metric((2, 2, .5), (1, 9, 90), (0, 0, 1))
    right = metric((1, 1, 1), (1, 9, 90), (0, 0, 1))
    fragmented = report.paired_document_bootstrap(left, right)
    merged = report.paired_document_bootstrap(metric((2, .5), (10, 90), (0, 1)),
                                             metric((1, 1), (10, 90), (0, 1)))
    assert fragmented["estimate"] == pytest.approx(-.35)  # Document-mean contrast would be +.25.
    assert fragmented["document_clusters"] == 2
    assert fragmented["ci95"] == merged["ci95"] == pytest.approx([-.5, 1.])
    assert fragmented == report.paired_document_bootstrap(left, right)


def test_paired_bootstrap_rejects_document_or_target_misalignment():
    with pytest.raises(ValueError, match="alignment"):
        report.paired_document_bootstrap(metric(), metric(docs=(0, 1, 1)))
    with pytest.raises(ValueError, match="alignment"):
        report.paired_document_bootstrap(metric(), metric(counts=(3, 2, 5)))


@pytest.mark.parametrize("case", ["missing_arm", "running", "failed", "source", "runtime", "data", "config", "schedule",
                                  "tokens", "updates", "cursor", "ce_exposure", "kl_exposure", "checkpoint", "storage", "final_missing",
                                  "final_mode", "final_sample", "curve_sample", "curve_alpha", "curve_endpoint", "document_alignment"])
def test_incomplete_or_incompatible_evidence_is_rejected(inputs, case):
    preflight, configuration, arms = inputs
    arm = arms["rt-nextlat"]
    if case == "missing_arm": del arms["rt"]
    elif case in ("running", "failed"): arm["status"] = case
    elif case == "source": arm["source_fingerprint"]["code"]["cdrm/pretrained/nextlat.py"] = "a" * 64
    elif case == "runtime": arm["source_fingerprint"]["runtime"]["gpu"] = "different"
    elif case == "data": arm["source_fingerprint"]["data_manifest_sha256"] = "a" * 64
    elif case == "config": arm["configuration"]["precision"] = "fp32"
    elif case == "schedule": preflight["schedule"]["total_tokens"] += 1
    elif case == "tokens": arm["counters"]["input_tokens"] += 1
    elif case == "updates": arm["counters"]["optimizer_updates"] -= 1
    elif case == "cursor": arm["data_cursor"] -= 1
    elif case == "ce_exposure": arm["counters"]["ce_positions"] -= 1
    elif case == "kl_exposure": arm["counters"]["kl_triples"] -= 1
    elif case == "checkpoint": arm["checkpoints"] = []
    elif case == "storage": arm["checkpoints"][0]["storage"]["sha256"] = "a" * 64
    elif case == "final_missing": arm["evaluations"].pop()
    elif case == "final_mode": arm["evaluations"][-1]["metrics"]["dev"]["mode"]["alpha"] = .5
    elif case == "final_sample": arm["evaluations"][-1]["rows_requested"] = 128
    elif case == "curve_sample": arm["evaluations"][0]["rows_requested"] = 512
    elif case == "curve_alpha": arm["evaluations"][0]["alpha"] = 1.
    elif case == "curve_endpoint": arm["evaluations"].pop(0)
    elif case == "document_alignment": arm["evaluations"][-1]["metrics"]["dev"]["document_records"][0]["document_id"] = 9
    with pytest.raises(ValueError):
        report.build_comparison(preflight, configuration, arms)


def test_nll_summary_cannot_disagree_with_recorded_token_sums(inputs):
    inputs[2]["ordinary"]["evaluations"][-1]["metrics"]["dev"]["mean_nll"] += .1
    with pytest.raises(ValueError, match="mean differs"):
        report.build_comparison(*inputs)


def test_exact_duplicate_resumed_evaluation_is_coalesced_but_conflict_rejected(inputs):
    arms = inputs[2]
    arms["ordinary"]["evaluations"].append(copy.deepcopy(arms["ordinary"]["evaluations"][1]))
    result = report.build_comparison(*inputs)
    assert len(result["curves"]["ordinary"]) == 4
    arms["ordinary"]["evaluations"][-1]["kind"] = "changed_event"
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        report.build_comparison(*inputs)


def amend(inputs):
    preflight, configuration, arms = inputs
    path = "scripts/olmo_o4_train.py"
    old = preflight["source_hashes"][path]
    configuration["source_hashes"][path] = "a" * 64
    amendment = {"changed_file": path, "old_sha256": old, "new_sha256": "a" * 64,
                 "reason": "Resume-only retained checkpoint upload retry; no numerical change.",
                 "previous_source_snapshot": "previous-source.py"}
    configuration["provenance_amendment"] = amendment
    for arm in arms.values():
        arm["configuration"].update(copy.deepcopy(configuration))
        arm["source_fingerprint"]["code"] = copy.deepcopy(configuration["source_hashes"])
    return amendment


def test_explicit_preflight_resume_only_amendment_is_preserved(inputs):
    amendment = amend(inputs)
    summary = report.build_comparison(*inputs)
    assert summary["preflight_provenance_amendment"] == amendment
    assert "provenance amendment" in report.markdown(summary)


@pytest.mark.parametrize("field", ["old_sha256", "new_sha256", "changed_file", "previous_source_snapshot"])
def test_inexact_preflight_source_amendment_is_rejected(inputs, field):
    amend(inputs)
    inputs[1]["provenance_amendment"][field] = ""
    with pytest.raises(ValueError, match="amendment"):
        report.build_comparison(*inputs)


def test_unrelated_preflight_code_change_is_not_excused_by_amendment(inputs):
    amend(inputs)
    inputs[1]["source_hashes"]["cdrm/pretrained/nextlat.py"] = "a" * 64
    with pytest.raises(ValueError, match="Unapproved"):
        report.build_comparison(*inputs)


def write_inputs(tmp_path, inputs):
    preflight, configuration, arms = inputs
    preflight_dir = tmp_path / "preflight"
    preflight_dir.mkdir()
    (preflight_dir / "report.json").write_text(json.dumps(preflight))
    (preflight_dir / "configuration.json").write_text(json.dumps(configuration))
    runs = tmp_path / "runs"
    for name, arm in arms.items():
        directory = runs / name
        directory.mkdir(parents=True)
        (directory / "report.json").write_text(json.dumps(arm))
    return SimpleNamespace(preflight=preflight_dir, runs=runs, output_dir=tmp_path / "output")


def test_incomplete_queue_produces_no_report_or_figure_files(tmp_path, inputs):
    inputs[2]["ordinary"]["status"] = "running"
    args = write_inputs(tmp_path, inputs)
    with pytest.raises(ValueError, match="not completed"):
        report.build_report(args)
    assert not args.output_dir.exists()


def test_report_writes_standalone_figures_and_preserves_recorded_protocol(tmp_path, inputs):
    args = write_inputs(tmp_path, inputs)
    args.output_dir.mkdir()
    protocol = args.output_dir / "protocol.md"
    protocol.write_text("Frozen prior protocol\n")
    summary = report.build_report(args)
    assert protocol.read_text() == "Frozen prior protocol\n"
    for name in ("results.md", "final-comparison.json", "learning-curves.pdf", "learning-curves.png"):
        assert (args.output_dir / name).stat().st_size > 0
    assert (args.output_dir / "learning-curves.pdf").read_bytes().startswith(b"%PDF")
    assert summary["report_source_sha256"] == report.sha(report.__file__)
    text = (args.output_dir / "results.md").read_text()
    assert "512-window" in text and "128-window" in text and "not equal compute" in text
    assert "gs://fast-chunks/test/rt-nextlat/" in text and "https://wandb.ai/" in text
