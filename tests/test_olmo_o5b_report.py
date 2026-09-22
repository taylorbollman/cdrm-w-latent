"""Matched finite/online inference records, exposure and original-document CIs."""
import copy
import json
import math
from types import SimpleNamespace

import pytest

from cdrm.pretrained.lm_schedule import build_schedule, alpha_for_update
from scripts import olmo_o5b_report as report


def metric(rows=512, *, shift=0., beta=0., index=0, online=False, records=True, short=False):
    values = []
    for i in range(rows):
        count = min(80 + i % 7, 63) if short else 80 + i % 7
        mean = 2. + (i % 11) * .01 + shift
        values.append({"batch_index": i // 8, "row_index": i % 8, "document_id": i // 3,
                       "ce_count": count, "ce_sum": mean * count, "mean_nll": mean, "next_token_correct": count // 2})
    count = sum(v["ce_count"] for v in values)
    total = math.fsum(v["ce_sum"] for v in values)
    hits = sum(v["next_token_correct"] for v in values)
    result = {"schema": "olmo-fbt-pass-evaluation-v1", "execution": "online" if online else "finite",
              "pass_index": None if online else index, "precision": "bf16_mixed",
              "mode": report._mode(beta, "online" if online else "finite"),
              "positions_per_chunk": 128, "batches": math.ceil(rows / 8), "documents": rows,
              "input_tokens": count + rows, "ce_count": count, "ce_sum": total, "mean_nll": total / count,
              "perplexity": math.exp(total / count), "next_token_correct": hits, "next_token_accuracy": hits / count}
    if records:
        result["document_records"] = values
    return result


def container(rows=512, *, shift=0., improvement=0., beta=0., records=True):
    passes = [metric(rows, shift=shift + improvement * index, beta=beta, index=index, records=records) for index in (0, 1)]
    keys = ("precision", "mode", "execution", "positions_per_chunk", "batches", "documents", "input_tokens")
    return {"schema": "olmo-fbt-evaluation-v1", **{k: passes[0][k] for k in keys}, "passes": passes}


def short_metrics(beta=0., shift=0., improvement=0.):
    return {split: {"pass0": metric(32, beta=beta, shift=shift, short=True),
                    "finite": metric(32, beta=beta, shift=shift + improvement, short=True, index=1),
                    "online": metric(32, beta=beta, shift=shift + improvement - .03, short=True, online=True)}
            for split in report.SPLITS}


@pytest.fixture
def inputs():
    schedule = build_schedule([8] * 12, batch_size=2, warmup_updates=1, ramp_min_tokens=16, ramp_min_updates=1)
    sources = {"scripts/olmo_o5b_train.py": "b" * 64, "cdrm/pretrained/olmo_fbt.py": "c" * 64}
    config = {"schema": "olmo-o5b-pilot-config-v1", "source_hashes": sources, "schedule": schedule,
              "checkpoint_sha256": "e" * 64, "data_manifest_sha256": "d" * 64, "physical_batch_size": 1,
              "precision": "bf16_mixed", "eval_rows": 128, "final_eval_rows": 512,
              "online_eval_rows": 32, "online_eval_length": 64, "num_passes": 2, "gamma": 1.,
              "rt_layers": [], "nextlat_enabled": False, "prefix_mixin": False, "hidden_jitter": 0.}
    runtime = {"torch": "test", "gpu": "H100", "cuda": "test"}
    preflight = {"schema": "olmo-o5b-preflight-v1", "status": "passed", "finished_utc": "2026-09-22",
                 "source_hashes": sources, "runtime": runtime, "schedule": schedule, "selected_batch_size": 2,
                 "checkpoint": {"sha256": "e" * 64}, "data_manifest_sha256": "d" * 64,
                 "baseline": {split: container() for split in report.SPLITS}, "initial_online": short_metrics(),
                 "cold_feedback": {split: container(128, beta=1., improvement=.1, records=False) for split in report.SPLITS}}
    arms = {}
    for arm in report.ARMS:
        is_fbt = arm == "fbt"
        events = []
        for update, tokens in enumerate(schedule["batch_token_prefix"]):
            beta = alpha_for_update(schedule, update, tokens) if is_fbt else 0.
            shift = (-.2 if is_fbt else -.1) * update / 3
            improvement = -.1 * beta
            events.append({"update": update, "input_tokens": tokens, "beta": beta, "full": False, "rows_requested": 128,
                           "metrics": {split: container(128, shift=shift, improvement=improvement, beta=beta, records=False)
                                       for split in report.SPLITS}})
        final = {**events[-1], "full": True, "rows_requested": 512,
                 "metrics": {split: container(512, shift=shift, improvement=improvement, beta=beta) for split in report.SPLITS}}
        prefix = "gs://fast-chunks/test"
        checkpoint = {"optimizer_updates": 3, "input_tokens": 48, "size_bytes": 1000, "sha256": "f" * 64,
                      "storage": {"uri": f"{prefix}/{arm}/update-000003.pt", "generation": "123", "size_bytes": 1000,
                                  "sha256": "f" * 64, "md5_base64": "test", "verification": "verified"}}
        arms[arm] = {"schema": "olmo-o5b-arm-v1", "arm": arm, "status": "completed", "finished_utc": "2026-09-22",
                     "started_utc": "2026-09-22", "configuration": config, "storage_prefix": prefix,
                     "source_fingerprint": {"checkpoint_sha256": "e" * 64, "data_manifest_sha256": "d" * 64,
                                            "code": sources, "runtime": runtime},
                     "counters": {"optimizer_updates": 3, "microbatches": 6, "documents": 6, "input_tokens": 48,
                                  "ce_positions": 42, "latent_pairs": 0, "kl_triples": 0}, "data_cursor": 6,
                     "checkpoints": [checkpoint], "evaluations": [*events, final],
                     "online_evaluations": [{"update": 3, "input_tokens": 48, "beta": beta, "rows_requested": 32,
                                             "max_length": 64, "metrics": short_metrics(beta, shift, improvement)}],
                     "wandb": {"run_url": f"https://wandb.ai/test/project/runs/{arm}"}}
    return json.loads(json.dumps([preflight, config, arms]))


def test_matched_objectives_and_separate_finite_online_comparisons(inputs):
    original = copy.deepcopy(inputs)
    value = report.build_comparison(*inputs)
    assert inputs == original  # Schema adaptation never mutates evaluator results.
    assert value["schema"] == "olmo-o5b-final-comparison-v1"
    assert value["comparisons"]["dev"]["fbt_pass1_minus_ordinary_pass1"]["estimate"] == pytest.approx(-.2)
    assert value["comparisons"]["dev"]["fbt_pass1_minus_fbt_pass0"]["estimate"] == pytest.approx(-.1)
    assert value["comparisons"]["dev"]["fbt_pass1_minus_fbt_pass0"]["document_clusters"] == 171
    assert value["online_comparisons"]["dev"]["fbt_online_minus_fbt_finite"]["estimate"] == pytest.approx(-.03)
    assert value["online_comparisons"]["dev"]["fbt_online_minus_fbt_finite"]["windows"] == 32
    assert all(row["rows_requested"] == 128 for rows in value["curves"].values() for row in rows)
    assert value["exposure"]["microbatches"] == 6


@pytest.mark.parametrize("case", ["missing_arm", "running", "source", "runtime", "data", "config", "schedule", "physical",
                                  "tokens", "cursor", "microbatches", "ce", "aux", "checkpoint", "storage", "final_missing",
                                  "final_mode", "final_sample", "final_count", "curve_sample", "curve_beta", "curve_endpoint",
                                  "document_alignment", "pass_alignment", "ordinary_pass_difference", "online_missing",
                                  "online_rows", "online_context", "online_execution", "online_alignment", "online_selection",
                                  "online_duplicates", "curve_duplicates", "nextlat", "gamma", "jitter", "pass_schema"])
def test_rejects_incompatible_or_incomplete_evidence(inputs, case):
    preflight, config, arms = inputs
    arm = arms["fbt"]
    final = arm["evaluations"][-1]
    metric1 = final["metrics"]["dev"]["passes"][1]
    online = arm["online_evaluations"][0]
    if case == "missing_arm": del arms["ordinary"]
    elif case == "running": arm["status"] = "running"
    elif case == "source": arm["source_fingerprint"]["code"]["scripts/olmo_o5b_train.py"] = "a" * 64
    elif case == "runtime": arm["source_fingerprint"]["runtime"]["gpu"] = "other"
    elif case == "data": arm["source_fingerprint"]["data_manifest_sha256"] = "a" * 64
    elif case == "config": arm["configuration"]["precision"] = "fp32"
    elif case == "schedule": preflight["schedule"]["total_tokens"] += 1
    elif case == "physical": config["physical_batch_size"] = 3
    elif case == "tokens": arm["counters"]["input_tokens"] += 1
    elif case == "cursor": arm["data_cursor"] -= 1
    elif case == "microbatches": arm["counters"]["microbatches"] = 3
    elif case == "ce": arm["counters"]["ce_positions"] += 1
    elif case == "aux": arm["counters"]["latent_pairs"] = 42
    elif case == "checkpoint": arm["checkpoints"] = []
    elif case == "storage": arm["checkpoints"][0]["storage"]["sha256"] = "a" * 64
    elif case == "final_missing": arm["evaluations"].pop()
    elif case == "final_mode": metric1["mode"]["beta"] = .5
    elif case == "final_sample": final["rows_requested"] = 128
    elif case == "final_count": metric1["documents"] = 511
    elif case == "curve_sample": arm["evaluations"][0]["rows_requested"] = 512
    elif case == "curve_beta": arm["evaluations"][0]["beta"] = 1.
    elif case == "curve_endpoint": arm["evaluations"].pop(0)
    elif case == "document_alignment":
        for value in final["metrics"]["dev"]["passes"]: value["document_records"][0]["document_id"] = 999
    elif case == "pass_alignment": metric1["document_records"][0]["document_id"] = 999
    elif case == "ordinary_pass_difference":
        ordinary = arms["ordinary"]["evaluations"][-1]["metrics"]["dev"]
        ordinary["passes"][1] = metric(shift=.1, index=1)
    elif case == "online_missing": arm["online_evaluations"] = []
    elif case == "online_rows": online["rows_requested"] = 31
    elif case == "online_context": online["max_length"] = 512
    elif case == "online_execution": online["metrics"]["dev"]["online"]["execution"] = "finite"
    elif case == "online_alignment": online["metrics"]["dev"]["online"]["document_records"][0]["document_id"] = 999
    elif case == "online_selection":
        for value in online["metrics"]["dev"].values(): value["document_records"][0]["document_id"] = 999
    elif case == "online_duplicates": arm["online_evaluations"].append(copy.deepcopy(online))
    elif case == "curve_duplicates": arm["evaluations"].append(copy.deepcopy(arm["evaluations"][0]))
    elif case == "nextlat": config["nextlat_enabled"] = True
    elif case == "gamma": config["gamma"] = .5
    elif case == "jitter": config["hidden_jitter"] = .02
    elif case == "pass_schema": metric1["schema"] = "olmo-lm-evaluation-v1"
    with pytest.raises(ValueError):
        report.build_comparison(*inputs)


def test_report_rejects_nll_disagreeing_with_selected_targets(inputs):
    inputs[2]["fbt"]["evaluations"][-1]["metrics"]["dev"]["passes"][1]["mean_nll"] += .1
    with pytest.raises(ValueError, match="mean differs"):
        report.build_comparison(*inputs)


def test_bootstrap_merges_windows_of_same_original_document():
    left, right = metric(6, shift=-.1), metric(6)
    result = report.contrast(left, right)
    assert result["document_clusters"] == 2
    assert result["ci95"] == pytest.approx([-.1, -.1])
    assert result == report.contrast(left, right)


def write_inputs(tmp_path, inputs):
    preflight, config, arms = inputs
    directory = tmp_path / "preflight"
    directory.mkdir()
    (directory / "report.json").write_text(json.dumps(preflight))
    (directory / "configuration.json").write_text(json.dumps(config))
    runs = tmp_path / "runs"
    for arm, value in arms.items():
        (runs / arm).mkdir(parents=True)
        (runs / arm / "report.json").write_text(json.dumps(value))
    return SimpleNamespace(preflight=directory, runs=runs, output_dir=tmp_path / "report")


def test_unfinished_queue_produces_no_report_files(tmp_path, inputs):
    inputs[2]["ordinary"]["status"] = "running"
    args = write_inputs(tmp_path, inputs)
    with pytest.raises(ValueError, match="not completed"):
        report.build_report(args)
    assert not args.output_dir.exists()


def test_report_generates_standalone_figures_and_preserves_protocol(tmp_path, inputs):
    args = write_inputs(tmp_path, inputs)
    args.output_dir.mkdir()
    protocol = args.output_dir / "protocol.md"
    protocol.write_text("Frozen protocol\n")
    validated = report.validate_runs(args.preflight, args.runs)
    result = report.build_report(args)
    assert all(result[key] == value for key, value in validated.items())
    assert protocol.read_text() == "Frozen protocol\n"
    for name in (*report.FIGURES, "results.md", "final-comparison.json"):
        assert (args.output_dir / name).stat().st_size > 0
    assert (args.output_dir / "online-comparison.pdf").read_bytes().startswith(b"%PDF")
    assert result["report_source_sha256"] == report.sha(report.__file__)
    text = (args.output_dir / "results.md").read_text()
    assert "512-window" in text and "128-window" in text and "maximum-64-token" in text
    assert "teacher-forced" in text and "not a claim of equal measured compute" in text
    assert "https://wandb.ai/" in text and "gs://fast-chunks/test/fbt/" in text
